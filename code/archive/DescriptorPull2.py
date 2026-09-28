#!/usr/bin/env python3
"""
Ultimate Ensemble for HOMO-LUMO Gap Prediction
Strategy: Multiple complementary models + ensemble averaging
Target: MAE < 1.0 eV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Advanced Feature Engineering
# ============================================================================

def extract_advanced_features(bond_matrix, descriptors):
    """Extract comprehensive features from bond matrix"""
    n_atoms = len(bond_matrix)
    
    # 1. Graph topology
    adj = (np.array(bond_matrix) > 0).astype(float)
    degrees = adj.sum(axis=1)
    
    # 2. Bond characteristics
    bonds = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if bond_matrix[i][j] > 0:
                bonds.append(bond_matrix[i][j])
    bonds = np.array(bonds) if bonds else np.array([0])
    
    # 3. Eigenvalues of adjacency matrix (spectral features)
    try:
        eigenvalues = np.linalg.eigvalsh(adj)
        spectral_gap = eigenvalues[-1] - eigenvalues[-2] if len(eigenvalues) > 1 else 0
        eigenvalue_sum = np.sum(np.abs(eigenvalues))
    except:
        spectral_gap = 0
        eigenvalue_sum = 0
    
    # 4. Path features (connectivity)
    adj2 = np.linalg.matrix_power(adj, 2)  # 2-hop connections
    adj3 = np.linalg.matrix_power(adj, 3)  # 3-hop connections
    
    features = [
        # Size features
        n_atoms,
        len(bonds),
        
        # Degree statistics
        np.mean(degrees) if len(degrees) > 0 else 0,
        np.max(degrees) if len(degrees) > 0 else 0,
        np.min(degrees) if len(degrees) > 0 else 0,
        np.std(degrees) if len(degrees) > 0 else 0,
        np.sum(degrees == 1),  # Terminal atoms
        np.sum(degrees == 2),  # Linear atoms
        np.sum(degrees == 3),  # Branching atoms
        np.sum(degrees >= 4),  # Highly connected atoms
        
        # Bond statistics
        np.mean(bonds),
        np.max(bonds),
        np.sum(bonds > 1),  # Double/triple bonds
        np.sum(bonds > 1.5),  # Aromatic bonds
        np.sum(bonds == 1),  # Single bonds
        
        # Spectral features
        spectral_gap,
        eigenvalue_sum / n_atoms if n_atoms > 0 else 0,
        
        # Path features
        np.sum(adj2) / (n_atoms ** 2) if n_atoms > 0 else 0,  # 2-hop density
        np.sum(adj3) / (n_atoms ** 2) if n_atoms > 0 else 0,  # 3-hop density
        
        # Ring features (cycles)
        np.trace(adj2) / 2,  # 2-cycles
        np.trace(adj3) / 3,  # 3-cycles
        
        # Molecular descriptors
        *descriptors
    ]
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Neural Network Models
# ============================================================================

class DeepNN(nn.Module):
    """Deep neural network"""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class WideNN(nn.Module):
    """Wide shallow network"""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class ResidualNN(nn.Module):
    """Residual network with skip connections"""
    def __init__(self, n_features):
        super().__init__()
        self.input_layer = nn.Linear(n_features, 256)
        
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.block1(x)
        x = F.relu(x)
        x = x + self.block2(x)
        x = F.relu(x)
        return self.output(x)


# ============================================================================
# Dataset
# ============================================================================

class AdvancedDataset(Dataset):
    def __init__(self, data_path, scaler=None, fit_scaler=False):
        self.data = []
        
        print(f"Loading {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                raw = json.loads(line.strip())
                if isinstance(raw, list):
                    self.data.append({
                        'bond_matrix': raw[0],
                        'descriptors': raw[2:16],
                        'gap': raw[16],
                    })
        
        print(f"Loaded {len(self.data)} molecules")
        print("Extracting advanced features...")
        
        # Extract features
        all_features = []
        for item in self.data:
            feats = extract_advanced_features(item['bond_matrix'], item['descriptors'])
            all_features.append(feats)
            item['features'] = feats
        
        all_features = np.array(all_features)
        print(f"Feature dimension: {all_features.shape[1]}")
        
        # Scale
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
        else:
            self.scaler = scaler
        
        if self.scaler:
            all_features = self.scaler.transform(all_features)
        
        for i, item in enumerate(self.data):
            item['scaled_features'] = all_features[i]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'features': torch.FloatTensor(item['scaled_features']),
            'gap': torch.FloatTensor([item['gap']])
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_nn(model, train_loader, val_loader, device, epochs=150):
    """Train neural network"""
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_mae = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            gap = batch['gap'].to(device)
            
            pred = model(features)
            loss = F.mse_loss(pred, gap) + 0.3 * F.l1_loss(pred, gap)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    gap = batch['gap'].to(device)
                    pred = model(features)
                    preds.extend(pred.cpu().numpy().flatten())
                    targets.extend(gap.cpu().numpy().flatten())
            
            val_mae = mean_absolute_error(targets, preds)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience = 0
            else:
                patience += 1
            
            if patience >= 6:  # 30 epochs patience
                break
    
    return best_val_mae


def predict_nn(model, loader, device):
    """Get predictions from neural network"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            pred = model(features)
            preds.extend(pred.cpu().numpy().flatten())
    return np.array(preds)


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cpu')
    
    print("="*60)
    print("ULTIMATE ENSEMBLE")
    print("="*60)
    print("Strategy: 5 diverse models + ensemble averaging")
    print("  1. Deep NN (4 layers, 256 hidden)")
    print("  2. Wide NN (2 layers, 512 hidden)")
    print("  3. Residual NN (skip connections)")
    print("  4. Gradient Boosting (sklearn)")
    print("  5. Random Forest (sklearn)")
    print()
    
    # Load data
    carbon_path = Path("carbon_only_dataset_enhanced_clean.json")
    
    print("="*60)
    print("CARBON PATHWAY")
    print("="*60)
    
    carbon_data = AdvancedDataset(carbon_path, scaler=None, fit_scaler=True)
    
    indices = list(range(len(carbon_data)))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    train_set = torch.utils.data.Subset(carbon_data, train_idx)
    test_set = torch.utils.data.Subset(carbon_data, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}\n")
    
    n_features = carbon_data.data[0]['scaled_features'].shape[0]
    
    # Train multiple models
    models = []
    
    print("Training Model 1: Deep NN...")
    model1 = DeepNN(n_features).to(device)
    mae1 = train_nn(model1, train_loader, test_loader, device)
    print(f"  Best Val MAE: {mae1:.4f} eV\n")
    models.append(('deep_nn', model1))
    
    print("Training Model 2: Wide NN...")
    model2 = WideNN(n_features).to(device)
    mae2 = train_nn(model2, train_loader, test_loader, device)
    print(f"  Best Val MAE: {mae2:.4f} eV\n")
    models.append(('wide_nn', model2))
    
    print("Training Model 3: Residual NN...")
    model3 = ResidualNN(n_features).to(device)
    mae3 = train_nn(model3, train_loader, test_loader, device)
    print(f"  Best Val MAE: {mae3:.4f} eV\n")
    models.append(('residual_nn', model3))
    
    # Extract features for sklearn models
    print("Training Model 4: Gradient Boosting...")
    X_train = np.array([carbon_data.data[i]['scaled_features'] for i in train_idx])
    y_train = np.array([carbon_data.data[i]['gap'] for i in train_idx])
    X_test = np.array([carbon_data.data[i]['scaled_features'] for i in test_idx])
    y_test = np.array([carbon_data.data[i]['gap'] for i in test_idx])
    
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                   max_depth=6, random_state=42, verbose=0)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    mae_gb = mean_absolute_error(y_test, pred_gb)
    print(f"  Val MAE: {mae_gb:.4f} eV\n")
    models.append(('gb', gb))
    
    print("Training Model 5: Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, 
                              min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    print(f"  Val MAE: {mae_rf:.4f} eV\n")
    models.append(('rf', rf))
    
    # Ensemble predictions
    print("="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    all_preds = []
    
    # NN predictions
    for name, model in models[:3]:
        preds = predict_nn(model, test_loader, device)
        all_preds.append(preds)
        mae = mean_absolute_error(y_test, preds)
        print(f"{name}: MAE = {mae:.4f} eV")
    
    # Sklearn predictions
    all_preds.append(pred_gb)
    print(f"gb: MAE = {mae_gb:.4f} eV")
    
    all_preds.append(pred_rf)
    print(f"rf: MAE = {mae_rf:.4f} eV")
    
    # Weighted ensemble (weight by inverse MAE)
    maes = [mae1, mae2, mae3, mae_gb, mae_rf]
    weights = [1/m for m in maes]
    weights = np.array(weights) / sum(weights)
    
    print(f"\nEnsemble weights: {weights}")
    
    ensemble_pred = sum(w * pred for w, pred in zip(weights, all_preds))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Ensemble MAE: {ensemble_mae:.4f} eV {'✓ <1.0eV!' if ensemble_mae < 1.0 else '✗'}")
    print(f"Ensemble R²:  {ensemble_r2:.4f}")
    print(f"\nImprovement over best single model: {min(maes) - ensemble_mae:.4f} eV")
    
    # Sample predictions
    print(f"\nSample Predictions:")
    for i in range(10):
        print(f"True: {y_test[i]:.3f} | Pred: {ensemble_pred[i]:.3f} | Error: {abs(y_test[i]-ensemble_pred[i]):.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()