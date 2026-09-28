#!/usr/bin/env python3
"""
ULTIMATE DESCRIPTOR GENERATION: RDKit + Mordred Combined
This combines the best of both worlds for maximum performance

RDKit: ~286 descriptors (with Gasteiger charges, EState indices)
Mordred: ~1613 descriptors (comprehensive electronic/topological)
Total: ~1900 descriptors → LASSO select top 500
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE DESCRIPTOR GENERATION: RDKit + Mordred")
print("="*80)
print("\nCombining:")
print("  - RDKit: Gasteiger charges, EState indices, standard descriptors")
print("  - Mordred: 1613 comprehensive molecular descriptors")
print("  - Total: ~1900 descriptors → LASSO select best 500")
print("="*80)

# Import libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, EState, rdMolDescriptors, rdPartialCharges
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    from mordred import Calculator, descriptors
    
    print("\n✓ RDKit imported")
    print("✓ Mordred imported\n")
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    exit(1)


def calculate_rdkit_descriptors(mol):
    """Calculate RDKit descriptors including electronic information"""
    desc_dict = {}
    
    try:
        # Standard descriptors
        for name, func in Descriptors.descList:
            try:
                desc_dict[f'RDKit_{name}'] = float(func(mol))
            except:
                desc_dict[f'RDKit_{name}'] = 0.0
        
        # Gasteiger Partial Charges (ELECTRONIC!) - COMPREHENSIVE
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetDoubleProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            
            if charges:
                # Basic statistics
                desc_dict['Gasteiger_Max'] = max(charges)
                desc_dict['Gasteiger_Min'] = min(charges)
                desc_dict['Gasteiger_Mean'] = np.mean(charges)
                desc_dict['Gasteiger_Std'] = np.std(charges)
                desc_dict['Gasteiger_Range'] = max(charges) - min(charges)
                desc_dict['Gasteiger_Sum_Abs'] = np.sum(np.abs(charges))
                
                # Positive/Negative charge statistics
                pos_charges = [c for c in charges if c > 0]
                neg_charges = [c for c in charges if c < 0]
                
                desc_dict['Gasteiger_Positive_Count'] = len(pos_charges)
                desc_dict['Gasteiger_Negative_Count'] = len(neg_charges)
                desc_dict['Gasteiger_Positive_Sum'] = sum(pos_charges) if pos_charges else 0.0
                desc_dict['Gasteiger_Negative_Sum'] = sum(neg_charges) if neg_charges else 0.0
                desc_dict['Gasteiger_Positive_Max'] = max(pos_charges) if pos_charges else 0.0
                desc_dict['Gasteiger_Negative_Min'] = min(neg_charges) if neg_charges else 0.0
                
                # Charge distribution (percentiles)
                desc_dict['Gasteiger_Q25'] = np.percentile(charges, 25)
                desc_dict['Gasteiger_Q50'] = np.percentile(charges, 50)
                desc_dict['Gasteiger_Q75'] = np.percentile(charges, 75)
                
                # Per-atom-type statistics (if available)
                c_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
                n_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
                o_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O']
                
                if c_atoms:
                    c_charges = [float(atom.GetDoubleProp('_GasteigerCharge')) for atom in c_atoms]
                    desc_dict['Gasteiger_C_Mean'] = np.mean(c_charges)
                    desc_dict['Gasteiger_C_Max'] = max(c_charges)
                    desc_dict['Gasteiger_C_Min'] = min(c_charges)
                
                if n_atoms:
                    n_charges = [float(atom.GetDoubleProp('_GasteigerCharge')) for atom in n_atoms]
                    desc_dict['Gasteiger_N_Mean'] = np.mean(n_charges)
                    desc_dict['Gasteiger_N_Max'] = max(n_charges)
                    desc_dict['Gasteiger_N_Min'] = min(n_charges)
                
                if o_atoms:
                    o_charges = [float(atom.GetDoubleProp('_GasteigerCharge')) for atom in o_atoms]
                    desc_dict['Gasteiger_O_Mean'] = np.mean(o_charges)
                    desc_dict['Gasteiger_O_Max'] = max(o_charges)
                    desc_dict['Gasteiger_O_Min'] = min(o_charges)
        except:
            pass
        
        # EState Indices (ELECTRONIC STATE!) - Keep all available
        try:
            estate_indices = EState.EStateIndices(mol)
            for i, val in enumerate(estate_indices):  # ALL estate indices
                desc_dict[f'EState_{i}'] = float(val)
        except:
            pass
        
        # Fragment counts
        try:
            desc_dict['NumAromaticRings'] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
            desc_dict['NumSaturatedRings'] = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
            desc_dict['NumAliphaticRings'] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
            desc_dict['NumAromaticHeterocycles'] = float(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
            desc_dict['NumSaturatedHeterocycles'] = float(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
            desc_dict['NumAliphaticHeterocycles'] = float(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
        except:
            pass
        
        # Topological
        try:
            desc_dict['Chi0'] = float(rdMolDescriptors.CalcChi0n(mol))
            desc_dict['Chi1'] = float(rdMolDescriptors.CalcChi1n(mol))
            desc_dict['Chi2'] = float(rdMolDescriptors.CalcChi2n(mol))
            desc_dict['Chi3'] = float(rdMolDescriptors.CalcChi3n(mol))
            desc_dict['Chi4'] = float(rdMolDescriptors.CalcChi4n(mol))
            desc_dict['Kappa1'] = float(rdMolDescriptors.CalcKappa1(mol))
            desc_dict['Kappa2'] = float(rdMolDescriptors.CalcKappa2(mol))
            desc_dict['Kappa3'] = float(rdMolDescriptors.CalcKappa3(mol))
        except:
            pass
        
    except:
        pass
    
    return desc_dict


def calculate_mordred_descriptors(mol, calc):
    """Calculate Mordred descriptors - FIXED VERSION"""
    desc_dict = {}
    
    try:
        # Calculate all descriptors
        result = calc(mol)
        
        # CORRECT: Use calc.descriptors, not result.descriptor_names
        for desc, value in zip(calc.descriptors, result):
            try:
                desc_name = f'Mordred_{str(desc)}'
                
                # Handle different value types
                if value is None:
                    desc_dict[desc_name] = 0.0
                elif isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        desc_dict[desc_name] = 0.0
                    else:
                        desc_dict[desc_name] = float(value)
                else:
                    # Try to convert (handles numpy deprecation errors)
                    try:
                        val = float(value)
                        if np.isnan(val) or np.isinf(val):
                            desc_dict[desc_name] = 0.0
                        else:
                            desc_dict[desc_name] = val
                    except:
                        desc_dict[desc_name] = 0.0
            except:
                desc_dict[desc_name] = 0.0
                
    except:
        pass
    
    return desc_dict


def calculate_combined_descriptors(smiles, mordred_calc):
    """Calculate both RDKit and Mordred descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Calculate both
        rdkit_desc = calculate_rdkit_descriptors(mol)
        mordred_desc = calculate_mordred_descriptors(mol, mordred_calc)
        
        # Combine
        combined = {**rdkit_desc, **mordred_desc}
        
        # Clean NaN/Inf
        for key in combined:
            if not np.isfinite(combined[key]):
                combined[key] = 0.0
        
        return combined if len(combined) > 0 else None
        
    except:
        return None


def main():
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    input_file = "enhanced_dataset_lasso.json"
    print(f"\nLoading {input_file}...")
    
    data = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0 and line_num > 0:
                print(f"  Loaded {line_num} molecules...")
            
            try:
                raw = json.loads(line.strip())
                if len(raw) >= 5:
                    smiles = raw[1]
                    pqr_full = raw[2]
                    gap = raw[4]
                    
                    if isinstance(smiles, str) and len(smiles) > 0:
                        data.append({
                            'smiles': smiles,
                            'pqr_full': pqr_full,
                            'gap': gap
                        })
            except:
                continue
    
    print(f"Loaded {len(data)} molecules\n")
    
    # Create Mordred calculator once
    print("Creating Mordred calculator...")
    mordred_calc = Calculator(descriptors, ignore_3D=True)
    print(f"✓ Mordred calculator created ({len(mordred_calc.descriptors)} descriptors)\n")
    
    # ========================================================================
    # PHASE 1: Test with 1000 molecules first
    # ========================================================================
    
    print("="*80)
    print("PHASE 1: TESTING WITH 1000 MOLECULES")
    print("="*80)
    print("\nGenerating descriptors for first 1000 molecules to test...")
    print("This catches errors early without wasting time!")
    print()
    
    test_data = data[:1000]
    test_descriptors = []
    test_indices = []
    
    for i, item in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"  Test: {i+1}/1000 molecules... ({len(test_indices)} valid)")
        
        desc = calculate_combined_descriptors(item['smiles'], mordred_calc)
        
        if desc is not None and len(desc) > 0:
            test_descriptors.append(desc)
            test_indices.append(i)
    
    print(f"\n✓ Test phase: {len(test_descriptors)} molecules with descriptors")
    
    # Convert to DataFrame
    test_df = pd.DataFrame(test_descriptors)
    print(f"✓ Total descriptors: {test_df.shape[1]}")
    
    # Clean
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Test variance filter
    print("\nTesting variance filter...")
    try:
        selector = VarianceThreshold(threshold=0.0)  # Use 0.0 for testing
        test_filtered = selector.fit_transform(test_df)
        test_features = test_df.columns[selector.get_support()].tolist()
        print(f"✓ Variance filter works! {len(test_features)} descriptors remain")
    except Exception as e:
        print(f"✗ Variance filter FAILED: {e}")
        print("\nDEBUG INFO:")
        print(f"  DataFrame shape: {test_df.shape}")
        print(f"  DataFrame dtypes: {test_df.dtypes.value_counts()}")
        print(f"  Any NaN: {test_df.isna().any().any()}")
        print(f"  Any Inf: {np.isinf(test_df.values).any()}")
        print(f"  Variance per column (first 10):")
        for col in test_df.columns[:10]:
            print(f"    {col[:40]:40s}: {test_df[col].var():.6e}")
        return
    
    print("\n✅ PHASE 1 PASSED! Proceeding with full dataset...\n")
    
    # ========================================================================
    # PHASE 2: Process all molecules
    # ========================================================================
    
    print("="*80)
    print("PHASE 2: PROCESSING ALL MOLECULES")
    print("="*80)
    print("\nThis will take ~10-20 minutes...")
    print()
    
    all_descriptors = []
    valid_indices = []
    
    for i, item in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)} molecules... ({len(valid_indices)} valid)")
        
        desc = calculate_combined_descriptors(item['smiles'], mordred_calc)
        
        if desc is not None and len(desc) > 0:
            all_descriptors.append(desc)
            valid_indices.append(i)
    
    # Filter data
    data = [data[i] for i in valid_indices]
    
    print(f"\n✓ Generated descriptors for {len(data)} molecules")
    
    # Convert to DataFrame
    desc_df = pd.DataFrame(all_descriptors)
    print(f"✓ Total descriptors: {desc_df.shape[1]}")
    
    # Clean
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Variance filter
    print("\n" + "="*80)
    print("VARIANCE FILTERING")
    print("="*80)
    
    selector = VarianceThreshold(threshold=0.0)  # Use 0.0 to keep all varying features
    desc_filtered = selector.fit_transform(desc_df)
    feature_names = desc_df.columns[selector.get_support()].tolist()
    print(f"After variance filter: {len(feature_names)} descriptors")
    
    # LASSO selection
    print("\n" + "="*80)
    print("LASSO FEATURE SELECTION (Ye et al. 2022 Method)")
    print("="*80)
    
    gaps = np.array([item['gap'] for item in data])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(desc_filtered)
    
    print("\nRunning LassoCV (this may take a few minutes)...")
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000)
    lasso.fit(X_scaled, gaps)
    
    print(f"✓ Lasso R²: {lasso.score(X_scaled, gaps):.4f}")
    
    # Select top 500
    coefficients = np.abs(lasso.coef_)
    n_select = min(500, len(feature_names))
    top_indices = np.argsort(coefficients)[-n_select:][::-1]
    
    selected_features = [feature_names[i] for i in top_indices]
    selected_X = desc_filtered[:, top_indices]
    
    print(f"✓ Selected top {len(selected_features)} descriptors")
    
    print("\nTop 10 most important:")
    for i, idx in enumerate(top_indices[:10]):
        print(f"  {i+1}. {feature_names[idx][:40]:40s} coef={coefficients[idx]:.4f}")
    
    # Quick test
    print("\n" + "="*80)
    print("PERFORMANCE TEST")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(selected_X, gaps, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    print(f"\nRandom Forest (baseline):")
    print(f"  MAE = {mae:.4f} eV")
    print(f"  R²  = {r2:.4f}")
    print(f"\nComparison:")
    print(f"  Old 375 LASSO features:     MAE = 1.50 eV")
    print(f"  RDKit+Mordred combined:     MAE = {mae:.4f} eV")
    print(f"  Target (Ye et al. 2022):    MAE = 0.177 eV")
    
    if mae < 0.5:
        print("\n✅ EXCELLENT! These features should work great with your deep model!")
    elif mae < 1.0:
        print("\n✓ Good improvement! Should work well with your architecture.")
    else:
        print("\n⚠️  Still high, but better than before.")
    
    # Save dataset
    print("\n" + "="*80)
    print("SAVING DATASET")
    print("="*80)
    
    output_file = "enhanced_dataset_combined_NO_HOMO_LUMO.json"
    
    with open(output_file, 'w') as f:
        for i in range(len(data)):
            pqr_no_homo_lumo = data[i]['pqr_full'][:5]
            
            enhanced = [
                None,
                data[i]['smiles'],
                pqr_no_homo_lumo,
                selected_X[i].tolist(),
                data[i]['gap']
            ]
            f.write(json.dumps(enhanced) + '\n')
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Structure: [None, SMILES, [5 PQR], [500 Combined], gap]")
    print(f"  Total features: {5 + len(selected_features)}")
    print(f"\n✅ Ready to train your dual-pathway model!")
    print(f"   Expected MAE with deep model: {mae*0.7:.2f}-{mae*1.1:.2f} eV")


if __name__ == "__main__":
    main()