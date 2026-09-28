#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected Descriptor Extraction Script
Fixes: 
1. Handles PQR JSON format (List of Dictionaries)
2. Silences RDKit Deprecation/Valence warnings
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import requests
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings

# --- SILENCE WARNINGS ---
warnings.filterwarnings('ignore') # Silence Python warnings

# Try RDKit and Silence RDKit Logs
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import RDLogger
    
    # This disables the specific "Deprecation" and "Valence" spam you were seeing
    RDLogger.DisableLog('rdApp.*') 
    
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


# ============================================================================
# PubChem API Descriptor Fetching
# ============================================================================

def fetch_pubchem_descriptors(smiles, max_retries=3):
    """
    Fetch molecular descriptors from PubChem API
    This is FREE and doesn't require PaDEL installation
    """
    for attempt in range(max_retries):
        try:
            # Step 1: Get CID from SMILES
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            cid = response.json()['IdentifierList']['CID'][0]
            
            # Step 2: Get properties
            props = [
                'MolecularWeight', 'XLogP', 'TPSA', 'Complexity',
                'HBondDonorCount', 'HBondAcceptorCount', 'RotatableBondCount',
                'HeavyAtomCount', 'AtomStereoCount', 'BondStereoCount',
                'CovalentUnitCount', 'IsotopeAtomCount'
            ]
            
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(props)}/JSON"
            prop_response = requests.get(prop_url, timeout=10)
            
            if prop_response.status_code == 200:
                data = prop_response.json()['PropertyTable']['Properties'][0]
                return data
            
            time.sleep(0.5)  # Rate limiting
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


# ============================================================================
# RDKit Comprehensive Descriptor Generation
# ============================================================================

def calculate_rdkit_descriptors(smiles):
    """
    Calculate comprehensive RDKit descriptors
    ~200 descriptors total
    """
    if not RDKIT_AVAILABLE or not isinstance(smiles, str):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        desc_dict = {}
        
        # Get all available descriptors
        descriptor_names = [name for name, _ in Descriptors.descList]
        
        for name in descriptor_names:
            try:
                calc = getattr(Descriptors, name)
                value = calc(mol)
                desc_dict[name] = value
            except:
                pass
        
        # Add Morgan fingerprint counts (additional features)
        # Using GetMorganFingerprintAsBitVect (even if deprecated) for compatibility
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for i in range(min(200, 1024)):  # First 200 bits
            desc_dict[f'MorganFP_{i}'] = fp[i]
            
        return desc_dict
    except Exception as e:
        return None


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("="*70)
    print("DESCRIPTOR EXTRACTION: PQR (DICT) + LASSO-SELECTED")
    print("="*70)
    print()
    
    # 1. Load PQR dataset
    pqr_path = Path("Pitt_Quantum_Repository_Data.json")
    
    if not pqr_path.exists():
        print(f"ERROR: {pqr_path} not found!")
        print("Please ensure the file is in the current directory")
        return
    
    print(f"Loading {pqr_path} (this may take a moment)...")
    
    # FIXED: Load entire file as one JSON object (List of Dictionaries)
    try:
        with open(pqr_path, 'r') as f:
            full_dataset = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not parse JSON file. {e}")
        return

    print(f"Structure: List containing {len(full_dataset)} molecules.")
    print("Parsing dictionary data...")
    
    data = []
    
    # FIXED: Parse the dictionary structure correctly
    for i, entry in enumerate(full_dataset):
        try:
            # 1. Get SMILES
            smiles = entry.get('smiles')
            if not smiles or not isinstance(smiles, str):
                continue
            
            # 2. Get PM7 Quantum Properties
            pm7 = entry.get('pm7', {})
            if not pm7:
                continue
                
            homo = float(pm7.get('homo', 0))
            lumo = float(pm7.get('lumo', 0))
            
            # 3. Calculate Gap
            gap = lumo - homo
            
            # 4. Extract standard PQR descriptors 
            # (Mapping dictionary keys to a list of features)
            pqr_descriptors = [
                float(entry.get('molecular mass', 0)),
                float(entry.get('exact mass', 0)),
                float(pm7.get('dipoleMoment', 0)),
                float(pm7.get('heatOfFormation', 0)),
                float(pm7.get('polarizability', 0)),
                homo,
                lumo
            ]
            
            data.append({
                'smiles': smiles,
                'pqr_descriptors': pqr_descriptors,
                'gap': gap
            })
            
        except Exception as e:
            # Skip malformed entries silently
            continue
    
    print(f"Successfully parsed {len(data)} valid molecules.\n")
    
    if len(data) == 0:
        print("ERROR: No valid molecules loaded!")
        return
    
    # ========== Generate Descriptors ==========
    print("="*70)
    print("GENERATING RDKit DESCRIPTORS")
    print("="*70)
    
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required for this fix. Please install it.")
        return

    all_external_descriptors = []
    valid_indices = []
    
    print(f"Processing {len(data)} molecules...")
    
    for i, item in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1} molecules...")
        
        # Calculate descriptors
        desc = calculate_rdkit_descriptors(item['smiles'])
        
        if desc is not None:
            all_external_descriptors.append(desc)
            valid_indices.append(i)
        else:
            pass # Skip failed calculations
            
    # Filter data to only include successes
    data = [data[i] for i in valid_indices]
    
    # Convert to DataFrame
    external_df = pd.DataFrame(all_external_descriptors)
    print(f"\nGenerated {external_df.shape[1]} raw RDKit descriptors")
    
    # Clean Data (Inf/NaN to 0)
    external_df = external_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ========== Variance Threshold ==========
    print("\n" + "="*70)
    print("VARIANCE THRESHOLD SELECTION")
    print("="*70)
    
    try:
        selector = VarianceThreshold(threshold=0.01)
        external_filtered = selector.fit_transform(external_df)
        feature_names = external_df.columns[selector.get_support()].tolist()
        print(f"Remaining descriptors: {len(feature_names)}")
    except ValueError:
        print("Warning: Variance threshold failed (maybe all features same?), keeping all.")
        external_filtered = external_df.values
        feature_names = external_df.columns.tolist()
    
    # ========== Lasso Feature Selection ==========
    print("\n" + "="*70)
    print("LASSO FEATURE SELECTION")
    print("="*70)
    
    # Get gaps for valid molecules
    gaps = np.array([item['gap'] for item in data])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(external_filtered)
    
    print("Running LassoCV (finding best descriptors)...")
    try:
        lasso = LassoCV(cv=3, random_state=42, n_jobs=-1)
        lasso.fit(X_scaled, gaps)
        
        print(f"Optimal alpha: {lasso.alpha_:.6f}")
        print(f"Lasso R²: {lasso.score(X_scaled, gaps):.4f}")
        
        # Select features
        coefficients = np.abs(lasso.coef_)
        n_features = min(500, len(feature_names))
        top_indices = np.argsort(coefficients)[-n_features:][::-1]
        
        selected_features = [feature_names[i] for i in top_indices]
        selected_X = external_filtered[:, top_indices]
        
        print(f"\nSelected top {len(selected_features)} descriptors")
        
    except Exception as e:
        print(f"Lasso failed: {e}. Saving all filtered descriptors.")
        selected_X = external_filtered
        selected_features = feature_names

    # ========== Save Enhanced Dataset ==========
    print("\n" + "="*70)
    print("SAVING ENHANCED DATASET")
    print("="*70)
    
    output_file = "enhanced_dataset_lasso.json"
    
    with open(output_file, 'w') as f:
        for i in range(len(data)):
            # Combine everything into the final list structure
            enhanced_entry = [
                None,                          # placeholder for bond_matrix
                data[i]['smiles'],             # smiles
                data[i]['pqr_descriptors'],    # PQR descriptors list
                selected_X[i].tolist(),        # Lasso descriptors list
                data[i]['gap']                 # gap
            ]
            f.write(json.dumps(enhanced_entry) + '\n')
            
    print(f"✓ Saved to: {output_file}")
    print("COMPLETE")

if __name__ == "__main__":
    main()