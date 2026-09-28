#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 17:52:58 2026

@author: isaacwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM9 Dataset Descriptor Extraction
Extracts molecular descriptors from QM9 .xyz files for validation

QM9 Format (line by line):
1. Number of atoms
2. Properties: tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
3-N. Atom lines: element, x, y, z, Mulliken_charge
N+1. Frequencies
N+2. SMILES
N+3. InChI

Generates output format: [bond_matrix, smiles, qm9_descriptors(7), lasso_descriptors(500), gap]
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit")


# ============================================================================
# QM9 XYZ Parser
# ============================================================================

def parse_qm9_xyz(filepath):
    """
    Parse a single QM9 .xyz file
    
    Returns:
        dict with keys: n_atoms, properties, coords, elements, charges, 
                       frequencies, smiles, inchi, gap
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Line 1: Number of atoms
        n_atoms = int(lines[0].strip())
        
        # Line 2: Properties
        props = lines[1].strip().split()
        # Format: tag idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv
        tag = props[0]
        idx = int(props[1])
        
        # Extract key properties (matching indices from format)
        properties = {
            'A': float(props[2]),
            'B': float(props[3]),
            'C': float(props[4]),
            'mu': float(props[5]),        # Dipole moment
            'alpha': float(props[6]),     # Polarizability
            'homo': float(props[7]),      # HOMO (eV)
            'lumo': float(props[8]),      # LUMO (eV)
            'gap': float(props[9]),       # Gap (eV)
            'r2': float(props[10]),       # Electronic spatial extent
            'zpve': float(props[11]),     # Zero point vibrational energy
            'U0': float(props[12]),       # Internal energy at 0K
            'U': float(props[13]),        # Internal energy at 298K
            'H': float(props[14]),        # Enthalpy at 298K
            'G': float(props[15]),        # Free energy at 298K
            'Cv': float(props[16])        # Heat capacity at 298K
        }
        
        # Lines 3 to n_atoms+2: Atom coordinates
        coords = []
        elements = []
        charges = []
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            charges.append(float(parts[4]))
        
        coords = np.array(coords)
        
        # Line n_atoms+3: Frequencies
        freq_line = 2 + n_atoms
        frequencies = [float(x) for x in lines[freq_line].strip().split()]
        
        # Line n_atoms+4: SMILES
        smiles_line = 2 + n_atoms + 1
        smiles_parts = lines[smiles_line].strip().split()
        smiles = smiles_parts[0] if smiles_parts else ""
        
        # Line n_atoms+5: InChI
        inchi_line = 2 + n_atoms + 2
        inchi = lines[inchi_line].strip() if len(lines) > inchi_line else ""
        
        return {
            'n_atoms': n_atoms,
            'properties': properties,
            'coords': coords,
            'elements': elements,
            'charges': charges,
            'frequencies': frequencies,
            'smiles': smiles,
            'inchi': inchi,
            'tag': tag,
            'idx': idx
        }
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def create_bond_matrix(coords, elements, cutoff=1.6):
    """
    Create bond-step matrix from 3D coordinates
    Uses distance-based bonding with element-specific radii
    """
    n_atoms = len(coords)
    
    # Covalent radii (Angstroms)
    radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57}
    
    # Distance matrix
    dist_matrix = cdist(coords, coords)
    
    # Bond matrix (1 if bonded, 0 otherwise)
    bond_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Bond if distance < sum of covalent radii * cutoff
            r1 = radii.get(elements[i], 0.7)
            r2 = radii.get(elements[j], 0.7)
            if dist_matrix[i, j] < (r1 + r2) * cutoff:
                bond_matrix[i, j] = 1
                bond_matrix[j, i] = 1
    
    # Convert to bond-step matrix (shortest path distances)
    # Simple approach: iterative matrix multiplication
    bond_step = bond_matrix.copy()
    for k in range(n_atoms):
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j and bond_step[i, j] == 0:
                    # Check if path through k exists
                    if bond_step[i, k] > 0 and bond_step[k, j] > 0:
                        if bond_step[i, j] == 0:
                            bond_step[i, j] = bond_step[i, k] + bond_step[k, j]
                        else:
                            bond_step[i, j] = min(bond_step[i, j], 
                                                 bond_step[i, k] + bond_step[k, j])
    
    return bond_step


# ============================================================================
# RDKit Descriptor Generation (same as PQR)
# ============================================================================

def calculate_rdkit_descriptors(smiles):
    """Generate ~200 RDKit descriptors"""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        desc_dict = {}
        
        # All RDKit descriptors
        descriptor_names = [name for name, _ in Descriptors.descList]
        
        for name in descriptor_names:
            try:
                calc = getattr(Descriptors, name)
                value = calc(mol)
                desc_dict[name] = value
            except:
                pass
        
        # Morgan fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for i in range(min(200, 1024)):
            desc_dict[f'MorganFP_{i}'] = fp[i]
            
        return desc_dict
    except Exception as e:
        return None


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("="*70)
    print("QM9 DATASET DESCRIPTOR EXTRACTION")
    print("="*70)
    print("Validation dataset for small molecules (≤9 heavy atoms)")
    print("Format matches PQR: [bond_matrix, smiles, descriptors(7), lasso(500), gap]")
    print()
    
    # QM9 dataset directory
    qm9_dir = Path("dsgdb9nsd.xyz")
    
    if not qm9_dir.exists():
        print(f"ERROR: {qm9_dir} not found!")
        print("Please ensure the QM9 dataset directory is in the current directory")
        print("Expected structure: dsgdb9nsd.xyz/dsgdb9nsd_000001.xyz, etc.")
        return
    
    # Get all .xyz files
    xyz_files = sorted(qm9_dir.glob("*.xyz"))
    
    if len(xyz_files) == 0:
        print(f"ERROR: No .xyz files found in {qm9_dir}")
        return
    
    print(f"Found {len(xyz_files)} QM9 molecules")
    print()
    
    # Parse all QM9 files
    print("="*70)
    print("PARSING QM9 XYZ FILES")
    print("="*70)
    
    data = []
    
    for i, xyz_file in enumerate(xyz_files):
        if (i + 1) % 1000 == 0:
            print(f"  Parsed {i+1}/{len(xyz_files)} files...")
        
        mol_data = parse_qm9_xyz(xyz_file)
        
        if mol_data is None:
            continue
        
        # Create bond-step matrix
        bond_matrix = create_bond_matrix(mol_data['coords'], mol_data['elements'])
        
        # Extract QM9 descriptors (7 total, matching PQR format)
        qm9_descriptors = [
            mol_data['properties']['mu'],      # Dipole moment
            mol_data['properties']['alpha'],   # Polarizability
            mol_data['properties']['homo'],    # HOMO
            mol_data['properties']['lumo'],    # LUMO
            mol_data['properties']['r2'],      # Electronic spatial extent
            mol_data['properties']['U0'],      # Internal energy at 0K
            mol_data['properties']['Cv']       # Heat capacity
        ]
        
        data.append({
            'bond_matrix': bond_matrix,
            'smiles': mol_data['smiles'],
            'qm9_descriptors': qm9_descriptors,
            'gap': mol_data['properties']['gap'],
            'filename': xyz_file.name
        })
    
    print(f"Successfully parsed {len(data)} molecules\n")
    
    if len(data) == 0:
        print("ERROR: No valid molecules loaded!")
        return
    
    # ========== Generate RDKit Descriptors ==========
    print("="*70)
    print("GENERATING RDKIT DESCRIPTORS")
    print("="*70)
    
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit is required. Please install it.")
        return
    
    all_rdkit_descriptors = []
    valid_indices = []
    
    print(f"Processing {len(data)} molecules...")
    
    for i, item in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)} molecules...")
        
        desc = calculate_rdkit_descriptors(item['smiles'])
        
        if desc is not None:
            all_rdkit_descriptors.append(desc)
            valid_indices.append(i)
    
    # Filter to valid molecules
    data = [data[i] for i in valid_indices]
    
    rdkit_df = pd.DataFrame(all_rdkit_descriptors)
    print(f"\nGenerated {rdkit_df.shape[1]} RDKit descriptors")
    
    # Clean
    rdkit_df = rdkit_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ========== Variance Threshold ==========
    print("\n" + "="*70)
    print("VARIANCE THRESHOLD SELECTION")
    print("="*70)
    
    try:
        selector = VarianceThreshold(threshold=0.01)
        rdkit_filtered = selector.fit_transform(rdkit_df)
        feature_names = rdkit_df.columns[selector.get_support()].tolist()
        print(f"After variance filtering: {len(feature_names)} descriptors")
    except ValueError:
        print("Warning: Variance threshold failed, keeping all.")
        rdkit_filtered = rdkit_df.values
        feature_names = rdkit_df.columns.tolist()
    
    # ========== Lasso Feature Selection ==========
    print("\n" + "="*70)
    print("LASSO FEATURE SELECTION")
    print("="*70)
    
    gaps = np.array([item['gap'] for item in data])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rdkit_filtered)
    
    print("Running LassoCV (selecting top 500 descriptors)...")
    try:
        lasso = LassoCV(cv=3, random_state=42, n_jobs=-1)
        lasso.fit(X_scaled, gaps)
        
        print(f"Optimal alpha: {lasso.alpha_:.6f}")
        print(f"Lasso R²: {lasso.score(X_scaled, gaps):.4f}")
        
        coefficients = np.abs(lasso.coef_)
        n_features = min(500, len(feature_names))
        top_indices = np.argsort(coefficients)[-n_features:][::-1]
        
        selected_features = [feature_names[i] for i in top_indices]
        selected_X = rdkit_filtered[:, top_indices]
        
        print(f"\nSelected top {len(selected_features)} descriptors")
        print(f"Top 10 most important:")
        for i in range(min(10, len(selected_features))):
            print(f"  {i+1}. {selected_features[i]} (|coef|: {coefficients[top_indices[i]]:.4f})")
        
    except Exception as e:
        print(f"Lasso failed: {e}. Using all filtered descriptors.")
        selected_X = rdkit_filtered
        selected_features = feature_names
    
    # ========== Save QM9 Dataset ==========
    print("\n" + "="*70)
    print("SAVING QM9 VALIDATION DATASET")
    print("="*70)
    
    output_file = "qm9_validation_dataset.json"
    
    with open(output_file, 'w') as f:
        for i in range(len(data)):
            # Format: [bond_matrix, smiles, qm9_desc(7), lasso_desc(500), gap]
            enhanced_entry = [
                data[i]['bond_matrix'].tolist(),  # Bond-step matrix
                data[i]['smiles'],                 # SMILES
                data[i]['qm9_descriptors'],        # 7 QM9 descriptors
                selected_X[i].tolist(),            # 500 Lasso descriptors
                data[i]['gap']                     # Gap (eV)
            ]
            f.write(json.dumps(enhanced_entry) + '\n')
    
    print(f"✓ Saved to: {output_file}")
    print(f"  Format: [bond_matrix, smiles, qm9_desc(7), lasso_desc({len(selected_features)}), gap]")
    print(f"  Total molecules: {len(data)}")
    
    # Save metadata
    metadata = {
        'dataset': 'QM9',
        'n_molecules': len(data),
        'n_qm9_descriptors': 7,
        'n_lasso_descriptors': len(selected_features),
        'qm9_descriptor_names': [
            'dipole_moment', 'polarizability', 'homo', 'lumo', 
            'electronic_spatial_extent', 'internal_energy_0K', 'heat_capacity'
        ],
        'lasso_descriptor_names': selected_features,
        'lasso_alpha': float(lasso.alpha_) if 'lasso' in locals() else None,
        'lasso_r2': float(lasso.score(X_scaled, gaps)) if 'lasso' in locals() else None
    }
    
    with open('qm9_metadata.json', 'w') as f:                              fc
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: qm9_metadata.json")
    
    # ========== Summary Statistics ==========
    print("\n" + "="*70)
    print("QM9 DATASET STATISTICS")
    print("="*70)
    
    gaps_array = np.array(gaps)
    print(f"\nGap Distribution:")
    print(f"  Mean: {np.mean(gaps_array):.4f} eV")
    print(f"  Std:  {np.std(gaps_array):.4f} eV")
    print(f"  Min:  {np.min(gaps_array):.4f} eV")
    print(f"  Max:  {np.max(gaps_array):.4f} eV")
    
    n_atoms_list = [len(item['bond_matrix']) for item in data]
    print(f"\nMolecule Sizes:")
    print(f"  Mean atoms: {np.mean(n_atoms_list):.1f}")
    print(f"  Min atoms:  {np.min(n_atoms_list)}")
    print(f"  Max atoms:  {np.max(n_atoms_list)}")
    
    print(f"\nDescriptor Summary:")
    print(f"  QM9 descriptors: 7")
    print(f"  Lasso-selected descriptors: {len(selected_features)}")
    print(f"  Total: {7 + len(selected_features)}")
    
    print("\n✓ QM9 validation dataset ready for model evaluation!")
    print(f"  Use this dataset to validate your model on small molecules")
    print(f"  Expected MAE: Paper reports 0.09-0.13 eV for QM9")


if __name__ == "__main__":
    main()
    