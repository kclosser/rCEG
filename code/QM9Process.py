#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 16:44:20 2025

@author: isaacwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert QM9 (.mat) into HLGModel training JSONL files expected by your code:
  - carbon_only_dataset_enhanced_clean.json
  - noncarbon_only_dataset_enhanced_clean.json
  - dataset_enhanced_clean.json  (adds has_carbon flag)

Assumptions / mapping:
- HOMO/LUMO present or computable in the .mat (gap = LUMO - HOMO).
- Polarizability: use QM9 alpha (if available). Otherwise, compute via RDKit Gasteiger polar surface approx is NOT used; we skip if missing.
- Surface area: RDKit Labute ASA (approx).
- Volume: RDKit 3D volume via ETKDG + ComputeMolVolume.
- SMILES: prefer 'smiles'/'SMILES' key; skip entries without a SMILES.
"""

import os, json, math, time
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.AllChem import ETKDGv3
from scipy.io import loadmat
import h5py

from typing import Any, Dict, List, Tuple, Optional

# ---------- Bond-step matrix ----------
def bond_weight(b):
    return {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 0.5,
        Chem.BondType.AROMATIC: 0.75,
        Chem.BondType.TRIPLE: 0.33
    }.get(b.GetBondType(), 1.0)

def bond_step_matrix(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    mat = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(mat, 0.0)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w = bond_weight(b)
        mat[i, j] = mat[j, i] = w
    # Floyd–Warshall
    for k in range(n):
        dik = mat[:, k][:, None]
        mkj = mat[k, :][None, :]
        mat = np.minimum(mat, dik + mkj)
    mat[np.isinf(mat)] = -1.0
    return mat

# ---------- Clean descriptor block (no TPSA, no nitro) ----------
def compute_clean_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    n_atoms = mol.GetNumAtoms() or 1

    # non-aromatic double bonds
    db_nonaro = sum(
        1 for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
    )

    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_rings = rdMolDescriptors.CalcNumRings(mol)

    # simple connectivity proxy used in your prior code
    conductance = num_aromatic_rings + db_nonaro

    num_sp2_atoms = sum(1 for a in mol.GetAtoms()
                        if a.GetHybridization() == Chem.HybridizationType.SP2)
    sp2_fraction = float(num_sp2_atoms) / float(n_atoms)

    num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    mw = Descriptors.MolWt(mol)
    abs_formal_charge = abs(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    num_radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

    return dict(
        db_nonaro=db_nonaro,
        mw=mw,
        h_acceptors=h_acceptors,
        conductance=conductance,
        num_aromatic_rings=num_aromatic_rings,
        num_rings=num_rings,
        num_heteroatoms=num_heteroatoms,
        num_sp2_atoms=num_sp2_atoms,
        sp2_fraction=sp2_fraction,
        abs_formal_charge=abs_formal_charge,
        num_radical_electrons=num_radical_electrons
    )

# ---------- 3D metrics: surface area & volume ----------
def add_conformer_and_geometry(mol: Chem.Mol) -> Tuple[float, float]:
    """
    Returns (surface_area, volume).
    Surface area: Labute ASA (approx) via rdMolDescriptors.CalcLabuteASA
    Volume: AllChem.ComputeMolVolume (needs 3D conformer)
    """
    mol = Chem.AddHs(mol)
    params = ETKDGv3()
    params.randomSeed = 42
    try:
        AllChem.EmbedMolecule(mol, params)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # If embedding fails, return sentinel to trigger QC skip
        return (float('nan'), float('nan'))
    # surface area (Å^2), volume (Å^3)
    sa = rdMolDescriptors.CalcLabuteASA(mol)
    vol = AllChem.ComputeMolVolume(mol)
    return (float(sa), float(vol))

# ---------- QC ----------
def sane_values(polarizability: Optional[float], surface_area: float, volume: float, mw: float) -> bool:
    if not (np.isfinite(mw) and mw > 10.0):
        return False
    if polarizability is not None:
        if not (np.isfinite(polarizability) and polarizability >= 0.0):
            return False
    if not (np.isfinite(surface_area) and surface_area > 0.0):
        return False
    if not (np.isfinite(volume) and volume > 0.0):
        return False
    return True

# ---------- .mat loaders (v5 and v7.3/HDF5) ----------
def _mat_as_dict(fname: str) -> Dict[str, Any]:
    try:
        data = loadmat(fname, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        # likely v7.3 HDF5; use h5py
        out = {}
        with h5py.File(fname, "r") as f:
            def fetch(name):
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    val = obj[()]
                    # bytes -> str
                    if isinstance(val, (bytes, np.bytes_)):
                        return val.decode("utf-8")
                    return val
                return obj
            for k in f.keys():
                out[k] = fetch(k)
        return out

def _maybe_cell_to_list(x):
    # Robustly convert MATLAB cell/string arrays -> Python list[str]
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        flat = np.array(x).ravel()
        out = []
        for v in flat:
            if isinstance(v, (bytes, np.bytes_)):
                out.append(v.decode("utf-8"))
            elif isinstance(v, str):
                out.append(v)
            else:
                try:
                    out.append(str(v))
                except Exception:
                    out.append(None)
        return out
    if isinstance(x, (bytes, np.bytes_)):
        return [x.decode("utf-8")]
    if isinstance(x, str):
        return [x]
    return None

# ---------- Main conversion ----------
def main(qm9_mat_path: str):
    data = _mat_as_dict(qm9_mat_path)

    # Heuristic key discovery
    # SMILES candidates
    smiles_key = next((k for k in data.keys() if k.lower() in {"smiles", "smile", "s"}), None)
    # HOMO/LUMO or gap
    homo_key = next((k for k in data.keys() if k.lower() in {"homo"}), None)
    lumo_key = next((k for k in data.keys() if k.lower() in {"lumo"}), None)
    gap_key  = next((k for k in data.keys() if k.lower() in {"gap", "homo_lumo_gap", "hlgap"}), None)
    # polarizability
    alpha_key = next((k for k in data.keys() if k.lower() in {"alpha", "polarizability"}), None)

    if smiles_key is None:
        raise RuntimeError("QM9 .mat file must contain a SMILES list (e.g., 'smiles'). Not found.")
    smiles_list = _maybe_cell_to_list(data[smiles_key])
    if smiles_list is None:
        raise RuntimeError("Could not parse SMILES array from the .mat file.")

    N = len(smiles_list)

    # Targets
    if gap_key is not None:
        gaps = np.array(data[gap_key]).astype(float).reshape(-1)
    else:
        if homo_key is None or lumo_key is None:
            raise RuntimeError("Need either 'gap' or both 'homo' and 'lumo' in the .mat.")
        homo = np.array(data[homo_key]).astype(float).reshape(-1)
        lumo = np.array(data[lumo_key]).astype(float).reshape(-1)
        if len(homo) != N or len(lumo) != N:
            raise RuntimeError("HOMO/LUMO length mismatch with SMILES.")
        gaps = lumo - homo

    if len(gaps) != N:
        raise RuntimeError("Gap vector length mismatch with SMILES.")

    # Polarizability (optional; if missing, we skip molecule)
    alphas = None
    if alpha_key is not None:
        alphas = np.array(data[alpha_key]).astype(float).reshape(-1)
        if len(alphas) != N:
            # if malformed, drop usage
            alphas = None

    out_c = open("carbon_only_dataset_enhanced_clean.json", "w")
    out_n = open("noncarbon_only_dataset_enhanced_clean.json", "w")
    out_all = open("dataset_enhanced_clean.json", "w")

    kept_carbon = kept_non = 0
    skipped = 0

    for i, smi in enumerate(smiles_list):
        if smi is None or len(str(smi).strip()) == 0:
            skipped += 1
            continue

        gap = float(gaps[i])

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue

        # Geometry-dependent features
        try:
            sa, vol = add_conformer_and_geometry(Chem.Mol(mol))
        except Exception:
            skipped += 1
            continue

        # Polarizability: prefer QM9 alpha; if absent, skip (to keep parity with your pipeline)
        pol = None
        if alphas is not None and np.isfinite(alphas[i]):
            pol = float(alphas[i])

        # Compute your “clean” descriptor set
        desc = compute_clean_descriptors(mol)

        if not sane_values(pol, sa, vol, desc["mw"]):
            skipped += 1
            continue

        bmat = bond_step_matrix(mol)
        smiles_canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        entry_17 = [
            bmat.tolist(), smiles_canon,
            desc["db_nonaro"], desc["mw"], desc["h_acceptors"],
            pol if pol is not None else float('nan'),  # we already QC’d finiteness above
            sa, vol,
            desc["conductance"], desc["num_aromatic_rings"], desc["num_rings"],
            desc["num_heteroatoms"], desc["num_sp2_atoms"], desc["sp2_fraction"],
            desc["abs_formal_charge"], desc["num_radical_electrons"],
            gap
        ]

        # final NaN/Inf guard (except matrix/smiles):
        flat = np.array([x for j, x in enumerate(entry_17) if j not in (0, 1)], dtype=float)
        if not np.all(np.isfinite(flat)):
            skipped += 1
            continue

        has_carbon = int(any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()))
        if has_carbon:
            json.dump(entry_17, out_c); out_c.write("\n")
            kept_carbon += 1
        else:
            json.dump(entry_17, out_n); out_n.write("\n")
            kept_non += 1

        json.dump(entry_17 + [has_carbon], out_all); out_all.write("\n")

        if (i + 1) % 500 == 0:
            print(f"... {i+1}/{N} processed (C={kept_carbon}, NC={kept_non}, skipped={skipped})")
        time.sleep(0.005)

    out_c.close(); out_n.close(); out_all.close()

    print("\n=== QM9 → JSONL DONE ===")
    print(f"Kept carbon:    {kept_carbon}")
    print(f"Kept noncarbon: {kept_non}")
    print(f"Skipped:        {skipped}")
    print("Wrote:")
    print("  carbon_only_dataset_enhanced_clean.json")
    print("  noncarbon_only_dataset_enhanced_clean.json")
    print("  dataset_enhanced_clean.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python qm9_to_enhanced_clean.py <path_to_qm9.mat>")
        sys.exit(1)
    main(sys.argv[1])
