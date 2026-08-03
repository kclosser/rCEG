#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:42:47 2025

@author: isaacwang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import numpy as np
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from typing import Dict, Any

# ---------- PubChem ----------
url_template = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/record/SDF"

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# ---------- Graph features ----------
def get_bond_step_matrix(mol: Chem.Mol) -> np.ndarray:
    """
    Weighted bond-step matrix (Schnet-ish): single=1.0, double=0.5, aromatic=0.75, triple=0.33
    Floyd–Warshall for shortest path. Unconnected -> -1.
    """
    n = mol.GetNumAtoms()
    mat = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(mat, 0.0)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        w = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 0.5,
            Chem.BondType.AROMATIC: 0.75,
            Chem.BondType.TRIPLE: 0.33
        }.get(b.GetBondType(), 1.0)
        mat[i, j] = mat[j, i] = w
    # shortest paths
    for k in range(n):
        dik = mat[:, k][:, None]
        mkj = mat[k, :][None, :]
        mat = np.minimum(mat, dik + mkj)
    mat[np.isinf(mat)] = -1.0
    return mat

def get_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

# ---------- External PM7 props ----------
def extract_molecular_properties(data: Any, cid: int) -> Dict[str, float]:
    """
    Pull PM7 polarizability / surface area / volume from your JSON blob.
    """
    for item in data:
        if item.get("pubchem_cid") == cid:
            pm7 = item.get("pm7", {})
            pol = pm7.get("polarizability")
            sa = pm7.get("surfaceArea")
            vol = pm7.get("volume")
            if all(x is not None for x in (pol, sa, vol)):
                return {"polarizability": float(pol), "surface_area": float(sa), "volume": float(vol)}
    raise ValueError(f"No matching PM7 data for CID {cid}")

# ---------- Descriptor computation (CLEAN) ----------
def compute_clean_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Produce a de-duplicated, physically sensible descriptor set:
      1) db_nonaro            (count of non-aromatic double bonds)
      2) mw                   (MolWt)
      3) h_acceptors          (NumHAcceptors)
      4) polarizability       (PM7; must be >=0)
      5) surface_area         (PM7; >0)
      6) volume               (PM7; >0)
      7) conductance          (proxy retained from your original: non-aromatic double + aromatic ring count)
      8) num_aromatic_rings
      9) num_rings
     10) num_heteroatoms
     11) num_sp2_atoms
     12) sp2_fraction         (= num_sp2_atoms / num_atoms)
     13) tpsa
     14) abs_formal_charge    (ions)
     15) num_radical_electrons (radicals)
    """
    n_atoms = mol.GetNumAtoms() or 1

    # bonds
    db_nonaro = 0
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic():
            db_nonaro += 1

    # ring/aromaticity
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_rings = rdMolDescriptors.CalcNumRings(mol)

    # “conductance” proxy as before, but now excludes aromatic double bonds counted above
    conductance = num_aromatic_rings + db_nonaro

    # atoms & hybridization
    num_sp2_atoms = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    sp2_fraction = float(num_sp2_atoms) / float(n_atoms)

    num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)

    # charges & radicals
    abs_formal_charge = abs(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    num_radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

    return {
        "db_nonaro": db_nonaro,
        "mw": mw,
        "h_acceptors": h_acceptors,
        "conductance": conductance,
        "num_aromatic_rings": num_aromatic_rings,
        "num_rings": num_rings,
        "num_heteroatoms": num_heteroatoms,
        "num_sp2_atoms": num_sp2_atoms,
        "sp2_fraction": sp2_fraction,
        "tpsa": tpsa,
        "abs_formal_charge": abs_formal_charge,
        "num_radical_electrons": num_radical_electrons,
    }

# ---------- QC ----------
def sane_pm7(polarizability: float, surface_area: float, volume: float) -> bool:
    return (
        polarizability is not None and surface_area is not None and volume is not None and
        np.isfinite(polarizability) and np.isfinite(surface_area) and np.isfinite(volume) and
        polarizability >= 0.0 and surface_area > 0.0 and volume > 0.0
    )

def sane_core(mw: float, tpsa: float) -> bool:
    return (
        np.isfinite(mw) and mw > 10.0 and
        np.isfinite(tpsa) and tpsa >= 0.0
    )

# ---------- Single-CID reader ----------
def read_molecule_data_clean(cid: int, json_data: Any) -> Dict[str, Any]:
    try:
        data_url = url_template.format(cid)
        r = session.get(data_url, timeout=15)
        r.raise_for_status()
        mol = Chem.MolFromMolBlock(r.text)
        if mol is None:
            raise RuntimeError(f"Cannot parse SDF for CID {cid}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"SDF fetch failed for CID {cid}: {e}")

    # Keep ions/radicals; just sanitize
    mol = Chem.AddHs(mol, addCoords=True)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        Chem.SanitizeMol(mol)
    except Exception:
        # If sanitize fails (often due to valence), try without embed
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e:
            raise RuntimeError(f"Sanitize failed CID {cid}: {e}")

    pm7 = extract_molecular_properties(json_data, cid)
    if not sane_pm7(pm7["polarizability"], pm7["surface_area"], pm7["volume"]):
        raise RuntimeError(f"PM7 QC failed (pol/sa/vol) for CID {cid}: {pm7}")

    desc = compute_clean_descriptors(mol)
    if not sane_core(desc["mw"], desc["tpsa"]):
        raise RuntimeError(f"Core QC failed (MW/TPSA) for CID {cid}: mw={desc['mw']} tpsa={desc['tpsa']}")

    bond_matrix = get_bond_step_matrix(mol)
    smiles = get_smiles(mol)

    return {
        "bond_step_matrix": bond_matrix.tolist(),
        "smiles": smiles,
        # PM7
        "polarizability": float(pm7["polarizability"]),
        "surface_area": float(pm7["surface_area"]),
        "volume": float(pm7["volume"]),
        # Clean descriptors
        **desc,
    }

# ---------- HOMO–LUMO gaps ----------
def load_homo_lumo_gaps(filename: str):
    with open(filename, "r") as f:
        content = f.read()
        return eval(content)  # you already use this format: [(cid, gap), ...]

# ---------- Main ----------
def main():
    # Inputs
    pm7_json = Path("Pitt_Quantum_Repository_Data.json")
    gaps_txt = Path("cid_homo_lumo_gap.txt")
    out_path = Path("carbon_only_dataset_enhanced_clean.json")  # <— change if needed

    with pm7_json.open("r") as f:
        json_data = json.load(f)
    homo_lumo_gaps = load_homo_lumo_gaps(str(gaps_txt))

    # We will write JSONL with this **cleaned** 18-element layout:
    # [bond_step_matrix, smiles,
    #  db_nonaro, mw, h_acceptors, polarizability, surface_area, volume,
    #  conductance, num_aromatic_rings, num_rings, num_heteroatoms,
    #  num_sp2_atoms, sp2_fraction, tpsa, abs_formal_charge, num_radical_electrons,
    #  gap]
    kept = skipped_no_carbon = skipped_qc = skipped_err = 0

    with out_path.open("w") as fout:
        total = len(homo_lumo_gaps)
        for idx, (cid, gap) in enumerate(homo_lumo_gaps, start=1):
            try:
                props = read_molecule_data_clean(cid, json_data)
                mol = Chem.MolFromSmiles(props["smiles"])
                if mol is None:
                    raise RuntimeError("Invalid SMILES after clean read.")
                # Keep only carbon-containing molecules for this file
                if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
                    skipped_no_carbon += 1
                    continue

                entry = [
                    props["bond_step_matrix"], props["smiles"],
                    props["db_nonaro"], props["mw"], props["h_acceptors"],
                    props["polarizability"], props["surface_area"], props["volume"],
                    props["conductance"], props["num_aromatic_rings"], props["num_rings"],
                    props["num_heteroatoms"], props["num_sp2_atoms"], props["sp2_fraction"],
                    props["tpsa"], props["abs_formal_charge"], props["num_radical_electrons"],
                    float(gap),
                ]

                # Final NaN/Inf guard
                flat = np.array([x for i, x in enumerate(entry) if i not in (0, 1)], dtype=float)
                if not np.all(np.isfinite(flat)):
                    skipped_qc += 1
                    continue

                json.dump(entry, fout)
                fout.write("\n")
                kept += 1

            except RuntimeError as e:
                skipped_qc += 1
                # print(f"[QC] CID {cid}: {e}")
            except Exception as e:
                skipped_err += 1
                # print(f"[ERR] CID {cid}: {e}")

            if idx % 50 == 0:
                print(f"... {idx}/{total} processed (kept={kept}, qc={skipped_qc}, nocarbon={skipped_no_carbon}, err={skipped_err})")
            time.sleep(0.08)  # be gentle to PubChem

    print("\n=== Done ===")
    print(f"Kept: {kept}")
    print(f"Skipped (no carbon): {skipped_no_carbon}")
    print(f"Skipped (QC): {skipped_qc}")
    print(f"Skipped (errors): {skipped_err}")
    print(f"Wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
