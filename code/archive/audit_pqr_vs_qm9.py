#!/usr/bin/env python3
"""
audit_pqr_vs_qm9.py

Purpose:
Diagnose why PQR small-organic subsets still perform near ~1 eV instead of
matching sub-0.2 eV results reported in QM9-like studies.

This script checks:
1. Whether PQR contains a strict QM9-like subset.
2. Whether that subset is actually easier under non-leakage features.
3. Whether random split vs scaffold-like split changes MAE.
4. Whether duplicate/conflicting labels remain.
5. Whether target spread and molecule composition differ from expected QM9-like behavior.

Usage:
cd ~/Downloads/Closser
python audit_pqr_vs_qm9.py --data enhanced_dataset_lasso_STRICT.jsonl --outdir runs/pqr_qm9_audit
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ALLOWED_QM9_ATOMS = {1, 6, 7, 8, 9}  # H, C, N, O, F


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                rows.append(obj)
    return rows


def safe_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else smiles
    except Exception:
        return smiles


def rdkit_2d(mol):
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        Crippen.MolMR(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHeteroatoms(mol),
        Lipinski.FractionCSP3(mol),
        sum(a.GetIsAromatic() for a in atoms),
        sum(b.GetIsAromatic() for b in bonds),
        sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
        sum(b.GetBondTypeAsDouble() == 3 for b in bonds),
        sum(a.GetAtomicNum() == 6 for a in atoms),
        sum(a.GetAtomicNum() == 7 for a in atoms),
        sum(a.GetAtomicNum() == 8 for a in atoms),
        sum(a.GetAtomicNum() == 9 for a in atoms),
    ]


def build_table(entries):
    rows = []
    X = []
    drops = Counter()

    for i, e in enumerate(entries):
        smiles = e[1]
        gap = fnum(e[4])
        if not math.isfinite(gap) or not (0.1 < gap < 25):
            drops["bad_gap"] += 1
            continue

        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            drops["bad_smiles"] += 1
            continue

        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            multi_fragment = True
        else:
            multi_fragment = False

        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        heavy = mol.GetNumHeavyAtoms()
        has_c = any(a == 6 for a in atoms)
        allowed_atoms_only = all(a in ALLOWED_QM9_ATOMS for a in atoms)

        canon_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        mol_no_stereo = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_no_stereo)
        canon_no_stereo = Chem.MolToSmiles(mol_no_stereo, canonical=True, isomericSmiles=False)

        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []

        # Non-leakage: use only first 5 PQR values, not HOMO/LUMO.
        feats = []
        feats.extend([fnum(v) for v in pqr[:5]])
        feats.extend([fnum(v) for v in desc])
        feats.extend(rdkit_2d(mol))

        X.append(feats)

        rows.append({
            "idx": i,
            "smiles": smiles,
            "canon_iso": canon_iso,
            "canon_no_stereo": canon_no_stereo,
            "scaffold": safe_scaffold(canon_no_stereo),
            "gap": gap,
            "heavy_atoms": heavy,
            "num_atoms": mol.GetNumAtoms(),
            "allowed_qm9_atoms_only": allowed_atoms_only,
            "has_carbon": has_c,
            "formal_charge": formal_charge,
            "radicals": radicals,
            "multi_fragment": multi_fragment,
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })

    max_len = max(len(x) for x in X)
    Xmat = np.full((len(X), max_len), np.nan, dtype=np.float32)

    for i, x in enumerate(X):
        arr = np.array(x, dtype=np.float64)
        arr[~np.isfinite(arr)] = np.nan
        arr = np.clip(arr, -1e6, 1e6)
        Xmat[i, :len(arr)] = arr.astype(np.float32)

    df = pd.DataFrame(rows)
    print("Drops:", dict(drops))
    print("Rows:", len(df))
    return df, Xmat


def add_subset_flags(df):
    df = df.copy()

    # Strictest QM9-like version.
    df["qm9_like_strict"] = (
        (df["allowed_qm9_atoms_only"])
        & (df["has_carbon"])
        & (~df["multi_fragment"])
        & (df["formal_charge"] == 0)
        & (df["radicals"] == 0)
        & (df["heavy_atoms"] <= 9)
    )

    # Slightly looser small-organic version.
    df["small_organic_loose"] = (
        (df["allowed_qm9_atoms_only"])
        & (df["has_carbon"])
        & (~df["multi_fragment"])
        & (df["formal_charge"] == 0)
        & (df["heavy_atoms"] <= 15)
    )

    # Organic but not necessarily QM9-sized.
    df["organic_allowed_atoms"] = (
        (df["allowed_qm9_atoms_only"])
        & (df["has_carbon"])
        & (~df["multi_fragment"])
        & (df["formal_charge"] == 0)
    )

    return df


def duplicate_report(df, key, name):
    g = df.groupby(key)["gap"].agg(["count", "mean", "std", "min", "max"])
    g["spread"] = g["max"] - g["min"]
    out = {
        "name": name,
        "unique_molecules": len(g),
        "duplicate_groups": int((g["count"] > 1).sum()),
        "conflict_gt_0p05": int(((g["count"] > 1) & (g["spread"] > 0.05)).sum()),
        "conflict_gt_0p10": int(((g["count"] > 1) & (g["spread"] > 0.10)).sum()),
        "max_spread": float(g["spread"].max()),
    }
    return out, g.sort_values("spread", ascending=False)


def make_model(kind="et"):
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            early_stopping=True,
            random_state=42,
            loss="absolute_error",
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.35,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", RobustScaler()),
        ("model", model),
    ])


def eval_subset(df, X, subset_name, split_type, model_kind, outdir):
    mask = df[subset_name].to_numpy()
    n = int(mask.sum())
    if n < 1000:
        return {
            "subset": subset_name,
            "split": split_type,
            "model": model_kind,
            "n": n,
            "status": "too_few_rows",
        }

    sub = df[mask].reset_index(drop=True)
    Xs = X[mask]
    y = sub["gap"].to_numpy(np.float32)

    # Collapse exact stereo-preserved duplicates by group split, not by dropping, to avoid leakage.
    groups = sub["canon_iso"].to_numpy()

    if split_type == "random":
        tr, te = train_test_split(np.arange(n), test_size=0.20, random_state=42)
    elif split_type == "molecule_group":
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(Xs, y, groups))
    elif split_type == "scaffold_group":
        scaffolds = sub["scaffold"].to_numpy()
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(Xs, y, scaffolds))
    else:
        raise ValueError(split_type)

    model = make_model(model_kind)
    model.fit(Xs[tr], y[tr])
    pred = model.predict(Xs[te])

    mae = mean_absolute_error(y[te], pred)
    r2 = r2_score(y[te], pred)

    return {
        "subset": subset_name,
        "split": split_type,
        "model": model_kind,
        "n": n,
        "train_n": len(tr),
        "test_n": len(te),
        "target_mean": float(y.mean()),
        "target_sd": float(y.std()),
        "test_mean": float(y[te].mean()),
        "test_sd": float(y[te].std()),
        "mae": float(mae),
        "r2": float(r2),
        "status": "ok",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_qm9_audit")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl(args.data)
    df, X = build_table(entries)
    df = add_subset_flags(df)

    # Composition summary.
    comp_rows = []
    for subset in ["qm9_like_strict", "small_organic_loose", "organic_allowed_atoms"]:
        m = df[subset]
        d = df[m]
        comp_rows.append({
            "subset": subset,
            "n": int(m.sum()),
            "pct_total": float(m.mean()),
            "gap_mean": float(d["gap"].mean()) if len(d) else np.nan,
            "gap_sd": float(d["gap"].std()) if len(d) else np.nan,
            "gap_min": float(d["gap"].min()) if len(d) else np.nan,
            "gap_max": float(d["gap"].max()) if len(d) else np.nan,
            "heavy_mean": float(d["heavy_atoms"].mean()) if len(d) else np.nan,
            "heavy_max": float(d["heavy_atoms"].max()) if len(d) else np.nan,
        })

    pd.DataFrame(comp_rows).to_csv(outdir / "subset_composition.csv", index=False)
    print("\nSubset composition:")
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # Atom composition.
    atom_summary = {
        "total_rows": len(df),
        "allowed_qm9_atoms_only": int(df["allowed_qm9_atoms_only"].sum()),
        "has_carbon": int(df["has_carbon"].sum()),
        "multi_fragment": int(df["multi_fragment"].sum()),
        "charged": int((df["formal_charge"] != 0).sum()),
        "radicals": int((df["radicals"] != 0).sum()),
        "heavy_atoms_gt_9": int((df["heavy_atoms"] > 9).sum()),
        "heavy_atoms_gt_15": int((df["heavy_atoms"] > 15).sum()),
    }
    pd.DataFrame([atom_summary]).to_csv(outdir / "global_composition.csv", index=False)
    print("\nGlobal composition:")
    print(pd.DataFrame([atom_summary]).to_string(index=False))

    # Duplicate/conflict reports.
    dup_rows = []
    for key, name in [
        ("canon_iso", "stereo_preserved"),
        ("canon_no_stereo", "stereo_removed"),
    ]:
        row, detail = duplicate_report(df, key, name)
        dup_rows.append(row)
        detail.head(200).to_csv(outdir / f"top_duplicate_conflicts_{name}.csv")

    pd.DataFrame(dup_rows).to_csv(outdir / "duplicate_conflict_summary.csv", index=False)
    print("\nDuplicate/conflict summary:")
    print(pd.DataFrame(dup_rows).to_string(index=False))

    # Export strict subset for Chemprop.
    qm9 = df[df["qm9_like_strict"]].copy()
    qm9[["canon_iso", "gap"]].rename(columns={"canon_iso": "smiles"}).drop_duplicates("smiles").to_csv(
        outdir / "pqr_qm9_like_strict.csv", index=False
    )

    # Model audit.
    results = []
    for subset in ["qm9_like_strict", "small_organic_loose", "organic_allowed_atoms"]:
        for split_type in ["random", "molecule_group", "scaffold_group"]:
            for model_kind in ["et", "hgb"]:
                print(f"\nRunning {subset} | {split_type} | {model_kind}")
                row = eval_subset(df, X, subset, split_type, model_kind, outdir)
                print(row)
                results.append(row)

    res = pd.DataFrame(results)
    res.to_csv(outdir / "subset_model_results.csv", index=False)

    print("\nFinal model audit:")
    print(res.to_string(index=False))
    print(f"\nWrote audit files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
