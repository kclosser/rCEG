#!/usr/bin/env python3
"""
pqr_gap_model_morgan.py

Next-step HOMO-LUMO gap model after diagnostics showed:
- entry[4] exactly equals LUMO - HOMO, so labels are internally consistent.
- HOMO/LUMO leakage test gets ~0.036 eV MAE, proving the target is learnable if the right info is present.
- PQR + current LASSO descriptors plateau around ~0.96-1.05 eV, so the descriptor representation is missing structural information.

This script adds structural information directly from SMILES:
1. Morgan fingerprints, ECFP-like circular fingerprints.
2. RDKit physicochemical descriptors.
3. Original PQR/LASSO descriptors.
4. Mazouin-style chemical class experts.

Usage:
    cd ~/Downloads/Closser

    # recommended: use strict-cleaned dataset from the diagnostic script
    python pqr_gap_model_morgan.py \
      --data enhanced_dataset_lasso_STRICT.jsonl \
      --outdir runs/pqr_gap_morgan

    # quick test on 30k rows
    python pqr_gap_model_morgan.py \
      --data enhanced_dataset_lasso_STRICT.jsonl \
      --outdir runs/pqr_gap_morgan_quick \
      --max-rows 30000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.DisableLog("rdApp.*")
    RDKIT = True
except Exception:
    RDKIT = False


def load_json_or_jsonl(path: str | Path) -> List[Any]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            if len(obj) >= 5 and isinstance(obj[1], str):
                return [obj]
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                    out.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], list):
                    out.extend(obj)
            except Exception as e:
                print(f"Skipping line {line_no}: {e}")
    return out


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "nitrile": "C#N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "aromatic": "a",
}


def has_smarts(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return int(patt is not None and mol.HasSubstructMatch(patt))


def chem_class(mol):
    flags = {k: has_smarts(mol, v) for k, v in SMARTS.items()}
    if not any(flags.values()):
        return "saturated"
    if flags["aromatic"] or flags["carbonyl"] or flags["amide"] or flags["ester"]:
        return "aromatic_or_carbonyl"
    if flags["alkene"] or flags["alkyne"] or flags["nitrile"]:
        return "single_unsaturated"
    return "other_unsaturated"


def rdkit_features(mol):
    """Compact, interpretable RDKit descriptors. No HOMO/LUMO leakage."""
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
        rdMolDescriptors.CalcNumAtoms(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        Crippen.MolMR(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHeteroatoms(mol),
        Lipinski.FractionCSP3(mol),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        sum(1 for b in mol.GetBonds() if b.GetIsAromatic()),
        sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2),
        sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 3),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17),
    ]


def morgan_bits(mol, generator, n_bits):
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def build_dataset(entries, n_bits=2048, max_rows=None, seed=42):
    if not RDKIT:
        raise RuntimeError("RDKit is required for this script.")

    rng = np.random.default_rng(seed)
    if max_rows is not None and len(entries) > max_rows:
        idx = rng.choice(len(entries), size=max_rows, replace=False)
        entries = [entries[i] for i in idx]
        print(f"Subsampled to {len(entries):,} rows")

    gen2 = GetMorganGenerator(radius=2, fpSize=n_bits)
    gen3 = GetMorganGenerator(radius=3, fpSize=n_bits)

    rows = []
    X_parts = []
    drops = Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drops["bad_entry_shape"] += 1
            continue

        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < 0.10 or gap > 25:
            drops["bad_gap"] += 1
            continue

        mol = Chem.MolFromSmiles(e[1], sanitize=True)
        if mol is None:
            drops["invalid_smiles"] += 1
            continue
        if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms()):
            drops["isolated_hydrogen"] += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            drops["multi_fragment"] += 1
            continue

        canon = Chem.MolToSmiles(mol, canonical=True)
        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []

        # IMPORTANT: only first 5 PQR descriptors. Do NOT use HOMO/LUMO columns.
        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso = [fnum(v) for v in desc]

        fp2 = morgan_bits(mol, gen2, n_bits)
        fp3 = morgan_bits(mol, gen3, n_bits)
        rdk = np.array(rdkit_features(mol), dtype=np.float32)

        # Use Morgan radius 2 and 3 because HOMO-LUMO gap can depend on local conjugation
        # and slightly wider electronic environments.
        x = np.concatenate([
            np.array(pqr5, dtype=np.float32),
            np.array(lasso, dtype=np.float32),
            rdk,
            fp2,
            fp3,
        ])

        rows.append({
            "idx": i,
            "smiles": canon,
            "gap": gap,
            "chem_class": chem_class(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
        X_parts.append(x)

    if not rows:
        raise RuntimeError("No usable molecules.")

    max_len = max(len(x) for x in X_parts)
    X = np.full((len(X_parts), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(X_parts):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    y = df["gap"].to_numpy(np.float32)
    groups = df["smiles"].to_numpy()

    print(f"Usable rows: {len(df):,}")
    print(f"Drops: {dict(drops)}")
    print("Class distribution:")
    print(df["chem_class"].value_counts().to_string())
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]:,} columns")

    return df, X, y, groups


def group_split(groups, y, test_size=0.15, val_size=0.15, seed=42):
    idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(gss1.split(idx, y, groups))
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_rel, val_rel = next(gss2.split(trainval, y[trainval], groups[trainval]))
    return trainval[tr_rel], trainval[val_rel], test


def make_model(kind, seed, k_best=None):
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=1000,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=80,
            random_state=seed,
            loss="absolute_error",
        )
    elif kind == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.30,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(kind)

    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
    ]
    # HGB slows down badly if we keep every fingerprint bit. KBest is supervised,
    # fit only inside training folds, so this is fine.
    if k_best is not None:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(name, y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"{name:35s} MAE={mae:.4f} eV   R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def fit_predict_models(df, X, y, train_idx, val_idx, test_idx, seed=42, min_class_n=1500):
    results = []
    base_models = {}

    configs = [
        ("hgb_k1500", "hgb", 1500),
        ("hgb_k3000", "hgb", 3000),
        ("extratrees", "extratrees", None),
        ("rf", "rf", None),
    ]

    P_val = []
    P_test = []
    pred_names = []

    for name, kind, kbest in configs:
        print(f"\nTraining global {name}...")
        model = make_model(kind, seed, k_best=kbest)
        model.fit(X[train_idx], y[train_idx])
        base_models[name] = model
        pv = model.predict(X[val_idx])
        pt = model.predict(X[test_idx])
        P_val.append(pv)
        P_test.append(pt)
        pred_names.append(name)
        results.append(evaluate("VAL " + name, y[val_idx], pv))
        results.append(evaluate("TEST " + name, y[test_idx], pt))

    # Class experts: only train HGB/extratrees where enough class-specific samples exist.
    experts = {}
    train_classes = df.iloc[train_idx]["chem_class"].to_numpy()
    all_classes = sorted(df["chem_class"].unique())

    for cls in all_classes:
        cls_train = train_idx[train_classes == cls]
        if len(cls_train) < min_class_n:
            print(f"\nSkipping class expert {cls}: only {len(cls_train):,} train rows")
            continue

        print(f"\nTraining class expert for {cls}: {len(cls_train):,} train rows")
        for kind, kbest in [("hgb", 1200), ("extratrees", None)]:
            name = f"expert_{cls}_{kind}"
            model = make_model(kind, seed + len(experts) + 10, k_best=kbest)
            model.fit(X[cls_train], y[cls_train])
            experts[name] = (cls, model)

            # fallback global best so non-class rows still have a prediction
            pv = base_models["hgb_k3000"].predict(X[val_idx])
            pt = base_models["hgb_k3000"].predict(X[test_idx])

            val_mask = df.iloc[val_idx]["chem_class"].to_numpy() == cls
            test_mask = df.iloc[test_idx]["chem_class"].to_numpy() == cls
            if val_mask.any():
                pv[val_mask] = model.predict(X[val_idx][val_mask])
            if test_mask.any():
                pt[test_mask] = model.predict(X[test_idx][test_mask])

            P_val.append(pv)
            P_test.append(pt)
            pred_names.append(name)
            results.append(evaluate("VAL " + name, y[val_idx], pv))
            results.append(evaluate("TEST " + name, y[test_idx], pt))

    P_val = np.vstack(P_val).T
    P_test = np.vstack(P_test).T

    print("\nTraining ridge stacker on validation predictions...")
    stacker = RidgeCV(alphas=np.logspace(-6, 3, 30))
    stacker.fit(P_val, y[val_idx])
    val_stack = stacker.predict(P_val)
    test_stack = stacker.predict(P_test)
    results.append(evaluate("VAL stacked", y[val_idx], val_stack))
    results.append(evaluate("TEST stacked", y[test_idx], test_stack))

    return results, base_models, experts, stacker, pred_names, test_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_gap_morgan")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-class-n", type=int, default=1500)
    args = ap.parse_args()

    if not RDKIT:
        raise RuntimeError("RDKit is required. Run this in your conda environment with RDKit installed.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PQR HOMO-LUMO gap model: PQR/LASSO + RDKit descriptors + Morgan fingerprints")
    print("=" * 80)

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded entries: {len(entries):,}")

    df, X, y, groups = build_dataset(entries, n_bits=args.n_bits, max_rows=args.max_rows, seed=args.seed)
    train_idx, val_idx, test_idx = group_split(groups, y, seed=args.seed)

    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Split target means: train={y[train_idx].mean():.3f}, val={y[val_idx].mean():.3f}, test={y[test_idx].mean():.3f}")

    results, base_models, experts, stacker, pred_names, test_pred = fit_predict_models(
        df, X, y, train_idx, val_idx, test_idx,
        seed=args.seed,
        min_class_n=args.min_class_n,
    )

    pd.DataFrame(results).to_csv(outdir / "metrics.csv", index=False)

    test_report = df.iloc[test_idx].copy()
    test_report["pred_gap"] = test_pred
    test_report["abs_error"] = np.abs(test_pred - y[test_idx])
    test_report.sort_values("abs_error", ascending=False).to_csv(outdir / "test_error_report.csv", index=False)

    test_report.groupby("chem_class").agg(
        n=("gap", "size"),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        mean_gap=("gap", "mean"),
    ).sort_values("mae", ascending=False).to_csv(outdir / "mae_by_class.csv")

    joblib.dump(
        {
            "base_models": base_models,
            "experts": experts,
            "stacker": stacker,
            "pred_names": pred_names,
            "n_bits": args.n_bits,
        },
        outdir / "pqr_gap_morgan_model.joblib",
    )

    print("\nSaved:")
    print(outdir / "metrics.csv")
    print(outdir / "mae_by_class.csv")
    print(outdir / "test_error_report.csv")
    print(outdir / "pqr_gap_morgan_model.joblib")
    print("\nIf this still cannot get below ~0.5 eV, the next real step is a graph neural network")
    print("or SchNet/D-MPNN-style model trained directly from molecular structure, not just descriptors.")


if __name__ == "__main__":
    main()
