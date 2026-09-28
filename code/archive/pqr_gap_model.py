#!/usr/bin/env python3
"""
pqr_gap_model.py

Fast, defensible HOMO-LUMO gap predictor for a PQR-style dataset.

Why this version is different from the older "three-paper ensemble":
1. It does NOT train a separate BERT + SchNet + KRR for every tiny subclass.
   That is extremely slow on CPU and creates many underpowered experts.
2. It uses the Mazouin idea faithfully: split molecules by chemically meaningful
   structural classes before fitting expert models.
3. It uses the Ye idea faithfully for the descriptor baseline: nonlinear models
   on selected knowledge-based descriptors, with direct gap prediction.
4. It only uses Su/PorphyBERT as an optional future extension, because their
   result depends on real pretraining + fine-tuning. A randomly initialized
   char-transformer is not PorphyBERT and will usually hurt.

Expected usage:
    python pqr_gap_model.py --data enhanced_dataset_lasso.json --outdir runs/pqr_gap

Input accepted:
    JSON list:       [entry, entry, ...]
    JSON-lines file: one entry per line

Expected entry format:
    entry[0] = anything / None
    entry[1] = SMILES
    entry[2] = [mol_weight, exact_mass, dipole_moment, heat_formation, polarizability, HOMO, LUMO]
    entry[3] = list of LASSO/PaDEL descriptors
    entry[4] = HOMO-LUMO gap target in eV
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


# -----------------------------
# Loading
# -----------------------------

def load_json_or_jsonl(path: str | Path) -> List[Any]:
    """
    Loads either:
    1. normal JSON list:
         [entry, entry, entry]
    2. JSON-lines:
         entry
         entry
         entry

    Your enhanced_dataset_lasso.json appears to be JSON-lines, where each
    molecule is stored as one JSON array per line.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    data: List[Any] = []

    # First try normal JSON.
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    # Then try JSON-lines.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip().rstrip(",")
            if not line:
                continue

            try:
                obj = json.loads(line)

                # One molecule entry:
                # [None, smiles, pqr_list, descriptor_list, gap]
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                    data.append(obj)

                # A line may contain a list of entries.
                elif isinstance(obj, list) and obj and isinstance(obj[0], list):
                    data.extend(obj)

                else:
                    data.append(obj)

            except json.JSONDecodeError as e:
                print(f"Skipping line {line_no}: could not parse JSON ({e})")

    if not data:
        raise ValueError(f"Could not parse {path}. No JSON-lines entries were found.")

    return data


# -----------------------------
# Chemistry features/classification
# -----------------------------

SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "nitrile": "C#N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "aromatic": "a",
    "hetero_aromatic": "[a;!#6]",
}


def _has_smarts(mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    return bool(patt is not None and mol.HasSubstructMatch(patt))


def classify_molecule(smiles: str) -> Dict[str, Any]:
    """
    Mazouin-style chemical routing:
    - saturated: no aromatic/double/triple bonds
    - single_unsaturated: simple C=C/C#C without aromatic or carbonyl dominance
    - aromatic_or_carbonyl: aromatic rings and/or carbonyl-like groups
    - other_unsaturated: fallback for mixed hetero/charged cases

    This is intentionally based on structure, not the target value, to avoid leakage.
    """
    out = {
        "chem_class": "unknown",
        "aromatic": 0,
        "carbonyl": 0,
        "amide": 0,
        "ester": 0,
        "nitrile": 0,
        "alkene": 0,
        "alkyne": 0,
        "hetero_atoms": 0,
        "rings": 0,
        "heavy_atoms": 0,
        "pi_atoms": 0,
        "valid_smiles": 0,
    }

    if not RDKIT_AVAILABLE:
        # Fallback, less accurate but still usable.
        s = str(smiles)
        aromatic = int(any(c in s for c in ["c", "n", "o", "s"]))
        carbonyl = int("=O" in s)
        unsat = int("=" in s or "#" in s or aromatic)
        out["aromatic"] = aromatic
        out["carbonyl"] = carbonyl
        out["chem_class"] = (
            "aromatic_or_carbonyl" if aromatic or carbonyl else
            "single_unsaturated" if unsat else
            "saturated"
        )
        return out

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return out

    out["valid_smiles"] = 1
    out["heavy_atoms"] = mol.GetNumHeavyAtoms()
    out["rings"] = rdMolDescriptors.CalcNumRings(mol)
    out["hetero_atoms"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))

    flags = {k: int(_has_smarts(mol, v)) for k, v in SMARTS.items()}
    out.update(flags)

    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    double_triple_bonds = 0
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            continue
        if b.GetBondTypeAsDouble() >= 2:
            double_triple_bonds += 1
    out["pi_atoms"] = aromatic_atoms + 2 * double_triple_bonds

    has_aromatic = out["aromatic"] == 1
    has_carbonyl_family = any(out[x] for x in ["carbonyl", "amide", "ester"])
    has_simple_unsat = any(out[x] for x in ["alkene", "alkyne", "nitrile"])

    if not has_aromatic and not has_carbonyl_family and not has_simple_unsat:
        out["chem_class"] = "saturated"
    elif has_aromatic or has_carbonyl_family:
        out["chem_class"] = "aromatic_or_carbonyl"
    elif has_simple_unsat:
        out["chem_class"] = "single_unsaturated"
    else:
        out["chem_class"] = "other_unsaturated"

    return out


# -----------------------------
# Dataset construction
# -----------------------------

@dataclass
class DatasetBundle:
    df: pd.DataFrame
    X_desc: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def build_dataset(entries: List[Any], drop_zero_gaps: bool = True) -> DatasetBundle:
    rows = []
    descs = []
    n_bad = Counter()

    for i, entry in enumerate(entries):
        try:
            smiles = str(entry[1])
            pqr_raw = list(entry[2]) if entry[2] is not None else []
            lasso = list(entry[3]) if entry[3] is not None else []
            gap = float(entry[4])

            if not math.isfinite(gap):
                n_bad["nonfinite_gap"] += 1
                continue

            # Many molecular datasets use 0 as failed/missing quantum output.
            # These zero-gap points create the horizontal line you showed and will
            # destroy MAE. Keep them only if you know they are real.
            if drop_zero_gaps and gap <= 0.05:
                n_bad["zero_or_placeholder_gap"] += 1
                continue

            if gap < 0 or gap > 25:
                n_bad["unphysical_gap"] += 1
                continue

            pqr = np.array(pqr_raw[:5], dtype=float)
            if pqr.size < 5:
                pqr = np.pad(pqr, (0, 5 - pqr.size), constant_values=np.nan)

            lasso_arr = np.array(lasso, dtype=float)
            desc = np.concatenate([pqr, lasso_arr])
            desc = np.nan_to_num(desc, nan=np.nan, posinf=np.nan, neginf=np.nan)

            chem = classify_molecule(smiles)
            row = {
                "idx": i,
                "smiles": smiles,
                "gap": gap,
                "mol_weight": pqr[0],
                "exact_mass": pqr[1],
                "dipole_moment": pqr[2],
                "heat_formation": pqr[3],
                "polarizability": pqr[4],
                **chem,
            }
            rows.append(row)
            descs.append(desc)

        except Exception as e:
            n_bad[type(e).__name__] += 1

    if not rows:
        raise RuntimeError("No usable entries were parsed.")

    # Ragged descriptor arrays are padded with NaN.
    max_len = max(len(x) for x in descs)
    X = np.full((len(descs), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(descs):
        X[i, : len(x)] = x

    df = pd.DataFrame(rows)

    # Clip extreme numeric descriptors based only on robust global limits.
    # This is not target leakage. It prevents heat_formation/polarizability outliers
    # like the ones in your plot from dominating tree splits.
    for col in ["mol_weight", "exact_mass", "dipole_moment", "heat_formation", "polarizability"]:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        lo, hi = np.nanpercentile(vals, [0.1, 99.9])
        df[col] = vals.clip(lo, hi)

    # For descriptors too, clip to reduce single failed descriptors controlling the model.
    lo = np.nanpercentile(X, 0.1, axis=0)
    hi = np.nanpercentile(X, 99.9, axis=0)
    X = np.where(X < lo, lo, X)
    X = np.where(X > hi, hi, X)

    y = df["gap"].to_numpy(dtype=np.float32)

    # Group by canonical smiles when possible so duplicates do not leak across splits.
    if RDKIT_AVAILABLE:
        canon = []
        for s in df["smiles"]:
            mol = Chem.MolFromSmiles(s)
            canon.append(Chem.MolToSmiles(mol) if mol is not None else s)
        groups = np.array(canon)
    else:
        groups = df["smiles"].to_numpy()

    feature_names = [f"desc_{i}" for i in range(X.shape[1])]
    print(f"Parsed usable molecules: {len(df):,}")
    print(f"Dropped counts: {dict(n_bad)}")
    print("Class distribution:")
    print(df["chem_class"].value_counts(dropna=False).to_string())
    print(f"Target gap: mean={y.mean():.3f}, sd={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}")

    return DatasetBundle(df=df, X_desc=X, y=y, groups=groups, feature_names=feature_names)


# -----------------------------
# Splitting
# -----------------------------

def group_train_val_test_split(groups: np.ndarray, y: np.ndarray, test_size=0.15, val_size=0.15, seed=42):
    n = len(y)
    idx = np.arange(n)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(idx, y, groups=groups))

    relative_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    tv_groups = groups[trainval_idx]
    tr_rel, val_rel = next(gss2.split(trainval_idx, y[trainval_idx], groups=tv_groups))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[val_rel]
    return train_idx, val_idx, test_idx


# -----------------------------
# Model definitions
# -----------------------------

def make_descriptor_pipeline(kind: str, seed: int = 42):
    """
    Descriptor models, close to Ye's descriptor/RF logic but with stronger
    modern sklearn defaults and robust preprocessing.
    """
    if kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=700,
            max_features=0.45,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
            bootstrap=False,
        )
    elif kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
            bootstrap=True,
        )
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=1200,
            learning_rate=0.035,
            l2_regularization=0.03,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=60,
            random_state=seed,
            loss="absolute_error",
        )
    elif kind == "elastic":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-5, 1, 20),
            cv=5,
            max_iter=20000,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(kind)

    # VarianceThreshold removes constant descriptors. SelectFromModel is only
    # applied for the linear model, not trees, because trees can handle many
    # descriptors and selection can accidentally discard class-specific signals.
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
    ]

    if kind == "elastic":
        steps += [("scale", RobustScaler()), ("model", model)]
    else:
        steps += [("model", model)]

    return Pipeline(steps)


def fit_global_models(X_train, y_train, seed=42) -> Dict[str, Any]:
    models = {}
    for kind in ["extra_trees", "random_forest", "hgb", "elastic"]:
        print(f"Training global {kind}...")
        pipe = make_descriptor_pipeline(kind, seed=seed)
        pipe.fit(X_train, y_train)
        models[f"global_{kind}"] = pipe
    return models


def fit_class_experts(df, X, y, train_idx, min_class_n=800, seed=42) -> Dict[str, Dict[str, Any]]:
    """
    Train selected-learning experts by class, but only where enough data exists.
    Tiny classes fall back to global models. This prevents your old issue where
    experts had 4, 7, 22, or 110 samples.
    """
    experts: Dict[str, Dict[str, Any]] = {}
    train_df = df.iloc[train_idx].copy()

    for cls, n in train_df["chem_class"].value_counts().items():
        if n < min_class_n:
            print(f"Skipping expert {cls}: only {n} train samples")
            continue

        cls_train_idx = train_idx[df.iloc[train_idx]["chem_class"].to_numpy() == cls]
        Xc, yc = X[cls_train_idx], y[cls_train_idx]
        print(f"Training class expert for {cls}: {len(cls_train_idx):,} samples")

        models = {}
        # Fewer models per expert keeps it fast while still capturing nonlinear structure.
        for kind in ["extra_trees", "hgb"]:
            pipe = make_descriptor_pipeline(kind, seed=seed + len(experts))
            pipe.fit(Xc, yc)
            models[kind] = pipe

        experts[cls] = models

    return experts


def predict_matrix(models: Dict[str, Any], experts: Dict[str, Dict[str, Any]], df: pd.DataFrame, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Returns matrix of base predictions:
    - global model predictions for every row
    - class expert predictions where available; otherwise global_extra_trees fallback
    """
    cols = []
    preds = []

    for name, model in models.items():
        p = model.predict(X)
        preds.append(p)
        cols.append(name)

    fallback = models["global_extra_trees"]

    for expert_name in sorted(experts.keys()):
        for model_name, model in experts[expert_name].items():
            p = fallback.predict(X)
            mask = df["chem_class"].to_numpy() == expert_name
            if mask.any():
                p[mask] = model.predict(X[mask])
            preds.append(p)
            cols.append(f"expert_{expert_name}_{model_name}")

    return np.vstack(preds).T, cols


def fit_stacker(P_val: np.ndarray, y_val: np.ndarray) -> RidgeCV:
    # RidgeCV avoids overfitting the base predictions.
    alphas = np.logspace(-6, 3, 30)
    stacker = RidgeCV(alphas=alphas, fit_intercept=True)
    stacker.fit(P_val, y_val)
    return stacker


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:28s} MAE={mae:.4f} eV   R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


# -----------------------------
# Diagnostics
# -----------------------------

def save_error_report(outdir: Path, df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    rep = df_test.copy()
    rep["pred_gap"] = y_pred
    rep["abs_error"] = np.abs(y_pred - y_true)
    rep = rep.sort_values("abs_error", ascending=False)
    rep.to_csv(outdir / "test_error_report.csv", index=False)

    by_class = rep.groupby("chem_class").agg(
        n=("gap", "size"),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        mean_gap=("gap", "mean"),
    ).sort_values("mae", ascending=False)
    by_class.to_csv(outdir / "mae_by_class.csv")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to enhanced_dataset_lasso.json/jsonl")
    parser.add_argument("--outdir", default="pqr_gap_run", help="Output directory")
    parser.add_argument("--keep-zero-gaps", action="store_true", help="Do not drop gap<=0.05 eV values")
    parser.add_argument("--min-class-n", type=int, default=800, help="Minimum train samples to fit a class expert")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PQR HOMO-LUMO gap model: descriptor ensemble + Mazouin-style routing")
    print(f"RDKit available: {RDKIT_AVAILABLE}")
    print("=" * 80)

    entries = load_json_or_jsonl(args.data)
    bundle = build_dataset(entries, drop_zero_gaps=not args.keep_zero_gaps)
    df, X, y, groups = bundle.df, bundle.X_desc, bundle.y, bundle.groups

    train_idx, val_idx, test_idx = group_train_val_test_split(groups, y, seed=args.seed)

    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    print("Target means by split:",
          f"train={y[train_idx].mean():.3f}",
          f"val={y[val_idx].mean():.3f}",
          f"test={y[test_idx].mean():.3f}")

    # Fit base global models.
    global_models = fit_global_models(X[train_idx], y[train_idx], seed=args.seed)

    # Fit Mazouin-style selected-learning class experts.
    experts = fit_class_experts(df, X, y, train_idx, min_class_n=args.min_class_n, seed=args.seed)

    # Build predictions for stacking.
    print("Building validation prediction matrix...")
    P_val, pred_cols = predict_matrix(global_models, experts, df.iloc[val_idx].reset_index(drop=True), X[val_idx])
    print("Building test prediction matrix...")
    P_test, _ = predict_matrix(global_models, experts, df.iloc[test_idx].reset_index(drop=True), X[test_idx])

    # Evaluate base models.
    results = []
    print("\nValidation performance:")
    for j, col in enumerate(pred_cols):
        results.append(evaluate(f"VAL {col}", y[val_idx], P_val[:, j]))

    stacker = fit_stacker(P_val, y[val_idx])
    val_stack = stacker.predict(P_val)
    test_stack = stacker.predict(P_test)

    print("\nFinal stacked performance:")
    results.append(evaluate("VAL stacked", y[val_idx], val_stack))
    results.append(evaluate("TEST stacked", y[test_idx], test_stack))

    print("\nTest base performance:")
    for j, col in enumerate(pred_cols):
        results.append(evaluate(f"TEST {col}", y[test_idx], P_test[:, j]))

    # Save everything.
    joblib.dump(
        {
            "global_models": global_models,
            "experts": experts,
            "stacker": stacker,
            "pred_cols": pred_cols,
            "feature_names": bundle.feature_names,
            "rdkit_available": RDKIT_AVAILABLE,
        },
        outdir / "pqr_gap_model.joblib",
    )

    pd.DataFrame(results).to_csv(outdir / "metrics.csv", index=False)
    save_error_report(outdir, df.iloc[test_idx].reset_index(drop=True), y[test_idx], test_stack)

    print("\nSaved:")
    print(f"  {outdir / 'pqr_gap_model.joblib'}")
    print(f"  {outdir / 'metrics.csv'}")
    print(f"  {outdir / 'test_error_report.csv'}")
    print(f"  {outdir / 'mae_by_class.csv'}")
    print("\nImportant: a true ~0.1 eV MAE is only realistic if the target labels are clean,")
    print("the train/test split is not distribution-shifted, and zero/failed quantum outputs are removed.")


if __name__ == "__main__":
    main()
