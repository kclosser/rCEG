#!/usr/bin/env python3
"""
calibrate_pqr_to_reference.py

Purpose:
Compare PQR HOMO-LUMO gaps against a QM9-like reference gap file and learn a
legitimate calibration map if overlapping molecules exist.

This does NOT blindly change labels. It only calibrates PQR if there is direct
evidence from shared molecules.

Inputs:
  --pqr enhanced_dataset_lasso_STRICT.jsonl
  --ref qm9_gap_reference.csv   # columns: smiles,gap

Outputs:
  overlap_report.csv
  calibration_metrics.csv
  pqr_calibrated_to_reference.csv
"""

import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def canon(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def load_pqr(path):
    rows = []
    feats = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            e = json.loads(line)
            if not (isinstance(e, list) and len(e) >= 5):
                continue

            smi = e[1]
            c = canon(smi)
            gap = fnum(e[4])
            if c is None or not math.isfinite(gap):
                continue

            pqr = e[2] if isinstance(e[2], list) else []
            desc = e[3] if isinstance(e[3], list) else []

            # Do not use HOMO/LUMO. Use first 5 PQR + LASSO descriptors.
            x = []
            x.extend([fnum(v) for v in pqr[:5]])
            x.extend([fnum(v) for v in desc])

            rows.append({"smiles": c, "pqr_gap": gap})
            feats.append(x)

    max_len = max(len(x) for x in feats)
    X = np.full((len(feats), max_len), np.nan, dtype=float)
    for i, x in enumerate(feats):
        arr = np.array(x, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        arr = np.clip(arr, -1e6, 1e6)
        X[i, :len(arr)] = arr

    df = pd.DataFrame(rows)
    for j in range(X.shape[1]):
        df[f"d{j}"] = X[:, j]

    # If exact duplicate canonical SMILES exist, median them.
    desc_cols = [c for c in df.columns if c.startswith("d")]
    df = df.groupby("smiles", as_index=False)[["pqr_gap"] + desc_cols].median()
    return df


def load_ref(path):
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError("Reference file must contain a 'smiles' column.")

    # Accept common gap column names.
    gap_col = None
    for c in ["gap", "homo_lumo_gap", "HOMO_LUMO_gap", "deltaE", "DeltaEHL", "ΔEHL"]:
        if c in df.columns:
            gap_col = c
            break

    if gap_col is None:
        raise ValueError("Reference file must contain a gap column named gap or similar.")

    rows = []
    for _, r in df.iterrows():
        c = canon(str(r["smiles"]))
        g = fnum(r[gap_col])
        if c is not None and math.isfinite(g):
            rows.append({"smiles": c, "ref_gap": g})

    out = pd.DataFrame(rows)
    out = out.groupby("smiles", as_index=False)["ref_gap"].median()
    return out


def make_model(kind):
    if kind == "linear":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LinearRegression()),
        ])

    if kind == "ridge":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", RidgeCV(alphas=np.logspace(-6, 4, 40))),
        ])

    if kind == "hgb":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-12)),
            ("model", HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.03,
                early_stopping=True,
                loss="absolute_error",
                random_state=42,
            )),
        ])

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.35,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqr", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--outdir", default="runs/pqr_reference_calibration")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pqr = load_pqr(args.pqr)
    ref = load_ref(args.ref)

    print(f"PQR molecules:       {len(pqr):,}")
    print(f"Reference molecules: {len(ref):,}")

    overlap = pqr.merge(ref, on="smiles", how="inner")
    print(f"Exact canonical overlap: {len(overlap):,}")

    if len(overlap) < 100:
        print("\nToo little overlap for reliable calibration.")
        print("You need either a QM9 file with matching SMILES or a newly computed reference subset.")
        overlap.to_csv(outdir / "overlap_report.csv", index=False)
        return

    overlap["diff_ref_minus_pqr"] = overlap["ref_gap"] - overlap["pqr_gap"]
    overlap.to_csv(outdir / "overlap_report.csv", index=False)

    print("\nRaw PQR vs reference:")
    print(f"  PQR mean/sd: {overlap.pqr_gap.mean():.3f}/{overlap.pqr_gap.std():.3f}")
    print(f"  REF mean/sd: {overlap.ref_gap.mean():.3f}/{overlap.ref_gap.std():.3f}")
    print(f"  Mean difference REF-PQR: {overlap.diff_ref_minus_pqr.mean():.3f}")
    print(f"  SD difference REF-PQR:   {overlap.diff_ref_minus_pqr.std():.3f}")

    desc_cols = [c for c in overlap.columns if c.startswith("d")]

    # Calibration features:
    # baseline uses PQR gap only; stronger models use PQR gap + descriptors.
    X_gap = overlap[["pqr_gap"]].to_numpy()
    X_full = overlap[["pqr_gap"] + desc_cols].to_numpy()
    y = overlap["ref_gap"].to_numpy()

    tr, te = train_test_split(np.arange(len(overlap)), test_size=0.25, random_state=42)

    rows = []
    trained = {}

    for name, X in [
        ("gap_only_linear", X_gap),
        ("gap_only_ridge", X_gap),
        ("gap_plus_desc_ridge", X_full),
        ("gap_plus_desc_hgb", X_full),
        ("gap_plus_desc_et", X_full),
    ]:
        kind = "linear" if "linear" in name else "ridge" if "ridge" in name else "hgb" if "hgb" in name else "et"
        model = make_model(kind)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        mae = mean_absolute_error(y[te], pred)
        r2 = r2_score(y[te], pred)
        rows.append({"model": name, "n_overlap": len(overlap), "test_mae": mae, "test_r2": r2})
        trained[name] = (model, X.shape[1])
        print(f"{name:22s} calibration MAE={mae:.4f} eV  R2={r2:.4f}")

    metrics = pd.DataFrame(rows).sort_values("test_mae")
    metrics.to_csv(outdir / "calibration_metrics.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    best_model, nfeat = trained[best_name]

    print(f"\nBest calibration model: {best_name}")

    # Apply best calibration to all PQR molecules.
    desc_cols_pqr = [c for c in pqr.columns if c.startswith("d")]
    if nfeat == 1:
        X_all = pqr[["pqr_gap"]].to_numpy()
    else:
        X_all = pqr[["pqr_gap"] + desc_cols_pqr].to_numpy()

    pqr["calibrated_gap"] = best_model.predict(X_all)
    pqr[["smiles", "pqr_gap", "calibrated_gap"]].to_csv(outdir / "pqr_calibrated_to_reference.csv", index=False)

    print(f"Wrote calibrated PQR labels to {outdir / 'pqr_calibrated_to_reference.csv'}")
    print("\nImportant:")
    print("Use calibrated labels only if overlap calibration MAE is low and the mapping is stable.")
    print("If calibration MAE is still high, PQR cannot be safely transformed into QM9-like labels.")


if __name__ == "__main__":
    main()
