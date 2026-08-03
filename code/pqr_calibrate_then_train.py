#!/usr/bin/env python3
"""
pqr_calibrate_then_train.py

Pipeline:
1. Load PQR strict-clean dataset.
2. Load QM9/reference gap CSV.
3. Find exact canonical SMILES overlap.
4. Learn calibration: reference_gap ≈ f(PQR_gap, non-leakage descriptors).
5. If calibration is good, create calibrated PQR labels and train a model.
6. If calibration is bad or overlap is too small, export a recomputation subset.

Reference CSV must contain:
  smiles,gap

The script does NOT use HOMO/LUMO as model features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, AllChem

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold


ALLOWED_QM9_ATOMS = {1, 6, 7, 8, 9}


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def canon(smiles, isomeric=True):
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def qm9_like(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return (
        all(a in ALLOWED_QM9_ATOMS for a in atoms)
        and any(a == 6 for a in atoms)
        and len(Chem.GetMolFrags(mol)) == 1
        and sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0
        and sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        and mol.GetNumHeavyAtoms() <= 9
    )


def rdkit_features(mol):
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
        sum(a.GetAtomicNum() == 16 for a in atoms),
        sum(a.GetAtomicNum() == 17 for a in atoms),
    ]


def load_pqr(path):
    rows = []
    feats = []
    drops = Counter()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue

            e = json.loads(line)
            if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
                drops["bad_shape"] += 1
                continue

            smiles_raw = e[1]
            pqr_gap = fnum(e[4])
            if not math.isfinite(pqr_gap):
                drops["bad_gap"] += 1
                continue

            mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
            if mol is None:
                drops["bad_smiles"] += 1
                continue

            smiles_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            smiles_noiso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

            pqr = e[2] if isinstance(e[2], list) else []
            lasso = e[3] if isinstance(e[3], list) else []

            # First 5 PQR only. Do NOT include HOMO/LUMO.
            x = []
            x.extend([fnum(v) for v in pqr[:5]])
            x.extend([fnum(v) for v in lasso])
            x.extend(rdkit_features(mol))

            arr = np.array(x, dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            arr = np.clip(arr, -1e6, 1e6)

            rows.append({
                "smiles": smiles_iso,
                "smiles_noiso": smiles_noiso,
                "pqr_gap": pqr_gap,
                "qm9_like": qm9_like(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
            feats.append(arr.astype(np.float32))

    max_len = max(len(x) for x in feats)
    X = np.full((len(feats), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(feats):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    desc = pd.DataFrame(X, columns=[f"d{i}" for i in range(X.shape[1])])
    df = pd.concat([df, desc], axis=1)

    desc_cols = [c for c in df.columns if c.startswith("d")]

    # Group duplicate stereo-preserved SMILES safely.
    # Numeric columns use median; text columns use first.
    agg = {"smiles_noiso": "first", "pqr_gap": "median", "qm9_like": "max", "heavy_atoms": "median"}
    for c in desc_cols:
        agg[c] = "median"

    df = df.groupby("smiles", as_index=False).agg(agg)

    print(f"PQR rows loaded: {len(df):,}")
    print(f"PQR drops: {dict(drops)}")
    print(f"PQR QM9-like rows: {int(df['qm9_like'].sum()):,}")
    return df


def load_ref(path):
    ref = pd.read_csv(path)

    if "smiles" not in ref.columns:
        raise ValueError("Reference CSV must have a 'smiles' column.")

    gap_col = None
    for c in ["gap", "homo_lumo_gap", "HOMO_LUMO_gap", "deltaE", "DeltaEHL", "gap_ev"]:
        if c in ref.columns:
            gap_col = c
            break

    if gap_col is None:
        raise ValueError("Reference CSV needs a gap column named gap, homo_lumo_gap, deltaE, etc.")

    rows = []
    for _, r in ref.iterrows():
        c_iso = canon(r["smiles"], isomeric=True)
        c_noiso = canon(r["smiles"], isomeric=False)
        g = fnum(r[gap_col])
        if c_iso is not None and c_noiso is not None and math.isfinite(g):
            rows.append({"smiles": c_iso, "smiles_noiso": c_noiso, "ref_gap": g})

    out = pd.DataFrame(rows)

    # Group duplicate reference SMILES safely.
    # Text columns use first; numeric gap uses median.
    out = out.groupby("smiles", as_index=False).agg({
        "smiles_noiso": "first",
        "ref_gap": "median",
    })

    print(f"Reference rows loaded: {len(out):,}")
    return out


def make_calibrator(kind):
    if kind == "linear":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LinearRegression()),
        ])
    if kind == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", RidgeCV(alphas=np.logspace(-6, 4, 50))),
        ])
    if kind == "hgb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-12)),
            ("model", HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.03,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                loss="absolute_error",
                random_state=42,
            )),
        ])
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", ExtraTreesRegressor(
            n_estimators=600,
            max_features=0.35,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def make_gap_model(kind):
    return make_calibrator(kind)


def evaluate(name, y, pred):
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name:32s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def write_recompute_subset(pqr, outdir, max_n=500):
    subset = pqr[pqr["qm9_like"]].copy()
    subset = subset.sort_values("pqr_gap").copy()

    if len(subset) > max_n:
        # stratified-ish across the PQR gap range
        bins = pd.qcut(subset["pqr_gap"], q=min(20, max(2, len(subset)//20)), duplicates="drop")
        subset = subset.groupby(bins, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, max_n // max(1, bins.nunique()))), random_state=42)
        )
        subset = subset.head(max_n)

    recompute_csv = outdir / "pqr_qm9_like_recompute_subset.csv"
    subset[["smiles", "pqr_gap", "heavy_atoms"]].to_csv(recompute_csv, index=False)

    xyz_dir = outdir / "qm9_like_xyz_for_recompute"
    xyz_dir.mkdir(exist_ok=True)

    manifest = []
    for i, row in subset.reset_index(drop=True).iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + i
        ok = AllChem.EmbedMolecule(m, params)
        if ok != 0:
            continue

        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass

        conf = m.GetConformer()
        path = xyz_dir / f"pqr_{i:05d}.xyz"
        with open(path, "w") as f:
            f.write(f"{m.GetNumAtoms()}\n")
            f.write(f"smiles={smi} pqr_gap={row['pqr_gap']}\n")
            for atom in m.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                f.write(f"{atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n")

        manifest.append({"id": f"pqr_{i:05d}", "smiles": smi, "pqr_gap": row["pqr_gap"], "xyz": str(path)})

    pd.DataFrame(manifest).to_csv(outdir / "recompute_manifest.csv", index=False)

    # ORCA input template: user must run ORCA locally/on cluster after installing.
    template = outdir / "ORCA_QM9_like_template.inp"
    template.write_text("""! B3LYP 6-31G(2df,p) Opt TightSCF

%pal nprocs 4 end

* xyz 0 1
# Replace this block with coordinates from the .xyz file.
*
""")

    print(f"\nCalibration failed or uncertain, so I exported recomputation files:")
    print(f"  {recompute_csv}")
    print(f"  {xyz_dir}/")
    print(f"  {outdir / 'recompute_manifest.csv'}")
    print(f"  {template}")
    print("\nUse these to recompute a PQR subset using the same/similar quantum protocol as the reference.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqr", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--outdir", default="runs/pqr_calibrate_then_train")
    ap.add_argument("--min-overlap", type=int, default=100)
    ap.add_argument("--good-calibration-mae", type=float, default=0.20)
    ap.add_argument("--recompute-n", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pqr = load_pqr(args.pqr)
    ref = load_ref(args.ref)

    desc_cols = [c for c in pqr.columns if c.startswith("d")]

    # Exact stereo-preserved overlap.
    overlap_iso = pqr.merge(ref[["smiles", "ref_gap"]], on="smiles", how="inner")
    overlap_iso.to_csv(outdir / "overlap_exact_isomeric.csv", index=False)

    # Non-isomeric overlap for diagnosis only.
    pqr_noiso = pqr.groupby("smiles_noiso", as_index=False)[["pqr_gap"] + desc_cols].median()
    ref_noiso = ref.groupby("smiles_noiso", as_index=False)[["ref_gap"]].median()
    overlap_noiso = pqr_noiso.merge(ref_noiso, on="smiles_noiso", how="inner")
    overlap_noiso.to_csv(outdir / "overlap_nonisomeric_diagnostic.csv", index=False)

    print(f"\nExact isomeric overlap:   {len(overlap_iso):,}")
    print(f"Non-isomeric overlap:     {len(overlap_noiso):,}")

    if len(overlap_iso) < args.min_overlap:
        print("\nNot enough exact overlap for reliable calibration.")
        write_recompute_subset(pqr, outdir, max_n=args.recompute_n)
        return

    overlap_iso["diff_ref_minus_pqr"] = overlap_iso["ref_gap"] - overlap_iso["pqr_gap"]

    print("\nOverlap label comparison:")
    print(f"  PQR gap mean/sd: {overlap_iso['pqr_gap'].mean():.3f}/{overlap_iso['pqr_gap'].std():.3f}")
    print(f"  REF gap mean/sd: {overlap_iso['ref_gap'].mean():.3f}/{overlap_iso['ref_gap'].std():.3f}")
    print(f"  REF-PQR mean/sd: {overlap_iso['diff_ref_minus_pqr'].mean():.3f}/{overlap_iso['diff_ref_minus_pqr'].std():.3f}")

    X_gap = overlap_iso[["pqr_gap"]].to_numpy()
    X_full = overlap_iso[["pqr_gap"] + desc_cols].to_numpy()
    y_ref = overlap_iso["ref_gap"].to_numpy()

    tr, te = train_test_split(np.arange(len(overlap_iso)), test_size=0.25, random_state=42)

    calibrators = {
        "gap_only_linear": ("linear", X_gap),
        "gap_only_ridge": ("ridge", X_gap),
        "gap_plus_desc_ridge": ("ridge", X_full),
        "gap_plus_desc_hgb": ("hgb", X_full),
        "gap_plus_desc_et": ("et", X_full),
    }

    cal_rows = []
    trained = {}

    print("\nCalibration models:")
    for name, (kind, X) in calibrators.items():
        model = make_calibrator(kind)
        model.fit(X[tr], y_ref[tr])
        pred = model.predict(X[te])
        row = evaluate(name, y_ref[te], pred)
        row["n_overlap"] = len(overlap_iso)
        row["n_features"] = X.shape[1]
        cal_rows.append(row)
        trained[name] = (model, X.shape[1])

    cal = pd.DataFrame(cal_rows).sort_values("mae")
    cal.to_csv(outdir / "calibration_metrics.csv", index=False)

    best = cal.iloc[0]
    best_name = best["name"]
    best_mae = float(best["mae"])
    best_model, nfeat = trained[best_name]

    print(f"\nBest calibration: {best_name}, MAE={best_mae:.4f} eV")

    if best_mae > args.good_calibration_mae:
        print("\nCalibration is not good enough to safely transform all PQR labels.")
        write_recompute_subset(pqr, outdir, max_n=args.recompute_n)
        return

    print("\nCalibration is good enough. Applying calibrated labels to PQR...")

    if nfeat == 1:
        X_all = pqr[["pqr_gap"]].to_numpy()
    else:
        X_all = pqr[["pqr_gap"] + desc_cols].to_numpy()

    pqr["calibrated_gap"] = best_model.predict(X_all)
    pqr[["smiles", "pqr_gap", "calibrated_gap", "qm9_like", "heavy_atoms"]].to_csv(
        outdir / "pqr_calibrated_labels.csv", index=False
    )

    # Train final model on calibrated labels.
    X_model = pqr[desc_cols].to_numpy()
    y_cal = pqr["calibrated_gap"].to_numpy()
    groups = pqr["smiles"].to_numpy()

    idx = np.arange(len(pqr))
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(idx, y_cal, groups))

    print("\nTraining final calibrated-label models:")
    final_rows = []
    for name, kind in [("calibrated_hgb", "hgb"), ("calibrated_et", "et")]:
        model = make_gap_model(kind)
        model.fit(X_model[tr_idx], y_cal[tr_idx])
        pred = model.predict(X_model[te_idx])
        row = evaluate(name, y_cal[te_idx], pred)
        final_rows.append(row)
        joblib.dump(model, outdir / f"{name}.joblib")

    pd.DataFrame(final_rows).to_csv(outdir / "calibrated_model_metrics.csv", index=False)
    joblib.dump(best_model, outdir / "best_calibrator.joblib")

    print(f"\nSaved calibrated training outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
