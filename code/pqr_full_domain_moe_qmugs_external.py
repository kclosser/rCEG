#!/usr/bin/env python3
"""
pqr_full_domain_moe.py

Full PQR domain-aware calibrated mixture-of-experts model.

Purpose:
- Train on the full PQR dataset, not only QM9-overlap molecules.
- Use QM9 overlap and optional recomputed PQR reference labels as real anchors.
- Use calibrated pseudo-labels for the rest of PQR.
- Evaluate honestly only on held-out real reference labels.
- Report MAE by domain, confidence, and label source.

No HOMO/LUMO leakage:
- pqr[5] and pqr[6] are never used as features.
- pqr_gap is used only for calibration/teacher pseudo-label generation.
- Final predictors use molecular features only, not pqr_gap or calibrated_gap.
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
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer


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


def classify_domain(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    allowed_qm9 = {1, 6, 7, 8, 9}
    allowed = all(a in allowed_qm9 for a in atoms)
    has_carbon = any(a == 6 for a in atoms)
    heavy = mol.GetNumHeavyAtoms()
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    single = len(Chem.GetMolFrags(mol)) == 1

    if allowed and has_carbon and single and charge == 0 and radicals == 0 and heavy <= 9:
        return "qm9_like_small_organic"
    if allowed and has_carbon and single and charge == 0 and radicals == 0 and heavy <= 20:
        return "near_qm9_larger_organic"
    if allowed and has_carbon and single and charge == 0:
        return "large_neutral_organic"
    if charge != 0 or radicals != 0:
        return "charged_or_radical"
    return "heteroatom_rich_non_qm9"


def rdkit_features(mol):
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
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
        sum(a.GetAtomicNum() == 35 for a in atoms),
        sum(a.GetAtomicNum() == 53 for a in atoms),
        sum(a.GetAtomicNum() == 15 for a in atoms),
        sum(a.GetAtomicNum() == 5 for a in atoms),
    ]


def bond_step_features(mol):
    try:
        dm = Chem.GetDistanceMatrix(mol).astype(float)
        upper = dm[np.triu_indices_from(dm, k=1)]
        upper = upper[np.isfinite(upper)]
        if len(upper) == 0:
            return [0.0] * 16

        out = [
            np.max(upper),
            np.mean(upper),
            np.std(upper),
            np.median(upper),
            np.percentile(upper, 25),
            np.percentile(upper, 75),
        ]

        for d in range(1, 11):
            out.append(float(np.sum(upper == d)) / len(upper))

        return out
    except Exception:
        return [np.nan] * 16


def load_pqr(path):
    rows = []
    feats = []
    audit_rows = []
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
            pqr = e[2] if isinstance(e[2], list) else []
            lasso = e[3] if isinstance(e[3], list) else []

            if not math.isfinite(pqr_gap):
                drops["bad_gap"] += 1
                continue

            mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
            if mol is None:
                drops["bad_smiles"] += 1
                continue

            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

            # Non-leakage features only:
            # pqr[:5] excludes pqr[5]=HOMO and pqr[6]=LUMO.
            x = []
            x.extend([fnum(v) for v in pqr[:5]])
            x.extend([fnum(v) for v in lasso])
            x.extend(rdkit_features(mol))
            x.extend(bond_step_features(mol))

            arr = np.array(x, dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            arr = np.clip(arr, -1e6, 1e6)

            rows.append({
                "smiles": smiles,
                "pqr_gap": pqr_gap,
                "domain": classify_domain(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
            feats.append(arr.astype(np.float32))

            if isinstance(pqr, list) and len(pqr) >= 7:
                audit_rows.append({
                    "smiles": smiles,
                    "gap": pqr_gap,
                    "homo": fnum(pqr[5]),
                    "lumo": fnum(pqr[6]),
                    "lumo_minus_homo": fnum(pqr[6]) - fnum(pqr[5]),
                })

    max_len = max(len(x) for x in feats)
    X = np.full((len(feats), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(feats):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df = pd.concat([df, Xdf], axis=1)

    feat_cols = [c for c in df.columns if c.startswith("x")]

    agg = {"pqr_gap": "median", "domain": "first", "heavy_atoms": "median"}
    for c in feat_cols:
        agg[c] = "median"

    df = df.groupby("smiles", as_index=False).agg(agg)
    audit = pd.DataFrame(audit_rows).drop_duplicates("smiles")

    print(f"PQR unique molecules: {len(df):,}")
    print(f"PQR drops: {dict(drops)}")
    print("PQR domains:")
    print(df["domain"].value_counts().to_string())

    return df, audit


def load_reference(path, source_name):
    ref = pd.read_csv(path)
    if "smiles" not in ref.columns:
        raise ValueError(f"{path} must contain a smiles column.")

    gap_col = None
    for c in ["gap", "ref_gap", "homo_lumo_gap", "HOMO_LUMO_gap", "deltaE", "DeltaEHL", "gap_ev"]:
        if c in ref.columns:
            gap_col = c
            break

    if gap_col is None:
        raise ValueError(f"{path} must contain a gap/ref_gap column.")

    rows = []
    for _, r in ref.iterrows():
        smi = canon(r["smiles"], isomeric=True)
        g = fnum(r[gap_col])
        if smi is not None and math.isfinite(g):
            rows.append({"smiles": smi, "ref_gap": g, "ref_source": source_name})

    out = pd.DataFrame(rows)
    out = out.groupby("smiles", as_index=False).agg({
        "ref_gap": "median",
        "ref_source": "first",
    })

    print(f"Reference {source_name}: {len(out):,} unique molecules")
    return out


def leakage_filter(df, audit, threshold, outdir):
    feat_cols = [c for c in df.columns if c.startswith("x")]
    merged = df[["smiles"] + feat_cols].merge(audit, on="smiles", how="inner")

    targets = ["gap", "homo", "lumo", "lumo_minus_homo"]
    bad = set()
    records = []

    for c in feat_cols:
        vals = pd.to_numeric(merged[c], errors="coerce")
        if vals.notna().sum() < 100:
            continue

        for t in targets:
            corr = vals.corr(merged[t])
            ac = abs(corr) if pd.notna(corr) else np.nan
            records.append({"feature": c, "target": t, "abs_corr": ac, "corr": corr})
            if pd.notna(ac) and ac >= threshold:
                bad.add(c)

    audit_df = pd.DataFrame(records).sort_values("abs_corr", ascending=False)
    audit_df.to_csv(outdir / "feature_leakage_audit.csv", index=False)

    keep = [c for c in feat_cols if c not in bad]
    print(f"Leakage filter dropped {len(bad)} features at abs_corr >= {threshold}")
    print(audit_df.head(20).to_string(index=False))
    return keep


def pipe(model, scale=True):
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
    ]
    if scale:
        steps.append(("scale", RobustScaler(with_centering=False)))
    steps.append(("model", model))
    return Pipeline(steps)


def et(seed, n=500):
    return pipe(ExtraTreesRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    ), scale=False)


def rf(seed, n=400):
    return pipe(RandomForestRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    ), scale=False)


def hgb(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            validation_fraction=0.15,
            n_iter_no_change=80,
            early_stopping=True,
            loss="absolute_error",
            random_state=seed,
        )),
    ])


def _continuous_scaler(kind):
    """
    Scaling is fitted inside each sklearn Pipeline, so validation/test
    statistics never influence training preprocessing.
    """
    if kind == "robust":
        # Wider quantile range avoids allowing a very narrow IQR to
        # exaggerate small differences in sparse/count descriptors.
        return RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(10.0, 90.0),
        )

    if kind == "standard":
        return StandardScaler(
            with_mean=True,
            with_std=True,
        )

    if kind == "power":
        # Yeo-Johnson supports zero and negative descriptor values.
        # It reduces skew and then standardizes to zero mean/unit variance.
        return PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        )

    raise ValueError(f"Unknown scaler kind: {kind}")


def ridge_scaled(kind="robust"):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", _continuous_scaler(kind)),
        ("model", RidgeCV(alphas=np.logspace(-6, 5, 60))),
    ])


def linear_scaled(kind="robust"):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", _continuous_scaler(kind)),
        ("model", LinearRegression()),
    ])


# Preserve the original function names for inverse calibration and any
# other existing code paths.
def ridge():
    return ridge_scaled("robust")


def lin():
    return linear_scaled("robust")


def metric(name, y, pred):
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name:38s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2), "n": int(len(y))}


def add_confidence(pqr, ref_train, ref_val, ref_test, feat_cols):
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_ref = scaler.fit_transform(imp.fit_transform(ref_train[feat_cols]))
    X_all = scaler.transform(imp.transform(pqr[feat_cols]))

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_ref)

    d_all, _ = nn.kneighbors(X_all)
    pqr["calibration_distance"] = d_all[:, 0]

    X_eval_ref = scaler.transform(imp.transform(pd.concat([ref_val, ref_test])[feat_cols]))
    d_eval, _ = nn.kneighbors(X_eval_ref)
    ref_dist = d_eval[:, 0]

    high_thr = float(np.quantile(ref_dist, 0.90))
    med_thr = float(np.quantile(ref_dist, 0.975))

    def label(d):
        if d <= high_thr:
            return "high_in_domain"
        if d <= med_thr:
            return "medium_near_domain"
        return "low_out_of_domain"

    pqr["calibration_confidence"] = [label(d) for d in pqr["calibration_distance"]]

    print("\nApplicability-domain thresholds:")
    print(f"  high <= {high_thr:.4f}")
    print(f"  medium <= {med_thr:.4f}")
    print(pqr["calibration_confidence"].value_counts().to_string())

    return pqr


def qedges(y, k):
    edges = np.percentile(y, np.linspace(0, 100, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign(y, edges):
    return np.digitize(y, edges[1:-1], right=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqr", required=True)
    ap.add_argument("--qm9-ref", required=True)
    ap.add_argument("--extra-ref", default=None, help="Optional recomputed PQR reference CSV with smiles,gap columns.")
    ap.add_argument("--outdir", default="runs/pqr_full_domain_moe")
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--leakage-corr-threshold", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument(
        "--external-ref",
        default=None,
        help=(
            "External reference CSV containing smiles and gap. "
            "Matching molecules are excluded completely from training."
        ),
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pqr, audit = load_pqr(args.pqr)
    qm9 = load_reference(args.qm9_ref, "qm9_overlap")

    refs = [qm9]
    if args.extra_ref:
        extra = load_reference(args.extra_ref, "recomputed_pqr_reference")
        refs.append(extra)

    ref = pd.concat(refs, ignore_index=True)
    ref = ref.groupby("smiles", as_index=False).agg({
        "ref_gap": "median",
        "ref_source": lambda x: "+".join(sorted(set(map(str, x)))),
    })

    # --------------------------------------------------------
    # Strict published external benchmark.
    #
    # External molecules are captured with their existing PQR
    # feature vectors and then removed from:
    #   - leakage filtering
    #   - calibration
    #   - pseudo-label generation
    #   - final expert training
    #
    # The external labels are never accessed during training.
    # --------------------------------------------------------
    external_test = None

    if args.external_ref:
        external_ref = load_reference(
            args.external_ref,
            "qmugs_dft_external",
        )

        external_test = pqr.merge(
            external_ref,
            on="smiles",
            how="inner",
        )

        external_smiles = set(external_test["smiles"])

        if len(external_test) < 50:
            raise RuntimeError(
                f"Only {len(external_test)} external overlap molecules found."
            )

        pqr = pqr[
            ~pqr["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        audit = audit[
            ~audit["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        ref = ref[
            ~ref["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        print(
            f"\nSTRICT EXTERNAL HOLDOUT: "
            f"{len(external_test):,} QMugs molecules removed "
            f"from all model-development stages."
        )

    feat_cols = leakage_filter(pqr, audit, args.leakage_corr_threshold, outdir)

    # Hard no-leakage assertion.
    # Final model input features must be x-columns only.
    banned_tokens = [
        "homo",
        "lumo",
        "lumo_minus_homo",
        "homo_lumo",
        "gap",
        "ref_gap",
        "pqr_gap",
        "training_label",
        "pred_qm9_aligned_gap",
        "inverse_reconstructed_pqr_gap",
        "cycle_error",
    ]

    bad_features = [
        c for c in feat_cols
        if any(tok in str(c).lower() for tok in banned_tokens)
    ]

    non_x_features = [c for c in feat_cols if not str(c).startswith("x")]

    if bad_features or non_x_features:
        raise RuntimeError(
            f"Leakage/non-x features found. bad_features={bad_features}, non_x_features={non_x_features}"
        )

    print(f"No-leakage feature check passed: {len(feat_cols)} x-features used.")

    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    print(f"\nReference-labeled PQR molecules available: {len(ref_pqr):,}")
    print(ref_pqr["ref_source"].value_counts().to_string())

    if len(ref_pqr) < 300:
        raise RuntimeError("Too few reference-labeled molecules for honest validation.")

    # Split real-reference molecules into calibration train, stack validation, final test.
    idx = np.arange(len(ref_pqr))
    cal_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed + 1)

    ref_cal = ref_pqr.iloc[cal_idx].copy()
    ref_val = ref_pqr.iloc[val_idx].copy()
    ref_test = ref_pqr.iloc[test_idx].copy()

    print(f"\nReference split: cal_train={len(ref_cal):,}, stack_val={len(ref_val):,}, final_test={len(ref_test):,}")

    # Calibration: PQR gap + features -> real reference gap.
    Xcal_gap = ref_cal[["pqr_gap"]].to_numpy()
    Xval_gap = ref_val[["pqr_gap"]].to_numpy()
    Xtest_gap = ref_test[["pqr_gap"]].to_numpy()

    Xcal_full = ref_cal[["pqr_gap"] + feat_cols].to_numpy()
    Xval_full = ref_val[["pqr_gap"] + feat_cols].to_numpy()
    Xtest_full = ref_test[["pqr_gap"] + feat_cols].to_numpy()

    ycal = ref_cal["ref_gap"].to_numpy()
    yval = ref_val["ref_gap"].to_numpy()
    ytest = ref_test["ref_gap"].to_numpy()

    calibrators = {
        "gap_linear_robust": (
            linear_scaled("robust"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_robust": (
            ridge_scaled("robust"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_standard": (
            ridge_scaled("standard"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_power": (
            ridge_scaled("power"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),

        # Descriptor-assisted regularized calibration baselines.
        "desc_ridge_robust": (
            ridge_scaled("robust"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),
        "desc_ridge_standard": (
            ridge_scaled("standard"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),
        "desc_ridge_power": (
            ridge_scaled("power"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),

        "desc_hgb": (hgb(args.seed + 10), Xcal_full, Xval_full, Xtest_full, "full"),
        "desc_et": (et(args.seed + 20, n=700), Xcal_full, Xval_full, Xtest_full, "full"),
        "desc_rf": (rf(args.seed + 30, n=500), Xcal_full, Xval_full, Xtest_full, "full"),
    }

    cal_models = {}
    cal_rows = []
    val_preds = {}
    test_preds = {}

    print("\nCalibration models:")
    for name, (model, Xc, Xv, Xt, mode) in calibrators.items():
        model.fit(Xc, ycal)
        pv = model.predict(Xv)
        pt = model.predict(Xt)

        row_v = metric("VAL cal_" + name, yval, pv)
        row_t = metric("TEST cal_" + name, ytest, pt)
        row_v["test_mae"] = row_t["mae"]
        row_v["test_r2"] = row_t["r2"]
        row_v["mode"] = mode
        cal_rows.append(row_v)

        cal_models[name] = (model, mode)
        val_preds[name] = pv
        test_preds[name] = pt

    cal_df = pd.DataFrame(cal_rows).sort_values("mae")
    cal_df.to_csv(outdir / "calibration_metrics.csv", index=False)

    top = list(cal_df.head(3)["name"].str.replace("VAL cal_", "", regex=False))
    maes = np.array([cal_df[cal_df["name"] == "VAL cal_" + n]["mae"].iloc[0] for n in top])
    weights = 1 / np.maximum(maes, 1e-6)
    weights = weights / weights.sum()

    print("\nCalibration ensemble:")
    for n, w in zip(top, weights):
        print(f"  {n}: weight={w:.3f}")

    ens_val = sum(w * val_preds[n] for n, w in zip(top, weights))
    ens_test = sum(w * test_preds[n] for n, w in zip(top, weights))
    ens_rows = [
        metric("VAL calibration_ensemble", yval, ens_val),
        metric("TEST calibration_ensemble", ytest, ens_test),
    ]
    pd.DataFrame(ens_rows).to_csv(outdir / "calibration_ensemble_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Inverse calibration / cycle-consistency model:
    #   real reference gap + features -> PQR-style gap
    #
    # This lets us test:
    #   PQR gap -> forward calibration -> QM9-aligned gap
    #           -> inverse calibration -> reconstructed PQR gap
    #
    # cycle_error = |reconstructed_pqr_gap - original_pqr_gap|
    #
    # This is a confidence diagnostic, not an external proof of correctness.
    # ------------------------------------------------------------------
    print("\nInverse calibration models for cycle-consistency:")

    ycal_pqr = ref_cal["pqr_gap"].to_numpy()
    yval_pqr = ref_val["pqr_gap"].to_numpy()
    ytest_pqr = ref_test["pqr_gap"].to_numpy()

    Xinv_cal_gap = ycal.reshape(-1, 1)
    Xinv_val_gap = yval.reshape(-1, 1)
    Xinv_test_gap = ytest.reshape(-1, 1)

    Xinv_cal_full = np.column_stack([ycal, ref_cal[feat_cols].to_numpy()])
    Xinv_val_full = np.column_stack([yval, ref_val[feat_cols].to_numpy()])
    Xinv_test_full = np.column_stack([ytest, ref_test[feat_cols].to_numpy()])

    inverse_candidates = {
        "inv_gap_linear": (lin(), Xinv_cal_gap, Xinv_val_gap, Xinv_test_gap, "gap"),
        "inv_gap_ridge": (ridge(), Xinv_cal_gap, Xinv_val_gap, Xinv_test_gap, "gap"),
        "inv_desc_hgb": (hgb(args.seed + 410), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
        "inv_desc_et": (et(args.seed + 420, n=700), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
        "inv_desc_rf": (rf(args.seed + 430, n=500), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
    }

    inv_models = {}
    inv_val_preds = {}
    inv_test_preds = {}
    inv_rows = []

    for name, (model, Xc, Xv, Xt, mode) in inverse_candidates.items():
        model.fit(Xc, ycal_pqr)
        pv = model.predict(Xv)
        pt = model.predict(Xt)

        row_v = metric("VAL " + name, yval_pqr, pv)
        row_t = metric("TEST " + name, ytest_pqr, pt)

        row_v["test_mae"] = row_t["mae"]
        row_v["test_r2"] = row_t["r2"]
        row_v["mode"] = mode

        inv_rows.append(row_v)
        inv_models[name] = (model, mode)
        inv_val_preds[name] = pv
        inv_test_preds[name] = pt

    inv_df = pd.DataFrame(inv_rows).sort_values("mae")
    inv_df.to_csv(outdir / "inverse_calibration_metrics.csv", index=False)

    inv_top = list(inv_df.head(3)["name"].str.replace("VAL ", "", regex=False))
    inv_maes = np.array([inv_df[inv_df["name"] == "VAL " + n]["mae"].iloc[0] for n in inv_top])
    inv_weights = 1 / np.maximum(inv_maes, 1e-6)
    inv_weights = inv_weights / inv_weights.sum()

    print("\nInverse calibration ensemble:")
    for n, w in zip(inv_top, inv_weights):
        print(f"  {n}: weight={w:.3f}")

    inv_ens_val = sum(w * inv_val_preds[n] for n, w in zip(inv_top, inv_weights))
    inv_ens_test = sum(w * inv_test_preds[n] for n, w in zip(inv_top, inv_weights))

    inv_ens_rows = [
        metric("VAL inverse_ensemble", yval_pqr, inv_ens_val),
        metric("TEST inverse_ensemble", ytest_pqr, inv_ens_test),
    ]
    pd.DataFrame(inv_ens_rows).to_csv(outdir / "inverse_calibration_ensemble_metrics.csv", index=False)

    # Generate calibrated labels for all PQR.
    Xall_gap = pqr[["pqr_gap"]].to_numpy()
    Xall_full = pqr[["pqr_gap"] + feat_cols].to_numpy()

    pseudo = np.zeros(len(pqr), dtype=float)
    for n, w in zip(top, weights):
        model, mode = cal_models[n]
        if mode == "gap":
            pseudo += w * model.predict(Xall_gap)
        else:
            pseudo += w * model.predict(Xall_full)

    # Forward calibrated label: PQR-style -> QM9/reference-aligned.
    pqr["pred_qm9_aligned_gap"] = pseudo
    pqr["training_label"] = pseudo
    pqr["label_source"] = "calibrated_pseudo_label"

    # Inverse reconstruction: QM9/reference-aligned -> reconstructed PQR-style.
    Xinv_all_gap = pseudo.reshape(-1, 1)
    Xinv_all_full = np.column_stack([pseudo, pqr[feat_cols].to_numpy()])

    inv_recon = np.zeros(len(pqr), dtype=float)
    for n, w in zip(inv_top, inv_weights):
        model, mode = inv_models[n]
        if mode == "gap":
            inv_recon += w * model.predict(Xinv_all_gap)
        else:
            inv_recon += w * model.predict(Xinv_all_full)

    pqr["inverse_reconstructed_pqr_gap"] = inv_recon
    pqr["cycle_error"] = np.abs(pqr["inverse_reconstructed_pqr_gap"] - pqr["pqr_gap"])

    def _cycle_conf(e):
        if e <= 0.20:
            return "high_cycle_consistency"
        if e <= 0.50:
            return "medium_cycle_consistency"
        return "low_cycle_consistency"

    pqr["cycle_confidence"] = [_cycle_conf(e) for e in pqr["cycle_error"].to_numpy()]

    print("\nForward-inverse cycle consistency across all PQR:")
    print(pqr["cycle_error"].describe().to_string())
    print("\nCycle confidence counts:")
    print(pqr["cycle_confidence"].value_counts().to_string())

    # Use real reference labels for calibration-training anchors only.
    ref_map = dict(ref_cal[["smiles", "ref_gap"]].values)
    pqr.loc[pqr["smiles"].isin(ref_map), "training_label"] = pqr.loc[pqr["smiles"].isin(ref_map), "smiles"].map(ref_map)
    pqr.loc[pqr["smiles"].isin(ref_map), "label_source"] = "real_reference_anchor"

    # Do not train on validation/test reference molecules.
    holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
    train_pool = pqr[~pqr["smiles"].isin(holdout)].copy()

    pqr = add_confidence(pqr, ref_cal, ref_val, ref_test, feat_cols)

    def _combine_conf(row):
        cal = row["calibration_confidence"]
        cyc = row["cycle_confidence"]

        if cal == "low_out_of_domain" or cyc == "low_cycle_consistency":
            return "low_confidence"
        if cal == "medium_near_domain" or cyc == "medium_cycle_consistency":
            return "medium_confidence"
        return "high_confidence"

    pqr["final_confidence"] = pqr.apply(_combine_conf, axis=1)

    print("\nFinal combined confidence counts:")
    print(pqr["final_confidence"].value_counts().to_string())

    Xtrain = train_pool[feat_cols].to_numpy()
    ytrain = train_pool["training_label"].to_numpy()

    Xv_final = ref_val[feat_cols].to_numpy()
    Xt_final = ref_test[feat_cols].to_numpy()

    print(f"\nFull final training pool: {len(train_pool):,}")
    print("Training label sources:")
    print(train_pool["label_source"].value_counts().to_string())

    # Final global experts.
    print("\nTraining final full-PQR experts...")
    g_hgb = hgb(args.seed + 100)
    g_et = et(args.seed + 110, n=700)
    g_rf = rf(args.seed + 120, n=500)

    g_hgb.fit(Xtrain, ytrain)
    g_et.fit(Xtrain, ytrain)
    g_rf.fit(Xtrain, ytrain)

    # Regularized scaled experts. Because preprocessing is inside each
    # pipeline, every scaler is fitted on Xtrain only.
    g_ridge_robust = ridge_scaled("robust")
    g_ridge_standard = ridge_scaled("standard")

    print("Training scaled regularized global experts...")
    g_ridge_robust.fit(Xtrain, ytrain)
    g_ridge_standard.fit(Xtrain, ytrain)

    pred_v = {}
    pred_t = {}

    pred_v["global_hgb"] = g_hgb.predict(Xv_final)
    pred_t["global_hgb"] = g_hgb.predict(Xt_final)

    pred_v["global_et"] = g_et.predict(Xv_final)
    pred_t["global_et"] = g_et.predict(Xt_final)

    pred_v["global_rf"] = g_rf.predict(Xv_final)
    pred_t["global_rf"] = g_rf.predict(Xt_final)

    pred_v["global_ridge_robust"] = g_ridge_robust.predict(Xv_final)
    pred_t["global_ridge_robust"] = g_ridge_robust.predict(Xt_final)

    pred_v["global_ridge_standard"] = g_ridge_standard.predict(Xv_final)
    pred_t["global_ridge_standard"] = g_ridge_standard.predict(Xt_final)

    # Preserve the original tree-only mean for routing and direct
    # comparison with the prior model.
    pred_v["global_mean"] = (
        pred_v["global_hgb"]
        + pred_v["global_et"]
        + pred_v["global_rf"]
    ) / 3
    pred_t["global_mean"] = (
        pred_t["global_hgb"]
        + pred_t["global_et"]
        + pred_t["global_rf"]
    ) / 3

    # Domain experts.
    print("\nTraining chemical-domain experts...")
    dom_v = np.zeros(len(ref_val))
    dom_t = np.zeros(len(ref_test))
    domain_models = {}

    train_domains = train_pool["domain"].to_numpy()
    val_domains = ref_val["domain"].to_numpy()
    test_domains = ref_test["domain"].to_numpy()

    for domain in sorted(train_pool["domain"].unique()):
        mtr = train_domains == domain
        mv = val_domains == domain
        mt = test_domains == domain
        n = int(mtr.sum())

        if n >= 500:
            print(f"  {domain}: {n:,} rows")
            model = et(args.seed + 200 + len(domain), n=500)
            model.fit(Xtrain[mtr], ytrain[mtr])
            domain_models[domain] = model

            if mv.any():
                dom_v[mv] = model.predict(Xv_final[mv])
            if mt.any():
                dom_t[mt] = model.predict(Xt_final[mt])
        else:
            if mv.any():
                dom_v[mv] = pred_v["global_mean"][mv]
            if mt.any():
                dom_t[mt] = pred_t["global_mean"][mt]

    pred_v["domain_expert"] = dom_v
    pred_t["domain_expert"] = dom_t

    # --------------------------------------------------------
    # Frozen-model external QMugs evaluation.
    # --------------------------------------------------------
    if external_test is not None:
        from sklearn.model_selection import KFold

        Xext = external_test[feat_cols].to_numpy()
        yext = external_test["ref_gap"].to_numpy()

        ext_hgb = g_hgb.predict(Xext)
        ext_et = g_et.predict(Xext)
        ext_rf = g_rf.predict(Xext)
        ext_global_mean = (ext_hgb + ext_et + ext_rf) / 3.0

        ext_domain = ext_global_mean.copy()
        ext_domains = external_test["domain"].to_numpy()

        for domain, model in domain_models.items():
            mask = ext_domains == domain
            if mask.any():
                ext_domain[mask] = model.predict(Xext[mask])

        external_report = external_test[
            ["smiles", "ref_gap", "domain", "heavy_atoms"]
        ].copy()

        external_report["pred_global_hgb"] = ext_hgb
        external_report["pred_global_et"] = ext_et
        external_report["pred_global_rf"] = ext_rf
        external_report["pred_global_mean"] = ext_global_mean
        external_report["pred_domain_expert"] = ext_domain

        for name in [
            "global_hgb",
            "global_et",
            "global_rf",
            "global_mean",
            "domain_expert",
        ]:
            external_report["abs_err_" + name] = np.abs(
                external_report["pred_" + name]
                - external_report["ref_gap"]
            )

        # Secondary diagnostic:
        # five-fold cross-fitted constant offset correction. This estimates
        # performance after correcting only the computational-method offset.
        aligned = np.zeros(len(external_report), dtype=float)

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        )

        raw_pred = external_report[
            "pred_domain_expert"
        ].to_numpy()

        for fit_idx, eval_idx in kf.split(raw_pred):
            offset = np.median(
                yext[fit_idx] - raw_pred[fit_idx]
            )
            aligned[eval_idx] = raw_pred[eval_idx] + offset

        external_report[
            "pred_domain_expert_crossfit_offset_aligned"
        ] = aligned

        external_report[
            "abs_err_domain_expert_crossfit_offset_aligned"
        ] = np.abs(aligned - yext)

        external_report.to_csv(
            outdir / "qmugs_external_predictions.csv",
            index=False,
        )

        external_metrics = []

        for name in [
            "global_hgb",
            "global_et",
            "global_rf",
            "global_mean",
            "domain_expert",
        ]:
            pred = external_report["pred_" + name].to_numpy()
            external_metrics.append({
                "evaluation": "strict_frozen_external",
                "model": name,
                "mae": float(mean_absolute_error(yext, pred)),
                "rmse": float(
                    np.sqrt(np.mean((yext - pred) ** 2))
                ),
                "r2": float(r2_score(yext, pred)),
                "n": int(len(yext)),
            })

        external_metrics.append({
            "evaluation": "five_fold_crossfit_offset_aligned",
            "model": "domain_expert",
            "mae": float(mean_absolute_error(yext, aligned)),
            "rmse": float(
                np.sqrt(np.mean((yext - aligned) ** 2))
            ),
            "r2": float(r2_score(yext, aligned)),
            "n": int(len(yext)),
        })

        pd.DataFrame(external_metrics).to_csv(
            outdir / "qmugs_external_metrics.csv",
            index=False,
        )

        domain_rows = []

        for domain, group in external_report.groupby("domain"):
            domain_rows.append({
                "domain": domain,
                "n": len(group),
                "mae_domain_expert": float(
                    group["abs_err_domain_expert"].mean()
                ),
                "mae_offset_aligned": float(
                    group[
                        "abs_err_domain_expert_crossfit_offset_aligned"
                    ].mean()
                ),
                "mean_reference_gap_ev": float(
                    group["ref_gap"].mean()
                ),
            })

        pd.DataFrame(domain_rows).to_csv(
            outdir / "qmugs_external_mae_by_domain.csv",
            index=False,
        )

        print("\nSTRICT QMugs external performance:")
        print(
            pd.DataFrame(external_metrics)
            .sort_values("mae")
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Piecewise split architecture:
    #   For each chemical domain, identify an x-feature whose relationship
    #   with the training label is better approximated by two local lines
    #   than by one global line. The best breakpoint is selected using only
    #   x-features and training labels.
    #
    #   Then each domain is split into left/right branches at that breakpoint,
    #   and a separate local expert is trained for each branch.
    #
    #   This replaces the old calibrated-gap regime split. It is designed to
    #   capture nonlinear descriptor-target relationships without using HOMO,
    #   LUMO, pqr_gap, ref_gap, or any direct gap-derived feature as input.
    # ------------------------------------------------------------------
    print("\nTraining piecewise descriptor-split architecture experts...")

    def _line_mae(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(y) < 50 or np.nanstd(x) < 1e-12:
            return np.inf
        A = np.column_stack([x, np.ones(len(x))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        return float(np.mean(np.abs(pred - y)))

    def _best_piecewise_split(Xdf, y, candidate_cols, min_leaf=300, max_features=40):
        y = np.asarray(y, dtype=float)

        # Rank candidate x-features by absolute correlation with the training label.
        ranked = []
        for c in candidate_cols:
            x = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 2 * min_leaf:
                continue
            if np.nanstd(x[ok]) < 1e-12:
                continue
            corr = np.corrcoef(x[ok], y[ok])[0, 1]
            if np.isfinite(corr):
                ranked.append((c, abs(corr), corr))

        ranked = sorted(ranked, key=lambda z: z[1], reverse=True)[:max_features]

        best = None

        for c, abs_corr, corr in ranked:
            x = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            x_ok = x[ok]
            y_ok = y[ok]

            if len(y_ok) < 2 * min_leaf:
                continue

            base_mae = _line_mae(x_ok, y_ok)

            # Candidate breakpoints between the 20th and 80th percentiles.
            qs = np.linspace(0.20, 0.80, 25)
            cuts = np.unique(np.quantile(x_ok, qs))

            for cut in cuts:
                left = x_ok <= cut
                right = ~left

                if left.sum() < min_leaf or right.sum() < min_leaf:
                    continue

                mae_left = _line_mae(x_ok[left], y_ok[left])
                mae_right = _line_mae(x_ok[right], y_ok[right])

                if not np.isfinite(mae_left) or not np.isfinite(mae_right):
                    continue

                piece_mae = (left.sum() * mae_left + right.sum() * mae_right) / len(y_ok)
                improvement = base_mae - piece_mae

                rec = {
                    "feature": c,
                    "corr": float(corr),
                    "abs_corr": float(abs_corr),
                    "split_value": float(cut),
                    "single_line_mae": float(base_mae),
                    "piecewise_line_mae": float(piece_mae),
                    "improvement": float(improvement),
                    "left_n": int(left.sum()),
                    "right_n": int(right.sum()),
                }

                if best is None or rec["improvement"] > best["improvement"]:
                    best = rec

        return best

    piece_v = np.zeros(len(ref_val))
    piece_t = np.zeros(len(ref_test))

    # Fallback to domain expert if a split branch is too small or not helpful.
    piece_v[:] = pred_v["domain_expert"]
    piece_t[:] = pred_t["domain_expert"]

    piece_rows = []

    for domain in sorted(train_pool["domain"].unique()):
        mtr = train_domains == domain
        mv = val_domains == domain
        mt = test_domains == domain

        n_domain = int(mtr.sum())
        if n_domain < 1000:
            print(f"  {domain}: n={n_domain:,}; using domain-expert fallback")
            continue

        X_domain_df = train_pool.loc[mtr, feat_cols]
        y_domain = ytrain[mtr]

        split = _best_piecewise_split(
            X_domain_df,
            y_domain,
            feat_cols,
            min_leaf=max(250, min(750, n_domain // 12)),
            max_features=50,
        )

        if split is None or split["improvement"] <= 0:
            print(f"  {domain}: no useful piecewise split found; using fallback")
            continue

        feature = split["feature"]
        cut = split["split_value"]

        x_train = pd.to_numeric(train_pool.loc[mtr, feature], errors="coerce").to_numpy(dtype=float)
        left_train = x_train <= cut
        right_train = x_train > cut

        print(
            f"  {domain}: split {feature} <= {cut:.5g}; "
            f"linear MAE {split['single_line_mae']:.3f} -> {split['piecewise_line_mae']:.3f}; "
            f"left={left_train.sum():,}, right={right_train.sum():,}"
        )

        split["domain"] = domain
        split["domain_n"] = n_domain
        piece_rows.append(split)

        # Train branch experts.
        left_model = et(args.seed + 700 + (abs(hash(domain + feature + 'L')) % 10000), n=500)
        right_model = et(args.seed + 800 + (abs(hash(domain + feature + 'R')) % 10000), n=500)

        X_domain = Xtrain[mtr]
        left_model.fit(X_domain[left_train], y_domain[left_train])
        right_model.fit(X_domain[right_train], y_domain[right_train])

        # Route validation molecules in this domain.
        if mv.any():
            val_indices = np.where(mv)[0]
            x_val = pd.to_numeric(ref_val.loc[mv, feature], errors="coerce").to_numpy(dtype=float)
            left_val = x_val <= cut
            right_val = x_val > cut

            if left_val.any():
                piece_v[val_indices[left_val]] = left_model.predict(Xv_final[val_indices[left_val]])
            if right_val.any():
                piece_v[val_indices[right_val]] = right_model.predict(Xv_final[val_indices[right_val]])

        # Route test molecules in this domain.
        if mt.any():
            test_indices = np.where(mt)[0]
            x_test = pd.to_numeric(ref_test.loc[mt, feature], errors="coerce").to_numpy(dtype=float)
            left_test = x_test <= cut
            right_test = x_test > cut

            if left_test.any():
                piece_t[test_indices[left_test]] = left_model.predict(Xt_final[test_indices[left_test]])
            if right_test.any():
                piece_t[test_indices[right_test]] = right_model.predict(Xt_final[test_indices[right_test]])

    piece_df = pd.DataFrame(piece_rows)
    piece_df.to_csv(outdir / "piecewise_split_architecture_branches.csv", index=False)

    pred_v["piecewise_split_expert"] = piece_v
    pred_t["piecewise_split_expert"] = piece_t

    # Visualization of learned piecewise splits.
    try:
        import matplotlib.pyplot as plt

        if len(piece_df):
            nplot = min(6, len(piece_df))
            plot_df = piece_df.sort_values("improvement", ascending=False).head(nplot)

            fig, axes = plt.subplots(nplot, 1, figsize=(8, 3.2 * nplot))
            if nplot == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, plot_df.iterrows()):
                domain = row["domain"]
                feature = row["feature"]
                cut = row["split_value"]

                mask = train_pool["domain"].to_numpy() == domain
                d = train_pool.loc[mask, [feature, "training_label"]].copy()
                d = d.replace([np.inf, -np.inf], np.nan).dropna()

                if len(d) > 5000:
                    d = d.sample(5000, random_state=args.seed)

                ax.scatter(d[feature], d["training_label"], s=4, alpha=0.25)
                ax.axvline(cut, linestyle="--", linewidth=1.5)

                # Draw separate local linear fits for left and right.
                for side_mask in [d[feature] <= cut, d[feature] > cut]:
                    sub = d[side_mask]
                    if len(sub) > 50 and sub[feature].std() > 1e-12:
                        xs = sub[feature].to_numpy(dtype=float)
                        ys = sub["training_label"].to_numpy(dtype=float)
                        A = np.column_stack([xs, np.ones(len(xs))])
                        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                        grid = np.linspace(xs.min(), xs.max(), 100)
                        ax.plot(grid, coef[0] * grid + coef[1], linewidth=1.5)

                ax.set_title(
                    f"{domain}: {feature} split at {cut:.4g} "
                    f"(linear MAE improvement={row['improvement']:.3f} eV)"
                )
                ax.set_xlabel(feature)
                ax.set_ylabel("Training label / reference-aligned gap (eV)")
                ax.grid(True, alpha=0.25)

            fig.suptitle("Learned Piecewise Descriptor Splits", fontsize=14, weight="bold")
            fig.tight_layout()
            fig.savefig(outdir / "figure_piecewise_descriptor_splits.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved piecewise split figure to: {outdir / 'figure_piecewise_descriptor_splits.png'}")
    except Exception as exc:
        print(f"WARNING: Could not generate piecewise split figure: {exc}")

    # Stacker trained only on stack-validation real reference labels.
    names = list(pred_v.keys())
    Pv = np.column_stack([pred_v[n] for n in names])
    Pt = np.column_stack([pred_t[n] for n in names])

    stacker = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-6, 5, 60))),
    ])
    stacker.fit(Pv, yval)

    pred_v["stack_reference_validated"] = stacker.predict(Pv)
    pred_t["stack_reference_validated"] = stacker.predict(Pt)

    # Final metrics.
    print("\nFINAL metrics on real held-out reference labels:")
    final_rows = []
    for n in pred_v:
        final_rows.append(metric("VAL " + n, yval, pred_v[n]))
        final_rows.append(metric("TEST " + n, ytest, pred_t[n]))

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(outdir / "FINAL_reference_holdout_metrics.csv", index=False)

    preprocessing_rows = [
        {
            "component": "ExtraTrees and RandomForest",
            "transformation": "Median imputation + variance filtering; no scaling",
            "reason": "Tree split thresholds are scale invariant",
        },
        {
            "component": "HistGradientBoosting",
            "transformation": "Median imputation + variance filtering; no scaling",
            "reason": "Tree-based boosting does not require feature scaling",
        },
        {
            "component": "Linear/Ridge calibration",
            "transformation": "Robust, Standard, and Yeo-Johnson variants compared",
            "reason": "Scaling and skew correction can affect regularized linear models",
        },
        {
            "component": "Global Ridge experts",
            "transformation": "RobustScaler and StandardScaler variants",
            "reason": "Provides scaled regularized alternatives to tree experts",
        },
        {
            "component": "Reference-validated stacker",
            "transformation": "StandardScaler fitted on validation predictions",
            "reason": "Makes Ridge penalties comparable between model outputs",
        },
        {
            "component": "Applicability-domain distance",
            "transformation": "Median imputation + RobustScaler",
            "reason": "Euclidean distances require comparable feature scales",
        },
    ]
    pd.DataFrame(preprocessing_rows).to_csv(
        outdir / "preprocessing_and_scaling_summary.csv",
        index=False,
    )

    # Metrics by domain / source / confidence on final test.
    test_report = ref_test[["smiles", "ref_gap", "pqr_gap", "domain", "ref_source"]].copy()
    for n, p in pred_t.items():
        test_report["pred_" + n] = p
        test_report["abs_err_" + n] = np.abs(p - test_report["ref_gap"])

    test_report.to_csv(outdir / "final_test_predictions.csv", index=False)

    best_col = "abs_err_stack_reference_validated"
    group_rows = []
    for group_col in ["domain", "ref_source"]:
        for key, g in test_report.groupby(group_col):
            group_rows.append({
                "group_type": group_col,
                "group": key,
                "n": len(g),
                "mae_stack": float(g[best_col].mean()),
                "mae_domain_expert": float(g["abs_err_domain_expert"].mean()),
                "mae_global_et": float(g["abs_err_global_et"].mean()),
            })

    pd.DataFrame(group_rows).to_csv(outdir / "mae_by_domain_and_source.csv", index=False)

    # Save all-PQR predictions.
    all_preds = pqr[[
        "smiles",
        "domain",
        "pqr_gap",
        "pred_qm9_aligned_gap",
        "inverse_reconstructed_pqr_gap",
        "cycle_error",
        "cycle_confidence",
        "training_label",
        "label_source",
        "heavy_atoms",
        "calibration_distance",
        "calibration_confidence",
        "final_confidence",
    ]].copy()

    all_preds.to_csv(outdir / "all_pqr_predictions_with_label_source.csv", index=False)

    # Recompute candidates: low-confidence / broad domains / high cycle error.
    cand = all_preds[
        (all_preds["final_confidence"].isin(["low_confidence", "medium_confidence"]))
        | (all_preds["cycle_error"] > 0.50)
    ].copy()

    cand = cand.sort_values(
        ["final_confidence", "cycle_error", "calibration_distance"],
        ascending=[True, False, False],
    )

    cand.to_csv(outdir / "recommended_next_recompute_candidates.csv", index=False)

    if args.save_models:
        joblib.dump({
            "calibration_models": cal_models,
            "calibration_top": top,
            "calibration_weights": weights,
            "global_hgb": g_hgb,
            "global_et": g_et,
            "global_rf": g_rf,
            "stacker": stacker,
            "feature_columns": feat_cols,
            "regime_edges": edges,
        }, outdir / "full_domain_moe_model_bundle.joblib")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Key files:")
    print("  FINAL_reference_holdout_metrics.csv")
    print("  mae_by_domain_and_source.csv")
    print("  all_pqr_predictions_with_label_source.csv")
    print("  recommended_next_recompute_candidates.csv")


if __name__ == "__main__":
    main()
