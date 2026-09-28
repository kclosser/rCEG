#!/usr/bin/env python3
"""
Ultimate PQR calibrated-regime model.

Core idea:
1. Load PQR.
2. Load QM9/reference gaps.
3. Find exact overlapping molecules.
4. Train calibration models only on overlap-train.
5. Hold out overlap-test as real external validation.
6. Convert PQR labels to QM9-calibrated labels.
7. Train global + regime experts on calibrated PQR labels.
8. Stack models using overlap-validation real QM9 labels.
9. Report final MAE on overlap-test real QM9 labels.

No HOMO/LUMO leakage:
- pqr[5] and pqr[6] are never used as model features.
- They are used only to audit descriptor leakage.
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

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


HARTREE_TO_EV = 27.211386245988


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def canon(smiles, isomeric=True):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


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



def classify_domain(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    allowed_qm9 = {1, 6, 7, 8, 9}
    allowed = all(a in allowed_qm9 for a in atoms)
    has_carbon = any(a == 6 for a in atoms)
    heavy = mol.GetNumHeavyAtoms()
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    single_fragment = len(Chem.GetMolFrags(mol)) == 1

    if allowed and has_carbon and single_fragment and charge == 0 and radicals == 0 and heavy <= 9:
        return "qm9_like_small_organic"
    if allowed and has_carbon and single_fragment and charge == 0 and radicals == 0 and heavy <= 20:
        return "near_qm9_larger_organic"
    if allowed and has_carbon and single_fragment and charge == 0:
        return "large_neutral_organic"
    if charge != 0 or radicals != 0:
        return "charged_or_radical"
    return "heteroatom_rich_non_qm9"


def load_pqr(path):
    rows = []
    feats = []
    audit = []
    drops = Counter()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue

            e = json.loads(line)
            if not (isinstance(e, list) and len(e) >= 5):
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
            smiles_noiso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

            # FEATURES: first five PQR values only.
            # DO NOT use pqr[5] or pqr[6], which are HOMO/LUMO.
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
                "smiles_noiso": smiles_noiso,
                "pqr_gap": pqr_gap,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "domain": classify_domain(mol),
            })
            feats.append(arr.astype(np.float32))

            if len(pqr) >= 7:
                audit.append({
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

    agg = {"smiles_noiso": "first", "pqr_gap": "median", "heavy_atoms": "median", "domain": "first"}
    for c in feat_cols:
        agg[c] = "median"

    df = df.groupby("smiles", as_index=False).agg(agg)

    print(f"PQR loaded: {len(df):,} unique stereo-preserved molecules")
    print(f"Drops: {dict(drops)}")
    print(f"Feature columns before leakage filter: {len(feat_cols):,}")

    audit_df = pd.DataFrame(audit).drop_duplicates("smiles")
    return df, audit_df


def load_ref(path):
    ref = pd.read_csv(path)

    if "smiles" not in ref.columns:
        raise ValueError("Reference CSV must contain a smiles column.")

    gap_col = None
    for c in ["gap", "homo_lumo_gap", "HOMO_LUMO_gap", "deltaE", "DeltaEHL", "gap_ev"]:
        if c in ref.columns:
            gap_col = c
            break

    if gap_col is None:
        raise ValueError("Reference CSV needs a gap column.")

    rows = []
    for _, r in ref.iterrows():
        try:
            smi = canon(r["smiles"], isomeric=True)
            smi_no = canon(r["smiles"], isomeric=False)
            g = fnum(r[gap_col])
            if smi is not None and math.isfinite(g):
                rows.append({"smiles": smi, "smiles_noiso": smi_no, "ref_gap": g})
        except Exception:
            pass

    out = pd.DataFrame(rows)
    out = out.groupby("smiles", as_index=False).agg({
        "smiles_noiso": "first",
        "ref_gap": "median",
    })

    print(f"Reference loaded: {len(out):,} molecules")
    return out


def filter_leakage_features(df, audit_df, threshold=0.98):
    feat_cols = [c for c in df.columns if c.startswith("x")]

    merged = df[["smiles"] + feat_cols].merge(audit_df, on="smiles", how="inner")
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

    audit = pd.DataFrame(records).sort_values("abs_corr", ascending=False)
    keep = [c for c in feat_cols if c not in bad]

    print(f"Leakage audit: dropped {len(bad)} feature columns with abs_corr >= {threshold}")
    print("Top audit rows:")
    print(audit.head(15).to_string(index=False))

    return keep, audit


def make_preprocessor_model(model):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", RobustScaler(with_centering=False)),
        ("model", model),
    ])


def make_et(seed=42, n=700):
    return make_preprocessor_model(ExtraTreesRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    ))


def make_rf(seed=42, n=500):
    return make_preprocessor_model(RandomForestRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    ))


def make_hgb(seed=42):
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


def make_ridge():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", RidgeCV(alphas=np.logspace(-6, 4, 50))),
    ])


def metric(name, y, pred):
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name:34s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def quantile_edges(y, k):
    edges = np.percentile(y, np.linspace(0, 100, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign(y, edges):
    return np.digitize(y, edges[1:-1], right=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqr", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--outdir", default="runs/pqr_ultimate_calibrated_regime")
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--leakage-corr-threshold", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pqr, audit_df = load_pqr(args.pqr)
    ref = load_ref(args.ref)

    feat_cols, leak_audit = filter_leakage_features(pqr, audit_df, args.leakage_corr_threshold)
    leak_audit.to_csv(outdir / "feature_leakage_audit.csv", index=False)

    overlap = pqr.merge(ref[["smiles", "ref_gap"]], on="smiles", how="inner")
    overlap.to_csv(outdir / "overlap_exact_isomeric.csv", index=False)

    print(f"\nExact PQR/QM9 overlap: {len(overlap):,}")
    print(f"PQR overlap mean/sd: {overlap.pqr_gap.mean():.3f}/{overlap.pqr_gap.std():.3f}")
    print(f"REF overlap mean/sd: {overlap.ref_gap.mean():.3f}/{overlap.ref_gap.std():.3f}")
    print(f"REF-PQR mean/sd: {(overlap.ref_gap - overlap.pqr_gap).mean():.3f}/{(overlap.ref_gap - overlap.pqr_gap).std():.3f}")

    if len(overlap) < 200:
        raise RuntimeError("Not enough exact overlap for a serious calibration/regime model.")

    # Split real overlap into calibration train, stack validation, final real test.
    idx = np.arange(len(overlap))
    cal_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed + 1)

    cal = overlap.iloc[cal_idx].copy()
    val = overlap.iloc[val_idx].copy()
    test = overlap.iloc[test_idx].copy()

    print(f"\nOverlap split: calibration={len(cal):,}, stack_val={len(val):,}, final_test={len(test):,}")

    X_cal_full = cal[["pqr_gap"] + feat_cols].to_numpy()
    X_val_full = val[["pqr_gap"] + feat_cols].to_numpy()
    X_test_full = test[["pqr_gap"] + feat_cols].to_numpy()

    X_cal_gap = cal[["pqr_gap"]].to_numpy()
    X_val_gap = val[["pqr_gap"]].to_numpy()
    X_test_gap = test[["pqr_gap"]].to_numpy()

    y_cal = cal["ref_gap"].to_numpy()
    y_val = val["ref_gap"].to_numpy()
    y_test = test["ref_gap"].to_numpy()

    calibrators = {
        "cal_gap_linear": (make_preprocessor_model(LinearRegression()), X_cal_gap, X_val_gap, X_test_gap),
        "cal_gap_ridge": (make_ridge(), X_cal_gap, X_val_gap, X_test_gap),
        "cal_desc_ridge": (make_ridge(), X_cal_full, X_val_full, X_test_full),
        "cal_desc_hgb": (make_hgb(args.seed + 10), X_cal_full, X_val_full, X_test_full),
        "cal_desc_et": (make_et(args.seed + 20, n=900), X_cal_full, X_val_full, X_test_full),
        "cal_desc_rf": (make_rf(args.seed + 30, n=600), X_cal_full, X_val_full, X_test_full),
    }

    print("\nCalibration models evaluated on real overlap-validation:")
    cal_rows = []
    cal_models = {}
    cal_val_preds = {}
    cal_test_preds = {}

    for name, (model, Xc, Xv, Xt) in calibrators.items():
        model.fit(Xc, y_cal)
        pv = model.predict(Xv)
        pt = model.predict(Xt)
        row = metric("VAL " + name, y_val, pv)
        test_row = metric("TEST " + name, y_test, pt)
        row["test_mae"] = test_row["mae"]
        row["test_r2"] = test_row["r2"]
        cal_rows.append(row)
        cal_models[name] = model
        cal_val_preds[name] = pv
        cal_test_preds[name] = pt

    cal_metrics = pd.DataFrame(cal_rows).sort_values("mae")
    cal_metrics.to_csv(outdir / "calibration_model_metrics.csv", index=False)

    # Ensemble top three calibrators by validation MAE.
    top_names = list(cal_metrics.head(3)["name"].str.replace("VAL ", "", regex=False))
    val_maes = np.array([cal_metrics[cal_metrics["name"] == "VAL " + n]["mae"].iloc[0] for n in top_names])
    weights = 1 / np.maximum(val_maes, 1e-6)
    weights = weights / weights.sum()

    print("\nCalibration ensemble:")
    for n, w in zip(top_names, weights):
        print(f"  {n}: weight={w:.3f}")

    cal_ens_val = sum(w * cal_val_preds[n] for n, w in zip(top_names, weights))
    cal_ens_test = sum(w * cal_test_preds[n] for n, w in zip(top_names, weights))

    ens_rows = []
    ens_rows.append(metric("VAL calibration_ensemble", y_val, cal_ens_val))
    ens_rows.append(metric("TEST calibration_ensemble", y_test, cal_ens_test))
    pd.DataFrame(ens_rows).to_csv(outdir / "calibration_ensemble_metrics.csv", index=False)

    # Apply calibration ensemble to all PQR.
    X_all_full = pqr[["pqr_gap"] + feat_cols].to_numpy()
    X_all_gap = pqr[["pqr_gap"]].to_numpy()

    pseudo = np.zeros(len(pqr), dtype=float)
    for n, w in zip(top_names, weights):
        if n in ["cal_gap_linear", "cal_gap_ridge"]:
            pseudo += w * cal_models[n].predict(X_all_gap)
        else:
            pseudo += w * cal_models[n].predict(X_all_full)

    pqr["calibrated_gap"] = pseudo

    # Applicability-domain scoring:
    # distance to the exact PQR/QM9 calibration-training overlap in feature space.
    ad_imp = SimpleImputer(strategy="median")
    ad_scale = RobustScaler()
    X_ad_cal = ad_imp.fit_transform(cal[feat_cols].to_numpy())
    X_ad_cal = ad_scale.fit_transform(X_ad_cal)

    X_ad_all = ad_imp.transform(pqr[feat_cols].to_numpy())
    X_ad_all = ad_scale.transform(X_ad_all)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_ad_cal)
    dist_all, _ = nn.kneighbors(X_ad_all)
    pqr["calibration_distance"] = dist_all[:, 0]

    # Use overlap validation/test distances to set confidence thresholds.
    X_ad_val = ad_scale.transform(ad_imp.transform(val[feat_cols].to_numpy()))
    X_ad_test = ad_scale.transform(ad_imp.transform(test[feat_cols].to_numpy()))
    d_val, _ = nn.kneighbors(X_ad_val)
    d_test, _ = nn.kneighbors(X_ad_test)
    reference_dist = np.concatenate([d_val[:, 0], d_test[:, 0]])

    high_thr = float(np.quantile(reference_dist, 0.90))
    med_thr = float(np.quantile(reference_dist, 0.975))

    def _conf(d):
        if d <= high_thr:
            return "high_in_domain"
        if d <= med_thr:
            return "medium_near_domain"
        return "low_out_of_domain"

    pqr["calibration_confidence"] = [
        _conf(d) for d in pqr["calibration_distance"].to_numpy()
    ]

    print("\nCalibration applicability domain:")
    print(f"  high-confidence threshold:   {high_thr:.4f}")
    print(f"  medium-confidence threshold: {med_thr:.4f}")
    print(pqr["calibration_confidence"].value_counts().to_string())

    # Replace calibration-overlap molecules with real QM9 values.
    ref_map = dict(cal[["smiles", "ref_gap"]].values)
    pqr["used_real_ref_label"] = pqr["smiles"].isin(ref_map)
    pqr.loc[pqr["used_real_ref_label"], "calibrated_gap"] = pqr.loc[pqr["used_real_ref_label"], "smiles"].map(ref_map)

    # Exclude overlap validation and final test from final training.
    holdout_smiles = set(val["smiles"]) | set(test["smiles"])
    train_pool = pqr[~pqr["smiles"].isin(holdout_smiles)].copy()

    X_train = train_pool[feat_cols].to_numpy()
    y_train = train_pool["calibrated_gap"].to_numpy()

    X_val_final = val[feat_cols].to_numpy()
    X_test_final = test[feat_cols].to_numpy()

    print(f"\nFinal training pool: {len(train_pool):,}")
    print(f"Stack validation real QM9 molecules: {len(val):,}")
    print(f"Final test real QM9 molecules: {len(test):,}")

    # Global final models.
    print("\nTraining final global calibrated models...")
    global_hgb = make_hgb(args.seed + 100)
    global_et = make_et(args.seed + 110, n=900)
    global_rf = make_rf(args.seed + 120, n=600)

    global_hgb.fit(X_train, y_train)
    global_et.fit(X_train, y_train)
    global_rf.fit(X_train, y_train)

    pred_val = {}
    pred_test = {}

    pred_val["global_hgb"] = global_hgb.predict(X_val_final)
    pred_test["global_hgb"] = global_hgb.predict(X_test_final)

    pred_val["global_et"] = global_et.predict(X_val_final)
    pred_test["global_et"] = global_et.predict(X_test_final)

    pred_val["global_rf"] = global_rf.predict(X_val_final)
    pred_test["global_rf"] = global_rf.predict(X_test_final)

    pred_val["global_mean"] = (pred_val["global_hgb"] + pred_val["global_et"] + pred_val["global_rf"]) / 3
    pred_test["global_mean"] = (pred_test["global_hgb"] + pred_test["global_et"] + pred_test["global_rf"]) / 3

    # Domain-aware experts: one expert per chemical/calibration domain.
    print("\nTraining domain-aware experts...")
    domain_val = np.zeros(len(val), dtype=float)
    domain_test = np.zeros(len(test), dtype=float)

    train_domains = train_pool["domain"].to_numpy()
    val_domains = val["domain"].to_numpy()
    test_domains = test["domain"].to_numpy()

    for domain in sorted(train_pool["domain"].unique()):
        train_mask = train_domains == domain
        val_mask = val_domains == domain
        test_mask = test_domains == domain
        n_domain = int(train_mask.sum())

        if n_domain >= 500:
            print(f"  domain {domain}: training expert on {n_domain:,} rows")
            expert = make_et(args.seed + 700 + len(domain), n=500)
            expert.fit(X_train[train_mask], y_train[train_mask])

            if val_mask.any():
                domain_val[val_mask] = expert.predict(X_val_final[val_mask])
            if test_mask.any():
                domain_test[test_mask] = expert.predict(X_test_final[test_mask])
        else:
            print(f"  domain {domain}: only {n_domain:,} rows, using global_mean fallback")
            if val_mask.any():
                domain_val[val_mask] = pred_val["global_mean"][val_mask]
            if test_mask.any():
                domain_test[test_mask] = pred_test["global_mean"][test_mask]

    pred_val["domain_expert"] = domain_val
    pred_test["domain_expert"] = domain_test

    # Regime experts routed by global_mean.
    print("\nTraining calibrated-regime experts...")
    edges = quantile_edges(y_train, args.regimes)
    train_reg = assign(y_train, edges)
    val_route = assign(pred_val["global_mean"], edges)
    test_route = assign(pred_test["global_mean"], edges)

    regime_val = np.zeros(len(val), dtype=float)
    regime_test = np.zeros(len(test), dtype=float)

    for r in range(args.regimes):
        mask = train_reg == r
        print(f"  regime {r}: {edges[r]:.3f} to {edges[r+1]:.3f}, n={int(mask.sum()):,}")
        expert = make_et(args.seed + 300 + r, n=500)
        expert.fit(X_train[mask], y_train[mask])

        mv = val_route == r
        mt = test_route == r
        if mv.any():
            regime_val[mv] = expert.predict(X_val_final[mv])
        if mt.any():
            regime_test[mt] = expert.predict(X_test_final[mt])

    pred_val["regime_by_global"] = regime_val
    pred_test["regime_by_global"] = regime_test

    # Stack using real QM9 labels on overlap-validation only.
    P_val = np.column_stack([pred_val[k] for k in pred_val.keys()])
    P_test = np.column_stack([pred_test[k] for k in pred_val.keys()])

    stacker = RidgeCV(alphas=np.logspace(-6, 4, 50))
    stacker.fit(P_val, y_val)

    pred_val["stack_real_qm9"] = stacker.predict(P_val)
    pred_test["stack_real_qm9"] = stacker.predict(P_test)

    print("\nFINAL evaluation against real QM9/reference labels:")
    final_rows = []
    for k in pred_val:
        final_rows.append(metric("VAL " + k, y_val, pred_val[k]))
        final_rows.append(metric("TEST " + k, y_test, pred_test[k]))

    pd.DataFrame(final_rows).to_csv(outdir / "FINAL_real_qm9_holdout_metrics.csv", index=False)

    report = test[["smiles", "pqr_gap", "ref_gap"]].copy()
    for k, v in pred_test.items():
        report["pred_" + k] = v
        report["abs_err_" + k] = np.abs(v - report["ref_gap"])
    report.to_csv(outdir / "final_test_predictions.csv", index=False)

    pqr[[
        "smiles",
        "domain",
        "pqr_gap",
        "calibrated_gap",
        "used_real_ref_label",
        "heavy_atoms",
        "calibration_distance",
        "calibration_confidence",
    ]].to_csv(outdir / "pqr_calibrated_labels_with_confidence.csv", index=False)

    if args.save_models:
        joblib.dump({
            "calibration_models": cal_models,
            "calibration_top_names": top_names,
            "calibration_weights": weights,
            "global_hgb": global_hgb,
            "global_et": global_et,
            "global_rf": global_rf,
            "stacker": stacker,
            "feature_columns": feat_cols,
            "regime_edges": edges,
        }, outdir / "ultimate_model_bundle.joblib")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Most important file:")
    print("  FINAL_real_qm9_holdout_metrics.csv")


if __name__ == "__main__":
    main()
