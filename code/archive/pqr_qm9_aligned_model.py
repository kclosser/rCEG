#!/usr/bin/env python3
"""
pqr_qm9_aligned_model.py

Goal:
Build a QM9-aligned PQR HOMO-LUMO gap model.

Core idea:
1. Define strict QM9-like PQR molecules.
2. Train a QM9-anchor model on that subset.
3. Measure every molecule's similarity to the QM9-like subset.
4. Train domain-specific and gap-regime experts.
5. Stack all realistic predictions using validation data.

This does NOT use HOMO/LUMO as features.
It preserves stereochemistry to avoid the conflict problem caused by removing stereo.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
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


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            e = json.loads(line)
            if isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str):
                out.append(e)
    return out


def safe_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return s if s else smiles
    except Exception:
        return smiles


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


def bond_step_summary(mol):
    try:
        dm = Chem.GetDistanceMatrix(mol).astype(float)
        upper = dm[np.triu_indices_from(dm, k=1)]
        upper = upper[np.isfinite(upper)]
        if len(upper) == 0:
            return [0.0] * 16

        vals = [
            np.max(upper),
            np.mean(upper),
            np.std(upper),
            np.median(upper),
            np.percentile(upper, 25),
            np.percentile(upper, 75),
        ]

        for d in range(1, 11):
            vals.append(float(np.sum(upper == d)) / len(upper))

        return vals
    except Exception:
        return [np.nan] * 16


def classify_domain(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    allowed = all(a in ALLOWED_QM9_ATOMS for a in atoms)
    has_c = any(a == 6 for a in atoms)
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    heavy = mol.GetNumHeavyAtoms()
    single = len(Chem.GetMolFrags(mol)) == 1

    if allowed and has_c and single and charge == 0 and radicals == 0 and heavy <= 9:
        return "qm9_like_strict"
    if allowed and has_c and single and charge == 0 and radicals == 0 and heavy <= 15:
        return "small_organic_loose"
    if allowed and has_c and single and charge == 0:
        return "organic_allowed_atoms"
    if charge != 0 or radicals != 0:
        return "charged_or_radical"
    return "non_qm9_chemistry"


def build_dataset(entries):
    rows = []
    Xs = []
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

        if len(Chem.GetMolFrags(mol)) > 1:
            drops["fragment"] += 1
            continue

        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        domain = classify_domain(mol)

        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []

        feats = []
        feats.extend([fnum(v) for v in pqr[:5]])   # no HOMO/LUMO
        feats.extend([fnum(v) for v in desc])
        feats.extend(rdkit_features(mol))
        feats.extend(bond_step_summary(mol))

        arr = np.array(feats, dtype=np.float64)
        arr[~np.isfinite(arr)] = np.nan
        arr = np.clip(arr, -1e6, 1e6)

        Xs.append(arr.astype(np.float32))

        rows.append({
            "idx": i,
            "smiles": canon,
            "scaffold": safe_scaffold(canon),
            "gap": gap,
            "domain": domain,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })

    max_len = max(len(x) for x in Xs)
    X = np.full((len(Xs), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    y = df["gap"].to_numpy(np.float32)

    print("Drops:", dict(drops))
    print(f"Usable rows: {len(df):,}")
    print("Domain distribution:")
    print(df["domain"].value_counts().to_string())
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"Feature matrix: {X.shape}")

    return df, X, y


class CleanPCA:
    def __init__(self, n_components=40):
        self.n_components = n_components

    def fit(self, X):
        self.imp = SimpleImputer(strategy="median")
        self.scale = RobustScaler()
        X1 = self.imp.fit_transform(X)
        X1 = np.clip(X1, -1e6, 1e6)
        X2 = self.scale.fit_transform(X1)
        n = min(self.n_components, X2.shape[1], X2.shape[0] - 1)
        self.pca = PCA(n_components=n, random_state=42)
        self.pca.fit(X2)
        return self

    def transform(self, X):
        X1 = self.imp.transform(X)
        X1 = np.clip(X1, -1e6, 1e6)
        X2 = self.scale.transform(X1)
        return self.pca.transform(X2)


def make_et(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", ExtraTreesRegressor(
            n_estimators=650,
            max_features=0.35,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_hgb(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=70,
            loss="absolute_error",
            random_state=seed,
        )),
    ])


def group_split(df, y, group_col, test_size, seed):
    idx = np.arange(len(df))
    groups = df[group_col].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(idx, y, groups))
    return tr, te


def weighted_knn_predict(Z_train, y_train, Z_query, k=25):
    k = min(k, len(y_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Z_train)
    dist, ind = nn.kneighbors(Z_query)

    w = 1.0 / (dist + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    return (w * y_train[ind]).sum(axis=1)


def make_gap_regimes(y_train, k):
    edges = np.percentile(y_train, np.linspace(0, 100, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_regime(y_or_pred, edges):
    return np.digitize(y_or_pred, edges[1:-1], right=False)


def eval_metrics(name, y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"{name:32s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_qm9_aligned")
    ap.add_argument("--test-group", choices=["smiles", "scaffold"], default="smiles")
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--min-domain-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl(args.data)
    df, X, y = build_dataset(entries)

    trainval_idx, test_idx = group_split(df, y, args.test_group, 0.15, args.seed)
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]

    tr_rel, val_rel = group_split(trainval_df, y_trainval, args.test_group, 0.1765, args.seed + 1)

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[val_rel]

    print(f"\nSplit sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    Xtr, Xva, Xte = X[train_idx], X[val_idx], X[test_idx]
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    dtr = df.iloc[train_idx]["domain"].to_numpy()
    dva = df.iloc[val_idx]["domain"].to_numpy()
    dte = df.iloc[test_idx]["domain"].to_numpy()

    # PCA coordinate system used for QM9-likeness and kNN features.
    pcamap = CleanPCA(n_components=40).fit(Xtr)
    Ztr = pcamap.transform(Xtr)
    Zva = pcamap.transform(Xva)
    Zte = pcamap.transform(Xte)

    qm_mask_tr = dtr == "qm9_like_strict"
    if qm_mask_tr.sum() < 200:
        print("WARNING: too few QM9-like training rows; QM9 anchor may be weak.")

    qm_centroid = Ztr[qm_mask_tr].mean(axis=0)
    qm_std = Ztr[qm_mask_tr].std(axis=0)
    qm_std[qm_std < 1e-8] = 1.0

    def qm_distance(Z):
        return np.sqrt(np.mean(((Z - qm_centroid) / qm_std) ** 2, axis=1))

    qm_dist_tr = qm_distance(Ztr)
    qm_dist_va = qm_distance(Zva)
    qm_dist_te = qm_distance(Zte)

    # Global models.
    print("\nTraining global models...")
    global_et = make_et(args.seed)
    global_hgb = make_hgb(args.seed + 10)

    global_et.fit(Xtr, ytr)
    global_hgb.fit(Xtr, ytr)

    pred = {}
    pred["global_et_val"] = global_et.predict(Xva)
    pred["global_et_test"] = global_et.predict(Xte)
    pred["global_hgb_val"] = global_hgb.predict(Xva)
    pred["global_hgb_test"] = global_hgb.predict(Xte)

    global_mean_val = (pred["global_et_val"] + pred["global_hgb_val"]) / 2
    global_mean_test = (pred["global_et_test"] + pred["global_hgb_test"]) / 2

    # QM9-anchor model.
    print("\nTraining QM9-anchor model...")
    if qm_mask_tr.sum() >= 200:
        qm_anchor = make_et(args.seed + 20)
        qm_anchor.fit(Xtr[qm_mask_tr], ytr[qm_mask_tr])
        qm_anchor_val = qm_anchor.predict(Xva)
        qm_anchor_test = qm_anchor.predict(Xte)
    else:
        qm_anchor = None
        qm_anchor_val = global_mean_val.copy()
        qm_anchor_test = global_mean_test.copy()

    # Domain experts.
    print("\nTraining domain experts...")
    domain_experts = {}
    domain_pred_val = np.zeros(len(yva), dtype=np.float32)
    domain_pred_test = np.zeros(len(yte), dtype=np.float32)

    for domain in sorted(df["domain"].unique()):
        mask = dtr == domain
        n = int(mask.sum())

        if n >= args.min_domain_n:
            print(f"  domain {domain}: training expert on {n:,} rows")
            model = make_et(args.seed + 100 + len(domain_experts))
            model.fit(Xtr[mask], ytr[mask])
            domain_experts[domain] = model
        else:
            print(f"  domain {domain}: n={n:,}, using global fallback")

    for domain in sorted(df["domain"].unique()):
        va_mask = dva == domain
        te_mask = dte == domain

        model = domain_experts.get(domain)
        if model is None:
            domain_pred_val[va_mask] = global_mean_val[va_mask]
            domain_pred_test[te_mask] = global_mean_test[te_mask]
        else:
            domain_pred_val[va_mask] = model.predict(Xva[va_mask])
            domain_pred_test[te_mask] = model.predict(Xte[te_mask])

    # Gap-regime experts routed by global prediction.
    print("\nTraining gap-regime experts...")
    edges = make_gap_regimes(ytr, args.regimes)
    reg_tr = assign_regime(ytr, edges)
    reg_va_pred = assign_regime(global_mean_val, edges)
    reg_te_pred = assign_regime(global_mean_test, edges)

    regime_pred_val = np.zeros(len(yva), dtype=np.float32)
    regime_pred_test = np.zeros(len(yte), dtype=np.float32)

    for r in range(args.regimes):
        mask = reg_tr == r
        n = int(mask.sum())
        print(f"  regime {r}: {edges[r]:.3f} to {edges[r+1]:.3f}, n={n:,}")
        model = make_et(args.seed + 300 + r)
        model.fit(Xtr[mask], ytr[mask])

        va_mask = reg_va_pred == r
        te_mask = reg_te_pred == r
        if va_mask.any():
            regime_pred_val[va_mask] = model.predict(Xva[va_mask])
        if te_mask.any():
            regime_pred_test[te_mask] = model.predict(Xte[te_mask])

    # KNN features.
    print("\nBuilding kNN similarity predictors...")
    knn_all_val = weighted_knn_predict(Ztr, ytr, Zva, k=35)
    knn_all_test = weighted_knn_predict(Ztr, ytr, Zte, k=35)

    if qm_mask_tr.sum() >= 50:
        knn_qm_val = weighted_knn_predict(Ztr[qm_mask_tr], ytr[qm_mask_tr], Zva, k=25)
        knn_qm_test = weighted_knn_predict(Ztr[qm_mask_tr], ytr[qm_mask_tr], Zte, k=25)
    else:
        knn_qm_val = global_mean_val.copy()
        knn_qm_test = global_mean_test.copy()

    # Domain one-hot features.
    domains = sorted(df["domain"].unique())
    domain_to_i = {d: i for i, d in enumerate(domains)}

    def onehot(ds):
        out = np.zeros((len(ds), len(domains)), dtype=np.float32)
        for i, d in enumerate(ds):
            out[i, domain_to_i[d]] = 1.0
        return out

    # Auto-disable QM9-anchor features if they are worse than the global model.
    # This prevents a bad out-of-domain QM9 anchor from contaminating the stacker.
    from sklearn.metrics import mean_absolute_error as _mae

    global_val_mae = _mae(yva, global_mean_val)
    qm_anchor_val_mae = _mae(yva, qm_anchor_val)
    knn_qm_val_mae = _mae(yva, knn_qm_val)

    use_qm_anchor = qm_anchor_val_mae <= global_val_mae
    use_knn_qm = knn_qm_val_mae <= global_val_mae

    print(f"\nFeature gating:")
    print(f"  global_mean VAL MAE = {global_val_mae:.4f}")
    print(f"  qm_anchor   VAL MAE = {qm_anchor_val_mae:.4f} -> {'KEEP' if use_qm_anchor else 'DROP'}")
    print(f"  knn_qm      VAL MAE = {knn_qm_val_mae:.4f} -> {'KEEP' if use_knn_qm else 'DROP'}")

    stack_features_val = [
        pred["global_et_val"],
        pred["global_hgb_val"],
        global_mean_val,
        domain_pred_val,
        regime_pred_val,
        knn_all_val,
        qm_dist_va,
    ]

    stack_features_test = [
        pred["global_et_test"],
        pred["global_hgb_test"],
        global_mean_test,
        domain_pred_test,
        regime_pred_test,
        knn_all_test,
        qm_dist_te,
    ]

    if use_qm_anchor:
        stack_features_val.append(qm_anchor_val)
        stack_features_test.append(qm_anchor_test)

    if use_knn_qm:
        stack_features_val.append(knn_qm_val)
        stack_features_test.append(knn_qm_test)

    stack_features_val.append(onehot(dva))
    stack_features_test.append(onehot(dte))

    # Stack realistic predictions using validation only.
    P_val = np.column_stack(stack_features_val)
    P_test = np.column_stack(stack_features_test)

    stacker = RidgeCV(alphas=np.logspace(-6, 4, 50))
    stacker.fit(P_val, yva)

    stack_val = stacker.predict(P_val)
    stack_test = stacker.predict(P_test)

    print("\nOverall results:")
    results = []
    results.append(eval_metrics("VAL global_et", yva, pred["global_et_val"]))
    results.append(eval_metrics("TEST global_et", yte, pred["global_et_test"]))
    results.append(eval_metrics("VAL global_hgb", yva, pred["global_hgb_val"]))
    results.append(eval_metrics("TEST global_hgb", yte, pred["global_hgb_test"]))
    results.append(eval_metrics("VAL global_mean", yva, global_mean_val))
    results.append(eval_metrics("TEST global_mean", yte, global_mean_test))
    results.append(eval_metrics("VAL qm_anchor", yva, qm_anchor_val))
    results.append(eval_metrics("TEST qm_anchor", yte, qm_anchor_test))
    results.append(eval_metrics("VAL domain_expert", yva, domain_pred_val))
    results.append(eval_metrics("TEST domain_expert", yte, domain_pred_test))
    results.append(eval_metrics("VAL regime_by_global", yva, regime_pred_val))
    results.append(eval_metrics("TEST regime_by_global", yte, regime_pred_test))
    results.append(eval_metrics("VAL knn_all", yva, knn_all_val))
    results.append(eval_metrics("TEST knn_all", yte, knn_all_test))
    results.append(eval_metrics("VAL knn_qm_anchor", yva, knn_qm_val))
    results.append(eval_metrics("TEST knn_qm_anchor", yte, knn_qm_test))
    results.append(eval_metrics("VAL QM9_ALIGNED_STACK", yva, stack_val))
    results.append(eval_metrics("TEST QM9_ALIGNED_STACK", yte, stack_test))

    pd.DataFrame(results).to_csv(outdir / "metrics.csv", index=False)

    # Subgroup results.
    subgroup_rows = []
    for domain in domains:
        mask = dte == domain
        if mask.sum() == 0:
            continue
        subgroup_rows.append({
            "domain": domain,
            "n_test": int(mask.sum()),
            "target_mean": float(yte[mask].mean()),
            "target_sd": float(yte[mask].std()),
            "global_mean_mae": float(mean_absolute_error(yte[mask], global_mean_test[mask])),
            "domain_expert_mae": float(mean_absolute_error(yte[mask], domain_pred_test[mask])),
            "stack_mae": float(mean_absolute_error(yte[mask], stack_test[mask])),
            "qm_distance_mean": float(qm_dist_te[mask].mean()),
        })

    pd.DataFrame(subgroup_rows).to_csv(outdir / "test_mae_by_domain.csv", index=False)

    test_report = df.iloc[test_idx].copy()
    test_report["pred_global_mean"] = global_mean_test
    test_report["pred_qm_anchor"] = qm_anchor_test
    test_report["pred_domain"] = domain_pred_test
    test_report["pred_regime"] = regime_pred_test
    test_report["pred_stack"] = stack_test
    test_report["qm_distance"] = qm_dist_te
    test_report["abs_error_stack"] = np.abs(stack_test - yte)
    test_report.sort_values("abs_error_stack", ascending=False).to_csv(outdir / "test_error_report.csv", index=False)

    joblib.dump({
        "global_et": global_et,
        "global_hgb": global_hgb,
        "qm_anchor": qm_anchor,
        "domain_experts": domain_experts,
        "stacker": stacker,
        "domains": domains,
        "edges": edges,
        "pcamap": pcamap,
    }, outdir / "qm9_aligned_model.joblib")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Main files:")
    print("  metrics.csv")
    print("  test_mae_by_domain.csv")
    print("  test_error_report.csv")
    print("  qm9_aligned_model.joblib")


if __name__ == "__main__":
    main()
