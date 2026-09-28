#!/usr/bin/env python3
"""
pqr_gap_oof_gated_moe.py

OOF-prediction-gated mixture of experts.

Your regime experiment showed:
- ORACLE_ROUTE test MAE ~0.166 eV
- realistic soft/hard route ~1.0 eV

So the experts work if the correct regime is known. The bottleneck is routing.
This script improves routing by adding out-of-fold base-model predicted gaps as
router features.

Usage:
cd ~/Downloads/Closser
python pqr_gap_oof_gated_moe.py --data enhanced_dataset_lasso_STRICT.jsonl --outdir runs/pqr_oof_gated_quick --max-rows 30000 --n-regimes 8
"""

from __future__ import annotations

import argparse, json, math
from collections import Counter
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV

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


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.1, hi=99.9):
        self.lo = lo
        self.hi = hi
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        self.lo_ = np.where(np.isfinite(self.lo_), self.lo_, -1e6)
        self.hi_ = np.where(np.isfinite(self.hi_), self.hi_, 1e6)
        same = self.hi_ <= self.lo_
        self.hi_[same] = self.lo_[same] + 1e-6
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.clip(X, self.lo_, self.hi_)


SMARTS = {
    "aromatic": "a", "hetero_aromatic": "[a;!#6]", "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N", "ester": "C(=O)O", "acid": "C(=O)[OX2H1]",
    "nitrile": "C#N", "alkene": "C=C", "alkyne": "C#C",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]", "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][#6]", "fluoro": "[F]", "chloro": "[Cl]",
    "sulfur": "[S]", "nitro": "[NX3](=O)=O", "phenol_like": "aO",
    "aniline_like": "aN",
}


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


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
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                    out.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], list):
                    out.extend(obj)
            except Exception:
                pass
    return out


def smarts_flags(mol):
    vals = []
    for smarts in SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        vals.append(float(patt is not None and mol.HasSubstructMatch(patt)))
    return vals


def rdkit_features(mol):
    atoms, bonds = list(mol.GetAtoms()), list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol), Descriptors.NumRadicalElectrons(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol), Crippen.MolMR(mol), Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHeteroatoms(mol), Lipinski.FractionCSP3(mol),
        sum(a.GetIsAromatic() for a in atoms), sum(b.GetIsAromatic() for b in bonds),
        sum(b.GetBondTypeAsDouble() == 2 for b in bonds), sum(b.GetBondTypeAsDouble() == 3 for b in bonds),
        sum(a.GetAtomicNum() == 6 for a in atoms), sum(a.GetAtomicNum() == 7 for a in atoms),
        sum(a.GetAtomicNum() == 8 for a in atoms), sum(a.GetAtomicNum() == 9 for a in atoms),
        sum(a.GetAtomicNum() == 16 for a in atoms), sum(a.GetAtomicNum() == 17 for a in atoms),
    ]


def morgan_arr(mol, gen, n_bits):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def coarse_class(mol):
    flags = dict(zip(SMARTS.keys(), smarts_flags(mol)))
    if flags["aromatic"] or flags["carbonyl"] or flags["amide"] or flags["ester"]:
        return "aromatic_or_carbonyl"
    if flags["alkene"] or flags["alkyne"] or flags["nitrile"]:
        return "single_unsaturated"
    return "saturated"


def build_dataset(entries, n_bits=2048, max_rows=None, seed=42):
    rng = np.random.default_rng(seed)
    if max_rows is not None and len(entries) > max_rows:
        idx = rng.choice(len(entries), max_rows, replace=False)
        entries = [entries[i] for i in idx]
        print(f"Subsampled to {len(entries):,} raw rows")

    gen2, gen3 = GetMorganGenerator(radius=2, fpSize=n_bits), GetMorganGenerator(radius=3, fpSize=n_bits)
    rows, Xs, drops = [], [], Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drops["bad_entry_shape"] += 1; continue
        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < 0.10 or gap > 25:
            drops["bad_gap"] += 1; continue
        mol = Chem.MolFromSmiles(e[1], sanitize=True)
        if mol is None:
            drops["invalid_smiles"] += 1; continue
        if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms()):
            drops["isolated_hydrogen"] += 1; continue
        if len(Chem.GetMolFrags(mol)) > 1:
            drops["multi_fragment"] += 1; continue

        canon = Chem.MolToSmiles(mol, canonical=True)
        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []
        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso = [fnum(v) for v in desc]

        x = np.concatenate([
            np.array(pqr5, dtype=np.float32),
            np.array(lasso, dtype=np.float32),
            np.array(rdkit_features(mol), dtype=np.float32),
            np.array(smarts_flags(mol), dtype=np.float32),
            morgan_arr(mol, gen2, n_bits),
            morgan_arr(mol, gen3, n_bits),
        ])
        rows.append({
            "idx": i, "smiles": canon, "gap": gap, "coarse_class": coarse_class(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(), "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
        Xs.append(x)

    max_len = max(len(x) for x in Xs)
    X = np.full((len(Xs), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x
    df = pd.DataFrame(rows)
    y, groups = df["gap"].to_numpy(np.float32), df["smiles"].to_numpy()

    print(f"Usable rows: {len(df):,}")
    print(f"Drops: {dict(drops)}")
    print("Coarse class distribution:")
    print(df["coarse_class"].value_counts().to_string())
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"Feature matrix: {X.shape}")
    return df, X, y, groups


def group_split(groups, y, test_size=0.15, val_size=0.15, seed=42):
    idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(gss1.split(idx, y, groups))
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_rel, va_rel = next(gss2.split(trainval, y[trainval], groups[trainval]))
    return trainval[tr_rel], trainval[va_rel], test


def make_hgb_reg(seed=42, k_best=2500):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper()),
        ("variance", VarianceThreshold(1e-12)),
        ("select", SelectKBest(f_regression, k=k_best)),
        ("model", HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.03,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=80,
            random_state=seed, loss="absolute_error",
        )),
    ])


def make_et_reg(seed=42):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper()),
        ("variance", VarianceThreshold(1e-12)),
        ("model", ExtraTreesRegressor(
            n_estimators=500, max_features=0.30, min_samples_leaf=1, n_jobs=-1, random_state=seed,
        )),
    ])


def make_router(kind="hgb", seed=42, k_best=2500):
    if kind == "hgb":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clip", PercentileClipper()),
            ("variance", VarianceThreshold(1e-12)),
            ("select", SelectKBest(f_classif, k=k_best)),
            ("model", HistGradientBoostingClassifier(
                max_iter=900, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.03,
                early_stopping=True, validation_fraction=0.12, n_iter_no_change=70, random_state=seed,
            )),
        ])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper()),
        ("variance", VarianceThreshold(1e-12)),
        ("model", ExtraTreesClassifier(
            n_estimators=600, max_features=0.35, min_samples_leaf=1,
            n_jobs=-1, random_state=seed, class_weight="balanced",
        )),
    ])


def eval_pred(name, y_true, pred):
    mae, r2 = mean_absolute_error(y_true, pred), r2_score(y_true, pred)
    print(f"{name:38s} MAE={mae:.4f} eV   R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def make_regimes(y_train, n_regimes):
    edges = np.percentile(y_train, np.linspace(0, 100, n_regimes + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    return np.array(out)


def apply_regimes(y, edges):
    return np.digitize(y, edges[1:-1], right=False)


def make_oof_predictions(X_train, y_train, groups_train, X_val, X_test, args):
    n = len(y_train)
    oof_hgb, oof_et = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    n_splits = min(args.oof_folds, len(np.unique(groups_train)))
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\nBuilding OOF base predictions with {n_splits} folds...")
    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups_train), 1):
        print(f"  OOF fold {fold}/{n_splits}")
        hgb, et = make_hgb_reg(seed=args.seed + fold, k_best=args.k_best), make_et_reg(seed=args.seed + fold)
        hgb.fit(X_train[tr], y_train[tr])
        et.fit(X_train[tr], y_train[tr])
        oof_hgb[va], oof_et[va] = hgb.predict(X_train[va]), et.predict(X_train[va])

    print("\nTraining full base models for val/test meta features...")
    full_hgb, full_et = make_hgb_reg(seed=args.seed, k_best=args.k_best), make_et_reg(seed=args.seed)
    full_hgb.fit(X_train, y_train)
    full_et.fit(X_train, y_train)

    val_hgb, test_hgb = full_hgb.predict(X_val), full_hgb.predict(X_test)
    val_et, test_et = full_et.predict(X_val), full_et.predict(X_test)

    def meta(a, b):
        m, d = (a + b) / 2.0, np.abs(a - b)
        return np.vstack([a, b, m, d]).T.astype(np.float32)

    return {
        "full_hgb": full_hgb, "full_et": full_et,
        "train_meta": meta(oof_hgb, oof_et),
        "val_meta": meta(val_hgb, val_et),
        "test_meta": meta(test_hgb, test_et),
    }


def expand_with_meta(X, meta):
    mean_pred, disagreement = meta[:, 2:3], meta[:, 3:4]
    meta_aug = np.hstack([meta, mean_pred ** 2, np.sqrt(np.maximum(mean_pred, 0)), disagreement ** 2]).astype(np.float32)
    return np.hstack([X, meta_aug])


def gaussian_weights_from_pred(pred_gap, edges, temperature=0.75):
    finite_edges = edges.copy()
    finite_edges[0] = edges[1] - (edges[2] - edges[1])
    finite_edges[-1] = edges[-2] + (edges[-2] - edges[-3])
    centers = (finite_edges[:-1] + finite_edges[1:]) / 2.0
    logits = -0.5 * ((pred_gap.reshape(-1, 1) - centers.reshape(1, -1)) / temperature) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_oof_gated_moe")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--n-regimes", type=int, default=8)
    ap.add_argument("--k-best", type=int, default=3000)
    ap.add_argument("--expert-k-best", type=int, default=2000)
    ap.add_argument("--router-k-best", type=int, default=3000)
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--min-regime-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RDKIT:
        raise RuntimeError("RDKit is required.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PQR OOF-prediction-gated mixture of experts")
    print("=" * 80)

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded entries: {len(entries):,}")
    df, X, y, groups = build_dataset(entries, n_bits=args.n_bits, max_rows=args.max_rows, seed=args.seed)

    train_idx, val_idx, test_idx = group_split(groups, y, seed=args.seed)
    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Split means: train={y[train_idx].mean():.3f}, val={y[val_idx].mean():.3f}, test={y[test_idx].mean():.3f}")

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    groups_train = groups[train_idx]

    edges = make_regimes(y_train, args.n_regimes)
    train_reg, val_reg, test_reg = apply_regimes(y_train, edges), apply_regimes(y_val, edges), apply_regimes(y_test, edges)
    n_reg = len(edges) - 1

    print("\nRegime edges from TRAIN only:")
    for k in range(n_reg):
        print(f"  regime {k}: {edges[k]:8.3f} to {edges[k+1]:8.3f} eV, n_train={(train_reg==k).sum():,}")

    base = make_oof_predictions(X_train, y_train, groups_train, X_val, X_test, args)
    train_meta, val_meta, test_meta = base["train_meta"], base["val_meta"], base["test_meta"]

    results = []
    for nm, col in [("base_hgb", 0), ("base_et", 1), ("base_mean", 2)]:
        results.append(eval_pred("VAL " + nm, y_val, val_meta[:, col]))
        results.append(eval_pred("TEST " + nm, y_test, test_meta[:, col]))

    Xr_train, Xr_val, Xr_test = expand_with_meta(X_train, train_meta), expand_with_meta(X_val, val_meta), expand_with_meta(X_test, test_meta)

    print("\nTraining routers with molecular features + OOF predicted-gap coordinates...")
    routers, val_ps, test_ps = {}, {}, {}
    router_rows = []
    for name, kind in [("hgb_router", "hgb"), ("et_router", "et")]:
        print(f"  training {name}")
        router = make_router(kind, seed=args.seed, k_best=min(args.router_k_best, Xr_train.shape[1]))
        router.fit(Xr_train, train_reg)
        routers[name] = router

        val_pred, test_pred = router.predict(Xr_val), router.predict(Xr_test)
        print(f"  {name} VAL acc={accuracy_score(val_reg, val_pred):.4f}, bal={balanced_accuracy_score(val_reg, val_pred):.4f}")
        print(f"  {name} TEST acc={accuracy_score(test_reg, test_pred):.4f}, bal={balanced_accuracy_score(test_reg, test_pred):.4f}")

        val_p, test_p = np.zeros((len(y_val), n_reg), dtype=np.float32), np.zeros((len(y_test), n_reg), dtype=np.float32)
        pval, ptest = router.predict_proba(Xr_val), router.predict_proba(Xr_test)
        classes = list(router.named_steps["model"].classes_)
        for j, c in enumerate(classes):
            val_p[:, int(c)] = pval[:, j]
            test_p[:, int(c)] = ptest[:, j]
        val_ps[name], test_ps[name] = val_p, test_p
        router_rows.append({
            "router": name,
            "val_accuracy": accuracy_score(val_reg, val_pred),
            "val_balanced_accuracy": balanced_accuracy_score(val_reg, val_pred),
            "test_accuracy": accuracy_score(test_reg, test_pred),
            "test_balanced_accuracy": balanced_accuracy_score(test_reg, test_pred),
        })

    val_ps["gaussian_base_mean"] = gaussian_weights_from_pred(val_meta[:, 2], edges, temperature=0.75)
    test_ps["gaussian_base_mean"] = gaussian_weights_from_pred(test_meta[:, 2], edges, temperature=0.75)
    router_rows.append({
        "router": "gaussian_base_mean",
        "val_accuracy": accuracy_score(val_reg, val_ps["gaussian_base_mean"].argmax(axis=1)),
        "val_balanced_accuracy": balanced_accuracy_score(val_reg, val_ps["gaussian_base_mean"].argmax(axis=1)),
        "test_accuracy": accuracy_score(test_reg, test_ps["gaussian_base_mean"].argmax(axis=1)),
        "test_balanced_accuracy": balanced_accuracy_score(test_reg, test_ps["gaussian_base_mean"].argmax(axis=1)),
    })

    print("\nTraining expert regressors per true target regime...")
    val_expert, test_expert = np.zeros((len(y_val), n_reg), dtype=np.float32), np.zeros((len(y_test), n_reg), dtype=np.float32)
    expert_rows = []
    for k in range(n_reg):
        mask, n = train_reg == k, int((train_reg == k).sum())
        print(f"  regime {k}: expert on {n:,} rows")
        expert = make_hgb_reg(seed=args.seed + 100 + k, k_best=min(args.expert_k_best, X_train.shape[1]))
        expert.fit(X_train[mask], y_train[mask])
        val_expert[:, k], test_expert[:, k] = expert.predict(X_val), expert.predict(X_test)
        vm, tm = val_reg == k, test_reg == k
        expert_rows.append({
            "regime": k, "n_train": n,
            "val_oracle_mae": mean_absolute_error(y_val[vm], val_expert[vm, k]) if vm.any() else np.nan,
            "val_base_mae_same": mean_absolute_error(y_val[vm], val_meta[vm, 2]) if vm.any() else np.nan,
            "test_oracle_mae": mean_absolute_error(y_test[tm], test_expert[tm, k]) if tm.any() else np.nan,
            "test_base_mae_same": mean_absolute_error(y_test[tm], test_meta[tm, 2]) if tm.any() else np.nan,
        })

    P_val_list, P_test_list = [val_meta[:,0], val_meta[:,1], val_meta[:,2]], [test_meta[:,0], test_meta[:,1], test_meta[:,2]]
    pred_names = ["base_hgb", "base_et", "base_mean"]

    for name in val_ps:
        vmix = (val_ps[name] * val_expert).sum(axis=1)
        tmix = (test_ps[name] * test_expert).sum(axis=1)
        results.append(eval_pred("VAL mix_" + name, y_val, vmix))
        results.append(eval_pred("TEST mix_" + name, y_test, tmix))
        P_val_list.append(vmix); P_test_list.append(tmix); pred_names.append("mix_" + name)

    hard_val_reg, hard_test_reg = routers["hgb_router"].predict(Xr_val), routers["hgb_router"].predict(Xr_test)
    hard_val, hard_test = val_expert[np.arange(len(y_val)), hard_val_reg], test_expert[np.arange(len(y_test)), hard_test_reg]
    results.append(eval_pred("VAL hard_hgb_router", y_val, hard_val))
    results.append(eval_pred("TEST hard_hgb_router", y_test, hard_test))
    P_val_list.append(hard_val); P_test_list.append(hard_test); pred_names.append("hard_hgb_router")

    oracle_val, oracle_test = val_expert[np.arange(len(y_val)), val_reg], test_expert[np.arange(len(y_test)), test_reg]
    results.append(eval_pred("VAL ORACLE_ROUTE", y_val, oracle_val))
    results.append(eval_pred("TEST ORACLE_ROUTE", y_test, oracle_test))

    print("\nTraining ridge stacker over realistic predictions...")
    P_val, P_test = np.vstack(P_val_list).T, np.vstack(P_test_list).T
    stacker = RidgeCV(alphas=np.logspace(-6, 3, 30))
    stacker.fit(P_val, y_val)
    val_stack, test_stack = stacker.predict(P_val), stacker.predict(P_test)
    results.append(eval_pred("VAL stacked_realistic", y_val, val_stack))
    results.append(eval_pred("TEST stacked_realistic", y_test, test_stack))

    pd.DataFrame(results).to_csv(outdir / "metrics.csv", index=False)
    pd.DataFrame(expert_rows).to_csv(outdir / "regime_expert_report.csv", index=False)
    pd.DataFrame(router_rows).to_csv(outdir / "router_report.csv", index=False)
    pd.DataFrame(confusion_matrix(val_reg, hard_val_reg, labels=list(range(n_reg)))).to_csv(outdir / "val_confusion_hgb_router.csv", index=False)
    pd.DataFrame(confusion_matrix(test_reg, hard_test_reg, labels=list(range(n_reg)))).to_csv(outdir / "test_confusion_hgb_router.csv", index=False)

    test_report = df.iloc[test_idx].copy()
    test_report["true_regime"] = test_reg
    test_report["pred_regime_hgb"] = hard_test_reg
    test_report["pred_base_mean"] = test_meta[:, 2]
    test_report["pred_stack"] = test_stack
    test_report["pred_oracle"] = oracle_test
    test_report["abs_error_stack"] = np.abs(test_stack - y_test)
    test_report["abs_error_base_mean"] = np.abs(test_meta[:, 2] - y_test)
    test_report["abs_error_oracle"] = np.abs(oracle_test - y_test)
    test_report.sort_values("abs_error_stack", ascending=False).to_csv(outdir / "test_error_report.csv", index=False)
    test_report.groupby("true_regime").agg(
        n=("gap", "size"),
        base_mean_mae=("abs_error_base_mean", "mean"),
        stack_mae=("abs_error_stack", "mean"),
        oracle_mae=("abs_error_oracle", "mean"),
        mean_gap=("gap", "mean"),
        sd_gap=("gap", "std"),
    ).to_csv(outdir / "test_mae_by_true_regime.csv")

    joblib.dump({"base": base, "routers": routers, "edges": edges, "stacker": stacker, "pred_names": pred_names}, outdir / "oof_gated_moe_model.joblib")

    print("\nSaved:")
    for fn in ["metrics.csv", "router_report.csv", "regime_expert_report.csv", "val_confusion_hgb_router.csv", "test_confusion_hgb_router.csv", "test_error_report.csv", "test_mae_by_true_regime.csv", "oof_gated_moe_model.joblib"]:
        print(" ", outdir / fn)

    print("\nInterpretation:")
    print("If router accuracy rises and realistic MAE improves, frame your novelty as OOF-predicted electronic-regime routing.")
    print("If ORACLE remains ~0.16 eV but realistic routing stays ~1 eV, the bottleneck is still regime classification.")
    print("Then the honest path toward ~0.1 eV is a stronger structure model/router: D-MPNN, SchNet, or pretrained SMILES model.")


if __name__ == "__main__":
    main()
