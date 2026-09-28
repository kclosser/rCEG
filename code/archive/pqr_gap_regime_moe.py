#!/usr/bin/env python3
"""
pqr_gap_regime_moe.py

Target-regime mixture-of-experts for HOMO-LUMO gap prediction.

Why this is the next experiment
-------------------------------
Your learned leaf router did reduce target spread somewhat, but not enough:
the leaves still had sd_gap often ~1.5-2.2 eV. That means a purely structural
decision-tree router is not discovering sufficiently uniform regimes.

This script uses a stronger, still valid architecture:

1. Define HOMO-LUMO gap regimes on TRAIN ONLY using y-quantiles.
   Example: low gap, mid-low gap, ..., high gap.

2. Train a classifier to predict the regime from SMILES-derived features.
   At test time, the model does NOT know y; it only predicts regime probabilities.

3. Train one regressor expert per true training regime.

4. Final prediction is a probability-weighted mixture:
      prediction = sum_k P(regime=k | features) * expert_k(features)

This directly operationalizes your novelty:
"First identify which hidden electronic regime a molecule belongs to, then
use the specialized expert for that smoother subspace."

Usage
-----
cd ~/Downloads/Closser

# quick
python pqr_gap_regime_moe.py \
  --data enhanced_dataset_lasso_STRICT.jsonl \
  --outdir runs/pqr_regime_moe_quick \
  --max-rows 30000 \
  --n-regimes 8

# full
python pqr_gap_regime_moe.py \
  --data enhanced_dataset_lasso_STRICT.jsonl \
  --outdir runs/pqr_regime_moe_full \
  --n-regimes 8
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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


SMARTS = {
    "aromatic": "a",
    "hetero_aromatic": "[a;!#6]",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "acid": "C(=O)[OX2H1]",
    "nitrile": "C#N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][#6]",
    "fluoro": "[F]",
    "chloro": "[Cl]",
    "sulfur": "[S]",
    "nitro": "[NX3](=O)=O",
    "phenol_like": "aO",
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
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
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

    gen2 = GetMorganGenerator(radius=2, fpSize=n_bits)
    gen3 = GetMorganGenerator(radius=3, fpSize=n_bits)

    rows, Xs = [], []
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

        # Only first 5 PQR descriptors. HOMO/LUMO are not used.
        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso = [fnum(v) for v in desc]
        rdk = rdkit_features(mol)
        flags = smarts_flags(mol)
        fp2 = morgan_arr(mol, gen2, n_bits)
        fp3 = morgan_arr(mol, gen3, n_bits)

        x = np.concatenate([
            np.array(pqr5, dtype=np.float32),
            np.array(lasso, dtype=np.float32),
            np.array(rdk, dtype=np.float32),
            np.array(flags, dtype=np.float32),
            fp2,
            fp3,
        ])

        rows.append({
            "idx": i,
            "smiles": canon,
            "gap": gap,
            "coarse_class": coarse_class(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
        Xs.append(x)

    if not rows:
        raise RuntimeError("No usable rows.")

    max_len = max(len(x) for x in Xs)
    X = np.full((len(Xs), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    y = df["gap"].to_numpy(np.float32)
    groups = df["smiles"].to_numpy()

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
    train = trainval[tr_rel]
    val = trainval[va_rel]
    return train, val, test


def make_regressor(kind="hgb", seed=42, k_best=2500):
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
    elif kind == "et":
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.30,
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
    if k_best is not None:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    steps.append(("model", model))
    return Pipeline(steps)


def make_classifier(seed=42, k_best=2500):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
        ("select", SelectKBest(f_classif, k=k_best)),
        ("model", HistGradientBoostingClassifier(
            max_iter=700,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=60,
            random_state=seed,
        )),
    ])


def eval_pred(name, y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"{name:35s} MAE={mae:.4f} eV   R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def make_regimes(y_train, n_regimes):
    # Quantile bins on training target only.
    qs = np.linspace(0, 100, n_regimes + 1)
    edges = np.percentile(y_train, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # remove duplicate edges if target has repeated values
    unique_edges = [edges[0]]
    for e in edges[1:]:
        if e > unique_edges[-1]:
            unique_edges.append(e)
    edges = np.array(unique_edges)
    return edges


def apply_regimes(y, edges):
    # labels 0..K-1
    return np.digitize(y, edges[1:-1], right=False)


def train_regime_moe(df, X, y, train_idx, val_idx, test_idx, args):
    results = []

    # 1. Global baseline.
    print("\nTraining global regressors...")
    global_hgb = make_regressor("hgb", seed=args.seed, k_best=args.k_best)
    global_hgb.fit(X[train_idx], y[train_idx])
    val_global = global_hgb.predict(X[val_idx])
    test_global = global_hgb.predict(X[test_idx])
    results.append(eval_pred("VAL global_hgb", y[val_idx], val_global))
    results.append(eval_pred("TEST global_hgb", y[test_idx], test_global))

    # 2. Target regimes.
    edges = make_regimes(y[train_idx], args.n_regimes)
    train_reg = apply_regimes(y[train_idx], edges)
    val_reg = apply_regimes(y[val_idx], edges)
    test_reg = apply_regimes(y[test_idx], edges)
    n_reg = len(edges) - 1

    print("\nRegime edges from TRAIN only:")
    for k in range(n_reg):
        print(f"  regime {k}: {edges[k]:8.3f} to {edges[k+1]:8.3f} eV, n_train={(train_reg==k).sum():,}")

    # 3. Classifier router.
    print("\nTraining regime classifier router...")
    clf = make_classifier(seed=args.seed, k_best=args.k_best)
    clf.fit(X[train_idx], train_reg)

    val_reg_pred = clf.predict(X[val_idx])
    test_reg_pred = clf.predict(X[test_idx])

    print(f"VAL regime accuracy:      {accuracy_score(val_reg, val_reg_pred):.4f}")
    print(f"VAL balanced accuracy:    {balanced_accuracy_score(val_reg, val_reg_pred):.4f}")
    print(f"TEST regime accuracy:     {accuracy_score(test_reg, test_reg_pred):.4f}")
    print(f"TEST balanced accuracy:   {balanced_accuracy_score(test_reg, test_reg_pred):.4f}")

    val_proba = clf.predict_proba(X[val_idx])
    test_proba = clf.predict_proba(X[test_idx])

    # Some classifiers may omit a class if subsampling created a weird split.
    clf_classes = list(clf.named_steps["model"].classes_)

    # 4. Expert per target regime.
    print("\nTraining one expert per target regime...")
    val_expert_matrix = np.zeros((len(val_idx), n_reg), dtype=np.float32)
    test_expert_matrix = np.zeros((len(test_idx), n_reg), dtype=np.float32)
    expert_rows = []

    for k in range(n_reg):
        mask = train_reg == k
        n = int(mask.sum())

        # fallback to global for tiny bins
        if n < args.min_regime_n:
            print(f"  regime {k}: n={n}, too small -> global fallback")
            val_expert_matrix[:, k] = val_global
            test_expert_matrix[:, k] = test_global
            expert_rows.append({"regime": k, "n_train": n, "used": False})
            continue

        print(f"  regime {k}: training expert on {n:,} rows")
        expert = make_regressor("hgb", seed=args.seed + 100 + k, k_best=args.expert_k_best)
        expert.fit(X[train_idx][mask], y[train_idx][mask])
        val_expert_matrix[:, k] = expert.predict(X[val_idx])
        test_expert_matrix[:, k] = expert.predict(X[test_idx])

        # oracle-in-regime eval: only on points truly in that regime
        vm = val_reg == k
        tm = test_reg == k
        row = {"regime": k, "n_train": n, "used": True}
        if vm.any():
            row["val_true_regime_mae"] = mean_absolute_error(y[val_idx][vm], val_expert_matrix[vm, k])
            row["val_global_same_points_mae"] = mean_absolute_error(y[val_idx][vm], val_global[vm])
        if tm.any():
            row["test_true_regime_mae"] = mean_absolute_error(y[test_idx][tm], test_expert_matrix[tm, k])
            row["test_global_same_points_mae"] = mean_absolute_error(y[test_idx][tm], test_global[tm])
        expert_rows.append(row)

    # 5. Mixture predictions.
    # Reconstruct full proba matrix in case classes are omitted.
    val_p = np.zeros((len(val_idx), n_reg), dtype=np.float32)
    test_p = np.zeros((len(test_idx), n_reg), dtype=np.float32)
    for j, cls in enumerate(clf_classes):
        val_p[:, int(cls)] = val_proba[:, j]
        test_p[:, int(cls)] = test_proba[:, j]

    val_mix = (val_p * val_expert_matrix).sum(axis=1)
    test_mix = (test_p * test_expert_matrix).sum(axis=1)

    # Hard routing.
    val_hard = val_expert_matrix[np.arange(len(val_idx)), val_reg_pred]
    test_hard = test_expert_matrix[np.arange(len(test_idx)), test_reg_pred]

    # Oracle routing: an upper bound if router were perfect.
    val_oracle = val_expert_matrix[np.arange(len(val_idx)), val_reg]
    test_oracle = test_expert_matrix[np.arange(len(test_idx)), test_reg]

    results.append(eval_pred("VAL regime_soft_mixture", y[val_idx], val_mix))
    results.append(eval_pred("TEST regime_soft_mixture", y[test_idx], test_mix))
    results.append(eval_pred("VAL regime_hard_route", y[val_idx], val_hard))
    results.append(eval_pred("TEST regime_hard_route", y[test_idx], test_hard))
    results.append(eval_pred("VAL regime_ORACLE_ROUTE", y[val_idx], val_oracle))
    results.append(eval_pred("TEST regime_ORACLE_ROUTE", y[test_idx], test_oracle))

    # 6. Stacker.
    print("\nTraining ridge stacker over global + hard + soft mixture...")
    P_val = np.vstack([val_global, val_mix, val_hard]).T
    P_test = np.vstack([test_global, test_mix, test_hard]).T
    stacker = RidgeCV(alphas=np.logspace(-6, 3, 30))
    stacker.fit(P_val, y[val_idx])
    val_stack = stacker.predict(P_val)
    test_stack = stacker.predict(P_test)
    results.append(eval_pred("VAL stacked", y[val_idx], val_stack))
    results.append(eval_pred("TEST stacked", y[test_idx], test_stack))

    # diagnostics
    expert_df = pd.DataFrame(expert_rows)
    cm_val = pd.DataFrame(confusion_matrix(val_reg, val_reg_pred, labels=list(range(n_reg))))
    cm_test = pd.DataFrame(confusion_matrix(test_reg, test_reg_pred, labels=list(range(n_reg))))

    test_report = df.iloc[test_idx].copy()
    test_report["true_regime"] = test_reg
    test_report["pred_regime"] = test_reg_pred
    test_report["pred_global"] = test_global
    test_report["pred_soft_mixture"] = test_mix
    test_report["pred_hard_route"] = test_hard
    test_report["pred_stack"] = test_stack
    test_report["abs_error_stack"] = np.abs(test_stack - y[test_idx])
    test_report["abs_error_soft"] = np.abs(test_mix - y[test_idx])
    test_report["abs_error_global"] = np.abs(test_global - y[test_idx])

    objects = {
        "global_hgb": global_hgb,
        "classifier": clf,
        "edges": edges,
        "stacker": stacker,
    }
    outputs = {
        "results": pd.DataFrame(results),
        "expert_report": expert_df,
        "cm_val": cm_val,
        "cm_test": cm_test,
        "test_report": test_report,
    }
    return objects, outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_regime_moe")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--n-regimes", type=int, default=8)
    ap.add_argument("--k-best", type=int, default=3000)
    ap.add_argument("--expert-k-best", type=int, default=2000)
    ap.add_argument("--min-regime-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RDKIT:
        raise RuntimeError("RDKit is required.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PQR target-regime mixture of experts")
    print("=" * 80)

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded entries: {len(entries):,}")

    df, X, y, groups = build_dataset(entries, n_bits=args.n_bits, max_rows=args.max_rows, seed=args.seed)
    train_idx, val_idx, test_idx = group_split(groups, y, seed=args.seed)
    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Split means: train={y[train_idx].mean():.3f}, val={y[val_idx].mean():.3f}, test={y[test_idx].mean():.3f}")

    objects, outputs = train_regime_moe(df, X, y, train_idx, val_idx, test_idx, args)

    outputs["results"].to_csv(outdir / "metrics.csv", index=False)
    outputs["expert_report"].to_csv(outdir / "regime_expert_report.csv", index=False)
    outputs["cm_val"].to_csv(outdir / "val_regime_confusion_matrix.csv", index=False)
    outputs["cm_test"].to_csv(outdir / "test_regime_confusion_matrix.csv", index=False)
    outputs["test_report"].sort_values("abs_error_stack", ascending=False).to_csv(outdir / "test_error_report.csv", index=False)

    outputs["test_report"].groupby("true_regime").agg(
        n=("gap", "size"),
        global_mae=("abs_error_global", "mean"),
        soft_mae=("abs_error_soft", "mean"),
        stack_mae=("abs_error_stack", "mean"),
        mean_gap=("gap", "mean"),
        sd_gap=("gap", "std"),
    ).to_csv(outdir / "test_mae_by_true_regime.csv")

    joblib.dump(objects, outdir / "regime_moe_model.joblib")

    print("\nSaved:")
    for fn in [
        "metrics.csv",
        "regime_expert_report.csv",
        "val_regime_confusion_matrix.csv",
        "test_regime_confusion_matrix.csv",
        "test_error_report.csv",
        "test_mae_by_true_regime.csv",
        "regime_moe_model.joblib",
    ]:
        print(" ", outdir / fn)

    print("\nInterpretation:")
    print("1. If ORACLE_ROUTE is much better than soft/hard routing, the experts work but the classifier router is weak.")
    print("2. If ORACLE_ROUTE is not better than global, even target-defined regimes do not help with these features.")
    print("3. If soft/hard improves, this is your strongest split-architecture result so far.")


if __name__ == "__main__":
    main()
