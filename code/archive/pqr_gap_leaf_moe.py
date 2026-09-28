#!/usr/bin/env python3
"""
pqr_gap_leaf_moe.py

Purpose
-------
Your diagnostics showed:
- Labels are internally correct: gap = LUMO - HOMO.
- HOMO/LUMO leakage test gets very low MAE, so the target is learnable.
- PQR/LASSO + Morgan/RDKit still stays near ~1 eV, so the issue is not just cleaning.
- Your goal is a split architecture that makes the target space more uniform.

This script implements that idea more directly:

1. Build structural features from SMILES:
   - original PQR + LASSO descriptors
   - RDKit descriptors
   - Morgan fingerprints
   - SMARTS motif flags

2. Learn a ROUTER on TRAINING DATA ONLY:
   - a shallow DecisionTreeRegressor splits molecules into leaves
   - leaves are chosen to reduce target variance
   - this is not leakage because the router is fit only on train features/labels
     and then applied to validation/test features

3. Train a global model plus local expert models:
   - global model predicts the broad trend
   - each leaf expert predicts either the direct target or residual correction
   - validation decides whether direct expert, residual expert, or global fallback is best

This is a stronger, more defensible version of your novelty:
"Mazouin-style selected learning, but with learned structural routing instead of
manual MW/dipole bins."

Usage
-----
cd ~/Downloads/Closser

# Quick test first
python pqr_gap_leaf_moe.py \
  --data enhanced_dataset_lasso_STRICT.jsonl \
  --outdir runs/pqr_leaf_moe_quick \
  --max-rows 30000

# Full run
python pqr_gap_leaf_moe.py \
  --data enhanced_dataset_lasso_STRICT.jsonl \
  --outdir runs/pqr_leaf_moe_full
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor, export_text
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


SMARTS_PATTERNS = {
    # Mazouin-inspired electronic structure classes
    "aromatic_atom": "a",
    "benzene": "c1ccccc1",
    "hetero_aromatic": "[a;!#6]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "acid": "C(=O)[OX2H1]",
    "nitrile": "C#N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "azo": "N=N",
    "nitro": "[NX3](=O)=O",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][#6]",
    "fluoro": "[F]",
    "chloro": "[Cl]",
    "sulfur": "[S]",
    "phosphorus": "[P]",
    # conjugation proxies
    "conj_carbonyl_alkene": "C=CC=O",
    "enone": "C=CC(=O)",
    "aniline_like": "aN",
    "phenol_like": "aO",
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


def mol_ok_and_canon(smiles):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None, None, "invalid_smiles"
    if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms()):
        return None, None, "isolated_hydrogen"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, None, "multi_fragment"
    return Chem.MolToSmiles(mol, canonical=True), mol, None


def smarts_flags(mol):
    flags = []
    for name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        flags.append(float(patt is not None and mol.HasSubstructMatch(patt)))
    return flags


def rdkit_numeric(mol):
    bonds = list(mol.GetBonds())
    atoms = list(mol.GetAtoms())
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
        mol.GetNumAtoms(),
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
        sum(b.GetBondTypeAsDouble() == 1 for b in bonds),
        sum(b.GetBondTypeAsDouble() == 1.5 for b in bonds),
        sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
        sum(b.GetBondTypeAsDouble() == 3 for b in bonds),
        sum(a.GetAtomicNum() == 6 for a in atoms),
        sum(a.GetAtomicNum() == 7 for a in atoms),
        sum(a.GetAtomicNum() == 8 for a in atoms),
        sum(a.GetAtomicNum() == 9 for a in atoms),
        sum(a.GetAtomicNum() == 16 for a in atoms),
        sum(a.GetAtomicNum() == 17 for a in atoms),
    ]


def morgan_array(mol, gen, n_bits):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def coarse_chem_class(mol):
    flags = dict(zip(SMARTS_PATTERNS.keys(), smarts_flags(mol)))
    if flags["aromatic_atom"] or flags["carbonyl"] or flags["amide"] or flags["ester"]:
        return "aromatic_or_carbonyl"
    if flags["alkene"] or flags["alkyne"] or flags["nitrile"]:
        return "single_unsaturated"
    return "saturated"


def build_dataset(entries, n_bits=2048, max_rows=None, seed=42):
    rng = np.random.default_rng(seed)
    if max_rows is not None and len(entries) > max_rows:
        idx = rng.choice(len(entries), max_rows, replace=False)
        entries = [entries[i] for i in idx]
        print(f"Subsampled raw entries to {len(entries):,}")

    gen2 = GetMorganGenerator(radius=2, fpSize=n_bits)
    gen3 = GetMorganGenerator(radius=3, fpSize=n_bits)

    rows = []
    X_full = []
    X_route = []
    drops = Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drops["bad_entry_shape"] += 1
            continue

        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < 0.10 or gap > 25:
            drops["bad_gap"] += 1
            continue

        canon, mol, err = mol_ok_and_canon(e[1])
        if err:
            drops[err] += 1
            continue

        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []

        # Do not include HOMO/LUMO. Use first 5 PQR only.
        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso = [fnum(v) for v in desc]
        rdk = rdkit_numeric(mol)
        flags = smarts_flags(mol)
        fp2 = morgan_array(mol, gen2, n_bits)
        fp3 = morgan_array(mol, gen3, n_bits)

        # Full model features.
        x = np.concatenate([
            np.array(pqr5, dtype=np.float32),
            np.array(lasso, dtype=np.float32),
            np.array(rdk, dtype=np.float32),
            np.array(flags, dtype=np.float32),
            fp2,
            fp3,
        ])

        # Router features should be lower-dimensional and interpretable.
        # Use PQR5 + RDKit numeric + SMARTS flags + compact Morgan bits.
        route = np.concatenate([
            np.array(pqr5, dtype=np.float32),
            np.array(rdk, dtype=np.float32),
            np.array(flags, dtype=np.float32),
            fp2[:512],
            fp3[:512],
        ])

        rows.append({
            "old_index": i,
            "smiles": canon,
            "gap": gap,
            "coarse_class": coarse_chem_class(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
        X_full.append(x)
        X_route.append(route)

    if not rows:
        raise RuntimeError("No usable rows.")

    max_full = max(len(x) for x in X_full)
    X = np.full((len(X_full), max_full), np.nan, dtype=np.float32)
    for i, x in enumerate(X_full):
        X[i, :len(x)] = x

    max_route = max(len(x) for x in X_route)
    R = np.full((len(X_route), max_route), np.nan, dtype=np.float32)
    for i, x in enumerate(X_route):
        R[i, :len(x)] = x

    df = pd.DataFrame(rows)
    y = df["gap"].to_numpy(np.float32)
    groups = df["smiles"].to_numpy()

    print(f"Usable rows: {len(df):,}")
    print(f"Drops: {dict(drops)}")
    print("Coarse class distribution:")
    print(df["coarse_class"].value_counts().to_string())
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"Full feature matrix:   {X.shape}")
    print(f"Router feature matrix: {R.shape}")
    return df, X, R, y, groups


def group_split(groups, y, test_size=0.15, val_size=0.15, seed=42):
    idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(gss1.split(idx, y, groups))
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_rel, val_rel = next(gss2.split(trainval, y[trainval], groups[trainval]))
    train = trainval[tr_rel]
    val = trainval[val_rel]
    return train, val, test


def preprocess_route(R_train, R_all, k=700):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
        ("select", SelectKBest(f_regression, k=min(k, R_train.shape[1]))),
    ])
    Rtr = pipe.fit_transform(R_train[0], R_train[1])
    Rall = pipe.transform(R_all)
    return pipe, Rtr, Rall


def make_global_model(kind="hgb", seed=42, k_best=3000):
    if kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=1100,
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
            n_estimators=600,
            max_features=0.30,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=500,
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
    if k_best is not None:
        steps.append(("select", SelectKBest(f_regression, k=k_best)))
    steps.append(("model", model))
    return Pipeline(steps)


def eval_pred(name, y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"{name:38s} MAE={mae:.4f} eV   R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def target_sd_report(df, y, labels, split_name):
    tmp = pd.DataFrame({"leaf": labels, "gap": y})
    rep = tmp.groupby("leaf").agg(
        n=("gap", "size"),
        mean_gap=("gap", "mean"),
        sd_gap=("gap", "std"),
        min_gap=("gap", "min"),
        max_gap=("gap", "max"),
    ).sort_values("sd_gap", ascending=False)
    print(f"\n{split_name} leaf target spread:")
    print(rep.head(20).to_string())
    return rep


def fit_leaf_moe(df, X, R, y, train_idx, val_idx, test_idx, args):
    # ---------------------------
    # 1. Global baseline
    # ---------------------------
    print("\nTraining global baselines...")
    global_models = {}
    P_val = []
    P_test = []
    pred_names = []
    results = []

    for name, kind, kbest in [
        ("global_hgb", "hgb", args.k_best),
        ("global_et", "et", None),
    ]:
        model = make_global_model(kind, seed=args.seed, k_best=kbest)
        model.fit(X[train_idx], y[train_idx])
        global_models[name] = model
        pv = model.predict(X[val_idx])
        pt = model.predict(X[test_idx])
        P_val.append(pv)
        P_test.append(pt)
        pred_names.append(name)
        results.append(eval_pred("VAL " + name, y[val_idx], pv))
        results.append(eval_pred("TEST " + name, y[test_idx], pt))

    global_best = global_models["global_hgb"]
    train_global_pred = global_best.predict(X[train_idx])
    val_global_pred = global_best.predict(X[val_idx])
    test_global_pred = global_best.predict(X[test_idx])
    train_resid = y[train_idx] - train_global_pred

    # ---------------------------
    # 2. Fit learned router
    # ---------------------------
    print("\nFitting target-variance router on TRAIN only...")
    route_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(1e-12)),
        ("select", SelectKBest(f_regression, k=min(args.router_k, R.shape[1]))),
    ])
    R_train = route_pipe.fit_transform(R[train_idx], y[train_idx])
    R_val = route_pipe.transform(R[val_idx])
    R_test = route_pipe.transform(R[test_idx])
    R_all = route_pipe.transform(R)

    best_router = None
    best_router_score = float("inf")
    router_records = []

    for leaves in args.leaf_grid:
        router = DecisionTreeRegressor(
            max_leaf_nodes=leaves,
            min_samples_leaf=args.min_leaf,
            random_state=args.seed,
            criterion="squared_error",
        )
        router.fit(R_train, y[train_idx])
        val_leaf = router.apply(R_val)

        # Estimate whether leaves actually reduce target spread on validation.
        leaf_mae = 0.0
        total = 0
        train_leaf = router.apply(R_train)
        leaf_means = {}
        for lf in np.unique(train_leaf):
            mask = train_leaf == lf
            leaf_means[lf] = float(np.mean(y[train_idx][mask]))
        pred = np.array([leaf_means.get(lf, float(np.mean(y[train_idx]))) for lf in val_leaf])
        val_mae = mean_absolute_error(y[val_idx], pred)

        router_records.append({"max_leaf_nodes": leaves, "val_leaf_mean_mae": val_mae})
        print(f"  router leaves={leaves:3d}: VAL leaf-mean MAE={val_mae:.4f}")

        if val_mae < best_router_score:
            best_router_score = val_mae
            best_router = router

    router = best_router
    train_leaf = router.apply(R_train)
    val_leaf = router.apply(R_val)
    test_leaf = router.apply(R_test)
    all_leaf = router.apply(R_all)

    print(f"\nSelected router with {router.get_n_leaves()} leaves; validation leaf-mean MAE={best_router_score:.4f}")
    print("Leaf IDs:", sorted(np.unique(train_leaf))[:20], "..." if len(np.unique(train_leaf)) > 20 else "")

    # Report whether your split actually made y more uniform.
    train_leaf_report = target_sd_report(df.iloc[train_idx], y[train_idx], train_leaf, "TRAIN")
    val_leaf_report = target_sd_report(df.iloc[val_idx], y[val_idx], val_leaf, "VAL")

    # Save human-readable router rules if possible.
    feature_names = [f"route_feature_{i}" for i in range(R_train.shape[1])]
    try:
        tree_text = export_text(router, feature_names=feature_names, max_depth=6)
    except Exception:
        tree_text = "(could not export tree)"

    # ---------------------------
    # 3. Local experts per leaf
    # ---------------------------
    print("\nTraining local leaf experts...")
    direct_val_pred = val_global_pred.copy()
    direct_test_pred = test_global_pred.copy()
    resid_val_pred = val_global_pred.copy()
    resid_test_pred = test_global_pred.copy()
    chosen_val_pred = val_global_pred.copy()
    chosen_test_pred = test_global_pred.copy()

    expert_info = []
    unique_leaves = sorted(np.unique(train_leaf))

    for lf in unique_leaves:
        tr_mask = train_leaf == lf
        va_mask = val_leaf == lf
        te_mask = test_leaf == lf
        ntr = int(tr_mask.sum())
        nva = int(va_mask.sum())
        nte = int(te_mask.sum())

        if ntr < args.min_expert_n or nva < 20:
            expert_info.append({
                "leaf": int(lf), "n_train": ntr, "n_val": nva, "n_test": nte,
                "used": False, "reason": "too_small",
            })
            continue

        # Direct target expert.
        direct = make_global_model("hgb", seed=args.seed + int(lf), k_best=min(args.expert_k_best, X.shape[1]))
        direct.fit(X[train_idx][tr_mask], y[train_idx][tr_mask])

        # Residual expert: predict correction to global model.
        resid = make_global_model("hgb", seed=args.seed + int(lf) + 1000, k_best=min(args.expert_k_best, X.shape[1]))
        resid.fit(X[train_idx][tr_mask], train_resid[tr_mask])

        dv = direct.predict(X[val_idx][va_mask])
        dt = direct.predict(X[test_idx][te_mask]) if nte else np.array([])

        rv_corr = resid.predict(X[val_idx][va_mask])
        rt_corr = resid.predict(X[test_idx][te_mask]) if nte else np.array([])
        rv = val_global_pred[va_mask] + rv_corr
        rt = test_global_pred[te_mask] + rt_corr if nte else np.array([])

        gv = val_global_pred[va_mask]

        direct_mae = mean_absolute_error(y[val_idx][va_mask], dv)
        resid_mae = mean_absolute_error(y[val_idx][va_mask], rv)
        global_mae = mean_absolute_error(y[val_idx][va_mask], gv)

        # Only use a local expert if it truly beats the global model on validation.
        best_kind = "global"
        best_mae = global_mae
        if direct_mae + args.min_improvement < best_mae:
            best_kind = "direct"
            best_mae = direct_mae
        if resid_mae + args.min_improvement < best_mae:
            best_kind = "residual"
            best_mae = resid_mae

        # Store raw expert predictions too.
        direct_val_pred[va_mask] = dv
        if nte:
            direct_test_pred[te_mask] = dt
        resid_val_pred[va_mask] = rv
        if nte:
            resid_test_pred[te_mask] = rt

        if best_kind == "direct":
            chosen_val_pred[va_mask] = dv
            if nte:
                chosen_test_pred[te_mask] = dt
        elif best_kind == "residual":
            chosen_val_pred[va_mask] = rv
            if nte:
                chosen_test_pred[te_mask] = rt

        expert_info.append({
            "leaf": int(lf), "n_train": ntr, "n_val": nva, "n_test": nte,
            "used": best_kind != "global",
            "chosen": best_kind,
            "val_global_mae": global_mae,
            "val_direct_mae": direct_mae,
            "val_residual_mae": resid_mae,
            "val_best_mae": best_mae,
            "train_gap_mean": float(np.mean(y[train_idx][tr_mask])),
            "train_gap_sd": float(np.std(y[train_idx][tr_mask])),
        })

        print(
            f"  leaf {int(lf):>4d}: n_train={ntr:>5d}, n_val={nva:>4d}, n_test={nte:>4d} | "
            f"global={global_mae:.3f}, direct={direct_mae:.3f}, residual={resid_mae:.3f} -> {best_kind}"
        )

    P_val += [direct_val_pred, resid_val_pred, chosen_val_pred]
    P_test += [direct_test_pred, resid_test_pred, chosen_test_pred]
    pred_names += ["leaf_direct_all", "leaf_residual_all", "leaf_chosen_by_val"]

    results.append(eval_pred("VAL leaf_direct_all", y[val_idx], direct_val_pred))
    results.append(eval_pred("TEST leaf_direct_all", y[test_idx], direct_test_pred))
    results.append(eval_pred("VAL leaf_residual_all", y[val_idx], resid_val_pred))
    results.append(eval_pred("TEST leaf_residual_all", y[test_idx], resid_test_pred))
    results.append(eval_pred("VAL leaf_chosen_by_val", y[val_idx], chosen_val_pred))
    results.append(eval_pred("TEST leaf_chosen_by_val", y[test_idx], chosen_test_pred))

    # ---------------------------
    # 4. Stacker
    # ---------------------------
    print("\nTraining stacker over global + leaf predictions...")
    P_val = np.vstack(P_val).T
    P_test = np.vstack(P_test).T
    stacker = RidgeCV(alphas=np.logspace(-6, 3, 30))
    stacker.fit(P_val, y[val_idx])
    val_stack = stacker.predict(P_val)
    test_stack = stacker.predict(P_test)
    results.append(eval_pred("VAL stacked", y[val_idx], val_stack))
    results.append(eval_pred("TEST stacked", y[test_idx], test_stack))

    objects = {
        "global_models": global_models,
        "route_pipe": route_pipe,
        "router": router,
        "stacker": stacker,
        "pred_names": pred_names,
        "router_records": router_records,
        "tree_text": tree_text,
        "expert_info": expert_info,
    }
    preds = {
        "val_stack": val_stack,
        "test_stack": test_stack,
        "test_leaf": test_leaf,
        "all_leaf": all_leaf,
    }
    reports = {
        "results": results,
        "expert_info": expert_info,
        "train_leaf_report": train_leaf_report,
        "val_leaf_report": val_leaf_report,
        "router_records": router_records,
    }
    return objects, preds, reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_leaf_moe")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-best", type=int, default=3000)
    ap.add_argument("--router-k", type=int, default=700)
    ap.add_argument("--expert-k-best", type=int, default=1500)
    ap.add_argument("--min-leaf", type=int, default=400)
    ap.add_argument("--min-expert-n", type=int, default=600)
    ap.add_argument("--min-improvement", type=float, default=0.01)
    ap.add_argument("--leaf-grid", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48])
    args = ap.parse_args()

    if not RDKIT:
        raise RuntimeError("RDKit is required.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PQR learned split architecture: structural router + local experts")
    print("=" * 80)

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded entries: {len(entries):,}")

    df, X, R, y, groups = build_dataset(entries, n_bits=args.n_bits, max_rows=args.max_rows, seed=args.seed)
    train_idx, val_idx, test_idx = group_split(groups, y, seed=args.seed)
    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Split target means: train={y[train_idx].mean():.3f}, val={y[val_idx].mean():.3f}, test={y[test_idx].mean():.3f}")

    objects, preds, reports = fit_leaf_moe(df, X, R, y, train_idx, val_idx, test_idx, args)

    pd.DataFrame(reports["results"]).to_csv(outdir / "metrics.csv", index=False)
    pd.DataFrame(reports["expert_info"]).to_csv(outdir / "leaf_expert_report.csv", index=False)
    pd.DataFrame(reports["router_records"]).to_csv(outdir / "router_selection.csv", index=False)
    reports["train_leaf_report"].to_csv(outdir / "train_leaf_target_spread.csv")
    reports["val_leaf_report"].to_csv(outdir / "val_leaf_target_spread.csv")

    test_report = df.iloc[test_idx].copy()
    test_report["leaf"] = preds["test_leaf"]
    test_report["pred_gap"] = preds["test_stack"]
    test_report["abs_error"] = np.abs(test_report["pred_gap"].to_numpy() - y[test_idx])
    test_report.sort_values("abs_error", ascending=False).to_csv(outdir / "test_error_report.csv", index=False)

    test_report.groupby("leaf").agg(
        n=("gap", "size"),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        mean_gap=("gap", "mean"),
        sd_gap=("gap", "std"),
    ).sort_values("mae", ascending=False).to_csv(outdir / "test_mae_by_leaf.csv")

    test_report.groupby("coarse_class").agg(
        n=("gap", "size"),
        mae=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"),
        mean_gap=("gap", "mean"),
        sd_gap=("gap", "std"),
    ).sort_values("mae", ascending=False).to_csv(outdir / "test_mae_by_coarse_class.csv")

    with open(outdir / "router_tree.txt", "w") as f:
        f.write(objects["tree_text"])

    joblib.dump(objects, outdir / "leaf_moe_model.joblib")

    print("\nSaved:")
    for fn in [
        "metrics.csv",
        "leaf_expert_report.csv",
        "router_selection.csv",
        "train_leaf_target_spread.csv",
        "val_leaf_target_spread.csv",
        "test_error_report.csv",
        "test_mae_by_leaf.csv",
        "test_mae_by_coarse_class.csv",
        "router_tree.txt",
        "leaf_moe_model.joblib",
    ]:
        print(" ", outdir / fn)

    print("\nHow to read the results:")
    print("1. Open train_leaf_target_spread.csv and val_leaf_target_spread.csv.")
    print("   If leaf sd_gap is still near 2 eV, the router cannot find uniform subspaces.")
    print("2. Open leaf_expert_report.csv.")
    print("   If most leaves choose 'global', splitting does not help with current features.")
    print("3. If leaves are more uniform but experts still fail, we need a real graph neural network.")
    print("4. If leaves are not more uniform, the structural features still do not expose the hidden variables.")


if __name__ == "__main__":
    main()
