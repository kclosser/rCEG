#!/usr/bin/env python3
"""
pqr_diagnose_and_strict_clean.py

Purpose:
Your model plateaued around ~0.94-1.02 eV MAE after ordinary cleaning.
This script checks whether the problem is:
1. target label mismatch,
2. descriptor/target row misalignment,
3. duplicate SMILES with conflicting labels,
4. weak descriptor representation,
5. too-broad chemical space.

It also writes a stricter cleaned file that DROPS conflicting duplicate molecules
instead of taking the median.

Usage:
    cd ~/Downloads/Closser
    python pqr_diagnose_and_strict_clean.py --data enhanced_dataset_lasso.json
    python pqr_gap_model.py --data enhanced_dataset_lasso_STRICT.jsonl --outdir runs/pqr_gap_strict
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV

try:
    from rdkit import Chem
    from rdkit import RDLogger
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


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def canon_smiles(smiles):
    if not RDKIT:
        return str(smiles), True, False, False
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None, False, False, False
    isolated_h = any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms())
    multi_frag = len(Chem.GetMolFrags(mol)) > 1
    return Chem.MolToSmiles(mol, canonical=True), True, isolated_h, multi_frag


def featurize(entries, include_homo_lumo=False):
    rows = []
    Xs = []
    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            continue
        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []
        gap = fnum(e[4])
        if not math.isfinite(gap):
            continue
        base = [fnum(v) for v in pqr[:7 if include_homo_lumo else 5]]
        x = np.array(base + [fnum(v) for v in desc], dtype=float)
        rows.append({
            "idx": i,
            "smiles": e[1],
            "gap": gap,
            "homo": fnum(pqr[5]) if len(pqr) > 5 else np.nan,
            "lumo": fnum(pqr[6]) if len(pqr) > 6 else np.nan,
            "calc_gap": (fnum(pqr[6]) - fnum(pqr[5])) if len(pqr) > 6 else np.nan,
        })
        Xs.append(x)
    max_len = max(len(x) for x in Xs)
    X = np.full((len(Xs), max_len), np.nan)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x
    return pd.DataFrame(rows), X


def model_score(df, X, y, label, seed=42, max_n=30000, shuffle_y=False):
    good = np.isfinite(y) & (y > 0.1) & (y < 25)
    df = df.loc[good].reset_index(drop=True)
    X = X[good]
    y = y[good]

    if len(y) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_n, replace=False)
        X, y, df = X[idx], y[idx], df.iloc[idx].reset_index(drop=True)

    if shuffle_y:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed)

    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_features=0.45, n_jobs=-1,
            random_state=seed, min_samples_leaf=1
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
            early_stopping=True, random_state=seed, loss="absolute_error"
        ),
        "Ridge": RidgeCV(alphas=np.logspace(-6, 3, 20)),
    }

    print(f"\nSanity model test: {label}")
    print(f"Rows used: {len(y):,}; target sd={np.std(y):.3f}")

    for name, m in models.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(with_centering=False)),
            ("model", m),
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        print(f"  {name:10s} MAE={mean_absolute_error(y[te], pred):.4f} eV   R2={r2_score(y[te], pred):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="enhanced_dataset_lasso_STRICT.jsonl")
    ap.add_argument("--conflict-threshold", type=float, default=0.05)
    ap.add_argument("--gap-min", type=float, default=0.10)
    ap.add_argument("--gap-max", type=float, default=20.0)
    ap.add_argument("--max-n", type=int, default=30000)
    args = ap.parse_args()

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded entries: {len(entries):,}")
    print(f"RDKit available: {RDKIT}")

    df, X = featurize(entries, include_homo_lumo=False)
    y = df["gap"].to_numpy(float)

    # 1. HOMO/LUMO target consistency.
    valid_hl = np.isfinite(df["calc_gap"].to_numpy(float)) & np.isfinite(y)
    diff = np.abs(df.loc[valid_hl, "calc_gap"].to_numpy(float) - y[valid_hl])
    print("\nTarget consistency check: entry[4] vs entry[2][6] - entry[2][5]")
    print(f"Rows with HOMO/LUMO available: {valid_hl.sum():,}")
    if len(diff):
        print(f"Median abs mismatch: {np.median(diff):.6f} eV")
        print(f"95th pct mismatch:   {np.percentile(diff,95):.6f} eV")
        print(f"Max mismatch:        {np.max(diff):.6f} eV")
        print(f"Rows mismatch >0.01: {(diff > 0.01).sum():,}")
        print(f"Rows mismatch >0.10: {(diff > 0.10).sum():,}")

    # 2. Duplicate conflicts.
    groups = defaultdict(list)
    meta = {}
    drop_counts = Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drop_counts["bad_entry_shape"] += 1
            continue
        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < args.gap_min or gap > args.gap_max:
            drop_counts["bad_gap"] += 1
            continue
        c, valid, isolated_h, multi_frag = canon_smiles(e[1])
        if not valid:
            drop_counts["invalid_smiles"] += 1
            continue
        if isolated_h:
            drop_counts["isolated_hydrogen"] += 1
            continue
        if multi_frag:
            drop_counts["multi_fragment"] += 1
            continue
        groups[c].append((gap, e))
        meta[c] = c

    strict_entries = []
    duplicate_groups = 0
    conflict_groups = 0
    conflict_rows = 0

    for c, vals in groups.items():
        if len(vals) > 1:
            duplicate_groups += 1
        gaps = np.array([v[0] for v in vals], dtype=float)
        if len(vals) > 1 and (gaps.max() - gaps.min() > args.conflict_threshold):
            conflict_groups += 1
            conflict_rows += len(vals)
            continue

        # If duplicates are consistent, keep one with median gap.
        chosen_gap, chosen_e = vals[int(np.argmin(np.abs(gaps - np.median(gaps))))]
        e = list(chosen_e)
        e[1] = c
        e[4] = float(np.median(gaps))
        strict_entries.append(e)

    print("\nDuplicate/conflict check")
    print(f"Canonical duplicate groups: {duplicate_groups:,}")
    print(f"Conflicting duplicate groups dropped: {conflict_groups:,}")
    print(f"Rows inside conflicting groups dropped: {conflict_rows:,}")

    with open(args.out, "w", encoding="utf-8") as f:
        for e in strict_entries:
            f.write(json.dumps(e) + "\n")
    print(f"\nWrote strict-clean dataset: {args.out}")
    print(f"Strict-clean rows: {len(strict_entries):,}")
    yy = np.array([e[4] for e in strict_entries], dtype=float)
    print(f"Strict target mean/sd/min/max: {yy.mean():.3f}/{yy.std():.3f}/{yy.min():.3f}/{yy.max():.3f}")

    # 3. Model sanity checks.
    df_base, X_base = featurize(strict_entries, include_homo_lumo=False)
    y_base = df_base["gap"].to_numpy(float)
    model_score(df_base, X_base, y_base, "PQR + LASSO descriptors only", max_n=args.max_n)

    df_hl, X_hl = featurize(strict_entries, include_homo_lumo=True)
    y_hl = df_hl["gap"].to_numpy(float)
    model_score(df_hl, X_hl, y_hl, "INCLUDING HOMO/LUMO columns as a leakage test", max_n=args.max_n)

    model_score(df_base, X_base, y_base, "SHUFFLED TARGET control", max_n=args.max_n, shuffle_y=True)

    print("\nInterpretation guide")
    print("--------------------")
    print("A) If INCLUDING HOMO/LUMO gets near 0 MAE: labels are mathematically consistent.")
    print("B) If INCLUDING HOMO/LUMO is still high: entry[4] is not LUMO-HOMO, or rows/columns are corrupted.")
    print("C) If SHUFFLED TARGET is close to the real descriptor score: descriptors are not aligned/predictive enough.")
    print("D) If strict clean still gives ~1 eV, the PQR/LASSO descriptors are missing key structural information.")
    print("   At that point we need Morgan fingerprints / RDKit descriptors / bond-step graph features from SMILES.")


if __name__ == "__main__":
    main()
