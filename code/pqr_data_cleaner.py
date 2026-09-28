#!/usr/bin/env python3
"""
pqr_data_cleaner.py

Audits and cleans enhanced_dataset_lasso.json/jsonl before ML training.

Main cleaning choices:
1. Parse JSON-lines correctly.
2. Drop invalid RDKit SMILES.
3. Drop molecules with disconnected/isolated hydrogen atoms, which cause:
      WARNING: not removing hydrogen atom without neighbors
4. Drop multi-fragment molecules by default unless --keep-fragments is used.
5. Drop zero/near-zero HOMO-LUMO gaps.
6. Drop extreme/unphysical gaps using percentile trimming.
7. Remove descriptor columns that are mostly missing, constant, or almost constant.
8. Deduplicate canonical SMILES; by default, keep the median target row.

Usage:
    cd ~/Downloads/Closser
    python pqr_data_cleaner.py --data enhanced_dataset_lasso.json --out enhanced_dataset_lasso_CLEAN.jsonl
    python pqr_gap_model.py --data enhanced_dataset_lasso_CLEAN.jsonl --outdir runs/pqr_gap_clean
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
            # Could be a full dataset or one entry. A real entry has a SMILES string at index 1.
            if len(obj) >= 5 and isinstance(obj[1], str):
                return [obj]
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                    data.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], list):
                    data.extend(obj)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_no}: {e}")
    return data


def canonicalize(smiles: str):
    if not RDKIT:
        return str(smiles), None, True, False, False

    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        return None, None, False, False, False

    isolated_h = any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms())
    multi_fragment = len(Chem.GetMolFrags(mol)) > 1
    canon = Chem.MolToSmiles(mol, canonical=True)
    return canon, mol, True, isolated_h, multi_fragment


def safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="enhanced_dataset_lasso_CLEAN.jsonl")
    ap.add_argument("--report", default="cleaning_report.csv")
    ap.add_argument("--keep-fragments", action="store_true", help="Keep multi-fragment/salt-like SMILES.")
    ap.add_argument("--gap-min", type=float, default=0.10, help="Drop gaps below this value.")
    ap.add_argument("--gap-max", type=float, default=20.0, help="Hard upper limit for gap.")
    ap.add_argument("--trim-percentiles", default="0.25,99.75", help="Extra target trim percentiles, e.g. 0.25,99.75")
    ap.add_argument("--missing-col-frac", type=float, default=0.30, help="Drop descriptor columns with more missing than this.")
    ap.add_argument("--dedupe", choices=["median", "first", "none"], default="median")
    args = ap.parse_args()

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded raw entries: {len(entries):,}")
    print(f"RDKit available: {RDKIT}")

    rows = []
    drops = Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5):
            drops["bad_entry_shape"] += 1
            continue

        smiles = str(e[1])
        gap = safe_float(e[4])
        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []

        if not math.isfinite(gap):
            drops["nonfinite_gap"] += 1
            continue
        if gap < args.gap_min:
            drops["gap_below_min_or_placeholder"] += 1
            continue
        if gap > args.gap_max:
            drops["gap_above_hard_max"] += 1
            continue

        canon, mol, valid, isolated_h, multi_fragment = canonicalize(smiles)
        if not valid:
            drops["invalid_smiles"] += 1
            continue
        if isolated_h:
            drops["isolated_hydrogen"] += 1
            continue
        if multi_fragment and not args.keep_fragments:
            drops["multi_fragment"] += 1
            continue

        x = np.array([safe_float(v) for v in (list(pqr)[:5] + list(desc))], dtype=float)
        missing_frac = np.mean(~np.isfinite(x)) if x.size else 1.0
        if x.size == 0 or missing_frac > 0.50:
            drops["too_many_missing_descriptors_in_row"] += 1
            continue

        rows.append({
            "old_index": i,
            "entry": e,
            "canon": canon,
            "smiles": smiles,
            "gap": gap,
            "x": x,
        })

    if not rows:
        raise RuntimeError("No rows survived basic cleaning.")

    gaps = np.array([r["gap"] for r in rows], dtype=float)
    lo_p, hi_p = [float(v) for v in args.trim_percentiles.split(",")]
    lo, hi = np.percentile(gaps, [lo_p, hi_p])
    kept = []
    for r in rows:
        if r["gap"] < lo or r["gap"] > hi:
            drops["target_percentile_trim"] += 1
        else:
            kept.append(r)
    rows = kept

    # Descriptor column quality report.
    max_len = max(len(r["x"]) for r in rows)
    X = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        X[i, :len(r["x"])] = r["x"]

    missing_col = np.mean(~np.isfinite(X), axis=0)
    col_std = np.nanstd(X, axis=0)
    good_cols = (missing_col <= args.missing_col_frac) & np.isfinite(col_std) & (col_std > 1e-12)
    good_idx = np.where(good_cols)[0]

    print(f"Descriptor columns before cleaning: {X.shape[1]:,}")
    print(f"Descriptor columns retained:        {len(good_idx):,}")

    cleaned_entries = []
    if args.dedupe == "none":
        selected_rows = rows
    else:
        by_canon = defaultdict(list)
        for r in rows:
            by_canon[r["canon"]].append(r)

        selected_rows = []
        duplicate_groups = 0
        duplicate_target_conflict = 0

        for canon, group in by_canon.items():
            if len(group) == 1:
                selected_rows.append(group[0])
                continue

            duplicate_groups += 1
            ys = np.array([g["gap"] for g in group])
            if ys.max() - ys.min() > 0.05:
                duplicate_target_conflict += 1

            if args.dedupe == "first":
                chosen = group[0]
            else:
                med = float(np.median(ys))
                chosen = min(group, key=lambda g: abs(g["gap"] - med))
                chosen["entry"][4] = med
            selected_rows.append(chosen)

        drops["duplicate_rows_removed"] += len(rows) - len(selected_rows)
        print(f"Duplicate canonical SMILES groups: {duplicate_groups:,}")
        print(f"Duplicate groups with target spread >0.05 eV: {duplicate_target_conflict:,}")

    for r in selected_rows:
        e = list(r["entry"])
        pqr5 = list(e[2])[:5] if isinstance(e[2], list) else []
        desc = list(e[3]) if isinstance(e[3], list) else []

        combined = np.array([safe_float(v) for v in (pqr5 + desc)], dtype=float)
        new_combined = []
        for j in good_idx:
            if j < len(combined):
                v = combined[j]
            else:
                v = np.nan
            new_combined.append(None if not math.isfinite(v) else float(v))

        # Keep original format, but put retained descriptors into entry[3].
        # Keep entry[2] as the original first five physical descriptors.
        e[1] = r["canon"]
        e[2] = [None if not math.isfinite(safe_float(v)) else float(v) for v in pqr5]
        e[3] = new_combined[5:] if len(new_combined) > 5 else []
        e[4] = float(r["gap"])
        cleaned_entries.append(e)

    with open(args.out, "w", encoding="utf-8") as f:
        for e in cleaned_entries:
            f.write(json.dumps(e) + "\n")

    report = pd.DataFrame({
        "rule": list(drops.keys()),
        "count": list(drops.values()),
    }).sort_values("count", ascending=False)
    report.to_csv(args.report, index=False)

    final_gaps = np.array([e[4] for e in cleaned_entries], dtype=float)
    print("\nCleaning summary")
    print("----------------")
    print(f"Final entries: {len(cleaned_entries):,}")
    print(f"Target gap mean/sd/min/max: {final_gaps.mean():.3f} / {final_gaps.std():.3f} / {final_gaps.min():.3f} / {final_gaps.max():.3f}")
    print("\nDropped rows:")
    print(report.to_string(index=False))
    print(f"\nWrote cleaned data: {args.out}")
    print(f"Wrote report:       {args.report}")


if __name__ == "__main__":
    main()
