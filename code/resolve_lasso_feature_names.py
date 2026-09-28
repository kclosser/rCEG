#!/usr/bin/env python3
"""
resolve_lasso_feature_names.py   (spec task E4)

Recover the human-readable names of the 375 "LASSO" descriptors stored in
enhanced_dataset_lasso_STRICT.jsonl record[3].

Why not just re-run the selection, as the spec suggests?
-------------------------------------------------------
The spec says to re-run LASSO in MordredPull2.py. That is the wrong script.
MordredPull2.py (and MordredPull.py) both print "Old 375 LASSO features" as a
PRIOR baseline they were trying to beat -- they emit a NEW 500-feature
RDKit+Mordred set that was never adopted. The 375-feature block actually present
in the training data was produced by DescriptorPull3.py, whose pool is:

    RDKit Descriptors.descList   (~200 descriptors)
  + MorganFP_0 .. MorganFP_199   (first 200 bits of a 1024-bit r=2 Morgan FP)

then VarianceThreshold(0.01), StandardScaler, LassoCV(cv=3, random_state=42),
and finally the top min(500, n_surviving) features ordered by DESCENDING
absolute LASSO coefficient.

Re-running that would depend on LassoCV reconverging identically under a
different RDKit/sklearn version, which is not guaranteed. Instead this script
IDENTIFIES each stored column by matching its values against a freshly computed
descriptor pool over a sample of molecules. That is exact and verifiable.

Outputs
-------
lasso_selected_feature_names.json   ordered list of 375 resolved names
lasso_name_resolution_report.csv    per-column match evidence and confidence
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

N_MORGAN_BITS = 200
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024


def descriptor_pool(smiles):
    """Reproduce DescriptorPull3.calculate_rdkit_descriptors exactly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    out = {}
    for name, fn in Descriptors.descList:
        try:
            out[name] = float(fn(mol))
        except Exception:
            pass

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        for i in range(min(N_MORGAN_BITS, MORGAN_NBITS)):
            out[f"MorganFP_{i}"] = float(fp[i])
    except Exception:
        pass

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="enhanced_dataset_lasso_STRICT.jsonl")
    ap.add_argument("--n-sample", type=int, default=400,
                    help="molecules used to fingerprint each column")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load a sample of molecules with their stored LASSO vectors ----
    smiles_list, stored = [], []
    with open(args.data, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not (isinstance(rec, list) and len(rec) >= 5 and isinstance(rec[1], str)):
                continue
            lasso = rec[3]
            if not isinstance(lasso, list) or not lasso:
                continue
            if Chem.MolFromSmiles(rec[1]) is None:
                continue
            smiles_list.append(rec[1])
            stored.append(lasso)
            if len(smiles_list) >= args.n_sample:
                break

    width = max(len(v) for v in stored)
    S = np.full((len(stored), width), np.nan, dtype=float)
    for i, v in enumerate(stored):
        S[i, :len(v)] = [float(x) if x is not None else np.nan for x in v]

    print(f"Loaded {len(smiles_list)} molecules; stored LASSO block width = {width}")

    # ---- compute the candidate pool ----
    pool_rows = []
    for smi in smiles_list:
        d = descriptor_pool(smi)
        pool_rows.append(d if d is not None else {})

    pool = pd.DataFrame(pool_rows)
    pool = pool.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pool_names = list(pool.columns)
    P = pool.to_numpy(dtype=float)
    print(f"Candidate pool: {len(pool_names)} descriptors "
          f"({sum(1 for n in pool_names if n.startswith('MorganFP_'))} Morgan bits)")

    # ---- match each stored column to a pool column ----
    records, resolved = [], []
    used = {}

    for j in range(width):
        col = S[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 10:
            records.append({"x_index": j + 5, "lasso_index": j, "resolved_name": None,
                            "confidence": "unresolved", "n_candidates": 0,
                            "note": "too few finite stored values"})
            resolved.append(None)
            continue

        cands = []
        for k, name in enumerate(pool_names):
            if np.allclose(col[ok], P[ok, k], rtol=args.rtol, atol=args.atol, equal_nan=False):
                cands.append(name)

        if len(cands) == 1:
            name, conf = cands[0], "unique"
        elif len(cands) > 1:
            # Several pool columns are numerically identical on this sample
            # (e.g. constant-valued Morgan bits). Prefer a not-yet-used name.
            fresh = [c for c in cands if c not in used]
            name = fresh[0] if fresh else cands[0]
            conf = "ambiguous"
        else:
            name, conf = None, "unresolved"

        if name is not None:
            used[name] = used.get(name, 0) + 1

        records.append({"x_index": j + 5, "lasso_index": j, "resolved_name": name,
                        "confidence": conf, "n_candidates": len(cands),
                        "note": "" if conf == "unique" else ";".join(cands[:6])})
        resolved.append(name)

    rep = pd.DataFrame(records)
    rep.to_csv(outdir / "lasso_name_resolution_report.csv", index=False)

    n_uni = int((rep.confidence == "unique").sum())
    n_amb = int((rep.confidence == "ambiguous").sum())
    n_unr = int((rep.confidence == "unresolved").sum())

    print(f"\nResolution: unique={n_uni}  ambiguous={n_amb}  unresolved={n_unr}  of {width}")

    with open(outdir / "lasso_selected_feature_names.json", "w") as fh:
        json.dump({
            "source_script": "DescriptorPull3.py",
            "method": "value-matching against a freshly computed RDKit descList + 200 Morgan bit pool",
            "pool_size": len(pool_names),
            "n_sample_molecules": len(smiles_list),
            "ordering": "descending |LassoCV coefficient| (DescriptorPull3 argsort[-n:][::-1])",
            "x_index_offset": 5,
            "n_resolved_unique": n_uni,
            "n_ambiguous": n_amb,
            "n_unresolved": n_unr,
            "names": resolved,
        }, fh, indent=2)

    morgan = [n for n in resolved if n and n.startswith("MorganFP_")]
    print(f"Morgan fingerprint bits among the selected features: {len(morgan)}")
    if morgan:
        print("  -> NOTE: the model's feature set DOES contain Morgan bits.")
        print("     Examples:", ", ".join(morgan[:8]))

    print(f"\nWrote {outdir/'lasso_selected_feature_names.json'}")
    print(f"Wrote {outdir/'lasso_name_resolution_report.csv'}")


if __name__ == "__main__":
    main()
