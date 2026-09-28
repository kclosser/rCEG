#!/usr/bin/env python3
"""
audit_corpus_composition.py   (spec task A2)

Characterises the strict 84,143-molecule PQR corpus for manuscript Tables 2, 3
and 5, and quantifies the QM9 overlap.

Emits a long-format corpus_composition.csv with columns:
    section, metric, group, value, note

Domain assignment reuses classify_domain() from the live model script so the
audit cannot drift from what the model actually does.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MODEL_SCRIPT = "pqr_full_domain_moe_qmugs_external.py"


def load_classify_domain():
    """Import classify_domain from the live model script, not a copy."""
    spec = importlib.util.spec_from_file_location("_rceg_model", MODEL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="enhanced_dataset_lasso_STRICT.jsonl")
    ap.add_argument("--qm9-ref", default="qm9_gap_reference.csv")
    ap.add_argument("--outdir", default="runs/rceg_final_consolidated")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    classify_domain = load_classify_domain()

    domains = Counter()
    elements = Counter()
    charged_by_domain = Counter()
    radical_by_domain = Counter()
    heavy_counts = []
    gaps = []
    canon_smiles = set()
    n_lines = n_parsed = n_badsmiles = 0

    print(f"Scanning {args.data} ...")
    with open(args.data, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not (isinstance(rec, list) and len(rec) >= 5 and isinstance(rec[1], str)):
                continue

            mol = Chem.MolFromSmiles(rec[1], sanitize=True)
            if mol is None:
                n_badsmiles += 1
                continue

            n_parsed += 1
            canon_smiles.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

            d = classify_domain(mol)
            domains[d] += 1

            heavy_counts.append(mol.GetNumHeavyAtoms())
            try:
                gaps.append(float(rec[4]))
            except Exception:
                pass

            syms = {a.GetSymbol() for a in mol.GetAtoms()}
            for s in syms:
                elements[s] += 1
            if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) is False:
                elements["H(implicit)"] += 0  # placeholder, H usually implicit

            if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
                charged_by_domain[d] += 1
            if sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) > 0:
                radical_by_domain[d] += 1

            if n_parsed % 20000 == 0:
                print(f"  {n_parsed:,} parsed ...")

    heavy = np.array(heavy_counts, dtype=float)
    rows = []

    def add(section, metric, group, value, note=""):
        rows.append({"section": section, "metric": metric, "group": group,
                     "value": value, "note": note})

    add("corpus", "lines_read", "strict_file", n_lines)
    add("corpus", "molecules_parsed", "strict_file", n_parsed)
    add("corpus", "smiles_parse_failures", "strict_file", n_badsmiles)
    add("corpus", "unique_canonical_smiles", "strict_file", len(canon_smiles))

    for d, c in sorted(domains.items()):
        add("domain", "domain_count", d, c)
    add("domain", "domain_count", "TOTAL", sum(domains.values()))

    add("heavy_atoms", "min", "all", int(np.min(heavy)))
    add("heavy_atoms", "p05", "all", float(np.percentile(heavy, 5)))
    add("heavy_atoms", "median", "all", float(np.median(heavy)))
    add("heavy_atoms", "p95", "all", float(np.percentile(heavy, 95)))
    add("heavy_atoms", "max", "all", int(np.max(heavy)))
    add("heavy_atoms", "mean", "all", round(float(np.mean(heavy)), 3))

    if gaps:
        ga = np.array(gaps)
        for k, v in [("min", ga.min()), ("median", np.median(ga)),
                     ("mean", ga.mean()), ("sd", ga.std()), ("max", ga.max())]:
            add("gap_ev", k, "all", round(float(v), 4))

    for el, c in sorted(elements.items(), key=lambda kv: -kv[1]):
        if c > 0:
            add("elements", "molecules_containing", el, c)

    for d in sorted(domains):
        add("charge_radical", "n_charged", d, charged_by_domain.get(d, 0))
        add("charge_radical", "n_radical", d, radical_by_domain.get(d, 0))
    add("charge_radical", "n_charged", "TOTAL", sum(charged_by_domain.values()))
    add("charge_radical", "n_radical", "TOTAL", sum(radical_by_domain.values()))

    # ---------------- QM9 overlap ----------------
    if Path(args.qm9_ref).exists():
        q = pd.read_csv(args.qm9_ref, usecols=["smiles"])
        qc = set()
        for s in q["smiles"]:
            m = Chem.MolFromSmiles(str(s))
            if m is not None:
                qc.add(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True))
        ov = canon_smiles & qc
        add("qm9", "qm9_molecules_parsed", "qm9", len(qc))
        add("qm9", "overlap_with_pqr", "qm9_and_pqr", len(ov))
        add("qm9", "overlap_fraction_of_qm9", "qm9", round(len(ov) / max(1, len(qc)), 6))
        add("qm9", "overlap_fraction_of_pqr", "pqr", round(len(ov) / max(1, len(canon_smiles)), 6))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "corpus_composition.csv", index=False)

    # ---------------- acceptance ----------------
    total = sum(domains.values())
    print("\n" + "=" * 66)
    print("ACCEPTANCE (spec A2)")
    print("=" * 66)
    print(f"[{'PASS' if total == 84143 else 'FAIL'}] domain counts sum to 84143 -> {total}")

    expected = {"near_qm9_larger_organic": 23515, "large_neutral_organic": 18082,
                "charged_or_radical": 6883, "heteroatom_rich_non_qm9": 20765}
    for d, exp in expected.items():
        act = domains.get(d, 0)
        print(f"[{'PASS' if act == exp else 'FAIL'}] {d:26s} expected={exp:6d} actual={act:6d}")
    print(f"[----] qm9_like_small_organic    actual={domains.get('qm9_like_small_organic',0)} (not previously quoted)")

    print(f"\nHeavy atoms: min={int(heavy.min())} max={int(heavy.max())} "
          f"median={np.median(heavy):.0f} p05={np.percentile(heavy,5):.0f} p95={np.percentile(heavy,95):.0f}")
    print(f"Wrote {outdir/'corpus_composition.csv'}")


if __name__ == "__main__":
    main()
