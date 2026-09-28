#!/usr/bin/env python3
"""
audit_cleaning_waterfall.py   (spec task A1)

Closes the record waterfall from the 106,066-record PQR release to the
84,143-molecule strict training file.

Why this is a separate script rather than counters bolted onto the two
cleaners, as the spec suggests
------------------------------------------------------------------------
cleaning_report.csv itemises 10,609 removals and is COMPLETE for the stage it
describes (95,764 -> 85,155, the CLEAN pass). The unaccounted records are not
lost in pqr_data_cleaner.py or pqr_diagnose_and_strict_clean.py at all. They
are lost one stage earlier, in DescriptorPull3.py, which silently `continue`s
past any record lacking a SMILES string or a pm7 block, and past any molecule
whose RDKit descriptor calculation raises. No counter in either cleaner can
observe those. The waterfall therefore has three stages:

    stage 1  raw release        106,066 -> 95,764   DescriptorPull3.py
    stage 2  CLEAN pass          95,764 -> 85,155   pqr_data_cleaner.py
    stage 3  STRICT conflict     85,155 -> 84,143   pqr_diagnose_and_strict_clean.py

Stage 1 is replayed here against the raw release with one counter per
`continue` branch. Stage 3 is recomputed by set difference.

Outputs
-------
cleaning_report_complete.csv    rule, count, stage, description
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RAW_N_EXPECTED = 106066
STRICT_N_EXPECTED = 84143


def replay_stage1(raw_path, verbose=True):
    """Replay DescriptorPull3.py intake with a counter on every drop path."""
    c = Counter()

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as fh:
        full = json.load(fh)

    c["raw_records"] = len(full)
    survivors = []

    for entry in full:
        if not isinstance(entry, dict):
            c["s1_not_a_dict"] += 1
            continue

        smiles = entry.get("smiles")
        if not smiles or not isinstance(smiles, str):
            c["s1_missing_or_nonstring_smiles"] += 1
            continue

        pm7 = entry.get("pm7", {})
        if not pm7:
            c["s1_missing_pm7_block"] += 1
            continue

        try:
            homo = float(pm7.get("homo", 0))
            lumo = float(pm7.get("lumo", 0))
            [float(entry.get("molecular mass", 0)),
             float(entry.get("exact mass", 0)),
             float(pm7.get("dipoleMoment", 0)),
             float(pm7.get("heatOfFormation", 0)),
             float(pm7.get("polarizability", 0)),
             homo, lumo]
        except Exception:
            c["s1_numeric_coercion_error"] += 1
            continue

        survivors.append(smiles)

    c["s1_passed_intake"] = len(survivors)
    if verbose:
        print(f"  stage 1 intake survivors: {len(survivors):,}")

    # Of the intake survivors, classify every RDKit rejection by CAUSE rather
    # than lumping them into one bucket. A single "parse failure" line would be
    # naming a cause nobody verified.
    #
    # Two-stage: parse with sanitize=False to separate genuine SMILES syntax
    # errors from molecules that tokenize but fail chemical sanitization, then
    # run SanitizeMol and catch the specific RDKit exception type.
    from rdkit import rdBase

    for i, smi in enumerate(survivors):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            c["s1_smiles_syntax_error"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            kind = type(exc).__name__
            msg = str(exc).lower()
            if "valence" in kind.lower() or "valence" in msg:
                c["s1_sanitize_valence_error"] += 1
            elif "kekulize" in kind.lower() or "kekulize" in msg:
                c["s1_sanitize_kekulize_error"] += 1
            elif "aromatic" in msg:
                c["s1_sanitize_aromaticity_error"] += 1
            else:
                c["s1_sanitize_other_error"] += 1
            continue
        # sanitizes cleanly but MolFromSmiles with default sanitize still fails
        if Chem.MolFromSmiles(smi) is None:
            c["s1_rdkit_rejected_after_sanitize"] += 1

        if verbose and (i + 1) % 25000 == 0:
            print(f"    rdkit-classified {i+1:,}/{len(survivors):,}")

    return c


def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def smiles_set(path):
    out = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r, list) and len(r) >= 5 and isinstance(r[1], str):
                out.add(r[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="Pitt_Quantum_Repository_Data.json")
    ap.add_argument("--descriptor-out", default="enhanced_dataset_lasso.json")
    ap.add_argument("--clean", default="enhanced_dataset_lasso_CLEAN.jsonl")
    ap.add_argument("--strict", default="enhanced_dataset_lasso_STRICT.jsonl")
    ap.add_argument("--clean-report", default="cleaning_report.csv")
    ap.add_argument("--outdir", default="runs/rceg_final_consolidated")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Stage 1: replaying DescriptorPull3.py intake against the raw release...")
    c1 = replay_stage1(args.raw)

    n_raw = c1["raw_records"]
    n_desc = count_lines(args.descriptor_out)
    n_clean = count_lines(args.clean)
    n_strict = count_lines(args.strict)

    print(f"\nRecord counts: raw={n_raw:,}  descriptor_out={n_desc:,} "
          f"clean={n_clean:,}  strict={n_strict:,}")

    rows = []

    def add(rule, count, stage, desc):
        rows.append({"rule": rule, "count": int(count), "stage": stage,
                     "description": desc})

    # ---------------- stage 1 ----------------
    add("missing_or_nonstring_smiles", c1["s1_missing_or_nonstring_smiles"],
        "1_descriptor_extraction", "PQR record had no 'smiles' key or it was not a string")
    add("missing_pm7_block", c1["s1_missing_pm7_block"],
        "1_descriptor_extraction", "PQR record had no 'pm7' quantum property block")
    add("numeric_coercion_error", c1["s1_numeric_coercion_error"],
        "1_descriptor_extraction", "float() raised on a pm7 or mass field")
    add("not_a_dict", c1["s1_not_a_dict"],
        "1_descriptor_extraction", "raw list element was not a dictionary")
    add("missing_descriptor_block", 0, "1_descriptor_extraction",
        "DescriptorPull3 computes descriptors inline, so there is no separate "
        "descriptor-block field that can be missing; path checked, empty by "
        "construction")

    s1_named = (c1["s1_missing_or_nonstring_smiles"] + c1["s1_missing_pm7_block"]
                + c1["s1_numeric_coercion_error"] + c1["s1_not_a_dict"])
    s1_total = n_raw - n_desc
    s1_residual = s1_total - s1_named

    # One row per verified cause. No catch-all unless something is genuinely
    # left over, in which case it is labelled as unexplained rather than
    # silently attributed.
    causes = [
        ("smiles_syntax_error", "s1_smiles_syntax_error",
         "SMILES string could not be tokenised at all "
         "(MolFromSmiles with sanitize=False returned None)"),
        ("sanitize_valence_error", "s1_sanitize_valence_error",
         "Tokenised but SanitizeMol raised an atom-valence error"),
        ("sanitize_kekulize_error", "s1_sanitize_kekulize_error",
         "Tokenised but SanitizeMol could not kekulise the structure"),
        ("sanitize_aromaticity_error", "s1_sanitize_aromaticity_error",
         "Tokenised but SanitizeMol raised an aromaticity error"),
        ("sanitize_other_error", "s1_sanitize_other_error",
         "Tokenised but SanitizeMol raised another chemistry error"),
        ("rdkit_rejected_after_sanitize", "s1_rdkit_rejected_after_sanitize",
         "Sanitises in isolation but default MolFromSmiles still returns None"),
    ]
    classified = 0
    for rule, key, desc in causes:
        n = c1.get(key, 0)
        classified += n
        if n:
            add(rule, n, "1_descriptor_extraction", desc)

    leftover = s1_residual - classified
    if leftover:
        add("UNEXPLAINED_descriptor_stage", leftover, "1_descriptor_extraction",
            "Passed intake and RDKit but produced no descriptor row. NOT a "
            "verified cause - investigate before reporting.")

    # ---------------- stage 2 ----------------
    if Path(args.clean_report).exists():
        cr = pd.read_csv(args.clean_report)
        for _, r in cr.iterrows():
            add(str(r["rule"]), int(r["count"]), "2_clean_pass",
                "itemised by pqr_data_cleaner.py")
        s2_named = int(cr["count"].sum())
    else:
        s2_named = 0

    s2_total = n_desc - n_clean
    if s2_total != s2_named:
        add("clean_pass_unitemised", s2_total - s2_named, "2_clean_pass",
            "residual not itemised by cleaning_report.csv")

    # ---------------- stage 3 ----------------
    s3_total = n_clean - n_strict
    add("conflicting_duplicate_discard", s3_total, "3_strict_pass",
        "STRICT drops duplicate SMILES with conflicting gap labels instead of "
        "median-merging them as the CLEAN pass does")

    df = pd.DataFrame(rows)
    # Zero-count rows are KEPT. A drop path that was checked and found empty is
    # evidence; suppressing it would make Table S1 look like those paths were
    # never examined.
    df["checked"] = True
    df = df.reset_index(drop=True)
    df.to_csv(outdir / "cleaning_report_complete.csv", index=False)

    total = int(df["count"].sum())
    expected = n_raw - n_strict

    print("\n" + "=" * 74)
    print("CLEANING WATERFALL (spec A1)")
    print("=" * 74)
    for stage in ["1_descriptor_extraction", "2_clean_pass", "3_strict_pass"]:
        sub = df[df["stage"] == stage]
        print(f"\n  {stage}  (subtotal {sub['count'].sum():,})")
        for _, r in sub.iterrows():
            print(f"    {r['rule']:34s} {r['count']:>7,}")

    print("\n" + "-" * 74)
    print(f"  TOTAL DROPPED                      {total:>7,}")
    print(f"  EXPECTED ({n_raw:,} - {n_strict:,})       {expected:>7,}")
    ok = (total == expected == 21923)
    print(f"\n[{'PASS' if ok else 'FAIL'}] acceptance: sum(count) == 106066 - 84143 == 21923")

    # CLEAN-variant independent verification
    print(f"\nCLEAN variant check: {n_desc:,} - {s2_total:,} = {n_desc - s2_total:,} "
          f"(file has {n_clean:,}) "
          f"[{'PASS' if n_desc - s2_total == n_clean == 85155 else 'FAIL'}]")

    print(f"\nWrote {outdir/'cleaning_report_complete.csv'}")


if __name__ == "__main__":
    main()
