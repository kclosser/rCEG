#!/usr/bin/env python3
"""
audit_recompute_outcomes.py   (spec task A3)

Reconciles the Psi4 recomputation bookkeeping across three files:

    recompute_manifest.csv            molecules targeted for recomputation
    pqr_recomputed_reference.csv      converged results
    pqr_recomputed_reference_bad.csv  failure records

The manuscript states 950 targets and 41 failures, while the bad file holds 84
records. The spec's hypothesis was that the bad file logs retried ATTEMPTS
rather than distinct molecules. This script tests that directly and emits a
per-domain reconciliation.

Outputs
-------
recompute_audit.csv            per-domain reconciliation + a TOTAL row
recompute_failure_modes.csv    every non-converged molecule with its cause
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST = "runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/recompute_manifest.csv"

# Ordered: first pattern to match wins.
FAILURE_PATTERNS = [
    ("disk_or_scratch_exhausted",
     r"no space left on device|psio_error|wt_toclen|error writing to file"),
    ("scf_non_convergence",
     r"scf.{0,40}(did not converge|not converged|failed)|iterations exceeded|convergence.{0,20}fail"),
    ("geometry_embedding_failure",
     r"embed_failed|bad_smiles|could not embed|conformer"),
    ("basis_assignment_failure",
     r"basis|bse|no basis set"),
    ("memory_exhausted",
     r"out of memory|memory allocation|bad_alloc"),
    ("orbital_extraction_failure",
     r"could not identify homo/lumo"),
    ("manually_marked_failed",
     r"manually_marked_failed"),
]


def classify_failure(reason):
    text = str(reason).lower()
    for label, pattern in FAILURE_PATTERNS:
        if re.search(pattern, text):
            return label
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--converged", default="pqr_recomputed_reference.csv")
    ap.add_argument("--bad", default="pqr_recomputed_reference_bad.csv")
    ap.add_argument("--corpus", default=None,
                    help="Optional corpus_composition.csv for fraction_of_domain_population")
    ap.add_argument("--outdir", default="runs/rceg_final_consolidated")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest)
    good = pd.read_csv(args.converged)
    bad = pd.read_csv(args.bad)

    # Domain is authoritative from the manifest; the bad file's domain column
    # is partially blank.
    dom = dict(zip(man["id"], man["domain"]))
    bad["domain_resolved"] = bad["id"].map(dom)
    good["domain_resolved"] = good["id"].map(dom).fillna(good.get("domain"))

    man_ids = set(man["id"])
    good_ids = set(good["id"])
    bad_ids = set(bad["id"])

    retried_then_converged = bad_ids & good_ids
    never_converged = man_ids - good_ids
    failed_logged = never_converged & bad_ids
    failed_unlogged = never_converged - bad_ids

    bad["failure_mode"] = bad["reason"].map(classify_failure)
    bad["later_converged"] = bad["id"].isin(good_ids)

    rows = []
    pop = None
    if args.corpus and Path(args.corpus).exists():
        cc = pd.read_csv(args.corpus)
        sub = cc[cc["metric"] == "domain_count"] if "metric" in cc.columns else cc
        pop = dict(zip(sub.get("group", sub.iloc[:, 0]), sub.get("value", sub.iloc[:, 1])))

    for domain in sorted(man["domain"].dropna().unique()):
        d_man = set(man.loc[man["domain"] == domain, "id"])
        d_good = d_man & good_ids
        d_never = d_man - good_ids
        d_badrec = bad[bad["id"].isin(d_man)]

        modes = (d_badrec[~d_badrec["later_converged"]]["failure_mode"]
                 .value_counts().to_dict())
        mode_str = "; ".join(f"{k}={v}" for k, v in sorted(modes.items())) or "none"

        rows.append({
            "domain": domain,
            "n_targeted": len(d_man),
            "n_converged": len(d_good),
            "n_failed": len(d_never),
            "n_failure_records": int(len(d_badrec)),
            "n_records_that_later_converged": int(d_badrec["later_converged"].sum()),
            "n_failed_with_no_record": len(d_never - bad_ids),
            "fraction_of_domain_population": (
                round(len(d_man) / pop[domain], 6) if pop and domain in pop else ""
            ),
            "failure_mode": mode_str,
        })

    total_modes = (bad[~bad["later_converged"]]["failure_mode"]
                   .value_counts().to_dict())
    rows.append({
        "domain": "TOTAL",
        "n_targeted": len(man_ids),
        "n_converged": len(good_ids),
        "n_failed": len(never_converged),
        "n_failure_records": len(bad),
        "n_records_that_later_converged": len(retried_then_converged),
        "n_failed_with_no_record": len(failed_unlogged),
        "fraction_of_domain_population": "",
        "failure_mode": "; ".join(f"{k}={v}" for k, v in sorted(total_modes.items())),
    })

    audit = pd.DataFrame(rows)
    audit.to_csv(outdir / "recompute_audit.csv", index=False)

    detail = pd.DataFrame({"id": sorted(never_converged)})
    detail["domain"] = detail["id"].map(dom)
    rmap = dict(zip(bad["id"], bad["reason"]))
    mmap = dict(zip(bad["id"], bad["failure_mode"]))
    detail["reason"] = detail["id"].map(rmap).fillna("NO RECORD IN BAD FILE")
    detail["failure_mode"] = detail["id"].map(mmap).fillna("unrecorded")
    detail.to_csv(outdir / "recompute_failure_modes.csv", index=False)

    # ---------------- reconciliation report ----------------
    print("=" * 72)
    print("RECOMPUTE RECONCILIATION (spec A3)")
    print("=" * 72)
    print(audit.to_string(index=False))

    print("\nHypothesis test - does the bad file log retried ATTEMPTS?")
    print(f"  bad-file records                 : {len(bad)}")
    print(f"  distinct molecule ids in it      : {len(bad_ids)}")
    print(f"  ids that LATER converged (retry) : {len(retried_then_converged)}")
    print(f"  ids that never converged         : {len(bad_ids - good_ids)}")
    verdict = "CONFIRMED" if retried_then_converged else "REFUTED"
    print(f"  verdict: {verdict}")

    print("\nAcceptance checks (spec A3):")
    checks = [
        ("n_targeted sums to 950", len(man_ids), 950),
        ("n_converged sums to 909", len(good_ids), 909),
        ("n_failed sums to 41", len(never_converged), 41),
    ]
    for label, actual, expected in checks:
        flag = "PASS" if actual == expected else "FAIL"
        print(f"  [{flag}] {label:28s} actual={actual}")

    print("\nPer-domain targeting (manuscript asserts 250/250/250/100/100):")
    for _, r in audit[audit.domain != "TOTAL"].iterrows():
        print(f"  {r.domain:26s} targeted={r.n_targeted:4d}  failed={r.n_failed:3d}")

    if failed_unlogged:
        print(f"\nWARNING: {len(failed_unlogged)} targeted molecules have NO outcome "
              f"record in either file: {sorted(failed_unlogged)}")

    print(f"\nWrote {outdir/'recompute_audit.csv'}")
    print(f"Wrote {outdir/'recompute_failure_modes.csv'}")


if __name__ == "__main__":
    main()
