#!/usr/bin/env python3
"""
run_C4_perclass_repeats.py   (closes the Table 7 single-split problem)

Re-runs the frozen ten-repeat protocol and collects what the original run did
not retain: **per-chemical-class test metrics for every repeat**.

Why this is needed
------------------
Table 7 as published is a single split. Its overall MAE is 0.386 eV while the
headline, averaged over ten repeats, is 0.338 +- 0.026 eV. Both are correct,
but a reader who reconstructs the overall figure from the class rows recovers
0.386 and will ask why the representative split sits ~1.8 SD above the mean.
Averaging the class rows over the same ten repeats removes the question and
supplies the per-class spreads Reviewer 3 asked for.

This calls the model module's own `run_repeat`, so the numbers are the frozen
architecture's, not a reimplementation. The only addition is that per-class
metrics are computed from the returned test predictions and kept per repeat.

Second question answered here
-----------------------------
The calibration mapping (Stage 1) reaches 0.331 eV on held-out molecules while
being handed the semiempirical gap; rCEG reaches 0.338 eV without it. Those two
figures are measured on different molecule sets (849 pooled val+test versus 424
test), so they are not directly comparable as published. This script evaluates
both **on the same test partition in the same repeat**, which makes the
comparison honest.

The calibration ensemble is refitted here with the same seed and split that
`run_repeat` used internally. Fitting is deterministic, so the refitted object
is the same one; this is a way to obtain its test predictions, not a second
model.

    PYTHONHASHSEED=0 python3 run_C4_perclass_repeats.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rceg_inputs import (add_input_args, build_inputs, excise_external,
                         load_model_module)

OUT = Path("runs/revision2")
CONFIGS = ["piecewise_split", "domain_expert"]


def per_class(m, y_true, y_pred, domains, config, repeat):
    rows = []
    for dom in sorted(pd.unique(domains)):
        k = domains == dom
        if int(k.sum()) < 2:
            continue
        mt = m.full_metrics(y_true[k], y_pred[k])
        rows.append({"repeat": repeat, "config": config, "domain": dom,
                     "n": int(k.sum()), **mt})
    mt = m.full_metrics(y_true, y_pred)
    rows.append({"repeat": repeat, "config": config, "domain": "ALL",
                 "n": int(len(y_true)), **mt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    add_input_args(ap)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    m = load_model_module()
    pqr, audit, ref = build_inputs(m, args)
    pqr, audit, ref, _ = excise_external(m, pqr, audit, ref, args.external_ref)

    feat = m.leakage_filter(pqr, audit, 0.94, OUT)
    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random",
                                   seed=args.seed)
    print(f"\nreference pool {len(ref_pqr):,}, {len(feat)} features, "
          f"{args.repeats} repeats\n", flush=True)

    rows, overall, cal_rows = [], [], []
    for rep in range(args.repeats):
        seed = args.seed + 1000 * rep
        t0 = time.perf_counter()

        results, ref_test, pred_t = m.run_repeat(
            pqr, ref_pqr, feat, seed, rep, "random", groups,
            want_permutation=False)

        y = ref_test["ref_gap"].to_numpy()
        dom = ref_test["domain"].to_numpy()
        for cfg in CONFIGS:
            if cfg in pred_t:
                rows += per_class(m, y, np.asarray(pred_t[cfg]), dom, cfg, rep)

        # Stage 1 evaluated on the SAME test partition, for a fair comparison.
        ci, vi, _ti = m.grouped_three_way_split(groups, seed)
        cal_predict, *_ = m.fit_calibration_ensemble(
            ref_pqr.iloc[ci].copy(), ref_pqr.iloc[vi].copy(), feat, seed)
        cal_pred = cal_predict(ref_test)
        rows += per_class(m, y, np.asarray(cal_pred), dom,
                          "calibration_stage1", rep)
        cal_rows.append({
            "repeat": rep, "seed": seed, "n_test": int(len(y)),
            "calibration_stage1_mae": m.full_metrics(y, cal_pred)["mae"],
            **{f"{c}_mae": results[c]["mae"] for c in CONFIGS if c in results},
        })

        for cfg in CONFIGS:
            if cfg in results:
                overall.append({"repeat": rep, "config": cfg, **results[cfg]})
        print(f"  repeat {rep}: " + "  ".join(
            f"{c}={results[c]['mae']:.4f}" for c in CONFIGS if c in results) +
            f"  cal={cal_rows[-1]['calibration_stage1_mae']:.4f}"
            f"  [{time.perf_counter()-t0:.0f}s]", flush=True)

    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / "C4_perclass_per_repeat.csv", index=False)
    pd.DataFrame(overall).to_csv(OUT / "C4_overall_per_repeat.csv", index=False)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "C4_calibration_vs_rceg.csv", index=False)

    summ = (pc.groupby(["config", "domain"])
              .agg(n=("n", "mean"),
                   mae=("mae", "mean"), mae_sd=("mae", "std"),
                   rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                   r2=("r2", "mean"), r2_sd=("r2", "std"),
                   median_abs_error=("median_abs_error", "mean"),
                   n_repeats=("mae", "size"))
              .reset_index())
    summ.to_csv(OUT / "C4_perclass_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print(summ[summ.config == "piecewise_split"].round(4).to_string(index=False))
    print("\ncalibration (Stage 1, given the semiempirical gap) vs rCEG, "
          "same test partition:")
    for c in ["calibration_stage1_mae"] + [f"{c}_mae" for c in CONFIGS]:
        if c in cal.columns:
            print(f"  {c:28s} {cal[c].mean():.4f} +- {cal[c].std():.4f}")
    print(f"\nwrote {OUT/'C4_perclass_summary.csv'} and three companions")


if __name__ == "__main__":
    main()
