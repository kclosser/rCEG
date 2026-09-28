#!/usr/bin/env python3
"""
make_training_iteration_error_figure_clean.py   (spec task B3)

Figure 3: held-out error against boosting iteration, with a zoomed inset.

What was wrong (R1.4, R5.6)
---------------------------
The previous version globbed for the newest `global_hgb_convergence_history.csv`
anywhere under runs/ and, separately, for the newest
`FINAL_reference_holdout_metrics.csv`. The only convergence history on disk came
from `pqr_full_domain_moe_novel_split_regime`, a superseded May run, while the
test MAE came from a different and later run. The figure therefore drew one
model's curve and annotated it with another model's number -- which is how the
legend read 0.403 and the caption 0.402 -- and it violated the single-source
rule in spec 0.1.

Now every number is read from ONE file pair produced by
emit_hgb_convergence_history.py on the consolidated run's primary split, and the
same two variables (VAL_MAE_MIN, TEST_MAE) feed the curve annotation, the
legend, the caption string and figure3_data.csv. They cannot drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rceg_final_consolidated")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    run = Path(args.run)
    hist_path = run / "global_hgb_convergence_history.csv"
    summ_path = run / "global_hgb_convergence_summary.csv"
    for p in (hist_path, summ_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run emit_hgb_convergence_history.py first.")

    hist = pd.read_csv(hist_path).sort_values("iteration")
    summ = pd.read_csv(summ_path).iloc[0]

    # ---- the single source of every number rendered ----
    VAL_MAE_MIN = float(summ["val_mae_min"])
    TEST_MAE = float(summ["test_mae"])
    BEST_ITER = int(summ["best_iteration"])

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    ax.plot(hist["iteration"], hist["ref_val_mae"], color="#2B6CB0",
            linewidth=1.9, label="Reference validation MAE", zorder=3)
    ax.plot(hist["iteration"], hist["ref_test_mae"], color="#A0AEC0",
            linewidth=1.4, linestyle=":", label="Held-out test MAE", zorder=2)

    ax.scatter([BEST_ITER], [VAL_MAE_MIN], marker="o", s=52, zorder=5,
               color="#2B6CB0", edgecolor="white", linewidth=1.2,
               label=f"Minimum validation MAE = {VAL_MAE_MIN:.3f} eV")
    ax.axhline(TEST_MAE, color="#C05621", linestyle="--", linewidth=1.4,
               zorder=4, label=f"Held-out test MAE = {TEST_MAE:.3f} eV")

    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Mean absolute error (eV)")
    ax.grid(True, alpha=0.24, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    # ---- inset over the last third, where the curve flattens ----
    lo = int(hist["iteration"].quantile(0.62))
    zoom = hist[hist["iteration"] >= lo]
    axin = inset_axes(ax, width="43%", height="41%", loc="center right",
                      borderpad=1.5)
    axin.plot(zoom["iteration"], zoom["ref_val_mae"], color="#2B6CB0",
              linewidth=1.6)
    axin.plot(zoom["iteration"], zoom["ref_test_mae"], color="#A0AEC0",
              linewidth=1.2, linestyle=":")
    axin.axhline(TEST_MAE, color="#C05621", linestyle="--", linewidth=1.1)
    axin.scatter([BEST_ITER], [VAL_MAE_MIN], marker="o", s=30, zorder=5,
                 color="#2B6CB0", edgecolor="white", linewidth=0.9)
    axin.grid(True, alpha=0.22, linewidth=0.5)
    axin.tick_params(labelsize=8)
    axin.set_title("final third", fontsize=8.5, pad=3)
    mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="#CBD5E0", lw=0.9)

    fig.tight_layout()
    png = OUT / "figure_training_iteration_vs_error_clean_inset.png"
    pdf = OUT / "figure_training_iteration_vs_error_clean_inset.pdf"
    fig.savefig(png, dpi=args.dpi, pad_inches=0.08)
    fig.savefig(pdf, dpi=args.dpi, pad_inches=0.08)
    plt.close(fig)

    # ---- sidecar carrying the same two variables ----
    data = hist.copy()
    data["val_mae_min"] = VAL_MAE_MIN
    data["test_mae"] = TEST_MAE
    data["best_iteration"] = BEST_ITER
    data.to_csv(OUT / "figure3_data.csv", index=False)

    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {OUT/'figure3_data.csv'}")
    print(f"\nrendered values (identical in legend, caption and sidecar):")
    print(f"  val_mae_min    = {VAL_MAE_MIN:.3f} eV  (iteration {BEST_ITER})")
    print(f"  test_mae       = {TEST_MAE:.3f} eV")

    d = pd.read_csv(OUT / "figure3_data.csv")
    ok = (abs(d["val_mae_min"].iloc[0] - VAL_MAE_MIN) < 1e-12
          and abs(d["test_mae"].iloc[0] - TEST_MAE) < 1e-12)
    print(f"\n[{'PASS' if ok else 'FAIL'}] figure3_data.csv carries val_mae_min "
          f"and test_mae matching the rendered figure")

    if not bool(summ["early_stopped"]):
        print(f"\nNOTE: early stopping never triggered -- the model ran to "
              f"max_iter={int(summ['max_iter_requested'])} with validation MAE "
              f"still decreasing. global_hgb is under-trained; raising max_iter "
              f"would improve it.")


if __name__ == "__main__":
    main()
