#!/usr/bin/env python3
"""
make_figure6_domain_error_distributions.py   (spec task B4)

New Figure 6, requested by R3.8 and R5.6: the distribution of SIGNED prediction
error per chemical domain on the held-out reference test set, rather than a
single aggregate MAE.

The point of the figure is that the aggregate number hides both the spread and
the bias. Reading it, a reviewer can see immediately that the error is not
uniform across chemistry and that one domain carries a systematic offset.

Emits figure6_data.csv with the raw per-molecule residuals so the distribution
is reproducible, per the spec's requirement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 600
PRETTY = {
    "qm9_like_small_organic": "QM9-like\nsmall organic",
    "near_qm9_larger_organic": "Near-QM9\nlarger organic",
    "large_neutral_organic": "Large neutral\norganic",
    "heteroatom_rich_non_qm9": "Heteroatom-rich\nnon-QM9",
    "charged_or_radical": "Charged or\nradical",
}
ORDER = ["qm9_like_small_organic", "near_qm9_larger_organic",
         "large_neutral_organic", "heteroatom_rich_non_qm9",
         "charged_or_radical"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rceg_final_consolidated")
    ap.add_argument("--outdir", default="paper_figures_revised")
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    src = run / "figure6_data.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run the consolidated model first "
            "(it emits figure6_data.csv from the primary repeat).")

    df = pd.read_csv(src)
    if "signed_err" not in df.columns:
        df["signed_err"] = df["pred"] - df["ref_gap"]
    df["abs_err"] = df["signed_err"].abs()

    doms = [d for d in ORDER if d in set(df["domain"])]
    data = [df.loc[df.domain == d, "signed_err"].to_numpy() for d in doms]
    stats = pd.DataFrame([{
        "domain": d,
        "n": len(v),
        "mae": float(np.mean(np.abs(v))),
        "median_abs_error": float(np.median(np.abs(v))),
        "mean_signed_error": float(np.mean(v)),
        "sd_signed_error": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
    } for d, v in zip(doms, data)])

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    parts = ax.violinplot(data, positions=np.arange(len(doms)),
                          showextrema=False, showmedians=False, widths=0.82)
    for pc in parts["bodies"]:
        pc.set_facecolor("#7FA8D4")
        pc.set_edgecolor("#2B6CB0")
        pc.set_alpha(0.55)
        pc.set_linewidth(0.9)

    bp = ax.boxplot(data, positions=np.arange(len(doms)), widths=0.18,
                    patch_artist=True, showfliers=False, zorder=3,
                    medianprops=dict(color="#1A365D", linewidth=1.6),
                    boxprops=dict(facecolor="white", edgecolor="#1A365D",
                                  linewidth=1.0),
                    whiskerprops=dict(color="#1A365D", linewidth=1.0),
                    capprops=dict(color="#1A365D", linewidth=1.0))

    # zero line: the reference for unbiased prediction
    ax.axhline(0.0, color="#C05621", linestyle="--", linewidth=1.3,
               zorder=2, label="zero error")

    # per-domain MAE overlaid
    ax.scatter(np.arange(len(doms)), stats["mae"], marker="D", s=46,
               color="#C05621", edgecolor="white", linewidth=0.8, zorder=5,
               label="domain MAE")

    # A handful of residuals reach -7.2 eV. Showing the full range would squash
    # the informative bulk into a thin band, so the axis is limited and the
    # number of points outside it is stated on the figure rather than dropped.
    all_err = np.concatenate(data)
    ymax = float(np.ceil(max(np.percentile(np.abs(v), 97.5) for v in data) * 1.6))
    n_outside = int((np.abs(all_err) > ymax).sum())

    for i, row in stats.iterrows():
        ax.text(i, ymax * 0.90, f"n = {row.n:,}", ha="center", va="center",
                fontsize=9.5, color="#2D3748")
        ax.text(i, stats["mae"].iloc[i], f"  {row.mae:.3f}", ha="left",
                va="center", fontsize=9, color="#7B341E")

    if n_outside:
        ax.text(0.012, 0.025,
                f"{n_outside} of {len(all_err)} residuals fall outside the "
                f"axis range (min {all_err.min():.2f}, max {all_err.max():.2f} eV); "
                f"all are retained in figure6_data.csv",
                transform=ax.transAxes, fontsize=8.4, color="#4A5568",
                style="italic")

    ax.set_xticks(np.arange(len(doms)))
    ax.set_xticklabels([PRETTY.get(d, d) for d in doms])
    ax.set_ylabel("Signed prediction error (eV)\npredicted − reference")
    ax.set_ylim(-ymax, ymax)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    fig.tight_layout()
    png = out / "figure6_domain_error_distributions.png"
    pdf = out / "figure6_domain_error_distributions.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf, dpi=args.dpi)
    plt.close(fig)

    # sidecars: raw residuals plus the summary that annotates the figure
    keep = [c for c in ["smiles", "domain", "ref_source", "ref_gap", "pred",
                        "signed_err", "abs_err"] if c in df.columns]
    df[keep].to_csv(out / "figure6_data.csv", index=False)
    stats.to_csv(out / "figure6_data_summary.csv", index=False)

    print(stats.round(4).to_string(index=False))
    print(f"\nwrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {out/'figure6_data.csv'} ({len(df):,} per-molecule residuals)")

    tot = int(stats["n"].sum())
    print(f"\n[{'PASS' if tot == len(df) else 'FAIL'}] annotated n sums to the "
          f"plotted rows: {tot} vs {len(df)}")
    worst = stats.loc[stats["mean_signed_error"].abs().idxmax()]
    print(f"largest systematic bias: {worst.domain} "
          f"({worst.mean_signed_error:+.3f} eV)")


if __name__ == "__main__":
    main()
