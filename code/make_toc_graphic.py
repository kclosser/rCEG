#!/usr/bin/env python3
"""
make_toc_graphic.py   (spec task B5)

Table-of-contents / graphical abstract.

Addresses R1.5 and R1.10. The previous graphic used inconsistent HOMO-LUMO
orthography, which the manuscript-wide text sweep could not reach because it is
an image. Every occurrence here uses the hyphen-minus form with lower-case
"gap": "HOMO-LUMO gap".

Counts are the audited values: 106,066 PQR records reduce to the 84,143-molecule
strict corpus, itemised in cleaning_report_complete.csv.

Sized for ACS at 3.30 x 1.80 inches, rendered at 600 dpi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("paper_figures_revised")
OUT.mkdir(exist_ok=True)

TITLE = "HOMO-LUMO gap"          # hyphen-minus, lower-case "gap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rceg_final_consolidated")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    # headline number, read rather than typed
    # Read the FROZEN architecture, not whichever config happens to have the
    # lowest MAE. The architecture was selected on validation (see
    # frozen_architecture.csv); picking by test MAE here would silently
    # reintroduce test-set selection into a published figure.
    run = Path(args.run)
    frozen = pd.read_csv(run / "frozen_architecture.csv").iloc[0]
    cfg = str(frozen["frozen_architecture"])
    summ = pd.read_csv(run / "repeated_split_summary.csv")
    row = summ[summ.config == cfg]
    if len(row):
        mae, sd = float(row.iloc[0].mae_mean), float(row.iloc[0].mae_sd)
    else:
        mae, sd = float(frozen["test_mae"]), float(frozen["test_sd"])
    print(f"TOC headline uses the frozen architecture '{cfg}'")

    fig, ax = plt.subplots(figsize=(3.30, 1.80))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.945, f"Scalable {TITLE} prediction", ha="center", va="center",
            fontsize=8.6, weight="bold", color="#1A365D")

    def chip(x, w, y, h, text, fc, ec, fs=5.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012,rounding_size=0.03",
                                    facecolor=fc, edgecolor=ec, linewidth=0.85))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.35,
                weight="bold" if bold else "normal")

    def arr(p, q, c="#4A5568"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                     color=c, linewidth=1.0,
                                     shrinkA=1.2, shrinkB=1.2))

    y, h = 0.60, 0.20
    chip(0.030, 0.205, y, h, "106,066\nPQR records", "#EBF4FF", "#3182CE", 5.4, True)
    arr((0.235, y + h / 2), (0.278, y + h / 2))
    chip(0.278, 0.200, y, h, "84,143\nclean molecules", "#EBF4FF", "#3182CE", 5.4, True)
    arr((0.478, y + h / 2), (0.521, y + h / 2))
    chip(0.521, 0.205, y, h, "chemical\ndescriptors", "#F0FFF4", "#38A169", 5.4)
    arr((0.726, y + h / 2), (0.769, y + h / 2))
    chip(0.769, 0.200, y, h, "domain-split\nexperts", "#FFFFF0", "#B7791F", 5.4)

    # headline result
    ax.add_patch(FancyBboxPatch((0.055, 0.115), 0.40, 0.30,
                                boxstyle="round,pad=0.014,rounding_size=0.035",
                                facecolor="#FFFAF0", edgecolor="#DD6B20",
                                linewidth=1.25))
    ax.text(0.255, 0.325, f"{mae:.3f} ± {sd:.3f} eV", ha="center", va="center",
            fontsize=9.4, weight="bold", color="#7B341E")
    ax.text(0.255, 0.195, "held-out reference MAE\n10 repeated splits",
            ha="center", va="center", fontsize=4.9, color="#7B341E",
            linespacing=1.35)

    # throughput contrast, from the measured timing table
    ax.add_patch(FancyBboxPatch((0.520, 0.115), 0.425, 0.30,
                                boxstyle="round,pad=0.014,rounding_size=0.035",
                                facecolor="#F7FAFC", edgecolor="#4A5568",
                                linewidth=1.25))
    ax.text(0.7325, 0.325, "42 ms  vs  119 s", ha="center", va="center",
            fontsize=8.4, weight="bold", color="#2D3748")
    ax.text(0.7325, 0.195, f"per {TITLE}\nprediction vs B3LYP",
            ha="center", va="center", fontsize=4.9, color="#2D3748",
            linespacing=1.35)

    fig.tight_layout(pad=0.30)
    png = OUT / "toc_graphic.png"
    pdf = OUT / "toc_graphic.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"wrote {png}\nwrote {pdf}")
    print(f"headline: {cfg} = {mae:.3f} +/- {sd:.3f} eV")

    # orthography check across every string drawn
    drawn = [t.get_text() for t in ax.texts]
    bad = [t for t in drawn
           if ("HOMO" in t or "LUMO" in t)
           and ("HOMO-LUMO gap" not in t or "–" in t or "—" in t
                or "Gap" in t or "GAP" in t)]
    print(f"[{'PASS' if not bad else 'FAIL'}] orthography is "
          f"'HOMO-LUMO gap' everywhere" + (f"; offenders {bad}" if bad else ""))


if __name__ == "__main__":
    main()
