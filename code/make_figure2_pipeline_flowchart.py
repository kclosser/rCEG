#!/usr/bin/env python3
"""
make_figure2_pipeline_flowchart.py   (spec task B2)

Figure 2: the rCEG pipeline, redrawn to R3.7's structural requirements.

What R3.7 asked for, and how it is satisfied here
--------------------------------------------------
* Two visually distinct outer containers, "Stage 1 - Calibration model" and
  "Stage 2 - rCEG prediction model", with an explicit statement that they are
  separate models never applied jointly at prediction time.
* The full data flow with counts at every reduction.
* Explicit marking of which molecules are excluded from which stage.
* A separate inference path for a new molecule, entering Stage 2 only.
* The leakage-audit box retained, since the manuscript text now defines it.
* No residual dual-pathway or neural-network iconography.

Every count is the MEASURED value from runs/rceg_final_consolidated, not an
approximation:

    106,066  PQR release
     21,923  removed by cleaning (itemised in cleaning_report_complete.csv)
     84,143  strict corpus
      9,753  QMugs external holdout, excised before any model development
     74,390  development corpus
      2,123  reference-labelled  ->  1,274 calibration / 425 val / 424 test
     73,541  training pool = 1,274 real anchors + 72,267 pseudo-labelled
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("paper_figures_revised")
OUT.mkdir(exist_ok=True)

C_DATA = "#EBF4FF"; E_DATA = "#3182CE"
C_PROC = "#F0FFF4"; E_PROC = "#38A169"
C_EXCL = "#FFF5F5"; E_EXCL = "#E53E3E"
C_AUDIT = "#FFFAF0"; E_AUDIT = "#DD6B20"
C_S1 = "#FAF5FF"; E_S1 = "#805AD5"
C_S2 = "#FFFFF0"; E_S2 = "#B7791F"
C_INF = "#F7FAFC"; E_INF = "#4A5568"


def box(ax, x, y, w, h, text, fc, ec, fs=8.6, bold=False, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.014",
                                facecolor=fc, edgecolor=ec, linewidth=1.3,
                                zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=z + 1, weight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, p, q, ec="#2D3748", style="-|>", lw=1.35, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=ec, linewidth=lw, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5, zorder=z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13.2, 10.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- data preparation, shared ----------------
    box(ax, 0.035, 0.905, 0.30, 0.062,
        "Pitt Quantum Repository\n106,066 records", C_DATA, E_DATA, 9.2, True)
    arrow(ax, (0.185, 0.905), (0.185, 0.868))

    box(ax, 0.035, 0.806, 0.30, 0.062,
        "Cleaning and strict de-duplication\n21,923 records removed",
        C_PROC, E_PROC)
    box(ax, 0.355, 0.806, 0.245, 0.062,
        "Excluded\n10,162 unparseable SMILES\n10,749 other rules",
        C_EXCL, E_EXCL, 7.8)
    arrow(ax, (0.335, 0.837), (0.355, 0.837), E_EXCL, ls=(0, (3, 2)))
    arrow(ax, (0.185, 0.806), (0.185, 0.769))

    box(ax, 0.035, 0.707, 0.30, 0.062,
        "Strict corpus\n84,143 molecules", C_DATA, E_DATA, 9.2, True)
    arrow(ax, (0.185, 0.707), (0.185, 0.670))

    box(ax, 0.035, 0.608, 0.30, 0.062,
        "QMugs external holdout excised\nbefore any model development",
        C_PROC, E_PROC)
    box(ax, 0.355, 0.608, 0.245, 0.062,
        "Held out\n9,753 QMugs molecules\nnever seen in training",
        C_EXCL, E_EXCL, 7.8)
    arrow(ax, (0.335, 0.639), (0.355, 0.639), E_EXCL, ls=(0, (3, 2)))
    arrow(ax, (0.185, 0.608), (0.185, 0.571))

    box(ax, 0.035, 0.509, 0.30, 0.062,
        "Development corpus\n74,390 molecules", C_DATA, E_DATA, 9.2, True)

    # leakage audit hangs off the development corpus, which is what it audits
    box(ax, 0.400, 0.509, 0.29, 0.062,
        "Leakage audit\nHOMO and LUMO structurally excluded\n"
        "max |r| over 427 features = 0.53",
        C_AUDIT, E_AUDIT, 7.8)
    arrow(ax, (0.335, 0.540), (0.400, 0.540), E_AUDIT, ls=(0, (3, 2)))

    # ---------------- Stage 1 container ----------------
    ax.add_patch(FancyBboxPatch((0.028, 0.266), 0.455, 0.202,
                                boxstyle="round,pad=0.012,rounding_size=0.016",
                                facecolor=C_S1, edgecolor=E_S1, linewidth=2.1,
                                linestyle="-", zorder=1))
    ax.text(0.2555, 0.458, "STAGE 1 — Calibration model", ha="center",
            va="center", fontsize=10.4, weight="bold", color=E_S1, zorder=2)

    arrow(ax, (0.185, 0.509), (0.185, 0.452))
    box(ax, 0.045, 0.376, 0.20, 0.058,
        "Reference-labelled\n2,123 molecules", C_DATA, E_DATA, 8.4, True)
    box(ax, 0.265, 0.376, 0.205, 0.058,
        "Split  1,274 calibration\n425 validation / 424 test", C_PROC, E_PROC, 8.0)
    arrow(ax, (0.245, 0.405), (0.265, 0.405))

    box(ax, 0.045, 0.286, 0.425, 0.062,
        "Calibrate  PQR PM7 gap + descriptors  →  DFT reference gap\n"
        "fitted on the 1,274 calibration anchors only",
        "#FFFFFF", E_S1, 8.4)
    arrow(ax, (0.3675, 0.376), (0.3675, 0.348))

    # ---------------- Stage 2 container ----------------
    ax.add_patch(FancyBboxPatch((0.520, 0.266), 0.452, 0.202,
                                boxstyle="round,pad=0.012,rounding_size=0.016",
                                facecolor=C_S2, edgecolor=E_S2, linewidth=2.1,
                                zorder=1))
    ax.text(0.746, 0.458, "STAGE 2 — rCEG prediction model", ha="center",
            va="center", fontsize=10.4, weight="bold", color=E_S2, zorder=2)

    box(ax, 0.535, 0.376, 0.42, 0.058,
        "Training pool  73,541 molecules\n"
        "1,274 real reference anchors + 72,267 calibrated pseudo-labels",
        C_DATA, E_DATA, 8.0)
    box(ax, 0.535, 0.286, 0.42, 0.062,
        "Domain experts and piecewise descriptor splits\n"
        "inputs are molecular descriptors only — no gap of any kind",
        "#FFFFFF", E_S2, 8.4)
    arrow(ax, (0.745, 0.376), (0.745, 0.348))

    # calibration hands pseudo-labels to stage 2
    arrow(ax, (0.470, 0.317), (0.535, 0.405), E_S1, lw=1.7)
    ax.text(0.5025, 0.345, "pseudo-\nlabels", ha="center", va="center",
            fontsize=7.6, color=E_S1, style="italic", zorder=6)

    # the separation statement, clear of both containers
    ax.text(0.50, 0.228,
            "The two models are never applied jointly. Stage 1 exists only to "
            "produce training labels;\nStage 2 alone is used for prediction and "
            "never receives a PQR gap at inference.",
            ha="center", va="center", fontsize=8.8, style="italic",
            color="#4A5568", zorder=6)

    # ---------------- evaluation ----------------
    box(ax, 0.520, 0.140, 0.20, 0.058,
        "Evaluation\n424 held-out\nreference molecules", C_DATA, E_DATA, 8.0, True)
    box(ax, 0.762, 0.140, 0.20, 0.058,
        "External test\n9,753 QMugs\nmolecules", C_DATA, E_DATA, 8.0, True)
    arrow(ax, (0.620, 0.286), (0.620, 0.198))
    arrow(ax, (0.862, 0.286), (0.862, 0.198))
    arrow(ax, (0.478, 0.390), (0.478, 0.169), E_DATA, ls=(0, (2, 2)), lw=1.1)
    arrow(ax, (0.478, 0.169), (0.520, 0.169), E_DATA, ls=(0, (2, 2)), lw=1.1)

    # ---------------- inference path ----------------
    ax.add_patch(FancyBboxPatch((0.028, 0.018), 0.944, 0.098,
                                boxstyle="round,pad=0.010,rounding_size=0.014",
                                facecolor=C_INF, edgecolor=E_INF, linewidth=1.5,
                                linestyle=(0, (5, 3)), zorder=1))
    ax.text(0.050, 0.100, "INFERENCE, new molecule", ha="left", va="center",
            fontsize=9.2, weight="bold", color=E_INF, zorder=6)

    # The semiempirical box is drawn OPTIONAL. The structure-only ablation
    # (B6) showed the five PQR scalars can be dropped for +0.0001 eV
    # (paired p = 0.814), so inference needs only a SMILES string. Drawing
    # this step as required conceded a reviewer objection the paper answers.
    y = 0.034
    steps = [(0.055, 0.140, "SMILES"),
             (0.225, 0.180, "PM7 semiempirical\nproperties  (optional)"),
             (0.435, 0.175, "RDKit descriptors"),
             (0.640, 0.190, "Stage 2 experts\n(Stage 1 not used)"),
             (0.860, 0.108, "Predicted gap")]
    for i, (x, w, t) in enumerate(steps):
        optional = i == 1
        box(ax, x, y, w, 0.040, t, "#FFFFFF", E_INF, 7.8, z=4)
        if optional:
            # dashed overlay marks the box as skippable
            ax.add_patch(FancyBboxPatch(
                (x, y), w, 0.040,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                facecolor="none", edgecolor=E_INF, linewidth=1.9,
                linestyle=(0, (3, 2)), zorder=6))
    for i in range(len(steps) - 1):
        x0 = steps[i][0] + steps[i][1]
        arrow(ax, (x0, y + 0.020), (steps[i + 1][0], y + 0.020), E_INF, lw=1.15)

    # bypass: SMILES straight to descriptors, the structure-only route
    sm_r = steps[0][0] + steps[0][1]
    dsc_l = steps[2][0]
    ax.add_patch(FancyArrowPatch(
        (sm_r, y + 0.038), (dsc_l, y + 0.038),
        arrowstyle="-|>", mutation_scale=12, color=E_INF, linewidth=1.15,
        linestyle=(0, (3, 2)), shrinkA=1.5, shrinkB=1.5, zorder=5,
        connectionstyle="arc3,rad=-0.38"))
    ax.text((sm_r + dsc_l) / 2, y + 0.083,
            "structure-only route: no quantum calculation",
            ha="center", va="center", fontsize=7.0, style="italic",
            color=E_INF, zorder=6)

    arrow(ax, (0.735, 0.286), (0.735, 0.074), E_S2, ls=(0, (4, 3)), lw=1.4)

    fig.tight_layout()
    png = OUT / "figure2_pipeline_flowchart.png"
    pdf = OUT / "figure2_pipeline_flowchart.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
