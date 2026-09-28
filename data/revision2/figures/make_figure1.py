#!/usr/bin/env python3
"""
make_figure1.py   (revision-2 Figure 1)

Three-panel scatter of the raw PQR HOMO-LUMO gap against three scalar
descriptors, for the reference-labelled molecules only.

Why this exists as a separate script
------------------------------------
The Figure 1 embedded in the manuscript is the JULY build
(`figure1_reference_descriptor_scatter_clean.png`): 743 points, a legend
reading 641 and 102, a band of 16 records at exactly 0.0 eV, and a
"Displayed complete-case subset" block added to explain the shortfall. It was
plotted from the pre-cleaning descriptor table and joined on raw SMILES.

That was already fixed in September:
`figure1_reference_descriptor_scatter_final.png` plots 2,137 points, reads
1,228 and 909, and has no zero-gap records. **The data was never the problem
by the time this task was raised.**

The one thing that version still gets wrong is the axis labels, which read
"PM7 polarizability" and "PM7 heat of formation". Whether the source data is
PM6 or PM7 is unresolved: the PQR project FAQ states PM6 as implemented in
MOPAC, while the distributed field is named `pm7`. Naming the method on the
axis forces a guess. This script drops the prefix so the labels are correct
either way, and lets the Methods name the method once.

Input
-----
`figure1_data.csv` from the September build: 2,137 rows, already the strict
corpus joined to the reference pool on RDKit-canonical SMILES, carrying the
gap and all three descriptors, with zero nulls and zero zero-gap records.
Those invariants are re-asserted here rather than assumed.

The 2,137 points are 1,228 + 909; the 14 molecules labelled by both sources
appear once in each series, which is intended and the caption says so.

    PYTHONHASHSEED=0 python3 runs/revision2/figures/make_figure1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("runs/revision2/figures")
DEFAULT_SRC = Path("paper_figures_revised/figure1_data.csv")

# Method prefix deliberately omitted -- see the module docstring.
PANELS = [
    ("mol_weight", "Molecular weight", "Da"),
    ("polarizability", "Polarizability", "Å$^3$"),
    ("heat_formation", "Heat of formation", "kcal mol$^{-1}$"),
]

SERIES = [
    ("PQR–QM9 overlap", "#4C78A8", "o", 1228),
    ("Psi4-recomputed PQR", "#F58518", "^", 909),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--clip-percentile", type=float, default=None,
                    metavar="P",
                    help="clip each x axis to the [P, 100-P] percentile, as "
                         "the September build did with P=0.5. Off by default: "
                         "clipping pushes a few points outside the frame "
                         "where they are silently not drawn, and this figure "
                         "has a history of count mismatches. Use 0.5 to "
                         "reproduce the earlier framing.")
    ap.add_argument("--suffix", default="",
                    help="appended to output filenames, for variants")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"MISSING INPUT: {src}")
    df = pd.read_csv(src)
    OUT.mkdir(parents=True, exist_ok=True)

    cols = ["smiles", "series", "gap"] + [c for c, _, _ in PANELS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"{src} lacks required columns: {missing}")
    df = df[cols].copy()

    # ---- invariants. Report, never repair. ----------------------------
    problems = []
    nulls = df[[c for c, _, _ in PANELS] + ["gap"]].isna().sum()
    if int(nulls.sum()):
        problems.append(f"nulls present: {nulls[nulls > 0].to_dict()}")
    n_zero = int((df["gap"].abs() < 0.01).sum())
    if n_zero:
        problems.append(f"{n_zero} records at |gap| < 0.01 eV")
    counts = df["series"].value_counts().to_dict()
    for label, _, _, want in SERIES:
        got = counts.get(label, 0)
        if got != want:
            problems.append(f"series '{label}': {got} rows, expected {want}")
    if len(df) != 2137:
        problems.append(f"{len(df)} rows, expected 2,137")

    print(f"source              {src}")
    print(f"rows                {len(df):,}")
    for label, _, _, want in SERIES:
        print(f"  {label:22s} {counts.get(label, 0):,} (expected {want:,})")
    print(f"nulls in plotted columns  {int(nulls.sum())}")
    print(f"records at |gap| < 0.01   {n_zero}")
    print(f"gap range           {df['gap'].min():.3f} – {df['gap'].max():.3f} eV")

    if problems:
        print("\nSTOPPING. The input does not satisfy the stated invariants:")
        for p in problems:
            print(f"  - {p}")
        sys.exit("Record this in DISCREPANCIES.md rather than adjusting a "
                 "filter to make the counts come out.")

    # ---- plot ---------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)

    for ax, (col, name, unit) in zip(axes, PANELS):
        for label, colour, marker, _ in SERIES:
            sub = df[df["series"] == label]
            ax.scatter(sub[col], sub["gap"], s=16, alpha=0.55, c=colour,
                       marker=marker, linewidths=0, label=label, zorder=3)
        if args.clip_percentile is not None:
            P = args.clip_percentile
            lo, hi = df[col].quantile([P / 100.0, 1 - P / 100.0])
            pad = 0.02 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            hidden = int(((df[col] < lo - pad) | (df[col] > hi + pad)).sum())
            if hidden:
                print(f"  NOTE {col}: {hidden} point(s) fall outside the "
                      f"clipped axis and are not drawn")
        ax.set_xlabel(f"{name} ({unit})")
        ax.grid(True, alpha=0.25, zorder=0)
    axes[0].set_ylabel("HOMO–LUMO gap (eV)")

    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="none",
                          markersize=8, alpha=0.75,
                          label=f"{l} (n={n:,})")
               for l, c, m, n in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()

    stem = f"figure1_reference_descriptor_scatter{args.suffix}"
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf, dpi=args.dpi)
    plt.close(fig)

    df.to_csv(OUT / "figure1_reference_descriptor_scatter_data.csv",
              index=False)

    print(f"\nwrote {png} ({args.dpi} dpi)")
    print(f"wrote {pdf}")
    print(f"wrote {OUT/'figure1_reference_descriptor_scatter_data.csv'} "
          f"({len(df):,} rows)")


if __name__ == "__main__":
    main()
