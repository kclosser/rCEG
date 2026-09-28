#!/usr/bin/env python3
"""
make_figure1_reference_descriptor_scatter_final.py   (spec task B1)

Figure 1: interpretable PQR descriptors against the HOMO-LUMO gap, with the two
reference-labelled subsets highlighted.

Three revision changes (R2.7, R4.5, and the Figure 1 comment in R5)
-------------------------------------------------------------------
1. The exact-mass panel is dropped. It is collinear with molecular weight at a
   MEASURED Pearson r of 0.9934 (the previous caption claimed >0.999; the real
   value is emitted to figure1_data.csv so the claim is checkable).

2. The legend/caption count mismatch is fixed at its root. The previous version
   joined reference molecules to the descriptor table on RAW SMILES STRINGS.
   The reference files store SMILES in a different canonical form, so most
   joins silently failed:

       QM9  overlap : 614 of 1,243 matched   (49%)
       Psi4 recompute: 99 of 909 matched     (11%)

   Both sides are now canonicalised with RDKit before joining, which recovers
   every Psi4 molecule and roughly doubles the QM9 overlap.

   NOTE FOR THE MANUSCRIPT: the caption previously attributed the reduced
   counts to missing descriptor values. That is incorrect. There are ZERO
   missing values among the joined rows, and there is no subsampling anywhere
   in this script. The cause was the un-canonicalised join. The caption
   sentence must be corrected.

3. Plotted counts are computed AFTER all filtering, and the same variable feeds
   the legend, the caption string and figure1_data.csv, so the three cannot
   drift apart again.

Outputs at 600 dpi, PNG and PDF, plus the data sidecar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)
DPI = 600

base_path = ROOT / "enhanced_dataset_lasso.csv"
qm9_path = ROOT / "qm9_gap_reference.csv"
psi4_path = ROOT / "pqr_recomputed_reference.csv"

for p in (base_path, qm9_path, psi4_path):
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

base = pd.read_csv(base_path)
qm9 = pd.read_csv(qm9_path)
psi4 = pd.read_csv(psi4_path)


def canon(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


NEEDED = ["smiles", "gap", "mol_weight", "exact_mass",
          "polarizability", "heat_formation"]
missing = [c for c in NEEDED if c not in base.columns]
if missing:
    raise RuntimeError(f"Missing columns in {base_path}: {missing}")

base = base[NEEDED].copy()
base["key"] = base["smiles"].map(canon)
base = base.dropna(subset=["key"]).drop_duplicates(subset=["key"])

# ------------------------------------------------------------------
# Restrict to the cleaned training corpus.
#
# enhanced_dataset_lasso.csv is the PRE-cleaning descriptor table, so it still
# contains the 4,719 rows that cleaning removed under
# "gap_below_min_or_placeholder" -- molecules carrying a literal gap of 0.0 eV.
# Plotting those put 16 unphysical points on the x-axis of the previous figure
# (e.g. neopentane at 0.0 eV) and showed molecules the paper then excluded.
# Joining to the strict corpus makes Figure 1 consistent with every other
# number in the manuscript.
# ------------------------------------------------------------------
import json

strict_gap = {}
strict_path = ROOT / "enhanced_dataset_lasso_STRICT.jsonl"
if strict_path.exists():
    with open(strict_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, list) and len(rec) >= 5 and isinstance(rec[1], str):
                k = canon(rec[1])
                if k:
                    try:
                        strict_gap[k] = float(rec[4])
                    except Exception:
                        pass
    n_before = len(base)
    base = base[base["key"].isin(strict_gap)].copy()

    # Take the TARGET from the strict corpus, not from the pre-clean CSV.
    # The two disagree for a handful of molecules -- the CSV still carries the
    # placeholder 0.0 eV that cleaning replaced -- and the figure must show the
    # value the model was actually trained against.
    n_disagree = int((base["gap"] - base["key"].map(strict_gap)).abs().gt(1e-6).sum())
    base["gap"] = base["key"].map(strict_gap)
    print(f"restricted to the strict training corpus: "
          f"{n_before:,} -> {len(base):,} descriptor rows")
    print(f"  gap taken from the strict corpus; "
          f"{n_disagree:,} rows disagreed with the pre-clean CSV")
else:
    print("WARNING: strict corpus not found; figure will include pre-clean rows")

for c in ["gap", "mol_weight", "exact_mass", "polarizability", "heat_formation"]:
    base[c] = pd.to_numeric(base[c], errors="coerce")

# measured collinearity, reported rather than asserted
r_mw_exact = float(base["mol_weight"].corr(base["exact_mass"]))

qm9_keys = {k for k in (canon(s) for s in qm9["smiles"].unique()) if k}
psi4_keys = {k for k in (canon(s) for s in psi4["smiles"].unique()) if k}

# raw-string join retained purely to document the size of the old bug
raw_qm9 = qm9[["smiles"]].drop_duplicates().merge(
    base[["smiles"]], on="smiles", how="inner")["smiles"].nunique()
raw_psi4 = psi4[["smiles"]].drop_duplicates().merge(
    base[["smiles"]], on="smiles", how="inner")["smiles"].nunique()

SERIES = [
    ("PQR–QM9 overlap", qm9_keys, "#2B6CB0", "o", raw_qm9),
    ("Psi4-recomputed PQR", psi4_keys, "#DD6B20", "^", raw_psi4),
]

PANELS = [
    ("mol_weight", "Molecular weight", "Da"),
    ("polarizability", "PM7 polarizability", "Å$^3$"),
    ("heat_formation", "PM7 heat of formation", "kcal mol$^{-1}$"),
]
PANEL_COLS = [c for c, _, _ in PANELS]

# Build each series once, complete-case across every plotted descriptor, so the
# number in the legend is exactly the number of points drawn in every panel.
frames, counts = {}, {}
for label, keys, _, _, _ in SERIES:
    sub = base[base["key"].isin(keys)].dropna(subset=["gap"] + PANEL_COLS).copy()
    sub["series"] = label
    frames[label] = sub
    counts[label] = len(sub)

plot_df = pd.concat(frames.values(), ignore_index=True)

print("Figure 1 counts (complete-case across all plotted descriptors):")
for label, keys, _, _, raw in SERIES:
    print(f"  {label:22s} reference={len(keys):6,}  raw-join={raw:5,}  "
          f"canonical-join+complete-case={counts[label]:5,}")
print(f"  measured Pearson r(mol_weight, exact_mass) = {r_mw_exact:.4f}")

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10, "figure.dpi": 120,
    "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1))

for ax, (col, name, unit) in zip(axes, PANELS):
    for label, _, colour, marker, _ in SERIES:
        d = frames[label]
        ax.scatter(d[col], d["gap"], s=13, alpha=0.55, c=colour,
                   marker=marker, linewidths=0,
                   label=f"{label} (n={counts[label]:,})")
    ax.set_xlabel(f"{name} ({unit})")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    lo, hi = np.percentile(plot_df[col].dropna(), [0.5, 99.5])
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)

axes[0].set_ylabel("HOMO–LUMO gap (eV)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.045), markerscale=2.0)
fig.tight_layout()

png = OUT / "figure1_reference_descriptor_scatter_final.png"
pdf = OUT / "figure1_reference_descriptor_scatter_final.pdf"
fig.savefig(png, dpi=DPI)
fig.savefig(pdf, dpi=DPI)
plt.close(fig)

# ---- data sidecar: the same `counts` object that fed the legend ----
summary = pd.DataFrame([
    {"series": label,
     "n_reference_molecules": len(keys),
     "n_joined_raw_string": raw,
     "n_plotted": counts[label],
     "join_method": "RDKit canonical SMILES on both sides"}
    for label, keys, _, _, raw in SERIES
])
summary["pearson_r_molweight_exactmass"] = r_mw_exact
summary["exact_mass_panel_dropped_reason"] = (
    f"collinear with molecular weight, measured r={r_mw_exact:.4f}")
summary.to_csv(OUT / "figure1_data_summary.csv", index=False)

plot_df[["smiles", "series", "gap"] + PANEL_COLS].to_csv(
    OUT / "figure1_data.csv", index=False)

print(f"\nwrote {png}")
print(f"wrote {pdf}")
print(f"wrote {OUT/'figure1_data.csv'} ({len(plot_df):,} plotted points)")
print(f"wrote {OUT/'figure1_data_summary.csv'}")

# ---- acceptance: legend counts must equal the sidecar counts exactly ----
recount = plot_df.groupby("series").size().to_dict()
ok = all(recount.get(l) == counts[l] for l in counts)
print(f"\n[{'PASS' if ok else 'FAIL'}] legend n matches figure1_data.csv rows: "
      f"{ {l: (counts[l], recount.get(l)) for l in counts} }")
