from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

base_path = ROOT / "enhanced_dataset_lasso.csv"
qm9_path = ROOT / "qm9_gap_reference.csv"
psi4_path = ROOT / "pqr_recomputed_reference.csv"

for p in [base_path, qm9_path, psi4_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

base = pd.read_csv(base_path)
qm9 = pd.read_csv(qm9_path)
psi4 = pd.read_csv(psi4_path)

# ------------------------------------------------------------
# Pick merge column
# ------------------------------------------------------------
join_col = None
for c in ["smiles", "id"]:
    if c in base.columns and c in qm9.columns and c in psi4.columns:
        join_col = c
        break

if join_col is None:
    raise RuntimeError(
        "Could not find a common join column. Expected 'smiles' or 'id' "
        "in enhanced_dataset_lasso.csv, qm9_gap_reference.csv, and "
        "pqr_recomputed_reference.csv."
    )

# ------------------------------------------------------------
# Required descriptor columns
# ------------------------------------------------------------
needed_base = [
    join_col,
    "gap",
    "mol_weight",
    "exact_mass",
    "polarizability",
    "heat_formation",
]

missing = [c for c in needed_base if c not in base.columns]
if missing:
    raise RuntimeError(f"Missing columns in enhanced_dataset_lasso.csv: {missing}")

base_keep = base[needed_base].copy()

# ------------------------------------------------------------
# Build plotted reference subsets
# ------------------------------------------------------------
qm9_ref = (
    qm9[[join_col]]
    .drop_duplicates()
    .merge(base_keep, on=join_col, how="inner")
)

psi4_ref = (
    psi4[[join_col]]
    .drop_duplicates()
    .merge(base_keep, on=join_col, how="inner")
)

qm9_ref["subset"] = "PQR–QM9 overlap"
psi4_ref["subset"] = "Psi4-recomputed PQR"

plot_df = pd.concat([qm9_ref, psi4_ref], ignore_index=True)

for c in ["gap", "mol_weight", "exact_mass", "polarizability", "heat_formation"]:
    plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

plot_df = plot_df.dropna(subset=["gap"])

# ------------------------------------------------------------
# Counts for caption
# ------------------------------------------------------------
n_qm9 = len(qm9_ref)
n_psi4 = len(psi4_ref)
n_total_points = len(plot_df)
n_unique = plot_df[join_col].nunique()
n_overlap_between_reference_sets = n_qm9 + n_psi4 - n_unique

print("\nFigure counts:")
print(f"PQR–QM9 overlap plotted: {n_qm9}")
print(f"Psi4-recomputed PQR plotted: {n_psi4}")
print(f"Total plotted label entries: {n_total_points}")
print(f"Unique molecules: {n_unique}")
print(f"Overlap between reference groups: {n_overlap_between_reference_sets}")

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
specs = [
    ("mol_weight", "Molecular weight", "Da"),
    ("exact_mass", "Exact mass", "Da"),
    ("polarizability", "PM7 polarizability", "Å³"),
    ("heat_formation", "PM7 heat of formation", "kcal/mol"),
]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.4, 6.4),
    layout="constrained",
)

axes = axes.ravel()

for ax, (col, label, unit) in zip(axes, specs):
    dfc = plot_df[[col, "gap", "subset"]].dropna().copy()

    # Trim only extreme display outliers so the plot remains readable.
    # Full data are still exported to CSV.
    xlo = dfc[col].quantile(0.005)
    xhi = dfc[col].quantile(0.995)
    ylo = dfc["gap"].quantile(0.005)
    yhi = dfc["gap"].quantile(0.995)

    visible = dfc[
        (dfc[col] >= xlo)
        & (dfc[col] <= xhi)
        & (dfc["gap"] >= ylo)
        & (dfc["gap"] <= yhi)
    ]

    q = visible[visible["subset"] == "PQR–QM9 overlap"]
    p = visible[visible["subset"] == "Psi4-recomputed PQR"]

    ax.scatter(
        q[col],
        q["gap"],
        s=16,
        alpha=0.68,
        linewidths=0,
        label=f"PQR–QM9 overlap (n={n_qm9:,})",
    )

    ax.scatter(
        p[col],
        p["gap"],
        s=20,
        alpha=0.70,
        marker="^",
        linewidths=0,
        label=f"Psi4-recomputed PQR (n={n_psi4:,})",
    )

    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Raw PQR HOMO–LUMO gap (eV)")
    ax.grid(alpha=0.20)

    # No title or in-panel text.
    # The descriptor name and unit are already in the x-axis label.

# One shared legend at the bottom.
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=2,
    frameon=False,
    fontsize=10,
    markerscale=1.3,
)

# No figure title. The caption should provide the title.

png = OUT / "figure1_reference_descriptor_scatter_final.png"
pdf = OUT / "figure1_reference_descriptor_scatter_final.pdf"
csv = OUT / "figure1_reference_descriptor_scatter_final_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

plot_df.to_csv(csv, index=False)

print("\nSaved:")
print(png)
print(pdf)
print(csv)

print("\nSuggested caption:")
print(
    f"Figure 1. Reference-set descriptor distributions used for calibration and "
    f"evaluation. Raw PQR HOMO–LUMO gaps are plotted against four scalar molecular "
    f"descriptors for the PQR–QM9 overlap molecules and the Psi4-recomputed PQR "
    f"molecules. Blue circles represent PQR–QM9 overlap labels (n={n_qm9:,}), and "
    f"orange triangles represent Psi4-recomputed PQR labels (n={n_psi4:,}). "
    f"Molecular weight and exact mass are reported in daltons, PM7 polarizability "
    f"in Å³, and PM7 heat of formation in kcal/mol."
)

if n_overlap_between_reference_sets > 0:
    print(
        f"\nOptional extra caption sentence: Because "
        f"{n_overlap_between_reference_sets:,} molecules were present in both "
        f"reference groups, the two plotted groups contain "
        f"{n_total_points:,} label entries corresponding to "
        f"{n_unique:,} unique molecules."
    )
