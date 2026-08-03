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

print("Base columns:", list(base.columns))
print("QM9 columns:", list(qm9.columns))
print("Psi4 columns:", list(psi4.columns))

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

# Use SMILES if available; otherwise try id.
join_col = None
for c in ["smiles", "id"]:
    if c in base.columns and c in qm9.columns and c in psi4.columns:
        join_col = c
        break

if join_col is None:
    raise RuntimeError("Could not find common join column: expected smiles or id.")

# Need raw PQR gap and named descriptors from base.
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

qm9_ref = qm9[[join_col]].drop_duplicates().merge(base_keep, on=join_col, how="inner")
psi4_ref = psi4[[join_col]].drop_duplicates().merge(base_keep, on=join_col, how="inner")

qm9_ref["subset"] = "PQR–QM9 overlap"
psi4_ref["subset"] = "Psi4-recomputed PQR"

plot_df = pd.concat([qm9_ref, psi4_ref], ignore_index=True)

for c in ["gap", "mol_weight", "exact_mass", "polarizability", "heat_formation"]:
    plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

plot_df = plot_df.dropna(subset=["gap"])

# Print counts so you can verify caption.
n_qm9 = len(qm9_ref)
n_psi4 = len(psi4_ref)
n_unique = plot_df[join_col].nunique()
n_total_points = len(plot_df)

print()
print("Figure counts:")
print("PQR-QM9 overlap plotted:", n_qm9)
print("Psi4-recomputed PQR plotted:", n_psi4)
print("Total plotted points:", n_total_points)
print("Unique molecules:", n_unique)

specs = [
    ("mol_weight", "Molecular weight", "Da"),
    ("exact_mass", "Exact mass", "Da"),
    ("polarizability", "PM7 polarizability", "Å³"),
    ("heat_formation", "PM7 heat of formation", "kcal/mol"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
axes = axes.ravel()

for ax, (col, label, unit) in zip(axes, specs):
    dfc = plot_df[[col, "gap", "subset"]].dropna().copy()

    # Keep central 99% for readability but do not say this in every panel.
    xlo = dfc[col].quantile(0.005)
    xhi = dfc[col].quantile(0.995)
    ylo = dfc["gap"].quantile(0.005)
    yhi = dfc["gap"].quantile(0.995)
    visible = dfc[(dfc[col] >= xlo) & (dfc[col] <= xhi) & (dfc["gap"] >= ylo) & (dfc["gap"] <= yhi)]

    q = visible[visible["subset"] == "PQR–QM9 overlap"]
    p = visible[visible["subset"] == "Psi4-recomputed PQR"]

    ax.scatter(
        q[col],
        q["gap"],
        s=13,
        alpha=0.65,
        linewidths=0,
        label=f"PQR–QM9 overlap (n={n_qm9:,})",
    )

    ax.scatter(
        p[col],
        p["gap"],
        s=15,
        alpha=0.65,
        marker="^",
        linewidths=0,
        label=f"Psi4-recomputed PQR (n={n_psi4:,})",
    )

    ax.set_xlabel(f"{label} ({unit})", fontsize=10)
    ax.set_ylabel("Raw PQR HOMO–LUMO gap (eV)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.20)

    # Move panel title to a small simple descriptor label, not a separate figure title.
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )

# One larger legend for the whole figure.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=10,
    markerscale=1.3,
)

# No overall title. Caption handles the figure title.
fig.tight_layout(rect=[0, 0.08, 1, 1])

png = OUT / "figure1_reference_descriptor_scatter_clean.png"
pdf = OUT / "figure1_reference_descriptor_scatter_clean.pdf"
csv = OUT / "figure1_reference_descriptor_scatter_clean_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

plot_df.to_csv(csv, index=False)

print()
print("Saved:")
print(png)
print(pdf)
print(csv)
