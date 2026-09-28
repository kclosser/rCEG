from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

PQR = ROOT / "enhanced_dataset_lasso.csv"

if not PQR.exists():
    raise FileNotFoundError(f"Missing {PQR}")

df = pd.read_csv(PQR)

# Keep only named, interpretable descriptors. Do NOT use HOMO/LUMO.
needed = [
    "smiles",
    "gap",
    "mol_weight",
    "exact_mass",
    "dipole_moment",
    "heat_formation",
    "polarizability",
]

missing = [c for c in needed if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

df = df[needed].copy()

for c in needed:
    if c != "smiles":
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["gap"])

# Try to highlight real reference / recomputed molecules if available.
df["subset"] = "Full PQR dataset"

reference_files = [
    ROOT / "qm9_gap_reference.csv",
    ROOT / "pqr_recomputed_reference.csv",
]

for ref_path in reference_files:
    if ref_path.exists() and "smiles" in pd.read_csv(ref_path, nrows=1).columns:
        ref = pd.read_csv(ref_path, usecols=["smiles"])
        label = (
            "QM9-overlap reference"
            if "qm9" in ref_path.name.lower()
            else "Psi4-recomputed reference"
        )
        df.loc[df["smiles"].isin(set(ref["smiles"])), "subset"] = label

descriptors = [
    ("mol_weight", "Molecular weight", "Da"),
    ("exact_mass", "Exact mass", "Da"),
    ("dipole_moment", "Dipole moment", "Debye"),
    ("heat_formation", "Heat of formation", "kcal/mol"),
    ("polarizability", "Polarizability", "Å³"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
axes = axes.ravel()

for ax, (col, label, unit) in zip(axes, descriptors):
    plot_df = df[[col, "gap", "subset"]].dropna()

    # Display 1st to 99th percentile so one extreme value does not ruin the scale.
    lo = plot_df[col].quantile(0.01)
    hi = plot_df[col].quantile(0.99)
    visible = plot_df[(plot_df[col] >= lo) & (plot_df[col] <= hi)]

    # Full dataset as background.
    full = visible[visible["subset"] == "Full PQR dataset"]
    ax.scatter(
        full[col],
        full["gap"],
        s=5,
        alpha=0.16,
        linewidths=0,
        label="Full PQR dataset",
    )

    # Highlight reference subsets if available.
    for subset, marker, size in [
        ("QM9-overlap reference", "o", 13),
        ("Psi4-recomputed reference", "^", 16),
    ]:
        sub = visible[visible["subset"] == subset]
        if len(sub):
            ax.scatter(
                sub[col],
                sub["gap"],
                s=size,
                alpha=0.55,
                linewidths=0.25,
                edgecolors="black",
                marker=marker,
                label=subset,
            )

    # Linear trend for descriptive visualization only.
    if len(visible) > 10 and visible[col].std() > 0:
        m, b = np.polyfit(visible[col], visible["gap"], 1)
        xs = np.linspace(visible[col].min(), visible[col].max(), 200)
        ax.plot(xs, m * xs + b, linewidth=1.3)

    r = visible[col].corr(visible["gap"])

    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Raw PQR HOMO–LUMO gap (eV)")
    ax.set_title(f"{label} vs gap, r = {r:.2f}")
    ax.grid(alpha=0.22)

    ax.text(
        0.02,
        0.02,
        "x-axis shows 1st–99th percentile",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
    )

# Remove unused sixth panel.
axes[-1].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Interpretable Molecular Descriptors versus HOMO–LUMO Gap",
    y=0.98,
    fontsize=15,
)

fig.tight_layout(rect=[0, 0.07, 1, 0.95])

out_png = OUT / "figure_named_descriptors_vs_gap.png"
out_pdf = OUT / "figure_named_descriptors_vs_gap.pdf"
out_csv = OUT / "figure_named_descriptors_vs_gap_data.csv"

fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

df.to_csv(out_csv, index=False)

print("Saved:")
print(out_png)
print(out_pdf)
print(out_csv)
