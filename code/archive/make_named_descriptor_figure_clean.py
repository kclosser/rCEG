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
df["subset"] = "Full PQR dataset"

# Highlight reference subsets if files exist.
reference_specs = [
    (ROOT / "qm9_gap_reference.csv", "QM9-overlap reference"),
    (ROOT / "pqr_recomputed_reference.csv", "Psi4-recomputed reference"),
]

for ref_path, label in reference_specs:
    if ref_path.exists():
        ref_head = pd.read_csv(ref_path, nrows=1)
        if "smiles" in ref_head.columns:
            ref = pd.read_csv(ref_path, usecols=["smiles"])
            df.loc[df["smiles"].isin(set(ref["smiles"])), "subset"] = label

descriptors = [
    ("mol_weight", "Molecular weight", "Da"),
    ("exact_mass", "Exact mass", "Da"),
    ("dipole_moment", "Dipole moment", "Debye"),
    ("heat_formation", "Heat of formation", "kcal/mol"),
    ("polarizability", "Polarizability", "Å³"),
]

# -----------------------------
# Figure 1: cleaner full dataset
# -----------------------------

fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
axes = axes.ravel()

for ax, (col, label, unit) in zip(axes, descriptors):
    plot_df = df[[col, "gap", "subset"]].dropna()

    # Visual limits only. Data still saved fully.
    xlo = plot_df[col].quantile(0.01)
    xhi = plot_df[col].quantile(0.99)
    ylo = plot_df["gap"].quantile(0.01)
    yhi = plot_df["gap"].quantile(0.99)

    visible = plot_df[
        (plot_df[col] >= xlo)
        & (plot_df[col] <= xhi)
        & (plot_df["gap"] >= ylo)
        & (plot_df["gap"] <= yhi)
    ]

    full = visible[visible["subset"] == "Full PQR dataset"]

    # Use density instead of thousands of blue points.
    hb = ax.hexbin(
        full[col],
        full["gap"],
        gridsize=55,
        mincnt=1,
        bins="log",
        alpha=0.75,
    )

    # Reference subsets as overlays.
    for subset, marker, size in [
        ("QM9-overlap reference", "o", 14),
        ("Psi4-recomputed reference", "^", 18),
    ]:
        sub = visible[visible["subset"] == subset]
        if len(sub):
            ax.scatter(
                sub[col],
                sub["gap"],
                s=size,
                alpha=0.70,
                marker=marker,
                linewidths=0.35,
                edgecolors="black",
                label=subset,
            )

    # Descriptive trend line.
    if len(visible) > 10 and visible[col].std() > 0:
        m, b = np.polyfit(visible[col], visible["gap"], 1)
        xs = np.linspace(xlo, xhi, 200)
        ax.plot(xs, m * xs + b, linewidth=1.4, color="black", alpha=0.85)

    r = visible[col].corr(visible["gap"])

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("PQR HOMO–LUMO gap (eV)")
    ax.set_title(f"{label}\nr = {r:.2f}")
    ax.grid(alpha=0.18)

axes[-1].axis("off")

# One shared colorbar for density.
cbar = fig.colorbar(hb, ax=axes[:5], shrink=0.72, pad=0.02)
cbar.set_label("Full PQR molecule density, log scale")

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

fig.suptitle(
    "Interpretable Molecular Descriptors versus HOMO–LUMO Gap",
    y=0.985,
    fontsize=16,
)

fig.text(
    0.5,
    0.055,
    "Axes display the central 98% of values for visualization; full data are retained in the exported CSV.",
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.08, 0.95, 0.95])

out_png = OUT / "figure_named_descriptors_vs_gap_clean_density.png"
out_pdf = OUT / "figure_named_descriptors_vs_gap_clean_density.pdf"
out_csv = OUT / "figure_named_descriptors_vs_gap_clean_density_data.csv"

fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

df.to_csv(out_csv, index=False)

print("Saved clean full-dataset descriptor figure:")
print(out_png)
print(out_pdf)
print(out_csv)

# ------------------------------------
# Figure 2: reference-subset-only view
# ------------------------------------

ref_df = df[df["subset"] != "Full PQR dataset"].copy()

if len(ref_df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    axes = axes.ravel()

    for ax, (col, label, unit) in zip(axes, descriptors):
        plot_df = ref_df[[col, "gap", "subset"]].dropna()

        for subset, marker, size in [
            ("QM9-overlap reference", "o", 22),
            ("Psi4-recomputed reference", "^", 28),
        ]:
            sub = plot_df[plot_df["subset"] == subset]
            if len(sub):
                ax.scatter(
                    sub[col],
                    sub["gap"],
                    s=size,
                    alpha=0.75,
                    marker=marker,
                    linewidths=0.35,
                    edgecolors="black",
                    label=subset,
                )

        if len(plot_df) > 10 and plot_df[col].std() > 0:
            m, b = np.polyfit(plot_df[col], plot_df["gap"], 1)
            xs = np.linspace(plot_df[col].min(), plot_df[col].max(), 200)
            ax.plot(xs, m * xs + b, linewidth=1.4, color="black", alpha=0.85)

        r = plot_df[col].corr(plot_df["gap"]) if len(plot_df) > 2 else np.nan

        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("PQR HOMO–LUMO gap (eV)")
        ax.set_title(f"{label}\nr = {r:.2f}")
        ax.grid(alpha=0.18)

    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.suptitle(
        "Interpretable Descriptor Coverage of Real-Reference Molecules",
        y=0.985,
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    out_png2 = OUT / "figure_named_descriptors_reference_only.png"
    out_pdf2 = OUT / "figure_named_descriptors_reference_only.pdf"
    out_csv2 = OUT / "figure_named_descriptors_reference_only_data.csv"

    fig.savefig(out_png2, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf2, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    ref_df.to_csv(out_csv2, index=False)

    print("\nSaved reference-only descriptor figure:")
    print(out_png2)
    print(out_pdf2)
    print(out_csv2)
