from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

# Try to find the needed files from either the revised figures folder or runs.
data_candidates = [
    OUT / "figure_piecewise_splits_named_axes_data.csv",
    *ROOT.glob("runs/**/all_pqr_predictions_with_label_source.csv"),
]

branch_candidates = [
    *ROOT.glob("runs/**/piecewise_split_architecture_branches.csv"),
]

data_path = next((p for p in data_candidates if p.exists()), None)
branch_path = next((p for p in branch_candidates if p.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "Could not find piecewise plot data. Expected either:\n"
        "paper_figures_revised/figure_piecewise_splits_named_axes_data.csv\n"
        "or runs/**/all_pqr_predictions_with_label_source.csv"
    )

if branch_path is None:
    raise FileNotFoundError(
        "Could not find piecewise_split_architecture_branches.csv under runs/."
    )

print("Using data:", data_path)
print("Using branches:", branch_path)

df = pd.read_csv(data_path)
branches = pd.read_csv(branch_path)

# Identify y-axis column.
target_candidates = [
    "training_label",
    "calibrated_gap",
    "pred_qm9_aligned_gap",
    "final_prediction",
    "reference_aligned_gap",
]

target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    raise RuntimeError(
        "Could not identify reference-aligned target column. "
        f"Available columns: {list(df.columns)}"
    )

# Clean names for paper.
domain_label = {
    "charged_or_radical": "Charged/radical",
    "near_qm9_larger_organic": "Near-QM9 larger organic",
    "qm9_like_small_organic": "QM9-like small organic",
    "heteroatom_rich_non_qm9": "Heteroatom-rich non-QM9",
    "large_neutral_organic": "Large neutral organic",
}

# If you eventually recover real descriptor names, replace these labels.
feature_label = {
    "x47": "LASSO descriptor 43, internal feature value",
    "x13": "LASSO descriptor 9, internal feature value",
    "x15": "LASSO descriptor 11, internal feature value",
    "x260": "LASSO descriptor 256, internal feature value",
    "x316": "LASSO descriptor 312, internal feature value",
}

# Preferred panel order matching your existing figure.
preferred_order = [
    "charged_or_radical",
    "near_qm9_larger_organic",
    "qm9_like_small_organic",
    "heteroatom_rich_non_qm9",
    "large_neutral_organic",
]

branches["order"] = branches["domain"].map(
    {d: i for i, d in enumerate(preferred_order)}
).fillna(999)

branches = branches.sort_values("order").reset_index(drop=True)

n = len(branches)

fig, axes = plt.subplots(
    n,
    1,
    figsize=(7.2, 10.4),
    sharey=False,
)

if n == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, branches.iterrows()):
    domain = str(row["domain"])
    feature = str(row["feature"])
    split_value = float(row["split_value"])

    # Some branch files call this differently depending on script version.
    improvement = None
    for c in [
        "linear_mae_improvement",
        "mae_improvement",
        "improvement",
    ]:
        if c in row.index:
            try:
                improvement = float(row[c])
            except Exception:
                improvement = None
            break

    if feature not in df.columns:
        print(f"Skipping {domain} because {feature} is not in data.")
        ax.axis("off")
        continue

    plot_df = df.copy()

    if "domain" in plot_df.columns:
        plot_df = plot_df[plot_df["domain"].astype(str) == domain]

    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df[target_col] = pd.to_numeric(plot_df[target_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature, target_col])

    if len(plot_df) > 6000:
        plot_df = plot_df.sample(6000, random_state=42)

    ax.scatter(
        plot_df[feature],
        plot_df[target_col],
        s=5,
        alpha=0.18,
        linewidths=0,
    )

    ax.axvline(
        split_value,
        linestyle="--",
        linewidth=1.3,
    )

    # Separate descriptive linear fits on each side of the learned split.
    left = plot_df[plot_df[feature] <= split_value]
    right = plot_df[plot_df[feature] > split_value]

    for side in [left, right]:
        if len(side) < 20 or side[feature].std() == 0:
            continue

        slope, intercept = np.polyfit(
            side[feature],
            side[target_col],
            1,
        )

        grid = np.linspace(
            side[feature].min(),
            side[feature].max(),
            150,
        )

        ax.plot(
            grid,
            slope * grid + intercept,
            linewidth=1.4,
        )

    # No subplot title. Put the information into x-axis label instead.
    dlabel = domain_label.get(domain, domain.replace("_", " ").title())
    flabel = feature_label.get(feature, feature)

    if improvement is not None and np.isfinite(improvement):
        xlabel = (
            f"{dlabel}: {flabel}; split threshold = {split_value:.4g}; "
            f"linear MAE improvement = {improvement:.3f} eV"
        )
    else:
        xlabel = (
            f"{dlabel}: {flabel}; split threshold = {split_value:.4g}"
        )

    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel("Reference-aligned gap (eV)", fontsize=8.5)
    ax.grid(alpha=0.22)
    ax.tick_params(axis="both", labelsize=8)

    # Small in-panel label only, not a title.
    ax.text(
        0.01,
        0.94,
        dlabel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
            "pad": 2,
        },
    )

# Remove overall title per professor's comment.
fig.tight_layout(h_pad=1.1)

out_png = OUT / "figure_piecewise_splits_clean_no_titles.png"
out_pdf = OUT / "figure_piecewise_splits_clean_no_titles.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

# Save figure metadata for the paper.
meta_rows = []

for _, row in branches.iterrows():
    feature = str(row["feature"])
    domain = str(row["domain"])

    improvement = None
    for c in ["linear_mae_improvement", "mae_improvement", "improvement"]:
        if c in row.index:
            try:
                improvement = float(row[c])
            except Exception:
                improvement = None
            break

    meta_rows.append({
        "domain": domain,
        "domain_label": domain_label.get(domain, domain),
        "selected_feature": feature,
        "x_axis_meaning": (
            "Numerical value of the selected molecular descriptor feature "
            "for each molecule."
        ),
        "split_threshold": row["split_value"],
        "linear_mae_improvement_ev": improvement,
    })

pd.DataFrame(meta_rows).to_csv(
    OUT / "figure_piecewise_splits_clean_metadata.csv",
    index=False,
)

print("Saved:")
print(out_png)
print(out_pdf)
print(OUT / "figure_piecewise_splits_clean_metadata.csv")
