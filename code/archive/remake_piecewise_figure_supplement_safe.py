from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

data_path = OUT / "figure_piecewise_splits_named_axes_data.csv"
branch_candidates = list(ROOT.glob("runs/**/piecewise_split_architecture_branches.csv"))

if not data_path.exists():
    raise FileNotFoundError(f"Missing {data_path}")

if not branch_candidates:
    raise FileNotFoundError("Could not find piecewise_split_architecture_branches.csv under runs/")

branch_path = sorted(branch_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

df = pd.read_csv(data_path)
branches = pd.read_csv(branch_path)

target_candidates = [
    "training_label",
    "calibrated_gap",
    "pred_qm9_aligned_gap",
    "final_prediction",
    "reference_aligned_gap",
]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    raise RuntimeError(f"Could not identify y-axis target column. Columns: {list(df.columns)}")

domain_label = {
    "charged_or_radical": "Charged/radical",
    "near_qm9_larger_organic": "Near-QM9 larger organic",
    "qm9_like_small_organic": "QM9-like small organic",
    "heteroatom_rich_non_qm9": "Heteroatom-rich non-QM9",
    "large_neutral_organic": "Large neutral organic",
}

# Honest labels: these are model features, not physical units.
feature_label = {
    "x47": "MorganFP_80 fingerprint value",
    "x13": "BCUT2D_MWLOW descriptor value",
    "x15": "MolLogP descriptor value",
    "x260": "Internal LASSO feature 256",
    "x316": "Internal LASSO feature 312",
}

feature_note = {
    "x47": "hashed fingerprint bit/count; unitless",
    "x13": "RDKit BCUT descriptor; unitless/eigenvalue-like descriptor",
    "x15": "estimated logP; unitless",
    "x260": "unit unavailable",
    "x316": "unit unavailable",
}

preferred_order = [
    "charged_or_radical",
    "near_qm9_larger_organic",
    "qm9_like_small_organic",
    "heteroatom_rich_non_qm9",
    "large_neutral_organic",
]

branches["order"] = branches["domain"].map({d: i for i, d in enumerate(preferred_order)}).fillna(999)
branches = branches.sort_values("order").reset_index(drop=True)

fig, axes = plt.subplots(len(branches), 1, figsize=(7.2, 11.4), sharey=False)

if len(branches) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, branches.iterrows()):
    domain = str(row["domain"])
    feature = str(row["feature"])
    split_value = float(row["split_value"])

    improvement = None
    for c in ["linear_mae_improvement", "mae_improvement", "improvement"]:
        if c in row.index:
            try:
                improvement = float(row[c])
            except Exception:
                improvement = None
            break

    plot_df = df.copy()
    if "domain" in plot_df.columns:
        plot_df = plot_df[plot_df["domain"].astype(str) == domain]

    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df[target_col] = pd.to_numeric(plot_df[target_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature, target_col])

    if len(plot_df) > 6000:
        plot_df = plot_df.sample(6000, random_state=42)

    ax.scatter(plot_df[feature], plot_df[target_col], s=5, alpha=0.18, linewidths=0)
    ax.axvline(split_value, linestyle="--", linewidth=1.2)

    for side in [plot_df[plot_df[feature] <= split_value], plot_df[plot_df[feature] > split_value]]:
        if len(side) < 20 or side[feature].std() == 0:
            continue
        slope, intercept = np.polyfit(side[feature], side[target_col], 1)
        grid = np.linspace(side[feature].min(), side[feature].max(), 150)
        ax.plot(grid, slope * grid + intercept, linewidth=1.3)

    dlabel = domain_label.get(domain, domain.replace("_", " ").title())
    flabel = feature_label.get(feature, feature)
    fnote = feature_note.get(feature, "unitless/internal feature")

    ax.set_xlabel(f"{flabel} ({fnote})", fontsize=8.2)
    ax.set_ylabel("Reference-aligned gap (eV)", fontsize=8.2)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.22)

    if improvement is not None and np.isfinite(improvement):
        annotation = f"{dlabel}\nSplit = {split_value:.4g}; diagnostic MAE improvement = {improvement:.3f} eV"
    else:
        annotation = f"{dlabel}\nSplit = {split_value:.4g}"

    ax.text(
        0.01,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.75", boxstyle="round,pad=0.25"),
    )

fig.tight_layout(h_pad=1.35)

out_png = OUT / "supplement_piecewise_internal_features.png"
out_pdf = OUT / "supplement_piecewise_internal_features.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

print("Saved:")
print(out_png)
print(out_pdf)
