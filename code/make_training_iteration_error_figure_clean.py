from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

hits = sorted(
    ROOT.glob("runs/**/global_hgb_convergence_history.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not hits:
    raise FileNotFoundError("Could not find global_hgb_convergence_history.csv under runs/.")

path = hits[0]
print("Using:", path)

df = pd.read_csv(path)

iter_col = next(
    (c for c in ["iteration", "iter", "n_iter", "boosting_iteration"] if c in df.columns),
    None,
)
val_col = next(
    (c for c in ["val_mae", "validation_mae", "valid_mae", "mae"] if c in df.columns),
    None,
)

if iter_col is None or val_col is None:
    raise RuntimeError(f"Could not identify columns. Columns: {list(df.columns)}")

df = df.sort_values(iter_col).copy()
best = df.loc[df[val_col].idxmin()]

# Find final held-out test MAE if available.
test_mae = None
metric_hits = sorted(
    ROOT.glob("runs/**/FINAL_reference_holdout_metrics.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

for mpath in metric_hits:
    metrics = pd.read_csv(mpath)
    if {"name", "mae"}.issubset(metrics.columns):
        row = metrics[metrics["name"].astype(str).eq("TEST global_hgb")]
        if len(row):
            test_mae = float(row.iloc[0]["mae"])
            break

fig, ax = plt.subplots(figsize=(7.0, 4.35))

# Main curve: line only, no giant marker clutter.
ax.plot(
    df[iter_col],
    df[val_col],
    linewidth=2.0,
    label="Validation MAE",
)

# Minimum marker.
ax.scatter(
    best[iter_col],
    best[val_col],
    marker="*",
    s=135,
    zorder=5,
    label=f"Minimum validation MAE = {best[val_col]:.3f} eV",
)

# Test MAE.
if test_mae is not None:
    ax.axhline(
        test_mae,
        linestyle="--",
        linewidth=1.5,
        label=f"Held-out test MAE = {test_mae:.3f} eV",
    )

ax.set_xlabel("Boosting iteration")
ax.set_ylabel("Mean absolute error (eV)")
ax.grid(alpha=0.18)

# Cleaner axes.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Smaller inset, no connector lines.
zoom = df[(df[iter_col] >= 300) & (df[iter_col] <= 700)].copy()

axins = inset_axes(
    ax,
    width="38%",
    height="36%",
    loc="upper center",
    bbox_to_anchor=(0.12, -0.04, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=1.0,
)

axins.plot(
    zoom[iter_col],
    zoom[val_col],
    linewidth=1.5,
)

axins.scatter(
    best[iter_col],
    best[val_col],
    marker="*",
    s=70,
    zorder=5,
)

if test_mae is not None:
    axins.axhline(
        test_mae,
        linestyle="--",
        linewidth=1.0,
    )

axins.set_xlim(300, 700)

ymin = min(zoom[val_col].min(), test_mae if test_mae is not None else zoom[val_col].min())
ymax = max(zoom[val_col].max(), test_mae if test_mae is not None else zoom[val_col].max())
pad = max((ymax - ymin) * 0.22, 0.004)
axins.set_ylim(ymin - pad, ymax + pad)

axins.set_title("Plateau region", fontsize=8)
axins.set_xlabel("Iteration", fontsize=7)
axins.set_ylabel("MAE", fontsize=7)
axins.tick_params(axis="both", labelsize=7)
axins.grid(alpha=0.16)

for spine in ["top", "right"]:
    axins.spines[spine].set_visible(False)

# Put legend below plot, not over data.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
    fontsize=8.5,
)

fig.tight_layout(rect=[0, 0.08, 1, 1])

png = OUT / "figure_training_iteration_vs_error_clean_inset.png"
pdf = OUT / "figure_training_iteration_vs_error_clean_inset.pdf"
csv = OUT / "figure_training_iteration_vs_error_clean_inset_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

df.to_csv(csv, index=False)

print("Saved:")
print(png)
print(pdf)
print(csv)
