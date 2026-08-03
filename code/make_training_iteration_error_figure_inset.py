from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

hits = sorted(
    ROOT.glob("runs/**/global_hgb_convergence_history.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not hits:
    raise FileNotFoundError(
        "Could not find global_hgb_convergence_history.csv under runs/."
    )

path = hits[0]
print("Using:", path)

df = pd.read_csv(path)
print("Columns:", list(df.columns))

iter_col = next(
    (c for c in ["iteration", "iter", "n_iter", "boosting_iteration"] if c in df.columns),
    None,
)

val_col = next(
    (c for c in ["val_mae", "validation_mae", "valid_mae", "mae"] if c in df.columns),
    None,
)

if iter_col is None or val_col is None:
    raise RuntimeError(
        f"Could not identify iteration/validation MAE columns. Columns: {list(df.columns)}"
    )

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

fig, ax = plt.subplots(figsize=(7.2, 4.5))

ax.plot(
    df[iter_col],
    df[val_col],
    marker="o",
    markersize=2.2,
    linewidth=1.35,
    label="Validation MAE",
)

ax.scatter(
    [best[iter_col]],
    [best[val_col]],
    marker="*",
    s=145,
    zorder=5,
    label=f"Minimum validation MAE = {best[val_col]:.3f} eV",
)

if test_mae is not None:
    ax.axhline(
        test_mae,
        linestyle="--",
        linewidth=1.15,
        label=f"Final held-out test MAE = {test_mae:.3f} eV",
    )

ax.set_xlabel("Boosting iteration")
ax.set_ylabel("Mean absolute error (eV)")
ax.grid(alpha=0.22)
ax.legend(fontsize=8, loc="upper right")

# No separate title. Caption will explain the figure.

# Inset zoom from 300 to 700.
zoom = df[(df[iter_col] >= 300) & (df[iter_col] <= 700)].copy()

axins = inset_axes(
    ax,
    width="43%",
    height="43%",
    loc="center right",
    borderpad=2.0,
)

axins.plot(
    zoom[iter_col],
    zoom[val_col],
    marker="o",
    markersize=1.6,
    linewidth=1.0,
)

axins.scatter(
    [best[iter_col]],
    [best[val_col]],
    marker="*",
    s=80,
    zorder=5,
)

if test_mae is not None:
    axins.axhline(
        test_mae,
        linestyle="--",
        linewidth=0.9,
    )

axins.set_xlim(300, 700)

# Tight y-limits around the zoom region.
ymin = min(zoom[val_col].min(), test_mae if test_mae is not None else zoom[val_col].min())
ymax = max(zoom[val_col].max(), test_mae if test_mae is not None else zoom[val_col].max())
pad = max((ymax - ymin) * 0.18, 0.005)
axins.set_ylim(ymin - pad, ymax + pad)

axins.set_xlabel("Iteration", fontsize=7)
axins.set_ylabel("MAE", fontsize=7)
axins.tick_params(axis="both", labelsize=7)
axins.grid(alpha=0.20)
axins.set_title("Iterations 300–700", fontsize=8)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", linewidth=0.8)

fig.tight_layout()

png = OUT / "figure_training_iteration_vs_error_with_inset.png"
pdf = OUT / "figure_training_iteration_vs_error_with_inset.pdf"
csv = OUT / "figure_training_iteration_vs_error_with_inset_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

df.to_csv(csv, index=False)

print("Saved:")
print(png)
print(pdf)
print(csv)
