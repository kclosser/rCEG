from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

# Find convergence history.
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

train_col = next(
    (c for c in ["train_mae", "training_mae"] if c in df.columns),
    None,
)

if iter_col is None or val_col is None:
    raise RuntimeError(
        f"Could not identify iteration/validation MAE columns. Columns: {list(df.columns)}"
    )

df = df.sort_values(iter_col).copy()

fig, ax = plt.subplots(figsize=(7.6, 4.9))

if train_col is not None:
    ax.plot(
        df[iter_col],
        df[train_col],
        marker="o",
        markersize=2.5,
        linewidth=1.3,
        label="Training MAE",
    )

ax.plot(
    df[iter_col],
    df[val_col],
    marker="o",
    markersize=2.5,
    linewidth=1.5,
    label="Validation MAE",
)

best = df.loc[df[val_col].idxmin()]
ax.scatter(
    [best[iter_col]],
    [best[val_col]],
    marker="*",
    s=180,
    zorder=5,
    label=f"Lowest validation MAE = {best[val_col]:.3f} eV",
)

# Add final held-out test MAE if available.
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
            ax.axhline(
                test_mae,
                linestyle="--",
                linewidth=1.2,
                label=f"Final held-out test MAE = {test_mae:.3f} eV",
            )
            break

ax.set_xlabel("Boosting iteration")
ax.set_ylabel("MAE (eV)")
ax.set_title("Model Error Across Training Iterations")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

png = OUT / "figure_training_iteration_vs_error.png"
pdf = OUT / "figure_training_iteration_vs_error.pdf"
csv = OUT / "figure_training_iteration_vs_error_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

df.to_csv(csv, index=False)

print("Saved:")
print(png)
print(pdf)
print(csv)
