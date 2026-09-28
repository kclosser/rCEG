from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

candidates = sorted(
    ROOT.glob("runs/**/global_hgb_convergence_history.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not candidates:
    raise FileNotFoundError(
        "Could not find global_hgb_convergence_history.csv under runs/. "
        "Copy or regenerate the model run outputs first."
    )

history_path = candidates[0]
print("Using convergence file:", history_path)

hist = pd.read_csv(history_path)
print("Columns:", list(hist.columns))

iter_col = None
for c in ["iteration", "iter", "n_iter", "boosting_iteration"]:
    if c in hist.columns:
        iter_col = c
        break

mae_col = None
for c in ["val_mae", "validation_mae", "test_mae", "mae"]:
    if c in hist.columns:
        mae_col = c
        break

if iter_col is None or mae_col is None:
    raise RuntimeError(
        f"Could not identify iteration and MAE columns. Columns: {list(hist.columns)}"
    )

hist = hist.sort_values(iter_col).copy()
best = hist.loc[hist[mae_col].idxmin()]

fig, ax = plt.subplots(figsize=(7.5, 4.8))

ax.plot(
    hist[iter_col],
    hist[mae_col],
    marker="o",
    markersize=3,
    linewidth=1.4,
    label="Validation MAE",
)

ax.scatter(
    [best[iter_col]],
    [best[mae_col]],
    marker="*",
    s=180,
    zorder=5,
    label=f"Lowest validation MAE = {best[mae_col]:.3f} eV",
)

# Optional held-out test MAE reference line, if available.
metrics_files = sorted(
    ROOT.glob("runs/**/FINAL_reference_holdout_metrics.csv"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

for metrics_path in metrics_files:
    metrics = pd.read_csv(metrics_path)
    if "name" in metrics.columns and "mae" in metrics.columns:
        m = metrics[metrics["name"].astype(str).eq("TEST global_hgb")]
        if len(m):
            test_mae = float(m.iloc[0]["mae"])
            ax.axhline(
                test_mae,
                linestyle="--",
                linewidth=1.2,
                label=f"Final held-out test MAE = {test_mae:.3f} eV",
            )
            break

ax.set_xlabel("Boosting iteration")
ax.set_ylabel("Validation MAE (eV)")
ax.set_title("Model Error Across Training Iterations")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

out_png = OUT / "figure_error_vs_training_iteration.png"
out_pdf = OUT / "figure_error_vs_training_iteration.pdf"
out_csv = OUT / "figure_error_vs_training_iteration_data.csv"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

hist.to_csv(out_csv, index=False)

print("Saved:")
print(out_png)
print(out_pdf)
print(out_csv)
