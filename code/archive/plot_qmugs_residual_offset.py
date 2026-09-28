from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

RUN = Path("runs/pqr_qmugs_strict_external_LEAK094")
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN / "qmugs_external_predictions.csv")

y = pd.to_numeric(df["ref_gap"], errors="coerce").to_numpy(dtype=float)
pred = pd.to_numeric(
    df["pred_domain_expert"],
    errors="coerce",
).to_numpy(dtype=float)

valid = np.isfinite(y) & np.isfinite(pred)
y = y[valid]
pred = pred[valid]

residual = y - pred

mean_residual = float(np.mean(residual))
median_residual = float(np.median(residual))

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_offsets = []

for fit_idx, eval_idx in kf.split(pred):
    offset = float(
        np.median(y[fit_idx] - pred[fit_idx])
    )
    fold_offsets.append(offset)

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    residual,
    bins=70,
    alpha=0.8,
)

ax.axvline(
    median_residual,
    linestyle="-",
    linewidth=2,
    label=f"Overall median = {median_residual:.3f} eV",
)

ax.axvline(
    mean_residual,
    linestyle="--",
    linewidth=2,
    label=f"Overall mean = {mean_residual:.3f} eV",
)

for i, offset in enumerate(fold_offsets, start=1):
    ax.axvline(
        offset,
        linestyle=":",
        linewidth=1,
        alpha=0.8,
        label=f"Fold {i} offset = {offset:.3f} eV",
    )

ax.set_xlabel(
    "Signed residual: QMugs gap − frozen rCEG prediction (eV)"
)
ax.set_ylabel("Molecule count")
ax.set_title(
    "Distribution of QMugs Target-Scale Residuals"
)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)

fig.text(
    0.5,
    0.01,
    (
        "The concentration of residuals near 4.17 eV indicates a "
        "dominant common reference-scale offset."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

figure_path = OUT / "figure_qmugs_signed_residual_offset.png"

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

figure_data = pd.DataFrame({
    "qmugs_gap_ev": y,
    "frozen_rceg_prediction_ev": pred,
    "signed_residual_ev": residual,
})

figure_data.to_csv(
    OUT / "figure_qmugs_signed_residual_offset_data.csv",
    index=False,
)

pd.DataFrame({
    "fold": np.arange(1, 6),
    "median_offset_ev": fold_offsets,
}).to_csv(
    OUT / "figure_qmugs_signed_residual_fold_offsets.csv",
    index=False,
)

print("Overall mean:", mean_residual)
print("Overall median:", median_residual)
print("Fold offsets:", fold_offsets)
print("\nSaved:")
print(figure_path)
