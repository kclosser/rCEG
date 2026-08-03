from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

RUN = Path("runs/pqr_qmugs_strict_external_LEAK094")
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(
    RUN / "qmugs_external_predictions.csv"
)

required = {
    "ref_gap",
    "pred_domain_expert",
}

missing = required - set(pred.columns)
if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

y = pred["ref_gap"].to_numpy(dtype=float)
p = pred["pred_domain_expert"].to_numpy(dtype=float)

valid = np.isfinite(y) & np.isfinite(p)
y = y[valid]
p = p[valid]

median_offset = float(np.median(y - p))
p_offset = p + median_offset

linear = LinearRegression()
linear.fit(
    p.reshape(-1, 1),
    y,
)

p_linear = linear.predict(
    p.reshape(-1, 1)
)

raw_mae = mean_absolute_error(y, p)
offset_mae = mean_absolute_error(y, p_offset)
linear_mae = mean_absolute_error(y, p_linear)

raw_r2 = r2_score(y, p)
offset_r2 = r2_score(y, p_offset)
linear_r2 = r2_score(y, p_linear)

corr = np.corrcoef(y, p)[0, 1]

lo = min(
    y.min(),
    p.min(),
    p_offset.min(),
    p_linear.min(),
)

hi = max(
    y.max(),
    p.max(),
    p_offset.max(),
    p_linear.max(),
)

fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(
    y,
    p,
    s=7,
    alpha=0.15,
    label=f"Raw frozen predictions: MAE={raw_mae:.3f} eV",
)

ax.scatter(
    y,
    p_offset,
    s=7,
    alpha=0.15,
    label=f"Offset-aligned: MAE={offset_mae:.3f} eV",
)

ax.plot(
    [lo, hi],
    [lo, hi],
    linewidth=1.5,
    label="Ideal prediction",
)

ax.set_xlabel("QMugs DFT HOMO–LUMO gap (eV)")
ax.set_ylabel("rCEG prediction (eV)")
ax.set_title(
    "QMugs External Transfer Before and After Reference-Scale Alignment"
)

ax.grid(alpha=0.25)
ax.legend(fontsize=9)

diagnostic_text = (
    f"n = {len(y):,}\n"
    f"Pearson r = {corr:.3f}\n"
    f"Median offset = {median_offset:.3f} eV\n"
    f"Linear slope = {linear.coef_[0]:.3f}\n"
    f"Linear intercept = {linear.intercept_:.3f} eV\n"
    f"Linear MAE = {linear_mae:.3f} eV\n"
    f"Linear R² = {linear_r2:.3f}"
)

ax.text(
    0.03,
    0.97,
    diagnostic_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.9,
    },
)

fig.tight_layout()

fig.savefig(
    OUT / "figure_qmugs_scale_alignment_diagnostic.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

figure_data = pd.DataFrame({
    "qmugs_reference_gap_ev": y,
    "raw_frozen_prediction_ev": p,
    "offset_aligned_prediction_ev": p_offset,
    "linear_aligned_prediction_ev": p_linear,
})

figure_data.to_csv(
    OUT / "figure_qmugs_scale_alignment_diagnostic_data.csv",
    index=False,
)

summary = pd.DataFrame([
    {
        "evaluation": "raw frozen",
        "mae_ev": raw_mae,
        "r2": raw_r2,
        "pearson_r": corr,
        "n": len(y),
    },
    {
        "evaluation": "median-offset aligned",
        "mae_ev": offset_mae,
        "r2": offset_r2,
        "pearson_r": corr,
        "n": len(y),
    },
    {
        "evaluation": "in-sample linear diagnostic",
        "mae_ev": linear_mae,
        "r2": linear_r2,
        "pearson_r": corr,
        "n": len(y),
    },
])

summary.to_csv(
    OUT / "qmugs_scale_alignment_summary.csv",
    index=False,
)

print(summary.to_string(index=False))
print("\nLinear slope:", linear.coef_[0])
print("Linear intercept:", linear.intercept_)
print("\nSaved figure and figure-data files to:")
print(OUT)
