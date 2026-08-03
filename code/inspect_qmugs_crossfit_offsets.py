from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

RUN = Path("runs/pqr_qmugs_strict_external_LEAK094")
PREDICTIONS = RUN / "qmugs_external_predictions.csv"

df = pd.read_csv(PREDICTIONS)

required = {"ref_gap", "pred_domain_expert"}
missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )

y = pd.to_numeric(df["ref_gap"], errors="coerce").to_numpy(dtype=float)
pred = pd.to_numeric(
    df["pred_domain_expert"],
    errors="coerce",
).to_numpy(dtype=float)

valid = np.isfinite(y) & np.isfinite(pred)
y = y[valid]
pred = pred[valid]

# Positive residual means QMugs is larger than the frozen prediction.
residual = y - pred

print("Overall residual statistics")
print("=" * 60)
print(f"n:                       {len(residual):,}")
print(f"Mean signed residual:    {np.mean(residual):.9f} eV")
print(f"Median signed residual:  {np.median(residual):.9f} eV")
print(f"Residual SD:             {np.std(residual):.9f} eV")
print(f"Raw MAE:                 {mean_absolute_error(y, pred):.9f} eV")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

aligned = np.zeros(len(y), dtype=float)
fold_rows = []

for fold, (fit_idx, eval_idx) in enumerate(kf.split(pred), start=1):
    fit_residual = y[fit_idx] - pred[fit_idx]

    offset = float(np.median(fit_residual))

    aligned[eval_idx] = pred[eval_idx] + offset

    fold_mae = mean_absolute_error(
        y[eval_idx],
        aligned[eval_idx],
    )

    fold_rows.append({
        "fold": fold,
        "calibration_n": len(fit_idx),
        "held_out_n": len(eval_idx),
        "median_offset_ev": offset,
        "calibration_mean_residual_ev": float(
            np.mean(fit_residual)
        ),
        "held_out_aligned_mae_ev": float(fold_mae),
    })

folds = pd.DataFrame(fold_rows)

print("\nExact five cross-fitted offsets")
print("=" * 60)
print(folds.to_string(index=False))

print("\nOffset summary")
print("=" * 60)
print(folds["median_offset_ev"].describe().to_string())

print("\nCombined cross-fitted metrics")
print("=" * 60)
print(
    f"MAE: {mean_absolute_error(y, aligned):.9f} eV"
)
print(
    f"R2:  {r2_score(y, aligned):.9f}"
)

folds.to_csv(
    RUN / "qmugs_crossfit_fold_offsets.csv",
    index=False,
)

print("\nWrote:")
print(RUN / "qmugs_crossfit_fold_offsets.csv")
