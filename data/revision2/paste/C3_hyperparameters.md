## C3 — hyperparameters, complete listing

Read from the frozen model source (`pqr_full_domain_moe_qmugs_external.py`), not transcribed from any earlier document.

### One-sentence Methods summary

> Domain and branch experts are ExtraTrees regressors with 500 trees, `max_features` = 0.35 and no minimum-leaf constraint; the calibration ensemble combines the three best of eight candidates (ridge and linear variants on the semiempirical gap alone, and ridge, HistGradientBoosting, ExtraTrees and RandomForest on the full descriptor vector) weighted by inverse validation MAE; a domain receives its own expert at 500 training molecules and is considered for a piecewise split at 1,000, with at least 250 molecules required on each side of a breakpoint.

### Shared pipeline

Every learner is wrapped in: median imputation → variance threshold (1e-12) → optional scaling → estimator. Tree learners are **not** scaled (split thresholds are scale-invariant); linear learners are (ridge penalties are not).

### Prediction stage

| Component | Estimator | Hyperparameters |
|---|---|---|
| Domain expert (one per class ≥ `min_domain`) | `ExtraTreesRegressor` | `n_estimators=500`, `max_features=0.35`, `min_samples_leaf=1`, `random_state = seed + 200 + crc32(domain)`, `n_jobs=1` at predict |
| Piecewise branch experts (left / right of the breakpoint) | `ExtraTreesRegressor` | identical, seeded `seed + 700/800 + crc32(domain + feature + side)` |

### Calibration stage — all eight candidates

| Candidate | Estimator | Inputs | Hyperparameters |
|---|---|---|---|
| `gap_linear_robust` | `LinearRegression` | semiempirical gap only | RobustScaler, quantile range (10, 90) |
| `gap_ridge_robust` | `RidgeCV` | gap only | `alphas = logspace(-6, 5, 60)`, RobustScaler |
| `gap_ridge_standard` | `RidgeCV` | gap only | same grid, StandardScaler |
| `gap_ridge_power` | `RidgeCV` | gap only | same grid, Yeo-Johnson |
| `desc_ridge_robust` | `RidgeCV` | gap + 427 descriptors | same grid, RobustScaler |
| `desc_hgb` | `HistGradientBoostingRegressor` | gap + descriptors | `max_iter=700`, `learning_rate=0.035`, `max_leaf_nodes=31`, `l2_regularization=0.03`, `loss='absolute_error'`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=80`, `random_state=seed+10` |
| `desc_et` | `ExtraTreesRegressor` | gap + descriptors | `n_estimators=700`, `max_features=0.35`, `min_samples_leaf=1`, `random_state=seed+20` |
| `desc_rf` | `RandomForestRegressor` | gap + descriptors | `n_estimators=500`, `max_features=0.35`, `min_samples_leaf=1`, `random_state=seed+30` |

Candidates are ranked by MAE on the held-out validation partition; the **top 3** are combined with weights ∝ 1/MAE, renormalised. Scores are quantised to 12 decimal places before ranking — a determinism guard, see below.

### Ablation baselines (not part of the frozen architecture)

| Configuration | Estimator | Hyperparameters |
|---|---|---|
| `global_et` | `ExtraTreesRegressor` | `n_estimators=500`, `max_features=0.35`, `min_samples_leaf=1` |
| `global_rf` | `RandomForestRegressor` | `n_estimators=400`, `max_features=0.35`, `min_samples_leaf=1` |
| `global_hgb` | `HistGradientBoostingRegressor` | as `desc_hgb` above |
| `global_ridge` | `RidgeCV` | `alphas = logspace(-6, 5, 60)`, Robust and Standard variants |

### Architecture thresholds

| Parameter | Value | Meaning |
|---|---|---|
| `min_domain` | **500** | Minimum training molecules before a class gets its own expert; smaller classes fall back to the global-mean model |
| `min_piecewise` | **1,000** | Minimum training molecules before a class is considered for a piecewise split |
| `min_leaf` (branch size) | **`max(250, min(750, n_domain // 12))`** | Minimum molecules on each side of a candidate breakpoint |
| `max_features` (breakpoint search) | **50** | Candidate descriptors, ranked by absolute correlation with the training label |
| `top_k` (calibration ensemble) | **3** | Candidates retained |
| Leakage excision threshold | **\|r\| ≥ 0.94** | Against gap, HOMO, LUMO and LUMO−HOMO |
| Variance threshold | **1e-12** | Near-constant column removal |

### Determinism configuration

| Guard | Setting |
|---|---|
| Hash seed | `PYTHONHASHSEED=0` on every command |
| Seed derivation | `zlib.crc32` of a string key, not Python's salted `hash()` |
| Calibration score quantisation | 12 decimal places before ranking and weighting |
| Pseudo-label rounding | 9 decimal places |
| Predict-time parallelism | `n_jobs` forced to 1 by `DeterministicPipeline` |

These were added after a reproducibility audit found a 1e-17 difference in calibration weights — from `RandomForestRegressor` with `n_jobs=-1` accumulating tree predictions in a non-fixed order — being amplified by near-tied ExtraTrees splits into up to **0.096 eV** on individual predictions. Byte-identical reruns were verified afterwards.
