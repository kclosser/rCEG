## C6 — which baseline does Figure 3 plot?

**Both circulating values are real. They are different quantities measured on different things.**

| Value | What it is | Basis |
|---|---|---|
| **0.405 ± 0.024 eV** | `global_hgb` test MAE in the ablation table | **Ten repeated splits** |
| **0.458 eV** | `global_hgb` test MAE at the best validation iteration of the warm-start curve | **One split** (the primary split, seed 42) |
| 0.456 eV | the same model fitted with early stopping rather than warm-start checkpointing | One split |

**Figure 3 plots the single-split warm-start curve, so the figure's own annotation is 0.458 eV.** The 0.405 eV in the ablation table is the ten-repeat mean and is the right number *for that table*. Neither is wrong; the caption simply has to say which it is showing.

### Caption-ready values

| Quantity | Value |
|---|---|
| Test MAE of the plotted configuration | **0.4579 eV** |
| Minimum reference-validation MAE | **0.4369 eV** |
| Iteration at which it occurs | **700** |
| Iteration cap (`max_iter`) | 700 |
| Early stopping triggered? | **no** |
| Slope of the validation curve over the last 100 iterations | **-0.00527 eV** |

### The gap to rCEG — quote it on a consistent basis

| Comparison | Baseline | rCEG | Gap |
|---|---|---|---|
| Ten repeated splits (ablation table) | 0.405 | 0.338 | **0.066 eV** |
| The single split Figure 3 plots | 0.458 | 0.386 | **0.072 eV** |

The often-quoted 0.066 eV is the **ten-repeat** comparison. Do not pair it with the figure's 0.458 eV, and do not compare 0.458 against the ten-repeat headline of 0.338 — that mixes a single split with an average and overstates the gap.

### ⚠️ The baseline is truncated, not converged

The best iteration equals the cap (700 = 700) and validation MAE is still falling at the cutoff, by 0.0053 eV over the last 100 iterations. `global_hgb` is shown **under-trained**, which slightly flatters the ablation.

**It does not change the conclusion.** At the observed — and decelerating — rate, closing the 0.066 eV gap would need roughly 1,258 further iterations. But the figure should not be described as showing convergence; say it is shown at a fixed 700-iteration budget.

Legend, caption text and `figure3_data.csv` all read these values from the same row of `global_hgb_convergence_summary.csv`, so the earlier 0.403-versus-0.402 drift between legend and caption cannot recur.
