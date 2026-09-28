## B2 — per-domain absolute-error distribution (deciles)

Reviewer 3 asked for error distributions, not only summary statistics. Computed on the 424 held-out reference test molecules; raw per-molecule residuals are in `figure6_data.csv` so the box plot is reproducible.

| domain                  |   n |    mae |   rmse |    p10 |    p20 |    p30 |    p40 |    p50 |    p60 |    p70 |    p80 |    p90 |    p95 |    max |
|:------------------------|----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| qm9_like_small_organic  | 254 | 0.2526 | 0.621  | 0.0137 | 0.0276 | 0.0449 | 0.0787 | 0.1174 | 0.1625 | 0.2243 | 0.3351 | 0.4995 | 0.7994 | 7.2273 |
| near_qm9_larger_organic |  20 | 0.4481 | 0.5431 | 0.0635 | 0.1641 | 0.2871 | 0.3568 | 0.3746 | 0.5114 | 0.576  | 0.6613 | 0.8275 | 0.9359 | 1.1965 |
| large_neutral_organic   |  42 | 0.5046 | 0.648  | 0.1077 | 0.1573 | 0.227  | 0.3141 | 0.4242 | 0.5323 | 0.6227 | 0.7337 | 0.9419 | 1.2897 | 1.9285 |
| heteroatom_rich_non_qm9 |  51 | 0.5395 | 0.7094 | 0.0815 | 0.1594 | 0.243  | 0.3331 | 0.4979 | 0.582  | 0.7712 | 0.8541 | 0.9685 | 1.2911 | 2.5688 |
| charged_or_radical      |  57 | 0.7329 | 1.0947 | 0.0885 | 0.1556 | 0.2591 | 0.4003 | 0.4579 | 0.7358 | 0.8591 | 1.1132 | 1.3953 | 1.9473 | 4.7781 |
