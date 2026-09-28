## B4 — is the PQR→reference correction domain-independent?

Reviewer 3 asked for the PQR-reference gap differences reported separately by domain, and for a determination of whether the correction is approximately domain-independent. Computed on the 425 held-out reference test molecules.

| domain                  |   n |   mean |     sd |   median |
|:------------------------|----:|-------:|-------:|---------:|
| charged_or_radical      |  35 | 6.8275 | 2.5293 |   6.8975 |
| heteroatom_rich_non_qm9 |  51 | 4.5172 | 1.561  |   3.9897 |
| large_neutral_organic   |  40 | 5.0049 | 1.3162 |   4.6393 |
| near_qm9_larger_organic |  16 | 5.8221 | 2.2475 |   6.009  |
| qm9_like_small_organic  | 283 | 5.1924 | 1.804  |   5.8766 |

Spread of the mean correction across domains: **2.310 eV**.

| test | statistic | p |
|---|---|---|
| one-way ANOVA | F = 9.226 | 3.74e-07 |
| Kruskal-Wallis | H = 24.995 | 5.04e-05 |
| Levene (equal variance) | W = 6.820 | 2.51e-05 |

**Answer: the correction is NOT domain-independent at the 95% level.**

The mean PQR-minus-reference difference differs significantly across the five chemical domains (ANOVA p = 3.7e-07, Kruskal-Wallis p = 5.0e-05), spanning 2.31 eV from the smallest to the largest. A single global offset therefore could not serve all of PQR, which is precisely why a descriptor-conditioned calibration is required rather than a constant shift.

**This supports the method rather than conceding a weakness.** The companion number to quote alongside it is the residual after calibration: the raw discrepancy varies by 2.31 eV across domains, but the descriptor-conditioned calibration removes 89-95% of it in every domain, leaving a residual spread of 0.43 eV (see `calibration_by_domain_and_source.csv`).

