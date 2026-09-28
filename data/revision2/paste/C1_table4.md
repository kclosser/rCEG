## C1 / Table 4 — calibration accuracy and offset, per chemical class

Reviewer 3 asked whether the semiempirical-to-DFT correction is domain-independent. Reviewer 2 asked for the calibration model's own accuracy on reference molecules it never saw. Both are answered below.

### C1.1 Calibration accuracy on HELD-OUT reference molecules

These are the validation and test partitions pooled (n = 849 per repeat, 425 + 424), averaged over the ten repeated splits. **The calibration ensemble never saw these molecules.** This is the number Reviewer 2 asked for.

| Chemical class | n | MAE (eV) | RMSE (eV) | R² |
|---|---|---|---|---|
| qm9 like small organic | 524 | **0.226** | 0.433 | 0.922 |
| near qm9 larger organic | 39 | **0.356** | 0.445 | 0.879 |
| large neutral organic | 91 | **0.473** | 0.624 | 0.647 |
| heteroatom rich non qm9 | 93 | **0.506** | 0.753 | 0.499 |
| charged or radical | 102 | **0.572** | 0.861 | 0.869 |
| **all** | 849 | **0.331** | 0.569 | 0.934 |

By reference source:

| Source | n | MAE (eV) | RMSE (eV) | R² |
|---|---|---|---|---|
| qm9 overlap | 482 | 0.212 | 0.408 | 0.932 |
| qm9 overlap+recomputed pqr reference | 5 | 0.162 | 0.202 | 0.974 |
| recomputed pqr reference | 362 | 0.492 | 0.731 | 0.864 |

Counts are means over ten repeats, hence non-integer. **No standard deviation across repeats is available for these columns**: the run aggregated them before writing, and the per-repeat rows were not retained. The offsets in §C1.2, which were recomputed here, do carry one.

### C1.2 Raw semiempirical-minus-reference offset, by class and by set

The offset is a property of the **data**, not of a fitted model, so it is meaningful on both partitions and neither is in-sample. Recomputed here over all ten repeats.

| Chemical class | Calibration set |  | Held-out set |  |
|---|---|---|---|---|
| | n | mean ± SD (eV) | n | mean ± SD (eV) |
| qm9 like small organic | 789 | 5.205 ± 1.815 | 524 | 5.226 ± 1.828 |
| near qm9 larger organic | 61 | 5.404 ± 1.826 | 39 | 5.349 ± 1.799 |
| large neutral organic | 136 | 4.983 ± 1.464 | 91 | 4.915 ± 1.435 |
| heteroatom rich non qm9 | 141 | 4.760 ± 1.676 | 93 | 4.737 ± 1.605 |
| charged or radical | 147 | 6.396 ± 2.223 | 102 | 6.613 ± 2.281 |

- **Calibration set:** per-class mean offset spans 4.760 – 6.396 eV, spread **1.637 eV**.
- **Held-out set:** spans 4.737 – 6.613 eV, spread **1.876 eV**.

The two sets agree closely class by class — the largest disagreement is 0.217 eV — so the structure is a property of the chemistry, not of one partition. **Name the set in every row and never average the two spreads.**

### C1.3 Does the offset differ significantly across classes?

**Yes, decisively.** On the held-out molecules of a single split (n = 849, five classes):

| Test | Statistic | p |
|---|---|---|
| One-way ANOVA | F = 37.38 | 8.3 × 10⁻²⁹ |
| Kruskal–Wallis | H = 91.35 | 6.8 × 10⁻¹⁹ |

Effect size η² = **0.150**: chemical class accounts for about 15% of the variance in the semiempirical-to-DFT offset.

**This is the justification for conditioning the calibration on descriptors rather than subtracting a constant**, and the manuscript should say so explicitly rather than treating it as an implementation detail. A single global offset would misplace `charged_or_radical` by roughly +1.2 eV and `heteroatom_rich_non_qm9` by roughly −0.6 eV relative to the corpus mean — several times the model's own 0.338 eV error.

### C1.4 A correction to earlier revision documents

Earlier drafts of the SI and summary described the 1.876 eV spread as coming from the **calibration set**. It does not: `calibration_by_domain_and_source.csv` is computed on held-out reference molecules only (`ref_val` + `ref_test`), as the emitting code states. The correct pairing is **1.637 eV on the calibration set and 1.876 eV on the held-out set**, both averaged over ten repeats, and both are in the table above.

A third figure, 2.310 eV, appears in `B4_domain_independence_test.csv`. That file has n = 425 (a single partition), carries no MAE, and has no generating script in the repository. **It is superseded by the table above and should not be quoted.**
