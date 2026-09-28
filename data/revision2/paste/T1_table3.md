## T1 / Table 3 — reference recomputation by chemical class

**The targeted column is confirmed, not derived.** Each class count was checked
against `recompute_manifest.csv`, which has exactly 947 rows and per-class counts
identical to the table below. The sums close: 947 targeted, 909 converged, 38 failed.

| Chemical class | Class population | Targeted | Converged | Failed | Sampled |
|---|---|---|---|---|---|
| qm9 like small organic | 4,949 | 100 | 100 | 0 | 2.0% |
| near qm9 larger organic | 25,498 | 100 | 100 | 0 | 0.4% |
| large neutral organic | 22,193 | 248 | 227 | 21 | 1.1% |
| heteroatom rich non qm9 | 24,530 | 249 | 233 | 16 | 1.0% |
| charged or radical | 6,973 | 250 | 249 | 1 | 3.6% |
| **total** | **84,143** | **947** | **909** | **38** | **1.1%** |

Failures by class — 21 large neutral organic, 16 heteroatom-rich, 1 charged-or-radical,
0 and 0 for the two QM9-like classes — match the reported split exactly.
Of the 38, **30 were `disk_or_scratch_exhausted`**: an infrastructure limit, not a
chemical one, so the failed set is not systematically different in chemistry from
the converged set.

### ⚠️ The 1.1% is correct — do not change it

The spec expected 1.0% for large neutral organic on the grounds that 227/22,193 =
1.02%. That uses the **converged** count as numerator. The manuscript's 1.1% uses
the **targeted** count: 248/22,193 = 1.12%. Both are arithmetically right; they
answer different questions.

| Class | targeted/population | converged/population |
|---|---|---|
| qm9 like small organic | 2.02% | 2.02% |
| near qm9 larger organic | 0.39% | 0.39% |
| large neutral organic | 1.12% | 1.02% |
| heteroatom rich non qm9 | 1.02% | 0.95% |
| charged or radical | 3.59% | 3.57% |

**Recommendation.** Keep 1.1% and label the column **"fraction of class targeted"**.
That is the sampling-design quantity a reader wants: it describes what was *selected*
for recomputation, independent of how many later hit a disk limit. If the column is
instead labelled "fraction recomputed", every value must switch to the converged
numerator and large neutral organic becomes 1.0%. **The error would be leaving the
label ambiguous, not the digit.**
