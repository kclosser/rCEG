## C3 / Table 9 — computational cost per chemical domain

One molecule per domain at that domain's **median heavy-atom count**, so the table shows typical rather than best or worst cases. All timings on Apple M4 Pro, 12 logical cores, **single-threaded**.

| domain | molecule | heavy atoms | basis | semiempirical | descriptors | inference | end-to-end in PQR | end-to-end novel | B3LYP reference | speedup in PQR / novel |
|---|---|---|---|---|---|---|---|---|---|---|
| qm9 like small organic | `pqr_0871` | 8 | 6-31G(d,p) | 44 ms | 1.7 ms | 9.50 ms | 11.2 ms | 55.1 ms | 10.9 s | **972x** / 198x |
| near qm9 larger organic | `pqr_0845` | 14 | 6-31G(d,p) | 48 ms | 2.7 ms | 9.50 ms | 12.2 ms | 60.5 ms | 40.0 s | **3,280x** / 662x |
| heteroatom rich non qm9 | `pqr_0399` | 18 | def2-SVP | 45 ms | 2.9 ms | 9.50 ms | 12.4 ms | 57.1 ms | 42.4 s | **3,409x** / 742x |
| charged or radical | `pqr_0166` | 22 | def2-SVP | 26 ms | 5.6 ms | 9.50 ms | 15.1 ms | 41.3 ms | 256.2 s | **16,979x** / 6,205x |
| large neutral organic | `pqr_0502` | 27 | 6-31G(d,p) | 65 ms | 4.5 ms | 9.50 ms | 14.0 ms | 79.3 ms | 129.8 s | **9,294x** / 1,638x |

Median across domains: B3LYP **42.4 s**, end-to-end in-PQR **12.4 ms**, novel **57.1 ms**. Consistent with task T1, which measured the same B3LYP single point over 20 molecules at a 116.8 s median.

### Notes

- The **semiempirical column is GFN2-xTB**, a labelled proxy. PM7 via MOPAC was unavailable on the benchmark machine; it must not be reported as a PM7 timing. The `method` column of `timing_semiempirical.csv` names exactly what was run.

- **Both end-to-end columns are given deliberately.** Reviewer 4's objection was that a molecule outside PQR needs a semiempirical calculation first — that is the *novel* column. Reporting only the in-PQR column is the omission objected to.

- The structure-only ablation (B6) showed the five PQR scalars can be dropped with no measurable accuracy loss (paired p = 0.81), so the semiempirical step is avoidable entirely; in that variant a novel molecule costs the same as an in-PQR one.

### Two deliberate deviations from the specified protocol

1. **B3LYP used 3 repeats, not 5.** A single call on the largest representative takes minutes with small run-to-run variance, and T1 independently supplies n = 20 for the same quantity. Millisecond-scale measurements used the full 5.

2. **Inference was timed in a separate environment**, because the psi4 conda environment has no scikit-learn. Same physical machine, single-threaded in both.

