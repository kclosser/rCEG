## T1 — DFT reference timing, measured locally

Reviewer 4 asked for a DFT reference time alongside the inference time. The manuscript quoted 119 s from the production recomputation logs, but those ran under SLURM on account `csn100` while the rCEG timings were measured on a laptop — a cross-hardware comparison. The reference calculation has now been re-timed **on the same machine as every other number in Table 9**.

Hardware: Apple M4 Pro, 12 logical cores. Single-threaded, `scf_type df`, `e_convergence`/`d_convergence` 1e-6, basis by composition (6-31G(d,p) for CHNOF, def2-SVP otherwise) — identical settings to the production recomputation.

| quantity | value |
|---|---|
| B3LYP single point, local, n=20 | **116.8 s** median (IQR 27.8-204.0) |
| same, from the cluster production logs, n=332 | 119 s median (IQR 51-273) |
| range across 7-37 heavy atoms | 7.0-4668 s |

**The two agree to within 2%.** The local median of 116.8 s essentially reproduces the 119 s cluster figure, so the reported speedup does not change — but it can now be stated as a like-for-like measurement on one machine rather than across two.

### Speedup, restated on one machine

| scenario | cost | vs DFT |
|---|---|---|
| B3LYP reference single point | 116.8 s | 1x |
| rCEG, molecule already in PQR (descriptors 3.1 ms + single-molecule inference 38.3 ms) | 42.2 ms | **2,767x** |
| rCEG, novel molecule (GFN2-xTB 38 ms + descriptors + inference) | 80 ms | **1,466x** |

**Which inference number this uses.** The 42.2 ms total is descriptors
(3.06 ms) plus **single-molecule cold latency** (38.3 ms). An earlier version
of this fragment wrote the parenthetical as "inference 0.40 ms", which is the
*batched, amortized* figure and does not sum to the total — 3.1 + 0.40 = 3.5,
not 42.2. Three inference figures exist in the run outputs and they are easy
to confuse:

| Value | What it measures | Source |
|---|---|---|
| 0.40 ms | batched, amortized over 20 molecules | `timing_benchmark.csv` |
| 9.50 ms | the figure used in Table 9 | `C3_timings.csv` |
| 38.3 ms | single-molecule cold latency (used here) | `timing_benchmark.csv` |

Batching changes the number by a factor of ~95, so whichever is quoted must
be labelled. See `DISCREPANCIES.md` §D2 — the choice is still open and it
moves the speedup ratios.

**Both rows must appear.** Reviewer 4's objection was that a molecule outside PQR needs a semiempirical calculation first. That is correct, and it is the second row. Quoting only the first would be the omission they objected to.

The semiempirical figure is a **GFN2-xTB** single point used as a labelled proxy: PM7 via MOPAC was unavailable on the benchmark machine, and xtb is the same class of method at the same cost scale. It must not be reported as a PM7 timing. See `timing_semiempirical.csv`, whose `method` column names exactly what was run.

**Note on B6.** The structure-only ablation showed the five PQR scalars can be dropped with no measurable accuracy loss (p = 0.81), so the semiempirical step is avoidable entirely — in which case a novel molecule costs the same 42.2 ms as one already in PQR. That variant is the stronger answer to Reviewer 4 and should be mentioned here.

