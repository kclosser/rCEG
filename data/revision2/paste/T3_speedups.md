## T3 — inference latency re-measured, and the Table 9 figure reproduced

Hardware: **Apple M4 Pro, 12 logical cores, 25.8 GB**, Python 3.14.0. Single-threaded (`n_jobs=1`) to match every other timing in Table 9. Descriptor generation is excluded and reported separately, as it is in the table's own column.

### The tabulated 9.50 ms is reproduced

Median single-molecule latency across the five representatives is **9.611 ms**, against the **9.496 ms** Table 9 reports — a difference of 1.2%. **Table 9's inference column is the single-molecule latency**, and it is now regenerable from `run_C3_inference.py`.

| Chemical class | Heavy | Single-molecule (ms) | p10–p90 | Batched, per molecule (ms) |
|---|---|---|---|---|
| qm9 like small organic | 8 | **9.57** | 9.46–10.30 | 0.073 |
| near qm9 larger organic | 14 | **9.61** | 9.56–10.39 | 0.114 |
| large neutral organic | 27 | **9.72** | 9.61–10.13 | 0.133 |
| heteroatom rich non qm9 | 18 | **9.58** | 9.46–10.29 | 0.131 |
| charged or radical | 22 | **9.79** | 9.55–10.93 | 0.102 |

n = 50 timed calls per molecule after warm-up for the single-molecule figures; batches of 1,000.

### Latency is flat in molecule size — which justifies the single value

Single-molecule latency spans only **9.57–9.79 ms** from 8 to 27 heavy atoms, a 2% spread. Inference cost is set by the fitted forest, not by the molecule, so reporting one value for all five rows was a defensible simplification rather than an error. It should still be stated as such in the caption.

### Speedups under both definitions

| Chemical class | B3LYP | End-to-end, single | Speedup | End-to-end, batched | Speedup |
|---|---|---|---|---|---|
| qm9 like small organic | 10.9 s | 11.3 ms | **965×** | 1.8 ms | 6,008× |
| near qm9 larger organic | 40.0 s | 12.3 ms | **3,249×** | 2.8 ms | 14,147× |
| large neutral organic | 129.8 s | 14.2 ms | **9,150×** | 4.6 ms | 28,178× |
| heteroatom rich non qm9 | 42.4 s | 12.5 ms | **3,387×** | 3.1 ms | 13,812× |
| charged or radical | 256.2 s | 15.4 ms | **16,655×** | 5.7 ms | 45,020× |

- **Single-molecule (answering one query): 965× – 16,655×.** This reproduces the published 972×–16,979× range.
- **Batched (screening a library): 6,008× – 45,020×.**

**Recommendation: keep the single-molecule definition.** It is what the published range already uses, it is the conservative choice, and it answers the question a reader actually asks — what does one prediction cost. Quote the batched figure only if the paper makes a screening-throughput claim, and label it explicitly if so.

### One number still unexplained

`timing_benchmark.csv` records `inference_single_molecule_latency` = **38.3 ms**, four times what is measured here under the same definition on the same hardware. That measurement routed through a different code path (the full multi-model stack rather than a single domain expert) and is not reproduced by this script. It should not be quoted. The two figures this fragment establishes — 9.6 ms single, 0.11 ms batched — are the ones with a committed script behind them.
