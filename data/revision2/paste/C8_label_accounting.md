## C8 — label accounting, reconciled

The Methods state **82,020 pseudo-labelled training targets**. That is 84,143 − 2,123: the strict corpus minus the reference pool. It omits the 9,753 QMugs molecules excised before any fitting, so it counts 9,753 molecules that were never available for training. **It is wrong under every reading.**

### The chain

| Step | Count |
|---|---|
| PQR release | 106,066 |
| − removed by cleaning | 21,923 |
| **= strict corpus** | **84,143** |
| − QMugs external holdout, excised before any model development | 9,753 |
| **= development corpus** | **74,390** |

Within the development corpus, by `label_source` in `all_pqr_predictions_with_label_source.csv`:

| Label source | n |
|---|---|
| `real_reference_anchor` | 1,273 |
| `calibrated_pseudo_label` | **73,117** |
| **total** | **74,390** |

### Two defensible figures — pick by what the sentence claims

| Reading | Anchors | Pseudo-labelled | Total |
|---|---|---|---|
| (a) Labels assigned across the development corpus | 1,273 | **73,117** | 74,390 |
| (b) Molecules actually trained on, per repeat | 1,274 | **72,267** | 73,541 |

They differ because (b) removes the 849 held-out reference molecules (425 validation + 424 test) that are never trained on, and because the two code paths partition the reference pool slightly differently — see `paste/C2_table2.md` and DISCREPANCIES §B9.

### Which number the manuscript should carry

**72,267.** The sentence says *pseudo-labelled training targets*, and (b) is the count of molecules carrying a pseudo-label that the model actually trains on. It is also the figure already drawn in Figure 2, whose flow reads 73,541 training pool = 1,274 real anchors + 72,267 pseudo-labelled — so adopting it makes the text and the figure agree.

If the sentence is instead rewritten to describe the labelling step rather than the training step, 73,117 is correct. **Do not use 82,020 under either reading**, and state the 9,753 excision explicitly wherever corpus sizes are given, since its absence is what produced the error.
