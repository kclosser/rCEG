## A7 — zero-gap contamination

- STRICT corpus records with |pqr_gap| < 0.01 eV: **0** of 84,143
- of those, inside the reference-labelled pool: **0**

**Finding: the STRICT corpus is clean.** No placeholder zero-gap records survive cleaning and none sit in the reference pool. The 0 eV band in the earlier Figure 1 came from plotting `enhanced_dataset_lasso.csv`, the PRE-cleaning descriptor table, which still holds the 4,719 rows removed under `gap_below_min_or_placeholder`. Figure 1 has been repointed at the strict corpus and the band is gone. **No reference-pool rebuild is needed; Phase B is unblocked.**
