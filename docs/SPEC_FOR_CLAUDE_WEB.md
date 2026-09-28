# rCEG — manuscript revision spec

**Paste this whole file into Claude web, then paste or upload the manuscript.**
Self-contained: every number you need is below, verified against the run that
produced it. No filesystem access required.

**Manuscript:** *A split-architecture machine learning model for scalable
HOMO–LUMO gap prediction: rCEG* — Isaac Wang, Kristina D. Closser.
Revision 2 for *ACS Omega*.

**Your job:** apply the edits in §3 and draft the response to reviewers. All
measurement is finished. Nothing below requires you to compute or estimate a
value.

---

## 0. Rules — read before writing anything

1. **Never invent a number.** If you want a value not in §2, say so and stop.
2. **If the manuscript disagrees with §2, the manuscript is wrong.** These come
   from the frozen run. Change the paper, not the number.
3. **Do not soften a finding to protect a claim.** Several edits make the paper
   more modest. That is deliberate and it is what survives review.
4. **Name the molecule set.** Several quantities exist on more than one
   partition and differ between them. Where §2 names a set, carry that name
   into the text.
5. **Ignore two older documents** if shown them: `RCEG_SYSTEM_SPEC.md` and
   `REVISION_BRIEF_FOR_CLAUDE_WEB.md`. Both predate the final analysis.
6. **Flag, don't guess.** If an instruction cannot be applied because the
   manuscript text differs from what is described, quote what you found and
   ask.

---

## 1. What the model actually is

**Two stages.** Earlier drafts describe something different — see **E3**.

**Stage 1 — calibration.** An ensemble maps the cheap semiempirical gap onto
the DFT reference scale. **Eight candidates, four of them descriptor-based**
(see E11); the best three on validation MAE are combined with weights ∝ 1/MAE.
Its output produces **pseudo-labels** for the unlabelled corpus.

**Stage 2 — prediction.** One ExtraTrees expert per chemical class. Within a
class large enough to support it, a **single learned breakpoint** on one
descriptor splits the class and a separate expert is fitted either side. This
piecewise split is the paper's named contribution.

**There is no meta-learner** — evaluated and dropped (E3).
**Stage 1 and Stage 2 are never applied jointly.** Stage 1 exists only to make
training labels; Stage 2 alone predicts, and never receives a gap of any kind.
At inference the model needs **only a SMILES string** (E10).

Classes: `qm9_like_small_organic`, `near_qm9_larger_organic`,
`large_neutral_organic`, `heteroatom_rich_non_qm9`, `charged_or_radical`.

---

## 2. Every number you may use

Ten repeated splits, seeds 42 + 1000·i, selection on validation only, unless a
row says otherwise.

### 2.1 Headline

| Quantity | Value |
|---|---|
| Test MAE | **0.338 ± 0.026 eV** |
| Test R² | **0.930** |
| Validation MAE (the selection criterion) | 0.344 ± 0.016 eV |
| Selected configuration | `piecewise_split` |
| Repeats | 10 |

⚠️ **Quote accuracy as 0.3 eV in the abstract and conclusions** — see **E5**.

### 2.2 Corpus, and the partitions

| Quantity | Value |
|---|---|
| PQR release | 106,066 |
| Removed by cleaning | 21,923 |
| **Strict corpus** | **84,143** |
| QMugs external holdout, excised before any model development | 9,753 |
| **Development corpus** | **74,390** |
| Reference-labelled pool | **2,123** |

**Two partitions of the reference pool exist and both are real.**

| Partition | Produced by | Governs |
|---|---|---|
| **1,274 / 425 / 424** | repeated-split protocol | Headline, Tables 6, 7, 8 |
| **1,273 / 425 / 425** | legacy single-run path | **Table 2's confidence bands only** |

Both sum to 2,123. Use the first everywhere except Table 2 — see **E12**.

**Training pool:** 73,541 = 1,274 real anchors + **72,267 pseudo-labelled**.
⚠️ The manuscript says 82,020 pseudo-labels. Wrong — see **E9**.

Class populations: near-qm9 larger organic 25,498 · heteroatom-rich non-qm9
24,530 · large neutral organic 22,193 · charged or radical 6,973 · qm9-like
small organic 4,949.

### 2.3 Cleaning waterfall (Table S1)

| Rule | Removed |
|---|---|
| `missing_or_nonstring_smiles` | 140 |
| `sanitize_valence_error` | **10,089** |
| `sanitize_kekulize_error` | 73 |
| `gap_below_min_or_placeholder` | 4,719 |
| `duplicate_rows_removed` | 3,328 |
| `multi_fragment` | 2,005 |
| `target_percentile_trim` | 446 |
| `isolated_hydrogen` | 101 |
| `gap_above_hard_max` | 10 |
| `conflicting_duplicate_discard` | 1,012 |
| **total** | **21,923** |

Four further rules were checked and removed nothing (`missing_pm7_block`,
`numeric_coercion_error`, `not_a_dict`, `missing_descriptor_block`) — keep them
in the SI table; a rule checked and not firing is part of the audit.
106,066 − 21,923 = 84,143, closing exactly.

### 2.4 Per-class performance (Table 7)

Ten repeated splits, with standard deviations across repeats.

| Class | n | MAE (eV) | RMSE (eV) | R² | Median abs err |
|---|---|---|---|---|---|
| qm9-like small organic | 264 | **0.237 ± 0.025** | 0.464 ± 0.115 | 0.909 ± 0.037 | 0.116 |
| near-qm9 larger organic | 20 | **0.389 ± 0.051** | 0.471 ± 0.061 | 0.850 ± 0.057 | 0.366 |
| large neutral organic | 44 | **0.480 ± 0.065** | 0.635 ± 0.078 | 0.621 ± 0.113 | 0.385 |
| heteroatom-rich non-qm9 | 47 | **0.529 ± 0.066** | 0.731 ± 0.106 | 0.490 ± 0.198 | 0.434 |
| charged or radical | 49 | **0.537 ± 0.106** | 0.781 ± 0.178 | 0.888 ± 0.059 | 0.368 |
| **all** | 424 | **0.338 ± 0.026** | 0.574 ± 0.070 | 0.930 ± 0.013 | 0.182 |

**The overall row reconciles exactly with the headline (0.338 ± 0.026 eV)**, so
a reader reconstructing it from the class rows recovers the reported figure.

⚠️ **The two weakest classes are statistically tied** — see **E2**.

By reference source (single split, not yet re-run): QM9-labelled **0.227 eV**
(n = 232); Psi4-recomputed **0.580 eV** (n = 191). ⚠️ The draft carries 0.229
for the first; correct to **0.227**. A third row with n = 1 should be
footnoted or dropped.

The median absolute error is well below the MAE in every class. Quote both.

### 2.4b Stage 1 versus rCEG, same test partition

| Model | Test MAE (eV) |
|---|---|
| Calibration (Stage 1, **given** the semiempirical gap) | **0.327 ± 0.028** |
| rCEG `piecewise_split` (descriptors only) | 0.338 ± 0.026 |
| rCEG `domain_expert` (descriptors only) | 0.335 ± 0.027 |

rCEG is **+0.0116 ± 0.0049 eV** behind the calibration mapping, worse in
**10 of 10** repeats, paired p < 0.0001. See **E15**.

### 2.5 Calibration, per class (Table 4)

**Accuracy on held-out reference molecules** (validation + test pooled,
n = 849 per repeat, mean over ten repeats). The ensemble never saw these:

| Class | n | MAE | RMSE | R² |
|---|---|---|---|---|
| qm9-like small organic | 524 | 0.226 | 0.433 | 0.922 |
| near-qm9 larger organic | 39 | 0.356 | 0.445 | 0.879 |
| large neutral organic | 91 | 0.473 | 0.624 | 0.647 |
| heteroatom-rich non-qm9 | 93 | 0.506 | 0.753 | 0.499 |
| charged or radical | 102 | 0.572 | 0.861 | 0.869 |
| **all** | 849 | **0.331** | 0.569 | 0.934 |

No SDs are available for these columns — the run aggregated before writing.

**Raw semiempirical-minus-reference offset, by class and set:**

| Class | Calibration set | Held-out set |
|---|---|---|
| qm9-like small organic | 5.205 | 5.226 |
| near-qm9 larger organic | 5.404 | 5.349 |
| large neutral organic | 4.983 | 4.915 |
| heteroatom-rich non-qm9 | 4.760 | 4.737 |
| charged or radical | 6.396 | 6.613 |
| **spread** | **1.637 eV** | **1.876 eV** |

⚠️ **Do not write "1.876 calibration / 2.310 held-out."** That pairing was
wrong. 1.876 is the held-out spread; the calibration spread is 1.637. The
figure 2.310 came from a file with no generating script and is **superseded —
do not quote it**.

**The offset differs across classes decisively:** one-way ANOVA **F = 37.38,
p = 8.3 × 10⁻²⁹**; Kruskal–Wallis H = 91.35, p = 6.8 × 10⁻¹⁹; **η² = 0.150**.
Class accounts for ~15% of the variance. A single global offset would misplace
`charged_or_radical` by ~+1.2 eV and `heteroatom_rich_non_qm9` by ~−0.6 eV —
several times the model's own error. See **E13**.

### 2.6 Confidence bands (Table 2)

Assigned over the **74,390-molecule development corpus** — not 84,143.

| Axis | Band | Rule | n | % |
|---|---|---|---|---|
| Applicability | `high_in_domain` | ≤ 90th pct, distance ≤ **20.08** | 60,653 | 81.53% |
| | `medium_near_domain` | ≤ 97.5th pct, ≤ **30.46** | 10,588 | 14.23% |
| | `low_out_of_domain` | > 97.5th pct | 3,149 | 4.23% |
| Cycle | `high_cycle_consistency` | error ≤ 0.20 eV | 10,684 | 14.36% |
| | `medium_cycle_consistency` | 0.20 < error ≤ 0.50 eV | 12,186 | 16.38% |
| | `low_cycle_consistency` | error > 0.50 eV | 51,520 | 69.26% |
| Combined | high / medium / low | pessimistic: worst axis wins | 9,451 / 12,755 / 52,184 | 12.70 / 17.15 / 70.15% |

Distance is 1-nearest-neighbour Euclidean to the calibration set in
RobustScaler space after median imputation; thresholds are percentiles of
held-out reference distances. See **E12**.

### 2.7 Ablation (Table 6)

| Configuration | MAE ± SD (eV) | R² |
|---|---|---|
| `domain_expert` | 0.335 ± 0.027 | 0.933 |
| `full_rceg` (meta-learner) | 0.338 ± 0.023 | 0.937 |
| **`piecewise_split`** (selected) | **0.338 ± 0.026** | 0.930 |
| `no_exact_mass` | 0.347 ± 0.027 | 0.928 |
| `global_et` | 0.347 ± 0.027 | 0.928 |
| `global_rf` | 0.349 ± 0.025 | 0.929 |
| `global_mean` | 0.361 ± 0.026 | 0.926 |
| `global_hgb` | 0.405 ± 0.024 | 0.915 |
| `domain_permuted_piecewise` | 0.437 ± 0.023 | 0.903 |
| `domain_permuted` | 0.440 ± 0.026 | 0.902 |
| `global_ridge` | 0.684 ± 0.024 | 0.820 |

The two `domain_permuted` rows are the control: class labels shuffled with
sizes preserved, so only the chemical meaning of the partition is destroyed.

### 2.8 Split protocols (Table 8)

| Protocol | MAE ± SD | R² | Median max Tanimoto to train |
|---|---|---|---|
| random | 0.335 ± 0.026 | 0.933 | 0.455 |
| scaffold | 0.360 ± 0.025 | 0.925 | 0.347 |
| cluster | 0.337 ± 0.021 | 0.933 | 0.421 |

Scaffold costs **+7.3%** while cutting median nearest-neighbour similarity from
0.455 to 0.347. Real but modest — the direct answer to the near-duplicate
objection.

### 2.9 Recomputation (Table 3)

| Class | Population | Targeted | Converged | Failed | Targeted % |
|---|---|---|---|---|---|
| qm9-like small organic | 4,949 | 100 | 100 | 0 | 2.0% |
| near-qm9 larger organic | 25,498 | 100 | 100 | 0 | 0.4% |
| large neutral organic | 22,193 | 248 | 227 | 21 | 1.1% |
| heteroatom-rich non-qm9 | 24,530 | 249 | 233 | 16 | 1.0% |
| charged or radical | 6,973 | 250 | 249 | 1 | 3.6% |
| **total** | **84,143** | **947** | **909** | **38** | **1.1%** |

30 of 38 failures were disk/scratch exhaustion — infrastructure, not chemistry
— so the failed set is not chemically distinct. **Label the last column
"fraction of class targeted"**; it uses the targeted count, which is why large
neutral organic reads 1.1% and not 1.0%. Both are arithmetically right; the
ambiguity is the defect.

### 2.10 Computational cost (Table 9)

All single-threaded, Apple M4 Pro, 12 logical cores, 25.8 GB.

| Class | Heavy | Descriptors | Inference | End-to-end | B3LYP | Speedup |
|---|---|---|---|---|---|---|
| qm9-like small organic | 8 | 1.7 ms | 9.5 ms | 11.2 ms | 10.9 s | 972× |
| near-qm9 larger organic | 14 | 2.7 ms | 9.5 ms | 12.2 ms | 40.1 s | 3,280× |
| heteroatom-rich non-qm9 | 18 | 2.9 ms | 9.5 ms | 12.4 ms | 42.4 s | 3,409× |
| charged or radical | 22 | 5.6 ms | 9.5 ms | 15.1 ms | 256.2 s | 16,979× |
| large neutral organic | 27 | 4.5 ms | 9.5 ms | 14.0 ms | 129.8 s | 9,294× |

- Inference is **single-molecule latency**, re-measured independently at
  9.611 ms (1.2% from the tabulated 9.496). Near-constant in molecule size
  (9.57–9.79 ms), which is why one value appears in all five rows — say so.
- **Speedup range: 972× – 16,979×.** ⚠️ Quote the range, never one figure (E6).
- Batched inference gives 0.073–0.133 ms/molecule and 6,008×–45,020×. Use only
  if the paper makes a screening-throughput claim, and label it.
- B3LYP re-timed on the same machine: 116.8 s (n=20) vs 119 s from cluster logs
  (n=332) — within 2%, so no conclusion changes, but state the hardware.
- The charged-or-radical representative (`pqr_0166`) is an **open-shell
  dication** (charge +2, multiplicity 3); its 256 s cost drives the 16,979×
  upper bound — see **E6**.

### 2.11 Structure-only ablation

| Configuration | Features | MAE | Δ vs full | paired p |
|---|---|---|---|---|
| full | 427 | 0.3384 ± 0.026 | — | — |
| drop x0–x4 | 422 | 0.3385 ± 0.025 | +0.0001 (+0.04%) | **0.814** |
| drop x2–x4 | 424 | 0.3390 ± 0.026 | +0.0006 (+0.16%) | 0.230 |

**Inference needs only SMILES.** Required caveat: training labels are
calibrated pseudo-labels derived from a semiempirical quantity, so the claim is
about *inference*, not training. State both halves.

### 2.12 External validation (QMugs, 9,753 molecules)

Excised by SMILES before any fitting.

| Evaluation | MAE | R² |
|---|---|---|
| Frozen transfer | 4.19 eV | −14.5 |
| 5-fold cross-fit, offset-aligned | **0.521 eV** | **0.600** |

Per class, offset-aligned: qm9-like small organic 0.436 (n=134) · large neutral
organic 0.504 (n=4,016) · near-qm9 larger organic 0.520 (n=1,939) ·
heteroatom-rich non-qm9 0.543 (n=3,664). Spread 0.107 eV — **near-uniform**,
which supports the transfer claim.

The five cross-fit fold offsets are 4.1755 / 4.1786 / 4.1820 / 4.1857 / 4.1863
eV, median 4.1820, spread **0.0108 eV**: the alignment estimates one genuine
protocol constant, not per-fold noise.

⚠️ **QMugs contains no charged-or-radical molecules**, so the weakest class is
untested externally. Say so.

Framing: rCEG transfers in **ranking and shape but not absolute placement**; a
new reference protocol needs a handful of labelled anchors to re-align. State
as a limitation.

### 2.13 Limits on achievable accuracy

| Bound | Value |
|---|---|
| Geometry sensitivity (UFF vs B3LYP-optimised) | mean **0.440 eV**, median 0.279, n = 37 of 50 |
| Cross-protocol label floor (QM9 vs Psi4, 14 molecules) | **0.295 eV** |
| Conformational floor (9,697 molecules, ≥3 conformers) | **0.076 eV** |

Together these make the strongest argument in the paper: **the model operates
near the noise floor of its own reference data.**

### 2.14 Descriptors and leakage

- 427 features: `x0–x4` semiempirical scalars, `x5–x379` LASSO block,
  `x380–x410` RDKit, `x411–x426` bond-step topology.
- **176 of 375 LASSO-selected descriptors (47%) are Morgan fingerprint bits.**
- **None of the five learned breakpoints is a fingerprint bit**: HallKierAlpha
  (−0.400), TPSA (12.530), VSA_EState6 (0.000), Ipc (1,186.695), BCUT2D_MRLOW
  (−0.451). All resolved uniquely.
- Leakage audit: largest |r| between any feature and any leakage-relevant
  target is **0.534**; 5 features exceed 0.5; none approaches the 0.94 excision
  threshold.
- Open-shell agreement, semiempirical vs DFT: **r = 0.074** (n=99) against
  r = 0.694 for neutral closed-shell (n=660).

### 2.15 Methods details

- Reference: B3LYP, **6-31G(d,p)** for H/C/N/O/F-only molecules, **def2-SVP**
  otherwise. Both must be stated.
- **RKS** closed-shell, **UKS** open-shell. Not RHF/UHF.
- `scf_type df`, `e_convergence` 1e-6, `d_convergence` 1e-6, `maxiter` 150.
- Of 947 recomputed, 247 carry non-zero charge and 99 are open-shell.
- Experts: ExtraTrees, 500 trees, `max_features` 0.35, `min_samples_leaf` 1.
- Thresholds: own expert at **500** training molecules; piecewise split
  considered at **1,000**; ≥ **250** molecules each side of a breakpoint.
- Determinism: `PYTHONHASHSEED=0`, `zlib.crc32` seeding, byte-identical reruns
  verified.

---

## 3. The edits

### E1 — Reference pool: 2,137 → 2,123
Every occurrence. 1,228 + 909 − 14 = 2,123, matching 1,274 + 425 + 424.

⚠️ **One 2,137 is legitimate:** Figure 1 plots 2,137 points because it shows
two series with the 14 dual-labelled molecules in both. A plot count, not a
pool count. Do not change it.

### E2 — Worst class: write it as a tie, not a single class

On the ten-repeat table the two weakest classes are **statistically
indistinguishable**: heteroatom-rich non-QM9 at 0.529 ± 0.066 eV and
charged-or-radical at 0.537 ± 0.106 eV, differing by 0.008 eV with paired
p = 0.84. Heteroatom-rich is worst in 6 of 10 repeats, charged-or-radical in 2,
large neutral organic in 2.

**Write something like:** *the two weakest classes are heteroatom-rich non-QM9
and charged-or-radical species, at 0.53 and 0.54 eV; the difference between
them is not significant across repeated splits.*

⚠️ **If the submitted text already names heteroatom-rich as weakest, it was
right** — the single-split table that appeared to contradict it was the
outlier (it put charged-or-radical at 0.733 eV, near the top of that class's
0.456–0.734 range). Check the text before editing it.

Still worth saying: charged-or-radical has by far the **largest spread**
(± 0.106 vs ± 0.066), so it is the least *predictable* class even where it is
not the least accurate. That is consistent with semiempirical and DFT barely
agreeing for open-shell species (r = 0.074 vs 0.694) — a property of the input
data, not a failure of the model.

### E3 — Remove the meta-learner from the Introduction
It was dropped: `full_rceg` is +0.0053 eV *worse* than `piecewise_split` on
validation, better in only 3 of 10 repeats, paired p = 0.12. Rewrite to the
two-stage form in §1. Check the abstract and any architecture caption. Keep it
as an ablation row only.

### E4 — The open-shell "0.2 eV"
No computed quantity matches it (§2.14: r = 0.074; mean absolute difference
8.3 eV). Locate what it referred to and label it, or delete it.

### E5 — Quote accuracy to one decimal place
Use **0.3 eV** in the abstract and conclusions.

State the reason in the Discussion: the mean geometry-induced shift is
**0.440 eV**, which *exceeds the model's own test MAE*. The third significant
figure is not meaningful when a defensible change of input geometry moves the
answer by more than the error quoted. Keep precise values in tables where the
protocol is fully specified.

Also state the accuracy is **conditional on the UFF-optimised input geometry**.

When reporting the geometry number, attach both caveats: only 37 of 50
optimisations converged, and the shortfall biases the estimate **both** ways —
optimiser failures concentrate in large flexible molecules whose gaps move
least (raising the mean), while `gau_loose` convergence stops nearer the start
(lowering it).

### E6 — Speedup as a range
Replace any single figure with **972× – 16,979×**, or name a class.

Add to the Table 9 caption that the charged-or-radical representative
(`pqr_0166`) is an **open-shell dication** (charge +2, multiplicity 3)
requiring an unrestricted reference, so its 256 s B3LYP cost — and the 16,979×
upper bound — reflects an expensive open-shell calculation rather than a
typical molecule of that size. Also state that inference is single-molecule
latency and near-constant in size.

### E7 — Narrow the interpretability claim
176 of 375 selected descriptors are fingerprint bits, so the feature set as a
whole is not interpretable. The claim that survives is narrower and stronger:
**none of the five learned breakpoints falls on a fingerprint bit** — each is a
named physicochemical descriptor. Make that distinction explicitly.

### E8 — Figure 2: swap in the regenerated file
Figure 2 has been **regenerated**. Its inference path previously listed PM7
semiempirical properties as *required*, conceding the objection §2.11 answers.
It now draws that box dashed and labelled "(optional)", with a bypass arc
annotated "structure-only route: no quantum calculation".

Swap the file in and add a caption clause: *semiempirical inputs are not
required at inference.*

### E9 — Pseudo-label count: 82,020 → 72,267
82,020 is 84,143 − 2,123 and omits the 9,753 QMugs excision, counting molecules
never available for training.

Use **72,267** — the training-pool count, matching Figure 2's
73,541 = 1,274 + 72,267. (If the sentence is rewritten to describe labelling
rather than training, 73,117 over the development corpus is correct.) State the
9,753 excision explicitly wherever corpus sizes appear.

### E10 — Add the structure-only result
A **new result** answering the objection that rCEG needs PQR-derived inputs.
Add a short Results subsection from §2.11. Claim: *at inference, rCEG needs
only a SMILES string.* Caveat that must travel with it: *training labels derive
from a semiempirical quantity, so this is about inference, not training.*

### E11 — Calibration candidates: ten/six → eight/four
The Methods state ten candidates, six descriptor-based. The frozen model has
**eight, four descriptor-based**: four on the gap alone
(`gap_linear_robust`, `gap_ridge_robust`, `gap_ridge_standard`,
`gap_ridge_power`) and four on the full descriptor vector (`desc_ridge_robust`,
`desc_hgb`, `desc_et`, `desc_rf`). Top three by validation MAE are combined
with weights ∝ 1/MAE.

Nothing else changes — a counting error in prose, not in the model.

### E12 — Table 2: state the denominator and the partition
Two clauses:
1. Bands are assigned over the **74,390-molecule development corpus**, not
   84,143.
2. They derive from the **legacy single-run partition (1,273 / 425 / 425)**,
   whereas the accuracy tables use the repeated-split partition
   (1,274 / 425 / 424).

Without the second, a reader reconciling Table 2 against Table 7 finds
partitions that do not match. Add the applicability cut-offs (20.08, 30.46)
from §2.6 to the previously blank row.

### E13 — State why calibration is descriptor-conditioned
Currently treated as an implementation detail. It is a **requirement**, and
§2.5 now proves it: F = 37.38, p = 8.3 × 10⁻²⁹, η² = 0.150. A constant
correction would misplace charged species by ~+1.2 eV. Write this as the
justification for the design.

### E14 — Figure 3 caption
Two corrections:
1. It plots `global_hgb` at **0.458 eV** on the single split shown — *not* the
   ten-repeat 0.405 eV, and *not* the headline 0.338 eV. Say which.
2. The baseline **has not converged**: best iteration equals the 700-iteration
   cap, validation MAE still falling at −0.005 eV per 100 iterations. Describe
   it as a fixed 700-iteration budget. **Do not call it a convergence plot.**
   (This does not change the conclusion: the 0.066 eV ten-repeat gap would need
   ~1,300 further iterations to close, at a decelerating rate.)

### E15 — State the Stage 1 versus rCEG comparison

Currently absent from the paper, and worth adding.

The calibration mapping is **handed** the semiempirical gap and reaches
0.327 ± 0.028 eV. rCEG reaches 0.338 ± 0.026 eV from descriptors alone, on the
same test molecules in the same repeats.

**State it accurately: this is not parity.** rCEG is behind by
+0.0116 ± 0.0049 eV, in 10 of 10 repeats, paired p < 0.0001. The correct claim
is that **rCEG gives up about 12 meV — roughly 4% of the error — in exchange
for needing no quantum calculation at inference at all.** Quantifying that
trade is stronger than leaving it unstated, and it corroborates the
structure-only result (E10) from a different direction.

Do not quote the older pairing of 0.331 against 0.338: those were measured on
different molecule sets (n = 849 pooled versus n = 424 test) and are not
comparable.

---

## 4. The one open decision

**PM6 versus PM7.**

- The distributed PQR data exposes properties under a field named `pm7`, and no
  other method string appears in the records.
- The **PQR project FAQ states PM6 as implemented in MOPAC**, and two referees
  independently described the data as PM6.

Project documentation outranks a field label, so the balance favours **PM6**.

**Recommended:** verify the FAQ, cite it, use **PM6** throughout, and add one
sentence noting the distributed data exposes these properties under a field
named `pm7` so a reader checking the raw download is not confused.

**Do not** find-and-replace without that verification. Flag it in your output.

---

## 5. Response to reviewers — the strongest points

1. **"Not structure-only" is answered outright.** Dropping every semiempirical
   input costs +0.0001 eV, p = 0.814 (§2.11).
2. **The weak class is a data limitation, not a model failure.** Semiempirical
   and DFT barely agree on open-shell molecules, r = 0.074 vs 0.694 (§2.14).
3. **Descriptor-conditioned calibration is required, not preferred.**
   F = 37.38, p = 8.3 × 10⁻²⁹; a constant correction misplaces charged species
   by ~1.2 eV (§2.5). This answers the domain-independence question decisively
   rather than adequately.
4. **The model sits near its own reference noise floor.** Two protocols
   disagree by 0.295 eV; conformers span 0.076 eV; geometry choice moves the
   answer 0.440 eV (§2.13).
5. **Transfer is real but conditional.** Offset-aligned MAE 0.521 eV on 9,753
   held-out QMugs molecules, near-uniform across classes, fold offset stable to
   ±0.005 eV (§2.12).

Concede plainly, without hedging: the 2,137 error, the worst-class
inconsistency, the leftover meta-learner text, the unattributed 0.2 eV, the
miscounted calibration candidates, the 82,020 pseudo-label figure, the
cross-hardware timing comparison (re-measured, unchanged), and the
interpretability overreach.

---

## 6. Do not use these numbers

| Wrong | Correct |
|---|---|
| 0.318 eV headline | **0.338 eV** |
| 2,137 reference pool | **2,123** (Figure 1's 2,137 plot count is fine) |
| "1.876 calibration / 2.310 held-out" | **1.637 calibration / 1.876 held-out**; drop 2.310 |
| 82,020 pseudo-labels | **72,267** |
| Ten calibration candidates, six descriptor-based | **Eight, four** |
| 0.229 eV QM9 source row | **0.227 eV** |
| Per-class Table 7 from a single split | **ten-repeat values in §2.4** |
| "calibration 0.331 vs rCEG 0.338" | not comparable; use **0.327 vs 0.338** (§2.4b) |
| "0.2 eV" open-shell agreement | no such quantity; use r = 0.074 |
| 0.347 eV anywhere | not a current value |
| 38.3 ms inference | **9.5 ms** single-molecule |
| A single speedup figure | **972× – 16,979×** |
| Confidence bands over 84,143 | over **74,390** |
| RHF/UHF for the DFT reference | **RKS/UKS** |
| PM7 without qualification | see §4 |

---

## 7. What to hand back

1. The revised manuscript, changes marked or listed.
2. A response-to-reviewers letter.
3. Anything you could not apply, with the manuscript text you found quoted.
   **A reported blocker is worth more than a plausible edit.**
