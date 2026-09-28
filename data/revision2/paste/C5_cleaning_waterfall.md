## C5 — cleaning waterfall, itemised

From the 106,066-record PQR release to the 84,143-molecule strict corpus. Every rule that was checked is listed, including those that removed nothing — a rule firing zero times is part of the audit, not an omission.

| Stage | Rule | Removed | Running total |
|---|---|---|---|
| — | *PQR release as downloaded* | — | 106,066 |
| descriptor extraction | `missing_or_nonstring_smiles` | 140 | 105,926 |
| descriptor extraction | `missing_pm7_block` | 0 | 105,926 |
| descriptor extraction | `numeric_coercion_error` | 0 | 105,926 |
| descriptor extraction | `not_a_dict` | 0 | 105,926 |
| descriptor extraction | `missing_descriptor_block` | 0 | 105,926 |
| descriptor extraction | `sanitize_valence_error` | 10,089 | 95,837 |
| descriptor extraction | `sanitize_kekulize_error` | 73 | 95,764 |
| clean pass | `gap_below_min_or_placeholder` | 4,719 | 91,045 |
| clean pass | `duplicate_rows_removed` | 3,328 | 87,717 |
| clean pass | `multi_fragment` | 2,005 | 85,712 |
| clean pass | `target_percentile_trim` | 446 | 85,266 |
| clean pass | `isolated_hydrogen` | 101 | 85,165 |
| clean pass | `gap_above_hard_max` | 10 | 85,155 |
| strict pass | `conflicting_duplicate_discard` | 1,012 | 84,143 |
| — | **strict corpus** | **21,923 removed** | **84,143** |

The waterfall closes exactly: 106,066 − 21,923 = 84,143, with no unexplained residual.

**10,089 of the 21,923 removals (46%) are RDKit valence-sanitisation failures** — records whose SMILES could not be parsed into a chemically valid molecule. The next largest are placeholder or sub-threshold gaps (4,719), exact duplicates (3,328) and multi-fragment entries (2,005).

For the Methods, the cleaned set is defined as: every PQR record whose SMILES sanitises to a single connected molecule, carries a physically meaningful gap, and is not a duplicate or a conflicting duplicate of another record.

*Note.* The strict corpus of 84,143 is the starting point for modelling, but the **development corpus is 74,390** after the 9,753 QMugs external molecules are excised. See `paste/C8_label_accounting.md`.
