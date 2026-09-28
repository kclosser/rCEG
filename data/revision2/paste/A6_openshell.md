## A6 — PQR vs DFT reference agreement, by species type

| subset                        |   n |   pearson_r |   spearman_rho |   mean_abs_diff_eV |   median_abs_diff_eV |   mean_signed_pqr_minus_ref |
|:------------------------------|----:|------------:|---------------:|-------------------:|---------------------:|----------------------------:|
| all recomputed                | 909 |      0.5894 |         0.5856 |             5.4416 |               4.9619 |                      5.4416 |
| domain == charged_or_radical  | 249 |      0.5047 |         0.2662 |             6.4865 |               6.495  |                      6.4865 |
| charge != 0                   | 246 |      0.5134 |         0.2772 |             6.4521 |               6.4728 |                      6.4521 |
| multiplicity > 1 (open-shell) |  99 |      0.0739 |         0.1243 |             8.2951 |               7.8779 |                      8.2951 |
| neutral closed-shell          | 660 |      0.694  |         0.6289 |             5.0474 |               4.4775 |                      5.0474 |

**Sentence.** Among the 99 open-shell recomputed molecules the PQR semiempirical gap is essentially uncorrelated with the DFT reference (Pearson r = 0.074, Spearman rho = 0.124), with a mean absolute difference of 8.30 eV; for neutral closed-shell species r = 0.694 and the difference is 5.05 eV.

**Draft check.** The manuscript cites 'r = 0.07' — reproduced. It also cites 'a distance of 0.2 eV', which matches NO quantity in this table; locate its source or remove it.
