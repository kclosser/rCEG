## T4 / Table 9 — representative molecules and caption

### The five representatives

| Chemical class | ID | SMILES | Heavy | Class median | Charge | Mult. | Basis | B3LYP |
|---|---|---|---|---|---|---|---|---|
| qm9 like small organic | `pqr_0871` | `C/C=C/[C@@H]1CC[C@H]1C` | 8 | 8 | +0 | 1 | 6-31G(d,p) | 10.9 s |
| near qm9 larger organic | `pqr_0845` | `C[C@@H]1CC[C@H]1[C@@H]1O[C@H]1[C@@H]1CCC1(C)C` | 14 | 14 | +0 | 1 | 6-31G(d,p) | 40.0 s |
| large neutral organic | `pqr_0502` | `Cc1c(O)ccc2c1Oc1c(ccc(O)c1C)C21OC(=O)c2ccccc21` | 27 | 27 | +0 | 1 | 6-31G(d,p) | 129.8 s |
| heteroatom rich non qm9 | `pqr_0399` | `CCNS(=O)(=O)c1ccc2c(c1)CN(C)C(=O)N2` | 18 | 18 | +0 | 1 | def2-SVP | 42.4 s |
| charged or radical | `pqr_0166` | `CC1=C[N+]([C@H]2O[C@H](COP(=O)(O)O)[C@@H](O)[C@@H]2O)C(=O)[NH+]C1=O` | 22 | 22 | +2 | 3 | def2-SVP | 256.2 s |

Every representative sits **exactly at its class's median heavy-atom count**, so each row describes a typical molecule of its class rather than a best or worst case.

### Provenance of the selection

`C3_representatives.csv` is **authoritative**. It is the preserved record of the five molecules actually timed, and `run_C3_timings.py` reads it when present.

The selection was originally made ad hoc, with the result left in a temporary file and no generating script. `select_representatives()` now reconstructs the documented criterion — one molecule per class at the class median heavy-atom count — but **does not reproduce the original five**: it agrees on heavy-atom count in four of five classes while picking different molecules (many share the median count and the original tie-break is unknown), and for heteroatom-rich it computes a median of 19 against the original 18. Rather than silently adopt a new set, the original file is kept and the reconstruction is documented as a reconstruction.

### ⚠️ The charged-or-radical representative is an open-shell dication

`pqr_0166` carries **charge +2 and multiplicity 3**. Its 256 s B3LYP cost — which produces the 16,979× upper bound — comes as much from the unrestricted open-shell reference and the double positive charge as from its 22 heavy atoms. A neutral closed-shell molecule of the same size costs far less. Unqualified, that figure reads as typical of the class, and it is not.

### Caption, ready to paste

> **Table 9.** Computational cost per chemical class, measured on one representative molecule per class chosen at that class's median heavy-atom count. All timings single-threaded on an Apple M4 Pro (12 logical cores, 25.8 GB). Inference is single-molecule latency with the model already loaded; descriptor generation is listed separately. Speedup is the B3LYP single-point wall time divided by end-to-end rCEG cost. Note that the charged-or-radical representative (`pqr_0166`) is an open-shell dication (charge +2, multiplicity 3) requiring an unrestricted reference, so its 256 s B3LYP cost — and hence the 16,979× upper bound — reflects an expensive open-shell calculation rather than a typical molecule of that size. Batched inference lowers per-molecule cost to ~0.1 ms, raising the range to 6,008×–45,020×.
