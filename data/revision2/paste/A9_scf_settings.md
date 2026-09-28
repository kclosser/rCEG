## A9 — Psi4 SCF settings actually used

Read directly from `run_psi4_recompute_batch.py` lines 83–94. These are **explicitly
set**, not Psi4 defaults:

| Option | Value |
|---|---|
| `scf_type` | `df` (density-fitted) |
| `e_convergence` | `1e-6` |
| `d_convergence` | `1e-6` |
| `maxiter` | `150` |
| `reference` | `uks` when multiplicity > 1, otherwise `rks` |
| `basis` | `6-31G(d,p)` for H/C/N/O/F-only molecules, `def2-SVP` otherwise |

**Methods clause, ready to paste:**

> Single-point energies were evaluated with Psi4 at the B3LYP level using density-fitted
> SCF (`scf_type df`), with energy and density convergence thresholds of 1×10⁻⁶ and a
> 150-iteration limit. A restricted Kohn–Sham reference (RKS) was used for closed-shell
> species and an unrestricted reference (UKS) for open-shell species. The basis set was
> assigned by composition: 6-31G(d,p) for molecules containing only H, C, N, O and F, and
> def2-SVP otherwise.

**Two notes.**

1. The reference is **RKS/UKS**, not RHF/UHF. These are Kohn–Sham references for a DFT
   functional; writing RHF/UHF in a B3LYP Methods section would be wrong. (Psi4 accepts
   `rhf`/`uhf` as aliases for `rks`/`uks` when the method is a functional, so the two
   spellings are functionally identical, but only one is correct in prose.)
2. The geometry-optimisation driver written for task C1 passes `rhf`/`uhf`. Functionally
   identical under Psi4's aliasing, so the C1 numbers are comparable to the original
   single points, but the spelling should be harmonised before deposit.
