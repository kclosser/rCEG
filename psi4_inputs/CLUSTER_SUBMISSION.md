# A5 geometry sensitivity — cluster submission instructions

Everything except the Psi4 execution is done and verified. `psi4` is not installed
on the local machine, so these two stages must run on the cluster.

## What is already done

- `a5a_optimize_manifest.csv` — 50 molecules, 10 from each of the five domains,
  seeded (`--seed 42`) and therefore reproducible.
- `a5b_conformer_manifest.csv` — 20 molecules, 4 per domain.
- All 250 geometries (50 + 20×10) were pre-flight built locally with ETKDGv3 + UFF:
  **250/250 succeeded**, 15–83 atoms. No embedding failures will surface on the cluster.
- Basis assignment resolved: 38 molecules at `6-31G(d,p)`, 12 at `def2-SVP`,
  following the same composition rule as `run_psi4_recompute_batch.py`.
- The `collect` stage was validated against synthetic results and correctly reports
  `GEOM_SHIFT`, `GEOM_SD`, `GEOM_MAX`, per-domain means, and `CONF_SPREAD`.

## Submit

```bash
cd ~/Downloads/Closser          # or wherever the repo lives on the cluster
mkdir -p logs

# A5a - 50 B3LYP optimisations, 10 array tasks of 5 molecules each
sbatch --array=0-9 run_geometry_sensitivity.sh optimize

# A5b - 200 B3LYP single points, one job
sbatch run_geometry_sensitivity.sh conformers

# once both finish
python3 run_geometry_sensitivity.py --stage collect
```

`collect` writes into `runs/rceg_final_consolidated/`:

| file | fills |
|---|---|
| `geometry_sensitivity.csv` | `[[GEOM_SHIFT]] [[GEOM_SD]] [[GEOM_MAX]]` |
| `geometry_sensitivity_by_domain.csv` | `[[GEOM_DOMAIN]]` |
| `conformer_ensemble_spread.csv` | per-conformer raw gaps |
| `conformer_ensemble_summary.csv` | `[[CONF_SPREAD]]` |

## Read this before submitting — scratch space

Task A3 audited the earlier reference recomputations and found that **30 of the 38
failures were `disk_or_scratch_exhausted`** — Psi4 raising *"No space left on device"*
inside `PSI_SCRATCH` — not chemical non-convergence. The failures were an
infrastructure problem.

`run_geometry_sensitivity.sh` therefore:

1. sets `PSI_SCRATCH` to a per-array-task directory under `/scratch/$USER/`, never
   letting it default to `/tmp`;
2. **refuses to start** if fewer than 20 GB are free there;
3. calls `psi4.core.clean()` after every molecule;
4. removes its scratch directory on exit via a shell trap.

Please do not strip those guards. If the pre-flight check fails, request more scratch
rather than lowering `MIN_KB`.

## Resumability

Both stages skip any molecule that already has a `.json` or `.failed.json` result, so a
requeued or timed-out array task resumes cleanly. Re-running `sbatch` after a partial
failure is safe.

## Acceptance (spec A5)

- A5a needs **≥ 45 of 50** converged
- A5b needs **≥ 18 of 20** molecules with converged conformers

Non-convergences are written to `*.failed.json` with the exception text and are counted
in the collected CSVs. They are reported, never silently dropped.

## One limitation worth knowing

The 50-molecule sample is stratified by chemical *domain*, as the spec specifies. That
yields only **1 open-shell molecule** (multiplicity 3); the `charged_or_radical` domain
is mostly charged closed-shell species. So this experiment will characterise geometry
sensitivity for closed-shell chemistry well, but will not settle whether open-shell
species are more geometry-sensitive. Given that the pseudo-label diagnosis found
open-shell species to be the single worst-behaved subset (PM7 vs B3LYP correlation
r = 0.074), a follow-up stratified on multiplicity rather than domain would be
worthwhile if reviewers press on it.
