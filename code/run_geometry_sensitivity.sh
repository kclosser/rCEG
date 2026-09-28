#!/bin/bash
# run_geometry_sensitivity.sh   (spec task A5)
#
# Fresh driver for the geometry-sensitivity calculations. Deliberately NOT
# derived from HLGClaude.sh / HLGModel.sh, which drive the abandoned
# neural-network track.
#
# Submit as an array job so the 50 optimisations run in parallel chunks:
#
#     sbatch --array=0-9 run_geometry_sensitivity.sh optimize
#     sbatch            run_geometry_sensitivity.sh conformers
#     python3 run_geometry_sensitivity.py --stage collect
#
# Both psi4 stages are resumable: a molecule with an existing .json or
# .failed.json result is skipped, so a requeued array task picks up where it
# stopped.
#
# SCRATCH NOTE -- this is the important part.
# The audit in task A3 found that 30 of the 38 failed reference recomputations
# died with Psi4 "No space left on device" in PSI_SCRATCH, not for any chemical
# reason. This driver therefore (a) puts scratch on a per-task directory under
# /scratch rather than letting it default to /tmp, (b) checks free space before
# starting, and (c) clears scratch between molecules. Do not remove these.

#SBATCH --job-name="rceg_geom"
#SBATCH --output="logs/rceg_geom_%A_%a.out"
#SBATCH --error="logs/rceg_geom_%A_%a.err"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32000M
#SBATCH --account=csn100
#SBATCH --export=ALL
#SBATCH --constraint="lustre"
#SBATCH -t 24:00:00

set -euo pipefail

STAGE="${1:-optimize}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs

# ---- scratch, sized and isolated per task ----
export PSI_SCRATCH="/scratch/${USER}/rceg_geom_${JOB_ID}_${TASK_ID}"
mkdir -p "$PSI_SCRATCH"
trap 'rm -rf "$PSI_SCRATCH"' EXIT

AVAIL_KB=$(df -Pk "$PSI_SCRATCH" | awk 'NR==2 {print $4}')
MIN_KB=$((20 * 1024 * 1024))          # require 20 GB
echo "PSI_SCRATCH=$PSI_SCRATCH"
echo "scratch free: $((AVAIL_KB / 1024 / 1024)) GB"
if [ "$AVAIL_KB" -lt "$MIN_KB" ]; then
  echo "ERROR: fewer than 20 GB free in $PSI_SCRATCH." >&2
  echo "This is the exact condition that killed 30 of 38 earlier jobs." >&2
  exit 1
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8

echo "stage=$STAGE  array task $TASK_ID of $N_TASKS  host=$(hostname)"
echo "started $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

case "$STAGE" in
  optimize)
    # 50 molecules split across the array
    TOTAL=50
    CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
    START=$(( TASK_ID * CHUNK ))
    END=$(( START + CHUNK ))
    echo "handling manifest rows [$START, $END)"
    python3 -u run_geometry_sensitivity.py \
        --stage optimize \
        --start "$START" --end "$END" \
        --threads 8 --memory "24 GB" \
        --geom-maxiter 60
    ;;
  conformers)
    python3 -u run_geometry_sensitivity.py \
        --stage conformers \
        --threads 8 --memory "24 GB"
    ;;
  *)
    echo "unknown stage: $STAGE (expected optimize|conformers)" >&2
    exit 2
    ;;
esac

echo "finished $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "scratch high-water usage:"
du -sh "$PSI_SCRATCH" 2>/dev/null || true
