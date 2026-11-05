#!/bin/bash
#SBATCH --job-name=hdb-ablation
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-89
#SBATCH --output=logs/%x-%A_%a.out

# CPU-only: this project trains gradient-boosted trees, not neural nets.
# Each array task runs one ablation configuration and writes its own file under
# outputs/parts/, because tasks run concurrently and appending to one shared
# table loses rows. Run python src/run_ablation.py --merge afterwards.
#
# The array bound must match the registry, which python src/run_ablation.py
# --count prints. Submit with
#   sbatch --array=0-$(($(python src/run_ablation.py --count) - 1)) \
#          scripts/slurm_ablation.sbatch
# to take the bound from the registry rather than from the directive above. A
# task given an index the registry does not hold exits with an error instead of
# silently scoring the wrong config.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR" outputs logs

echo "job      $SLURM_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "host     $(hostname)"
git rev-parse HEAD 2>/dev/null || true

# Build outputs/base.parquet once on the login node with python src/data.py.
uv run --frozen python src/run_ablation.py --index "$SLURM_ARRAY_TASK_ID"
