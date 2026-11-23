#!/bin/bash
#SBATCH --job-name=dara
#SBATCH --partition=core64
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
# One array task = one row of a job table in configs/sweeps/. Submit from the repository root, e.g.
#   sbatch --array=1-$(($(wc -l < configs/sweeps/main.tsv)-1)) --export=ALL,TSV=configs/sweeps/main.tsv scripts/hpc/run_sweep.slurm
# (add --reservation=<name> if you have one). Set DARA_VENV to your virtual environment, or activate it before sbatch.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
# Python environment: $DARA_VENV if set, else a .venv folder in the repository, else whatever is on PATH.
# Batch jobs do not read your interactive shell setup, so pass it explicitly:  --export=ALL,TSV=...,DARA_VENV=/path/to/venv
if [ -n "${DARA_VENV:-}" ]; then
  [ -f "$DARA_VENV/bin/activate" ] || { echo "DARA_VENV=$DARA_VENV has no bin/activate"; exit 2; }
  source "$DARA_VENV/bin/activate"
elif [ -f "$REPO/.venv/bin/activate" ]; then
  source "$REPO/.venv/bin/activate"
fi
python -c "import papermill, torch" 2>/dev/null || {
  echo "python at $(command -v python) cannot import papermill/torch: set DARA_VENV (see the header of this script)"; exit 2; }
echo "python: $(command -v python)"
TSV=${TSV:?set TSV=configs/sweeps/<table>.tsv}
mapfile -d '' ARGS < <(python scripts/hpc/row_args.py "$TSV" "$SLURM_ARRAY_TASK_ID")
RUN=${ARGS[0]}; ARGS=("${ARGS[@]:1}")
mkdir -p logs/executed
echo "$(date '+%F %T') task ${SLURM_ARRAY_TASK_ID} of ${TSV}: ${RUN} on $(hostname)"
papermill notebooks/dara_abr.ipynb "logs/executed/${RUN}.ipynb" --cwd "$REPO" "${ARGS[@]}"
echo "$(date '+%F %T') done: ${RUN}"
