#!/bin/bash
# Runs jobs of a table on this machine, one after another (smoke tests, or a machine without SLURM).
#   bash scripts/hpc/run_local.sh configs/sweeps/smoke_train.tsv          # every job
#   bash scripts/hpc/run_local.sh configs/sweeps/smoke_train.tsv 2 4      # jobs 2..4
set -uo pipefail
cd "$(dirname "$0")/../.."
if [ -n "${DARA_VENV:-}" ]; then source "$DARA_VENV/bin/activate"; elif [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
python -c "import papermill, torch" 2>/dev/null || { echo "python cannot import papermill/torch: activate your venv or set DARA_VENV"; exit 2; }
TSV=${1:?usage: run_local.sh <table.tsv> [first] [last]}
N=$(($(wc -l < "$TSV") - 1)); FIRST=${2:-1}; LAST=${3:-$N}
mkdir -p logs/executed logs/local
FAIL=0
for i in $(seq "$FIRST" "$LAST"); do
  mapfile -d '' ARGS < <(python scripts/hpc/row_args.py "$TSV" "$i")
  RUN=${ARGS[0]}; ARGS=("${ARGS[@]:1}")
  echo "$(date '+%T') [$i/$N] $RUN"
  if papermill notebooks/dara_abr.ipynb "logs/executed/${RUN}.ipynb" --cwd "$(pwd)" "${ARGS[@]}" > "logs/local/${RUN}.log" 2>&1; then
    echo "           ok"
  else
    echo "           FAILED (see logs/local/${RUN}.log)"; FAIL=$((FAIL + 1))
  fi
done
echo "finished: $FAIL failed"
exit $FAIL
