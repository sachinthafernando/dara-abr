#!/bin/bash
# Bundles what is needed to check the analysis elsewhere: results/, figures/, configs/, the small per-run files
# (summaries, per-session scores, training and DARA logs, configs) and the landscape/generalisation tables.
# Checkpoints, counterfactual arrays and executed notebooks are left out (large). Output: dara_results_<date>.tgz
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="dara_results_$(date +%Y%m%d_%H%M).tgz"
LIST=$(mktemp)
find runs -type f \( -name triage_summary.csv -o -name gt_per_episode.csv -o -name ppo_log.csv -o -name dara_log.csv \
     -o -name config.json -o -name diag_per_state.csv -o -name 'gen_*.csv' -o -name 'landscape_*.csv' -o -name 'rt_budget*.json' \) \
     -not -path '*/smoke/*' > "$LIST"
find results figures configs -type f >> "$LIST"
ls data/processed/MANIFEST.json >> "$LIST" 2>/dev/null || true
tar czf "$OUT" -T "$LIST"
rm -f "$LIST"
echo "written $OUT ($(du -h "$OUT" | cut -f1))"
