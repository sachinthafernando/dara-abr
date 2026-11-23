#!/bin/bash
# Every analysis step, in order, with the default paths (runs/ -> results/ -> figures/). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/analysis/evaluate_training.py
python scripts/analysis/evaluate_generalisation.py
python scripts/analysis/make_tables.py
python scripts/analysis/make_figures.py
python scripts/analysis/make_overview.py
python scripts/analysis/make_paper_figures.py
