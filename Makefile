# Convenience targets. Run `make help`.
PY ?= python
.PHONY: help install-analysis install summary test smoke sweeps analysis paper check-notebook

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-17s %s\n", $$1, $$2}'

install-analysis: ## level 1: dependencies to inspect the results (no PyTorch)
	$(PY) -m pip install -r requirements-analysis.txt

install:         ## levels 2-3: full dependencies (install PyTorch for your platform first)
	$(PY) -m pip install -r requirements.txt && $(PY) -m ipykernel install --user --name python3

summary:         ## level 1: benchmark tables from the shipped results -> results/BENCHMARK.md
	$(PY) scripts/analysis/benchmark_summary.py

test:            ## level 1: check the shipped results against the published numbers
	$(PY) -m pytest

check-notebook:  ## confirm the notebook is the one that produced the results
	@$(PY) -c "import hashlib;h=hashlib.md5(open('notebooks/dara_abr.ipynb','rb').read()).hexdigest();print(h, 'OK' if h=='9d323ac8b76368b3a28e6cc3b0407023' else 'DIFFERENT')"

sweeps:          ## write every job table to configs/sweeps/
	$(PY) scripts/sweeps/make_sweeps.py

smoke: sweeps    ## level 2: smoke-test the generator jobs on this machine (no datasets needed)
	bash scripts/hpc/run_local.sh configs/sweeps/smoke_train.tsv 1 6

analysis:        ## level 3: statistics, tables, figures and report from runs/
	bash scripts/analysis/run_all.sh
	$(PY) scripts/analysis/threshold_view.py --threshold 0.15

paper: analysis  ## level 3: paper figures, Table I and the benchmark summary
	$(PY) scripts/analysis/make_paper.py
	$(PY) scripts/analysis/benchmark_summary.py
