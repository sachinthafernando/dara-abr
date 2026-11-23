# Changelog

## [1.0.0] - 2026-11-23

First public release: the code, experiment definitions and results of the thesis and the DARA conference paper.

### Code (unchanged from the version that produced the results)
- `notebooks/dara_abr.ipynb` (md5 `9d323ac8b76368b3a28e6cc3b0407023`), `src/dara_abr/`, `scripts/data/`,
  `scripts/hpc/`, `scripts/sweeps/`, `scripts/analysis/` (all except `benchmark_summary.py`), `configs/preregistration/`,
  `tests/make_fixtures.py`, `tests/test_holdout.py`.

### Added
- `README.md` rewritten for public use: three reproduction levels, benchmark table, dataset links and licences.
  The previous README is kept as `docs/PIPELINE.md`.
- `scripts/analysis/benchmark_summary.py`: benchmark tables from the shipped results (no training, no GPU),
  including a recomputation of the transfer tests from per-session scores; writes `results/BENCHMARK.md`.
- `results/sessions/`: per-session scores of the transfer (41,000 sessions) and per-scenario (28,600 sessions)
  experiments, previously loose CSVs in the repository root (repeated header rows removed).
- `tests/test_results.py` (pytest) checking the shipped results against the published numbers; `pytest.ini`.
- `requirements-analysis.txt`, `environment.yml`, `Makefile`, `CITATION.cff`, `LICENSE` (pending), CI workflow,
  `docs/REPRODUCING.md`, `docs/RELEASE_NOTES_v1.0.0.md`, `docs/PUBLISHING.md`, `data/raw/*/.gitkeep`, `runs/` and `logs/` placeholders.

### Changed
- `configs/sweeps/*.tsv` regenerated with `make_sweeps.py` without `--skip-done`, so they list every job of the study
  (the cluster copies had finished jobs removed).
- `.gitignore`: `results/`, `figures/` and `overview/` are now tracked.

### Removed
- `__pycache__/`, the intermediate `figures/paper/_work/` and the empty `figures/paper_t01/`.
- `data/processed/*.npz` (derived from third-party datasets; rebuilt by `scripts/data/prepare_traces.py`).
