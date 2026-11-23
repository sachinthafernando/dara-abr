# Pipeline reference

The complete description of the experiment pipeline as used for the paper and the thesis (originally the
repository README). In this document **DARA-L** is the method called **DARA** in the paper and thesis.

Counterfactual reward diagnosis for learning-based adaptive bitrate (ABR) streaming on 5G mmWave.

DARA checks, before and during training, whether the training reward of an ABR agent prefers actions that go badly
under the intended QoE (misalignment *M*, dominance *D*), and if it does, switches on a freeze constraint whose
budget comes from the diagnosis itself (DARA-L, a PPO-Lagrangian). The repository contains the environment, the
diagnosis, the baselines (Pensieve, COREL, RobustMPC), the experiment definitions, and the scripts that turn the
runs into the paper's tables and figures.

## Layout

```
configs/
  preregistration/paper_v1.json  rules fixed before the runs: seeds, comparisons, tests, which conditions go in figures
  sweeps/*.tsv                   job tables (one row = one notebook execution), written by make_sweeps.py
data/
  raw/                           third-party datasets (not included; see data/README.md)
  processed/                     traces built from raw data; content_params.json (included); MANIFEST.json
notebooks/dara_abr.ipynb         environment, training, diagnosis, evaluation (MODE = run | landscape | generalize)
src/dara_abr/                    network generator, experiment registry, statistics, result loaders
scripts/
  data/                          inspect_datasets.py -> prepare_traces.py -> analyse_traces.py (+ calibrate_content.py)
  sweeps/make_sweeps.py          job tables from the registry
  hpc/                           run_sweep.slurm (SLURM array), run_local.sh, check_jobs.py, import_runs.py
  analysis/                      evaluate_training.py, evaluate_generalisation.py, make_tables.py, make_figures.py
runs/      train/<run>/, landscape/, generalize/        (outputs of the notebook)
logs/      slurm/, executed/ (executed notebooks)        (one per job)
results/   data/, training/, generalisation/, tables/   (analysis outputs)
figures/   fig_*.pdf / .png                             (paper figures)
overview/  index.html (full results report), README.md, charts and CSVs  (written by make_overview.py)
tests/make_fixtures.py           synthetic stand-ins for the raw data, to test the pipeline without the datasets
tests/test_holdout.py            checks that held-out traces are used only by the final test sessions
```

Run names are `<environment>__<method>_s<seed>`, e.g. `ct4__dara_online_lag_s101`. Environments, methods, seeds and
test conditions are defined once in `src/dara_abr/registry.py`; labels and colours in every table and figure come
from there too.

## Setup

```bash
python -m venv ~/venvs/dara && source ~/venvs/dara/bin/activate
pip install torch            # the build matching your CUDA version
pip install -r requirements.txt
python -m ipykernel install --user --name python3
```

On a cluster whose login shell is csh/tcsh, wrap each command in `bash -c '...'`. Batch jobs do not inherit an
activated environment reliably, so the SLURM script activates one itself: `$DARA_VENV` if it reaches the job, else a
`.venv` folder in the repository. Safest is to pass it on the command line,
`--export=ALL,TSV=$TSV,DARA_VENV=/absolute/path/to/venv`, or create the venv as `dara-abr/.venv`.

## Running everything (from the repository root)

**1. Data** (see `data/README.md`): place the datasets, then

```bash
python scripts/data/inspect_datasets.py
python scripts/data/prepare_traces.py
python scripts/data/analyse_traces.py
```

**2. Job tables**

```bash
python scripts/sweeps/make_sweeps.py            # all seeds; --skip-done leaves out finished runs
```

| Table | Jobs | What for |
|---|---|---|
| `smoke_train.tsv`, `smoke_generalise.tsv` | 8 + 3 | every code path in minutes; run first |
| `landscape.tsv` | 1 | *M*, *D* over freeze-penalty cap × ladder, no training (cap-effect figure) |
| `main.tsv` | 75 | primary environment `ct4`, 15 methods × 5 seeds: every figure and Table I |
| `other_envs.tsv` | 75 | DARA-L and hand-picked budgets on three more environments: budget table only |
| `ablations.tsv` | 10 | optional |
| `generalise.tsv` | 26 | after `main.tsv`: test networks × chunk-size ladders |

Optional: finished runs from an earlier output folder, made with the same code and parameters, can be linked in
instead of re-trained (`python scripts/hpc/import_runs.py --from <old folder> --dry-run`, then without `--dry-run`,
then `make_sweeps.py --skip-done`). A clean run of every table is the stronger reproducibility test.

**3. Submit** (one array task per row; add `--reservation=<name>` if you have one)

```bash
TSV=configs/sweeps/smoke_train.tsv
sbatch --array=1-$(($(wc -l < $TSV)-1)) --export=ALL,TSV=$TSV scripts/hpc/run_sweep.slurm
python scripts/hpc/check_jobs.py $TSV           # finished / failed jobs, and the --array list to re-run
```

Order: `smoke_train` → `smoke_generalise` → `landscape`, `main`, `other_envs` (in parallel) → `generalise`.
Without SLURM: `bash scripts/hpc/run_local.sh configs/sweeps/smoke_train.tsv`.

**4. Analysis** (`bash scripts/analysis/run_all.sh` runs all five)

```bash
python scripts/analysis/evaluate_training.py        # results/training/
python scripts/analysis/evaluate_generalisation.py  # results/generalisation/
python scripts/analysis/make_tables.py              # results/tables/*.tex, tables.md
python scripts/analysis/make_figures.py             # figures/
python scripts/analysis/make_overview.py            # overview/index.html: every result, explained, with all numbers
python scripts/analysis/make_paper_figures.py       # figures/paper/: IEEE-format result figures + captions.md
```

Sensitivity view with a lower act threshold (not the pre-registered analysis; marked as such in the report):
`python scripts/analysis/threshold_view.py --threshold 0.15` -> `overview/views/t015/index.html`. Environments with
0.15 <= M < 0.2 use the runs trained with that threshold (`dara_online_lag_m015`); nothing in `runs/` or `results/` changes.

Session time line (one test session, chunk by chunk; needs the runs' `checkpoint.pt`, no training, a minute on CPU):

```bash
papermill notebooks/dara_abr.ipynb logs/executed/timeline_ct1.ipynb --cwd . -p MODE timeline -p OUT_ROOT runs \
  -p TL_TAG ct1 -p TL_RUNS "ct1__dara_online_lag_m015_s303,ct1__ppo_lagrangian_t0.5_s505,ct1__corel_official_s505"
python scripts/analysis/plot_session.py --timeline runs/timeline/timeline_ct1.csv     # figures/paper/fig_session_*
```
The replay is checked against each run's `gt_per_episode.csv` (`timeline_<tag>_check.csv`); add `robustmpc` to
`TL_RUNS` for RobustMPC on the same sessions, and `--episode N` to choose the session.

Transfer of the ladder-×1 models (exploratory, no training): `configs/sweeps/generalise_ct1.tsv` plays the ct1
checkpoints on the generator, Lumos5G, Irish 5G static and driving and the harsh mmWave scenarios, ladders ×1 and ×4;
then `python scripts/analysis/plot_transfer.py --gen runs/generalise_ct1 --env ct1` ->
`figures/transfer_ct1/fig_transfer_ct1.pdf`, `transfer_table.csv`, `transfer_tests.csv`, `caption.md`.

Per-scenario comparison (exploratory, no training, like COREL's per-scenario figure): `configs/sweeps/scenarios_ct1.tsv`
plays the ct1 checkpoints on each of the eight generator scenarios and on walking / driving / dead-zone groups, ladder ×1;
then `python scripts/analysis/plot_scenarios.py --gen runs/scenarios_ct1 --env ct1` -> `figures/scenarios_ct1/`
(`fig_scenarios_groups`, `fig_scenarios_all`, `scenario_table.csv`, `scenario_tests.csv`, `caption.md`; 95% CIs from a
hierarchical bootstrap over seeds and sessions). Generalisation CSVs now also record each session's scenario, mean
quality level and mean bitrate.

### All paper figures in one command

```bash
python scripts/analysis/make_paper.py            # -> figures/paper/: Fig. 1-8, table1.tex/.md/.csv, captions.md
```
Defaults: DARA-L at act threshold 0.15 (the threshold view is built first if missing), ct8 and ctd1 left out
(`--exclude-envs ""` keeps them), no best-fixed-tier reference (`--with-best-tier` adds it). Inputs: the threshold view,
`runs/generalise_ct1`, `runs/scenarios_ct1` and `runs/timeline/timeline_ct1_all.csv[.gz]`; a figure whose input is
missing is skipped and reported.

| Output | Script | Data |
|---|---|---|
| Fig. network | make_figures.py | generator + `data/processed/*.npz` |
| Fig. cap effect | make_figures.py | `runs/landscape`, `results/training` (ct4) |
| Fig. tier share | make_figures.py | `runs/train` (ct4) |
| Fig. budget | make_figures.py | `results/training` (ct4) |
| Fig. trade-off | make_figures.py | `runs/generalize` (training condition) |
| Fig. generalisation | make_figures.py | `results/generalisation` (mmWave conditions) |
| Table I (training environment) | make_tables.py | `results/training` |
| Table II (all test conditions) | make_tables.py | `results/generalisation` |
| Table III (budget rule) | make_tables.py | `results/training` (all environments) |

## Second batch (configs/preregistration/paper_v2.json)

| Table | Jobs (after `--skip-done`) | What for |
|---|---|---|
| `smoke_v2.tsv` | 5 | held-out traces, threshold 0.15, ladder ×8, COREL on traces; run first |
| `precheck.tsv` | 2 | M before training on Irish driving and Lumos5G (no training) |
| `set1.tsv` | 150 | 11 methods × 5 seeds on ladders ×1, ×2, ×4, ×8 (+ DARA-L with threshold 0.15 on ×1) |
| `set2.tsv` | 50 | DARA-L, PPO-Lagrangian 0.5 s, COREL, Pensieve on harsh mmWave, Irish driving, Lumos5G |

Irish driving (`ctd1`) and Lumos5G (`ctl`) hold out traces for testing (`TRACE_HOLDOUT_FRAC`, 25% and 20%,
stratified by name prefix, seed 7; the list is written to each run's `holdout_traces.json`). Training, DARA's
diagnosis and learning gate, and COREL/Pensieve validation use only the training traces; only the final 100 test
sessions use the held-out ones (`python tests/test_holdout.py` checks this). Set 3 (the ×4 models on unseen
networks) is the first batch's `generalise.tsv`.

## Evaluation protocol (summary of `configs/preregistration/paper_v1.json`)

* 5 training seeds (101, 202, 303, 404, 505) per trained method; 100 test sessions (seed 99) shared by every method.
  Training, validation (Pensieve/COREL checkpoint choice, DARA's learning gate) and test sessions come from separate
  random streams.
* Score *J*₀ per session: Σ q(tier) − 1.075·freeze(s) − |Δq|, with q = (1.00, 1.92, 2.61, 3.30, 4.00).
* Each comparison is reported three ways: Wilcoxon signed-rank on seed-averaged sessions (Benjamini–Hochberg within
  a family), Welch t-test on seed means, and a hierarchical bootstrap interval (seeds and sessions resampled).
  "Better"/"worse" needs both the session test and the interval.
* Figures use the primary training environment only. The generalisation figure shows mmWave networks (training
  generator with unseen ladders, Lumos5G); every condition, including both Irish 5G sets, is in Table II.

## Adding a sweep category

Add an entry to `GROUPS` (and, if needed, `TRAINING_ENVS` or `METHODS`) in `src/dara_abr/registry.py`, then extend
`make_sweeps.py` with one `write(...)` line. Ladder- or dataset-specific categories (e.g. training on ×1 or ×8, or on
Lumos5G traces) are new `TRAINING_ENVS` entries; nothing else changes.
