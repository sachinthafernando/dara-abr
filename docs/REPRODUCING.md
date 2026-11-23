# Where each result comes from

This page maps the results of the thesis and the paper to the shipped file that holds the numbers, and to the
command that rebuilds that file from `runs/`. All values are at act threshold τ = 0.15 (`views/t015`). "Rebuild"
assumes the jobs of the README, Level 3, have finished. In the code and in the generated files, DARA is
`dara_online_lag` / "DARA-L".

## Paper

| Result | Shipped file | Rebuild |
|---|---|---|
| Fig. 1, overview | `figures/paper/fig_overview.*` | `make_paper.py` |
| Fig. 2, *M* vs. shortfall of plain PPO | `figures/paper/fig_diagnosis.*`; `results/views/t015/training/diagnosis_validity*.csv` | `threshold_view.py --threshold 0.15`, `make_paper.py` |
| Table I, training environments | `figures/paper/table1.{tex,md,csv}` | `make_paper.py` (`make_table1.py`) |
| Fig. 3, *J*₀ and freeze per environment | `figures/paper/fig_qoe_components.*`; `results/views/t015/training/training_table.csv` | `make_paper.py` |
| Fig. 4, *J*₀ vs. freeze budget | `figures/paper/fig_budget_sweep.*`; `results/views/t015/training/budget_test.csv` | `make_paper.py` |
| Fig. 5, transfer of ×1 models | `figures/paper/fig_transfer_ct1.*`, `transfer_table.csv`, `transfer_tests.csv`; sessions in `results/sessions/transfer_ct1_sessions.csv.gz` | sweep `generalise_ct1.tsv`, then `plot_transfer.py --gen runs/generalise_ct1 --env ct1` |
| Fig. 6, per-scenario bitrate vs. freeze | `figures/paper/fig_scenarios_*.*`, `scenario_table.csv`, `scenario_tests.csv`; sessions in `results/sessions/scenarios_ct1_sessions.csv.gz` | sweep `scenarios_ct1.tsv`, then `plot_scenarios.py --gen runs/scenarios_ct1 --env ct1` |
| Fig. 7, quality switching | `figures/paper/fig_switching.*`, `switching_table.csv` | `plot_switching.py` (via `make_paper.py`) |
| Fig. 8, one test session | `figures/paper/fig_session_ct1_ep25.*` | notebook `MODE timeline` (see `docs/PIPELINE.md`), `plot_session.py` |

## Thesis

| Result | Shipped file | Rebuild |
|---|---|---|
| Table 3.1, network statistics; Table 3.2, tier feasibility | `results/data/network_stats.csv`, `results/data/ladder_feasibility.csv` | `scripts/data/analyse_traces.py` |
| Fig. 3.2, 3.7 (environment, actor-critic) | diagrams, no data | – |
| Table 4.1, hyperparameters | notebook parameter cell; `configs/sweeps/*.tsv` | – |
| Table 4.2, training environments | `src/dara_abr/registry.py` (`TRAINING_ENVS`) | – |
| Table 5.2, generator ×4: *M* and plain PPO per reward variant | `results/views/t015/training/training_table.csv` (methods `fixed`, `cap*`, `static_best`, `ppo_ref_reward`), `diagnosis_validity.csv` | `threshold_view.py --threshold 0.15` |
| Fig. 5.6, *M* vs. shortfall | as paper Fig. 2 | as paper Fig. 2 |
| Table 5.3, all training environments | `figures/paper/table1.csv` (6 environments); `overview/views/t015/summary.csv` (all 8); `results/BENCHMARK.md` | `make_paper.py --exclude-envs ""`; `benchmark_summary.py` |
| Fig. 5.7, budget sweep | as paper Fig. 4 | as paper Fig. 4 |
| Fig. 5.9, *J*₀ and freeze per environment | as paper Fig. 3 | as paper Fig. 3 |
| Fig. 5.10, Table 5.4, transfer of ×1 models | as paper Fig. 5 | as paper Fig. 5 |
| Table 5.5, ×4 models on unseen networks | `results/generalisation/gen_table.csv`, `gen_tests.csv` | sweep `generalise.tsv`, then `evaluate_generalisation.py` |
| Fig. 5.11, per-scenario groups | as paper Fig. 6 | as paper Fig. 6 |
| Fig. 5.12, quality switching | as paper Fig. 7 | as paper Fig. 7 |
| Fig. 5.13, one test session | as paper Fig. 8 | as paper Fig. 8 |

### Not in this release
- **Thesis Fig. 5.5** (training reward vs. *J*₀ of the constant tiers) and **Fig. 5.8** (Lagrange multiplier during
  training) were drawn with a thesis-only plotting script, which is not in v1.0.0.
  - Fig. 5.5 is drawn from `runs/landscape/`.
  - Fig. 5.8 is drawn from the λ logs in `runs/train/<run>/`.
  - The final λ values are in `training_table.csv` (`final_lambda`).
- **Thesis Fig. 3.3–3.5 and the print versions of the network figures** were drawn with separate scripts.
  - The underlying statistics are in `results/data/`.
  - `make_figures.py` draws the network figure in `figures/fig_network.*`.
- **Pipeline-validation experiments** (thesis E00–E03: Figs. 4.1–5.4, Table 5.1) were run with an earlier version of
  the code, before the DARA study. They are not part of this release.

## Checking a reproduction
1. Before you start: `md5sum notebooks/dara_abr.ipynb` should print `9d323ac8b76368b3a28e6cc3b0407023`.
2. After your runs and the analysis commands: `git diff --stat results/ figures/ overview/` lists every file that
   changed.
3. GPU training is not bit-for-bit deterministic. Compare your seed means with the shipped `J0 ± J0_seed_sd`, not the
   exact digits.
4. `pytest` checks the shipped values. If it fails after a re-run, your results differ from the published ones;
   inspect the diff to see by how much.
