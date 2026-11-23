# DARA v1.0.0

First public release of the code and results behind the MSc by Research thesis *Deep Reinforcement Learning for
Adaptive Bitrate Streaming in 5G mmWave Networks* (University of Hertfordshire, 2026) and the paper *DARA: Improving
RL-based adaptive bitrate streaming through reward diagnosis*.

## Highlights
- **DARA** diagnoses misaligned ABR training rewards before and during training. It uses counterfactual fixed-tier
  rollouts, with no policy needed, and fixes the problem with a PPO-Lagrangian freeze budget derived from the
  diagnosis.
- **Benchmark.** DARA is compared with PPO-Lagrangian (hand-picked budgets), COREL, Pensieve, RobustMPC and plain
  PPO on 8 training environments: the mmWave generator with ladders ×1–×8, harsh mmWave, Lumos5G, and Irish 5G
  static and driving. The comparisons use 5 seeds and 100 shared test sessions.
- **Results shipped.** Every result table, the paper figures, a self-contained HTML report and 69,600 per-session
  scores are included. Inspecting them needs no GPU and no datasets.
- **One-command checks.** `python scripts/analysis/benchmark_summary.py` and `pytest`.
- **Smoke tests.** Every code path runs in about a minute per job on a CPU. The release was checked with Python 3.13
  and PyTorch 2.15.

## Headline numbers (τ = 0.15)
- *M* measured before training predicts the shortfall of plain PPO: Spearman ρ = 0.87 (95% CI 0.78–0.92, 22 settings).
- On generator ×4, plain PPO scores *J*₀ = 65.9 with 244.5 s of freezing per session. DARA scores 188.7 with 25.6 s.
  The best hand-picked budget scores 192.1.
- DARA is within 1–4% of the best hand-picked budget on generator ×1, ×2, ×4 and Irish 5G static, and 9% above it on
  Lumos5G.

## Notes
- The code is byte-identical to the version that produced the results. In the code and the generated outputs, DARA
  is named `dara_online_lag` / "DARA-L".
- The datasets are not redistributed; see the README for links and licences.
- The licence is pending (see `LICENSE`).
