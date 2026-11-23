# Suggested captions

> Note: DARA's act threshold is lowered from 0.2 to 0.15 after the results were seen. Environments with 0.15 <= M < 0.2 use the dara_online_lag_m015 runs as DARA-L.

**fig_qoe_bars.** Normalised QoE (mean $J_0$ divided by the best mean $J_0$ of any method or fixed rule run in that environment) in each training environment, ordered by misalignment $M$; error bars: s.d. over 5 training seeds; 100 shared test sessions. DARA-L is within 5% of the best in 5 of 8 environments (×1, Lumos5G ×4, ×2, Irish st. ×1, ×4).

**fig_qoe_components.** QoE components per training environment: (a) mean quality level (0 = lowest, 4 = highest tier) and (b) freeze time per session (log scale). DARA-L freezes less than PPO-Lagrangian (0.5 s) in 7 of 8 environments.

**fig_qoe_cdf.** CDF of per-session QoE over the 100 test sessions (each learned method averaged over its seeds; x-axis starts at the 1.5th percentile) in (a) ×4, (b) ×8, (c) Lumos5G ×4.

**fig_diagnosis.** Misalignment $M$, measured before training, against the shortfall of plain PPO after training (5 seeds; bars: s.d.), for every reward setting (ladder × freeze-penalty cap). Spearman $\rho$ = 0.87 (95% bootstrap interval 0.78–0.92, 22 settings).

**fig_budget_sweep.** QoE against the freeze budget: PPO-Lagrangian with hand-picked budgets (bars: s.d. over seeds) and DARA-L at its derived budget (star). DARA-L is within 5% of the best hand-picked budget in 4 of 6 environments where it acted (×1, ×2, ×4, Irish st. ×1).

**fig_training.** Training dynamics: QoE on training sessions (top) and Lagrange multiplier $\lambda$ (bottom), mean ± s.d. over 5 seeds; shaded band: episodes at which DARA activated the constraint (range over seeds). PPO (original reward) has no constraint.

**fig_unseen.** Normalised QoE on unseen mmWave conditions (synthetic generator with unseen chunk-size ladders; real Lumos5G traces) for models trained on the generator with ladder ×4; error bars: s.d. over seeds. All conditions, including the non-mmWave Irish 5G traces, are reported in Table II.

**fig_tradeoff.** Quality–freeze trade-off in the primary training environment (×4): mean over the 100 test sessions per seed (small markers) and over seeds (large markers, bars: s.d.); dotted: always the same tier. Up and to the left is better.
