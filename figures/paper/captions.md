# Paper figures (DARA-L at act threshold 0.15; left out: ct8,ctd1; no best-fixed-tier reference)

**Fig. 1 (fig_overview).** DARA and DARA-L. During PPO training, DARA periodically measures, from counterfactual rollouts that need no policy, how often the training reward's preferred tier is worse for the intended QoE $J_0$ than the intended-best tier (misalignment $M$). If $M \ge \tau$ ($\tau$ = 0.15), it switches on a freeze constraint whose budget $\hat b$ is the freeze per chunk of the intended-best tier.

**Fig. 2 (fig_diagnosis).** Misalignment $M$, measured before training, against the shortfall of plain PPO after training (5 seeds; bars: s.d.), for every reward setting (ladder × freeze-penalty cap). Spearman $\rho$ = 0.87 (95% bootstrap interval 0.66–0.89, 17 settings).

**Table I (table1.tex).** Training environments: misalignment before training, DARA's derived budget, and QoE $J_0$ after training (note under the table).

**Fig. 3 (fig_qoe_components).** Per training environment: (a) mean QoE $J_0$ per session over the 100 test sessions (error bars: s.d. over 5 training seeds; RobustMPC is deterministic) and (b) freeze time per session (log scale). DARA-L freezes less than PPO-Lagrangian (0.5 s) in 6 of 6 environments. COREL and Pensieve were not trained on Irish 5G static.

**Fig. 4 (fig_budget_sweep).** QoE against the freeze budget: PPO-Lagrangian with hand-picked budgets (bars: s.d. over seeds) and DARA-L at its derived budget (star). DARA-L is within 5% of the best hand-picked budget in 4 of 5 environments where it acted (×1, ×2, ×4, Irish st. ×1).

**Fig. 5 (fig_transfer_ct1).** Models trained on the mmWave generator with ladder ×1, tested without retraining on other networks and ladders: mean QoE $J_0$ and freeze time per session (log scale) over 100 test sessions per condition; error bars: s.d. over 5 training seeds. Generator ×1 is the training condition.

**Fig. 6 (fig_scenarios_groups).** Average bitrate and freeze time per session for each network scenario (models trained on the full generator mix, ladder ×1; 100 test sessions per scenario; freeze axis reversed, so the top right is better). Error bars: 95% confidence intervals (hierarchical bootstrap over 5 training seeds and the test sessions). Groups: walking = urban pedestrian, crowded stationary; driving = urban driving, motorway, suburban open; dead zone = tunnel, under bridge, urban canyon.

**Fig. 7 (fig_switching).** Quality changes over all replayed test sessions (100–500 sessions per method; 5 training seeds per learned method): (a) quality switches per session, by switch size in tiers; (b) share of chunks played at each tier; (c) switching term of $J_0$, $\sum|\Delta q|$ per session (bars: mean; dots: training seeds).

**Fig. 8 (fig_session_ct1_ep25).** Test session 25 of the shared 100 (77 chunks of 4 s, crowded stationary scenario): (a) network throughput; (b) DARA-L, (c) PPO-Lag. (0.5 s), (d) COREL, (e) Pensieve, (f) RobustMPC: quality level of each chunk over the session (drawn while the chunk downloads), stalls (red, with duration) and quality switches (from→to). Seeds and session chosen as the most typical by $J_0$, freeze and switching; every learned method stalls at least once in it.

Also written: fig_scenarios_all (the eight generator scenarios one by one), transfer/scenario/switching tables and significance tests (CSV).
