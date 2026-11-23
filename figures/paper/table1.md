# Table I

| Training network | Ladder | M | Acts | Budget (s/chunk) | PPO, original reward | DARA-L | Best hand-picked PPO-Lag. (budget) | RobustMPC |
|---|---|---|---|---|---|---|---|---|
| mmWave generator | ×1 | 0.16 | 5/5 | 0.16 | 284.9 ± 7.6 | 287.5 ± 5.1 | **290.5 ± 5.9 (0.25)** | 223.2 |
| Lumos5G (real) | ×4 | 0.23 | 5/5 | 0.14 | – | **253.8 ± 2.5** | 232.4 ± 1.4 (0.5)† | 219.5 |
| mmWave generator | ×2 | 0.33 | 5/5 | 0.23 | 220.5 ± 2.8 | 238.1 ± 8.8 | **247.9 ± 4.1 (0.25)** | 186.7 |
| Irish 5G static | ×1 | 0.37 | 5/5 | 0.33 | – | 220.9 ± 6.3 | **226.9 ± 4.5 (0.5)** | 196.2 |
| mmWave generator | ×4 | 0.57 | 5/5 | 0.38 | 65.9 ± 9.3 | 188.7 ± 4.6 | **192.1 ± 3.6 (0.5)** | 147.1 |
| Harsh mmWave | ×4 | 0.78 | 5/5 | 0.45 | – | 99.8 ± 11.2 | **136.9 ± 5.2 (1)** | 95.1 |

Mean J_0 per session over 100 shared test sessions; learned methods: mean ± s.d. over 5 training seeds. M: misalignment of the training reward, measured before training. DARA switched the freeze constraint on in 5/5 seeds in every environment, during training (between episodes 2000 and 5000). Best hand-picked: the best of the PPO-Lagrangian budgets trained in that environment, chosen afterwards on the test sessions (an oracle choice; 3–4 budgets, each a full training run). Bold: best learned method in the row. †: only one hand-picked budget (0.5 s) was trained there. –: not trained.
