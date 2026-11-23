# fig_transfer_ct1

Models trained on the mmWave generator with ladder ×1, tested without retraining on other networks and ladders: mean QoE $J_0$ and freeze time per session (log scale) over 100 test sessions per condition; error bars: s.d. over 5 training seeds; black tick: best always-one-tier rule on the same sessions (chosen on the test sessions, an upper reference). Generator ×1 is the training condition.

## Numbers (mean J0 | freeze s/session)

| condition | DARA-L (thr. 0.2) | DARA-L (thr. 0.15) | PPO-Lag. (0.5 s) | COREL | Pensieve | RobustMPC | Best fixed tier |
|---|---|---|---|---|---|---|---|
| Generator (training net.), x1 | 286.3 \| 35.1 | 287.5 \| 17.4 | 285.4 \| 33.7 | 260.0 \| 9.2 | 259.4 \| 11.0 | 223.2 \| 6.1 | 292.7 \| 38.2 |
| Lumos5G, x1 | 295.5 \| 3.4 | 294.5 \| 2.2 | 294.3 \| 3.3 | 280.3 \| 1.4 | 279.8 \| 1.7 | 269.1 \| 1.4 | 303.1 \| 3.6 |
| Irish 5G static, x1 | 196.0 \| 95.8 | 225.1 \| 37.3 | 197.9 \| 92.5 | 174.2 \| 20.9 | 194.7 \| 23.2 | 196.2 \| 16.7 | 204.9 \| 45.0 |
| Irish 5G driving, x1 | -729.0 \| 955.1 | -42.3 \| 252.3 | -677.2 \| 903.5 | 19.4 \| 118.7 | -6.6 \| 171.7 | 98.4 \| 45.1 | 48.3 \| 91.8 |
| Harsh mmWave, x1 | 218.0 \| 94.1 | 250.0 \| 34.7 | 218.8 \| 91.4 | 208.8 \| 18.6 | 215.2 \| 22.7 | 179.8 \| 15.0 | 227.8 \| 40.3 |
| Generator (training net.), x4 | 82.7 \| 221.1 | 184.7 \| 54.4 | 92.4 \| 205.7 | 127.4 \| 14.2 | 135.4 \| 26.8 | 147.1 \| 17.6 | 170.3 \| 44.4 |
| Lumos5G, x4 | 251.6 \| 41.5 | 237.3 \| 6.6 | 251.6 \| 37.1 | 159.3 \| 2.6 | 167.4 \| 3.8 | 223.2 \| 3.6 | 252.3 \| 50.9 |
| Irish 5G static, x4 | -293.5 \| 547.8 | 85.9 \| 112.0 | -252.8 \| 503.8 | 58.7 \| 40.1 | 63.9 \| 66.3 | 108.5 \| 26.2 | 99.2 \| 44.4 |
| Irish 5G driving, x4 | -3828.9 \| 3832.5 | -835.1 \| 954.6 | -3767.2 \| 3770.2 | -379.6 \| 447.3 | -627.4 \| 704.6 | -151.3 \| 233.1 | -170.4 \| 228.9 |
| Harsh mmWave, x4 | -251.2 \| 527.1 | 98.6 \| 117.5 | -219.0 \| 490.6 | 77.2 \| 36.0 | 75.7 \| 65.4 | 95.1 \| 29.2 | 117.0 \| 37.6 |

Paired Wilcoxon tests (DARA-L (thr. 0.15) against each method, sessions averaged over seeds, BH-corrected within J0 and within freeze): transfer_tests.csv
