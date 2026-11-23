## Table I: training environment
| Method | J0 (mean ± seed s.d.) | Freeze (s) | ΔJ0 DARA-L − method [95% CI] | Sessions DARA-L better | q (session) | p (seed) | Verdict |
|---|---|---|---|---|---|---|---|
| DARA-L (ours) | 188.7 ± 4.6 | 25.6 | -- | -- | -- | -- | -- |
| COREL | 174.7 ± 22.4 | 14.4 | +14.0 [-3.8, +33.1] | 0.73 | <1e-4 | 0.24 | inconclusive (tests disagree) |
| Pensieve | 145.6 ± 19.5 | 15.1 | +43.1 [+27.2, +60.5] | 0.96 | <1e-4 | 0.007 | better |
| RobustMPC | 147.1 | 17.6 | +41.6 [+33.9, +49.6] | 0.87 | <1e-4 | -- | better |
| Best single tier (2) | 170.3 | 44.4 | +18.4 [+7.8, +29.6] | 0.55 | 0.13 | -- | inconclusive (tests disagree) |
| DARA, cap only | 158.1 ± 20.0 | 133.0 | +30.6 [+6.8, +63.9] | 0.49 | 0.37 | 0.025 | inconclusive (tests disagree) |
| Original reward (c = 0.1) | 65.9 ± 9.3 | 244.5 | +122.8 [+54.0, +208.3] | 0.52 | 0.008 | <1e-4 | better |
| PPO, cap c = ∞ | 174.7 ± 9.7 | 106.0 | +13.9 [-0.9, +33.7] | 0.46 | 0.98 | 0.029 | no significant difference |
| PPO-Lagrangian, 0.25 s | 164.5 ± 13.2 | 14.0 | +24.2 [+12.8, +36.7] | 0.88 | <1e-4 | 0.012 | better |
| PPO-Lagrangian, 0.5 s | 192.1 ± 3.6 | 38.1 | -3.4 [-9.3, +1.9] | 0.24 | <1e-4 | 0.23 | inconclusive (tests disagree) |
| PPO-Lagrangian, 1 s | 183.0 ± 9.1 | 92.1 | +5.7 [-7.3, +21.5] | 0.41 | 0.22 | 0.26 | no significant difference |
| PPO-Lagrangian, 2 s | 109.3 ± 17.3 | 190.9 | +79.4 [+35.8, +135.4] | 0.51 | 0.012 | 0.000 | better |

## Table II: generalisation (DARA-L minus method)
| Network | Ladder | COREL | Pensieve | RobustMPC | PPO-Lagrangian, 0.5 s |
|---|---|---|---|---|---|
| Training generator | ×1 | +15.6? | +58.4+ | +48.3+ | -3.8? |
| Training generator (training) | ×4 | +14.0? | +43.1+ | +41.6+ | -3.4? |
| Training generator | ×8 | +0.2= | +29.7+ | +25.7+ | +1.8? |
| Lumos5G | ×1 | -4.4? | +46.0+ | +8.6? | -4.6? |
| Lumos5G | ×4 | +5.1= | +49.1+ | +5.7? | -8.8? |
| Lumos5G | ×8 | +6.4= | +51.2+ | +3.8= | -3.9? |
| Irish 5G static | ×1 | +41.1+ | +63.0+ | +12.9+ | +5.1? |
| Irish 5G static | ×4 | +33.0+ | +33.4+ | +8.5+ | +6.7? |
| Irish 5G static | ×8 | +10.2? | +13.1? | -8.8? | +19.2? |
| Irish 5G driving | ×1 | -8.4? | -6.8= | -65.7- | +93.2+ |
| Irish 5G driving | ×4 | -167.2- | -142.7- | -214.5- | +596.1+ |
| Irish 5G driving | ×8 | -434.5- | -443.3- | -516.8- | +1300.3+ |

## Table III: budget rule
| Training environment | Derived budget (s/chunk) | DARA-L J_0 | Best hand budget | Its J_0 | Gap | Within 5% |
|---|---|---|---|---|---|---|
| mmWave generator, ladder ×1 | nan | 286.3 | 0.25 | 290.5 | -1.5% | n/a |
| mmWave generator, ladder ×2 | 0.231 | 238.1 | 0.25 | 247.9 | -4.0% | yes |
| mmWave generator, ladder ×4 | 0.384 | 188.7 | 0.5 | 192.1 | -1.8% | yes |
| mmWave generator, ladder ×8 | 0.513 | 100.2 | 1 | 129.8 | -22.9% | no |
| Irish 5G driving traces, ladder ×1 (4 of 16 traces held out) | 0.561 | 88.8 | 0.5 | 101.5 | -12.5% | no |
| Harsh mmWave scenarios, ladder ×4 | 0.453 | 99.8 | 1 | 136.9 | -27.1% | no |
| Lumos5G traces, ladder ×4 (20% of traces held out) | 0.144 | 253.8 | 0.5 | 232.4 | +9.2% | yes |
| Irish 5G static traces, ladder ×1 | 0.327 | 220.9 | 0.5 | 226.9 | -2.6% | yes |

## Table IV: every training environment (J0, freeze s; DARA-L [budget]; mark = DARA-L vs method)
| Training environment | DARA-L [budget] | PPO-Lag., 0.5 s | COREL | Pensieve | RobustMPC |
|---|---|---|---|---|---|
| mmWave ×1 | 286.3 (35) [--] | 285.4 (34) ? | 260.0 (9) + | 259.4 (11) + | 223.2 (6) + |
| mmWave ×2 | 238.1 (20) [0.23] | 247.7 (53) ? | 191.7 (6) + | 204.2 (14) + | 186.7 (11) + |
| mmWave ×4 | 188.7 (26) [0.38] | 192.1 (38) ? | 174.7 (14) ? | 145.6 (15) + | 147.1 (18) + |
| mmWave ×8 | 100.2 (18) [0.51] | 117.6 (32) ? | 122.5 (17) - | 98.2 (24) = | 104.9 (33) = |
| harsh mmWave ×4 | 99.8 (19) [0.45] | 129.8 (30) - | 110.8 (17) ? | 69.7 (14) + | 95.1 (29) = |
| Irish driving ×1 | 88.8 (32) [0.56] | 101.5 (32) ? | 69.9 (28) ? | 71.5 (29) + | 120.5 (33) - |
| Lumos5G ×4 | 253.8 (16) [0.14] | 232.4 (67) ? | 259.1 (16) - | 246.7 (23) = | 219.5 (4) + |
| Irish static ×1 | 220.9 (21) [0.33] | 226.9 (41) ? | -- | -- | 196.2 (17) + |
