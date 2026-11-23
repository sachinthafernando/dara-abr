#!/usr/bin/env python3
"""Benchmark summary from the result files shipped with the repository (no training, no GPU, no datasets).

Reads
  results/views/t015/training/*.csv          training-environment results (act threshold 0.15, as in the paper/thesis)
  results/sessions/transfer_ct1_sessions.csv.gz   per-session scores of the ladder-x1 models on unseen networks
and writes
  results/BENCHMARK.md                       the tables quoted in the README
  results/benchmark/*.csv                    the same tables as CSV

The transfer comparison is recomputed here from the per-session scores (seed-averaged per session, paired Wilcoxon
signed-rank test against DARA, Benjamini-Hochberg within each test condition), so it also serves as a worked example
of the evaluation protocol in configs/preregistration/paper_v1.json.

Usage:  python scripts/analysis/benchmark_summary.py [--view results/views/t015] [--out results]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--view", default="results/views/t015", help="folder with training/ results (default: threshold 0.15)")
ap.add_argument("--sessions", default="results/sessions/transfer_ct1_sessions.csv.gz")
ap.add_argument("--out", default="results")
a = ap.parse_args()

TRAIN = ROOT / a.view / "training"
OUT = ROOT / a.out
(OUT / "benchmark").mkdir(parents=True, exist_ok=True)

# Paper / thesis names. In code and in the analysis outputs DARA is "dara_online_lag" / "DARA-L".
NAME = {"dara_online_lag": "DARA", "dara_online_lag_m015": "DARA", "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)",
        "corel_official": "COREL", "pensieve_official": "Pensieve", "robustmpc": "RobustMPC",
        "fixed": "PPO (original reward)", "ppo_ref_reward": "PPO on J0 terms"}
ENV = {"ct1": ("mmWave generator", "×1"), "ctl": ("Lumos5G (real)", "×4"), "ct2": ("mmWave generator", "×2"),
       "cts1": ("Irish 5G static", "×1"), "ct4": ("mmWave generator", "×4"), "ct8": ("mmWave generator", "×8"),
       "cth4": ("Harsh mmWave", "×4"), "ctd1": ("Irish 5G driving", "×1")}


def pm(m, s):
    if pd.isna(m):
        return "–"
    return f"{m:.1f}" if pd.isna(s) else f"{m:.1f} ± {s:.1f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


# ------------------------------------------------------------------ 1. training environments
T = pd.read_csv(TRAIN / "training_table.csv")
B = pd.read_csv(TRAIN / "budget_test.csv").set_index("env")
DV = pd.read_csv(TRAIN / "diagnosis_validity.csv")
# misalignment of the original reward, measured before training (one value per environment; from the DARA run logs,
# collected in the overview summary). Falls back to the diagnosis-validity table for the generator ladders.
SUM = ROOT / a.view.replace("results/", "overview/", 1) / "summary.csv"
M = pd.read_csv(SUM).set_index("env").M if SUM.exists() else DV[DV.method == "fixed"].set_index("env").M
envs = [e for e in ENV if e in set(T.env)]

rows, long = [], []
for e in envs:
    t = T[T.env == e].set_index("method")
    g = lambda m, c="J0": t[c].get(m, np.nan)
    hand = t[t.index.str.startswith("ppo_lagrangian_t")]
    best_hand = hand.J0.idxmax() if len(hand) else None
    tiers = t[t.index.str.match(r"^tier\d$")]
    best_tier = tiers.J0.idxmax() if len(tiers) else None
    mval = B.loc[e].get("derived_budget", np.nan) if e in B.index else np.nan
    row = {
        "Training network": ENV[e][0], "Ladder": ENV[e][1],
        "M": "" if pd.isna(M.get(e)) else f"{M[e]:.2f}",
        "DARA budget (s)": "" if pd.isna(mval) else f"{mval:.2f}",
        "DARA": pm(g("dara_online_lag"), g("dara_online_lag", "J0_seed_sd")),
        "PPO-Lag. 0.5 s": pm(g("ppo_lagrangian_t0.5"), g("ppo_lagrangian_t0.5", "J0_seed_sd")),
        "Best hand budget": "–" if best_hand is None else
            f"{pm(hand.J0[best_hand], hand.J0_seed_sd[best_hand])} ({best_hand.split('_t')[1]} s)",
        "COREL": pm(g("corel_official"), g("corel_official", "J0_seed_sd")),
        "Pensieve": pm(g("pensieve_official"), g("pensieve_official", "J0_seed_sd")),
        "RobustMPC": pm(g("robustmpc"), np.nan),
        "PPO, original reward": pm(g("fixed"), g("fixed", "J0_seed_sd")),
        "Best fixed tier": "–" if best_tier is None else f"{tiers.J0[best_tier]:.1f} ({best_tier})",
    }
    rows.append(row)
    for m in ["dara_online_lag", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc", "fixed"]:
        if m in t.index:
            long.append(dict(env=e, method=NAME[m], J0=t.J0[m], J0_seed_sd=t.J0_seed_sd[m], freeze_s=t.freeze_s[m],
                             mean_tier=t.mean_tier[m], n_seeds=t.n_seeds[m]))
train_tab = pd.DataFrame(rows)
train_long = pd.DataFrame(long)
train_tab.to_csv(OUT / "benchmark" / "training_environments.csv", index=False)
train_long.to_csv(OUT / "benchmark" / "training_environments_long.csv", index=False)

freeze = train_long.pivot(index="env", columns="method", values="freeze_s").reindex(envs)
freeze = freeze[[c for c in ["DARA", "PPO-Lag. (0.5 s)", "COREL", "Pensieve", "RobustMPC", "PPO (original reward)"]
                 if c in freeze.columns]]
freeze_tab = freeze.round(1).astype(object).where(freeze.notna(), "–")
freeze_tab.insert(0, "Training network", [f"{ENV[e][0]} {ENV[e][1]}" for e in envs])
freeze_tab = freeze_tab.reset_index(drop=True)

# ------------------------------------------------------------------ 2. diagnosis validity
S = pd.read_csv(TRAIN / "diagnosis_validity_summary.csv").iloc[0]

# ------------------------------------------------------------------ 3. transfer (recomputed from sessions)
X = pd.read_csv(ROOT / a.sessions)
X = X[X.method == "policy"].copy()
X["m"] = X.run.str.split("__").str[1].str.replace(r"_s\d+$", "", regex=True)
X = X[X.m.isin(["dara_online_lag_m015", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"])]
NET = {"seen": "Generator (training network)", "lumos5g": "Lumos5G", "irish_static": "Irish 5G static",
       "irish_driving": "Irish 5G driving", "gen_harsh": "Harsh mmWave"}
per_sess = X.groupby(["network", "content", "m", "episode"]).agg(J0=("J0", "mean"), fr=("rebuffer", "mean")).reset_index()
per_seed = X.groupby(["network", "content", "m", "seed"]).J0.mean().groupby(["network", "content", "m"]).std()
REF = "dara_online_lag_m015"
trows, tests = [], []
for (net, lad), d in per_sess.groupby(["network", "content"], sort=False):
    w = d.pivot(index="episode", columns="m", values="J0")
    fr = d.groupby("m").fr.mean()
    r = {"Test network": NET.get(net, net), "Ladder": lad.replace("x", "×")}
    ps = []
    for m in ["dara_online_lag_m015", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]:
        r[NAME[m]] = f"{w[m].mean():.1f} ({fr[m]:.1f} s)"
        if m != REF:
            p = wilcoxon(w[REF], w[m]).pvalue
            ps.append(p)
            tests.append(dict(network=net, content=lad, comparison=NAME[m], diff=w[REF].mean() - w[m].mean(), p=p))
    trows.append(r)
tests = pd.DataFrame(tests)
# Benjamini-Hochberg within each test condition (family = the three comparisons against DARA)
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out
tests["q"] = tests.groupby(["network", "content"]).p.transform(bh)
tests["result"] = np.where(tests.q >= 0.05, "no significant difference",
                           np.where(tests["diff"] > 0, "DARA better", "DARA worse"))
tests.to_csv(OUT / "benchmark" / "transfer_tests_recomputed.csv", index=False)
transfer_tab = pd.DataFrame(trows)
transfer_tab.to_csv(OUT / "benchmark" / "transfer_ct1.csv", index=False)
wins = tests.result.value_counts()

# ------------------------------------------------------------------ write
md = f"""# Benchmark summary

Generated by `scripts/analysis/benchmark_summary.py` from the result files in this repository.
Score: mean QoE $J_0$ per session (higher is better) over the 100 shared test sessions (evaluation seed 99);
learned methods: mean ± s.d. over 5 training seeds (101, 202, 303, 404, 505). RobustMPC and fixed tiers are
deterministic. DARA act threshold τ = 0.15 (`{a.view}`).

## 1. Training environments (QoE $J_0$)

{md_table(train_tab)}

M: misalignment of the original training reward, measured before training (DARA acts when M ≥ 0.15).
Best hand budget: best of 1–4 PPO-Lagrangian budgets trained in that environment, chosen afterwards on the test
sessions (an oracle choice; each budget is a full training run). DARA derives its budget during training instead.
COREL and Pensieve were not trained on every environment (–).

### Freeze (rebuffering) time per session, seconds

{md_table(freeze_tab)}

## 2. Does the diagnosis predict training failure?

Spearman ρ between the misalignment M measured before training and the shortfall of plain PPO after training:
**ρ = {S.rho:.3f}** (95% bootstrap interval {S.ci_lo:.3f}–{S.ci_hi:.3f}, {int(S.n)} reward settings).
Settings flagged by DARA (M ≥ {S.threshold:g}) fell short of the best fixed tier by {100*S.mean_shortfall_flagged:.0f}% on
average; unflagged settings by {100*S.mean_shortfall_unflagged:.1f}%.

## 3. Transfer to unseen networks (models trained on generator ×1)

QoE $J_0$ per session, with mean freeze time per session in brackets. Recomputed from
`{a.sessions}` (seed-averaged per session).

{md_table(transfer_tab)}

Paired Wilcoxon signed-rank tests of DARA against each baseline over the 100 seed-averaged test sessions,
Benjamini–Hochberg q < 0.05 within each condition: DARA better in {wins.get('DARA better', 0)},
worse in {wins.get('DARA worse', 0)}, no significant difference in {wins.get('no significant difference', 0)} of
{len(tests)} comparisons (`results/benchmark/transfer_tests_recomputed.csv`). The paper's own tests, which also
require a hierarchical-bootstrap interval to agree, are in `figures/paper/transfer_tests.csv`.
"""
(OUT / "BENCHMARK.md").write_text(md, encoding="utf-8")
print(md)
