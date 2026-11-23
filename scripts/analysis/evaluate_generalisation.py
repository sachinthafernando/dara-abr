#!/usr/bin/env python3
"""Generalisation results: trained policies (primary environment) played unchanged on the test conditions.

Inputs : runs/generalize/gen_*.csv (bare policies; the reference job gives RobustMPC and always-one-tier)
Outputs: results/generalisation/{gen_table.csv, gen_tests.csv, gen_summary.csv, report.txt}

Comparisons: DARA-L against each comparator in the pre-registration (and any other trained method present), per
condition; Benjamini-Hochberg over the pre-registered comparisons (all conditions), separately over the rest. The training condition (seen/x4) is included
for completeness and marked; Table I reports it from the training runs.
Usage:  python scripts/analysis/evaluate_generalisation.py [--gen runs] [--prereg ...]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr.results import load_generalisation, matrix  # noqa: E402
from dara_abr.stats import compare, bh, verdict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "results" / "generalisation"))
ap.add_argument("--n-boot", type=int, default=10000)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
PR = json.load(open(a.prereg))
REF, ALPHA = PR["proposed_method"], PR["statistics"]["alpha"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
lines = []


def say(s=""):
    print(s); lines.append(str(s))


G = load_generalisation(a.gen)
if G.empty:
    raise SystemExit(f"no generalize/gen_*.csv under {a.gen}")
conds = [f"{n}/{c}" for n in R.NETWORKS for c in R.CONTENTS if f"{n}/{c}" in set(G.condition)]
G["network"] = G.condition.str.split("/").str[0]
pol, ref = G[G.run != "reference"], G[G.run == "reference"]
say(f"trained methods: {sorted(pol.train_method.unique())}  seeds: {sorted(pol.seed.dropna().astype(int).unique())}")
say(f"conditions: {conds}")

# ---------------------------------------------------------------------------------------------- table
rows = []
mpc = ref[ref.method == "robustmpc"].groupby("condition").J0.mean()
for (c, m), g in pol.groupby(["condition", "train_method"]):
    X = matrix(g)
    rows.append(dict(condition=c, method=m, n_seeds=X.shape[0], J0=X.mean(), J0_seed_sd=X.mean(1).std(ddof=1) if X.shape[0] > 1 else np.nan,
                     freeze_s=g.rebuffer.mean(), mean_tier=g.mean_tier.mean(), J0_minus_mpc=X.mean() - mpc.get(c, np.nan)))
for (c, m), g in ref.groupby(["condition", "method"]):
    rows.append(dict(condition=c, method=m, n_seeds=0, J0=g.J0.mean(), freeze_s=g.rebuffer.mean(),
                     mean_tier=g.mean_tier.mean(), J0_minus_mpc=g.J0.mean() - mpc.get(c, np.nan)))
T = pd.DataFrame(rows)
T["network"] = T.condition.str.split("/").str[0]
T["family"] = T.network.map(lambda n: R.NETWORKS.get(n, {}).get("family", ""))
T["training_condition"] = T.condition == R.TRAINING_CONDITION
T.to_csv(OUT / "gen_table.csv", index=False)
W = T.pivot_table(index="method", columns="condition", values="J0").reindex(columns=conds)
say("\n== mean J0 per test condition (trained policies: mean over seeds; reference policies: deterministic) ==")
say(W.round(1).to_string())

# ---------------------------------------------------------------------------------------------- paired tests
wanted = PR["comparisons"]["generalisation"]["against"]
others = [m for m in wanted if m in set(pol.train_method) or m == "robustmpc"]
others += [m for m in sorted(pol.train_method.unique()) if m not in others and m != REF]
tests = []
for c in conds:
    x = pol[(pol.train_method == REF) & (pol.condition == c)]
    if x.empty:
        continue
    A = matrix(x)
    for m in others:
        y = ref[(ref.method == m) & (ref.condition == c)] if m == "robustmpc" else pol[(pol.train_method == m) & (pol.condition == c)]
        if y.empty:
            continue
        tests.append(dict(condition=c, comparison=m, prereg=m in wanted, **compare(A, matrix(y), n_boot=a.n_boot)))
P = pd.DataFrame(tests)
if P.empty:
    raise SystemExit(f"{REF} has no generalisation rows")
P["network"] = P.condition.str.split("/").str[0]
P["family"] = P.network.map(lambda n: R.NETWORKS.get(n, {}).get("family", ""))
P["training_condition"] = P.condition == R.TRAINING_CONDITION
P["q_session"] = np.nan                      # BH families: pre-registered comparators, and everything else
for fam, idx in P.groupby("prereg").groups.items():
    P.loc[idx, "q_session"] = bh(P.loc[idx, "p_session"].values)
P["verdict"] = [verdict(d, q, lo, hi, ALPHA, "DARA-L") for d, q, lo, hi in zip(P["diff"], P.q_session, P.ci_lo, P.ci_hi)]
P["mark"] = P.verdict.map(lambda v: "+" if v.endswith("better") else "-" if v.endswith("worse") else "=" if v.startswith("no") else "?")
P.to_csv(OUT / "gen_tests.csv", index=False)
say(f"\n== {REF} minus each alternative per condition (+ better, - worse, = no significant difference, "
    "? tests disagree; BH over the pre-registered comparators, separately over the others) ==")
tab = P.assign(cell=P["diff"].round(1).astype(str) + P.mark).pivot(index="condition", columns="comparison", values="cell")
say(tab.reindex(index=conds, columns=others).to_string())

# ---------------------------------------------------------------------------------------------- summary by network family
def agg(df):
    return pd.Series(dict(mean_diff=df["diff"].mean(), better=int((df.mark == "+").sum()), worse=int((df.mark == "-").sum()),
                          no_difference=int((df.mark == "=").sum()), disagree=int((df.mark == "?").sum()), n=len(df)))


U = P[~P.training_condition]
parts = {"all unseen": U}
for fam in dict.fromkeys(U.family):
    parts[R.FAMILIES.get(fam, fam)] = U[U.family == fam]
Sm = pd.concat({k: v.groupby("comparison")[["diff", "mark"]].apply(agg) for k, v in parts.items() if len(v)},
               names=["conditions"]).reset_index()
Sm.to_csv(OUT / "gen_summary.csv", index=False)
say("\n== summary over unseen conditions ==")
say(Sm.round(1).to_string(index=False))
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
print(f"\nwritten to {OUT}")
