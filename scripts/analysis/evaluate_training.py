#!/usr/bin/env python3
"""Training-environment results: per-method table, paired comparisons of DARA-L against every alternative, and the
budget-rule test. Rules (comparisons, alpha, tolerance) come from the pre-registration; nothing is tuned here.

Inputs : runs/train/<env>__<method>_s<seed>/{gt_per_episode.csv, triage_summary.csv}
Outputs: results/training/{training_table.csv, training_tests.csv, budget_test.csv, report.txt}

Benjamini-Hochberg families: the pre-registered comparisons on the primary environment; the other comparisons there;
and each other training environment.
Usage:  python scripts/analysis/evaluate_training.py [--runs runs/train] [--prereg configs/preregistration/paper_v2.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr.results import load_training, matrix  # noqa: E402
from dara_abr.stats import compare, bh, verdict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "results" / "training"))
ap.add_argument("--n-boot", type=int, default=10000)
ap.add_argument("--landscape", default=str(ROOT / "runs" / "landscape" / "landscape_content_u1_integrated.csv"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
PR = json.load(open(a.prereg))
REF = PR["proposed_method"]
ALPHA = PR["statistics"]["alpha"]
TOL = PR["budget_rule_test"]["tolerance_rel"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
lines = []


def say(s=""):
    print(s); lines.append(str(s))


S, REFS, META = load_training(a.runs)
if S.empty:
    raise SystemExit(f"no finished runs under {a.runs}")
say(f"runs: {META.shape[0]}  environments: {sorted(S.env.unique())}  seeds found: {sorted(S.seed.unique())}")

# ---------------------------------------------------------------------------------------------- table
rows = []
for (env, m), g in S.groupby(["env", "method"]):
    X = matrix(g)
    mt = META[(META.env == env) & (META.method == m)]
    tiers = [np.dot(np.arange(5), t) for t in mt.policy_tier_share if t is not None] if "policy_tier_share" in mt else []
    lag_on = mt.final_lagrange == 1 if "final_lagrange" in mt else pd.Series(False, index=mt.index)
    rows.append(dict(env=env, method=m, label=R.label(m), role=R.METHODS.get(m, {}).get("role", ""),
                     n_seeds=X.shape[0], J0=X.mean(), J0_seed_sd=X.mean(1).std(ddof=1) if X.shape[0] > 1 else np.nan,
                     seed_means=json.dumps([round(v, 1) for v in X.mean(1)]),
                     freeze_s=matrix(g, "rebuffer").mean(), mean_tier=np.mean(tiers) if tiers else np.nan,
                     budget=mt.loc[lag_on, "final_lag_target"].mean() if lag_on.any() else np.nan,
                     dara_acted=f"{int(lag_on.sum())}/{len(mt)}" if m.startswith("dara") and "lag" in m else "",
                     final_lambda=mt.final_lambda.mean() if "final_lambda" in mt else np.nan))
for (env, m), g in REFS.groupby(["env", "method"]):
    rows.append(dict(env=env, method=m, label=R.label(m), role="reference", n_seeds=0, J0=g.J0.mean(),
                     freeze_s=g.rebuffer.mean()))
ROLE_ORDER = ["proposed", "diagnosis", "baseline", "hand_tuned", "reward_ablation", "ablation", "reference"]
T = pd.DataFrame(rows)
T["_r"] = T.role.map(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else len(ROLE_ORDER))
T = T.sort_values(["env", "_r", "J0"], ascending=[True, True, False]).drop(columns="_r")
T.to_csv(OUT / "training_table.csv", index=False)
say(f"\n== training environments: score J0 per session (mean over seeds and the {S.episode.nunique()} shared test sessions) ==")
say(T.drop(columns=["label"]).round(3).to_string(index=False))

# ---------------------------------------------------------------------------------------------- paired comparisons
tests = []
for env in sorted(S.env.unique()):
    if not ((S.env == env) & (S.method == REF)).any():
        continue
    A = matrix(S[(S.env == env) & (S.method == REF)])
    others = [m for m in sorted(S[S.env == env].method.unique()) if m != REF]
    if env == PR["primary_training_environment"]:
        wanted = PR["comparisons"]["training_environment"]["against"]
        others = [m for m in others if m in wanted] + [m for m in others if m not in wanted]
    for m in others:
        B = matrix(S[(S.env == env) & (S.method == m)])
        tests.append(dict(env=env, comparison=m, label=R.label(m), **compare(A, B, n_boot=a.n_boot)))
    rf = REFS[REFS.env == env]
    if len(rf):
        tiers = {k: g for k, g in rf.groupby("method") if k.startswith("tier")}
        if "robustmpc" in set(rf.method):
            tests.append(dict(env=env, comparison="robustmpc", label="RobustMPC",
                              **compare(A, matrix(rf[rf.method == "robustmpc"]), n_boot=a.n_boot)))
        if tiers:
            best = max(tiers, key=lambda k: tiers[k].J0.mean())
            tests.append(dict(env=env, comparison=f"best_single_tier ({best})", label=f"Best single tier ({best})",
                              **compare(A, matrix(tiers[best]), n_boot=a.n_boot)))
P = pd.DataFrame(tests)
if len(P):
    def registered(env, comp):
        c = comp.split(" ")[0]
        if env == PR["primary_training_environment"] and c in PR["comparisons"]["training_environment"]["against"]:
            return True
        return any(env in st.get("envs", []) and c in st.get("against", []) for st in PR.get("sets", {}).values())
    P["prereg"] = [registered(e, c) for e, c in zip(P.env, P.comparison)]
    # BH families: the pre-registered comparisons on the primary environment form one family; every other
    # comparison set (non-registered ones on the primary environment, each other environment) is its own family
    P["family"] = [f"{e}:{'prereg' if r else 'other'}" for e, r in zip(P.env, P.prereg)]
    P["q_session"] = np.nan
    for fam, idx in P.groupby("family").groups.items():
        P.loc[idx, "q_session"] = bh(P.loc[idx, "p_session"].values)
    P["verdict"] = [verdict(d, q, lo, hi, ALPHA, "DARA-L") for d, q, lo, hi in zip(P["diff"], P.q_session, P.ci_lo, P.ci_hi)]
    P.to_csv(OUT / "training_tests.csv", index=False)
    say(f"\n== {REF} minus each alternative on the same test sessions "
        "(session: Wilcoxon on seed-averaged sessions, BH per family; seed: Welch t-test on seed means; "
        "CI: hierarchical bootstrap over seeds and sessions) ==")
    cols = ["env", "comparison", "diff", "ci_lo", "ci_hi", "share_sessions_better", "p_session", "q_session", "p_seed",
            "n_seeds_a", "n_seeds_b", "family", "verdict"]
    say(P[cols].round(4).to_string(index=False))

# ---------------------------------------------------------------------------------------------- budget-rule test
B = []
for env in sorted(S.env.unique()):
    d = T[(T.env == env)]
    hand = d[d.method.str.match(r"ppo_lagrangian_t[0-9.]+$")].copy()
    if hand.empty or not (d.method == REF).any():
        continue
    hand["budget"] = hand.method.str.extract(r"_t([0-9.]+)$")[0].astype(float)
    hand = hand.sort_values("budget")
    dl = d[d.method == REF].iloc[0]
    best = hand.loc[hand.J0.idxmax()]
    gap = dl.J0 - best.J0
    acted = np.isfinite(dl.budget)
    interp = np.interp(np.log(dl.budget), np.log(hand.budget), hand.J0) if acted else np.nan
    cmp = P[(P.env == env) & (P.comparison == best.method)].iloc[0] if len(P) else None
    B.append(dict(env=env, primary=env == PR["primary_training_environment"], derived_budget=dl.budget,
                  dara_acted=dl.dara_acted, J0_daral=dl.J0, best_hand_budget=best.budget, J0_best_hand=best.J0,
                  gap=gap, gap_rel=gap / abs(best.J0), passes=(bool(gap / abs(best.J0) >= -TOL) if acted else None),
                  note="" if acted else "DARA did not act in any seed: no derived budget",
                  rank=int((hand.J0 > dl.J0).sum()) + 1, n_hand=len(hand),
                  J0_hand_curve_at_derived=interp, J0_by_hand_budget=json.dumps({f"{b:g}": round(j, 1) for b, j in zip(hand.budget, hand.J0)}),
                  q_vs_best=cmp.q_session if cmp is not None else np.nan, verdict_vs_best=cmp.verdict if cmp is not None else ""))
BT = pd.DataFrame(B)
if len(BT):
    BT.to_csv(OUT / "budget_test.csv", index=False)
    say(f"\n== budget rule: DARA-L's derived budget vs hand-picked budgets (pass = within {TOL:.0%} of the best) ==")
    say(BT.round(3).to_string(index=False))
# ---------------------------------------------------------------------------------------------- C1: does M predict failure?
# every trained reward setting (original reward and cap variants) on the mmWave generator ladders: M before training
# (landscape, same ladder and cap) against the trained shortfall relative to the best single tier on the same sessions
LS = pd.read_csv(a.landscape) if Path(a.landscape).exists() else None
if LS is not None:
    LS = LS[(LS.quality_k == 0) & np.isclose(LS.q_top, R.QUALITY[-1])]
    if "rebuf_scale" in LS:
        LS = LS[LS.rebuf_scale == 1.0]
    pts = []
    for env, e in R.TRAINING_ENVS.items():
        if e["net"].get("NET_SOURCE") != "generator" or e["net"].get("NET_SCENARIOS"):
            continue
        rf = REFS[REFS.env == env]
        tiers = {k: g for k, g in rf.groupby("method") if k.startswith("tier")}
        if not tiers:
            continue
        best = max(tiers, key=lambda k: tiers[k].J0.mean()); bJ = tiers[best].J0.mean()
        for c in R.CAPS:
            m = R.cap_key(c)
            g = S[(S.env == env) & (S.method == m)]
            l = LS[np.isclose(LS.ladder_scale, e["ladder_scale"]) & np.isclose(LS.rn_cap, c)]
            if g.empty or l.empty:
                continue
            per_seed = g.groupby("seed").J0.mean()
            pts.append(dict(env=env, ladder_scale=e["ladder_scale"], cap=c, method=m, M=float(l.M.iloc[0]),
                            D=float(l.D.iloc[0]), best_tier=best, J0_best_tier=bJ, J0=per_seed.mean(),
                            shortfall=(bJ - per_seed.mean()) / abs(bJ), n_seeds=len(per_seed),
                            seed_shortfalls=json.dumps([round((bJ - v) / abs(bJ), 4) for v in per_seed])))
    DV = pd.DataFrame(pts)
    if len(DV) >= 4:
        from scipy.stats import spearmanr
        rho = spearmanr(DV.M, DV.shortfall).correlation
        rng = np.random.default_rng(0); boots = []
        seeds = [np.array(json.loads(x)) for x in DV.seed_shortfalls]
        for _ in range(2000):
            sf = [rng.choice(v, len(v)).mean() for v in seeds]
            boots.append(spearmanr(DV.M, sf).correlation)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        thr = PR.get("act_threshold", {}).get("primary", 0.2)
        DV["flagged"] = DV.M >= thr
        DV.to_csv(OUT / "diagnosis_validity.csv", index=False)
        say(f"\n== C1: M before training vs trained shortfall from the best single tier ({len(DV)} settings) ==")
        say(DV.drop(columns=["seed_shortfalls"]).round(3).to_string(index=False))
        say(f"Spearman rho(M, shortfall) = {rho:.3f}, 95% bootstrap interval [{lo:.3f}, {hi:.3f}] (seeds resampled per setting)")
        fl, nf = DV[DV.flagged], DV[~DV.flagged]
        say(f"settings with M >= {thr}: {len(fl)}, mean shortfall {fl.shortfall.mean():+.3f}; "
            f"M < {thr}: {len(nf)}, mean shortfall {nf.shortfall.mean():+.3f}")
        pd.DataFrame([dict(rho=rho, ci_lo=lo, ci_hi=hi, n=len(DV), threshold=thr,
                           mean_shortfall_flagged=fl.shortfall.mean(), mean_shortfall_unflagged=nf.shortfall.mean())]
                     ).to_csv(OUT / "diagnosis_validity_summary.csv", index=False)
else:
    say(f"\n(C1 skipped: {a.landscape} not found)")

(OUT / "report.txt").write_text("\n".join(lines) + "\n")
print(f"\nwritten to {OUT}")
