#!/usr/bin/env python3
"""Results at a glance: a self-contained report that explains every result to a reader new to the project.

The report tells the results as one story, from "where does the training reward go wrong" to "how does DARA-L compare".
Every chart states its question and, underneath, the answer in plain words. All numbers and answers are computed from
the result files, so the report stays correct when the results change.

  overview/index.html        the report (open in a browser; images and tables embedded, works offline)
  overview/README.md         the same text for GitHub, linking the PNGs and CSVs
  overview/*.png, *.csv      every chart, and the numbers behind it
  overview/summary.csv       one row per training environment: difficulty, budget, score of every main method

  Overview (01-08)           the story: difficulty, what fails, the diagnosis, overall standing, budgets, significance
  Per environment (10_env_*) every method trained there: seeds, score breakdown, quality levels, session spread, table
  Training (11, 12)          score and penalty weight during training, with the episode DARA switched on
  Statistics (20)            every comparison with its 95% interval, session share, q and p values
  Budgets and caps (30, 31)  every hand-picked budget; the reward-cap sweep
  Unseen networks (40, 41)   score, freezing and quality per network and ladder; significance
  Networks (50, 51)          throughput of every network; how often each quality level is sustainable

Inputs : results/training/*.csv, results/generalisation/*.csv  (evaluate_training.py, evaluate_generalisation.py)
         runs/train/*/dara_log.csv (optional: misalignment M of the trace environments)
Usage  : python scripts/analysis/make_overview.py [--results results] [--runs runs/train] [--out overview]
"""
import argparse, base64, glob, html, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401  (older matplotlib + pandas 2)

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "overview"))
a = ap.parse_args()
RES, OUT = Path(a.results), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
for old in list(OUT.glob("*.png")) + list(OUT.glob("*.csv")):
    old.unlink()
PR = json.load(open(a.prereg))
REF = PR["proposed_method"]
THR = PR.get("act_threshold", {}).get("primary", 0.2)
TOL = PR.get("budget_rule_test", {}).get("tolerance_rel", 0.05)

# ------------------------------------------------------------------------------------------------ style
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8986", "#e6e5e1", "#fcfcfb"
BLUE, BLUE_L, RED, GREY = "#2a78d6", "#86b6ef", "#e34948", "#a9a8a4"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                     "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
                     "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
                     "axes.linewidth": 0.7, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "savefig.dpi": 150})
MAIN = [REF, "ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc"]
MARK = {REF: "D", "ppo_lagrangian_t0.5": "o", "corel_official": "s", "pensieve_official": "^", "robustmpc": "v"}
SHORT = {REF: "DARA-L (ours)", "ppo_lagrangian_t0.5": "PPO-Lagrangian, 0.5 s budget", "corel_official": "COREL",
         "pensieve_official": "Pensieve", "robustmpc": "RobustMPC"}
EXCLUDE_ROLES = {"ablation", "diagnosis"}       # not counted as "the best anyone achieved"


def read(path):
    p = RES / path
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None
    return pd.read_csv(p)


TT, TS = read("training/training_table.csv"), read("training/training_tests.csv")
BT, DV, SV = read("training/budget_test.csv"), read("training/diagnosis_validity.csv"), read("training/diagnosis_validity_summary.csv")
GT, GS = read("generalisation/gen_tests.csv"), read("generalisation/gen_table.csv")
if TT is None:
    sys.exit("results/training/training_table.csv is needed (run scripts/analysis/evaluate_training.py)")


def short_env(e):
    return R.TRAINING_ENVS.get(e, {}).get("short", e)


def long_env(e):
    t = R.TRAINING_ENVS.get(e, {})
    return t.get("label", e).split(" (")[0] if t else e


# ================================================================================================ per-environment summary
def env_M(e):
    """misalignment M at the training reward (cap 0.1), as DARA measured it at the start of training"""
    ms = []
    for f in sorted(glob.glob(str(Path(a.runs) / f"{e}__{REF}_s*" / "dara_log.csv"))):
        d = pd.read_csv(f)
        if len(d) and "M" in d:
            ms.append(float(d.M.iloc[0]))
    if ms:
        return float(np.mean(ms))
    if DV is not None:
        d = DV[(DV.env == e) & (DV.method == "fixed")]
        if len(d):
            return float(d.M.iloc[0])
    return np.nan


rows = []
for e in TT.env.unique():
    d = TT[TT.env == e].set_index("method")
    pool = d[~d.role.isin(EXCLUDE_ROLES)]
    best = pool.J0.idxmax()
    tiers = d[d.index.str.fullmatch(r"tier[0-4]")]
    r = dict(env=e, environment=long_env(e), M=env_M(e),
             dara_acted=d.loc[REF, "dara_acted"] if REF in d.index else "",
             derived_budget=d.loc[REF, "budget"] if REF in d.index else np.nan,
             best_method=best, best_J0=float(pool.loc[best, "J0"]),
             best_fixed_tier=tiers.J0.idxmax() if len(tiers) else "", best_fixed_tier_J0=tiers.J0.max() if len(tiers) else np.nan,
             plain_ppo_J0=d.loc["fixed", "J0"] if "fixed" in d.index else np.nan,
             plain_ppo_mean_tier=d.loc["fixed", "mean_tier"] if "fixed" in d.index else np.nan)
    for m in MAIN:
        r[f"{m}_J0"] = d.loc[m, "J0"] if m in d.index else np.nan
        r[f"{m}_freeze_s"] = d.loc[m, "freeze_s"] if m in d.index else np.nan
    r["dara_l_final_lambda"] = d.loc[REF, "final_lambda"] if REF in d.index else np.nan
    r["ppo_lag05_final_lambda"] = d.loc["ppo_lagrangian_t0.5", "final_lambda"] if "ppo_lagrangian_t0.5" in d.index else np.nan
    if BT is not None and (BT.env == e).any():
        b = BT[BT.env == e].iloc[0]
        r.update(best_hand_budget=b.best_hand_budget, best_hand_J0=b.J0_best_hand, n_hand_budgets=int(b.n_hand),
                 gap_to_best_hand=b.gap_rel)
    rows.append(r)
S = pd.DataFrame(rows).sort_values("M", na_position="last").reset_index(drop=True)
S.round(4).to_csv(OUT / "summary.csv", index=False)
ENVS = list(S.env)                       # easiest (lowest M) first
SIDX = S.set_index("env")


def verdict(e, comp):
    if TS is None:
        return None
    p = TS[(TS.env == e) & (TS.comparison.str.startswith("best_single_tier") if comp == "best_tier" else TS.comparison == comp)]
    return None if p.empty else p.iloc[0]


def pct(x):
    return f"{100 * x:+.0f}%"


sections = []   # dicts: part, name, question, answer sentences, how-to-read, optional table (DataFrame)
PART = "Overview"


def save(fig, name, question, answer, how, table=None):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    sections.append(dict(part=PART, name=name, q=question, ans=[x for x in answer if x], how=how, table=table))
    print("written", OUT / f"{name}.png")


def place_labels(fig, ax, items, fontsize=8.8):
    """annotate points without overlapping each other or the points: try a few offsets per label, keep the first free one"""
    fig.canvas.draw()
    r_ = fig.canvas.get_renderer()
    taken = [ax.transData.transform(xy) for _, xy in items]
    taken = [matplotlib.transforms.Bbox.from_extents(x - 9, y - 9, x + 9, y + 9) for x, y in taken]
    for txt, xy in items:
        best = None
        for dx, dy, ha in ((10, 8, "left"), (10, -16, "left"), (-10, 8, "right"), (-10, -16, "left" if False else "right"),
                           (10, 22, "left"), (10, -30, "left"), (-10, 22, "right"), (-10, -30, "right")):
            an = ax.annotate(txt, xy, textcoords="offset points", xytext=(dx, dy), fontsize=fontsize, color=INK, ha=ha)
            bb = an.get_window_extent(r_)
            if not any(bb.overlaps(t) for t in taken) and ax.bbox.contains(*bb.p0) and ax.bbox.contains(*bb.p1):
                best = an
                break
            an.remove()
        if best is None:
            best = ax.annotate(txt, xy, textcoords="offset points", xytext=(10, 8), fontsize=fontsize, color=INK)
        taken.append(best.get_window_extent(r_))


def ylabels(ax, envs):
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([f"{short_env(e)}" for e in envs])
    ax.invert_yaxis()


# ================================================================================================ 1 difficulty
if S.M.notna().any():
    d = S[S.M.notna()]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(d) + 1.2))
    cols = [BLUE if m >= THR else GREY for m in d.M]
    ax.barh(range(len(d)), d.M, color=cols, height=0.62, edgecolor=SURF, lw=1.5)
    ax.axvline(THR, color=INK2, ls="--", lw=1)
    ax.text(THR, -0.85, f" threshold {THR:g}", fontsize=9, color=INK2, va="bottom")
    for i, r in enumerate(d.itertuples()):
        txt = f"M = {r.M:.2f}   " + (f"DARA acts; derived budget {r.derived_budget:.2f} s per chunk" if r.M >= THR and pd.notna(r.derived_budget)
                                      else "DARA leaves training alone")
        ax.text(r.M + 0.01, i, txt, va="center", fontsize=9, color=INK)
    ylabels(ax, list(d.env))
    ax.set_xlim(0, 1.45); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Misalignment M: share of test situations where the training reward prefers a quality the viewer would rate badly")
    ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    acted = d[d.M >= THR]
    ans = [f"{len(d)} training environments, sorted from least to most misaligned. DARA intervenes in the {len(acted)} with M ≥ {THR:g} "
           f"and leaves {len(d) - len(acted)} alone ({', '.join(short_env(e) for e in d[d.M < THR].env) or 'none'})."]
    lm = [SIDX.loc[e, "M"] for e in R.LADDER_ENVS if e in SIDX.index and pd.notna(SIDX.loc[e, "M"])]
    if len(lm) > 2 and all(x < y for x, y in zip(lm, lm[1:])):
        ans.append("On the same network, M rises with chunk size (" + " → ".join(f"{m:.2f}" for m in lm) + " for ladders "
                   + ", ".join(short_env(e).split()[-1] for e in R.LADDER_ENVS if e in SIDX.index) + "): bigger chunks take "
                   "longer to download, so the top quality the reward keeps paying for causes more freezing.")
    save(fig, "01_difficulty", "How badly does the training reward mislead the agent in each environment?", ans,
         "Each bar is one training environment. M is measured before any training, by trying every quality level in "
         "360 test situations and comparing what the training reward prefers with what the real score (J0) prefers.")

# ================================================================================================ 2 what goes wrong without DARA
lad = [e for e in R.LADDER_ENVS if e in SIDX.index and pd.notna(SIDX.loc[e, "plain_ppo_J0"])]
if lad:
    fig, ax = plt.subplots(figsize=(10, 4.3))
    series = [("plain_ppo_J0", "Plain PPO on the training reward (no fix)", RED),
              (f"{REF}_J0", "DARA-L (diagnosis + freeze limit)", BLUE),
              ("best_fixed_tier_J0", "Best 'always the same quality' rule", GREY)]
    w = 0.26
    x = np.arange(len(lad))
    for k, (col, lbl, c) in enumerate(series):
        v = SIDX.loc[lad, col].to_numpy(float)
        ax.bar(x + (k - 1) * w, v, w * 0.92, color=c, label=lbl, edgecolor=SURF, lw=1)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + (6 if vi >= 0 else -6), f"{vi:.0f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{short_env(e)}\n(M = {SIDX.loc[e, 'M']:.2f})" for e in lad])
    ax.set_ylabel("Score J0 per session (higher is better)")
    ax.grid(True, axis="y", color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    worst = min(lad, key=lambda e: SIDX.loc[e, "plain_ppo_J0"])
    ans = [f"On the easiest ladder ({short_env(lad[0])}) plain training already works ({SIDX.loc[lad[0], 'plain_ppo_J0']:.0f} vs "
           f"{SIDX.loc[lad[0], 'best_fixed_tier_J0']:.0f} for the best fixed quality).",
           f"As chunks grow, plain training collapses: on {short_env(worst)} it scores {SIDX.loc[worst, 'plain_ppo_J0']:.0f}. "
           f"Its average quality level there is {SIDX.loc[worst, 'plain_ppo_mean_tier']:.1f} of 4 (almost always the top), so the "
           f"video keeps freezing. DARA-L scores {SIDX.loc[worst, f'{REF}_J0']:.0f} there."]
    save(fig, "02_without_fix", "What happens if the agent is simply trained on the reward?", ans,
         "Same network (mmWave generator), four chunk-size ladders. A negative score means freezing cost more than the "
         "picture quality earned.")

# ================================================================================================ 3 diagnosis predicts failure
if DV is not None and len(DV):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(THR, 1, color="#f4f3f0", zorder=0)
    ax.axvline(THR, color=INK2, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.7)
    scales = sorted(DV.ladder_scale.unique())
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"][-len(scales):]
    for sc, col in zip(scales, ramp):
        d = DV[DV.ladder_scale == sc]
        ax.scatter(d.M, 100 * d.shortfall, s=70, color=col, edgecolor=SURF, lw=1, zorder=3, label=f"ladder ×{sc:g}")
    ax.set_xlim(0, max(0.8, DV.M.max() + 0.05))
    ax.text(THR - 0.01, 0.95, "DARA: no problem", transform=ax.get_xaxis_transform(), ha="right", fontsize=9, color=INK2)
    ax.text(THR + 0.01, 0.95, "DARA: problem, intervene", transform=ax.get_xaxis_transform(), fontsize=9, color=INK2)
    ax.set_xlabel("Misalignment M measured BEFORE training")
    ax.set_ylabel("How far plain training fell short\nof the best fixed quality (%)")
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    ans = []
    if SV is not None and len(SV):
        r = SV.iloc[0]
        ans.append(f"Yes. Over {int(r.n)} reward settings, the higher M was before training, the worse the trained agent did "
                   f"(rank correlation {r.rho:.2f}, 95% interval {r.ci_lo:.2f} to {r.ci_hi:.2f}).")
        ans.append(f"Settings DARA flags (M ≥ {THR:g}) fell {100 * r.mean_shortfall_flagged:.0f}% short of the best fixed quality "
                   f"on average (over 100% means the score went negative); settings it passes fell {100 * r.mean_shortfall_unflagged:.1f}% short.")
    fl = DV[DV.flagged.astype(str) == "True"]
    small = fl[fl.shortfall < 0.05]
    if len(small):
        ans.append(f"Caveat: {len(small)} of {len(fl)} flagged settings lost less than 5% (all on "
                   f"{', '.join(sorted(set(short_env(e) for e in small.env)))}): a high M warns of failure but does not guarantee it.")
    save(fig, "03_diagnosis", "Can the diagnosis tell, before training, whether training will fail?", ans,
         "Each dot: one reward setting (ladder × how much the reward penalises freezing), trained with plain PPO on 5 seeds. "
         "Up = trained agent did badly. A good diagnosis puts the low dots left of the line and the high dots right of it.")

# ================================================================================================ 4 where DARA-L stands
fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(ENVS) + 1.6))
for i, e in enumerate(ENVS):
    best = SIDX.loc[e, "best_J0"]
    vals = {m: SIDX.loc[e, f"{m}_J0"] / best for m in MAIN if pd.notna(SIDX.loc[e, f"{m}_J0"])}
    if vals:
        ax.plot([100 * min(vals.values()), 100 * max(vals.values())], [i, i], color=GRID, lw=2, zorder=1)
    for m, v in vals.items():
        ax.scatter(100 * v, i, marker=MARK[m], s=130 if m == REF else 70, color=R.colour(m), edgecolor=SURF, lw=1.2,
                   zorder=4 if m == REF else 3)
ax.axvline(100, color=INK2, lw=0.8)
lo_ = min(100 * SIDX.loc[e, f"{m}_J0"] / SIDX.loc[e, "best_J0"] for e in ENVS for m in MAIN if pd.notna(SIDX.loc[e, f"{m}_J0"]))
ax.set_xlim(max(0, 10 * np.floor(lo_ / 10) - 5), 103)
ylabels(ax, ENVS)
ax.set_xlabel("Score as % of the best result anyone achieved in that environment (100% = the best)")
ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.text(1.0, 1.01, "harder ↓", transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
hs = [Line2D([], [], ls="", marker=MARK[m], ms=10 if m == REF else 8, color=R.colour(m), mec=SURF, label=SHORT[m]) for m in MAIN]
ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, frameon=False)
rel = pd.Series({e: SIDX.loc[e, f"{REF}_J0"] / SIDX.loc[e, "best_J0"] for e in ENVS})
near = rel[rel >= 0.95]; far = rel[rel < 0.9]
first = [e for e in ENVS if pd.notna(SIDX.loc[e, f"{REF}_J0"]) and
         SIDX.loc[e, f"{REF}_J0"] >= max(SIDX.loc[e, f"{m}_J0"] for m in MAIN if pd.notna(SIDX.loc[e, f"{m}_J0"]))]
ans = [f"DARA-L is within 5% of the best result in {len(near)} of {len(rel)} environments "
       f"({', '.join(short_env(e) for e in near.index) or 'none'}).",
       f"It falls more than 10% behind in {len(far)}: " +
       (", ".join(f"{short_env(e)} ({100 * v:.0f}%)" for e, v in far.items()) if len(far) else "none") + ".",
       f"It has the highest mean of the five methods shown in {len(first)} of {len(ENVS)} environments"
       + (f" ({', '.join(short_env(e) for e in first)})" if first else "") +
       ". Small gaps are often not statistically significant (chart 7)."]
save(fig, "04_overall", "Overall: how close does each method get to the best possible result?", ans,
     "One row per training environment, easiest at the top (lowest M). 'The best result anyone achieved' is the highest "
     "mean score of any method or fixed rule we ran there, including every hand-picked budget and reward variant.")

# ================================================================================================ 5 budget rule vs difficulty
if BT is not None:
    d = S[S.gap_to_best_hand.notna() & S.derived_budget.notna() & S.M.notna()] if "gap_to_best_hand" in S else S.iloc[:0]
    if len(d):
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.axhspan(-100 * TOL, 100 * max(TOL, d.gap_to_best_hand.max() + 0.02), color="#e3eefb", zorder=0)
        ax.axhline(0, color=INK2, lw=0.8)
        labels_ = []
        for n_, r in enumerate(d.sort_values("M").itertuples()):
            ok = r.gap_to_best_hand >= -TOL
            few = r.n_hand_budgets < 3
            ax.scatter(r.M, 100 * r.gap_to_best_hand, s=110, marker="D", zorder=3,
                       color=SURF if few else (BLUE if ok else RED), edgecolor=BLUE if ok else RED, lw=2)
            labels_.append((f"{short_env(r.env)}: {r.derived_budget:.2f} s vs {r.best_hand_budget:g} s",
                            (r.M, 100 * r.gap_to_best_hand)))
        ax.set_xlim(THR - 0.05, 1.0)
        ax.set_xlabel("Misalignment M of the environment (harder →)")
        ax.set_ylabel("DARA-L's score minus the best\nhand-picked budget's score (%)")
        ax.grid(True, color=GRID); ax.set_axisbelow(True)
        hs = [Patch(fc="#e3eefb", label=f"pass zone: within {100 * TOL:.0f}% of the best hand-picked budget, or better"),
              Line2D([], [], ls="", marker="D", ms=9, color=BLUE, label="passes"),
              Line2D([], [], ls="", marker="D", ms=9, color=RED, label="fails"),
              Line2D([], [], ls="", marker="D", ms=9, color=SURF, mec=INK2, mew=1.5, label="only one hand-picked budget to compare with")]
        place_labels(fig, ax, labels_)
        ax.text(0.01, 0.99, "label: DARA-L's budget vs the best hand-picked budget (s of freezing per chunk)",
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
        ax.legend(handles=hs, frameon=False, loc="lower left")
        ax.set_ylim(100 * d.gap_to_best_hand.min() - 6, 100 * max(TOL, d.gap_to_best_hand.max()) + 6)
        ok = d[d.gap_to_best_hand >= -TOL]; bad = d[d.gap_to_best_hand < -TOL]
        set_envs = {e for k in ("set1", "set2") for e in PR.get("sets", {}).get(k, {}).get("envs", [])}
        c3 = d[d.env.isin(set_envs) & (d.n_hand_budgets >= 3)]
        ans = [f"DARA-L picks a good budget in {len(ok)} of {len(d)} environments where it acts, but misses in "
               f"{len(bad)} ({', '.join(short_env(e) for e in bad.env) or 'none'}).",
               f"The misses are the hardest environments (M ≥ {bad.M.min():.2f})." if len(bad) else "",
               ("Same rule as the pre-registered test, applied to this view" if PR.get("deviation") else "Pre-registered test")
               + f" (at least 3 hand-picked budgets): passes {int((c3.gap_to_best_hand >= -TOL).sum())} "
               f"of {len(c3)}." if len(c3) else ""]
        hard = bad[bad.best_hand_budget > bad.derived_budget]
        if len(hard):
            ans.append("In " + ", ".join(short_env(e) for e in hard.env) +
                       " the best hand-picked budget allowed more freezing than DARA-L's: the derived budget was too strict.")
        save(fig, "05_budget", "Does DARA-L choose a freeze budget as good as tuning it by hand?", [x for x in ans if x],
             "PPO-Lagrangian is the same learner as DARA-L but with the freeze budget picked by hand; we trained several "
             "budgets per environment and took the best. DARA-L gets its budget from the diagnosis, with no tuning. "
             "Only environments where DARA acts are shown.")

# ================================================================================================ 6 why: DARA-L vs PPO-Lagrangian 0.5
pl = "ppo_lagrangian_t0.5"
d = S[S[f"{REF}_J0"].notna() & S[f"{pl}_J0"].notna()]
if len(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(d) + 1.8), sharey=True)
    for ax, (col, xl) in zip(axes, (("_J0", "Score J0 per session (higher is better)"), ("_freeze_s", "Freezing per session (s, lower is better)"))):
        for i in range(len(d)):
            a_, b_ = d.iloc[i][f"{REF}{col}"], d.iloc[i][f"{pl}{col}"]
            ax.plot([a_, b_], [i, i], color=GRID, lw=3, zorder=1)
            ax.scatter(b_, i, marker="o", s=70, color=R.colour(pl), edgecolor=SURF, zorder=2)
            ax.scatter(a_, i, marker="D", s=90, color=BLUE, edgecolor=SURF, zorder=3)
        ax.set_xlabel(xl); ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ylabels(axes[0], list(d.env))
    hs = [Line2D([], [], ls="", marker="D", ms=9, color=BLUE, label=SHORT[REF]),
          Line2D([], [], ls="", marker="o", ms=8, color=R.colour(pl), label=SHORT[pl])]
    fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    cautious = d[(d[f"{REF}_freeze_s"] < d[f"{pl}_freeze_s"] - 5) & (d[f"{REF}_J0"] < d[f"{pl}_J0"] - 5)]
    ans = ["The fixed 0.5 s budget is the strongest single competitor: it is the same learner, with a budget that happens "
           "to suit several of these environments."]
    if len(cautious):
        ans.append(f"Where DARA-L scores lower ({', '.join(short_env(e) for e in cautious.env)}), it freezes LESS than PPO-Lagrangian but "
                   "also picks lower quality: it is too cautious.")
        lam = cautious[["env", "dara_l_final_lambda", "ppo_lag05_final_lambda"]].dropna()
        if len(lam):
            ans.append("A likely reason (exploratory): DARA switches the freeze limit on part-way through training, after the "
                       "agent has learned to freeze a lot, and the limit's penalty weight then grows much larger than PPO-Lagrangian's ("
                       + ", ".join(f"{short_env(r.env)}: {r.dara_l_final_lambda:.0f} vs {r.ppo_lag05_final_lambda:.0f}" for r in lam.itertuples()) + ").")
    gain = d[d[f"{REF}_J0"] > d[f"{pl}_J0"] + 5]
    if len(gain):
        ans.append("Where a 0.5 s budget is wrong for the network, DARA-L's own budget helps: " +
                   ", ".join(f"{short_env(r['env'])} {r[f'{REF}_J0']:.0f} vs {r[f'{pl}_J0']:.0f}" for _, r in gain.iterrows()) + ".")
    save(fig, "06_vs_fixed_budget", "Why does DARA-L not beat a hand-picked budget of 0.5 s?", ans,
         "Same rows as chart 4. Left: score; right: seconds of freezing per session. Each line joins the two methods in "
         "one environment; where only the diamond is visible, the two values are almost equal.")

# ================================================================================================ 7 significance grid
VERD = {"DARA-L better": ("better", "#86b6ef", None), "DARA-L worse": ("worse", "#f19c9b", None),
        "no significant difference": ("no difference", "#e2e1dc", None),
        "inconclusive (tests disagree)": ("unclear", SURF, "///")}


def grid_chart(ax, rows_, cols_, cell, rowlab, collab):
    for i, r in enumerate(rows_):
        for j, c in enumerate(cols_):
            v, diff = cell(r, c)
            if v is None:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=SURF, ec=GRID, lw=1))
                ax.text(j + .5, i + .5, "not run", ha="center", va="center", fontsize=8.5, color=MUTED)
                continue
            t, fc, h = VERD[v]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=fc, ec="#bdbcb7" if h else SURF, lw=1.5, hatch=h))
            ax.text(j + .5, i + .5, f"{t}\n{diff:+.0f}", ha="center", va="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF if h else fc, ec="none"))
    ax.set_xlim(0, len(cols_)); ax.set_ylim(len(rows_), 0)
    ax.set_xticks(np.arange(len(cols_)) + .5); ax.set_xticklabels(collab, fontweight="bold"); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows_)) + .5); ax.set_yticklabels(rowlab)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def grid_legend(fig):
    hs = [Patch(fc=VERD[k][1], ec="#bdbcb7", hatch=VERD[k][2], label=l) for k, l in (
        ("DARA-L better", "DARA-L significantly better"), ("DARA-L worse", "DARA-L significantly worse"),
        ("no significant difference", "no significant difference"), ("inconclusive (tests disagree)", "unclear: the two tests disagree"))]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))


if TS is not None:
    comps = ["ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(ENVS) + 1.4))

    def cell(e, c):
        p = TS[(TS.env == e) & (TS.comparison == c)]
        return (None, None) if p.empty else (p.verdict.iloc[0], float(p["diff"].iloc[0]))
    grid_chart(ax, ENVS, comps, cell, [short_env(e) for e in ENVS], [SHORT[c].replace(" budget", "") for c in comps])
    fig.tight_layout(rect=(0, 0.06, 1, 1)); grid_legend(fig)
    ans = []
    for c in comps:
        v = TS[TS.env.isin(ENVS) & (TS.comparison == c)].verdict.value_counts()
        n = int(v.sum())
        ans.append(f"vs {SHORT[c]}: significantly better in {v.get('DARA-L better', 0)} of {n}, worse in "
                   f"{v.get('DARA-L worse', 0)}, no difference in {v.get('no significant difference', 0)}, unclear in "
                   f"{v.get('inconclusive (tests disagree)', 0)}.")
    save(fig, "07_significance", "Which differences are real, and which could be chance?", ans,
         "Number = DARA-L's mean score minus the other method's. 'Significant' needs two tests to agree: one over the 100 "
         "test sessions and one that also accounts for the 5 training seeds. 'Unclear' usually means DARA-L wins most "
         "sessions by a little but loses a few by a lot, or the 5 training runs vary too much to tell.")

# ================================================================================================ 8 unseen networks
if GS is not None and len(GS):
    meths = [m for m in MAIN if m in set(GS.method)]
    conds = [f"{n}/{c}" for n in R.GEN_NETWORKS for c in ("x1", "x4", "x8") if f"{n}/{c}" in set(GS.condition)]
    tab = GS[GS.method.isin(meths)].pivot_table(index="condition", columns="method", values="J0").reindex(conds)[meths]
    rank = tab.rank(axis=1, ascending=False, method="min")
    shade = {1: ("#1c5cab", "white"), 2: ("#3987e5", "white"), 3: ("#86b6ef", INK), 4: ("#cde2fb", INK), 5: ("#f0efec", INK)}
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(conds) + 1.4))
    for i, cnd in enumerate(conds):
        for j, m in enumerate(meths):
            v = tab.loc[cnd, m]
            if pd.isna(v):
                continue
            k = int(rank.loc[cnd, m]); fc, tc = shade.get(k, shade[5])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=fc, ec=SURF, lw=2))
            ax.text(j + .5, i + .5, f"{v:.0f}  ({k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'})",
                    ha="center", va="center", fontsize=9.5, color=tc, fontweight="bold" if k == 1 else "normal")
    ax.set_xlim(0, len(meths)); ax.set_ylim(len(conds), 0)
    ax.set_xticks(np.arange(len(meths)) + .5); ax.set_xticklabels([SHORT[m].replace(" budget", "") for m in meths], fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(conds)) + .5)
    ax.set_yticklabels([f"{R.NETWORKS[c.split('/')[0]]['label']}, ladder {R.CONTENTS[c.split('/')[1]]}"
                        + ("  ← training" if c == R.TRAINING_CONDITION else "") for c in conds])
    for i in range(1, len(R.GEN_NETWORKS)):
        k = sum(1 for c in conds if c.split("/")[0] in R.GEN_NETWORKS[:i])
        if 0 < k < len(conds):
            ax.axhline(k, color=INK2, lw=1.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    unseen = [c for c in conds if c != R.TRAINING_CONDITION]
    rk = rank.loc[unseen, REF] if REF in rank else pd.Series(dtype=float)
    ans = [f"On {len(unseen)} unseen conditions DARA-L ranks 1st in {int((rk == 1).sum())}, in the top two in "
           f"{int((rk <= 2).sum())}, and last in {int((rk == len(meths)).sum())} (average rank {rk.mean():.1f} of {len(meths)})."]
    for m in meths:
        if m != REF:
            r_ = rank.loc[unseen, m]
            ans.append(f"{SHORT[m]}: average rank {r_.mean():.1f}.")
    bad = [c for c in unseen if rank.loc[c, REF] >= len(meths) - 1]
    if bad:
        nm = lambda c: f"{R.NETWORKS[c.split('/')[0]]['label']} {R.CONTENTS[c.split('/')[1]]}"
        ans.append("DARA-L ranks low (4th or 5th) on " + ", ".join(nm(c) for c in bad) + ".")
    save(fig, "08_unseen", "Do the models still work on networks and chunk sizes they never saw?", ans,
         "All five methods were trained once on the mmWave generator with ladder ×4, then tested without retraining. "
         "Number = mean score; rank 1st = best in that row. Lumos5G: real mmWave traces; Irish 5G: slower, mid-band 5G "
         "(a deliberately hard test). Scores can be negative when freezing dominates.")

# ################################################################################################ DETAILS
# Everything below adds the full picture: every method in every environment, training curves, statistics with
# intervals, the hand-picked budget curves, the reward-cap sweep, the unseen-network details and the networks.
SEQ5 = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # quality levels 0 (lowest) .. 4 (highest)
VCOL = {"DARA-L better": BLUE, "DARA-L worse": RED, "no significant difference": MUTED, "inconclusive (tests disagree)": "#c98500"}
VTXT = {"DARA-L better": "DARA-L better", "DARA-L worse": "DARA-L worse", "no significant difference": "no difference",
        "inconclusive (tests disagree)": "unclear"}
RUNS = Path(a.runs)
RUN_DIRS = {}
for dd in sorted(RUNS.glob("*__*_s*")) if RUNS.exists() else []:
    pr = R.parse_run(dd.name)
    if pr:
        RUN_DIRS.setdefault((pr["env"], pr["method"]), []).append(dd)


def order_key(m):
    if m.startswith("dara"):
        return {REF: 0, "dara_online_lag_m015": 1, "dara_preflight_lag": 2, "dara_online": 3}.get(m, 4)
    if m.startswith("ppo_lagrangian_t"):
        return 10 + float(m.split("_t")[1])
    if m == "fixed":
        return 20.1
    if m.startswith("cap"):
        return 20 + float(m[3:]) + 0.2
    return {"static_best": 30, "ppo_ref_reward": 35, "corel_official": 40, "pensieve_official": 41,
            "robustmpc": 50}.get(m, 60)


def mlabel(m):
    if m == REF:
        return "DARA-L (ours)"
    if m == "fixed":
        return "Plain PPO, cap 0.1 (original reward)"
    if m == "static_best":
        return "Plain PPO, no cap"
    if m.startswith("tier"):
        return f"Always quality {m[-1]} (best fixed)"
    return R.label(m).replace("PPO, cap c = ", "Plain PPO, cap ").replace(" (ours)", "")


def runs_table(e, m, fname):
    out = []
    for dd in RUN_DIRS.get((e, m), []):
        f = dd / fname
        if f.exists():
            out.append(pd.read_csv(f))
    return out


def lst(x):
    try:
        return json.loads(x) if isinstance(x, str) else []
    except ValueError:
        return []


def env_methods(e):
    d = TT[TT.env == e]
    ms = [m for m in d.method if not m.startswith("tier")]
    bt = SIDX.loc[e, "best_fixed_tier"] if e in SIDX.index else ""
    return sorted(ms, key=order_key) + ([bt] if bt else [])


def row_colour(m):
    if m.startswith("tier"):
        return GREY
    if m == "fixed" or m.startswith("cap") or m == "static_best":
        return "#8a8986"
    return R.colour(m)


def session_stats(e, m):
    """per-session J0, quality earned (minus switching) and freeze penalty, averaged over seeds"""
    if m.startswith("tier") or m == "robustmpc":
        any_run = next((v for (ee, _), v in RUN_DIRS.items() if ee == e), None)
        if not any_run:
            return None
        g = pd.read_csv(any_run[0] / "gt_per_episode.csv")
        c = m if m.startswith("tier") else "robustmpc"
        ref, reb = g[f"{c}_ref"].to_numpy(float), g[f"{c}_rebuffer"].to_numpy(float)
        return ref, ref + 1.075 * reb, 1.075 * reb
    gs = runs_table(e, m, "gt_per_episode.csv")
    if not gs:
        return None
    ref = np.mean([g.policy_ref.to_numpy(float) for g in gs], axis=0)
    reb = np.mean([g.policy_rebuffer.to_numpy(float) for g in gs], axis=0)
    return ref, ref + 1.075 * reb, 1.075 * reb


def tier_share(e, m):
    if m.startswith("tier"):
        v = np.zeros(5); v[int(m[-1])] = 1; return v
    col = "mpc_tier_share" if m == "robustmpc" else "policy_tier_share"
    src = next((v for (ee, _), v in RUN_DIRS.items() if ee == e), []) if m == "robustmpc" else RUN_DIRS.get((e, m), [])
    vals = []
    for dd in src[:1] if m == "robustmpc" else src:
        f = dd / "triage_summary.csv"
        if f.exists():
            t = pd.read_csv(f)
            if col in t and len(lst(t[col].iloc[0])) == 5:
                vals.append(lst(t[col].iloc[0]))
    return np.mean(vals, axis=0) if vals else None


# ================================================================================================ B: one page per environment
PART = "Every method, environment by environment"
for e in ENVS:
    ms = env_methods(e)
    d = TT[TT.env == e].set_index("method")
    n = len(ms)
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.36 * n + 1.9), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1.15, 1.0, 1.15]))
    y = np.arange(n)
    stats = {m: session_stats(e, m) for m in ms}
    # (a) score with every seed
    ax = axes[0]
    for i, m in enumerate(ms):
        c = row_colour(m)
        sm = lst(d.loc[m, "seed_means"]) if m in d.index and "seed_means" in d else []
        if sm:
            ax.scatter(sm, [i] * len(sm), s=18, color=c, alpha=0.55, edgecolor="none", zorder=2)
        j0 = float(d.loc[m, "J0"])
        ax.scatter([j0], [i], marker="|", s=260, color=INK if m != REF else BLUE, lw=2.2, zorder=3)
        ax.text(1.0, i, f"{j0:.0f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_title("(a) Score J0: bar = mean, dots = the 5 seeds", loc="left", fontsize=10)
    # (b) what the score is made of
    ax = axes[1]
    for i, m in enumerate(ms):
        st = stats[m]
        if st is None:
            continue
        q, f = st[1].mean(), st[2].mean()
        ax.barh(i, q, color="#9ec5f4", height=0.6, edgecolor=SURF)
        ax.barh(i, -f, color="#f19c9b", height=0.6, edgecolor=SURF)
        ax.scatter(q - f, i, marker="D", s=26, color=INK, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title("(b) Quality earned (blue) minus freezing cost (red) = J0 (◆)", loc="left", fontsize=10)
    # (c) quality levels used
    ax = axes[2]
    for i, m in enumerate(ms):
        ts = tier_share(e, m)
        if ts is None:
            continue
        left = 0
        for k, v in enumerate(ts):
            ax.barh(i, 100 * v, left=left, color=SEQ5[k], height=0.6, edgecolor=SURF, lw=0.8)
            left += 100 * v
    ax.set_xlim(0, 100)
    ax.set_title("(c) Share of chunks at each quality level (%)", loc="left", fontsize=10)
    ax.legend(handles=[Patch(fc=SEQ5[k], label=f"q{k}") for k in range(5)], ncol=5, frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.04 - 0.4 / max(n, 6)), handlelength=1.2, columnspacing=0.8)
    # (d) spread over the 100 test sessions
    ax = axes[3]
    data = [stats[m][0] if stats[m] is not None else np.array([np.nan]) for m in ms]
    kw = dict(positions=y, widths=0.55, patch_artist=True, showfliers=True, whis=(5, 95),
              flierprops=dict(marker=".", ms=3, mfc=MUTED, mec="none"), medianprops=dict(color=INK, lw=1.5))
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)          # matplotlib >= 3.10
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kw)
    fin = [x[np.isfinite(x)] for x in data if np.isfinite(x).any()]
    lo_, hi_ = min(np.percentile(x, 2) for x in fin), max(np.percentile(x, 99.5) for x in fin)
    ax.set_xlim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.05 * (hi_ - lo_))
    for patch, m in zip(bp["boxes"], ms):
        patch.set_facecolor(row_colour(m)); patch.set_alpha(0.55); patch.set_edgecolor(INK2)
    ax.set_title("(d) Score per test session (box: middle half; whiskers: 5–95%)", loc="left", fontsize=10)
    ax.text(0.0, -0.06 - 0.5 / max(n, 6), "axis cut at the 2nd percentile; the table gives the worst 5%", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    for ax in axes:
        ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels([mlabel(m) for m in ms]); axes[0].invert_yaxis()
    for t, m in zip(axes[0].get_yticklabels(), ms):
        if m == REF:
            t.set_fontweight("bold")
    for ax, xl in zip(axes, ("J0 per session", "points per session", "% of chunks", "J0 per session")):
        ax.set_xlabel(xl)
    fig.tight_layout(w_pad=1.2)
    # table and sentences
    rows_ = []
    for m in ms:
        r = d.loc[m] if m in d.index else None
        st = stats[m]
        p = TS[(TS.env == e) & ((TS.comparison == m) | TS.comparison.str.endswith(f"({m})"))] if TS is not None else pd.DataFrame()
        rows_.append({"method": mlabel(m), "J0 (mean)": f"{r.J0:.1f}" if r is not None else "",
                      "seed s.d.": "" if r is None or pd.isna(r.get("J0_seed_sd")) else f"{r.J0_seed_sd:.1f}",
                      "freeze s/session": f"{r.freeze_s:.1f}" if r is not None else "",
                      "mean quality level": "" if r is None or pd.isna(r.get("mean_tier")) else f"{r.mean_tier:.2f}",
                      "worst 5% of sessions (J0)": "" if st is None else f"{np.percentile(st[0], 5):.0f}",
                      "budget (s/chunk)": "" if r is None or pd.isna(r.get("budget")) else f"{r.budget:.3g}",
                      "final λ": f"{r.final_lambda:.2f}" if r is not None and pd.notna(r.get("final_lambda")) and (m.startswith("ppo_lagrangian") or m.startswith("dara")) else "",
                      "DARA-L minus this [95% CI]": "" if p.empty else f"{p['diff'].iloc[0]:+.1f} [{p.ci_lo.iloc[0]:+.1f}, {p.ci_hi.iloc[0]:+.1f}]",
                      "verdict": "" if p.empty else VTXT.get(p.verdict.iloc[0], p.verdict.iloc[0])})
    tab = pd.DataFrame(rows_)
    pool = [m for m in ms if m in d.index and d.loc[m, "role"] not in EXCLUDE_ROLES]
    top3 = sorted(pool, key=lambda m: -float(d.loc[m, "J0"]))[:3]
    worst5 = {m: np.percentile(stats[m][0], 5) for m in ms if stats[m] is not None and m in MAIN}
    ans = [f"Top three by mean score: " + ", ".join(f"{mlabel(m)} ({float(d.loc[m, 'J0']):.1f})" for m in top3) + "."]
    if REF in d.index:
        r = d.loc[REF]
        ans.append(f"DARA-L: {r.J0:.1f} (seeds {', '.join(f'{x:.0f}' for x in lst(r.seed_means))}), freezing {r.freeze_s:.1f} s per "
                   f"session, average quality level {r.mean_tier:.2f} of 4, " +
                   (f"budget {r.budget:.3f} s per chunk, DARA acted in {r.dara_acted} seeds." if pd.notna(r.budget) else
                    f"DARA did not act ({r.dara_acted} seeds), so DARA-L trained as plain PPO."))
    if worst5:
        best_tail = max(worst5, key=worst5.get)
        ans.append(f"In the worst 5% of test sessions, {mlabel(best_tail)} holds up best ({worst5[best_tail]:.0f}); DARA-L: "
                   f"{worst5.get(REF, float('nan')):.0f}.")
    save(fig, f"10_env_{e}", f"{short_env(e)}: {R.TRAINING_ENVS.get(e, {}).get('label', e)}", ans,
         "Every method trained in this environment, scored on the same 100 test sessions. (a) mean and each seed; "
         "(b) the score split into quality earned (sum of quality levels minus switching penalties) and the cost of "
         "freezing (1.075 per second); (c) how often each quality level was chosen (q0 lowest, q4 highest); "
         "(d) how the score varies across the 100 test sessions. Plain PPO, cap c = the reward variants without DARA.",
         table=tab)

# ================================================================================================ C: training dynamics
PART = "How training unfolded"
GRIDX = np.arange(0, 10001, 100)


def curve(e, m, col):
    out = []
    for g in runs_table(e, m, "ppo_log.csv"):
        if col in g and "episode" in g and len(g) > 5:
            v = pd.Series(np.interp(GRIDX, g.episode, g[col], left=np.nan, right=np.nan)).rolling(5, min_periods=1).mean()
            out.append(v.to_numpy())
    return np.array(out) if out else None


def switch_eps(e):
    eps = []
    for dd in RUN_DIRS.get((e, REF), []):
        f = dd / "dara_log.csv"
        if f.exists():
            L = pd.read_csv(f)
            on = L[L.lagrange == 1]
            if len(on):
                eps.append(int(on.episode.iloc[0]))
    return eps


cm = [(REF, BLUE, "DARA-L"), ("ppo_lagrangian_t0.5", R.colour("ppo_lagrangian_t0.5"), "PPO-Lagrangian 0.5 s"),
      ("fixed", "#8a8986", "Plain PPO (cap 0.1, no fix)")]
if RUN_DIRS:
    for col, nm, yl, q, how in (
            ("mean_ref_20", "11_training_score", "Score J0 on training sessions",
             "How did the score develop during training?",
             "Moving average of J0 on the sessions the agent trains on (not the test sessions), mean of 5 seeds with the "
             "min–max range shaded. The grey band marks the episodes at which DARA switched the freeze limit on (earliest "
             "to latest seed)."),
            ("lam", "12_training_lambda", "Penalty weight λ on freezing",
             "How strongly was freezing penalised as training went on?",
             "λ is the weight the Lagrangian puts on freezing beyond the budget; it rises while the agent freezes more than "
             "allowed and falls when it freezes less. DARA-L's λ is 0 until DARA switches the limit on.")):
        ncol = 4; nrow = int(np.ceil(len(ENVS) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.3 * nrow), squeeze=False)
        notes_ = []
        for ax, e in zip(axes.ravel(), ENVS):
            sw = switch_eps(e)
            if sw:
                ax.axvspan(min(sw), max(sw) + 50, color="#e2e1dc", zorder=0)
            for m, c, lb in cm:
                if col == "lam" and m == "fixed":
                    continue
                cv = curve(e, m, col)
                if cv is None:
                    continue
                mu = np.nanmean(cv, axis=0)
                ax.fill_between(GRIDX, np.nanmin(cv, axis=0), np.nanmax(cv, axis=0), color=c, alpha=0.15, lw=0)
                ax.plot(GRIDX, mu, color=c, lw=1.6, label=lb)
            if col == "lam":
                lam_d, lam_p = curve(e, REF, "lam"), curve(e, "ppo_lagrangian_t0.5", "lam")
                if lam_d is not None and lam_p is not None:
                    notes_.append(f"{short_env(e)}: final λ {np.nanmean(lam_d[:, -1]):.1f} (DARA-L) vs {np.nanmean(lam_p[:, -1]):.1f}")
            ax.set_title(short_env(e), loc="left", fontweight="bold", fontsize=10)
            ax.grid(True, color=GRID); ax.set_axisbelow(True)
            if ax in axes[-1]:
                ax.set_xlabel("Training episode")
            if ax in axes[:, 0]:
                ax.set_ylabel(yl)
        for ax in axes.ravel()[len(ENVS):]:
            ax.axis("off")
        hs = [Line2D([], [], color=c, lw=2, label=lb) for m, c, lb in cm if not (col == "lam" and m == "fixed")]
        hs.append(Patch(fc="#e2e1dc", label="DARA switched the freeze limit on (range over seeds)"))
        fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        if col == "mean_ref_20":
            ans = ["DARA switched the limit on at episode " + "; ".join(
                f"{short_env(e)} {min(switch_eps(e))}–{max(switch_eps(e))}" for e in ENVS if switch_eps(e)) + " (of 10,000).",
                   "Before that, DARA-L trains without any freeze limit, exactly like plain PPO."]
            dips, gaps = [], []
            for e in ENVS:
                cd, cp = curve(e, REF, col), curve(e, "ppo_lagrangian_t0.5", col)
                sw = switch_eps(e)
                if cd is None or cp is None or not sw:
                    continue
                pre = GRIDX < min(sw)
                lo_d, lo_p = np.nanmin(np.nanmean(cd, 0)[pre]), np.nanmin(np.nanmean(cp, 0)[pre])
                if lo_d < lo_p - 50:
                    dips.append(f"{short_env(e)} {int(round(lo_d))} vs {int(round(lo_p))}")
                last = GRIDX >= 9000
                gaps.append((e, np.nanmean(np.nanmean(cd, 0)[last]) - np.nanmean(np.nanmean(cp, 0)[last])))
            if dips:
                ans.append("In that unconstrained phase DARA-L's training score falls far below PPO-Lagrangian's (lowest "
                           "point: " + ", ".join(dips) + "): the agent learns to stream at high quality and freeze.")
            if gaps:
                ans.append("Over the last 1,000 episodes DARA-L minus PPO-Lagrangian 0.5 s on training sessions: " +
                           ", ".join(f"{short_env(e)} {g:+.0f}" for e, g in gaps) + ".")
        else:
            big = []
            for e in ENVS:
                cd, cp = curve(e, REF, "lam"), curve(e, "ppo_lagrangian_t0.5", "lam")
                if cd is not None and cp is not None and np.nanmean(cd[:, -1]) > 2 * max(np.nanmean(cp[:, -1]), 0.5):
                    big.append(short_env(e))
            ans = notes_ + ([f"DARA-L's final λ is more than twice PPO-Lagrangian's in {', '.join(big)}. A large λ makes the "
                             "agent avoid freezing at the cost of quality (exploratory reading, not a pre-registered test)."]
                            if big else [])
        save(fig, nm, q, ans, how)

# ================================================================================================ D: statistics
PART = "Statistics in full"
if TS is not None:
    ncol = 2; nrow = int(np.ceil(len(ENVS) / ncol))
    hh = [max(2.2, 0.32 * len(TS[TS.env == e]) + 1.0) for e in ENVS]
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, sum(max(hh[i:i + 2]) for i in range(0, len(hh), 2))), squeeze=False,
                             gridspec_kw=dict(height_ratios=[max(hh[i:i + 2]) for i in range(0, len(hh), 2)]))
    for ax, e in zip(axes.ravel(), ENVS):
        p = TS[TS.env == e].copy()
        p["key"] = p.comparison.str.replace(r"best_single_tier \((tier\d)\)", r"\1", regex=True)
        p = p.sort_values("key", key=lambda s: s.map(order_key) if True else s)
        yy = np.arange(len(p))
        for i, r in enumerate(p.itertuples()):
            c = VCOL.get(r.verdict, MUTED)
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=2)
            ax.scatter(r.diff, i, color=c, s=40, zorder=3, marker="o" if r.prereg else "s")
            ax.text(1.0, i, f" {r.share_sessions_better * 100:.0f}% sessions, q={r.q_session:.3f}"
                    + (f", p_seed={r.p_seed:.3f}" if pd.notna(r.p_seed) else ""), transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(yy); ax.set_yticklabels([mlabel(k) for k in p.key], fontsize=8.5); ax.invert_yaxis()
        ax.set_title(short_env(e), loc="left", fontweight="bold", fontsize=10)
        ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
        ax.set_xlabel("DARA-L minus method (J0), 95% interval")
    for ax in axes.ravel()[len(ENVS):]:
        ax.axis("off")
    hs = [Line2D([], [], color=VCOL[k], marker="o", lw=2, label=VTXT[k]) for k in VCOL]
    hs.append(Line2D([], [], color=MUTED, marker="s", ls="", label="square: comparison not pre-registered"))
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 0.93, 1), w_pad=14)
    full = TS.copy()
    full["environment"] = full.env.map(short_env); full["method"] = full.comparison.map(
        lambda c: mlabel(c.split("(")[1].rstrip(")")) if c.startswith("best_single_tier") else mlabel(c))
    full = full[["environment", "method", "diff", "ci_lo", "ci_hi", "diff_median", "share_sessions_better", "p_session",
                 "q_session", "p_seed", "prereg", "verdict"]].round(4)
    full.columns = ["environment", "compared with", "mean diff", "CI low", "CI high", "median session diff",
                    "share of sessions DARA-L better", "p (sessions)", "q (BH)", "p (seeds)", "pre-registered", "verdict"]
    cnt = TS.verdict.value_counts()
    save(fig, "20_statistics", "Every comparison with its confidence interval", [
        f"{len(TS)} comparisons: DARA-L better in {cnt.get('DARA-L better', 0)}, worse in {cnt.get('DARA-L worse', 0)}, no "
        f"difference in {cnt.get('no significant difference', 0)}, unclear in {cnt.get('inconclusive (tests disagree)', 0)}.",
        "An interval that crosses 0 means the 5 seeds disagree too much to be sure of the direction, even when the "
        "per-session test is significant: that is what most 'unclear' verdicts are."],
        "Dot = DARA-L's mean score minus the method's (positive = DARA-L better); line = 95% bootstrap interval over "
        "seeds and sessions. Right: share of the 100 sessions where DARA-L scored higher, the per-session test (q, "
        "corrected for multiple comparisons) and the test over seed means (p_seed).", table=full)

# ================================================================================================ E: budgets and reward caps
PART = "Budgets and reward settings"
benvs = [e for e in ENVS if ((TT.env == e) & TT.method.str.match(r"ppo_lagrangian_t[0-9.]+$")).any()]
if benvs:
    ncol = 4; nrow = int(np.ceil(len(benvs) / ncol))
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(17, 5.2 * nrow), squeeze=False)
    for k, e in enumerate(benvs):
        d = TT[TT.env == e].set_index("method")
        hand = d[d.index.str.match(r"ppo_lagrangian_t[0-9.]+$")].copy()
        hand["b"] = hand.index.str.extract(r"_t([0-9.]+)$", expand=False).astype(float).to_numpy()
        hand = hand.sort_values("b")
        for j, (col, yl) in enumerate((("J0", "Score J0"), ("freeze_s", "Freezing s/session"))):
            ax = axes[2 * (k // ncol) + j, k % ncol]
            ax.errorbar(hand.b.to_numpy(float), hand[col].to_numpy(float),
                        yerr=hand.J0_seed_sd.to_numpy(float) if col == "J0" else None, fmt="-o", color=R.colour("ppo_lagrangian_t0.5"),
                        ms=5, capsize=3, lw=1.5)
            if REF in d.index and pd.notna(d.loc[REF, "budget"]):
                ax.errorbar([float(d.loc[REF, "budget"])], [float(d.loc[REF, col])],
                            yerr=[float(d.loc[REF, "J0_seed_sd"])] if col == "J0" else None, fmt="D", ms=8, color=BLUE, mec=SURF, capsize=3)
            elif REF in d.index:
                ax.axhline(float(d.loc[REF, col]), color=BLUE, ls=":", lw=1.5)
            if "robustmpc" in d.index:
                ax.axhline(float(d.loc["robustmpc", col]), color=R.colour("robustmpc"), ls="--", lw=1)
            ax.set_xscale("log", base=2)
            xs = sorted(set(hand.b)); ax.set_xticks(xs); ax.set_xticklabels([f"{b:g}" for b in xs]); ax.minorticks_off()
            xa = xs + ([float(d.loc[REF, "budget"])] if REF in d.index and pd.notna(d.loc[REF, "budget"]) else [])
            ax.set_xlim(min(xa) / 1.4, max(xa) * 1.5)
            ax.grid(True, color=GRID); ax.set_axisbelow(True)
            if j == 0:
                ax.set_title(short_env(e), loc="left", fontweight="bold", fontsize=10)
            ax.set_ylabel(yl, fontsize=9)
            if j == 1:
                ax.set_xlabel("Freeze budget (s per chunk)", fontsize=9)
    for k in range(len(benvs), nrow * ncol):
        axes[2 * (k // ncol), k % ncol].axis("off"); axes[2 * (k // ncol) + 1, k % ncol].axis("off")
    hs = [Line2D([], [], color=R.colour("ppo_lagrangian_t0.5"), marker="o", label="PPO-Lagrangian, hand-picked budget (± seed s.d.)"),
          Line2D([], [], color=BLUE, marker="D", ls="", label="DARA-L at its derived budget"),
          Line2D([], [], color=BLUE, ls=":", label="DARA-L when DARA did not act"),
          Line2D([], [], color=R.colour("robustmpc"), ls="--", label="RobustMPC")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tab = None
    if BT is not None:
        tab = BT.assign(environment=BT.env.map(short_env))[["environment", "derived_budget", "dara_acted", "J0_daral",
                                                              "best_hand_budget", "J0_best_hand", "gap_rel", "n_hand",
                                                              "J0_hand_curve_at_derived", "J0_by_hand_budget"]].round(3)
        tab.columns = ["environment", "derived budget", "DARA acted", "DARA-L J0", "best hand budget", "its J0", "gap",
                       "hand budgets", "hand curve at derived budget (interpolated)", "J0 per hand budget"]
    ans = []
    if BT is not None:
        for r in BT.itertuples():
            if pd.notna(r.J0_hand_curve_at_derived):
                ans.append(f"{short_env(r.env)}: DARA-L {r.J0_daral:.0f}; a hand-picked budget equal to DARA-L's would score about "
                           f"{r.J0_hand_curve_at_derived:.0f} (interpolated), the best hand-picked one {r.J0_best_hand:.0f}.")
    save(fig, "30_budget_curves", "Score and freezing for every hand-picked budget", ans +
         ["Where DARA-L sits below the interpolated curve, the loss comes from how DARA-L trains (limit switched on late), not "
          "from the budget value itself."],
         "Top row of each pair: score; bottom row: freezing. The interpolated value is a straight line between the two "
         "nearest hand-picked budgets, so it is only a rough guide.", table=tab)

if DV is not None and len(DV):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    capx = lambda c: "∞ (no cap)" if c >= 1e5 else f"{c:g}"
    caps = sorted(DV.cap.unique()); xpos = {c: i for i, c in enumerate(caps)}
    ramp = {s: c for s, c in zip(sorted(DV.ladder_scale.unique()), ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])}
    for sc, g in DV.groupby("ladder_scale"):
        g = g.sort_values("cap")
        axes[0].plot([xpos[c] for c in g.cap], g.M, "-o", color=ramp[sc], label=f"ladder ×{sc:g}")
        axes[1].plot([xpos[c] for c in g.cap], g.J0, "-o", color=ramp[sc], label=f"ladder ×{sc:g}")
        e = g.env.iloc[0]
        axes[1].axhline(g.J0_best_tier.iloc[0], color=ramp[sc], ls=":", lw=1)
        if (e, REF) and e in SIDX.index and pd.notna(SIDX.loc[e, f"{REF}_J0"]):
            axes[1].scatter([len(caps) - 0.5], [SIDX.loc[e, f"{REF}_J0"]], marker="D", s=60, color=ramp[sc], edgecolor=INK, zorder=4)
    axes[0].axhline(THR, color=INK2, ls="--", lw=1); axes[0].text(0, THR + 0.01, f"DARA acts above {THR:g}", fontsize=8.5, color=INK2)
    for ax in axes:
        ax.set_xticks(range(len(caps))); ax.set_xticklabels([capx(c) for c in caps])
        ax.set_xlabel("Cap on the freezing penalty in the training reward (smaller = freezing costs less)")
        ax.grid(True, color=GRID); ax.set_axisbelow(True)
    axes[1].set_xlim(-0.4, len(caps) - 0.1)
    axes[0].set_ylabel("Misalignment M"); axes[1].set_ylabel("Score J0 of plain PPO")
    axes[0].legend(frameon=False); axes[1].legend(handles=axes[1].get_legend_handles_labels()[0] + [
        Line2D([], [], color=INK2, ls=":", label="best fixed quality"),
        Line2D([], [], marker="D", ls="", color="#86b6ef", mec=INK, label="DARA-L (right edge)")], frameon=False, fontsize=8.5)
    best_nc = DV[DV.cap >= 1e5]
    ans = ["Loosening or removing the cap lowers M and, in step, raises the plain-PPO score: the cap is what makes the reward "
           "ignore freezing."]
    if len(best_nc):
        ans.append("With no cap at all: " + ", ".join(f"ladder ×{r.ladder_scale:g} M {r.M:.2f}, J0 {r.J0:.0f}" for r in best_nc.itertuples()) + ".")
    tab = DV.assign(environment=DV.env.map(short_env), cap=DV.cap.map(capx))[["environment", "cap", "M", "D", "J0", "J0_best_tier", "shortfall", "flagged"]].round(3)
    save(fig, "31_reward_caps", "What does the cap on the freezing penalty do to the reward and to training?", ans,
         "Left: misalignment M before training for each cap. Right: the score plain PPO reaches with that reward; dotted "
         "lines are the best fixed quality on each ladder; diamonds at the right edge are DARA-L (which uses cap 0.1).", table=tab)

# ================================================================================================ F: unseen networks in detail
PART = "Unseen networks in detail"
if GS is not None and len(GS):
    nets = [n for n in R.GEN_NETWORKS if n in set(GS.network)]
    lads = [c for c in ("x1", "x4", "x8") if any(GS.condition.str.endswith("/" + c))]
    meths = [m for m in MAIN if m in set(GS.method)]
    fig, axes = plt.subplots(len(nets), 3, figsize=(16, 3.0 * len(nets)), squeeze=False)
    for i, nnet in enumerate(nets):
        for j, (col, yl) in enumerate((("J0", "Score J0"), ("freeze_s", "Freezing s/session"), ("mean_tier", "Mean quality level"))):
            ax = axes[i, j]
            for m in meths:
                g = GS[(GS.network == nnet) & (GS.method == m)].set_index("condition")
                ys = [g.loc[f"{nnet}/{c}", col] if f"{nnet}/{c}" in g.index else np.nan for c in lads]
                ax.plot(range(len(lads)), ys, "-", marker=MARK[m], color=R.colour(m), ms=8 if m == REF else 6, lw=1.6 if m == REF else 1.1, label=SHORT[m])
            if col == "J0":
                bt = GS[(GS.network == nnet) & GS.method.str.fullmatch(r"tier[0-4]")]
                ys = [bt[bt.condition == f"{nnet}/{c}"].J0.max() for c in lads]
                ax.plot(range(len(lads)), ys, ":", color=GREY, lw=1.5, label="best fixed quality")
            ax.set_xticks(range(len(lads))); ax.set_xticklabels([f"ladder {R.CONTENTS[c]}" for c in lads])
            if col == "freeze_s" and GS[(GS.network == nnet) & GS.method.isin(meths)].freeze_s.max() > 200:
                ax.set_yscale("log")
            ax.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_ylabel(yl, fontsize=9)
            if j == 0:
                ax.set_title(R.NETWORKS[nnet]["label"], loc="left", fontweight="bold", fontsize=10)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    tab = GS[GS.method.isin(meths + [m for m in GS.method.unique() if m.startswith("tier")])].copy()
    tab["method"] = tab.method.map(mlabel)
    tab = tab[["condition", "method", "J0", "J0_seed_sd", "freeze_s", "mean_tier"]].round(2)
    ans = []
    for nnet in nets:
        g = GS[(GS.network == nnet) & GS.method.isin(meths)]
        best = g.loc[g.groupby("condition").J0.idxmax()]
        ans.append(f"{R.NETWORKS[nnet]['label']}: best per ladder: " + ", ".join(
            f"{R.CONTENTS[c.split('/')[1]]} {SHORT[m]}" for c, m in zip(best.condition, best.method)) + ".")
    save(fig, "40_unseen_detail", "Score, freezing and quality on every unseen network and ladder", ans,
         "Models trained on the mmWave generator with ladder ×4, tested without retraining. Each row is one test network, "
         "each column one measure; the x-axis is the chunk-size ladder.", table=tab)
if GT is not None and len(GT):
    comps = [c for c in ["ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc"] if c in set(GT.comparison)]
    conds = [f"{n}/{c}" for n in R.GEN_NETWORKS for c in ("x1", "x4", "x8") if f"{n}/{c}" in set(GT.condition)]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(conds) + 1.4))

    def gcell(cnd, comp):
        g = GT[(GT.condition == cnd) & (GT.comparison == comp)]
        return (None, None) if g.empty else (g.verdict.iloc[0], float(g["diff"].iloc[0]))
    grid_chart(ax, conds, comps, gcell, [f"{R.NETWORKS[c.split('/')[0]]['label']}, ladder {R.CONTENTS[c.split('/')[1]]}" for c in conds],
               [SHORT[c].replace(" budget", "") for c in comps])
    fig.tight_layout(rect=(0, 0.06, 1, 1)); grid_legend(fig)
    tab = GT[GT.comparison.isin(comps)].assign(method=lambda x: x.comparison.map(lambda c: SHORT[c]))[
        ["condition", "method", "diff", "ci_lo", "ci_hi", "share_sessions_better", "q_session", "p_seed", "verdict"]].round(4)
    save(fig, "41_unseen_significance", "On unseen networks, which differences are significant?", [
        f"{(GT[GT.comparison.isin(comps)].verdict == 'DARA-L better').sum()} significantly better, "
        f"{(GT[GT.comparison.isin(comps)].verdict == 'DARA-L worse').sum()} significantly worse, out of "
        f"{len(GT[GT.comparison.isin(comps)])} comparisons."],
        "As chart 7, for the unseen test conditions.", table=tab)

# ================================================================================================ G: the networks
PART = "The networks"
NS, LF = read("data/network_stats.csv"), read("data/ladder_feasibility.csv")
if NS is not None:
    keep = NS[~NS.network.str.startswith("generator: ") | NS.network.str.contains("training mix|tunnel|urban_canyon|suburban")]
    keep = keep.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(keep) + 1.4))
    for i, r in enumerate(keep.itertuples()):
        c = {"mmwave_synthetic": BLUE, "mmwave_real": "#1baf7a", "non_mmwave_5g": "#eb6834"}.get(r.family, MUTED)
        ax.plot([r.p10, r.p90], [i, i], color=c, lw=3, alpha=0.45, solid_capstyle="round")
        ax.scatter(r.median, i, color=c, s=60, zorder=3, edgecolor=SURF)
        ax.text(1.0, i, f" median {r.median:.0f}, below 5 Mbit/s {100 * r.share_below_5:.0f}% of the time",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    for k_, (b, lb) in enumerate(((3, "q0 ×1"), (44, "q4 ×1"), (176, "q4 ×4"), (352, "q4 ×8"))):
        ax.axvline(b, color=GRID, lw=1, ls="--")
        ax.text(b, len(keep) - 0.4 + 0.45 * (k_ % 2), lb, fontsize=7.5, color=MUTED, ha="center")
    ax.set_xscale("log"); ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep.network.str.replace("generator: ", "Generator: "))
    ax.set_xlabel("Throughput (Mbit/s, log scale): dot = median, bar = 10th to 90th percentile")
    ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    gm = NS[NS.network == "generator: training mix"]
    lm, ir = NS[NS.network == "Lumos5G: all"], NS[NS.network == "Irish 5G: driving"]
    ans = []
    if len(gm) and len(lm):
        ans.append(f"Median throughput: training generator {gm['median'].iloc[0]:.0f} Mbit/s, Lumos5G {lm['median'].iloc[0]:.0f}, "
                   + (f"Irish 5G driving {ir['median'].iloc[0]:.0f}." if len(ir) else ""))
    ans.append("Dashed lines: bitrate of the lowest and highest quality on each ladder. Where most of a network's bar is left "
               "of a quality's line, that quality cannot be sustained.")
    save(fig, "50_networks", "How fast are the networks used for training and testing?", ans,
         "Generator rows are synthetic sessions of the mmWave generator (training mix and selected scenarios); Lumos5G is "
         "real mmWave; Irish 5G is real mid-band 5G.", table=NS.round(2))
if LF is not None:
    nets_ = [n for n in LF.network.unique() if not n.startswith("generator: ") or "training mix" in n or "tunnel" in n]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.4 * len(nets_) + 1.6), sharey=True)
    for ax, lad in zip(axes, ("x1", "x4", "x8")):
        g = LF[LF.ladder == lad].set_index("network").reindex(nets_).dropna(axis=1, how="all").drop(columns="ladder", errors="ignore")
        M_ = g.to_numpy(float)
        ax.imshow(M_, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M_):
            if np.isfinite(v):
                ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(M_.shape[1])); ax.set_xticklabels([c.replace("tier", "q").replace("_", "\n") for c in g.columns], fontsize=8.5)
        ax.set_title(f"ladder {R.CONTENTS.get(lad, lad)}", loc="left", fontweight="bold", fontsize=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(nets_))); axes[0].set_yticklabels([n.replace("generator: ", "Generator: ") for n in nets_])
    fig.tight_layout()
    save(fig, "51_feasibility", "How often can each network carry each quality level?", [
        "Share of seconds in which the network is at least as fast as the quality level's bitrate. Bigger ladders push every "
        "quality's bitrate up, so the shares shrink from left to right: this is why ×8 is hard."],
        "Rows: networks; columns: the five quality levels with their bitrate (Mbit/s) on that ladder.", table=None)


# ================================================================================================ bottom line
bottom = []
if SV is not None and len(SV):
    r = SV.iloc[0]
    bottom.append(("The diagnosis works.", f"Misalignment M, measured before training, predicts how badly training will "
                   f"go (rank correlation {r.rho:.2f}, interval {r.ci_lo:.2f}–{r.ci_hi:.2f})."))
quiet = S[(S.M < THR) & S.dara_acted.astype(str).str.startswith("0/")]
if len(quiet):
    bottom.append(("No false alarms.", f"Where M is below {THR:g} ({', '.join(short_env(e) for e in quiet.env)}), DARA "
                   "never intervened, and there was nothing to fix."))
if "gap_to_best_hand" in S:
    d = S[S.gap_to_best_hand.notna() & S.derived_budget.notna()]
    ok = d[d.gap_to_best_hand >= -TOL]
    bottom.append(("The automatic budget works on moderate problems, not on the hardest.",
                   f"Within {100 * TOL:.0f}% of the best hand-tuned budget in {len(ok)} of {len(d)} environments; it is too "
                   f"strict on {', '.join(short_env(e) for e in d[d.gap_to_best_hand < -TOL].env) or 'none'}."))
if TS is not None:
    v = TS[TS.env.isin(ENVS) & (TS.comparison == "ppo_lagrangian_t0.5")].verdict.value_counts()
    bottom.append(("Not better than a well-chosen fixed budget.", f"Against PPO-Lagrangian with a 0.5 s budget, DARA-L is "
                   f"significantly better in {v.get('DARA-L better', 0)} environments, worse in {v.get('DARA-L worse', 0)}, "
                   f"and the tests are unclear in the other {int(v.sum()) - v.get('DARA-L better', 0) - v.get('DARA-L worse', 0)}. "
                   "Its advantage is that it needs no tuning."))
    for c, nm in (("pensieve_official", "Pensieve"), ("robustmpc", "RobustMPC"), ("corel_official", "COREL")):
        v = TS[TS.env.isin(ENVS) & (TS.comparison == c)].verdict.value_counts()
        bottom.append((f"vs {nm}:", f"better in {v.get('DARA-L better', 0)} of {int(v.sum())} environments, worse in "
                       f"{v.get('DARA-L worse', 0)}."))

# ================================================================================================ write report
def esc(s):
    return html.escape(str(s))


css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--card:#ffffff;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#222220}}
body{background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:1240px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px} h2{font-size:21px;margin:48px 0 6px} .sub{color:var(--ink2);margin:0 0 24px}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:16px 0}
.box li{margin:6px 0} .ans{border-left:4px solid var(--accent);padding:4px 0 4px 14px;margin:10px 0}
.ans p{margin:4px 0} .how{color:var(--ink2);font-size:14px;margin:8px 0 12px}
img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fcfcfb}
table{border-collapse:collapse;font-size:14px;width:100%} td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
th{color:var(--ink2);font-weight:600} .num{text-align:right}
h1.part{font-size:24px;margin:72px 0 0;padding-top:16px;border-top:2px solid var(--line)}
details{margin:10px 0 0} summary{cursor:pointer;color:var(--accent);font-size:14px}
.tw{overflow-x:auto;max-height:520px;overflow-y:auto} .tw table{font-size:12.5px;white-space:nowrap}
.warn{border-color:#eda100;background:rgba(237,161,0,.08)} .toc li{margin:3px 0} .toc ol{font-size:14px} a{color:var(--accent)}
"""
glossary = [
    ("The task", "A video player picks one of five quality levels for each 4-second chunk of video while the 5G connection "
                 "speeds up and slows down. Higher quality looks better but takes longer to download; if the download is too "
                 "slow, the video freezes."),
    ("Score J0", "One number per viewing session, higher is better: the quality of every chunk added up, minus 1.075 per "
                 "second of freezing, minus a small penalty for each jump in quality. Every method is scored on the same 100 "
                 "test sessions."),
    ("The problem", "Learning agents are trained on a reward that only approximates J0. If the reward still pays for top "
                    "quality when the network cannot deliver it, the agent learns to freeze."),
    ("DARA", "A check run before and during training: it measures how often the training reward prefers a quality that J0 "
             "rates badly (misalignment M, 0 = never, 1 = always)."),
    ("DARA-L (ours)", f"If M ≥ {THR:g}, DARA adds a limit on freezing (a Lagrangian constraint) with a budget computed from the "
                      "check itself, no tuning."),
    ("PPO-Lagrangian", "The same learner with the freezing budget picked by hand (e.g. 0.5 s per chunk)."),
    ("COREL, Pensieve", "Published learning-based methods (their released algorithms). COREL is given a near-perfect "
                        "forecast of the network, as in its released code."),
    ("RobustMPC", "A classic, non-learning controller."),
    ("Ladder ×1 … ×8", "How large the video chunks are. ×8 chunks are 8 times bigger than ×1, so the same network can "
                       "deliver fewer high-quality chunks: a harder problem."),
    ("Seeds", "Every learned method was trained 5 times with different random seeds; results are averages."),
]
envtab = ["<table><tr><th>Environment</th><th>Network</th><th class=num>M</th><th>DARA acted</th>"
          "<th class=num>Derived budget (s/chunk)</th></tr>"]
for r in S.itertuples():
    envtab.append(f"<tr><td>{esc(short_env(r.env))}</td><td>{esc(R.TRAINING_ENVS.get(r.env, {}).get('label', r.env))}</td>"
                  f"<td class=num>{'' if pd.isna(r.M) else f'{r.M:.2f}'}</td><td>{esc(r.dara_acted)}</td>"
                  f"<td class=num>{'–' if pd.isna(r.derived_budget) else f'{r.derived_budget:.2f}'}</td></tr>")
envtab.append("</table>")

def table_html(t):
    if t is None or len(t) == 0:
        return ""
    return ("<details><summary>All numbers (" + str(len(t)) + " rows)</summary><div class=tw>" +
            t.to_html(index=False, border=0, na_rep="", escape=True) + "</div></details>")


parts = list(dict.fromkeys(s_["part"] for s_ in sections))
toc = "".join(f"<li><a href='#p{k}'>{esc(p_)}</a><ol>" + "".join(
    f"<li><a href='#s{s_['name']}'>{esc(s_['q'])}</a></li>" for s_ in sections if s_["part"] == p_) + "</ol></li>"
    for k, p_ in enumerate(parts))
H = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
     f"<title>DARA-L results</title><style>{css}</style></head><body><main>",
     "<h1>DARA-L results</h1>",
     f"<p class=sub>Generated from <code>results/</code> and <code>runs/</code> by <code>scripts/analysis/make_overview.py</code>. "
     f"{len(ENVS)} training environments, 5 training seeds per learned method, 100 test sessions each.</p>",
     (f"<div class='box warn'><b>Sensitivity view, not the pre-registered analysis.</b> {esc(PR['deviation'])}"
      + "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in PR.get("view_log", [])) + "</ul></div>" if PR.get("deviation") else ""),
     "<div class=box><b>Bottom line</b><ul>" + "".join(f"<li><b>{esc(h)}</b> {esc(t)}</li>" for h, t in bottom) + "</ul></div>",
     "<div class=box><b>Contents</b><ol class=toc>" + toc + "</ol></div>",
     "<h2>The words used here</h2><div class=box><ul>" +
     "".join(f"<li><b>{esc(k)}.</b> {esc(v)}</li>" for k, v in glossary) + "</ul></div>",
     "<h2>The training environments</h2>", "".join(envtab)]
md = ["# DARA-L results", "", "Generated by `scripts/analysis/make_overview.py`. Open `index.html` for the formatted "
      "report with every table.", ""] + ([f"> **Sensitivity view, not the pre-registered analysis.** {PR['deviation']}", ""]
                                          if PR.get("deviation") else []) + [ "## Bottom line", ""] + [f"* **{h}** {t}" for h, t in bottom] + \
     ["", "## The words used here", ""] + [f"* **{k}.** {v}" for k, v in glossary]
n_ = 0
for k, p_ in enumerate(parts):
    H.append(f"<h1 class=part id='p{k}'>{esc(p_)}</h1>")
    md += ["", f"## {p_}", ""]
    for s_ in [x for x in sections if x["part"] == p_]:
        n_ += 1
        img = base64.b64encode((OUT / f"{s_['name']}.png").read_bytes()).decode()
        H += [f"<h2 id='s{s_['name']}'>{n_}. {esc(s_['q'])}</h2>",
              "<div class=ans>" + "".join(f"<p>{esc(x)}</p>" for x in s_["ans"]) + "</div>",
              f"<p class=how><b>How to read it.</b> {esc(s_['how'])}</p>",
              f"<img loading=lazy alt='{esc(s_['q'])}' src='data:image/png;base64,{img}'>", table_html(s_["table"])]
        md += [f"### {n_}. {s_['q']}", ""] + [f"> {x}" for x in s_["ans"]] + ["", f"*How to read it.* {s_['how']}", "",
                                                                               f"![{s_['q']}]({s_['name']}.png)", ""]
        if s_["table"] is not None and len(s_["table"]):
            s_["table"].to_csv(OUT / f"{s_['name']}.csv", index=False)
            md += [f"Numbers: [{s_['name']}.csv]({s_['name']}.csv)", ""]
H.append("</main></body></html>")
(OUT / "index.html").write_text("\n".join(H), encoding="utf-8")
(OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
print("written", OUT / "index.html")
