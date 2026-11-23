#!/usr/bin/env python3
"""IEEE-format result figures (Times, 8 pt, single column 3.5 in / double column 7.16 in, no titles, vector PDF + PNG),
using standard chart types: grouped bars with error bars, CDFs, line plots with shaded spread, scatter.

  fig_qoe_bars        normalised QoE (J0 / best result in that environment), every training environment     double
  fig_qoe_components  (a) mean J0 per session, (b) freeze time per session, every training environment          double
  fig_qoe_cdf         CDF of per-session QoE in three environments                                           double
  fig_diagnosis       misalignment M before training vs trained shortfall (claim C1)                         single
  fig_budget_sweep    QoE vs freeze budget: hand-picked budgets and DARA-L's derived budget                    double
  fig_training        QoE and penalty weight lambda during training                                          double
  fig_unseen          normalised QoE on unseen mmWave conditions (models trained on mmWave x4)                double
  fig_tradeoff        mean quality level vs freeze time, primary environment, seeds and means                 single
  captions.md         suggested caption for each figure, with the key numbers computed from the results

Inputs : results/training, results/generalisation, runs/train (CDF, training curves); see make_overview.py.
Usage  : python scripts/analysis/make_paper_figures.py [--results results] [--runs runs/train] [--prereg ...] [--out figures/paper]
         For the threshold-0.15 view: --results results/views/t015 --runs runs/views/t015/train
                                      --prereg configs/preregistration/view_t015.json --out figures/paper_t015
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Tight layout not applied")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401  (older matplotlib + pandas 2)

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "figures" / "paper"))
ap.add_argument("--exclude-envs", default="", help="comma-separated training environments to leave out of every figure")
ap.add_argument("--no-best-tier", action="store_true", help="do not draw the best always-one-tier rule")
ap.add_argument("--only", default="", help="comma-separated figure names to write (default: all)")
a = ap.parse_args()
EXCL = {e for e in a.exclude_envs.split(",") if e}
ONLY = {f for f in a.only.split(",") if f}
RES, RUNS, OUT = Path(a.results), Path(a.runs), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
PR = json.load(open(a.prereg))
REF = PR["proposed_method"]
THR = PR.get("act_threshold", {}).get("primary", 0.2)
PRIMARY = PR.get("primary_training_environment", "ct4")

# ------------------------------------------------------------------------------------------------ IEEE style
W1, W2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 1.6,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "errorbar.capsize": 1.5})
INK, INK2 = "#0b0b0b", "#52514e"
MAIN = [REF, "ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc"]
NAME = {REF: "DARA-L", "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)", "corel_official": "COREL", "pensieve_official": "Pensieve",
        "robustmpc": "RobustMPC", "best_tier": "Best fixed tier", "fixed": "PPO (original reward)"}
COL = {m: R.colour(m) for m in MAIN}
COL["best_tier"] = "#a9a8a4"; COL["fixed"] = "#52514e"
HATCH = {REF: "", "ppo_lagrangian_t0.5": "////", "corel_official": "\\\\\\\\", "pensieve_official": "....",
         "robustmpc": "xxxx", "best_tier": "----"}
MARK = {REF: "D", "ppo_lagrangian_t0.5": "o", "corel_official": "s", "pensieve_official": "^", "robustmpc": "v",
        "best_tier": "x", "fixed": "P"}
LS = {REF: "-", "ppo_lagrangian_t0.5": "--", "corel_official": "-.", "pensieve_official": ":", "robustmpc": (0, (5, 1, 1, 1, 1, 1)),
      "fixed": (0, (1, 1))}
ENV_SHORT = {"ct1": "×1", "ct2": "×2", "ct4": "×4", "ct8": "×8", "cth4": "Harsh ×4", "ctl": "Lumos5G ×4",
             "ctd1": "Irish dr. ×1", "cts1": "Irish st. ×1"}
captions = []


def read(p):
    p = RES / p
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None
    return pd.read_csv(p)


def save(fig, name, caption):
    if ONLY and name not in ONLY:
        plt.close(fig)
        return
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    captions.append((name, caption))
    print("written", OUT / f"{name}.pdf")


def sub(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")


TT, TS = read("training/training_table.csv"), read("training/training_tests.csv")
BT, DV, SV = read("training/budget_test.csv"), read("training/diagnosis_validity.csv"), read("training/diagnosis_validity_summary.csv")
GS = read("generalisation/gen_table.csv")
if TT is None:
    sys.exit("results/training/training_table.csv is needed")

RUN_DIRS = {}
for d in sorted(RUNS.glob("*__*_s*")) if RUNS.exists() else []:
    p = R.parse_run(d.name)
    if p:
        RUN_DIRS.setdefault((p["env"], p["method"]), []).append(d)


def env_M(e):
    ms = [float(pd.read_csv(d / "dara_log.csv").M.iloc[0]) for d in RUN_DIRS.get((e, REF), []) if (d / "dara_log.csv").exists()]
    if ms:
        return float(np.mean(ms))
    if DV is not None and ((DV.env == e) & (DV.method == "fixed")).any():
        return float(DV[(DV.env == e) & (DV.method == "fixed")].M.iloc[0])
    return np.nan


ENVS = sorted(TT.env.unique(), key=lambda e: (np.nan_to_num(env_M(e), nan=9), e))
ENVS = [e for e in ENVS if e not in EXCL]
if EXCL:
    TT = TT[~TT.env.isin(EXCL)]
    if BT is not None:
        BT = BT[~BT.env.isin(EXCL)]
    if DV is not None and DV.env.isin(EXCL).any():
        # C1 over the remaining reward settings, computed exactly as in evaluate_training.py
        from scipy.stats import spearmanr
        DV = DV[~DV.env.isin(EXCL)].reset_index(drop=True)
        rho = spearmanr(DV.M, DV.shortfall).correlation
        rng_ = np.random.default_rng(0); boots = []
        seeds_ = [np.array(json.loads(x)) for x in DV.seed_shortfalls]
        for _ in range(2000):
            boots.append(spearmanr(DV.M, [rng_.choice(v, len(v)).mean() for v in seeds_]).correlation)
        lo_, hi_ = np.nanpercentile(boots, [2.5, 97.5])
        SV = pd.DataFrame([dict(rho=rho, ci_lo=lo_, ci_hi=hi_, n=len(DV))])
        print(f"C1 without {sorted(EXCL)}: rho = {rho:.3f} [{lo_:.3f}, {hi_:.3f}], {len(DV)} settings")
BEST = [] if a.no_best_tier else ["best_tier"]


def mpc_tier(e):
    """RobustMPC's mean quality level, from the tier shares logged with any run in the environment"""
    for (ee, _), dirs in RUN_DIRS.items():
        if ee == e:
            f = dirs[0] / "triage_summary.csv"
            if f.exists():
                t = pd.read_csv(f)
                if "mpc_tier_share" in t:
                    return float(np.dot(json.loads(t.mpc_tier_share.iloc[0]), range(5)))
    return np.nan


def cell(e, m, col="J0"):
    d = TT[TT.env == e].set_index("method")
    if m == "best_tier":
        t = d[d.index.str.fullmatch(r"tier[0-4]")]
        if not len(t):
            return np.nan
        k = t.J0.idxmax()
        return float(k[-1]) if col == "mean_tier" else float(t[col].loc[k])
    if m == "robustmpc" and col == "mean_tier":
        return mpc_tier(e)
    return float(d.loc[m, col]) if m in d.index else np.nan


def best_any(e):
    d = TT[(TT.env == e) & ~TT.role.isin(["ablation", "diagnosis"])]
    return float(d.J0.max())


def grouped_bars(ax, groups, methods, value, err=None, width=0.8):
    k = len(methods); w = width / k
    for j, m in enumerate(methods):
        x = np.arange(len(groups)) + (j - (k - 1) / 2) * w
        v = np.array([value(g, m) for g in groups], float)
        e = np.array([err(g, m) for g in groups], float) if err else None
        ax.bar(x, v, w * 0.92, color=COL[m], hatch=HATCH.get(m, ""), edgecolor="white" if not HATCH.get(m) else INK2,
               lw=0.4, label=NAME[m], zorder=2)
        if e is not None and np.isfinite(e).any():
            ax.errorbar(x, v, yerr=np.where(np.isfinite(e), e, 0), fmt="none", ecolor=INK, elinewidth=0.5, capsize=1, zorder=3)
    ax.set_xticks(np.arange(len(groups)))
    ax.grid(axis="x", visible=False)


# ================================================================================================ fig_qoe_bars
meths = MAIN + BEST
fig, ax = plt.subplots(figsize=(W2, 1.9))
grouped_bars(ax, ENVS, meths, lambda e, m: cell(e, m) / best_any(e),
             lambda e, m: cell(e, m, "J0_seed_sd") / best_any(e) if m not in ("robustmpc", "best_tier") else np.nan)
ax.set_xticklabels([ENV_SHORT.get(e, e) + (f"\n$M$ = {env_M(e):.2f}" if np.isfinite(env_M(e)) else "") for e in ENVS])
ax.set_ylabel("Normalised QoE")
lo = min(cell(e, m) / best_any(e) for e in ENVS for m in meths if np.isfinite(cell(e, m)))
ax.set_ylim(max(0, np.floor(lo * 10) / 10 - 0.05), 1.12); ax.axhline(1, color=INK2, lw=0.5, ls=":")
ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.4)
r_ = {e: cell(e, REF) / best_any(e) for e in ENVS}
near = [ENV_SHORT[e] for e in ENVS if r_[e] >= 0.95]
save(fig, "fig_qoe_bars",
     "Normalised QoE (mean $J_0$ divided by the best mean $J_0$ of any method or fixed rule run in that environment) in "
     f"each training environment, ordered by misalignment $M$; error bars: s.d. over {len(PR['seeds'])} training seeds; "
     f"100 shared test sessions. DARA-L is within 5% of the best in {len(near)} of {len(ENVS)} environments "
     f"({', '.join(near)}).")

# ================================================================================================ fig_qoe_components
fig, axes = plt.subplots(2, 1, figsize=(W2, 3.1), sharex=True)
grouped_bars(axes[0], ENVS, meths, lambda e, m: cell(e, m),
             lambda e, m: cell(e, m, "J0_seed_sd") if m not in ("robustmpc", "best_tier") else np.nan)
axes[0].set_ylabel("Mean $J_0$ per session"); axes[0].set_ylim(0, None)
grouped_bars(axes[1], ENVS, meths, lambda e, m: cell(e, m, "freeze_s"))
axes[1].set_ylabel("Freeze time (s/session)")
axes[1].set_yscale("log"); axes[1].set_ylim(bottom=1)
axes[1].set_xticklabels([ENV_SHORT.get(e, e) for e in ENVS])
sub(axes[0], "(a)"); sub(axes[1], "(b)")
axes[0].legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.05), columnspacing=1.0, handlelength=1.4)
fig.tight_layout(h_pad=0.6)
fr = {e: (cell(e, REF, "freeze_s"), cell(e, "ppo_lagrangian_t0.5", "freeze_s")) for e in ENVS}
less = [ENV_SHORT[e] for e, (x, y) in fr.items() if np.isfinite(y) and x < y - 1]
save(fig, "fig_qoe_components",
     "Per training environment: (a) mean QoE $J_0$ per session over the 100 test sessions (error bars: s.d. over "
     f"{len(PR['seeds'])} training seeds; RobustMPC{'' if a.no_best_tier else ' and the best fixed tier'} "
     f"{'is' if a.no_best_tier else 'are'} deterministic) and (b) freeze time per "
     f"session (log scale). DARA-L freezes less than PPO-Lagrangian (0.5 s) in {len(less)} of {len(ENVS)} environments.")

# ================================================================================================ fig_qoe_cdf


def sessions(e, m):
    if m in ("robustmpc", "best_tier"):
        any_ = next((v for (ee, _), v in RUN_DIRS.items() if ee == e), None)
        if not any_:
            return None
        g = pd.read_csv(any_[0] / "gt_per_episode.csv")
        if m == "robustmpc":
            return g.robustmpc_ref.to_numpy(float)
        t = max(range(5), key=lambda k: g[f"tier{k}_ref"].mean())
        return g[f"tier{t}_ref"].to_numpy(float)
    gs = [pd.read_csv(d / "gt_per_episode.csv").policy_ref.to_numpy(float) for d in RUN_DIRS.get((e, m), [])
          if (d / "gt_per_episode.csv").exists()]
    return np.mean(gs, axis=0) if gs else None


cdf_envs = [e for e in [PRIMARY, "ct8", "ctl"] if e in ENVS]
if RUN_DIRS and cdf_envs:
    fig, axes = plt.subplots(1, len(cdf_envs), figsize=(W2, 1.9), squeeze=False)
    for k, (ax, e) in enumerate(zip(axes[0], cdf_envs)):
        for m in MAIN + BEST:
            v = sessions(e, m)
            if v is None:
                continue
            v = np.sort(v)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[m], ls=LS.get(m, (0, (3, 1))) if m != "best_tier" else (0, (2, 2)),
                    lw=1.3 if m == REF else 0.9, label=NAME[m])
        allv = np.concatenate([v_ for v_ in (sessions(e, m) for m in MAIN + BEST) if v_ is not None])
        ax.set_xlim(np.percentile(allv, 1.5), np.percentile(allv, 100) * 1.02)
        ax.set_xlabel("QoE $J_0$ per session"); ax.set_ylim(0, 1)
        if k == 0:
            ax.set_ylabel("CDF")
        sub(ax, f"({'abc'[k]}) {ENV_SHORT[e]}")
    axes[0, 0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0.0, 1.12), columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_qoe_cdf", "CDF of per-session QoE over the 100 test sessions (each learned method averaged over its seeds; "
         "x-axis starts at the 1.5th percentile) in "
         + ", ".join(f"({'abc'[k]}) {ENV_SHORT[e]}" for k, e in enumerate(cdf_envs)) + ".")

# ================================================================================================ fig_diagnosis
if DV is not None and len(DV):
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    mk = {1: "o", 2: "s", 4: "^", 8: "D"}
    shade = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 8: "#0d366b"}
    for sc, g in DV.groupby("ladder_scale"):
        sd = [np.std(json.loads(x), ddof=1) if len(json.loads(x)) > 1 else 0 for x in g.seed_shortfalls]
        ax.errorbar(g.M.to_numpy(float), 100 * g.shortfall.to_numpy(float), yerr=100 * np.array(sd), fmt=mk.get(sc, "o"),
                    ms=3.8, color=shade.get(sc, INK), mec="white", mew=0.4, elinewidth=0.6, label=f"Ladder ×{sc:g}")
    ax.axvline(THR, color=INK2, ls="--", lw=0.7)
    ax.text(THR + 0.01, 0.97, f"act threshold {THR:g}", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, va="top")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel("Misalignment $M$ before training"); ax.set_ylabel("Shortfall from best fixed tier (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9), handletextpad=0.3)
    cap = "Misalignment $M$, measured before training, against the shortfall of plain PPO after training (5 seeds; bars: s.d.), for every reward setting (ladder × freeze-penalty cap)."
    if SV is not None and len(SV):
        r = SV.iloc[0]
        ax.text(0.98, 0.04, f"Spearman $\\rho$ = {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", transform=ax.transAxes,
                ha="right", fontsize=6.5, color=INK2)
        cap += f" Spearman $\\rho$ = {r.rho:.2f} (95% bootstrap interval {r.ci_lo:.2f}–{r.ci_hi:.2f}, {int(r.n)} settings)."
    fig.tight_layout()
    save(fig, "fig_diagnosis", cap)

# ================================================================================================ fig_budget_sweep
benvs = [e for e in ENVS if ((TT.env == e) & TT.method.str.match(r"ppo_lagrangian_t[0-9.]+$")).sum() >= 3]
if benvs:
    nc = min(3, len(benvs)); nr = int(np.ceil(len(benvs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(W2, 1.75 * nr + 0.35), squeeze=False)
    for k, e in enumerate(benvs):
        ax = axes[k // nc, k % nc]
        d = TT[TT.env == e].set_index("method")
        h = d[d.index.str.match(r"ppo_lagrangian_t[0-9.]+$")].copy()
        h["b"] = h.index.str.extract(r"_t([0-9.]+)$", expand=False).astype(float).to_numpy()
        h = h.sort_values("b")
        ax.errorbar(h.b.to_numpy(float), h.J0.to_numpy(float), yerr=h.J0_seed_sd.to_numpy(float), fmt="-o", ms=3,
                    color=COL["ppo_lagrangian_t0.5"], elinewidth=0.6, label="PPO-Lag., hand-picked budget")
        if REF in d.index and np.isfinite(d.loc[REF, "budget"]):
            ax.errorbar([d.loc[REF, "budget"]], [d.loc[REF, "J0"]], yerr=[d.loc[REF, "J0_seed_sd"]], fmt="*", ms=8,
                        color=COL[REF], mec="white", mew=0.4, elinewidth=0.6, zorder=4, label="DARA-L, derived budget")
        elif REF in d.index:
            ax.axhline(d.loc[REF, "J0"], color=COL[REF], ls=":", lw=1, label="DARA-L (did not act)")
        if "robustmpc" in d.index:
            ax.axhline(d.loc["robustmpc", "J0"], color=COL["robustmpc"], ls="--", lw=0.8, label="RobustMPC")
        ax.set_xscale("log", base=2)
        xs = sorted(h.b); ax.set_xticks(xs); ax.set_xticklabels([f"{b:g}" for b in xs]); ax.minorticks_off()
        ax.grid(axis="x", visible=False)
        if k // nc == nr - 1:
            ax.set_xlabel("Freeze budget (s/chunk)")
        if k % nc == 0:
            ax.set_ylabel("QoE $J_0$")
        sub(ax, f"({'abcdefgh'[k]}) {ENV_SHORT[e]}")
    for k in range(len(benvs), nr * nc):
        axes[k // nc, k % nc].axis("off")
    hs, ls_ = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls_:
                hs.append(hh); ls_.append(ll)
    fig.legend(hs, ls_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.9, w_pad=0.9)
    cap = "QoE against the freeze budget: PPO-Lagrangian with hand-picked budgets (bars: s.d. over seeds) and DARA-L at its derived budget (star)."
    if BT is not None:
        sel = BT[BT.env.isin(benvs) & BT.derived_budget.notna()]
        ok = sel[sel.gap_rel >= -PR.get("budget_rule_test", {}).get("tolerance_rel", 0.05)]
        cap += (f" DARA-L is within 5% of the best hand-picked budget in {len(ok)} of {len(sel)} environments where it acted "
                f"({', '.join(ENV_SHORT[e] for e in ok.env)}).")
    save(fig, "fig_budget_sweep", cap)

# ================================================================================================ fig_training
GX = np.arange(0, 10001, 100)


def curves(e, m, col):
    out = []
    for d in RUN_DIRS.get((e, m), []):
        f = d / "ppo_log.csv"
        if f.exists():
            g = pd.read_csv(f)
            if col in g and len(g) > 5:
                out.append(pd.Series(np.interp(GX, g.episode, g[col], left=np.nan, right=np.nan)).rolling(5, min_periods=1).mean().to_numpy())
    return np.array(out) if out else None


def switch_on(e):
    eps = []
    for d in RUN_DIRS.get((e, REF), []):
        f = d / "dara_log.csv"
        if f.exists():
            L = pd.read_csv(f); on = L[L.lagrange == 1]
            if len(on):
                eps.append(int(on.episode.iloc[0]))
    return eps


tenvs = [e for e in [PRIMARY, "ct8", "cth4", "ctd1"] if e in ENVS and (e, REF) in RUN_DIRS]
if tenvs:
    fig, axes = plt.subplots(2, len(tenvs), figsize=(W2, 3.0), sharex=True, squeeze=False)
    for k, e in enumerate(tenvs):
        sw = switch_on(e)
        for row, (col, yl) in enumerate((("mean_ref_20", "Training QoE $J_0$"), ("lam", "Penalty weight $\\lambda$"))):
            ax = axes[row, k]
            if sw:
                ax.axvspan(min(sw), max(sw) + 50, color="#e2e1dc", lw=0, zorder=0)
            for m in [REF, "ppo_lagrangian_t0.5"] + (["fixed"] if row == 0 else []):
                cv = curves(e, m, col)
                if cv is None:
                    continue
                mu = np.nanmean(cv, 0)
                ax.fill_between(GX, mu - np.nanstd(cv, 0), mu + np.nanstd(cv, 0), color=COL[m], alpha=0.18, lw=0)
                ax.plot(GX, mu, color=COL[m], ls=LS.get(m, "-"), lw=1.0, label=NAME[m])
            if k == 0:
                ax.set_ylabel(yl)
            if row == 1:
                ax.set_xlabel("Training episode"); ax.set_xticks([0, 5000, 10000]); ax.set_xticklabels(["0", "5k", "10k"])
            if row == 0:
                sub(ax, f"({'abcd'[k]}) {ENV_SHORT[e]}")
    hs = [Line2D([], [], color=COL[m], ls=LS.get(m, "-"), label=NAME[m]) for m in [REF, "ppo_lagrangian_t0.5", "fixed"]]
    hs.append(Patch(fc="#e2e1dc", label="DARA activates the constraint"))
    fig.legend(handles=hs, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    save(fig, "fig_training", "Training dynamics: QoE on training sessions (top) and Lagrange multiplier $\\lambda$ (bottom), "
         "mean ± s.d. over 5 seeds; shaded band: episodes at which DARA activated the constraint (range over seeds). "
         "PPO (original reward) has no constraint.")

# ================================================================================================ fig_unseen
if GS is not None and len(GS):
    conds = [c for c in PR.get("generalisation", {}).get("figure", {}).get("conditions", [])] or \
            [f"{n}/{c}" for n in PR["generalisation"]["figure"]["networks"] for c in PR["generalisation"]["figure"]["contents"]]
    conds = [c for c in conds if c not in PR["generalisation"]["figure"].get("exclude", []) and c in set(GS.condition)]
    gm = [m for m in MAIN if m in set(GS.method)] + ["best_tier"]

    def gval(c, m, col="J0"):
        g = GS[GS.condition == c]
        if m == "best_tier":
            t = g[g.method.str.fullmatch(r"tier[0-4]")]
            return float(t[col].loc[t.J0.idxmax()]) if len(t) else np.nan
        r = g[g.method == m]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def gbest(c):
        g = GS[(GS.condition == c) & (GS.method.isin(MAIN) | GS.method.str.fullmatch(r"tier[0-4]"))]
        return float(g.J0.max())
    fig, ax = plt.subplots(figsize=(W2, 1.8))
    grouped_bars(ax, conds, gm, lambda c, m: gval(c, m) / gbest(c),
                 lambda c, m: gval(c, m, "J0_seed_sd") / gbest(c) if m not in ("robustmpc", "best_tier") else np.nan)
    ax.set_xticklabels([f"{R.NETWORKS[c.split('/')[0]]['label'].replace('Training generator', 'mmWave generator')}, {R.CONTENTS[c.split('/')[1]]}"
                        for c in conds])
    lo = min(gval(c, m) / gbest(c) for c in conds for m in gm if np.isfinite(gval(c, m)))
    ax.set_ylim(max(0, np.floor(lo * 10) / 10 - 0.05), 1.12); ax.axhline(1, color=INK2, lw=0.5, ls=":")
    ax.set_ylabel("Normalised QoE")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.4)
    save(fig, "fig_unseen", "Normalised QoE on unseen mmWave conditions (synthetic generator with unseen chunk-size ladders; real "
         "Lumos5G traces) for models trained on the generator with ladder ×4; error bars: s.d. over seeds. All conditions, "
         "including the non-mmWave Irish 5G traces, are reported in Table II.")

# ================================================================================================ fig_tradeoff
if PRIMARY in ENVS:
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    d = TT[TT.env == PRIMARY].set_index("method")
    tiers = d[d.index.str.fullmatch(r"tier[0-4]")].sort_index()
    ax.plot(tiers.freeze_s, [float(k[-1]) for k in tiers.index],
            color=COL["best_tier"], lw=0.8, ls=":", marker="x", ms=3, label="Fixed tier 0–4")
    for m in MAIN:
        if m not in d.index:
            continue
        x, y = float(d.loc[m, "freeze_s"]), cell(PRIMARY, m, "mean_tier")
        per = []
        for rd in RUN_DIRS.get((PRIMARY, m), []):
            f = rd / "triage_summary.csv"
            if f.exists():
                t = pd.read_csv(f)
                if "policy_tier_share" in t:
                    sh = json.loads(t.policy_tier_share.iloc[0])
                    per.append((float(t.policy_rebuffer.iloc[0]), float(np.dot(sh, range(5)))))
        if per:
            px, py = zip(*per)
            ax.scatter(px, py, marker=MARK[m], s=8, color=COL[m], alpha=0.45, lw=0)
            ax.errorbar(x, y, xerr=np.std(px, ddof=1), yerr=np.std(py, ddof=1), fmt=MARK[m], ms=5, color=COL[m], mec="white",
                        mew=0.5, elinewidth=0.7, label=NAME[m], zorder=4)
        else:
            ax.plot(x, y, MARK[m], ms=5, color=COL[m], mec="white", mew=0.5, label=NAME[m], zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Freeze time (s/session, log scale)"); ax.set_ylabel("Mean quality level")
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax.set_ylim(-0.2, 4.6)
    fig.tight_layout()
    save(fig, "fig_tradeoff", f"Quality–freeze trade-off in the primary training environment ({ENV_SHORT[PRIMARY]}): mean over the 100 "
         "test sessions per seed (small markers) and over seeds (large markers, bars: s.d.); dotted: always the same tier. "
         "Up and to the left is better.")

(OUT / "captions.md").write_text("# Suggested captions\n\n" + (f"> Note: {PR['deviation']}\n\n" if PR.get("deviation") else "") + "\n\n".join(f"**{n}.** {c}" for n, c in captions) + "\n")
print("written", OUT / "captions.md")
