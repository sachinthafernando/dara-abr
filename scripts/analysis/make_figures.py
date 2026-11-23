#!/usr/bin/env python3
"""Paper figures (IEEE two-column: 3.5 in single, 7.16 in double; no titles; PDF for the paper, PNG for checking).

Result figures use the primary training environment, except the two *_ladders figures (set 1 of paper_v2.json).

  fig_network        (a) one generator session coloured by channel state, (b) throughput per channel state,
                     (c) throughput distribution of the training generator and the real test networks
  fig_cap_effect     cap sweep: (a) M and D before training, (b) J0 after training, (c) freeze time
  fig_tier_share     share of chunks per quality tier for the trained policies and RobustMPC
  fig_budget         (a) J0 and (b) freeze time against the freeze budget: hand-picked budgets vs DARA-L's derived one
  fig_tradeoff       mean quality tier against freeze time, 95% intervals
  fig_generalisation J0 minus RobustMPC on the mmWave test conditions named in the pre-registration
  fig_diagnosis_ladders  M before training against the trained shortfall, every reward setting on the four ladders (C1)
  fig_budget_ladders     J0 and freeze against the budget on each ladder: hand-picked budgets, DARA-L, RobustMPC (C3)

Inputs (each figure is skipped with a WARNING if its input is missing):
  runs/train (tier shares), runs/generalize (trade-off), runs/landscape/landscape_content_u1_integrated.csv,
  results/training/*.csv, results/generalisation/*.csv, data/processed/*.npz
Usage:  python scripts/analysis/make_figures.py [--only fig_network fig_budget ...]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401  (older matplotlib + pandas 2)
from dara_abr.results import load_training, load_generalisation  # noqa: E402
from dara_abr.urban_generator import generate_session, SCENARIO_MIX, REGIME_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--gen", default=str(ROOT / "runs"))
ap.add_argument("--landscape", default=str(ROOT / "runs" / "landscape" / "landscape_content_u1_integrated.csv"))
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "figures"))
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
RES = Path(a.results)
PR = json.load(open(a.prereg))
ENV = PR["primary_training_environment"]
REF = PR["proposed_method"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "savefig.dpi": 300, "pdf.fonttype": 42})
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ5 = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]                  # quality tiers, light -> dark
SEQ6 = ["#c9dcf5", "#9ec5f4", "#6ba4ea", "#3987e5", "#1c5cab", "#0d366b"]       # channel states, outage -> strong LoS
W1, W2 = 3.5, 7.16
TIER_MBPS = np.array(R.ladder(R.TRAINING_ENVS[ENV]["ladder_scale"])) * 8 / 4.0  # nominal bitrates, training ladder


def want(name):
    return a.only is None or name in a.only


def warn(name, why):
    print(f"WARNING: skipping {name}: {why}")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig); print("written", OUT / f"{name}.pdf")


def tag(ax, s, x=0.0):
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def style(ax, grid="both"):
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


TT = csv(RES / "training" / "training_table.csv")
T4 = TT[TT.env == ENV].set_index("method") if TT is not None else None
need_runs = any(want(f) for f in ("fig_tier_share",))
S, REFS, META = load_training(a.runs) if need_runs and Path(a.runs).exists() else (None, None, None)
G = load_generalisation(a.gen) if any(want(f) for f in ("fig_tradeoff",)) and Path(a.gen).exists() else pd.DataFrame()
GT = csv(RES / "generalisation" / "gen_table.csv")

# ======================================================================================== fig_network
if want("fig_network"):
    rng = np.random.default_rng(7)
    ex = generate_session("urban_driving", 300, rng)
    pool = []
    for scn in SCENARIO_MIX:
        for _ in range(40):
            g = generate_session(scn, 600, rng)
            pool.append(pd.DataFrame(dict(state=g["regime"], mbps=g["throughput"], scenario=scn)))
    pool = pd.concat(pool, ignore_index=True)
    w = pd.Series(SCENARIO_MIX); w = w / w.sum()
    # the training mix: each scenario sampled in proportion to its weight in SCENARIO_MIX
    mixdf = pool.groupby("scenario", group_keys=False)[["state", "mbps", "scenario"]].apply(
        lambda d: d.sample(frac=min(1.0, w[d.name] / w.max()), random_state=0))
    nets = [("Training generator (synthetic)", mixdf.mbps.to_numpy(), INK, "-")]
    for fname, label, col, ls, prefix in ((R.LUMOS_TRACES, "Lumos5G (real mmWave)", "#6250d6", "-", ""),
                                          (R.IRISH_TRACES, "Irish 5G static", "#0f8a8a", "--", "Static_"),
                                          (R.IRISH_TRACES, "Irish 5G driving", "#0f8a8a", ":", "Driving_")):
        p = ROOT / fname
        if not p.exists():
            warn("fig_network (c) " + label, f"{fname} not found"); continue
        z = np.load(p)
        keys = [k for k in z.files if k.startswith(prefix)]
        if keys:
            nets.append((label, np.concatenate([z[k] for k in keys]), col, ls))
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw=dict(width_ratios=[1.35, 1, 1.1]))
    ax = axes[0]
    t = np.arange(len(ex["throughput"])); y = np.maximum(ex["throughput"], 0.1)
    ax.plot(t, y, color=INK, lw=0.9)
    xt = ax.get_xaxis_transform()                    # x in data, y in axes coordinates: channel-state strip
    for s_ in range(len(REGIME_NAMES)):
        on = ex["regime"] == s_
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]]); ends = np.flatnonzero(on & ~np.r_[on[1:], False]) + 1
        ax.broken_barh([(a_, b_ - a_) for a_, b_ in zip(starts, ends)], (0.0, 0.07), transform=xt, color=SEQ6[s_], lw=0)
    ax.set_xlim(0, t[-1]); ax.set_yscale("log"); ax.set_ylim(0.03, 3000)
    for mb in TIER_MBPS:
        ax.axhline(mb, color=INK2, lw=0.4, ls=":")
    ax.text(1.01, np.log10(TIER_MBPS[2] / 0.03) / np.log10(3000 / 0.03), "tiers", transform=ax.transAxes, fontsize=6,
            color=INK2, rotation=90, va="center")
    ax.text(0.01, 0.085, "channel state", transform=ax.transAxes, fontsize=6, color=INK2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Throughput (Mbit/s)"); style(ax, "y"); tag(ax, "(a)")
    ax = axes[1]
    data = [np.maximum(mixdf.mbps[mixdf.state == s].to_numpy(), 0.1) for s in range(len(REGIME_NAMES))]
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
    for patch, c in zip(bp["boxes"], SEQ6):
        patch.set_facecolor(c); patch.set_edgecolor(INK2); patch.set_linewidth(0.6)
    share = mixdf.state.value_counts(normalize=True).reindex(range(len(REGIME_NAMES))).fillna(0)
    for i, s in enumerate(share):
        ax.text(i + 1, 2500, f"{100 * s:.0f}%", ha="center", fontsize=6, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(0.1, 4000)
    ax.set_xticks(range(1, len(REGIME_NAMES) + 1))
    ax.set_xticklabels(["Outage", "Deep\nNLoS", "NLoS", "Mixed", "LoS", "Strong\nLoS"], fontsize=6.5)
    ax.set_xlabel("Channel state (share of time above)"); style(ax, "y"); tag(ax, "(b)")
    ax = axes[2]
    for label, v, col, ls in nets:
        v = np.sort(np.maximum(v, 0.1))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.3, label=label)
    for mb in TIER_MBPS:
        ax.axvline(mb, color=INK2, lw=0.4, ls=":")
    ax.set_xscale("log"); ax.set_xlim(0.1, 3000); ax.set_ylim(0, 1)
    ax.set_xlabel("Throughput (Mbit/s)"); ax.set_ylabel("CDF"); style(ax); tag(ax, "(c)")
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0)
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=2.2)
    save(fig, "fig_network")

# ======================================================================================== fig_cap_effect
if want("fig_cap_effect"):
    L = csv(a.landscape)
    if L is None:
        warn("fig_cap_effect", f"{a.landscape} not found (run configs/sweeps/landscape.tsv)")
    elif T4 is None:
        warn("fig_cap_effect", "results/training/training_table.csv not found (run evaluate_training.py)")
    else:
        L = L[(L.quality_k == 0) & np.isclose(L.q_top, R.QUALITY[-1]) & np.isclose(L.ladder_scale, R.TRAINING_ENVS[ENV]["ladder_scale"])]
        if "rebuf_scale" in L:
            L = L[L.rebuf_scale == 1.0]
        caps = [c for c in R.CAPS if R.cap_key(c) in T4.index and np.isclose(L.rn_cap, c).any()]
        if len(caps) < 3:
            warn("fig_cap_effect", f"only {len(caps)} caps have both a landscape row and trained runs")
        else:
            x = np.arange(len(caps)); xl = [R.cap_label(c) for c in caps]
            Mv = np.array([float(L[np.isclose(L.rn_cap, c)].M.iloc[0]) for c in caps])
            Dv = np.array([float(L[np.isclose(L.rn_cap, c)].D.iloc[0]) for c in caps])
            J = np.array([float(T4.loc[R.cap_key(c), "J0"]) for c in caps])
            Js = np.array([float(T4.loc[R.cap_key(c), "J0_seed_sd"]) for c in caps])
            Fz = np.array([float(T4.loc[R.cap_key(c), "freeze_s"]) for c in caps])
            refs = [(m, R.label(m).replace(" (ours)", ""), R.colour(m), "-") for m in (REF, "ppo_lagrangian_t0.5") if m in T4.index]
            fig, axes = plt.subplots(1, 3, figsize=(W2, 1.95))
            ax = axes[0]
            ax.plot(x, Mv, "-o", ms=3.5, lw=1.4, color="#6250d6", label="Misalignment $M$")
            ax.plot(x, Dv, "--s", ms=3, lw=1.0, color=INK2, label="Dominance $D$")
            ax.axhline(0.2, color=INK2, lw=0.6, ls=":"); ax.text(len(caps) - 1, 0.215, "$M$ = 0.2", ha="right", va="bottom", fontsize=6.5, color=INK2)
            ax.set_ylim(0, 1.05); ax.set_ylabel("Signal (before training)"); ax.legend(frameon=False, loc="upper right")
            ax = axes[1]
            ax.errorbar(x, J, yerr=Js, fmt="-o", ms=3.5, lw=1.4, color=INK2, capsize=1.5, elinewidth=0.8, label="PPO with cap $c$")
            for m, lab, col, ls in refs:
                ax.axhline(float(T4.loc[m, "J0"]), color=col, lw=1.2, ls=ls, label=lab)
            if "robustmpc" in T4.index:
                ax.axhline(float(T4.loc["robustmpc", "J0"]), color=R.colour("robustmpc"), lw=1.0, ls="--", label="RobustMPC")
            ax.set_ylabel("$J_0$ after training"); hb, lb = ax.get_legend_handles_labels()
            ax = axes[2]
            ax.plot(x, Fz, "-o", ms=3.5, lw=1.4, color=INK2)
            for m, lab, col, ls in refs:
                ax.axhline(float(T4.loc[m, "freeze_s"]), color=col, lw=1.2, ls=ls)
            ax.set_ylim(bottom=0); ax.set_ylabel("Freeze time per session (s)")
            for i, ax in enumerate(axes):
                ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("Freeze-penalty cap $c$"); style(ax); tag(ax, "(" + "abc"[i] + ")")
            fig.tight_layout(w_pad=1.0)
            fig.legend(hb, lb, ncol=4, loc="lower center", bbox_to_anchor=(0.66, 0.99), frameon=False)
            save(fig, "fig_cap_effect")

# ======================================================================================== fig_tier_share
if want("fig_tier_share"):
    if META is None or META.empty:
        warn("fig_tier_share", f"no runs under {a.runs}")
    else:
        m4 = META[META.env == ENV]
        order = ["fixed", "dara_online", REF, "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]
        rows, shares = [], []
        for k in order:
            v = [t for t in m4[m4.method == k].policy_tier_share if t is not None]
            if v:
                rows.append(R.label(k)); shares.append(np.mean(v, 0))
        mpc = [t for t in m4.mpc_tier_share if t is not None]
        if mpc:
            rows.append("RobustMPC"); shares.append(np.mean(mpc, 0))
        if not rows:
            warn("fig_tier_share", f"no tier shares for {ENV}")
        else:
            fig, ax = plt.subplots(figsize=(W1, 0.27 * len(rows) + 0.55))
            y = np.arange(len(rows))[::-1]; left = np.zeros(len(rows))
            for t in range(5):
                vals = np.array([s[t] for s in shares])
                ax.barh(y, vals, left=left, height=0.62, color=SEQ5[t], edgecolor=SURF, linewidth=1.0, label=f"tier {t}")
                for yi, l0, v in zip(y, left, vals):
                    if v >= 0.12:
                        ax.text(l0 + v / 2, yi, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if t >= 2 else INK)
                left += vals
            ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlim(0, 1); ax.set_xlabel("Share of chunks")
            ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, handlelength=1.0, columnspacing=0.8)
            fig.tight_layout(); save(fig, "fig_tier_share")

# ======================================================================================== fig_budget
if want("fig_budget"):
    if T4 is None:
        warn("fig_budget", "results/training/training_table.csv not found")
    else:
        hand = T4[T4.index.str.match(r"ppo_lagrangian_t[0-9.]+$")].copy()
        if hand.empty or REF not in T4.index:
            warn("fig_budget", "hand-picked budgets or DARA-L missing")
        else:
            hand["b"] = hand.index.str.extract(r"_t([0-9.]+)$", expand=False).astype(float)
            hand = hand.sort_values("b"); dl = T4.loc[REF]
            if not np.isfinite(float(dl.budget)):
                print("WARNING: fig_budget: DARA-L has no derived budget (DARA did not act); drawing the hand-picked curve only")
            fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.45, 1.9))
            for ax, col, ylab in ((axes[0], "J0", "$J_0$ per session"), (axes[1], "freeze_s", "Freeze time per session (s)")):
                err = hand.J0_seed_sd.to_numpy(float) if col == "J0" else None
                ax.errorbar(hand.b.to_numpy(float), hand[col].to_numpy(float), yerr=err, fmt="-o", ms=3.5, lw=1.2,
                            color=R.colour("ppo_lagrangian_t0.5"), capsize=1.5, elinewidth=0.8, label="PPO-Lagrangian, hand-picked budget")
                if np.isfinite(float(dl.budget)):
                    ax.errorbar([float(dl.budget)], [float(dl[col])], yerr=[float(dl.J0_seed_sd)] if col == "J0" else None,
                                fmt="D", ms=5, color=R.colour(REF), mec=SURF, mew=0.8, capsize=1.5, elinewidth=0.8, zorder=4,
                                label="DARA-L (derived budget)")
                if "robustmpc" in T4.index:
                    ax.axhline(float(T4.loc["robustmpc", col]), color=R.colour("robustmpc"), lw=1.0, ls="--", label="RobustMPC")
                ax.set_xscale("log", base=2); bt = sorted(hand.b.astype(float))
                ax.set_xticks(bt); ax.set_xticklabels([f"{b:g}" for b in bt]); ax.minorticks_off()
                ax.set_xlabel("Freeze budget (s per chunk)"); ax.set_ylabel(ylab); style(ax)
            if axes[1].get_ylim()[0] > 0:
                axes[1].set_ylim(bottom=0)
            tag(axes[0], "(a)"); tag(axes[1], "(b)")
            h, l = axes[0].get_legend_handles_labels()
            fig.tight_layout(w_pad=1.2)
            fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5)
            save(fig, "fig_budget")

# ======================================================================================== fig_tradeoff
if want("fig_tradeoff"):
    if G.empty:
        warn("fig_tradeoff", f"no generalisation output under {a.gen}")
    else:
        seen = G[G.condition == R.TRAINING_CONDITION]
        items = [(REF, R.colour(REF)), ("corel_official", R.colour("corel_official")),
                 ("pensieve_official", R.colour("pensieve_official")), ("ppo_lagrangian_t0.5", R.colour("ppo_lagrangian_t0.5"))]
        rng = np.random.default_rng(0)

        def ci(v):
            bs = rng.choice(v, (2000, len(v))).mean(1)
            return np.percentile(bs, [2.5, 97.5])
        pts = []
        for k, col in items:
            d = seen[(seen.run != "reference") & (seen.train_method == k)]
            if len(d):
                g = d.groupby("episode")[["rebuffer", "mean_tier"]].mean()
                pts.append((R.label(k) + (" (hand-tuned)" if k.startswith("ppo_lag") else ""), col, g.rebuffer.values, g.mean_tier.values))
        d = seen[(seen.run == "reference") & (seen.method == "robustmpc")].sort_values("episode")
        if len(d):
            pts.append(("RobustMPC", R.colour("robustmpc"), d.rebuffer.values, d.mean_tier.values))
        fig, ax = plt.subplots(figsize=(W1, 2.3))
        for lab, col, rb, mt in pts:
            x, y = rb.mean(), mt.mean(); xl, xh = ci(rb); yl, yh = ci(mt)
            ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", ms=5, color=col, mec=SURF,
                        mew=1.0, elinewidth=1.0, capsize=2, zorder=3)
            ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
        ax.set_xlabel("Freeze time per session (s)"); ax.set_ylabel("Mean quality tier (0–4)"); style(ax)
        ax.annotate("better", xy=(0.04, 0.95), xytext=(0.20, 0.80), xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8), fontsize=7, color=INK2, ha="center")
        ax.set_xlim(left=0); fig.tight_layout(); save(fig, "fig_tradeoff")

# ======================================================================================== fig_generalisation
if want("fig_generalisation"):
    if GT is None:
        warn("fig_generalisation", "results/generalisation/gen_table.csv not found (run evaluate_generalisation.py)")
    else:
        F = PR["generalisation"]["figure"]
        conds = [f"{n}/{c}" for n in F["networks"] for c in F["contents"] if f"{n}/{c}" not in F["exclude"]]
        conds = [c for c in conds if c in set(GT.condition)]
        rows = [m for m in (REF, "corel_official", "pensieve_official", "ppo_lagrangian_t0.5") if m in set(GT.method)]
        if not conds or not rows:
            warn("fig_generalisation", "no figure conditions or methods in gen_table.csv")
        else:
            V = np.array([[GT[(GT.method == m) & (GT.condition == c)].J0_minus_mpc.mean() for c in conds] for m in rows])
            vmax = max(np.nanmax(np.abs(V)), 1.0)
            cmap = LinearSegmentedColormap.from_list("div", ["#c0362c", "#e34948", "#f3b0a9", "#f0efec", "#9ec5f4", "#2a78d6", "#1c5cab"])
            fig, ax = plt.subplots(figsize=(max(W1, 0.62 * len(conds) + 1.6), 0.36 * len(rows) + 0.95))
            im = ax.imshow(V, cmap=cmap, norm=SymLogNorm(linthresh=20, vmin=-vmax, vmax=vmax, base=10), aspect="auto")
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    ax.text(j, i, f"{V[i, j]:+.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(V[i, j]) > 0.55 * vmax and abs(V[i, j]) > 60 else INK)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([R.label(m).replace("PPO-Lagrangian, 0.5 s", "PPO-Lag., 0.5 s") for m in rows])
            ax.set_xticks(range(len(conds))); ax.set_xticklabels([R.CONTENTS[c.split("/")[1]] for c in conds], fontsize=6.5)
            pos = 0
            for n in F["networks"]:
                k = sum(1 for c in conds if c.startswith(n + "/"))
                if not k:
                    continue
                ax.text(pos + (k - 1) / 2, -0.75, R.NETWORKS[n]["label"], ha="center", va="bottom", fontsize=7, color=INK)
                if pos > 0:
                    ax.axvline(pos - 0.5, color=SURF, lw=2.5)
                pos += k
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("$J_0$ − RobustMPC"); cb.outline.set_visible(False)
            nice = [v for v in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if v <= vmax]
            ticks = sorted([-v for v in nice] + [0] + nice)
            cb.set_ticks(ticks); cb.set_ticklabels([f"{t:+d}" if t else "0" for t in ticks]); cb.ax.minorticks_off()
            fig.tight_layout(); save(fig, "fig_generalisation")

# ======================================================================================== fig_diagnosis_ladders (C1)
if want("fig_diagnosis_ladders"):
    DV = csv(RES / "training" / "diagnosis_validity.csv")
    SV = csv(RES / "training" / "diagnosis_validity_summary.csv")
    if DV is None or DV.empty:
        warn("fig_diagnosis_ladders", "results/training/diagnosis_validity.csv not found (run evaluate_training.py)")
    else:
        thr = PR.get("act_threshold", {}).get("primary", 0.2)
        scales = sorted(DV.ladder_scale.unique())
        ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"][-len(scales):] if len(scales) <= 4 else SEQ5
        fig, ax = plt.subplots(figsize=(W1, 2.4))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(thr, color=INK2, lw=0.8, ls="--")
        ax.text(thr, 1.0, f" act if M ≥ {thr:g}", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, va="top")
        for sc, col in zip(scales, ramp):
            d = DV[DV.ladder_scale == sc]
            sd = [np.std(json.loads(x), ddof=1) if len(json.loads(x)) > 1 else 0 for x in d.seed_shortfalls]
            ax.errorbar(d.M.to_numpy(float), 100 * d.shortfall.to_numpy(float), yerr=100 * np.array(sd, float), fmt="o", ms=4.5, color=col, mec=SURF, mew=0.6,
                        elinewidth=0.8, capsize=1.5, label=f"ladder ×{sc:g}", zorder=3)
        ax.set_xlabel("Misalignment $M$ before training"); ax.set_ylabel("Trained shortfall from best tier (%)")
        style(ax); ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.0, handletextpad=0.5)
        if SV is not None and len(SV):
            r = SV.iloc[0]
            ax.text(0.98, 0.45, f"Spearman ρ = {r.rho:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", transform=ax.transAxes,
                    ha="right", fontsize=6.5, color=INK2)
        fig.tight_layout(); save(fig, "fig_diagnosis_ladders")

# ======================================================================================== fig_budget_ladders (C3)
if want("fig_budget_ladders"):
    if TT is None:
        warn("fig_budget_ladders", "results/training/training_table.csv not found")
    else:
        envs = [e for e in R.LADDER_ENVS if ((TT.env == e) & TT.method.str.match(r"ppo_lagrangian_t[0-9.]+$")).any()]
        if not envs:
            warn("fig_budget_ladders", "no hand-picked budgets in the ladder environments")
        else:
            fig, axes = plt.subplots(2, len(envs), figsize=(W2, 3.4), squeeze=False)
            for j, e in enumerate(envs):
                d = TT[TT.env == e].set_index("method")
                hand = d[d.index.str.match(r"ppo_lagrangian_t[0-9.]+$")].copy()
                hand["b"] = hand.index.str.extract(r"_t([0-9.]+)$", expand=False).astype(float)
                hand = hand.sort_values("b")
                for i, (col, ylab) in enumerate((("J0", "$J_0$ per session"), ("freeze_s", "Freeze per session (s)"))):
                    ax = axes[i, j]
                    ax.errorbar(hand.b.to_numpy(float), hand[col].to_numpy(float),
                                yerr=hand.J0_seed_sd.to_numpy(float) if col == "J0" else None, fmt="-o", ms=3.2, lw=1.1,
                                color=R.colour("ppo_lagrangian_t0.5"), capsize=1.5, elinewidth=0.7, label="PPO-Lagrangian, hand-picked budget")
                    if REF in d.index:
                        dl = d.loc[REF]
                        if np.isfinite(float(dl.budget)):
                            ax.errorbar([float(dl.budget)], [float(dl[col])], yerr=[float(dl.J0_seed_sd)] if col == "J0" else None,
                                        fmt="D", ms=4.5, color=R.colour(REF), mec=SURF, mew=0.8, capsize=1.5, zorder=4,
                                        label="DARA-L (derived budget)")
                        else:
                            ax.axhline(float(dl[col]), color=R.colour(REF), lw=1.1, ls=":", label="DARA-L (did not act)")
                    if "robustmpc" in d.index:
                        ax.axhline(float(d.loc["robustmpc", col]), color=R.colour("robustmpc"), lw=0.9, ls="--", label="RobustMPC")
                    ax.set_xscale("log", base=2); bt = sorted(hand.b.astype(float))
                    ax.set_xticks(bt); ax.set_xticklabels([f"{b:g}" for b in bt]); ax.minorticks_off(); style(ax)
                    if i == 1:
                        ax.set_xlabel("Budget (s per chunk)"); ax.set_ylim(bottom=0)
                    if j == 0:
                        ax.set_ylabel(ylab)
                axes[0, j].text(0.0, 1.03, f"({'abcd'[j]}) {R.TRAINING_ENVS[e]['short']}", transform=axes[0, j].transAxes,
                                fontsize=7, va="bottom")
            hs, ls_ = {}, {}
            for ax in axes.ravel():
                for h, l in zip(*ax.get_legend_handles_labels()):
                    hs.setdefault(l, h)
            fig.tight_layout(w_pad=0.8, h_pad=0.6)
            fig.legend(list(hs.values()), list(hs.keys()), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       frameon=False, fontsize=6.5)
            save(fig, "fig_budget_ladders")
print(f"figures in {OUT}")
