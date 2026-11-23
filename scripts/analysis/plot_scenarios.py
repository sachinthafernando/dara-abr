#!/usr/bin/env python3
"""Per-scenario comparison in the style of COREL's Fig. 5: one panel per network scenario, each method one point at
(freeze time per session, mean bitrate per session) with 95% confidence intervals on both axes. The freeze axis is
reversed, so better is towards the top right. IEEE double column, no title, vector PDF + PNG.

  fig_scenarios_groups   walking / driving / dead-zone groups of generator scenarios            (1 x 3 panels)
  fig_scenarios_all      each of the eight generator scenarios                                  (2 x 4 panels)
  scenario_table.csv     every point: mean, 95% CI of bitrate, freeze, J0 and quality; seeds; sessions
  scenario_tests.csv     DARA-L against each method per scenario: paired Wilcoxon on seed-averaged sessions, BH within
                         each measure
  caption.md

95% CI: hierarchical bootstrap of the mean (resample training seeds, then test sessions; RobustMPC: sessions only),
2000 resamples, percentile interval.

Input: generalize/gen_*.csv of configs/sweeps/scenarios_ct1.tsv
Usage: python scripts/analysis/plot_scenarios.py --gen runs/scenarios_ct1 --env ct1 [--y bitrate|quality|J0]
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Tight layout not applied")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401
from dara_abr.results import load_generalisation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs" / "scenarios_ct1"))
ap.add_argument("--env", default="ct1")
ap.add_argument("--content", default="x1")
ap.add_argument("--methods", default="dara_online_lag_m015,ppo_lagrangian_t0.5,corel_official,pensieve_official")
ap.add_argument("--y", default="bitrate", choices=["bitrate", "quality", "J0"])
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default=str(ROOT / "figures" / "scenarios_ct1"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
METHODS = [m for m in a.methods.split(",") if m]
REF_M = METHODS[0]
YCOL, YLAB = {"bitrate": ("mean_bitrate_mbps", "Average bitrate (Mbps)"),
              "quality": ("mean_quality", "Mean quality level $q$"),
              "J0": ("J0", "Mean $J_0$ per session")}[a.y]

W2 = 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True})
INK = "#0b0b0b"
NAME = {"dara_online_lag_m015": "DARA-L", "dara_online_lag": "DARA-L", "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)",
        "corel_official": "COREL", "pensieve_official": "Pensieve", "robustmpc": "RobustMPC"}
COL = {m: R.colour(m) for m in METHODS + ["robustmpc"]}
COL["dara_online_lag_m015"] = R.colour("dara_online_lag")
MARK = {"dara_online_lag_m015": "D", "dara_online_lag": "D", "ppo_lagrangian_t0.5": "o", "corel_official": "s",
        "pensieve_official": "^", "robustmpc": "v"}
GROUPS = ["grp_walking", "grp_driving", "grp_deadzone"]
SCNS = ["scn_urban_pedestrian", "scn_crowded_stationary", "scn_urban_driving", "scn_motorway", "scn_suburban_open",
        "scn_tunnel", "scn_under_bridge", "scn_urban_canyon"]
LABEL = {"grp_walking": "Walking", "grp_driving": "Driving", "grp_deadzone": "Dead zone",
         "scn_urban_pedestrian": "Urban pedestrian", "scn_crowded_stationary": "Crowded stationary",
         "scn_urban_driving": "Urban driving", "scn_motorway": "Motorway", "scn_suburban_open": "Suburban open",
         "scn_tunnel": "Tunnel", "scn_under_bridge": "Under bridge", "scn_urban_canyon": "Urban canyon"}
GROUP_OF = {"grp_walking": "urban pedestrian, crowded stationary",
            "grp_driving": "urban driving, motorway, suburban open",
            "grp_deadzone": "tunnel, under bridge, urban canyon"}

G = load_generalisation(a.gen)
if G.empty:
    sys.exit(f"no generalize/gen_*.csv under {a.gen}")
if YCOL not in G:
    sys.exit(f"column {YCOL} missing: these results were produced before the notebook recorded it; re-run the sweep")
G = G[G.content == a.content]
ref = G[(G.run == "reference") & (G.method == "robustmpc")]
pol = G[(G.run != "reference") & (G.env == a.env) & G.train_method.isin(METHODS)]
METHODS = [m for m in METHODS if m in set(pol.train_method)]
BARS = METHODS + ["robustmpc"]
rng = np.random.default_rng(0)


def sessions(net, m):
    """(seeds, sessions) arrays for each measure; RobustMPC is one 'seed'."""
    if m == "robustmpc":
        d = ref[ref.network == net].sort_values("episode")
        return {c: d[c].to_numpy(float)[None, :] for c in ("rebuffer", "mean_bitrate_mbps", "mean_quality", "J0")}
    d = pol[(pol.network == net) & (pol.train_method == m)]
    return {c: d.pivot_table(index="seed", columns="episode", values=c).dropna(axis=1).to_numpy(float)
            for c in ("rebuffer", "mean_bitrate_mbps", "mean_quality", "J0")}


def boot_ci(X):
    """95% percentile CI of the mean of a (seeds, sessions) array: resample seeds, then the shared sessions."""
    S, N = X.shape
    if S == 0 or N == 0:
        return np.nan, np.nan
    si = rng.integers(0, S, size=(a.boot, S))
    ni = rng.integers(0, N, size=(a.boot, N))
    means = np.array([X[si[b]][:, ni[b]].mean() for b in range(a.boot)])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


rows = []
for net in GROUPS + SCNS:
    if not ((G.network == net).any()):
        continue
    for m in BARS:
        X = sessions(net, m)
        if X["J0"].size == 0:
            continue
        r = dict(network=net, scenario=LABEL.get(net, net), method=m, n_seeds=X["J0"].shape[0], sessions=X["J0"].shape[1])
        for c, k in (("mean_bitrate_mbps", "bitrate"), ("rebuffer", "freeze_s"), ("J0", "J0"), ("mean_quality", "quality")):
            lo, hi = boot_ci(X[c])
            r.update({k: X[c].mean(), f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(r)
T = pd.DataFrame(rows)
T.round(4).to_csv(OUT / "scenario_table.csv", index=False)
YK = {"bitrate": "bitrate", "quality": "quality", "J0": "J0"}[a.y]

# ------------------------------------------------------------------------------------------------ tests
tests = []
try:
    from scipy.stats import wilcoxon
    for net in T.network.unique():
        b = {c: v.mean(axis=0) for c, v in sessions(net, REF_M).items()}
        for m in [x for x in BARS if x != REF_M]:
            o = {c: v.mean(axis=0) for c, v in sessions(net, m).items()}
            for c in ("J0", "rebuffer", "mean_bitrate_mbps"):
                n = min(len(b[c]), len(o[c]))
                d = b[c][:n] - o[c][:n]
                p = wilcoxon(d).pvalue if n > 5 and np.any(d != 0) else 1.0
                tests.append(dict(network=net, scenario=LABEL.get(net, net), method=REF_M, against=m, value=c, sessions=n,
                                  mean_difference=d.mean(), p=p))
    TS = pd.DataFrame(tests)
    for c in TS.value.unique():
        k = TS.value == c
        p = TS.loc[k, "p"].to_numpy(); o = np.argsort(p); q = np.empty_like(p)
        q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        TS.loc[k, "p_bh"] = np.minimum(q, 1)
    TS["verdict"] = ["no difference" if r.p_bh >= 0.05 else ("higher" if r.mean_difference > 0 else "lower")
                     for r in TS.itertuples()]
    TS.round(5).to_csv(OUT / "scenario_tests.csv", index=False)
except ImportError:
    TS = pd.DataFrame()
    print("scipy not installed: scenario_tests.csv skipped")


# ------------------------------------------------------------------------------------------------ figures
def panel(ax, net):
    d = T[T.network == net]
    for m in BARS:
        r = d[d.method == m]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.errorbar(r.freeze_s, r[YK], xerr=[[r.freeze_s - r.freeze_s_lo], [r.freeze_s_hi - r.freeze_s]],
                    yerr=[[r[YK] - r[f"{YK}_lo"]], [r[f"{YK}_hi"] - r[YK]]], fmt=MARK[m], color=COL[m], mfc=COL[m],
                    mec=INK, mew=0.4, ms=4.5, elinewidth=0.7, capsize=1.5, zorder=3)
    lo = d.freeze_s_lo.min(); hi = d.freeze_s_hi.max(); pad = 0.08 * (hi - lo if hi > lo else 1)
    ax.set_xlim(hi + pad, max(lo - pad, 0) if lo - pad > 0 else lo - pad)       # reversed: less freeze to the right
    ylo = d[f"{YK}_lo"].min(); yhi = d[f"{YK}_hi"].max(); yp = 0.08 * (yhi - ylo if yhi > ylo else 1)
    ax.set_ylim(ylo - yp, yhi + yp)
    ax.set_xlabel("Freeze time (s/session)")


def legend(fig, top):
    h = [Line2D([], [], ls="none", marker=MARK[m], color=COL[m], mfc=COL[m], mec=INK, mew=0.4, ms=4.5, label=NAME[m])
         for m in BARS]
    fig.legend(handles=h, ncol=len(h), loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.4,
               handletextpad=0.3)
    fig.tight_layout(rect=[0, 0, 1, top], w_pad=0.8, h_pad=0.9)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("written", OUT / f"{name}.pdf")


nets = [n for n in GROUPS if n in set(T.network)]
if nets:
    fig, axes = plt.subplots(1, len(nets), figsize=(W2, 2.2), squeeze=False)
    for i, (ax, n) in enumerate(zip(axes.flat, nets)):
        panel(ax, n)
        ax.set_ylabel(YLAB if i == 0 else "")
        ax.text(-0.02, 1.03, f"({'abc'[i]}) {LABEL[n]}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    legend(fig, 0.90)
    save(fig, "fig_scenarios_groups")

nets = [n for n in SCNS if n in set(T.network)]
if nets:
    nc = 4; nr = int(np.ceil(len(nets) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(W2, 2.0 * nr + 0.3), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(nets):
            ax.set_visible(False); continue
        panel(ax, nets[i])
        ax.set_ylabel(YLAB if i % nc == 0 else "")
        ax.text(-0.02, 1.03, f"({'abcdefgh'[i]}) {LABEL[nets[i]]}", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8)
    legend(fig, 0.93)
    save(fig, "fig_scenarios_all")

# ------------------------------------------------------------------------------------------------ caption
ns = sorted(set(T.n_seeds[T.method != "robustmpc"]))
sess = int(T.sessions.min()) if len(T) else 0
yl = {"bitrate": "average bitrate", "quality": "mean quality level", "J0": "mean QoE $J_0$"}[a.y]
cap = (f"{yl[0].upper() + yl[1:]} and freeze time per session for each network scenario (models trained on the full "
       f"generator mix, ladder ×1; {sess} test sessions per scenario; freeze axis reversed, so the top right is better). "
       f"Error bars: 95% confidence intervals (hierarchical bootstrap over {'/'.join(map(str, ns))} training seeds and "
       f"the test sessions). Groups: " + "; ".join(f"{LABEL[g].lower()} = {GROUP_OF[g]}" for g in GROUPS) + ".")
L = ["# fig_scenarios_groups / fig_scenarios_all", "", cap, "",
     f"## Numbers: {yl} | freeze s/session | J0 (mean over seeds and sessions)", "",
     "| scenario | " + " | ".join(NAME[m] for m in BARS) + " |", "|" + "---|" * (len(BARS) + 1)]
for net in T.network.unique():
    d = T[T.network == net].set_index("method")
    L.append(f"| {LABEL.get(net, net)} | " + " | ".join(
        f"{d.loc[m, YK]:.1f} \\| {d.loc[m, 'freeze_s']:.1f} \\| {d.loc[m, 'J0']:.1f}" if m in d.index else "–" for m in BARS) + " |")
if len(TS):
    L += ["", f"Paired Wilcoxon, {NAME[REF_M]} against each method (BH within J0, freeze, bitrate): scenario_tests.csv"]
(OUT / "caption.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
