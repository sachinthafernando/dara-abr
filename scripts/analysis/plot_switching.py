#!/usr/bin/env python3
"""How each method changes quality, over ALL replayed test sessions (not one chosen session). IEEE, no title.

  (a) quality switches per session, stacked by switch size (1, 2, 3 or 4 tiers): bar height = switches per session
  (b) share of chunks played at each tier
  switching_table.csv : switches per session, share of each switch size, tier shares, mean sum |dq| per session
                        (the switching term of J0), per method; seeds pooled (each seed's sessions weigh equally)
  caption.md

Input: one or more timeline CSVs written by the notebook's MODE='timeline' (per-chunk rows).
Usage: python scripts/analysis/plot_switching.py --timeline runs/timeline/timeline_ct1.csv [more.csv] --out figures/switching_ct1
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument("--timeline", nargs="+", required=True)
ap.add_argument("--methods", default="dara_online_lag_m015,ppo_lagrangian_t0.5,corel_official,pensieve_official,robustmpc")
ap.add_argument("--out", default=str(ROOT / "figures" / "switching_ct1"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
ORDER = a.methods.split(",")

W1 = 3.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True})
NAME = {"dara_online_lag_m015": "DARA-L", "dara_online_lag": "DARA-L", "ppo_lagrangian_t0.5": "PPO-Lag.\n(0.5 s)",
        "ppo_lagrangian_t0.25": "PPO-Lag.\n(0.25 s)", "ppo_lagrangian_t1": "PPO-Lag.\n(1 s)", "corel_official": "COREL",
        "pensieve_official": "Pensieve", "robustmpc": "RobustMPC"}
SIZE_COL = ["#c9dcf3", "#7fa9df", "#2a62b0", "#0b2f66"]            # switch of 1, 2, 3, 4 tiers: light -> dark
TIER_COL = ["#f1efe8", "#d6d3c9", "#a9a59a", "#6f6b62", "#2f2c27"]  # tier 0 .. 4

T = pd.concat([pd.read_csv(f) for f in a.timeline], ignore_index=True)


def key(run):
    if run == "robustmpc":
        return "robustmpc", 0
    p = R.parse_run(run)
    return (p["method"], p["seed"]) if p else (run, 0)


T[["mkey", "seed"]] = pd.DataFrame([key(r) for r in T.method], index=T.index)
cnt = T.groupby(["method", "episode"]).chunk.transform("size")
bad = cnt < T.n_chunks                                           # a session cut short (e.g. a truncated file)
if bad.any():
    print("WARNING: dropping incomplete sessions:",
          T[bad].groupby("method").episode.unique().apply(list).to_dict())
    T = T[~bad].copy()
T = T.sort_values(["method", "episode", "chunk"])
T["dtier"] = (T.tier - T.prev_tier).abs()
T["dq"] = T.groupby(["method", "episode"]).quality.diff().abs()
first = T.chunk == T.groupby(["method", "episode"]).chunk.transform("min")
T.loc[first, "dtier"] = 0                                       # the first chunk's 'previous tier' is the random start

rows = []
for m in [x for x in ORDER if x in set(T.mkey)]:
    g = T[T.mkey == m]
    n_sess = g.groupby("seed").episode.nunique().sum()
    sw = g[g.dtier > 0]
    r = dict(method=m, seeds=g.seed.nunique(), sessions=int(n_sess), switches_per_session=len(sw) / n_sess,
             sum_abs_dq_per_session=g.dq.fillna(0).groupby([g.seed, g.episode]).sum().mean())
    for k in range(1, 5):
        r[f"switch_{k}_per_session"] = (sw.dtier == k).sum() / n_sess
        r[f"switch_{k}_share"] = (sw.dtier == k).mean() if len(sw) else 0.0
    for t in range(5):
        r[f"tier{t}_share"] = (g.tier == t).mean()
    rows.append(r)
S = pd.DataFrame(rows)
PS = (T.assign(dq0=T.dq.fillna(0)).groupby(["mkey", "seed", "episode"]).dq0.sum().groupby(["mkey", "seed"]).mean()
      .rename("sum_abs_dq").reset_index())
PS.round(4).to_csv(OUT / "switching_per_seed.csv", index=False)
S["sum_abs_dq_seed_sd"] = [PS[PS.mkey == m].sum_abs_dq.std(ddof=1) if (PS.mkey == m).sum() > 1 else np.nan
                           for m in S.method]
S.round(4).to_csv(OUT / "switching_table.csv", index=False)

LBL = [NAME.get(m, m).replace("\n", " ") for m in S.method]
y = np.arange(len(S))[::-1]                                      # first method on top
fig, axes = plt.subplots(1, 3, figsize=(2 * W1 * 1.02, 0.30 * len(S) + 1.05), sharey=True,
                         gridspec_kw=dict(wspace=0.12, width_ratios=[1.15, 1.15, 0.9]))
left = np.zeros(len(S))
for k in range(1, 5):
    v = S[f"switch_{k}_per_session"].to_numpy()
    axes[0].barh(y, v, 0.62, left=left, color=SIZE_COL[k - 1], edgecolor="white", lw=0.4, zorder=2,
                 label=f"{k} tier{'s' if k > 1 else ''}")
    left += v
axes[0].set_xlabel("Quality switches per session")
axes[0].legend(title="Switch size", title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4,
               handlelength=1.0, columnspacing=0.8)
left = np.zeros(len(S))
for t in range(5):
    v = 100 * S[f"tier{t}_share"].to_numpy()
    axes[1].barh(y, v, 0.62, left=left, color=TIER_COL[t], edgecolor="white", lw=0.4, zorder=2, label=str(t))
    left += v
axes[1].set_xlabel("Chunks at each tier (%)"); axes[1].set_xlim(0, 100)
h, l = axes[1].get_legend_handles_labels()
axes[1].legend(h, l, title="Tier", title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5,
               handlelength=1.0, columnspacing=0.8)
axes[2].barh(y, S.sum_abs_dq_per_session, 0.62, color="#9fb7d9", edgecolor="white", lw=0.4, zorder=2)
for i, m in enumerate(S.method):
    v = PS[PS.mkey == m].sum_abs_dq.to_numpy()
    if len(v) > 1:
        axes[2].scatter(v, np.full(len(v), y[i]) + np.linspace(-0.18, 0.18, len(v)), s=7, color="#0b2f66", zorder=3, lw=0)
axes[2].set_xlabel(r"$\sum|\Delta q|$ per session")
axes[0].set_yticks(y); axes[0].set_yticklabels(LBL)
for ax, s_ in zip(axes, "abc"):
    ax.grid(axis="y", visible=False)
    ax.text(0.0, 1.03, f"({s_})", transform=ax.transAxes, fontsize=8, va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_switching.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=300 if ext == "png" else None)
plt.close(fig)

seeds = sorted(set(S.seeds[S.method != "robustmpc"]))
cap = (f"Quality changes over all replayed test sessions ({int(S.sessions.min())}–{int(S.sessions.max())} sessions per "
       f"method; {'/'.join(map(str, seeds))} training seed{'s' if max(seeds) > 1 else ''} per learned method): (a) quality "
       "switches per session, by switch size in tiers; (b) share of chunks played at each tier; (c) switching term of $J_0$, "
       "$\\sum|\\Delta q|$ per session (bars: mean; dots: training seeds).")
L = ["# fig_switching", "", cap, "", "| method | switches/session | 1 | 2 | 3 | 4 tiers | sum abs dq per session | tier 4 share |",
     "|---|---|---|---|---|---|---|---|"]
for r in S.itertuples():
    L.append(f"| {NAME.get(r.method, r.method).replace(chr(10), ' ')} | {r.switches_per_session:.2f} | "
             + " | ".join(f"{getattr(r, f'switch_{k}_share'):.0%}" for k in range(1, 5))
             + f" | {r.sum_abs_dq_per_session:.2f} | {r.tier4_share:.0%} |")
(OUT / "caption.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
