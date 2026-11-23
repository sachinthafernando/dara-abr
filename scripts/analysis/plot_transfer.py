#!/usr/bin/env python3
"""Transfer figure: models trained on one environment (default: generator, ladder x1) played, without retraining, on
other networks and chunk-size ladders. IEEE double-column figure, no title, vector PDF + PNG.

  Panels : (a) mean J0 per session   (b) freeze time per session (log scale); ladders separated by a dashed line
  Groups : network (generator = the training network, Lumos5G, Irish 5G static, Irish 5G driving, harsh mmWave scenarios)
  Bars   : each learned method (mean over training seeds, error bar = s.d. over seeds) and RobustMPC;

Also writes transfer_table.csv (every number in the figure, plus mean tier and switches), transfer_tests.csv
(per condition: each method against the first method, paired Wilcoxon on the seed-averaged sessions, BH-corrected)
and caption.md.

Input: the generalize/gen_*.csv files of a generalisation sweep, e.g. configs/sweeps/generalise_ct1.tsv
Usage: python scripts/analysis/plot_transfer.py --gen runs/generalise_ct1 --env ct1 --out figures/transfer_ct1
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
warnings.filterwarnings("ignore", message="Tight layout not applied")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401
from dara_abr.results import load_generalisation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=str(ROOT / "runs" / "generalise_ct1"), help="folder holding generalize/gen_*.csv")
ap.add_argument("--env", default="ct1", help="training environment of the models")
ap.add_argument("--methods", default="dara_online_lag_m015,ppo_lagrangian_t0.25,ppo_lagrangian_t0.5,ppo_lagrangian_t1,"
                                      "corel_official,pensieve_official")
ap.add_argument("--networks", default="seen,lumos5g,irish_static,gen_harsh")
ap.add_argument("--contents", default="x1,x4")
ap.add_argument("--out", default=str(ROOT / "figures" / "transfer_ct1"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
METHODS = [m for m in a.methods.split(",") if m]
NETS, CONTS = a.networks.split(","), a.contents.split(",")

W2 = 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True})
INK, INK2 = "#0b0b0b", "#52514e"
NAME = {"dara_online_lag": "DARA-L (thr. 0.2)", "dara_online_lag_m015": "DARA-L",
        "ppo_lagrangian_t0.25": "PPO-Lag. (0.25 s)", "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)",
        "ppo_lagrangian_t1": "PPO-Lag. (1 s)", "corel_official": "COREL", "pensieve_official": "Pensieve",
        "robustmpc": "RobustMPC"}
COL = {m: R.colour(m) for m in METHODS + ["robustmpc"]}
COL["dara_online_lag"] = "#8db7ea"
COL["dara_online_lag_m015"] = R.colour("dara_online_lag")   # DARA-L is reported at act threshold 0.15
COL.update({"ppo_lagrangian_t0.25": "#f4bfd3", "ppo_lagrangian_t1": "#a8406b"})   # light -> dark = tighter -> looser
HATCH = {"dara_online_lag_m015": "", "dara_online_lag": "", "ppo_lagrangian_t0.5": "////", "ppo_lagrangian_t0.25": "////", "ppo_lagrangian_t1": "////", "corel_official": "\\\\\\\\",
         "pensieve_official": "....", "robustmpc": "xxxx"}
NET_NAME = {"seen": "Generator", "lumos5g": "Lumos5G", "irish_static": "Irish 5G\nstatic",
            "irish_driving": "Irish 5G\ndriving", "gen_harsh": "Harsh\nmmWave", "gen_half": "Generator\n0.5× rate"}
CONT_NAME = {"x1": "ladder ×1", "x4": "ladder ×4", "x8": "ladder ×8", "x4_bursty": "ladder ×4, bursty"}

G = load_generalisation(a.gen)
if G.empty:
    sys.exit(f"no generalize/gen_*.csv under {a.gen}")
ref = G[G.run == "reference"]
pol = G[(G.run != "reference") & (G.env == a.env) & G.train_method.isin(METHODS)]
METHODS = [m for m in METHODS if m in set(pol.train_method)]
missing = sorted(set(a.methods.split(",")) - set(METHODS))
if missing:
    print("WARNING: no results for", ", ".join(missing))
CONDS = [(n, c) for c in CONTS for n in NETS if ((G.network == n) & (G.content == c)).any()]

# ------------------------------------------------------------------------------------------------ table
rows = []
for n, c in CONDS:
    cr = ref[(ref.network == n) & (ref.content == c)]
    g = cr[cr.method == "robustmpc"]
    if len(g):
        rows.append(dict(network=n, content=c, method="robustmpc", detail="", n_seeds=1, sessions=len(g), J0=g.J0.mean(),
                         J0_seed_sd=np.nan, freeze_s=g.rebuffer.mean(), freeze_seed_sd=np.nan, mean_tier=g.mean_tier.mean(),
                         switches=g.switches.mean()))
    for m in METHODS:
        g = pol[(pol.network == n) & (pol.content == c) & (pol.train_method == m)]
        if not len(g):
            continue
        s = g.groupby("seed")[["J0", "rebuffer", "mean_tier", "switches"]].mean()
        rows.append(dict(network=n, content=c, method=m, detail="seeds " + " ".join(str(int(x)) for x in s.index),
                         n_seeds=len(s), sessions=int(g.groupby("seed").size().min()), J0=s.J0.mean(),
                         J0_seed_sd=s.J0.std(ddof=1) if len(s) > 1 else np.nan, freeze_s=s.rebuffer.mean(),
                         freeze_seed_sd=s.rebuffer.std(ddof=1) if len(s) > 1 else np.nan, mean_tier=s.mean_tier.mean(),
                         switches=s.switches.mean()))
T = pd.DataFrame(rows)
T.round(4).to_csv(OUT / "transfer_table.csv", index=False)


def cell(n, c, m, col="J0"):
    r = T[(T.network == n) & (T.content == c) & (T.method == m)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ------------------------------------------------------------------------------------------------ paired tests
tests = []
try:
    from scipy.stats import wilcoxon
    base = METHODS[0] if "dara_online_lag_m015" not in METHODS else "dara_online_lag_m015"
    for n, c in CONDS:
        def sess(m):
            if m == "robustmpc":
                g = ref[(ref.network == n) & (ref.content == c) & (ref.method == "robustmpc")]
                return g.set_index("episode")[["J0", "rebuffer"]]
            g = pol[(pol.network == n) & (pol.content == c) & (pol.train_method == m)]
            return g.groupby("episode")[["J0", "rebuffer"]].mean()
        b = sess(base)
        for m in [x for x in METHODS if x != base] + ["robustmpc"]:
            o = sess(m)
            idx = b.index.intersection(o.index)
            for col in ("J0", "rebuffer"):
                d = b.loc[idx, col].to_numpy() - o.loc[idx, col].to_numpy()
                p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else 1.0
                tests.append(dict(network=n, content=c, method=base, against=m, value=col, sessions=len(idx),
                                  mean_difference=d.mean(), median_difference=np.median(d), p=p))
    TS = pd.DataFrame(tests)
    if len(TS):
        for col in ("J0", "rebuffer"):                         # Benjamini-Hochberg within each measure
            k = TS.value == col
            p = TS.loc[k, "p"].to_numpy(); o = np.argsort(p); q = np.empty_like(p)
            q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
            TS.loc[k, "p_bh"] = np.minimum(q, 1)
        TS["verdict"] = [("no difference" if r.p_bh >= 0.05 else
                          (("higher" if r.mean_difference > 0 else "lower")))
                         for r in TS.itertuples()]
        TS.round(5).to_csv(OUT / "transfer_tests.csv", index=False)
except ImportError:
    print("scipy not installed: transfer_tests.csv skipped")

# ------------------------------------------------------------------------------------------------ figure
bars = METHODS + ["robustmpc"]
k = len(bars); width = 0.84; w = width / k
X = np.arange(len(CONDS))
fig, axes = plt.subplots(2, 1, figsize=(W2, 3.3), sharex=True)
for ax, (col, sdcol, lab) in zip(axes, [("J0", "J0_seed_sd", "Mean $J_0$ per session"),
                                        ("freeze_s", "freeze_seed_sd", "Freeze time (s/session)")]):
    for j, m in enumerate(bars):
        x = X + (j - (k - 1) / 2) * w
        v = np.array([cell(n, c, m, col) for n, c in CONDS])
        e = np.array([cell(n, c, m, sdcol) for n, c in CONDS])
        if col == "freeze_s":
            v = np.maximum(v, 0.05)                         # log axis: a zero freeze is drawn at the floor
        ax.bar(x, v, w * 0.92, color=COL[m], hatch=HATCH.get(m, ""), edgecolor=INK2 if HATCH.get(m) else "white",
               lw=0.4, zorder=2)
        if np.isfinite(e).any():
            ax.errorbar(x, v, yerr=np.where(np.isfinite(e), e, 0), fmt="none", ecolor=INK, elinewidth=0.5,
                        capsize=1, zorder=3)
    if col == "J0":
        lo = np.nanmin([cell(n, c, m, "J0") - np.nan_to_num(cell(n, c, m, "J0_seed_sd")) for n, c in CONDS for m in bars])
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_ylim(min(0, lo * 1.08), None)
    else:
        pos = [cell(n, c, m, "freeze_s") for n, c in CONDS for m in bars]
        pos = [x for x in pos if np.isfinite(x) and x > 0.05]
        ax.set_yscale("log"); ax.set_ylim(min(1.0, 0.5 * min(pos)) if pos else 0.05, None)
    ax.set_ylabel(lab)
    ax.set_xlim(-0.55, len(CONDS) - 0.45)
    ax.grid(axis="x", visible=False)
    for b in [i for i in range(1, len(CONDS)) if CONDS[i][1] != CONDS[i - 1][1]]:     # divider between ladders
        ax.axvline(b - 0.5, color=INK2, lw=0.6, ls="--")
axes[1].set_xticks(X)
try:
    TRAIN_C = f"x{R.TRAINING_ENVS[a.env]['ladder_scale']:g}"
except Exception:
    TRAIN_C = None
axes[1].set_xticklabels([NET_NAME.get(n, n) + ("\n(training)" if (n, c) == ("seen", TRAIN_C) else "") for n, c in CONDS])
for c in CONTS:                                              # ladder label centred under its groups
    idx = [i for i, (_, cc) in enumerate(CONDS) if cc == c]
    if idx:
        axes[1].text(np.mean(idx), -0.34, f"Ladder {CONT_NAME.get(c, c).split()[-1]}", transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
axes[0].text(-0.02, 1.02, "(a)", transform=axes[0].transAxes, fontsize=8, va="bottom", ha="left")
axes[1].text(-0.02, 1.02, "(b)", transform=axes[1].transAxes, fontsize=8, va="bottom", ha="left")
handles = [Patch(facecolor=COL[m], hatch=HATCH.get(m, ""), edgecolor=INK2 if HATCH.get(m) else "white", lw=0.4,
                 label=NAME.get(m, m)) for m in bars]
axes[0].legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, 1.06), columnspacing=1.2,
               handlelength=1.4)
fig.tight_layout(h_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_transfer_{a.env}.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=300 if ext == "png" else None)
plt.close(fig)

# ------------------------------------------------------------------------------------------------ caption
ns = sorted(set(T.n_seeds[T.method.isin(METHODS)]))
sess = int(T.sessions.min())
env_name = {"ct1": "the mmWave generator with ladder ×1"}.get(a.env, a.env)
cap = (f"Models trained on {env_name}, tested without retraining on other networks and ladders: mean QoE $J_0$ and "
       f"freeze time per session (log scale) over {sess} test sessions per condition; error bars: s.d. over "
       f"{'/'.join(map(str, ns))} training seed{'s' if max(ns) > 1 else ''}. Generator ×1 is the training condition.")
lines = [f"# fig_transfer_{a.env}", "", cap, "", "## Numbers (mean J0 | freeze s/session)", ""]
hdr = "| condition | " + " | ".join(NAME.get(m, m) for m in bars) + " |"
lines += [hdr, "|" + "---|" * (len(bars) + 1)]
for n, c in CONDS:
    lines.append(f"| {NET_NAME.get(n, n).replace(chr(10), ' ')}, {c} | " +
                 " | ".join(f"{cell(n, c, m):.1f} \\| {cell(n, c, m, 'freeze_s'):.1f}" for m in bars) + " |")
if tests:
    lines += ["", f"Paired Wilcoxon tests ({NAME.get(TS.method.iloc[0])} against each method, sessions averaged over seeds, "
              "BH-corrected within J0 and within freeze): transfer_tests.csv"]
(OUT / "caption.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("written", OUT / f"fig_transfer_{a.env}.pdf")
