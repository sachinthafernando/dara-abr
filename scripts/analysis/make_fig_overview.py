#!/usr/bin/env python3
"""Fig. 1: DARA / DARA-L overview (IEEE single column, no title). Boxes follow the notebook's implementation:
collect_diag_states -> counterfactual(mode 'hold') -> landscape_signals (M, intended_freeze_per_chunk) -> act_lag.

Usage: python scripts/analysis/make_fig_overview.py [--threshold 0.15] [--out figures/paper]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--n-states", type=int, default=360)
ap.add_argument("--horizon", type=int, default=10)
ap.add_argument("--margin", type=float, default=0.5)
ap.add_argument("--every", type=int, default=1000)
ap.add_argument("--out", default=str(ROOT / "figures" / "paper"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman",
                                                            "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
INK, INK2 = "#0b0b0b", "#52514e"
FILL_D, FILL_T, EDGE_D = "#e8f0fb", "#fbeef3", "#2a78d6"

W, H = 3.5, 3.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text, fill, edge=INK2, lw=0.6, fs=7, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=fill, ec=edge, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.07
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(x0, y0, x1, y1, label=None, lx=0.0, ly=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, mutation_scale=7, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=7, color=INK2, ha="center", va="center")


X0, BW = 0.40, 3.02                      # main column
y = H - 0.08
# 0. input
h = 0.30; y -= h
box(X0, y, BW, h, "PPO training on the ABR environment with training reward $r$", "#f4f3ef")
yin = y
# DARA group
h = 0.42; y -= 0.13 + h
box(X0, y, BW, h, f"(1) Diagnostic states: {a.n_states} snapshots of the environment,\nstratified by throughput, "
                   "volatility, buffer and trend", FILL_D, EDGE_D)
arrow(X0 + BW / 2, yin, X0 + BW / 2, y + h)
y1 = y
h = 0.50; y -= 0.13 + h
box(X0, y, BW, h, f"(2) Counterfactual rollouts: from each state $s$, hold tier $a$ for {a.horizon} chunks\n"
                   r"$\rightarrow\ Q_r(s,a)$: return of the training reward" "\n"
                   r"$\rightarrow\ Q_J(s,a)$: return of the intended QoE $J_0$, and its freeze", FILL_D, EDGE_D)
arrow(X0 + BW / 2, y1, X0 + BW / 2, y + h)
y2 = y
h = 0.42; y -= 0.13 + h
box(X0, y, BW, h, r"(3) Misalignment: $M = \Pr_s\!\left[\,Q_J(s,a_r^*) < \max_a Q_J(s,a) - \delta\,\right]$," "\n"
                   r"$a_r^* = \arg\max_a Q_r(s,a)$,  $\delta$ = " + f"{a.margin:g}", FILL_D, EDGE_D)
arrow(X0 + BW / 2, y2, X0 + BW / 2, y + h)
y3 = y
# decision diamond
dh, dw = 0.36, 1.0; cy = y - 0.13 - dh / 2; cx = X0 + BW / 2
ax.add_patch(Polygon([(cx, cy + dh / 2), (cx + dw / 2, cy), (cx, cy - dh / 2), (cx - dw / 2, cy)], closed=True,
                     fc="white", ec=EDGE_D, lw=0.7))
ax.text(cx, cy, r"$M \geq \tau$ ?" + f"\n($\\tau$ = {a.threshold:g})", ha="center", va="center", fontsize=7.2,
        linespacing=1.1)
arrow(cx, y3, cx, cy + dh / 2)
# no branch
nb_x = cx + dw / 2 + 0.30
box(nb_x, cy - 0.17, X0 + BW - nb_x, 0.34, "keep PPO\nwith $r$", "#f4f3ef", fs=7)
arrow(cx + dw / 2, cy, nb_x, cy, "no", ly=0.07)
# yes branch
h = 0.40; y = cy - dh / 2 - 0.15 - h
box(X0, y, BW, h, r"(4) Derived freeze budget: $\hat b$ = mean freeze per chunk" "\n"
                   r"of the intended-best tier $\arg\max_a Q_J(s,a)$", FILL_D, EDGE_D)
arrow(cx, cy - dh / 2, cx, y + h, "yes", lx=0.13)
y4 = y
h = 0.42; y -= 0.13 + h
box(X0, y, BW, h, r"(5) PPO-Lagrangian: reward $r - \lambda\,F$, $F$ = freeze (s)" "\n"
                   r"$\lambda \leftarrow \max\!\left(0,\ \lambda + \eta\,(\bar F_{\mathrm{chunk}} - \hat b)\right)$",
    FILL_T, "#c2577f")
arrow(cx, y4, cx, y + h)
# DARA bracket over steps 1-4 (structural label)
top, bot = y1 + 0.42, y4
bx = X0 - 0.12
ax.plot([bx + 0.06, bx, bx, bx + 0.06], [top, top, bot, bot], color=EDGE_D, lw=0.8)
ax.text(bx - 0.06, (top + bot) / 2, "DARA", rotation=90, ha="center", va="center", fontsize=8, color=EDGE_D)
ax.text(bx - 0.20, (top + bot) / 2, f"checked every {a.every} episodes", rotation=90, ha="center", va="center",
        fontsize=6.5, color=INK2)
# bracket DARA-L over 1-5
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_overview.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=300 if ext == "png" else None)
print("written", OUT / "fig_overview.pdf")
