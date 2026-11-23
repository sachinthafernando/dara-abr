#!/usr/bin/env python3
"""One test session on a time line: network throughput, and for each method the quality level of every chunk, where
it stalled and for how long, and every quality switch (from -> to). IEEE double-column figure, PDF + PNG, plus a
table of the events.

Input: the timeline written by the notebook in MODE='timeline' (runs/timeline/timeline_<tag>.csv and _network.csv).
Session: --episode N, or chosen automatically as a typical session in which every learned method stalls at least once
(40-110 chunks; the one whose freeze per method is closest to that method's average over all sessions).

Usage: python scripts/analysis/plot_session.py --timeline runs/timeline/timeline_ct1.csv [--episode 17]
       [--labels "ct1__dara_online_lag_m015_s101=DARA-L,ct1__ppo_lagrangian_t0.5_s101=PPO-Lag. (0.5 s),..."]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402
from dara_abr import mpl_compat  # noqa: E402,F401  (older matplotlib + pandas 2)

ap = argparse.ArgumentParser()
ap.add_argument("--timeline", required=True)
ap.add_argument("--network", default=None, help="default: <timeline>_network.csv")
ap.add_argument("--episode", type=int, default=None)
ap.add_argument("--labels", default="", help="run=label pairs, comma-separated; default: from the run names")
ap.add_argument("--out", default=str(ROOT / "figures" / "paper"))
a = ap.parse_args()
TLP = Path(a.timeline)
TL = pd.read_csv(TLP)
NW = pd.read_csv(a.network or TLP.with_name(TLP.stem + "_network.csv"))
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                           "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5, "axes.axisbelow": True})
INK, INK2, STALL = "#0b0b0b", "#52514e", "#e34948"
K0 = 8.0                                                    # the session clock starts after the K = 8 history seconds

# ------------------------------------------------------------------------------------------------ labels and order
lab = dict(x.split("=", 1) for x in a.labels.split(",") if "=" in x)
methods = list(dict.fromkeys(TL.method))
if lab:                                                     # --labels also selects the runs to draw, in that order
    methods = [m for m in lab if m in set(methods)]
    TL = TL[TL.method.isin(methods)].copy()


def label(m):
    if m in lab:
        return lab[m]
    if m == "robustmpc":
        return "RobustMPC"
    p = R.parse_run(m)
    if not p:
        return m
    k = p["method"]
    return {"dara_online_lag": "DARA-L", "dara_online_lag_m015": "DARA-L (threshold 0.15)",
            "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)", "corel_official": "COREL",
            "pensieve_official": "Pensieve"}.get(k, R.label(k))


def colour(m):
    p = R.parse_run(m)
    return R.colour("robustmpc" if m == "robustmpc" else (p["method"] if p else m)).replace(
        R.colour("dara_online_lag_m015"), R.colour("dara_online_lag"))


env = (R.parse_run(methods[0]) or {}).get("env", "")

# ------------------------------------------------------------------------------------------------ choose the session
per = TL.groupby(["method", "episode"]).agg(freeze=("stall", "sum"), stalls=("stall", lambda s: int((s > 0.05).sum())),
                                            j0=("j0_chunk", "sum"), n=("chunk", "size")).reset_index()
learned = [m for m in methods if m != "robustmpc"]
if a.episode is not None:
    ep = a.episode
    why = "chosen by --episode"
else:
    piv = per.pivot(index="episode", columns="method", values="stalls")[learned]
    nchunks = per.groupby("episode").n.first()
    mean_f = per.groupby("method").freeze.mean()
    cand = piv.index[(piv > 0).all(axis=1) & nchunks.loc[piv.index].between(40, 110)]
    why = "typical session in which every learned method stalls (40-110 chunks)"
    if len(cand) == 0:
        cand = piv.index[(piv > 0).sum(axis=1) >= 2]
        why = "typical session in which at least two learned methods stall"
    if len(cand) == 0:
        cand = piv.index
        why = "typical session"
    # typical = close to each method's average freeze AND average J0, preferring sessions in which the methods
    # rank in the same order as on average
    fz = per[per.method.isin(learned)].pivot(index="episode", columns="method", values="freeze").loc[cand]
    jj = per[per.method.isin(learned)].pivot(index="episode", columns="method", values="j0").loc[cand]
    mean_j = per.groupby("method").j0.mean()[learned]
    score = (abs(fz - mean_f[learned]) / mean_f[learned].clip(lower=0.5)).sum(axis=1) + \
            (abs(jj - mean_j) / mean_j.abs().clip(lower=10)).sum(axis=1)
    # +1 for every pair of methods whose J0 order in the session is the reverse of a clear (> 5%) average difference
    for i_, a_ in enumerate(learned):
        for b_ in learned[i_ + 1:]:
            gap_ = mean_j[a_] - mean_j[b_]
            if abs(gap_) > 0.05 * max(abs(mean_j[a_]), abs(mean_j[b_])):
                score = score + (np.sign(jj[a_] - jj[b_]) != np.sign(gap_)).astype(float)
    why += "; close to each method's average J0 and freeze"
    ep = int(score.idxmin())
S = TL[TL.episode == ep].copy()
n_chunks = int(S.n_chunks.iloc[0]); scen = S.scenario.iloc[0]
print(f"session {ep} ({why}): {n_chunks} chunks of 4 s, scenario {scen}")
print(per[per.episode == ep].assign(mean_freeze=per.groupby("method").freeze.transform("mean"),
                                     mean_j0=per.groupby("method").j0.transform("mean"))[["method", "j0", "mean_j0", "freeze", "mean_freeze"]].round(1).to_string(index=False))

# ------------------------------------------------------------------------------------------------ figure
nm = len(methods)
fig, axes = plt.subplots(nm + 1, 1, figsize=(7.16, 1.05 + 1.25 * nm), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.0] + [1.15] * nm))
end = float(S.dl_end.max() + 4.0 - K0)
nw = NW[NW.episode == ep]
ax = axes[0]
ax.plot(nw.second.to_numpy(float) - K0, nw.mbps.to_numpy(float), color="#3987e5", lw=0.7)
ax.set_yscale("log"); ax.set_ylabel("Throughput\n(Mbit/s)")
rates = None
try:
    import json
    cfg = json.load(open(Path(TLP).parents[1] / "train" / learned[0] / "config.json")) if learned else None
except Exception:
    cfg = None
base = (cfg or {}).get("BASE_CHUNK_MB")
if base:
    unit = 8.0 if (cfg or {}).get("UNITS_FIX", 1) else 1.0
    rates = [b * unit / 4.0 for b in base]
    for k, r_ in enumerate(rates):
        ax.axhline(r_, color=INK2, lw=0.4, ls=":")
        if k in (0, len(rates) - 1):
            ax.text(end, r_, f" q{k} {r_:g}", fontsize=6, color=INK2, va="center", ha="left", clip_on=False)
ax.text(0.0, 1.03, f"(a) Network throughput (scenario: {scen.replace('_', ' ')}); dotted: nominal bitrate of each quality level",
        transform=ax.transAxes, fontsize=7.5)
events = []
LABELS = []                                                 # (axes, text, colour, size, candidate positions)
gap = 0.035 * end                                           # labels closer than this are moved apart
for i, m in enumerate(methods, 1):
    ax = axes[i]
    d = S[S.method == m].sort_values("chunk")
    t0 = d.dl_start.to_numpy(float) - K0
    t1 = np.append(t0[1:], d.dl_end.to_numpy(float)[-1] - K0)
    q = d.tier.to_numpy(int)
    for x0, x1, qq in zip(t0, t1, q):
        ax.plot([x0, x1], [qq, qq], color=colour(m), lw=1.6, solid_capstyle="butt")
    for k in range(1, len(q)):
        if q[k] != q[k - 1]:
            ax.plot([t0[k], t0[k]], [q[k - 1], q[k]], color=colour(m), lw=0.8)
    # stalls: playback stops when the buffer runs out during a download
    nst, flip = 0, 0
    for r in d.itertuples():
        if r.stall > 0.05:
            nst += 1
            s0 = r.dl_start - K0 + r.buffer_before
            ax.axvspan(s0, s0 + r.stall, color=STALL, alpha=0.35, lw=0)
            xm = s0 + r.stall / 2
            LABELS.append((ax, f"{r.stall:.1f} s", STALL, 5.8,
                           [(xm, 4.4), (xm, 5.25), (xm, 6.1)]))
            events.append(dict(method=label(m), time_s=round(s0, 1), chunk=int(r.chunk), event="stall",
                               detail=f"{r.stall:.2f} s at quality q{int(r.tier)}"))
    # switches
    for k in range(1, len(q)):
        if q[k] != q[k - 1]:
            hi_, lo_ = max(q[k], q[k - 1]), min(q[k], q[k - 1])
            # labels stay directly above or below the switch they describe (never shifted sideways)
            cands = [(t0[k], lo_ - 0.62), (t0[k], lo_ - 1.35)]
            if hi_ < 4:
                cands = [(t0[k], hi_ + 0.18)] + cands + [(t0[k], hi_ + 0.9)]
            cands += [(t0[k], 4.4), (t0[k], 5.25), (t0[k], 6.1)]   # last resort: the free rows above level 4
            LABELS.append((ax, f"{q[k-1]}→{q[k]}", INK2, 5.5, cands))
            events.append(dict(method=label(m), time_s=round(t0[k], 1), chunk=int(d.chunk.iloc[k]), event="switch",
                               detail=f"q{q[k-1]} -> q{q[k]}"))
    first = d.iloc[0]
    if int(first.prev_tier) != int(first.tier):
        events.append(dict(method=label(m), time_s=0.0, chunk=0, event="switch",
                           detail=f"q{int(first.prev_tier)} -> q{int(first.tier)} (from the session's start level)"))
    nsw = int((np.diff(q) != 0).sum())
    j0 = float(d.j0_chunk.sum()); fr = float(d.stall.sum())
    ax.set_ylim(-1.9, 7.0); ax.set_yticks(range(5)); ax.set_ylabel("Quality")
    ax.text(0.0, 1.03, f"({'bcdefgh'[i-1]}) {label(m)}: $J_0$ = {j0:.0f}, freeze {fr:.1f} s in {nst} stall"
            f"{'s' if nst != 1 else ''}, {nsw} switch{'es' if nsw != 1 else ''}, mean quality {q.mean():.2f}",
            transform=ax.transAxes, fontsize=7.5)
axes[-1].set_xlabel("Session time (s)")
axes[-1].set_xlim(0, end)
fig.legend(handles=[Patch(fc=STALL, alpha=0.35, label="stall (label: its duration)"),
                    Patch(fc="none", ec="none", label="a→b: quality switch from level a to level b")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(h_pad=1.1)
# place every label at the first candidate position that overlaps no earlier label (greedy, in display space)
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
placed = {}
for ax_, txt, col, fs, cands in LABELS:
    boxes = placed.setdefault(id(ax_), [])
    keep = None
    for x, y in cands:
        t = ax_.text(x, y, txt, fontsize=fs, color=col, ha="center", va="bottom")
        bb = t.get_window_extent(rnd).expanded(1.12, 1.1)
        inside = ax_.bbox.y0 <= bb.y0 and bb.y1 <= ax_.bbox.y1
        if inside and not any(bb.overlaps(b) for b in boxes):
            keep = t; break
        t.remove()
    if keep is None:
        x, y = cands[0]
        keep = ax_.text(x, y, txt, fontsize=fs, color=col, ha="center", va="bottom")
    boxes.append(keep.get_window_extent(rnd).expanded(1.08, 1.15))
stem = f"fig_session_{env or 'env'}_ep{ep}"
fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
E = pd.DataFrame(events).sort_values(["method", "time_s"])
E.to_csv(OUT / f"{stem}_events.csv", index=False)
cap = (f"Test session {ep} of the shared 100 ({n_chunks} chunks of 4 s, {scen.replace('_', ' ')} scenario): (a) network "
       "throughput; " + ", ".join(f"({'bcdefgh'[i]}) {label(m)}" for i, m in enumerate(methods)) +
       ": quality level of each chunk over the session (drawn while the chunk downloads), stalls (red, with duration) "
       "and quality switches (from→to).")
(OUT / f"{stem}_caption.md").write_text(cap + "\n")
print("written", OUT / f"{stem}.pdf"); print(cap)
print(E.to_string(index=False))
