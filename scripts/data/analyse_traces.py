#!/usr/bin/env python3
"""Step 3 of the data pipeline: describe every network the agents are trained or tested on.

Networks: the mmWave generator (training mix, each scenario, each channel state), Lumos5G (all, walking, driving)
and Irish 5G (static, driving). For each: hours, throughput quantiles, share below 5 Mbit/s, lag-1
autocorrelation of log10 throughput, and for each chunk-size ladder (x1, x4, x8) the share of seconds in which
the network could sustain each tier's nominal bitrate (mean chunk size x 8 / 4 s).

Writes results/data/{network_stats.csv, channel_states.csv, ladder_feasibility.csv, network_report.txt}.
Generator samples: --sessions sessions of --seconds s per scenario, seed 2026 (deterministic).

Usage:  python scripts/data/analyse_traces.py [--sessions 200 --seconds 600]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr.urban_generator import generate_session, SCENARIO_MIX, REGIME_NAMES  # noqa: E402
from dara_abr import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", type=int, default=200)
ap.add_argument("--seconds", type=int, default=600)
ap.add_argument("--out", default=str(ROOT / "results" / "data"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)


def lag1(x):
    x = np.log10(np.maximum(np.asarray(x, float), 0.1))
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def stats(name, family, traces):
    v = np.concatenate(traces)
    return dict(network=name, family=family, traces=len(traces), hours=round(len(v) / 3600, 2),
                mean=v.mean(), p10=np.quantile(v, .1), median=np.median(v), p90=np.quantile(v, .9), max=v.max(),
                share_below_5=(v < 5).mean(), lag1_log10=np.nanmean([lag1(t) for t in traces]))


# ---------------------------------------------------------------------------------- generator samples
rng = np.random.default_rng(2026)
gen, states = {}, []
for scn in SCENARIO_MIX:
    tr = []
    for _ in range(a.sessions):
        g = generate_session(scn, a.seconds, rng)
        tr.append(g["throughput"])
        states.append(pd.DataFrame(dict(scenario=scn, state=g["regime"], mbps=g["throughput"])))
    gen[scn] = tr
S = pd.concat(states, ignore_index=True)
w = np.array(list(SCENARIO_MIX.values()), float); w /= w.sum()
mix = []                         # the training mix: scenarios in proportion to SCENARIO_MIX
for scn, wi in zip(SCENARIO_MIX, w):
    mix += gen[scn][:max(1, int(round(wi * a.sessions)))]

NETS = [("generator: training mix", "mmwave_synthetic", mix)]
NETS += [(f"generator: {s}", "mmwave_synthetic", gen[s]) for s in SCENARIO_MIX]
for fname, fam, label in ((R.LUMOS_TRACES, "mmwave_real", "Lumos5G"), (R.IRISH_TRACES, "non_mmwave_5g", "Irish 5G")):
    p = ROOT / fname
    if not p.exists():
        print(f"(skipping {label}: {fname} not found - run prepare_traces.py)")
        continue
    z = np.load(p)
    NETS.append((f"{label}: all", fam, [z[k] for k in z.files]))
    for pre in sorted({k.split("_")[0] for k in z.files}):
        NETS.append((f"{label}: {pre.lower()}", fam, [z[k] for k in z.files if k.startswith(pre + "_")]))

T = pd.DataFrame([stats(*n) for n in NETS])
T.round(3).to_csv(OUT / "network_stats.csv", index=False)

CS = S.groupby("state").mbps.agg(share="size", mean="mean", median="median",
                                 p10=lambda x: np.quantile(x, .1), p90=lambda x: np.quantile(x, .9))
CS["share"] /= CS["share"].sum()
CS.index = [REGIME_NAMES[i] for i in CS.index]
occ = S.groupby(["scenario", "state"]).size().unstack(fill_value=0)
occ = (occ.T / occ.sum(axis=1)).T.rename(columns=dict(enumerate(REGIME_NAMES)))
CS.round(3).to_csv(OUT / "channel_states.csv", index_label="state")
occ.round(3).to_csv(OUT / "channel_state_occupancy.csv")

F = []
for name, fam, tr in NETS:
    v = np.concatenate(tr)
    for sc in (1, 4, 8):
        br = np.array(R.ladder(sc)) * 8 / 4.0
        F.append(dict(network=name, ladder=f"x{sc}", **{f"tier{t}_{br[t]:g}Mbps": (v >= br[t]).mean() for t in range(5)}))
FT = pd.DataFrame(F)
FT.round(3).to_csv(OUT / "ladder_feasibility.csv", index=False)

pd.set_option("display.width", 220)
rep = ["== networks (throughput in Mbit/s) ==", T.round(2).to_string(index=False),
       "\n== channel states of the generator (all scenarios pooled) ==", CS.round(2).to_string(),
       "\n== channel-state occupancy per scenario ==", occ.round(3).to_string(),
       "\n== ladder feasibility: share of seconds with throughput >= tier bitrate ==",
       "\n".join(f"-- ladder {lad} --\n" + g.drop(columns="ladder").dropna(axis=1).round(3).to_string(index=False)
                 for lad, g in FT.groupby("ladder"))]
(OUT / "network_report.txt").write_text("\n".join(rep) + "\n")
print("\n".join(rep))
print(f"\nwritten to {OUT}")
