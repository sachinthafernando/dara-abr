"""Loaders for the notebook outputs.

runs/train/<run>/gt_per_episode.csv   per test session: policy and reference-policy scores (training environment)
runs/train/<run>/triage_summary.csv   one row per horizon; is_main_H == 1 is the reported one
runs/generalize/gen_<run>.csv         per test session and condition (MODE='generalize')
runs/landscape/landscape_*.csv        diagnosis signals over reward settings (MODE='landscape')
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

from .registry import parse_run


def load_training(runs_dir):
    """Returns (sessions, refs, meta).
    sessions: env, method, seed, episode, J0, rebuffer          (trained policies, training environment)
    refs    : env, method, episode, J0, rebuffer                (always-one-tier and RobustMPC, same sessions)
    meta    : one row per run with the summary fields used in the paper"""
    S, R, M = [], [], []
    for f in sorted(glob.glob(str(Path(runs_dir) / "*" / "gt_per_episode.csv"))):
        d = Path(f).parent
        info = parse_run(d.name)
        if info is None:
            continue
        g = pd.read_csv(f)
        S.append(pd.DataFrame(dict(env=info["env"], method=info["method"], seed=info["seed"], episode=g.episode,
                                   J0=g.policy_ref, rebuffer=g.get("policy_rebuffer", np.nan))))
        for c in g.columns:
            if c.endswith("_ref") and c != "policy_ref":
                k = c[:-4]
                R.append(pd.DataFrame(dict(env=info["env"], method=k, seed=info["seed"], episode=g.episode, J0=g[c],
                                           rebuffer=g.get(f"{k}_rebuffer", np.nan))))
        ts = d / "triage_summary.csv"
        row = dict(run=d.name, **info)
        if ts.exists():
            s = pd.read_csv(ts)
            s = s[s.is_main_H == 1].iloc[0] if "is_main_H" in s else s.iloc[0]
            for k in ("policy_tier_share", "mpc_tier_share", "final_lagrange", "final_lag_target", "final_lambda",
                      "final_mu", "final_rn_cap", "dara_changes", "dara_first_change_ep", "M", "D", "triage", "algo"):
                row[k] = s.get(k, np.nan)
        M.append(row)
    sessions = pd.concat(S, ignore_index=True) if S else pd.DataFrame(columns=["env", "method", "seed", "episode", "J0", "rebuffer"])
    refs = pd.concat(R, ignore_index=True) if R else pd.DataFrame(columns=["env", "method", "seed", "episode", "J0", "rebuffer"])
    # reference policies are deterministic and identical across runs of one environment: keep one copy
    if len(refs):
        refs = refs.sort_values("seed").drop_duplicates(["env", "method", "episode"]).drop(columns="seed")
    meta = pd.DataFrame(M)
    for c in ("policy_tier_share", "mpc_tier_share"):
        if c in meta:
            meta[c] = meta[c].map(lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else None)
    return sessions, refs, meta


def load_generalisation(root):
    """Bare-policy and reference rows of every generalize/gen_*.csv under root (runtime-guard variants dropped)."""
    files = sorted(glob.glob(str(Path(root) / "generalize" / "gen_*.csv")))
    if not files:
        return pd.DataFrame()
    G = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    G["condition"] = G.network + "/" + G.content
    ref = G.run == "reference"
    G = G[ref | (G.method == "policy")].copy()
    info = G.loc[G.run != "reference", "run"].map(parse_run)
    G.loc[info.index, "env"] = info.map(lambda i: i["env"] if i else None)
    G.loc[info.index, "train_method"] = info.map(lambda i: i["method"] if i else None)
    G.loc[info.index, "seed"] = info.map(lambda i: i["seed"] if i else np.nan)
    G.loc[G.run == "reference", "train_method"] = G.loc[G.run == "reference", "method"]
    return G


def matrix(df, value="J0"):
    """(seeds, sessions) array from long rows with columns seed, episode."""
    if "seed" not in df or df.seed.isna().all():
        return df.sort_values("episode")[value].to_numpy()[None, :]
    W = df.pivot_table(index="seed", columns="episode", values=value)
    return W.dropna(axis=1).to_numpy()
