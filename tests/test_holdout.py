#!/usr/bin/env python3
"""Checks the held-out trace split of the notebook's environment, without training.

For a trace network with TRACE_HOLDOUT_FRAC > 0: the held-out set is deterministic, non-empty, disjoint from the
training traces, and only split='test' environments ever sample it; every other split never does.
Usage: python tests/test_holdout.py [--file data/processed/irish5g_download.npz --prefix Driving --frac 0.25]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--file", default="data/processed/irish5g_download.npz")
ap.add_argument("--prefix", default="Driving")
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--sessions", type=int, default=200)
a = ap.parse_args()

import os
os.chdir(ROOT)
nb = json.load(open(ROOT / "notebooks" / "dara_abr.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
G = globals()
exec(cells[1], G)
exec(cells[2], G)
G.update(MODE="landscape", OUT_ROOT="/tmp/holdout_test", LADDER="content", UNITS_FIX=1, DL_MODEL="integrated",
         NET_SOURCE="trace", TRACE_FILE=a.file, TRACE_PREFIX=a.prefix, TRACE_HOLDOUT_FRAC=a.frac,
         BASE_CHUNK_MB=[1.5, 3.0, 6.0, 12.0, 22.0])
exec(cells[3], G)
exec(cells[5], G)
keys = sorted(k for k in load_traces(a.file) if k.startswith(a.prefix))
hold = holdout_traces(keys, C)
assert hold and len(hold) < len(keys), (hold, keys)
assert hold == holdout_traces(keys, C), "held-out set must be deterministic"
seen = {}
for split, seed in (("train", 101), ("eval", 99), ("eval", 2026), ("test", 99)):
    env = ABREnv(C, None, split, seed=seed)
    used = {env.scenario}
    for _ in range(a.sessions - 1):
        env.reset(); used.add(env.scenario)
    seen[(split, seed)] = used
    bad = used & set(hold) if split != "test" else used - set(hold)
    assert not bad, f"split {split}: sampled {sorted(bad)}"
print(f"traces with prefix {a.prefix!r}: {len(keys)}; held out ({len(hold)}): {hold}")
for k, v in seen.items():
    print(f"  {k[0]:5s} seed {k[1]:5d}: {len(v)} distinct traces used, all {'held-out' if k[0] == 'test' else 'training'}")
print("OK")
