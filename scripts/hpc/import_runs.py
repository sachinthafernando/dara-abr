#!/usr/bin/env python3
"""Bring finished training runs from an earlier output folder into runs/train/ under the current naming, so that
`make_sweeps.py --skip-done` leaves them out and only the missing runs are submitted.

Only runs whose method is in the registry and whose triage_summary.csv and checkpoint.pt exist are imported.
Old names p2_<env>__<method>_s<seed> become <env>__<method>_s<seed>; the hand-tuned 0.5 s Lagrangian
(p2_ct4__ppo_lagrangian_sX) becomes ct4__ppo_lagrangian_t0.5_sX. Symlinks by default (no copying); --copy copies.

Use it only for runs made with the same code and parameters (same notebook version, same environment); runs
trained on trace networks depend on the trace file and should be re-run after prepare_traces.py.

Usage:  python scripts/hpc/import_runs.py --from ../ph2_l/dara_phase2_content [--envs ct4] [--copy] [--dry-run]
"""
import argparse, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="src", required=True)
ap.add_argument("--envs", nargs="*", default=["ct4"])
ap.add_argument("--copy", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
dst_root = ROOT / "runs" / "train"; dst_root.mkdir(parents=True, exist_ok=True)
n = 0
for d in sorted(Path(a.src).iterdir()):
    info = R.parse_run(d.name)
    if not d.is_dir() or info is None or d.name.startswith("smoke_"):
        continue
    if info["env"] not in a.envs or info["method"] not in R.METHODS:
        continue
    if not ((d / "triage_summary.csv").exists() and (d / "checkpoint.pt").exists()):
        print(f"skip (unfinished): {d.name}"); continue
    if R.TRAINING_ENVS.get(info["env"], {}).get("net", {}).get("NET_SOURCE") == "trace":
        print(f"skip (trace-trained, re-run instead): {d.name}"); continue
    new = dst_root / R.run_name(**info)
    if new.exists():
        print(f"exists: {new.name}"); continue
    print(f"{d.name} -> runs/train/{new.name}")
    n += 1
    if a.dry_run:
        continue
    if a.copy:
        shutil.copytree(d, new)
    else:
        os.symlink(d.resolve(), new)
print(f"{'would import' if a.dry_run else 'imported'} {n} runs")
