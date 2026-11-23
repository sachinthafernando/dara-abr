#!/usr/bin/env python3
"""Which jobs of a table have finished? Lists missing outputs and the end of each failed job's executed notebook.
Usage: python scripts/hpc/check_jobs.py configs/sweeps/main.tsv"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
missing = []
for i, r in enumerate(rows, 1):
    out = ROOT / r["OUT_ROOT"]
    ok = {"run": (out / r["RUN_NAME"] / "triage_summary.csv").exists(),
          "generalize": (out / "generalize" / f"gen_{r['RUN_NAME'][4:]}.csv").exists(),
          "landscape": (out / "landscape" / "landscape_content_u1_integrated.csv").exists()}.get(r["MODE"], False)
    if not ok:
        missing.append((i, r["RUN_NAME"]))
print(f"{path}: {len(rows) - len(missing)}/{len(rows)} finished")
for i, name in missing:
    ex = ROOT / "logs" / "executed" / f"{name}.ipynb"
    err = ""
    if ex.exists():
        nb = json.load(open(ex))
        for c in nb["cells"]:
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    err = f"{o.get('ename')}: {o.get('evalue')}"[:200]
    print(f"  job {i:3d} {name:45s} {'not started' if not ex.exists() else (err or 'running or killed')}")
if missing:
    print("re-run only these:  --array=" + ",".join(str(i) for i, _ in missing))
