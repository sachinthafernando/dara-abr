#!/usr/bin/env python3
"""Prints the papermill arguments for one row of a job table, NUL-separated (for bash `mapfile -d ''`).

Row numbers start at 1 (the first job). Empty cells are skipped (notebook default); values starting with "[" are
passed raw (-r, parsed as YAML lists), everything else as -p. The first item printed is the run name.
Usage: python scripts/hpc/row_args.py configs/sweeps/main.tsv 3
"""
import csv, sys

path, i = sys.argv[1], int(sys.argv[2])
with open(path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if not 1 <= i <= len(rows):
    sys.exit(f"{path} has {len(rows)} jobs; asked for job {i}")
row = rows[i - 1]
out = [row["RUN_NAME"]]
for k, v in row.items():
    v = (v or "").strip()
    if not v:
        continue
    out += (["-r", k, v] if v.startswith("[") else ["-p", k, v])
sys.stdout.write("\0".join(out) + "\0")
