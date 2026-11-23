#!/usr/bin/env python3
"""Step 1 of the data pipeline: check that the raw datasets are where and what the pipeline expects.

For each dataset in scripts/data/dataset_spec.json it reports: files found, columns (and any required column that
is missing, with the closest names found), row counts, missing values, throughput range in Mbit/s, sampling
interval, trace count and lengths, and the radio-status values. Nothing is written except the report.

Exit code 0 = every dataset present has the required columns; 2 = a required column is missing (fix
dataset_spec.json or the files before running prepare_traces.py). Missing datasets are reported, not errors.

Usage:  python scripts/data/inspect_datasets.py [--only lumos5g irish5g puffer] [--out results/data/inspection.txt]
"""
import argparse, difflib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr.datasets import load_spec, find_files, read_lumos, read_irish, to_traces  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", default=["lumos5g", "irish5g", "puffer"])
ap.add_argument("--out", default=str(ROOT / "results" / "data" / "inspection.txt"))
a = ap.parse_args()
SPEC = load_spec()
PREP = SPEC["trace_preparation"]
lines, status = [], 0


def say(s=""):
    print(s); lines.append(str(s))


def check_columns(name, cols, required, optional=()):
    global status
    miss = [c for c in required if c not in cols]
    say(f"  columns ({len(cols)}): {list(cols)}")
    for c in miss:
        close = difflib.get_close_matches(c, list(cols), n=3, cutoff=0.4)
        say(f"  MISSING required column {c!r}; closest found: {close}")
        status = 2
    absent_opt = [c for c in optional if c not in cols]
    if absent_opt:
        say(f"  optional columns not present (fine): {absent_opt}")
    if not miss:
        say("  required columns: OK")
    return not miss


def throughput_summary(mbps, plausible_max):
    v = pd.Series(mbps).astype(float)
    say(f"  throughput (Mbit/s): n={v.notna().sum():,}  missing={v.isna().mean():.1%}  min={v.min():.3g}  "
        f"p10={v.quantile(.1):.3g}  median={v.median():.3g}  mean={v.mean():.3g}  p90={v.quantile(.9):.3g}  max={v.max():.4g}")
    say(f"  share exactly 0: {(v == 0).mean():.1%}   share below 1 Mbit/s: {(v < 1).mean():.1%}   "
        f"share above plausible max {plausible_max:g}: {(v > plausible_max).mean():.2%}")
    if v.max() > plausible_max:
        say("  WARNING: values above the plausible maximum - check the unit (throughput_to_mbps in dataset_spec.json)")
    if v.median() < 0.05:
        say("  WARNING: median below 0.05 Mbit/s - the unit conversion may be wrong")


def trace_summary(df):
    tr, log = to_traces(df, PREP["max_gap_s"], PREP["min_trace_s"])
    L = np.array([len(x) for x in tr.values()]) if tr else np.array([0])
    say(f"  source traces: {df.trace.nunique()}   duplicate seconds averaged: {int(log.duplicates.sum()):,}   "
        f"missing seconds: {int(log.missing.sum()):,}")
    say(f"  after cleaning (gaps <= {PREP['max_gap_s']} s interpolated, parts < {PREP['min_trace_s']} s dropped): "
        f"{len(tr)} traces, {L.sum() / 3600:.2f} h; length min/median/max = {L.min()}/{int(np.median(L))}/{L.max()} s")
    prefixes = pd.Series([k.split("_")[0] for k in tr]).value_counts().to_dict()
    say(f"  traces per prefix (usable as TRACE_PREFIX): {prefixes}")
    dropped = log[log.parts_kept == 0]
    if len(dropped):
        say(f"  source traces dropped entirely (too short): {len(dropped)}")


# ------------------------------------------------------------------------------------------------ Lumos5G
if "lumos5g" in a.only:
    s = SPEC["lumos5g"]
    say("=" * 100); say(f"Lumos5G   expected: {s['files']}")
    files = find_files(s)
    if not files:
        say("  not found (place Lumos5G-v1.0.csv in data/raw/lumos5g/)")
    else:
        say(f"  files: {len(files)}: {[Path(f).name for f in files][:5]}")
        head = pd.read_csv(files[0], nrows=5)
        if check_columns("lumos5g", head.columns, s["required_columns"], s.get("optional_columns", ())):
            df, _ = read_lumos(s)
            say(f"  rows: {len(df):,}")
            raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files])
            for k in s["trace_keys"][:2]:
                say(f"  {k} values: {raw[k].value_counts().to_dict()}")
            if s.get("status_column") in raw:
                say(f"  {s['status_column']} values: {raw[s['status_column']].value_counts().to_dict()}")
            steps = raw.sort_values(s["trace_keys"] + [s["order_column"]]).groupby(s["trace_keys"])[s["order_column"]].diff().dropna()
            say(f"  {s['order_column']} step within a trace: " + str(steps.value_counts().head(6).to_dict()))
            throughput_summary(df.mbps, s["plausible_max_mbps"])
            trace_summary(df)
            say("  first rows:"); say(head.head(3).to_string())

# ------------------------------------------------------------------------------------------------ Irish 5G
if "irish5g" in a.only:
    s = SPEC["irish5g"]
    say("=" * 100); say(f"Irish 5G   expected: {s['files']}")
    files = find_files(s)
    if not files:
        say("  not found (unzip 5G-production-dataset.zip into data/raw/irish5g/)")
    else:
        mob = pd.Series([Path(f).parent.name for f in files]).value_counts().to_dict()
        say(f"  files: {len(files)} {mob}: {[Path(f).name for f in files][:4]} ...")
        heads = {f: list(pd.read_csv(f, nrows=2).columns) for f in files}
        ok = check_columns("irish5g", heads[files[0]], s["required_columns"], s.get("optional_columns", ()))
        bad = [Path(f).name for f, c in heads.items() if any(r not in c for r in s["required_columns"])]
        if bad:
            say(f"  files missing a required column: {bad}"); status = 2; ok = False
        if ok:
            df, _ = read_irish(s)
            say(f"  rows: {len(df):,}   unparsed timestamps: {df.t.isna().mean():.2%} "
                f"(format {s['timestamp_format']!r}; example {pd.read_csv(files[0], nrows=1)[s['timestamp_column']].iloc[0]!r})")
            if df.t.isna().mean() > 0.01:
                say("  WARNING: many timestamps did not parse - check timestamp_format"); status = 2
            steps = df.sort_values(["trace", "t"]).groupby("trace").t.diff().dropna()
            say("  time step between rows (s): " + str(steps.value_counts().head(6).to_dict()))
            if s.get("status_column"):
                say(f"  {s['status_column']} values: {df.status.value_counts().to_dict()}")
            throughput_summary(df.mbps, s["plausible_max_mbps"])
            trace_summary(df)
            say("  first rows:"); say(pd.read_csv(files[0], nrows=3).to_string())

# ------------------------------------------------------------------------------------------------ Puffer
if "puffer" in a.only:
    s = SPEC["puffer"]
    say("=" * 100); say(f"Puffer   expected: {s['files']}   ({s['only_for']})")
    files = find_files(s)
    if not files:
        say("  not found (optional)")
    else:
        say(f"  files: {len(files)}: {[Path(f).name for f in files][:4]}")
        h = pd.read_csv(files[0], nrows=200000)
        say(f"  header: {list(h.columns)}  (expected {s['n_columns']} columns: {s['expected_header']})")
        if h.shape[1] != s["n_columns"]:
            say("  WARNING: unexpected number of columns"); status = 2
        else:
            say(f"  formats in first 200k rows: {h.iloc[:, 1].nunique()}  channels: {sorted(h.iloc[:, 2].astype(str).unique())}")
            say(f"  chunk size (bytes): median {h.iloc[:, 4].median():,.0f}")

say("=" * 100)
say("RESULT: " + ("OK - run scripts/data/prepare_traces.py next" if status == 0 else
                  "a required column or format is wrong - see the lines marked MISSING / WARNING above"))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(lines) + "\n")
print(f"report written to {a.out}")
sys.exit(status)
