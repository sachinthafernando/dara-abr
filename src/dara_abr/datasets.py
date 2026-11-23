"""Reading the raw throughput datasets into per-second traces (Mbit/s).

Both readers return a long DataFrame with columns
    trace   : trace name (prefix = mobility, so the notebook can select e.g. TRACE_PREFIX="Driving")
    t       : seconds (integer grid after cleaning)
    mbps    : throughput, Mbit/s
    status  : radio status column if present (Lumos5G nrStatus, Irish NetworkMode), else ""
and `to_traces` turns it into {name: np.ndarray} with short gaps interpolated and long gaps splitting a trace.
Column names, units and file locations come from scripts/data/dataset_spec.json.
"""
import glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

from . import ROOT


def load_spec(path=None):
    return json.load(open(path or ROOT / "scripts" / "data" / "dataset_spec.json"))


def find_files(spec_ds, root=ROOT):
    files = []
    for pat in spec_ds["files"]:
        files += glob.glob(str(Path(root) / pat), recursive=True)
    return sorted(set(files))


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_lumos(spec_ds, root=ROOT):
    files = find_files(spec_ds, root)
    if not files:
        raise FileNotFoundError(f"no Lumos5G files match {spec_ds['files']}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    miss = [c for c in spec_ds["required_columns"] if c not in df.columns]
    if miss:
        raise KeyError(f"Lumos5G: missing columns {miss}; found {list(df.columns)}")
    keys = spec_ds["trace_keys"]
    out = pd.DataFrame(dict(
        mbps=pd.to_numeric(df[spec_ds["throughput_column"]], errors="coerce") * float(spec_ds["throughput_to_mbps"]),
        t=pd.to_numeric(df[spec_ds["order_column"]], errors="coerce") * float(spec_ds["sample_period_s"]),
        status=df[spec_ds["status_column"]].astype(str) if spec_ds.get("status_column") in df else ""))
    parts = []
    for k in keys:
        v = df[k].astype(str).str.strip()
        parts.append(v.str.capitalize() if k == keys[0] else v)
    out["trace"] = parts[0]
    for k, v in zip(keys[1:], parts[1:]):
        out["trace"] = out["trace"] + "_" + (("run" + v) if k == "run_num" else v)
    return out, files


def read_irish(spec_ds, root=ROOT):
    files = find_files(spec_ds, root)
    if not files:
        raise FileNotFoundError(f"no Irish 5G files match {spec_ds['files']}")
    rows = []
    for f in files:
        p = Path(f)
        mob = p.parent.name.capitalize() if spec_ds.get("mobility_from_folder") else "All"
        d = pd.read_csv(f, low_memory=False)
        miss = [c for c in spec_ds["required_columns"] if c not in d.columns]
        if miss:
            raise KeyError(f"{f}: missing columns {miss}; found {list(d.columns)}")
        ts = pd.to_datetime(d[spec_ds["timestamp_column"]].astype(str).str.strip(), format=spec_ds["timestamp_format"],
                            errors="coerce")
        t = (ts - ts.min()).dt.total_seconds()
        rows.append(pd.DataFrame(dict(
            trace=f"{mob}_{p.stem}", t=t,
            mbps=pd.to_numeric(d[spec_ds["throughput_column"]], errors="coerce") * float(spec_ds["throughput_to_mbps"]),
            status=d[spec_ds["status_column"]].astype(str) if spec_ds.get("status_column") in d else "")))
    return pd.concat(rows, ignore_index=True), files


def to_traces(df, max_gap_s=3, min_trace_s=120, period=1.0):
    """Regular 1-s traces. Duplicate seconds are averaged; gaps up to max_gap_s are interpolated linearly;
    longer gaps split the trace into parts (_p0, _p1, ...; no suffix when a trace stays whole); parts shorter than
    min_trace_s are dropped. Returns (traces, log) where log lists what was done per source trace."""
    traces, log = {}, []
    for name, g0 in df.dropna(subset=["t"]).groupby("trace", sort=True):
        g = g0.assign(sec=np.round(g0.t / period).astype(int)).groupby("sec").mbps.mean()
        n_dup = int(len(g0) - len(g))
        full = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        isna = full.isna().to_numpy()
        # run lengths of missing values
        segs, start = [], 0
        i = 0
        while i < len(full):
            if isna[i]:
                j = i
                while j < len(full) and isna[j]:
                    j += 1
                if j - i > max_gap_s:
                    segs.append((start, i)); start = j
                i = j
            else:
                i += 1
        segs.append((start, len(full)))
        kept = []
        for a, b in segs:
            s = full.iloc[a:b].interpolate(limit_direction="both")
            if len(s) >= min_trace_s and s.notna().all():
                kept.append(s.to_numpy(float))
        for k, arr in enumerate(kept):
            traces[name if len(kept) == 1 else f"{name}_p{k}"] = np.maximum(arr, 0.0)
        log.append(dict(trace=name, seconds_raw=int(len(full)), duplicates=n_dup, missing=int(isna.sum()),
                        parts_kept=len(kept), seconds_kept=int(sum(len(x) for x in kept))))
    return traces, pd.DataFrame(log)
