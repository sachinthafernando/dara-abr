#!/usr/bin/env python3
"""Step 2 of the data pipeline: raw datasets -> per-second throughput traces used by the notebook.

Writes
  data/processed/lumos5g.npz          {trace name: Mbit/s per second}; names start with Walking_ / Driving_
  data/processed/irish5g_download.npz {trace name: Mbit/s per second}; names start with Static_ / Driving_
  data/processed/MANIFEST.json        source files (with md5), cleaning rules, trace counts and lengths,
                                      throughput summaries and radio-status shares
  results/data/prepare_log_<dataset>.csv  what happened to each source trace

Cleaning (dataset_spec.json, trace_preparation): duplicate seconds averaged; gaps up to max_gap_s interpolated;
longer gaps split a trace; parts shorter than min_trace_s dropped. Values are not clipped or rescaled.

Usage:  python scripts/data/prepare_traces.py [--only lumos5g irish5g] [--compare-irish OLD.npz]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr.datasets import load_spec, read_lumos, read_irish, to_traces, md5  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*", default=["lumos5g", "irish5g"])
ap.add_argument("--compare-irish", default=None, help="an earlier irish5g_download.npz to compare against")
a = ap.parse_args()
SPEC = load_spec()
PREP = SPEC["trace_preparation"]
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "results" / "data"; LOG.mkdir(parents=True, exist_ok=True)
man_path = OUT / "MANIFEST.json"
MAN = json.load(open(man_path)) if man_path.exists() else {}

TARGETS = {"lumos5g": (read_lumos, "lumos5g.npz"), "irish5g": (read_irish, "irish5g_download.npz")}


def summarise(tr):
    v = np.concatenate(list(tr.values()))
    L = np.array([len(x) for x in tr.values()])
    by_prefix = {}
    for p in sorted({k.split("_")[0] for k in tr}):
        vv = np.concatenate([x for k, x in tr.items() if k.startswith(p + "_")])
        by_prefix[p] = dict(traces=int(sum(k.startswith(p + "_") for k in tr)), hours=round(len(vv) / 3600, 3),
                            mean_mbps=round(float(vv.mean()), 2), median_mbps=round(float(np.median(vv)), 2))
    return dict(traces=len(tr), hours=round(L.sum() / 3600, 3), length_s=dict(min=int(L.min()), median=int(np.median(L)), max=int(L.max())),
                mbps=dict(mean=round(float(v.mean()), 2), median=round(float(np.median(v)), 2),
                          p10=round(float(np.quantile(v, .1)), 2), p90=round(float(np.quantile(v, .9)), 2),
                          max=round(float(v.max()), 2), share_below_1=round(float((v < 1).mean()), 4)),
                by_prefix=by_prefix)


for name in a.only:
    reader, fname = TARGETS[name]
    s = SPEC[name]
    t0 = time.time()
    df, files = reader(s)
    tr, log = to_traces(df, PREP["max_gap_s"], PREP["min_trace_s"])
    if not tr:
        raise SystemExit(f"{name}: no trace survived cleaning")
    np.savez_compressed(OUT / fname, **tr)
    log.to_csv(LOG / f"prepare_log_{name}.csv", index=False)
    st = df.status.astype(str)
    MAN[name] = dict(output=f"data/processed/{fname}", created=time.strftime("%Y-%m-%d %H:%M"),
                     citation=s["citation"], source=s["source"],
                     files=[dict(path=str(Path(f).relative_to(ROOT)), md5=md5(f)) for f in files],
                     unit_conversion=f"{s['throughput_column']} x {s['throughput_to_mbps']} -> Mbit/s",
                     cleaning=PREP, status_column=s.get("status_column"),
                     status_share={k: round(float(v), 4) for k, v in st.value_counts(normalize=True).items()} if st.ne("").any() else {},
                     **summarise(tr))
    print(f"{name}: {len(tr)} traces, {MAN[name]['hours']} h -> {OUT / fname}  ({time.time() - t0:.1f}s)")
    print("  " + json.dumps(MAN[name]["by_prefix"]))

json.dump(MAN, open(man_path, "w"), indent=2)
print(f"manifest: {man_path}")

if a.compare_irish and "irish5g" in a.only:
    old = np.load(a.compare_irish); new = np.load(OUT / "irish5g_download.npz")
    rows = []
    for k in sorted(set(old.files) | set(new.files)):
        o = old[k] if k in old.files else None; n = new[k] if k in new.files else None
        rows.append(dict(trace=k, in_old=o is not None, in_new=n is not None,
                         len_old=len(o) if o is not None else np.nan, len_new=len(n) if n is not None else np.nan,
                         mean_old=float(o.mean()) if o is not None else np.nan, mean_new=float(n.mean()) if n is not None else np.nan,
                         identical=bool(o is not None and n is not None and len(o) == len(n) and np.allclose(o, n))))
    C = pd.DataFrame(rows)
    C.to_csv(LOG / "irish_compare.csv", index=False)
    print(f"\ncomparison with {a.compare_irish}: {int(C.identical.sum())}/{len(C)} traces identical; "
          f"only in old: {int((~C.in_new).sum())}, only in new: {int((~C.in_old).sum())}")
    print(C.round(2).to_string(index=False))
