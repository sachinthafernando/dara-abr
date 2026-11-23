#!/usr/bin/env python3
"""Synthetic stand-ins for the raw datasets, with the column layout dataset_spec.json expects.

For testing the pipeline only (inspect -> prepare -> analyse -> notebook smoke run). The values are random and
must never be used for results. Writes into <dest>/data/raw/ (default: a scratch copy, never the real data folder).

Usage:  python tests/make_fixtures.py --dest /tmp/dara_test_repo
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dest", required=True)
a = ap.parse_args()
raw = Path(a.dest) / "data" / "raw"
rng = np.random.default_rng(0)

# Lumos5G-like: one CSV, per-second rows, runs per mobility/direction
rows = []
for mob, dirs in (("walking", ("CW", "ACW")), ("driving", ("CW",))):
    for d in dirs:
        for run in range(3):
            n = int(rng.integers(150, 400))
            tp = np.exp(rng.normal(5.5, 1.2, n)).clip(0, 1900)
            seq = np.arange(n)
            if run == 1:
                seq = np.delete(seq, range(60, 70))           # a 10-s gap -> trace split
            for i, s in enumerate(seq):
                rows.append(dict(run_num=run + (0 if mob == "walking" else 10), seq_num=int(s), abstractSignalStr=2,
                                 latitude=44.97, longitude=-93.26, movingSpeed=1.2, compassDirection=90,
                                 nrStatus="CONNECTED" if tp[i] > 50 else "NOT_RESTRICTED", lte_rssi=-60, lte_rsrp=-90,
                                 lte_rsrq=-10, lte_rssnr=10, nr_ssRsrp=-90, nr_ssRsrq=-11, nr_ssSinr=10,
                                 Throughput=round(float(tp[i]), 2), mobility_mode=mob, trajectory_direction=d, tower_id=17))
p = raw / "lumos5g"; p.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(p / "Lumos5G-v1.0.csv", index=False)

# Irish-5G-like: one CSV per trace under 5G-production-dataset/Download/{Static,Driving}
for mob, k in (("Static", 2), ("Driving", 3)):
    q = raw / "irish5g" / "5G-production-dataset" / "Download" / mob; q.mkdir(parents=True, exist_ok=True)
    for i in range(k):
        n = int(rng.integers(300, 700))
        t0 = pd.Timestamp("2019-12-14 10:16:30") + pd.Timedelta(days=i)
        ts = [t0 + pd.Timedelta(seconds=int(s)) for s in range(n)]
        dl = (np.exp(rng.normal(9.5, 1.5, n))).round()                       # kbit/s
        d = pd.DataFrame(dict(Timestamp=[t.strftime("%Y.%m.%d_%H.%M.%S") for t in ts], Longitude=-8.47, Latitude=51.89,
                              Speed=0 if mob == "Static" else 40, Operatorname="B", CellID=1,
                              NetworkMode=np.where(rng.random(n) < 0.6, "5G", "LTE"), RSRP=-90, RSRQ=-10, SNR=10, CQI=10,
                              RSSI=-60, DL_bitrate=dl, UL_bitrate=100, State="D", NRxRSRP=-90, NRxRSRQ=-10,
                              ServingCell_Lon=-8.47, ServingCell_Lat=51.89, ServingCell_Distance=500))
        d = pd.concat([d, d.iloc[[5]]]).sort_index()                         # one duplicated second
        d.to_csv(q / f"B_{t0.strftime('%Y.%m.%d_%H.%M.%S')}.csv", index=False)
print(f"synthetic fixtures written under {raw}")
