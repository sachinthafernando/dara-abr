# Data

Third-party datasets are **not** included. Download them from their original sources, place them as below, then run
the data pipeline (`scripts/data/`). Column names, units and file locations the scripts expect are in
`scripts/data/dataset_spec.json`; edit that file if your copy differs.

| Dataset | Used for | Licence | Put it here | Source |
|---|---|---|---|---|
| Lumos5G v1.0 (Narayanan et al., IMC 2020) | real mmWave training/test network; calibration source of the network generator | CC BY 4.0 | `data/raw/lumos5g/Lumos5G-v1.0.csv` | [IEEE DataPort](https://ieee-dataport.org/node/3598) (free IEEE login), [project page](https://lumos5g.umn.edu/) |
| Irish 5G production dataset (Raca et al., MMSys 2020) | real commercial 5G networks (static, driving); training environments `cts1`, `ctd1` | GPL-3.0 | `data/raw/irish5g/` (unzip `5G-production-dataset.zip` so that `.../Download/Static/*.csv` and `.../Download/Driving/*.csv` exist) | [github.com/uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) |
| Puffer `video_size` (Yan et al., NSDI 2020) | only to re-calibrate the chunk-size generator | see source | `data/raw/puffer/video_size_*.csv` | [puffer.stanford.edu/results](https://puffer.stanford.edu/results/) |

The md5 of the exact files used for the paper and thesis are in `processed/MANIFEST.json`; `prepare_traces.py`
rewrites that file, so `git diff data/processed/MANIFEST.json` shows whether your copy is identical.
The generator environments (`ct1`, `ct2`, `ct4`, `ct8`, `cth4`) need no dataset.

## Expected columns

**Lumos5G** (one row per second): `run_num, seq_num, abstractSignalStr, latitude, longitude, movingSpeed,
compassDirection, nrStatus, lte_rssi, lte_rsrp, lte_rsrq, lte_rssnr, nr_ssRsrp, nr_ssRsrq, nr_ssSinr, Throughput,
mobility_mode, trajectory_direction, tower_id`. Required: `run_num, seq_num, Throughput, mobility_mode,
trajectory_direction`. `Throughput` is taken as Mbit/s (iPerf downlink). A trace is one
`mobility_mode` + `trajectory_direction` + `run_num`, ordered by `seq_num`.

**Irish 5G** (one CSV per session, about one row per second): `Timestamp` (format `2019.12.14_10.16.30`),
`DL_bitrate` in kbit/s, plus radio fields (`NetworkMode`, `RSRP`, `SNR`, ...). Required: `Timestamp, DL_bitrate`.
Mobility (`Static` / `Driving`) is taken from the folder name.

**Puffer**: five columns read by position: time (ns), format, channel, video_ts, size.

## Pipeline

```bash
python scripts/data/inspect_datasets.py     # 1. check files and columns; exit code 2 = something to fix
python scripts/data/prepare_traces.py       # 2. -> data/processed/{lumos5g,irish5g_download}.npz + MANIFEST.json
python scripts/data/analyse_traces.py       # 3. network statistics -> results/data/
```

Cleaning (step 2): duplicate seconds are averaged, gaps of up to 3 s are interpolated, longer gaps split a trace,
and parts shorter than 120 s are dropped. Values are not clipped or rescaled. `MANIFEST.json` records the md5 of
every source file, so a result can be traced to the exact data it used.

## Included derived files

* `processed/content_params.json`: parameters of the content-complexity chunk-size generator, calibrated on two days of
  Puffer `video_size` data (fit day 2026-10-05, held-out day 2026-10-06; channels abc, cbs, fox, kpyx, kron, nbc, pbs;
  reference format 1280x720-26). `scripts/data/calibrate_content.py` reproduces the regime, dwell-time and title
  parameters; the tier exponents and the slow component were derived in a second step that is not in this release; the fitted
  values are in the file, so the Puffer data is not needed to reproduce the results.
* The mmWave network generator's constants are in `src/dara_abr/urban_generator.py` (calibrated on Lumos5G).
