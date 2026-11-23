#!/usr/bin/env python3
"""Re-run the analysis as if DARA's act threshold were lower (a sensitivity view; the pre-registered threshold stays 0.2).

DARA acts when M >= threshold. In an environment with M >= the pre-registered threshold, DARA acts identically at the
lower threshold, so DARA-L's runs are reused. In an environment with  threshold <= M < pre-registered threshold, DARA-L
is replaced by the runs trained with the lower threshold (e.g. dara_online_lag_m015); if those runs do not exist the
environment is reported and left unchanged. Environments with M below the lower threshold are unchanged.

Writes (nothing in runs/ or results/ is modified):
  runs/views/t<thr>/train/        symbolic links to the run folders that make up the view
  configs/preregistration/view_t<thr>.json   copy of the pre-registration with the lower threshold, marked as a deviation
then runs evaluate_training.py and make_overview.py into results/views/t<thr>/ and overview/views/t<thr>/.

Usage:  python scripts/analysis/threshold_view.py --threshold 0.15
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--results", default=str(ROOT / "results"), help="for generalisation and data results (unchanged)")
ap.add_argument("--landscape", default=str(ROOT / "runs" / "landscape" / "landscape_content_u1_integrated.csv"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--view-root", default=str(ROOT))
a = ap.parse_args()

PR = json.load(open(a.prereg))
REF = PR["proposed_method"]
T0 = PR.get("act_threshold", {}).get("primary", 0.2)
thr = a.threshold
tag = f"t{thr:g}".replace(".", "")
alt = f"{REF}_m{round(thr * 100):03d}"                               # 0.15 -> dara_online_lag_m015
VR = Path(a.view_root)
RUNS_OUT = VR / "runs" / "views" / tag / "train"
RES_OUT = VR / "results" / "views" / tag
OV_OUT = VR / "overview" / "views" / tag
PR_OUT = VR / "configs" / "preregistration" / f"view_{tag}.json"

runs = {}
for d in sorted(Path(a.runs).glob("*__*_s*")):
    p = R.parse_run(d.name)
    if p:
        runs.setdefault((p["env"], p["method"]), []).append(d)


def env_M(env):
    ms = []
    for d in runs.get((env, REF), []):
        f = d / "dara_log.csv"
        if f.exists():
            L = pd.read_csv(f)
            if len(L):
                ms.append(float(L.M.iloc[0]))
    return sum(ms) / len(ms) if ms else None


if RUNS_OUT.exists():
    shutil.rmtree(RUNS_OUT)
RUNS_OUT.mkdir(parents=True)
log = []
for env in sorted({e for e, _ in runs}):
    M = env_M(env)
    swap = M is not None and thr <= M < T0
    if swap and (env, alt) not in runs:
        log.append(f"{env}: M = {M:.3f} is between {thr:g} and {T0:g}, but no {alt} runs exist: left unchanged (train them first)")
        swap = False
    elif swap:
        log.append(f"{env}: M = {M:.3f}: DARA-L replaced by {alt} ({len(runs[(env, alt)])} seeds)")
    else:
        log.append(f"{env}: M = {M if M is None else round(M, 3)}: unchanged (DARA acts the same at {thr:g} and {T0:g})")
    for (e, m), dirs in runs.items():
        if e != env:
            continue
        if m == alt:                      # either it became DARA-L, or it is not part of this view
            continue
        src_dirs = runs[(env, alt)] if (m == REF and swap) else dirs
        for d in src_dirs:
            seed = R.parse_run(d.name)["seed"]
            os.symlink(d.resolve(), RUNS_OUT / R.run_name(env, m, seed))
print("\n".join(log))

pr = json.loads(json.dumps(PR))
pr.setdefault("act_threshold", {})["primary"] = thr
pr["name"] = PR.get("name", "") + f" | VIEW: act threshold {thr:g} (post-hoc sensitivity analysis)"
pr["deviation"] = (f"DARA's act threshold is lowered from {T0:g} to {thr:g} after the results "
                   f"were seen. Environments with {thr:g} <= M < {T0:g} use the {alt} runs as DARA-L.")
pr["view_log"] = log
PR_OUT.parent.mkdir(parents=True, exist_ok=True)
PR_OUT.write_text(json.dumps(pr, indent=2))
print("written", PR_OUT)

py = sys.executable
RES_OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([py, str(ROOT / "scripts" / "analysis" / "evaluate_training.py"), "--runs", str(RUNS_OUT), "--prereg", str(PR_OUT),
                "--out", str(RES_OUT / "training"), "--landscape", a.landscape], check=True)
for sub in ("generalisation", "data"):                     # unchanged: the ct4 models and the networks do not depend on it
    src = Path(a.results) / sub
    dst = RES_OUT / sub
    if src.exists() and not dst.exists():
        os.symlink(src.resolve(), dst)
subprocess.run([py, str(ROOT / "scripts" / "analysis" / "make_overview.py"), "--results", str(RES_OUT), "--runs", str(RUNS_OUT),
                "--prereg", str(PR_OUT), "--out", str(OV_OUT)], check=True)
print(f"\nview written: {OV_OUT / 'index.html'}")
