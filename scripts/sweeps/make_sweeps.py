#!/usr/bin/env python3
"""Writes the job tables (one row = one notebook execution) for every sweep group into configs/sweeps/.

  smoke_train.tsv       one tiny run (SMOKE=1) of every code path: plain PPO, DARA, DARA-L, PPO-Lagrangian,
                        Pensieve, COREL, a trace-trained environment; plus a small landscape job
  smoke_generalise.tsv  the generalisation path on two smoke checkpoints and the reference job (run after smoke_train)
  landscape.tsv         diagnosis signals over freeze-penalty cap x chunk-size ladder, no training (one job)
  main.tsv              the primary training environment (ct4): original reward, cap sweep, DARA, DARA-L,
                        hand-tuned PPO-Lagrangian budgets, Pensieve, COREL; all seeds          -> every figure
  other_envs.tsv        DARA-L and hand-tuned budgets on ct2, cth4, cts1                      -> budget-rule table
  ablations.tsv         PPO trained on the score, DARA-L pre-flight                            -> text / supplement
  generalise.tsv        reference job + GEN_METHODS on the primary environment, all seeds  (after main.tsv)
  generalise_budgets.tsv  exploratory: hand-picked budgets 0.25 / 1 / 2 s on the test networks
  generalise_ct1.tsv    exploratory: the ladder-x1 models on 5 networks x ladders x1, x4 (out: runs/generalise_ct1)
  second batch (configs/preregistration/paper_v2.json):
  smoke_v2.tsv          new code paths in minutes (held-out traces, threshold 0.15, ladder x8, COREL on traces)
  precheck.tsv          M before training on Irish driving and Lumos5G (landscape, no training)
  set1.tsv              11 methods x 5 seeds on ladders x1, x2, x4, x8 (+ DARA-L threshold 0.15 on x1)
  set2.tsv              DARA-L, PPO-Lagrangian 0.5 s, COREL, Pensieve x 5 seeds on cth4, ctd1, ctl
  (use --skip-done: runs that already exist in runs/train are left out)

Columns are notebook parameters; an empty cell means "notebook default" (the runner skips it).
Usage:  python scripts/sweeps/make_sweeps.py [--seeds 101 202 303 404 505] [--skip-done] [--with-ablations]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="*", type=int, default=R.SEEDS)
ap.add_argument("--skip-done", action="store_true", help="leave out runs whose output already exists")
ap.add_argument("--with-ablations", action="store_true", help="also generalise the ablation methods")
a = ap.parse_args()
OUT = ROOT / "configs" / "sweeps"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN, TOP = "runs/train", "runs"
SM_TRAIN, SM_TOP = "runs/smoke/train", "runs/smoke"
lst = lambda v: json.dumps([float(x) for x in v])

COLS = ["RUN_NAME", "OUT_ROOT", "MODE", "SEED", "SMOKE", "REGIME", "EXPECTED_LABEL", "LADDER", "UNITS_FIX", "DL_MODEL",
        "QUALITY", "QUALITY_K", "BASE_CHUNK_MB", "RN_CAP", "CLIP_LO", "DARA_MODE", "ENT_ADAPT", "LAGRANGE", "LAG_TARGET",
        "ALGO", "REWARD_MODE", "NET_SOURCE", "NET_SCENARIOS", "NET_SCALE", "TRACE_FILE", "TRACE_PREFIX", "TRACE_HOLDOUT_FRAC", "M_HIGH", "GT_EPISODES",
        "GEN_CKPT", "GEN_NETWORKS", "GEN_CONTENT", "GEN_EPISODES", "RT_MODES",
        "LS_RN_CAP", "LS_QUALITY_K", "LS_QTOP", "LS_LADDER_SCALE", "LS_REBUF_SCALE"]
COMMON = dict(LADDER="content", UNITS_FIX=1, DL_MODEL="integrated", QUALITY=lst(R.QUALITY), QUALITY_K=0.0, CLIP_LO=-60.0)


def train_row(env, method, seed, out=TRAIN, smoke=False):
    e, m = R.TRAINING_ENVS[env], R.METHODS[method]
    name = R.run_name(env, method, seed)
    r = dict(COMMON, RUN_NAME=("smoke_" if smoke else "") + name, OUT_ROOT=out, MODE="run", SEED=seed,
             SMOKE=1 if smoke else "", REGIME=f"{env}__{method}", EXPECTED_LABEL="PROBE",
             BASE_CHUNK_MB=lst(R.ladder(e["ladder_scale"])), RN_CAP=0.1, DARA_MODE="none", ENT_ADAPT=0, LAGRANGE=0,
             LAG_TARGET=0.5, ALGO="dara", REWARD_MODE="surrogate", NET_SOURCE="generator", NET_SCENARIOS="[]",
             NET_SCALE=1.0, GT_EPISODES=R.EVAL_SESSIONS)
    for k, v in e["net"].items():
        r[k] = json.dumps(v) if isinstance(v, list) else v
    r.update(m["params"])
    return r


def gen_row(run, out=TOP, ckpt_root=TRAIN, smoke=False, networks=None, contents=None):
    return dict(COMMON, RUN_NAME=f"gen_{run}", OUT_ROOT=out, MODE="generalize", SMOKE=1 if smoke else "",
                GEN_CKPT=f"{ckpt_root}/{run}" if run != "reference" else "none",
                BASE_CHUNK_MB=lst(R.ladder(R.TRAINING_ENVS[R.PRIMARY_ENV]["ladder_scale"])),
                GEN_NETWORKS=json.dumps(networks or R.GEN_NETWORKS), GEN_CONTENT=json.dumps(contents or R.GEN_CONTENT),
                GEN_EPISODES=3 if smoke else R.EVAL_SESSIONS, RT_MODES='["none"]')


def landscape_row(out=TOP, smoke=False, env=None):
    r = dict(COMMON, RUN_NAME="landscape_content" + (f"_{env}" if env else "") + ("_smoke" if smoke else ""), OUT_ROOT=out,
             MODE="landscape", SEED=101, SMOKE=1 if smoke else "", BASE_CHUNK_MB=lst(R.BASE_CHUNK_MB), RN_CAP=0.1,
             GT_EPISODES=5 if smoke else R.EVAL_SESSIONS,
             LS_RN_CAP=lst([0.1, 1e6] if (smoke or env) else R.CAPS), LS_QUALITY_K="[0.0]", LS_QTOP="[4.0]",
             LS_LADDER_SCALE="[4.0]" if smoke else "[1.0, 2.0, 4.0, 8.0]", LS_REBUF_SCALE="[1.0]")
    if env:                                       # pre-check of one training environment: its network and ladder
        e = R.TRAINING_ENVS[env]
        r["LS_LADDER_SCALE"] = lst([e["ladder_scale"]])
        for k, v in e["net"].items():
            r[k] = json.dumps(v) if isinstance(v, list) else v
    return r


def done(r):
    if r["MODE"] == "run":
        return (ROOT / r["OUT_ROOT"] / r["RUN_NAME"] / "triage_summary.csv").exists()
    if r["MODE"] == "generalize":
        return (ROOT / r["OUT_ROOT"] / "generalize" / f"gen_{r['RUN_NAME'][4:]}.csv").exists()
    return False


def write(name, rows, note):
    n0 = len(rows)
    if a.skip_done:
        rows = [r for r in rows if not done(r)]
    p = OUT / f"{name}.tsv"
    with open(p, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            extra = set(r) - set(COLS)
            assert not extra, f"columns missing from COLS: {extra}"
            f.write("\t".join(str(r.get(c, "")) for c in COLS) + "\n")
    skipped = f" ({n0 - len(rows)} already done, left out)" if a.skip_done and n0 != len(rows) else ""
    print(f"{str(p.relative_to(ROOT)):34s} {len(rows):4d} jobs{skipped}   {note}")
    return len(rows)


def expand(group):
    g = R.GROUPS[group]
    rows = []
    for env in g["envs"]:
        methods = []
        for m in g["methods"]:
            methods += [f"ppo_lagrangian_t{b:g}" for b in R.HAND_BUDGETS[env]] if m == "hand_budgets" else [m]
        methods += g.get("extra", {}).get(env, [])
        for s in a.seeds:
            rows += [train_row(env, m, s) for m in methods]
    return rows


print(f"seeds: {a.seeds}")
smoke = [train_row("ct4", m, 101, out=SM_TRAIN, smoke=True)
         for m in ("fixed", "dara_online", "dara_online_lag", "ppo_lagrangian_t0.5", "pensieve_official", "corel_official")]
smoke.append(train_row("cts1", "dara_online_lag", 101, out=SM_TRAIN, smoke=True))
smoke.append(landscape_row(out=SM_TOP, smoke=True))
write("smoke_train", smoke, "~minutes each; check every job finishes before anything else")
write("smoke_generalise", [gen_row("reference", out=SM_TOP, smoke=True)] +
      [gen_row("smoke_" + R.run_name("ct4", m, 101), out=SM_TOP, ckpt_root=SM_TRAIN, smoke=True)
       for m in ("dara_online_lag", "corel_official")], "after smoke_train")
write("landscape", [landscape_row()], "no training; feeds the cap-effect figure")
write("main", expand("main"), f"first batch: primary environment {R.PRIMARY_ENV}")
write("other_envs", expand("other_envs"), "first batch: budget-rule table")
write("ablations", expand("ablations"), "first batch: optional ablations")
gm = R.GEN_METHODS + (R.GEN_METHODS_ABLATION if a.with_ablations else [])
write("generalise", [gen_row("reference")] + [gen_row(R.run_name(R.PRIMARY_ENV, m, s)) for s in a.seeds for m in gm],
      "first batch: after main.tsv (set 3 of the second batch)")
write("generalise_budgets", [gen_row(R.run_name(R.PRIMARY_ENV, m, s)) for s in a.seeds for m in R.GEN_METHODS_BUDGETS],
      "exploratory: other hand-picked budgets on the test networks")
# ---- second batch (paper_v2.json)
sm2 = [train_row("ct1", "dara_online_lag_m015", 101, out=SM_TRAIN, smoke=True),
       train_row("ct8", "dara_online_lag", 101, out=SM_TRAIN, smoke=True),
       train_row("ctd1", "dara_online_lag", 101, out=SM_TRAIN, smoke=True),
       train_row("ctl", "corel_official", 101, out=SM_TRAIN, smoke=True),
       landscape_row(out=SM_TOP + "/precheck_ctd1", smoke=True, env="ctd1")]
write("smoke_v2", sm2, "second batch: new code paths (held-out traces, threshold 0.15, ladder x8)")
write("precheck", [landscape_row(out=f"runs/precheck/{e}", env=e) for e in ("ctd1", "ctl")],
      "second batch: M before training on Irish driving and Lumos5G (no training)")
write("set1", expand("set1"), "second batch: 11 methods on ladders x1, x2, x4, x8 (+ threshold 0.15 on x1)")
write("set2", expand("set2"), "second batch: 4 methods on harsh mmWave, Irish driving, Lumos5G")
# ---- exploratory (20 Oct, after the results): models trained on ladder x1 tested on other networks and ladders
G1_NETS, G1_CONT = ["seen", "lumos5g", "irish_static", "irish_driving", "gen_harsh"], ["x1", "x4"]
G1_METHODS = ["dara_online_lag", "dara_online_lag_m015", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]
write("generalise_ct1", [gen_row("reference", out="runs/generalise_ct1", networks=G1_NETS, contents=G1_CONT)] +
      [gen_row(R.run_name("ct1", m, s_), out="runs/generalise_ct1", networks=G1_NETS, contents=G1_CONT)
       for s_ in a.seeds for m in G1_METHODS],
      "exploratory: ladder-x1 models on Lumos5G, Irish 5G, harsh mmWave and the generator, ladders x1 and x4")
# ---- exploratory (20 Oct): ladder-x1 models per generator scenario (like COREL's per-scenario figure), ladder x1
SC_NETS = ["scn_" + s for s in ("urban_pedestrian", "crowded_stationary", "urban_driving", "motorway", "suburban_open",
                                "tunnel", "under_bridge", "urban_canyon")] + ["grp_walking", "grp_driving", "grp_deadzone"]
SC_METHODS = ["dara_online_lag_m015", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]
write("scenarios_ct1", [gen_row("reference", out="runs/scenarios_ct1", networks=SC_NETS, contents=["x1"])] +
      [gen_row(R.run_name("ct1", m, s_), out="runs/scenarios_ct1", networks=SC_NETS, contents=["x1"])
       for s_ in a.seeds for m in SC_METHODS],
      "exploratory: ladder-x1 models on each generator scenario and on walking / driving / dead-zone groups, ladder x1")
