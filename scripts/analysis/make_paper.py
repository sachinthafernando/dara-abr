#!/usr/bin/env python3
"""Every figure and table of the paper, in one command. IEEE format (Times, 8 pt, vector PDF + PNG, no titles).

  Fig. 1  fig_overview           DARA / DARA-L flowchart                                  make_fig_overview.py
  Fig. 2  fig_diagnosis          M before training vs shortfall after training (C1)        make_paper_figures.py
  Tab. I  table1.tex/.md/.csv    training environments: M, derived budget, J0              make_table1.py
  Fig. 3  fig_qoe_components     J0 and freeze per training environment                    make_paper_figures.py
  Fig. 4  fig_budget_sweep       J0 vs hand-picked budgets, DARA-L's derived budget        make_paper_figures.py
  Fig. 5  fig_transfer_ct1       ladder-x1 models on unseen networks and ladder x4         plot_transfer.py
  Fig. 6  fig_scenarios_groups   per-scenario trade-off (walking / driving / dead zone)    plot_scenarios.py
          fig_scenarios_all      the same for each of the eight generator scenarios        plot_scenarios.py
  Fig. 7  fig_switching          quality switches over all test sessions                   plot_switching.py
  Fig. 8  fig_session_ct1_epN    one typical test session, chunk by chunk                  plot_session.py
  captions.md                    a caption for each, with the numbers computed from the results

Defaults follow the paper's choices: DARA-L at act threshold 0.15 (the threshold view, built first if missing),
training environments ct8 (mmWave generator x8) and ctd1 (Irish 5G driving) left out, no best-fixed-tier reference.
A figure whose input is missing is skipped with a message; the rest are still made.

Inputs (cluster layout):
  results/views/t015, runs/views/t015/train, configs/preregistration/view_t015.json   (threshold_view.py)
  runs/generalise_ct1/generalize/gen_*.csv      configs/sweeps/generalise_ct1.tsv + generalise_ct1_budgets.tsv
  runs/scenarios_ct1/generalize/gen_*.csv       configs/sweeps/scenarios_ct1.tsv
  runs/timeline/timeline_ct1_all.csv[.gz]       notebook MODE=timeline, TL_TAG=ct1_all (+ _network.csv)

Usage:  python scripts/analysis/make_paper.py [--out figures/paper] [--exclude-envs ct8,ctd1] [--with-best-tier]
"""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "scripts" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--results", default=None, help="default: results/views/t<threshold>")
ap.add_argument("--runs", default=None, help="default: runs/views/t<threshold>/train")
ap.add_argument("--prereg", default=None, help="default: configs/preregistration/view_t<threshold>.json")
ap.add_argument("--transfer", default=str(ROOT / "runs" / "generalise_ct1"))
ap.add_argument("--scenarios", default=str(ROOT / "runs" / "scenarios_ct1"))
ap.add_argument("--timeline", default=str(ROOT / "runs" / "timeline" / "timeline_ct1_all.csv"))
ap.add_argument("--exclude-envs", default="ct8,ctd1")
ap.add_argument("--with-best-tier", action="store_true")
ap.add_argument("--out", default=str(ROOT / "figures" / "paper"))
a = ap.parse_args()

tag = f"t{a.threshold:g}".replace(".", "")
RES = Path(a.results or ROOT / "results" / "views" / tag)
RUNS = Path(a.runs or ROOT / "runs" / "views" / tag / "train")
PRE = Path(a.prereg or ROOT / "configs" / "preregistration" / f"view_{tag}.json")
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
WORK = OUT / "_work"; WORK.mkdir(exist_ok=True)
PY = sys.executable
done, skipped, caps = [], [], {}


def run(args, step):
    print(f"\n=== {step}: {' '.join(str(x) for x in args[1:3])} ...")
    r = subprocess.run([PY] + [str(x) for x in args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        skipped.append(f"{step}: failed (see output above)")
        return False
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("written", "WARNING", "C1"))))
    return True


def take(src_dir, names):
    for n in names:
        for ext in ("pdf", "png"):
            f = Path(src_dir) / f"{n}.{ext}"
            if f.exists():
                shutil.copy(f, OUT / f.name)
        if (Path(src_dir) / f"{n}.pdf").exists():
            done.append(n)


def first_para(path):
    p = Path(path)
    if not p.exists():
        return ""
    return next((l for l in p.read_text().split("\n") if l.strip() and not l.startswith(("#", "|"))), "")


# ------------------------------------------------------------------------------------------------ threshold view
if not (RES / "training" / "training_table.csv").exists() or not PRE.exists():
    print(f"{RES} or {PRE} missing: building the threshold view first")
    run([AN / "threshold_view.py", "--threshold", a.threshold], "threshold view")
PR = json.load(open(PRE)) if PRE.exists() else {}
THR = PR.get("act_threshold", {}).get("primary", a.threshold)
excl = ["--exclude-envs", a.exclude_envs] if a.exclude_envs else []
tier = [] if a.with_best_tier else ["--no-best-tier"]

# ------------------------------------------------------------------------------------------------ Fig. 1
if run([AN / "make_fig_overview.py", "--threshold", THR, "--out", WORK / "overview"], "Fig. 1 overview"):
    take(WORK / "overview", ["fig_overview"])
    caps["fig_overview"] = ("DARA and DARA-L. During PPO training, DARA periodically measures, from counterfactual "
                            "rollouts that need no policy, how often the training reward's preferred tier is worse for "
                            r"the intended QoE $J_0$ than the intended-best tier (misalignment $M$). If $M \ge \tau$ "
                            rf"($\tau$ = {THR:g}), it switches on a freeze constraint whose budget $\hat b$ is the "
                            "freeze per chunk of the intended-best tier.")

# ------------------------------------------------------------------------------------------------ Figs. 2-4
if (RES / "training" / "training_table.csv").exists():
    names = ["fig_diagnosis", "fig_qoe_components", "fig_budget_sweep"]
    if run([AN / "make_paper_figures.py", "--results", RES, "--runs", RUNS, "--prereg", PRE, "--out", WORK / "training",
            "--only", ",".join(names)] + excl + tier, "Figs. 2-4 training environments"):
        take(WORK / "training", names)
        cm = (WORK / "training" / "captions.md").read_text()
        for n in names:
            m = re.search(r"\*\*" + n + r"\.\*\* (.*)", cm)
            if m:
                caps[n] = m.group(1)
        if "fig_qoe_components" in caps and "cts1" not in a.exclude_envs:
            T = pd.read_csv(RES / "training" / "training_table.csv")
            miss = [m for m in ("corel_official", "pensieve_official") if not ((T.env == "cts1") & (T.method == m)).any()]
            if miss:
                caps["fig_qoe_components"] += " COREL and Pensieve were not trained on Irish 5G static."
    if run([AN / "make_table1.py", "--results", RES, "--runs", RUNS, "--prereg", PRE, "--out", WORK / "table"]
           + excl + tier, "Table I"):
        for f in ("table1.tex", "table1.md", "table1.csv"):
            shutil.copy(WORK / "table" / f, OUT / f)
        done.append("table1")
else:
    skipped.append(f"Figs. 2-4 and Table I: {RES / 'training' / 'training_table.csv'} not found")

# ------------------------------------------------------------------------------------------------ Fig. 5
if list((Path(a.transfer) / "generalize").glob("gen_*.csv")):
    if run([AN / "plot_transfer.py", "--gen", a.transfer, "--env", "ct1", "--out", WORK / "transfer"], "Fig. 5 transfer"):
        take(WORK / "transfer", ["fig_transfer_ct1"])
        caps["fig_transfer_ct1"] = first_para(WORK / "transfer" / "caption.md")
        for f in ("transfer_table.csv", "transfer_tests.csv"):
            shutil.copy(WORK / "transfer" / f, OUT / f)
else:
    skipped.append(f"Fig. 5: no generalize/gen_*.csv under {a.transfer}")

# ------------------------------------------------------------------------------------------------ Fig. 6
if list((Path(a.scenarios) / "generalize").glob("gen_*.csv")):
    if run([AN / "plot_scenarios.py", "--gen", a.scenarios, "--env", "ct1", "--out", WORK / "scenarios"],
           "Fig. 6 scenarios"):
        take(WORK / "scenarios", ["fig_scenarios_groups", "fig_scenarios_all"])
        caps["fig_scenarios_groups"] = first_para(WORK / "scenarios" / "caption.md")
        for f in ("scenario_table.csv", "scenario_tests.csv"):
            shutil.copy(WORK / "scenarios" / f, OUT / f)
else:
    skipped.append(f"Fig. 6: no generalize/gen_*.csv under {a.scenarios}")

# ------------------------------------------------------------------------------------------------ Figs. 7-8
TL = Path(a.timeline)
if not TL.exists() and Path(str(TL) + ".gz").exists():
    TL = Path(str(TL) + ".gz")
if TL.exists():
    if run([AN / "plot_switching.py", "--timeline", TL, "--out", WORK / "switching"], "Fig. 7 switching"):
        take(WORK / "switching", ["fig_switching"])
        caps["fig_switching"] = first_para(WORK / "switching" / "caption.md")
        shutil.copy(WORK / "switching" / "switching_table.csv", OUT / "switching_table.csv")
    # Fig. 8: for each method, the seed closest to the method's 5-seed mean on J0, freeze and switching together;
    # plot_session.py then picks the most typical session among those where every learned method stalls
    D = pd.read_csv(TL)
    D = D[D.groupby(["method", "episode"]).chunk.transform("size") == D.n_chunks]          # complete sessions only
    D = D.sort_values(["method", "episode", "chunk"])
    D["dq"] = D.groupby(["method", "episode"]).quality.diff().abs().fillna(0)
    E = D.groupby(["method", "episode"]).agg(J0=("j0_chunk", "sum"), fr=("stall", "sum"), dq=("dq", "sum"))
    Mn = E.groupby("method").mean()
    Mn["fam"] = [re.sub(r"_s\d+$", "", r) for r in Mn.index]
    order = ["dara_online_lag_m015", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]
    names = {"dara_online_lag_m015": "DARA-L", "ppo_lagrangian_t0.5": "PPO-Lag. (0.5 s)", "corel_official": "COREL",
             "pensieve_official": "Pensieve"}
    pick = []
    for m in order:
        g = Mn[Mn.fam.str.endswith("__" + m)]
        if len(g) == 0:
            continue
        z = ((g[["J0", "fr", "dq"]] - g[["J0", "fr", "dq"]].mean()) / g[["J0", "fr", "dq"]].std().replace(0, 1)).abs().sum(axis=1)
        pick.append(f"{z.idxmin()}={names[m]}")
    if "robustmpc" in Mn.index:
        pick.append("robustmpc=RobustMPC")
    net = Path(re.sub(r"\.csv(\.gz)?$", r"_network.csv\1", str(TL)))
    if not net.exists():
        net = Path(re.sub(r"\.csv(\.gz)?$", "_network.csv", str(TL)))
    if pick and net.exists():
        print("Fig. 8 seeds:", ", ".join(pick))
        if run([AN / "plot_session.py", "--timeline", TL, "--network", net, "--labels", ",".join(pick),
                "--out", WORK / "session"], "Fig. 8 example session"):
            for f in sorted((WORK / "session").glob("fig_session_*.pdf")):
                take(WORK / "session", [f.stem])
                caps[f.stem] = (first_para(WORK / "session" / f"{f.stem}_caption.md") +
                                " Seeds and session chosen as the most typical by $J_0$, freeze and switching; every "
                                "learned method stalls at least once in it.")
    else:
        skipped.append("Fig. 8: network file of the timeline not found" if pick else "Fig. 8: no runs in the timeline")
else:
    skipped.append(f"Figs. 7-8: {a.timeline} not found")

# ------------------------------------------------------------------------------------------------ captions
ORDER = [("Fig. 1", "fig_overview"), ("Fig. 2", "fig_diagnosis"), ("Table I", "table1"), ("Fig. 3", "fig_qoe_components"),
         ("Fig. 4", "fig_budget_sweep"), ("Fig. 5", "fig_transfer_ct1"), ("Fig. 6", "fig_scenarios_groups"),
         ("Fig. 7", "fig_switching")] + [("Fig. 8", n) for n in done if n.startswith("fig_session_")]
L = [f"# Paper figures (DARA-L at act threshold {THR:g}; left out: {a.exclude_envs or 'none'}"
     f"{'' if a.with_best_tier else '; no best-fixed-tier reference'})", ""]
for lab, n in ORDER:
    if n == "table1" and "table1" in done:
        L += [f"**{lab} (table1.tex).** Training environments: misalignment before training, DARA's derived budget, "
              "and QoE $J_0$ after training (note under the table).", ""]
    elif n in caps:
        L += [f"**{lab} ({n}).** {caps[n]}", ""]
if "fig_scenarios_all" in done:
    L += ["Also written: fig_scenarios_all (the eight generator scenarios one by one), transfer/scenario/switching "
          "tables and significance tests (CSV).", ""]
(OUT / "captions.md").write_text("\n".join(L))
print("\n=== done:", ", ".join(done))
for s in skipped:
    print("SKIPPED", s)
print("output:", OUT)
