#!/usr/bin/env python3
"""Table I of the paper: one row per training environment, ordered by misalignment M.

  M          misalignment of the training reward, measured before training (identical for every seed: it depends on
             the environment and the reward only)
  Acts       seeds in which DARA switched the constraint on, and the median episode at which it did
  Budget     DARA-L's derived freeze budget (s per chunk)
  J0 columns mean J0 per session over the 100 shared test sessions (learned methods: mean over seeds +- s.d.):
             PPO with the original reward (ladder environments only), DARA-L, the best hand-picked PPO-Lagrangian
             budget in that environment (its budget in brackets), RobustMPC, the best always-one-tier rule
  The highest J0 among the learned methods of each row is marked in bold (LaTeX, Markdown).

Writes table1.csv, table1.md and table1.tex (IEEE table*, booktabs) into --out.
Usage: python scripts/analysis/make_table1.py --results results/views/t015 --runs runs/views/t015/train --out figures/paper
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dara_abr import registry as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--runs", default=str(ROOT / "runs" / "train"))
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
ap.add_argument("--out", default=str(ROOT / "figures" / "paper"))
ap.add_argument("--exclude-envs", default="", help="comma-separated training environments to leave out")
ap.add_argument("--no-best-tier", action="store_true", help="leave out the best always-one-tier column")
a = ap.parse_args()
EXCL = {e for e in a.exclude_envs.split(",") if e}
TIER = not a.no_best_tier
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
PR = json.load(open(a.prereg))
REF = PR["proposed_method"]
T = pd.read_csv(Path(a.results) / "training" / "training_table.csv")

NET = {"ct1": "mmWave generator", "ct2": "mmWave generator", "ct4": "mmWave generator", "ct8": "mmWave generator",
       "cth4": "Harsh mmWave", "ctl": "Lumos5G (real)", "cts1": "Irish 5G static", "ctd1": "Irish 5G driving"}


def env_info(e):
    ms, eps = [], []
    global ALL_EPS
    for d in sorted(Path(a.runs).glob(f"{e}__{REF}_s*")):
        f = d / "dara_log.csv"
        if not f.exists():
            continue
        L = pd.read_csv(f)
        if len(L):
            ms.append(float(L.M.iloc[0]))
            on = L[L.action.astype(str).str.startswith("Lagrangian on")]
            if len(on):
                eps.append(int(on.episode.iloc[0])); ALL_EPS.append(int(on.episode.iloc[0]))
    return (np.mean(ms) if ms else np.nan), (int(np.median(eps)) if eps else None)


def cell(e, m, col="J0"):
    r = T[(T.env == e) & (T.method == m)]
    return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan


ALL_EPS = []
rows = []
for e in [e for e in T.env.unique() if e not in EXCL]:
    M, ep = env_info(e)
    d = T[T.env == e]
    hand = d[d.method.str.startswith("ppo_lagrangian_t")]
    hb = hand.loc[hand.J0.idxmax()] if len(hand) else None
    tiers = d[d.method.str.match(r"^tier\d$")]
    bt = tiers.loc[tiers.J0.idxmax()] if len(tiers) else None
    acted = d.loc[d.method == REF, "dara_acted"]
    rows.append(dict(
        env=e, network=NET.get(e, e), ladder=f"×{R.TRAINING_ENVS[e]['ladder_scale']:g}" if e in R.TRAINING_ENVS else "",
        M=M, acted=str(acted.iloc[0]) if len(acted) else "", switch_on_episode=ep, budget=cell(e, REF, "budget"),
        J0_original=cell(e, "fixed"), J0_original_sd=cell(e, "fixed", "J0_seed_sd"),
        J0_daral=cell(e, REF), J0_daral_sd=cell(e, REF, "J0_seed_sd"),
        J0_best_hand=float(hb.J0) if hb is not None else np.nan,
        J0_best_hand_sd=float(hb.J0_seed_sd) if hb is not None else np.nan,
        best_hand_budget=float(hb.budget) if hb is not None else np.nan,
        n_hand_budgets=len(hand), J0_robustmpc=cell(e, "robustmpc"),
        J0_best_tier=float(bt.J0) if bt is not None else np.nan, best_tier=str(bt.method) if bt is not None else ""))
X = pd.DataFrame(rows).sort_values("M").reset_index(drop=True)
X.round(4).to_csv(OUT / "table1.csv", index=False)


def pm(v, s):
    if not np.isfinite(v):
        return "–"
    return f"{v:.1f}" + (f" ± {s:.1f}" if np.isfinite(s) else "")


def fmt(r, bold, tex):
    learned = {"o": r.J0_original, "d": r.J0_daral, "h": r.J0_best_hand}
    top = max((v for v in learned.values() if np.isfinite(v)), default=np.nan)
    def b(k, txt):
        if bold and np.isfinite(learned[k]) and learned[k] == top:
            return (r"\textbf{" + txt + "}") if tex else f"**{txt}**"
        return txt
    sep = r" $\pm$ " if tex else " ± "
    def pm_(v, s):
        return pm(v, s).replace(" ± ", sep)
    hand = (pm_(r.J0_best_hand, r.J0_best_hand_sd) + f" ({r.best_hand_budget:g})") if np.isfinite(r.J0_best_hand) else "–"
    if np.isfinite(r.J0_best_hand) and r.n_hand_budgets == 1:
        hand += "$^\\dagger$" if tex else "†"
    if tex:
        def pm_(v, s_):
            if not np.isfinite(v):
                return "--"
            return f"{v:.1f}" + (r"{\tiny$\,\pm$" + f"{s_:.1f}" + "}" if np.isfinite(s_) else "")
        hand = (pm_(r.J0_best_hand, r.J0_best_hand_sd) + f" ({r.best_hand_budget:g})") if np.isfinite(r.J0_best_hand) else "--"
        if np.isfinite(r.J0_best_hand) and r.n_hand_budgets == 1:
            hand += "$^\\dagger$"
        return [f"{r.network} {r.ladder}", f"{r.M:.2f}", f"{r.budget:.2f}",
                b("o", pm_(r.J0_original, r.J0_original_sd)), b("d", pm_(r.J0_daral, r.J0_daral_sd)), b("h", hand),
                f"{r.J0_robustmpc:.1f}"] + ([f"{r.J0_best_tier:.1f}"] if TIER else [])
    return [f"{r.network}", r.ladder, f"{r.M:.2f}", f"{r.acted}", f"{r.budget:.2f}",
            b("o", pm_(r.J0_original, r.J0_original_sd)), b("d", pm_(r.J0_daral, r.J0_daral_sd)), b("h", hand),
            f"{r.J0_robustmpc:.1f}"] + ([f"{r.J0_best_tier:.1f}"] if TIER else [])


HDR = ["Training network", "Ladder", "$M$", "Acts", "Budget (s/chunk)", "PPO, original reward", "DARA-L",
       "Best hand-picked PPO-Lag. (budget)", "RobustMPC"] + (["Best fixed tier"] if TIER else [])
acts = sorted(set(X.acted))
note = ("Mean $J_0$ per session over 100 shared test sessions; learned methods: mean ± s.d. over 5 training seeds. "
        "$M$: misalignment of the training reward, measured before training. "
        + (f"DARA switched the freeze constraint on in {acts[0]} seeds in every environment"
           if len(acts) == 1 else "DARA switched the freeze constraint on in " + ", ".join(f"{r.env} {r.acted}" for r in X.itertuples()))
        + f", during training (between episodes {min(ALL_EPS)} and {max(ALL_EPS)}). Best hand-picked: the best of the "
        "PPO-Lagrangian budgets trained in that environment, chosen afterwards on the test sessions (an oracle choice; "
        f"{'–'.join(map(str, sorted({int(n) for n in X.n_hand_budgets if n > 1})))} budgets, each a full training run). "
        "Bold: best learned method in the row."
        + (" †: only one hand-picked budget (0.5 s) was trained there." if (X.n_hand_budgets == 1).any() else "")
        + (" –: not trained." if X.J0_original.isna().any() else ""))
md = ["| " + " | ".join(h.replace("$", "") for h in HDR) + " |", "|" + "---|" * len(HDR)]
md += ["| " + " | ".join(fmt(r, True, False)) + " |" for r in X.itertuples()]
(OUT / "table1.md").write_text("# Table I\n\n" + "\n".join(md) + "\n\n" + note.replace("$", "") + "\n")
tex = [r"\begin{table*}[t]", r"\centering",
       r"\caption{Training environments: misalignment before training, DARA's decision and derived budget, and QoE $J_0$ after training.}",
       r"\label{tab:envs}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{lrr" + "r" * (5 if TIER else 4) + "}", r"\toprule",
       r"Training environment & $M$ & Budget & \multicolumn{" + str(5 if TIER else 4) + r"}{c}{Mean $J_0$ per session} \\",
       r"\cmidrule(lr){4-" + str(8 if TIER else 7) + "}",
       r" & & (s/chunk) & PPO, orig.\ reward & DARA-L & Best hand-picked (budget) & RobustMPC"
       + (r" & Best fixed tier" if TIER else "") + r" \\",
       r"\midrule"]
import re  # noqa: E402
tex += [re.sub(r"(^|[ {])-(?=\d)", r"\1$-$", " & ".join(fmt(r, True, True)).replace("×", r"$\times$")) + r" \\"
        for r in X.itertuples()]
tex += [r"\bottomrule", r"\end{tabular}", "",
        r"\vspace{2pt}\parbox{\textwidth}{\scriptsize " + note.replace("±", r"$\pm$").replace("†", r"$^\dagger$")
        .replace("–", "--") + "}", r"\end{table*}"]
(OUT / "table1.tex").write_text("\n".join(tex) + "\n")
print("\n".join(md)); print(); print(note)
