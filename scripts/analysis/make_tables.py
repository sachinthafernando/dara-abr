#!/usr/bin/env python3
"""Paper tables (LaTeX, booktabs) and Markdown copies for reading, from the evaluation CSVs.

  table_training.tex        Table I   primary environment: every method vs DARA-L, with intervals and tests
  table_generalisation.tex  Table II  every test condition (incl. both Irish 5G sets): DARA-L minus each comparator
  table_budget.tex          Table III budget rule across training environments
Inputs : results/training/*.csv, results/generalisation/*.csv   Outputs: results/tables/
Usage:  python scripts/analysis/make_tables.py
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
ap.add_argument("--prereg", default=str(ROOT / "configs" / "preregistration" / "paper_v2.json"))
a = ap.parse_args()
RES = Path(a.results); OUT = RES / "tables"; OUT.mkdir(parents=True, exist_ok=True)
PR = json.load(open(a.prereg))
ENV = PR["primary_training_environment"]
md = []


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    return r"$<10^{-4}$" if p < 1e-4 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def neg(s):
    """typeset minus signs in a numeric cell"""
    return s.replace("-", "$-$")


def write(name, tex, note):
    (OUT / f"{name}.tex").write_text(tex)
    print(f"written {OUT / name}.tex  ({note})")


# ---------------------------------------------------------------------------------------------- Table I
tt, ts = RES / "training" / "training_table.csv", RES / "training" / "training_tests.csv"
if tt.exists() and ts.exists():
    T = pd.read_csv(tt); P = pd.read_csv(ts)
    T, P = T[T.env == ENV], P[P.env == ENV]
    order = [PR["proposed_method"]] + [c for c in PR["comparisons"]["training_environment"]["against"]]
    rows = []
    for m in order:
        key = m
        if m == "best_single_tier":
            hit = P[P.comparison.str.startswith("best_single_tier")]
            if hit.empty:
                continue
            key = hit.comparison.iloc[0].split("(")[1].rstrip(")")
        t = T[T.method == key]
        if t.empty:
            continue
        t = t.iloc[0]
        p = P[P.comparison.str.startswith(m)] if m != PR["proposed_method"] else None
        lab = R.label(key) if m != "best_single_tier" else f"Best single tier ({key[-1]})"
        js = f"{t.J0:.1f}" + (f" $\\pm$ {t.J0_seed_sd:.1f}" if np.isfinite(t.get("J0_seed_sd", np.nan)) else "")
        if p is None or p.empty:
            rows.append((lab, js, f"{t.freeze_s:.1f}", "--", "--", "--", "--", "--"))
        else:
            p = p.iloc[0]
            rows.append((lab, js, f"{t.freeze_s:.1f}", neg(f"{p['diff']:+.1f} [{p.ci_lo:+.1f}, {p.ci_hi:+.1f}]"),
                         f"{p.share_sessions_better:.2f}", fmt_p(p.q_session), fmt_p(p.p_seed), p.verdict.replace("DARA-L ", "")))
    head = ["Method", r"$J_0$ (mean $\pm$ seed s.d.)", "Freeze (s)", r"$\Delta J_0$ DARA-L $-$ method [95\% CI]",
            "Sessions DARA-L better", "$q$ (session)", "$p$ (seed)", "Verdict"]
    tex = ("\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Training environment (" + R.TRAINING_ENVS[ENV]["label"].replace("×", r"$\times$") +
           "): 100 test sessions, " + f"{len(PR['seeds'])}" + " seeds per trained method.}\n\\label{tab:training}\n"
           "\\begin{tabular}{lrrrrrrl}\\toprule\n" + " & ".join(head) + " \\\\\\midrule\n" +
           "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\n\\end{table*}\n")
    write("table_training", tex, "Table I")
    head_md = ["Method", "J0 (mean ± seed s.d.)", "Freeze (s)", "ΔJ0 DARA-L − method [95% CI]", "Sessions DARA-L better",
               "q (session)", "p (seed)", "Verdict"]
    md += ["## Table I: training environment", "| " + " | ".join(head_md) + " |", "|" + "---|" * len(head_md)]
    md += ["| " + " | ".join(x.replace("$-$", "-").replace("$\\pm$", "±").replace("$<10^{-4}$", "<1e-4") for x in r) + " |"
           for r in rows]

# ---------------------------------------------------------------------------------------------- Table II
gt = RES / "generalisation" / "gen_tests.csv"
if gt.exists():
    G = pd.read_csv(gt)
    comps = [c for c in PR["comparisons"]["generalisation"]["against"] if c in set(G.comparison)]
    conds = [f"{n}/{c}" for n in PR["generalisation"]["table"]["networks"]
             for c in sorted(PR["generalisation"]["contents"], key=lambda c: list(R.CONTENTS).index(c))]
    conds = [c for c in conds if c in set(G.condition)]
    rows = []
    for c in conds:
        n, ct = c.split("/")
        r = [R.NETWORKS[n]["label"] + (" (training)" if c == R.TRAINING_CONDITION else ""), R.CONTENTS[ct].replace("×", r"$\times$")]
        for m in comps:
            g = G[(G.condition == c) & (G.comparison == m)]
            r.append(neg(f"{g['diff'].iloc[0]:+.1f}{g['mark'].iloc[0]}") if len(g) else "--")
        rows.append(r)
    head = ["Network", "Ladder"] + [R.label(m) for m in comps]
    tex = ("\\begin{table}[t]\\centering\\footnotesize\n\\caption{DARA-L minus each method, mean $J_0$ per session on "
           "unseen conditions. + / $-$: significantly better / worse (session $q<0.05$ and bootstrap CI excluding 0); "
           "=: no significant difference; ?: the two tests disagree.}\n\\label{tab:generalisation}\n"
           "\\begin{tabular}{ll" + "r" * len(comps) + "}\\toprule\n" + " & ".join(head) + " \\\\\\midrule\n" +
           "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    write("table_generalisation", tex, "Table II")
    md += ["", "## Table II: generalisation (DARA-L minus method)", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(x.replace("$\\times$", "×").replace("$-$", "-") for x in r) + " |" for r in rows]

# ---------------------------------------------------------------------------------------------- Table III
bt = RES / "training" / "budget_test.csv"
if bt.exists():
    B = pd.read_csv(bt)
    rows = [(R.TRAINING_ENVS.get(r.env, {}).get("label", r.env).replace("×", r"$\times$"), f"{r.derived_budget:.3f}",
             f"{r.J0_daral:.1f}", f"{r.best_hand_budget:g}", f"{r.J0_best_hand:.1f}", f"{100 * r.gap_rel:+.1f}\\%",
             "n/a" if pd.isna(r.passes) else ("yes" if str(r.passes) == "True" else "no")) for r in B.itertuples()]
    head = ["Training environment", "Derived budget (s/chunk)", "DARA-L $J_0$", "Best hand budget", "Its $J_0$", "Gap", "Within 5\\%"]
    tex = ("\\begin{table}[t]\\centering\\footnotesize\n\\caption{Budget rule: DARA-L's derived freeze budget against "
           "hand-picked budgets.}\n\\label{tab:budget}\n\\begin{tabular}{lrrrrrl}\\toprule\n" + " & ".join(head) +
           " \\\\\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
    write("table_budget", tex, "Table III")
    md += ["", "## Table III: budget rule", "| " + " | ".join(h.replace("$", "").replace("\\%", "%") for h in head) + " |",
           "|" + "---|" * len(head)]
    md += ["| " + " | ".join(x.replace("$\\times$", "×").replace("\\%", "%") for x in r) + " |" for r in rows]


# ---------------------------------------------------------------------------------------------- Table IV: every environment
if tt.exists() and ts.exists():
    T = pd.read_csv(tt); P = pd.read_csv(ts)
    order = [e for e in list(R.LADDER_ENVS) + list(R.HARSH_ENVS) + ["cts1"] if e in set(T.env)]
    cols = [PR["proposed_method"], "ppo_lagrangian_t0.5", "corel_official", "pensieve_official", "robustmpc"]
    mark = {"better": "+", "worse": "-", "no significant difference": "=", "inconclusive (tests disagree)": "?"}
    rows = []
    for e in order:
        d = T[T.env == e].set_index("method")
        r = [R.TRAINING_ENVS.get(e, {}).get("short", e).replace("×", r"$\times$")]
        for m in cols:
            if m not in d.index:
                r.append("--"); continue
            cell = f"{d.loc[m, 'J0']:.1f} ({d.loc[m, 'freeze_s']:.0f})"
            if m != PR["proposed_method"]:
                p = P[(P.env == e) & (P.comparison == m)]
                if len(p):
                    cell += " " + mark.get(p.verdict.iloc[0].replace("DARA-L ", ""), "")
            else:
                b = d.loc[m].get("budget", np.nan)
                cell += f" [{b:.2f}]" if np.isfinite(b) else " [--]"
            r.append(neg(cell))
        rows.append(r)
    head = ["Training environment", "DARA-L [budget]", "PPO-Lag., 0.5 s", "COREL", "Pensieve", "RobustMPC"]
    tex = ("\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Every training environment: mean $J_0$ (freeze s per "
           "session) over the 100 test sessions and 5 seeds. DARA-L's derived budget (s per chunk) in brackets; [--]: DARA did "
           "not act. Marks give DARA-L against that method: + / $-$ DARA-L significantly better / worse, = no significant "
           "difference, ? tests disagree.}\n\\label{tab:environments}\n\\begin{tabular}{l" + "r" * (len(head) - 1) +
           "}\\toprule\n" + " & ".join(head) + " \\\\\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows) +
           "\n\\bottomrule\\end{tabular}\n\\end{table*}\n")
    write("table_environments", tex, "Table IV")
    md += ["", "## Table IV: every training environment (J0, freeze s; DARA-L [budget]; mark = DARA-L vs method)",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(x.replace("$\\times$", "×").replace("$-$", "-") for x in r) + " |" for r in rows]

(OUT / "tables.md").write_text("\n".join(md) + "\n")
print(f"written {OUT / 'tables.md'}")
