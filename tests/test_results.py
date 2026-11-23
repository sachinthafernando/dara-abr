"""Regression tests for the shipped results: they must reproduce the numbers published in the paper and thesis.

Run with `pytest` from the repository root (needs only numpy, pandas, scipy).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
T015 = ROOT / "results" / "views" / "t015" / "training"


def J0(env, method):
    t = pd.read_csv(T015 / "training_table.csv")
    return float(t[(t.env == env) & (t.method == method)].J0.iloc[0])


def test_diagnosis_validity():
    s = pd.read_csv(T015 / "diagnosis_validity_summary.csv").iloc[0]
    assert round(s.rho, 3) == 0.873
    assert (round(s.ci_lo, 3), round(s.ci_hi, 3)) == (0.778, 0.919)
    assert int(s.n) == 22 and s.threshold == 0.15
    assert round(100 * s.mean_shortfall_flagged) == 93
    assert round(100 * s.mean_shortfall_unflagged, 1) == 1.1


@pytest.mark.parametrize("env,method,value", [
    ("ct4", "dara_online_lag", 188.7), ("ct4", "ppo_lagrangian_t0.5", 192.1), ("ct4", "fixed", 65.9),
    ("ct4", "ppo_ref_reward", 191.3), ("ct4", "corel_official", 174.7), ("ct4", "pensieve_official", 145.6),
    ("ct4", "robustmpc", 147.1), ("ct1", "dara_online_lag", 287.5), ("ctl", "dara_online_lag", 253.8),
    ("ct2", "dara_online_lag", 238.1), ("cts1", "dara_online_lag", 220.9), ("cth4", "dara_online_lag", 99.8),
    ("ct8", "dara_online_lag", 100.2), ("ctd1", "dara_online_lag", 88.8),
])
def test_training_table(env, method, value):
    assert round(J0(env, method), 1) == value


def test_primary_environment_freeze():
    t = pd.read_csv(T015 / "training_table.csv").set_index(["env", "method"])
    assert round(t.loc[("ct4", "dara_online_lag"), "freeze_s"], 1) == 25.6
    assert round(t.loc[("ct4", "fixed"), "freeze_s"], 1) == 244.5
    assert round(t.loc[("ct4", "ppo_ref_reward"), "freeze_s"], 1) == 71.8


def test_derived_budgets_and_activation():
    b = pd.read_csv(T015 / "budget_test.csv").set_index("env")
    assert round(b.loc["ct4", "derived_budget"], 3) == 0.384
    assert (b.dara_acted == "5/5").all()
    gap = b.gap_rel * 100
    assert set(gap[(gap > -4.05) & (gap < 0)].index) == {"ct1", "ct2", "ct4", "cts1"}
    assert round(gap["ctl"]) == 9


def test_session_files_reproduce_transfer_table():
    s = pd.read_csv(ROOT / "results" / "sessions" / "transfer_ct1_sessions.csv.gz")
    assert len(s) == 41000
    p = s[s.method == "policy"].copy()
    p["m"] = p.run.str.split("__").str[1].str.replace(r"_s\d+$", "", regex=True)
    got = p.groupby(["network", "content", "m", "seed"]).J0.mean().groupby(["network", "content", "m"]).mean()
    ref = pd.read_csv(ROOT / "figures" / "paper" / "transfer_table.csv")
    ref = ref[ref.method != "robustmpc"].set_index(["network", "content", "method"]).J0
    common = ref.index.intersection(got.index)
    assert len(common) >= 30
    assert np.allclose(got.loc[common].values, ref.loc[common].values, atol=1e-3)


def test_scenario_sessions():
    s = pd.read_csv(ROOT / "results" / "sessions" / "scenarios_ct1_sessions.csv.gz")
    assert len(s) == 28600 and s.network.nunique() == 11


def test_benchmark_summary_runs(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analysis" / "benchmark_summary.py"),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = (out / "BENCHMARK.md").read_text(encoding="utf-8")
    assert "188.7 ± 4.6" in md and "ρ = 0.873" in md
