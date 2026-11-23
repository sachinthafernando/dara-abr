"""Single source of truth for the experiments: training environments, methods, test conditions and run names.

The sweep generator (scripts/sweeps/make_sweeps.py) turns these entries into notebook parameters; the analysis
scripts use the same keys, labels and colours, so a method is named and coloured the same everywhere.

Run names: <env>__<method>_s<seed>   e.g. ct4__dara_online_lag_s101
           (older runs named p2_<env>__<method>_s<seed> are parsed too)
"""
import re

SEEDS = [101, 202, 303, 404, 505]
EVAL_SEED = 99                    # the 100 test sessions shared by every method
EVAL_SESSIONS = 100

BASE_CHUNK_MB = [1.5, 3.0, 6.0, 12.0, 22.0]       # ladder x1: mean chunk size per tier (MB, 4-s chunks)
QUALITY = [1.0, 1.92, 2.61, 3.30, 4.00]
HARSH_SCENARIOS = ["urban_canyon", "tunnel", "under_bridge", "crowded_stationary"]

PROCESSED = "data/processed"
CONTENT_MODEL = f"{PROCESSED}/content_params.json"
IRISH_TRACES = f"{PROCESSED}/irish5g_download.npz"
LUMOS_TRACES = f"{PROCESSED}/lumos5g.npz"


def ladder(scale):
    return [round(scale * b, 6) for b in BASE_CHUNK_MB]


# ------------------------------------------------------------------------------------------- training environments
# primary=True: the environment of Table I and the generalisation test. Set 1 (ct1, ct2, ct4, ct8: the mmWave
# generator with four chunk-size ladders) feeds the per-ladder figures; set 2 (cth4, ctd1, ctl) harsh and real networks.
# holdout: share of the traces held out for the test sessions (trace networks only; see TRACE_HOLDOUT_FRAC).
TRAINING_ENVS = {
    "ct1":  dict(label="mmWave generator, ladder ×1", short="mmWave ×1", ladder_scale=1, primary=False,
                 net=dict(NET_SOURCE="generator")),
    "ct2":  dict(label="mmWave generator, ladder ×2", short="mmWave ×2", ladder_scale=2, primary=False,
                 net=dict(NET_SOURCE="generator")),
    "ct4":  dict(label="mmWave generator, ladder ×4", short="mmWave ×4", ladder_scale=4, primary=True,
                 net=dict(NET_SOURCE="generator")),
    "ct8":  dict(label="mmWave generator, ladder ×8", short="mmWave ×8", ladder_scale=8, primary=False,
                 net=dict(NET_SOURCE="generator")),
    "cth4": dict(label="Harsh mmWave scenarios, ladder ×4", short="harsh mmWave ×4", ladder_scale=4, primary=False,
                 net=dict(NET_SOURCE="generator", NET_SCENARIOS=HARSH_SCENARIOS)),
    "ctd1": dict(label="Irish 5G driving traces, ladder ×1 (4 of 16 traces held out)", short="Irish driving ×1",
                 ladder_scale=1, primary=False,
                 net=dict(NET_SOURCE="trace", TRACE_FILE=IRISH_TRACES, TRACE_PREFIX="Driving", TRACE_HOLDOUT_FRAC=0.25)),
    "ctl":  dict(label="Lumos5G traces, ladder ×4 (20% of traces held out)", short="Lumos5G ×4", ladder_scale=4,
                 primary=False,
                 net=dict(NET_SOURCE="trace", TRACE_FILE=LUMOS_TRACES, TRACE_PREFIX="none", TRACE_HOLDOUT_FRAC=0.2)),
    # earlier runs (budget table of the first five-seed batch)
    "cts1": dict(label="Irish 5G static traces, ladder ×1", short="Irish static ×1", ladder_scale=1, primary=False,
                 net=dict(NET_SOURCE="trace", TRACE_FILE=IRISH_TRACES, TRACE_PREFIX="Static")),
}
LADDER_ENVS = ["ct1", "ct2", "ct4", "ct8"]
HARSH_ENVS = ["cth4", "ctd1", "ctl"]
PRIMARY_ENV = "ct4"

# ------------------------------------------------------------------------------------------- methods
# role: proposed | diagnosis | reward_ablation | hand_tuned | baseline | ablation
# params: notebook parameters on top of the common ones (see sweep generator)
CAPS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 1e6]


def cap_key(c):
    return {0.1: "fixed", 1e6: "static_best"}.get(c, f"cap{c:g}")


def cap_label(c):
    return "∞" if c >= 1e5 else f"{c:g}"


METHODS = {
    "dara_online_lag":    dict(label="DARA-L (ours)", role="proposed", colour="#2a78d6",
                               params=dict(DARA_MODE="online_lag")),
    "dara_online":        dict(label="DARA, cap only", role="diagnosis", colour="#6250d6",
                               params=dict(DARA_MODE="online")),
    "corel_official":     dict(label="COREL", role="baseline", colour="#eda100",
                               params=dict(ALGO="corel_official")),
    "pensieve_official":  dict(label="Pensieve", role="baseline", colour="#1baf7a",
                               params=dict(ALGO="pensieve_official")),
    "ppo_ref_reward":     dict(label="PPO on the score", role="ablation", colour="#8a8986",
                               params=dict(REWARD_MODE="ref")),
    "dara_preflight_lag": dict(label="DARA-L, pre-flight", role="ablation", colour="#86b6ef",
                               params=dict(DARA_MODE="preflight_lag")),
    # sensitivity of the act threshold (primary rule: M >= 0.2); only differs from DARA-L where 0.15 <= M < 0.2
    "dara_online_lag_m015": dict(label="DARA-L, threshold 0.15", role="ablation", colour="#1c5cab",
                                 params=dict(DARA_MODE="online_lag", M_HIGH=0.15)),
}
for _c in CAPS:
    METHODS[cap_key(_c)] = dict(label=("Original reward (c = 0.1)" if _c == 0.1 else f"PPO, cap c = {cap_label(_c)}"),
                                role="reward_ablation", colour="#52514e", params=dict(RN_CAP=float(_c)), cap=_c)
HAND_BUDGETS = {"ct1": [0.25, 0.5, 1.0], "ct2": [0.125, 0.25, 0.5, 1.0], "ct4": [0.25, 0.5, 1.0, 2.0],
                "ct8": [0.25, 0.5, 1.0], "cth4": [0.125, 0.25, 0.5, 1.0], "cts1": [0.125, 0.25, 0.5, 1.0],
                "ctd1": [0.5], "ctl": [0.5]}
for _b in sorted({b for v in HAND_BUDGETS.values() for b in v}):
    METHODS[f"ppo_lagrangian_t{_b:g}"] = dict(label=f"PPO-Lagrangian, {_b:g} s", role="hand_tuned", colour="#e87ba4",
                                             params=dict(LAGRANGE=1, LAG_TARGET=float(_b)), budget=_b)
REFERENCE_POLICIES = {"robustmpc": dict(label="RobustMPC", colour="#eb6834")}
for _t in range(5):
    REFERENCE_POLICIES[f"tier{_t}"] = dict(label=f"Always tier {_t}", colour="#a9a8a4")

# ------------------------------------------------------------------------------------------- sweep groups
SET1_METHODS = ["dara_online_lag", "ppo_lagrangian_t0.25", "ppo_lagrangian_t0.5", "ppo_lagrangian_t1",
                "corel_official", "pensieve_official", "fixed", "cap0.25", "cap0.5", "cap1", "static_best"]
SET2_METHODS = ["dara_online_lag", "ppo_lagrangian_t0.5", "corel_official", "pensieve_official"]
GROUPS = {
    # set 1: every method on the four ladders of the mmWave generator (ct4 is complete from the first batch)
    "set1": dict(envs=LADDER_ENVS, methods=SET1_METHODS, extra={"ct1": ["dara_online_lag_m015"]}),
    # set 2: the main methods on harsh and real networks (cth4 DARA-L and 0.5 s exist from the first batch)
    "set2": dict(envs=HARSH_ENVS, methods=SET2_METHODS),
    # first batch (kept so its tables can be regenerated)
    "main": dict(envs=["ct4"], methods=["fixed"] + [cap_key(c) for c in CAPS[1:]] +
                 ["dara_online", "dara_online_lag", "ppo_lagrangian_t0.25", "ppo_lagrangian_t0.5",
                  "ppo_lagrangian_t1", "ppo_lagrangian_t2", "pensieve_official", "corel_official"]),
    "other_envs": dict(envs=["ct2", "cth4", "cts1"], methods=["dara_online_lag", "hand_budgets"]),
    "ablations": dict(envs=["ct4"], methods=["ppo_ref_reward", "dara_preflight_lag"]),
}

# ------------------------------------------------------------------------------------------- test conditions
NETWORKS = {
    "seen":          dict(label="Training generator", family="mmwave_synthetic"),
    "lumos5g":       dict(label="Lumos5G", family="mmwave_real"),
    "irish_static":  dict(label="Irish 5G static", family="non_mmwave_5g"),
    "irish_driving": dict(label="Irish 5G driving", family="non_mmwave_5g"),
    # optional synthetic stress tests (not run by default)
    "gen_harsh":     dict(label="Harsh mmWave scenarios", family="synthetic_stress"),
    "gen_half":      dict(label="Generator, throughput ×0.5", family="synthetic_stress"),
}
CONTENTS = {"x1": "×1", "x4": "×4", "x8": "×8", "x4_bursty": "×4 bursty"}
FAMILIES = {"mmwave_synthetic": "Synthetic mmWave", "mmwave_real": "Real mmWave (Lumos5G)",
            "non_mmwave_5g": "Non-mmWave 5G (Irish)", "synthetic_stress": "Synthetic stress"}
# trained policies (primary environment) that are played on the test conditions
GEN_METHODS = ["dara_online_lag", "corel_official", "pensieve_official", "ppo_lagrangian_t0.5", "dara_online"]
GEN_METHODS_ABLATION = ["ppo_ref_reward", "dara_preflight_lag"]
# exploratory (added 15 Oct, after the first results; not in the pre-registration): the other hand-picked budgets
# on the test networks, to place DARA-L's derived budget on the training-vs-shift trade-off
GEN_METHODS_BUDGETS = ["ppo_lagrangian_t0.25", "ppo_lagrangian_t1", "ppo_lagrangian_t2"]
GEN_NETWORKS = ["seen", "lumos5g", "irish_static", "irish_driving"]
GEN_CONTENT = ["x4", "x1", "x8"]
TRAINING_CONDITION = "seen/x4"

# ------------------------------------------------------------------------------------------- run names
_NAME = re.compile(r"^(?:smoke_)?(?:p2_)?(?P<env>[a-z0-9]+)__(?P<method>.+)_s(?P<seed>\d+)$")


def run_name(env, method, seed):
    return f"{env}__{method}_s{seed}"


def parse_run(name):
    m = _NAME.match(str(name))
    if not m:
        return None
    method = m["method"]
    if method == "ppo_lagrangian":            # older runs: the hand-tuned 0.5 s Lagrangian had no suffix
        method = "ppo_lagrangian_t0.5"
    return dict(env=m["env"], method=method, seed=int(m["seed"]))


def label(key):
    if key in METHODS:
        return METHODS[key]["label"]
    if key in REFERENCE_POLICIES:
        return REFERENCE_POLICIES[key]["label"]
    return key


def colour(key):
    if key in METHODS:
        return METHODS[key]["colour"]
    return REFERENCE_POLICIES.get(key, {}).get("colour", "#52514e")
