"""One-off patch (kept for the record): held-out traces for trace-based training environments.

TRACE_HOLDOUT_FRAC > 0 splits the traces selected by TRACE_PREFIX into training traces and held-out test traces
(stratified by the name prefix before the first '_', seeded by TRACE_HOLDOUT_SEED, deterministic). Environments built
with split='test' use only the held-out traces; every other split (training, BC data, DARA's diagnostic states and
learning gate, COREL/Pensieve validation, landscape) uses only the training traces. The final test sessions
(ground_truth and the generalisation player) use split='test', which is identical to split='eval' when no traces
are held out, so generator environments and existing trace environments are unchanged.
Usage: python scripts/dev/patch_holdout.py notebooks/dara_abr.ipynb
"""
import json, sys

p = sys.argv[1]
nb = json.load(open(p))
cells = nb["cells"]


def get(i): return "".join(cells[i]["source"])
def put(i, s): cells[i]["source"] = s.splitlines(keepends=True)


def rep(i, old, new):
    s = get(i)
    assert s.count(old) == 1, (i, old[:60], s.count(old))
    put(i, s.replace(old, new))


rep(2, 'TRACE_PREFIX    = ""\n', 'TRACE_PREFIX    = ""\n'
    '# trace only: share of the selected traces held out for testing (0 = none; stratified by name prefix, seeded);\n'
    '# held-out traces are used only by the final test sessions, never for training, diagnosis or validation\n'
    'TRACE_HOLDOUT_FRAC = 0.0\nTRACE_HOLDOUT_SEED = 7\n')

rep(5, '''class ABREnv:''', '''def holdout_traces(keys, cfg):
    """Deterministic held-out subset of `keys` (sorted names): within each name prefix, round(frac * n) traces
    (at least one when frac > 0), chosen with TRACE_HOLDOUT_SEED."""
    frac = float(getattr(cfg, "TRACE_HOLDOUT_FRAC", 0.0) or 0.0)
    if frac <= 0:
        return []
    groups = {}
    for k in sorted(keys):
        groups.setdefault(k.split("_")[0], []).append(k)
    out = []
    for gi, (g, ks) in enumerate(sorted(groups.items())):
        n = max(1, int(round(frac * len(ks))))
        assert n < len(ks), f"holding out {n} of {len(ks)} traces with prefix {g!r} leaves none for training"
        rng = np.random.default_rng([int(getattr(cfg, "TRACE_HOLDOUT_SEED", 7)), gi])
        out += [ks[j] for j in sorted(rng.choice(len(ks), size=n, replace=False))]
    return sorted(out)

class ABREnv:''')
rep(5, '''            tr = load_traces(cfg.TRACE_FILE)
            self._traces = {k: v for k, v in tr.items() if k.startswith(cfg.TRACE_PREFIX)}
            assert self._traces, f"no traces start with {cfg.TRACE_PREFIX!r} in {cfg.TRACE_FILE}"
            self._names = sorted(self._traces)''', '''            tr = load_traces(cfg.TRACE_FILE)
            keys = sorted(k for k in tr if k.startswith(cfg.TRACE_PREFIX))
            assert keys, f"no traces start with {cfg.TRACE_PREFIX!r} in {cfg.TRACE_FILE}"
            hold = holdout_traces(keys, cfg)
            if hold:                                     # 'test' = held-out traces only; everything else = the rest
                keys = hold if split == "test" else [k for k in keys if k not in set(hold)]
            self._traces = {k: tr[k] for k in keys}
            self._names = keys''')
rep(5, '''print("environment defined | unit factor =", 8.0 if UNITS_FIX else 1.0)''',
    '''print("environment defined | unit factor =", 8.0 if UNITS_FIX else 1.0)
if MODE == "run" and NET_SOURCE == "trace" and float(TRACE_HOLDOUT_FRAC) > 0:
    _keys = sorted(k for k in load_traces(TRACE_FILE) if k.startswith(TRACE_PREFIX))
    _hold = holdout_traces(_keys, C)
    json.dump(dict(trace_file=TRACE_FILE, prefix=TRACE_PREFIX, frac=float(TRACE_HOLDOUT_FRAC), seed=int(TRACE_HOLDOUT_SEED),
                   held_out=_hold, training=[k for k in _keys if k not in set(_hold)]),
              open(RUN_DIR / "holdout_traces.json", "w"), indent=1)
    print(f"held-out test traces ({len(_hold)} of {len(_keys)}):", _hold)''')

rep(17, '''        key += "_tr" + hashlib.md5(f"{cfg.TRACE_FILE}|{cfg.TRACE_PREFIX}|{float(cfg.NET_SCALE)}".encode()).hexdigest()[:8]''',
    '''        key += "_tr" + hashlib.md5(f"{cfg.TRACE_FILE}|{cfg.TRACE_PREFIX}|{float(cfg.NET_SCALE)}".encode()).hexdigest()[:8]
        if float(getattr(cfg, "TRACE_HOLDOUT_FRAC", 0.0) or 0.0) > 0:
            key += f"_ho{float(cfg.TRACE_HOLDOUT_FRAC):g}s{int(cfg.TRACE_HOLDOUT_SEED)}"''')

rep(23, '''def run_episodes(cfg, norm, n, seed, kind, model=None, tier=None):
    env = ABREnv(cfg, norm, "eval", seed=seed);''', '''def run_episodes(cfg, norm, n, seed, kind, model=None, tier=None, split="eval"):
    env = ABREnv(cfg, norm, split, seed=seed);''')
s = get(23)
s = s.replace('run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "policy", model=model)',
              'run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "policy", model=model, split="test")')
s = s.replace('run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "tier", tier=b)',
              'run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "tier", tier=b, split="test")')
s = s.replace('run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "mpc")',
              'run_episodes(cfg, norm, cfg.GT_EPISODES, cfg.EVAL_SEED, "mpc", split="test")')
assert s.count('split="test"') == 3
put(23, s)
rep(36, '''    env = ABREnv(cfg, norm, "eval", seed=seed); mpc = MPCPolicy(cfg) if kind == "mpc" else None''',
    '''    env = ABREnv(cfg, norm, "test", seed=seed); mpc = MPCPolicy(cfg) if kind == "mpc" else None''')
rep(22, "Computed without the diagnostic signals: full evaluation episodes (seed `EVAL_SEED`, identical across policies)",
    "Computed without the diagnostic signals: full test episodes (seed `EVAL_SEED`, identical across policies; on a trace "
    "network with `TRACE_HOLDOUT_FRAC > 0` they use only the held-out traces)")
json.dump(nb, open(p, "w"), indent=1)
print("patched", p)
