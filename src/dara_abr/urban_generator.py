"""
Urban 5G mmWave Channel Generator
=================================
Regime-switching (Markov-modulated) channel simulator calibrated against
real Lumos5G measurements, extended to dense-urban scenarios that the
original dataset under-represents.

Design basis:
  - 6-state regime model (OUTAGE / DEEP_NLOS / NLOS / MIXED / LOS / STRONG_LOS)
  - Transition matrix + dwell distributions measured from Lumos5G
  - Within-regime fading: AR(1) in log10 domain, per-regime sigma from Lumos5G
  - Scenario-specific transition biasing + blockage event injection
  - SINR coupled to throughput via measured linear fit + residual noise
"""
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# CALIBRATED CONSTANTS (measured from Lumos5G-v1_0.csv)
# ─────────────────────────────────────────────────────────────────────────
REGIME_NAMES = ['OUTAGE', 'DEEP_NLOS', 'NLOS', 'MIXED', 'LOS', 'STRONG_LOS']
N_REG = 6

# log10(throughput) mean + within-regime fading std, per regime
REG_LOG_MEAN = np.array([-0.520, 1.316, 2.055, 2.631, 2.932, 3.155])
REG_LOG_STD  = np.array([ 0.656, 0.307, 0.113, 0.105, 0.081, 0.054])
REG_BOUNDS   = [(0.0,5.0),(5.0,60.0),(60.0,250.0),(250.0,600.0),(600.0,1200.0),(1200.0,1920.0)]

# Baseline transition matrix measured from Lumos5G (row = from-state)
BASE_TRANS = np.array([
    [0.5468, 0.2120, 0.2273, 0.0093, 0.0045, 0.0000],
    [0.1412, 0.5526, 0.2793, 0.0231, 0.0037, 0.0000],
    [0.0133, 0.0532, 0.8219, 0.0849, 0.0249, 0.0017],
    [0.0134, 0.0396, 0.1136, 0.6479, 0.1840, 0.0015],
    [0.0028, 0.0059, 0.0243, 0.0824, 0.8090, 0.0756],
    [0.0002, 0.0011, 0.0068, 0.0077, 0.2440, 0.7402],
])

# SINR per regime — measured directly from Lumos5G nr_ssSinr (mean, std)
# NOTE: derived empirically per-regime rather than by inverting the weak
# global SINR->throughput regression (r=0.468), which amplifies error.
REG_SINR_MEAN = np.array([ 5.05,  5.91,  8.33, 11.81, 13.37, 15.85])
REG_SINR_STD  = np.array([ 7.59,  7.94,  7.36,  5.48,  4.44,  2.82])
SINR_MIN, SINR_MAX = -60.0, 30.0
SINR_AR_PHI = 0.65   # SINR is temporally correlated, not white noise

# AR(1) memory coefficient for within-regime fading
AR_PHI = 0.72

# ─────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────
# regime_bias      : multiplicative weight on transitioning INTO each regime
# dwell_scale      : multiplies expected dwell time (>1 = more persistent regimes)
# speed_kmh        : (min, max) UE speed
# fade_scale       : multiplies within-regime fading std (mobility -> more volatile)
# blockage_events  : list of (prob_per_second, min_dur, max_dur, target_regime)
SCENARIOS = {
    'urban_pedestrian': dict(
        desc='Walking, dense CBD. Frequent human-body and street-furniture blockage.',
        regime_bias=[1.25, 1.35, 1.15, 1.0, 0.95, 0.80],
        dwell_scale=0.95, speed_kmh=(2, 6), fade_scale=0.60,
        blockage_events=[(0.012, 1, 4, 0), (0.022, 2, 7, 1)],
    ),
    'urban_driving': dict(
        desc='City roads, traffic, building canyons, frequent beam re-acquisition.',
        regime_bias=[1.20, 1.40, 1.25, 1.05, 0.90, 0.70],
        dwell_scale=0.80, speed_kmh=(15, 50), fade_scale=0.72,
        blockage_events=[(0.010, 1, 3, 0), (0.028, 2, 8, 1)],
    ),
    'urban_canyon': dict(
        desc='Narrow street between tall buildings. Sustained NLOS, rare LOS.',
        regime_bias=[1.15, 1.65, 1.50, 1.00, 0.65, 0.40],
        dwell_scale=1.25, speed_kmh=(3, 30), fade_scale=0.65,
        blockage_events=[(0.010, 2, 5, 0), (0.035, 4, 16, 1)],
    ),
    'motorway': dict(
        desc='High-speed motorway. Fast cell transitions, Doppler, sparse gNB.',
        regime_bias=[1.10, 1.30, 1.20, 1.10, 1.00, 0.75],
        dwell_scale=0.55, speed_kmh=(90, 130), fade_scale=0.95,
        blockage_events=[(0.012, 1, 4, 0), (0.024, 2, 8, 1)],
    ),
    'tunnel': dict(
        desc='Road/rail tunnel. Deep sustained outage with weak leaky-feeder coverage.',
        regime_bias=[2.60, 2.10, 0.70, 0.30, 0.15, 0.05],
        dwell_scale=2.0, speed_kmh=(40, 90), fade_scale=0.55,
        blockage_events=[(0.030, 8, 40, 0), (0.035, 10, 35, 1)],
    ),
    'under_bridge': dict(
        desc='Passing beneath bridges/overpasses. Short sharp total blockage.',
        regime_bias=[1.35, 1.25, 1.10, 1.00, 0.95, 0.75],
        dwell_scale=0.85, speed_kmh=(30, 80), fade_scale=0.75,
        blockage_events=[(0.020, 2, 6, 0), (0.018, 2, 5, 1)],
    ),
    'crowded_stationary': dict(
        desc='Standing in crowd (station, event). Dense human-body blockage.',
        regime_bias=[1.25, 1.45, 1.20, 1.00, 0.90, 0.65],
        dwell_scale=1.05, speed_kmh=(0, 1), fade_scale=0.58,
        blockage_events=[(0.018, 1, 3, 0), (0.030, 2, 6, 1)],
    ),
    'suburban_open': dict(
        desc='Open suburban road, good LOS. Benign reference scenario.',
        regime_bias=[0.35, 0.45, 0.85, 1.25, 1.85, 1.80],
        dwell_scale=1.45, speed_kmh=(20, 60), fade_scale=0.60,
        blockage_events=[(0.003, 1, 2, 0), (0.006, 2, 4, 1)],
    ),
}

# Relative sampling weight of each scenario in the final dataset
SCENARIO_MIX = {
    'urban_pedestrian'  : 0.19,
    'urban_driving'     : 0.19,
    'urban_canyon'      : 0.11,
    'motorway'          : 0.12,
    'tunnel'            : 0.05,
    'under_bridge'      : 0.08,
    'crowded_stationary': 0.08,
    'suburban_open'     : 0.18,
}


def _build_transition(scn, rng):
    """Apply scenario regime-bias and dwell-scaling to the base matrix."""
    bias = np.array(scn['regime_bias'], dtype=float)
    T = BASE_TRANS * bias[None, :]           # bias transitions INTO each regime

    # dwell_scale > 1 -> strengthen self-transition (more persistent regimes)
    ds = scn['dwell_scale']
    for i in range(N_REG):
        T[i, i] *= ds
    T = T / T.sum(axis=1, keepdims=True)
    return T


def generate_session(scenario, n_seconds, rng):
    """
    Generate one synthetic session.
    Returns dict of arrays: throughput (Mbps), sinr (dB), regime (int), speed (km/h)
    """
    scn = SCENARIOS[scenario]
    T   = _build_transition(scn, rng)

    # ── UE speed: base speed + slow random walk (traffic, stops, acceleration)
    v_lo, v_hi = scn['speed_kmh']
    base_speed = rng.uniform(v_lo, v_hi)
    speed = np.empty(n_seconds)
    sp = base_speed
    for t in range(n_seconds):
        sp += rng.normal(0, max(0.5, base_speed * 0.05))
        sp = float(np.clip(sp, max(0.0, v_lo * 0.4), v_hi * 1.25))
        speed[t] = sp

    # ── regime sequence via Markov chain + injected blockage events
    regime = np.empty(n_seconds, dtype=int)
    # start weighted by scenario bias
    start_w = np.array(scn['regime_bias'], dtype=float)
    start_w = start_w / start_w.sum()
    cur = int(rng.choice(N_REG, p=start_w))

    forced = np.full(n_seconds, -1, dtype=int)   # forced blockage overlay
    for (prob, dmin, dmax, target) in scn['blockage_events']:
        t = 0
        while t < n_seconds:
            if rng.random() < prob:
                dur = int(rng.integers(dmin, dmax + 1))
                # speed shortens blockage duration (drive through it faster)
                spd_factor = np.clip(60.0 / max(speed[t], 1.0), 0.35, 2.2)
                dur = max(1, int(round(dur * spd_factor)))
                end = min(n_seconds, t + dur)
                forced[t:end] = target
                t = end
            else:
                t += 1

    for t in range(n_seconds):
        if forced[t] >= 0:
            cur = forced[t]
        else:
            cur = int(rng.choice(N_REG, p=T[cur]))
        regime[t] = cur

    # ── within-regime fading: AR(1) in log10 domain
    fade_scale = scn['fade_scale']
    log_tp = np.empty(n_seconds)
    resid  = 0.0
    for t in range(n_seconds):
        r = regime[t]
        sigma = REG_LOG_STD[r] * fade_scale
        # extra volatility from speed (Doppler / fast beam changes)
        sigma *= (1.0 + 0.004 * speed[t])
        resid = AR_PHI * resid + rng.normal(0, sigma * np.sqrt(1 - AR_PHI**2))
        log_tp[t] = REG_LOG_MEAN[r] + resid

    tp = np.power(10.0, log_tp)
    # clamp into regime bounds (soft) then global bounds
    for t in range(n_seconds):
        lo, hi = REG_BOUNDS[regime[t]]
        tp[t] = np.clip(tp[t], lo * 0.75, hi * 1.15)
    tp = np.clip(tp, 0.0, 1920.0)

    # ── SINR: per-regime empirical mean/std, AR(1)-correlated over time
    sinr = np.empty(n_seconds)
    s_resid = 0.0
    for t in range(n_seconds):
        r = regime[t]
        sd = REG_SINR_STD[r]
        s_resid = SINR_AR_PHI * s_resid + rng.normal(0, sd * np.sqrt(1 - SINR_AR_PHI**2))
        sinr[t] = REG_SINR_MEAN[r] + s_resid
    sinr = np.clip(sinr, SINR_MIN, SINR_MAX)

    return dict(throughput=tp, sinr=sinr, regime=regime, speed=speed)
