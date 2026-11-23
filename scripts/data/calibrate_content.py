#!/usr/bin/env python3
"""
puffer_content_fit.py -- calibrate the DARA-ABR content-complexity chunk-size generator
from Puffer `video_size` CSV files (columns: time (ns GMT), format, channel, video_ts, size).

Decomposition fitted (per source chunk i of a channel, encoded in format f):

    size_f(i) = base_f  x  kappa_f(title(i))  x  m_f(i)

    base_f   : mean chunk size of format f over the whole day (the fixed ladder)
    kappa_f  : title complexity = mean size of the B-minute wall-clock block (programme proxy)
               divided by base_f
    m_f(i)   : within-title multiplier = size / block mean  (scene / motion complexity)

Chunks are joined in groups of --join (default 2 x 2.002 s = 4.004 s, the env chunk length).
A Gaussian HMM is fitted to log m of the reference format; Viterbi paths give the regime
dwell-time distributions (semi-Markov), the jump matrix, and within-regime AR(1) correlation.
Cross-format coupling is summarised by the slope gamma_f in  log m_f = gamma_f * log m_ref.

The second file (--test) is held out: the fitted model is scored on it, and the generator
(semi-Markov regimes + AR(1) within regime) is simulated and compared with fit and test days.

Usage (csh-safe):
  bash -c 'python3 puffer_content_fit.py --fit DAY1.csv --test DAY2.csv --out content_model --bic 3,4,5,6,7,8'
"""
import argparse
import json
import os
import sys
import time as _time

import numpy as np
import pandas as pd

TICKS_PER_S = 90000.0
NAMES = ["time", "format", "channel", "video_ts", "size"]


# ----------------------------------------------------------------------------- loading
def load(paths, chunksize=5_000_000, channels=None):
    parts = []
    for p in paths:
        n = 0
        for ch in pd.read_csv(p, header=0, names=NAMES, chunksize=chunksize,
                              dtype={"time": "int64", "format": "category", "channel": "category",
                                     "video_ts": "int64", "size": "int64"}):
            if channels:
                ch = ch[ch["channel"].isin(channels)]
            parts.append(ch)
            n += len(ch)
        print(f"  read {p}: {n:,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    for c in ("format", "channel"):
        df[c] = df[c].astype(str).astype("category")
    n0 = len(df)
    df = df[df["size"] > 0]
    df = df.drop_duplicates(subset=["channel", "format", "video_ts"], keep="first")
    print(f"  kept {len(df):,} rows after size>0 and de-duplication (dropped {n0 - len(df):,})", flush=True)
    return df.reset_index(drop=True)


def chunk_step(df):
    """Most common positive video_ts increment (90 kHz ticks)."""
    d = df.sort_values(["channel", "format", "video_ts"])
    diffs = d.groupby(["channel", "format"], observed=True)["video_ts"].diff().dropna()
    diffs = diffs[diffs > 0].astype(np.int64)
    step = int(diffs.mode().iloc[0])
    share = float((diffs == step).mean())
    print(f"  chunk step = {step} ticks = {step / TICKS_PER_S:.4f} s ({100 * share:.1f}% of increments)", flush=True)
    return step


# ----------------------------------------------------------------------------- decomposition
def build_chunks(df, step, join, block_min, min_block_frac=0.5):
    """Returns (chunks4, blocks, fmt_stats). chunks4 has one row per joined chunk."""
    df = df.copy()
    # a channel restart can shift the timestamp grid: treat each (channel, video_ts mod step) as its own stream
    df["off"] = (df["video_ts"] % step).astype(np.int64)
    cnt = df.groupby(["channel", "off"], observed=True)["size"].transform("size")
    small = cnt < 200
    if small.any():
        print(f"  dropped {int(small.sum()):,} rows in tiny off-grid fragments", flush=True)
    df = df[~small].copy()
    noffs = df.groupby("channel", observed=True)["off"].transform("nunique")
    ch = df["channel"].astype(str)
    df["channel"] = np.where(noffs > 1, ch + "@" + df["off"].astype(str), ch)
    if (noffs > 1).any():
        print(f"  {int(df.loc[noffs > 1, 'channel'].nunique())} restart segments split into separate streams", flush=True)
    df["idx"] = (df["video_ts"] // step).astype(np.int64)
    block_ns = int(block_min * 60 * 1e9)
    df["block"] = (df["time"] // block_ns).astype(np.int64)

    expected = block_min * 60.0 / (step / TICKS_PER_S)
    g = df.groupby(["channel", "format", "block"], observed=True)["size"]
    blocks = g.agg(["mean", "count"]).reset_index().rename(columns={"mean": "block_mean", "count": "n_src"})
    blocks = blocks[blocks["n_src"] >= min_block_frac * expected]
    df = df.merge(blocks[["channel", "format", "block", "block_mean"]], on=["channel", "format", "block"], how="inner")

    fmt_mean = df.groupby("format", observed=True)["size"].mean()
    blocks["base"] = blocks["format"].map(fmt_mean).astype(float)
    blocks["kappa"] = blocks["block_mean"] / blocks["base"]

    df["pid"] = df["idx"] // join
    agg = df.groupby(["channel", "format", "pid"], observed=True).agg(
        size=("size", "sum"), bm=("block_mean", "sum"), n=("size", "size"),
        block=("block", "first"), time=("time", "first")).reset_index()
    agg = agg[agg["n"] == join].drop(columns="n")
    agg["m"] = agg["size"] / agg["bm"]
    agg["logm"] = np.log(agg["m"])
    agg = agg.sort_values(["channel", "format", "pid"]).reset_index(drop=True)
    for c in ("channel", "format"):
        agg[c] = agg[c].astype(str)
        blocks[c] = blocks[c].astype(str)
    blocks = blocks.reset_index(drop=True)

    dur = join * step / TICKS_PER_S
    fs = df.groupby("format", observed=True)["size"].agg(["mean", "count"])
    fs.index = fs.index.astype(str)
    fs["mbps"] = fs["mean"] * 8 / (step / TICKS_PER_S) / 1e6
    fs = fs.sort_values("mbps")
    print(f"  {len(agg):,} joined chunks of {dur:.3f} s; {len(blocks):,} valid {block_min}-min blocks", flush=True)
    return agg, blocks, fs, dur


def sequences(agg, fmt, min_run):
    """Gap-free runs of joined chunks for one format: list of (channel, pid array, logm array)."""
    out = []
    sub = agg[agg["format"] == fmt]
    for chn, s in sub.groupby("channel", observed=True):
        pid = s["pid"].to_numpy()
        x = s["logm"].to_numpy()
        cuts = np.flatnonzero(np.diff(pid) != 1) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(pid)]):
            if b - a >= min_run:
                out.append((str(chn), pid[a:b], x[a:b]))
    return out


# ----------------------------------------------------------------------------- Gaussian HMM (padded, scaled)
def pad(seqs):
    L = max(len(x) for x in seqs)
    X = np.zeros((len(seqs), L))
    M = np.zeros((len(seqs), L), dtype=bool)
    for i, x in enumerate(seqs):
        X[i, :len(x)] = x
        M[i, :len(x)] = True
    return X, M


def log_emis(X, mu, sd):
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * ((X[..., None] - mu) / sd) ** 2


def forward_backward(X, M, pi, A, mu, sd, want_post=True):
    S, L = X.shape
    K = len(mu)
    lB = log_emis(X, mu, sd)
    mx = lB.max(axis=2, keepdims=True)
    B = np.exp(lB - mx)
    loglik = float(np.sum(mx[..., 0] * M))
    alpha = np.zeros((S, L, K))
    c = np.ones((S, L))
    a = pi * B[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, :1]
    for t in range(1, L):
        a = (alpha[:, t - 1] @ A) * B[:, t]
        ct = a.sum(1)
        m = M[:, t]
        c[m, t] = ct[m]
        alpha[:, t] = np.where(m[:, None], a / np.where(m, ct, 1.0)[:, None], alpha[:, t - 1])
    loglik += float(np.sum(np.log(c) * M))
    if not want_post:
        return loglik, None, None
    beta = np.ones((S, L, K))
    xi = np.zeros((K, K))
    for t in range(L - 2, -1, -1):
        m1 = M[:, t + 1]
        w = B[:, t + 1] * beta[:, t + 1] / c[:, t + 1:t + 2]
        b = w @ A.T
        beta[:, t] = np.where(m1[:, None], b, 1.0)
        if m1.any():
            xi += A * (alpha[m1, t].T @ w[m1])
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    gamma *= M[..., None]
    return loglik, gamma, xi


def segment(seqs, seg):
    """Cut long sequences into pieces of ~seg chunks so the EM loop runs over seg steps, not the
    full sequence length. Only the transitions at the cuts are lost (about 1/seg of them)."""
    out = []
    for x in seqs:
        n = len(x)
        k = max(1, int(round(n / seg)))
        for piece in np.array_split(x, k):
            if len(piece) >= 2:
                out.append(piece)
    return out


def fit_hmm(seqs, K, iters=200, tol=1e-5, seed=0, seg=500, verbose=True):
    t0 = _time.time()
    X, M = pad(segment(seqs, seg) if seg else seqs)
    allx = X[M]
    qs = np.quantile(allx, (np.arange(K) + 0.5) / K)
    mu = qs.copy()
    sd = np.full(K, max(np.std(allx) / K, 0.02))
    A = np.full((K, K), 0.1 / max(K - 1, 1))
    np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    rng = np.random.default_rng(seed)
    mu += rng.normal(0, 1e-3, K)
    prev = -np.inf
    sd_floor = 0.01
    nobs = M.sum()
    for it in range(iters):
        ll, gamma, xi = forward_backward(X, M, pi, A, mu, sd)
        pi = gamma[:, 0].sum(0) + 1e-6
        pi /= pi.sum()
        A = xi + 1e-6
        A /= A.sum(1, keepdims=True)
        w = gamma.sum((0, 1)) + 1e-12
        mu = (gamma * X[..., None]).sum((0, 1)) / w
        sd = np.sqrt((gamma * (X[..., None] - mu) ** 2).sum((0, 1)) / w)
        sd = np.maximum(sd, sd_floor)
        if verbose and (it % 10 == 0):
            print(f"    K={K} iter {it:3d}  loglik/chunk {ll / nobs:.5f}  ({_time.time() - t0:.0f} s)", flush=True)
        if it > 5 and (ll - prev) / nobs < tol:
            break
        prev = ll
    o = np.argsort(mu)
    mu, sd, pi, A = mu[o], sd[o], pi[o], A[np.ix_(o, o)]
    # exact log-likelihood on the uncut sequences
    Xf, Mf = pad(seqs)
    ll, _, _ = forward_backward(Xf, Mf, pi, A, mu, sd, want_post=False)
    n = int(Mf.sum())
    p = (K - 1) + K * (K - 1) + 2 * K
    if verbose:
        print(f"    K={K} converged after {it + 1} iterations in {_time.time() - t0:.0f} s", flush=True)
    return dict(K=K, pi=pi, A=A, mu=mu, sd=sd, loglik=ll, n=n, n_params=p,
                bic=-2 * ll + p * np.log(n), iters=it + 1)


def viterbi(x, hmm):
    lA = np.log(hmm["A"])
    lB = log_emis(np.asarray(x), hmm["mu"], hmm["sd"])
    L, K = lB.shape
    d = np.log(hmm["pi"]) + lB[0]
    bp = np.zeros((L, K), dtype=np.int16)
    for t in range(1, L):
        s = d[:, None] + lA
        bp[t] = s.argmax(0)
        d = s.max(0) + lB[t]
    z = np.zeros(L, dtype=np.int64)
    z[-1] = int(d.argmax())
    for t in range(L - 1, 0, -1):
        z[t - 1] = bp[t, z[t]]
    return z


def runs_of(z):
    cuts = np.flatnonzero(np.diff(z) != 0) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(z)]
    return [(int(z[a]), int(b - a), a == 0, b == len(z)) for a, b in zip(starts, ends)]


# ----------------------------------------------------------------------------- statistics helpers
def acf(x_list, lags):
    out = []
    for k in lags:
        a, b = [], []
        for x in x_list:
            if len(x) > k:
                a.append(x[:-k] - x.mean())
                b.append(x[k:] - x.mean())
        a, b = np.concatenate(a), np.concatenate(b)
        out.append(float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))))
    return out


def regime_tables(seqs, hmm):
    K = hmm["K"]
    paths = [viterbi(x, hmm) for x in seqs]
    dwell = {k: [] for k in range(K)}
    jumps = np.zeros((K, K))
    pairs = {k: ([], []) for k in range(K)}
    for x, z in zip(seqs, paths):
        rs = runs_of(z)
        for (k, n, first, last) in rs:
            if not first and not last:
                dwell[k].append(n)
        ch = np.flatnonzero(np.diff(z) != 0)
        for t in ch:
            jumps[z[t], z[t + 1]] += 1
        same = np.flatnonzero(np.diff(z) == 0)
        for t in same:
            k = z[t]
            pairs[k][0].append(x[t] - hmm["mu"][k])
            pairs[k][1].append(x[t + 1] - hmm["mu"][k])
    rho = np.zeros(K)
    for k in range(K):
        a, b = np.asarray(pairs[k][0]), np.asarray(pairs[k][1])
        rho[k] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else 0.0
    J = jumps.copy()
    rowsum = J.sum(1, keepdims=True)
    J = np.where(rowsum > 0, J / np.maximum(rowsum, 1), (1 - np.eye(K)) / max(K - 1, 1))
    zall = np.concatenate(paths)
    share = np.bincount(zall, minlength=K) / len(zall)
    return dict(paths=paths, dwell=dwell, jump=J, jump_counts=jumps, rho=rho, share=share)


def sample_generator(model, lengths, rng):
    """Semi-Markov regimes (empirical dwell pmf) + AR(1) log-multiplier within regime."""
    K = len(model["mu"])
    mu, sd, rho = map(np.asarray, (model["mu"], model["sd"], model["rho"]))
    share = np.asarray(model["share"])
    J = np.asarray(model["jump"])
    pmfs = [np.asarray(p) for p in model["dwell_pmf"]]
    out = []
    for L in lengths:
        x = np.empty(L)
        k = rng.choice(K, p=share)
        t = 0
        first = True
        while t < L:
            d = 1 + rng.choice(len(pmfs[k]), p=pmfs[k])
            if first:  # stationary start: residual life of the first regime
                d = 1 + rng.integers(d)
                first = False
            e = mu[k] + sd[k] * rng.standard_normal()
            for j in range(min(d, L - t)):
                if j > 0:
                    e = mu[k] + rho[k] * (e - mu[k]) + sd[k] * np.sqrt(1 - rho[k] ** 2) * rng.standard_normal()
                x[t] = e
                t += 1
            k = rng.choice(K, p=J[k])
        out.append(x)
    return out


def summary_stats(xs, hmm, label):
    allx = np.concatenate(xs)
    m = np.exp(allx)
    q = np.quantile(m, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rt = regime_tables(xs, hmm)
    dw = [n for k in rt["dwell"] for n in rt["dwell"][k]]
    row = dict(source=label, n_chunks=int(len(allx)), sd_logm=float(allx.std()),
               cv_m=float(m.std() / m.mean()),
               **{f"m_q{int(p * 100):02d}": float(v) for p, v in zip([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], q)},
               **{f"acf{k}": v for k, v in zip([1, 2, 5, 10], acf(xs, [1, 2, 5, 10]))},
               mean_dwell_chunks=float(np.mean(dw)) if dw else float("nan"))
    for k in range(hmm["K"]):
        row[f"share_s{k}"] = float(rt["share"][k])
    return row, rt


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True, help="Puffer video_size CSV(s) used for fitting")
    ap.add_argument("--test", nargs="+", default=None, help="held-out Puffer video_size CSV(s)")
    ap.add_argument("--out", default="content_model")
    ap.add_argument("--states", default="6", help="number of regimes, or 'bic' to take the BIC minimum")
    ap.add_argument("--bic", default="", help="comma list of K to compare, e.g. 3,4,5,6,7,8")
    ap.add_argument("--ref_format", default="auto", help="reference format; auto = median-bitrate format")
    ap.add_argument("--join", type=int, default=2, help="source chunks per env chunk (2 x 2.002 s = 4.004 s)")
    ap.add_argument("--block_min", type=float, default=30.0, help="title-proxy block length in wall-clock minutes")
    ap.add_argument("--min_run", type=int, default=30, help="minimum gap-free run length (joined chunks)")
    ap.add_argument("--channels", default="", help="optional comma list of channels to keep")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--em_seg", type=int, default=500, help="EM segment length in chunks (speed); 0 = no cutting")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    chans = [c for c in a.channels.split(",") if c] or None
    t0 = _time.time()

    print("[1] fit day", flush=True)
    dfF = load(a.fit, channels=chans)
    step = chunk_step(dfF)
    aggF, blkF, fsF, dur = build_chunks(dfF, step, a.join, a.block_min)
    del dfF
    fmts = list(fsF.index)
    ref = a.ref_format if a.ref_format != "auto" else fmts[len(fmts) // 2]
    if ref not in fmts:
        sys.exit(f"reference format {ref} not in {fmts}")
    print(f"  formats by bitrate: {fmts}\n  reference format: {ref}", flush=True)
    seqF = sequences(aggF, ref, a.min_run)
    xsF = [s[2] for s in seqF]
    print(f"  {len(xsF)} fit sequences, {sum(len(x) for x in xsF):,} chunks", flush=True)

    # --- model order
    Ks = [int(k) for k in a.bic.split(",") if k]
    bic_rows, fits = [], {}
    if a.states == "bic" and not Ks:
        Ks = [3, 4, 5, 6, 7, 8]
    for K in Ks:
        h = fit_hmm(xsF, K, iters=a.iters, seed=a.seed, seg=a.em_seg)
        fits[K] = h
        bic_rows.append(dict(K=K, loglik=h["loglik"], loglik_per_chunk=h["loglik"] / h["n"],
                             n_params=h["n_params"], bic=h["bic"], iters=h["iters"]))
        print(f"  K={K}: loglik/chunk {h['loglik'] / h['n']:.4f}  BIC {h['bic']:.1f}", flush=True)
    K = int(min(bic_rows, key=lambda r: r["bic"])["K"]) if a.states == "bic" else int(a.states)

    print(f"[2] fitting HMM K={K}", flush=True)
    hmm = fits.get(K) or fit_hmm(xsF, K, iters=a.iters, seed=a.seed, seg=a.em_seg)
    rtF = regime_tables(xsF, hmm)

    maxd = max([max(v) for v in rtF["dwell"].values() if v] + [1])
    dwell_pmf, regimes = [], []
    for k in range(K):
        d = np.asarray(rtF["dwell"][k], dtype=int)
        if len(d) >= 5:
            pmf = np.bincount(d - 1, minlength=maxd).astype(float)
        else:  # too few complete runs: geometric from the HMM self-transition
            p = hmm["A"][k, k]
            pmf = (1 - p) * p ** np.arange(maxd)
        pmf = pmf / pmf.sum()
        dwell_pmf.append(pmf)
        allxk = np.concatenate([x[z == k] for x, z in zip(xsF, rtF["paths"])])
        regimes.append(dict(state=k, share=float(rtF["share"][k]), mu_logm=float(hmm["mu"][k]),
                            sd_logm=float(hmm["sd"][k]), m_median=float(np.exp(np.median(allxk))) if len(allxk) else np.nan,
                            stay_prob_hmm=float(hmm["A"][k, k]), resid_acf1=float(rtF["rho"][k]),
                            n_complete_runs=int(len(d)),
                            dwell_mean_chunks=float(d.mean()) if len(d) else np.nan,
                            dwell_median_chunks=float(np.median(d)) if len(d) else np.nan,
                            dwell_p90_chunks=float(np.quantile(d, 0.9)) if len(d) else np.nan,
                            dwell_max_chunks=int(d.max()) if len(d) else 0,
                            dwell_mean_s=float(d.mean() * dur) if len(d) else np.nan))
    pd.DataFrame(regimes).to_csv(f"{a.out}/regimes.csv", index=False)
    pd.DataFrame(hmm["A"]).to_csv(f"{a.out}/transition_matrix.csv", index_label="from")
    pd.DataFrame(rtF["jump"]).to_csv(f"{a.out}/jump_matrix.csv", index_label="from")
    rl = [dict(state=k, length_chunks=n) for k in range(K) for n in rtF["dwell"][k]]
    pd.DataFrame(rl).to_csv(f"{a.out}/run_lengths.csv", index=False)
    if bic_rows:
        pd.DataFrame(bic_rows).to_csv(f"{a.out}/bic.csv", index=False)

    print("[3] cross-format coupling, title complexity, caps", flush=True)
    wide = aggF.set_index(["channel", "pid", "format"])["logm"].unstack("format")
    cf_rows = []
    for f in fmts:
        ok = wide[ref].notna() & wide[f].notna()
        xr, xf = wide.loc[ok, ref].to_numpy(), wide.loc[ok, f].to_numpy()
        slope = float(np.sum(xr * xf) / np.sum(xr * xr)) if len(xr) else np.nan
        r = float(np.corrcoef(xr, xf)[0, 1]) if len(xr) > 2 else np.nan
        mf = np.exp(aggF.loc[aggF["format"] == f, "logm"].to_numpy())
        cf_rows.append(dict(format=f, mean_mbps=float(fsF.loc[f, "mbps"]), n_joined=int(len(mf)),
                            corr_logm_with_ref=r, gamma_slope_vs_ref=slope, r2=r ** 2 if r == r else np.nan,
                            sd_logm=float(np.log(mf).std()),
                            sd_ratio_to_ref=float(xf.std() / xr.std()) if len(xr) else np.nan,
                            cv_m=float(mf.std() / mf.mean()),
                            m_p95=float(np.quantile(mf, 0.95)), m_p99=float(np.quantile(mf, 0.99)),
                            m_max=float(mf.max())))
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(f"{a.out}/cross_format.csv", index=False)
    cf[["format", "mean_mbps", "m_p95", "m_p99", "m_max"]].to_csv(f"{a.out}/caps.csv", index=False)

    blkF["logk"] = np.log(blkF["kappa"])
    kref = blkF[blkF["format"] == ref].sort_values(["channel", "block"])
    kw = blkF.set_index(["channel", "block", "format"])["logk"].unstack("format")
    kcf = []
    for f in fmts:
        ok = kw[ref].notna() & kw[f].notna()
        if ok.sum() > 2:
            xr, xf = kw.loc[ok, ref].to_numpy(), kw.loc[ok, f].to_numpy()
            kcf.append(dict(format=f, corr_logk_with_ref=float(np.corrcoef(xr, xf)[0, 1]),
                            slope_vs_ref=float(np.polyfit(xr, xf, 1)[0]), n_blocks=int(ok.sum())))
    pd.DataFrame(kcf).to_csv(f"{a.out}/kappa_cross_format.csv", index=False)
    lag = []
    for _, s in kref.groupby("channel", observed=True):
        b, v = s["block"].to_numpy(), s["logk"].to_numpy()
        ok = np.flatnonzero(np.diff(b) == 1)
        lag += [(v[i], v[i + 1]) for i in ok]
    lag = np.asarray(lag)
    k_ac1 = float(np.corrcoef(lag[:, 0], lag[:, 1])[0, 1]) if len(lag) > 2 else np.nan
    kc = kref.groupby("channel", observed=True)["kappa"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    kc.to_csv(f"{a.out}/kappa_by_channel.csv", index=False)
    kref[["channel", "block", "block_mean", "n_src", "kappa"]].to_csv(f"{a.out}/kappa_blocks_ref.csv", index=False)
    print(f"  kappa ({ref}): {len(kref)} blocks, sd(log kappa) {kref['logk'].std():.3f}, "
          f"block-to-block corr {k_ac1:.3f}", flush=True)

    model = dict(
        source=dict(fit_files=a.fit, test_files=a.test, ref_format=ref, join=a.join,
                    chunk_seconds=dur, source_chunk_seconds=step / TICKS_PER_S, block_minutes=a.block_min,
                    channels=sorted(aggF["channel"].astype(str).unique().tolist())),
        regimes=dict(K=K, mu=hmm["mu"].tolist(), sd=hmm["sd"].tolist(), rho=rtF["rho"].tolist(),
                     share=rtF["share"].tolist(), A_hmm=hmm["A"].tolist(), jump=rtF["jump"].tolist(),
                     dwell_pmf=[p.tolist() for p in dwell_pmf],
                     dwell_note="dwell_pmf[k][j] = P(regime k lasts j+1 joined chunks); censored runs excluded"),
        kappa=dict(format=ref, n_blocks=int(len(kref)), sd_log=float(kref["logk"].std()),
                   block_lag1_corr=k_ac1,
                   log_quantiles=np.quantile(kref["logk"], np.linspace(0.01, 0.99, 99)).tolist()),
        formats=cf.to_dict(orient="records"),
        hmm_fit=dict(loglik=hmm["loglik"], n=hmm["n"], bic=hmm["bic"], iters=hmm["iters"]),
    )

    print("[4] validation", flush=True)
    lengths = [len(x) for x in xsF]
    val, xsT, rtT, seqT = [], None, None, None
    rowF, _ = summary_stats(xsF, hmm, "fit_day")
    rowF["loglik_per_chunk"] = hmm["loglik"] / hmm["n"]
    val.append(rowF)
    if a.test:
        dfT = load(a.test, channels=chans)
        aggT, blkT, fsT, _ = build_chunks(dfT, step, a.join, a.block_min)
        del dfT
        seqT = sequences(aggT, ref, a.min_run)
        xsT = [s[2] for s in seqT]
        if xsT:
            XT, MT = pad(xsT)
            llT, _, _ = forward_backward(XT, MT, hmm["pi"], hmm["A"], hmm["mu"], hmm["sd"], want_post=False)
            rowT, rtT = summary_stats(xsT, hmm, "test_day_heldout")
            rowT["loglik_per_chunk"] = llT / MT.sum()
            val.append(rowT)
            lengths = [len(x) for x in xsT]
            np.savez_compressed(f"{a.out}/test_traces.npz",
                                **{f"{c}_{i}": np.exp(x) for i, (c, _, x) in enumerate(seqT)})
            model["test_day"] = dict(n_sequences=len(xsT), n_chunks=int(sum(lengths)),
                                     loglik_per_chunk=float(llT / MT.sum()),
                                     kappa_sd_log=float(np.log(blkT.loc[blkT["format"] == ref, "kappa"]).std()))
        else:
            print("  no usable test sequences for the reference format", flush=True)
    xsG = sample_generator(model["regimes"], lengths, rng)
    XG, MG = pad(xsG)
    llG, _, _ = forward_backward(XG, MG, hmm["pi"], hmm["A"], hmm["mu"], hmm["sd"], want_post=False)
    rowG, rtG = summary_stats(xsG, hmm, "generator_sample")
    rowG["loglik_per_chunk"] = llG / MG.sum()
    val.append(rowG)
    vdf = pd.DataFrame(val)
    vdf.to_csv(f"{a.out}/validation.csv", index=False)
    model["validation"] = val
    with open(f"{a.out}/content_model.json", "w") as fh:
        json.dump(model, fh, indent=1, default=float)

    print("[5] figures", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        cols = plt.cm.viridis(np.linspace(0, 0.9, K))

        i = int(np.argmax(lengths if xsT is None else [len(x) for x in xsF]))
        x, z = xsF[i], rtF["paths"][i]
        n = min(len(x), 450)
        tt = np.arange(n) * dur / 60
        fig, ax = plt.subplots(figsize=(7, 2.4))
        for (k, ln, _, _), st in zip(runs_of(z[:n]), np.r_[0, np.flatnonzero(np.diff(z[:n]) != 0) + 1]):
            ax.axvspan(tt[st], tt[min(st + ln, n - 1)], color=cols[k], alpha=0.25, lw=0)
        ax.plot(tt, np.exp(x[:n]), color="k", lw=0.7)
        ax.set_xlabel("Video time (min)")
        ax.set_ylabel(r"Chunk-size multiplier $m$")
        fig.tight_layout()
        fig.savefig(f"{a.out}/example_sequence.png", dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(1, K, figsize=(1.5 * K, 2.4), sharey=True)
        for k in range(K):
            ax = axs[k] if K > 1 else axs
            bins = np.arange(1, 32)
            for rt, lab, ls in [(rtF, "fit", "-"), (rtT, "test", "--"), (rtG, "generator", ":")]:
                if rt is None or not rt["dwell"][k]:
                    continue
                h, _ = np.histogram(np.minimum(rt["dwell"][k], 30), bins=bins, density=True)
                ax.step(bins[:-1], h, where="post", ls=ls, lw=1, label=lab)
            ax.set_xlabel(f"regime {k}")
            if k == 0:
                ax.set_ylabel("Probability")
        fig.supxlabel("Regime dwell time (chunks; last bin = 30 or more)", fontsize=9)
        (axs[-1] if K > 1 else axs).legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{a.out}/regime_durations.png", dpi=200)
        plt.close(fig)

        lags = list(range(1, 21))
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for xs, lab, ls in [(xsF, "fit day", "-"), (xsT, "test day", "--"), (xsG, "generator", ":")]:
            if xs:
                ax.plot(lags, acf(xs, lags), ls=ls, label=lab)
        ax.set_xlabel("Lag (chunks)")
        ax.set_ylabel(r"ACF of $\log m$")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{a.out}/acf_validation.png", dpi=200)
        plt.close(fig)
    except Exception as e:  # figures are optional
        print(f"  figure step skipped: {e}", flush=True)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print("\nregimes:\n", pd.DataFrame(regimes).to_string(index=False))
        print("\ncross-format:\n", cf.to_string(index=False))
        print("\nvalidation:\n", vdf.to_string(index=False))
    print(f"\ndone in {(_time.time() - t0) / 60:.1f} min -> {a.out}/", flush=True)


if __name__ == "__main__":
    main()
