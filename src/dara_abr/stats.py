"""Statistics used in the paper.

Every method is evaluated on the same test sessions, so comparisons are paired by session. Three views are
reported for each comparison of method A against method B:

  session level : Wilcoxon signed-rank test on the per-session score, averaged over each method's seeds
                  (are these trained models better on typical sessions?)
  seed level    : Welch t-test on the per-seed mean scores (does the method reliably produce better models?)
  hierarchical  : 95% bootstrap interval of the mean difference, resampling seeds within each method and the
                  shared sessions together (both sources of variation at once)

Benjamini-Hochberg q-values are computed within a family of comparisons. A claim of "better" or "worse"
requires q < alpha at the session level AND a bootstrap interval that excludes zero (see the pre-registration).
"""
import numpy as np
from scipy.stats import wilcoxon, ttest_ind


def bh(p):
    p = np.asarray(p, float)
    k = len(p)
    if k == 0:
        return p
    o = np.argsort(p)
    q = np.empty(k)
    q[o] = np.minimum.accumulate((p[o] * k / np.arange(1, k + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def session_test(x, y):
    """x, y: per-session scores (seed-averaged), same sessions in the same order."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0      # identical arrays: no difference, p = 1
    return dict(diff=float(d.mean()), diff_median=float(np.median(d)), share_sessions_better=float(np.mean(d > 0)),
                p_session=p)


def seed_test(a, b):
    """a, b: per-seed mean scores. Welch t-test; NaN when either side has fewer than two seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(p_seed=np.nan)
    return dict(p_seed=float(ttest_ind(a, b, equal_var=False).pvalue))


def hierarchical_ci(A, B, n_boot=10000, level=0.95, seed=0):
    """A: (seeds_A, sessions), B: (seeds_B, sessions); a reference policy has a single row.
    Returns the bootstrap interval of mean(A) - mean(B)."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    n = min(A.shape[1], B.shape[1])
    A, B = A[:, :n], B[:, :n]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    step = 1000
    for i in range(0, n_boot, step):
        m = min(step, n_boot - i)
        sa = rng.integers(0, A.shape[0], size=(m, A.shape[0]))
        sb = rng.integers(0, B.shape[0], size=(m, B.shape[0]))
        ss = rng.integers(0, n, size=(m, n))
        a_bar = A[sa].mean(1)                     # (m, sessions): seed-resampled session means
        b_bar = B[sb].mean(1)
        out[i:i + m] = np.take_along_axis(a_bar - b_bar, ss, 1).mean(1)
    lo, hi = np.percentile(out, [50 * (1 - level), 50 * (1 + level)])
    return dict(ci_lo=float(lo), ci_hi=float(hi))


def compare(A, B, n_boot=10000, seed=0):
    """Full comparison of A against B. A, B: (seeds, sessions) arrays."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    n = min(A.shape[1], B.shape[1])
    r = session_test(A[:, :n].mean(0), B[:, :n].mean(0))
    r.update(seed_test(A[:, :n].mean(1), B[:, :n].mean(1)) if B.shape[0] > 1 else dict(p_seed=np.nan))
    r.update(hierarchical_ci(A, B, n_boot=n_boot, seed=seed))
    r.update(n_seeds_a=A.shape[0], n_seeds_b=B.shape[0], n_sessions=n)
    return r


def verdict(diff, q, ci_lo, ci_hi, alpha=0.05, name="A"):
    sig = q < alpha
    excl = (ci_lo > 0) or (ci_hi < 0)
    if sig and excl:
        return f"{name} better" if diff > 0 else f"{name} worse"
    if not sig and not excl:
        return "no significant difference"
    return "inconclusive (tests disagree)"
