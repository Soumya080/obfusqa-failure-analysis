"""Statistics. Every comparison is PAIRED: the same items appear in both conditions.

paired_bootstrap_diff  - CI for acc(A) - acc(B). Each resample draws item indices with replacement
                         and uses the SAME indices for A and B. That pairing is what makes the CI tight.
mcnemar_exact          - test using only discordant items (right in A / wrong in B, and the reverse).
wilson_ci              - CI for a proportion (used for attribution shares and audit rates).
length_share           - (acc(BASE) - acc(C5)) / (acc(BASE) - acc(CO)): fraction of the CO drop that
                         length alone reproduces. Reported with a bootstrap CI; undefined if BASE ~ CO.
"""
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def paired_bootstrap_diff(a, b, n_boot=10000, seed=0, alpha=0.05):
    a, b = _arr(a), _arr(b)
    assert a.shape == b.shape, "paired arrays must be aligned item by item"
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi), "n": n}


def mcnemar_exact(a, b):
    from statsmodels.stats.contingency_tables import mcnemar

    a, b = _arr(a).astype(int), _arr(b).astype(int)
    both = int(((a == 1) & (b == 1)).sum())
    a_only = int(((a == 1) & (b == 0)).sum())   # right in A, wrong in B
    b_only = int(((a == 0) & (b == 1)).sum())   # wrong in A, right in B
    neither = int(((a == 0) & (b == 0)).sum())
    res = mcnemar([[both, a_only], [b_only, neither]], exact=True)
    return {"a_only": a_only, "b_only": b_only, "p": float(res.pvalue)}


def wilson_ci(k, n, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint

    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": k, "n": n}
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return {"p": k / n, "lo": float(lo), "hi": float(hi), "k": int(k), "n": int(n)}


def length_share(base, c5, co, n_boot=10000, seed=0, alpha=0.05):
    base, c5, co = _arr(base), _arr(c5), _arr(co)
    rng = np.random.default_rng(seed)
    n = len(base)
    point_den = base.mean() - co.mean()
    point = (base.mean() - c5.mean()) / point_den if point_den > 0 else float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    num = base[idx].mean(1) - c5[idx].mean(1)
    den = base[idx].mean(1) - co[idx].mean(1)
    ok = den > 1e-9
    ratios = num[ok] / den[ok]
    if ok.mean() < 0.95:
        print(f"[length_share] warning: BASE-CO gap is near zero in {1 - ok.mean():.0%} of resamples; ratio unstable")
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2]) if len(ratios) else (float("nan"), float("nan"))
    return {"share": float(point), "lo": float(lo), "hi": float(hi), "n": n}


def compare(a, b, label_a, label_b, n_boot=10000, seed=0):
    """One row of the results table."""
    d = paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed)
    m = mcnemar_exact(a, b)
    return {"comparison": f"{label_a} - {label_b}", "acc_a": float(np.mean(a)), "acc_b": float(np.mean(b)),
            "diff": d["diff"], "ci_lo": d["lo"], "ci_hi": d["hi"],
            "a_only": m["a_only"], "b_only": m["b_only"], "mcnemar_p": m["p"], "n": d["n"]}


def coverage_self_test(n_items=256, p_a=0.7, p_b=0.5, rho=0.5, n_sims=200, n_boot=2000, seed=1):
    """Simulate paired right/wrong data with a KNOWN true difference and check the 95% CI covers it
    about 95% of the time. Run this once; it should print a coverage between ~0.92 and ~0.98."""
    rng = np.random.default_rng(seed)
    true_diff = p_a - p_b
    hits = 0
    for s in range(n_sims):
        z = rng.random(n_items)
        a = (z < p_a).astype(int)
        # b is correlated with a: with prob rho copy a's latent draw, else draw fresh
        z2 = np.where(rng.random(n_items) < rho, z, rng.random(n_items))
        b = (z2 < p_b).astype(int)
        r = paired_bootstrap_diff(a, b, n_boot=n_boot, seed=s)
        hits += r["lo"] <= true_diff <= r["hi"]
    cov = hits / n_sims
    print(f"[stats] bootstrap CI coverage on simulated data: {cov:.2f} (target ~0.95)")
    return cov
