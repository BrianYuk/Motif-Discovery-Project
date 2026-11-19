"""Gibbs site sampler for motif discovery (Lawrence et al., 1993), ZOOPS version.

State: one site position per sequence, or None for "no site". Each step removes
one sequence, rebuilds the PWM from the sites in all the others, and draws a new
position for the removed sequence in proportion to its likelihood ratio R_n(a),
with "no site" as one more option. Occasional phase shifts slide every site a few
positions so the sampler does not lock onto a shifted copy of the motif.
"""
import numpy as np
from scipy.special import logsumexp

from .em import log_ratio_scores, windows

DEFAULT_PSEUDOCOUNT = 0.25


def _pwm_from_counts(counts, n_sites, pseudocount):
    return (counts + pseudocount) / (n_sites + 4 * pseudocount)


def _site_counts(win, sites):
    width = win.shape[-1]
    counts = np.zeros((4, width))
    cols = np.arange(width)
    for n, a in enumerate(sites):
        if a is not None:
            np.add.at(counts, (win[n, a], cols), 1.0)
    return counts


def zoops_score(win, sites, log_bg, pseudocount):
    """Objective shared with EM: sum_n log[(1-g) + (g/m) sum_a R_n(a)], with the PWM
    and g estimated from the current alignment."""
    n_seqs, m = win.shape[0], win.shape[1]
    k = sum(a is not None for a in sites)
    pwm = _pwm_from_counts(_site_counts(win, sites), k, pseudocount)
    gamma = float(np.clip((k + 1) / (n_seqs + 2), 1e-3, 1 - 1e-6))
    S = log_ratio_scores(win, np.log(pwm), log_bg)
    log_site = logsumexp(np.log(gamma / m) + S, axis=1)
    return float(np.logaddexp(np.log1p(-gamma), log_site).sum()), pwm


def _draw(logits, rng):
    """Sample an index with probability proportional to exp(logits).

    Same steps and the same single random number as rng.choice(len(p), p=p) with
    p = exp(logits - logsumexp(logits)), without its per-call overhead."""
    top = logits.max()
    p = np.exp(logits - (top + np.log(np.exp(logits - top).sum())))
    cdf = p.cumsum()
    cdf /= cdf[-1]
    return int(cdf.searchsorted(rng.random(), side="right"))


def gibbs_single(X, width, rng, background, pseudocount=DEFAULT_PSEUDOCOUNT,
                 n_sweeps=200, burn_in=50, shift_every=5, max_shift=2, zoops=True):
    win = windows(X, width)
    n_seqs, m = win.shape[0], win.shape[1]
    log_bg = np.log(np.asarray(background, dtype=float))
    cols = np.arange(width)
    log_gamma_scale = np.log(1.0 / m)

    sites = [int(a) for a in rng.integers(0, m, size=n_seqs)]
    counts = _site_counts(win, sites)
    k_total = sum(a is not None for a in sites)     # kept up to date, never recounted
    best_score, best_sites = -np.inf, list(sites)
    logits = np.empty(m + 1)

    for sweep in range(1, n_sweeps + 1):
        for n in rng.permutation(n_seqs):
            a_old = sites[n]
            if a_old is not None:
                counts[win[n, a_old], cols] -= 1.0   # one cell per column, so plain indexing is exact
                k_total -= 1
            k = k_total
            log_odds = np.log((counts + pseudocount) / (k + 4 * pseudocount)) - log_bg[:, None]
            S = log_odds[win[n], cols].sum(axis=-1)                       # m
            if zoops:
                gamma = (k + 1) / (n_seqs - 1 + 2)                        # Beta(1,1) prior
                logits[0] = np.log1p(-gamma)
                logits[1:] = np.log(gamma) + log_gamma_scale + S
            else:
                logits[0] = -np.inf
                logits[1:] = S
            choice = _draw(logits, rng)
            sites[n] = None if choice == 0 else choice - 1
            if sites[n] is not None:
                counts[win[n, sites[n]], cols] += 1.0
                k_total += 1

        if shift_every and sweep % shift_every == 0:
            sites = _best_phase_shift(win, sites, log_bg, pseudocount, max_shift)
            counts = _site_counts(win, sites)
            k_total = sum(a is not None for a in sites)

        if sweep > burn_in or sweep == n_sweeps:
            score, _ = zoops_score(win, sites, log_bg, pseudocount)
            if score > best_score:
                best_score, best_sites = score, list(sites)

    score, pwm = zoops_score(win, best_sites, log_bg, pseudocount)
    return {"pwm": pwm, "sites": best_sites, "score": score}


def _best_phase_shift(win, sites, log_bg, pseudocount, max_shift):
    """Try sliding every site by -max_shift..max_shift; keep the shift that scores best."""
    m = win.shape[1]
    best_score, _ = zoops_score(win, sites, log_bg, pseudocount)
    best = sites
    for s in range(-max_shift, max_shift + 1):
        if s == 0:
            continue
        shifted = [None if a is None or not 0 <= a + s < m else a + s for a in sites]
        score, _ = zoops_score(win, shifted, log_bg, pseudocount)
        if score > best_score:
            best_score, best = score, shifted
    return list(best)


def find_motif_gibbs(X, width, seed=0, n_starts=1, background=(0.25,) * 4, **kwargs) -> dict:
    """Run the sampler from `n_starts` seeds and keep the best-scoring alignment."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        res = gibbs_single(X, width, rng, background, **kwargs)
        if best is None or res["score"] > best["score"]:
            best = res
    return {"pwm": best["pwm"], "sites": best["sites"], "score": best["score"], "extra": {}}
