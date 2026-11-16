"""Expectation maximization for motif discovery (MEME-style ZOOPS model).

Model: each sequence holds zero or one site; with probability gamma it holds one,
placed uniformly at one of m = L - W + 1 start positions; every other base is
drawn from the background q. All likelihoods are kept in log space.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import logsumexp

DEFAULT_PSEUDOCOUNT = 0.25  # per base, so one pseudo-site per column


def windows(X: np.ndarray, width: int) -> np.ndarray:
    """(N, L) int array -> (N, m, W) view of every W-long window."""
    return sliding_window_view(X, width, axis=1)


def log_ratio_scores(win: np.ndarray, log_pwm: np.ndarray, log_bg: np.ndarray) -> np.ndarray:
    """log R_n(a) = sum_j log(theta[x, j] / q[x]) for every window -> (N, m)."""
    width = win.shape[-1]
    log_odds = log_pwm - log_bg[:, None]              # 4 x W
    return log_odds[win, np.arange(width)].sum(axis=-1)


def e_step(S: np.ndarray, gamma: float):
    """Posterior z[n, a] that the site starts at a, and the log-likelihood ratio."""
    m = S.shape[1]
    log_num = np.log(gamma / m) + S                    # N x m
    log_site = logsumexp(log_num, axis=1)              # N
    log_den = np.logaddexp(np.log1p(-gamma), log_site)
    z = np.exp(log_num - log_den[:, None])
    return z, float(log_den.sum())


def m_step(win: np.ndarray, z: np.ndarray, pseudocount: float):
    """Re-estimate the PWM from expected counts, and gamma from expected sites."""
    width = win.shape[-1]
    counts = np.empty((4, width))
    for b in range(4):
        counts[b] = (z[..., None] * (win == b)).sum(axis=(0, 1))
    pwm = (counts + pseudocount) / (counts.sum(axis=0) + 4 * pseudocount)
    gamma = float(z.sum(axis=1).mean())
    return pwm, gamma


def init_from_window(win: np.ndarray, rng: np.random.Generator, weight: float = 0.5) -> np.ndarray:
    """Start from one random window: `weight` on its base, the rest spread evenly."""
    n = rng.integers(win.shape[0])
    a = rng.integers(win.shape[1])
    word = win[n, a]
    width = win.shape[-1]
    pwm = np.full((4, width), (1 - weight) / 3)
    pwm[word, np.arange(width)] = weight
    return pwm


def screen_starts(win, rng, log_bg, n_candidates, gamma0):
    """MEME-style start search: score `n_candidates` random starting words by the
    likelihood after one E-step and return the best one."""
    best_pwm, best_ll = None, -np.inf
    for _ in range(n_candidates):
        pwm = init_from_window(win, rng)
        _, ll = e_step(log_ratio_scores(win, np.log(pwm), log_bg), gamma0)
        if ll > best_ll:
            best_pwm, best_ll = pwm, ll
    return best_pwm


def em_single(X, width, rng, background, pseudocount=DEFAULT_PSEUDOCOUNT,
              max_iter=500, tol=1e-6, gamma0=0.5, zoops=True, n_candidates=1):
    win = windows(X, width)
    log_bg = np.log(np.asarray(background, dtype=float))
    gamma = gamma0 if zoops else 1.0 - 1e-9
    if n_candidates > 1:
        pwm = screen_starts(win, rng, log_bg, n_candidates, gamma)
    else:
        pwm = init_from_window(win, rng)
    prev = -np.inf
    loglik = prev
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        S = log_ratio_scores(win, np.log(pwm), log_bg)
        z, loglik = e_step(S, gamma)
        pwm, gamma_new = m_step(win, z, pseudocount)
        if zoops:
            gamma = float(np.clip(gamma_new, 1e-3, 1 - 1e-6))
        if abs(loglik - prev) < tol * max(1.0, abs(loglik)):
            break
        prev = loglik
    # Final posteriors under the final parameters.
    S = log_ratio_scores(win, np.log(pwm), log_bg)
    z, loglik = e_step(S, gamma)
    return {"pwm": pwm, "gamma": gamma, "z": z, "loglik": loglik, "n_iter": n_iter}


def shift_pwm(pwm: np.ndarray, s: int) -> np.ndarray:
    """Slide the matrix s columns (positive = right), filling new columns with uniform."""
    width = pwm.shape[1]
    out = np.full_like(pwm, 0.25)
    if s >= 0:
        out[:, s:] = pwm[:, :width - s]
    else:
        out[:, :width + s] = pwm[:, -s:]
    return out


def em_from(X, width, pwm, gamma, background, pseudocount=DEFAULT_PSEUDOCOUNT,
            max_iter=500, tol=1e-6, zoops=True, **_):
    """EM started from a given matrix (used for phase-shift trials)."""
    win = windows(X, width)
    log_bg = np.log(np.asarray(background, dtype=float))
    prev, n_iter = -np.inf, 0
    for n_iter in range(1, max_iter + 1):
        z, loglik = e_step(log_ratio_scores(win, np.log(pwm), log_bg), gamma)
        pwm, gamma_new = m_step(win, z, pseudocount)
        if zoops:
            gamma = float(np.clip(gamma_new, 1e-3, 1 - 1e-6))
        if abs(loglik - prev) < tol * max(1.0, abs(loglik)):
            break
        prev = loglik
    z, loglik = e_step(log_ratio_scores(win, np.log(pwm), log_bg), gamma)
    return {"pwm": pwm, "gamma": gamma, "z": z, "loglik": loglik, "n_iter": n_iter}


def phase_shift_search(X, width, res, background, max_shift, trial_iter=20, **kwargs):
    """Gibbs-style phase shifts for EM: restart briefly from the converged matrix slid
    by -max_shift..max_shift columns; finish the best trial if it beats the original."""
    best_trial = None
    for s in range(-max_shift, max_shift + 1):
        if s == 0:
            continue
        trial = em_from(X, width, shift_pwm(res["pwm"], s), res["gamma"], background,
                        **{**kwargs, "max_iter": trial_iter})
        if best_trial is None or trial["loglik"] > best_trial["loglik"]:
            best_trial = trial
    if best_trial is not None and best_trial["loglik"] > res["loglik"]:
        final = em_from(X, width, best_trial["pwm"], best_trial["gamma"], background, **kwargs)
        if final["loglik"] > res["loglik"]:
            final["n_iter"] += res["n_iter"]
            return final
    return res


def find_motif_em(X, width, seed=0, n_starts=1, background=(0.25,) * 4,
                  site_threshold=0.5, max_shift=None, **kwargs) -> dict:
    """Run EM from `n_starts` starts and keep the highest likelihood. With
    n_candidates > 1, each start is the best of that many screened starting words.

    Returns the PWM, one predicted site per sequence (the most probable start if its
    posterior exceeds `site_threshold`, else None) and the log-likelihood ratio.
    """
    rng = np.random.default_rng(seed)
    if max_shift is None:
        max_shift = width // 2
    best = None
    for _ in range(n_starts):
        res = em_single(X, width, rng, background, **kwargs)
        if max_shift:
            res = phase_shift_search(X, width, res, background, max_shift,
                                     **{k: v for k, v in kwargs.items() if k != "n_candidates"})
        if best is None or res["loglik"] > best["loglik"]:
            best = res
    z = best["z"]
    a_hat = z.argmax(axis=1)
    sites = [int(a) if z[n, a] > site_threshold else None for n, a in enumerate(a_hat)]
    return {"pwm": best["pwm"], "sites": sites, "score": best["loglik"],
            "extra": {"gamma": best["gamma"], "n_iter": best["n_iter"]}}
