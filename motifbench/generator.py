"""Synthetic benchmark generator: plant a PWM into random DNA and record where."""
from dataclasses import dataclass, field

import numpy as np

from .alphabet import decode


@dataclass
class Dataset:
    sequences: list          # list of ACGT strings, all the same length
    sites: list              # start position per sequence, or None if no site was planted
    pwm: np.ndarray          # 4 x W planted matrix
    params: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.pwm.shape[1]


def sample_site(pwm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one W-long site, one base per column from that column's probabilities."""
    cum = np.cumsum(pwm, axis=0)
    u = rng.random(pwm.shape[1])
    return (u[None, :] > cum).sum(axis=0).astype(np.int8)


def generate_dataset(
    pwm: np.ndarray,
    n_seqs: int,
    seq_len: int = 200,
    frac_with_site: float = 1.0,
    seed: int = 0,
    background=(0.25, 0.25, 0.25, 0.25),
) -> Dataset:
    """Random background sequences; a chosen share get one planted site (forward strand).

    frac_with_site = 0 gives the no-motif control. The number of sequences with a
    site is round(frac * n), and which sequences get one is random.
    """
    pwm = np.asarray(pwm, dtype=float)
    width = pwm.shape[1]
    if seq_len < width:
        raise ValueError("sequences must be at least as long as the motif")
    rng = np.random.default_rng(seed)
    X = rng.choice(4, size=(n_seqs, seq_len), p=np.asarray(background)).astype(np.int8)
    n_sites = int(round(frac_with_site * n_seqs))
    carriers = set(rng.choice(n_seqs, size=n_sites, replace=False).tolist())
    sites = []
    for n in range(n_seqs):
        if n in carriers:
            pos = int(rng.integers(0, seq_len - width + 1))
            X[n, pos:pos + width] = sample_site(pwm, rng)
            sites.append(pos)
        else:
            sites.append(None)
    params = dict(n_seqs=n_seqs, seq_len=seq_len, frac_with_site=frac_with_site, seed=seed)
    return Dataset([decode(row) for row in X], sites, pwm, params)
