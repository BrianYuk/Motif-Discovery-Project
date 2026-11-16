from collections import Counter

import numpy as np
import pytest

from motifbench import (find_motif, generate_dataset, get_pwm, information_content,
                        pwm_similarity, site_metrics)
from motifbench.alphabet import encode, revcomp_pwm
from motifbench.shuffle import dinucleotide_shuffle

STRONG = "MA0841.2"   # NFE2, 16.6 bits


# -- generator ---------------------------------------------------------------
def test_generator_records_sites_and_share():
    ds = generate_dataset(get_pwm(STRONG), 50, seq_len=100, frac_with_site=0.4, seed=3)
    assert len(ds.sequences) == 50 and all(len(s) == 100 for s in ds.sequences)
    assert sum(s is not None for s in ds.sites) == 20
    assert all(0 <= s <= 100 - 10 for s in ds.sites if s is not None)


def test_generator_is_deterministic_and_control_has_no_sites():
    a = generate_dataset(get_pwm(STRONG), 30, seed=7)
    b = generate_dataset(get_pwm(STRONG), 30, seed=7)
    assert a.sequences == b.sequences and a.sites == b.sites
    control = generate_dataset(get_pwm(STRONG), 30, frac_with_site=0.0, seed=7)
    assert all(s is None for s in control.sites)


def test_planted_sites_score_like_the_motif():
    pwm = get_pwm(STRONG)
    ds = generate_dataset(pwm, 200, frac_with_site=1.0, seed=1)
    lo = np.log2(pwm / 0.25)
    scores = [lo[encode(seq[a:a + 10]), np.arange(10)].sum() for seq, a in zip(ds.sequences, ds.sites)]
    assert np.mean(scores) > 10        # planted sites look like the motif


# -- shuffle -----------------------------------------------------------------
def test_dinucleotide_shuffle_keeps_pair_counts():
    rng = np.random.default_rng(0)
    seq = generate_dataset(get_pwm(STRONG), 1, seq_len=300, seed=2).sequences[0]
    for _ in range(5):
        sh = dinucleotide_shuffle(seq, rng)
        assert len(sh) == len(seq) and sh[0] == seq[0] and sh[-1] == seq[-1]
        pairs = lambda s: Counter(s[i:i + 2] for i in range(len(s) - 1))  # noqa: E731
        assert pairs(sh) == pairs(seq)
        assert sh != seq


# -- metrics -----------------------------------------------------------------
def test_information_content_limits():
    assert information_content(np.full((4, 5), 0.25)) == pytest.approx(0.0)
    one_hot = np.eye(4)[:, [0, 1, 2, 3, 0]] * 0.997 + 0.001
    assert information_content(one_hot) > 9.5        # close to 2 bits x 5


def test_similarity_identity_strand_and_offset():
    pwm = get_pwm(STRONG)
    assert pwm_similarity(pwm, pwm)["score"] == pytest.approx(1.0)
    rc = pwm_similarity(revcomp_pwm(get_pwm("MA0466.4")), get_pwm("MA0466.4"))
    assert rc["strand"] in "+-" and rc["score"] > 0.99
    shifted = pwm_similarity(pwm[:, 2:], pwm)
    assert shifted["offset"] == 2 and 0.7 < shifted["score"] < 0.9   # penalised partial overlap


def test_site_metrics_counts():
    m = site_metrics([5, None, 40, 7], [6, 3, None, 20], tolerance=2)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 3)


# -- methods -----------------------------------------------------------------
@pytest.mark.parametrize("method,kwargs", [("em", {"n_starts": 5}), ("gibbs", {"n_sweeps": 60, "burn_in": 10})])
def test_classical_methods_recover_strong_motif(method, kwargs):
    pwm = get_pwm(STRONG)
    ds = generate_dataset(pwm, 60, seq_len=120, frac_with_site=1.0, seed=11)
    res = find_motif(ds.sequences, 10, method, seed=1, **kwargs)
    assert pwm_similarity(res["pwm"], pwm)["score"] > 0.9
    assert site_metrics(res["sites"], ds.sites)["f1"] > 0.9


@pytest.mark.parametrize("method,kwargs", [("em", {"n_starts": 2}), ("gibbs", {"n_sweeps": 20, "burn_in": 5})])
def test_methods_are_deterministic_given_seed(method, kwargs):
    ds = generate_dataset(get_pwm(STRONG), 30, seq_len=80, frac_with_site=0.5, seed=4)
    a = find_motif(ds.sequences, 10, method, seed=9, **kwargs)
    b = find_motif(ds.sequences, 10, method, seed=9, **kwargs)
    assert np.array_equal(a["pwm"], b["pwm"]) and a["sites"] == b["sites"]


def test_cnn_recovers_strong_motif_and_is_deterministic():
    pytest.importorskip("torch")
    pwm = get_pwm(STRONG)
    ds = generate_dataset(pwm, 300, seq_len=100, frac_with_site=1.0, seed=5)
    a = find_motif(ds.sequences, 10, "cnn", seed=2, epochs=30)
    b = find_motif(ds.sequences, 10, "cnn", seed=2, epochs=30)
    assert np.allclose(a["pwm"], b["pwm"]) and a["sites"] == b["sites"]
    assert pwm_similarity(a["pwm"], pwm)["score"] > 0.7


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        find_motif(["ACGTACGTAC"] * 3, 4, "hmm")


def test_align_sites_uses_offset():
    from motifbench.metrics import align_sites
    assert align_sites([10, None, 7], 3, "+") == [7, None, 4]
    assert align_sites([10, None], 3, "-") == [10, None]


def test_em_phase_shift_recovers_out_of_phase_motif():
    # Seed and data chosen from a case where EM without phase shifts converged 3-4
    # columns out of phase (similarity 0.6-0.7, raw site F1 0).
    pwm = get_pwm(STRONG)
    ds = generate_dataset(pwm, 600, frac_with_site=0.5, seed=7)
    res = find_motif(ds.sequences, 10, "em", seed=3, n_candidates=20)
    sim = pwm_similarity(res["pwm"], pwm)
    assert sim["offset"] == 0 and sim["score"] > 0.95
