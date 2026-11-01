import numpy as np

from upgradebench.core import blend, correct, detection, member, noise_of_mean, one_hot, signal_for, solo_accuracy
from upgradebench.run import add_grid, replace_grid, summarise


def test_signal_inverts_the_accuracy():
    for acc in [0.3, 0.52, 0.6, 0.85]:
        assert abs(solo_accuracy(signal_for(acc)) - acc) < 1e-6


def test_chance_level_at_zero_signal():
    assert abs(solo_accuracy(0.0) - 1 / 18) < 1e-3


def test_integral_matches_simulation_and_rho_does_not_move_solo_accuracy():
    rng = np.random.default_rng(0)
    n, a = 90_000, signal_for(0.55)
    h = one_hot(n)
    s, e = rng.standard_normal(h.shape), rng.standard_normal(h.shape)
    for rho in [0.0, 0.5, 0.9]:
        assert abs(correct(member(a, rho, h, s, e)).mean() - 0.55) < 0.006


def test_balanced_rows():
    assert (one_hot(180).sum(0) == 10).all()


def test_noise_of_mean_matches_simulation():
    rng = np.random.default_rng(1)
    n, k, rho = 200_000, 4, 0.5
    s = rng.standard_normal(n)
    m = sum(np.sqrt(rho) * s + np.sqrt(1 - rho) * rng.standard_normal(n) for _ in range(k)) / k
    assert abs(m.std() - noise_of_mean(k, rho)) < 0.005


def test_share_zero_and_one_are_the_parts():
    r, t = np.ones((3, 2)), np.zeros((3, 2))
    assert (blend(r, t, 0.0) == r).all() and (blend(r, t, 1.0) == t).all()


def test_detection_arithmetic():
    before = np.array([True] * 50 + [False] * 50)
    after = before.copy()
    after[50:60] = True
    d = detection(before, after, n_public=100)
    assert abs(d["d_blend"] - 0.1) < 1e-12 and abs(d["flip"] - 0.1) < 1e-12
    assert abs(d["sd_public"] - np.sqrt((0.1 - 0.01) / 100)) < 1e-12
    assert abs(d["rows_needed"] - 2.8**2 * 0.1 / 0.01) < 1e-9


def test_replace_grid_gain_lands_and_weight_scales_it():
    df = replace_grid(quick=True, n=18_000)
    assert abs(df.d_solo - df.gain).max() < 0.02
    g = summarise(df, ["k", "rho", "rho_t", "share", "gain"])
    big = g[g.gain == g.gain.max()].groupby("share").d_blend.mean()
    assert big.iloc[-1] > big.iloc[0] > 0


def test_add_grid_runs_and_best_share_never_loses():
    df = add_grid(quick=True, n=18_000)
    assert (df.d_best >= -1e-12).all()
    assert df.blend_after.between(0, 1).all()
