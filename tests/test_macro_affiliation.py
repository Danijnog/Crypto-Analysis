import numpy as np

from crypto_analysis import macro_affiliation as ma


def planted(seed=0, p_in=0.9, p_out=0.01):
    """Six communities of 15 topics; the first two share topics 11-14. Every pair is observed."""
    rng = np.random.default_rng(seed)
    comms = [set(range(0, 15)), set(range(11, 26))] + [set(range(26 + 15 * i, 41 + 15 * i)) for i in range(4)]
    n = 86
    u, v = np.triu_indices(n, 1)
    same = np.array([any(a in s and b in s for s in comms) for a, b in zip(u, v)])
    y = (rng.random(len(u)) < np.where(same, p_in, p_out)).astype(int)
    return n, u, v, y, comms


def test_gradient_matches_finite_differences():
    n, u, v, y, _ = planted()
    F = np.random.default_rng(1).random((n, 3))
    eps = ma.background(n, int(y.sum()))
    G = ma.gradient(F, u, v, y, eps)
    h = 1e-6
    for i, c in [(0, 0), (12, 1), (40, 2)]:
        E = np.zeros_like(F)
        E[i, c] = h
        fd = (ma.loglik(F + E, u, v, y, eps) - ma.loglik(F - E, u, v, y, eps)) / (2 * h)
        assert np.isclose(G[i, c], fd, rtol=1e-4)


def test_recovers_planted_overlap():
    n, u, v, y, truth = planted()
    table = ma.choose_k(n, u, v, y, ks=(3, 6, 9, 12))
    K = int(table.groupby("K").heldout_ll.mean().idxmax())
    assert K == 6
    eps = ma.background(n, int(y.sum()))
    F, _ = ma.fit(ma.initial(n, u, v, y, K), u, v, y, eps)
    found = [set(c) for c in ma.memberships(F, ma.delta(eps)) if len(c) > 1]
    for t in truth:
        assert max(len(t & f) / len(t | f) for f in found) > 0.8
    shared = {11, 12, 13, 14}
    assert all(sum(s in f for f in found) == 2 for s in shared)


def test_unjudged_pairs_are_missing():
    """Dropping a pair from the observations is not the same as observing it as a non-edge."""
    n, u, v, y, _ = planted()
    F = np.random.default_rng(2).random((n, 3))
    keep = ~((u == 0) & (v == 1))
    assert ma.loglik(F, u[keep], v[keep], y[keep], 0.04) != ma.loglik(F, u, v, np.where(keep, y, 0), 0.04)


def test_isolated_topic_is_own_community():
    F = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    assert ma.memberships(F, 0.5) == [[0, 1], [2]]


def test_weights_scale_pairs():
    """A pair with weight 2 counts as the same pair observed twice."""
    n, u, v, y, _ = planted()
    F = np.random.default_rng(3).random((n, 3))
    w = np.ones(len(y))
    w[0] = 2.0
    twice = ma.loglik(F, np.r_[u, u[:1]], np.r_[v, v[:1]], np.r_[y, y[:1]], 0.04)
    assert np.isclose(ma.loglik(F, u, v, y, 0.04, w), twice)
    G2 = ma.gradient(F, np.r_[u, u[:1]], np.r_[v, v[:1]], np.r_[y, y[:1]], 0.04)
    assert np.allclose(ma.gradient(F, u, v, y, 0.04, w), G2)
