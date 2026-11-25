"""Atlas metrics against hand-worked values."""
import pytest

from surgery.atlas import metrics as m


def test_l_on():
    # chance 0.5, peak 0.9 -> threshold 0.7: first reached at layer 3 (0.7)
    assert m.l_on([0.5, 0.55, 0.6, 0.7, 0.9, 0.8], 0.5) == 3
    # chance 0.25, peak 0.85 -> threshold 0.55; layer 2 is 0.5 (no), layer 3 is 0.6
    assert m.l_on([0.25, 0.3, 0.5, 0.6, 0.85], 0.25) == 3
    # peak 0.52 is under 5 points above chance 0.5
    assert m.l_on([0.5, 0.52, 0.51], 0.5) is None
    # exactly 5 points above counts
    assert m.l_on([0.5, 0.55], 0.5) == 1


def test_l_lens():
    # chance 0.4, final 0.8 -> threshold 0.6; lens curve crosses at layer 2 even though it ends lower
    assert m.l_lens([0.4, 0.45, 0.6, 0.9, 0.8], 0.4, 0.8) == 2
    assert m.l_lens([0.4, 0.5], 0.4, 0.43) is None
    # never reaching the threshold gives None
    assert m.l_lens([0.4, 0.45, 0.5], 0.4, 0.9) is None


def test_fragile():
    assert m.fragile([0.0, 0.05, 0.049, 0.3, -0.2], 0.05) == [1, 3]
    assert m.fragile([], 0.05) == []


def test_shared_order():
    bi = [0.9, 0.3, 0.1, 0.2, 0.1, 0.05]  # 6 blocks: block 0 and 5 excluded
    assert m.shared_order(bi, 6) == [2, 4, 3, 1]  # tie 0.1 broken by index
    with pytest.raises(ValueError):
        m.shared_order(bi, 5)


# Fake sweep: base 0.80, tolerance 0.05 -> floor 0.75.
SWEEP = {0: [0.80, 0.80], 6: [0.79, 0.77], 9: [0.76, 0.74], 12: [0.70, 0.72], 14: [0.78, 0.60], 16: [0.30, 0.31]}


def test_k_star_hand_worked():
    # seed means: 0 -> 0.80, 6 -> 0.78, 9 -> 0.75 (exactly the floor, passes), 12 -> 0.71, 14 -> 0.69, 16 -> 0.305
    assert m.k_star(SWEEP, 0.80) == 9
    # largest qualifying k wins even if a smaller one failed
    assert m.k_star({0: 0.8, 6: 0.7, 12: 0.78}, 0.80) == 12
    # tighter tolerance
    assert m.k_star(SWEEP, 0.80, tol=0.03) == 6  # floor 0.77: 9 (0.75) fails
    # nothing qualifies; values off the grid are ignored
    assert m.k_star({0: 0.5, 7: 0.99}, 0.8) is None


def test_transfer_hand_worked():
    heal = {"a": {"a": 0.9, "b": 0.6}, "b": {"a": 0.5, "b": 0.8}}
    cut = {"a": 0.4, "b": 0.5}
    out = m.transfer(heal, cut)
    assert out["a"]["a"] == pytest.approx(0.5)
    assert out["a"]["b"] == pytest.approx(0.1)
    assert out["b"]["a"] == pytest.approx(0.1)
    assert out["b"]["b"] == pytest.approx(0.3)
