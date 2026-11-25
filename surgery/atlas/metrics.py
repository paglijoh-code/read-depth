"""M9 atlas: pure summary functions over per-layer curves and sweeps (docs/ATLAS-phase0.md, Brief B).

No model, no I/O. Accuracies are fractions in 0..1. Layer index 0 is the embedding output
for probe and lens curves, and block 0 for lesion drops and block influence.
"""
from __future__ import annotations

EPS = 1e-9
MIN_GAP = 0.05
K_GRID = (0, 6, 9, 12, 14, 16)


def _first_reaching(curve, threshold):
    for i, v in enumerate(curve):
        if v >= threshold - EPS:
            return i
    return None


def l_on(curve, chance: float, fraction: float = 0.5):
    """Smallest layer whose accuracy reaches chance + `fraction` (default half) of (peak - chance); None if the peak is under 5 points above chance."""
    peak = max(curve)
    if peak - chance < MIN_GAP - EPS:
        return None
    return _first_reaching(curve, chance + fraction * (peak - chance))


def l_lens(curve, chance: float, final: float):
    """As l_on but against the model's final accuracy; None if final is under 5 points above chance."""
    if final - chance < MIN_GAP - EPS:
        return None
    return _first_reaching(curve, chance + 0.5 * (final - chance))


def fragile(drops, threshold: float = 0.05) -> list[int]:
    """Blocks whose lesion costs at least `threshold` accuracy."""
    return [i for i, d in enumerate(drops) if d >= threshold - EPS]


def shared_order(bi, n_layers: int) -> list[int]:
    """Blocks by ascending block influence (least important first), excluding block 0 and the last block; ties by index."""
    if len(bi) != n_layers:
        raise ValueError(f"expected {n_layers} block-influence values, got {len(bi)}")
    return sorted(range(1, n_layers - 1), key=lambda i: (bi[i], i))


def _mean(x):
    if isinstance(x, (int, float)):
        return float(x)
    x = list(x)
    return sum(x) / len(x)


def k_star(acc_by_k, base: float, grid=K_GRID, tol: float = 0.05):
    """Largest grid k whose seed-mean accuracy is at least base - tol. `acc_by_k[k]` is a number or a list of per-seed
    accuracies. Grid values missing from `acc_by_k` are ignored; None if no grid value qualifies."""
    best = None
    for k in grid:
        if k in acc_by_k and _mean(acc_by_k[k]) >= base - tol - EPS:
            best = k
    return best


def transfer(score_after_heal, cut_only):
    """score_after_heal[a][b] = score on task b after healing on task a; cut_only[b] = score on b with the cut and no healing.
    Returns {a: {b: score_after_heal[a][b] - cut_only[b]}}."""
    return {a: {b: s - cut_only[b] for b, s in row.items()} for a, row in score_after_heal.items()}
