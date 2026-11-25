"""Deterministic discovery subset shared by M3 intervention sweeps."""
import random


def sweep_indices(n_total: int, n: int = 500, seed: int = 0) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), n))
