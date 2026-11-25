"""Planted toy model for the P0 pipeline: a 4-class key token is copied into a planted 3-dim subspace U at block k.

Same adapter interface as p0_harness.Adapter (collect_ids, patched_logits, lens_logits, n_blocks, d); the toy
recomputes the (tiny) last-token path instead of caching. Blocks before k never see the key; later blocks leave U alone
and the readout is linear in U, so use onset must sit at block k.
"""
from __future__ import annotations

import numpy as np

VERT = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)  # centred class vertices
QUERY, KEY_POS = 20, 1


class ToyAdapter:
    def __init__(self, d=32, B=8, k=4, seed=0, eps=0.3, scale=1.0):
        rng = np.random.default_rng(seed)
        self.d, self.n_blocks, self.k, self.eps = d, B, k, eps
        # planted subspace U: axis-aligned, so the standardised probe recovers it exactly (a rotated U would only
        # blur the probe subspace, not the onset logic)
        self.Ub = np.eye(d, 3, dtype=np.float32)
        self.Ucls = (VERT @ self.Ub.T * scale).astype(np.float32)  # [4, d] class vectors written at block k
        self.E = rng.standard_normal((32, d)).astype(np.float32)
        self.W = (rng.standard_normal((B + 1, d, d)) / np.sqrt(d)).astype(np.float32)
        self.perp = (np.eye(d) - self.Ub @ self.Ub.T).astype(np.float32)

    def _step(self, l, h, key):
        upd = self.eps * (np.tanh((h @ self.perp) @ self.W[l]) @ self.perp)  # never touches U
        return h + upd + (self.Ucls[key] if l == self.k else 0.0)

    def _logits(self, h):
        return (self.Ub.T @ h) @ VERT.T

    def _run(self, ids, start=0, h=None):
        key = ids[KEY_POS]
        if h is None:
            h = self.E[ids[-1]] @ self.perp
        hs = {start: h}
        for l in range(start + 1, self.n_blocks + 1):
            h = self._step(l, h, key)
            hs[l] = h
        return hs

    def collect_ids(self, ids_list, opt_ids=None, batch_size=None):
        S = np.zeros((len(ids_list), self.n_blocks + 1, self.d), dtype=np.float32)
        Lg = np.zeros((len(ids_list), 4), dtype=np.float32)
        for i, ids in enumerate(ids_list):
            hs = self._run(ids)
            S[i] = np.stack([hs[l] for l in range(self.n_blocks + 1)])
            Lg[i] = self._logits(hs[self.n_blocks])
        return S, Lg

    def patched_logits(self, ids, rows, opt_ids=None, chunk=None):
        out = []
        for L, v in rows:
            hs = self._run(ids) if L is None else self._run(ids, L, np.asarray(v, dtype=np.float32))
            out.append(self._logits(hs[self.n_blocks]))
        return np.array(out, dtype=np.float32).reshape(len(rows), 4)

    def lens_logits(self, states, opt_ids=None, batch_size=None):
        return np.einsum("dk,nbd->nbk", self.Ub @ VERT.T, states)


def make_toy_items(n, seed=0):
    """Token-id sequences [distractor, key, d, d, d, query]; answer twin changes the key, surface twin a distractor."""
    rng = np.random.default_rng(seed)
    ids, y, an, y_an, sf = [], [], [], [], []
    for i in range(n):
        key = i % 4
        x = [int(t) for t in rng.integers(4, 20, 6)]
        x[KEY_POS], x[-1] = key, QUERY
        k2 = int(rng.choice([c for c in range(4) if c != key]))
        a = list(x)
        a[KEY_POS] = k2
        s = list(x)
        pos = int(rng.choice([0, 2, 3, 4]))
        s[pos] = int(rng.choice([t for t in range(4, 20) if t != x[pos]]))
        ids.append(x), y.append(key), an.append(a), y_an.append(k2), sf.append(s)
    perm = rng.permutation(n)
    return ([ids[i] for i in perm], [y[i] for i in perm], [an[i] for i in perm], [y_an[i] for i in perm],
            [sf[i] for i in perm])
