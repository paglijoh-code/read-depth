"""Tokenizer checks for the atlas task families (spec docs/ATLAS-families.md, shared rules 4, 6, 7, 10).

All functions take loaded HF tokenizers and always tokenize without special tokens. `load_tokenizers` uses the local
cache only (no network). Batch tokenization keeps `check_items` fast enough for 80 tasks x 600 items x 7 vocabularies.
"""
from __future__ import annotations

import hashlib

from surgery.atlas.ccheck import MODELS

TOKENIZER_IDS = [m[0] for m in MODELS]
CTX = "Then"  # neutral one-word context for vetting (sentencepiece tokenizers treat a bare leading space oddly)


def load_tokenizers(ids=None, strict=False) -> dict:
    """{id: tokenizer} from the local cache. Ids that cannot load are left out (strict=True raises instead)."""
    from transformers import AutoTokenizer
    out = {}
    for i in (ids or TOKENIZER_IDS):
        try:
            out[i] = AutoTokenizer.from_pretrained(i, local_files_only=True)
        except Exception:
            if strict:
                raise
    return out


def distinct(toks: dict) -> dict:
    """One tokenizer per distinct vocabulary (first id wins)."""
    seen, out = set(), {}
    for i, t in toks.items():
        key = hashlib.md5(repr(sorted(t.get_vocab().items())).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out[i] = t
    return out


def _ids(tok, s) -> list:
    return list(tok(s, add_special_tokens=False).input_ids)


def _batch(tok, strs) -> list:
    return tok(list(strs), add_special_tokens=False)["input_ids"]


def single_token_in_context(tok, prompt: str, option: str) -> bool:
    """tok(prompt + option) == tok(prompt) + [one id]."""
    p, po = _ids(tok, prompt), _ids(tok, prompt + option)
    return len(po) == len(p) + 1 and po[:-1] == p


def align(tok, a: str, b: str) -> dict:
    """{len_equal, n_diff, last_equal}; n_diff is the number of differing positions (None if lengths differ)."""
    x, y = _ids(tok, a), _ids(tok, b)
    eq = len(x) == len(y)
    return {"len_equal": eq, "n_diff": sum(i != j for i, j in zip(x, y)) if eq else None,
            "last_equal": bool(x) and bool(y) and x[-1] == y[-1]}


def vet_words(words, toks: dict, prefix: str = " ") -> list:
    """The words that are a single token (with `prefix`) after a neutral context in every tokenizer."""
    words = list(words)
    ok = [True] * len(words)
    for tok in toks.values():
        base = _ids(tok, CTX)
        got = _batch(tok, [CTX + prefix + w for w in words])
        for k, g in enumerate(got):
            if not (len(g) == len(base) + 1 and g[:-1] == base):
                ok[k] = False
    return [w for w, o in zip(words, ok) if o]


def check_items(items, options, toks: dict) -> dict:
    """Check all items, answer twins and surface twins on every tokenizer.

    Per item: surface twin has the same length and exactly one differing position, not the last; answer twin has the
    same length and the same last token; every option is a single token in context on every prompt (item and twins),
    and the 4 option ids differ. Returns {per_tokenizer: {id: counts}, bad_ids: [item ids], ok: bool}.
    """
    bad, per = set(), {}
    prompts = {x["id"]: (x["prompt"], x["ans_twin"]["prompt"], x["surf_twin"]["prompt"]) for x in items}
    uniq = sorted({p for tri in prompts.values() for p in tri})
    for tid, tok in toks.items():
        enc = dict(zip(uniq, _batch(tok, uniq)))
        opt_enc = {o: dict(zip(uniq, _batch(tok, [p + o for p in uniq]))) for o in options}
        cnt = {"surf_bad": 0, "ans_bad": 0, "opt_bad": 0, "dup_bad": 0}
        for iid, (p, pa, ps) in prompts.items():
            e, ea, es = enc[p], enc[pa], enc[ps]
            if not (len(e) == len(es) and sum(u != v for u, v in zip(e, es)) == 1 and e[-1] == es[-1]):
                cnt["surf_bad"] += 1
                bad.add(iid)
            if not (len(e) == len(ea) and e[-1] == ea[-1]):
                cnt["ans_bad"] += 1
                bad.add(iid)
            for q in (p, pa, ps):
                eq, last = enc[q], []
                for o in options:
                    g = opt_enc[o][q]
                    if not (len(g) == len(eq) + 1 and g[:-1] == eq):
                        cnt["opt_bad"] += 1
                        bad.add(iid)
                    last.append(g[-1])
                if len(set(last)) != len(options):
                    cnt["dup_bad"] += 1
                    bad.add(iid)
        per[tid] = cnt
    return {"per_tokenizer": per, "bad_ids": sorted(bad), "ok": not bad}
