"""Build the general-text set for the capability studies (docs/CONTRACT-capability.md), deterministically.

    uv run python scripts/build_general.py

Writes var/data/general/{train,dev}.jsonl (rows {"id","source","group","text"}) and manifest.json. Sources:
Dolly-15k Q&A ("Question: ...\\nAnswer: ..."), top-level Python stdlib *.py cut into ~128-token chunks, and two
Project Gutenberg books (#84, #1661) cut the same way. Dev is 10%, split by group (Dolly by row, code by file,
books by contiguous 50-chunk blocks) so no group spans both. Rows that contain a canary prompt, a Q&A that is a
near-duplicate of one (word-set Jaccard >= 0.6), or any 12-word run of data/ppl_heldout.txt are dropped
(surgery.capability.drop_reason). Downloads are cached in var/data/general/raw and pinned by sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import sysconfig
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from surgery.canaries import CANARIES_PATH, PPL_PATH, load_canaries, load_ppl_text  # noqa: E402
from surgery.capability import drop_reason, filter_sets  # noqa: E402

OUT = ROOT / "var" / "data" / "general"
RAW = OUT / "raw"
MODEL = "Qwen/Qwen3-0.6B"
SEED = 20261104
CHUNK = 128
MIN_CHUNK = 32          # a trailing chunk shorter than this is dropped
BLOCK = 50              # book chunks per dev/train block
DEV_FRAC = 0.10
TARGETS = {"dolly": 6000, "code": 5000, "book": 5000}  # rows kept (train + dev); books cannot reach theirs
UA = {"User-Agent": "surgery-research/0.1 (build_general.py)"}

SOURCES = {
    "dolly": ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
              "dolly.jsonl"),
    "gutenberg84": ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "pg84.txt"),
    "gutenberg1661": ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "pg1661.txt"),
}
# sha256 of the downloads, pinned from the first build (a None would mean not pinned yet: the build prints the value)
SHA256 = {
    "dolly": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
    "gutenberg84": "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42",
    "gutenberg1661": "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
}
START, END = "*** START OF THE PROJECT GUTENBERG EBOOK", "*** END OF THE PROJECT GUTENBERG EBOOK"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch(name: str) -> bytes:
    url, fname = SOURCES[name]
    path = RAW / fname
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    got, want = sha(raw), SHA256[name]
    if want is None:
        print(f"{name}: sha256 {got} (not pinned yet)")
    elif got != want:
        sys.exit(f"{name}: sha256 {got} != pinned {want}")
    return raw


def book_body(raw: bytes) -> str:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    a = text.index(START)
    a = text.index("\n", a) + 1
    return text[a:text.index(END)].strip()


def chunk_ids(tok, text: str) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out = []
    for i in range(0, len(ids), CHUNK):
        piece = ids[i:i + CHUNK]
        if len(piece) >= MIN_CHUNK:
            out.append(tok.decode(piece))
    return out


def split_groups(items: list[dict], rng: random.Random) -> set:
    """Dev groups: shuffled groups are added until they hold DEV_FRAC of the items."""
    sizes = Counter(it["group"] for it in items)
    groups = sorted(sizes)
    rng.shuffle(groups)
    dev, n = set(), 0
    for g in groups:
        if n >= DEV_FRAC * len(items):
            break
        dev.add(g)
        n += sizes[g]
    return dev


def keep(items, target, rng, drops, source, flt):
    """Filter, split by group, subsample each side to its share of `target`. Returns (train, dev)."""
    ok = []
    for it in items:
        why = drop_reason(it["text"], it.pop("question", None), *flt)
        if why:
            drops[(source, why)] += 1
        else:
            ok.append(it)
    dev_groups = split_groups(ok, rng)
    train = [it for it in ok if it["group"] not in dev_groups]
    dev = [it for it in ok if it["group"] in dev_groups]
    n_dev = round(DEV_FRAC * target)
    train = rng.sample(train, min(len(train), target - n_dev))
    dev = rng.sample(dev, min(len(dev), n_dev))
    return sorted(train, key=lambda r: r["id"]), sorted(dev, key=lambda r: r["id"])


def write_rows(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    path.write_bytes(data)
    return sha(data)


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = [c["prompt"] for c in load_canaries()]
    flt = filter_sets(prompts, load_ppl_text())
    drops, train, dev, candidates = Counter(), [], [], {}

    # (a) Dolly
    dolly = []
    for i, line in enumerate(fetch("dolly").decode("utf-8").splitlines()):
        r = json.loads(line)
        dolly.append({"id": f"dolly-{i:05d}", "source": "dolly", "group": f"dolly:{i}",
                      "text": f"Question: {r['instruction']}\nAnswer: {r['response']}", "question": r["instruction"]})
    candidates["dolly"] = len(dolly)
    t, d = keep(dolly, TARGETS["dolly"], rng, drops, "dolly", flt)
    train += t
    dev += d

    # (b) Python stdlib, top-level files, no test modules
    code = []
    for path in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")):
        if path.name.startswith("test") or path.name.endswith("_test.py"):
            continue
        for j, text in enumerate(chunk_ids(tok, path.read_text(encoding="utf-8", errors="replace"))):
            code.append({"id": f"code-{path.stem}-{j:04d}", "source": "code", "group": f"code:{path.name}", "text": text,
                         "question": None})
    candidates["code"] = len(code)
    t, d = keep(code, TARGETS["code"], rng, drops, "code", flt)
    train += t
    dev += d

    # (c) books
    books = []
    for gid in ("84", "1661"):
        for j, text in enumerate(chunk_ids(tok, book_body(fetch(f"gutenberg{gid}")))):
            books.append({"id": f"book{gid}-{j:04d}", "source": "book", "group": f"book{gid}:{j // BLOCK}",
                          "text": text, "question": None})
    candidates["book"] = len(books)
    t, d = keep(books, TARGETS["book"], rng, drops, "book", flt)
    train += t
    dev += d

    assert not ({r["group"] for r in train} & {r["group"] for r in dev}), "a group spans train and dev"
    for r in train + dev:
        r.pop("question", None)
    sha_train, sha_dev = write_rows(OUT / "train.jsonl", train), write_rows(OUT / "dev.jsonl", dev)

    counts = {s: {"candidates": candidates[s], "train": sum(r["source"] == s for r in train),
                  "dev": sum(r["source"] == s for r in dev)} for s in TARGETS}
    drop_counts = {f"{s}:{why}": n for (s, why), n in sorted(drops.items())}
    for s, c in counts.items():
        print(f"{s}: {c['candidates']} candidates, {c['train']} train, {c['dev']} dev")
    print("dropped:", drop_counts or "none")
    print(f"train.jsonl {len(train)} rows sha256 {sha_train}")
    print(f"dev.jsonl {len(dev)} rows sha256 {sha_dev}")
    manifest = {"seed": SEED, "model_tokenizer": MODEL, "chunk_tokens": CHUNK, "python": sys.version.split()[0],
                "counts": counts, "dropped": drop_counts, "train_rows": len(train), "dev_rows": len(dev),
                "train_sha256": sha_train, "dev_sha256": sha_dev,
                "source_sha256": {n: sha((RAW / SOURCES[n][1]).read_bytes()) for n in SOURCES},
                "canaries_sha256": sha(Path(CANARIES_PATH).read_bytes()), "ppl_sha256": sha(Path(PPL_PATH).read_bytes())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
