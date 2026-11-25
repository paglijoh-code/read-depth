"""Build data/canaries500.jsonl (500 multiple-choice canaries) and data/ppl_heldout.txt, deterministically.

    uv run python scripts/build_canaries.py

Canaries: 100 items in each of syntax, factual, reasoning, instruction, code. Procedural generation with fixed
seeds; code output answers come from executing the snippet; factual items are hand-written. The perplexity text
is a slice of Project Gutenberg #1342 (public domain), the download pinned by sha256.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "canaries500.jsonl"
PPL_OUT = ROOT / "data" / "ppl_heldout.txt"
SEED = 20260101
N_PER_CAT = 100

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
GUTENBERG_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"
PPL_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
PPL_SKIP_CHARS = 20000   # skip front matter / table of contents region after the marker
PPL_CHARS = 200000


def item(cat: str, idx: int, rng: random.Random, prompt: str, correct: str, wrong: list[str], source: str) -> dict:
    opts = [correct] + list(wrong)
    assert len(set(opts)) == len(opts) and 2 <= len(opts) <= 4, (prompt, opts)
    rng.shuffle(opts)
    return {"id": f"{cat}-{idx:03d}", "category": cat, "prompt": prompt, "options": opts,
            "answer": opts.index(correct), "source": source}


def add(items: list[dict], it: dict) -> None:
    """Append unless an item with the same (prompt, sorted options) exists; callers loop until full."""
    key = (it["prompt"], tuple(sorted(it["options"])))
    if key not in {(i["prompt"], tuple(sorted(i["options"]))) for i in items}:
        items.append(it)


def nums_near(rng: random.Random, v: int, k: int = 2) -> list[str]:
    out: set[int] = set()
    while len(out) < k:
        d = rng.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10])
        if v + d != v and v + d >= 0:
            out.add(v + d)
    return [str(x) for x in sorted(out)]


# ---------------------------------------------------------------- syntax
def balanced(s: str) -> bool:
    st, pairs = [], {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in "([{":
            st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]:
                return False
    return not st


def rand_balanced(rng: random.Random, pairs: int) -> str:
    if pairs == 0:
        return ""
    k = rng.randrange(pairs)
    o, c = rng.choice(["()", "[]", "{}"])
    return o + rand_balanced(rng, k) + c + rand_balanced(rng, pairs - 1 - k)


def break_brackets(rng: random.Random, s: str) -> str:
    while True:
        t = list(s)
        kind = rng.choice(["swap", "drop", "flip"])
        i = rng.randrange(len(t))
        if kind == "drop":
            del t[i]
        elif kind == "swap":
            t[i] = rng.choice("()[]{}")
        else:
            t.insert(i, rng.choice(")]}"))
        r = "".join(t)
        if not balanced(r):
            return r


def json_valid(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


KEYS = ["name", "id", "age", "city", "tags", "price", "active", "count", "color", "size"]
VALS = ['"red"', "42", "true", "[1, 2, 3]", '"Paris"', "3.5", "null", '["a", "b"]', "7", '"north"']


def rand_json(rng: random.Random) -> str:
    ks = rng.sample(KEYS, rng.randint(2, 3))
    return "{" + ", ".join(f'"{k}": {rng.choice(VALS)}' for k in ks) + "}"


def break_json(rng: random.Random, s: str) -> str:
    while True:
        kind = rng.choice(["single", "trail", "unquoted", "missing_colon", "py_bool"])
        if kind == "single":
            r = s.replace('"', "'")
        elif kind == "trail":
            r = s[:-1] + ",}"
        elif kind == "unquoted":
            m = re.search(r'"(\w+)":', s)
            r = s[:m.start()] + m.group(1) + ":" + s[m.end():]
        elif kind == "missing_colon":
            r = re.sub(r'("\w+"): ', r"\1 ", s, count=1)
        else:
            r = s.replace("true", "True").replace("null", "None")
        if r != s and not json_valid(r):
            return r


PY_TEMPLATES = [
    "def {f}({a}):\n    return {a} + {n}",
    "for {a} in range({n}):\n    print({a})",
    "if {a} > {n}:\n    print({a})",
    "print([i * {n} for i in range(5)])",
    "while {a} < {n}:\n    {a} += 1",
    "def {f}({a}, b):\n    if {a} > b:\n        return {a}\n    return b",
    "class {C}:\n    def {f}(self):\n        return {n}",
    "{a} = {{'k': {n}}}\nprint({a}['k'])",
]


def rand_py(rng: random.Random) -> str:
    t = rng.choice(PY_TEMPLATES)
    return t.format(f=rng.choice(["add", "run", "calc", "go", "show"]), a=rng.choice(["x", "n", "val", "item"]),
                    n=rng.randint(2, 9), C=rng.choice(["Box", "Item", "Node"]))


def break_py(rng: random.Random, s: str) -> str:
    while True:
        kind = rng.choice(["colon", "paren", "indent", "kw"])
        if kind == "colon" and ":\n" in s:
            r = s.replace(":\n", "\n", 1)
        elif kind == "paren" and ")" in s:
            i = s.index(")")
            r = s[:i] + s[i + 1:]
        elif kind == "indent" and "\n    " in s:
            r = s.replace("\n    ", "\n", 1)
        elif kind == "kw":
            r = s.replace("return", "retrun", 1) if "return" in s else s.replace("print", "prnt)", 1)
        else:
            continue
        try:
            ast.parse(r)
        except (SyntaxError, IndentationError):
            return r


FORMAT_KINDS = {
    "an ISO 8601 date (YYYY-MM-DD)": (r"\d{4}-\d{2}-\d{2}", lambda r: f"{r.randint(1990, 2030)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
                                      lambda r, g: r.choice([g.replace("-", "/"), g.replace("-", ".", 1), g[2:], g.replace("-", "")[:8]])),
    "a six-digit hex color code": (r"#[0-9a-f]{6}", lambda r: "#" + "".join(r.choice("0123456789abcdef") for _ in range(6)),
                                   lambda r, g: r.choice([g[1:], "#" + g[1:4], "#" + g[1:6] + "g", "0x" + g[1:]])),
    "a dotted-quad IPv4 address": (r"(\d{1,3}\.){3}\d{1,3}", lambda r: ".".join(str(r.randint(1, 254)) for _ in range(4)),
                                   lambda r, g: r.choice([g.rsplit(".", 1)[0], g.replace(".", ",", 1), g.replace(".", "..", 1), g + ".7.1"])),
    "a valid email address": (r"[a-z]+@[a-z]+\.[a-z]{2,3}", lambda r: f"{r.choice(['sam', 'lee', 'ann', 'joe', 'kim'])}@{r.choice(['mail', 'site', 'corp'])}.{r.choice(['com', 'org', 'net'])}",
                              lambda r, g: r.choice([g.replace("@", ""), g.replace("@", "@@"), g.replace(".", "", 1) if g.count(".") else g, g.replace("@", " at ")])),
}


def build_syntax(rng: random.Random) -> list[dict]:
    items, n = [], 0
    for q in range(100):
        kind = q % 4
        if kind == 0:
            ok = rand_balanced(rng, rng.randint(2, 4))
            bad = set()
            while len(bad) < 2:
                bad.add(break_brackets(rng, ok))
            assert balanced(ok)
            it = item("syntax", n, rng, "Which of these strings has balanced brackets?", ok, sorted(bad), "procedural:brackets")
        elif kind == 1:
            ok = rand_json(rng)
            bad = set()
            while len(bad) < 2:
                bad.add(break_json(rng, ok))
            assert json_valid(ok)
            it = item("syntax", n, rng, "Which of these is valid JSON?", ok, sorted(bad), "procedural:json")
        elif kind == 2:
            ok = rand_py(rng)
            bad = set()
            while len(bad) < 2:
                bad.add(break_py(rng, ok))
            ast.parse(ok)
            it = item("syntax", n, rng, "Which of these is valid Python code?", ok, sorted(bad), "procedural:python")
        else:
            desc = rng.choice(list(FORMAT_KINDS))
            pat, gen, brk = FORMAT_KINDS[desc]
            ok = gen(rng)
            bad = set()
            while len(bad) < 2:
                b = brk(rng, ok)
                if b != ok and not re.fullmatch(pat, b):
                    bad.add(b)
            assert re.fullmatch(pat, ok)
            it = item("syntax", n, rng, f"Which of these is {desc}?", ok, sorted(bad), "procedural:format")
        items.append(it)
        n += 1
    return items


# ---------------------------------------------------------------- factual (hand-written)
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
            ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Russia", "Moscow"), ("China", "Beijing"), ("India", "New Delhi"),
            ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Kenya", "Nairobi"),
            ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
            ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("Thailand", "Bangkok"), ("Peru", "Lima"),
            ("Cuba", "Havana"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
            ("South Korea", "Seoul"), ("Chile", "Santiago"), ("Vietnam", "Hanoi"), ("Morocco", "Rabat")]
ELEMENTS = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("sodium", "Na"), ("potassium", "K"), ("lead", "Pb"),
            ("copper", "Cu"), ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"), ("nitrogen", "N"),
            ("helium", "He"), ("calcium", "Ca"), ("tin", "Sn"), ("mercury", "Hg"), ("zinc", "Zn"),
            ("sulfur", "S"), ("chlorine", "Cl"), ("neon", "Ne"), ("magnesium", "Mg"), ("nickel", "Ni"),
            ("aluminum", "Al"), ("silicon", "Si"), ("phosphorus", "P")]
# (question, answer, [two wrong answers])
FACTS = [
    ("What is the largest planet in the Solar System?", "Jupiter", ["Saturn", "Mars"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Earth"]),
    ("Which planet is known as the Red Planet?", "Mars", ["Neptune", "Mercury"]),
    ("Which planet has the most prominent rings?", "Saturn", ["Mars", "Venus"]),
    ("What is the chemical formula of water?", "H2O", ["CO2", "NaCl"]),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide", ["helium", "methane"]),
    ("What is the hardest natural substance?", "diamond", ["quartz", "granite"]),
    ("How many days are in a leap year?", "366", ["365", "364"]),
    ("How many minutes are in an hour?", "60", ["100", "30"]),
    ("How many sides does a hexagon have?", "6", ["5", "8"]),
    ("How many continents are there?", "7", ["5", "9"]),
    ("How many legs does a spider have?", "8", ["6", "10"]),
    ("How many degrees are in a right angle?", "90", ["180", "45"]),
    ("What is the largest ocean on Earth?", "the Pacific Ocean", ["the Atlantic Ocean", "the Indian Ocean"]),
    ("What is the longest river in Africa?", "the Nile", ["the Congo", "the Niger"]),
    ("Which is the tallest mountain above sea level?", "Mount Everest", ["Mount Fuji", "Mont Blanc"]),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare", ["Charles Dickens", "Mark Twain"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Pablo Picasso", "Vincent van Gogh"]),
    ("Who developed the theory of general relativity?", "Albert Einstein", ["Isaac Newton", "Niels Bohr"]),
    ("Who was the first President of the United States?", "George Washington", ["Abraham Lincoln", "Thomas Jefferson"]),
    ("Who composed the Ninth Symphony with the Ode to Joy?", "Ludwig van Beethoven", ["Johann Sebastian Bach", "Franz Schubert"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie"]),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100", ["90", "212"]),
    ("What is the freezing point of water in degrees Celsius?", "0", ["32", "-10"]),
    ("What is the SI unit of force?", "the newton", ["the joule", "the watt"]),
    ("What is the SI unit of electrical resistance?", "the ohm", ["the volt", "the ampere"]),
    ("What is the SI unit of frequency?", "the hertz", ["the pascal", "the tesla"]),
    ("How many meters are in a kilometer?", "1000", ["100", "10000"]),
    ("How many grams are in a kilogram?", "1000", ["100", "10"]),
    ("Which organ pumps blood around the human body?", "the heart", ["the liver", "the lung"]),
    ("Which blood cells carry oxygen?", "red blood cells", ["white blood cells", "platelets"]),
    ("How many bones are in the adult human body?", "206", ["106", "306"]),
    ("Which vitamin is produced in the skin by sunlight?", "vitamin D", ["vitamin C", "vitamin K"]),
    ("What language is primarily spoken in Brazil?", "Portuguese", ["Spanish", "French"]),
    ("What language is primarily spoken in Egypt?", "Arabic", ["Turkish", "Persian"]),
    ("In which year did World War II end?", "1945", ["1918", "1939"]),
    ("In which year did humans first land on the Moon?", "1969", ["1959", "1979"]),
    ("In which year did the Berlin Wall fall?", "1989", ["1961", "1991"]),
    ("Which empire built the Colosseum in Rome?", "the Roman Empire", ["the Ottoman Empire", "the Persian Empire"]),
    ("Which ancient civilization built the pyramids at Giza?", "the Egyptians", ["the Romans", "the Aztecs"]),
    ("What is the currency of Japan?", "the yen", ["the won", "the peso"]),
    ("What is the currency of the United Kingdom?", "the pound sterling", ["the euro", "the franc"]),
    ("Which animal is the largest mammal?", "the blue whale", ["the elephant", "the giraffe"]),
    ("Which animal is the fastest on land?", "the cheetah", ["the lion", "the horse"]),
    ("What do bees produce?", "honey", ["silk", "wax paper"]),
    ("What is a baby kangaroo called?", "a joey", ["a cub", "a calf"]),
    ("Which part of a plant absorbs water from the soil?", "the roots", ["the leaves", "the petals"]),
    ("What is the process by which water vapor becomes liquid called?", "condensation", ["evaporation", "sublimation"]),
    ("What force pulls objects toward the Earth?", "gravity", ["magnetism", "friction"]),
    ("Which gas makes up most of Earth's atmosphere?", "nitrogen", ["oxygen", "argon"]),
    ("What is the center of an atom called?", "the nucleus", ["the electron", "the shell"]),
    ("What is the square root of 81?", "9", ["8", "7"]),
    ("How many letters are in the English alphabet?", "26", ["24", "28"]),
    ("Which sport uses a shuttlecock?", "badminton", ["tennis", "squash"]),
    ("What is the first month of the year?", "January", ["March", "December"]),
    ("Which metal is liquid at room temperature?", "mercury", ["iron", "copper"]),
    ("Which ocean lies between Africa and Australia?", "the Indian Ocean", ["the Arctic Ocean", "the Atlantic Ocean"]),
    ("Which desert is the largest hot desert in the world?", "the Sahara", ["the Gobi", "the Mojave"]),
    ("What is the largest country in the world by area?", "Russia", ["Canada", "China"]),
    ("Which instrument has black and white keys?", "the piano", ["the flute", "the violin"]),
]


def build_factual(rng: random.Random) -> list[dict]:
    items = []
    caps = CAPITALS[:]
    rng.shuffle(caps)
    els = ELEMENTS[:]
    rng.shuffle(els)
    facts = FACTS[:]
    rng.shuffle(facts)
    n_caps, n_els = 20, 20
    n = 0
    for country, cap in caps[:n_caps]:
        wrong = rng.sample([c for _, c in CAPITALS if c != cap], 2)
        items.append(item("factual", n, rng, f"What is the capital of {country}?", cap, wrong, "handwritten:capitals")); n += 1
    for name, sym in els[:n_els]:
        wrong = rng.sample([s for _, s in ELEMENTS if s != sym], 2)
        items.append(item("factual", n, rng, f"What is the chemical symbol for {name}?", sym, wrong, "handwritten:elements")); n += 1
    for q, a, w in facts[:N_PER_CAT - n]:
        items.append(item("factual", n, rng, q, a, w, "handwritten:facts")); n += 1
    assert len(items) == N_PER_CAT, len(items)
    return items


# ---------------------------------------------------------------- reasoning
NAMES = ["Ann", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jo", "Kai", "Lena", "Mo", "Nia", "Omar", "Pia"]
THINGS = ["apples", "marbles", "stickers", "coins", "pencils", "books", "cards", "shells"]


def build_reasoning(rng: random.Random) -> list[dict]:
    items = []
    for n in range(N_PER_CAT):
        kind = n % 3
        if kind == 0:
            while True:
                v = rng.randint(2, 12)
                text, cur = f"Start with {v}.", v
                for _ in range(3):
                    op = rng.choice(["add", "subtract", "multiply"])
                    if op == "add":
                        k = rng.randint(2, 9); cur += k; text += f" Add {k}."
                    elif op == "subtract":
                        k = rng.randint(2, 9); cur -= k; text += f" Subtract {k}."
                    else:
                        k = rng.randint(2, 4); cur *= k; text += f" Multiply by {k}."
                if cur >= 0:
                    break
            items.append(item("reasoning", n, rng, text + " What is the result?", str(cur), nums_near(rng, cur), "procedural:arith"))
        elif kind == 1:
            k = rng.choice([3, 4])
            names = rng.sample(NAMES, k)  # names[0] is the greatest
            word = rng.choice([("taller than", "tallest", "shortest"), ("older than", "oldest", "youngest"),
                               ("faster than", "fastest", "slowest"), ("heavier than", "heaviest", "lightest")])
            order = names
            facts = [f"{order[i]} is {word[0]} {order[i + 1]}." for i in range(k - 1)]
            rng.shuffle(facts)
            if rng.random() < 0.5:
                q, ans = f"Who is the {word[1]}?", order[0]
            else:
                q, ans = f"Who is the {word[2]}?", order[-1]
            wrong = [x for x in names if x != ans][:2]
            items.append(item("reasoning", n, rng, " ".join(facts) + " " + q, ans, wrong, "procedural:ordering"))
        else:
            a, b, c = rng.randint(10, 30), rng.randint(2, 9), rng.randint(2, 9)
            who, thing = rng.choice(NAMES), rng.choice(THINGS)
            form = rng.randrange(3)
            if form == 0:
                text, ans = f"{who} has {a} {thing}, gives away {b}, then finds {c} more. How many {thing} does {who} have now?", a - b + c
            elif form == 1:
                text, ans = f"{who} buys {b} packs with {c} {thing} in each pack, and already had {a}. How many {thing} does {who} have in total?", a + b * c
            else:
                text, ans = f"{who} has {a * b} {thing} and splits them equally among {b} friends. How many {thing} does each friend get?", a
            items.append(item("reasoning", n, rng, text, str(ans), nums_near(rng, ans), "procedural:word"))
    return items


# ---------------------------------------------------------------- instruction following
BASES = ["The cat sat on the warm windowsill", "We walked along the river after lunch", "Rain fell softly on the old roof",
         "She opened the door and smiled", "Our team finished the report on time", "The market was busy this morning",
         "He plays the guitar every evening", "Fresh bread smells wonderful", "The train arrives at noon today",
         "They planted tomatoes in the garden", "The library closes early on Sundays", "A small bird landed on the fence",
         "Please send the files before Friday", "The soup needs a little more salt", "Children laughed in the park nearby",
         "The museum opens a new exhibit tomorrow"]
INSTR_KINDS = ["lower", "upper", "words", "question", "exclaim", "start", "keyword", "bullets", "ends_period", "quotes"]


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", s)


def build_instruction(rng: random.Random) -> list[dict]:
    items = []
    tries = 0
    while len(items) < N_PER_CAT:
        n, tries = len(items), tries + 1
        assert tries < 5000, "template capacity too small"
        kind = INSTR_KINDS[n % len(INSTR_KINDS)]
        base = rng.choice(BASES)
        w = base.split()
        if kind == "lower":
            p, ok, bad = "Reply entirely in lowercase letters.", base.lower() + ".", [base + ".", base.upper() + "."]
        elif kind == "upper":
            p, ok, bad = "Reply entirely in capital letters.", base.upper() + ".", [base + ".", base.lower() + "."]
        elif kind == "words":
            k = rng.randint(3, 5)
            base = rng.choice([b for b in BASES if len(b.split()) >= k + 2])
            w = base.split()
            p = f"Reply with exactly {k} words."
            ok, bad = " ".join(w[:k]), [" ".join(w[:k + 2]), " ".join(w[:k - 1])]
            assert len(words(ok)) == k
        elif kind == "question":
            p, ok, bad = "Reply with a question ending in a question mark.", "Is it true that " + base[0].lower() + base[1:] + "?", [base + ".", base + "!"]
        elif kind == "exclaim":
            p, ok, bad = "Reply with a sentence ending in an exclamation mark.", base + "!", [base + ".", base + "?"]
        elif kind == "start":
            lead = rng.choice(["Sure", "Certainly", "Absolutely"])
            p = f"Begin your reply with the word \"{lead}\"."
            others = [x for x in ["Sure", "Certainly", "Absolutely", "Maybe", "Well", "Perhaps"] if x != lead]
            ok, bad = f"{lead}, {base[0].lower() + base[1:]}.", [f"{o}, {base[0].lower() + base[1:]}." for o in rng.sample(others, 2)]
        elif kind == "keyword":
            kw = rng.choice([k for k in ["blue", "garden", "quickly", "tomorrow"] if k not in base])
            p = f"Include the word \"{kw}\" in your reply."
            ok = f"{base} {kw}."
            bad = [f"{base} today.", f"{base} again."]
            assert kw not in bad[0] + bad[1]
        elif kind == "bullets":
            k = rng.choice([2, 3])
            p = f"Reply with a list of exactly {k} bullet points, each starting with \"- \"."
            ok = "\n".join(f"- {x}" for x in rng.sample(BASES, k))
            bad = ["\n".join(f"- {x}" for x in rng.sample(BASES, k + 1)), "\n".join(f"- {x}" for x in rng.sample(BASES, k - 1))]
        elif kind == "ends_period":
            p, ok, bad = "Reply with a statement and do not use any exclamation marks.", base + ".", [base + "!", base + "!!"]
        else:
            p = "Reply with a sentence wrapped in double quotation marks."
            ok, bad = f"\"{base}.\"", [f"{base}.", f"'{base}.'"]
        add(items, item("instruction", n, rng, f"Instruction: {p}\nWhich response follows the instruction?", ok, bad, f"procedural:{kind}"))
    return items


# ---------------------------------------------------------------- code
def run_py(code: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return buf.getvalue().strip()


def out_snippets(rng: random.Random) -> tuple[str, list[str]]:
    """Return (correct snippet, list of variant snippets that differ in one detail)."""
    k = rng.randrange(8)
    a, b, c = rng.randint(1, 9), rng.randint(2, 9), rng.randint(1, 9)
    word = rng.choice(["python", "garden", "window", "bridge", "planet", "silver"])
    lst = [rng.randint(1, 9) for _ in range(5)]
    if k == 0:
        fmt = "x = {a}\ny = {b}\nprint(x * y + {c})"
        vs = [dict(a=a, b=b, c=c), dict(a=a, b=b, c=c + 1), dict(a=a + 1, b=b, c=c), dict(a=a, b=b + 1, c=c)]
    elif k == 1:
        fmt = "s = '{w}'\nprint(s[{i}:{j}])"
        vs = [dict(w=word, i=1, j=4), dict(w=word, i=0, j=3), dict(w=word, i=2, j=5), dict(w=word, i=1, j=3)]
    elif k == 2:
        fmt = "nums = {l}\nprint(sum(nums[{i}:]))"
        vs = [dict(l=lst, i=1), dict(l=lst, i=2), dict(l=lst, i=3), dict(l=lst, i=0)]
    elif k == 3:
        fmt = "s = '{w}'\nprint(s.{m}())"
        vs = [dict(w=word, m="upper"), dict(w=word, m="title"), dict(w=word, m="capitalize"), dict(w=word[::-1], m="upper")]
        fmt = "s = '{w}'\nprint(s.{m}())"
    elif k == 4:
        fmt = "nums = {l}\nprint(max(nums) - min(nums) + {a})"
        vs = [dict(l=lst, a=a), dict(l=lst, a=a + 1), dict(l=lst, a=a + 2), dict(l=lst + [0], a=a + 3)]
    elif k == 5:
        fmt = "print(len([i for i in range({a}) if i % {b} == 0]))"
        vs = [dict(a=a + 10, b=b), dict(a=a + 11, b=b), dict(a=a + 10, b=b + 1), dict(a=a + 12, b=b)]
    elif k == 6:
        fmt = "d = {{'a': {a}, 'b': {b}}}\nd['c'] = d['a'] + d['b']\nprint(d['c'] * {c})"
        vs = [dict(a=a, b=b, c=c), dict(a=a, b=b, c=c + 1), dict(a=a + 1, b=b, c=c), dict(a=a, b=b + 2, c=c)]
    else:
        fmt = "print('{w}'.count('{ch}') + len('{w}'))"
        ch = rng.choice(word)
        vs = [dict(w=word, ch=ch), dict(w=word + 'x', ch=ch), dict(w=word[:-1], ch=ch), dict(w=word, ch="z")]
    snips = [fmt.format(**v) for v in vs]
    return snips[0], snips[1:]


def build_expr(rng: random.Random) -> tuple[int, str, list[str]]:
    while True:
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
        exprs = [f"{a} + {b} * {c}", f"({a} + {b}) * {c}", f"{a} * {b} - {c}", f"{a} * {b} + {c}", f"{a} ** 2 - {b}",
                 f"{a * b} // {c}", f"{a * b} % {c + 3}", f"({a} - {b}) * {c}"]
        rng.shuffle(exprs)
        vals = [eval(e) for e in exprs]
        pick = [i for i in range(len(exprs)) if vals.count(vals[i]) == 1 and vals[i] >= 0]
        if pick and len(exprs) >= 3:
            i = pick[0]
            others = [e for j, e in enumerate(exprs) if j != i and vals[j] != vals[i]]
            if len(others) >= 2:
                return vals[i], exprs[i], others[:2]


COMPLEXITY = [
    ("def f(items):\n    for x in items:\n        print(x)", "O(n)"),
    ("def f(items):\n    return items[0]", "O(1)"),
    ("def f(items):\n    for x in items:\n        for y in items:\n            print(x, y)", "O(n^2)"),
    ("def f(n):\n    while n > 1:\n        n = n // 2", "O(log n)"),
    ("def f(items):\n    return sum(items)", "O(n)"),
    ("def f(items):\n    return len(items) + 5", "O(1)"),
    ("def f(n):\n    i = 1\n    while i < n:\n        i *= 2", "O(log n)"),
    ("def f(grid):\n    for row in grid:\n        for cell in row:\n            print(cell)", "O(n^2)"),
]
CX_OPTIONS = ["O(1)", "O(log n)", "O(n)", "O(n^2)"]


def build_code(rng: random.Random) -> list[dict]:
    items = []
    tries = 0
    while len(items) < N_PER_CAT:
        n, tries = len(items), tries + 1
        assert tries < 5000, "template capacity too small"
        if n % 4 in (0, 1):
            while True:
                good, variants = out_snippets(rng)
                gout = run_py(good)
                wrong = []
                for v in variants:
                    o = run_py(v)
                    if o != gout and o not in wrong:
                        wrong.append(o)
                if len(wrong) >= 2:
                    break
            add(items, item("code", n, rng, f"What does this Python code print?\n{good}", gout, wrong[:2], "procedural:exec"))
        elif n % 4 == 2:
            v, good, others = build_expr(rng)
            add(items, item("code", n, rng, f"Which Python expression evaluates to {v}?", good, others, "procedural:eval"))
        else:
            code, ans = COMPLEXITY[(n // 4) % len(COMPLEXITY)]
            # vary names to avoid byte-identical prompts
            nm = ["items", "data", "seq", "arr", "values", "xs", "lst", "vals", "nums", "rows"][(n // 4) // len(COMPLEXITY) % 10]
            code = code.replace("items", nm).replace("def f(", f"def {rng.choice(['f', 'g', 'h', 'process', 'scan'])}(")
            wrong = [o for o in CX_OPTIONS if o != ans]
            rng.shuffle(wrong)
            add(items, item("code", n, rng, f"What is the time complexity of this function?\n{code}", ans, wrong[:2], "procedural:complexity"))
    return items


# ---------------------------------------------------------------- perplexity text
def build_ppl() -> None:
    req = urllib.request.Request(GUTENBERG_URL, headers={"User-Agent": "surgery-build/1.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != GUTENBERG_SHA256:
        sys.exit(f"gutenberg download sha256 {got} != pinned {GUTENBERG_SHA256}")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    i = text.index(PPL_START_MARKER)
    i = text.index("\n", i) + 1 + PPL_SKIP_CHARS
    chunk = text[i:i + PPL_CHARS]
    chunk = chunk[:chunk.rindex("\n\n")].strip() + "\n"
    PPL_OUT.write_text(chunk, encoding="utf-8")


def main() -> None:
    items = []
    for name, fn in [("syntax", build_syntax), ("factual", build_factual), ("reasoning", build_reasoning),
                     ("instruction", build_instruction), ("code", build_code)]:
        cat = fn(random.Random(f"{SEED}-{name}"))
        assert len(cat) == N_PER_CAT, (name, len(cat))
        assert len({(i["prompt"], tuple(sorted(i["options"]))) for i in cat}) == N_PER_CAT, f"duplicate items in {name}"
        items += cat
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    if "--no-ppl" not in sys.argv:
        build_ppl()
    print("canaries sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
    print("ppl text sha256", hashlib.sha256(PPL_OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
