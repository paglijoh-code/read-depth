"""M9 atlas: the 14 tagged tasks, their fixed option sets and prompts (docs/SPEC-research.md, section 3).

This file is the interface between the task generators (`surgery/atlas/tasks.py`) and the
measurements (`surgery/atlas/measure.py`). It is fixed before any model run; changing an option set,
a system prompt or a split size after a run opens a new study (CONTRACT.md, "Locked").

Every task has ONE fixed option list. A record's `answer` is an index into that list. The model is
scored on the first token of its answer, restricted to the first tokens of the options, which must be
distinct (checked by `option_token_ids`). The question lives in the system message and the item lives
in the user message, so the two tasks of a matched pair see identical user text.

Record schema (one JSON object per line in var/data/atlas/{task}_{split}.jsonl):
    id           str   unique within the task; the two tasks of a matched pair use the same id per base item
    task         str   key of TASKS
    prompt       str   user message text
    answer       int   index into TASKS[task].options
    template_id  str   grouping key; no template appears in two splits of one task (or of its pair)
    rung         str | None   ladder step for tasks 8, 9, 11 (e.g. "H1", "ops1", "depth2"); None otherwise
    meta         dict  generator details, free-form, never shown to the model
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path("var/data/atlas")
SPLITS = {"discovery": 1000, "train": 800, "lrsel": 200, "selection": 500, "test": 500}
SPLIT_SEED = 9          # never 0 or 1: those are M2's seeds and their test splits are used
MIN_TEMPLATES_PER_SPLIT = 20
TEMPLATE_CAP = 0.05     # at most this share of one split from one template
ROUTING_MAX_TOKENS = 512  # user text cut from the oldest end, as the routing studies did


@dataclass(frozen=True)
class TaskSpec:
    number: int
    name: str
    kind: str           # appraise | execute
    needs: str          # surface | procedure | fact
    steps: str          # serial steps, as in the spec's table
    options: tuple[str, ...]
    system: str
    pair: str | None = None          # name of the matched-pair partner, if any
    rungs: tuple[str, ...] = field(default_factory=tuple)


def _sys(question: str, options: tuple[str, ...]) -> str:
    return (f"{question}\nAnswer with exactly one of: {', '.join(options)}. Output only the answer.")


_ROUTING_HEAD = ("You read a conversation and answer one question about its latest request, "
                 "judging from the whole conversation.\nQuestion: ")
_JUDGEMENT_Q = ("Does a good result depend on working out what to do (a recommendation, a choice, a review, "
                "an unknown cause)? YES, or NO if it is carrying out something already decided or has one findable answer.")
_WORK_Q = ("How much does the request set off, whatever the topic? LIGHT is a few steps at most; HEAVY is several "
           "steps, each depending on the last. If torn, HEAVY.")

# The planner prompt of M2-M7, unchanged, so tasks 6-8 stay comparable with earlier milestones.
from surgery.prompt import SYSTEM as PLANNER_SYSTEM  # noqa: E402
from surgery.task.generator import LABELS as RULE_LABELS  # noqa: E402
from surgery.task.generator import HARD_LABELS  # noqa: E402

DIGITS = tuple(str(i) for i in range(10))
STEPS_CONCEPTS = ("tree", "bush", "cactus", "fern", "grass", "vine", "log", "flower", "moss", "seaweed")

_T = [
    TaskSpec(1, "route_judgement", "appraise", "surface", "0", ("YES", "NO"),
             _ROUTING_HEAD + _JUDGEMENT_Q + "\nAnswer with exactly one word: YES, NO."),
    TaskSpec(2, "route_work", "appraise", "surface", "0", ("LIGHT", "HEAVY"),
             _ROUTING_HEAD + _WORK_Q + "\nAnswer with exactly one word: LIGHT, HEAVY."),
    TaskSpec(3, "code_shape", "appraise", "surface", "0", ("loop", "branch", "neither"),
             _sys("You are shown a short Python program. Does it contain a loop (for or while), a branch "
                  "(if), or neither? No program contains both.", ("loop", "branch", "neither")),
             pair="code_output"),
    TaskSpec(4, "fact_topic", "appraise", "surface", "0", ("geography", "history", "science", "other"),
             _sys("You are shown a yes-or-no question. Which subject is it about? Do not answer the question itself.",
                  ("geography", "history", "science", "other")),
             pair="fact_check"),
    TaskSpec(5, "hard_hops", "appraise", "surface", "0", ("0", "1", "2"),
             _sys("You are shown the state of a coding session and a numbered list of workflow rules, some of which "
                  "use derived flags. How long is the longest chain of flags defined from other flags? 0 means there "
                  "are no flags, 1 means flags are defined only from facts, 2 means some flag is defined from another "
                  "flag. Do not apply the rules.", ("0", "1", "2")),
             pair="hard"),
    TaskSpec(6, "rule", "execute", "procedure", "1", tuple(RULE_LABELS), PLANNER_SYSTEM),
    TaskSpec(7, "sensible", "execute", "procedure", "1", tuple(RULE_LABELS), PLANNER_SYSTEM),
    # The capability study scored `hard` under the same planner prompt (scripts/capability_sweep.py).
    TaskSpec(8, "hard", "execute", "procedure", "1/2/3", tuple(HARD_LABELS), PLANNER_SYSTEM,
             pair="hard_hops", rungs=("H1", "H2", "H3")),
    TaskSpec(9, "arith", "execute", "procedure", "1/2", DIGITS,
             _sys("You are shown an arithmetic expression. What is its value? The value is always a single digit.",
                  DIGITS), pair="arith_shape", rungs=("ops1", "ops2")),
    TaskSpec(10, "fact_check", "execute", "fact", "0", ("yes", "no"),
             _sys("You are shown a yes-or-no question about a fact. Is the statement in it true?", ("yes", "no")),
             pair="fact_topic"),
    TaskSpec(11, "syntax_valid", "execute", "surface", "1-3", ("yes", "no"),
             _sys("You are shown a string of brackets or a JSON value. Is it well-formed: every bracket closed by the "
                  "matching kind in the right order, and for JSON, valid JSON?", ("yes", "no")),
             pair="syntax_depth", rungs=("depth1", "depth2", "depth3")),
    TaskSpec(12, "code_output", "execute", "procedure", "2", DIGITS,
             _sys("You are shown a short Python program. What does it print? It always prints a single digit.",
                  DIGITS), pair="code_shape"),
    TaskSpec(13, "arith_shape", "appraise", "surface", "0", ("plus", "minus", "times", "mixed"),
             _sys("You are shown an arithmetic expression. Which operation does it use? Say mixed if it uses two "
                  "different operations. Do not work out its value.", ("plus", "minus", "times", "mixed")),
             pair="arith"),
    TaskSpec(14, "syntax_depth", "appraise", "surface", "0", ("1", "2", "3"),
             _sys("You are shown a string of brackets or a JSON value. What is its greatest nesting depth: the most "
                  "brackets open at once, reading left to right, whether or not the string is well-formed? Do not "
                  "judge whether it is well-formed.", ("1", "2", "3")),
             pair="syntax_valid"),
    # Phase 2 stage A (docs/ATLAS-phase2-stageA.md): four appraisal tasks that hide their answer in order or binding,
    # and three execute tasks that are nearly a copy or a lookup.
    TaskSpec(15, "arith_order", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown an arithmetic expression, written with spaces. Is the first number larger than the last "
                  "number? Do not work out its value.", ("yes", "no"))),
    TaskSpec(16, "syntax_first_close", "appraise", "surface", "0", ("round", "square", "curly"),
             _sys("You are shown a string of brackets, written with spaces. Which kind of bracket is closed first, reading "
                  "left to right: round ( ), square [ ] or curly { }?", ("round", "square", "curly"))),
    TaskSpec(17, "code_uses_first", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown a short Python program. Does its print line use the variable assigned on the first "
                  "line? Do not work out what it prints.", ("yes", "no"))),
    TaskSpec(18, "hard_target_read", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown the state of a coding session. Has the target file been read (read=yes in the file "
                  "list)? Do not decide the next action.", ("yes", "no"))),
    TaskSpec(19, "arith_id", "execute", "procedure", "1", DIGITS,
             _sys("You are shown an arithmetic expression. What is its value? The value is always a single digit.",
                  DIGITS)),
    TaskSpec(20, "code_literal", "execute", "procedure", "1", DIGITS,
             _sys("You are shown a short Python program. What does it print? It always prints a single digit.",
                  DIGITS)),
    TaskSpec(21, "arith_long", "execute", "procedure", "1/2", DIGITS,
             _sys("You are shown an arithmetic expression. What is its value?", DIGITS)),
    # Phase 2 stage C (docs/ATLAS-phase2-stageC.md): four appraisal tasks that need binding across distance.
    TaskSpec(22, "code_traces_first", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown a short Python program. Does the value printed come from the constant assigned on the "
                  "first line? Do not work out what it prints.", ("yes", "no"))),
    TaskSpec(23, "syntax_outer_pair", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown a string of brackets, written with spaces. Does the first bracket close at the very end "
                  "of the string?", ("yes", "no"))),
    TaskSpec(24, "target_read_bare", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown the state of a coding session. Has the target file been read (read=yes in the file "
                  "list)? Do not decide the next action.", ("yes", "no"))),
    TaskSpec(25, "bind_compare", "appraise", "surface", "0", ("yes", "no"),
             _sys("You are shown some variables and a question about two of them. Is the first variable named in the "
                  "question larger than the second? Do not add anything up.", ("yes", "no"))),
    # Robustness E2 (docs/ATLAS-robustness.md): the do-task's exact input plus one proposed answer; is it correct?
    TaskSpec(26, "rule_check", "appraise", "procedure", "1", ("yes", "no"),
             _sys("You are the planner of a coding agent. You are given the state of a coding session and a numbered "
                  "list of workflow rules, highest priority first, then a proposed next action. The correct action is "
                  "the one given by the first rule whose condition holds. Is the proposed action correct?", ("yes", "no"))),
    TaskSpec(27, "hard_check", "appraise", "procedure", "1/2/3", ("yes", "no"),
             _sys("You are the planner of a coding agent. You are given the state of a coding session, some named flags "
                  "and a numbered list of workflow rules, highest priority first, then a proposed next action. The "
                  "correct action is the one given by the first rule whose condition holds. Is the proposed action "
                  "correct?", ("yes", "no")), rungs=("H1", "H2", "H3")),
    TaskSpec(28, "arith_check", "appraise", "procedure", "1/2", ("yes", "no"),
             _sys("You are shown an arithmetic expression and a proposed value. Is the proposed value the value of the "
                  "expression?", ("yes", "no")), rungs=("ops1", "ops2")),
    TaskSpec(29, "code_output_check", "appraise", "procedure", "2", ("yes", "no"),
             _sys("You are shown a short Python program and a proposed output. Is the proposed output what the program "
                  "prints?", ("yes", "no"))),
    TaskSpec(30, "code_literal_check", "appraise", "procedure", "1", ("yes", "no"),
             _sys("You are shown a short Python program and a proposed output. Is the proposed output what the program "
                  "prints?", ("yes", "no"))),
    # E2 revision R1 (docs/ATLAS-robustness.md): binary doing tasks, the do-task's exact input plus "Is the answer one of: A?"
    # (A = a fixed half of the do-task's options, drawn with seed 0; no proposal). Same format as the check tasks, still doing.
    TaskSpec(31, "rule_bin", "execute", "procedure", "1", ("yes", "no"),
             _sys("You are the planner of a coding agent. You are given the state of a coding session and a numbered "
                  "list of workflow rules, highest priority first, then a list of actions. The correct action is the one "
                  "given by the first rule whose condition holds. Is the correct action one of the listed actions?",
                  ("yes", "no"))),
    TaskSpec(32, "hard_bin", "execute", "procedure", "1/2/3", ("yes", "no"),
             _sys("You are the planner of a coding agent. You are given the state of a coding session, some named flags "
                  "and a numbered list of workflow rules, highest priority first, then a list of actions. The correct "
                  "action is the one given by the first rule whose condition holds. Is the correct action one of the "
                  "listed actions?", ("yes", "no")), rungs=("H1", "H2", "H3")),
    TaskSpec(33, "arith_bin", "execute", "procedure", "1/2", ("yes", "no"),
             _sys("You are shown an arithmetic expression and a list of digits. Is the value of the expression one of "
                  "the listed digits? The value is always a single digit.", ("yes", "no")), rungs=("ops1", "ops2")),
    TaskSpec(34, "code_output_bin", "execute", "procedure", "2", ("yes", "no"),
             _sys("You are shown a short Python program and a list of digits. Is the digit the program prints one of the "
                  "listed digits? It always prints a single digit.", ("yes", "no"))),
    TaskSpec(35, "code_literal_bin", "execute", "procedure", "1", ("yes", "no"),
             _sys("You are shown a short Python program and a list of digits. Is the digit the program prints one of the "
                  "listed digits? It always prints a single digit.", ("yes", "no"))),
    # Steps study v2 (docs/ATLAS-steps.md): generators in surgery/atlas/steps.py
    TaskSpec(36, "steps_copy", "execute", "procedure", "1-4", DIGITS,
             _sys("You are shown a short Python program. What digit does it print?", DIGITS),
             rungs=("L1_far", "L1_middle", "L1_near", "L2_far", "L2_middle", "L2_near", "L3_far", "L3_middle", "L3_near",
                    "L4_far", "L4_middle", "L4_near", "L1_far_unit", "L1_middle_unit", "L1_near_unit", "L4_far_unit",
                    "L4_middle_unit", "L4_near_unit", "L1_far_pad_keep", "L1_far_pad")),
    TaskSpec(37, "steps_ops", "execute", "procedure", "1-4", DIGITS,
             _sys("You are shown two arithmetic expressions, A and B, on separate lines, and a question naming one of "
                  "them. What is the value of that expression? The value is always a single digit.", DIGITS),
             rungs=("s1", "s2", "s3", "s4")),
    TaskSpec(38, "steps_describe", "execute", "fact", "1-4", STEPS_CONCEPTS,
             _sys("Which of these is being described?", STEPS_CONCEPTS),
             rungs=tuple(f"n{n}_{p}" for n in (1, 2, 3, 4) for p in ("far", "near"))),
]
TASKS: dict[str, TaskSpec] = {t.name: t for t in _T}
# The 14 tasks of Phases 0 and 1, in number order: everything that reproduces those phases iterates over this.
PHASE1_TASKS = tuple(t.name for t in sorted(_T, key=lambda t: t.number) if t.number <= 14)
PAIRS = [("code_shape", "code_output"), ("fact_topic", "fact_check"), ("hard_hops", "hard"),
         ("arith_shape", "arith"), ("syntax_depth", "syntax_valid")]  # (appraise, execute)
assert sorted(t.number for t in TASKS.values()) == list(range(1, 39))
for a, b in PAIRS:
    assert TASKS[a].pair == b and TASKS[b].pair == a and TASKS[a].kind == "appraise" and TASKS[b].kind == "execute"


def build_prompt(tok, task: str, user_text: str) -> str:
    """Chat template, thinking off: system = the task's question, user = the item."""
    return tok.apply_chat_template(
        [{"role": "system", "content": TASKS[task].system}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def option_token_ids(tok, task: str) -> list[int]:
    """First token of each option, tokenized as the start of the answer (no leading space). Raises unless distinct."""
    opts = TASKS[task].options
    ids = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{task}: options share a first token: {dict(zip(opts, ids))}")
    return ids


def chance(answers: list[int]) -> float:
    """Frequency of the commonest option (the spec's definition of chance)."""
    from collections import Counter
    return max(Counter(answers).values()) / len(answers)


def data_path(task: str, split: str, root: Path = DATA_DIR) -> Path:
    return Path(root) / f"{task}_{split}.jsonl"


def read_records(task: str, split: str, root: Path = DATA_DIR) -> list[dict]:
    with open(data_path(task, split, root), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Steps study v3 (docs/ATLAS-steps3.md, Stage 2): generators in surgery/atlas/steps3.py
STEPS3_LETTERS = tuple("ABCDEFGHIJ")
_T3 = [
    TaskSpec(39, "steps3_swap", "execute", "procedure", "1-4", STEPS3_LETTERS,
             _sys("You are shown a row of ten slots numbered 0 to 9, each holding a different letter, then swap "
                  "instructions applied one after another in the order given, then a question about one slot. Which "
                  "letter is in that slot after all the swaps?", STEPS3_LETTERS),
             rungs=("m1", "m2", "m3", "m4")),
    TaskSpec(40, "steps3_total", "execute", "procedure", "1-4", DIGITS,
             _sys("You are shown a short Python program. What digit does it print?", DIGITS),
             rungs=("m1", "m2", "m3", "m4")),
    TaskSpec(41, "steps3_order", "execute", "procedure", "1-4", DIGITS,
             _sys("You are shown a short Python program. What digit does it print?", DIGITS),
             rungs=("m1", "m2", "m3", "m4")),
    TaskSpec(42, "steps3_mix", "execute", "procedure", "1-7", DIGITS,
             _sys("You are shown a short Python program. What digit does it print?", DIGITS),
             rungs=tuple(f"h{h}_m{m}" for h in (0, 2, 4) for m in (1, 2, 3))),
]
TASKS.update({t.name: t for t in _T3})
