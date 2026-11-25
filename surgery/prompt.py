"""Shared prompt construction and single-token label scoring for Qwen3.

Chat template with thinking disabled; the next-token logits at the start of the
assistant answer are restricted to the first token of each label. For the Qwen3
tokenizer the six first tokens are distinct (READ, EDIT, RUN, SEARCH, ASK, DONE),
checked by `label_token_ids`.
"""
from __future__ import annotations

from surgery.task.generator import LABELS

SYSTEM = (
    "You are the planner of a coding agent. You are given the state of a coding session "
    "and a numbered list of workflow rules, highest priority first. Apply the first rule "
    "whose condition holds. Answer with exactly one of: " + ", ".join(LABELS) + ". "
    "Output only the label."
)


def build_prompt(tok, record_prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": record_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def label_token_ids(tok) -> list[int]:
    ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"label first tokens are not distinct: {ids}")
    return ids
