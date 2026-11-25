# Data: generated next-action task

`sample50.jsonl` is 50 hand-auditable examples (fixed seed 12345, committed). The 500-example eval set is not
committed; regenerate it (deterministic):

    uv run python -m surgery.task.generator --n 500 --seed 0 --out var/data/eval500.jsonl
    uv run python -m surgery.task.generator --sample50     # rewrites data/sample50.jsonl

`scripts/sweep_skip.py` generates `var/data/eval500.jsonl` itself if it is missing.

## Record schema (one JSON object per line)

`id`, `prompt` (rendered text the model sees), `label` (one of READ_FILE, EDIT_FILE, RUN_TEST, SEARCH, ASK_USER,
DONE), `wrong_label` (contrastive: action of the highest-precedence firing rule that disagrees with the label, else
the naive-workflow action), `knobs` (generator settings), `state` (structured: `task`, `files`, `tests`, `history`,
`rules` = active rules in precedence order, `disabled` = decoy rules that never fire).

## Decision procedure (surgery/task/generator.py, `next_action`)

1. Derive yes/no status facts (target_read, test_failing, asked_user, repeat_failure, ...) from files, tests, history.
2. Active rules are `if <facts ANDed> then ACTION`, listed highest priority first; actions are random and distinct.
3. The first active rule whose condition holds decides the label; lower firing rules are ignored (precedence).
4. Disabled rules are listed too but never apply. They hold exactly the actions the active rules lack, so each
   action word appears once in every prompt and the bag of words says nothing about the label.

## Prompt and scoring

Qwen3 chat template, `enable_thinking=False`, a system message asking for exactly one label. Score = argmax of the
next-token logits restricted to the first token of each label. These are distinct for the Qwen3 tokenizer:
READ_FILE -> `READ` (9560), EDIT_FILE -> `EDIT` (16949), RUN_TEST -> `RUN` (47390), SEARCH -> `SEARCH` (43545),
ASK_USER -> `ASK` (7384), DONE -> `DONE` (70396); no relabelling was needed. Logit difference = logit(label) -
logit(wrong_label).

## Tuned knobs (the generator defaults) and baseline

| knob | value |
|---|---|
| n_rules (max active rules; count drawn uniformly from min_rules=1 .. n_rules) | 2 |
| cond_size (max facts per condition) | 2 |
| n_distractor_files / n_distractor_tests | 2 / 2 |
| history_extra (repeated reads / re-runs) | 2 |
| conflict_rate (target share of examples whose firing rules disagree; realised ~0.19) | 0.2 |
| n_off_extra (extra decoy rules) | 0 |

Measured on `eval500` (seed 0), unmodified Qwen3-0.6B, fp32 on MPS: **baseline accuracy 0.674** (n=500).
Per label: READ_FILE 0.82, EDIT_FILE 0.58, RUN_TEST 0.69, SEARCH 0.73, ASK_USER 0.77, DONE 0.45.
Label counts: READ_FILE 84, EDIT_FILE 83, RUN_TEST 83, SEARCH 83, ASK_USER 83, DONE 84.
Bag-of-words (uni+bigram LR) 5-fold CV: 0.08 on sample50 (majority 0.18), ~0.17-0.20 on 2000 examples.

## Tuning history (200-500 examples each, baseline accuracy, chance = 0.167)

- Natural-language rules with fixed condition->action pairs: BoW 0.48 (state alone predicts the label); rejected.
- Random actions per rule, 8 rules: model 0.16-0.28 (cannot apply 6+ rules). 4-6 rules: 0.2-0.3.
- 3 active rules: 0.41 (500). 2 active rules (min 1), conflict 0.5: 0.55-0.59. Distractor counts hardly matter.
- Rendering the disabled decoys before the active rules, and ending every rule line with `[end]`, removed
  decoy-copying and a positional BoW leak. Lowering conflict_rate to 0.2 gave the 0.674 above (0.1 gave 0.71).

## Canaries and perplexity

`canaries500.jsonl`: 500 fixed multiple-choice items, 100 each of `syntax`, `factual`, `reasoning`, `instruction`,
`code`. Schema: `id`, `category`, `prompt`, `options` (3 each), `answer` (index), `source`. They detect collateral
damage from interventions and are never optimized on. Scored by option log-likelihood (no generation) in
`surgery/canaries.py`: prompt `Question: <prompt>\nAnswer:`, option tokenized separately with a leading space, ids
concatenated, option log-probs summed, argmax. Regenerate (byte-identical, fixed seed): `uv run python scripts/build_canaries.py`.

Sources: all procedural (fixed seed) except `factual`, which is hand-written well-known facts (capitals, element
symbols, planets, units, history, science; no external dataset, no licence issue). `code` output items are answered by
executing the snippet in the build script. Answer positions are shuffled with the seed (roughly uniform).

`ppl_heldout.txt`: ~203 KB (47,295 Qwen3 tokens) slice of *Pride and Prejudice* by Jane Austen, Project Gutenberg
#1342, public domain in the US. Source `https://www.gutenberg.org/cache/epub/1342/pg1342.txt` (download sha256
`3f6bb9d6...cf9d75`, pinned in the build script); the slice is 200,000 characters starting 20,000 characters after the
Gutenberg start marker, cut at a paragraph boundary. Its sha256 is pinned as `PPL_SHA256` in `surgery/canaries.py` and
verified on load. WikiText-2 raw was not used (no stdlib-readable raw copy at a stable URL). Perplexity uses strided
windows (1024 / stride 512).

Baseline, Qwen3-0.6B fp32 on MPS: accuracy 0.634 overall (syntax 0.83, factual 0.91, reasoning 0.54, instruction 0.42,
code 0.47; chance about 0.33), perplexity 26.97.
