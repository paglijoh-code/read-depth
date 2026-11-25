# Read depth

Code for the read-depth studies by Johnny Paglino: at which layer of a small language model a linear probe can first read a task's answer, what that says about cutting layers, and whether models agree on the order of tasks (the depth atlas). Project page: https://johnnypaglino.web.app/research/read-depth/

Hobby research, built with AI models (mostly Anthropic's Claude). Not peer reviewed. All tasks included here are synthetic.

- `surgery/`: the library. `surgery/atlas/` has the tasks (`tasks.py`, `families/`), read/use measures, the frozen gates (`gates.py`), the simulations (`csim.py`) and the decision tree (`tree.py`).
- `scripts/`: the experiment and analysis scripts.
- `tests/`: unit tests.
- `data/`: samples and fixed check items.
- `paper2/`: the depth atlas paper (draft) and the scripts that make its figures and tables.

## Run

    uv sync
    uv run pytest -q tests

Tests that need downloaded models, or saved results, skip when those are absent.

## Not included

Raw results and trained adapters, so `scripts/` that only analyse saved results cannot run without rerunning the experiments first. The two routing tasks of the first study need a dataset that is not released. Cloud machine scripts and working notes are left out, so some comments and docstrings name files under `docs/` and `reports/` that are not here.

## Licence

MIT (`LICENSE`) for the code. The paper's text and figures are (c) Johnny Paglino, all rights reserved. `data/ppl_heldout.txt` is public-domain text.
