"""Depth-atlas task families (spec docs/ATLAS-families.md). Each batch module registers its tasks on import;
add a batch with one more name in the import line below."""
from . import core
from . import b1, b2, b3, b4, b5, b6, b7, b8, b9, b10  # noqa: F401
from .core import (TYPE_LABELS, Task, check_options, generate, get_task, rival_bow, rival_near_miss,  # noqa: F401
                   rival_position, rivals, split_items)

TASK_NAMES = list(core.REGISTRY)
FAMILIES: dict = {}
for _n in TASK_NAMES:
    _t = core.REGISTRY[_n]
    _lab, _tasks = FAMILIES.get(_t.family, (_t.type_label, ()))
    assert _lab == _t.type_label
    FAMILIES[_t.family] = (_lab, _tasks + (_n,))
assert all(len(v[1]) == 2 for v in FAMILIES.values()), FAMILIES
