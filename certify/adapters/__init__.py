from .example import delayed_cue
from .mikasa import TASKS as _MIKASA, memchain, tmaze

TASKS = dict(_MIKASA, example=delayed_cue)

__all__ = ["TASKS", "memchain", "tmaze", "delayed_cue"]
