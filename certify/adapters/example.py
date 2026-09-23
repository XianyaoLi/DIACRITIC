"""A minimal adapter written from scratch, to show what the tool needs (no external dependency).

Delayed-cue task: a hidden bit is shown at step 0, then hidden for `delay` steps, and the final action must repeat
it. The expert waits (action 0) and answers on the last step. Certified requirement: 1 bit while the cue is hidden,
0 once it has been acted on.
"""
from __future__ import annotations
import numpy as np

from ..core import Adapter


class DelayedCue:
    def __init__(self, delay: int):
        self.delay, self.T = delay, delay + 2

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self.bit, self.t = int(rng.integers(2)), 0
        return np.array([1.0, float(self.bit)])            # (cue visible, cue value)

    def step(self, a):
        self.t += 1
        done = self.t == self.T
        r = 1.0 if done and a == self.bit else 0.0
        return np.array([0.0, 0.0]), r, done


def delayed_cue(delay: int = 5, **_) -> Adapter:
    env = DelayedCue(delay)
    return Adapter(reset=env.reset, step=env.step, latent=lambda: env.bit,
                   oracle=lambda t, lat: lat if t == env.T - 1 else 0, n_latent=2, name=f"delayed-cue-{delay}",
                   latent_fields=("bit",))
