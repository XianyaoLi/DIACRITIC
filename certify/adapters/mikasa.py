"""Adapters for the two community benchmarks certified in the paper, loaded from an unmodified MIKASA-Base checkout.

    tmaze     Passive T-maze (Ni et al., 2023): the goal side is shown in the first observation and the agent must
              turn towards it after `corridor_length` steps. Hidden value: goal side (2 values).
    memchain  bsuite MemoryChain ("MemoryLength"): a context of `num_bits` bits is shown in the first observation
              and a query index on the last step; the final action must repeat the queried bit.
              Hidden value: (context, query) (2^bits * bits values).

The environment code is imported from `<mikasa>/mikasa_base/...` exactly as installed; nothing in it is modified.
Set the checkout with --mikasa PATH or the environment variable MIKASA_BASE (default: ../external/MIKASA-Base).
"""
from __future__ import annotations
import importlib.util
import os
import sys
import types

import numpy as np

from ..core import Adapter

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "MIKASA-Base")


def _root(mikasa: str | None) -> str:
    root = mikasa or os.environ.get("MIKASA_BASE") or _DEFAULT
    if not os.path.isdir(os.path.join(root, "mikasa_base")):
        raise FileNotFoundError(f"MIKASA-Base checkout not found at {root!r}; pass --mikasa PATH or set MIKASA_BASE")
    return root


def _load(root: str, name: str, *rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "mikasa_base", *rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _wrap(env):
    def reset(seed):
        o, _ = env.reset(seed=seed)
        return o

    def step(a):
        o, r, term, trunc, _ = env.step(a)
        return o, float(r), bool(term or trunc)
    return reset, step


def tmaze(L: int = 10, mikasa: str | None = None, **_) -> Adapter:
    root = _root(mikasa)
    tm = _load(root, "certify_tmaze_ext", "Passive_T_Maze", "env", "env_passive_t_maze.py")
    env = tm.TMazeClassicPassive(corridor_length=L, penalty=0.0)
    a_right = next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (1, 0))
    a_y = {s: next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (0, s)) for s in (1, -1)}
    reset, step = _wrap(env)
    return Adapter(reset=reset, step=step, latent=lambda: int(env.goal_y),
                   oracle=lambda t, lat: a_right if t < L else a_y[lat],
                   n_latent=2, name=f"tmaze-L{L}", latent_fields=("goal",))


def memchain(L: int = 10, bits: int = 1, mikasa: str | None = None, **_) -> Adapter:
    root = _root(mikasa)
    if "certify_bsx" not in sys.modules:
        pkg = types.ModuleType("certify_bsx")
        pkg.__path__ = [os.path.join(root, "mikasa_base", "Bsuite", "env")]
        sys.modules["certify_bsx"] = pkg
        _load(root, "certify_bsx.base", "Bsuite", "env", "base.py")
        _load(root, "certify_bsx.discounting_chain", "Bsuite", "env", "discounting_chain.py")
        _load(root, "certify_bsx.memory_chain", "Bsuite", "env", "memory_chain.py")
        _load(root, "certify_bsx.bsuite_env", "Bsuite", "env", "bsuite_env.py")
    bw = sys.modules["certify_bsx.bsuite_env"]
    env = bw.BsuiteGymWrapper("MemoryLength", memory_length=L, num_bits=bits)
    T = L + 1
    reset, step = _wrap(env)
    return Adapter(reset=reset, step=step,
                   latent=lambda: (tuple(int(b) for b in env._env._context), int(env._env._query)),
                   oracle=lambda t, lat: int(lat[0][lat[1]]) if t == T - 1 else 0,   # earlier actions have no effect
                   n_latent=(2 ** bits) * bits, name=f"memchain-L{L}-b{bits}", latent_fields=("context_query",))


TASKS = {"tmaze": tmaze, "memchain": memchain}
