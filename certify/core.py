"""Certify the minimal recurrent behavioral memory of an environment with enumerable hidden variables.

Pipeline (the one used for the community benchmarks in the paper):
  1. drive the environment with a scripted expert until every hidden value has been observed once,
     recording the observation/action sequence of each episode verbatim as symbols;
  2. build the induced finite POMDP (one deterministic history per hidden value, uniform prior unless given);
  3. run the exact partition-refinement solver: (A2) and (A4) checks, transitivity of the compatibility
     relation at every step, and the per-step rates H(G_{E,t}|O_t), H(Gamma_t|O_t), H(Gamma^s_t|O_t), H(H_t|O_t).

When the compatibility relation is transitive at every step, H(Gamma_t|O_t) is the exact minimal recurrent
behavioral memory of the induced finite model. When it fails at some step, the exact minimum is not computed;
the sandwich bracket [H(G_{E,t}|O_t), H(Gamma^s_t|O_t)] is reported instead and the failing steps are listed.

Requirements on the environment (see README): the hidden variable is enumerable and fixed within an episode,
the expert is deterministic given (step, hidden value), and the observation sequence is a deterministic function
of the hidden value and the actions. Episodes must have a fixed length.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "toy"))          # exact solver shipped with the supplementary code
from toy_env import Env                                        # noqa: E402
from gamma_solver import GammaSolver                           # noqa: E402


@dataclass
class Adapter:
    """What the tool needs from an environment.

    reset(seed) -> first observation           (must also fix the hidden value for the episode)
    step(action) -> (observation, reward, done)
    latent() -> hashable                       (the hidden value of the current episode; valid after reset)
    oracle(t, latent) -> action                (scripted expert; t counts from 0)
    n_latent: number of distinct hidden values, or None if unknown (then `patience` stops the enumeration)
    symbol(obs) -> hashable                    (observation -> symbol; default: the rounded raw observation)
    success(reward) -> bool                    (whether the episode's final reward means the task was solved)
    """
    reset: Callable[[int], object]
    step: Callable[[object], Tuple[object, float, bool]]
    latent: Callable[[], Hashable]
    oracle: Callable[[int, Hashable], object]
    n_latent: Optional[int] = None
    name: str = "env"
    symbol: Callable[[object], Hashable] = lambda o: tuple(np.round(np.asarray(o, np.float64).ravel(), 6).tolist())
    success: Callable[[float], bool] = lambda r: r > 0
    latent_fields: Tuple[str, ...] = ("latent",)


@dataclass
class Certificate:
    name: str
    T: int
    hidden_values: int
    a2: List[bool]
    a4: List[bool]
    transitive: List[bool]
    H_G: List[float]
    H_Gamma: List[float]          # nan at steps where transitivity fails
    H_GammaS: List[float]
    H_H: List[float]
    histories_per_step: List[int]
    exact: bool = field(init=False)

    def __post_init__(self):
        self.exact = all(self.a2) and all(self.transitive)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items()}
        for k in ("H_G", "H_Gamma", "H_GammaS", "H_H"):
            d[k] = [None if np.isnan(x) else round(float(x), 6) for x in d[k]]
        return d

    def table(self) -> str:
        head = f"{'t':>3} {'|hist|':>7} {'A2':>3} {'A4':>3} {'trans':>5} {'H(G|O)':>8} {'H(Gam|O)':>9} {'H(GamS|O)':>10} {'H(H|O)':>8}"
        rows = [head, "-" * len(head)]
        for t in range(self.T):
            g = "  nan" if np.isnan(self.H_Gamma[t]) else f"{self.H_Gamma[t]:9.3f}"
            rows.append(f"{t + 1:3d} {self.histories_per_step[t]:7d} {'ok' if self.a2[t] else 'X':>3} {'ok' if self.a4[t] else 'X':>3} "
                        f"{'yes' if self.transitive[t] else 'NO':>5} {self.H_G[t]:8.3f} {g:>9} {self.H_GammaS[t]:10.3f} {self.H_H[t]:8.3f}")
        verdict = ("exact: the compatibility relation is transitive at every step; H(Gamma|O) is the minimal recurrent behavioral memory"
                   if self.exact else
                   "NOT exact: transitivity fails at steps " + ", ".join(str(t + 1) for t, v in enumerate(self.transitive) if not v)
                   + "; the minimum lies in [H(G|O), H(GamS|O)] at those steps and is not certified by this tool")
        if not all(self.a2):
            verdict = "NOT exact: A2 fails; history does not determine the fixed expert action law."
        return "\n".join(rows + ["", f"{self.name}: T={self.T}, hidden values={self.hidden_values}", verdict])


def record(adapter: Adapter, max_episodes: int = 100_000, patience: int = 200, seed0: int = 10_000) -> Dict[Hashable, Tuple[Tuple, Tuple]]:
    """Drive the environment with the oracle until every hidden value has been recorded once.

    Returns {latent: (symbol sequence, action sequence)}. Every episode must be solved by the oracle and all
    episodes must have the same length; a hidden value seen twice must reproduce the same sequences.
    """
    table: Dict[Hashable, Tuple[Tuple, Tuple]] = {}
    T = None
    since_new = 0
    for i in range(max_episodes):
        o = adapter.reset(seed0 + i)
        lat = adapter.latent()
        syms, acts, r, done, t = [], [], 0.0, False, 0
        while not done:
            a = adapter.oracle(t, lat)
            syms.append(adapter.symbol(o))
            acts.append(a)
            o, r, done = adapter.step(a)
            t += 1
        if not adapter.success(r):
            raise RuntimeError(f"the oracle did not solve the episode for hidden value {lat!r}")
        if T is None:
            T = t
        elif t != T:
            raise RuntimeError(f"episodes must have a fixed length: got {t} and {T}")
        rec = (tuple(syms), tuple(acts))
        if lat in table:
            if table[lat] != rec:
                raise RuntimeError(f"hidden value {lat!r} produced two different observation/action sequences; "
                                   "the environment is not a deterministic function of the hidden value")
            since_new += 1
        else:
            table[lat] = rec
            since_new = 0
        if adapter.n_latent is not None:
            if len(table) == adapter.n_latent:
                break
        elif since_new >= patience:
            break
    if adapter.n_latent is not None and len(table) != adapter.n_latent:
        raise RuntimeError(f"recorded {len(table)} of {adapter.n_latent} hidden values in {max_episodes} episodes")
    return table


def _clean(xs: Sequence[float]) -> List[float]:
    return [float(x) if np.isnan(x) else (0.0 if abs(x) < 1e-12 else float(x)) for x in xs]     # -0.0 from log terms -> 0.0


def certify_env(fenv: Env) -> Certificate:
    """Solve a finite POMDP given directly as a `toy_env.Env` (stochastic experts and observations allowed)."""
    solver = GammaSolver(fenv).solve()
    failed_a2 = [t for t in range(1, fenv.T + 1) if not solver.a2_ok[t]]
    if failed_a2:
        raise ValueError(f"A2 fails at steps {failed_a2}: history does not determine the expert action law; "
                         "the fixed-expert memory requirement is not certified. "
                         "GammaSolver may still be used directly for history-conditional diagnostics.")
    R = solver.rates()
    T = fenv.T
    ts = range(1, T + 1)
    return Certificate(name=fenv.name, T=T, hidden_values=len(fenv.prior),
                       a2=[bool(solver.a2_ok[t]) for t in ts], a4=[bool(solver.a4_ok[t]) for t in ts],
                       transitive=[bool(solver.transitive[t]) for t in ts],
                       H_G=_clean(R["H(G|O)"]), H_Gamma=_clean(R["H(Gamma|O)"]),
                       H_GammaS=_clean(R["H(GammaS|O)"]), H_H=_clean(R["H(H|O)"]),
                       histories_per_step=[len(solver.levels[t]) for t in ts])


def certify(adapter: Adapter, prior: Optional[Dict[Hashable, float]] = None, **record_kwargs) -> Certificate:
    """Record every hidden value of an environment, build the induced finite POMDP and solve it exactly."""
    tab = record(adapter, **record_kwargs)
    T = len(next(iter(tab.values()))[1])
    if prior is None:
        prior = {k: 1.0 / len(tab) for k in tab}
    fenv = Env(T, prior,
               lambda lat, t, acts: {tab[lat][0][t - 1]: 1.0},
               lambda lat, t, hist: {tab[lat][1][t - 1]: 1.0},
               name=adapter.name, latent_fields=adapter.latent_fields)
    return certify_env(fenv)
