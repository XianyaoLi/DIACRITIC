"""
Generic finite-horizon POMDP-with-hidden-latent specification + exact history enumeration.

An environment is fully specified by
  * a prior over latents  (theta_beh, theta_nui, ...)  -- anything hashable,
  * an observation kernel   obs(latent, t, actions_so_far) -> {o: prob}   (deterministic if one entry),
  * an expert policy        expert(latent, t, history)   -> {a: prob}     (P_E(A_t | s_t)),
  * horizon T.
Time runs t = 1..T.  History H_t = (o_1, a_1, o_2, ..., a_{t-1}, o_t).  The expert takes A_t after seeing O_t.
Everything downstream (behavioral quotient G_t, behavioral memory state Gamma_t, rates) is computed from these
primitives only -- nothing about Gamma is specified by hand.

Two concrete toys are defined at the bottom:
  * toy_reveal   : the reveal toy (reveal bits of g, decide at T).   Expected H(Gamma_t|O_t) = [0,0,1,2].
  * toy_gap   : an A'-style toy: reveal theta -> gap1 -> use beta_1(theta) -> gap2 -> use beta_2(theta).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from math import log2, ceil
from typing import Callable, Dict, Hashable, List, Tuple, Any
import numpy as np


@dataclass
class Env:
    T: int
    prior: Dict[Hashable, float]                                   # latent -> prob
    obs: Callable[[Hashable, int, Tuple], Dict[Hashable, float]]   # (latent, t, actions a_1..a_{t-1}) -> {o: p}
    expert: Callable[[Hashable, int, Tuple], Dict[Hashable, float]]  # (latent, t, history) -> {a: p}
    name: str = "env"
    latent_fields: Tuple[str, ...] = ()                            # names of latent tuple entries (for reporting)

    def latent_field(self, latent, name):
        return latent[self.latent_fields.index(name)]


@dataclass
class HistoryNode:
    t: int
    hist: Tuple            # (o_1, a_1, ..., a_{t-1}, o_t)
    prob: float            # P(H_t = hist) under expert occupancy
    post: Dict[Hashable, float]   # posterior over latents given hist (normalized)
    o: Hashable            # O_t
    act_dist: Dict[Hashable, float] = field(default_factory=dict)  # P_E(A_t | hist) (mixture over posterior)
    a2_ok: bool = True     # all consistent latents share the same action distribution (Assumption A2)
    children: Dict[Tuple, "HistoryNode"] = field(default_factory=dict)  # (a, o') -> node at t+1
    cont: Dict[Tuple, float] = field(default_factory=dict)             # (a, o') -> P(a, o' | hist)


def _actions_of(hist: Tuple) -> Tuple:
    return tuple(hist[1::2])


def enumerate_histories(env: Env, tol: float = 1e-15) -> Dict[int, List[HistoryNode]]:
    """Exact forward enumeration of all positive-probability histories under expert occupancy."""
    levels: Dict[int, List[HistoryNode]] = {t: [] for t in range(1, env.T + 1)}
    # t = 1
    roots: Dict[Tuple, Dict[Hashable, float]] = {}
    for lat, p in env.prior.items():
        for o, po in env.obs(lat, 1, ()).items():
            if po <= 0:
                continue
            roots.setdefault((o,), {})
            roots[(o,)][lat] = roots[(o,)].get(lat, 0.0) + p * po
    frontier = []
    for h, joint in roots.items():
        Z = sum(joint.values())
        node = HistoryNode(1, h, Z, {l: v / Z for l, v in joint.items()}, h[-1])
        levels[1].append(node); frontier.append(node)
    for t in range(1, env.T + 1):
        for node in frontier:
            _fill_actions(env, node)
        if t == env.T:
            break
        new_frontier = []
        for node in frontier:
            # joint over (a, o', latent)
            joint: Dict[Tuple, Dict[Hashable, float]] = {}
            for lat, pl in node.post.items():
                pa = env.expert(lat, t, node.hist)
                for a, p_a in pa.items():
                    if p_a <= 0:
                        continue
                    acts = _actions_of(node.hist) + (a,)
                    for o2, p_o in env.obs(lat, t + 1, acts).items():
                        if p_o <= 0:
                            continue
                        joint.setdefault((a, o2), {})
                        joint[(a, o2)][lat] = joint[(a, o2)].get(lat, 0.0) + pl * p_a * p_o
            for (a, o2), jl in joint.items():
                Z = sum(jl.values())
                if Z <= tol:
                    continue
                child = HistoryNode(t + 1, node.hist + (a, o2), node.prob * Z, {l: v / Z for l, v in jl.items()}, o2)
                node.children[(a, o2)] = child
                node.cont[(a, o2)] = Z
                levels[t + 1].append(child); new_frontier.append(child)
        frontier = new_frontier
    return levels


def _fill_actions(env: Env, node: HistoryNode):
    dists = []
    mix: Dict[Hashable, float] = {}
    for lat, pl in node.post.items():
        pa = env.expert(lat, node.t, node.hist)
        key = tuple(sorted((a, round(p, 12)) for a, p in pa.items() if p > 0))
        dists.append(key)
        for a, p in pa.items():
            mix[a] = mix.get(a, 0.0) + pl * p
    node.act_dist = {a: p for a, p in mix.items() if p > 0}
    node.a2_ok = len(set(dists)) == 1


# ----------------------------------------------------------------------------------------------
# Concrete toys
# ----------------------------------------------------------------------------------------------
def _bits(x: int, L: int) -> List[int]:
    return [(x >> l) & 1 for l in range(L)]


def toy_reveal(M: int = 16, R: int = 4, N: int = 4, p_correct: float = 1.0, leak: float = 0.0) -> Env:
    """
    Reveal toy (deterministic reveal).
      latent = (u in [M], z in [N]);  g(u) = ceil((u+1) R / M) - 1 in [R];  L = log2 R reveal bits.
      t=1: O = z (nuisance revealed), expert uniform over 2 approach actions (behaviorally irrelevant randomness).
      t=2..1+L: O = bit_l(g); expert 'noop'.
      t=T=L+2: O = constant; expert action = g  (softmax/stochastic version: correct w.p. p_correct, else uniform).
      leak > 0: appendix variant -- reveal bit 0 is lost with prob leak (history no longer determines g: (A2) fails).
    With M=16,R=4,N=4 the outline predicts H(G|O)=[0,0,0,2], H(Gamma|O)=[0,0,1,2], H(H|O)=[1,3,4,5].
    """
    L = ceil(log2(R)) if R > 1 else 0
    T = L + 2
    prior = {(u, z): 1.0 / (M * N) for u in range(M) for z in range(N)}
    g_of = lambda u: ceil((u + 1) * R / M) - 1

    def obs(lat, t, acts):
        u, z = lat
        if t == 1:
            return {("z", z): 1.0}
        if 2 <= t <= 1 + L:
            b = _bits(g_of(u), L)[t - 2]
            if leak > 0 and t == 2:      # Bernoulli loss of bit 0: with prob `leak` it is never revealed -> (A2) fails
                return {("bit", 0, b): 1 - leak, ("bit", 0, "lost"): leak}
            return {("bit", t - 2, b): 1.0}
        return {("decide",): 1.0}

    def expert(lat, t, hist):
        u, z = lat
        if t == 1:
            return {("approach", 0): 0.5, ("approach", 1): 0.5}
        if t < T:
            return {("noop",): 1.0}
        g = g_of(u)
        if p_correct >= 1.0:
            return {("act", g): 1.0}
        d = {("act", r): (1 - p_correct) / R for r in range(R)}
        d[("act", g)] += p_correct
        return d

    return Env(T, prior, obs, expert, name=f"reveal(M={M},R={R},N={N},p={p_correct},leak={leak})", latent_fields=("u", "z"))


def toy_gap(M: int = 16, R1: int = 2, R2: int = 2, N: int = 4, gap1: int = 1, gap2: int = 3,
            p_correct: float = 1.0, reveal: str = "theta", nui_action: bool = True) -> Env:
    """
    A'-style toy (behavioral compatibility and recurrent-minimality test bed).
      latent = (theta in [M], z in [N]).  beta_1(theta) = theta mod R1 ;  beta_2(theta) = (theta // R1) mod R2.
      (crossed binning: the join beta_1 v beta_2 has R1*R2 classes; in-class distinctions M/(R1 R2) are harmless.)
      t=1                : O = z (nuisance), expert uniform over 2 approach actions (if nui_action).
      reveal phase (B steps): O = bits of theta ('theta') or of the join (beta_1,beta_2) ('join'); expert noop.
      gap1 steps         : O = 'wait', expert noop.        <- Gamma must carry (beta_1, beta_2); G is trivial.
      use1               : O = 'use1', expert action beta_1(theta).
      gap2 steps         : O = 'wait', expert noop.        <- beta_1 expired: Gamma = beta_2 only.
      use2               : O = 'use2', expert action beta_2(theta).
    Observations never depend on theta after the reveal phase => continuation supports are homogeneous (A4).
    """
    if reveal == "theta":
        B = ceil(log2(M)) if M > 1 else 0
        code = lambda th: th
    elif reveal == "join":
        B = ceil(log2(R1 * R2)) if R1 * R2 > 1 else 0
        code = lambda th: (th % R1) + R1 * ((th // R1) % R2)
    else:
        raise ValueError(reveal)
    t_rev0 = 2
    t_gap1 = t_rev0 + B
    t_use1 = t_gap1 + gap1
    t_gap2 = t_use1 + 1
    t_use2 = t_gap2 + gap2
    T = t_use2
    prior = {(th, z): 1.0 / (M * N) for th in range(M) for z in range(N)}
    b1 = lambda th: th % R1
    b2 = lambda th: (th // R1) % R2

    def obs(lat, t, acts):
        th, z = lat
        if t == 1:
            return {("z", z): 1.0}
        if t_rev0 <= t < t_gap1:
            return {("bit", t - t_rev0, _bits(code(th), B)[t - t_rev0]): 1.0}
        if t == t_use1:
            return {("use1",): 1.0}
        if t == t_use2:
            return {("use2",): 1.0}
        return {("wait", t): 1.0}

    def stoch(g, R):
        if p_correct >= 1.0:
            return {("act", g): 1.0}
        d = {("act", r): (1 - p_correct) / R for r in range(R)}
        d[("act", g)] += p_correct
        return d

    def expert(lat, t, hist):
        th, z = lat
        if t == 1 and nui_action:
            return {("approach", 0): 0.5, ("approach", 1): 0.5}
        if t == t_use1:
            return stoch(b1(th), R1)
        if t == t_use2:
            return stoch(b2(th), R2)
        return {("noop",): 1.0}

    env = Env(T, prior, obs, expert,
              name=f"gap(M={M},R1={R1},R2={R2},N={N},gap1={gap1},gap2={gap2},p={p_correct},reveal={reveal})",
              latent_fields=("theta", "z"))
    env.phases = dict(reveal=(t_rev0, t_gap1 - 1), gap1=(t_gap1, t_use1 - 1), use1=t_use1,
                      gap2=(t_gap2, t_use2 - 1), use2=t_use2)
    env.beta = (b1, b2)
    return env


if __name__ == "__main__":
    for env in [toy_reveal(), toy_gap()]:
        lv = enumerate_histories(env)
        print(env.name, "T =", env.T, " #histories per t:", [len(lv[t]) for t in range(1, env.T + 1)],
              " total prob at T = %.6f" % sum(n.prob for n in lv[env.T]))
