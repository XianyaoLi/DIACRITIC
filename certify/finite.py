"""Finite instances from the paper used as self-checks of the solver branches that the community benchmarks
do not exercise (both benchmarks are transitive at every step).

  remark_i     the three-history instance of Appendix A.2, Remark (i): (A4) fails and the compatibility relation is
               NOT transitive at t=2, so Gamma_2 is not constructed; the tool reports the bracket [0, log2 3]
               (the exact minimum, h2(1/3) = 0.918 bit, is obtained by the closed-partition dynamic program, not here).
  rereveal     the A'-style gap toy with beta_2 re-revealed one step before its use: Gamma needs 1 bit (not 2) through
               the first gap and grasp and 0 in the second gap, while the strong congruence keeps 2 bits.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "toy"))
from toy_env import Env, toy_gap        # noqa: E402


def remark_i() -> Env:
    prior = {th: 1 / 3 for th in range(3)}

    def obs(th, t, acts):
        if t == 1:
            return {("rev", th): 1.0}
        if t == 2:
            return {("c",): 1.0}
        return {("y",) if th == 1 else ("x",): 1.0}

    def expert(th, t, hist):
        if t < 3:
            return {("noop",): 1.0}
        return {("act", th): 1.0}
    return Env(3, prior, obs, expert, name="remark-i-nontransitive")


def rereveal(M: int = 16, R1: int = 2, R2: int = 2, N: int = 4, gap1: int = 1, gap2: int = 3) -> Env:
    env = toy_gap(M=M, R1=R1, R2=R2, N=N, gap1=gap1, gap2=gap2)
    t_use2, b2, old_obs = env.phases["use2"], env.beta[1], env.obs

    def obs(lat, t, acts):
        if t == t_use2 - 1:
            return {("b2", b2(lat[0])): 1.0}
        return old_obs(lat, t, acts)
    env.obs = obs
    env.name = "gap-toy-rereveal"
    return env


FINITE = {"remark_i": remark_i, "rereveal": rereveal}
