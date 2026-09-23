"""
T0 unit tests for the Gamma solver (first gate of the project).
  1. reveal toy (toy_reveal): automatic H(Gamma_t|O_t) == [0,0,1,2]; H(G|O) == [0,0,0,2]; Gamma^(J) monotone.
  2. gap toy: predicted temporal profile (anticipatory memory during gap1, expiry after use1).
  3. non-transitive environment (A4 violated): solver must flag transitive=False and NOT produce a Gamma;
     strong congruence still gives a valid upper bound.
  4. A2 violation (Bernoulli leak): flagged.
  5. "future observations are free": re-revealing beta_2 right before use2 makes Gamma trivial during gap2,
     even though A4 fails there (A4 is sufficient, not necessary, for transitivity).
"""
import numpy as np
from math import log2
from toy_env import Env, toy_reveal, toy_gap
from gamma_solver import GammaSolver

def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)

def env_nontransitive() -> Env:
    """theta in {0,1,2}. t=1: O=theta (so t=1 histories differ by O, not compared).  t=2: O=const, noop.
    t=3: O='y' if theta==1 else 'x'; expert acts theta-dependently for theta in {0,2}.
    At t=2: h0,h2 share continuation ('noop','x') and conflict; h1 shares none -> h0~h1, h1~h2, h0!~h2."""
    prior = {th: 1/3 for th in range(3)}
    def obs(th, t, acts):
        if t == 1: return {("rev", th): 1.0}
        if t == 2: return {("c",): 1.0}
        return {("y",) if th == 1 else ("x",): 1.0}
    def expert(th, t, hist):
        if t < 3: return {("noop",): 1.0}
        return {("act", th): 1.0}
    return Env(3, prior, obs, expert, name="nontransitive")

def toy_gap_rereveal(**kw) -> Env:
    env = toy_gap(**kw)
    t_use2 = env.phases["use2"]; b2 = env.beta[1]
    old_obs = env.obs
    def obs(lat, t, acts):
        if t == t_use2 - 1:
            return {("b2", b2(lat[0])): 1.0}
        return old_obs(lat, t, acts)
    env.obs = obs; env.name += "+rereveal_b2"
    return env

if __name__ == "__main__":
    print("\n[1] reveal toy")
    s = GammaSolver(toy_reveal()).solve(); r = s.report()
    assert close(r["rates"]["H(Gamma|O)"], [0, 0, 1, 2]), r["rates"]["H(Gamma|O)"]
    assert close(r["rates"]["H(G|O)"], [0, 0, 0, 2])
    assert all(r["a2"]) and all(r["a4"]) and all(r["transitive"]) and r["gammaJ_monotone"]
    assert close(r["rates"]["H(GammaS|O)"], r["rates"]["H(Gamma|O)"])
    print("PASS: H(Gamma|O) = [0,0,1,2] computed from the behavioral compatibility definition automatically")

    print("\n[2] gap toy (A'-style)")
    env = toy_gap(M=16, R1=2, R2=2, N=4, gap1=1, gap2=3)
    s = GammaSolver(env).solve(); r = s.report()
    ph = env.phases
    HG, HGam = r["rates"]["H(G|O)"], r["rates"]["H(Gamma|O)"]
    for t in range(ph["gap1"][0], ph["gap1"][1] + 1):
        assert abs(HGam[t-1] - log2(4)) < 1e-9 and abs(HG[t-1]) < 1e-9        # anticipatory memory = join
    assert abs(HG[ph["use1"]-1] - 1) < 1e-9 and abs(HGam[ph["use1"]-1] - 2) < 1e-9
    for t in range(ph["gap2"][0], ph["gap2"][1] + 1):
        assert abs(HGam[t-1] - 1) < 1e-9 and abs(HG[t-1]) < 1e-9              # beta_1 expired
    assert abs(HG[ph["use2"]-1] - 1) < 1e-9 and abs(HGam[ph["use2"]-1] - 1) < 1e-9
    assert all(r["a4"]) and all(r["transitive"]) and r["gammaJ_monotone"]
    # max |Gamma_t|_o during gap1 = |beta1 v beta2| = 4 > R1 = R2 = 2   (codebook lower bound)
    assert max(s.label_stats(ph["gap1"][0]).values()) == 4
    print("PASS: Gamma profile = anticipatory join during gap1, expiry after use1; max|Gamma|_o = 4 > R")

    print("\n[3] non-transitive environment (A4 violated)")
    s = GammaSolver(env_nontransitive()).solve(); r = s.report(gammaJ=False)
    assert r["a4"][1] is False and r["transitive"][1] is False and s.Gamma[2] is None
    assert abs(r["rates"]["H(GammaS|O)"][1] - log2(3)) < 1e-9      # strong: 3 classes (upper bound)
    assert abs(r["rates"]["H(G|O)"][1]) < 1e-9                       # lower bound 0
    # minimum compatible cover here has 2 cells ({0,1},{2}) -> H = h2(1/3) = 0.918 < log2 3: strict sandwich
    print("PASS: solver refuses to build Gamma (no transitive closure); sandwich [0, log2 3] reported")

    print("\n[4] A2 violation (leak)")
    s = GammaSolver(toy_reveal(leak=0.3)).solve(); r = s.report(gammaJ=False)
    assert r["a2"][-1] is False and all(r["a2"][:-1])
    print("PASS: A2 violation flagged at decision time T")

    print("\n[5] future observations are free: re-reveal beta_2 before use2")
    env = toy_gap_rereveal(M=16, R1=2, R2=2, N=4, gap1=1, gap2=3)
    s = GammaSolver(env).solve(); r = s.report(gammaJ=False)
    ph = env.phases; HGam = r["rates"]["H(Gamma|O)"]; HGS = r["rates"]["H(GammaS|O)"]
    # beta_2 will be re-observed at use2-1  =>  during gap1/use1 only beta_1 must be carried (1 bit, not 2),
    # and during gap2 nothing at all.  The strong congruence (no "free future") still charges 2 bits: strict sandwich.
    for t in range(ph["gap1"][0], ph["use1"] + 1):
        assert abs(HGam[t-1] - 1) < 1e-9 and abs(HGS[t-1] - 2) < 1e-9, t
    for t in range(ph["gap2"][0], ph["gap2"][1]):
        assert abs(HGam[t-1]) < 1e-9, t
    # A4 fails only at the step whose continuation observation reveals beta_2; the relation is transitive throughout.
    assert [not a for a in r["a4"]] == [t == ph["use2"] - 2 for t in range(1, env.T + 1)]
    assert all(r["transitive"])
    print("PASS: re-observable info leaves Gamma early (2->1 bit in gap1, 0 in gap2); strong bound stays at 2; A4 fails only at use2-2, relation still transitive")
    print("\nALL T0 TESTS PASSED")
