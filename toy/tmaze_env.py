"""
T-maze (cue -> corridor -> junction): the standard memory task of the RL memory literature (Ni et al. 2023 and relatives), as a
toy_env.Env so that the exact solver and the toy trainer apply unchanged.  No instrumented probe channel: the cue is the ordinary
observation at the start, then a corridor of L identical cells, then one junction where the expert turns according to the cue.
Latent = (cue in [R], z in [N]) with z an exogenous texture shown at the start (layer-1 nuisance).
Observations: ("cue", c, z) at t=1, ("corridor",) for t=2..L+1, ("junction",) at t=L+2.   Actions: "fwd", ("turn", c).
(A4) holds automatically: the remembered latent (the cue) influences only the expert's action, never the observation kernel,
so H(Gamma_t|O_t) = log2 R through the corridor (anticipatory memory), H(G_t|O_t) = 0 until the junction, and both drop to 0 after.
"""
from __future__ import annotations
from math import log2
from toy_env import Env


def tmaze(R=2, N=2, L=10):
    T = L + 2
    prior = {(c, z): 1.0 / (R * N) for c in range(R) for z in range(N)}
    def obs(lat, t, acts):
        c, z = lat
        if t == 1: return {("cue", c, z): 1.0}
        if t == T: return {("junction",): 1.0}
        return {("corridor",): 1.0}
    def expert(lat, t, hist):
        c, z = lat
        return {("turn", c): 1.0} if t == T else {"fwd": 1.0}
    env = Env(T, prior, obs, expert, name=f"tmaze(R={R},N={N},L={L})", latent_fields=("theta", "z"))
    env.phases = dict(reveal=(1, 1), gap1=(2, T - 1), use1=T, gap2=(T, T), use2=T)   # single class: use2 == use1 (summaries use gap1/use1)
    return env


if __name__ == "__main__":
    from gamma_solver import GammaSolver
    for R, L in [(2, 5), (2, 10), (2, 20), (2, 40), (2, 80), (4, 20)]:
        env = tmaze(R=R, L=L); s = GammaSolver(env).solve(); r = s.report(gammaJ=False, verbose=False); rt = r["rates"]
        print(f"{env.name:22s} |hist| max {max(len(s.levels[t]) for t in s.levels):3d}  A2 {'ok' if all(r['a2']) else 'X'} A4 {'ok' if all(r['a4']) else 'X'} trans {'yes' if all(r['transitive']) else 'NO'} | "
              f"H(Gam|O) corridor {sum(rt['H(Gamma|O)'][1:L+1])/L:.2f} [{log2(R):.2f}] junction {rt['H(Gamma|O)'][L+1]:.2f} | H(G|O) corridor {sum(rt['H(G|O)'][1:L+1])/L:.2f} junction {rt['H(G|O)'][L+1]:.2f} | H(H|O) corridor {sum(rt['H(H|O)'][1:L+1])/L:.2f}")
