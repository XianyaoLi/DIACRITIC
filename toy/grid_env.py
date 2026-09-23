"""
Second exact-Gamma domain that is not a robot arm: a 2-D grid corridor with signposts, wide halls with a hidden lateral drift,
and two junctions ("Signpost corridor", toy_env.Env instance -> same enumerator / solver / trainer as the other toys).

Layout (the agent walks left->right; time t = 1..T):
   start | signposts (B cells, one bit of theta each) | hall 1 (gap1 cells, width X+1) | junction 1 | hall 2 (gap2 cells) | junction 2
Latent = (theta in [M], w in [W], z in [N]):
   theta  hidden goal code; the expert turns b1(theta) = theta % R1 at junction 1 and b2(theta) = (theta // R1) % R2 at junction 2.
          In-class distinctions (M / (R1 R2) of them) never change the expert -> "world complexity" beyond the R1 R2 behaviours.
   w      hidden lateral DRIFT in the halls: every hall step moves the agent sideways by drift(w) in {-(W//2),...,+(W//2)} cells
          (W = 1: no drift = exact regime).  The drift changes the transitions the agent experiences (agent-centric), is revealed
          by a signpost at the start and re-revealed by the hall observations (lateral position x is observed), and the expert
          ignores it: layer-2 nuisance in the outline's sense.
   z      exogenous texture shown at the start (layer-1 nuisance).
Observations (symbols):  ("start", z, w) | ("sign", i, bit) | ("hall", k, x)  (hall id k, lateral cell x) | ("junction", k)
Actions: "fwd", "left", "right" in the halls (left/right = sidestep; the expert chooses uniformly among {"left","right"} on the
first `n_rand` cells of each hall and "fwd" otherwise -- behaviourally equivalent randomness), ("turn", b) at the junctions,
"fwd" elsewhere.  Lateral position: x' = clip(x + step(a) + drift(w), 0, X).
Why this separates the nearest neighbours: to invert a hall action from (o_t, o_{t+j}) one needs the drift (x'-x = a + drift is
ambiguous without w), so a multi-step-inverse representation must carry w; the expert never needs it, so Gamma does not.
"""
from __future__ import annotations
from math import ceil, log2
from toy_env import Env, _bits


def grid_corridor(M=16, R1=2, R2=2, W=1, N=2, gap1=3, gap2=6, n_rand=1, X=4, reveal_w=True):
    B = ceil(log2(M)) if M > 1 else 0
    t_sign0 = 2; t_hall1 = t_sign0 + B; t_j1 = t_hall1 + gap1; t_hall2 = t_j1 + 1; t_j2 = t_hall2 + gap2; T = t_j2
    prior = {(th, w, z): 1.0 / (M * W * N) for th in range(M) for w in range(W) for z in range(N)}
    b1 = lambda th: th % R1; b2 = lambda th: (th // R1) % R2
    drift = lambda w: w - (W // 2)
    step = {"left": -1, "right": +1, "fwd": 0}
    x0 = X // 2

    def hall_steps(t):
        """list of (hall id, cell index within the hall) for hall time steps <= t"""
        out = []
        for s in range(t_hall1, min(t, t_j1 - 1) + 1): out.append((1, s - t_hall1))
        for s in range(t_hall2, min(t, t_j2 - 1) + 1): out.append((2, s - t_hall2))
        return out

    def x_at(lat, t, acts):
        """lateral position when observing at time t inside a hall: the junction is a single cell, so each hall starts at the
        centre line x0 and only the actions taken inside the *current* hall (a_s for hall steps s < t) move the agent sideways"""
        th, w, z = lat; x = x0; s0 = t_hall1 if t < t_j1 else t_hall2
        for s in range(s0, t):
            a = acts[s - 1]                                        # a_s (1-indexed actions: acts[0] = a_1)
            x = min(X, max(0, x + step[a] + drift(w)))
        return x

    def obs(lat, t, acts):
        th, w, z = lat
        if t == 1: return {("start", z, w if reveal_w else -1): 1.0}
        if t_sign0 <= t < t_hall1: return {("sign", t - t_sign0, _bits(th, B)[t - t_sign0]): 1.0}
        if t == t_j1: return {("junction", 1): 1.0}
        if t == t_j2: return {("junction", 2): 1.0}
        k = 1 if t < t_j1 else 2
        return {("hall", k, x_at(lat, t, acts)): 1.0}

    def expert(lat, t, hist):
        th, w, z = lat
        if t == t_j1: return {("turn", b1(th)): 1.0}
        if t == t_j2: return {("turn", b2(th)): 1.0}
        if (t_hall1 <= t < t_hall1 + n_rand) or (t_hall2 <= t < t_hall2 + n_rand): return {"left": 0.5, "right": 0.5}
        return {"fwd": 1.0}

    env = Env(T, prior, obs, expert, name=f"grid(M={M},R1={R1},R2={R2},W={W},N={N},gap1={gap1},gap2={gap2},nrand={n_rand},X={X})",
              latent_fields=("theta", "w", "z"))
    env.phases = dict(reveal=(t_sign0, t_hall1 - 1), gap1=(t_hall1, t_j1 - 1), use1=t_j1, gap2=(t_hall2, t_j2 - 1), use2=t_j2)
    env.beta = (b1, b2)
    return env


if __name__ == "__main__":
    import sys, time
    from gamma_solver import GammaSolver
    from toy_env import enumerate_histories
    print("solver gate: A2 / A4 / transitivity and the predicted ladder  (exact regime expected at W=1; sandwich at W>1)")
    for (M, R1, R2, W, g1, g2) in [(16, 2, 2, 1, 3, 6), (16, 2, 2, 3, 3, 6), (32, 2, 2, 1, 3, 6), (32, 2, 2, 3, 3, 6), (16, 2, 2, 1, 3, 15), (16, 4, 2, 1, 3, 6)]:
        env = grid_corridor(M=M, R1=R1, R2=R2, W=W, gap1=g1, gap2=g2); t0 = time.time()
        s = GammaSolver(env).solve(); r = s.report(gammaJ=False, verbose=False); rt = r["rates"]; ph = env.phases
        mean = lambda xs, rng: sum(xs[t - 1] for t in range(rng[0], rng[1] + 1)) / (rng[1] - rng[0] + 1)
        print(f"{env.name:55s} |hist| max {max(len(s.levels[t]) for t in s.levels):5d}  A2 {'ok' if all(r['a2']) else 'X '} A4 {'ok' if all(r['a4']) else 'X '} trans {'yes' if all(r['transitive']) else 'NO '} | "
              f"H(Gam|O) gap1 {mean(rt['H(Gamma|O)'], ph['gap1']):.2f} [{log2(R1*R2):.2f}] use1 {rt['H(Gamma|O)'][ph['use1']-1]:.2f} gap2 {mean(rt['H(Gamma|O)'], ph['gap2']):.2f} [{log2(R2):.2f}] | "
              f"H(G|O) gap1 {mean(rt['H(G|O)'], ph['gap1']):.2f} use1 {rt['H(G|O)'][ph['use1']-1]:.2f} [{log2(R1):.2f}] | H(GamS|O) gap1 {mean(rt['H(GammaS|O)'], ph['gap1']):.2f} gap2 {mean(rt['H(GammaS|O)'], ph['gap2']):.2f} | H(H|O) gap1 {mean(rt['H(H|O)'], ph['gap1']):.2f}  ({time.time()-t0:.0f}s)")
