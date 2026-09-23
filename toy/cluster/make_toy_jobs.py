"""Job lines for the grid-corridor (second exact-Gamma domain) sweeps on the cluster CPU nodes.  One line = one toy_diacritic.py run:
   <tag> <M> <R1> <R2> <W> <gap1> <gap2> <variant> <beta> <lam> <seed> <steps> <l2> <lr> <mik> <aux_theta> <bound>
Config (local sweep 2026-08-29, M=16 gap 3/6, 6 seeds): raw VQ + tanh-bounded residual state + beta=0.01: -R 5/6, -RF 6/6 sufficient, no codebook
collapse; raw VQ unbounded collapses on 1-2/6 seeds (and sys-ID always); unit-sphere VQ kills -R (0/4).
Grids: grid_ladder (exact regime, M-sweep, all variants) | grid_horizon (reliability vs reveal->use distance) | grid_drift (layer-2 nuisance:
hidden drift W-sweep, ours vs multi-step-inverse vs sys-ID) | grid_smoke.  Usage: python toy/cluster/make_toy_jobs.py <grid> > toy/cluster/jobs_<grid>.txt"""
import sys, itertools
g = sys.argv[1]; J = []
def add(tag, M=16, R1=2, R2=2, W=1, gap1=3, gap2=6, variant="diacritic", beta=0.01, lam=0, seed=0, steps=6000, l2=0, lr=3e-4, mik=0, aux=0, bound="tanh", K=16, env="grid"):
    J.append(f"{tag} {M} {R1} {R2} {W} {gap1} {gap2} {variant} {beta} {lam} {seed} {steps} {l2} {lr} {mik} {aux} {bound} {K} {env}")
S8 = range(8)
if g == "grid_smoke":
    add("grid_smoke", steps=300); add("grid_smoke", variant="mik", steps=300, W=3); add("grid_smoke", variant="sysid", steps=300)
elif g == "grid_ladder":     # exact regime (W=1): ladder 2 -> 1 -> 0 and P-I (rate flat in M) for every variant, gap1=3 gap2=6
    for M, seed in itertools.product([4, 16, 64], S8):
        add("grid_ladder", M=M, seed=seed, variant="diacritic", beta=0.01, lam=0)          # -R
        add("grid_ladder", M=M, seed=seed, variant="diacritic", beta=0.01, lam=1)          # -RF
        add("grid_ladder", M=M, seed=seed, variant="diacritic", beta=0.03, lam=0)          # -R, stronger rate pressure (expiry)
        add("grid_ladder", M=M, seed=seed, variant="diacritic", beta=0.0, lam=0)           # beta = 0 control
        add("grid_ladder", M=M, seed=seed, variant="sysid", beta=0.01, lam=0)              # system identification (aux theta)
        add("grid_ladder", M=M, seed=seed, variant="mik", beta=0.01, lam=0)                # multi-step inverse, same rate term
        add("grid_ladder", M=M, seed=seed, variant="bypass", beta=0.01, lam=0)             # continuous carrier
        add("grid_ladder", M=M, seed=seed, variant="uncond", beta=0.01, lam=0)
elif g == "grid_horizon":    # learning horizon: reveal->use distance via gap2 (and gap1), -R / -RF / scaffold-RF
    for gap2, seed in itertools.product([1, 3, 6, 10, 15, 20], S8):
        add("grid_horizon", gap2=gap2, seed=seed, variant="diacritic", beta=0.01, lam=0)
        add("grid_horizon", gap2=gap2, seed=seed, variant="diacritic", beta=0.01, lam=1)
        add("grid_horizon", gap2=gap2, seed=seed, variant="scaffold", beta=0.01, lam=1)
    for gap1, seed in itertools.product([8, 15], S8):
        add("grid_horizon", gap1=gap1, gap2=6, seed=seed, variant="diacritic", beta=0.01, lam=0)
        add("grid_horizon", gap1=gap1, gap2=6, seed=seed, variant="diacritic", beta=0.01, lam=1)
elif g == "grid_drift":      # layer-2 nuisance: hidden lateral drift with W modes (revealed, agent-centric, expert-irrelevant); general regime for W>1
    for W, seed in itertools.product([1, 3, 5, 7], S8):
        add("grid_drift", W=W, seed=seed, variant="diacritic", beta=0.01, lam=0)
        add("grid_drift", W=W, seed=seed, variant="diacritic", beta=0.01, lam=1)
        add("grid_drift", W=W, seed=seed, variant="mik", beta=0.01, lam=0)                 # inverse objective + same rate term
        add("grid_drift", W=W, seed=seed, variant="mik", beta=0.0, lam=0)                  # inverse objective, no rate term
        add("grid_drift", W=W, seed=seed, variant="sysid", beta=0.01, lam=0)
        add("grid_drift", W=W, seed=seed, variant="bypass", beta=0.01, lam=0)
elif g == "grid_bypass":     # bypass tuning (the continuous-carrier ablation needs a larger lr, as on the gap toy): M=16 W=1 and W=5
    for lr, W, seed in itertools.product([1e-3, 3e-3], [1, 5], S8):
        add("grid_bypass", W=W, seed=seed, variant="bypass", beta=0.01, lam=0, lr=lr)
elif g == "grid_fair":       # matched-D fairness cells: multi-step inverse JOINT with imitation (+rate), and K=64 codebook controls for mik-only and sys-ID
    for W, seed in itertools.product([1, 5], S8):
        add("grid_fair", W=W, seed=seed, variant="diacritic", beta=0.01, lam=0, mik=1.0)     # inverse head + imitation + rate (joint)
        add("grid_fair", W=W, seed=seed, variant="mik", beta=0.01, lam=0, K=64)              # inverse only, K=64
    for seed in S8:
        add("grid_fair", W=1, seed=seed, variant="sysid", beta=0.01, lam=0, K=64)            # sys-ID, K=64 (M=16 needs 4 bits of theta + 2 of behaviour)
        add("grid_fair", W=1, seed=seed, variant="diacritic", beta=0.01, lam=0, K=64)        # ours, K=64
elif g == "tmaze":           # standard T-maze (cue -> corridor L -> junction), no probe channel; (A4) automatic; horizon sweep over L
    for L, seed in itertools.product([5, 10, 20, 40, 80], S8):
        add("tmaze", R1=2, gap1=L, seed=seed, variant="diacritic", beta=0.01, lam=0, env="tmaze")
        add("tmaze", R1=2, gap1=L, seed=seed, variant="diacritic", beta=0.01, lam=1, env="tmaze")
        add("tmaze", R1=2, gap1=L, seed=seed, variant="scaffold", beta=0.01, lam=1, env="tmaze")
    for L, seed in itertools.product([10, 40], S8):
        add("tmaze", R1=4, gap1=L, seed=seed, variant="diacritic", beta=0.01, lam=0, env="tmaze")   # 2-bit cue
elif g == "tmaze2":          # complete the 2-bit cue sweep (R=4) at L=5,20,80 (L=10,40 exist in grid tmaze)
    for L, seed in itertools.product([5, 20, 80], S8):
        add("tmaze", R1=4, gap1=L, seed=seed, variant="diacritic", beta=0.01, lam=0, env="tmaze")
else: raise SystemExit(g)
print("\n".join(J))
