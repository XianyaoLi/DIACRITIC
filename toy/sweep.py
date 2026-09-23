"""Generate job lists; run with:  python sweep.py <name> | xargs -P <n> -I{} python run_job.py {} <name>"""
import json, sys, itertools
name = sys.argv[1]; jobs = []
if name == "gap_sweep":
    for gap2, lam, seed in itertools.product([1, 3, 6, 10, 15], [0.0, 1.0], range(8)):
        jobs.append(dict(env="gap", gap2=gap2, lam=lam, beta=0.03, seed=seed, steps=4000, p=1.0))
elif name == "beta_sweep_reveal":
    for beta, lam, seed in itertools.product([0.0, 0.003, 0.01, 0.03, 0.1, 0.3], [0.0, 1.0], range(3)):
        jobs.append(dict(env="reveal", lam=lam, beta=beta, seed=seed, steps=2000, p=1.0))
elif name == "beta_sweep_gap":
    for beta, gap2, lam, seed in itertools.product([0.0, 0.01, 0.1, 0.3], [3, 10], [0.0, 1.0], range(3)):
        jobs.append(dict(env="gap", gap2=gap2, lam=lam, beta=beta, seed=seed, steps=3000, p=1.0))
elif name == "fs_reveal":      # finite-sample + stochastic expert: the regime with "something to forget"
    for beta, lam, seed in itertools.product([0.0, 0.003, 0.01, 0.03, 0.1, 0.3], [0.0, 1.0], range(4)):
        jobs.append(dict(env="reveal", lam=lam, beta=beta, seed=seed, steps=2000, p=0.7, n_train=48))
elif name == "fs_gap":
    for n_train, beta, lam, seed in itertools.product([128, 512], [0.0, 0.01, 0.03, 0.1], [0.0, 1.0], range(3)):
        jobs.append(dict(env="gap", gap2=6, lam=lam, beta=beta, seed=seed, steps=6000, p=0.8, n_train=n_train, lr=3e-4, l2=False))
elif name == "r_tuning":      # fair best-case search for DIACRITIC-R (BPTT only)
    for gap2, lr, l2, seed in itertools.product([1, 6], [3e-4, 1e-3, 3e-3], [True, False], range(4)):
        jobs.append(dict(env="gap", gap2=gap2, lam=0.0, beta=0.03, seed=seed, steps=6000, p=1.0, lr=lr, l2=l2))
    for gap2, commit, seed in itertools.product([1, 6], [0.05, 1.0], range(4)):
        jobs.append(dict(env="gap", gap2=gap2, lam=0.0, beta=0.03, seed=seed, steps=6000, p=1.0, commit=commit))
elif name == "rf_extra":
    for beta, seed in itertools.product([0.1, 0.3], range(4)):
        jobs.append(dict(env="gap", gap2=6, lam=1.0, beta=beta, seed=seed, steps=4000, p=1.0))
    for seed in range(4):
        jobs.append(dict(env="gap", gap2=15, lam=1.0, beta=0.03, seed=seed, steps=10000, p=1.0))
elif name == "fair_ext":     # complete the fair comparison: -R without L2 at long gaps; -RF without L2; -RF L2 at small lr
    for gap2, lr, seed in itertools.product([10, 15], [3e-4, 1e-3], range(4)):
        jobs.append(dict(env="gap", gap2=gap2, lam=0.0, beta=0.03, seed=seed, steps=6000, p=1.0, lr=lr, l2=False))
    for gap2, lr, seed in itertools.product([6, 15], [3e-4, 1e-3], range(4)):
        jobs.append(dict(env="gap", gap2=gap2, lam=1.0, beta=0.03, seed=seed, steps=6000, p=1.0, lr=lr, l2=False))
    for gap2, seed in itertools.product([6, 15], range(4)):
        jobs.append(dict(env="gap", gap2=gap2, lam=1.0, beta=0.03, seed=seed, steps=6000, p=1.0, lr=3e-4, l2=True))
elif name == "gap_formal":   # unified fair config for both variants: lr=3e-4, raw (non-L2) VQ, 6000 steps, 8 seeds
    for gap2, lam, seed in itertools.product([1, 3, 6, 10, 15], [0.0, 1.0], range(8)):
        jobs.append(dict(env="gap", gap2=gap2, lam=lam, beta=0.03, seed=seed, steps=6000, p=1.0, lr=3e-4, l2=False))
elif name == "ablation":     # factorized ablation on the gap toy (gap2=6), unified config, -R objective
    base = dict(env="gap", gap2=6, lam=0.0, steps=6000, lr=3e-4, l2=False)
    regimes = [dict(beta=0.03, p=1.0, n_train=0), dict(beta=0.1, p=0.8, n_train=512)]
    for v, rg, seed in itertools.product(["diacritic", "bypass", "uncond", "nonpersistent", "continuous"], regimes, range(4)):
        jobs.append(dict(base, variant=v, seed=seed, **rg))
    for rg, seed in itertools.product(regimes, range(4)):
        jobs.append(dict(base, variant="diacritic", seed=seed, **dict(rg, beta=0.0)))
    for beta, rg, seed in itertools.product([0.003, 0.01, 0.3], regimes, range(4)):
        jobs.append(dict(base, variant="continuous", seed=seed, **dict(rg, beta=beta)))
elif name == "bypass_beta":  # bypass must first *succeed* (weak/no rate pressure) to show that its C-rate under-counts Gamma
    base = dict(env="gap", gap2=6, lam=0.0, steps=6000, lr=3e-4, l2=False, variant="bypass")
    for beta, rg, seed in itertools.product([0.0, 0.003, 0.01], [dict(p=1.0, n_train=0), dict(p=0.8, n_train=512)], range(4)):
        jobs.append(dict(base, beta=beta, seed=seed, **rg))
elif name == "bypass_tune":
    for lr, l2, beta, seed in itertools.product([1e-3, 3e-3], [True, False], [0.0, 0.003], range(3)):
        jobs.append(dict(env="gap", gap2=6, lam=0.0, steps=6000, lr=lr, l2=l2, variant="bypass", beta=beta, seed=seed, p=1.0, n_train=0))
elif name == "refine_toy":   # D1: future-sufficiency-triggered code refinement vs -R / -RF on the gap toy (unified config)
    base = dict(env="gap", steps=6000, lr=3e-4, l2=False, beta=0.03, p=1.0)
    for gap2, seed in itertools.product([6, 10, 15], range(4)):
        jobs.append(dict(base, gap2=gap2, lam=0.0, seed=seed))                                   # -R
        jobs.append(dict(base, gap2=gap2, lam=1.0, seed=seed))                                   # -RF
        jobs.append(dict(base, gap2=gap2, lam=0.0, seed=seed, refine="ce"))                      # -R + refine(ce)
        jobs.append(dict(base, gap2=gap2, lam=1.0, seed=seed, refine="bfs"))                     # -RF + refine(bfs)
elif name == "scaffold_toy":   # D2: continuous scaffold annealed to a sole discrete carrier, vs -R / -RF (same unified config)
    base = dict(env="gap", steps=6000, lr=3e-4, l2=False, beta=0.03, p=1.0)
    for gap2, seed in itertools.product([6, 10, 15], range(4)):
        jobs.append(dict(base, gap2=gap2, lam=0.0, seed=seed, variant="scaffold"))
        jobs.append(dict(base, gap2=gap2, lam=1.0, seed=seed, variant="scaffold"))
elif name == "leak_reveal":   # appendix: (A2) violated by Bernoulli reveal loss; solver flags it; code targets p(a|h)
    for leak, beta, lam, seed in itertools.product([0.3], [0.03, 0.1], [0.0, 1.0], range(4)):
        jobs.append(dict(env="reveal", lam=lam, beta=beta, seed=seed, steps=2000, p=1.0, leak=leak))
else:
    raise SystemExit(name)
for j in jobs: print(json.dumps(j))
