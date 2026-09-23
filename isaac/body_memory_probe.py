"""Closed-loop 'body as memory' probe: does the learned policy write a hidden class into its own body pose?
For each phase, class-conditional mean tcp position separation (cm) by beta_1 / beta_2 with per-axis t-tests, on closed-loop
trajectories written by `aprime_env.py --policy ... --out <dir>`.  On expert data this is the visibility rule of aprime_data.py;
on closed-loop data a separation >> sensor noise in a gap phase means the policy stores the class in the observation it controls
(a memory channel the conditional rate does not charge).   Usage: python isaac/body_memory_probe.py <eval dir> [more dirs]"""
import sys, glob, json, os
import numpy as np
from scipy import stats
for d in sys.argv[1:]:
    eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "ep*_env*.npz")))]
    if not eps: print(d, "no trajectories"); continue
    ph = list(eps[0]["phase"]); idx = {p: [i for i, q in enumerate(ph) if q == p] for p in dict.fromkeys(ph)}
    labs = {"b1": np.array([int(e["b1"]) for e in eps]), "b2": np.array([int(e["b2"]) for e in eps])}
    succ = np.mean([bool(e["success"]) for e in eps]); out = {}
    print(f"\n{d}: {len(eps)} episodes, success {succ:.3f}")
    print("  phase   class  sep(cm)  min p(axis)   verdict")
    for phase in ["gap1", "grasp", "gap2", "place"]:
        if phase not in idx: continue
        tcp = np.stack([e["obs"][idx[phase], :3].mean(0) for e in eps])
        for name, lab in labs.items():
            cls = np.unique(lab); means = [tcp[lab == c].mean(0) for c in cls]
            sep = max(np.linalg.norm(means[i] - means[j]) for i in range(len(cls)) for j in range(i + 1, len(cls))) * 100
            p = min(stats.ttest_ind(tcp[lab == cls[0]], tcp[lab == cls[1]], axis=0).pvalue) if len(cls) == 2 else float("nan")
            flag = "BODY MEMORY" if (phase in ("gap1", "gap2") and sep > 0.5 and p < 1e-3) else ""
            print(f"  {phase:6s}  {name}   {sep:6.2f}   {p:9.1e}   {flag}"); out[f"{phase}_{name}"] = dict(sep_cm=float(sep), p=float(p))
    json.dump(dict(dir=d, episodes=len(eps), success=float(succ), probe=out), open(os.path.join(d, "body_memory_probe.json"), "w"), indent=1)
