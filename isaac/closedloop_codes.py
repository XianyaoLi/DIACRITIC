"""Theorem quantities on the policy's OWN occupancy: from closed-loop trajectories (aprime_env.py --policy --out), compute per phase
H(C|Ō), S_Γ-style sufficiency of the closed-loop code for β₁ / β₂ (H(β|Ō) - H(β|C,Ō)) / H(β|Ō), and compare with the teacher-forced
values.  Ō = (phase, probe, visible classes) as in aprime_data.  Usage: python isaac/closedloop_codes.py <eval dir> [more]"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toy")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamma_solver import cond_entropy
from aprime_data import visibility
for d in sys.argv[1:]:
    eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "ep*_env*.npz")))]
    eps = [dict(obs=e["obs"], code=e["code"], b1=int(e["b1"]), b2=int(e["b2"]), phase=list(e["phase"]), sym_obs=[eval(x) for x in e["sym_obs"]]) for e in eps if len(e["code"])]
    if not eps: print(d, "no codes logged"); continue
    meta = dict(R1=2, R2=2); vis = visibility(eps, meta)
    T = len(eps[0]["phase"]); w = [1.0 / len(eps)] * len(eps); ph = eps[0]["phase"]
    print(f"\n{d}: {len(eps)} closed-loop episodes")
    print("  phase   H(C|O)  suff(b1)  suff(b2)   [closed-loop code, own occupancy]")
    for phase in ["scan", "gap1", "grasp", "gap2", "place"]:
        idx = [t for t, p in enumerate(ph) if p == phase]; rows = []
        for t in idx:
            o = [(e["sym_obs"][t][0], e["sym_obs"][t][1], e["b1"] if vis["b1"][t] else -1, e["b2"] if vis["b2"][t] else -1) for e in eps]
            c = [int(e["code"][t]) for e in eps]; b1 = [e["b1"] for e in eps]; b2 = [e["b2"] for e in eps]
            HC = cond_entropy(c, o, w)
            def suff(b):
                Hb = cond_entropy(b, o, w); return (Hb - cond_entropy(b, list(zip(c, o)), w)) / Hb if Hb > 1e-9 else float("nan")
            rows.append((HC, suff(b1), suff(b2)))
        r = np.nanmean(np.array(rows, float), 0); print(f"  {phase:6s}  {r[0]:5.2f}    {r[1]:5.2f}     {r[2]:5.2f}")
