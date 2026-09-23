"""Compact per-eval-dir summary (run where the trajectories live, e.g. on the cluster): closed-loop success/side/slot + own-occupancy code
sufficiency + body-memory separation -> <dir>/summary.json (small; pull these instead of the npz trajectories).
Usage: python isaac/eval_summarize_dir.py <eval dir> [more]"""
import sys, os, glob, json, ast
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "toy")); sys.path.insert(0, HERE)
from gamma_solver import cond_entropy
from aprime_data import visibility
for d in sys.argv[1:]:
    f = os.path.join(d, "closedloop_eval.jsonl")
    if not os.path.isfile(f): continue
    ev = json.loads(open(f).readlines()[-1]); out = dict(model=os.path.basename(d.rstrip("/")), success=ev["success_rate"], side=ev["side_acc"], slot=ev["slot_acc"], err=ev["action_err_cm"], episodes=ev["episodes"])
    eps = []
    for g in sorted(glob.glob(os.path.join(d, "ep*_env*.npz"))):
        z = np.load(g, allow_pickle=True)
        if len(z["code"]) == 0: continue
        eps.append(dict(obs=z["obs"], code=z["code"], b1=int(z["b1"]), b2=int(z["b2"]), phase=list(z["phase"]), sym_obs=[ast.literal_eval(x) for x in z["sym_obs"]]))
    if eps:
        R1 = int(max(e["b1"] for e in eps)) + 1; R2 = int(max(e["b2"] for e in eps)) + 1
        vis = visibility(eps, dict(R1=R1, R2=R2)); ph = eps[0]["phase"]; w = [1.0 / len(eps)] * len(eps)
        for phase in ["gap1", "gap2"]:
            idx = [t for t, p in enumerate(ph) if p == phase]; hc, s1, s2 = [], [], []
            for t in idx:
                o = [(e["sym_obs"][t][0], e["sym_obs"][t][1], e["b1"] if vis["b1"][t] else -1, e["b2"] if vis["b2"][t] else -1) for e in eps]
                c = [int(e["code"][t]) for e in eps]; b1 = [e["b1"] for e in eps]; b2 = [e["b2"] for e in eps]
                hc.append(cond_entropy(c, o, w))
                for b, acc in ((b1, s1), (b2, s2)):
                    Hb = cond_entropy(b, o, w); acc.append((Hb - cond_entropy(b, list(zip(c, o)), w)) / Hb if Hb > 1e-9 else float("nan"))
            out[f"cl_HC_{phase}"] = float(np.nanmean(hc)); out[f"cl_suff_b1_{phase}"] = float(np.nanmean(s1)); out[f"cl_suff_b2_{phase}"] = float(np.nanmean(s2))
            tcp = np.stack([e["obs"][idx, :3].mean(0) for e in eps]); lab = np.array([e["b2"] for e in eps])
            means = [tcp[lab == c].mean(0) for c in range(R2) if (lab == c).any()]
            out[f"body_sep_b2_{phase}_cm"] = float(max(np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))) * 100) if len(means) > 1 else 0.0
    json.dump(out, open(os.path.join(d, "summary.json"), "w")); print(json.dumps(out))
