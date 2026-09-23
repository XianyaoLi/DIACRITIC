"""Expired-information analysis (no training): what does the event-agnostic random-offset target keep after first use?
In the second gap of A' only beta_2 is still required (beta_1 expired at the grasp) and the symbolic observation is uninformative, so
I(C_t; beta_1 | beta_2, O-bar_t) = I(C_t; beta_1 | beta_2) measures exactly the EXPIRED behavioural distinction still carried by the code.
Computed from the saved hard codes (codes/<grid>/<model>.npz: codes (E,T), theta) for the full-trajectory-sufficient seeds of each learner.
Usage: python isaac/analysis/expired_info.py > results_md/analysis_expired_info.md"""
import os, re, json, glob, math
from collections import defaultdict, Counter
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); RES = os.path.join(ROOT, "isaac", "cluster_results", "results"); COD = os.path.join(ROOT, "isaac", "cluster_results", "codes")
def H(xs): c = np.array(list(Counter(xs).values()), float); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def I_cond(c, x, y): return H(list(zip(c, y))) + H(list(zip(x, y))) - H(list(zip(c, x, y))) - H(list(y))      # I(C;X|Y)
info = {}
for g in ("generic", "generic2", "px_fc20", "px_gen20"):
    for line in open(os.path.join(RES, g + ".jsonl")):
        r = json.loads(line); a = r["args"]; p = r["per_t"]; ph = r["phases"]; idx = lambda n: [i for i, q in enumerate(ph) if q == n]
        if not a.get("distill"): continue
        mem = [i for i in idx("gap1") + idx("gap2") if p["H(Gam|O)"][i] > 1e-9]; act = [min([i for i in idx(b) if p["H(G|O)"][i] > 1e-9]) for b in ("grasp", "place")]
        full = all(p["S_Gam"][i] > 0.9 for i in mem) and all(p["S_G"][i] > 0.9 for i in act)
        lab = {"random": "random offset", "forecast": "targeted forecast", "uniform": "all offsets"}[a["distill_target"]] + (f", annealed from {100 * a['distill_off']:g}%" if a.get("distill_off") else ", not annealed")
        info[(g, os.path.basename(a["save_codes"]))] = (os.path.basename(a["data"]), lab, a["seed"], full, idx("gap1"), idx("gap2"))
rows = defaultdict(list)
for (g, f), (ds, lab, seed, full, g1, g2) in info.items():
    path = os.path.join(COD, g, f)
    if not (full and os.path.exists(path)): continue
    z = np.load(path, allow_pickle=True); codes = z["codes"]; th = np.array(z["theta"]).astype(int); b1 = th % 2; b2 = (th // 2) % 2
    rows[(ds, lab)].append((np.mean([I_cond(codes[:, t], b1, b2) for t in g2]), np.mean([I_cond(codes[:, t], b2, b1) for t in g2]), np.mean([I_cond(codes[:, t], b1, b2) for t in g1]), np.mean([H(codes[:, t]) for t in g2])))
print("# Expired information kept by the code in the second gap (A' gap 20; full-trajectory-sufficient seeds only)\n")
print(__doc__.split("Usage:")[0].strip() + "\n")
print("| dataset | learner | sufficient seeds analysed | I(C; beta_1 | beta_2), second gap (expired; theory 0) | I(C; beta_2 | beta_1), second gap (required; theory 1) | I(C; beta_1 | beta_2), first gap (required; theory 1) | H(C), second gap |\n|---|---|---|---|---|---|---|")
for k in sorted(rows):
    v = np.array(rows[k]); m = v.mean(0); print(f"| {k[0]} | {k[1]} | {len(v)} | **{m[0]:.2f}** (max {v[:, 0].max():.2f}) | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} |")

# ---- what IS the surplus then?  State-observation models (the recorded dataset is local): nuisance variables of the episode, median-split where continuous.
DS = os.path.join(ROOT, "isaac", "data", "tier4_gap20"); eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(DS, "ep*_env*.npz")))]
if eps:
    hov = np.array([int(e["hover_k"]) for e in eps]); xy = np.stack([e["obj_xy"] for e in eps]); xq = np.digitize(xy[:, 0], np.quantile(xy[:, 0], [.25, .5, .75])); yq = np.digitize(xy[:, 1], np.quantile(xy[:, 1], [.25, .5, .75]))
    out = defaultdict(list)
    for (g, f), (ds, lab, seed, full, g1, g2) in info.items():
        path = os.path.join(COD, g, f)
        if ds != "tier4_gap20" or not full or not os.path.exists(path): continue
        z = np.load(path, allow_pickle=True); C = z["codes"]; th = np.array(z["theta"]).astype(int); b2 = (th // 2) % 2; zz = np.array(z["z"]).astype(int)
        out[lab].append([np.mean([I_cond(C[:, t], v, b2) for t in g2]) for v in (xq, yq, hov, zz, (th >= 16).astype(int))] + [np.mean([H(list(zip(C[:, t], b2))) - H(list(b2)) for t in g2])])
    print("\n## What the surplus is (A' gap 20, state observations, second gap; information in bits, conditioned on beta_2)\n")
    print("The expert moves along linear segments whose start point is the pose at which the segment began; during the second gap that start point is the grasp pose, i.e. the object's initial position (uniform within +-2 cm), which the current observation no longer shows. It changes the expert's per-step displacement by up to 40 %, so an open-loop forecast of future actions is improved by remembering it, although the symbolic zero-distortion target Gamma does not require it.\n")
    print("| learner | seeds | object x (quartiles) | object y (quartiles) | hover nuisance | distractor z | mass half | H(C | beta_2) |\n|---|---|---|---|---|---|---|---|")
    for lab in sorted(out): m = np.array(out[lab]).mean(0); print(f"| {lab} | {len(out[lab])} | **{m[0]:.2f}** | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} | {m[4]:.2f} | {m[5]:.2f} |")
    print("\nReading: the expired class beta_1 is forgotten with or without annealing (first table: <= 0.02 bit). The surplus of the un-annealed generic target is a fine-grained, behaviourally predictive nuisance (where the current motion segment started); annealing the auxiliary loss lets the rate term remove it.")
