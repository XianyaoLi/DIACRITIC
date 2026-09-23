"""Surplus decomposition (no training): how much of the un-annealed event-agnostic code's second-gap surplus is explained by episode-level nuisances?
Surplus at a second-gap step t = H(C_t | beta_2) (Gamma_t = beta_2 there and the symbolic observation is uninformative).  For a feature set X we
report a CROSS-VALIDATED LOWER bound on I(C_t; X | beta_2) = H(C_t | beta_2) - H(C_t | beta_2, X): the second term is upper-bounded by the held-out
cross-entropy of a classifier predicting the code from (beta_2, X) (5-fold, gradient-boosted trees), so the explained fraction is conservative
(a plug-in estimate on binned features would be biased upwards with 576 episodes).
Usage: python isaac/analysis/surplus_decomposition.py > results_md/analysis_surplus_decomposition.md"""
import os, glob, json, math, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); RES = os.path.join(ROOT, "isaac/cluster_results/results"); COD = os.path.join(ROOT, "isaac/cluster_results/codes")
eps = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(ROOT, "isaac/data/tier4_gap20/ep*_env*.npz")))]
ph = list(eps[0]["phase"]); g2 = [i for i, p in enumerate(ph) if p == "gap2"]; grasp = [i for i, p in enumerate(ph) if p == "grasp"]; obs = np.stack([e["obs"] for e in eps])
xy = np.stack([e["obj_xy"] for e in eps]); hov = np.array([int(e["hover_k"]) for e in eps])
def H(xs): c = np.array(list(Counter(xs).values()), float); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def cv_ce(X, y, groups_b2):
    """held-out cross-entropy (bits) of y given X, 5-fold; classes unseen in a training fold get a small floor probability"""
    yv, yi = np.unique(y, return_inverse=True); ce = np.zeros(len(y)); skf = StratifiedKFold(5, shuffle=True, random_state=0)
    if len(yv) == 1: return 0.0
    for tr, te in skf.split(X, np.minimum(yi, 10 ** 6)):
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=0).fit(X[tr], yi[tr]); P = np.full((len(te), len(yv)), 1e-3)
        P[:, clf.classes_] = np.maximum(clf.predict_proba(X[te]), 1e-3); P /= P.sum(1, keepdims=True); ce[te] = -np.log2(P[np.arange(len(te)), yi[te]])
    return float(ce.mean())
info = {}
for g in ("generic", "generic2"):
    for line in open(os.path.join(RES, g + ".jsonl")):
        r = json.loads(line); a = r["args"]
        if os.path.basename(a["data"]) != "tier4_gap20" or a.get("distill_target") != "random" or (a.get("distill_w") or 1) != 1: continue
        p = r["per_t"]; idx = lambda n: [i for i, q in enumerate(r["phases"]) if q == n]
        mem = [i for i in idx("gap1") + idx("gap2") if p["H(Gam|O)"][i] > 1e-9]; act = [min([i for i in idx(b) if p["H(G|O)"][i] > 1e-9]) for b in ("grasp", "place")]
        if all(p["S_Gam"][i] > 0.9 for i in mem) and all(p["S_G"][i] > 0.9 for i in act):
            info[(g, os.path.basename(a["save_codes"]))] = "not annealed" if not a.get("distill_off") else f"annealed from {100 * a['distill_off']:g}%"
t = g2[len(g2) // 2]
act = np.stack([e["act"] for e in eps]); place = [i for i, p in enumerate(ph) if p == "place"]
FEATS = [("expert's own future displacements (mean action over the rest of gap 2 and over the place block)", lambda th, z: np.c_[act[:, [i for i in g2 if i >= t], :3].mean(1), act[:, place[:3], :3].mean(1)]),
         ("object x, y (true initial position)", lambda th, z: xy),
         ("+ hover nuisance, distractor, mass", lambda th, z: np.c_[xy, hov, z, th]),
         ("+ noisy observations at the grasp steps (tcp, gripper, object pose)", lambda th, z: np.c_[xy, hov, z, th, obs[:, grasp, :15].reshape(len(eps), -1)]),
         ("+ noisy observations of the whole history up to t", lambda th, z: np.c_[xy, hov, z, th, obs[:, :t + 1, :17].reshape(len(eps), -1)])]
print("# Decomposition of the second-gap surplus of the event-agnostic code (A' gap 20, state observations, full-trajectory-sufficient seeds)\n")
print(__doc__.split("Usage:")[0].strip() + f"\n\nStep analysed: t = {t} (middle of the second gap); 576 episodes; all values in bits, means over seeds (min–max in brackets).\n")
print("| learner | seeds | surplus H(C \\| beta_2) | " + " | ".join(n for n, _ in FEATS) + " |\n|---|---|---|" + "---|" * len(FEATS))
rows = {}
for (g, f), lab in sorted(info.items(), key=lambda kv: kv[1]):
    path = os.path.join(COD, g, f)
    if not os.path.exists(path): continue
    zc = np.load(path, allow_pickle=True); C = zc["codes"][:, t]; th = np.array(zc["theta"]).astype(int); b2 = (th // 2) % 2; zz = np.array(zc["z"]).astype(int)
    sur = H(list(zip(C, b2))) - H(list(b2)); ex = []
    for _, fx in FEATS:
        X = np.c_[b2, fx(th, zz)].astype(float); ex.append(max(0.0, sur - cv_ce(X, C, b2)))
    rows.setdefault(lab, []).append([sur] + ex)
for lab, v in rows.items():
    v = np.array(v); m = v.mean(0)
    print(f"| {lab} | {len(v)} | {m[0]:.2f} [{v[:,0].min():.2f}–{v[:,0].max():.2f}] | " + " | ".join(f"{m[i]:.2f} ({100 * m[i] / max(m[0], 1e-9):.0f}%) [{v[:,i].min():.2f}–{v[:,i].max():.2f}]" for i in range(1, v.shape[1])) + " |")
