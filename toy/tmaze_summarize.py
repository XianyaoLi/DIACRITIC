"""Summarise T-maze runs (toy/results/tmaze.jsonl): per (R, L, variant, lam): seeds sufficient (S_Gam > 0.9 at every corridor step and err = 0
at the junction), corridor rate H(C|O) vs log2 R, surplus, and a logistic slope of P(sufficient) on L per variant.
Usage: python toy/tmaze_summarize.py toy/results/tmaze.jsonl"""
import sys, re, collections, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from grid_summarize import read_records
from scipy.optimize import minimize
rows = [r for f in sys.argv[1:] for r in read_records(f)]
g = collections.defaultdict(list)
for r in rows:
    m = re.search(r"tmaze\(R=(\d+),N=\d+,L=(\d+)\)", r["env"]); a = r["args"]
    g[(int(m.group(1)), int(m.group(2)), a["variant"], int(a["lam"] > 0))].append(r)
def suff(r):
    res = r["res"]; T = len(res["err"]); cor = range(1, T - 1)
    return all(res["S_Gam"][i] > 0.9 for i in cor) and res["err"][T - 1] < 1e-6
print(f"{'R':>2s} {'L':>3s} {'variant':10s} lam n | suff | corridor H(C|O) [log2 R] (suff seeds) | surplus H(C|Gam,O) | err junction (all)")
per_var = collections.defaultdict(list)
for k, v in sorted(g.items()):
    S = [r for r in v if suff(r)]; T = len(v[0]["res"]["err"]); cor = range(1, T - 1)
    hc = np.mean([np.mean([r["res"]["H(C|O)"][i] for i in cor]) for r in (S or v)]); sur = np.mean([np.mean([r["res"]["H(C|Gam,O)"][i] for i in cor]) for r in (S or v)])
    print(f"{k[0]:2d} {k[1]:3d} {k[2]:10s} {k[3]}   {len(v)} | {len(S)}/{len(v)} |   {hc:.2f} [{np.log2(k[0]):.2f}]  |  {sur:.2f}  |  {np.mean([r['res']['err'][T-1] for r in v]):.2f}")
    for r in v: per_var[(k[0], k[2], k[3])].append((k[1], int(suff(r))))
print("\nlogistic slope of P(sufficient) on corridor length L:")
for k, R in sorted(per_var.items()):
    X = np.array([[1, l] for l, _ in R], float); y = np.array([s for _, s in R], float)
    if 0 < y.mean() < 1:
        w = minimize(lambda w: -np.sum(y * (X @ w) - np.logaddexp(0, X @ w)) + 1e-3 * w[1] ** 2, np.zeros(2), method="BFGS").x
        print(f"  R={k[0]} {k[1]:10s} lam={k[2]}: slope {w[1]:+.3f}/step  (n={len(R)})")
    else: print(f"  R={k[0]} {k[1]:10s} lam={k[2]}: P(sufficient) constant {y.mean():.2f} (n={len(R)})")
