"""Horizon law on the signpost corridor: logistic regression of P(class sufficient) on the reveal->use distance, pooled over the W=1
unsupervised sole-carrier runs (variants diacritic / scaffold) of the grid_horizon and grid_ladder grids.  Rows = (run, class):
beta_1 (distance = last sign -> junction 1 = gap1 + 1; outcome err(use1) = 0 and S_Gam > 0.9 in hall 1) and beta_2 (distance = last sign ->
junction 2; outcome err(use2) = 0 and S_Gam > 0.9 in both halls).  Cluster-robust SE by configuration + configuration bootstrap.
Usage: python toy/grid_horizon_law.py toy/results/grid_horizon.jsonl toy/results/grid_ladder.jsonl"""
import sys, re, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from grid_summarize import read_records
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist
rows = []
for f in sys.argv[1:]:
    for r in read_records(f):
        a = r["args"]; m = re.search(r"M=(\d+),R1=\d+,R2=\d+,W=(\d+),N=\d+,gap1=(\d+),gap2=(\d+)", r["env"])
        if int(m.group(2)) != 1 or a["variant"] not in ("diacritic", "scaffold") or int(m.group(1)) != 16: continue
        ph = r["phases"]; res = r["res"]; rev_end = ph["reveal"][1]
        h1 = list(range(ph["gap1"][0] - 1, ph["gap1"][1])); h2 = list(range(ph["gap2"][0] - 1, ph["gap2"][1]))
        ok1 = res["err"][ph["use1"] - 1] < 1e-6 and all(res["S_Gam"][i] > 0.9 for i in h1)
        ok2 = res["err"][ph["use2"] - 1] < 1e-6 and all(res["S_Gam"][i] > 0.9 for i in h1 + h2)
        cfg = (a["variant"], a["beta"], a["lam"], int(m.group(3)), int(m.group(4)))
        base = dict(lam=int(a["lam"] > 0), scaffold=int(a["variant"] == "scaffold"), beta_k=a["beta"] * 100, cfg=cfg, run=len(rows) // 2)
        rows.append(dict(base, dist=ph["use1"] - rev_end, cls=0, y=int(ok1))); rows.append(dict(base, dist=ph["use2"] - rev_end, cls=1, y=int(ok2)))
y = np.array([r["y"] for r in rows], float); cfg_keys = sorted({r["cfg"] for r in rows}); cid = np.array([cfg_keys.index(r["cfg"]) for r in rows])
print(f"{len(rows)} rows from {len(rows)//2} runs in {len(cfg_keys)} configurations; distances {sorted({r['dist'] for r in rows})}")
def design(cols, rs=None): rs = rows if rs is None else rs; return np.column_stack([np.ones(len(rs))] + [[r[c] for r in rs] for c in cols])
def mle(X, yy): return minimize(lambda w: -np.sum(yy * (X @ w) - np.logaddexp(0, X @ w)) + 1e-3 * np.sum(w[1:] ** 2), np.zeros(X.shape[1]), method="BFGS").x
def sandwich(X, yy, w, groups):
    p = 1 / (1 + np.exp(-(X @ w))); bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 2e-3 * np.eye(X.shape[1])); S = X * (yy - p)[:, None]
    G = len(np.unique(groups)); meat = sum(np.outer(S[groups == g].sum(0), S[groups == g].sum(0)) for g in np.unique(groups)) * G / (G - 1) * (len(yy) - 1) / (len(yy) - X.shape[1])
    return np.sqrt(np.diag(bread @ meat @ bread)), G
def boot(cols, groups, B=2000):
    rng = np.random.default_rng(0); ug = np.unique(groups); idx_of = {g: np.where(groups == g)[0] for g in ug}; W = []
    for _ in range(B):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug), replace=True)]); W.append(mle(design(cols, [rows[i] for i in idx]), y[idx]))
    return np.array(W)
def fit(cols, sel=None):
    global rows, y, cid
    if sel is not None:
        keep = [i for i, r in enumerate(rows) if sel(r)]; R, Y, C = [rows[i] for i in keep], y[keep], cid[keep]
    else: R, Y, C = rows, y, cid
    rows0, y0, cid0 = rows, y, cid; rows, y, cid = R, Y, C
    X = design(cols); w = mle(X, y); se, G = sandwich(X, y, w, cid); Wb = boot(cols, cid)
    print(f"\n[{' + '.join(cols)}] n={len(rows)} G={G}" + ("" if sel is None else "  (subset)"))
    for i, c in enumerate(cols, 1):
        lo, hi = np.percentile(Wb[:, i], [2.5, 97.5]); z = w[i] / se[i]
        print(f"   {c:9s} coef {w[i]:+.3f}  config-cluster SE {se[i]:.3f} p={2*(1-tdist.cdf(abs(z), G-1)):.1e}  boot CI ({lo:+.3f},{hi:+.3f})")
    rows, y, cid = rows0, y0, cid0
fit(["dist"]); fit(["dist", "lam", "scaffold", "beta_k"]); fit(["dist", "lam", "scaffold", "beta_k", "cls"])
fit(["dist"], sel=lambda r: r["scaffold"] == 0); fit(["dist"], sel=lambda r: r["scaffold"] == 1)
print("\nempirical P(sufficient) by distance and variant:")
for lab, f in (("-R", lambda r: r["lam"] == 0 and not r["scaffold"]), ("-RF", lambda r: r["lam"] == 1 and not r["scaffold"]), ("scaffold-RF", lambda r: r["scaffold"] == 1)):
    print(f"  {lab:12s} " + "  ".join(f"d={d}:{np.mean([r['y'] for r in rows if r['dist']==d and f(r)]):.2f}(n={sum(1 for r in rows if r['dist']==d and f(r))})" for d in sorted({r['dist'] for r in rows if f(r)})))
