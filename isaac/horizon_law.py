"""Horizon law: pooled logistic regression of P(sufficient) on reveal->use distance, log N, beta, variant, class identity
(numpy/scipy only).  Rows: one per (run, class): beta_1 (horizon = steps from reveal to grasp; outcome S_G grasp > 0.9) and
beta_2 (horizon to first place step; outcome S_Gam gap2 > 0.9 & S_G place > 0.9).  A' tier-4 state-observation runs of the unsupervised sole-carrier learners only (variants diacritic / scaffold incl. refine, curriculum,
step-budget and coverage variants); bypass / Transformer (memory outside the code), privileged (oracle) heads, pixels and task A are excluded.
Inference: the two rows of a run are not independent (same weights), and runs of one configuration share everything but the
seed, so standard errors are cluster-robust (sandwich) with clusters = run and, more conservatively, clusters = configuration
(dataset, variant, beta, lam, N, refine, curric); plus a cluster bootstrap (resampling runs / configurations with replacement).
Usage: python isaac/horizon_law.py <jsonl files...>"""
import sys, json, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist
rows = []
for f in sys.argv[1:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        a = r.get("args", {}); ds = os.path.basename(a.get("data", "")); ph = r.get("phases")
        if not ph or "gap2" not in ph or not ds.startswith("tier4") or ds.endswith("_px"): continue          # A' tier-4, state observations
        if a.get("variant", "diacritic") not in ("diacritic", "scaffold") or a.get("aux_theta", 0) or a.get("aux_join", 0) or "smoke" in a.get("out", ""): continue   # unsupervised sole-carrier learners only
        reveal_end = max(i for i, p in enumerate(ph) if p == "scan"); h1 = ph.index("grasp") - reveal_end; h2 = ph.index("place") - reveal_end
        d = r["summary"]; var = a.get("variant", "diacritic"); N = a.get("n_train", 448)
        cfg = (ds, var, a["beta"], a.get("lam", 0), N, a.get("refine", ""), bool(a.get("init_from")))
        base = dict(logN=np.log(N), beta_k=a["beta"] * 1000, lam=int(a.get("lam", 0) > 0), scaffold=int(var == "scaffold"), refine=int(bool(a.get("refine"))), curric=int(bool(a.get("init_from"))), run=len(rows) // 2, cfg=cfg)
        rows.append(dict(base, dist=h1, cls=0, y=int(d["S_G_grasp"] > 0.9)))
        rows.append(dict(base, dist=h2, cls=1, y=int(d["S_Gam_gap2"] > 0.9 and d["S_G_place"] > 0.9)))
y = np.array([r["y"] for r in rows], float); run_id = np.array([r["run"] for r in rows])
cfg_keys = sorted({r["cfg"] for r in rows}); cfg_id = np.array([cfg_keys.index(r["cfg"]) for r in rows])
print(f"{len(rows)} (run, class) rows from {len(rows)//2} runs in {len(cfg_keys)} configurations; distances {sorted({r['dist'] for r in rows})}")

def design(cols, rs=None):
    rs = rows if rs is None else rs
    return np.column_stack([np.ones(len(rs))] + [[r[c] for r in rs] for c in cols])
def mle(X, yy):
    nll = lambda w: -np.sum(yy * (X @ w) - np.logaddexp(0, X @ w)) + 1e-3 * np.sum(w[1:] ** 2)
    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
def sandwich(X, yy, w, groups):
    p = 1 / (1 + np.exp(-(X @ w))); bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 2e-3 * np.eye(X.shape[1]))
    S = X * (yy - p)[:, None]; G = len(np.unique(groups)); meat = np.zeros((X.shape[1],) * 2)
    for g in np.unique(groups): s = S[groups == g].sum(0); meat += np.outer(s, s)
    meat *= G / (G - 1) * (len(yy) - 1) / (len(yy) - X.shape[1])          # small-sample (CR1) correction
    return np.sqrt(np.diag(bread @ meat @ bread)), G
def boot(cols, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx_of = {g: np.where(groups == g)[0] for g in ug}; W = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True); idx = np.concatenate([idx_of[g] for g in pick])
        W.append(mle(design(cols, [rows[i] for i in idx]), y[idx]))
    return np.array(W)
def fit(cols):
    X = design(cols); w = mle(X, y); p = 1 / (1 + np.exp(-(X @ w)))
    ll0 = -np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())); pr2 = 1 - (-np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / ll0
    se_run, G_run = sandwich(X, y, w, run_id); se_cfg, G_cfg = sandwich(X, y, w, cfg_id)
    Wb_run = boot(cols, run_id); Wb_cfg = boot(cols, cfg_id)
    print(f"\n[{' + '.join(cols)}]  n={len(rows)}  pseudo-R2={pr2:.3f}   (SE: cluster-robust by run G={G_run} / by configuration G={G_cfg}; bootstrap 2000 resamples of runs / configurations)")
    for i, c in enumerate(cols, 1):
        zr = w[i] / se_run[i]; zc = w[i] / se_cfg[i]; pc = 2 * (1 - tdist.cdf(abs(zc), G_cfg - 1))
        lo_r, hi_r = np.percentile(Wb_run[:, i], [2.5, 97.5]); lo_c, hi_c = np.percentile(Wb_cfg[:, i], [2.5, 97.5])
        print(f"   {c:9s} coef {w[i]:+.3f} | run-cluster SE {se_run[i]:.3f} p={2*(1-norm.cdf(abs(zr))):.1e}  boot CI ({lo_r:+.3f},{hi_r:+.3f})"
              f" | config-cluster SE {se_cfg[i]:.3f} p={pc:.1e} (t, df={G_cfg-1})  boot CI ({lo_c:+.3f},{hi_c:+.3f})")
fit(["dist"]); fit(["dist", "logN", "beta_k", "lam", "scaffold", "refine", "curric"]); fit(["dist", "logN", "beta_k", "lam", "scaffold", "refine", "curric", "cls"])
print("\nempirical P(sufficient) by distance:")
for dd in sorted({r["dist"] for r in rows}):
    ys = [r["y"] for r in rows if r["dist"] == dd]; print(f"   d={dd:2d}: {np.mean(ys):.2f}  (n={len(ys)})")
