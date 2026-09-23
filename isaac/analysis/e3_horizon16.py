"""E3: controlled within-learner horizon sweep (16 seeds) -- does P(sufficient) fall with the reveal->use distance inside every learner?

Data.  Runs with n_train=448, beta=0.002, steps=10000, no refine / init_from / aux heads, datasets tier4_gap6 / gap10 / gap20
(gap2 = 6/10/20, gap1 = 2) and tier4_g1_10 / g1_20 (gap1 = 10/20, gap2 = 6); learners -R (diacritic, lam 0), -RF (diacritic, lam 1),
scaffold-RF (scaffold, lam 1).  All *.jsonl* under isaac/cluster_results/results and isaac/results are scanned; identical
(dataset, variant, lam, seed) keys are de-duplicated by keeping the FIRST occurrence in the file order below (the first row inside a
file wins too); the discarded duplicates are listed with a flag saying whether they are numerically identical re-runs.

Rows.  Exactly as isaac/horizon_law.py: one row per (run, class).  beta_1: distance = first 'grasp' step - last 'scan' step, outcome
S_G_grasp > thr.  beta_2: distance = first 'place' step - last 'scan' step, outcome S_Gam_gap2 > thr and S_G_place > thr.  The summary
fields are nanmean of per_t['S_Gam'] over the gap2 steps and of per_t['S_G'] over the grasp / place steps (train_diacritic.py pm());
they are recomputed here from per_t (and asserted equal to the stored summary) so that thr = 0.8 / 0.95 uses the same aggregation.

Inference.  Logistic regression by numpy/scipy (BFGS on the negative log-likelihood with the same 1e-3 ridge on the non-intercept
weights as horizon_law.py -- negligible away from separation, keeps the slope finite under quasi-separation).  Standard errors are
cluster-robust (CR1 sandwich) with clusters = run (the two class rows of a run share the weights).  Wald tests use those SEs; the
likelihood-ratio test uses the unpenalised log-likelihood at the fitted weights; interval-valued effects use the delta method with the
cluster-robust covariance; raw proportions use Wilson intervals and Newcombe (method 10) intervals for their differences.
Usage: python isaac/analysis/e3_horizon16.py            (prints a Markdown report to stdout; no files are written)"""
import os, sys, glob, json, itertools
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                        # .../isaac
R_CLUSTER, R_LOC = os.path.join(ROOT, "cluster_results", "results"), os.path.join(ROOT, "results")
PRIORITY = [os.path.join(R_CLUSTER, "gap_refine.jsonl"), os.path.join(R_CLUSTER, "scaffold.jsonl"), os.path.join(R_LOC, "horizon_gap1.jsonl"),
            os.path.join(R_CLUSTER, "horizon16.jsonl")]
DATASETS = ["tier4_gap6", "tier4_gap10", "tier4_gap20", "tier4_g1_10", "tier4_g1_20"]
LEARNERS = {("diacritic", 0): "-R", ("diacritic", 1): "-RF", ("scaffold", 1): "scaffold-RF"}
LORDER = ["-R", "-RF", "scaffold-RF"]
SEEDS = list(range(16)); Z = norm.ppf(0.975)

def file_list():
    fs = sorted(glob.glob(os.path.join(R_CLUSTER, "*.jsonl*")) + glob.glob(os.path.join(R_LOC, "**", "*.jsonl*"), recursive=True))
    return [f for f in PRIORITY if os.path.exists(f)] + [f for f in fs if f not in PRIORITY]

def matches(a):
    ds = os.path.basename(a.get("data", ""))
    if ds not in DATASETS or a.get("n_train") != 448 or a.get("beta") != 0.002 or a.get("steps") != 10000: return None
    if a.get("refine") not in ("", None) or a.get("init_from") or (a.get("aux_theta") or 0) or (a.get("aux_join") or 0) or (a.get("aux_mik") or 0): return None
    if "smoke" in a.get("out", "") or a.get("pixels"): return None
    key = (a.get("variant", "diacritic"), int(a.get("lam", 0) > 0))
    if key not in LEARNERS or a.get("lam", 0) not in (0, 0.0, 1, 1.0): return None
    return ds, LEARNERS[key], a.get("seed")

def load():
    runs, dups, src = {}, [], {}
    for f in file_list():
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            m = matches(r.get("args", {}))
            if m is None or not r.get("phases") or "gap2" not in r["phases"] or "summary" not in r: continue
            rel = os.path.relpath(f, ROOT)
            if m in runs:
                same = all(abs(runs[m]["summary"][k] - r["summary"][k]) < 1e-9 for k in ("S_Gam_gap2", "S_G_grasp", "S_G_place"))
                dups.append((m, rel, src[m], same, r.get("slurm_job"))); continue
            runs[m] = r; src[m] = rel
    return runs, src, dups

def agg(r):
    """recompute the paper's summary fields from per_t with train_diacritic.py's aggregation and assert equality"""
    ph, pt = r["phases"], r["per_t"]
    pm = lambda k, p: float(np.nanmean([pt[k][i] for i, q in enumerate(ph) if q == p]))
    out = dict(S_Gam_gap2=pm("S_Gam", "gap2"), S_G_grasp=pm("S_G", "grasp"), S_G_place=pm("S_G", "place"))
    for k, v in out.items(): assert abs(v - r["summary"][k]) < 1e-9, (k, v, r["summary"][k])
    return out

def build_rows(runs, thr):
    rows = []
    for i, (key, r) in enumerate(sorted(runs.items(), key=lambda kv: (DATASETS.index(kv[0][0]), LORDER.index(kv[0][1]), kv[0][2]))):
        ds, learner, seed = key; ph = r["phases"]; rev = max(j for j, p in enumerate(ph) if p == "scan"); d = agg(r)
        base = dict(ds=ds, learner=learner, seed=seed, run=i, RF=int(learner == "-RF"), SC=int(learner == "scaffold-RF"))
        rows.append(dict(base, dist=ph.index("grasp") - rev, cls=0, y=int(d["S_G_grasp"] > thr)))
        rows.append(dict(base, dist=ph.index("place") - rev, cls=1, y=int(d["S_Gam_gap2"] > thr and d["S_G_place"] > thr)))
    return rows

# ---------------------------------------------------------------- statistics
def wilson(k, n):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / den; h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h
def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
def design(rows, cols): return np.column_stack([np.ones(len(rows))] + [[float(r[c]) if not callable(c) else c(r) for r in rows] for c in cols])
def mle(X, y):
    nll = lambda w: -np.sum(y * (X @ w) - np.logaddexp(0, X @ w)) + 1e-3 * np.sum(w[1:] ** 2)
    res = minimize(nll, np.zeros(X.shape[1]), method="BFGS", options=dict(gtol=1e-8, maxiter=10000)); return res.x
def loglik(X, y, w): return float(np.sum(y * (X @ w) - np.logaddexp(0, X @ w)))
def sandwich(X, y, w, groups):
    p = 1 / (1 + np.exp(-(X @ w))); bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 2e-3 * np.eye(X.shape[1]))
    S = X * (y - p)[:, None]; ug = np.unique(groups); G = len(ug); meat = np.zeros((X.shape[1],) * 2)
    for g in ug: s = S[groups == g].sum(0); meat += np.outer(s, s)
    meat *= G / (G - 1) * (len(y) - 1) / (len(y) - X.shape[1])                                # CR1 small-sample correction
    return bread @ meat @ bread, G
def delta(fun, w, V, eps=1e-6):
    g = np.array([(fun(w + eps * e) - fun(w - eps * e)) / (2 * eps) for e in np.eye(len(w))]); v = fun(w)
    se = float(np.sqrt(g @ V @ g)); return v, se, v - Z * se, v + Z * se
def fit(rows, cols):
    X = design(rows, cols); y = np.array([r["y"] for r in rows], float); g = np.array([r["run"] for r in rows])
    w = mle(X, y); V, G = sandwich(X, y, w, g); return dict(X=X, y=y, w=w, V=V, G=G, se=np.sqrt(np.diag(V)), ll=loglik(X, y, w), n=len(y), cols=["const"] + [c if isinstance(c, str) else c.__name__ for c in cols])
def wald(w, V, idx):
    b = w[idx]; Vb = V[np.ix_(idx, idx)]; stat = float(b @ np.linalg.solve(Vb, b)); return stat, 1 - chi2.cdf(stat, len(idx))
def pstr(p): return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
def sig(v): return 1 / (1 + np.exp(-v))

# ---------------------------------------------------------------- report pieces
def counts_table(rows, thr, runs):
    out = [f"\n### Sufficient counts per (dataset, learner), threshold {thr}\n",
           "Cells: k/n for seeds 0-7, seeds 8-15 and pooled 0-15; Wilson 95 % interval on the pooled proportion.  d1 / d2 = reveal->use distance of beta_1 / beta_2.\n",
           "| dataset | d1 / d2 | learner | n seeds | beta_1 0-7 | beta_1 8-15 | beta_1 pooled | Wilson 95 % | beta_2 0-7 | beta_2 8-15 | beta_2 pooled | Wilson 95 % |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    short = []
    for ds in DATASETS:
        for L in LORDER:
            rs = [r for r in rows if r["ds"] == ds and r["learner"] == L]
            if not rs: short.append((ds, L, 0, [])); out.append(f"| {ds} | - | {L} | 0 | - | - | - | - | - | - | - | - |"); continue
            d1 = {r["dist"] for r in rs if r["cls"] == 0}.pop(); d2 = {r["dist"] for r in rs if r["cls"] == 1}.pop()
            seeds = sorted({r["seed"] for r in rs}); n = len(seeds)
            if n < 16: short.append((ds, L, n, seeds))
            cell = []
            for cls in (0, 1):
                for blk in ((0, 8), (8, 16), (0, 16)):
                    rr = [r for r in rs if r["cls"] == cls and blk[0] <= r["seed"] < blk[1]]; k = sum(r["y"] for r in rr)
                    cell.append(f"{k}/{len(rr)}" if rr else "-")
                    if blk == (0, 16): lo, hi = wilson(k, len(rr)); cell.append(f"{k/len(rr):.3f} [{lo:.3f}, {hi:.3f}]")
            out.append(f"| {ds} | {d1} / {d2} | {L} | {n} | " + " | ".join(cell) + " |")
    return "\n".join(out), short

def by_distance(rows):
    out = ["\n### Empirical P(sufficient) by distance and learner (threshold 0.9)\n", "| class | distance | datasets | " + " | ".join(f"{L}: k/n (p)" for L in LORDER) + " |", "|---|---|---|" + "---|" * len(LORDER)]
    for cls in (0, 1):
        for d in sorted({r["dist"] for r in rows if r["cls"] == cls}):
            dss = sorted({r["ds"] for r in rows if r["cls"] == cls and r["dist"] == d}, key=DATASETS.index); cell = []
            for L in LORDER:
                rr = [r for r in rows if r["cls"] == cls and r["dist"] == d and r["learner"] == L]; k = sum(r["y"] for r in rr)
                cell.append(f"{k}/{len(rr)} ({k/len(rr):.2f})" if rr else "-")
            out.append(f"| beta_{cls+1} | {d} | {', '.join(dss)} | " + " | ".join(cell) + " |")
    return "\n".join(out)

def per_learner(rows, thr):
    out = [f"\n### Per-learner logistic slopes of P(sufficient) on distance, threshold {thr}\n",
           "Model per learner: logit P(y=1) = b0 + b1 * dist + b2 * cls (both classes pooled, class indicator cls = 1 for beta_2).  SE: cluster-robust, clusters = run.  "
           "OR/step = exp(b1).  Marginal effect = change in P per +10 steps at the learner's pooled covariate mean: derivative form 10 * b1 * p(1-p) with p at the mean of dist and cls, and finite-difference form P(mean dist + 5) - P(mean dist - 5); delta-method CIs.\n",
           "| learner | rows / runs | b1 (dist) | SE | 95 % CI | z | p | OR / step [95 % CI] | b2 (cls) [SE] | mean dist | P at mean | dP per 10 steps, derivative [95 % CI] | P(mean+5) - P(mean-5) [95 % CI] | log-lik |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fits = {}
    for L in LORDER:
        rs = [r for r in rows if r["learner"] == L]
        if not rs: continue
        F = fit(rs, ["dist", "cls"]); fits[L] = F; w, se, V = F["w"], F["se"], F["V"]; z = w[1] / se[1]; p = 2 * (1 - norm.cdf(abs(z)))
        xbar = F["X"].mean(0); me = lambda ww: 10 * ww[1] * sig(xbar @ ww) * (1 - sig(xbar @ ww)); mv, mse, mlo, mhi = delta(me, w, V)
        fd = lambda ww: sig(xbar @ ww + 5 * ww[1]) - sig(xbar @ ww - 5 * ww[1]); fv, fse, flo, fhi = delta(fd, w, V)
        out.append(f"| {L} | {F['n']} / {F['G']} | {w[1]:+.4f} | {se[1]:.4f} | [{w[1]-Z*se[1]:+.4f}, {w[1]+Z*se[1]:+.4f}] | {z:+.2f} | {pstr(p)} | "
                   f"{np.exp(w[1]):.3f} [{np.exp(w[1]-Z*se[1]):.3f}, {np.exp(w[1]+Z*se[1]):.3f}] | {w[2]:+.3f} [{se[2]:.3f}] | {xbar[1]:.2f} | {sig(xbar @ w):.3f} | {mv:+.3f} [{mlo:+.3f}, {mhi:+.3f}] | {fv:+.3f} [{flo:+.3f}, {fhi:+.3f}] | {F['ll']:.2f} |")
    return "\n".join(out), fits

def pooled_model(rows):
    RFd = lambda r: float(r["RF"] * r["dist"]); SCd = lambda r: float(r["SC"] * r["dist"]); RFd.__name__, SCd.__name__ = "RF x dist", "SC x dist"
    full = fit(rows, ["dist", "cls", "RF", "SC", RFd, SCd]); red = fit(rows, ["dist", "cls", "RF", "SC"])
    w, se, V = full["w"], full["se"], full["V"]
    out = ["\n### Pooled model with learner dummies and learner x distance interactions (threshold 0.9)\n",
           "logit P = b0 + b_d * dist + b_c * cls + b_RF * RF + b_SC * SC + g_RF * (RF x dist) + g_SC * (SC x dist); reference learner -R.  SE cluster-robust by run.\n",
           f"n = {full['n']} rows, G = {full['G']} runs; log-lik full = {full['ll']:.3f}, reduced (common slope) = {red['ll']:.3f}.\n",
           "| term | coef | SE | 95 % CI | z | p |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(full["cols"]):
        z = w[i] / se[i]; out.append(f"| {c} | {w[i]:+.4f} | {se[i]:.4f} | [{w[i]-Z*se[i]:+.4f}, {w[i]+Z*se[i]:+.4f}] | {z:+.2f} | {pstr(2*(1-norm.cdf(abs(z))))} |")
    lr = 2 * (full["ll"] - red["ll"]); plr = 1 - chi2.cdf(lr, 2); wst, pw = wald(w, V, [5, 6])
    out += [f"\nLikelihood-ratio test, one common slope vs learner-specific slopes: LR = {lr:.3f} on 2 df, p = {pstr(plr)} (model-based, ignores clustering).",
            f"Cluster-robust Wald test of g_RF = g_SC = 0: chi2 = {wst:.3f} on 2 df, p = {pstr(pw)}.",
            "\nImplied slopes and pairwise slope contrasts (cluster-robust, delta method); Holm-adjusted p over the three contrasts:\n",
            "| quantity | estimate | SE | 95 % CI | z | p | Holm p |", "|---|---|---|---|---|---|---|"]
    slopes = {"-R": w[1], "-RF": w[1] + w[5], "scaffold-RF": w[1] + w[6]}
    cons = [("slope -RF - slope -R", np.array([0, 0, 0, 0, 0, 1, 0.])), ("slope scaffold-RF - slope -R", np.array([0, 0, 0, 0, 0, 0, 1.])), ("slope scaffold-RF - slope -RF", np.array([0, 0, 0, 0, 0, -1, 1.]))]
    res = []
    for name, c in cons:
        est = float(c @ w); s = float(np.sqrt(c @ V @ c)); z = est / s; res.append((name, est, s, z, 2 * (1 - norm.cdf(abs(z)))))
    order = np.argsort([r[4] for r in res]); holm = [0] * 3; running = 0
    for rank, j in enumerate(order): running = max(running, (3 - rank) * res[j][4]); holm[j] = min(1, running)
    for L in LORDER:
        c = np.array([0, 1, 0, 0, 0, int(L == "-RF"), int(L == "scaffold-RF")], float); est = float(c @ w); s = float(np.sqrt(c @ V @ c)); z = est / s
        out.append(f"| slope {L} | {est:+.4f} | {s:.4f} | [{est-Z*s:+.4f}, {est+Z*s:+.4f}] | {z:+.2f} | {pstr(2*(1-norm.cdf(abs(z))))} | - |")
    for (name, est, s, z, p), h in zip(res, holm): out.append(f"| {name} | {est:+.4f} | {s:.4f} | [{est-Z*s:+.4f}, {est+Z*s:+.4f}] | {z:+.2f} | {pstr(p)} | {pstr(h)} |")
    # supplementary: fully interacted model (learner x dist AND learner x cls), whose learner-specific slopes coincide with the per-learner fits
    RFc = lambda r: float(r["RF"] * r["cls"]); SCc = lambda r: float(r["SC"] * r["cls"]); RFc.__name__, SCc.__name__ = "RF x cls", "SC x cls"
    full2 = fit(rows, ["dist", "cls", "RF", "SC", RFd, SCd, RFc, SCc]); red2 = fit(rows, ["dist", "cls", "RF", "SC", RFc, SCc])
    lr2 = 2 * (full2["ll"] - red2["ll"]); w2, V2 = full2["w"], full2["V"]; ws2, pw2 = wald(w2, V2, [5, 6])
    out += [f"\nSupplementary, fully interacted model (adds RF x cls, SC x cls so that each learner has its own class offset; its learner-specific slopes equal the per-learner fits above): "
            f"LR common-slope vs learner-specific = {lr2:.3f} on 2 df, p = {pstr(1 - chi2.cdf(lr2, 2))}; cluster-robust Wald chi2 = {ws2:.3f}, p = {pstr(pw2)}; "
            f"slope contrasts -RF - -R = {w2[5]:+.4f} (SE {np.sqrt(V2[5,5]):.4f}), scaffold-RF - -R = {w2[6]:+.4f} (SE {np.sqrt(V2[6,6]):.4f}), scaffold-RF - -RF = {w2[6]-w2[5]:+.4f} (SE {np.sqrt(V2[5,5]+V2[6,6]-2*V2[5,6]):.4f})."]
    # pairwise LEVEL contrasts between learners at the beta_2 distances 19 and 23 (pooled model, delta method) and raw Newcombe differences
    out += ["\nPairwise learner contrasts of the beta_2 level at fixed distance (pooled model with learner x distance interactions, cls = 1; raw = pooled proportions, Newcombe 95 % CI):\n",
            "| distance (dataset) | contrast | model P_a | model P_b | model P_a - P_b [95 % CI] | p | raw k/n a | raw k/n b | raw RD [Newcombe 95 % CI] |", "|---|---|---|---|---|---|---|---|---|"]
    pairs = [("scaffold-RF", "-R"), ("scaffold-RF", "-RF"), ("-RF", "-R")]
    xrow = lambda L, d: np.array([1, d, 1, int(L == "-RF"), int(L == "scaffold-RF"), d * int(L == "-RF"), d * int(L == "scaffold-RF")], float)
    for d, ds in ((19, "tier4_gap6"), (23, "tier4_gap10")):
        for a, b in pairs:
            f = lambda ww: sig(xrow(a, d) @ ww) - sig(xrow(b, d) @ ww); est, s, lo, hi = delta(f, w, V); z = est / s
            ra = [r for r in rows if r["learner"] == a and r["cls"] == 1 and r["ds"] == ds]; rb = [r for r in rows if r["learner"] == b and r["cls"] == 1 and r["ds"] == ds]
            ka, kb = sum(r["y"] for r in ra), sum(r["y"] for r in rb); rd, rlo, rhi = newcombe(ka, len(ra), kb, len(rb))
            out.append(f"| {d} ({ds}) | {a} vs {b} | {sig(xrow(a, d) @ w):.3f} | {sig(xrow(b, d) @ w):.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pstr(2*(1-norm.cdf(abs(z))))} | {ka}/{len(ra)} | {kb}/{len(rb)} | {rd:+.3f} [{rlo:+.3f}, {rhi:+.3f}] |")
    return "\n".join(out), full, red

def bimodality(runs):
    out = ["\n### Distribution of the three sufficiency statistics over the 192 runs (why the threshold hardly matters)\n",
           "| statistic | <= 0.8 | (0.8, 0.9] | (0.9, 0.95] | > 0.95 | exactly 1.0 | largest value below 1.0 |", "|---|---|---|---|---|---|---|"]
    for k in ("S_G_grasp", "S_Gam_gap2", "S_G_place"):
        v = np.array([agg(r)[k] for r in runs.values()])
        out.append(f"| {k} | {(v <= 0.8).sum()} | {((v > 0.8) & (v <= 0.9)).sum()} | {((v > 0.9) & (v <= 0.95)).sum()} | {(v > 0.95).sum()} | {(v >= 1 - 1e-9).sum()} | {v[v < 1 - 1e-9].max():.4f} |")
    return "\n".join(out)

def effect_sizes(rows, fits):
    out = ["\n### Effect sizes: beta_2 sufficiency at distance 19 (gap6) vs 33 (gap20), threshold 0.9\n",
           "Model: per-learner fit (cls = 1); P CIs are delta-method intervals on the logit scale mapped back (cluster-robust covariance), the difference CI is delta-method on the probability scale.  Raw: pooled 16-seed proportions with Wilson CIs; risk difference with Newcombe (method 10) CI.\n",
           "| learner | model P(d=19) [95 % CI] | model P(d=33) [95 % CI] | model P19 - P33 [95 % CI] | raw gap6 k/n (p) [Wilson] | raw gap20 k/n (p) [Wilson] | raw RD [Newcombe 95 % CI] |", "|---|---|---|---|---|---|---|"]
    for L in LORDER:
        F = fits[L]; w, V = F["w"], F["V"]
        P = lambda d: (lambda ww: sig(ww[0] + ww[1] * d + ww[2])); dif = delta(lambda ww: P(19)(ww) - P(33)(ww), w, V)
        eta = lambda d: delta(lambda ww: ww[0] + ww[1] * d + ww[2], w, V); e19, e33 = eta(19), eta(33)            # CI on the logit scale, mapped back
        p19 = (sig(e19[0]), None, sig(e19[2]), sig(e19[3])); p33 = (sig(e33[0]), None, sig(e33[2]), sig(e33[3]))
        r6 = [r for r in rows if r["learner"] == L and r["cls"] == 1 and r["ds"] == "tier4_gap6"]; r20 = [r for r in rows if r["learner"] == L and r["cls"] == 1 and r["ds"] == "tier4_gap20"]
        k6, n6, k20, n20 = sum(r["y"] for r in r6), len(r6), sum(r["y"] for r in r20), len(r20); w6, w20 = wilson(k6, n6), wilson(k20, n20); rd, lo, hi = newcombe(k6, n6, k20, n20)
        out.append(f"| {L} | {p19[0]:.3f} [{p19[2]:.3f}, {p19[3]:.3f}] | {p33[0]:.3f} [{p33[2]:.3f}, {p33[3]:.3f}] | {dif[0]:+.3f} [{dif[2]:+.3f}, {dif[3]:+.3f}] | "
                   f"{k6}/{n6} ({k6/n6:.3f}) [{w6[0]:.3f}, {w6[1]:.3f}] | {k20}/{n20} ({k20/n20:.3f}) [{w20[0]:.3f}, {w20[1]:.3f}] | {rd:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    return "\n".join(out)

def main():
    runs, src, dups = load(); rows = build_rows(runs, 0.9)
    print("## E3 horizon sweep, 16 seeds: within-learner distance effect\n")
    print(f"Runs after filtering and de-duplication: {len(runs)}; (run, class) rows: {len(rows)}.  Files scanned: {len(file_list())} under isaac/cluster_results/results and isaac/results.\n")
    print("### Provenance: which file supplied which cell (first occurrence wins)\n")
    print("| dataset | learner | seeds 0-7 source (seeds) | seeds 8-15 source (seeds) |", "\n|---|---|---|---|")
    for ds in DATASETS:
        for L in LORDER:
            c = []
            for blk in ((0, 8), (8, 16)):
                byf = {}
                for (d, l, s), f in src.items():
                    if d == ds and l == L and blk[0] <= s < blk[1]: byf.setdefault(f, []).append(s)
                c.append("; ".join(f"{f} ({','.join(map(str, sorted(v)))})" for f, v in sorted(byf.items())) or "**none**")
            print(f"| {ds} | {L} | {c[0]} | {c[1]} |")
    print("\nDiscarded duplicates (same dataset, learner, seed found again later in the scan order):\n")
    print("| dataset | learner | seed | discarded file (slurm job) | kept file | identical S_Gam_gap2 / S_G_grasp / S_G_place? |", "\n|---|---|---|---|---|---|")
    for (ds, L, s), f, kept, same, job in sorted(dups, key=lambda t: (DATASETS.index(t[0][0]), LORDER.index(t[0][1]), t[0][2], t[1])):
        print(f"| {ds} | {L} | {s} | {f} ({job}) | {kept} | {'yes' if same else '**no**'} |")
    nsame = sum(1 for d in dups if d[3]); print(f"\n{len(dups)} duplicates discarded, {nsame} numerically identical, {len(dups)-nsame} differing (same seed, different SLURM job: GPU non-determinism).")
    tab, short = counts_table(rows, 0.9, runs); print(tab)
    print("\nCells with fewer than 16 seeds:\n"); print("| dataset | learner | n | seeds present |", "\n|---|---|---|---|")
    for ds, L, n, seeds in short: print(f"| {ds} | {L} | {n} | {','.join(map(str, seeds)) or '-'} |")
    print(by_distance(rows))
    pl, fits = per_learner(rows, 0.9); print(pl)
    pm, full, red = pooled_model(rows); print(pm)
    print(effect_sizes(rows, fits))
    print("\n## Sensitivity: thresholds 0.8 and 0.95 (same aggregation, recomputed from per_t)\n")
    print(bimodality(runs))
    for thr in (0.8, 0.95):
        rr = build_rows(runs, thr); tab, _ = counts_table(rr, thr, runs); print(tab); pl, _ = per_learner(rr, thr); print(pl)
    print("\n### Sensitivity summary: slope b1 per learner (cluster-robust 95 % CI) by threshold\n")
    print("| learner | thr 0.8 | thr 0.9 | thr 0.95 |", "\n|---|---|---|---|")
    cell = {L: [] for L in LORDER}
    for thr in (0.8, 0.9, 0.95):
        rr = build_rows(runs, thr)
        for L in LORDER:
            F = fit([r for r in rr if r["learner"] == L], ["dist", "cls"]); w, se = F["w"], F["se"]; z = w[1] / se[1]
            cell[L].append(f"{w[1]:+.4f} [{w[1]-Z*se[1]:+.4f}, {w[1]+Z*se[1]:+.4f}] p={pstr(2*(1-norm.cdf(abs(z))))}")
    for L in LORDER: print(f"| {L} | " + " | ".join(cell[L]) + " |")

if __name__ == "__main__": main()
