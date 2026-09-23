"""N1: resolution sweep of the side-information coarsening for the reported rate H(C_t | Obar_t).

Concern addressed: Obar_t = f_t(O_t) is a hand-built symbolic coarsening of the raw observation.  Does the learned code
carry information that a *finer* coarsening of the raw observation would explain away (i.e. is part of the "memory"
actually leaked from O_t)?

For the 16 fig3 runs (tier4_gap6_2k, variant diacritic, lam=0, beta in {0, 0.001}, 8 seeds each) we compute per step t
H(C_t | f_r(O_t)) for a nested sequence of coarsenings
    r0 = Obar_t                                                    (the paper's symbolic observation, O_lab[t])
    r1..r5 = (Obar_t, grid cell of tcp_pos at 10 / 5 / 2 / 1 / 0.5 cm)
    r6 = (Obar_t, 1 cm cell of tcp_pos, 1 cm cell of obj_pos_rel_tcp, gripper opening binned at 5 mm)
with (i) the plug-in estimator, (ii) Miller-Madow applied consistently to the joint and the marginal,
H_MM(X) = H_plugin(X) + (m-1)/(2 n ln 2), H(C|F) = H_MM(C,F) - H_MM(F), (iii) occupancy statistics of f_r (occupied
cells, mean / min episodes per cell, fraction of episodes in cells with <= 2 episodes), (iv) an episode bootstrap 95 %
interval (200 resamples) of the plug-in value, and (v) a permutation null: the codes are shuffled among the episodes
that share the same Obar_t (this preserves the joint law of (C_t, Obar_t) exactly, hence H(C|Obar) is unchanged), and
H_null(C | f_r) is recomputed.  The drop H(C|Obar) - H_null(C|f_r) is what a *non-leaking* code loses purely from bin
fragmentation; the excess  H_null(C|f_r) - H(C|f_r)  is the part of the drop not explained by fragmentation.

Fragmentation-free alternative: a 2-fold cross-validated conditional-entropy upper bound.  Per step t a multinomial
logistic regression and a small MLP predict C_t from the standardised raw 27-dim O_t plus the one-hot of Obar_t on half
of the episodes; the held-out cross-entropy (bits, predictive law mixed with uniform at 1e-3 so it is proper over the
K=16 code values) upper-bounds H(C_t | O_t) without any binning.

Runs are matched to code files by basename(args.save_codes) (every fig3 row stores it; the local file names are
tier4_gap6_2k_b{beta}_lam{lam}_{variant}_s{seed}.npz, without the aux/ref infix used by later tags).  Plug-in values at
r0 are asserted equal to the stored per_t['H(C|O)'] (this also checks the episode order of the code files).

Usage: python isaac/analysis/n1_resolution.py [--boot 200] [--perm 20] [--no-cv]
       [--json OUT.json]      (prints markdown; ~35 s, ~20 s with --no-cv)
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "toy"))
from aprime_data import prepare, solver_labels   # noqa: E402

LN2 = float(np.log(2.0))
DATA = os.path.join(HERE, "..", "data", "tier4_gap6_2k")
RES = os.path.join(HERE, "..", "cluster_results", "results", "fig3.jsonl")
CODES = os.path.join(HERE, "..", "cluster_results", "codes", "fig3")
THR = 0.9
RES_NAMES = ["r0:Obar", "r1:+tcp10cm", "r2:+tcp5cm", "r3:+tcp2cm", "r4:+tcp1cm", "r5:+tcp0.5cm", "r6:+tcp1cm+obj1cm+grip5mm"]


# ----------------------------------------------------------------------------- entropy helpers (dense int keys)
def dense(labels):
    """Hashable labels (list / 1-d array) -> dense int codes 0..m-1."""
    d = {}
    return np.array([d.setdefault(x, len(d)) for x in labels], dtype=np.int64)


def joint(*cols):
    """Dense joint key of several dense int columns."""
    out = np.zeros_like(cols[0])
    for c in cols:
        out = out * (int(c.max()) + 1) + c
    return np.unique(out, return_inverse=True)[1].astype(np.int64)


def H_from_counts(cnt, mm=False):
    n = cnt.sum(); p = cnt[cnt > 0] / n
    h = float(-(p * np.log2(p)).sum())
    if mm:
        h += (len(p) - 1) / (2.0 * n * LN2)
    return h


def H(x, mm=False, idx=None):
    if idx is not None: x = x[idx]
    return H_from_counts(np.bincount(x), mm)


def condH(x, y, mm=False, idx=None):
    """H(X|Y) = H(X,Y) - H(Y), plug-in or Miller-Madow on both terms."""
    return H(joint(x, y), mm, idx) - H(y, mm, idx)


def occupancy(y):
    cnt = np.bincount(y); cnt = cnt[cnt > 0]
    return dict(cells=int(len(cnt)), mean_per_cell=float(cnt.mean()), min_per_cell=int(cnt.min()),
                frac_eps_in_small_cells=float(cnt[cnt <= 2].sum() / cnt.sum()))


def grid(x, size):
    """(E,k) float -> dense int cell index at the given cell size (floor)."""
    cells = np.floor(x / size).astype(np.int64)
    cols = [np.unique(cells[:, j], return_inverse=True)[1] for j in range(cells.shape[1])]
    return joint(*cols)


# ----------------------------------------------------------------------------- CV upper bound
def cv_bound(X, y, K, seed=0, eps=1e-3):
    """2-fold CV held-out cross-entropy (bits) of a multinomial LR and a small MLP predicting y (0..K-1) from X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    E = len(y); perm = np.random.RandomState(seed).permutation(E); folds = [perm[: E // 2], perm[E // 2:]]
    out = {}
    for name, mk in (("LR", lambda: LogisticRegression(C=1.0, max_iter=3000)),
                     ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=seed, alpha=1.0, early_stopping=True))):
        ce = []
        for a, b in ((0, 1), (1, 0)):
            tr, te = folds[a], folds[b]
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if len(np.unique(y[tr])) == 1:                       # constant code on this fold: predict it with prob 1-eps
                P = np.full((len(te), K), eps / K); P[:, int(y[tr][0])] += 1 - eps
            else:
                m = mk().fit(Xtr, y[tr])
                P = np.full((len(te), K), eps / K)
                P[:, m.classes_] += (1 - eps) * m.predict_proba(Xte)
            ce.append(float(-np.log2(P[np.arange(len(te)), y[te]]).mean()))
        out[name] = float(np.mean(ce))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--no-cv", action="store_true"); ap.add_argument("--json", default="")
    args = ap.parse_args()
    t0 = time.time()

    eps, meta, vis, env, solver, key = prepare(DATA, augment=True)
    G_lab, Gam_lab, GamS_lab, O_lab = solver_labels(solver, eps, key)
    E, T = len(eps), len(eps[0]["phase"]); phases = [str(p) for p in eps[0]["phase"]]
    obs = np.stack([e["obs"] for e in eps])                                   # (E,T,27)
    gap1 = [t for t, p in enumerate(phases) if p == "gap1"]; gap2 = [t for t, p in enumerate(phases) if p == "gap2"]
    steps = gap1 + gap2
    print(f"dataset {os.path.basename(DATA)}: E={E}, T={T}, gap1 steps (0-based) {gap1}, gap2 steps {gap2}; solver {time.time()-t0:.1f}s\n")

    # coarsenings per step: list over r of dense keys (E,)
    F = {}
    for t in steps:
        ob = dense(O_lab[t]); tcp = obs[:, t, :3]; objrel = obs[:, t, 8:11]; grip = obs[:, t, 7:8]
        fs = [ob]
        for s in (0.10, 0.05, 0.02, 0.01, 0.005):
            fs.append(joint(ob, grid(tcp, s)))
        fs.append(joint(ob, grid(tcp, 0.01), grid(objrel, 0.01), grid(grip, 0.005)))
        F[t] = fs
    R = len(RES_NAMES)

    # occupancy table (run independent)
    print("### Occupancy of the coarsenings f_r (run independent; n = %d episodes)\n" % E)
    print("| r | phase | occupied cells (mean over steps) | mean eps/cell | min eps/cell (min over steps) | frac. episodes in cells with <=2 eps |")
    print("|---|---|---|---|---|---|")
    occ = {}
    for r in range(R):
        for ph, st in (("gap1", gap1), ("gap2", gap2)):
            o = [occupancy(F[t][r]) for t in st]
            occ[(r, ph)] = dict(cells=float(np.mean([x["cells"] for x in o])), mean_per_cell=float(np.mean([x["mean_per_cell"] for x in o])),
                                min_per_cell=int(min(x["min_per_cell"] for x in o)), frac_small=float(np.mean([x["frac_eps_in_small_cells"] for x in o])))
            d = occ[(r, ph)]
            print(f"| {RES_NAMES[r]} | {ph} | {d['cells']:.1f} | {d['mean_per_cell']:.2f} | {d['min_per_cell']} | {d['frac_small']:.3f} |")
    print()

    # ground-truth labels: is the memory content itself (Gamma_t) or the full latent (theta) decodable from finer coarsenings?
    perm_rng0 = np.random.RandomState(2)
    theta = dense([e["theta"] for e in eps])
    print("### Ground-truth labels (run independent): H(label | f_r), permutation null within Obar cells, and the CV upper bound on H(label | O_t)\n")
    print("| label | phase | r | H_plugin(label|f_r) | H_null(label|f_r) | excess = H_null - H_plugin | CV bound LR | CV bound MLP |")
    print("|---|---|---|---|---|---|---|---|")
    gt = {}
    for lname, lab in (("Gamma_t", [None] * 0), ("theta", None)):
        for ph, st in (("gap1", gap1), ("gap2", gap2)):
            vals = {r: [] for r in range(R)}; nulls = {r: [] for r in range(R)}; cv = []
            for t in st:
                y = dense(Gam_lab[t]) if lname == "Gamma_t" else theta
                ob = F[t][0]
                for r in range(R):
                    vals[r].append(condH(y, F[t][r]))
                    nl = []
                    for _ in range(args.perm):
                        yp = y.copy()
                        for cell in np.unique(ob):
                            m = np.nonzero(ob == cell)[0]; yp[m] = y[perm_rng0.permutation(m)]
                        nl.append(condH(yp, F[t][r]))
                    nulls[r].append(float(np.mean(nl)))
                if not args.no_cv:
                    X = np.concatenate([obs[:, t, :], np.eye(int(ob.max()) + 1)[ob]], 1)
                    cv.append(cv_bound(X, y, K=int(y.max()) + 1))
            for r in range(R):
                p, nl = float(np.mean(vals[r])), float(np.mean(nulls[r]))
                gt[(lname, ph, r)] = dict(plug=p, null=nl, excess=nl - p)
                cvs = f"{np.mean([c['LR'] for c in cv]):.3f} | {np.mean([c['MLP'] for c in cv]):.3f}" if cv else "-- | --"
                print(f"| {lname} | {ph} | {RES_NAMES[r]} | {p:.3f} | {nl:.3f} | {nl-p:+.3f} | {cvs if r == 0 else '(same) | (same)'} |")
            if cv: gt[(lname, ph, "cv")] = dict(LR=float(np.mean([c["LR"] for c in cv])), MLP=float(np.mean([c["MLP"] for c in cv])))
    print()

    # runs
    rows = [json.loads(l) for l in open(RES) if l.strip()]
    sel = [r for r in rows if os.path.basename(r["args"]["data"]) == "tier4_gap6_2k" and r["args"]["variant"] == "diacritic"
           and float(r["args"]["lam"]) == 0.0 and float(r["args"]["beta"]) in (0.0, 0.001)]
    sel.sort(key=lambda r: (float(r["args"]["beta"]), int(r["args"]["seed"])))
    assert len(sel) == 16, len(sel)
    rng = np.random.RandomState(0)
    boot_idx = [rng.randint(0, E, E) for _ in range(args.boot)]
    perm_rng = np.random.RandomState(1)

    per_run = []
    print("### Per-run values (mean over the steps of the phase; plug-in / Miller-Madow / permutation-null plug-in / bootstrap 95% of plug-in)\n")
    print("| beta | seed | sufficient | phase | " + " | ".join(RES_NAMES) + " | CV bound LR | CV bound MLP |")
    print("|---|---|---|---|" + "---|" * R + "---|---|")
    for r_ in sel:
        a, s = r_["args"], r_["summary"]
        fn = os.path.join(CODES, os.path.basename(a["save_codes"]))
        assert os.path.exists(fn), fn
        C = np.load(fn)["codes"]; assert C.shape == (E, T), C.shape
        suf = s["S_Gam_gap2"] > THR and s["S_G_place"] > THR
        rec = dict(beta=float(a["beta"]), seed=int(a["seed"]), sufficient=bool(suf), file=os.path.basename(fn), per_t={})
        for t in steps:
            c = dense(C[:, t]); vals = {}
            # alignment check against the stored per-step H(C|O)
            h0 = condH(c, F[t][0])
            assert abs(h0 - r_["per_t"]["H(C|O)"][t]) < 1e-6, (fn, t, h0, r_["per_t"]["H(C|O)"][t])
            ob = F[t][0]
            perms = []
            for _ in range(args.perm):                       # shuffle codes within each Obar cell
                cp = c.copy()
                for cell in np.unique(ob):
                    m = np.nonzero(ob == cell)[0]; cp[m] = c[perm_rng.permutation(m)]
                perms.append(cp)
            for r in range(R):
                f = F[t][r]; xy = joint(c, f); K1 = int(xy.max()) + 1; K2 = int(f.max()) + 1
                plug = condH(c, f); mm = condH(c, f, mm=True)
                bs = np.array([H_from_counts(np.bincount(xy[i], minlength=K1)) - H_from_counts(np.bincount(f[i], minlength=K2)) for i in boot_idx])
                null = float(np.mean([condH(cp, f) for cp in perms]))
                vals[r] = dict(plug=plug, mm=mm, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), null=null,
                               joint_cells=int(K1))
            if not args.no_cv:
                X = np.concatenate([obs[:, t, :], np.eye(int(ob.max()) + 1)[ob]], 1)
                vals["cv"] = cv_bound(X, c, K=16)
            rec["per_t"][t] = vals
        for ph, st in (("gap1", gap1), ("gap2", gap2)):
            cells = []
            for r in range(R):
                p = np.mean([rec["per_t"][t][r]["plug"] for t in st]); m = np.mean([rec["per_t"][t][r]["mm"] for t in st])
                nl = np.mean([rec["per_t"][t][r]["null"] for t in st])
                lo = np.mean([rec["per_t"][t][r]["lo"] for t in st]); hi = np.mean([rec["per_t"][t][r]["hi"] for t in st])
                cells.append(f"{p:.3f} / {m:.3f} / {nl:.3f} / [{lo:.3f},{hi:.3f}]")
            if not args.no_cv:
                cv = [f"{np.mean([rec['per_t'][t]['cv'][k] for t in st]):.3f}" for k in ("LR", "MLP")]
            else:
                cv = ["--", "--"]
            print(f"| {rec['beta']} | {rec['seed']} | {'yes' if suf else 'no'} | {ph} | " + " | ".join(cells) + f" | {cv[0]} | {cv[1]} |")
        per_run.append(rec)
    print()

    # group aggregates
    def agg(group, ph, r, k):
        st = gap1 if ph == "gap1" else gap2
        return float(np.mean([np.mean([rec["per_t"][t][r][k] for t in st]) for rec in group]))

    summary = {}
    for gname, group in (("sufficient", [x for x in per_run if x["sufficient"]]), ("insufficient", [x for x in per_run if not x["sufficient"]])):
        print(f"### Group means over {gname} seeds (n = {len(group)}: " + ", ".join(f"b{x['beta']}/s{x['seed']}" for x in group) + ")\n")
        print("| phase | r | H_plugin(C|f_r) | H_MM(C|f_r) | boot 95% (plug-in, mean of per-step bounds) | H_null(C|f_r) (perm. within Obar) | drop vs r0 (plug-in) | drop of the null (fragmentation only) | excess drop = H_null - H_plugin | occupied cells of f_r | joint (C,f_r) cells | mean eps/cell | min eps/cell | frac eps in cells <=2 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ph in ("gap1", "gap2"):
            base = agg(group, ph, 0, "plug") if group else float("nan")
            for r in range(R):
                if not group: continue
                p, m, nl = agg(group, ph, r, "plug"), agg(group, ph, r, "mm"), agg(group, ph, r, "null")
                lo, hi = agg(group, ph, r, "lo"), agg(group, ph, r, "hi"); jc = agg(group, ph, r, "joint_cells")
                d = occ[(r, ph)]
                summary[(gname, ph, r)] = dict(plug=p, mm=m, lo=lo, hi=hi, null=nl, drop=base - p, null_drop=base - nl, excess=nl - p)
                print(f"| {ph} | {RES_NAMES[r]} | {p:.3f} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {nl:.3f} | {base-p:.3f} | {base-nl:.3f} | {nl-p:+.3f} | {d['cells']:.1f} | {jc:.1f} | {d['mean_per_cell']:.2f} | {d['min_per_cell']} | {d['frac_small']:.3f} |")
        if not args.no_cv and group:
            for ph in ("gap1", "gap2"):
                st = gap1 if ph == "gap1" else gap2
                lr = float(np.mean([np.mean([rec["per_t"][t]["cv"]["LR"] for t in st]) for rec in group]))
                ml = float(np.mean([np.mean([rec["per_t"][t]["cv"]["MLP"] for t in st]) for rec in group]))
                summary[(gname, ph, "cv")] = dict(LR=lr, MLP=ml)
                print(f"\n{gname}, {ph}: 2-fold CV held-out cross-entropy upper bound on H(C_t|O_t) from the raw 27-dim O_t + one-hot(Obar_t): LR = {lr:.3f} bits, MLP = {ml:.3f} bits  (plug-in H(C|Obar) = {agg(group, ph, 0, 'plug'):.3f}).")
        print()

    # per-step profile for the sufficient group (gap1 and gap2 steps)
    suf_group = [x for x in per_run if x["sufficient"]]
    if suf_group:
        print("### Per-step profile, sufficient seeds (mean over seeds): plug-in H(C|f_r)\n")
        print("| t (1-based) | phase | " + " | ".join(RES_NAMES) + (" | CV LR | CV MLP |" if not args.no_cv else ""))
        print("|---|---|" + "---|" * R + ("---|---|" if not args.no_cv else ""))
        for t in steps:
            cells = [f"{np.mean([rec['per_t'][t][r]['plug'] for rec in suf_group]):.3f}" for r in range(R)]
            extra = ""
            if not args.no_cv:
                extra = f" {np.mean([rec['per_t'][t]['cv']['LR'] for rec in suf_group]):.3f} | {np.mean([rec['per_t'][t]['cv']['MLP'] for rec in suf_group]):.3f} |"
            print(f"| {t+1} | {phases[t]} | " + " | ".join(cells) + " |" + extra)
        print()

    if args.json:
        def conv(o):
            if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [conv(v) for v in o]
            if isinstance(o, (np.floating, np.integer)): return o.item()
            return o
        json.dump(dict(runs=conv(per_run), summary=conv({"|".join(map(str, k)): v for k, v in summary.items()}),
                       occupancy=conv({"|".join(map(str, k)): v for k, v in occ.items()})), open(args.json, "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
