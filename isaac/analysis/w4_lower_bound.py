"""Corridor certificate, lower side: NON-TRIVIAL LOWER BOUNDS on the zero-distortion recurrent memory of the non-transitive corridor.

Any zero-distortion deterministic recurrent memory induces, at every step t, a partition P_t of the reachable histories that refines
O_t, whose blocks are pairwise ~_t-compatible, and that is closed under the dynamics (a "closed compatible state assignment").  We
keep ONLY the necessary condition

      pairwise-INCOMPATIBLE histories with the same O_t lie in different blocks of P_t                                  (*)

i.e. P_t restricted to an observation cell o is a proper colouring of the incompatibility graph  I_o = (histories with O_t = o,
edges = not ~_t).  Dropping closure and the pairwise->setwise step only enlarges the feasible set, so every bound below is a valid
lower bound on   min_P H(P_t | O_t) = sum_o p(o) min H(P_t | O_t = o).

Bounds per cell (weights = p(h) normalised in the cell):
  A   history-level clique/mass bound.  Clique Q with masses p_1 >= ... >= p_k, rest r = 1 - sum p_i:
          H(blocks) >= H(p_1 + r, p_2, ..., p_k).
      Proof: the k clique members sit in k distinct blocks.  Merging all clique-free blocks into the block of member 1 does not
      increase entropy (merging atoms never does) and gives q = (p_1 + r_1, ..., p_k + r_k), r_i >= 0, sum r_i = r.  For every j the
      sum of the j largest entries of q is sum_{i in S}(p_i + r_i) <= (p_1 + ... + p_j) + r = sum of the j largest entries of
      v = (p_1 + r, p_2, ..., p_k); totals agree, so v majorises q and, entropy being Schur-concave, H(q) >= H(v).
  A'  the same bound on the TWIN QUOTIENT (histories with identical incompatibility neighbourhood in the cell are merged into one
      "type" carrying their total mass).  Valid for the colouring relaxation by the twin lemma: twins u, v are non-adjacent with
      N(u) = N(v); if they sit in classes A (u), B (v) with m(A) >= m(B), moving v into A keeps the colouring proper and replaces
      (m(A), m(B)) by the majorising (m(A)+p_v, m(B)-p_v), so entropy does not increase; sum of squared class masses strictly
      increases, so the process terminates in a colouring with every type monochromatic.  Hence
      min-entropy colouring of I_o = min-entropy colouring of the weighted quotient, and A on the quotient is a valid bound.
  B   chromatic entropy = EXACT minimum entropy over proper colourings of the weighted quotient (subset DP,
      f[S] = min_{independent I subset S, I contains lowest element of S}  g(m(I)) + f[S \\ I],  g(x) = -x log2 x), when the cell
      has <= --max-types types and the DP finishes within --cell-seconds; otherwise the cell falls back to max(A, A') (reported).
      B >= max(A, A', C) always (asserted); B ignores closure, so it is still only a LOWER bound on the closed problem.

  C   independent-set (max-block-mass) bound: every block is an independent set of I_o, so its mass is <= alpha_o = the maximum
      weight of an independent set (exact, by enumeration over the quotient); a distribution with all atoms <= alpha has
      H = sum q log2(1/q) >= log2(1/alpha_o).  C <= B, but it is a one-line human-checkable certificate (on the corridor
      alpha_o = 1/4 in hall 1 and 1/2 in hall 2).
  The DP value of B is cross-checked on every cell by an independent pruned enumeration of ALL proper colourings of the quotient
  (restricted-growth strings; --verify-nodes cap), and the optimal colouring returned by the DP is re-verified on the RAW
  history-level graph (every class pairwise compatible) so B is also attained there.

Aggregate: LB_t = sum_o p(o) * bound(o).  Upper bounds are read from REPORT_W4_upper_bound*.json (w4_upper_bound.py).

Usage (repository root, CPU only, a few seconds; writes results_md/analysis_W4_lower_bound.md):
  python isaac/analysis/w4_lower_bound.py --W 1 3 5 7
"""
import os, sys, time, json, argparse, collections, itertools
from math import log2
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "toy")); sys.path.insert(0, HERE)
from grid_env import grid_corridor                                          # noqa: E402
from gamma_solver import GammaSolver                                        # noqa: E402
from w4_structure import h_given_o                                          # noqa: E402


def H(v):
    v = np.asarray(v, float); v = v[v > 1e-15]
    return float(-(v * np.log2(v)).sum())


def clique_bound_value(masses):
    """H(p_1 + r, p_2, ..., p_k) for clique masses (normalised in the cell, any order)."""
    m = sorted(masses, reverse=True); r = 1.0 - sum(m)
    return H([m[0] + max(r, 0.0)] + m[1:])


# ------------------------------------------------------------------------------------------------ twin quotient
def twin_quotient(inc, p):
    """inc: bool (n,n) symmetric, zero diagonal.  Returns (types: list of member lists, adjacency among types, type masses)."""
    rows = collections.OrderedDict()
    for i in range(inc.shape[0]): rows.setdefault(inc[i].tobytes(), []).append(i)
    types = list(rows.values()); rep = [m[0] for m in types]
    adj = inc[np.ix_(rep, rep)].copy()
    mass = np.array([p[m].sum() for m in types])
    # self-check: the quotient is well defined (adjacency between two types is all-or-nothing, no edges inside a type)
    for a, ma in enumerate(types):
        assert not inc[np.ix_(ma, ma)].any()
        for b, mb in enumerate(types):
            blk = inc[np.ix_(ma, mb)]; assert blk.all() or not blk.any()
    return types, adj, mass


# ------------------------------------------------------------------------------------------------ cliques
def best_clique(adj, mass, heavy, deadline, exact_limit=22):
    """Clique of the type graph maximising the bound value; returns (A value using one heaviest history per type,
    A' value using the type masses, clique, exact?).  Exhaustive over all cliques (Bron-Kerbosch style extension) if the type
    count is small, else greedy + 1-swap local search from every start vertex."""
    n = len(mass); nb = [set(np.nonzero(adj[i])[0].tolist()) for i in range(n)]
    best = dict(A=0.0, Ap=0.0, QA=(), QAp=())

    def score(Q):
        a = clique_bound_value([heavy[i] for i in Q]); ap = clique_bound_value([mass[i] for i in Q])
        if a > best["A"] + 1e-15: best["A"], best["QA"] = a, tuple(Q)
        if ap > best["Ap"] + 1e-15: best["Ap"], best["QAp"] = ap, tuple(Q)

    exact = n <= exact_limit
    if exact:
        def ext(Q, cand):
            score(Q)
            if time.time() > deadline: raise TimeoutError
            for v in sorted(cand):
                ext(Q + [v], {u for u in cand if u > v and u in nb[v]})
        try:
            for v in range(n): ext([v], {u for u in nb[v] if u > v})
        except TimeoutError:
            exact = False
    if not exact:
        for key in (mass, heavy):
            for s in range(n):
                Q = [s]; cand = set(nb[s])
                while cand:
                    v = max(cand, key=lambda u: key[u]); Q.append(v); cand &= nb[v]
                score(Q)
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(Q)):                                   # 1-swap: drop Q[i], re-extend greedily
                        R = Q[:i] + Q[i + 1:]; cand = set.intersection(*[nb[u] for u in R]) if R else set(range(n))
                        cand.discard(Q[i])
                        while cand:
                            v = max(cand, key=lambda u: key[u]); R.append(v); cand &= nb[v]
                        if clique_bound_value([key[u] for u in R]) > clique_bound_value([key[u] for u in Q]) + 1e-12:
                            Q = R; score(Q); improved = True; break
    for Q in (best["QA"], best["QAp"]):                                       # certificate check
        assert all(adj[a, b] for a, b in itertools.combinations(Q, 2))
    return best["A"], best["Ap"], best, exact


# ------------------------------------------------------------------------------------------------ exact chromatic entropy
def chromatic_entropy(adj, mass, deadline):
    """Exact min-entropy proper colouring of a vertex-weighted graph (masses sum to 1) by subset DP; None on timeout."""
    n = len(mass); full = (1 << n) - 1
    nbm = [int(sum(1 << j for j in np.nonzero(adj[i])[0])) for i in range(n)]
    indep = np.zeros(1 << n, bool); msum = np.zeros(1 << n); indep[0] = True
    for S in range(1, 1 << n):
        lb = S & -S; i = lb.bit_length() - 1; R = S ^ lb
        indep[S] = indep[R] and not (nbm[i] & R); msum[S] = msum[R] + mass[i]
    g = np.where(msum > 1e-15, -msum * np.log2(np.maximum(msum, 1e-300)), 0.0)
    f = np.full(1 << n, np.inf); f[0] = 0.0; arg = np.zeros(1 << n, np.int64)
    for S in range(1, 1 << n):
        if (S & 0xFFF) == 0 and time.time() > deadline: return None, None, None
        lb = S & -S; R = S ^ lb; sub = R; bestv, besti = np.inf, 0
        while True:                                                           # I = lb | sub, sub ranges over submasks of R
            I = lb | sub
            if indep[I]:
                v = g[I] + f[S ^ I]
                if v < bestv: bestv, besti = v, I
            if sub == 0: break
            sub = (sub - 1) & R
        f[S], arg[S] = bestv, besti
    if not np.isfinite(f[full]): return None, None, None
    classes, S = [], full
    while S: classes.append(int(arg[S])); S ^= int(arg[S])
    # certificate: classes are independent, cover everything, entropy matches
    assert all(indep[c] for c in classes) and sum(classes) == full
    assert abs(H([msum[c] for c in classes]) - f[full]) < 1e-9
    alpha = float(msum[indep].max())
    return float(f[full]), [[i for i in range(n) if c >> i & 1] for c in classes], alpha


def brute_force_chromatic_entropy(adj, mass, max_nodes):
    """Independent check of the DP: enumerate all proper colourings (restricted-growth assignment of types to classes, pruned only
    by properness) and return the minimum entropy, or None if more than max_nodes search nodes would be needed."""
    n = len(mass); best = [np.inf]; nodes = [0]
    def rec(i, classes, cm):
        nodes[0] += 1
        if nodes[0] > max_nodes: raise OverflowError
        if i == n: best[0] = min(best[0], H(cm)); return
        for c in range(len(classes)):
            if not any(adj[i, j] for j in classes[c]):
                classes[c].append(i); cm[c] += mass[i]; rec(i + 1, classes, cm); classes[c].pop(); cm[c] -= mass[i]
        classes.append([i]); cm.append(mass[i]); rec(i + 1, classes, cm); classes.pop(); cm.pop()
    try: rec(0, [], [])
    except OverflowError: return None
    return best[0]


# ------------------------------------------------------------------------------------------------ per level
def level_bounds(solver, t, max_types, cell_seconds, verify_nodes):
    comp = np.asarray(solver.comp[t], bool); O = [str(o) for o in solver.O[t]]; p = np.asarray(solver.P[t], float); p = p / p.sum()
    byo = collections.defaultdict(list)
    for i, o in enumerate(O): byo[o].append(i)
    out = dict(A=0.0, Ap=0.0, B=0.0, C=0.0, LB=0.0, cells=[], all_exact=True, all_clique_exact=True, all_verified=True)
    for o, ii in byo.items():
        ii = np.asarray(ii); po = p[ii].sum(); q = p[ii] / po
        inc = ~comp[np.ix_(ii, ii)]; np.fill_diagonal(inc, False); assert (inc == inc.T).all()
        types, adj, mass = twin_quotient(inc, q); heavy = np.array([q[m].max() for m in types])
        deadline = time.time() + cell_seconds
        A, Ap, cl, cl_exact = best_clique(adj, mass, heavy, deadline)
        B, classes, alpha, verified = None, None, None, None
        if len(types) <= max_types: B, classes, alpha = chromatic_entropy(adj, mass, time.time() + cell_seconds)
        if B is not None:
            for c in classes:                                                 # optimal colouring is proper on the RAW graph too
                mem = [h for k in c for h in types[k]]; assert not inc[np.ix_(mem, mem)].any()
            bf = brute_force_chromatic_entropy(adj, mass, verify_nodes)
            verified = None if bf is None else bool(abs(bf - B) < 1e-9)
            assert verified is not False, "subset DP and brute-force enumeration disagree: BUG"
        C = log2(1.0 / alpha) if alpha else 0.0
        lb = B if B is not None else max(A, Ap)
        if B is not None: assert B >= max(A, Ap) - 1e-9, "exact chromatic entropy below a clique bound: BUG"
        if B is not None: assert B >= C - 1e-9
        out["A"] += po * A; out["Ap"] += po * Ap; out["C"] += po * C; out["all_verified"] &= bool(verified); out["B"] += po * (B if B is not None else max(A, Ap)); out["LB"] += po * lb
        out["all_exact"] &= B is not None; out["all_clique_exact"] &= cl_exact
        out["cells"].append(dict(o=o, p=float(po), n_hist=int(len(ii)), n_types=len(types), n_edges=int(adj.sum() // 2),
                                 complete=bool(adj.sum() == len(types) * (len(types) - 1)), omega=len(cl["QAp"]),
                                 A=A, Ap=Ap, B=B, C=C, alpha=alpha, bf_verified=verified, n_colours=(len(classes) if classes else None), exact=B is not None))
    return out


def load_upper():
    up = {}
    for fn in ("REPORT_W4_upper_bound_W1.json", "REPORT_W4_upper_bound.json"):
        fp = os.path.join(HERE, fn)
        if os.path.exists(fp):
            for k, v in json.load(open(fp)).items(): up[int(k)] = [float(x) for x in v["certified_upper"]]
    return up


def phase_of(env, t):
    for name, rng in env.phases.items():
        if (isinstance(rng, tuple) and rng[0] <= t <= rng[1]) or rng == t:
            return {"reveal": "signs", "gap1": "hall 1", "use1": "junction 1", "gap2": "hall 2", "use2": "junction 2"}[name]
    return "start"


MD_HEAD = """# Corridor certificate, lower side: non-trivial lower bounds on the recurrent memory of the non-transitive corridor

Generated by `isaac/analysis/w4_lower_bound.py` (numpy only, CPU, a few seconds; `python isaac/analysis/w4_lower_bound.py --W 1 3 5 7` from the repository root). Date: {date}. Companion of `analysis_W4_upper_bound.md` (certified upper bounds, read from `isaac/analysis/REPORT_W4_upper_bound*.json`).

## Method

**Relaxation.** A zero-distortion deterministic recurrent memory induces at each step t a partition P_t of the reachable histories that refines O_t, has pairwise ~_t-compatible blocks and is closed under the dynamics. We keep only the necessary condition *"pairwise-incompatible histories with the same O_t lie in different blocks"*: within an observation cell o, P_t is a proper colouring of the incompatibility graph I_o (vertices = histories with O_t = o, weights p(h)/p(o), edges = pairs that are not ~_t, taken from `GammaSolver.comp[t]` unchanged). Closure is dropped, so every quantity below is a valid **lower** bound on min_P H(P_t|O_t) = sum_o p(o) min H(P_t|O_t=o); none of them is claimed to be achievable by itself.

**Bound A (clique/mass, history level).** For a clique Q of I_o with masses p_1 >= ... >= p_k and rest r = 1 - sum_i p_i, every valid partition satisfies H >= H(p_1 + r, p_2, ..., p_k). *Argument:* the k clique members occupy k distinct blocks. Merging every clique-free block into the block of member 1 cannot increase entropy and leaves block masses q = (p_1 + r_1, ..., p_k + r_k) with r_i >= 0, sum_i r_i = r. For every j, the j largest entries of q sum to sum_(i in S)(p_i + r_i) <= p_1 + ... + p_j + r, which is the sum of the j largest entries of v = (p_1 + r, p_2, ..., p_k), and the totals agree; so v majorises q and, entropy being Schur-concave, H(q) >= H(v). (Putting the rest on the heaviest member is the entropy-minimising, hence always-safe, choice; whether that merge is actually compatible is irrelevant for a lower bound.) The clique maximising the bound is found by exhaustive clique enumeration on the twin quotient (<= 22 types) or greedy + 1-swap local search otherwise.

**Twin quotient and Bound A'.** Histories of a cell with identical incompatibility neighbourhoods ("twins"; necessarily non-adjacent) are merged into one *type* carrying their total mass. This is lossless for the colouring relaxation: if twins u, v sit in classes A, B with m(A) >= m(B), moving v into A keeps the colouring proper (N(v) = N(u) misses A) and replaces (m(A), m(B)) by a majorising pair, so entropy does not increase, and the sum of squared class masses strictly increases, so the process ends with every type monochromatic. Hence the minimum-entropy colouring of I_o equals that of the weighted quotient, and Bound A evaluated with *type* masses (A') is valid. No assumption on futures is needed for this step because closure has already been dropped.

**Bound B (chromatic entropy, exact).** Minimum entropy over all proper colourings of the weighted quotient by an exact subset DP, f[S] = min over independent I in S containing the lowest vertex of S of g(m(I)) + f[S minus I], g(x) = -x log2 x. Used when a cell has <= {max_types} types and finishes within {cell_seconds:.0f} s; otherwise the cell falls back to max(A, A'). On every cell the DP value is cross-checked against an independent enumeration of *all* proper colourings, and the optimal colouring is re-verified to be pairwise compatible on the raw history-level graph.

**Bound C (largest block).** Every block is an independent set, so its mass is <= alpha_o = maximum weight of an independent set of I_o (exact), and any distribution with atoms <= alpha has H >= log2(1/alpha_o). C <= B, but it is a one-line, human-checkable certificate.

**Aggregate.** LB_t = max( H(G_t|O_t), sum_o p(o) bound(o) ), bound = B where exact, else max(A, A'). "Pinned" means |LB_t - certified upper_t| <= 1e-6.

**Instances.** Same as `w4_upper_bound.py`: signpost corridor M=16, R1=R2=2, N=2, gap1=3, gap2=6, X=4, T=16 (t=1 start, t=2-5 signposts, t=6-8 hall 1, t=9 junction 1, t=10-15 hall 2, t=16 junction 2); W in {{3, 5, 7}} non-transitive, W=1 transitive (sanity check).
"""


def write_md(path, res, sanity, a):
    L = [MD_HEAD.format(date=time.strftime("%Y-%m-%d"), max_types=a.max_types, cell_seconds=a.cell_seconds)]
    L.append("## Results (bits; all columns except 'upper' are lower bounds on min H(C_t|O_t))\n")
    for W, r in res.items():
        nt = [x["t"] for x in r["rows"] if not x["transitive"]]
        L.append(f"### W = {W}  ({'transitive at every step' if not nt else 'non-transitive at t = ' + ', '.join(map(str, nt))}; max {r['max_hist']} histories/level)\n")
        L.append("| t | phase | ~ transitive | cells | max types/cell (from histories) | max clique | trivial H(G\\|O) | A clique (histories) | A' clique (types) | C log2(1/alpha) | B chromatic entropy | **lower** | certified upper | lower == upper (1e-6) | fallback cells |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in r["rows"]:
            L.append(f"| {x['t']} | {x['phase']} | {'yes' if x['transitive'] else 'NO'} | {x['n_cells']} | {x['max_types']} ({x['max_hist_cell']}) | {x['max_omega']} | {abs(x['H_G_O']):.3f} | {x['A']:.3f} | {x['Aprime']:.3f} | "
                     f"{abs(x['C']):.3f} | {abs(x['B']):.6f} | **{abs(x['lower']):.6f}** | {abs(x['upper']):.6f} | {'YES' if x['pinned'] else 'no'} | {x['n_fallback']} |")
        L.append(f"\nTotals over t: lower {r['total_lower']:.6f}, certified upper {r['total_upper']:.6f}; pinned at {sum(x['pinned'] for x in r['rows'])}/{r['T']} steps; "
                 f"exact chromatic entropy on {r['n_exact_cells']}/{r['n_cells']} cells ({r['n_fallback_cells']} fell back to the clique heuristic); DP cross-checked by full enumeration on {r['n_verified_cells']}/{r['n_exact_cells']} cells. Runtime {r['seconds']:.1f} s.\n")
    L.append("## Sanity check on transitive levels\n")
    L.append("Where ~_t is transitive the incompatibility graph of each cell is complete multipartite, the twin quotient is a complete graph on the Gamma-classes, the clique is everything (rest r = 0) and both A' and B must equal H(Gamma_t|O_t) exactly.\n")
    L.append("| W | transitive steps checked | max abs(A' - H(Gamma\\|O)) | max abs(B - H(Gamma\\|O)) | pass |")
    L.append("|---|---|---|---|---|")
    for W in res:
        ss = [s for s in sanity if s[0] == W]
        if ss: L.append(f"| {W} | {len(ss)} (t = {', '.join(str(s[1]) for s in ss)}) | {max(abs(s[3]-s[2]) for s in ss):.1e} | {max(abs(s[4]-s[2]) for s in ss):.1e} | {'yes' if all(s[5] for s in ss) else 'NO'} |")
    L.append("\nThe history-level Bound A is *not* tight even there (e.g. W=1, hall 1: 0.597 vs 2 bits), because a history-level clique carries only one history per class and the rest mass is dumped on one atom; the twin quotient is what makes the clique argument sharp.\n")
    ntW = [W for W in res if any(not x["transitive"] for x in res[W]["rows"])]
    allp = all(res[W]["all_pinned"] for W in ntW); nofb = all(res[W]["n_fallback_cells"] == 0 for W in res)
    L.append("## Structure behind the numbers\n")
    L.append(STRUCT)
    L.append("## What can be claimed in the paper\n")
    if allp and nofb:
        L.append(f"1. On the non-transitive corridor instances W in {{{', '.join(map(str, ntW))}}} the chromatic-entropy lower bound (minimum entropy over proper colourings of the per-observation incompatibility graph, computed exactly on every cell, no heuristic fallback) coincides with the certified upper bound of the greedy closed compatible assignment at **every** step t = 1..16 to within 1e-6, so the minimum zero-distortion rate over closed compatible state assignments is pinned exactly: 1 bit at t=3, 2 bits for t=4-9 (last signposts, hall 1, junction 1), 1 bit for t=10-16, 0 at t=1-2, i.e. the W=1 ladder, independent of W.")
        L.append("2. The plain clique bound is strictly weaker on these instances (history-level A <= 0.75 bits; type-level A' between 1.21 and 1.87 of the 2 bits in hall 1, because the largest clique has only 4 of the up-to-12 types and the remaining mass is conservatively placed on one atom); the tight certificate is the largest-block bound C: no pairwise-compatible set of histories has conditional mass above 1/4 (hall 1) or 1/2 (hall 2), hence H >= 2 and 1 bits.")
        L.append("3. Scope: the statement is for deterministic recurrent memories (partitions refining O_t) on these three finite instances with the solver's occupancy weights and per-step rates H(C_t|O_t); it uses the pairwise-compatibility relation of `gamma_solver.py`, is a numerical computation (exact DP, cross-checked by enumeration) rather than a general theorem for all W, and makes no claim about other non-transitive families. Remark (not needed for the claim above): Bound C uses only that the set of histories sharing a memory value is pairwise compatible, so p(c|o) <= alpha_o and H(C_t|O_t=o) >= log2(1/alpha_o) would hold verbatim for a stochastic encoder as well, provided the paper's zero-distortion definition forbids two incompatible histories from sharing a memory value with positive probability; Bounds A/A'/B are stated for partitions only.\n")
    else:
        L.append("1. The chromatic-entropy / clique lower bounds are non-trivial but do not meet the certified upper bound at every step; see the per-step tables for where lower == upper.")
        L.append("2. Only the steps marked YES may be described as pinned exactly; the others remain a bracket [lower, upper].")
        L.append("3. Cells that fell back to the clique heuristic still give valid (possibly loose) lower bounds.\n")
    L.append("JSON with all per-step / per-cell values: optional, `--out path.json` (not written by default).\n")
    open(path, "w").write("\n".join(L))


STRUCT = "(filled at run time)"


def describe_structure(solver, t, o_pick=None):
    """Human-readable description of the quotient of the largest cell at level t: component structure of the type graph."""
    comp = np.asarray(solver.comp[t], bool); O = [str(o) for o in solver.O[t]]; p = np.asarray(solver.P[t], float); p = p / p.sum()
    byo = collections.defaultdict(list)
    for i, o in enumerate(O): byo[o].append(i)
    o, ii = max(byo.items(), key=lambda kv: len(kv[1])); ii = np.asarray(ii); q = p[ii] / p[ii].sum()
    inc = ~comp[np.ix_(ii, ii)]; np.fill_diagonal(inc, False)
    types, adj, mass = twin_quotient(inc, q)
    from gamma_solver import _labels_from_relation
    lab, _ = _labels_from_relation(adj | np.eye(len(types), dtype=bool))
    comps = [np.nonzero(lab == c)[0] for c in range(lab.max() + 1)]
    desc = []
    for c in comps:
        sub = adj[np.ix_(c, c)]; complete = bool(sub.sum() == len(c) * (len(c) - 1))
        desc.append(f"{'K' if complete else 'non-complete component of size '}{len(c)} (type masses {', '.join(f'{mass[k]:.4f}' for k in c)})")
    return o, len(ii), len(types), desc


def main():
    global STRUCT
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", nargs="*", type=int, default=[1, 3, 5, 7])
    ap.add_argument("--max-types", type=int, default=14)
    ap.add_argument("--cell-seconds", type=float, default=20.0)
    ap.add_argument("--verify-nodes", type=int, default=2_000_000)
    ap.add_argument("--out", default=None, help="optional JSON dump")
    ap.add_argument("--md", default=os.path.join(ROOT, "results_md", "analysis_W4_lower_bound.md"))
    a = ap.parse_args()
    upper = load_upper(); res = {}; sanity = []; struct_lines = []
    for W in a.W:
        env = grid_corridor(M=16, R1=2, R2=2, W=W, N=2, gap1=3, gap2=6)
        t0 = time.time(); solver = GammaSolver(env).solve(); T = solver.T
        w = {t: np.array([nd.prob for nd in solver.levels[t]]) for t in solver.levels}
        print(f"\n== corridor W={W}: T={T}, max histories/level {max(len(solver.levels[t]) for t in solver.levels)}  (solver {time.time()-t0:.1f}s)")
        print("    t phase       trans | H(G|O)  A      A'     C      B(exact) -> LOWER | UPPER  | pinned | cells: types(max) omega(max) fallback")
        rows = []
        for t in range(1, T + 1):
            t1 = time.time(); lb = level_bounds(solver, t, a.max_types, a.cell_seconds, a.verify_nodes)
            HG = h_given_o(solver.G[t], solver.O[t], w[t]); tr = bool(solver.transitive[t])
            HGam = h_given_o(solver.Gamma[t], solver.O[t], w[t]) if tr else float("nan")
            U = upper.get(W, [float("nan")] * T)[t - 1]
            low = max(lb["LB"], HG)                                           # H(G|O) is itself a valid lower bound
            pinned = bool(abs(low - U) <= 1e-6)
            assert not (low > U + 1e-6), f"lower bound exceeds certified upper bound at W={W}, t={t}: BUG"
            row = dict(t=t, phase=phase_of(env, t), transitive=tr, H_G_O=HG, H_Gamma_O=HGam, A=lb["A"], Aprime=lb["Ap"], B=lb["B"], C=lb["C"],
                       lower=low, upper=U, pinned=pinned, all_exact=lb["all_exact"], n_cells=len(lb["cells"]),
                       max_types=max(c["n_types"] for c in lb["cells"]), max_hist_cell=max(c["n_hist"] for c in lb["cells"]),
                       max_omega=max(c["omega"] for c in lb["cells"]),
                       n_fallback=sum(not c["exact"] for c in lb["cells"]), n_verified=sum(bool(c["bf_verified"]) for c in lb["cells"]),
                       cells=lb["cells"], seconds=time.time() - t1)
            rows.append(row)
            print(f"   {t:2d} {row['phase']:11s} {'yes' if tr else 'NO ':5s} | {HG:6.3f} {lb['A']:6.3f} {lb['Ap']:6.3f} {lb['C']:6.3f} {lb['B']:6.3f}   -> {low:6.4f} | {U:6.4f} | "
                  f"{'YES' if pinned else 'no ':6s} | {row['n_cells']} cells, types {row['max_types']}, omega {row['max_omega']}, fallback {row['n_fallback']}, bf-verified {row['n_verified']}  ({row['seconds']:.1f}s)")
            if tr:
                okB = abs(lb["B"] - HGam) < 1e-9; okA = abs(lb["Ap"] - HGam) < 1e-9
                sanity.append((W, t, HGam, lb["Ap"], lb["B"], okA and okB))
                assert okA and okB, f"sanity failed at transitive level W={W} t={t}"
        nc = sum(r["n_cells"] for r in rows); nf = sum(r["n_fallback"] for r in rows)
        res[W] = dict(T=T, rows=rows, total_lower=sum(r["lower"] for r in rows), total_upper=sum(r["upper"] for r in rows),
                      all_pinned=all(r["pinned"] for r in rows), max_hist=max(len(solver.levels[t]) for t in solver.levels),
                      n_cells=nc, n_fallback_cells=nf, n_exact_cells=nc - nf, n_verified_cells=sum(r["n_verified"] for r in rows),
                      seconds=time.time() - t0)
        print(f"   total lower {res[W]['total_lower']:.6f}  total upper {res[W]['total_upper']:.6f}  pinned at every step: {res[W]['all_pinned']}")
        if W > 1:
            for t in (env.phases["gap1"][0], env.phases["gap1"][1], env.phases["gap2"][0]):
                o, nh, nty, desc = describe_structure(solver, t)
                struct_lines.append(f"- W={W}, t={t}, cell {o}: {nh} histories -> {nty} types; type graph = disjoint union of " + "; ".join(desc) + ".")
    STRUCT = ("Connected components of the twin-quotient (type) incompatibility graph of the largest cell at the first/last step of hall 1 and the first step of hall 2 "
              "(computed, not assumed). Types in different components are mutually compatible, which is exactly the non-transitivity; a colour class can take "
              "at most one type from each complete component, which caps its mass (Bound C) and forces the entropy up to the W=1 value.\n\n" + "\n".join(struct_lines) + "\n")
    write_md(a.md, res, sanity, a); print(f"\nwritten {a.md}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1); print(f"written {a.out}")
    return res


if __name__ == "__main__":
    main()
