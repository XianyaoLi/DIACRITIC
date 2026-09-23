"""W4: structure of the compatibility relation ~_t on the empirical finite POMDPs of Task A (taskA2_M*) and A' (tier4_gap6),
and an exact solver for the minimum-entropy CLOSED COMPATIBLE PARTITION sequence (the zero-distortion recurrent memory
without assumption (A4); paper Section 2.3 and Appendix A.3).

Problem solved exactly (per dataset, plug-in entropies in bits):
    minimise   sum_t w_t * H(P_t | O_t)
    over       partitions P_t of the reachable level-t histories, t = 1..T, such that
      (i)  every cell of P_t is a ~_t-compatible set (pairwise compatible under solver.comp[t]);
      (ii) closure / recurrent realisability: if h, h' lie in one cell of P_{t-1} and both are extended by the same
           (a_{t-1}, o_t) then h(a,o) and h'(a,o) lie in one cell of P_t.  Level 0 is a single cell (fixed C_0), so all
           level-1 histories with the same o_1 must share a cell (there is exactly one level-1 history per o_1 anyway).
Modelling notes (see REPORT_W4_structure.md):
  * cells are automatically inside one O_t value (compatibility requires O_t(h)=O_t(h')), so the joint (O_t,C_t)-cell
    formulation of the paper and the "code partition" formulation have the same value H(C_t|O_t); we optimise the joint cells.
  * weights: 'solver' = exact occupancy of the enumerated POMDP (uniform latent prior, as in the paper's H(G|O), H(Gamma^s|O));
    'empirical' = frequency of the symbolic history among the recorded episodes.
  * ties: among cost-equal candidates the coarser one (fewer cells) is kept; the number of optimal candidates is counted.
Algorithm: backward DP over levels with the partition as the state, exact reductions:
  (R1) future-isomorphism symmetry: histories with isomorphic probability-weighted future subtrees are interchangeable;
       the DP state is the multiset of cell types (memoised);
  (R2) factorisation over "future components": histories whose descendants are never ~-compatible at any later level are
       optimised independently (H(P_t|O_t) = H(K_t|O_t) + sum_K local_K);
  (R3, optional, --prune coarsest) valid only when ~_s is transitive for all s >= t: keep only the coarsest candidate.
Usage (from repo root, ~1-2 min):
  python isaac/analysis/w4_structure.py                # all datasets, exact DP on taskA2_M4
  ... --dp taskA2_M4 taskA2_M8 --weights empirical --per-step --selftest
"""
import os, sys, time, argparse, json, collections, itertools
from functools import lru_cache
from math import log2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "toy"))
from aprime_data import prepare, load_episodes            # noqa: E402
from gamma_solver import GammaSolver, _labels_from_relation, cond_entropy   # noqa: E402
from toy_env import Env                                                    # noqa: E402

TASKA = ["taskA2_M4", "taskA2_M8", "taskA2_M16", "taskA2_M32"]
APRIME = ["tier4_gap6"]
PY = sys.executable


# ------------------------------------------------------------------------------------------------ helpers
def bell(n):
    B = [1]
    for i in range(n):
        B.append(sum(__import__("math").comb(i, k) * B[k] for k in range(i + 1)))
    return B[n]


def maximal_cliques(adj):
    """Bron-Kerbosch with pivoting on a boolean adjacency matrix (diagonal ignored)."""
    n = adj.shape[0]
    nbrs = [set(np.nonzero(adj[i])[0].tolist()) - {i} for i in range(n)]
    out = []

    def bk(R, P, X):
        if not P and not X:
            out.append(frozenset(R)); return
        u = max(P | X, key=lambda v: len(nbrs[v] & P))
        for v in list(P - nbrs[u]):
            bk(R | {v}, P & nbrs[v], X & nbrs[v]); P = P - {v}; X = X | {v}
    bk(set(), set(range(n)), set())
    return out


def nontransitive_triples(comp):
    C = comp.astype(int); n = C.shape[0]
    two = (C @ C) > 0               # i-j-k path
    bad = two & ~comp
    n_pairs = int(bad.sum() // 2)
    n_trip = 0
    for i, k in zip(*np.nonzero(np.triu(bad, 1))):
        n_trip += int((C[i] & C[k]).sum())
    return n_pairs, n_trip


def count_vector_partitions(vec, cap_states=60000):
    """Number of multiset partitions of a multipartite number vec (= number of set partitions of a multiset with
    vec[i] indistinguishable atoms of type i, up to permuting identical atoms).  Returns None if too large to count."""
    vec = tuple(int(v) for v in vec if v > 0)
    if not vec:
        return 1
    ranges = [range(v + 1) for v in vec]
    if int(np.prod([v + 1 for v in vec])) > 400:
        return None
    parts = [p for p in itertools.product(*ranges) if any(p)]
    parts.sort(reverse=True)                                    # lexicographically decreasing
    pidx = {p: i for i, p in enumerate(parts)}
    n_states = [0]

    @lru_cache(maxsize=None)
    def f(rem, maxi):                                            # parts with index >= maxi (i.e. lex <= parts[maxi])
        n_states[0] += 1
        if not any(rem):
            return 1
        tot = 0
        for j in range(maxi, len(parts)):
            p = parts[j]
            if all(p[i] <= rem[i] for i in range(len(rem))):
                tot += f(tuple(r - q for r, q in zip(rem, p)), j)
        return tot
    return f(vec, 0)


def h_given_o(cells_of_node, O, w):
    return cond_entropy(list(cells_of_node), list(O), list(w))


# ------------------------------------------------------------------------------------------------ per-level structure
def level_structure(solver, weights, describe):
    """Per level t: sizes, compatibility structure, bracket.  weights[t] = np.array of node weights."""
    rows, a4_details = [], {}
    for t in range(1, solver.T + 1):
        nodes = solver.levels[t]; n = len(nodes); comp = solver.comp[t]; w = weights[t]
        lab, tr = _labels_from_relation(comp)
        n_comp = int(lab.max()) + 1
        cliques = maximal_cliques(comp)
        csz = collections.Counter(len(c) for c in cliques)
        n_pairs, n_trip = nontransitive_triples(comp)
        # candidate-partition bound: product over maximal cliques of Bell(size) when the graph is a union of cliques
        bell_bound = __import__('math').prod(bell(len(c)) for c in cliques) if tr else None
        G = solver.G[t]; GS = solver.GammaS[t]; Gam = solver.Gamma[t]
        hist = [nd.hist for nd in nodes]
        row = dict(t=t, n=n, n_obs=len(set(solver.O[t])), transitive=tr, a4=solver.a4_ok[t], n_components=n_comp,
                   n_maxcliques=len(cliques), clique_sizes=dict(sorted(csz.items())), nontrans_pairs=n_pairs,
                   nontrans_triples=n_trip, nG=len(set(G)), nGamma=(len(set(Gam)) if Gam is not None else None),
                   nGammaS=len(set(GS)), bell_bound=bell_bound,
                   H_G=h_given_o(G, solver.O[t], w), H_Gamma=(h_given_o(Gam, solver.O[t], w) if Gam is not None else float("nan")),
                   H_GammaS=h_given_o(GS, solver.O[t], w), H_H=h_given_o(hist, solver.O[t], w))
        rows.append(row)
        if not solver.a4_ok[t]:
            by_g = collections.defaultdict(list)
            for i, nd in enumerate(nodes):
                by_g[G[i]].append(i)
            det = []
            for g, idx in by_g.items():
                sup = collections.defaultdict(list)
                for i in idx:
                    sup[frozenset(nodes[i].cont.keys())].append(i)
                if len(sup) > 1:
                    det.append(dict(G=g, O=str(nodes[idx[0]].o), groups=[dict(support=sorted(map(str, S)), n=len(ii),
                                    mass=float(sum(w[i] for i in ii) / w.sum()), members=describe(t, ii)) for S, ii in sup.items()]))
            a4_details[t] = det
    return rows, a4_details


# ------------------------------------------------------------------------------------------------ exact DP
class ClosedPartitionDP:
    def __init__(self, solver, weights, obj_w=None, prune=None, tol=1e-9):
        self.s = solver; self.T = solver.T; self.w = weights; self.tol = tol
        self.obj_w = obj_w if obj_w is not None else {t: 1.0 for t in range(1, self.T + 1)}
        self.prune = prune
        self.Z = {t: float(weights[t].sum()) for t in weights}
        self._children()
        self._iso()
        self._future_components()
        self.memo, self.choice, self.ties, self.n_cand = {}, {}, {}, collections.Counter()
        self.n_states = collections.Counter()

    # children index maps
    def _children(self):
        s = self.s
        self.child = {t: [{u: s.index[t + 1][nd.children[u].hist] for u in nd.cont} for nd in s.levels[t]] for t in range(1, self.T)}
        self.child[self.T] = [dict() for _ in s.levels[self.T]]

    # (R1) future-isomorphism classes, computed backwards
    def _iso(self):
        s = self.s; self.iso = {}
        for t in range(self.T, 0, -1):
            keys = []
            for i, nd in enumerate(s.levels[t]):
                act = tuple(sorted((str(a), round(p, 12)) for a, p in nd.act_dist.items()))
                ch = tuple(sorted((str(u), self.iso[t + 1][j]) for u, j in self.child[t][i].items())) if t < self.T else ()
                keys.append((str(nd.o), act, round(float(self.w[t][i]), 12), ch))
            ids = {k: j for j, k in enumerate(sorted(set(keys), key=repr))}
            self.iso[t] = [ids[k] for k in keys]

    # (R2) future components: union of comp_t and "children in a common component at t+1"
    def _future_components(self):
        s = self.s; self.fc = {}
        for t in range(self.T, 0, -1):
            n = len(s.levels[t]); parent = list(range(n))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]; i = parent[i]
                return i

            def union(i, j):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            comp = s.comp[t]
            for i in range(n):
                for j in np.nonzero(comp[i])[0]:
                    union(i, int(j))
            if t < self.T:
                first = {}
                for i in range(n):
                    for j in self.child[t][i].values():
                        k = self.fc[t + 1][j]
                        if k in first:
                            union(i, first[k])
                        else:
                            first[k] = i
            roots = {}
            self.fc[t] = [roots.setdefault(find(i), len(roots)) for i in range(n)]

    # local cost of a partition (cells = list of index lists) of a component K at level t, global normalisation
    def _local_cost(self, t, cells):
        O = self.s.O[t]; w = self.w[t]; Z = self.Z[t]
        pK = collections.Counter(); pc = collections.Counter()
        for ci, c in enumerate(cells):
            for i in c:
                pK[O[i]] += w[i] / Z; pc[(ci, O[i])] += w[i] / Z
        return -sum(p * log2(p / pK[o]) for (ci, o), p in pc.items() if p > 0)

    def _atom_type(self, t, atom):
        return tuple(sorted(self.iso[t][i] for i in atom))

    def _key(self, t, atoms):
        return (t, tuple(sorted(self._atom_type(t, a) for a in atoms)))

    # candidate partitions of the atoms (blocks forced by closure), compatible, deduplicated by type
    def _candidates(self, t, atoms):
        comp = self.s.comp[t]
        types = [self._atom_type(t, a) for a in atoms]
        order = sorted(range(len(atoms)), key=lambda k: types[k])
        atoms = [atoms[k] for k in order]; types = [types[k] for k in order]
        m = len(atoms)
        for a in atoms:                       # atom internally compatible? (can fail on non-transitive instances)
            ii = list(a)
            if not comp[np.ix_(ii, ii)].all():
                return []
        A = np.zeros((m, m), dtype=bool)
        for x in range(m):
            for y in range(x, m):
                A[x, y] = A[y, x] = comp[np.ix_(list(atoms[x]), list(atoms[y]))].all()
        if self.prune == "coarsest":          # (R3): merge everything mutually compatible (valid under transitivity)
            lab, tr = _labels_from_relation(A)
            assert tr, "coarsest pruning requires a transitive relation"
            cells = [[i for x in range(m) if lab[x] == c for i in atoms[x]] for c in range(lab.max() + 1)]
            return [(cells, tuple(lab.tolist()))]
        out, seen = [], set()
        rgs = [0] * m

        def rec(x, ncell):
            if x == m:
                cells = [[] for _ in range(ncell)]
                for y in range(m):
                    cells[rgs[y]].extend(atoms[y])
                key = tuple(sorted(tuple(sorted(self.iso[t][i] for i in c)) for c in cells))
                if key not in seen:
                    seen.add(key); out.append((cells, tuple(rgs)))
                return
            lo = rgs[x - 1] if x > 0 and types[x] == types[x - 1] else 0     # identical atoms: non-decreasing labels
            for c in range(lo, ncell + 1):
                if all(A[x, y] for y in range(x) if rgs[y] == c):
                    rgs[x] = c
                    rec(x + 1, max(ncell, c + 1))
        rec(0, 0)
        return out

    # blocks at t+1 forced by the cells at t, grouped by future component
    def _forced_blocks(self, t, cells):
        if t == self.T:
            return {}
        byK = collections.defaultdict(list)
        for c in cells:
            blocks = collections.defaultdict(list)
            for i in c:
                for u, j in self.child[t][i].items():
                    blocks[str(u)].append(j)
            for b in blocks.values():
                byK[self.fc[t + 1][b[0]]].append(frozenset(b))
        return byK

    def value(self, t, atoms):
        """Min cost-to-go for the component covered by `atoms` (forced blocks) at level t."""
        key = self._key(t, atoms)
        if key in self.memo:
            return self.memo[key]
        self.n_states[t] += 1
        best, best_rgs, n_opt, best_cells = float("inf"), None, 0, None
        cands = self._candidates(t, list(atoms))
        self.n_cand[t] += len(cands)
        for cells, rgs in cands:
            cost = self.obj_w[t] * self._local_cost(t, cells)
            for K, blocks in self._forced_blocks(t, cells).items():
                cost += self.value(t + 1, tuple(blocks))
                if cost >= best + self.tol:
                    break
            if cost < best - self.tol:
                best, best_rgs, n_opt, best_cells = cost, rgs, 1, len(cells)
            elif abs(cost - best) <= self.tol:
                n_opt += 1
                if len(cells) < best_cells:
                    best_rgs, best_cells = rgs, len(cells)
        self.memo[key] = best; self.choice[key] = best_rgs; self.ties[key] = n_opt
        return best

    def solve(self):
        """Returns total objective (incl. the constant sum_t w_t H(K_t|O_t)) and the optimal partition labels per level."""
        s = self.s
        # constant part: component entropy
        const = sum(self.obj_w[t] * h_given_o(self.fc[t], s.O[t], self.w[t]) for t in range(1, self.T + 1))
        # level-1 forced blocks: one cell C_0 -> histories grouped by o_1, within future components
        byK = collections.defaultdict(lambda: collections.defaultdict(list))
        for i, nd in enumerate(s.levels[1]):
            byK[self.fc[1][i]][str(nd.o)].append(i)
        total = const
        roots = []
        for K, d in byK.items():
            atoms = tuple(frozenset(b) for b in d.values())
            total += self.value(1, atoms); roots.append(atoms)
        # reconstruct
        self.P = {t: -np.ones(len(s.levels[t]), dtype=int) for t in range(1, self.T + 1)}
        self.ncell = collections.Counter()
        stack = [(1, a) for a in roots]
        while stack:
            t, atoms = stack.pop()
            atoms = list(atoms)
            types = [self._atom_type(t, a) for a in atoms]
            order = sorted(range(len(atoms)), key=lambda k: types[k]); atoms = [atoms[k] for k in order]
            rgs = self.choice[self._key(t, atoms)]
            if rgs is None:
                raise RuntimeError("infeasible subproblem reached")
            cells = [[] for _ in range(max(rgs) + 1)]
            for y, c in enumerate(rgs):
                cells[c].extend(atoms[y])
            for c in cells:
                cid = self.ncell[t]; self.ncell[t] += 1
                for i in c:
                    self.P[t][i] = cid
            for K, blocks in self._forced_blocks(t, cells).items():
                stack.append((t + 1, tuple(blocks)))
        assert all((self.P[t] >= 0).all() for t in self.P)
        self.per_t = [h_given_o(self.P[t].tolist(), s.O[t], self.w[t]) for t in range(1, self.T + 1)]
        assert abs(sum(self.obj_w[t] * self.per_t[t - 1] for t in range(1, self.T + 1)) - total) < 1e-6, (total, self.per_t)
        self.total = total
        self.n_ties = sum(1 for k, v in self.ties.items() if v > 1)
        return total, self.per_t

    def check_closed_compatible(self):
        """Independent verification of (i) and (ii) for the reconstructed partition."""
        s = self.s
        for t in range(1, self.T + 1):
            P = self.P[t]
            for c in set(P.tolist()):
                ii = np.nonzero(P == c)[0]
                assert s.comp[t][np.ix_(ii, ii)].all(), f"cell not compatible at t={t}"
            if t < self.T:
                for i in range(len(P)):
                    for j in range(i + 1, len(P)):
                        if P[i] == P[j]:
                            for u, ci in self.child[t][i].items():
                                if u in self.child[t][j]:
                                    assert self.P[t + 1][ci] == self.P[t + 1][self.child[t][j][u]], f"closure violated t={t}"
        return True


# ------------------------------------------------------------------------------------------------ weights
def history_weights(solver, eps, key, kind):
    if kind == "solver":
        return {t: np.array([nd.prob for nd in solver.levels[t]], dtype=float) for t in solver.levels}
    W = {}
    for t in range(1, solver.T + 1):
        cnt = collections.Counter()
        for e in eps:
            hist = []
            for s_ in range(t):
                hist.append(e[key][s_])
                if s_ < t - 1:
                    hist.append(e["sym_act"][s_])
            cnt[solver.index[t][tuple(hist)]] += 1
        W[t] = np.array([cnt[i] / len(eps) for i in range(len(solver.levels[t]))], dtype=float)
    return W


def make_describe(env, eps, levels):
    th2mass = {e["theta"]: float(e["mass"]) for e in eps}
    masses = sorted(set(th2mass.values())); med = masses[len(masses) // 2] if masses else None
    fields = env.latent_fields

    def describe(t, idx):
        nodes = levels[t]
        lat = sorted({l for i in idx for l in nodes[i].post})
        th = sorted({l[fields.index("theta")] for l in lat})
        out = dict(thetas=th)
        if "g" in fields:
            out["g"] = sorted({l[fields.index("g")] for l in lat})
        if med is not None and th and all(x in th2mass for x in th):
            out["mass_half"] = sorted({int(th2mass[x] >= med) for x in th})
        out["hover_a1"] = sorted({str(nodes[i].hist[1]) for i in idx if len(nodes[i].hist) > 1})
        return out
    return describe


# ------------------------------------------------------------------------------------------------ learned rate
def learned_rates(results, data_basename, variant="diacritic", beta=0.001, aux_theta=0.0):
    rows = [json.loads(l) for l in open(results)]
    sel = [r for r in rows if os.path.basename(r["args"]["data"]) == data_basename and r["args"]["variant"] == variant
           and abs(r["args"]["beta"] - beta) < 1e-12 and abs(r["args"].get("aux_theta", 0.0) - aux_theta) < 1e-12]
    if not sel:
        return None
    H = np.array([r["per_t"]["H(C|O)"] for r in sel])
    return dict(n_seeds=len(sel), seeds=sorted(r["args"]["seed"] for r in sel), mean=H.mean(0), sd=H.std(0), min=H.min(0), max=H.max(0),
                theory=sel[0]["theory"], phases=sel[0]["phases"])


# ------------------------------------------------------------------------------------------------ self test
def remark_i_env():
    """Paper Remark (i): three histories at t=2 sharing (O,G); U(h1)={u,v1}, U(h3)={u,v3}, U(h2)={w}; conflict after u.
    Exact minimum at t=2 is h2(1/3)=0.918 bits; H(G|O)=0, H(Gamma^s|O)=log2(3)."""
    prior = {(m,): 1 / 3 for m in (1, 2, 3)}

    def obs(lat, t, acts):
        m = lat[0]
        if t == 1: return {("reveal", m): 1.0}
        if t == 2: return {("wait",): 1.0}
        return {("u",): 0.5, ("v", m): 0.5} if m in (1, 3) else {("w",): 1.0}

    def expert(lat, t, hist):
        m = lat[0]
        if t < 3: return {("noop",): 1.0}
        return {("act", "A" if m == 1 else "B"): 1.0} if hist[-1] == ("u",) else {("noop",): 1.0}
    return Env(3, prior, obs, expert, name="remark_i", latent_fields=("m",))


def selftest():
    env = remark_i_env(); solver = GammaSolver(env).solve()
    w = {t: np.array([nd.prob for nd in solver.levels[t]]) for t in solver.levels}
    print("selftest Remark (i):  transitive per t =", [solver.transitive[t] for t in (1, 2, 3)], " A4 =", [solver.a4_ok[t] for t in (1, 2, 3)])
    dp = ClosedPartitionDP(solver, w); tot, per = dp.solve(); dp.check_closed_compatible()
    HG = [h_given_o(solver.G[t], solver.O[t], w[t]) for t in (1, 2, 3)]
    HS = [h_given_o(solver.GammaS[t], solver.O[t], w[t]) for t in (1, 2, 3)]
    print("  H(G|O)    =", np.round(HG, 4)); print("  exact min =", np.round(per, 4), " (expected [0, 0.9183, 0.3333])")
    print("  H(GamS|O) =", np.round(HS, 4), " ties:", dp.n_ties, " states/level:", dict(dp.n_states), " cand/level:", dict(dp.n_cand))
    assert abs(per[1] - (-(1 / 3) * log2(1 / 3) - (2 / 3) * log2(2 / 3))) < 1e-9 and abs(per[2] - 1 / 3) < 1e-9
    # closure stress: transitive-prune must refuse here (non-transitive at t=2)
    try:
        ClosedPartitionDP(solver, w, prune="coarsest").solve(); print("  (coarsest prune unexpectedly succeeded)")
    except AssertionError:
        print("  coarsest-prune correctly refused on the non-transitive instance")


# ------------------------------------------------------------------------------------------------ main
def fmt(x):
    return "nan" if x != x else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=TASKA + APRIME)
    ap.add_argument("--dp", nargs="*", default=["taskA2_M4", "tier4_gap6"], help="datasets on which to run the exact (unpruned) DP")
    ap.add_argument("--dp-pruned", nargs="*", default=TASKA + APRIME, help="datasets on which to run the coarsest-pruned DP")
    ap.add_argument("--weights", default="solver", choices=["solver", "empirical"])
    ap.add_argument("--per-step", action="store_true", help="also minimise each H(P_t|O_t) separately (one DP per t)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--results", default=os.path.join(HERE, "..", "cluster_results", "results", "taskA.jsonl"))
    args = ap.parse_args()
    if args.selftest:
        selftest()
    for name in args.datasets:
        d = os.path.join(HERE, "..", "data", name)
        sag = name.startswith("taskA2")
        t0 = time.time()
        eps, meta, vis, env, solver, key = prepare(d, augment=True, sag_symbol=sag)
        W = history_weights(solver, eps, key, args.weights)
        describe = make_describe(env, eps, solver.levels)
        phases = eps[0]["phase"]
        print(f"\n##### {name}  (sag_symbol={sag}, weights={args.weights}, T={env.T}, latents={len(env.prior)}, E={len(eps)}, load {time.time()-t0:.1f}s)")
        rows, a4d = level_structure(solver, W, describe)
        print("| t | phase | n_t | #obs | transitive | A4 | #comp | #maxcliques (sizes) | non-trans pairs/triples | |G| | |Gamma| | |Gamma^s| | Bell-bound | H(G|O) | H(Gamma|O) | H(Gamma^s|O) | H(H|O) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['t']} | {phases[r['t']-1]} | {r['n']} | {r['n_obs']} | {'yes' if r['transitive'] else 'NO'} | {'ok' if r['a4'] else 'FAIL'} | {r['n_components']} | "
                  f"{r['n_maxcliques']} {r['clique_sizes']} | {r['nontrans_pairs']}/{r['nontrans_triples']} | {r['nG']} | {r['nGamma']} | {r['nGammaS']} | {r['bell_bound']} | "
                  f"{fmt(r['H_G'])} | {fmt(r['H_Gamma'])} | {fmt(r['H_GammaS'])} | {fmt(r['H_H'])} |")
        print("all levels transitive:", all(r["transitive"] for r in rows), "; A4 fails at t =", [r["t"] for r in rows if not r["a4"]])
        for t, det in a4d.items():
            print(f"A4 failure detail t={t} ({phases[t-1]}):")
            for g in det:
                print(f"  (O,G)-class O={g['O']}:")
                for grp in g["groups"]:
                    print(f"    support={grp['support']}  n={grp['n']} mass={grp['mass']:.4f} members={grp['members']}")
        # symmetry / state-count estimates for the unpruned DP
        dp0 = ClosedPartitionDP(solver, W)
        print("| t | future components (sizes) | iso classes per comp-component (atoms per type) | DP memo-state bound = sum over future components of #partition types (orbits) | Bell-bound (no symmetry) |")
        print("|---|---|---|---|---|")
        for t in range(1, solver.T + 1):
            fc = np.array(dp0.fc[t]); iso = np.array(dp0.iso[t])
            lab, _ = _labels_from_relation(solver.comp[t])
            fsz = sorted(collections.Counter(fc.tolist()).values(), reverse=True)
            typ, total_states = [], 0
            for K in range(fc.max() + 1):
                orbits_K = 1
                for c in sorted(set(lab[fc == K].tolist())):
                    cnt = sorted(collections.Counter(iso[lab == c].tolist()).values(), reverse=True)
                    typ.append(tuple(cnt))
                    o = count_vector_partitions(cnt)
                    orbits_K = None if (o is None or orbits_K is None) else orbits_K * o
                total_states = None if (orbits_K is None or total_states is None) else total_states + orbits_K
            tc = collections.Counter(typ)
            print(f"| {t} | {len(fsz)} {fsz if len(fsz) <= 8 else str(fsz[:8]) + '...'} | {dict(tc)} | {total_states if total_states is not None else 'not counted (>1e6)'} | {rows[t-1]['bell_bound']} |")
        # DP runs
        runs = {}
        if name in args.dp_pruned and all(r["transitive"] for r in rows):
            t0 = time.time(); dp = ClosedPartitionDP(solver, W, prune="coarsest"); tot, per = dp.solve(); dp.check_closed_compatible()
            runs["pruned"] = (tot, per, dp, time.time() - t0)
        if name in args.dp:
            t0 = time.time(); dp = ClosedPartitionDP(solver, W); tot, per = dp.solve(); dp.check_closed_compatible()
            runs["exact"] = (tot, per, dp, time.time() - t0)
            if args.per_step:
                ps = []
                for tt in range(1, solver.T + 1):
                    dpt = ClosedPartitionDP(solver, W, obj_w={t: float(t == tt) for t in range(1, solver.T + 1)})
                    ps.append(dpt.solve()[0])
                runs["per_step"] = ps
        lr = learned_rates(args.results, name) if sag else None
        if runs:
            for k, v in runs.items():
                if k == "per_step":
                    continue
                tot, per, dp, secs = v
                print(f"DP[{k}]: total = {tot:.6f} bits ({secs:.1f}s), memo states per level = {dict(sorted(dp.n_states.items()))}, "
                      f"candidates enumerated per level = {dict(sorted(dp.n_cand.items()))}, decisions with ties = {dp.n_ties}, "
                      f"max cells per level = {max(dp.ncell.values())}")
            hdr = "| t | phase | H(G|O) | " + ("exact min (unpruned DP)" if "exact" in runs else "min (coarsest-pruned DP)") + " | H(Gamma|O) | H(Gamma^s|O) |" + (" per-step min |" if "per_step" in runs else "") + (" learned H(C|O) mean (sd) [min,max] |" if lr else "")
            print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
            per = runs["exact"][1] if "exact" in runs else runs["pruned"][1]
            for t in range(1, solver.T + 1):
                r = rows[t - 1]
                line = f"| {t} | {phases[t-1]} | {fmt(r['H_G'])} | {fmt(per[t-1])} | {fmt(r['H_Gamma'])} | {fmt(r['H_GammaS'])} |"
                if "per_step" in runs:
                    line += f" {fmt(runs['per_step'][t-1])} |"
                if lr:
                    line += f" {lr['mean'][t-1]:.3f} ({lr['sd'][t-1]:.3f}) [{lr['min'][t-1]:.2f},{lr['max'][t-1]:.2f}] |"
                print(line)
            if "exact" in runs and "pruned" in runs:
                print("exact == pruned (Gamma) per step:", np.allclose(runs["exact"][1], runs["pruned"][1], atol=1e-9))
            if "exact" in runs:
                gam = [r["H_Gamma"] for r in rows]
                print("exact == H(Gamma|O) per step:", np.allclose(runs["exact"][1], gam, atol=1e-9))
            print("sum_t H(G|O) = %.4f ; sum_t exact = %.4f ; sum_t H(Gamma^s|O) = %.4f" % (sum(r["H_G"] for r in rows), sum(per), sum(r["H_GammaS"] for r in rows)))
            if lr:
                print(f"learned (n_seeds={lr['n_seeds']}, seeds={lr['seeds']}): sum_t mean H(C|O) = {lr['mean'].sum():.4f}")


if __name__ == "__main__":
    main()
