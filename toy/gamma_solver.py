"""
Backward partition-refinement solver for the behavioral memory state Gamma_t (behavioral compatibility definition),
computed *automatically* from the enumerated history tree -- nothing about Gamma is specified by hand.

For every t it computes, over all positive-probability histories h at time t:
  G_t(h)        behavioral quotient class = (O_t(h), P_E(A_t | h))                      [Def. 2.2 + (A2)]
  U_t(h)        continuation support      = {(a, o') : P(a, o' | h) > 0}
  comp_t        compatibility relation defined by backward recursion:
                  h ~ h'  iff  O_t(h)=O_t(h')  and  G_t(h)=G_t(h')  and
                               for all u in U_t(h) ∩ U_t(h'):  hu ~_{t+1} h'u
                (only common reachable continuations are checked -> "future observations are free")
  A4 check      continuation-support homogeneity:  same (O_t, G_t)  =>  same U_t
  transitivity  whether comp_t is an equivalence relation.  Only then is Gamma_t := H_t / comp_t a quotient.
                If it is not, the solver REPORTS this and does NOT take a transitive closure.
  strong_t      the strong (completely specified) congruence: additionally requires U_t(h)=U_t(h') and
                agreement on all continuations.  Always an equivalence; gives a valid zero-distortion recurrent
                realization, hence an UPPER bound on R^mem_t(0).  Under A4 it coincides with comp_t.
  Gamma^(J)_t   the J-step truncated recursion (base case at depth J uses only (O, G)); Gamma^(T-t) = Gamma.
Entropies H(X | O_t) are exact (weighted by the expert-occupancy history probabilities).
"""
from __future__ import annotations
from math import log2
from typing import Dict, List, Tuple, Hashable
import numpy as np
from toy_env import Env, HistoryNode, enumerate_histories


# ---------------------------------------------------------------- entropies
def cond_entropy(labels: List[Hashable], conds: List[Hashable], probs: List[float]) -> float:
    """H(X | Y) with X=labels, Y=conds, weights probs (need not be normalized)."""
    pj: Dict[Tuple, float] = {}
    pc: Dict[Hashable, float] = {}
    Z = sum(probs)
    for x, y, p in zip(labels, conds, probs):
        pj[(x, y)] = pj.get((x, y), 0.0) + p / Z
        pc[y] = pc.get(y, 0.0) + p / Z
    return -sum(p * log2(p / pc[y]) for (x, y), p in pj.items() if p > 0)


def _labels_from_relation(rel: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Connected components of a reflexive symmetric boolean relation; second output = is it transitive?"""
    n = rel.shape[0]
    lab = -np.ones(n, dtype=int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = k
        while stack:
            j = stack.pop()
            for m in np.nonzero(rel[j])[0]:
                if lab[m] < 0:
                    lab[m] = k; stack.append(m)
        k += 1
    transitive = True
    for c in range(k):
        idx = np.nonzero(lab == c)[0]
        if not rel[np.ix_(idx, idx)].all():
            transitive = False; break
    return lab, transitive


def _refine(nodes: List[HistoryNode], base: np.ndarray, next_rel: np.ndarray | None,
            next_index: Dict[Tuple, int] | None, mode: str) -> np.ndarray:
    """One backward step. base[i,j] = same (O,G). mode 'compat' checks common continuations only,
    'strong' additionally requires equal supports."""
    n = len(nodes)
    rel = base.copy()
    if next_rel is None:
        return rel
    supports = [set(nd.cont.keys()) for nd in nodes]
    for i in range(n):
        for j in range(i + 1, n):
            if not rel[i, j]:
                continue
            Si, Sj = supports[i], supports[j]
            if mode == "strong" and Si != Sj:
                rel[i, j] = rel[j, i] = False
                continue
            ok = True
            for u in Si & Sj:
                ci = next_index[nodes[i].children[u].hist]
                cj = next_index[nodes[j].children[u].hist]
                if not next_rel[ci, cj]:
                    ok = False; break
            rel[i, j] = rel[j, i] = ok
    return rel


class GammaSolver:
    def __init__(self, env: Env):
        self.env = env
        self.levels = enumerate_histories(env)
        self.T = env.T
        self.index = {t: {nd.hist: i for i, nd in enumerate(self.levels[t])} for t in self.levels}
        self._precompute()

    # ---- per-time primitives
    def _precompute(self):
        self.G, self.O, self.P, self.a2_ok, self.a4_ok, self.base = {}, {}, {}, {}, {}, {}
        for t, nodes in self.levels.items():
            gkey = [(nd.o, tuple(sorted((a, round(p, 12)) for a, p in nd.act_dist.items()))) for nd in nodes]
            gid = {k: i for i, k in enumerate(sorted(set(gkey), key=repr))}
            self.G[t] = [gid[k] for k in gkey]
            self.O[t] = [nd.o for nd in nodes]
            self.P[t] = [nd.prob for nd in nodes]
            self.a2_ok[t] = all(nd.a2_ok for nd in nodes)
            # A4: same (O,G) => same continuation support
            sup_by_class: Dict[int, set] = {}
            a4 = True
            for g, nd in zip(self.G[t], nodes):
                S = frozenset(nd.cont.keys())
                if g in sup_by_class and sup_by_class[g] != S:
                    a4 = False
                sup_by_class.setdefault(g, S)
            self.a4_ok[t] = a4
            g = np.array(self.G[t])
            self.base[t] = (g[:, None] == g[None, :])   # (O,G) equal (O is part of G's key)

    # ---- full-horizon relations (behavioral compatibility definition) and strong congruence
    def solve(self):
        self.comp, self.strong, self.transitive = {}, {}, {}
        self.Gamma, self.GammaS = {}, {}
        nxt_c = nxt_s = None
        for t in range(self.T, 0, -1):
            nodes = self.levels[t]
            nidx = self.index.get(t + 1)
            self.comp[t] = _refine(nodes, self.base[t], nxt_c, nidx, "compat")
            self.strong[t] = _refine(nodes, self.base[t], nxt_s, nidx, "strong")
            lab, tr = _labels_from_relation(self.comp[t])
            self.transitive[t] = tr
            self.Gamma[t] = lab if tr else None
            labs, trs = _labels_from_relation(self.strong[t])
            assert trs, "strong congruence must be transitive"
            self.GammaS[t] = labs
            nxt_c, nxt_s = self.comp[t], self.strong[t]
        return self

    # ---- finite-horizon Gamma^(J)
    def gamma_J(self, t: int, J: int) -> np.ndarray:
        """Relation for Gamma^(J)_t: recursion unrolled J steps (J >= T - t gives Gamma_t)."""
        J = min(J, self.T - t)
        rel = self.base[t + J]
        for s in range(t + J - 1, t - 1, -1):
            rel = _refine(self.levels[s], self.base[s], rel, self.index[s + 1], "compat")
        return rel

    # ---- reporting
    def H_given_O(self, t: int, labels) -> float:
        return cond_entropy(list(labels), self.O[t], self.P[t])

    def rates(self) -> Dict[str, List[float]]:
        out = {"H(G|O)": [], "H(Gamma|O)": [], "H(GammaS|O)": [], "H(H|O)": []}
        for t in range(1, self.T + 1):
            out["H(G|O)"].append(self.H_given_O(t, self.G[t]))
            out["H(Gamma|O)"].append(self.H_given_O(t, self.Gamma[t]) if self.Gamma[t] is not None else float("nan"))
            out["H(GammaS|O)"].append(self.H_given_O(t, self.GammaS[t]))
            out["H(H|O)"].append(self.H_given_O(t, [nd.hist for nd in self.levels[t]]))
        return out

    def gammaJ_table(self) -> Dict[int, List[float]]:
        """H(Gamma^(J)_t | O_t) for J = 0..T-t, plus monotonicity check (Gamma^(J) coarser than Gamma^(J+1))."""
        table, self.gammaJ_monotone = {}, True
        for t in range(1, self.T + 1):
            row, prev = [], None
            for J in range(0, self.T - t + 1):
                rel = self.gamma_J(t, J)
                lab, tr = _labels_from_relation(rel)
                if prev is not None:
                    # refinement: rel_J ⊇ rel_{J+1}
                    if not (prev >= rel).all():
                        self.gammaJ_monotone = False
                prev = rel
                row.append(self.H_given_O(t, lab) if tr else float("nan"))
            assert (prev == self.comp[t]).all(), "Gamma^(T-t) must equal Gamma"
            table[t] = row
        return table

    def label_stats(self, t: int, extra: Dict[str, List[Hashable]] | None = None):
        """Convenience: counts of Gamma classes per observation etc."""
        lab = self.Gamma[t]
        per_o: Dict[Hashable, set] = {}
        for l, o in zip(lab, self.O[t]):
            per_o.setdefault(o, set()).add(l)
        return {o: len(s) for o, s in per_o.items()}

    def report(self, gammaJ: bool = True, verbose: bool = True) -> Dict:
        rates = self.rates()
        T = self.T
        if verbose:
            print(f"=== {self.env.name}  (T={T}) ===")
            print(" t  |hist|  A2  A4  transitive  H(G|O)  H(Gamma|O)  H(GammaS|O)  H(H|O)  max|Gamma_t|_o")
            for t in range(1, T + 1):
                mx = max(self.label_stats(t).values()) if self.Gamma[t] is not None else -1
                print(f"{t:2d}  {len(self.levels[t]):5d}   {'ok' if self.a2_ok[t] else 'X '}  "
                      f"{'ok' if self.a4_ok[t] else 'X '}  {'yes' if self.transitive[t] else 'NO '}      "
                      f"{rates['H(G|O)'][t-1]:5.2f}    {rates['H(Gamma|O)'][t-1]:5.2f}       "
                      f"{rates['H(GammaS|O)'][t-1]:5.2f}       {rates['H(H|O)'][t-1]:5.2f}     {mx}")
        out = dict(rates=rates, a2=[self.a2_ok[t] for t in range(1, T + 1)],
                   a4=[self.a4_ok[t] for t in range(1, T + 1)],
                   transitive=[self.transitive[t] for t in range(1, T + 1)])
        if gammaJ:
            tab = self.gammaJ_table()
            out["gammaJ"] = tab; out["gammaJ_monotone"] = self.gammaJ_monotone
            if verbose:
                print(" H(Gamma^(J)_t | O_t), rows t, columns J=0..T-t   (monotone non-decreasing in J: "
                      f"{self.gammaJ_monotone})")
                for t in range(1, T + 1):
                    print(f"  t={t:2d}: " + "  ".join(f"{v:4.2f}" for v in tab[t]))
        return out


if __name__ == "__main__":
    import argparse
    from toy_env import toy_reveal, toy_gap
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="reveal")
    args = ap.parse_args()
    env = toy_reveal() if args.which == "reveal" else toy_gap()
    GammaSolver(env).solve().report()
