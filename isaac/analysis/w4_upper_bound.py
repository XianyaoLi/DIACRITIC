"""Corridor certificate, upper side: a CERTIFIED FEASIBLE UPPER BOUND on the zero-distortion recurrent memory for the non-transitive corridor family.

For the drifting corridor (Appendix "grid", W >= 3) the exact closed-partition DP of w4_structure.py does not terminate, so the paper
only reports the general bracket  H(G_E|O) <= R^mem(0) <= H(Gamma^s|O) = 2 + log2 W  in the halls.  Any *closed compatible state
assignment* is realisable by a deterministic recurrent code (paper Section 2.3, Appendix A.3), hence its per-step conditional entropy
H(P_t|O_t) is an achievable rate = an upper bound on the minimum.  This script constructs such assignments greedily and verifies them:

  (i)  compatibility: every cell of P_t is a ~_t-compatible set (pairwise, solver.comp[t]);
  (ii) closure (successor consistency): if h, h' share a cell of P_{t-1} and are both continued by the same (a_{t-1}, o_t) then
       their continuations share a cell of P_t  -> a deterministic transition C_t = F(C_{t-1}, O_t, A_{t-1}) exists.

Algorithm (forward greedy with closure propagation and rollback): levels t = 1..T in order; at level t the cells forced by earlier
merges are given; candidate merges are pairs of cells with the same O_t whose members are pairwise compatible; a merge is applied
tentatively, its implied successor merges are propagated recursively (union of the children cells for every common continuation,
each checked for compatibility), and the whole chain is rolled back if any implied merge is incompatible.  Orders: largest-cells-first,
plus random orders (seeded); the best total sum_t H(P_t|O_t) is kept.  The final assignment is re-verified from scratch by an
independent checker (same tests as w4_structure.ClosedPartitionDP.check_closed_compatible).  Plain graph colouring of the
incompatibility graph would NOT certify (ii); this construction does.

Usage (from the repository root, CPU only):
  python isaac/analysis/w4_upper_bound.py --W 1 3 5 7 --restarts 20
"""
import os, sys, time, argparse, collections, random, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "toy"))
from grid_env import grid_corridor                                          # noqa: E402
from gamma_solver import GammaSolver                                        # noqa: E402
from w4_structure import h_given_o                                          # noqa: E402  (same estimator as the exact DP)


class Assignment:
    """Cells per level with union-find-free explicit membership, an undo log for tentative merges, and closure propagation."""
    def __init__(self, solver):
        s = solver; self.s = s; self.T = s.T
        self.child = {t: [{u: s.index[t + 1][nd.children[u].hist] for u in nd.cont} for nd in s.levels[t]] for t in range(1, self.T)}
        self.child[self.T] = [dict() for _ in s.levels[self.T]]
        self.comp = {t: np.asarray(s.comp[t], bool) for t in range(1, self.T + 1)}
        self.O = {t: [str(o) for o in s.O[t]] for t in range(1, self.T + 1)}
        self.n = {t: len(s.levels[t]) for t in range(1, self.T + 1)}
        # initial cells = singletons, except level 1 where histories sharing o_1 must share a cell (single C_0)
        self.cell = {t: list(range(self.n[t])) for t in range(1, self.T + 1)}       # node -> cell id
        self.members = {t: {i: {i} for i in range(self.n[t])} for t in range(1, self.T + 1)}
        self.log = []
        byo = collections.defaultdict(list)
        for i, o in enumerate(self.O[1]): byo[o].append(i)
        for o, ii in byo.items():
            for j in ii[1:]:
                ok = self._merge(1, self.cell[1][ii[0]], self.cell[1][j]); assert ok, "level-1 forced merge infeasible"
        self.log = []

    # -- primitive merge of two cells at level t with closure propagation; returns False (after partial changes, to be rolled back) if infeasible
    def _merge(self, t, a, b):
        if a == b: return True
        A, B = self.members[t][a], self.members[t][b]
        ia, ib = np.fromiter(A, int), np.fromiter(B, int)
        if not self.comp[t][np.ix_(ia, ib)].all(): return False
        # apply: move B into A
        self.log.append((t, a, b, frozenset(B)))
        for j in B: self.cell[t][j] = a
        A |= B; del self.members[t][b]
        if t == self.T: return True
        # closure: for every continuation u, all children of members having u must share a cell at t+1
        byu = collections.defaultdict(set)
        for m in A:
            for u, c in self.child[t][m].items(): byu[u].add(self.cell[t + 1][c])
        for u, cells in byu.items():
            if len(cells) > 1:
                cells = sorted(cells); root = cells[0]
                for c in cells[1:]:
                    root = self.cell[t + 1][next(iter(self.members[t + 1][root]))]   # cell ids can change through nested merges
                    cc = self.cell[t + 1][next(iter(self.members[t + 1][c]))] if c in self.members[t + 1] else None
                    if cc is None or cc == root: continue
                    if not self._merge(t + 1, root, cc): return False
        return True

    def try_merge(self, t, a, b):
        mark = len(self.log)
        ok = self._merge(t, a, b)
        if not ok: self.rollback(mark)
        return ok

    def rollback(self, mark):
        while len(self.log) > mark:
            t, a, b, B = self.log.pop()
            for j in B: self.cell[t][j] = b
            self.members[t][a] -= B; self.members[t][b] = set(B)

    def cost(self, w):
        return [h_given_o(self.cell[t], self.O[t], w[t]) for t in range(1, self.T + 1)]

    def check(self):
        """Independent verification of compatibility and closure (same tests as ClosedPartitionDP.check_closed_compatible)."""
        for t in range(1, self.T + 1):
            P = np.asarray(self.cell[t])
            for c in set(P.tolist()):
                ii = np.nonzero(P == c)[0]
                assert self.comp[t][np.ix_(ii, ii)].all(), f"cell not compatible at t={t}"
                assert len({self.O[t][i] for i in ii}) == 1, f"cell spans two observations at t={t}"
            if t < self.T:
                for i in range(len(P)):
                    for j in range(i + 1, len(P)):
                        if P[i] == P[j]:
                            for u, ci in self.child[t][i].items():
                                if u in self.child[t][j]:
                                    assert self.cell[t + 1][ci] == self.cell[t + 1][self.child[t][j][u]], f"closure violated t={t}"
        return True


def greedy(solver, w, order="size", seed=0):
    asg = Assignment(solver); rng = random.Random(seed)
    for t in range(1, asg.T + 1):
        improved = True
        while improved:
            improved = False
            cells = list(asg.members[t].keys())
            byo = collections.defaultdict(list)
            for c in cells: byo[asg.O[t][next(iter(asg.members[t][c]))]].append(c)
            pairs = [(a, b) for cs in byo.values() for i, a in enumerate(cs) for b in cs[i + 1:]]
            if order == "size": pairs.sort(key=lambda ab: -(len(asg.members[t][ab[0]]) * len(asg.members[t][ab[1]])))
            else: rng.shuffle(pairs)
            for a, b in pairs:
                if a not in asg.members[t] or b not in asg.members[t]: continue
                if asg.try_merge(t, a, b): improved = True
            # one sweep over the (changing) pair list is a full pass; repeat until no merge succeeds
    return asg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", nargs="*", type=int, default=[1, 3, 5, 7])
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "REPORT_W4_upper_bound.json"))
    a = ap.parse_args()
    res = {}
    for W in a.W:
        env = grid_corridor(M=16, R1=2, R2=2, W=W, N=2, gap1=3, gap2=6)
        t0 = time.time(); solver = GammaSolver(env).solve(); T = solver.T
        w = {t: np.array([nd.prob for nd in solver.levels[t]]) for t in solver.levels}
        HG = [h_given_o(solver.G[t], solver.O[t], w[t]) for t in range(1, T + 1)]
        HS = [h_given_o(solver.GammaS[t], solver.O[t], w[t]) for t in range(1, T + 1)]
        trans = [bool(solver.transitive[t]) for t in range(1, T + 1)]
        HGam = [h_given_o(solver.Gamma[t], solver.O[t], w[t]) if solver.Gamma[t] is not None else float("nan") for t in range(1, T + 1)]
        sizes = [len(solver.levels[t]) for t in range(1, T + 1)]
        print(f"\n== corridor W={W}: T={T}, histories/level max {max(sizes)}, transitive at every t: {all(trans)}  (solver {time.time()-t0:.1f}s)")
        best, best_per, best_tag = float("inf"), None, None
        runs = [("size", 0)] + [("random", s) for s in range(1, a.restarts)]
        for order, seed in runs:
            t1 = time.time(); asg = greedy(solver, w, order, seed); per = asg.cost(w); tot = float(sum(per)); asg.check()
            print(f"   greedy order={order:6s} seed={seed:2d}: total {tot:.4f}  ({time.time()-t1:.1f}s)  per-step {np.round(per, 3).tolist()}")
            if tot < best - 1e-9: best, best_per, best_tag = tot, per, (order, seed)
        # phase labels from the env (hall steps) for the report
        phases = getattr(env, "phase_of_t", None)
        print(f"   BEST {best_tag}: total {best:.4f}")
        print("   t : H(G|O)  lower | certified upper (greedy) | H(Gamma^s|O) (paper's structural upper) | exact H(Gamma|O) if transitive")
        for t in range(1, T + 1):
            print(f"   {t:2d}: {HG[t-1]:6.3f} | {best_per[t-1]:6.3f} | {HS[t-1]:6.3f} | {'' if trans[t-1] else 'non-transitive'} {HGam[t-1] if trans[t-1] else ''}")
        res[W] = dict(T=T, sizes=sizes, transitive=trans, H_G_O=HG, H_GammaS_O=HS, H_Gamma_O=HGam, certified_upper=[float(x) for x in best_per],
                      total_upper=best, best_order=best_tag, restarts=len(runs))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
