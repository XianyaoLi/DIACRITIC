"""Phase 0 for Isaac task A': symbolic-level acceptance with the exact Gamma solver (no Isaac dependency).

A' = identify -> grasp (beta_1, R1 classes) -> uninformative gap -> place (beta_2, R2 classes).
The symbolic model is `toy_gap` (crossed binning of a hidden mode theta in [M]; nuisance z in [N]).
For each tier |beta_1 v beta_2| in {4, 8, 16} and each gap length we check, per time step,
(A2) history determines the expert's action distribution, (A4) continuation-support homogeneity,
transitivity of the compatibility relation, and report the three theory lines of Fig 3:
H(G_t|O_t) <= H(Gamma_t|O_t) <= H(Gamma^s_t|O_t), plus the exact predictions
  gap1: H(Gamma|O) = log2(R1 R2),  H(G|O) = 0 ;   use1: H(G|O) = log2 R1 ;
  gap2: H(Gamma|O) = log2 R2,      H(G|O) = 0 ;   use2: H(G|O) = log2 R2 .
Any tier/gap that fails A2/A4/transitivity is NOT in the exact-Gamma regime and must not be used for Fig 3.
Run:  python a_prime_symbolic.py | tee results_a_prime_symbolic.log
"""
import sys, json
from math import log2
from toy_env import toy_gap
from gamma_solver import GammaSolver

TIERS = [(2, 2), (4, 2), (4, 4)]          # (R1, R2): join sizes 4, 8, 16
M = 32                                     # hidden modes; in-class (harmless) distinctions = M/(R1 R2) = 8, 4, 2
N = 4                                      # nuisance modes
GAPS = [1, 3, 6, 10, 20]                   # gap2 lengths (gap1 fixed = 2)
GAP1 = 2

def mean_over(xs, rng):
    a, b = rng
    return sum(xs[t - 1] for t in range(a, b + 1)) / (b - a + 1)

rows = []
print(f"M={M} N={N} gap1={GAP1}   (each cell: solver checks over ALL t; rates at gap1 / use1 / gap2 / use2)")
print(" join (R1,R2)  gap2 | A2  A4  trans | H(G|O)  gap1 use1 gap2 use2 | H(Gam|O) gap1 use1 gap2 use2 [pred gap1, gap2] | H(GamS|O) gap1 gap2 | H(H|O) gap1 | verdict")
for (R1, R2) in TIERS:
    for g2 in GAPS:
        env = toy_gap(M=M, R1=R1, R2=R2, N=N, gap1=GAP1, gap2=g2)
        s = GammaSolver(env).solve(); r = s.report(gammaJ=False, verbose=False)
        ph = env.phases; rt = r["rates"]
        ok = all(r["a2"]) and all(r["a4"]) and all(r["transitive"])
        HG = rt["H(G|O)"]; HGam = rt["H(Gamma|O)"]; HGS = rt["H(GammaS|O)"]; HH = rt["H(H|O)"]
        pred1, pred2 = log2(R1 * R2), log2(R2)
        exact = (abs(mean_over(HGam, ph["gap1"]) - pred1) < 1e-9 and abs(mean_over(HGam, ph["gap2"]) - pred2) < 1e-9
                 and abs(mean_over(HG, ph["gap1"])) < 1e-9 and abs(HG[ph["use1"] - 1] - log2(R1)) < 1e-9)
        verdict = "EXACT-Γ regime, predictions met" if ok and exact else ("checks ok but prediction mismatch" if ok else "NOT exact-Γ (A2/A4/transitivity failed)")
        print(f"  {R1*R2:2d}  ({R1},{R2})   {g2:3d} | {'ok' if all(r['a2']) else 'X '}  {'ok' if all(r['a4']) else 'X '}  {'yes' if all(r['transitive']) else 'NO '}  | "
              f"{mean_over(HG, ph['gap1']):5.2f} {HG[ph['use1']-1]:4.2f} {mean_over(HG, ph['gap2']):4.2f} {HG[ph['use2']-1]:4.2f} | "
              f"{mean_over(HGam, ph['gap1']):5.2f} {HGam[ph['use1']-1]:4.2f} {mean_over(HGam, ph['gap2']):4.2f} {HGam[ph['use2']-1]:4.2f} [{pred1:.2f}, {pred2:.2f}] | "
              f"{mean_over(HGS, ph['gap1']):5.2f} {mean_over(HGS, ph['gap2']):4.2f} | {mean_over(HH, ph['gap1']):5.2f} | {verdict}")
        rows.append(dict(R1=R1, R2=R2, join=R1*R2, gap2=g2, T=env.T, a2=all(r["a2"]), a4=all(r["a4"]), transitive=all(r["transitive"]),
                         exact=exact, H_G=HG, H_Gam=HGam, H_GamS=HGS, H_H=HH, phases=ph))
json.dump(rows, open("results/a_prime_symbolic.json", "w"))
print("saved results/a_prime_symbolic.json")
