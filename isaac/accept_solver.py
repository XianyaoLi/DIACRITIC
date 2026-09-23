"""Acceptance (a): run the exact Gamma solver on the *recorded* symbolic observations/actions of an A' dataset.

Symbolic observations are augmented with the classes that are decodable from the raw observation at each step
(see aprime_data.visibility); pass --raw_sym to use the bare (phase, probe) symbols instead.
Usage:  python isaac/accept_solver.py isaac/data/tier4_gap6 [--raw_sym]
"""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from aprime_data import prepare


def main(d, augment=True, sag=False):
    eps, meta, vis, env, s, key = prepare(d, augment=augment, sag_symbol=sag)
    R1, R2, M, N = meta["R1"], meta["R2"], meta["M"], meta["N"]
    T = len(eps[0]["phase"]); phases = eps[0]["phase"]
    print(f"{len(eps)} episodes, T={T}, R1={R1} R2={R2} M={M} N={N}; expert success {np.mean([e['success'] for e in eps]):.3f}; symbols: {key}")
    print("visible b1 at t:", [t + 1 for t in range(T) if vis["b1"][t]], " visible b2 at t:", [t + 1 for t in range(T) if vis["b2"][t]])
    r = s.report(gammaJ=False, verbose=False); rt = r["rates"]
    print("\n t  phase    A2  A4  trans | H(G|O)  H(Gam|O)  H(GamS|O)  H(H|O)")
    for t in range(1, T + 1):
        print(f"{t:2d}  {phases[t-1]:8s} {'ok' if r['a2'][t-1] else 'X '}  {'ok' if r['a4'][t-1] else 'X '}  {'yes' if r['transitive'][t-1] else 'NO '}  | "
              f"{rt['H(G|O)'][t-1]:5.2f}    {rt['H(Gamma|O)'][t-1]:5.2f}      {rt['H(GammaS|O)'][t-1]:5.2f}     {rt['H(H|O)'][t-1]:5.2f}")
    ok = all(r["a2"]) and all(r["a4"]) and all(r["transitive"])
    def pm(k, ph):
        idx = [i for i, p in enumerate(phases) if p == ph]; return float(np.mean([rt[k][i] for i in idx]))
    pred1, pred2 = math.log2(R1 * R2), math.log2(R2)
    g1, g2, G1, G2 = pm("H(Gamma|O)", "gap1"), pm("H(Gamma|O)", "gap2"), pm("H(G|O)", "gap1"), pm("H(G|O)", "gap2")
    exact = abs(g1 - pred1) < 1e-6 and abs(g2 - pred2) < 1e-6 and abs(G1) < 1e-6 and abs(G2) < 1e-6
    print(f"\nchecks: A2 {all(r['a2'])}  A4 {all(r['a4'])}  transitive {all(r['transitive'])}")
    print(f"gap1: H(Gam|O)={g1:.3f} (pred {pred1:.3f}), H(G|O)={G1:.3f} | gap2: H(Gam|O)={g2:.3f} (pred {pred2:.3f}), H(G|O)={G2:.3f}")
    print("VERDICT:", ("EXACT-Gamma regime on recorded data" + (", predictions met" if exact else " (A′-layout predictions not applicable)")) if ok else (("GENERAL regime (A4 fails at t=" + ",".join(str(t + 1) for t, a in enumerate(r["a4"]) if not a) + "): report the sandwich [H(G|O), H(GammaS|O)]") if all(r["a2"]) and all(r["transitive"]) else "FAIL"))
    json.dump(dict(dataset=d, symbols=key, visible=vis, T=T, phases=phases, rates=rt, a2=r["a2"], a4=r["a4"], transitive=r["transitive"], ok=ok, exact=exact,
                   gap1=dict(HGam=g1, pred=pred1, HG=G1), gap2=dict(HGam=g2, pred=pred2, HG=G2)), open(os.path.join(d, "accept_solver.json"), "w"), indent=1)
    return ok and exact


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], augment="--raw_sym" not in sys.argv, sag="--sag" in sys.argv) else 1)
