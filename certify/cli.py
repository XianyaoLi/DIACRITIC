"""Command-line entry point.

    python -m certify --task memchain --L 30 --bits 2 [--mikasa PATH] [--out FILE.jsonl] [--quiet]
    python -m certify --task tmaze --L 50
    python -m certify --paper [--out FILE.jsonl]        # the 13 community-benchmark certifications reported in the paper,
                                                        # plus two finite instances that exercise the non-transitive branch
    python -m certify --finite remark_i | rereveal      # one of those finite instances
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time


from .adapters import TASKS
from .core import certify, certify_env
from .finite import FINITE

PAPER = [("memchain", dict(bits=b, L=L)) for b in (1, 2, 3) for L in (10, 30, 100)] + [("tmaze", dict(L=L)) for L in (10, 20, 50, 100)]


def expected_profile(task: str, cfg: dict):
    """Requirement profiles derived by hand for the paper (Appendix E.5); used only by --paper as a self-check."""
    if task == "memchain":          # context visible for two steps, b bits until the query, one bit at the query step
        b, L = cfg["bits"], cfg["L"]
        return [0.0, 0.0] + [float(b)] * (L - 2) + [1.0]
    if task == "tmaze":             # goal visible in the first observation, one bit along the corridor through the decision
        return [0.0] + [1.0] * cfg["L"]
    return None


def finite_checks():
    """The two finite instances of the paper that exercise what the benchmarks do not: a non-transitive step and a
    strict gap between Gamma and the strong congruence. Returns [(name, passed, summary)]."""
    out = []
    c = certify_env(FINITE["remark_i"]())
    ok = (not c.transitive[1]) and (not c.a4[1]) and math.isnan(c.H_Gamma[1]) and abs(c.H_G[1]) < 1e-9 and abs(c.H_GammaS[1] - math.log2(3)) < 1e-9 and c.transitive[0] and c.transitive[2]
    out.append((c.name, ok, "t=2: A4 fails, not transitive, Gamma not constructed, bracket [0.000, 1.585]" if ok else c.table()))
    e = FINITE["rereveal"](); c = certify_env(e); ph = e.phases
    ok = c.exact and all(abs(c.H_Gamma[t - 1] - 1) < 1e-9 and abs(c.H_GammaS[t - 1] - 2) < 1e-9 for t in range(ph["gap1"][0], ph["use1"] + 1)) \
        and all(abs(c.H_Gamma[t - 1]) < 1e-9 for t in range(ph["gap2"][0], ph["gap2"][1])) \
        and [not a for a in c.a4] == [t == ph["use2"] - 2 for t in range(1, e.T + 1)]
    out.append((c.name, ok, "exact; Gamma 1 bit vs strong congruence 2 bits through the first gap and grasp, 0 in the second gap; A4 fails only at use2-2" if ok else c.table()))
    return out


def run_one(task: str, cfg: dict, mikasa: str | None, out: str | None, quiet: bool):
    t0 = time.time()
    cert = certify(TASKS[task](mikasa=mikasa, **cfg))
    d = dict(kind="cert", task=task, **cfg, secs=round(time.time() - t0, 2), **cert.to_dict())
    if not quiet:
        print(cert.table())
        print(f"({d['secs']} s)\n")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(d) + "\n")
    return cert


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m certify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), help="benchmark adapter")
    ap.add_argument("--L", type=int, default=10, help="corridor length (tmaze), memory length (memchain) or delay (example)")
    ap.add_argument("--bits", type=int, default=1, help="context bits (memchain)")
    ap.add_argument("--mikasa", default=None, help="path to an unmodified MIKASA-Base checkout (default: $MIKASA_BASE or ../external/MIKASA-Base)")
    ap.add_argument("--out", default=None, help="append one JSON line per certificate to this file")
    ap.add_argument("--paper", action="store_true", help="run every community-benchmark certification of the paper and self-check the memory-chain profiles")
    ap.add_argument("--finite", choices=sorted(FINITE), help="solve one of the paper's finite instances (no external environment needed)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.finite:
        print(certify_env(FINITE[a.finite]()).table())
        return
    if a.paper:
        ok = True
        rows = []
        for task, cfg in PAPER:
            cert = run_one(task, cfg, a.mikasa, a.out, quiet=True)
            exp = expected_profile(task, cfg)
            match = exp is None or (cert.exact and len(cert.H_Gamma) == len(exp) and all(abs(x - y) < 1e-6 for x, y in zip(cert.H_Gamma, exp)))
            ok &= match and cert.exact and all(cert.a2)
            need = [t for t, h in enumerate(cert.H_Gamma) if h > 1e-9]
            rows.append((cert.name, cert.T, cert.hidden_values, "yes" if cert.exact else "NO", "ok" if all(cert.a2) else "X",
                         "ok" if all(cert.a4) else f"fails@{[t + 1 for t, v in enumerate(cert.a4) if not v]}",
                         f"{min(cert.H_Gamma[t] for t in need):.2f}-{max(cert.H_Gamma[t] for t in need):.2f} over steps {need[0] + 1}-{need[-1] + 1}",
                         "matches hand derivation" if match else "MISMATCH"))
        w = [max(len(str(r[i])) for r in rows + [("benchmark", "T", "hidden", "exact", "A2", "A4", "H(Gamma|O) where positive", "check")]) for i in range(8)]
        hdr = ("benchmark", "T", "hidden", "exact", "A2", "A4", "H(Gamma|O) where positive", "check")
        print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
        for r in rows:
            print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))
        print("\nfinite instances (non-transitive branch and strict sandwich):")
        for name, passed, summary in finite_checks():
            ok &= passed
            print(f"  {name:24s} {'PASS' if passed else 'FAIL'}  {summary}")
        print("\nALL PAPER CERTIFICATIONS REPRODUCED" if ok else "\nSOME CERTIFICATIONS DIFFER FROM THE PAPER", file=sys.stderr)
        sys.exit(0 if ok else 1)
    if not a.task:
        ap.error("--task, --finite or --paper is required")
    cfg = dict(L=a.L, bits=a.bits) if a.task == "memchain" else dict(delay=a.L) if a.task == "example" else dict(L=a.L)
    run_one(a.task, cfg, a.mikasa, a.out, a.quiet)


if __name__ == "__main__":
    main()
