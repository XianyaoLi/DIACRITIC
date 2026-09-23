"""N5: solver-side coverage of the recorded datasets.

For each dataset used in the paper: E = number of recorded episodes, the number of distinct latent keys seen
(theta, or (theta, g=b2) for split-latent task-A datasets) vs the number of possible keys (M, or M*R2), the number of
keys seen exactly once (n1) and the Good-Turing missing-mass estimate n1/E, the number of unseen keys (which
build_env() in aprime_data.py fills by copying the symbolic obs/act sequence of a seen key with the same
(b1,b2) = (theta % R1, (theta // R1) % R2), or with the same g for split datasets), and the number of distinct symbolic
histories per solver level (len(solver.index[t]); max over t and the full per-t profile).

Usage: python isaac/analysis/n5_coverage.py   (prints markdown)
"""
import os, sys, time, collections, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from aprime_data import prepare

DATASETS = ["tier4_gap6", "tier4_gap6_2k", "tier4_gap6_4k", "tier4_gap10", "tier4_gap20", "tier16_gap6",
            "pI_M4", "pI_M8", "pI_M16", "pI_M32", "taskA2_M4", "taskA2_M8", "taskA2_M16", "taskA2_M32"]

print("| dataset | E | M,R1,R2,N | key | keys seen / possible | unseen (copied) | n1 (seen once) | Good-Turing n1/E | min/max episodes per seen key | T | max #hist per level | #hist per level t=1..T |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
profiles = {}
for name in DATASETS:
    d = os.path.join(HERE, "..", "data", name)
    if not os.path.isdir(d):
        print(f"| {name} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | directory missing |"); continue
    sag = name.startswith("taskA2")
    try:
        t0 = time.time()
        eps, meta, vis, env, solver, key = prepare(d, sag_symbol=sag)
        secs = time.time() - t0
    except Exception as e:
        print(f"| {name} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | FAILED to load: {type(e).__name__}: {str(e)[:120]} |")
        traceback.print_exc(file=sys.stderr); continue
    M, R1, R2, N = meta["M"], meta["R1"], meta["R2"], meta["N"]
    split = len({(e["theta"], e["b2"]) for e in eps}) > len({e["theta"] for e in eps})   # same rule as build_env
    keyf = (lambda e: (e["theta"], e["b2"])) if split else (lambda e: e["theta"])
    cnt = collections.Counter(keyf(e) for e in eps)
    E = len(eps); possible = M * R2 if split else M
    n1 = sum(1 for v in cnt.values() if v == 1)
    unseen = possible - len(cnt)
    hist = {t: len(solver.index[t]) for t in sorted(solver.index)}
    prof = ",".join(str(hist[t]) for t in sorted(hist))
    print(f"| {name} | {E} | {M},{R1},{R2},{N} | {'(theta,g)' if split else 'theta'} | {len(cnt)} / {possible} | {unseen} | {n1} | {n1/E:.4f} | {min(cnt.values())}/{max(cnt.values())} | {env.T} | {max(hist.values())} | {prof} |")
    sys.stderr.write(f"{name}: {secs:.2f}s\n")
