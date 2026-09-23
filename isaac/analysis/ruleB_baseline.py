"""Audit-rule baseline (no training): false-alarm rate of audit rule B on CLEAN data.
Rule B = the policy is behaviourally correct at the memory-dependent decision (held-out class error at the first place step < 0.05, teacher-forced)
AND its code fails the full-trajectory sufficiency gate. This is an exploratory empirical signal, not a leakage certificate: the correctness threshold allows
nonzero error, and the full gate also checks the earlier grasp requirement. The rule was formulated on the injected-leak sweep (grid `leak`); here it is applied, unchanged, to every other
A'-family run of the project (no injected leak), which were all recorded before the rule existed.
Usage: python isaac/analysis/ruleB_baseline.py > results_md/analysis_ruleB_baseline.md"""
import os, json, glob, math
from collections import defaultdict
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster_results", "results")
def idx(r, n): return [i for i, q in enumerate(r["phases"]) if q == n]
def full(r):
    p = r["per_t"]; mem = [i for i in idx(r, "gap1") + idx(r, "gap2") if p["H(Gam|O)"][i] > 1e-9]
    act = [min(j) for j in ([i for i in idx(r, "grasp") if p["H(G|O)"][i] > 1e-9], [i for i in idx(r, "place") if p["H(G|O)"][i] > 1e-9]) if j]
    return bool(mem) and all(p["S_Gam"][i] > 0.9 for i in mem) and all(p["S_G"][i] > 0.9 for i in act)
def gap2_ok(r):      # the memory that must support the PLACE decision: S_Gamma > 0.9 at every second-gap step with a positive requirement
    p = r["per_t"]; return all(p["S_Gam"][i] > 0.9 for i in idx(r, "gap2") if p["H(Gam|O)"][i] > 1e-9)
fam = defaultdict(lambda: [0, 0, 0, 0]); per_grid = defaultdict(lambda: [0, 0, 0, 0])
for f in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
    g = os.path.basename(f)[:-6]
    if g.startswith(("smoke", "tf_", "rep_", "closedloop", "timing")) or g in ("leak", "readout", "readoutb", "readoutK", "weigh", "weigh2", "weigh3", "weighH8"): continue   # readout*: ledger predates the nested-visibility fix; weigh*: no place-step requirement
    seen = {}
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if "per_t" not in r or "gap2" not in r.get("phases", []) or "H(Gam|O)" not in r["per_t"] or "class_err_test" not in r["per_t"]: continue
        ds = os.path.basename(r["args"]["data"].rstrip("/"))
        if ds.startswith("taskA"): continue
        seen[json.dumps({k: v for k, v in r["args"].items() if k not in ("out", "save_codes", "save_model")}, sort_keys=True)] = r
    for r in seen.values():
        a = r["args"]; pl = idx(r, "place")
        if not pl or a.get("variant", "diacritic") in ("transformer", "bypass"): continue            # full-history variants: memory bypasses the code by construction (rule B fires by design)
        ce = r["per_t"]["class_err_test"][pl[0]]
        if ce is None or (isinstance(ce, float) and math.isnan(ce)): continue
        correct = ce < 0.05; suff = full(r); px = bool(a.get("pixels")); sup = bool(a.get("distill")) or (a.get("aux_join") or 0) > 0
        key = ("pixels" if px else "state") + ", " + ("forecast / privileged supervision" if sup else "unsupervised (-R, -RF, scaffold, controls)")
        for d in (fam[key], per_grid[g]): d[0] += 1; d[1] += int(correct); d[2] += int(correct and not suff); d[3] += int(correct and not gap2_ok(r))
print("# Audit rule B on clean data (no injected leak)\n\n" + __doc__.split("Usage:")[0].strip() + "\n")
print("Rule B' (matched form) pairs the PLACE decision with the memory that must support it: correct at the first place step AND S_Gamma <= 0.9 at some second-gap step.  Rule B compares the place decision with the WHOLE gate, so it also fires when a policy places correctly but lost the grasp class in the first gap -- a genuine insufficiency, not a side channel.\n")
print("| family | runs | behaviourally correct at the place decision | rule B fires (correct AND fails the full gate) | rule B' fires (correct AND second-gap S_Gamma fails) | false-alarm rate of B' among correct runs |\n|---|---|---|---|---|---|")
T = [0, 0, 0, 0]
for k in sorted(fam):
    n, c, b, b2 = fam[k]; T = [T[0] + n, T[1] + c, T[2] + b, T[3] + b2]; print(f"| {k} | {n} | {c} | {b} ({100 * b / max(c, 1):.1f}%) | {b2} | {100 * b2 / max(c, 1):.1f}% |")
print(f"| **all clean sole-carrier runs** | {T[0]} | {T[1]} | {T[2]} ({100 * T[2] / max(T[1], 1):.1f}%) | **{T[3]}** | **{100 * T[3] / max(T[1], 1):.1f}%** |")
L = {}
for l in open(os.path.join(RES, "leak.jsonl")):
    r = json.loads(l); L[(r["args"]["data"], r["args"]["seed"])] = r
by = defaultdict(lambda: [0, 0, 0])
for (ds, _), r in L.items():
    d = float(ds.split("_leak")[1]); c = r["per_t"]["class_err_test"][idx(r, "place")[0]] < 0.05; by[d][0] += int(c); by[d][1] += int(c and not full(r)); by[d][2] += int(c and not gap2_ok(r))
print("\nInjected-leak sweep under the same two rules (correct runs; B; B'): " + "; ".join(f"delta={d:g} mm: {v[0]}, {v[1]}, {v[2]}" for d, v in sorted(by.items())) + ".\n")
print("## Per grid (grids with at least one firing)\n\n| grid | runs | correct | rule B | rule B' |\n|---|---|---|---|---|")
for g in sorted(per_grid):
    if per_grid[g][2] or per_grid[g][3]: print(f"| {g} | {per_grid[g][0]} | {per_grid[g][1]} | {per_grid[g][2]} | {per_grid[g][3]} |")
