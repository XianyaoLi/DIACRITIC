"""Class error at the FIRST divergent step of each block (the only step where the action must come from memory; later steps
can be copied from a_{t-1}).  Architecture-agnostic learning check.  Usage: python isaac/first_div_err.py <jsonl>... """
import sys, json, os, collections, numpy as np
g = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        a = r["args"]
        if a.get("aux_theta", 0) or a.get("aux_join", 0) or a.get("init_from") or a.get("refine"): continue
        ce = r["per_t"]["class_err_test"]; div = [t for t, c in enumerate(ce) if c == c]
        if not div: continue
        b1 = [t for t in div if t < 20]; b2 = [t for t in div if t >= 20]
        g[(os.path.basename(a["data"]), a.get("variant", "diacritic"), a["beta"], a.get("lam", 0), bool(a.get("pixels")))].append((ce[b1[0]] if b1 else np.nan, ce[b2[0]] if b2 else np.nan, r["summary"]["mse_test"]))
print(f"{'dataset':15s} {'variant':12s} {'beta':7s} lam px  n | first-b1-step err (per seed)            | first-b2-step err (per seed)                    | learned(b2 err<.1)")
for k in sorted(g):
    v = np.array(g[k]); print(f"{k[0]:15s} {k[1]:12s} {k[2]:<7} {k[3]:<3} {int(k[4])}  {len(v):2d} | {' '.join(f'{x:.2f}' for x in v[:,0]):40s} | {' '.join(f'{x:.2f}' for x in v[:,1]):48s} | {(v[:,1] < 0.1).sum()}/{len(v)}")
