"""Group the pulled the cluster eval summaries (isaac/cluster_results/eval/*/summary.json) by model config (seed stripped) and print
closed-loop sufficiency (own occupancy, beta_2 in gap2), success | sufficient, success | not, and gap2 code rate | sufficient."""
import json, glob, re, collections, sys
import numpy as np
rows = [json.load(open(f)) for f in glob.glob('isaac/cluster_results/eval/*/summary.json')]
g = collections.defaultdict(list)
for r in rows: g[re.sub(r'_s\d+', '', r['model'])].append(r)
print(f"{'config (seed stripped)':66s}  n  CL-suff  succ|suff succ|not  gap2rate|suff  body(cm)")
for k, v in sorted(g.items()):
    ok = [r for r in v if r.get('cl_suff_b2_gap2', 0) > 0.9]; no = [r for r in v if r.get('cl_suff_b2_gap2', 0) <= 0.9]
    f = lambda xs, key: np.mean([x[key] for x in xs if key in x]) if xs else float('nan')
    print(f"{k:66s} {len(v):2d}  {len(ok)}/{len(v):<4d}  {f(ok,'success'):5.2f}     {f(no,'success'):5.2f}     {f(ok,'cl_HC_gap2'):5.2f}       {f(v,'body_sep_b2_gap2_cm'):4.2f}")
