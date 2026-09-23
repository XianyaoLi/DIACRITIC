import json, numpy as np, os, sys
from collections import defaultdict
files = sys.argv[1:] or ["ablation"]
rows=[json.loads(l) for f in files for l in open(f"results/{f}.jsonl")]
g=defaultdict(list)
for r in rows:
    j=r["job"]; g[("enum" if j["n_train"]==0 else "fs", j["variant"], j["beta"])].append(r)
print("(regime, variant, beta): n  succ  S_G@use1  S_Gam@gap1  H(C|O)@gap1[2]  H(C|O)@gap2[1]  H(C|Gam,O)  I(C;z|O)  D@use2")
for k,rs in sorted(g.items()):
    ph=rs[0]["phases"]; g1=ph["gap1"][0]-1; u1=ph["use1"]-1
    f=lambda key,i: np.nanmean([r["res"][key][i] for r in rs])
    hc2=np.mean([np.mean(r["res"]["H(C|O)"][ph["gap2"][0]-1:ph["gap2"][1]]) for r in rs])
    d2=[r["res"]["D_TV"][ph["use2"]-1] for r in rs]; ok=sum(d<0.05 for d in d2)
    hcg=[r['res']['mean']['H(C|Gam,O)'] for r in rs]
    print(f"  {str(k):40s} {len(rs)}  {ok}/{len(rs)}  {f('S_G',u1):.2f}     {f('S_Gam',g1):.2f}       {f('H(C|O)',g1):.2f}           {hc2:.2f}          {np.nanmean(hcg) if not all(np.isnan(hcg)) else float('nan'):.2f}       {np.mean([r['res']['mean']['I(C;z|O)'] for r in rs]):.2f}     {np.mean(d2):.3f}")
