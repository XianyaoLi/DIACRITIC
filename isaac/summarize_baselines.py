"""Tabulate runs by (dataset, variant, beta, lam) with the teacher-forced metrics that are meaningful for *every* architecture
(imitation mse, place-phase class error at the memory-dependent divergent steps) next to the code metrics that are only meaningful
for sole-carrier variants (rate, S_Gamma).  Usage: python isaac/summarize_baselines.py <jsonl> [<jsonl> ...]"""
import sys, json, os, collections, numpy as np
g = collections.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        a = r["args"]; g[(os.path.basename(a["data"]), a.get("variant", "diacritic"), a["beta"], a.get("lam", 0), bool(a.get("pixels")))].append(r)
def gm(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.nanmean([r["per_t"][key][i] for i in idx]))
print(f"{'dataset':15s} {'variant':12s} {'beta':7s} lam px  n | suff(S_Gam2>.9&S_G>.9) | gap1 H(C|O) [theory] | gap2 H(C|O) [theory] | S_Gam gap2 | class_err_b2 (per seed) | mse_test")
for k in sorted(g):
    rs = g[k]; ok = [r for r in rs if r["summary"]["S_Gam_gap2"] > 0.9 and r["summary"]["S_G_place"] > 0.9]
    hc1 = [gm(r, "H(C|O)", "gap1") for r in rs]; hc2 = [gm(r, "H(C|O)", "gap2") for r in rs]
    th1 = gm(rs[0], "H(Gam|O)", "gap1"); th2 = gm(rs[0], "H(Gam|O)", "gap2")
    ce = [r["summary"].get("class_err_b2", float("nan")) for r in rs]; mse = [r["summary"]["mse_test"] for r in rs]
    print(f"{k[0]:15s} {k[1]:12s} {k[2]:<7} {k[3]:<3} {int(k[4])}  {len(rs):2d} |        {len(ok)}/{len(rs)}          |  {np.mean(hc1):4.2f}±{np.std(hc1):4.2f} [{th1:4.2f}]  |  {np.mean(hc2):4.2f}±{np.std(hc2):4.2f} [{th2:4.2f}]  |   {np.mean([r['summary']['S_Gam_gap2'] for r in rs]):4.2f}     | {' '.join(f'{c:.2f}' for c in ce)} | {np.mean(mse):.4f}")
