"""Task A (general regime, split latent): per (variant, beta, M): sufficiency at the place step (S_G place > 0.9), class error,
H(C|Ō) averaged over the transport steps BEFORE the late reveal (should be ~0 for a minimal memory), I(C; mass|Ō) and I(C; theta|Ō)
over gap2, and the sandwich bounds from the solver (H(G|Ō), H(Γ|Ō), H(Γˢ|Ō)).  Usage: python isaac/summarize_taskA.py <jsonl...>"""
import sys, json, os, re, collections
import numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = collections.defaultdict(list)
for r in rows:
    a = r["args"]; M = int(re.search(r"M(\d+)", os.path.basename(a["data"])).group(1))
    tag = ("sysid" if a.get("aux_theta", 0) else ("mik" + ("-only" if a.get("mik_detach", 0) else "+imit") if a.get("aux_mik", 0) else a.get("variant", "diacritic"))) + ("-RF" if a.get("lam", 0) > 0 else "-R")
    if a.get("K", 16) != 16: tag += f"-K{a['K']}"
    g[(tag, a["beta"], M)].append(r)
def phase_mean(r, key, ph, upto=None):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; idx = idx[:upto] if upto else idx
    return float(np.nanmean([r["per_t"][key][i] for i in idx]))
print(f"{'variant':14s} {'beta':>6s} {'M':>3s} n | suff(place)  cls_err | H(C|O)@grasp [Gam, GamS]  H(C|O)@gap2-pre [Gam,GamS]  @place [Gam] | I(C;mass|O)@grasp @gap2 | H(C|Gam,O)")
for (tag, beta, M), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    ok = [r for r in rs if r["summary"]["S_G_place"] > 0.9]
    pre = lambda r, k: phase_mean(r, k, "gap2", upto=len([i for i, p in enumerate(r["phases"]) if p == "gap2"]) - 2)
    th = rs[0].get("theory", {})
    hg_pre = np.mean([th["H(Gamma|O)"][i] for i, p in enumerate(rs[0]["phases"]) if p == "gap2"][:-2]) if th else float("nan")
    hgs_pre = np.mean([th["H(GammaS|O)"][i] for i, p in enumerate(rs[0]["phases"]) if p == "gap2"][:-2]) if th else float("nan")
    hg_place = th["H(Gamma|O)"][rs[0]["phases"].index("place")] if th else float("nan")
    f = lambda fn: np.mean([fn(r) for r in rs])
    gi = [i for i, p in enumerate(rs[0]["phases"]) if p == "grasp"]
    hg_gr = np.mean([th["H(Gamma|O)"][i] for i in gi]) if th else float("nan"); hgs_gr = np.mean([th["H(GammaS|O)"][i] for i in gi]) if th else float("nan")
    print(f"{tag:14s} {beta:6.4f} {M:3d} {len(rs)} |   {len(ok)}/{len(rs)}      {f(lambda r: r['summary']['class_err_test']):.3f} |   {f(lambda r: phase_mean(r, 'H(C|O)', 'grasp')):.2f}    [{hg_gr:.2f}, {hgs_gr:.2f}]      "
          f"{f(lambda r: pre(r, 'H(C|O)')):.2f}    [{hg_pre:.2f}, {hgs_pre:.2f}]    {f(lambda r: r['per_t']['H(C|O)'][r['phases'].index('place')]):.2f}  [{hg_place:.2f}] |   {f(lambda r: phase_mean(r, 'I(C;mass|O)', 'grasp')):.3f}   {f(lambda r: phase_mean(r, 'I(C;mass|O)', 'gap2')):.3f}  |  {f(lambda r: r['summary']['HC_Gam_O']):.2f}")
