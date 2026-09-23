"""Appendix figure P-I on A': learned gap-1 rate vs log2 M with the full theta revealed (8 seeds per cell).  Successful runs (solid) stay at
H(Gamma|O) = 2, the mean over all seeds (hollow, dotted) is flat, and the system-identification rate grows with M toward the store-everything line.
Usage: python isaac/fig_pI.py isaac/results/pI_all.jsonl"""
import sys, json, os, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle; figstyle.appendix(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows:
    a = r["args"]; M = int(os.path.basename(a["data"]).split("M")[-1])
    tag = "sys-ID" if a.get("aux_theta", 0) else ("scaffold-RF" if a.get("variant") == "scaffold" else ("DIACRITIC$-$RF" if a.get("lam", 0) > 0 else "DIACRITIC$-$R"))
    if tag != "sys-ID" and a["beta"] == 0: continue
    g[(tag, a["beta"], M)].append(r)
def gm(r, key, ph="gap1"):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.mean([r["per_t"][key][i] for i in idx]))
def ok(r):   # full-trajectory gate (Section 3): S_Gamma > 0.9 at every step of both gaps with a positive requirement, S_G > 0.9 at the first grasp / place step
    p, ph = r["per_t"], r["phases"]; idx = lambda n: [i for i, q in enumerate(ph) if q == n]
    mem = [i for i in idx("gap1") + idx("gap2") if p["H(Gam|O)"][i] > 1e-9]; act = [min(j) for j in ([i for i in idx("grasp") if p["H(G|O)"][i] > 1e-9], [i for i in idx("place") if p["H(G|O)"][i] > 1e-9]) if j]
    return all(p["S_Gam"][i] > 0.9 for i in mem) and all(p["S_G"][i] > 0.9 for i in act)
fig, ax = plt.subplots(figsize=(figstyle.TEXTW, 2.35), gridspec_kw=dict(left=0.09, right=0.62, top=0.93, bottom=0.2))
Ms = sorted({k[2] for k in g}); x = np.log2(Ms); r0 = g[next(k for k in g if k[2] == Ms[0])][0]
ax.plot(x, [gm(g[next(k for k in g if k[2] == M)][0], "H(H|O)") for M in Ms], color=P["gray"], ls="--", lw=1.0, label=r"$H(H_t\mid\bar O_t)$ (store everything)")
ax.plot(x, [gm(g[next(k for k in g if k[2] == M)][0], "H(Gam|O)") for M in Ms], color=P["black"], lw=1.5, label=r"$H(\Gamma_t\mid\bar O_t)$ (theory)")
col = {"DIACRITIC$-$R": P["ours"], "DIACRITIC$-$RF": P["ours2"], "scaffold-RF": P["purple"], "sys-ID": P["base"]}; mk = {"DIACRITIC$-$R": "o", "DIACRITIC$-$RF": "s", "scaffold-RF": "^", "sys-ID": "D"}
for tag in ["DIACRITIC$-$R", "DIACRITIC$-$RF", "scaffold-RF", "sys-ID"]:
    keys = sorted({k for k in g if k[0] == tag}, key=lambda k: k[2]); beta = keys[0][1]
    ys = [np.mean([gm(r, "H(C|O)") for r in g[k] if ok(r)]) if any(ok(r) for r in g[k]) else np.nan for k in keys]
    ya = [np.mean([gm(r, "H(C|O)") for r in g[k]]) for k in keys]; ns = [sum(ok(r) for r in g[k]) for k in keys]
    b = f"$\\beta={beta:g}$".replace("0.001", "10^{-3}").replace("0.002", "2\\times10^{-3}")
    ax.plot([math.log2(k[2]) for k in keys], ys, marker=mk[tag], ms=4, color=col[tag], lw=1.3, label=f"{tag}, {b}: successful runs")
    ax.plot([math.log2(k[2]) for k in keys], ya, marker=mk[tag], ms=4, mfc="white", color=col[tag], ls=":", lw=0.9, label=f"{tag}: all seeds")
    print(tag, beta, "succ/8:", ns, "rate succ:", np.round(ys, 2), "rate all:", np.round(ya, 2))
ax.set_xticks(x); ax.set_xticklabels([f"{M}" for M in Ms]); ax.set_xlabel(r"$M$ (hidden modes revealed at identification)"); ax.set_ylabel("bits in gap$_1$"); ax.set_ylim(0.5, 6.3)
ax.set_title("P-I on A$'$: behavioral, not world, complexity sets the memory rate", loc="left")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.6, handlelength=2.0, labelspacing=0.45)
out = "paper/figs/pI_all.png"; figstyle.save(out)
