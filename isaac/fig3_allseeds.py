"""Fig 3 (main text): A' exact regime, the three-level staircase and the learned rate of every seed (tier 4, gap2 = 6, N = 1152).
(a) plain -R, beta = 0; (b) plain -R, beta = 1e-3.  Sufficient seeds blue, insufficient red; theory H(Gamma|O), H(G|O), H(H|O).
Usage: python isaac/fig3_allseeds.py isaac/cluster_results/results/fig3.jsonl"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle; figstyle.main(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
def pick(beta): return [r for r in rows if r["args"].get("variant", "diacritic") == "diacritic" and r["args"].get("lam", 0) == 0 and abs(r["args"]["beta"] - beta) < 1e-9]
def suff(r):   # full-trajectory gate: S_Gamma > 0.9 at every step of both gaps with a positive requirement, S_G > 0.9 at the first grasp and first place step
    p, ph = r["per_t"], r["phases"]; idx = lambda n: [i for i, q in enumerate(ph) if q == n]
    mem = [i for i in idx("gap1") + idx("gap2") if p["H(Gam|O)"][i] > 1e-9]; act = [min(j) for j in ([i for i in idx("grasp") if p["H(G|O)"][i] > 1e-9], [i for i in idx("place") if p["H(G|O)"][i] > 1e-9]) if j]
    return all(p["S_Gam"][i] > 0.9 for i in mem) and all(p["S_G"][i] > 0.9 for i in act)
fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXTW, 1.2), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.07, right=0.995, top=0.89, bottom=0.33))
for a, beta, lab in zip(ax, [0.0, 1e-3], ["(a) plain $-$R, $\\beta=0$", "(b) plain $-$R, $\\beta=10^{-3}$"]):
    rs = pick(beta); r0 = rs[0]; ph = r0["phases"]; T = len(ph); t = np.arange(T)
    for name in ("gap1", "gap2"):
        i0 = ph.index(name); i1 = T - 1 - ph[::-1].index(name); figstyle.phase_span(a, i0, i1, name.replace("gap", "gap$_") + "$", where="top")
    a.plot(t, r0["theory"]["H(H|O)"], color=P["gray"], ls="--", lw=0.9, label=r"$H(H_t\mid\bar O_t)$ (store everything)")
    a.plot(t, r0["theory"]["H(G|O)"], color=P["black"], ls=":", lw=1.2, label=r"$H(G_{E,t}\mid\bar O_t)$")
    a.plot(t, r0["theory"]["H(Gamma|O)"], color=P["black"], ls="-", lw=1.5, label=r"$H(\Gamma_t\mid\bar O_t)$")
    ns = sum(suff(r) for r in rs)
    for r in rs:
        a.plot(t, r["per_t"]["H(C|O)"], color=P["ours"] if suff(r) else P["base"], lw=0.8, alpha=0.75, zorder=3)
    a.plot([], [], color=P["ours"], lw=1.0, label="sufficient seed"); a.plot([], [], color=P["base"], lw=1.0, label="insufficient seed")
    a.set_title(f"{lab}: {ns}/{len(rs)} sufficient", loc="left"); a.set_xlabel("$t$ (macro step)"); a.set_xlim(-0.5, T - 0.5); a.set_xticks(range(0, T, 5))
ax[0].set_ylabel(r"$\hat H(C_t\mid\bar O_t)$ (bits)"); ax[0].set_ylim(-0.1, 4.5); ax[0].set_yticks([0, 1, 2, 3, 4])
H, L = ax[0].get_legend_handles_labels()
figstyle.legend_below(fig, H, L, ncol=5, y=-0.03, handlelength=1.8, columnspacing=1.1)
out = "paper/figs/fig3_allseeds.png"; figstyle.save(out)
