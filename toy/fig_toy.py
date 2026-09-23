"""Appendix figure 'Toy validation' (one figure, two rows).  Top: (a) R_E(0) = log2 R is independent of M (H(U_beh) fixed at log2 M);
(b) rate-distortion for R = 4: closed form, Blahut-Arimoto on the reduced source and on the full history source (M = 4, 8, 16), and the
deterministic frontier.  Bottom (gap toy, unified config, 8 seeds): (c) reliability vs gap2 length for -R and -RF; (d) post-use rate in gap2 of
successful runs vs the theory; (e) the per-step conditional rate at gap2 = 6 against the staircase.
Usage: python toy/fig_toy.py   (reads toy/results/gap_formal.jsonl; recomputes the exact quantities via toy/exact_rd.py)"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "isaac"))
import figstyle; figstyle.appendix(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import log2
from exact_rd import reduced_source, blahut_arimoto, rd_curve, closed_form_RD, history_source, deterministic_frontier_G
betas = np.concatenate([np.linspace(0, 2, 21), np.linspace(2.2, 12, 50)])
fig = plt.figure(figsize=(figstyle.TEXTW, 4.1))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.6, wspace=2.1, left=0.08, right=0.99, top=0.95, bottom=0.17)
a1 = fig.add_subplot(gs[0, :2]); a2 = fig.add_subplot(gs[0, 2:5]); a3 = fig.add_subplot(gs[1, :2]); a4 = fig.add_subplot(gs[1, 2:4]); a5 = fig.add_subplot(gs[1, 4:])
# (a)
Rs = [1, 2, 4, 8]; xs = np.arange(4)
a1.bar(xs - 0.19, [log2(16)] * 4, width=0.36, color=P["gray"], label=r"$H(U_{\rm beh})$, $M=16$ (world)")
a1.bar(xs + 0.19, [log2(r) for r in Rs], width=0.36, color=P["ours"], label=r"$H(G_E\mid O)=R_E(0)$ (behavior)")
for x_, r in zip(xs, Rs): a1.text(x_ + 0.19, log2(r) + 0.1, f"{log2(r):.0f}", ha="center", va="bottom", fontsize=6.5, color=P["ours"])
a1.set_xticks(xs); a1.set_xticklabels([f"$R={r}$" for r in Rs]); a1.set_ylabel("bits"); a1.set_ylim(0, 6.6); a1.grid(axis="x", visible=False)
a1.set_title("(a) world vs. behavioral uncertainty", loc="left"); a1.legend(loc="upper left", fontsize=6.3, handlelength=1.2)
# (b)
R = 4; p, d = reduced_source(R); rates, Ds = rd_curve(p, d, betas); cf = closed_form_RD(R, Ds)
a2.plot(Ds, cf, color=P["black"], lw=1.4, label="closed form (Cor. 2)")
a2.plot(Ds, rates, "o", ms=2.6, color=P["ours"], mfc="white", mew=0.8, label="BA, reduced source $G$")
for M, ls in zip([4, 8, 16], ["--", "-.", ":"]):
    pH, dH, _ = history_source(M, 4, 4); rH, DH = rd_curve(pH, dH, betas); a2.plot(DH, rH, ls=ls, lw=1.0, color=P["alt"], label=f"BA, history source, $M={M}$")
det = deterministic_frontier_G(4); a2.plot(det[:, 1], det[:, 0], "s-", ms=3, lw=1.0, color=P["base"], label=r"$R^{\det}_{G\mid O}$ (deterministic frontier)")
a2.annotate("closed form and the four BA curves\ncoincide (Theorem 4)", xy=(0.30, 0.78), xytext=(0.30, 1.45), fontsize=6.5, ha="left", color="#333333", arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, shrinkB=2))
a2.set_xlabel("Hamming distortion $D$"); a2.set_ylabel("rate (bits)"); a2.set_xlim(-0.02, 0.78); a2.set_ylim(-0.05, 2.15); a2.set_title("(b) rate–distortion, $R=4$ (BA = Blahut–Arimoto)", loc="left"); a2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.6, handlelength=1.8, borderaxespad=0, labelspacing=0.6)
# gap toy
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "gap_formal.jsonl"))]
def pm(res, key, ph, w): a, b = ph[w]; return float(np.nanmean(res[key][a - 1:b]))
succ = lambda r: r["res"]["D_TV"][r["phases"]["use2"] - 1] < 0.05
gaps = sorted({r["job"]["gap2"] for r in rows})
sty = {"R": dict(color=P["ours"], marker="o", label="DIACRITIC$-$R"), "RF": dict(color=P["base"], marker="s", label="DIACRITIC$-$RF (BFS)")}
for var, lam in (("R", 0.0), ("RF", 1.0)):
    grp = {g: [r for r in rows if r["job"]["gap2"] == g and r["job"]["lam"] == lam] for g in gaps}
    a3.plot(gaps, [np.mean([succ(r) for r in grp[g]]) for g in gaps], ls="-", ms=3.5, **sty[var])
    ok = {g: [pm(r["res"], "H(C|O)", r["phases"], "gap2") for r in grp[g] if succ(r)] for g in gaps}
    a4.errorbar(gaps, [np.mean(ok[g]) for g in gaps], [np.std(ok[g]) for g in gaps], ls="-", ms=3.5, capsize=2, **sty[var])
    rs6 = [r for r in grp[6] if succ(r)]; hc = np.array([r["res"]["H(C|O)"] for r in rs6]); T = hc.shape[1]
    a5.errorbar(range(1, T + 1), hc.mean(0), hc.std(0), ls="-", ms=3, capsize=1.5, lw=1.0, **sty[var])
r0 = next(r for r in rows if r["job"]["gap2"] == 6); ph = r0["phases"]; T = r0["T"]
a3.set_xlabel("gap$_2$ length (steps)"); a3.set_ylabel("$P$(near-zero distortion at use$_2$)"); a3.set_ylim(-0.05, 1.05); a3.set_xticks(gaps)
a3.set_title("(c) reliability vs. memory gap", loc="left")
a4.axhline(pm(r0["res"], "H(Gam|O)", ph, "gap2"), color=P["black"], lw=1.2, label=r"$H(\Gamma_t\mid O_t)$")
a4.axhline(0.0, color=P["black"], ls=":", lw=1.2, label=r"$H(G_{E,t}\mid O_t)$")
a4.set_xlabel("gap$_2$ length (steps)"); a4.set_ylabel(r"$\hat H(C_t\mid O_t)$ in gap$_2$ (bits)"); a4.set_xticks(gaps); a4.set_ylim(-0.1, 2.3)
a4.set_title("(d) rate after use$_1$, successful runs", loc="left")
tt = np.arange(1, T + 1)
a5.step(tt, r0["res"]["H(H|O)"], where="mid", color=P["gray"], ls="--", lw=0.9, label=r"$H(H_t\mid O_t)$")
a5.step(tt, r0["res"]["H(G|O)"], where="mid", color=P["black"], ls=":", lw=1.2, label=r"$H(G_{E,t}\mid O_t)$")
a5.step(tt, r0["res"]["H(Gam|O)"], where="mid", color=P["black"], ls="-", lw=1.4, label=r"$H(\Gamma_t\mid O_t)$")
for name, key in (("gap$_1$", "gap1"), ("gap$_2$", "gap2")):
    a, b = ph[key]; figstyle.phase_span(a5, a, b, name, where=4.8)
a5.set_xlabel("$t$"); a5.set_ylabel("bits"); a5.set_xlim(0.5, T + 0.5); a5.set_xticks([1, 5, 10, T]); a5.set_title("(e) rate over time, gap$_2=6$", loc="left"); a5.set_ylim(-0.1, 7.0)
H, L = [], []
for a in (a3, a4, a5):
    for h_, l_ in zip(*a.get_legend_handles_labels()):
        if l_ not in L: H.append(h_); L.append(l_)
figstyle.legend_below(fig, H, L, ncol=5, y=0.0, fontsize=6.6, handlelength=1.7)
out = os.path.join(HERE, "..", "paper", "figs", "fig_toy.png"); figstyle.save(out)
