"""Appendix figure: empirical rate-distortion on the robot (A', tier 4, gap 6; the 80 fig3 DIACRITIC policies, 128 closed-loop episodes each).
Left: post-use code rate vs closed-loop success for policies that reach sufficiency (many sit exactly at H(Gamma|O) = 1).
Right: closed-loop success of sufficient policies vs beta, -R and -RF, annotated with the mean post-use rate.
Usage: python isaac/fig_rd.py isaac/results/rd_closedloop_points.json"""
import sys, json, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle; figstyle.appendix(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pts = json.load(open(sys.argv[1]))
betas = sorted({p["beta"] for p in pts}); bcol = {0.0: "#c9d3e6", 0.0003: "#8fa6cf", 0.001: "#4f6fa8", 0.002: "#243f73"}
blab = {0.0: r"$\beta=0$", 0.0003: r"$\beta=3\times10^{-4}$", 0.001: r"$\beta=10^{-3}$", 0.002: r"$\beta=2\times10^{-3}$"}
fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXTW, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.3, left=0.08, right=0.99, top=0.9, bottom=0.34))
ax[0].axvline(1.0, color=P["black"], lw=1.0, ls="-", zorder=1); ax[0].text(1.03, 0.14, r"$H(\Gamma\mid\bar O)=1$", fontsize=6.5, ha="left", va="bottom", rotation=90, color="#333333")
for b in betas:
    for lam, m in ((0, "o"), (1, "s")):
        q = [p for p in pts if p["beta"] == b and p["lam"] == lam]
        if q: ax[0].scatter([p["rate"] for p in q], [p["succ"] for p in q], s=22, marker=m, color=bcol[b], edgecolor="white", linewidth=0.5, zorder=3)
for b in betas: ax[0].scatter([], [], s=22, color=bcol[b], label=blab[b])
ax[0].scatter([], [], s=22, marker="o", color="#666666", label="$-$R"); ax[0].scatter([], [], s=22, marker="s", color="#666666", label="$-$RF")
ax[0].set_xlabel(r"post-use code rate $\hat H(C_t\mid\bar O_t)$ in gap$_2$ (bits, own occupancy)"); ax[0].set_ylabel("closed-loop success"); ax[0].set_ylim(0.05, 1.05); ax[0].set_xlim(0.85, 2.4)
ax[0].set_title("(a) sufficient policies: rate vs. success", loc="left")
mean = {}
for lam, m, c, lab in ((0, "o", P["ours"], "$-$R"), (1, "s", P["base"], "$-$RF")):
    xs = [b for b in betas if any(p["beta"] == b and p["lam"] == lam for p in pts)]
    s = [[p["succ"] for p in pts if p["beta"] == b and p["lam"] == lam] for b in xs]; r = [[p["rate"] for p in pts if p["beta"] == b and p["lam"] == lam] for b in xs]
    xp = [max(b, 1e-4) for b in xs]
    ax[1].errorbar(xp, [np.mean(v) for v in s], [np.std(v) for v in s], marker=m, color=c, ms=4, lw=1.2, capsize=2, label=lab)
    for b, x_, v, rr in zip(xs, xp, s, r): mean[(lam, b)] = (x_, np.mean(v), np.mean(rr), c)
for (lam, b), (x_, y_, rr, c) in mean.items():
    other = mean.get((1 - lam, b)); above = other is None or y_ >= other[1]
    ax[1].annotate(f"{rr:.2f} bit", (x_, y_), xytext=(5, 6 if above else -11), textcoords="offset points", fontsize=5.8, color=c, ha="left")
ax[1].set_xscale("log"); ax[1].set_xticks([1e-4, 3e-4, 1e-3, 2e-3]); ax[1].set_xticklabels(["0", r"$3\!\times\!10^{-4}$", r"$10^{-3}$", r"$2\!\times\!10^{-3}$"]); ax[1].minorticks_off()
ax[1].set_xlabel(r"$\beta$"); ax[1].set_ylabel("success of sufficient policies"); ax[1].set_ylim(0.5, 1.06); ax[1].set_title("(b) distortion cost of rate pressure", loc="left"); ax[1].legend(loc="lower left", fontsize=6.6)
H, L = ax[0].get_legend_handles_labels(); figstyle.legend_below(fig, H, L, ncol=6, y=0.0, fontsize=6.6)
out = "paper/figs/fig_rd_closedloop.png"; figstyle.save(out)
