"""Fig 1 (two stacked full-width panels, printed 5.5 x 1.10 in): (a) the concept chain S_t -> G_{E,t} -> Gamma_t -> C_t;
(b) the A' episode timeline with the H(Gamma|O) and H(G_E|O) staircases (2 -> 1 -> 0).  Usage: python isaac/fig1_concept.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle; figstyle.apply(8.0); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
W, H = figstyle.TEXTW, 1.10
fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
# ---------------- panel (a): concept chain ----------------
ax = fig.add_axes([0, 0.50, 1, 0.50]); ax.set_xlim(0, W); ax.set_ylim(0, 0.55); ax.axis("off")
boxes = [("world state $S_t$", "hidden $\\theta$: $\\log_2 M$ bits", "#e9e9e9", "#8a8a8a"),
         ("behavioral quotient $G_{E,t}$", "$R_E(0)=H(G_{E,t}\\mid O_t)$", "#dbe5f3", P["ours"]),
         ("behavioral memory $\\Gamma_t$", "$R^{\\rm mem}_t(0)=H(\\Gamma_t\\mid O_t)$", "#b9cbe6", P["ours"]),
         ("learned code $C_t$", "sole carrier across time", "#f7dfcb", P["base2"])]
arrows = ["forget across states", "keep for later", "learn"]
bw, bh, gap, x0, y0 = 1.14, 0.34, 0.30, 0.14, 0.03
ax.text(0.02, 0.53, "(a)", fontsize=8, fontweight="bold", ha="left", va="top")
for i, (title, body, fc, ec) in enumerate(boxes):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec=ec, lw=0.8))
    ax.text(x + bw / 2, y0 + bh * 0.68, title, ha="center", va="center", fontsize=7.6, fontweight="bold")
    ax.text(x + bw / 2, y0 + bh * 0.27, body, ha="center", va="center", fontsize=7)
    if i < 3:
        xa, xb = x + bw + 0.02, x + bw + gap - 0.02
        ax.add_patch(FancyArrowPatch((xa, y0 + bh / 2), (xb, y0 + bh / 2), arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#333333"))
        ax.text((xa + xb) / 2, y0 + bh + 0.03, arrows[i], ha="center", va="bottom", fontsize=7, color="#333333")
# ---------------- panel (b): A' timeline ----------------
ax = fig.add_axes([0, 0, 1, 0.50]); ax.set_xlim(0, W); ax.set_ylim(0, 0.55); ax.axis("off")
ax.text(0.02, 0.53, "(b)", fontsize=8, fontweight="bold", ha="left", va="top")
segs = [("identify ($\\beta_1,\\beta_2$)", 4, "#dbe5f3"), ("gap$_1$", 2, "#c9d6ea"), ("grasp: use $\\beta_1$", 2, "#f7dfcb"), ("gap$_2$", 6, "#dbe5f3"), ("place: use $\\beta_2$", 2, "#f7dfcb"), ("done", 1, "#e9e9e9")]
tx0, tx1, ty, th = 1.02, W - 0.06, 0.30, 0.13; tot = sum(s[1] for s in segs); u = (tx1 - tx0) / tot
x = tx0; edges = [x]
for lab, w_, fc in segs:
    ax.add_patch(FancyBboxPatch((x, ty), w_ * u, th, boxstyle="square,pad=0", fc=fc, ec="#666666", lw=0.6))
    ax.text(x + w_ * u / 2, ty + th / 2, lab, ha="center", va="center", fontsize=7.5); x += w_ * u; edges.append(x)
for xe, lab in ((edges[1], "reveal"), (edges[2], "$\\beta_1$ used"), (edges[4], "$\\beta_2$ used")):
    ax.plot([xe, xe], [ty - 0.015, ty + th + 0.015], color="#222222", lw=0.8); ax.text(xe, ty + th + 0.025, lab, ha="center", va="bottom", fontsize=7, color="#222222")
# staircases below the bar; y scale 0..2 bits
y_of = lambda v: 0.035 + 0.095 * v
gam = [(0, 2), (2, 2), (2, 2), (1, 1), (1, 1), (0, 0)]; ge = [0, 0, 1, 0, 1, 0]
xs, ys = [], []
for i, (a, b) in enumerate(gam): xs += [edges[i], edges[i + 1]]; ys += [y_of(a), y_of(b)]
ax.plot(xs, ys, color=P["ours"], lw=1.5, solid_capstyle="round")
ax.plot([(edges[i] + edges[i + 1]) / 2 for i in range(1, 6)], [y_of(gam[i][1]) for i in range(1, 6)], "o", color=P["ours"], ms=3.2)
xs, ys = [], []
for i, v in enumerate(ge): xs += [edges[i], edges[i + 1]]; ys += [y_of(v) - 0.012, y_of(v) - 0.012]
ax.plot(xs, ys, color=P["base2"], lw=1.2, ls="--", dashes=(3, 2))
ax.plot([(edges[i] + edges[i + 1]) / 2 for i in range(6)], [y_of(v) - 0.012 for v in ge], "s", color=P["base2"], ms=3.2, mfc="white")
# small bit axis at the left of the staircase
ax.plot([tx0 - 0.04, tx0 - 0.04], [y_of(0), y_of(2)], color="#444444", lw=0.6)
for v in (0, 1, 2):
    ax.plot([tx0 - 0.06, tx0 - 0.04], [y_of(v), y_of(v)], color="#444444", lw=0.6); ax.text(tx0 - 0.08, y_of(v), f"{v}", ha="right", va="center", fontsize=7, color="#444444")
ax.text(tx0 - 0.04, y_of(2) + 0.03, "bits", ha="center", va="bottom", fontsize=7, color="#444444")
# legend at the far left
ax.plot([0.08, 0.28], [y_of(1.55), y_of(1.55)], color=P["ours"], lw=1.5); ax.plot([0.18], [y_of(1.55)], "o", color=P["ours"], ms=3.2); ax.text(0.33, y_of(1.55), "$H(\\Gamma_t\\mid\\bar O_t)$", ha="left", va="center", fontsize=7.5, color=P["ours"])
ax.plot([0.08, 0.28], [y_of(0.55), y_of(0.55)], color=P["base2"], lw=1.2, ls="--", dashes=(3, 2)); ax.plot([0.18], [y_of(0.55)], "s", color=P["base2"], ms=3.2, mfc="white"); ax.text(0.33, y_of(0.55), "$H(G_{E,t}\\mid\\bar O_t)$", ha="left", va="center", fontsize=7.5, color=P["base2"])
out = "paper/figs/fig1_concept.png"; plt.savefig(out, dpi=300, bbox_inches=None, pad_inches=0); plt.savefig(out.replace(".png", ".pdf"), bbox_inches=None, pad_inches=0); print("saved", out)
