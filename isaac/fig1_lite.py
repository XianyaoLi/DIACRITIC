"""Alternative concept layout: four boxes + three arrows; A' episode bar; two number rows. Original aspect (11 x 2.25 in)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 11.0, 2.25
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
GRAY, LB, B, OR, DG, EDGE, INK_B, INK_O = "#ececec", "#dbe8f6", "#a9c9ea", "#f8e4d0", "#dcdcdc", "#6b6b6b", "#1f4e9c", "#b3541e"
boxes = [("world state $S_t$", "hidden $\\theta$: $\\log_2 M$ bits", None, GRAY),
         ("behavioral quotient $G_{E,t}$", "what the current action depends on", "$R_E(0)=H(G_{E,t}\\,|\\,O_t)$", LB),
         ("behavioral memory $\\Gamma_t$", "what a future action will still need", "$R^{\\rm mem}_t(0)=H(\\Gamma_t\\,|\\,O_t)$", B),
         ("learned code $C_t$", "sole carrier, rate-penalized", "rate $-\\log r_\\eta(C_t\\,|\\,O_t)$", OR)]
arrows = [("forget", None), ("keep", "refines\n$G_{E,t}$"), ("learn", None)]
bw, bh, gap = 2.25, 0.92, 0.55
y0 = H - 0.10 - bh
xs = [0.2 + i * (bw + gap) for i in range(4)]
for (t, s, f, c), x in zip(boxes, xs):
    ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.14", fc=c, ec=EDGE, lw=0.8))
    ax.text(x + bw / 2, y0 + bh - 0.19, t, ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(x + bw / 2, y0 + bh - 0.45, s, ha="center", va="center", fontsize=10.5)
    if f: ax.text(x + bw / 2, y0 + 0.18, f, ha="center", va="center", fontsize=9.5, color="#333333")
for (lab, sub), x1, x2 in zip(arrows, xs[:-1], xs[1:]):
    xa, xb = x1 + bw + 0.07, x2 - 0.07; ym = y0 + bh / 2
    ax.add_patch(FancyArrowPatch((xa, ym), (xb, ym), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#333333"))
    ax.text((xa + xb) / 2, ym + 0.13, lab, ha="center", va="bottom", fontsize=10.5, fontweight="bold", style="italic")
    if sub: ax.text((xa + xb) / 2, ym - 0.14, sub, ha="center", va="top", fontsize=8.5, color="#333333")
segs = [("reveal $\\beta_1,\\beta_2$", 1.55, LB), ("wait", 1.15, LB), ("grasp: use $\\beta_1$", 1.55, OR),
        ("transport: $\\beta_1$ done", 2.45, LB), ("place: use $\\beta_2$", 2.45, OR), ("done", 0.75, DG)]
gam = ["0$\\rightarrow$2", "2", "2", "2$\\rightarrow$1", "1", "1$\\rightarrow$0"]; gE = ["0", "0", "1", "0", "1", "0"]
xL, xR, hb = 1.55, 10.8, 0.38
yb = y0 - 0.32 - hb
sc = (xR - xL) / sum(w for _, w, _ in segs)
yr1, yr2 = yb - 0.17, yb - 0.40
x = xL
for (lab, w, c), g, e in zip(segs, gam, gE):
    w *= sc
    ax.add_patch(Rectangle((x, yb), w, hb, fc=c, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, yb + hb / 2, lab, ha="center", va="center", fontsize=10.5)
    ax.text(x + w / 2, yr1, g, ha="center", va="center", fontsize=10.5, color=INK_B, fontweight="bold")
    ax.text(x + w / 2, yr2, e, ha="center", va="center", fontsize=10.5, color=INK_O, fontweight="bold")
    if lab == "wait":
        bx = x + w / 2 + 0.17
        ax.plot([bx, bx + 0.06, bx + 0.06, bx], [yr1 + 0.09, yr1 + 0.09, yr2 - 0.09, yr2 - 0.09], color=INK_B, lw=0.9)
        ax.text(bx + 0.11, (yr1 + yr2) / 2, "$\\Delta_t=2$", ha="left", va="center", fontsize=9.5, color=INK_B)
    if lab.startswith("grasp"): ax.text(x + w / 2 + 0.12, yr2, "($\\beta_1$ only)", ha="left", va="center", fontsize=8, color=INK_O)
    if lab.startswith("place"): ax.text(x + w / 2 + 0.12, yr2, "($\\beta_2$ only)", ha="left", va="center", fontsize=8, color=INK_O)
    x += w
ax.text(xL - 0.12, yb + hb / 2, "one episode of A$'$", ha="right", va="center", fontsize=10, fontweight="bold", color="#333333")
ax.text(xL - 0.12, yr1, "keep  $H(\\Gamma_t\\,|\\,\\bar O_t)$", ha="right", va="center", fontsize=9.5, color=INK_B)
ax.text(xL - 0.12, yr2, "now  $H(G_{E,t}\\,|\\,\\bar O_t)$", ha="right", va="center", fontsize=9.5, color=INK_O)
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"paper/figs/fig1_lite.{ext}", dpi=220, transparent=(ext == "svg"))
print("ok")
