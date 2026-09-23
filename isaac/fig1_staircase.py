"""Standalone staircase panel for Fig 1(b), drawn at the 2x slide scale (bar width 10.6 in) so it can be pasted under the timeline
in Google Slides: H(Gamma_t|Obar_t) solid blue with filled circles, H(G_{E,t}|Obar_t) dashed orange with hollow squares, a 0/1/2-bit axis
on the left, transparent background.  Segment widths 4:2:2:6:2:1 (identify, gap1, grasp, gap2, place, done) match the timeline bar.
Outputs paper/figs/fig1_staircase.{svg,pdf,png} (with legend) and *_bare.* (lines + axis only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import figstyle; P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42})
S = 2.0                     # slide scale: everything is 2x the printed size
BAR_W, H = 10.6, 1.15       # inches at slide scale (printed 5.3 x 0.52)
FS = 15                     # ~7.5 pt printed
segs = [4, 2, 2, 6, 2, 1]; tot = sum(segs)
def draw(legend, out):
    left = 2.05 if legend else 0.5
    fig = plt.figure(figsize=(left + BAR_W + 0.1, H)); fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, left + BAR_W + 0.1); ax.set_ylim(0, H); ax.axis("off"); ax.patch.set_alpha(0)
    u = BAR_W / tot; edges = [left + sum(segs[:i]) * u for i in range(7)]
    y_of = lambda v: 0.10 + 0.38 * v
    gam = [(0, 2), (2, 2), (2, 2), (1, 1), (1, 1), (0, 0)]; ge = [0, 0, 1, 0, 1, 0]
    xs, ys = [], []
    for i, (a, b) in enumerate(gam): xs += [edges[i], edges[i + 1]]; ys += [y_of(a), y_of(b)]
    ax.plot(xs, ys, color=P["ours"], lw=3.0, solid_capstyle="round", solid_joinstyle="round")
    ax.plot([(edges[i] + edges[i + 1]) / 2 for i in range(1, 6)], [y_of(gam[i][1]) for i in range(1, 6)], "o", color=P["ours"], ms=7)
    xs, ys = [], []
    for i, v in enumerate(ge): xs += [edges[i], edges[i + 1]]; ys += [y_of(v) - 0.025, y_of(v) - 0.025]
    ax.plot(xs, ys, color=P["base2"], lw=2.4, ls="--", dashes=(3, 2))
    ax.plot([(edges[i] + edges[i + 1]) / 2 for i in range(6)], [y_of(v) - 0.025 for v in ge], "s", color=P["base2"], ms=7, mfc="white", mew=1.6)
    # segment boundary guides (light, so they align with the bar above when pasted)
    for xe in edges[1:-1]: ax.plot([xe, xe], [y_of(0) - 0.06, y_of(2) + 0.06], color="#bbbbbb", lw=0.6, ls=":")
    # bit axis
    ax.plot([left - 0.08, left - 0.08], [y_of(0), y_of(2)], color="#444444", lw=1.2)
    for v in (0, 1, 2):
        ax.plot([left - 0.13, left - 0.08], [y_of(v), y_of(v)], color="#444444", lw=1.2); ax.text(left - 0.17, y_of(v), f"{v}", ha="right", va="center", fontsize=FS, color="#444444")
    ax.text(left - 0.08, y_of(2) + 0.07, "bits", ha="center", va="bottom", fontsize=FS, color="#444444")
    if legend:
        ax.plot([0.10, 0.50], [y_of(1.55)] * 2, color=P["ours"], lw=3.0); ax.plot([0.30], [y_of(1.55)], "o", color=P["ours"], ms=7); ax.text(0.58, y_of(1.55), "$H(\\Gamma_t\\mid\\bar O_t)$", ha="left", va="center", fontsize=FS, color=P["ours"])
        ax.plot([0.10, 0.50], [y_of(0.45)] * 2, color=P["base2"], lw=2.4, ls="--", dashes=(3, 2)); ax.plot([0.30], [y_of(0.45)], "s", color=P["base2"], ms=7, mfc="white", mew=1.6); ax.text(0.58, y_of(0.45), "$H(G_{E,t}\\mid\\bar O_t)$", ha="left", va="center", fontsize=FS, color=P["base2"])
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig); print("saved", out, "(.svg .pdf .png)", "size in:", round(left + BAR_W + 0.1, 2), "x", H)
draw(True, "paper/figs/fig1_staircase"); draw(False, "paper/figs/fig1_staircase_bare")
