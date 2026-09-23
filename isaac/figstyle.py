"""Shared matplotlib style for the paper figures: muted, print-safe palette; serif text matching the Times body of the paper; light frames.
Figures are drawn at their final printed width (paper text width = 5.5 in) so that font sizes are the sizes seen in print.
`main()` = compact settings for the page-limited main text, `appendix()` = roomier settings for the appendix."""
import matplotlib as mpl
PALETTE = {"ours": "#3b5b92", "ours2": "#7f9fd0", "base": "#b04a3a", "base2": "#d4907f", "base3": "#e8bfb4", "alt": "#3f8f8a", "alt2": "#8fc4bf",
           "gray": "#7a7a7a", "black": "#222222", "purple": "#7b5c9b", "amber": "#c58a2a"}
CYCLE = [PALETTE[k] for k in ("ours", "base", "alt", "purple", "amber", "gray", "ours2", "base2")]
TEXTW = 5.5     # paper \textwidth in inches
_BASE = {
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "axes.prop_cycle": mpl.cycler(color=CYCLE), "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "xtick.color": "#444444", "ytick.color": "#444444",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.45, "grid.color": "#888888", "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "lines.markeredgewidth": 0.8, "legend.frameon": False, "legend.handlelength": 1.8,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "errorbar.capsize": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titleweight": "normal", "axes.titlepad": 3.0, "axes.labelpad": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
}
def apply(size=8.5):
    mpl.rcParams.update(_BASE)
    mpl.rcParams.update({"font.size": size, "axes.titlesize": size, "axes.labelsize": size, "legend.fontsize": size - 1.0,
                         "xtick.labelsize": size - 1.0, "ytick.labelsize": size - 1.0, "legend.title_fontsize": size - 1.0})
def main(): apply(7.0)
def appendix(): apply(8.0)
def legend_below(fig, handles, labels, ncol, y=0.0, **kw):
    """One shared legend row under all panels (identity is stated once, outside the data)."""
    kw = dict(dict(columnspacing=1.2, handletextpad=0.5), **kw)
    return fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), **kw)
def report(path, target_w=TEXTW):
    """Print the printed height of a saved png when included at width = target_w (used to hold the main-text height cap)."""
    from PIL import Image; w, h = Image.open(path).size; print(f"saved {path}: {w}x{h}px -> {target_w:.2f} x {target_w*h/w:.2f} in at width={target_w:.2f}in")
def save(path_png, target_w=TEXTW):
    """Save the current figure as vector PDF (used by the paper) and PNG (preview), then report the printed height."""
    import matplotlib.pyplot as plt, os
    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    base = os.path.splitext(path_png)[0]; plt.savefig(base + ".pdf"); plt.savefig(base + ".png"); report(base + ".png", target_w)
SPAN = dict(color="#000000", alpha=0.06, lw=0, zorder=0)
def phase_span(ax, x0, x1, label=None, where="top", size=None, pad=0.5):
    """Uniform shaded phase band [x0-pad, x1+pad] with its name set in small gray text: `where`='top' (inside, at the top of the axes),
    'above' (just outside the top edge, used when the plot area is crowded), or a float y in data units."""
    import matplotlib as mpl
    ax.axvspan(x0 - pad, x1 + pad, **SPAN)
    if label is None: return
    size = size or mpl.rcParams["font.size"] - 1.0; xm = (x0 + x1) / 2; kw = dict(ha="center", fontsize=size, color="#444444", zorder=6)
    if where == "top": ax.text(xm, 0.965, label, transform=ax.get_xaxis_transform(), va="top", **kw)
    elif where == "above": ax.text(xm, 1.015, label, transform=ax.get_xaxis_transform(), va="bottom", **kw)
    else: ax.text(xm, where, label, va="center", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85), **kw)
