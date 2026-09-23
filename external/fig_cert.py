"""Appendix figure: certified requirement vs learned rate and sufficient seeds on the bsuite memory chain (3 panels: 1, 2, 3 context bits) against the delay.
Usage: python external/fig_cert.py   (reads external/results/cert.jsonl + external/results/cluster/cert.*.jsonl; writes paper/figs/fig_cert.pdf/.png)"""
import sys, os, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "isaac")); import figstyle; figstyle.appendix(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})   # same typeface as Figures 1-3
C = collections.defaultdict(dict)
for f in [os.path.join(HERE, "results/cert.jsonl")] + sorted(glob.glob(os.path.join(HERE, "results/cluster/cert.*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") == "run" and r["task"] == "memchain": C[(r["bits"], r["L"], r["mode"])][r["seed"]] = r
Ls = [10, 30, 100]; x = np.arange(3); wd = 0.32
fig, ax = plt.subplots(1, 3, figsize=(figstyle.TEXTW, 1.75), gridspec_kw=dict(wspace=0.5, left=0.07, right=0.94, top=0.86, bottom=0.36))
for k, b in enumerate((1, 2, 3)):
    a = ax[k]; a2 = a.twinx()
    for j, (m, col, lab) in enumerate((("plain", P["gray"], "sufficient seeds, plain imitation"), ("generic", P["purple"], "sufficient seeds, event-agnostic forecast"))):
        a2.bar(x + (j - 0.5) * wd, [sum(r["sufficient"] for r in C[(b, L, m)].values()) for L in Ls], wd * 0.92, color=col, alpha=0.45, label=lab, zorder=1)
    req = [np.mean([r["req_mid"] for m in ("plain", "generic") for r in C[(b, L, m)].values()]) for L in Ls]
    a.plot(x, req, color=P["black"], lw=1.6, label="certified requirement $H(\\Gamma_t\\mid O_t)$", zorder=4)
    for i, L in enumerate(Ls):
        ok = [r["rate_mid"] for m in ("plain", "generic") for r in C[(b, L, m)].values() if r["sufficient"]]
        if ok: a.plot([i] * len(ok), ok, "o", color=P["ours"], ms=3.5, mec="white", mew=0.4, zorder=5, label="learned rate, sufficient seeds" if i == 0 else "_")
    a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
    a.set_xticks(x); a.set_xticklabels([str(L) for L in Ls]); a.set_xlabel("delay (steps)"); a.set_ylim(0, 3.4); a.set_yticks([0, 1, 2, 3]); a2.set_ylim(0, 8.8); a2.set_yticks([0, 4, 8]); a2.grid(False)
    a.set_title(f"{b}-bit context", loc="left")
    if k == 0: a.set_ylabel("bits at mid-delay")
    if k == 2: a2.set_ylabel("sufficient\nseeds (of 8)", labelpad=3)
H, Lb = ax[0].get_legend_handles_labels(); h2, l2 = ax[0].figure.axes[3].get_legend_handles_labels()
figstyle.legend_below(fig, H + h2, Lb + l2, ncol=2, y=-0.02)
figstyle.save(os.path.join(HERE, "..", "paper", "figs", "fig_cert.png"))
