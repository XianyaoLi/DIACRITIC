"""Fig 2 (main text): Task A, general regime.  (a) code rate during grasp/transport vs M (4..512): ours flat; system identification with
K=16 (capacity stress) and with the largest codebook tested at each M (K=128..1024, capacity-relieved) tracks log2 M with slope ~1.
(b) the learned rate against the exact optimum: ~_t is transitive on every Task A dataset, so the exact minimum is H(Gamma|O)=0 during
grasp/transport (Theorem exact-trans) while the strong congruence charges H(Gamma^s|O)=1 bit; theory recomputed with the fixed loader.
Usage: python isaac/fig2_taskA.py isaac/cluster_results/results/taskA.jsonl isaac/cluster_results/results/sysK.jsonl isaac/cluster_results/results/taskA_big.jsonl"""
import sys, json, os, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toy"))
import figstyle; figstyle.main(); P = figstyle.PALETTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = collections.defaultdict(list)
for r in rows:
    a = r["args"]; M = int(re.search(r"M(\d+)", os.path.basename(a["data"])).group(1)); K = a.get("K", 16)
    if a.get("variant", "diacritic") != "diacritic" or a.get("aux_mik", 0) or a.get("n_train", 448) != 448: continue
    if a.get("aux_theta", 0): g[("sys", K, M)].append(r)
    elif a["beta"] > 0 and K == 16: g[("ours", 16, M)].append(r)
def ph_mean(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.nanmean([r["per_t"][key][i] for i in idx]))
def cell(k): return [ph_mean(r, "H(C|O)", "grasp") for r in g[k]]
Ms = sorted({k[2] for k in g if k[0] == "ours"}); x = np.log2(Ms)
# theory from the (fixed) loader: exact minimum H(Gamma|O) and the strong congruence during grasp, per M
from aprime_data import prepare
theo = {}
for M in Ms:
    eps, meta, vis, env, solver, key = prepare(f"isaac/data/taskA2_M{M}", augment=True, sag_symbol=True); r_ = solver.rates(); ph = eps[0]["phase"]
    gi = [i for i, p in enumerate(ph) if p == "grasp"]
    theo[M] = (np.mean([r_["H(G|O)"][i] for i in gi]), np.mean([r_["H(Gamma|O)"][i] for i in gi]), np.mean([r_["H(GammaS|O)"][i] for i in gi]), all(solver.transitive.values()))
    print(f"M={M}: H(G|O) {theo[M][0]:.3f} H(Gamma|O) {theo[M][1]:.3f} H(GammaS|O) {theo[M][2]:.3f} transitive {theo[M][3]}")
fig, ax = plt.subplots(1, 2, figsize=(figstyle.TEXTW, 1.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25, left=0.075, right=0.995, top=0.84, bottom=0.5))
ax[0].plot(x, x, color=P["black"], ls=":", lw=1.0, label=r"$\log_2 M$")
def series(keyf, **st):
    xs = [M for M in Ms if keyf(M) in g]; y = [np.mean(cell(keyf(M))) for M in xs]; e = [np.std(cell(keyf(M))) for M in xs]
    ax[0].errorbar(np.log2(xs), y, e, **st); return xs, y
series(lambda M: ("ours", 16, M), marker="o", ls="-", color=P["ours"], lw=1.5, ms=3.5, zorder=5, label=r"DIACRITIC, $K{=}16$, $\beta{=}10^{-3}$")
series(lambda M: ("sys", 16, M), marker="s", ls="-", color=P["base"], lw=1.2, ms=3.2, label=r"sys-ID, $K{=}16$ (capacity stress)")
bigK = {M: max(k[1] for k in g if k[0] == "sys" and k[2] == M and k[1] > 16) for M in Ms if any(k[0] == "sys" and k[2] == M and k[1] > 16 for k in g)}
xs, y = series(lambda M: ("sys", bigK.get(M, -1), M), marker="^", ls="--", color=P["base2"], lw=1.1, ms=3.2, label=r"sys-ID, largest $K$ (128–1024)")
sl = np.polyfit(np.log2(xs), y, 1)[0]; print("capacity-relieved sys-ID slope vs log2 M:", round(sl, 3), "points", dict(zip(xs, np.round(y, 2))))
ax[0].set_ylabel(r"$\hat H(C_t\mid\bar O_t)$ (bits)"); ax[0].set_title("(a) grasp-phase rate vs. world complexity", loc="left")
ax[0].set_ylim(-0.2, 9.6); ax[0].set_yticks([0, 2, 4, 6, 8])
lo = [theo[M][1] for M in Ms]; hi = [theo[M][2] for M in Ms]
ax[1].fill_between(x, lo, hi, color="#000000", alpha=0.06, lw=0, label="bracket between the two bounds")
ax[1].plot(x, hi, color=P["black"], ls="--", lw=0.9, label=r"$H(\Gamma^s\mid\bar O)$ (strong congruence)")
ax[1].plot(x, lo, color=P["black"], ls="-", lw=1.3, label=r"exact minimum $H(\Gamma\mid\bar O)=H(G_E\mid\bar O)$")
for k, st in ((("ours", 16), dict(marker="o", ls="-", color=P["ours"], lw=1.5, ms=3.5, zorder=5)), (("sys", 16), dict(marker="s", ls="-", color=P["base"], lw=1.2, ms=3.2))):
    xs = [M for M in Ms if (k[0], k[1], M) in g]; ax[1].errorbar(np.log2(xs), [np.mean(cell((k[0], k[1], M))) for M in xs], [np.std(cell((k[0], k[1], M))) for M in xs], **st)
ax[1].set_ylabel("bits"); ax[1].set_title("(b) learned rate against the exact optimum", loc="left"); ax[1].set_ylim(-0.15, 4.2); ax[1].set_yticks([0, 1, 2, 3, 4])
ax[1].text(np.log2(4.3), 1.55, r"DIACRITIC (blue) $\leq 0.10$ bit at every $M$ (exact minimum)", fontsize=5.8, color=P["ours"], ha="left", va="center")
for a in ax: a.set_xticks(x); a.set_xticklabels([str(M) for M in Ms]); a.set_xlabel(r"$M$ (hidden mass modes)")
H, L = [], []
for a in ax:
    for h_, l_ in zip(*a.get_legend_handles_labels()):
        if l_ not in L: H.append(h_); L.append(l_)
figstyle.legend_below(fig, H, L, ncol=4, y=-0.04, handlelength=1.4, columnspacing=0.7, handletextpad=0.35)
out = "paper/figs/fig2_taskA.png"; figstyle.save(out)
