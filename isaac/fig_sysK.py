"""Control on task A: system identification with a larger codebook (K = 16/64/128/256) vs ours (K = 16/64), and the recurrent
multi-step-inverse baseline.  Left: grasp/transport code rate vs log2 M (sys-ID should track log2 M + behaviour bits once K allows it);
middle: I(C; mass | O); right: fraction of seeds sufficient at the place step.  Usage: python isaac/fig_sysK.py taskA.jsonl sysK.jsonl mik.jsonl"""
import sys, json, os, re, collections
import numpy as np
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "isaac")); import figstyle; figstyle.appendix()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = collections.defaultdict(list)
for r in rows:
    a = r["args"]; M = int(re.search(r"M(\d+)", os.path.basename(a["data"])).group(1)); K = a.get("K", 16)
    if a.get("variant", "diacritic") != "diacritic": continue
    if a.get("aux_theta", 0): tag = f"sys-ID K={K}"
    elif a.get("aux_mik", 0): tag = "multi-step inverse only" if a.get("mik_detach", 0) else "multi-step inverse + imitation + rate"
    elif a["beta"] > 0: tag = f"ours β=1e-3 K={K}"
    else: continue
    g[(tag, M)].append(r)
def ph_mean(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.nanmean([r["per_t"][key][i] for i in idx]))
fig, ax = plt.subplots(1, 3, figsize=(figstyle.TEXTW, 2.15)); Ms = sorted({k[1] for k in g}); tags = sorted({k[0] for k in g})
ax[0].plot(np.log2(Ms), np.log2(Ms), "k:", label=r"$\log_2 M$"); ax[0].plot(np.log2(Ms), np.log2(Ms) + 0.0, alpha=0)
fam = {"ours β=1e-3 K=16": ("o-", figstyle.PALETTE["ours"]), "ours β=1e-3 K=64": ("o-", figstyle.PALETTE["ours2"]), "sys-ID K=16": ("s--", figstyle.PALETTE["base"]), "sys-ID K=64": ("s--", figstyle.PALETTE["base2"]), "sys-ID K=128": ("^--", figstyle.PALETTE["base2"]), "sys-ID K=256": ("v--", figstyle.PALETTE["base3"]), "multi-step inverse only": ("^-.", figstyle.PALETTE["alt"]), "multi-step inverse + imitation + rate": ("^-.", figstyle.PALETTE["purple"])}
sty = {t: fam.get(t, ("o-", figstyle.PALETTE["gray"]))[0] for t in tags}; col = {t: fam.get(t, ("o-", figstyle.PALETTE["gray"]))[1] for t in tags}
for tag in tags:
    xs = [M for M in Ms if (tag, M) in g]
    ax[0].plot(np.log2(xs), [np.mean([ph_mean(r, "H(C|O)", "grasp") for r in g[(tag, M)]]) for M in xs], sty[tag], color=col[tag], label=tag)
    ax[1].plot(np.log2(xs), [np.mean([ph_mean(r, "I(C;mass|O)", "grasp") for r in g[(tag, M)]]) for M in xs], sty[tag], color=col[tag])
    ax[2].plot(np.log2(xs), [np.mean([r["summary"]["S_G_place"] > 0.9 for r in g[(tag, M)]]) for M in xs], sty[tag], color=col[tag])
ax[0].set_xlabel(r"$\log_2 M$"); ax[0].set_ylabel("bits during grasp/transport"); ax[0].set_title("(a) code rate: paying for the world")
ax[1].set_xlabel(r"$\log_2 M$"); ax[1].set_ylabel(r"$I(C;\mathrm{mass}\mid\bar O)$"); ax[1].set_title("(b) world-state information kept")
ax[2].set_xlabel(r"$\log_2 M$"); ax[2].set_ylabel("fraction sufficient (place)"); ax[2].set_ylim(-0.05, 1.05); ax[2].set_title("(c) behavioral sufficiency")
h, l = ax[0].get_legend_handles_labels(); figstyle.legend_below(fig, h, l, ncol=5, y=-0.02, handlelength=1.6, columnspacing=1.0, fontsize=6.6)
plt.tight_layout(rect=(0, 0.13, 1, 1)); out = "paper/figs/fig_sysK.png"; figstyle.save(out)
for tag in tags:
    for M in Ms:
        if (tag, M) not in g: continue
        rs = g[(tag, M)]; print(f"{tag:40s} M={M:2d} n={len(rs)} suff {sum(r['summary']['S_G_place']>0.9 for r in rs)}/{len(rs)} rate@grasp {np.mean([ph_mean(r,'H(C|O)','grasp') for r in rs]):.2f} I(C;mass) {np.mean([ph_mean(r,'I(C;mass|O)','grasp') for r in rs]):.2f} cls_err {np.mean([r['summary']['class_err_test'] for r in rs]):.3f}")
