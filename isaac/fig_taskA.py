"""Task A figure: learned code rate during the grasp/transport phase (before any reveal is needed) vs log2 M, per variant, with the
sandwich [H(Gamma|O), H(GammaS|O)] and the store-everything line; plus I(C;mass|O) and sufficiency per M.
Usage: python isaac/fig_taskA.py isaac/cluster_results/results/taskA.jsonl"""
import sys, json, os, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows:
    a = r["args"]; M = int(re.search(r"M(\d+)", os.path.basename(a["data"])).group(1))
    tag = ("sys-ID" if a.get("aux_theta", 0) else a.get("variant", "diacritic")) + (f" β={a['beta']:g}")
    g[(tag, M)].append(r)
def ph_mean(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.nanmean([r["per_t"][key][i] for i in idx]))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
tags = sorted({k[0] for k in g}); Ms = sorted({k[1] for k in g})
th = {M: next(r for (t, m), rs in g.items() if m == M for r in rs).get("theory") for M in Ms}
for M in Ms:
    if th[M] is None: continue
ax[0].plot(np.log2(Ms), [np.mean([th[M]["H(Gamma|O)"][i] for i, p in enumerate(next(r for (t, m), rs in g.items() if m == M for r in rs)["phases"]) if p == "grasp"]) for M in Ms], "k-", lw=2, label=r"$H(\Gamma_t\mid\bar O_t)$ (lower bound)")
ax[0].plot(np.log2(Ms), [np.mean([th[M]["H(GammaS|O)"][i] for i, p in enumerate(next(r for (t, m), rs in g.items() if m == M for r in rs)["phases"]) if p == "grasp"]) for M in Ms], "k--", lw=2, label=r"$H(\Gamma^s_t\mid\bar O_t)$ (upper bound)")
ax[0].plot(np.log2(Ms), np.log2(Ms), "k:", label=r"$\log_2 M$ (store the mass)")
for tag in tags:
    xs = [M for M in Ms if (tag, M) in g]; ys = [np.mean([ph_mean(r, "H(C|O)", "grasp") for r in g[(tag, M)]]) for M in xs]
    ax[0].plot(np.log2(xs), ys, "o-", label=tag)
    ax[1].plot(np.log2(xs), [np.mean([ph_mean(r, "I(C;mass|O)", "grasp") for r in g[(tag, M)]]) for M in xs], "o-", label=tag)
    ax[2].plot(np.log2(xs), [np.mean([r["summary"]["S_G_place"] > 0.9 for r in g[(tag, M)]]) for M in xs], "o-", label=tag)
ax[0].set_xlabel(r"$\log_2 M$ (mass modes)"); ax[0].set_ylabel("bits during grasp/transport"); ax[0].set_title("Task A: learned rate vs the sandwich"); ax[0].legend(fontsize=7)
ax[1].set_xlabel(r"$\log_2 M$"); ax[1].set_ylabel(r"$I(C;\,\mathrm{mass}\mid\bar O)$ (bits)"); ax[1].set_title("world-state information kept"); ax[1].legend(fontsize=7)
ax[2].set_xlabel(r"$\log_2 M$"); ax[2].set_ylabel("fraction sufficient at the place step"); ax[2].set_ylim(-0.05, 1.05); ax[2].set_title("behavioural sufficiency (matched budget)"); ax[2].legend(fontsize=7)
plt.tight_layout(); out = os.path.splitext(sys.argv[1])[0] + "_fig.png"; plt.savefig(out, dpi=140); print("saved", out)
