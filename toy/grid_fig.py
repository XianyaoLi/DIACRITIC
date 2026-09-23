"""Figure for the signpost-corridor domain: (a) exact ladder H(C|O) vs t (W=1, M=16, sufficient seeds, -R and -RF) against H(Gamma|O), H(G|O), H(H|O);
(b) learning horizon: fraction of seeds sufficient vs reveal->use distance of beta_2 (gap2 sweep) per variant; (c) layer-2 nuisance: I(C;w|O) in the
halls vs log2 W per variant, with the strong-congruence excess H(GammaS|O) - 2.  Usage: python toy/grid_fig.py grid_ladder.jsonl grid_horizon.jsonl grid_drift.jsonl"""
import sys, json, re, collections
import numpy as np
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "isaac")); import figstyle; figstyle.appendix()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from grid_summarize import read_records
rows = [r for f in sys.argv[1:] for r in read_records(f)]
def meta(r):
    m = re.search(r"M=(\d+),R1=(\d+),R2=(\d+),W=(\d+),N=\d+,gap1=(\d+),gap2=(\d+)", r["env"]); a = r["args"]
    return dict(M=int(m.group(1)), W=int(m.group(4)), gap1=int(m.group(5)), gap2=int(m.group(6)), tag=re.sub(r"\.jsonl.*$", "", a["out"].split("/")[-1]),
                variant=a["variant"], beta=a["beta"], lam=a["lam"])
def rng(r, ph): a, b = r["phases"][ph]; return list(range(a - 1, b))
def suff(r):
    res = r["res"]; g = rng(r, "gap1") + rng(r, "gap2"); sg = [res["S_Gam"][i] for i in g if not np.isnan(res["S_Gam"][i])]
    return (all(v > 0.9 for v in sg) if sg else True) and res["err"][r["phases"]["use1"] - 1] < 1e-6 and res["err"][r["phases"]["use2"] - 1] < 1e-6
def label(m):
    v = m["variant"]
    if v == "diacritic": return ("β=0" if m["beta"] == 0 else ("-RF" if m["lam"] > 0 else "-R"))
    if v == "mik": return "multi-step inverse" + (" (no rate)" if m["beta"] == 0 else " (+rate)")
    return {"sysid": "sys-ID", "bypass": "GRU bypass", "uncond": "uncond", "scaffold": "scaffold-RF"}.get(v, v)
COL = {"-R": figstyle.PALETTE["ours"], "-RF": figstyle.PALETTE["base"], "scaffold-RF": figstyle.PALETTE["purple"], "GRU bypass": figstyle.PALETTE["amber"],
       "multi-step inverse (no rate)": figstyle.PALETTE["alt"], "multi-step inverse (+rate)": figstyle.PALETTE["alt2"], "sys-ID": figstyle.PALETTE["gray"], "uncond": figstyle.PALETTE["ours2"], "β=0": figstyle.PALETTE["ours2"]}
MK = {"-R": "o", "-RF": "s", "scaffold-RF": "^", "GRU bypass": "D", "multi-step inverse (no rate)": "v", "multi-step inverse (+rate)": "P", "sys-ID": "x", "uncond": "d", "β=0": "d"}
def lstyle(lab): return dict(color=COL.get(lab, figstyle.PALETTE["black"]), marker=MK.get(lab, "o"), ms=3.5)
fig, axs = plt.subplots(2, 2, figsize=(figstyle.TEXTW, 4.3), gridspec_kw=dict(wspace=0.28, hspace=0.42, left=0.09, right=0.99, top=0.95, bottom=0.09)); ax = [axs[0, 0], axs[0, 1], axs[1, 0]]; axs[1, 1].axis("off")
# (a) ladder
lad = [r for r in rows if meta(r)["tag"] == "grid_ladder" and meta(r)["M"] == 16 and meta(r)["W"] == 1 and meta(r)["variant"] == "diacritic" and meta(r)["beta"] > 0]
if lad:
    r0 = lad[0]; T = len(r0["theory"]["H(Gamma|O)"]); t = np.arange(1, T + 1)
    ax[0].plot(t, r0["theory"]["H(Gamma|O)"], "k-", lw=1.5, label=r"$H(\Gamma_t\mid\bar O_t)$"); ax[0].plot(t, r0["theory"]["H(G|O)"], "k:", lw=1.5, label=r"$H(G_{E,t}\mid\bar O_t)$")
    ax[0].plot(t, r0["theory"]["H(H|O)"], color="gray", ls="--", lw=1, label=r"$H(H_t\mid\bar O_t)$ (store everything)")
    for lab, col in (("-R", figstyle.PALETTE["ours"]), ("-RF", figstyle.PALETTE["base"])):
        rs = [r for r in lad if label(meta(r)) == lab and suff(r)]
        if rs: Y = np.array([r["res"]["H(C|O)"] for r in rs]); ax[0].errorbar(t, Y.mean(0), Y.std(0), fmt="o-", color=col, ms=3, capsize=2, label=f"$\\hat H(C_t\\mid\\bar O_t)$, {lab} sufficient seeds")
    ph = r0["phases"]
    for name, (a, b) in (("signs", ph["reveal"]), ("hall 1", ph["gap1"]), ("hall 2", ph["gap2"])): figstyle.phase_span(ax[0], a, b, name)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("bits"); ax[0].set_title("(a) corridor, exact regime: the ladder"); ax[0].set_ylim(-0.1, 6.9); ax[0].set_xticks([1, 5, 10, 15])
# (b) horizon
hor = [r for r in rows if meta(r)["tag"] == "grid_horizon" and meta(r)["gap1"] == 3]
g = collections.defaultdict(list)
for r in hor:
    m = meta(r); d = r["phases"]["use2"] - r["phases"]["reveal"][1]; g[(label(m), d)].append(suff(r))
for lab in sorted({k[0] for k in g}):
    ds = sorted({k[1] for k in g if k[0] == lab}); ax[1].plot(ds, [np.mean(g[(lab, d)]) for d in ds], ls="-", label=lab, **lstyle(lab))
ax[1].set_xlabel(r"reveal $\to$ use distance of $\beta_2$ (steps)"); ax[1].set_ylabel("fraction of seeds sufficient"); ax[1].set_ylim(-0.05, 1.05); ax[1].set_title("(b) learning horizon on the corridor"); pass
# (c) drift
dr = [r for r in rows if meta(r)["tag"] == "grid_drift"]
g2 = collections.defaultdict(list)
for r in dr:
    m = meta(r); g2[(label(m), m["W"])].append(r)
Ws = sorted({k[1] for k in g2})
if dr:
    ax[2].plot(np.log2(Ws), [np.mean([g2[k][0]["theory"]["H(GammaS|O)"][i] for i in rng(g2[k][0], "gap1")]) - 2.0 for k in [(next(kk[0] for kk in g2 if kk[1] == W), W) for W in Ws]], "k--", lw=1.4, label=r"$H(\Gamma^s\mid\bar O)-H(\Gamma\mid\bar O)$ (drift bits)")
for lab in sorted({k[0] for k in g2}):
    ws = [W for W in Ws if (lab, W) in g2]
    ax[2].plot(np.log2(ws), [np.mean([np.nanmean([r["res"]["I(C;w|O)"][i] for i in rng(r, "gap1") + rng(r, "gap2")]) for r in g2[(lab, W)]]) for W in ws], ls="-", label=lab, **lstyle(lab))
ax[2].set_xlabel(r"$\log_2 W$ (drift modes)"); ax[2].set_ylabel(r"$I(C;\,w\mid\bar O)$ in the halls (bits)"); ax[2].set_title("(c) agent-centric, behaviour-irrelevant drift")
ax[2].annotate("rate-penalised carriers,\nGRU bypass, sys-ID:\nall $\\leq 0.1$ bit (overlapping)", xy=(2.3, 0.1), xytext=(0.12, 1.0), fontsize=6.5, ha="left", color="#333333", arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, shrinkB=2))
H, L = [], []
for a_ in ax:
    for h_, l_ in zip(*a_.get_legend_handles_labels()):
        if l_ not in L: H.append(h_); L.append(l_)
axs[1, 1].legend(H, L, loc="center", bbox_to_anchor=(0.5, 0.42), ncol=1, fontsize=7.2, handlelength=2.0, labelspacing=0.55, borderaxespad=0)
out = "paper/figs/fig_grid.png"; figstyle.save(out)
