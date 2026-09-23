"""Aggregate isaac/results/*.jsonl: table per (variant, lam, beta) and a Fig-3-style figure (rate ladder vs theory).
Usage: python isaac/summarize.py isaac/results/beta_sweep_512.jsonl [more.jsonl ...]"""
import sys, json, math, os
from collections import defaultdict
import numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
g = defaultdict(list)
import os as _os
def _key(r):
    a = r["args"]; v = a.get("variant", "diacritic"); ref = a.get("refine", "") or ""; aux = a.get("aux_theta", 0) or 0
    tag = v + ("+ref" + ref if ref else "") + ("+sysid" if aux else "")
    return (_os.path.basename(a["data"]), tag, a.get("lam", 0.0), a["beta"])
for r in rows: g[_key(r)].append(r)
ms = lambda xs: (np.nanmean(xs), np.nanstd(xs))
print("dataset        variant        lam  beta   n | H(C|O)@gap1 [Gam] H(C|O)@gap2 [Gam] | S_Gam@gap1 S_Gam@gap2 | S_G grasp place | H(C|Gam,O) I(C;z|O) | cls_err mse_test | codes | reached theory: n  gap2-rate H(C|Gam,O) cls_err (successful runs only)")
for (ds, v, lam, beta), rs in sorted(g.items()):
    S = lambda k: ms([r["summary"][k] for r in rs])
    hg1, hg2 = rs[0]["summary"]["HGam_gap1"], rs[0]["summary"]["HGam_gap2"]
    ok = [r for r in rs if r["summary"]["S_Gam_gap2"] > 0.9 and r["summary"]["S_G_place"] > 0.9]     # reached the theory (sufficient memory)
    Sok = lambda k: np.mean([r["summary"][k] for r in ok]) if ok else float("nan")
    print(f"{ds:14s} {v:14s} {lam:4.1f} {beta:6.4f} {len(rs):2d} | {S('HC_gap1')[0]:5.2f}±{S('HC_gap1')[1]:4.2f} [{hg1:4.2f}] {S('HC_gap2')[0]:5.2f}±{S('HC_gap2')[1]:4.2f} [{hg2:4.2f}] | "
          f"{S('S_Gam_gap1')[0]:5.2f}±{S('S_Gam_gap1')[1]:4.2f} {S('S_Gam_gap2')[0]:5.2f}±{S('S_Gam_gap2')[1]:4.2f} | {S('S_G_grasp')[0]:4.2f} {S('S_G_place')[0]:4.2f}  | "
          f"{S('HC_Gam_O')[0]:5.2f}      {S('I_Cz_O')[0]:5.2f}   | {S('class_err_test')[0]:5.3f}  {S('mse_test')[0]:5.3f}  | {S('codes')[0]:4.1f} | "
          f"{len(ok)}/{len(rs)}  gap2 {Sok('HC_gap2'):4.2f} H(C|Gam,O) {Sok('HC_Gam_O'):4.2f} err {Sok('class_err_test'):5.3f}")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    phases = rows[0]["phases"]; T = len(phases)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    r0 = rows[0]["per_t"]
    ax.step(range(1, T + 1), r0["H(G|O)"], where="mid", color="k", ls=":", lw=2, label=r"$H(G_{E,t}\mid\bar O_t)$ (theory)")
    ax.step(range(1, T + 1), r0["H(Gam|O)"], where="mid", color="k", ls="-", lw=2, label=r"$H(\Gamma_t\mid\bar O_t)$ (theory)")
    SUCC = os.environ.get("SUCC_ONLY", "1") == "1"; BETAS = os.environ.get("BETAS"); LAMS = os.environ.get("LAMS")
    keep = {k: v for k, v in g.items() if (BETAS is None or k[3] in [float(b) for b in BETAS.split(",")]) and (LAMS is None or k[2] in [float(l) for l in LAMS.split(",")])}
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(keep), 1)))
    for ((ds, v, lam, beta), rs), c in zip(sorted(keep.items()), cols):
        if SUCC:
            rs = [r for r in rs if r["summary"]["S_Gam_gap2"] > 0.9 and r["summary"]["S_G_place"] > 0.9] or rs
        hc = np.array([r["per_t"]["H(C|O)"] for r in rs])
        ax.errorbar(range(1, T + 1), hc.mean(0), hc.std(0), fmt="o-", ms=3, color=c, capsize=2, label=f"{ds} {v}{'-RF' if lam > 0 else '-R'} β={beta} ($\\hat H(C_t\\mid\\bar O_t)$, n={len(rs)})")
    # phase bands
    last = None
    for t, p in enumerate(phases):
        if p != last:
            ax.axvline(t + 0.5, color="gray", lw=0.5, alpha=0.5); ax.text(t + 0.6, ax.get_ylim()[1] * 0.97, p, fontsize=7, va="top", color="gray"); last = p
    ax.set_xlabel("t (macro step)"); ax.set_ylabel("bits"); ax.set_title("A' (Franka, tier 4, gap2=6): learned conditional code rate vs theory" + (" — runs that reached sufficiency" if SUCC else "")); ax.legend(fontsize=7, ncol=2)
    out = os.path.splitext(sys.argv[1])[0] + ".png"; plt.tight_layout(); plt.savefig(out, dpi=140); print("saved", out)
except Exception as e:
    print("figure skipped:", e)
