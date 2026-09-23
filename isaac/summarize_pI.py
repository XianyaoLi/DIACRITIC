"""P-I (world vs behavioural complexity): learned gap1 code rate vs log2 M at fixed behavioural classes (R1 R2 = 4).
Prediction: slope d H(C|O)/d log2 M ≈ 0 at H(Gam|O) = 2 (rate-pressured, successful runs); H(H|O) = log2 M + 1 is the
"store everything" line.  Usage: python isaac/summarize_pI.py isaac/results/pI_msweep.jsonl"""
import sys, json, os, math
from collections import defaultdict
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
g = defaultdict(list)
for r in rows:
    M = int(os.path.basename(r["args"]["data"]).split("M")[-1]); tag = ("sysid" if r["args"].get("aux_theta", 0) else r["args"].get("variant", "diacritic")) + ("+curric" if r["args"].get("init_from") else ""); g[(M, r["args"]["beta"], r["args"].get("lam", 0.0), tag)].append(r)
def gap_mean(r, key, ph="gap1"):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.mean([r["per_t"][key][i] for i in idx]))
print("  variant     M  log2M  beta   lam  n | reached | gap1: H(H|O)  H(Gam|O)  H(C|O) all      H(C|O) succ | H(C|Gam,O) succ | cls_err succ")
pts = defaultdict(list)
for (M, beta, lam, tag), rs in sorted(g.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][2], kv[0][0])):
    ok = [r for r in rs if r["summary"]["S_Gam_gap2"] > 0.9 and r["summary"]["S_G_place"] > 0.9]
    HH = gap_mean(rs[0], "H(H|O)") if "H(H|O)" in rs[0]["per_t"] else math.log2(M) + 1.0; HGam = gap_mean(rs[0], "H(Gam|O)")
    hc_all = [gap_mean(r, "H(C|O)") for r in rs]; hc_ok = [gap_mean(r, "H(C|O)") for r in ok]
    hcg_ok = [gap_mean(r, "H(C|Gam,O)") for r in ok]; err_ok = [r["summary"]["class_err_test"] for r in ok]
    print(f"{tag:10s} {M:3d}  {math.log2(M):4.1f}  {beta:6.4f} {lam:4.1f} {len(rs):2d} |  {len(ok)}/{len(rs)}    |       {HH:4.2f}     {HGam:4.2f}    {np.mean(hc_all):4.2f}±{np.std(hc_all):4.2f}   "
          f"{(np.mean(hc_ok) if ok else float('nan')):4.2f}±{(np.std(hc_ok) if ok else 0):4.2f} |   {(np.mean(hcg_ok) if ok else float('nan')):4.2f}       |  {(np.mean(err_ok) if ok else float('nan')):.3f}")
    pts[(tag, beta, lam)].append((math.log2(M), np.mean(hc_ok) if ok else float("nan"), np.mean(hc_all), HH, HGam))
for (tag, beta, lam), v in pts.items():
    v = sorted(v); x = np.array([a[0] for a in v]); y = np.array([a[1] for a in v]); ya = np.array([a[2] for a in v])
    m = ~np.isnan(y)
    if m.sum() >= 2:
        print(f"slope dH(C|O)/dlog2M  {tag} beta={beta} lam={lam}: successful runs {np.polyfit(x[m], y[m], 1)[0]:+.2f}   all runs {np.polyfit(x, ya, 1)[0]:+.2f}   (store-everything line: +1.00, theory: 0.00)")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    v = sorted(next(iter(pts.values()))); x = [a[0] for a in v]
    ax.plot(x, [a[3] for a in v], "k--", label=r"$H(H_t\mid\bar O_t)$ (store everything)")
    ax.plot(x, [a[4] for a in v], "k-", lw=2, label=r"$H(\Gamma_t\mid\bar O_t)$ (theory)")
    for (tag, beta, lam), v in sorted(pts.items()):
        v = sorted(v); ax.plot([a[0] for a in v], [a[1] for a in v], "o-", label=f"{tag}-{'RF' if lam > 0 else 'R'} β={beta} (successful runs)")
        ax.plot([a[0] for a in v], [a[2] for a in v], "x:", alpha=0.6, label=f"  {tag} all seeds")
    ax.set_xlabel(r"$\log_2 M$ (hidden modes revealed at identification)"); ax.set_ylabel("bits in gap1"); ax.set_title("P-I: behavioural, not world, complexity sets the memory rate")
    ax.legend(fontsize=7); plt.tight_layout(); out = os.path.splitext(sys.argv[1])[0] + ".png"; plt.savefig(out, dpi=140); print("saved", out)
except Exception as e:
    print("figure skipped:", e)
