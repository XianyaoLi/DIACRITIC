"""Credit-assignment horizon curve: fraction of seeds reaching sufficiency vs the number of steps between the reveal of a class and its
first use, pooled from all A' result files.  beta_2 horizon = gap1 + grasp block + gap2 (reveal -> place); beta_1 horizon = gap1 (reveal -> grasp).
Usage: python isaac/horizon_curve.py isaac/cluster_results/results/*.jsonl isaac/results/*.jsonl"""
import sys, json, os, re, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
for f in sys.argv[1:]:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        a = r.get("args", {}); ds = os.path.basename(a.get("data", "")); ph = r.get("phases")
        if not ph or "gap2" not in ph: continue
        first_place = ph.index("place"); reveal_end = max(i for i, p in enumerate(ph) if p == "scan"); first_grasp = ph.index("grasp")
        var = a.get("variant", "diacritic") + ("-RF" if a.get("lam", 0) > 0 else "-R") + ("+ref" if a.get("refine") else "") + ("+curric" if a.get("init_from") else "") + ("+sysid" if a.get("aux_theta") else "")
        d = r["summary"]
        rows.append(dict(ds=ds, var=var, beta=a["beta"], h2=first_place - reveal_end, h1=first_grasp - reveal_end,
                         ok2=d["S_Gam_gap2"] > 0.9 and d["S_G_place"] > 0.9, ok1=d["S_G_grasp"] > 0.9))
g = collections.defaultdict(list)
for r in rows: g[(r["var"], r["beta"], r["ds"], r["h2"], r["h1"])].append(r)
print(f"{'variant':22s} {'beta':>6s} {'dataset':16s} h(beta1) reach(beta1)   h(beta2) reach(beta2)   n")
pts = collections.defaultdict(list)
for (var, beta, ds, h2, h1), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3])):
    f1 = np.mean([r["ok1"] for r in rs]); f2 = np.mean([r["ok2"] for r in rs])
    print(f"{var:22s} {beta:6.4f} {ds:16s}   {h1:3d}     {f1:4.2f}          {h2:3d}     {f2:4.2f}      {len(rs)}")
    if "pI" not in ds and "2k" not in ds: pts[(var, beta)].append((h2, f2, len(rs))); pts[(var, beta)].append((h1, f1, len(rs)))
fig, ax = plt.subplots(figsize=(7, 4.2))
for (var, beta), v in sorted(pts.items()):
    if beta not in (0.002, 0.001): continue
    v = sorted(set(v)); ax.plot([a[0] for a in v], [a[1] for a in v], "o-", label=f"{var} β={beta:g}")
ax.set_xlabel("steps from reveal to first use (β₁: grasp; β₂: place)"); ax.set_ylabel("fraction of seeds reaching sufficiency (teacher-forced)")
ax.set_title("credit-assignment horizon of anticipatory discrete memory (A′)"); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("isaac/results/fig_horizon.png", dpi=140); print("saved isaac/results/fig_horizon.png")
