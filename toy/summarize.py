"""Aggregate sweep results: per-phase S_Gamma, distortion at use2, rate profiles vs Gamma; figure for gap sweep."""
import json, sys, numpy as np, math
from collections import defaultdict
CURRENT=[]

def load(name):
    return [json.loads(l) for l in open(f"results/{name}.jsonl")]

def phase_mean(res, key, ph, which):
    v = res[key]
    if isinstance(ph[which], list):
        a, b = ph[which]; xs = v[a-1:b]
    else:
        xs = [v[ph[which]-1]]
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")

def agg(rows, keyf):
    g = defaultdict(list)
    for r in rows: g[keyf(r["job"])].append(r)
    return dict(sorted(g.items()))

def ms(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return (np.mean(xs), np.std(xs)) if xs else (float("nan"), float("nan"))

def gap_table(name="gap_sweep"):
    rows = load(name)
    groups = agg(rows, lambda j: (j["beta"], j["gap2"], ("RF" if j["lam"] > 0 else "R") + ("+" + j["refine"] if j.get("refine") else "")))
    print(f"\n=== {name}: per-phase means over seeds (gap toy: reveal->gap1->use1->gap2->use2) ===")
    print(" beta  gap2 var | S_Gam@gap1 S_Gam@use1 S_Gam@gap2 | D_TV@use1 D_TV@use2 | H(C|O)@gap1 [Gam] H(C|O)@gap2 [Gam] | H(C|Gam,O) I(C;z|O) | codes")
    for (beta, gap2, var), rs in groups.items():
        ph = rs[0]["phases"]
        f = lambda key, w: ms([phase_mean(r["res"], key, ph, w) for r in rs])
        HGam1 = phase_mean(rs[0]["res"], "H(Gam|O)", ph, "gap1"); HGam2 = phase_mean(rs[0]["res"], "H(Gam|O)", ph, "gap2")
        sg1, su1, sg2 = f("S_Gam", "gap1"), f("S_Gam", "use1"), f("S_Gam", "gap2")
        d1, d2 = f("D_TV", "use1"), f("D_TV", "use2")
        hc1, hc2 = f("H(C|O)", "gap1"), f("H(C|O)", "gap2")
        hcg = ms([r["res"]["mean"]["H(C|Gam,O)"] for r in rs]); iz = ms([r["res"]["mean"]["I(C;z|O)"] for r in rs])
        codes = np.mean([r["res"]["codes_used"] for r in rs])
        print(f"{beta:5.3f} {gap2:4d}  {var:6s} | {sg1[0]:5.2f}±{sg1[1]:4.2f} {su1[0]:5.2f}±{su1[1]:4.2f} {sg2[0]:5.2f}±{sg2[1]:4.2f} | "
              f"{d1[0]:6.3f}   {d2[0]:6.3f}±{d2[1]:5.3f} | {hc1[0]:5.2f}±{hc1[1]:4.2f} [{HGam1:4.2f}] {hc2[0]:5.2f}±{hc2[1]:4.2f} [{HGam2:4.2f}] | "
              f"{hcg[0]:5.2f}     {iz[0]:5.2f}   | {codes:4.1f}")
    return groups

def reveal_table(name="beta_sweep_reveal"):
    rows = load(name)
    groups = agg(rows, lambda j: (j["beta"], "RF" if j["lam"] > 0 else "R"))
    print(f"\n=== {name}: reveal toy, per-t H(C|O) (target Gamma=[0,0,1,2]) ===")
    print(" beta  var | H(C|O) per t (mean over seeds)      | S_Gam@3 S_G@4 | D_TV mean | H(C|Gam,O) I(C;z|O) | codes")
    for (beta, var), rs in groups.items():
        hc = np.mean([r["res"]["H(C|O)"] for r in rs], 0)
        sg3 = ms([r["res"]["S_Gam"][2] for r in rs]); sg4 = ms([r["res"]["S_G"][3] for r in rs])
        d = ms([r["res"]["mean"]["D_TV"] for r in rs]); hcg = ms([r["res"]["mean"]["H(C|Gam,O)"] for r in rs]); iz = ms([r["res"]["mean"]["I(C;z|O)"] for r in rs])
        print(f"{beta:5.3f}  {var:2s} | " + " ".join(f"{x:4.2f}" for x in hc) + f"   | {sg3[0]:5.2f}  {sg4[0]:5.2f} | {d[0]:6.3f}   | {hcg[0]:5.2f}      {iz[0]:5.2f}  | {np.mean([r['res']['codes_used'] for r in rs]):4.1f}")

def fs_gap_table(name="fs_gap"):
    """Finite-sample gap toy (n_train x beta x variant): per-phase rates vs Gamma, nuisance MI, distortion."""
    rows = load(name)
    groups = agg(rows, lambda j: (j.get("n_train", 0), j["beta"], "RF" if j["lam"] > 0 else "R"))
    ph = rows[0]["phases"]; r0 = rows[0]["res"]
    HGam1 = phase_mean(r0, "H(Gam|O)", ph, "gap1"); HGam2 = phase_mean(r0, "H(Gam|O)", ph, "gap2")
    print(f"\n=== {name}: finite-sample gap toy (gap2={rows[0]['job']['gap2']}, p={rows[0]['job']['p']}); "
          f"targets H(Gam|O)@gap1={HGam1:.2f}, @gap2={HGam2:.2f}, H(G|O)@gap=0 ===")
    print("  n   beta  var  n_seeds | S_Gam@gap1 S_Gam@gap2 | H(C|O)@gap1 H(C|O)@gap2 | H(C|Gam,O) I(C;z|O) | D_TV@use1 D_TV@use2 | succ | codes")
    for (n, beta, var), rs in groups.items():
        f = lambda key, w: ms([phase_mean(r["res"], key, ph, w) for r in rs])
        sg1, sg2 = f("S_Gam", "gap1"), f("S_Gam", "gap2"); hc1, hc2 = f("H(C|O)", "gap1"), f("H(C|O)", "gap2")
        hcg = ms([r["res"]["mean"]["H(C|Gam,O)"] for r in rs]); iz = ms([r["res"]["mean"]["I(C;z|O)"] for r in rs])
        d1, d2 = f("D_TV", "use1"), f("D_TV", "use2")
        succ = np.mean([r["res"]["D_TV"][ph["use2"]-1] < 0.05 for r in rs])
        codes = np.mean([r["res"]["codes_used"] for r in rs])
        print(f"{n:4d}  {beta:5.3f}  {var:2s}   {len(rs):2d}     | {sg1[0]:5.2f}±{sg1[1]:4.2f} {sg2[0]:5.2f}±{sg2[1]:4.2f} | "
              f"{hc1[0]:5.2f}±{hc1[1]:4.2f} {hc2[0]:5.2f}±{hc2[1]:4.2f} | {hcg[0]:5.2f}      {iz[0]:5.2f}   | "
              f"{d1[0]:6.3f}    {d2[0]:6.3f}   | {succ:4.2f} | {codes:4.1f}")
    return groups

def leak_table(name="leak_reveal"):
    """Reveal toy with Bernoulli reveal loss (A2 fails at T): solver flags a2=False; the learned code targets the
    observational conditional; report per-t H(C|O) vs H(Gam|O), S_Gam, D_TV."""
    rows = load(name)
    groups = agg(rows, lambda j: (j.get("leak", 0.0), j["beta"], "RF" if j["lam"] > 0 else "R"))
    print(f"\n=== {name}: reveal toy + Bernoulli leak (A2 violated at T) ===")
    print(" leak  beta  var | H(C|O) per t        | H(Gam|O) per t      | S_Gam@T  D_TV mean | H(C|Gam,O) I(C;z|O) | codes")
    for (leak, beta, var), rs in groups.items():
        hc = np.mean([r["res"]["H(C|O)"] for r in rs], 0); hg = rs[0]["res"]["H(Gam|O)"]
        sgT = ms([r["res"]["S_Gam"][-1] for r in rs]); d = ms([r["res"]["mean"]["D_TV"] for r in rs])
        hcg = ms([r["res"]["mean"]["H(C|Gam,O)"] for r in rs]); iz = ms([r["res"]["mean"]["I(C;z|O)"] for r in rs])
        print(f"{leak:5.2f} {beta:5.3f}  {var:2s} | " + " ".join(f"{x:4.2f}" for x in hc) + " | " + " ".join(f"{x:4.2f}" for x in hg)
              + f" | {sgT[0]:5.2f}    {d[0]:6.3f}   | {hcg[0]:5.2f}      {iz[0]:5.2f}  | {np.mean([r['res']['codes_used'] for r in rs]):4.1f}")

def gap_figure(groups, beta=0.03, out="results/fig_gap_sweep.png"):
    out = out.replace("fig_gap_sweep", "fig_" + CURRENT[0]) if CURRENT else out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gaps = sorted({g for (b, g, v) in groups if b == beta})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    succ_of = lambda r: r["res"]["D_TV"][r["phases"]["use2"]-1] < 0.05
    for var, c in [("R", "C0"), ("RF", "C3")]:
        sr = [np.mean([succ_of(r) for r in groups[(beta, g, var)]]) for g in gaps]
        ns = [len(groups[(beta, g, var)]) for g in gaps]
        ax[0].plot(gaps, sr, "o-", color=c, label=f"DIACRITIC-{var} (n={ns[0]} seeds)")
        # rate during gap2 for successful runs vs Gamma
        hc = [ms([phase_mean(r["res"], "H(C|O)", r["phases"], "gap2") for r in groups[(beta, g, var)] if succ_of(r)]) for g in gaps]
        ax[1].errorbar(gaps, [m for m, s in hc], [s for m, s in hc], fmt="o-", color=c, capsize=3, label=f"$H(C_t|O_t)$ in gap2, DIACRITIC-{var} (successful runs)")
    ax[0].set_xlabel("gap2 length (steps between use1 and use2)"); ax[0].set_ylabel("P(near-zero distortion at use2)"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(); ax[0].set_title("BFS vs BPTT-only: reliability vs memory gap")
    r0 = next(iter(groups.values()))[0]["res"]
    ax[1].axhline(phase_mean(r0, "H(Gam|O)", next(iter(groups.values()))[0]["phases"], "gap2"), color="k", ls="-", label=r"$H(\Gamma_t|O_t)$ in gap2 (theory)")
    ax[1].axhline(0.0, color="k", ls=":", label=r"$H(G_{E,t}|O_t)$ in gap2 (theory)")
    ax[1].set_xlabel("gap2 length"); ax[1].set_ylabel("bits"); ax[1].set_title("expired information: rate after use1"); ax[1].legend(fontsize=7)
    g = 6 if 6 in gaps else max(gaps)
    for var, c in [("R", "C0"), ("RF", "C3")]:
        rs = [r for r in groups[(beta, g, var)] if succ_of(r)]
        if not rs: continue
        hc = np.array([r["res"]["H(C|O)"] for r in rs])
        ax[2].errorbar(range(1, hc.shape[1] + 1), hc.mean(0), hc.std(0), fmt="o-", color=c, capsize=2, label=f"$H(C_t|O_t)$ DIACRITIC-{var} ({len(rs)} succ. runs)")
    r0 = groups[(beta, g, "RF")][0]["res"]; T = len(r0["H(G|O)"])
    ax[2].step(range(1, T + 1), r0["H(G|O)"], where="mid", color="k", ls=":", label=r"$H(G_{E,t}|O_t)$")
    ax[2].step(range(1, T + 1), r0["H(Gam|O)"], where="mid", color="k", ls="-", label=r"$H(\Gamma_t|O_t)$")
    ax[2].step(range(1, T + 1), r0["H(H|O)"] if "H(H|O)" in r0 else [np.nan]*T, where="mid", color="gray", ls="--", label=r"$H(H_t|O_t)$")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("bits"); ax[2].set_title(f"conditional context rate over time (gap2={g})"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out, dpi=140); print("saved", out)

if __name__ == "__main__":
    which = sys.argv[1:] or ["gap_sweep", "beta_sweep_reveal"]
    for w in which:
        if w.startswith("fs_gap"):
            pass
        if w.startswith("fs_gap"):
            fs_gap_table(w)
        elif w.startswith("leak"):
            leak_table(w)
        elif w.startswith("gap") or w == "beta_sweep_gap" or "refine" in w:
            CURRENT[:] = [w]; g = gap_table(w)
            try: gap_figure(g)
            except Exception as e: print('figure skipped:', e)
        elif w in ("ablation", "bypass_beta", "bypass_tune"):
            print(f"(use: python ablation_table.py {w})")
        else:
            reveal_table(w)
