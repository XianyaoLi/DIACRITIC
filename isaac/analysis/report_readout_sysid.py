"""Evidence tables for the readout task, the dual-carrier sys-ID control and the forecast label density, computed from the result
files exactly like report_evidence.py.  Usage: python isaac/analysis/report_readout_sysid.py > report_readout_sysid.md"""
import json, os, glob, collections, numpy as np, datetime
R = "isaac/cluster_results/results"; EV = "isaac/cluster_results/eval"
def load(t):
    f = f"{R}/{t}.jsonl"; return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
def cl(a):
    f = f"{EV}/{os.path.basename(a['save_model'])[:-3]}/closedloop_eval.jsonl"
    return json.loads(open(f).readline()) if os.path.exists(f) else None
def pm(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]; return float(np.nanmean([r["per_t"][key][i] for i in idx]))
def phase_all_above(r, key, phase, threshold=0.9):
    vals = [r["per_t"][key][i] for i, p in enumerate(r["phases"]) if p == phase and np.isfinite(r["per_t"][key][i])]
    return bool(vals) and all(v > threshold for v in vals)
late = lambda r: r["summary"]["S_Gam_gap2"] > 0.9 and r["summary"]["S_G_place"] > 0.9
full = lambda r: phase_all_above(r, "S_Gam", "gap1") and phase_all_above(r, "S_Gam", "gap2") and phase_all_above(r, "S_G", "grasp") and phase_all_above(r, "S_G", "place")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def f2(x): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
def mean(xs): return float(np.mean(xs)) if len(xs) else float("nan")
def first_step(r, phase): return next(i for i, p in enumerate(r["phases"]) if p == phase)

print(f"# Evidence tables: readout task, dual-carrier sys-ID, forecast label density ({datetime.date.today()})\n")
print("All numbers are recomputed from `isaac/cluster_results/results/*.jsonl` (teacher-forced metrics), `isaac/cluster_results/eval/*/closedloop_eval.jsonl` (128 own-occupancy episodes per policy) and `isaac/cluster_results/results/rep_*.jsonl` (offline re-evaluation on independently recorded datasets) by `isaac/analysis/report_readout_sysid.py`. Gates: late-gate sufficient = S_Gamma(gap2) > 0.9 and S_G(place) > 0.9; full-trajectory = S_Gamma > 0.9 at every gap step and S_G > 0.9 at every grasp and place step. Wilson 95% intervals in brackets.\n")

# ---------------------------------------------------------------- A. readout task
def readout_section(tag, title, note):
    rows = load(tag)
    if not rows: print(f"## {title}\n\n(no results yet)\n"); return
    print(f"## {title}\n\n{note}\n")
    print("| task | M | learner | K | late-gate | full-traj. | gap1 rate all (suff) | transport rate all (suff) | I(C;theta|O) gap2 | side acc | slot acc | action err (cm) | closed-loop success all (suff) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in rows:
        a = r["args"]; ds = os.path.basename(a["data"]); task, M = ds.split("_M")[0], int(ds.split("_M")[1])
        learner = "sys-ID" if a["aux_theta"] else ("forecast-distilled" if a.get("distill") else "unsupervised -R")
        g[(task, M, learner, a["K"])].append(r)
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3])):
        rs = g[k]; ok = [r for r in rs if late(r)]; fl = [r for r in rs if full(r)]; cls = [cl(r["args"]) for r in rs]; cc = [c for c in cls if c]
        so = [c["success_rate"] for c, r in zip(cls, rs) if c and r in ok]; lo, hi = wilson(len(ok), len(rs))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(ok)}/{len(rs)} [{lo:.2f},{hi:.2f}] | {len(fl)}/{len(rs)} | {f2(mean([r['summary']['HC_gap1'] for r in rs]))} ({f2(mean([r['summary']['HC_gap1'] for r in ok]))}) | {f2(mean([r['summary']['HC_gap2'] for r in rs]))} ({f2(mean([r['summary']['HC_gap2'] for r in ok]))}) | {f2(mean([r['summary']['I_Ctheta_O_gap2'] for r in rs]))} | {f2(mean([c['side_acc'] for c in cc]))} | {f2(mean([c['slot_acc'] for c in cc]))} | {f2(mean([c['action_err_cm'] for c in cc]))} | {f2(mean([c['success_rate'] for c in cc]))} ({f2(mean(so))}) |")
    # theory per task from the first run of each dataset
    print("\nTheory (exact solver on the recorded data; identical at every M): " + "; ".join(f"{t}: H(Gamma|O) gap1 {rs[0]['theory']['H(Gamma|O)'][first_step(rs[0],'gap1')]:.2f}, transport {rs[0]['theory']['H(Gamma|O)'][first_step(rs[0],'gap2')]:.2f}; H(H|O) transport (store everything) {rs[0]['theory']['H(H|O)'][first_step(rs[0],'gap2')]:.2f} at M={m}" for (t, m), rs in sorted({(k[0], k[1]): v for k, v in g.items()}.items()) if m in (32, 512)) + ".\n")
    # forecaster quality from the logs
    fq = collections.defaultdict(list)
    import re
    for f in glob.glob("isaac/cluster_results/logs/train_*.out"):
        s = open(f).read(); m = re.search(r"held-out per-dim forecast mse ([0-9.]+)", s); sp = re.search(r"spec: (\S+) ", s)
        if m and sp and sp.group(1).startswith(tag.replace("b", "b") if False else ("readoutb" if tag == "readoutb" else "readout")) and (tag == "readoutb") == sp.group(1).startswith("readoutb"): fq[sp.group(1)].append(float(m.group(1)))
    if fq: print("Full-history forecaster held-out mse (retrieval check, per dataset, mean over runs): " + ", ".join(f"{d} {np.mean(v):.3f}" for d, v in sorted(fq.items())) + ".\n")

readout_section("readout", "A0. Readout task, analog scalar display (first attempt; retrieval failure at 8 classes)",
                "Franka grasp-transport-place with the hidden mass class theta in [M] shown as ONE scalar channel theta/(M-1) for 3 scan steps (kinematic attach: the mass is never re-revealed). Behaviour: grasp side = quartile of theta (2 bits, used ~3 steps after the reveal), place slot = half (readout2, 1 bit, used ~20 steps later) or octile (readout3, 3 bits). N=448 of 1024 stratified episodes, beta=1e-3, 8 seeds per cell.")
# A (main): offline re-evaluation with the corrected visibility convention (nested classes), train-side (tf_*) and fresh (rep_*)
def readoutb_section():
    tf, rp = [], []
    for ds in ("readoutb2_M32", "readoutb2_M128", "readoutb2_M512", "readoutb3_M32", "readoutb3_M128", "readoutb3_M512"):
        for f, lst in ((f"{R}/tf_{ds}.jsonl", tf), (f"{R}/rep_{ds}.jsonl", rp)):
            if os.path.exists(f): lst += [json.loads(l) for l in open(f)]
    print("## A. Readout task, parallel binary display (main)\n")
    print("Same task as A0; the readout is a 9-channel binary display of theta (LSB-first bits, shown for 3 scan steps), so the class is linearly decodable and retrieval is not the bottleneck (full-history forecaster held-out mse 0.0001-0.0003). readoutb2: 2 -> 1 -> 0 bits (quartile grasp side, half slot); readoutb3: 3 -> 3 -> 0 bits (quartile grasp side, octile slot, 8 slots at >= 21 cm spacing); world complexity log2 M = 5, 7, 9 bits. Because the classes are nested (the slot is a function of the grasp quartile), the visibility convention uses unconditional class means for the nested factor; all numbers below are teacher-forced re-evaluations of the saved models under that convention (`isaac/eval_offline.py`), on the training recordings (train) and on independently recorded datasets (fresh), plus closed-loop success of the same policies.\n")
    if not tf: print("(offline re-evaluation not finished)\n"); return
    def key(r):
        a = r["args"]; ds = os.path.basename(a["data"]); learner = "sys-ID" if a["aux_theta"] else ("forecast-distilled" if a.get("distill") else "unsupervised -R")
        return (ds.split("_M")[0], int(ds.split("_M")[1]), learner, a["K"])
    def fullt(r):
        ph = r["phases"]; sg = r["per_t"]["S_Gam"]; sG = r["per_t"]["S_G"]
        ok = lambda v: (v != v) or v > 0.9
        return all(ok(sg[i]) for i, p in enumerate(ph) if p in ("gap1", "gap2")) and all(ok(sG[i]) for i, p in enumerate(ph) if p in ("grasp", "place"))
    g = collections.defaultdict(list); gr = collections.defaultdict(dict)
    for r in tf: g[key(r)].append(r)
    for r in rp: gr[key(r)][r["model"]] = r
    print("| task | M | learner | K | late-gate train -> fresh | full-traj. train | gap1 rate all (suff) [theory] | transport rate all (suff) [theory] | fresh gap1 rate | I(C;theta|O) gap2 | side acc | slot acc | closed-loop success all (suff) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3])):
        rs = g[k]; ok = [r for r in rs if r["sufficient"]]; fl = [r for r in rs if fullt(r)]; fr = [gr[k].get(r["model"]) for r in rs]; fr = [x for x in fr if x]
        cls = []
        for r in rs:
            f = f"{EV}/{r['model'][:-3]}/closedloop_eval.jsonl"; cls.append(json.loads(open(f).readline()) if os.path.exists(f) else None)
        cc = [c for c in cls if c]; so = [c["success_rate"] for c, r in zip(cls, rs) if c and r["sufficient"]]; lo, hi = wilson(len(ok), len(rs))
        th1 = rs[0]["HGam_gap1"]; th2 = rs[0]["HGam_gap2"]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(ok)}/{len(rs)} [{lo:.2f},{hi:.2f}] -> {sum(r['sufficient'] for r in fr)}/{len(fr)} | {len(fl)}/{len(rs)} | {f2(mean([r['HC_gap1'] for r in rs]))} ({f2(mean([r['HC_gap1'] for r in ok]))}) [{th1:.2f}] | {f2(mean([r['HC_gap2'] for r in rs]))} ({f2(mean([r['HC_gap2'] for r in ok]))}) [{th2:.2f}] | {f2(mean([r['HC_gap1'] for r in fr]))} | {f2(mean([r['I_theta_gap2'] for r in rs]))} | {f2(mean([c['side_acc'] for c in cc]))} | {f2(mean([c['slot_acc'] for c in cc]))} | {f2(mean([c['success_rate'] for c in cc]))} ({f2(mean(so))}) |")
    r0 = next(iter(g.values()))[0]
    print("\nSolver on the fresh recordings vs the training recordings: " + "; ".join(f"{ds}: max |dH(Gamma|O)| {json.loads(open(f'{R}/rep_{ds}.jsonl').readline())['solver_vs_orig']['max_abs_diff_HGam']:.3f}" for ds in ("readoutb2_M32", "readoutb2_M128", "readoutb2_M512", "readoutb3_M32", "readoutb3_M128", "readoutb3_M512") if os.path.exists(f"{R}/rep_{ds}.jsonl")) + ".\n")
readoutb_section()

# ---------------------------------------------------------------- B. dual carrier
rows = load("sysid_dual")
print("## B. Is the sys-ID closed-loop penalty a shared-carrier artefact? (grid sysid_dual, Task A)\n")
print("Task A, M=32 (N=448, every mass in training) and the M=512 full-coverage cell (N=1000). Configurations: (i) shared carrier with the theta head down-weighted to 0.1 (K = 256 / 1024); (ii) DUAL carrier: behavioural code C (K=16, read by the policy only) + identification code C2 (K2 = 256 / 1024, read by the theta head only), shared observation/action encoders, both rate-penalised; (iii) dual + stop-gradient (no theta gradient into the shared encoders). Reference rows from the earlier grids: plain DIACRITIC (K=16) and shared-carrier sys-ID at weight 1.0.\n")
print("| dataset | N | configuration | place-sufficient | behavioural code H(C|O) transport | I(C;mass|O) | I(C;theta|O) | identification code H(C2|O) | I(C2;theta|O) | closed-loop success (mean, min) | slot acc |\n|---|---|---|---|---|---|---|---|---|---|---|")
def ref_rows():
    out = []
    for t, sel in (("taskA", lambda a: os.path.basename(a["data"]) == "taskA2_M32" and a["n_train"] == 448 and a["beta"] == 0.001 and a["K"] in (16, 256)),
                   ("taskA_cov", lambda a: os.path.basename(a["data"]) == "taskA2_M512" and a["n_train"] == 1000)):
        for r in load(t):
            if sel(r["args"]) and not r["args"].get("sym_input") and not r["args"].get("distill"): out.append(r)
    return out
g = collections.defaultdict(list)
for r in ref_rows() + rows:
    a = r["args"]; ds = os.path.basename(a["data"]); K2 = a.get("dual_K", 0)
    if K2: cfg = f"dual carrier K2={K2}" + (" + stop-gradient" if a.get("dual_detach") else "")
    elif a["aux_theta"] == 0: cfg = f"DIACRITIC (reference, K={a['K']})"
    else: cfg = f"shared carrier, theta weight {a['aux_theta']:g}, K={a['K']}"
    g[(ds, a["n_train"], cfg)].append(r)
for k in sorted(g, key=lambda k: (k[0], k[1], k[2].startswith("DIAC") is False, k[2])):
    rs = g[k]; cls = [cl(r["args"]) for r in rs]; cc = [c for c in cls if c]; s = lambda key: mean([r["summary"].get(key, float("nan")) for r in rs])
    n_ok = sum(r["summary"]["S_G_place"] > 0.9 for r in rs)
    print(f"| {k[0]} | {k[1]} | {k[2]} | {n_ok}/{len(rs)} | {f2(s('HC_gap2'))} | {f2(s('I_Cmass_O_gap2'))} | {f2(s('I_Ctheta_O_gap2'))} | {f2(s('HC2_gap2'))} | {f2(s('I_C2theta_O_gap2'))} | {f2(mean([c['success_rate'] for c in cc]))}, {f2(min([c['success_rate'] for c in cc]) if cc else np.nan)} | {f2(mean([c['slot_acc'] for c in cc]))} |")
# replication
rep = []
for f in glob.glob(f"{R}/rep_sysid_dual_M*.jsonl"): rep += [json.loads(l) for l in open(f)]
if rep:
    print("\nFresh-data replication (teacher-forced re-evaluation of the same models on independently recorded datasets rep_taskA2_M32 / rep_taskA2_M512, solver rebuilt from the new recordings):\n")
    print("| dataset | configuration | place-sufficient train -> fresh | H(C|O) transport train -> fresh | I(C;mass|O) grasp, fresh |\n|---|---|---|---|---|")
    gg = collections.defaultdict(list)
    for r in rep:
        m = r.get("model", ""); name = os.path.basename(m).replace(".pt", "")
        cfg = ("dual" + ("+sg" if "dual" in name and name.split("_dual")[1].split("_")[0].endswith("d") else "")) if "_dual" in name else ("theta weight 0.1" if "_aux0.1_" in name else "other")
        gg[(os.path.basename(r.get("data", "")), cfg)].append(r)
    for k in sorted(gg):
        rs = gg[k]; tr = [r["train_summary"] for r in rs]
        n_tr = sum(1 for t in tr if t.get("S_G_place") is not None and t["S_G_place"] > 0.9); n_fr = sum(1 for r in rs if r.get("sufficient"))
        print(f"| {k[0]} | {k[1]} | {n_tr}/{len(rs)} -> {n_fr}/{len(rs)} | {f2(mean([t.get('HC_gap2', np.nan) for t in tr]))} -> {f2(mean([r.get('HC_gap2', np.nan) for r in rs]))} | {f2(mean([r.get('I_mass_grasp', np.nan) for r in rs]))} |")
print()

# ---------------------------------------------------------------- C. forecast label density
rows = load("fc_stride")
print("## C. Forecast supervision with sub-sampled labels (grid fc_stride, A' gap 20)\n")
print("Forecast distillation on tier4_gap20 (reveal -> place distance 33 steps, the hardest tested horizon) with the student's distillation loss applied only every s-th step after the last reveal step (the forecaster itself is trained on all steps); s=1 is the paper's configuration (grid distill).\n")
print("| stride s | labels per episode (student) | sufficient | gap1 rate (suff) | gap2 rate (suff) | class error b2 (first place step) | closed-loop success (mean, min) |\n|---|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for r in rows: g[r["args"]["distill_stride"]].append(r)
for r in load("distill"):
    a = r["args"]
    if os.path.basename(a["data"]) == "tier4_gap20" and a.get("distill_target", "forecast") == "forecast" and a.get("distill") and a["K"] == 16 and a["beta"] == 0.001: g[1].append(r)
for s_ in sorted(g):
    rs = g[s_]; ok = [r for r in rs if late(r)]; cls = [cl(r["args"]) for r in rs]; cc = [c for c in cls if c]
    nlab = {1: 3 + 20, 2: 2 + 10, 4: 1 + 5}.get(s_, "?")   # (rev..grasp, rev..place) supervised steps at gap1=2 (grasp 3 steps after the reveal) and gap 20
    ce = mean([r["per_t"]["class_err_test"][first_step(r, "place")] for r in rs])
    print(f"| {s_} | {nlab} | {len(ok)}/{len(rs)} | {f2(mean([r['summary']['HC_gap1'] for r in ok]))} | {f2(mean([r['summary']['HC_gap2'] for r in ok]))} | {f2(ce)} | {f2(mean([c['success_rate'] for c in cc]))}, {f2(min([c['success_rate'] for c in cc]) if cc else np.nan)} |")
print("\n## D. Certified upper bound on the non-transitive corridor\n\nSee the output of `isaac/analysis/w4_upper_bound.py`: an explicit closed compatible assignment attains 2 / 1 bits in the halls for W = 3, 5, 7, tightening the bracket from [0, 2+log2 W] to [0, 2] with the top achievable.\n")
