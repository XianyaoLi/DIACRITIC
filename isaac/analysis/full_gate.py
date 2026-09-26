"""Compare the relaxed, full, and strict empirical sufficiency gates on the result ledgers.
Relaxed = mean S_Gamma over gap2 > 0.9 and mean S_G over place > 0.9.
The paper's FULL gate checks S_Gamma at every positive-requirement gap step and S_G at the first
positive-requirement grasp and place decisions. The stricter gate checks all such episode/decision steps.
Passing a threshold is an empirical diagnostic, not a certificate of exact zero distortion.
Usage: python isaac/analysis/full_gate.py [results dir] [--training-only] > results_md/full_gate.md
  --training-only  count each trained model once: skip the teacher-forced re-evaluation ledgers (rep_*, tf_*) and the timing
                   benchmark; this is the sample of Appendix B.5 (A'-family training runs; the weighing task has no relaxed gate).
The normalisation-control ledgers (normctl, rep_normctl) are tabulated but never enter the totals.
"""
import sys, os, json, glob, math
from collections import defaultdict
import numpy as np
TRAIN_ONLY = "--training-only" in sys.argv[1:]; _args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = _args[0] if _args else os.path.join(os.path.dirname(__file__), "..", "cluster_results", "results")
CONTROL = ("normctl", "rep_normctl")
TH = 0.9
def gates(r):
    ph, p = r["phases"], r["per_t"]; SGam, SG, HC = p["S_Gam"], p["S_G"], p["H(C|O)"]
    # eval_offline ledgers (tf_*, rep_*) do not store the per-step requirement; S is NaN exactly where the requirement is zero
    HGam = p.get("H(Gam|O)") or [0.0 if math.isnan(x) else 1.0 for x in SGam]; HG = p.get("H(G|O)") or [0.0 if math.isnan(x) else 1.0 for x in SG]
    idx = lambda name: [i for i, q in enumerate(ph) if q == name]
    nm = lambda xs: float(np.nanmean(xs)) if len(xs) and not all(math.isnan(x) for x in xs) else float("nan")
    old = nm([SGam[i] for i in idx("gap2")]) > TH and nm([SG[i] for i in idx("place")]) > TH
    mem = [i for i in idx("gap1") + idx("gap2") if HGam[i] > 1e-9]
    allmem = [i for i in range(len(ph)) if HGam[i] > 1e-9]                      # every step with a positive exact requirement (incl. scan / grasp / transport steps)
    # decision steps = the FIRST grasp step and the FIRST place step with a positive behavioural requirement (the steps at which the remembered
    # class must be read out).  Later steps of a block are excluded: there the executed class is already visible in the raw observation (arm pose),
    # and on the readout family the symbolic convention deliberately does not expose it (nested classes), so S_G against O-bar is not a memory test.
    act = [min(js) for js in ([i for i in idx("grasp") if HG[i] > 1e-9], [i for i in idx("place") if HG[i] > 1e-9]) if js]
    act_all = [i for i in idx("grasp") + idx("place") if HG[i] > 1e-9]
    full = all(SGam[i] > TH for i in mem) and all(SG[i] > TH for i in act)
    strict = all(SGam[i] > TH for i in allmem) and all(SG[i] > TH for i in act_all)
    g1 = idx("gap1"); g2 = idx("gap2")
    return dict(has_theory=("H(Gam|O)" in p), old=bool(old), full=bool(full), strict=bool(strict), minS_gap1=min([SGam[i] for i in g1 if HGam[i] > 1e-9] or [float("nan")]), minS_gap2=min([SGam[i] for i in g2 if HGam[i] > 1e-9] or [float("nan")]),
                HC_gap1=nm([HC[i] for i in g1]), HC_gap2=nm([HC[i] for i in g2]), HGam_gap1=nm([HGam[i] for i in g1]), HGam_gap2=nm([HGam[i] for i in g2]))
def cell(r):
    a = dict(r["args"]); ds = os.path.basename((r.get("data") or a["data"]).rstrip("/")); tag = []
    if "model" in r:    # eval_offline ledger: recover the cell from the model file name
        m = r["model"]; a.setdefault("variant", "scaffold" if "_scaffold_" in m else "bypass" if "_bypass_" in m else "transformer" if "_transformer_" in m else "diacritic")
        a.setdefault("lam", float(m.split("_lam")[1].split("_")[0])); a.setdefault("steps", 10000)
        if a.get("distill"): a.setdefault("distill_target", m.split("_distill")[1].split("_")[0].replace(".pt", "") if "_distill" in m else "forecast")
        if "_symin" in m: a.setdefault("sym_input", int(m.split("_symin")[1][0]))
        if "_ds" in m and m.split("_ds")[1][0].isdigit(): a["distill_stride"] = int(m.split("_ds")[1].split("_")[0].replace(".pt", ""))
        for k_, suf in (("distill_source", "_src"), ("fc_corrupt", "_cor"), ("fc_steps", "_fcs"), ("distill_w", "_dw"), ("distill_off", "_off")):
            pass
        if "_fromzero" in m: a["fc_start"] = "zero"
        for k_, suf in (("distill_source", "_src"), ("fc_corrupt", "_cor"), ("fc_steps", "_fcs"), ("distill_w", "_dw"), ("distill_off", "_off")):
            if suf in m: v_ = m.split(suf)[1].split("_")[0].replace(".pt", ""); a[k_] = v_ if k_ == "distill_source" else float(v_)
    v = a["variant"]
    if a.get("aux_theta", 0) > 0: tag.append(f"sysID(K2={a.get('dual_K', 0)})" if a.get("dual_K", 0) else "sysID")
    if a.get("aux_join", 0) > 0: tag.append("oracle")
    if a.get("aux_mik", 0) > 0: tag.append("mik" + ("-only" if a.get("mik_detach") else "+imit"))
    if a.get("distill"): tag.append("distill:" + a.get("distill_target", "feat") + ("" if a.get("distill_source", "teacher") == "teacher" else "-gt") + (f"/s{a['distill_stride']}" if (a.get("distill_stride") or 1) > 1 else "")
                                    + (f"/cor{a['fc_corrupt']}" if (a.get("fc_corrupt") or 0) > 0 else "") + (f"/fcs{int(a['fc_steps'])}" if (a.get("fc_steps") or 3000) != 3000 else "") + (f"/w{a['distill_w']}" if (a.get("distill_w") or 1) != 1 else "") + (f"/off{a['distill_off']}" if (a.get("distill_off") or 0) > 0 else "") + ("/from0" if a.get("fc_start") == "zero" else ""))
    if a.get("sym_input", 0): tag.append(f"symin{a['sym_input']}" + (f"/hw{a['high_w']}" if a.get("high_w", 1) != 1 else ""))
    if a.get("refine"): tag.append("refine-" + a["refine"])
    if a.get("init_from"): tag.append("curric")
    if a.get("bfs_noobs"): tag.append("noobs")
    if a.get("pixels"): tag.append("px")
    name = ("-RF" if a["lam"] > 0 else "-R") if v == "diacritic" else (v + ("-RF" if a["lam"] > 0 else "-R") if v == "scaffold" else v)
    return (ds, name + ("[" + ",".join(tag) + "]" if tag else ""), a["K"], a["beta"], a["n_train"], a["steps"])
rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
    grid = os.path.basename(f)[:-6]
    if grid.startswith(("smoke", "closedloop")) or grid in ("readout", "readoutb"): continue
    if TRAIN_ONLY and grid.startswith(("rep_", "tf_", "timing")): continue    # readout family: training-time ledger predates the nested-class visibility fix -> use tf_* / rep_*
    seen = {}
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if "per_t" not in r or "phases" not in r or "gap2" not in r["phases"]: continue
        ds = os.path.basename((r.get("data") or r["args"]["data"]).rstrip("/"))
        if ds.startswith("taskA"): continue                                       # Task A has its own (placement) gate: exact requirement is zero
        seen[(cell(r), r["args"]["seed"])] = r                                    # last record wins (requeued jobs)
    for (c, seed), r in seen.items(): rows[grid][c].append(gates(r))
print("# Full-trajectory sufficiency gate\n")
print("Relaxed gate = mean S_Gamma(gap2) > 0.9 and mean S_G(place) > 0.9.  **Full gate** = S_Gamma(t) > 0.9 at every step of gap1 and gap2 with a positive exact requirement, and S_G > 0.9 at the first grasp step and the first place step (the two memory-dependent decisions)."
      "  Strict = S_Gamma > 0.9 at every step of the episode with H(Gamma|O) > 0 and S_G > 0.9 at every grasp/place step with H(G|O) > 0 (on the readout family this also penalises dropping a class while the arm pose shows it).  Rates (bits) are means over the seeds that pass the FULL gate; theory in parentheses.\n")
tot = dict(n=0, old=0, full=0, strict=0, lost=0, gained=0)
for grid in sorted(rows):
    print(f"\n## {grid}\n\n| dataset | learner | K | beta | N | seeds | relaxed gate | full gate | strict | H(C\\|O) gap1 (full-gate seeds) | gap2 | min_t S_Gamma gap1 of relaxed-gate seeds |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(rows[grid], key=str):
        g = rows[grid][c]; n = len(g); o = sum(x["old"] for x in g); fu = sum(x["full"] for x in g); st = sum(x["strict"] for x in g); ok = [x for x in g if x["full"]]
        if not grid.startswith(CONTROL): tot["n"] += n; tot["old"] += o; tot["full"] += fu; tot["strict"] += st; tot["lost"] += sum(x["old"] and not x["full"] for x in g); tot["gained"] += sum(x["full"] and not x["old"] for x in g)   # (one statement: all increments guarded)
        th = (lambda k_: f" ({g[0][k_]:.2f})" if g[0][k_] not in (0.0, 1.0) or "H(Gam|O)" in str(k_) else "")
        r1 = f"{np.mean([x['HC_gap1'] for x in ok]):.2f}" + (f" ({g[0]['HGam_gap1']:.2f})" if g[0].get("has_theory") else "") if ok else "--"; r2 = f"{np.mean([x['HC_gap2'] for x in ok]):.2f}" + (f" ({g[0]['HGam_gap2']:.2f})" if g[0].get("has_theory") else "") if ok else "--"
        ms = [x["minS_gap1"] for x in g if x["old"]]; msr = f"{min(ms):.2f}..{max(ms):.2f}" if ms else "--"
        print(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {n} | {o}/{n} | **{fu}/{n}** | {st}/{n} | {r1} | {r2} | {msr} |")
print(f"\n## Totals ({'training runs only: rep_*/tf_*/timing skipped' if TRAIN_ONLY else 'all ledgers'}; normalisation-control ledgers excluded)\n\n{tot['n']} A'-family runs: relaxed gate {tot['old']}, full gate {tot['full']}, strict {tot['strict']}; pass the relaxed gate but fail the full gate: {tot['lost']}; pass full but not paper: {tot['gained']}.")
