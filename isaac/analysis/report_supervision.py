"""Evidence tables for the supervision, pixel, full-gate, weighing and T-maze experiments, generated from the ledgers only (no hand-typed numbers).
Sources: isaac/cluster_results/results/{generic,generic2,teacher,px_fc,px_fc20,weigh,weigh2,weigh3}.jsonl (training-time per-step metrics),
rep_*.jsonl (teacher-forced re-evaluation of the saved models on independent recordings, isaac/eval_offline.py),
isaac/cluster_results/eval/<model>/summary.json (closed loop, 128 episodes per model), external/results/tmaze.jsonl.
Gates: relaxed = mean S_Gamma(gap2) > 0.9 and mean S_G(place) > 0.9;  FULL = S_Gamma(t) > 0.9 at every gap1/gap2 step with a positive exact
requirement and S_G > 0.9 at the first grasp and first place step with a positive requirement (on the weigh task: gap1 + the grasp decision).
Usage: python isaac/analysis/report_supervision.py > report_supervision.md"""
import os, re, json, glob, math
from collections import defaultdict
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."); RES = os.path.join(ROOT, "isaac", "cluster_results", "results"); EVAL = os.path.join(ROOT, "isaac", "cluster_results", "eval")
TH = 0.9
def nm(xs): xs = [x for x in xs if not math.isnan(x)]; return float(np.mean(xs)) if xs else float("nan")
def gates(per, ph):
    S, SG, HC = per["S_Gam"], per["S_G"], per["H(C|O)"]; pos = lambda v: [not math.isnan(x) for x in v]
    need, needg = (([h > 1e-9 for h in per["H(Gam|O)"]], [h > 1e-9 for h in per["H(G|O)"]]) if "H(Gam|O)" in per else (pos(S), pos(SG)))
    idx = lambda n: [i for i, q in enumerate(ph) if q == n]
    paper = nm([S[i] for i in idx("gap2")]) > TH and nm([SG[i] for i in idx("place")]) > TH
    mem = [i for i in idx("gap1") + idx("gap2") if need[i]]; act = [min(j) for j in ([i for i in idx("grasp") if needg[i]], [i for i in idx("place") if needg[i]]) if j]
    full = bool(mem) and all(S[i] > TH for i in mem) and all(SG[i] > TH for i in act)
    return bool(paper), bool(full), nm([HC[i] for i in idx("gap1")]), nm([HC[i] for i in idx("gap2")])
def label(a, model=""):
    m = model[:-3] if model.endswith(".pt") else model
    if not a.get("variant"): a = dict(a, variant=("transformer" if "_transformer_" in m else "bypass" if "_bypass_" in m else "diacritic"))
    if a["variant"] != "diacritic": return {"transformer": "full-history Transformer (VQ read-out)", "bypass": "GRU bypass (VQ read-out)"}.get(a["variant"], a["variant"])
    if (a.get("aux_theta") or 0) > 0: return "system identification (class head)"
    if not a.get("distill"): return "plain -R (unsupervised)"
    tgt = a.get("distill_target") or (re.search(r"_distill([a-z]+)", m).group(1) if "_distill" in m else "forecast")
    src = a.get("distill_source") or ("gt" if "_srcgt" in m else "teacher"); off = a.get("distill_off") or (float(re.search(r"_off([0-9.]+)", m).group(1)) if "_off" in m else 0); dw = a.get("distill_w") or (float(re.search(r"_dw([0-9.]+)", m).group(1)) if "_dw" in m else 1)
    cor = a.get("fc_corrupt") or (float(re.search(r"_cor([0-9.]+)", m).group(1)) if "_cor" in m else 0); fcs = a.get("fc_steps") or (int(float(re.search(r"_fcs([0-9.]+)", m).group(1))) if "_fcs" in m else 3000)
    start = a.get("fc_start") or ("zero" if "_fromzero" in m else "reveal"); st = a.get("distill_stride") or 1
    name = {"forecast": "targeted forecast", "random": "generic: random future offset", "uniform": "generic: all offsets (sum)", "feat": "teacher state"}[tgt]
    if tgt == "forecast" and start == "zero": name += ", supervised from step 0"
    if src == "gt": name += ", recorded actions as targets (no teacher)"
    if dw != 1: name += f", weight {dw:g}"
    if off: name += ", annealed" + ("" if abs(float(off) - 0.4) < 1e-9 else f" from {100 * float(off):g}%")
    if cor: name += f", {100 * cor:g}% wrong-class forecasts"
    if fcs != 3000: name += f", forecaster trained {int(fcs)} steps"
    if st > 1: name += f", stride {st}"
    return name
def load(grid):
    out = {}; f = os.path.join(RES, grid + ".jsonl")
    if not os.path.exists(f): return []
    for line in open(f):
        r = json.loads(line)
        if "per_t" not in r: continue
        a = r["args"]; model = r.get("model", os.path.basename(a.get("save_model") or "")); ds = os.path.basename((r.get("data") or a["data"]).rstrip("/"))
        p, fu, r1, r2 = gates(r["per_t"], r["phases"]); th = r.get("theory", {}).get("H(Gamma|O)")
        t1 = nm([th[i] for i, q in enumerate(r["phases"]) if q == "gap1"]) if th else float("nan"); t2 = nm([th[i] for i, q in enumerate(r["phases"]) if q == "gap2"]) if th else float("nan")
        per = r["per_t"]; sc = [i for i, q in enumerate(r["phases"]) if q == "scan"]
        out[(ds, label(a, model), a.get("n_train"), a.get("seed"))] = dict(ds=ds, lab=label(a, model), N=a.get("n_train"), seed=a.get("seed"), paper=p, full=fu, r1=r1, r2=r2, t1=t1, t2=t2, model=model.replace(".pt", ""),
                                                                          fc=r.get("fc") or {}, mse_scan=(nm([per["mse_test"][i] for i in sc]) if "mse_test" in per else float("nan")), cls=(r.get("summary") or {}).get("class_err_test"))
    return list(out.values())
def closed(model):
    f = os.path.join(EVAL, model, "summary.json")
    return json.load(open(f)) if os.path.exists(f) else None
def table(rows, title, cl=True, extra=None):
    print(f"\n### {title}\n"); cells = defaultdict(list)
    for r in rows: cells[(r["ds"], r["lab"], r["N"])].append(r)
    hdr = "| dataset | learner | N | seeds | relaxed gate | FULL gate | first-gap rate, full-gate seeds (theory) | second-gap rate (theory) |" + (" closed-loop success | side acc. | slot acc. | evaluated |" if cl else "") + (f" {extra[0]} |" if extra else "")
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for k in sorted(cells, key=lambda k: (k[0], k[2] or 0, k[1])):
        g = cells[k]; ok = [x for x in g if x["full"]]; n = len(g)
        th = lambda v: "" if math.isnan(v) else f" ({v:.2f})"
        row = f"| {k[0]} | {k[1]} | {k[2]} | {n} | {sum(x['paper'] for x in g)}/{n} | **{len(ok)}/{n}** | " + (f"{np.mean([x['r1'] for x in ok]):.2f}{th(g[0]['t1'])}" if ok else "--") + " | " + (f"{np.mean([x['r2'] for x in ok]):.2f}{th(g[0]['t2'])}" if ok and not math.isnan(ok[0]["r2"]) else "--") + " |"
        if cl:
            ev = [c for c in (closed(x["model"]) for x in g) if c]
            row += (f" {np.mean([c['success'] for c in ev]):.2f} | {np.mean([c['side'] for c in ev]):.2f} | {np.mean([c['slot'] for c in ev]):.2f} | {len(ev)}/{n} |" if ev else " -- | -- | -- | 0 |")
        if extra: row += " " + extra[1](g) + " |"
        print(row)

print("# Evidence tables: supervision, pixels, full gate, weighing task, T-maze (generated by isaac/analysis/report_supervision.py; do not edit by hand)\n")
print(__doc__.split("Usage:")[0].strip().replace("\n", "  \n"))
print("\nFrozen configuration: K=16, beta=1e-3, lambda=0, 10 000 steps, min(512, N) episodes sampled uniformly with replacement per step (448 at N=448), N=448 training episodes unless stated, seeds 0-7.  Closed loop = 128 episodes per model in Isaac (means over the evaluated seeds, sufficient or not).")
print("\n## 1. Future-behaviour supervision without decision-time knowledge (A', state observations)")
table(load("generic") + load("generic2") + load("frozen60"), "1a. Training recordings (grids `generic`, `generic2`, `frozen60`)")
rep = []
for f in sorted(glob.glob(os.path.join(RES, "rep_generic*.jsonl"))): rep += load(os.path.basename(f)[:-6])
table(rep, "1b. Same models, independent recordings (no retraining; theory column omitted: the new solver reproduces the 2 -> 1 -> 0 staircase)", cl=False)
fz = [r for r in load("frozen60") + load("generic2") if r["lab"] == "generic: random future offset, annealed from 60%"]
table(fz, "1c. FROZEN event-agnostic configuration (random future offset, auxiliary weight held to 60 % of training, zero from 80 %; chosen on pixel seeds 0-7, never retuned) across the five primary cells")
fzr = []
for f in sorted(glob.glob(os.path.join(RES, "rep_frozen60*.jsonl"))): fzr += load(os.path.basename(f)[:-6])
table(fzr, "1d. Frozen configuration, same models on independent recordings", cl=False)
print("\nReference cells of the paper under the FULL gate (from `analysis_full_gate.md`): unsupervised -R / -RF / scaffold 0-3/8 at gap 6 and 0-1/8 at gap 20; targeted forecast 8/8 (gap 6, 10, 20, M16), 5/8 (M32); teacher-state distillation 0-2/8.")
print("\n## 2. Pixel observations (A', 64x64 crop of a 128x128 camera + gripper / probe / phase channels; the forecaster reads the same pixels)")
table(load("px_fc") + load("px_fc20") + load("px_gen20"), "2a. Grids `px_fc` (gap 6), `px_fc20` and `px_gen20` (gap 20; 'annealed' = auxiliary weight held to 40 % of training and zero from 80 %, 'from 60%' = held to 60 %); plain and random+annealed cells have 16 seeds (0-15)",
      extra=("action error, closed loop (cm)", lambda g: (lambda ev: f"{np.mean([c['err'] for c in ev]):.1f}" if ev else "--")([c for c in (closed(x["model"]) for x in g) if c])))
print("\n### 2b. Pixel gap 20, closed loop split by offline sufficiency (n, success, slot accuracy)\n\n| learner | offline-sufficient seeds | other seeds |\n|---|---|---|")
_c = defaultdict(list)
for r in load("px_fc20") + load("px_gen20"):
    c = closed(r["model"])
    if c: _c[r["lab"]].append((r["full"], c["success"], c["slot"]))
for k in sorted(_c):
    f = lambda xs: f"{len(xs)}, {np.mean([x[1] for x in xs]):.2f}, {np.mean([x[2] for x in xs]):.2f}" if xs else "0, --, --"
    print(f"| {k} | {f([x for x in _c[k] if x[0]])} | {f([x for x in _c[k] if not x[0]])} |")
print("\n## 3. Dependence on the forecaster (A' gap 20)")
table(load("teacher"), "3a. Training recordings (grid `teacher`); last column = held-out class error of the forecaster's pending-action forecast", extra=("forecaster class error (held-out / as used in training)", lambda g: f"{nm([x['fc'].get('heldout_class_err', float('nan')) for x in g]):.3f} / {nm([x['fc'].get('train_class_err_as_used', float('nan')) for x in g]):.3f}"))
table(load("rep_teacher_gap20"), "3b. Same models on the independent recording (clean: no corrupted episode)", cl=False)
print("\n## 4. Weigh task (hidden mass revealed only by arm sag during a physical lift; Appendix E.3 of the paper)")
table(load("weigh") + load("weigh2") + load("weigh3"), "4a. Training recordings (grids `weigh`, `weigh2`, `weigh3`); gate = every gap1 step + the grasp decision; last column = held-out action MSE inside the weigh block",
      extra=("action MSE, weigh block", lambda g: f"{nm([x['mse_scan'] for x in g]):.4f}"))
rw = []
for f in sorted(glob.glob(os.path.join(RES, "rep_weigh*.jsonl"))): rw += load(os.path.basename(f)[:-6])
table(rw, "4b. Same models, independent recordings", cl=False)
print("\n### 4c. Closed loop split by offline sufficiency (success / side accuracy)\n\n| dataset | learner | N | offline-sufficient seeds: n, success, side | other seeds: n, success, side |\n|---|---|---|---|---|")
cells = defaultdict(list)
for r in load("weigh") + load("weigh2") + load("weigh3"):
    c = closed(r["model"])
    if c: cells[(r["ds"], r["lab"], r["N"])].append((r["full"], c["success"], c["side"]))
for k in sorted(cells, key=lambda k: (k[0], k[2] or 0, k[1])):
    s = [x for x in cells[k] if x[0]]; o = [x for x in cells[k] if not x[0]]; f = lambda xs: f"{len(xs)}, {np.mean([x[1] for x in xs]):.2f}, {np.mean([x[2] for x in xs]):.2f}" if xs else "0, --, --"
    print(f"| {k[0]} | {k[1]} | {k[2]} | {f(s)} | {f(o)} |")
print("\n## 5. External benchmark: Passive T-Maze (Ni et al. 2023; environment unmodified from MIKASA-Base), behaviour cloning of the scripted oracle, 100 closed-loop episodes per run\n")
T = defaultdict(list); tf = os.path.join(ROOT, "external", "results", "tmaze.jsonl")
for line in (open(tf) if os.path.exists(tf) else []):
    r = json.loads(line); T[(r["steps"], r["mode"], r["L"])].append(r)
names = {"gru": "standard GRU BC (continuous state, no bottleneck)", "gru_generic": "standard GRU BC + random-offset future-action head", "plain": "compact K=16 sole carrier, plain", "generic": "compact K=16 sole carrier + random-offset future-action head",
         "bypass": "paper's GRU-bypass variant (policy reads the VQ code)", "transformer": "paper's Transformer variant (policy reads the VQ code)"}
for steps in sorted({k[0] for k in T}):
    Ls = sorted({k[2] for k in T if k[0] == steps}); print(f"\n**{steps} training steps** — solved seeds (success >= 0.99) / runs, mean success in parentheses\n\n| learner | " + " | ".join(f"L={L}" for L in Ls) + " |\n|---|" + "---|" * len(Ls))
    for m in ("gru", "gru_generic", "plain", "generic", "bypass", "transformer"):
        if any((steps, m, L) in T for L in Ls):
            print(f"| {names[m]} | " + " | ".join((f"{sum(x['success'] >= 0.99 for x in T[(steps, m, L)])}/{len(T[(steps, m, L)])} ({np.mean([x['success'] for x in T[(steps, m, L)]]):.2f})" if (steps, m, L) in T else "--") for L in Ls) + " |")
print("\nFailure modes of the compact carrier on the T-maze (5000 steps, `generic`): cue kept at mid-corridor / kept at the junction — " + "; ".join(f"L={L}: {sum(x['I_code_goal_mid'] > 0.9 for x in T[(5000, 'generic', L)])}/{len(T[(5000, 'generic', L)])} / {sum(x['I_code_goal_last'] > 0.9 for x in T[(5000, 'generic', L)])}/{len(T[(5000, 'generic', L)])}" for L in sorted({k[2] for k in T if k[:2] == (5000, 'generic')})) + ".")
