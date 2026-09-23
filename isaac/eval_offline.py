"""Rate replication on independent data: evaluate saved DIACRITIC models (teacher-forced) on a dataset that was never used for training or
for building the solver, recomputing every theorem quantity with a solver built from the NEW recordings.  Also reports whether the new
data induce the same theoretical staircase as the original dataset.
Usage: python isaac/eval_offline.py --models isaac/cluster_results/models/fig3/*.pt --data isaac/data/rep_tier4_gap6 [--orig isaac/data/tier4_gap6_2k] [--sym aug|aug_sag] [--out file.jsonl]"""
import sys, os, glob, json, argparse, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "toy"))
from gamma_solver import cond_entropy
from aprime_data import prepare, solver_labels
from diacritic_model import load_model
ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--data", required=True); ap.add_argument("--orig", default="")
ap.add_argument("--sym", default="aug"); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args(); dev = torch.device(args.device)
eps, meta, vis, env, solver, sym_key = prepare(args.data, augment=True, sag_symbol=(args.sym == "aug_sag"))
R1, R2, M = meta["R1"], meta["R2"], meta["M"]; E, T = len(eps), len(eps[0]["phase"]); phases = eps[0]["phase"]
G_lab, Gam_lab, GamS_lab, O_lab = solver_labels(solver, eps, sym_key); th = solver.rates()
same = None
if args.orig:
    eps0, meta0, vis0, env0, solver0, _ = prepare(args.orig, augment=True, sag_symbol=(args.sym == "aug_sag")); th0 = solver0.rates()
    same = dict(max_abs_diff_HGam=float(np.max(np.abs(np.array(th["H(Gamma|O)"]) - np.array(th0["H(Gamma|O)"])))), max_abs_diff_HG=float(np.max(np.abs(np.array(th["H(G|O)"]) - np.array(th0["H(G|O)"])))),
                max_abs_diff_HGamS=float(np.max(np.abs(np.array(th["H(GammaS|O)"]) - np.array(th0["H(GammaS|O)"])))), transitive_new=all(solver.transitive.values()), transitive_orig=all(solver0.transitive.values()),
                a4_fail_new=[t for t, v in solver.a4_ok.items() if not v], a4_fail_orig=[t for t, v in solver0.a4_ok.items() if not v], visible_same=(vis == vis0))
    print("solver on new data vs original:", json.dumps(same))
O = torch.tensor(np.stack([e["obs"] for e in eps])); A = torch.tensor(np.stack([e["act"] for e in eps]))
z_lab = [e["z"] for e in eps]; theta_lab = [e["theta"] for e in eps]; mass_lab = [e["theta"] * 4 // M for e in eps]
div_steps = {t: ("b1" if isinstance(eps[0]["sym_act"][t], tuple) and eps[0]["sym_act"][t][0] == "grasp" else "b2") for t in range(T) if isinstance(eps[0]["sym_act"][t], tuple) and eps[0]["sym_act"][t][0] in ("grasp", "place")}
w = [1.0 / E] * E
def sym_onehot(binner):
    """apply the model's frozen binner f_t to the new raw observations -> (E,T,ds) one-hots (+ disagreement with the new symbolic labels)"""
    vocab = binner["vocab"]; ph_names = binner["phase_names"]; means = binner["means"]; sag = binner["sag"]; S = torch.zeros(E, T, len(vocab)); dis = 0; unseen = 0
    for i in range(E):
        for t in range(T):
            o = eps[i]["obs"][t]; phase = ph_names[int(np.argmax(o[19:27]))]; probe = int(o[18] > 0.5) if o[17] > 0.5 else -1; sym = [phase, probe]
            for which in ("b1", "b2"):
                if str(t) in means and which in means[str(t)]: mu = np.asarray(means[str(t)][which]); sym.append(int(np.argmin(((mu - o[None, :3]) ** 2).sum(-1))))
                else: sym.append(-1)
            if str(t) in sag: sym.append(int((o[2] < sag[str(t)][0]) == sag[str(t)][1]))
            key = str(tuple(sym)); j = vocab.get(key)
            if j is None: unseen += 1; j = 0
            S[i, t, j] = 1.0; dis += int(tuple(sym) != tuple(eps[i][sym_key][t]))
    return S, dis / (E * T), unseen
rows = []
for path in args.models:
    m, ck = load_model(path, args.device); ex = ck["extra"]; a = ex.get("args", {})
    On = ((O.to(dev) - ck["o_mu"].to(dev)) / ck["o_sd"].to(dev)); An = ((A.to(dev) - ck["a_mu"].to(dev)) / ck["a_sd"].to(dev))
    S = None; dis = unseen = None
    if m.ds > 0:
        S, dis, unseen = sym_onehot(ex["sym_binner"]); S = S.to(dev)
    SA = None
    if m.na > 0:
        av = ex["act_vocab"]; SA = torch.zeros(E, T, len(av))
        for i in range(E):
            for t in range(T): SA[i, t, av[str(eps[i]["sym_act"][t])]] = 1.0
        SA = SA.to(dev)
    with torch.no_grad(): ks, eqs, rates, vqs, preds = m.rollout(On, An, S, SA)
    ks = ks.cpu().numpy(); preds = preds.cpu()
    per = {}
    for t in range(T):
        c = list(ks[:, t]); o = O_lab[t]; g = G_lab[t]; gam = Gam_lab[t]
        HC = cond_entropy(c, o, w); HG = cond_entropy(g, o, w); HGam = cond_entropy(gam, o, w)
        per.setdefault("H(C|O)", []).append(HC); per.setdefault("S_G", []).append((HG - cond_entropy(g, list(zip(c, o)), w)) / HG if HG > 1e-9 else float("nan"))
        per.setdefault("S_Gam", []).append((HGam - cond_entropy(gam, list(zip(c, o)), w)) / HGam if HGam > 1e-9 else float("nan"))
        per.setdefault("I(C;mass|O)", []).append(cond_entropy(mass_lab, o, w) - cond_entropy(mass_lab, list(zip(c, o)), w))
        per.setdefault("I(C;theta|O)", []).append(cond_entropy(theta_lab, o, w) - cond_entropy(theta_lab, list(zip(c, o)), w))
    pm = lambda key, ph: float(np.nanmean([per[key][i] for i, p in enumerate(phases) if p == ph]))
    summ = dict(model=os.path.basename(path), HC_gap1=pm("H(C|O)", "gap1"), HC_gap2=pm("H(C|O)", "gap2"), HC_grasp=pm("H(C|O)", "grasp"), S_Gam_gap1=pm("S_Gam", "gap1"), S_Gam_gap2=pm("S_Gam", "gap2"),
                S_G_grasp=pm("S_G", "grasp"), S_G_place=pm("S_G", "place"), I_mass_grasp=pm("I(C;mass|O)", "grasp"), I_theta_gap2=pm("I(C;theta|O)", "gap2"), binner_disagree=dis, unseen=unseen,
                HGam_gap1=float(np.nanmean([th["H(Gamma|O)"][i] for i, p in enumerate(phases) if p == "gap1"])), HGam_gap2=float(np.nanmean([th["H(Gamma|O)"][i] for i, p in enumerate(phases) if p == "gap2"])),
                # sufficiency gate as in the paper: A' (R1 > 1) = S_Gamma(gap2) > 0.9 and S_G(place) > 0.9; Task A (R1 == 1, split latent) = S_G(place) > 0.9 only
                sufficient=bool(pm("S_Gam", "gap2") > 0.9 and pm("S_G", "place") > 0.9) if R1 > 1 else bool(pm("S_G", "place") > 0.9), train_summary={k: ex.get("summary", {}).get(k) for k in ("HC_gap1", "HC_gap2", "S_Gam_gap2", "S_G_grasp", "S_G_place")})
    summ.update(per_t={k: [float(x) for x in v] for k, v in per.items()}, phases=list(phases), args={k: ex.get("args", {}).get(k) for k in ("K", "aux_theta", "distill", "n_train", "beta", "data", "dual_K", "distill_stride", "seed")})
    rows.append(summ); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k not in ("train_summary", "per_t", "phases", "args")}))
if args.out:
    with open(args.out, "a") as f:
        for r in rows: f.write(json.dumps(dict(r, data=args.data, solver_vs_orig=same)) + "\n")
