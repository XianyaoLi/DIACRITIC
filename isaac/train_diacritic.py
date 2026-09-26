"""Acceptance (d): train DIACRITIC-R (recurrent conditional-rate bottleneck, hard VQ, sole discrete carrier) on the
recorded A' demonstrations (continuous observations / actions) and measure the theorem quantities with Gamma labels
obtained from the exact solver on the recorded symbolic histories.

Model (continuous version of toy/toy_diacritic.py, variant 'diacritic' / 'bypass' / 'nonpersistent' / 'uncond'):
    e~_t = e_{t-1} + f([e_{t-1}, g_o(o_t), g_a(a_{t-1})]),  C_t = argmin_k ||e~_t - E_k||,  STE backward,
    rate_t = -log r_eta(C_t | o_t)  (soft-to-hard gradient),  pi(a_t | o_t, C_t) Gaussian (MSE on standardised actions).
    Only C_t (the code) crosses time.
Metrics per t (empirical over episodes, symbolic O_t = (phase, probe bit)):
    H(C|O), H(Gam|O), H(G|O), S_Gam, S_G, H(C|Gam,O), I(C;z|O), codes; distortion = class error of the decoded
    expert action at the beta_1 / beta_2 dependent steps (nearest per-(t,class) mean expert action) on held-out episodes.

Usage:  python isaac/train_diacritic.py isaac/data/tier4_gap6 --beta 0.03 --seed 0 --out isaac/results/tier4_gap6.jsonl
"""
import sys, os, glob, json, ast, math, argparse, time
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy")); sys.path.insert(0, HERE)
from gamma_solver import cond_entropy

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--beta", type=float, default=0.03); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, default=16); ap.add_argument("--d", type=int, default=32); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--l2", type=int, default=1); ap.add_argument("--commit", type=float, default=0.25); ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--beta_warmup", type=int, default=1000); ap.add_argument("--variant", default="diacritic")
ap.add_argument("--n_train", type=int, default=448); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--log_every", type=int, default=500)
ap.add_argument("--bound", default="none", help="none | tanh (bounded residual state) ")
ap.add_argument("--lam", type=float, default=0.0, help="BFS weight (0 = DIACRITIC-R, >0 = DIACRITIC-RF)")
ap.add_argument("--J", type=int, default=100, help="BFS horizon (capped at T-1-t)")
ap.add_argument("--sym", default="raw", help="symbolic observation convention for the metrics: raw=(phase,probe) | aug=+visible classes (aprime_data)")
ap.add_argument("--pixels", action="store_true", help="pixel sanity check: CNN on the recorded camera images (downsampled to --px_res) + low-dim channels (gripper, probe, phase) instead of the state vector")
ap.add_argument("--px_res", type=int, default=64); ap.add_argument("--px_crop", default="40,112,16,112", help="y0,y1,x0,x1 crop of the 128x128 frame before downsampling ("" = none)")
ap.add_argument("--save_codes", default="", help="npz path: save hard codes (E,T), labels and split for re-evaluation")
ap.add_argument("--refine", default="", help="'' | bfs | mse : future-sufficiency-triggered code refinement (split the code whose members have inconsistent futures)")
ap.add_argument("--refine_every", type=int, default=200); ap.add_argument("--refine_start", type=int, default=1000); ap.add_argument("--refine_thresh", type=float, default=1.5); ap.add_argument("--refine_min", type=int, default=8)
ap.add_argument("--init_from", default="", help="warm start: load weights from a saved model (.pt) trained on a shorter gap (curriculum)")
ap.add_argument("--scaffold_hold", type=int, default=2000); ap.add_argument("--scaffold_end", type=int, default=6000)
ap.add_argument("--aux_join", type=float, default=0.0, help="oracle-Gamma capacity control: weight of a privileged head predicting the behavioural join (beta1,beta2) from (o_t, C_t) — diagnostic upper bound only")
ap.add_argument("--aux_theta", type=float, default=0.0, help="system-identification baseline: weight of an auxiliary head predicting the hidden mode theta from (o_t, C_t); forces world-state memory")
ap.add_argument("--aux_mik", type=float, default=0.0, help="multi-step-inverse (MusIK/ACSD-style) baseline: weight of a head predicting a_t from (C_t, o_t, o_{t+j}), j ~ U{1..mik_J}; the representation is shaped by the inverse objective")
ap.add_argument("--mik_J", type=int, default=8); ap.add_argument("--mik_detach", type=int, default=0, help="1: the policy head reads a detached (C_t, o_t) so the recurrent state is trained ONLY by the inverse objective (+VQ); 0: joint with imitation")
ap.add_argument("--save_model", default="", help="pt path: save weights + normalisation stats for closed-loop evaluation (aprime_env.py --policy)")
ap.add_argument("--sym_input", type=int, default=0, help="2: matched side information, hierarchical -- transition, prior AND the behavioural head see only f_t(o_t); a memory-free low-level controller pi_low(a | o_t, symbolic action) executes.  1: memory-write control only (transition + prior on f_t(o_t), policy head on raw o_t).  1: matched side information -- the recurrent transition and the conditional prior receive ONLY the symbolic observation f_t(o_t) (frozen per-step binner fitted on the training split; one-hot); the policy head keeps the raw observation")
ap.add_argument("--distill", default="", help="retrieve->commit control: path of a frozen full-history teacher (.pt); the student's (o_t, C_t) must regress the teacher's pre-quantisation state (non-privileged target)")
ap.add_argument("--distill_w", type=float, default=1.0)
ap.add_argument("--distill_target", default="forecast", help="forecast: the teacher's history-derived forecast of the pending class-dependent actions (grasp/place), per step, masked once used; feat: the teacher's pre-quantisation state; uniform: GENERIC target -- the forecast of all future actions a_{t+1..t+J} at every step, no knowledge of decision times or of the reveal step; random: same generic forecast, the student head is queried at one random offset j ~ U{1..J} per (episode, step)")
ap.add_argument("--high_w", type=float, default=1.0, help="hierarchical: weight of the behavioural head's cross-entropy (the only imitation signal that reaches the memory; 1 step in 33 is memory-dependent, so the mean CE is a weak signal at weight 1)")
ap.add_argument("--bfs_noobs", type=int, default=0, help="1: decision-centric control -- the -RF future decoder receives no future observation/action (open-loop a_{t+1..t+J} from (C_t, o_t))")
ap.add_argument("--dual_K", type=int, default=0, help="system-identification control with a SEPARATE identification carrier: the theta head reads (o_t, C2_t) from a second recurrent hard-VQ code with K2 entries; the policy reads (o_t, C_t) only; both carriers are rate-penalised (requires --aux_theta > 0)")
ap.add_argument("--dual_detach", type=int, default=0, help="1: the second carrier reads stop-gradient observation/action embeddings (no theta gradient into the shared encoders)")
ap.add_argument("--distill_stride", type=int, default=1, help="forecast distillation supervised only every s-th step after the last reveal (label-density control)")
ap.add_argument("--distill_source", default="teacher", help="teacher: regress the full-history forecaster's output (default) | gt: regress the recorded future expert actions directly (no forecaster)")
ap.add_argument("--distill_off", type=float, default=0.0, help="training-only supervision schedule: > 0 -> the distillation weight is held until this fraction of the training steps, then annealed linearly to 0 at 80 %; the last 20 % of training is imitation + rate only (lets the rate term prune what only the auxiliary target asked for)")
ap.add_argument("--fc_start", default="reveal", help="targeted forecast: supervise from the last identification step (reveal; default, as in the paper) | zero: from the first step of the episode -- no knowledge of WHEN the class becomes known; before that the forecaster's output is the conditional mean")
ap.add_argument("--fc_J", type=int, default=0, help="generic targets (uniform | random): number of future offsets j = 1..J (0 = all, J = T-1)")
ap.add_argument("--fc_steps", type=int, default=3000, help="training steps of the full-history forecaster (smaller = deliberately weak teacher)")
ap.add_argument("--norm_fit", default="all", choices=["all", "train"], help="episodes used for the observation/action normalisation statistics: all recorded episodes (default; the paper's runs) or the training split only (control)")
ap.add_argument("--fc_corrupt", type=float, default=0.0, help="teacher-dependence control: for this fraction of the training episodes the distillation target is the forecast of a training episode with a DIFFERENT behavioural class (consistently wrong forecast)")
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)

# ----------------------------------------------------------------------------- data + solver labels
from aprime_data import prepare, solver_labels
eps, meta, vis, env, solver, sym_key = prepare(args.data, augment=(args.sym in ("aug", "aug_sag")), sag_symbol=(args.sym == "aug_sag"))
R1, R2, M, N = meta["R1"], meta["R2"], meta["M"], meta["N"]
E, T = len(eps), len(eps[0]["phase"]); phases = eps[0]["phase"]
G_lab, Gam_lab, GamS_lab, O_lab = solver_labels(solver, eps, sym_key)
rates_theory = solver.rates()
z_lab = [e["z"] for e in eps]; theta_lab = [e["theta"] for e in eps]; mass_lab = [e["theta"] * 4 // M for e in eps]   # mass quartile (mass is monotone in theta)

O = torch.tensor(np.stack([e["obs"] for e in eps])); A = torch.tensor(np.stack([e["act"] for e in eps]))      # (E,T,do), (E,T,da)
if args.pixels:
    import glob as _glob
    IM = []
    for f in sorted(_glob.glob(os.path.join(args.data, "ep*_env*.npz"))):
        im = torch.tensor(np.load(f, allow_pickle=True)["img"])                      # (T,H,W,3) uint8
        if args.px_crop: y0, y1, x0, x1 = map(int, args.px_crop.split(",")); im = im[:, y0:y1, x0:x1]
        im = F.interpolate(im.permute(0, 3, 1, 2).float(), size=(args.px_res, args.px_res), mode="area").round().to(torch.uint8)
        IM.append(im)
    IM = torch.stack(IM).to(dev)                                                       # (E,T,3,r,r) uint8 on GPU
    LOW_IDX = [7, 17, 18] + list(range(19, 27))                                          # gripper, probe(2), phase one-hot(8)
    print(f"[pixels] images {tuple(IM.shape)} ({IM.numel()/1e6:.0f} MB uint8), low-dim channels {len(LOW_IDX)}", flush=True)
_nfit = np.random.RandomState(0).permutation(E)[:args.n_train] if args.norm_fit == "train" else np.arange(E)   # same split as below
o_mu, o_sd = O[_nfit].reshape(-1, O.shape[-1]).mean(0), O[_nfit].reshape(-1, O.shape[-1]).std(0) + 1e-6
a_mu, a_sd = A[_nfit].reshape(-1, A.shape[-1]).mean(0), A[_nfit].reshape(-1, A.shape[-1]).std(0) + 1e-6
On, An = ((O - o_mu) / o_sd).to(dev), ((A - a_mu) / a_sd).to(dev)
if args.pixels:
    On_low = On[:, :, LOW_IDX].contiguous()
    def OBS(idx, t=None):   # (imgs float in [0,1], low) for a batch of episodes
        im = IM[idx].float() / 255.0; lo = On_low[idx]
        return (im, lo) if t is None else (im[:, t], lo[:, t])
else:
    def OBS(idx, t=None): return On[idx] if t is None else On[idx][:, t]
perm = np.random.RandomState(0).permutation(E); tr_idx, te_idx = perm[:args.n_train], perm[args.n_train:]
# divergent steps and per-(t, class) mean expert actions for class decoding
div_steps = {t: ("b1" if isinstance(eps[0]["sym_act"][t], tuple) and eps[0]["sym_act"][t][0] == "grasp" else "b2")
             for t in range(T) if isinstance(eps[0]["sym_act"][t], tuple) and eps[0]["sym_act"][t][0] in ("grasp", "place")}
class_mean = {}
for t, which in div_steps.items():
    for c in range(R1 if which == "b1" else R2):
        rows = [i for i in tr_idx if eps[i][which] == c]
        class_mean[(t, c)] = An[rows, t].mean(0)
train_keys = {(eps[i]["theta"], eps[i]["b2"]) if len({(e["theta"], e["b2"]) for e in eps}) > len({e["theta"] for e in eps}) else eps[i]["theta"] for i in tr_idx}
all_keys = {(e["theta"], e["b2"]) if len({(e["theta"], e["b2"]) for e in eps}) > len({e["theta"] for e in eps}) else e["theta"] for e in eps}
train_modes = dict(seen_in_train=len(train_keys), recorded=len(all_keys), modes_M=M, unseen_in_train=len(all_keys - train_keys))
print(f"{E} episodes (train {len(tr_idx)} / test {len(te_idx)}), T={T}, obs {O.shape[-1]}, act {A.shape[-1]}, divergent steps {sorted(div_steps)}; training coverage {train_modes}", flush=True)

# ----------------------------------------------------------------------------- matched side information: frozen per-step binner f_t(o_t)
S = None; sym_binner = None; sym_disagree = None
if args.sym_input:
    if args.pixels: raise SystemExit("--sym_input not supported with --pixels")
    Oraw = np.stack([e["obs"] for e in eps]); ph_names = sorted({p for p in phases}, key=lambda p: int(np.argmax(Oraw[0, phases.index(p), 19:27])))
    ph_names = [None] * 8
    for t, p in enumerate(phases): ph_names[int(np.argmax(Oraw[0, t, 19:27]))] = p
    ph_names = [str(p) if p is not None else f"phase{i}" for i, p in enumerate(ph_names)]
    means, sag = {}, {}
    for t in range(T):
        mt = {}
        for which, R in (("b1", R1), ("b2", R2)):
            if vis[which][t]:
                mt[which] = [Oraw[[i for i in tr_idx if eps[i][which] == c], t, :3].mean(0).tolist() for c in range(R)]
        if mt: means[str(t)] = mt
        if sym_key == "sym_obs_aug" and len(eps[0]["sym_obs_aug"][t]) == 5:       # task A: sag symbol from the tcp height
            hi = [i for i in tr_idx if eps[i]["sym_obs_aug"][t][4] == 1]; lo = [i for i in tr_idx if eps[i]["sym_obs_aug"][t][4] == 0]
            z1, z0 = Oraw[hi, t, 2].mean(), Oraw[lo, t, 2].mean(); sag[str(t)] = [float((z1 + z0) / 2), bool(z1 < z0)]   # [threshold, symbol-1 is the LOWER side]
    def f_t(t, o):
        phase = ph_names[int(np.argmax(o[19:27]))]; probe = int(o[18] > 0.5) if o[17] > 0.5 else -1; sym = [phase, probe]
        for which in ("b1", "b2"):
            if str(t) in means and which in means[str(t)]: mu = np.asarray(means[str(t)][which]); sym.append(int(np.argmin(((mu - o[None, :3]) ** 2).sum(-1))))
            else: sym.append(-1)
        if str(t) in sag: sym.append(int((o[2] < sag[str(t)][0]) == sag[str(t)][1]))
        return tuple(sym)
    shat = [[f_t(t, eps[i]["obs"][t]) for t in range(T)] for i in range(E)]
    true = [[tuple(eps[i][sym_key][t]) for t in range(T)] for i in range(E)]
    dis_tr = float(np.mean([shat[i][t] != true[i][t] for i in tr_idx for t in range(T)])); dis_te = float(np.mean([shat[i][t] != true[i][t] for i in te_idx for t in range(T)]))
    sym_disagree = dict(train=dis_tr, test=dis_te)
    vocab = {str(k): j for j, k in enumerate(sorted({shat[i][t] for i in range(E) for t in range(T)} | {true[i][t] for i in range(E) for t in range(T)}, key=str))}
    S = torch.zeros(E, T, len(vocab))
    for i in range(E):
        for t in range(T): S[i, t, vocab[str(shat[i][t])]] = 1.0
    S = S.to(dev)
    sym_binner = dict(vocab=vocab, phase_names=ph_names, phase_slice=[19, 27], probe_idx=[17, 18], means=means, sag=sag, disagree=sym_disagree)
    print(f"[sym_input] vocabulary {len(vocab)} symbols; binner-vs-label disagreement train {dis_tr:.4f} test {dis_te:.4f}; {len(means)} steps with visible classes, {len(sag)} sag steps", flush=True)

SA = None; act_vocab = None; NA = 0
if args.sym_input == 3: NA = -1
if args.sym_input == 2:
    act_vocab = {str(a): j for j, a in enumerate(sorted({str(eps[i]["sym_act"][t]) for i in range(E) for t in range(T)}))}
    SA = torch.zeros(E, T, len(act_vocab))
    for i in range(E):
        for t in range(T): SA[i, t, act_vocab[str(eps[i]["sym_act"][t])]] = 1.0
    SA = SA.to(dev); SA_idx = SA.argmax(-1)
    print(f"[sym_input=2] hierarchical: {len(act_vocab)} symbolic actions {sorted(act_vocab)}", flush=True)

# ----------------------------------------------------------------------------- model (shared)
from diacritic_model import DIACRITIC, save_model, load_model

model = DIACRITIC((-len(LOW_IDX)) if args.pixels else O.shape[-1], A.shape[-1], args.K, args.d, args.hid, args.tau, bool(args.l2), args.commit, args.variant, args.bound, ds=(S.shape[-1] if S is not None else 0), na=(SA.shape[-1] if SA is not None else NA), K2=args.dual_K, dual_detach=bool(args.dual_detach)).to(dev)
if args.dual_K > 0: assert args.aux_theta > 0, "--dual_K is the identification carrier of the sys-ID control; set --aux_theta"
teacher = dist_head = fc_head = None; fc_mask = fc_tgt = None
if args.distill:
    if args.distill_target == "feat":
        teacher, _tck = load_model(args.distill, dev)
        for p_ in teacher.parameters(): p_.requires_grad_(False)
        with torch.no_grad(): teacher.rollout(On, An); T_ets = torch.stack(teacher._ets, 1).detach()          # (E,T,d) teacher states, all episodes
        dist_head = nn.Sequential(nn.Linear(2 * args.d, args.hid), nn.ReLU(), nn.Linear(args.hid, args.d)).to(dev); dist_dim = args.d
        print(f"[distill] teacher {args.distill} ({teacher.variant}); target = teacher pre-quantisation state (d={args.d}), weight {args.distill_w}", flush=True)
    else:
        # history-derived forecast of the PENDING class-dependent actions: for each divergent block (grasp -> b1, place -> b2) the expert
        # action at its first divergent step is the target at every earlier step t (masked once that step is reached), so the target
        # at t is exactly the anticipatory behaviour still to come.  The forecast head is fitted on the frozen teacher (recorded data only).
        blocks = []
        for which in ("b1", "b2"):
            ts = [t for t, w_ in div_steps.items() if w_ == which]
            if ts: blocks.append(min(ts))
        da = A.shape[-1]
        rev = max(i for i, p_ in enumerate(phases) if p_ == "scan")                                                    # last reveal step (targeted target + diagnostics only)
        generic = args.distill_target in ("uniform", "random")
        if generic:
            # GENERIC future-behaviour target: a_{t+1..t+J} at EVERY step t; no decision time, divergent step or reveal step enters the
            # construction -- the only mask is the end of the episode.
            J_fc = args.fc_J if args.fc_J > 0 else T - 1; dist_dim = da * J_fc
            fidx = torch.arange(T, device=dev)[:, None] + 1 + torch.arange(J_fc, device=dev)[None]                     # (T,J): index of a_{t+j}
            fvalid = (fidx < T).float(); fidx = fidx.clamp(max=T - 1)
            fc_tgt = An[:, fidx].reshape(E, T, dist_dim)                                                                 # (E,T,J*da), offset-major
            fc_mask = fvalid[None, :, :, None].expand(E, T, J_fc, da).reshape(E, T, dist_dim)
            fc_mask_student = fc_mask
            if args.distill_stride > 1: raise SystemExit("--distill_stride applies to the targeted forecast only")
        else:
            dist_dim = da * len(blocks)
            fc_tgt = torch.cat([An[:, tb][:, None, :].expand(E, T, da) for tb in blocks], -1)                          # (E,T,dist_dim)
            fc_mask = torch.cat([((torch.arange(T, device=dev) >= (0 if args.fc_start == "zero" else rev)) & (torch.arange(T, device=dev) < tb)).float()[None, :, None].expand(E, T, da) for tb in blocks], -1)
            if args.distill_stride > 1:   # label-density control: keep the supervision only at steps rev, rev+s, rev+2s, ... (the forecaster itself is trained on all steps)
                keep = (((torch.arange(T, device=dev) - rev) % args.distill_stride) == 0).float()[None, :, None]
                n_all = int(fc_mask[0].sum().item() // da); fc_mask_student = fc_mask * keep
                print(f"[distill] stride {args.distill_stride}: student supervised at {int(fc_mask_student[0].sum().item() // da)} of {n_all} (step, block) labels per episode", flush=True)
            else: fc_mask_student = fc_mask
        def fc_pending(F_, tb, bi):
            """(E, n_steps, da): the forecast of the action at the first divergent step tb, read at the steps rev..tb-1 (diagnostic)."""
            ts_ = list(range(rev, tb))
            if generic: return torch.stack([F_[:, t_, (tb - t_ - 1) * da:(tb - t_) * da] for t_ in ts_ if tb - t_ - 1 < J_fc], 1)
            return F_[:, ts_, bi * da:(bi + 1) * da]
        def fc_class_error(F_, rows):
            """class error of the pending-action forecast (nearest per-class mean expert action), averaged over rev..tb-1 and both blocks"""
            errs = []
            for bi, tb in enumerate(blocks):
                which = div_steps[tb]; R = R1 if which == "b1" else R2; P_ = fc_pending(F_, tb, bi)[rows]                # (n,steps,da)
                if P_.shape[1] == 0: continue
                mu = torch.stack([class_mean[(tb, c)] for c in range(R)], 0)                                           # (R,da)
                pred_c = (P_[:, :, None, :] - mu[None, None]).pow(2).sum(-1).argmin(-1).cpu().numpy()                    # (n,steps)
                true_c = np.array([eps[i][which] for i in rows])[:, None]; errs.append(float((pred_c != true_c).mean()))
            return float(np.mean(errs)) if errs else float("nan")
        # the forecaster is a full-history (causal Transformer) model trained from scratch on the recorded data for this objective: at every
        # step it retrieves the pending class-dependent actions from the history prefix ("retrieval is easy"); the frozen teacher's own
        # per-step readout is NOT used as a feature because nothing forced it to represent the pending classes at gap steps.
        fc_err = fc_cls = float("nan"); fc_info = {}
        if args.distill_source == "gt":
            fc_tgt_all = fc_tgt                                                                                          # recorded future expert actions, no forecaster
        else:
            fc_model = DIACRITIC((-len(LOW_IDX)) if args.pixels else O.shape[-1], A.shape[-1], args.K, args.d, args.hid, args.tau, bool(args.l2), args.commit, "transformer", "none").to(dev)
            fc_head = nn.Sequential(nn.Linear(args.d, args.hid), nn.ReLU(), nn.Linear(args.hid, dist_dim)).to(dev)
            def fc_forward(idx):
                Ab = An[idx]; ea_ = fc_model.g_a(torch.cat([torch.zeros_like(Ab[:, :1]), Ab[:, :-1]], 1))
                if args.pixels:   # the forecaster sees the same pixel observation as the student (CNN over all frames of the batch)
                    im, lo = OBS(idx); B_, T_ = lo.shape[:2]; eo_ = fc_model.g_o((im.reshape(B_ * T_, *im.shape[2:]), lo.reshape(B_ * T_, -1))).view(B_, T_, -1)
                else: eo_ = fc_model.g_o(On[idx])
                return fc_head(fc_model.tf_forward(torch.cat([eo_, ea_], -1)))
            opt_fc = torch.optim.Adam(list(fc_model.parameters()) + list(fc_head.parameters()), lr=1e-3); trf = torch.tensor(tr_idx, device=dev)
            fcB = 64 if args.pixels else 256
            _t_fc = time.time()
            for it_ in range(args.fc_steps):
                b_ = trf[torch.randint(0, len(trf), (fcB,), device=dev)]
                l_ = ((fc_forward(b_) - fc_tgt[b_]).pow(2) * fc_mask[b_]).sum(-1).mean(); opt_fc.zero_grad(); l_.backward(); opt_fc.step()
            with torch.no_grad():
                fc_tgt_all = torch.cat([fc_forward(torch.arange(i, min(i + fcB, E), device=dev)) for i in range(0, E, fcB)], 0).detach()   # (E,T,dist_dim)
                te_ = torch.tensor(te_idx, device=dev); fc_err = float(((fc_tgt_all[te_] - fc_tgt[te_]).pow(2) * fc_mask[te_]).sum() / fc_mask[te_].sum())
            for p_ in fc_model.parameters(): p_.requires_grad_(False)
            for p_ in fc_head.parameters(): p_.requires_grad_(False)
            del opt_fc
        try: fc_cls = fc_class_error(fc_tgt_all, list(te_idx)); fc_cls_tr = fc_class_error(fc_tgt_all, list(tr_idx))
        except Exception as ex_: fc_cls_tr = float("nan"); print(f"[distill] forecast class-error diagnostic unavailable: {ex_}", flush=True)
        n_corrupt = 0
        if args.fc_corrupt > 0:   # consistently wrong forecast for a fraction of the training episodes (donor = training episode of a different behavioural class)
            rs_ = np.random.RandomState(1000 + args.seed); bad = rs_.choice(tr_idx, int(round(args.fc_corrupt * len(tr_idx))), replace=False); fc_clean = fc_tgt_all.clone()
            for i in bad:
                donors = [j for j in tr_idx if (eps[j]["b1"], eps[j]["b2"]) != (eps[i]["b1"], eps[i]["b2"])]; fc_tgt_all[i] = fc_clean[donors[rs_.randint(len(donors))]]
            n_corrupt = len(bad); del fc_clean
            try: fc_cls_tr = fc_class_error(fc_tgt_all, list(tr_idx))
            except Exception: pass
        fc_info = dict(fc_secs=(time.time() - _t_fc) if args.distill_source != 'gt' else 0.0, fc_params=(sum(p_.numel() for p_ in fc_model.parameters()) + sum(p_.numel() for p_ in fc_head.parameters())) if args.distill_source != 'gt' else 0, student_params=sum(p_.numel() for p_ in model.parameters()), target=args.distill_target, source=args.distill_source, dim=int(dist_dim), fc_steps=args.fc_steps, heldout_mse=fc_err, heldout_class_err=fc_cls, train_class_err_as_used=fc_cls_tr, n_corrupt=int(n_corrupt))
        dist_head = nn.Sequential(nn.Linear(2 * args.d + (16 if args.distill_target == "random" else 0), args.hid), nn.ReLU(), nn.Linear(args.hid, da if args.distill_target == "random" else dist_dim)).to(dev)
        fc_jemb = nn.Embedding(J_fc + 1, 16).to(dev) if args.distill_target == "random" else None
        print(f"[distill] target={args.distill_target} source={args.distill_source}: " + (f"generic future actions a_(t+1..t+{J_fc}) at every step (dim {dist_dim}), no decision-time / reveal-step knowledge" if generic else f"pending divergent actions at steps {blocks} (dim {dist_dim}), supervised from {('step 0' if args.fc_start == 'zero' else 'the last reveal step ' + str(rev))} until each use")
              + f"; forecaster steps {args.fc_steps}, held-out per-dim forecast mse {fc_err:.4f}, pending-action class error held-out {fc_cls:.3f} / train-as-used {fc_cls_tr:.3f} ({n_corrupt} corrupted episodes); weight {args.distill_w}", flush=True)
if args.init_from:
    _ck = torch.load(args.init_from, map_location=dev, weights_only=False)
    model.load_state_dict(_ck["state_dict"]); print(f"[init] loaded weights from {args.init_from} (trained on {_ck['extra'].get('args', {}).get('data')})", flush=True)
aux_head = nn.Sequential(nn.Linear(2 * args.d, args.hid), nn.ReLU(), nn.Linear(args.hid, M)).to(dev) if args.aux_theta > 0 else None
theta_t = torch.tensor([e["theta"] for e in eps], device=dev)
join_head = nn.Sequential(nn.Linear(2 * args.d, args.hid), nn.ReLU(), nn.Linear(args.hid, R1 * R2)).to(dev) if args.aux_join > 0 else None
join_t = torch.tensor([e["b1"] + R1 * e["b2"] for e in eps], device=dev)
mik_head = mik_jemb = None
if args.aux_mik > 0:
    if args.pixels: raise SystemExit("--aux_mik not supported with --pixels")
    mik_head = nn.Sequential(nn.Linear(3 * args.d + 16, args.hid), nn.ReLU(), nn.Linear(args.hid, args.hid), nn.ReLU(), nn.Linear(args.hid, A.shape[-1])).to(dev)
    mik_jemb = nn.Embedding(args.mik_J + 1, 16).to(dev); model.detach_pi = bool(args.mik_detach)
opt = torch.optim.Adam(list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else []) + (list(join_head.parameters()) if join_head is not None else [])
                       + (list(mik_head.parameters()) + list(mik_jemb.parameters()) if mik_head is not None else []) + (list(dist_head.parameters()) if dist_head is not None else []) + (list(fc_jemb.parameters()) if args.distill and args.distill_target == 'random' else []), lr=args.lr)
t0 = time.time(); tr = torch.tensor(tr_idx, device=dev)
for it in range(args.steps):
    if args.variant == "scaffold":
        model.alpha = 1.0 if it < args.scaffold_hold else max(0.0, 1.0 - (it - args.scaffold_hold) / max(1, args.scaffold_end - args.scaffold_hold))
    b = tr[torch.randint(0, len(tr), (min(args.batch, len(tr)),), device=dev)]
    ks, eqs, rates, vqs, preds = model.rollout(OBS(b), An[b], S[b] if S is not None else None, SA[b] if SA is not None else None)
    mse = (preds - An[b]).pow(2).mean(-1)
    if NA == -1:                  # intent realisation: the behavioural head's own MSE is the imitation signal that reaches the memory
        mse = mse + args.high_w * (model._logits - An[b]).pow(2).mean(-1)
    if SA is not None:            # hierarchical: imitation = cross-entropy of the behavioural head; the low-level MSE trains pi_low only
        mse = mse + args.high_w * F.cross_entropy(model._logits.reshape(-1, model.na), SA_idx[b].reshape(-1), reduction="none").view(len(b), -1)
    dist = torch.zeros_like(mse)
    if args.distill:              # retrieve->commit: regress the history-derived target from (o_t, C_t)
        feat = torch.cat([model.g_s(S[b]) if S is not None else (torch.stack(model._eos, 1) if args.pixels else model.g_o(On[b])), eqs], -1)      # matched side information: the training-time head reads (f_t(o_t), C_t) only
        if args.distill_target == "feat": dist = args.distill_w * (dist_head(feat) - T_ets[b]).pow(2).sum(-1)
        elif args.distill_target == "random":   # one random future offset per (episode, step); offsets beyond the episode end are masked
            B_, T_ = feat.shape[:2]; jt = torch.randint(0, J_fc, (B_, T_), device=dev); gi = (jt[..., None] * da + torch.arange(da, device=dev)[None, None])
            tg = fc_tgt_all[b].gather(-1, gi); mk = fc_mask_student[b].gather(-1, gi)
            dist = args.distill_w * ((dist_head(torch.cat([feat, fc_jemb(jt + 1)], -1)) - tg).pow(2) * mk).sum(-1)
        else: dist = args.distill_w * ((dist_head(feat) - fc_tgt_all[b]).pow(2) * fc_mask_student[b]).sum(-1)
    if args.distill and args.distill_off > 0: dist = dist * float(np.clip((0.8 - it / args.steps) / max(0.8 - args.distill_off, 1e-6), 0.0, 1.0))
    beta_t = args.beta * min(1.0, (it + 1) / args.beta_warmup) if args.beta_warmup > 0 else args.beta
    bfs = torch.zeros_like(mse)
    if args.lam > 0 and args.pixels: raise SystemExit("BFS not supported with --pixels")
    if args.lam > 0:
        for t in range(On.shape[1] - 1):
            pr = model.bfs_pred(eqs, On[b], An[b], t, args.J, noobs=bool(args.bfs_noobs))
            if pr is None: continue
            Jt = pr.shape[1]; bfs[:, t] = (pr - An[b][:, t + 1:t + 1 + Jt]).pow(2).mean(-1).mean(-1)   # mean over j (toy finding)
    aux = torch.zeros_like(mse)
    if aux_head is not None:      # sys-ID baseline: theta must be decodable from (o_t, C_t) at every step -> the code carries log2 M bits
        _o = OBS(b); feat = torch.cat([(torch.stack([model.g_o((_o[0][:, t], _o[1][:, t])) for t in range(On.shape[1])], 1) if args.pixels else model.g_o(_o)), (model._eqs2 if args.dual_K > 0 else eqs)], -1)
        aux = F.cross_entropy(aux_head(feat).reshape(-1, M), theta_t[b].repeat_interleave(On.shape[1]), reduction="none").view(len(b), -1)
    if args.dual_K > 0:           # the identification carrier pays the same rate penalty and VQ losses as the behavioural carrier
        rates = rates + model._rates2; vqs = vqs + model._vqs2
    if join_head is not None:     # oracle-Gamma: the join must be decodable from (o_t, C_t) at every step (privileged; diagnostic only)
        _o = OBS(b); feat = torch.cat([(torch.stack([model.g_o((_o[0][:, t], _o[1][:, t])) for t in range(On.shape[1])], 1) if args.pixels else model.g_o(_o)), eqs], -1)
        aux = aux + args.aux_join * F.cross_entropy(join_head(feat).reshape(-1, R1 * R2), join_t[b].repeat_interleave(On.shape[1]), reduction="none").view(len(b), -1) / max(args.aux_theta, 1.0)
    mik = torch.zeros_like(mse)
    if mik_head is not None:      # multi-step inverse: a_t from (C_t, o_t, o_{t+j}); one random j per (episode, t), last step masked
        B_, T_ = mse.shape; ar = torch.arange(T_, device=dev)[None].expand(B_, -1)
        jt = torch.randint(1, args.mik_J + 1, (B_, T_), device=dev); jt = torch.minimum(jt, (T_ - 1 - ar).clamp(min=1)); idx = (ar + jt).clamp(max=T_ - 1)
        Ob = On[b]; go_now = model.g_o(Ob); go_fut = model.g_o(Ob[torch.arange(B_, device=dev)[:, None], idx])
        pr = mik_head(torch.cat([eqs, go_now, go_fut, mik_jemb(jt)], -1)); mik = (pr - An[b]).pow(2).mean(-1) * (ar < T_ - 1).float()
    loss = (mse + beta_t * rates + vqs + args.lam * bfs + (args.aux_theta if args.aux_theta > 0 else 1.0) * aux + args.aux_mik * mik + dist).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if args.refine and it >= args.refine_start and it % args.refine_every == 0:
        with torch.no_grad():      # neural partition refinement: split the code with the largest within-code future-inconsistency
            res = (bfs if args.refine == "bfs" else mse).detach(); gmean = float(res.mean()); best, bk, bt = 0.0, -1, -1
            Tn = res.shape[1]
            for t in range(Tn):
                for k in ks[:, t].unique().tolist():
                    m = ks[:, t] == k
                    if int(m.sum()) < args.refine_min: continue
                    r = float(res[m, t].mean())
                    if r > best: best, bk, bt = r, k, t
            used = torch.zeros(args.K, dtype=torch.bool, device=dev); used[ks.unique()] = True; free = (~used).nonzero().squeeze(-1)
            if bk >= 0 and best > args.refine_thresh * gmean and len(free) > 0:
                m = (ks[:, bt] == bk).nonzero().squeeze(-1); r = res[m, bt]; hi = m[r > r.median()]
                if len(hi) >= 2:
                    et = torch.stack(model._ets, 1)[hi, bt]
                    model.E_raw.data[free[0]] = et.mean(0) + 0.01 * torch.randn(args.d, device=dev)
                    n_refine = globals().get("n_refine", 0) + 1; globals()["n_refine"] = n_refine
                    if it % args.log_every == 0 or n_refine <= 5: print(f"  it {it:5d} refine #{n_refine}: split code {bk} at t={bt+1} (res {best:.4f} vs mean {gmean:.4f}, {len(hi)}/{len(m)}) -> code {int(free[0])}", flush=True)
    if it % 50 == 0 and it < args.steps // 2:
        with torch.no_grad():
            used = torch.zeros(args.K, dtype=torch.bool, device=dev); used[ks.unique()] = True
            if (~used).any():
                src = eqs.detach().reshape(-1, args.d); n = int((~used).sum())
                model.E_raw.data[~used] = src[torch.randint(0, len(src), (n,), device=dev)] + 0.01 * torch.randn(n, args.d, device=dev)
    if it % args.log_every == 0:
        print(f"  it {it:5d} loss {loss.item():.4f} mse {mse.mean().item():.4f} rate {rates.mean().item():.3f} bfs {bfs.mean().item():.4f} mik {mik.mean().item():.4f} dist {dist.mean().item():.4f} codes {ks.unique().numel()} ({time.time()-t0:.0f}s)", flush=True)

# ----------------------------------------------------------------------------- evaluation
model.alpha = 0.0     # scaffold off: the discrete code is the sole carrier at evaluation
model.eval()
logits_all = intent_all = None
with torch.no_grad():
    if args.pixels:
        parts = [model.rollout(OBS(torch.arange(i, min(i + 64, E), device=dev)), An[i:i + 64]) for i in range(0, E, 64)]
        ks, eqs, rates, vqs, preds = [torch.cat([p[j] for p in parts], 0) for j in range(5)]
    else:
        ks, eqs, rates, vqs, preds = model.rollout(On, An, S, SA)
        logits_all = model._logits.cpu() if SA is not None else None
        intent_all = model._logits.cpu() if NA == -1 else None
ks = ks.cpu().numpy(); preds = preds.cpu()
ks2 = model._ks2.cpu().numpy() if args.dual_K > 0 else None
w = [1.0 / E] * E
per_t = defaultdict(list)
for t in range(T):
    c = list(ks[:, t]); o = O_lab[t]; g = G_lab[t]; gam = Gam_lab[t]
    HC_O = cond_entropy(c, o, w); HG_O = cond_entropy(g, o, w); HGam_O = cond_entropy(gam, o, w)
    per_t["H(C|O)"].append(HC_O); per_t["H(G|O)"].append(HG_O); per_t["H(Gam|O)"].append(HGam_O)
    per_t["S_G"].append((HG_O - cond_entropy(g, list(zip(c, o)), w)) / HG_O if HG_O > 1e-9 else float("nan"))
    per_t["S_Gam"].append((HGam_O - cond_entropy(gam, list(zip(c, o)), w)) / HGam_O if HGam_O > 1e-9 else float("nan"))
    per_t["H(C|Gam,O)"].append(cond_entropy(c, list(zip(gam, o)), w))
    per_t["I(C;z|O)"].append(cond_entropy(z_lab, o, w) - cond_entropy(z_lab, list(zip(c, o)), w))
    per_t["I(C;theta|O)"].append(cond_entropy(theta_lab, o, w) - cond_entropy(theta_lab, list(zip(c, o)), w))
    per_t["I(C;mass|O)"].append(cond_entropy(mass_lab, o, w) - cond_entropy(mass_lab, list(zip(c, o)), w))
    per_t["rate_model"].append(float(rates[:, t].mean())); per_t["H(H|O)"].append(float(rates_theory["H(H|O)"][t]))
    if ks2 is not None:           # dual carrier: the identification code's own rate and world information, reported separately
        c2 = list(ks2[:, t]); per_t["H(C2|O)"].append(cond_entropy(c2, o, w)); per_t["I(C2;theta|O)"].append(cond_entropy(theta_lab, o, w) - cond_entropy(theta_lab, list(zip(c2, o)), w))
        per_t["I(C2;mass|O)"].append(cond_entropy(mass_lab, o, w) - cond_entropy(mass_lab, list(zip(c2, o)), w))
    per_t["mse_test"].append(float((preds[te_idx, t] - An[te_idx, t].cpu()).pow(2).mean()))
    if t in div_steps:
        which = div_steps[t]; R = R1 if which == "b1" else R2
        errs = []
        for i in te_idx:
            if logits_all is not None: errs.append(int(logits_all[i, t].argmax()) != int(SA_idx[i, t])); continue     # behavioural head vs recorded symbolic action
            if intent_all is not None:                                                                            # intent head decoded by nearest class mean
                dists = [(intent_all[i, t] - class_mean[(t, c)].cpu()).pow(2).sum().item() for c in range(R)]; errs.append(int(np.argmin(dists)) != eps[i][which]); continue
            dists = [(preds[i, t] - class_mean[(t, c)].cpu()).pow(2).sum().item() for c in range(R)]
            errs.append(int(np.argmin(dists)) != eps[i][which])
        per_t["class_err_test"].append(float(np.mean(errs))); per_t["class_err_b2"].append(float(np.mean(errs)) if which == "b2" else float("nan"))
    else:
        per_t["class_err_test"].append(float("nan")); per_t["class_err_b2"].append(float("nan"))
def pm(key, ph):
    idx = [i for i, p in enumerate(phases) if p == ph]; return float(np.nanmean([per_t[key][i] for i in idx]))
summary = dict(HC_gap1=pm("H(C|O)", "gap1"), HC_gap2=pm("H(C|O)", "gap2"), HGam_gap1=pm("H(Gam|O)", "gap1"), HGam_gap2=pm("H(Gam|O)", "gap2"),
               S_Gam_gap1=pm("S_Gam", "gap1"), S_Gam_gap2=pm("S_Gam", "gap2"), S_G_grasp=pm("S_G", "grasp"), S_G_place=pm("S_G", "place"),
               HC_Gam_O=float(np.nanmean(per_t["H(C|Gam,O)"])), I_Cz_O=float(np.nanmean(per_t["I(C;z|O)"])), I_Ctheta_O_gap2=pm("I(C;theta|O)", "gap2"), I_Cmass_O_gap2=pm("I(C;mass|O)", "gap2"),
               class_err_test=float(np.nanmean(per_t["class_err_test"])), class_err_b2=float(np.nanmean(per_t["class_err_b2"])), mse_test=float(np.mean(per_t["mse_test"])), codes=int(len(np.unique(ks))))
if ks2 is not None:
    summary.update(HC2_gap1=pm("H(C2|O)", "gap1"), HC2_gap2=pm("H(C2|O)", "gap2"), HC2_grasp=pm("H(C2|O)", "grasp"), I_C2theta_O_gap2=pm("I(C2;theta|O)", "gap2"), I_C2mass_O_gap2=pm("I(C2;mass|O)", "gap2"), codes2=int(len(np.unique(ks2))))
    print(f"[dual] identification carrier: H(C2|O) gap1 {summary['HC2_gap1']:.2f} grasp {summary['HC2_grasp']:.2f} gap2 {summary['HC2_gap2']:.2f} | I(C2;theta|O) gap2 {summary['I_C2theta_O_gap2']:.2f} | codes {summary['codes2']}", flush=True)
print("\n t  phase    H(G|O) H(Gam|O) H(C|O)  S_G  S_Gam H(C|Gam,O) I(C;z|O) cls_err")
for t in range(T):
    ce = per_t["class_err_test"][t]
    print(f"{t+1:2d}  {phases[t]:8s} {per_t['H(G|O)'][t]:5.2f}  {per_t['H(Gam|O)'][t]:5.2f}   {per_t['H(C|O)'][t]:5.2f}  "
          f"{per_t['S_G'][t] if not math.isnan(per_t['S_G'][t]) else float('nan'):4.2f} {per_t['S_Gam'][t] if not math.isnan(per_t['S_Gam'][t]) else float('nan'):4.2f}  "
          f"{per_t['H(C|Gam,O)'][t]:5.2f}      {per_t['I(C;z|O)'][t]:5.2f}   {'' if math.isnan(ce) else f'{ce:.2f}'}")
if args.save_model:
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    save_model(args.save_model, model, o_mu, o_sd, a_mu, a_sd, extra=dict(args=vars(args), summary=summary, obs_dim=int(O.shape[-1]), act_dim=int(A.shape[-1]), sym_binner=sym_binner, act_vocab=act_vocab))
if args.save_codes:
    os.makedirs(os.path.dirname(args.save_codes) or ".", exist_ok=True)
    np.savez_compressed(args.save_codes, codes=ks, te_idx=te_idx, tr_idx=tr_idx, preds=preds.numpy(), theta=[e["theta"] for e in eps], z=z_lab)
print("SUMMARY", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(dict(args=vars(args), sym=sym_key, visible=vis, phases=phases, sym_disagree=sym_disagree, train_modes=train_modes, theory={k: [float(x) for x in v] for k, v in rates_theory.items()}, per_t={k: [float(x) for x in v] for k, v in per_t.items()}, summary=summary, fc=(fc_info if args.distill and args.distill_target != 'feat' else None), n_refine=globals().get('n_refine', 0), secs=time.time() - t0, device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))) + "\n")
