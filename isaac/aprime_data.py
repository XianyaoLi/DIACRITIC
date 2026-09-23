"""Shared loader for recorded A' datasets + construction of the finite symbolic Env for the exact solver.

Symbolic observation at step t = (phase, probe_bit, visible_b1, visible_b2), where a class is *visible* at t when it
is decodable from the raw observation at t.  Rule (data-driven, auditable): the class-conditional means of the
end-effector position at step t are separated by more than `vis_thresh` (default 5 cm, >> 2 mm sensor noise; computed conditionally on the other factor).
Without this augmentation, information that the raw observation already provides (e.g. the slot under the gripper
after the place motion, the approach side during the grasp) would be wrongly counted as memory in H(C | O_sym).
"""
import os, glob, json, ast
import numpy as np
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy"))
from toy_env import Env
from gamma_solver import GammaSolver

TCP = slice(0, 3)   # obs layout: tcp_pos(3) first


def load_episodes(d):
    eps = []
    for f in sorted(glob.glob(os.path.join(d, "ep*_env*.npz"))):
        z = np.load(f, allow_pickle=True)
        eps.append(dict(obs=z["obs"].astype(np.float32), act=z["act"].astype(np.float32), theta=int(z["theta"]), z=int(z["z"]), mass=float(z["mass"]),
                        b1=int(z["b1"]), b2=int(z["b2"]), hover_k=int(z["hover_k"]), success=bool(z["success"]),
                        sym_obs=[ast.literal_eval(s) for s in z["sym_obs"]],
                        sym_act=[ast.literal_eval(s) if s.startswith("(") else s for s in z["sym_act"]], phase=list(z["phase"])))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return eps, meta


def visibility(eps, meta, vis_thresh=0.05):
    """Per step: is b1 / b2 decodable from the raw tcp position (class-mean separation > thresh)?"""
    T = len(eps[0]["phase"]); R1, R2 = meta["R1"], meta["R2"]
    tcp = np.stack([e["obs"][:, TCP] for e in eps])           # (E,T,3)
    vis = {"b1": [False] * T, "b2": [False] * T}
    for which, other, R, Ro in (("b1", "b2", R1, R2), ("b2", "b1", R2, R1)):
        lab = np.array([e[which] for e in eps]); lab_o = np.array([e[other] for e in eps])
        for t in range(T):
            seps = []
            for co in range(Ro):                      # condition on the other factor: removes its sampling variance
                m = lab_o == co
                means = np.stack([tcp[m & (lab == c), t].mean(0) for c in range(R) if (m & (lab == c)).any()])
                if len(means) > 1:
                    seps.append(max(np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))))
            if not seps:                              # nested classes (the other factor determines this one, e.g. readout tasks): unconditional means
                means = np.stack([tcp[lab == c, t].mean(0) for c in range(R) if (lab == c).any()])
                if len(means) > 1: seps.append(max(np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))))
            vis[which][t] = bool(seps and np.mean(seps) > vis_thresh)
    return vis


def sag_visibility(eps, R, thresh=0.006):
    """Weigh task: the mass class is *visible* at step t when the class-conditional means of the tcp height are pairwise separated by more
    than `thresh` (6 mm = 3 sd of the position noise): the arm sags under the held object.  Same data-driven convention as `visibility`."""
    T = len(eps[0]["phase"]); z = np.stack([e["obs"][:, 2] for e in eps]); lab = np.array([e["b1"] for e in eps]); out = []
    for t in range(T):
        m = sorted(z[lab == c, t].mean() for c in range(R) if (lab == c).any())
        out.append(bool(len(m) > 1 and min(b - a for a, b in zip(m, m[1:])) > thresh))
    return out


def augment_sym_obs(eps, vis, sag_symbol=False, sag_phases=("gap2", "place"), weigh_vis=None):
    """Symbolic obs = (phase, probe, visible b1, visible b2[, sag]).  sag_symbol (task A, physical grasp): a 2-level symbol of the
    tcp height during transport, thresholded at the largest gap between mass-sorted class means -> a deterministic function of the
    hidden mass class that the observation reveals (agent-centric, behaviourally irrelevant) -> A4 fails by design (general regime)."""
    # visibility-convention analogue for the mass: the leak probe shows the mass half is decodable from the raw tcp height in the
    # sag phases (physical grasp), so the symbolic observation carries the mass-half indicator there (deterministic in theta).
    med = None
    if sag_symbol:
        masses = sorted({float(e["mass"]) for e in eps}); med = masses[len(masses) // 2]
    for e in eps:
        e["sym_obs_aug"] = []
        for t, o in enumerate(e["sym_obs"]):
            sym = (o[0], o[1], e["b1"] if vis["b1"][t] else -1, e["b2"] if vis["b2"][t] else -1)
            if sag_symbol and e["phase"][t] in sag_phases:
                sym = sym + (int(float(e["mass"]) >= med),)
            if weigh_vis is not None:      # weigh task: the sag shows the mass class (= b1; b2 is a function of it) while the object is held
                sym = sym + ((e["b1"] if weigh_vis[t] else -1),)
            e["sym_obs_aug"].append(sym)
    return eps


def build_env(eps, meta, sym_key="sym_obs_aug"):
    """Finite Env whose obs/expert are the recorded symbolic sequences (deterministic functions of theta; checked)."""
    R1, R2, M, N = meta["R1"], meta["R2"], meta["M"], meta["N"]
    T = len(eps[0]["phase"])
    # latent key: theta alone (A'), or (theta, g) when the slot g is an independent latent (task A --split_latent): detect from data
    split = len({(e["theta"], e["b2"]) for e in eps}) > len({e["theta"] for e in eps})
    keyf = (lambda e: (e["theta"], e["b2"])) if split else (lambda e: e["theta"])
    obs_tab, act_tab = {}, {}
    for e in eps:
        k = keyf(e); o = tuple(e[sym_key]); a = tuple(e["sym_act"][1:])
        if k in obs_tab:
            assert obs_tab[k] == o and act_tab[k] == a, f"symbolic obs/act not a function of the latent ({k})"
        obs_tab[k], act_tab[k] = o, a
    if split:
        keys = [(th, g) for th in range(M) for g in range(R2)]
        # Unrecorded (theta, g) keys: the symbolic sequence is a known deterministic function of the key -- the early probe emits the
        # LSB-first binary code of theta during the scan steps, the sag symbol is the mass half of theta, everything else depends on g
        # only.  So we take a recorded donor with the same g and the same mass half and overwrite the scan-step probe bits with theta's
        # own bits (copying a donor verbatim, as an earlier version did, would give the unseen key another theta's probe bits and create a
        # spurious strong-congruence class).  The construction is verified below by rebuilding every recorded key from a different donor.
        phases = [p_ for p_ in eps[0]["phase"]]; B = sum(1 for p_ in phases if p_ == "scan"); half = M // 2
        def build_from(th, g, donor):
            bits = [(th >> i) & 1 for i in range(B)]; o = list(obs_tab[donor]); j = 0
            for t, p_ in enumerate(phases):
                if p_ == "scan": o[t] = (o[t][0], bits[j]) + tuple(o[t][2:]); j += 1
            return tuple(o), act_tab[donor]
        recorded = list(obs_tab)
        for k in recorded:      # self-check: every recorded key is reproduced exactly from a different donor with the same (g, mass half)
            donors = [d for d in recorded if d != k and d[1] == k[1] and (d[0] < half) == (k[0] < half)]
            if donors:
                o, a = build_from(k[0], k[1], donors[0]); assert o == obs_tab[k] and a == act_tab[k], f"symbolic reconstruction failed for key {k}"
        for k in keys:
            if k not in obs_tab:
                donors = [d for d in recorded if d[1] == k[1] and (d[0] < half) == (k[0] < half)]; assert donors, k
                obs_tab[k], act_tab[k] = build_from(k[0], k[1], donors[0])
        prior = {(th, g, z): 1.0 / (M * R2 * N) for th in range(M) for g in range(R2) for z in range(N)}
        env = Env(T, prior, lambda lat, t, acts: {obs_tab[(lat[0], lat[1])][t - 1]: 1.0},
                  lambda lat, t, hist: {("hover", 0): 0.5, ("hover", 1): 0.5} if t == 1 else {act_tab[(lat[0], lat[1])][t - 2]: 1.0},
                  name="A-recorded(split)", latent_fields=("theta", "g", "z"))
        return env, obs_tab, act_tab
    readout = meta.get("reveal") in ("readout", "readout_bits")
    for th in range(M):
        if th not in obs_tab:
            if meta.get("reveal") == "weigh":      # monotone mass classes, nothing displayed: any same-class recording has the same symbolic sequence
                src = next(s for s in obs_tab if s * R1 // M == th * R1 // M); obs_tab[th], act_tab[th] = obs_tab[src], act_tab[src]
            elif readout:      # monotone classes; the scan-step probe symbol is theta itself, so rebuild it from a same-class donor
                src = next(s for s in obs_tab if s * R1 // M == th * R1 // M and s * R2 // M == th * R2 // M)
                o = list(obs_tab[src]); phases_ = eps[0]["phase"]
                for t, p_ in enumerate(phases_):
                    if p_ == "scan": o[t] = (o[t][0], th) + tuple(o[t][2:])
                obs_tab[th], act_tab[th] = tuple(o), act_tab[src]
            else:
                src = next(s for s in obs_tab if s % R1 == th % R1 and (s // R1) % R2 == (th // R1) % R2)
                obs_tab[th], act_tab[th] = obs_tab[src], act_tab[src]
    prior = {(th, z): 1.0 / (M * N) for th in range(M) for z in range(N)}
    env = Env(T, prior, lambda lat, t, acts: {obs_tab[lat[0]][t - 1]: 1.0},
              lambda lat, t, hist: {("hover", 0): 0.5, ("hover", 1): 0.5} if t == 1 else {act_tab[lat[0]][t - 2]: 1.0},
              name="A'-recorded", latent_fields=("theta", "z"))
    return env, obs_tab, act_tab


def solver_labels(solver, eps, sym_key="sym_obs_aug"):
    """Per t (1-indexed list index t-1): per-episode labels G, Gamma, GammaS, O from the solver via the symbolic history prefix."""
    T = len(eps[0]["phase"]); G, Gam, GamS, O = [], [], [], []
    for t in range(1, T + 1):
        idx = []
        for e in eps:
            hist = []
            for s in range(t):
                hist.append(e[sym_key][s])
                if s < t - 1: hist.append(e["sym_act"][s])
            idx.append(solver.index[t][tuple(hist)])
        G.append([solver.G[t][i] for i in idx]); Gam.append([solver.Gamma[t][i] for i in idx])
        GamS.append([solver.GammaS[t][i] for i in idx]); O.append([solver.O[t][i] for i in idx])
    return G, Gam, GamS, O


def prepare(d, vis_thresh=0.05, augment=True, sag_symbol=False):
    eps, meta = load_episodes(d)
    vis = visibility(eps, meta, vis_thresh)
    wv = sag_visibility(eps, meta["R1"]) if meta.get("reveal") == "weigh" else None
    if wv is not None: vis = dict(vis, sag=wv)
    augment_sym_obs(eps, vis, sag_symbol=sag_symbol, weigh_vis=wv)
    key = "sym_obs_aug" if augment else "sym_obs"
    env, obs_tab, act_tab = build_env(eps, meta, key)
    solver = GammaSolver(env).solve()
    return eps, meta, vis, env, solver, key
