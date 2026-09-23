"""External benchmark: Passive T-Maze of Ni et al. (2023), taken unmodified from MIKASA-Base
(external/MIKASA-Base/mikasa_base/Passive_T_Maze/env/env_passive_t_maze.py, class TMazeClassicPassive: ambiguous position, the goal
cue is shown in the FIRST observation only, the decision is taken corridor_length steps later).  No exact-rate measurement is attempted
beyond H(C) on the corridor: the question is whether generic future-behaviour supervision changes closed-loop success and how that scales
with the memory horizon.

Behaviour cloning of the scripted oracle (move right corridor_length times, then up / down according to the cue) with the paper's K=16
sole-carrier recurrent policy (isaac/diacritic_model.py, unchanged):
    plain     imitation + rate
    generic   + training-only head: from (o_t, C_t) predict the expert action at ONE random future offset j ~ U{1..T-1} per (episode, step)
              (no knowledge of the decision step; targets are the recorded future actions -- the task is deterministic given the cue)
    bypass / transformer   full-history references (continuous GRU state / causal attention carry the memory instead of the code)
Closed loop: the environment is rolled out with the learned policy (argmax action); success = goal reward at the end of the episode.

Usage: python external/tmaze_bc.py --L 50 --mode generic --seed 0 --out external/results/tmaze.jsonl
"""
import sys, os, json, time, argparse, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "isaac"))
from diacritic_model import DIACRITIC
spec = importlib.util.spec_from_file_location("tmaze_ext", os.path.join(HERE, "MIKASA-Base", "mikasa_base", "Passive_T_Maze", "env", "env_passive_t_maze.py"))
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=50); ap.add_argument("--mode", default="plain", help="plain | generic | bypass | transformer | gru | gru_generic | rf (BFS future decoder WITH future observations) | scaffold (annealed continuous scaffold, zero at evaluation) | gru4 / gru8 (GRU policy with a 4- / 8-dimensional state)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--beta", type=float, default=1e-3); ap.add_argument("--K", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--n_demo", type=int, default=256); ap.add_argument("--n_eval", type=int, default=100); ap.add_argument("--distill_w", type=float, default=1.0)
ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device(args.device)

# ------------------------------------------------------------------ demonstrations from the scripted oracle, recorded in the external environment
env = tm.TMazeClassicPassive(corridor_length=args.L, penalty=0.0)
A_RIGHT = next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (1, 0))
A_Y = {+1: next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (0, 1)), -1: next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (0, -1))}
def oracle_episode(seed):
    o, _ = env.reset(seed=seed); g = int(env.goal_y); obs, act = [], []
    for t in range(env.episode_length):
        a = A_RIGHT if t < args.L else A_Y[g]
        obs.append(o); act.append(a); o, r, done, _, _ = env.step(a)
    return np.stack(obs), np.array(act), g, float(r)
demos = [oracle_episode(10_000 + i) for i in range(args.n_demo)]
assert all(d[3] == 1.0 for d in demos), "oracle must solve the task"
O = torch.tensor(np.stack([d[0] for d in demos]), dtype=torch.float32, device=dev)                  # (E,T,2)
A = F.one_hot(torch.tensor(np.stack([d[1] for d in demos]), device=dev), 4).float()                  # (E,T,4) one-hot actions
G = np.array([d[2] for d in demos]); E, T, _ = O.shape

class GRUPolicy(nn.Module):
    """Standard recurrent BC reference (no bottleneck, no quantisation): GRU over (o_t, a_{t-1}), policy reads the continuous hidden state.
    Same interface as DIACRITIC so that the training / evaluation code below is shared (codes = 0, rate = 0)."""
    def __init__(self, do, da, hid=128):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(do + da, hid), nn.ReLU()); self.cell = nn.GRUCell(hid, hid); self.pi = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da)); self.hid, self.da = hid, da
        self.g_o = lambda o: torch.zeros(*o.shape[:-1], 0, device=o.device); self.E_raw = nn.Parameter(torch.zeros(1, 1))
    def init_state(self, B, device): return torch.zeros(B, self.hid, device=device), torch.zeros(B, self.hid, device=device), torch.zeros(B, self.da, device=device)
    def step(self, o, e, h, a_prev):
        h = self.cell(self.inp(torch.cat([o, a_prev], -1)), h); z = torch.zeros(o.shape[0], device=o.device)
        return z.long(), h, None, z, self.pi(h), h
    def rollout(self, O, A):
        B, T, _ = O.shape; e, h, a_prev = self.init_state(B, O.device); hs, ps = [], []
        for t in range(T):
            _, _, _, _, p_, h = self.step(O[:, t], e, h, a_prev); hs.append(h); ps.append(p_); a_prev = A[:, t]
        z = torch.zeros(B, T, device=O.device); return z.long(), torch.stack(hs, 1), z, z, torch.stack(ps, 1)
GRU_MODE = args.mode in ("gru", "gru_generic", "gru4", "gru8")
variant = args.mode if args.mode in ("bypass", "transformer", "scaffold") else "diacritic"
model = GRUPolicy(O.shape[-1], 4, hid={"gru4": 4, "gru8": 8}.get(args.mode, 128)).to(dev) if GRU_MODE else DIACRITIC(O.shape[-1], 4, args.K, 32, 128, 1.0, True, 0.25, variant, "none").to(dev)
if variant == "transformer" and T > model.tf_pos.shape[0]: model.tf_pos = nn.Parameter(torch.zeros(T, model.tf_pos.shape[1], device=dev))
head = jemb = None
if args.mode in ("generic", "gru_generic"):
    head = nn.Sequential(nn.Linear((128 if GRU_MODE else 2 * 32) + 16, 128), nn.ReLU(), nn.Linear(128, 4)).to(dev); jemb = nn.Embedding(T, 16).to(dev)
opt = torch.optim.Adam(list(model.parameters()) + (list(head.parameters()) + list(jemb.parameters()) if head is not None else []), lr=args.lr)
t0 = time.time(); ar = torch.arange(T, device=dev)
for it in range(args.steps):
    b = torch.randint(0, E, (args.batch,), device=dev)
    if args.mode == "scaffold": model.alpha = 1.0 if it < 0.2 * args.steps else max(0.0, 1.0 - (it - 0.2 * args.steps) / (0.4 * args.steps))      # hold 20 %, anneal to 0 by 60 %
    ks, eqs, rates, vqs, preds = model.rollout(O[b], A[b])
    ce = F.cross_entropy(preds.reshape(-1, 4), A[b].argmax(-1).reshape(-1), reduction="none").view(len(b), T)
    dist = torch.zeros_like(ce)
    if head is not None:
        jt = torch.randint(1, T, (len(b), T), device=dev); idx = ar[None] + jt; valid = (idx < T).float(); idx = idx.clamp(max=T - 1)
        tgt = A[b].argmax(-1).gather(1, idx)
        logits = head(torch.cat([model.g_o(O[b]), eqs, jemb(jt)], -1))
        dist = args.distill_w * F.cross_entropy(logits.reshape(-1, 4), tgt.reshape(-1), reduction="none").view(len(b), T) * valid
    if args.mode == "rf":      # -RF: future-behaviour decoder that receives the future observations and actions (8 random start steps per update)
        bl = 0.0
        for t_ in np.random.randint(0, T - 1, 8):
            pr = model.bfs_pred(eqs, O[b], A[b], int(t_), 100); bl = bl + F.cross_entropy(pr.reshape(-1, 4), A[b][:, t_ + 1:t_ + 1 + pr.shape[1]].argmax(-1).reshape(-1))
        dist = dist + bl / 8
    beta_t = args.beta * min(1.0, (it + 1) / 1000)
    loss = (ce + beta_t * rates + vqs + dist).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if not GRU_MODE and it % 50 == 0 and it < args.steps // 2:      # dead-code revival, as in the paper's trainer
        with torch.no_grad():
            used = torch.zeros(args.K, dtype=torch.bool, device=dev); used[ks.unique()] = True
            if (~used).any():
                src = eqs.detach().reshape(-1, 32); n = int((~used).sum()); model.E_raw.data[~used] = src[torch.randint(0, len(src), (n,), device=dev)] + 0.01 * torch.randn(n, 32, device=dev)
    if it % 500 == 0: print(f"  it {it:5d} ce {ce.mean().item():.4f} rate {rates.mean().item():.3f} dist {dist.mean().item():.4f} ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ teacher-forced diagnostics + closed loop in the external environment
model.alpha = 0.0; model.eval()
with torch.no_grad():
    ks, eqs, rates, vqs, preds = model.rollout(O, A); ks = ks.cpu().numpy()
    tf_dec_acc = float((preds[:, -1].argmax(-1) == A[:, -1].argmax(-1)).float().mean())
def H(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
def I_code_goal(t):
    return H(ks[:, t]) + H(G) - H(ks[:, t] * 10 + (G > 0))
info_mid, info_last, rate_mid = I_code_goal(T // 2), I_code_goal(T - 1), H(ks[:, T // 2])
succ = 0; dec_ok = 0
with torch.no_grad():
    for i in range(args.n_eval):
        o, _ = env.reset(seed=500_000 + i); e, h, a_prev = model.init_state(1, dev); r = 0.0
        for t in range(env.episode_length):
            ot = torch.tensor(o, dtype=torch.float32, device=dev)[None]
            k, eq, oh, vq, pred, h = model.step(ot, e, h, a_prev); a = int(pred.argmax(-1)); e = eq; a_prev = F.one_hot(torch.tensor([a], device=dev), 4).float()
            if t == env.episode_length - 1: dec_ok += int(a == A_Y[int(env.goal_y)])
            o, r, done, _, _ = env.step(a)
        succ += int(r > 0)
res = dict(L=args.L, mode=args.mode, seed=args.seed, K=args.K, beta=args.beta, steps=args.steps, success=succ / args.n_eval, decision_acc=dec_ok / args.n_eval, tf_decision_acc=tf_dec_acc,
           I_code_goal_mid=info_mid, I_code_goal_last=info_last, H_code_mid=rate_mid, secs=time.time() - t0)
print("RESULT", json.dumps(res), flush=True)
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f: f.write(json.dumps(res) + "\n")
