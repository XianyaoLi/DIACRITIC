"""CERTIFY the behavioural-memory requirement of community memory benchmarks and compare the learned code rate with it.
Tasks (environment code unmodified, from external/MIKASA-Base):
    tmaze     Passive T-maze of Ni et al. (2023)  (TMazeClassicPassive; cue in the first observation, decision corridor_length steps later)
    memchain  bsuite MemoryChain / 'MemoryLength' (context of num_bits bits in the first observation, query index shown on the last step,
              the final action must repeat the queried bit)
Certification: the hidden variable is enumerable (goal; (context, query)), so we drive the EXTERNAL environment once per hidden value with the
scripted oracle, record the observation/action sequences verbatim as symbols, build the induced finite POMDP and run the exact solver
(toy/gamma_solver.py): (A2)/(A4)/transitivity checks and the per-step requirement H(Gamma_t | O_t).
Learning: behaviour cloning of the oracle with the paper's unchanged K=16 sole-carrier model (plain, or + event-agnostic random-offset
forecast of the recorded future actions); metrics per step with the solver's labels: H(C|O), S_Gamma; gate = S_Gamma > 0.9 at every step with
a positive requirement AND closed-loop success >= 0.99 in the external environment.  Rates are reported for seeds that pass the gate.
Usage: python external/cert_bc.py --task memchain --L 10 --bits 3 --mode generic --seed 0 --out external/results/cert.jsonl
       python external/cert_bc.py --task memchain --L 30 --bits 2 --certify_only"""
import sys, os, json, time, argparse, itertools, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "isaac")); sys.path.insert(0, os.path.join(HERE, "..", "toy"))
from diacritic_model import DIACRITIC
from toy_env import Env
from gamma_solver import GammaSolver, cond_entropy
def _load(name, *rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "MIKASA-Base", "mikasa_base", *rel)); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="memchain"); ap.add_argument("--L", type=int, default=10); ap.add_argument("--bits", type=int, default=1)
ap.add_argument("--mode", default="plain", help="plain | generic"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--beta", type=float, default=1e-3); ap.add_argument("--K", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--n_demo", type=int, default=512); ap.add_argument("--n_eval", type=int, default=200); ap.add_argument("--certify_only", action="store_true"); ap.add_argument("--out", default="")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device(args.device)

# ------------------------------------------------------------------ external environments + scripted oracles
if args.task == "tmaze":
    tm = _load("tmaze_ext", "Passive_T_Maze", "env", "env_passive_t_maze.py"); env = tm.TMazeClassicPassive(corridor_length=args.L, penalty=0.0); NA = 4
    A_RIGHT = next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (1, 0)); A_Y = {s: next(a for a, mv in enumerate(env.action_mapping) if tuple(mv) == (0, s)) for s in (1, -1)}
    def latent_of(): return int(env.goal_y)
    def oracle(t, lat): return A_RIGHT if t < args.L else A_Y[lat]
    T_EP = env.episode_length
else:
    import types; pkg = types.ModuleType("bsx"); pkg.__path__ = [os.path.join(HERE, "MIKASA-Base", "mikasa_base", "Bsuite", "env")]; sys.modules["bsx"] = pkg
    _load("bsx.base", "Bsuite", "env", "base.py"); _load("bsx.discounting_chain", "Bsuite", "env", "discounting_chain.py"); _load("bsx.memory_chain", "Bsuite", "env", "memory_chain.py")
    bw = _load("bsx.bsuite_env", "Bsuite", "env", "bsuite_env.py"); env = bw.BsuiteGymWrapper("MemoryLength", memory_length=args.L, num_bits=args.bits); NA = 2
    def latent_of(): return (tuple(int(b) for b in env._env._context), int(env._env._query))
    def oracle(t, lat): return int(lat[0][lat[1]]) if t == T_EP - 1 else 0          # actions before the last step have no effect; the oracle plays 0
    T_EP = args.L + 1
def episode(seed, policy=None):
    o, _ = env.reset(seed=seed); lat = latent_of(); obs, act, r, done, t = [], [], 0.0, False, 0
    while not done:
        a = oracle(t, lat) if policy is None else policy(o, t); obs.append(np.asarray(o, np.float32).ravel()); act.append(a)
        o, r, term, trunc, _ = env.step(a); done = bool(term or trunc); t += 1
    return np.stack(obs), np.array(act), lat, float(r)
demos = [episode(10_000 + i) for i in range(args.n_demo)]
assert all(d[3] > 0 for d in demos), "the oracle must solve the external task"; T = len(demos[0][1]); assert T == T_EP, (T, T_EP)

# ------------------------------------------------------------------ certification on the induced finite POMDP (symbols = the external env's own observations)
sym = lambda o: tuple(np.round(o, 6).tolist())
tab = {}
for ob, ac, lat, _ in demos: tab.setdefault(lat, (tuple(sym(x) for x in ob), tuple(int(a) for a in ac)))
n_lat = 2 if args.task == "tmaze" else (2 ** args.bits) * args.bits
assert len(tab) == n_lat, f"only {len(tab)} of {n_lat} hidden values were recorded; increase --n_demo"
fenv = Env(T, {k: 1.0 / n_lat for k in tab}, lambda lat, t, acts: {tab[lat][0][t - 1]: 1.0}, lambda lat, t, hist: {tab[lat][1][t - 1]: 1.0}, name=f"{args.task}-recorded", latent_fields=("latent",))
solver = GammaSolver(fenv).solve(); R = solver.rates()
cert = dict(task=args.task, L=args.L, bits=args.bits, T=T, hidden_values=n_lat, transitive=bool(all(solver.transitive.values())), a4_fail=[t for t, v in solver.a4_ok.items() if not v],
            H_Gamma=[round(float(x), 4) for x in R["H(Gamma|O)"]], H_G=[round(float(x), 4) for x in R["H(G|O)"]], H_H=[round(float(x), 4) for x in R["H(H|O)"]])
print("CERT", json.dumps(cert), flush=True)
if args.certify_only:
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); open(args.out, "a").write(json.dumps(dict(kind="cert", **cert)) + "\n")
    sys.exit(0)
def labels(ob, ac):
    G, Ol = [], []
    for t in range(1, T + 1):
        hist = []
        for s in range(t):
            hist.append(sym(ob[s]))
            if s < t - 1: hist.append(int(ac[s]))
        i = solver.index[t][tuple(hist)]; G.append(solver.Gamma[t][i]); Ol.append(solver.O[t][i])
    return G, Ol
LAB = [labels(d[0], d[1]) for d in demos]

# ------------------------------------------------------------------ behaviour cloning with the unchanged sole-carrier model
O = torch.tensor(np.stack([d[0] for d in demos]), device=dev); A = F.one_hot(torch.tensor(np.stack([d[1] for d in demos]), device=dev), NA).float(); E = len(demos)
model = DIACRITIC(O.shape[-1], NA, args.K, 32, 128, 1.0, True, 0.25, "diacritic", "none").to(dev); head = jemb = None
if args.mode == "generic": head = nn.Sequential(nn.Linear(2 * 32 + 16, 128), nn.ReLU(), nn.Linear(128, NA)).to(dev); jemb = nn.Embedding(T, 16).to(dev)
opt = torch.optim.Adam(list(model.parameters()) + (list(head.parameters()) + list(jemb.parameters()) if head is not None else []), lr=args.lr); ar = torch.arange(T, device=dev); t0 = time.time()
for it in range(args.steps):
    b = torch.randint(0, E, (args.batch,), device=dev); ks, eqs, rates, vqs, preds = model.rollout(O[b], A[b])
    ce = F.cross_entropy(preds.reshape(-1, NA), A[b].argmax(-1).reshape(-1), reduction="none").view(len(b), T); dist = torch.zeros_like(ce)
    if head is not None:      # event-agnostic: one random future offset per (episode, step), recorded future action as the target, no decision-time knowledge
        jt = torch.randint(1, T, (len(b), T), device=dev); idx = ar[None] + jt; valid = (idx < T).float(); idx = idx.clamp(max=T - 1)
        dist = F.cross_entropy(head(torch.cat([model.g_o(O[b]), eqs, jemb(jt)], -1)).reshape(-1, NA), A[b].argmax(-1).gather(1, idx).reshape(-1), reduction="none").view(len(b), T) * valid
    loss = (ce + args.beta * min(1.0, (it + 1) / 1000) * rates + vqs + dist).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    if it % 50 == 0 and it < args.steps // 2:
        with torch.no_grad():
            used = torch.zeros(args.K, dtype=torch.bool, device=dev); used[ks.unique()] = True
            if (~used).any(): src = eqs.detach().reshape(-1, 32); n = int((~used).sum()); model.E_raw.data[~used] = src[torch.randint(0, len(src), (n,), device=dev)] + 0.01 * torch.randn(n, 32, device=dev)
model.eval()
with torch.no_grad(): ks = model.rollout(O, A)[0].cpu().numpy()
w = [1.0 / E] * E; HC, HGam, S = [], [], []
for t in range(T):
    c = list(ks[:, t]); gam = [LAB[i][0][t] for i in range(E)]; o = [LAB[i][1][t] for i in range(E)]; hg = cond_entropy(gam, o, w)
    HC.append(cond_entropy(c, o, w)); HGam.append(hg); S.append((hg - cond_entropy(gam, list(zip(c, o)), w)) / hg if hg > 1e-9 else float("nan"))
def policy_factory():
    st = {}
    def pol(o, t):
        if t == 0: st["e"], st["h"], st["a"] = model.init_state(1, dev)
        with torch.no_grad(): k, eq, oh, vq, pred, st["h"] = model.step(torch.tensor(np.asarray(o, np.float32).ravel(), device=dev)[None], st["e"], st["h"], st["a"])
        a = int(pred.argmax(-1)); st["e"] = eq; st["a"] = F.one_hot(torch.tensor([a], device=dev), NA).float(); return a
    return pol
pol = policy_factory(); succ = float(np.mean([episode(500_000 + i, pol)[3] > 0 for i in range(args.n_eval)]))
need = [t for t in range(T) if HGam[t] > 1e-9]; suff = bool(all(S[t] > 0.9 for t in need) and succ >= 0.99); mid = need[len(need) // 2]
res = dict(kind="run", task=args.task, L=args.L, bits=args.bits, mode=args.mode, seed=args.seed, K=args.K, beta=args.beta, steps=args.steps, success=succ, sufficient=suff, min_S_Gam=float(np.nanmin([S[t] for t in need])),
           rate_mid=float(HC[mid]), req_mid=float(HGam[mid]), rate_mean_delay=float(np.mean([HC[t] for t in need])), req_mean_delay=float(np.mean([HGam[t] for t in need])), HC=[round(x, 3) for x in HC], secs=time.time() - t0)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "HC"}), flush=True)
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); open(args.out, "a").write(json.dumps(res) + "\n")
