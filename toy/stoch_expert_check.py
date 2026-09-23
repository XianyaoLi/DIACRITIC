"""Stochastic-expert check: Definition 1 groups hidden states by the expert's ACTION DISTRIBUTION, but every other experiment uses a deterministic expert.
Finite check with a stochastic expert: three equiprobable hidden states s in {0,1,2}; s is displayed at t=1, then hidden; at t=3 the expert draws
a in {L, R} with P(L | s) = p, p, 1-p (p = 0.7 by default; --p 0.9 for a sharper contrast).  States 0 and 1 induce the SAME non-degenerate law, so the behavioural quotient must merge them:
H(G|O) = H(Gamma|O) = h2(1/3) = 0.918 bit, not log2 3 = 1.585.
(1) exact solver on the finite POMDP; (2) the paper's sole-carrier learner (hard VQ, conditional-rate penalty) cloned from sampled demonstrations:
rate at the waiting step, KL between the learned and the true action law per hidden state, and whether states 0/1 share a code.
Usage: python toy/stoch_expert_check.py [--seeds 8] > results_md/analysis_stochastic_expert.md"""
import sys, os, json, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "isaac"))
from toy_env import Env
from gamma_solver import GammaSolver
from diacritic_model import DIACRITIC
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--p", type=float, default=0.7, help="P(L | s) for s = 0, 1; s = 2 uses 1 - p"); ap.add_argument("--betas", default="0,0.01,0.03,0.1"); args = ap.parse_args()
PL = [args.p, args.p, 1 - args.p]; T = 3; NOOP, L, R = 2, 0, 1
env = Env(T, {s: 1 / 3 for s in range(3)}, lambda s, t, acts: {(("show", s) if t == 1 else ("blank",)): 1.0},
          lambda s, t, hist: ({NOOP: 1.0} if t < 3 else {L: PL[s], R: 1 - PL[s]}), name="stochastic-expert", latent_fields=("s",))
sol = GammaSolver(env).solve(); Rt = sol.rates(); h2 = lambda p: -p * math.log2(p) - (1 - p) * math.log2(1 - p)
print("# Stochastic-expert check of Definition 1\n\n" + __doc__.split("Usage:")[0].strip() + "\n")
print(f"## Exact solver\n\nH(G|O) per step: {[round(float(x), 4) for x in Rt['H(G|O)']]}; H(Gamma|O): {[round(float(x), 4) for x in Rt['H(Gamma|O)']]}; H(H|O): {[round(float(x), 4) for x in Rt['H(H|O)']]}.  "
      f"h2(1/3) = {h2(1/3):.4f}, log2 3 = {math.log2(3):.4f}; transitive: {all(sol.transitive.values())}; (A4) failures: {[t for t, v in sol.a4_ok.items() if not v]}.\n")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run(beta, seed):
    torch.manual_seed(seed); rs = np.random.RandomState(seed); s = rs.randint(0, 3, args.n); a3 = np.where(rs.rand(args.n) < np.array(PL)[s], L, R)
    O = torch.zeros(args.n, T, 4); O[np.arange(args.n), 0, s] = 1.0; O[:, 1:, 3] = 1.0; A = torch.full((args.n, T), NOOP); A[:, 2] = torch.tensor(a3); O, A1 = O.to(dev), F.one_hot(A, 3).float().to(dev); A = A.to(dev)
    m = DIACRITIC(4, 3, 8, 16, 64, 1.0, True, 0.25, "diacritic", "none").to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for it in range(args.steps):
        b = torch.randint(0, args.n, (256,), device=dev); ks, eqs, rates, vqs, preds = m.rollout(O[b], A1[b])
        loss = (F.cross_entropy(preds.reshape(-1, 3), A[b].reshape(-1), reduction="none").view(len(b), T) + beta * min(1.0, (it + 1) / 1000) * rates + vqs).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 and it < args.steps // 2:
            with torch.no_grad():
                used = torch.zeros(8, dtype=torch.bool, device=dev); used[ks.unique()] = True
                if (~used).any(): src = eqs.detach().reshape(-1, 16); k = int((~used).sum()); m.E_raw.data[~used] = src[torch.randint(0, len(src), (k,), device=dev)] + 0.01 * torch.randn(k, 16, device=dev)
    m.eval()
    with torch.no_grad(): ks, _, _, _, preds = m.rollout(O, A1)
    c = ks[:, 1].cpu().numpy(); p = F.softmax(preds[:, 2], -1).cpu().numpy(); _, cnt = np.unique(c, return_counts=True); Hc = float(-(cnt / cnt.sum() * np.log2(cnt / cnt.sum())).sum())
    kl = []; codes = []
    for st in range(3):
        q = p[s == st].mean(0); q = np.array([q[L], q[R]]) / (q[L] + q[R]); pt = np.array([PL[st], 1 - PL[st]]); kl.append(float((pt * np.log2(pt / q)).sum())); codes.append(int(np.bincount(c[s == st]).argmax()))
    return Hc, max(kl), codes[0] == codes[1], codes[2] != codes[0]
print(f"## Learner, action law P(L|s) = {PL} (K=8 sole carrier, cross-entropy on sampled demonstrations, 6000 episodes)\n\n| beta | seeds | rate at the waiting step (bits; requirement 0.918) | max_s KL(true || learned) (bits) | states 0,1 share a code | state 2 separate |\n|---|---|---|---|---|---|")
for beta in [float(x) for x in args.betas.split(',')]:
    v = [run(beta, sd) for sd in range(args.seeds)]; Hs = np.array([x[0] for x in v])
    print(f"| {beta:g} | {len(v)} | {Hs.mean():.3f} [{Hs.min():.3f}–{Hs.max():.3f}] | {np.mean([x[1] for x in v]):.4f} (max {max(x[1] for x in v):.4f}) | {sum(x[2] for x in v)}/{len(v)} | {sum(x[3] for x in v)}/{len(v)} |", flush=True)
