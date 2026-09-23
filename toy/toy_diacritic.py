"""
Persistent discrete recurrent memory (DIACRITIC) on the enumerable toys, with exact evaluation.

Model (outline §3.2, frozen):   e_{t-1} = E[C_{t-1}],  Δ_t = f_ψ(e_{t-1}, enc(O_t), A_{t-1}),  ẽ_t = e_{t-1} + Δ_t,
                                C_t = VQ_K(ẽ_t)  (hard nearest code, forward deterministic; straight-through backward),
                                A_t ~ π_ω(· | O_t, C_t).   C_0 = code 0.  Nothing but C_t crosses time.
Objective (BCRB):  Σ_t [ CE(π_ω(A_t|O_t,C_t))                                  (j = 0, current imitation)
                         + λ Σ_{j≥1} CE(π_ω^{BFS}(A_{t+j} | C_t, O_t, O_{t+1:t+j}, A_{t:t+j-1}))   (BFS, C_t frozen)
                         + β (−log r_η(C_t | O_t)) ] + L_VQ.
Rate term: value uses the hard code; gradient to the encoder flows through the soft assignment
q(k|ẽ) = softmax(−‖ẽ−E_k‖²/τ) via a straight-through one-hot (soft-to-hard VQ).  The forward representation is
therefore deterministic from training to test, as the recurrent-realization definition requires.
Variants:  DIACRITIC-R  = λ=0 (BPTT through the recurrence only);  DIACRITIC-RF = λ>0 (+ BFS).
Evaluation: exact over the enumerated history tree; Gamma_t labels from gamma_solver (never hand-specified).
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from toy_env import Env, toy_reveal, toy_gap, enumerate_histories
from gamma_solver import GammaSolver, cond_entropy

torch.set_num_threads(int(__import__("os").environ.get("TORCH_THREADS", "4")))


# ------------------------------------------------------------------ data
class ToyData:
    """Episodes = terminal histories × final expert action, with exact weights; vocabularies; solver labels."""
    def __init__(self, env: Env):
        self.env, self.T = env, env.T
        self.solver = GammaSolver(env).solve()
        lv = self.solver.levels
        self.vo = {o: i for i, o in enumerate(sorted({nd.o for t in lv for nd in lv[t]}, key=repr))}
        acts = sorted({a for t in lv for nd in lv[t] for a in nd.act_dist}, key=repr)
        self.va = {a: i + 1 for i, a in enumerate(acts)}      # 0 = "no previous action"
        O, A, W, nodes = [], [], [], []
        for nd in lv[self.T]:
            for aT, p in nd.act_dist.items():
                h = nd.hist
                O.append([self.vo[o] for o in h[0::2]])
                A.append([self.va[a] for a in h[1::2]] + [self.va[aT]])
                W.append(nd.prob * p); nodes.append(nd)
        self.O = torch.tensor(O); self.A = torch.tensor(A); self.W = torch.tensor(W, dtype=torch.float)
        self.W /= self.W.sum()
        self.nodes = nodes
        # per-episode, per-t labels from the solver (prefix lookup)
        self.G, self.Gam, self.GamS, self.Ot, self.P_E = [], [], [], [], []
        for t in range(1, self.T + 1):
            idx = [self.solver.index[t][nd.hist[: 2 * t - 1]] for nd in nodes]
            self.G.append([self.solver.G[t][i] for i in idx])
            self.Gam.append([self.solver.Gamma[t][i] for i in idx] if self.solver.Gamma[t] is not None else None)
            self.GamS.append([self.solver.GammaS[t][i] for i in idx])
            self.Ot.append([self.solver.O[t][i] for i in idx])
            self.P_E.append([self.solver.levels[t][i].act_dist for i in idx])
        # nuisance latent expansion for I(C; z | O): (episode, latent) pairs with weights
        self.lat_rows = []
        for e, nd in enumerate(nodes):
            for lat, pl in nd.post.items():
                self.lat_rows.append((e, lat, pl))
        self.n_o, self.n_a = len(self.vo), len(self.va) + 1

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.W), size=n, p=self.W.numpy())
        w = torch.ones(n) / n
        return self.O[idx], self.A[idx], w


# ------------------------------------------------------------------ model
class DIACRITIC(nn.Module):
    def __init__(self, n_o, n_a, K=16, d=32, hid=64, tau=1.0):
        super().__init__()
        self.K, self.d, self.tau = K, d, tau
        self.commit = 0.25
        self.emb_o = nn.Embedding(n_o, d); self.emb_a = nn.Embedding(n_a, d)
        self.E_raw = nn.Parameter(torch.randn(K, d) * 0.5)
        self.l2 = True                # unit-sphere VQ: bounded distances, no residual scale blow-up
        self.f_trunk = nn.Sequential(nn.Linear(3 * d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.f_head = nn.Linear(hid, d)
        self.f = lambda x: self.f_head(self.f_trunk(x))
        self.pi = nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(), nn.Linear(hid, n_a))
        self.prior = nn.Embedding(n_o, K)                      # r_η(c | o)
        self.prior_uncond = nn.Parameter(torch.zeros(K))      # r_η(c)   (ablation 'uncond')
        self.gru = nn.GRUCell(2 * d, d); self.readout = nn.Linear(d, d)     # ablation 'bypass'
        self.logsig = nn.Linear(hid, d); self.prior_mu = nn.Embedding(n_o, d); self.prior_ls = nn.Embedding(n_o, d)  # 'continuous'
        self.variant = "diacritic"
        self.bound = "none"         # 'tanh': bounded residual state (raw VQ only) -- prevents the unbounded growth of the residual recurrence
        self.alpha = 0.0            # scaffold weight (annealed by train())
        # BFS decoder: GRU over continuation window tokens, initialised from (C_t, O_t); head shares nothing but is small
        self.bfs_init = nn.Linear(2 * d, hid)
        self.bfs_gru = nn.GRU(2 * d, hid, batch_first=True)
        self.bfs_head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n_a))
        self.bfs_head_o = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n_o))   # 'obsact' control: also predict O_{t+j}
        self.future = ""            # '' = BFS with the continuation as side information | 'act' = open-loop future actions from (C_t,O_t)
                                    # | 'obsact' = open-loop future observations AND actions (world-predictive control)
        # nearest-neighbour baselines: multi-step inverse head q(A_t | C_t, O_t, O_{t+j}) ('mik': the memory is trained only by it,
        # the policy head reads a detached (O_t, C_t)); system-identification head q(theta | C_t, O_t) ('sysid')
        self.mik_head = nn.Sequential(nn.Linear(3 * d + 16, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n_a)); self.mik_jemb = nn.Embedding(64, 16)
        self.sys_head = None; self.detach_pi = False

    @property
    def E(self):
        return F.normalize(self.E_raw, dim=-1) if self.l2 else self.E_raw

    def quantize(self, e_tilde):
        if self.l2:
            e_tilde = F.normalize(e_tilde, dim=-1)
        d2 = (e_tilde[:, None, :] - self.E[None]).pow(2).sum(-1)          # (B,K)
        k = d2.argmin(-1)
        e_k = self.E[k]
        e_q = e_tilde + (e_k - e_tilde).detach()                            # straight-through
        q = F.softmax(-d2 / self.tau, -1)
        onehot_st = F.one_hot(k, self.K).float() + q - q.detach()
        vq_loss = (e_tilde.detach() - e_k).pow(2).sum(-1) + self.commit * (e_tilde - e_k.detach()).pow(2).sum(-1)
        return k, e_q, onehot_st, vq_loss

    def rollout(self, O, A):
        self._ets = []          # pre-quantisation proposals per t (for future-sufficiency-triggered refinement)
        """Teacher-forced recurrent pass. O:(B,T) obs ids, A:(B,T) expert action ids (A[:,t-1] = a_t).
        Returns (ks, eqs, rates[bits], vqs, logits). For 'continuous', ks is None and eqs are the sampled/mean states."""
        B, T = O.shape; v = self.variant
        e = self.E[torch.zeros(B, dtype=torch.long)] if v != "continuous" else torch.zeros(B, self.d)   # C_0
        h = torch.zeros(B, self.d)                                            # bypass: continuous carrier
        a_prev = torch.zeros(B, dtype=torch.long)
        ks, eqs, rates, vqs, logits = [], [], [], [], []
        for t in range(T):
            eo, ea = self.emb_o(O[:, t]), self.emb_a(a_prev)
            if v == "bypass":
                h = self.gru(torch.cat([eo, ea], -1), h)
                e_tilde = self.readout(h)
            elif v == "nonpersistent":
                e_tilde = self.f(torch.cat([torch.zeros_like(e), eo, ea], -1))
            elif v == "continuous":
                tr = self.f_trunk(torch.cat([e, eo, ea], -1))
                mu = e + self.f_head(tr); ls = self.logsig(tr).clamp(-6, 2)
                z = mu + torch.exp(ls) * torch.randn_like(mu) if self.training else mu
                pm, pls = self.prior_mu(O[:, t]), self.prior_ls(O[:, t]).clamp(-6, 2)
                kl = (pls - ls + (torch.exp(2 * ls) + (mu - pm) ** 2) / (2 * torch.exp(2 * pls)) - 0.5).sum(-1)
                rates.append(kl / math.log(2)); vqs.append(torch.zeros(B)); ks.append(torch.zeros(B, dtype=torch.long))
                logits.append(self.pi(torch.cat([eo, z], -1))); eqs.append(z)
                e, a_prev = z, A[:, t]
                continue
            elif v == "scaffold":                                              # D2: continuous scaffold, annealed to 0 by train()
                h = self.gru(torch.cat([eo, ea], -1), h)
                e_tilde = e + self.f(torch.cat([e, eo, ea], -1)) + self.alpha * self.readout(h)
            else:
                e_tilde = e + self.f(torch.cat([e, eo, ea], -1))
            if self.bound == "tanh" and not self.l2: e_tilde = torch.tanh(e_tilde)
            self._ets.append(e_tilde.detach() if not self.l2 else F.normalize(e_tilde.detach(), dim=-1))
            k, e_q, oh, vq = self.quantize(e_tilde)
            logp = F.log_softmax(self.prior_uncond, -1)[None].expand(B, -1) if v == "uncond" else F.log_softmax(self.prior(O[:, t]), -1)
            rates.append(-(oh * logp).sum(-1) / math.log(2))
            pin = torch.cat([eo, e_q], -1); logits.append(self.pi(pin.detach() if self.detach_pi else pin))
            ks.append(k); eqs.append(e_q); vqs.append(vq)
            e, a_prev = e_q, A[:, t]                                          # only C_t crosses time (bypass: h does)
        return (torch.stack(ks, 1) if v != "continuous" else None), torch.stack(eqs, 1), torch.stack(rates, 1), torch.stack(vqs, 1), torch.stack(logits, 1)

    def mik_loss(self, eqs, O, A, J):
        """CE of the multi-step inverse head: predict A_t from (C_t, O_t, O_{t+j}), one random j in 1..J per (episode, t); last step masked."""
        B, T = O.shape; ar = torch.arange(T)[None].expand(B, -1)
        jt = torch.randint(1, J + 1, (B, T)); jt = torch.minimum(jt, (T - 1 - ar).clamp(min=1)); idx = (ar + jt).clamp(max=T - 1)
        feat = torch.cat([eqs, self.emb_o(O), self.emb_o(torch.gather(O, 1, idx)), self.mik_jemb(jt)], -1)
        ce = F.cross_entropy(self.mik_head(feat).reshape(-1, self.mik_head[-1].out_features), A.reshape(-1), reduction="none").view(B, T)
        return ce * (ar < T - 1).float()

    def bfs_logits(self, eqs, O, A, t, J, with_obs=False):
        """Predict A_{t+j}, j=1..J, from frozen C_t (=eqs[:,t]), O_t and the window (O_{t+1:t+j}, A_{t:t+j-1}).
        self.future in ('act', 'obsact'): the window is blanked (open-loop prediction from (C_t, O_t) alone); with_obs also returns O_{t+j} logits."""
        B, T = O.shape
        J = min(J, T - 1 - t)
        if J <= 0:
            return None
        h0 = torch.tanh(self.bfs_init(torch.cat([self.emb_o(O[:, t]), eqs[:, t]], -1)))[None]
        tok = torch.cat([self.emb_o(O[:, t + 1:t + 1 + J]), self.emb_a(A[:, t:t + J])], -1)    # (B,J,2d)
        if self.future in ("act", "obsact"): tok = torch.zeros_like(tok)
        out, _ = self.bfs_gru(tok, h0)
        if with_obs: return self.bfs_head(out), self.bfs_head_o(out)
        return self.bfs_head(out)                                            # (B,J,n_a): position j-1 predicts A_{t+j}


# ------------------------------------------------------------------ training
@dataclass
class Cfg:
    K: int = 16; d: int = 32; hid: int = 64; tau: float = 1.0
    beta: float = 0.03; lam: float = 1.0; J: int = 100; w_decay: float = 1.0
    steps: int = 3000; lr: float = 1e-3; seed: int = 0
    n_train: int = 0            # 0 = enumeration regime (all episodes, exact weights)
    bptt: bool = True           # False: detach recurrence (BFS-only credit assignment)
    reinit_every: int = 50
    reinit_frac: float = 0.5     # fraction of training during which dead codes are restarted
    beta_warmup: int = 1000      # linearly anneal beta from 0 over this many steps (prevents early codebook collapse)
    bfs_norm: str = "mean"       # 'mean' over j (default; 'sum' collapses the codebook on long horizons)
    commit: float = 0.25
    l2: bool = True              # unit-sphere codes and proposals (default); False = raw Euclidean VQ (unstable)
    refine: str = ""          # '' | 'bfs' | 'ce' : split the code whose members have the most inconsistent futures
    refine_every: int = 200; refine_start: int = 1000; refine_thresh: float = 1.5; refine_min: int = 8
    scaffold_hold: int = 1500; scaffold_end: int = 4000   # scaffold: alpha=1 until hold, linear to 0 at end
    bound: str = "none"          # 'tanh' = bounded residual state
    mik: float = 0.0; mik_J: int = 8      # >0: multi-step-inverse weight (variant 'mik' = inverse objective only, policy detached; 'diacritic'+mik>0 = joint)
    aux_theta: float = 0.0               # >0: system-identification head weight (variant 'sysid')
    future: str = ""                     # '' | 'act' | 'obsact'  (see DIACRITIC.bfs_logits)
    variant: str = "diacritic"   # diacritic | bypass (continuous GRU state crosses time, C_t = VQ readout) | uncond (prior r(c) not
                                 # conditioned on O) | nonpersistent (no e_{t-1}) | continuous (Gaussian persistent conditional bottleneck)


def train(data: ToyData, cfg: Cfg, log=False):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    O, A, W = (data.O, data.A, data.W) if cfg.n_train == 0 else data.sample(cfg.n_train, cfg.seed)
    if cfg.n_train != 0 and theta_id is not None:
        rng = np.random.default_rng(cfg.seed); theta_id = theta_id[torch.tensor(rng.choice(len(data.W), size=cfg.n_train, p=data.W.numpy()))]
    model = DIACRITIC(data.n_o, data.n_a, cfg.K, cfg.d, cfg.hid, cfg.tau); model.commit = cfg.commit; model.l2 = cfg.l2; model.variant = cfg.variant; model.bound = cfg.bound; model.future = cfg.future
    mik_w, aux_w = cfg.mik, cfg.aux_theta
    if cfg.variant == "mik": model.variant = "diacritic"; model.detach_pi = True; mik_w = mik_w or 1.0; cfg.mik = mik_w
    if cfg.variant == "sysid": model.variant = "diacritic"; aux_w = aux_w or 1.0; cfg.aux_theta = aux_w
    theta_id = None
    if aux_w > 0:
        ti = data.env.latent_fields.index("theta"); n_theta = len({lat[ti] for lat in data.env.prior})
        model.sys_head = nn.Sequential(nn.Linear(2 * cfg.d, cfg.hid), nn.ReLU(), nn.Linear(cfg.hid, n_theta))
        theta_id = torch.tensor([max(nd.post.items(), key=lambda kv: kv[1])[0][ti] for nd in data.nodes])   # theta is determined by the terminal history (A2)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    T = O.shape[1]
    for it in range(cfg.steps):
        if cfg.variant == "scaffold":
            model.alpha = 1.0 if it < cfg.scaffold_hold else max(0.0, 1.0 - (it - cfg.scaffold_hold) / max(1, cfg.scaffold_end - cfg.scaffold_hold))
        ks, eqs, rates, vqs, logits = model.rollout(O, A)
        if not cfg.bptt:
            raise NotImplementedError
        ce = F.cross_entropy(logits.reshape(-1, data.n_a), A.reshape(-1), reduction="none").view(-1, T)
        beta_t = cfg.beta * min(1.0, (it + 1) / cfg.beta_warmup) if cfg.beta_warmup > 0 else cfg.beta
        loss_t = ce + beta_t * rates + vqs
        if mik_w > 0: loss_t = loss_t + mik_w * model.mik_loss(eqs, O, A, cfg.mik_J)
        if aux_w > 0: loss_t = loss_t + aux_w * F.cross_entropy(model.sys_head(torch.cat([model.emb_o(O), eqs], -1)).reshape(-1, model.sys_head[-1].out_features), theta_id[:, None].expand(-1, T).reshape(-1), reduction="none").view(-1, T)
        bfs = torch.zeros_like(ce)
        if cfg.lam > 0:
            for t in range(T - 1):
                lg = model.bfs_logits(eqs, O, A, t, cfg.J, with_obs=(cfg.future == "obsact"))
                if lg is None:
                    continue
                lgo = None
                if cfg.future == "obsact": lg, lgo = lg
                Jt = lg.shape[1]
                tgt = A[:, t + 1:t + 1 + Jt]
                cej = F.cross_entropy(lg.reshape(-1, data.n_a), tgt.reshape(-1), reduction="none").view(-1, Jt)
                if lgo is not None: cej = cej + F.cross_entropy(lgo.reshape(-1, data.n_o), O[:, t + 1:t + 1 + Jt].reshape(-1), reduction="none").view(-1, Jt)
                wj = cfg.w_decay ** torch.arange(1, Jt + 1, dtype=torch.float)
                bfs[:, t] = (cej * wj).sum(-1) / (Jt if cfg.bfs_norm == "mean" else 1.0)
        loss = ((loss_t + cfg.lam * bfs).mean(1) * W).sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if cfg.refine and ks is not None and it >= cfg.refine_start and it % cfg.refine_every == 0:
            # neural partition refinement (the solver's backward refinement as a learning rule): a code whose members carry
            # systematically different future requirements shows a high within-code residual of the future-sufficiency
            # decoder (refine='bfs') or of the imitation loss (refine='ce'); split it along that residual.
            with torch.no_grad():
                res = (bfs if cfg.refine == "bfs" else ce).detach()                     # (B,T) per-episode residual
                if cfg.refine == "bfs" and cfg.lam <= 0: raise ValueError("refine='bfs' needs lam>0")
                gmean = float((res * W[:, None]).sum() / max(1e-9, W.sum() * res.shape[1]))
                best, bk, bt = 0.0, -1, -1
                for t in range(T):
                    for k in ks[:, t].unique().tolist():
                        m = ks[:, t] == k
                        if int(m.sum()) < cfg.refine_min: continue
                        r = float(res[m, t].mean())
                        if r > best: best, bk, bt = r, k, t
                used = torch.zeros(cfg.K, dtype=torch.bool); used[ks.unique()] = True
                free = (~used).nonzero().squeeze(-1)
                if bk >= 0 and best > cfg.refine_thresh * gmean and len(free) > 0:
                    m = (ks[:, bt] == bk).nonzero().squeeze(-1)
                    r = res[m, bt]; hi = m[r > r.median()]                                  # conflicting minority -> new code
                    if len(hi) >= 2:
                        et = torch.stack(model._ets, 1)[hi, bt]                                # their proposals at bt
                        model.E_raw.data[free[0]] = et.mean(0) + 0.01 * torch.randn(cfg.d)
                        if log: print(f"  it {it:5d} refine: split code {bk} at t={bt+1} (residual {best:.3f} vs mean {gmean:.3f}, {len(hi)}/{len(m)} members) -> code {int(free[0])}")
        if ks is not None and cfg.reinit_every and it % cfg.reinit_every == 0 and it < int(cfg.steps * cfg.reinit_frac):
            with torch.no_grad():       # dead-code restart: unused codes jump to random encoder outputs
                used = torch.zeros(cfg.K, dtype=torch.bool); used[ks.unique()] = True
                if (~used).any():
                    src = eqs.detach().reshape(-1, cfg.d)
                    model.E_raw.data[~used] = src[torch.randint(0, len(src), ((~used).sum().item(),))] + 0.01 * torch.randn((~used).sum().item(), cfg.d)
        if log and it % 500 == 0:
            print(f"  it {it:5d} loss {loss.item():.4f} ce {(ce.mean(1)*W).sum():.4f} rate {(rates.mean(1)*W).sum():.3f} "
                  f"bfs {(bfs.mean(1)*W).sum():.4f} codes {ks.unique().numel() if ks is not None else -1}")
    model.eval()
    return model


# ------------------------------------------------------------------ exact evaluation
def probe_entropy(z, o_ids, labels, w, n_o, steps=400, hid=32, seed=0):
    """Fixed-capacity recoverability probe: H_probe(label | z, o) in bits (weighted CE of a small MLP trained on all rows)."""
    torch.manual_seed(seed)
    lab = {l: i for i, l in enumerate(sorted(set(labels), key=repr))}; y = torch.tensor([lab[l] for l in labels])
    if len(lab) == 1: return 0.0
    x = torch.cat([z, F.one_hot(torch.tensor(o_ids), n_o).float()], -1); wt = torch.tensor(w, dtype=torch.float); wt = wt / wt.sum()
    net = nn.Sequential(nn.Linear(x.shape[1], hid), nn.ReLU(), nn.Linear(hid, len(lab))); opt = torch.optim.Adam(net.parameters(), 1e-2)
    with torch.enable_grad():
        for _ in range(steps):
            loss = (F.cross_entropy(net(x), y, reduction="none") * wt).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((F.cross_entropy(net(x), y, reduction="none") * wt).sum() / math.log(2))


@torch.no_grad()
def evaluate(model: DIACRITIC, data: ToyData) -> Dict:
    model.eval()
    ks, eqs, rates, vqs, logits = model.rollout(data.O, data.A)
    if ks is None:                                   # continuous bottleneck: probes instead of exact counting
        return evaluate_continuous(model, data, eqs, rates, logits)
    ks = ks.numpy(); probs = F.softmax(logits, -1).numpy(); W = data.W.numpy(); T = data.T
    inv_a = {i: a for a, i in data.va.items()}
    per_t = {k: [] for k in ["H(C|O)", "H(C|Gam,O)", "S_G", "S_Gam", "I(C;z|O)", "D_TV", "err", "H(G|O)", "H(Gam|O)", "H(H|O)", "rate_model"] + [f"I(C;{f}|O)" for f in data.env.latent_fields if f != "z"]}
    for t in range(T):
        c = list(ks[:, t]); o = data.Ot[t]; g = data.G[t]; gam = data.Gam[t]
        w = list(W)
        HG_O = cond_entropy(g, o, w); HC_O = cond_entropy(c, o, w)
        HG_CO = cond_entropy(g, list(zip(c, o)), w)
        per_t["H(C|O)"].append(HC_O); per_t["H(G|O)"].append(HG_O)
        per_t["H(H|O)"].append(cond_entropy([nd.hist[:2 * t + 1] for nd in data.nodes], o, w))
        per_t["S_G"].append((HG_O - HG_CO) / HG_O if HG_O > 1e-9 else float("nan"))
        if gam is not None:
            HGam_O = cond_entropy(gam, o, w); HGam_CO = cond_entropy(gam, list(zip(c, o)), w)
            per_t["H(Gam|O)"].append(HGam_O)
            per_t["S_Gam"].append((HGam_O - HGam_CO) / HGam_O if HGam_O > 1e-9 else float("nan"))
            per_t["H(C|Gam,O)"].append(cond_entropy(c, list(zip(gam, o)), w))
        else:
            per_t["H(Gam|O)"].append(float("nan")); per_t["S_Gam"].append(float("nan")); per_t["H(C|Gam,O)"].append(float("nan"))
        # nuisance MI: I(C; z | O) over (episode, latent) rows;  z = latent field 'z'
        zi = data.env.latent_fields.index("z") if "z" in data.env.latent_fields else None
        if zi is not None:
            rows = data.lat_rows
            cz = [c[e] for e, _, _ in rows]; z = [lat[zi] for _, lat, _ in rows]; oo = [o[e] for e, _, _ in rows]
            wz = [W[e] * pl for e, _, pl in rows]
            per_t["I(C;z|O)"].append(cond_entropy(z, oo, wz) - cond_entropy(z, list(zip(cz, oo)), wz))
        else:
            per_t["I(C;z|O)"].append(float("nan"))
        for f in data.env.latent_fields:          # same for every other latent field (theta: world-state information; w: agent-centric nuisance)
            if f == "z": continue
            fi = data.env.latent_fields.index(f); rows = data.lat_rows
            cz = [c[e] for e, _, _ in rows]; zz = [lat[fi] for _, lat, _ in rows]; oo = [o[e] for e, _, _ in rows]; wz = [W[e] * pl for e, _, pl in rows]
            per_t[f"I(C;{f}|O)"].append(cond_entropy(zz, oo, wz) - cond_entropy(zz, list(zip(cz, oo)), wz))
        # distortion: exact TV between expert P_E(.|h_t) and learner π(.|o_t,c_t); argmax error
        tv, err = 0.0, 0.0
        for e in range(len(W)):
            pe = data.P_E[t][e]; q = probs[e, t]
            tv += W[e] * 0.5 * sum(abs(pe.get(inv_a.get(i), 0.0) - q[i]) for i in range(len(q)))
            err += W[e] * float(pe.get(inv_a.get(int(q.argmax())), 0.0) <= 0.0)   # argmax outside expert support
        per_t["D_TV"].append(tv); per_t["err"].append(err)
        per_t["rate_model"].append(float((rates[:, t].numpy() * W).sum()))
    out = {k: [float(x) for x in v] for k, v in per_t.items()}
    out["codes_used"] = int(len(np.unique(ks)))
    out["mean"] = {k: float(np.nanmean(v)) for k, v in out.items() if isinstance(v, list)}
    return out


@torch.no_grad()
def evaluate_continuous(model, data, eqs, rates, logits):
    probs = F.softmax(logits, -1).numpy(); W = data.W.numpy(); T = data.T; inv_a = {i: a for a, i in data.va.items()}
    keys = ["H(C|O)", "H(C|Gam,O)", "S_G", "S_Gam", "I(C;z|O)", "D_TV", "err", "H(G|O)", "H(Gam|O)", "H(H|O)", "rate_model"]
    per_t = {k: [] for k in keys}
    o_ids = data.O.numpy()
    for t in range(T):
        o = data.Ot[t]; g = data.G[t]; gam = data.Gam[t]; w = list(W); z = eqs[:, t]
        HG_O = cond_entropy(g, o, w); per_t["H(G|O)"].append(HG_O)
        per_t["H(H|O)"].append(cond_entropy([nd.hist[:2 * t + 1] for nd in data.nodes], o, w))
        kl = float((rates[:, t].numpy() * W).sum()); per_t["rate_model"].append(kl); per_t["H(C|O)"].append(kl)   # KL upper bound (bits)
        Hp_G = probe_entropy(z, o_ids[:, t], g, w, data.n_o)
        per_t["S_G"].append((HG_O - Hp_G) / HG_O if HG_O > 1e-9 else float("nan"))
        if gam is not None:
            HGam_O = cond_entropy(gam, o, w); per_t["H(Gam|O)"].append(HGam_O)
            Hp = probe_entropy(z, o_ids[:, t], gam, w, data.n_o)
            per_t["S_Gam"].append(max(0.0, (HGam_O - Hp) / HGam_O) if HGam_O > 1e-9 else float("nan"))
        else:
            per_t["H(Gam|O)"].append(float("nan")); per_t["S_Gam"].append(float("nan"))
        per_t["H(C|Gam,O)"].append(float("nan"))
        zi = data.env.latent_fields.index("z") if "z" in data.env.latent_fields else None
        if zi is not None:      # nuisance recoverability probe: I_probe(Z; z | O) = H(z|O) - H_probe(z | Z, O)
            rows = data.lat_rows; zz = torch.stack([z[e] for e, _, _ in rows]); lat = [l[zi] for _, l, _ in rows]
            oo = [o_ids[e, t] for e, _, _ in rows]; wz = [W[e] * pl for e, _, pl in rows]
            per_t["I(C;z|O)"].append(cond_entropy(lat, [o[e] for e, _, _ in rows], wz) - probe_entropy(zz, oo, lat, wz, data.n_o))
        else:
            per_t["I(C;z|O)"].append(float("nan"))
        tv, err = 0.0, 0.0
        for e in range(len(W)):
            pe = data.P_E[t][e]; q = probs[e, t]
            tv += W[e] * 0.5 * sum(abs(pe.get(inv_a.get(i), 0.0) - q[i]) for i in range(len(q)))
            err += W[e] * float(pe.get(inv_a.get(int(q.argmax())), 0.0) <= 0.0)
        per_t["D_TV"].append(tv); per_t["err"].append(err)
    out = {k: [float(x) for x in v] for k, v in per_t.items()}
    out["codes_used"] = -1
    out["mean"] = {k: float(np.nanmean(v)) for k, v in out.items() if isinstance(v, list)}
    return out


def fmt_row(name, vals):
    return f"{name:>12s} " + " ".join(f"{v:5.2f}" if not (isinstance(v, float) and math.isnan(v)) else "   - " for v in vals)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reveal"); ap.add_argument("--gap2", type=int, default=3); ap.add_argument("--gap1", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.03); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--p_correct", type=float, default=1.0); ap.add_argument("--J", type=int, default=100)
    ap.add_argument("--out", default=""); ap.add_argument("--variant", default="diacritic"); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--l2", type=int, default=0)
    ap.add_argument("--M", type=int, default=16); ap.add_argument("--R1", type=int, default=2); ap.add_argument("--R2", type=int, default=2); ap.add_argument("--W", type=int, default=1); ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--mik", type=float, default=0.0); ap.add_argument("--aux_theta", type=float, default=0.0); ap.add_argument("--beta_warmup", type=int, default=1000); ap.add_argument("--bound", default="none")
    ap.add_argument("--future", default="", help="'' (BFS with side information) | act (open-loop future actions) | obsact (open-loop future observations + actions)")
    a = ap.parse_args()
    if a.env == "grid":
        from grid_env import grid_corridor
        env = grid_corridor(M=a.M, R1=a.R1, R2=a.R2, W=a.W, N=a.N, gap1=a.gap1, gap2=a.gap2)
    elif a.env == "tmaze":
        from tmaze_env import tmaze
        env = tmaze(R=a.R1, N=a.N, L=a.gap1)
    elif a.env == "rereveal":          # gap toy whose beta_2 is re-revealed one step before use2: H(Gamma|O) drops to 0 in gap2 while Gamma^s keeps 1 bit
        from test_t0 import toy_gap_rereveal
        env = toy_gap_rereveal(M=a.M, R1=a.R1, R2=a.R2, N=a.N, gap1=a.gap1, gap2=a.gap2, p_correct=a.p_correct)
    else:
        env = toy_reveal(p_correct=a.p_correct) if a.env == "reveal" else toy_gap(gap1=a.gap1, gap2=a.gap2, p_correct=a.p_correct)
    data = ToyData(env)
    cfg = Cfg(K=a.K, beta=a.beta, lam=a.lam, steps=a.steps, seed=a.seed, n_train=a.n_train, J=a.J, variant=a.variant, lr=a.lr, l2=bool(a.l2), mik=a.mik, aux_theta=a.aux_theta, beta_warmup=a.beta_warmup, bound=a.bound, future=a.future)
    t0 = time.time()
    model = train(data, cfg, log=True)
    res = evaluate(model, data)
    print(f"{env.name}  variant={'RF' if a.lam>0 else 'R'} beta={a.beta} seed={a.seed}  ({time.time()-t0:.0f}s)  codes_used={res['codes_used']}")
    print(fmt_row("t", list(range(1, env.T + 1))))
    for k in ["H(G|O)", "H(Gam|O)", "H(C|O)", "H(C|Gam,O)", "S_G", "S_Gam", "I(C;z|O)"] + [f"I(C;{f}|O)" for f in env.latent_fields if f != "z"] + ["D_TV", "err"]:
        print(fmt_row(k, res[k]))
    if a.env in ("grid", "tmaze"): print(fmt_row("H(GamS|O)", data.solver.rates()["H(GammaS|O)"]))
    if a.out:
        rec = dict(env=env.name, args=vars(a), cfg=cfg.__dict__, res=res, theory=data.solver.rates(), phases=getattr(env, "phases", None),
                   a4=[bool(x) for x in data.solver.a4_ok.values()], transitive=[bool(x) for x in data.solver.transitive.values()], secs=time.time() - t0,
                   host=__import__("os").uname().nodename, slurm_job=__import__("os").environ.get("SLURM_JOB_ID"))
        if ".jsonl" in a.out:            # .jsonl or a shard .jsonl.<task>.<i>: one line per run
            __import__("os").makedirs(__import__("os").path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        else: json.dump(rec, open(a.out, "w"))
