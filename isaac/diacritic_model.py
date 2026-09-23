"""DIACRITIC model (continuous-observation version), shared by train_diacritic.py (teacher-forced training / exact metrics)
and aprime_env.py --policy (closed-loop evaluation inside Isaac).  Pure torch; importable from both the conda env and the
Isaac Sim python.

  e~_t = e_{t-1} + f([e_{t-1}, g_o(o_t), g_a(a_{t-1})]),  C_t = argmin_k ||e~_t - E_k||  (unit-sphere codes by default),
  STE backward, rate_t = -log r_eta(C_t | o_t) with soft-to-hard gradient, pi(a_t | o_t, C_t).  Only C_t crosses time.
Variants: diacritic | bypass (continuous GRU state crosses time, C_t = VQ read-out) | nonpersistent | uncond (prior not
conditioned on o).  Optional BFS decoder for the -RF objective.
"""
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class DIACRITIC(nn.Module):
    def __init__(self, do, da, K=16, d=32, hid=128, tau=1.0, l2=True, commit=0.25, variant="diacritic", bound="none", ds=0, na=0, K2=0, dual_detach=False):
        super().__init__()
        self.cfg = dict(do=do, da=da, K=K, d=d, hid=hid, tau=tau, l2=l2, commit=commit, variant=variant, bound=bound, ds=ds, na=na, K2=K2, dual_detach=dual_detach)
        self.K2, self.dual_detach = K2, dual_detach
        self.K, self.d, self.tau, self.l2, self.commit, self.variant, self.bound, self.ds, self.na = K, d, tau, l2, commit, variant, bound, ds, na
        self.detach_pi = False   # MIK baseline: policy head reads a detached (o_t, C_t) so imitation does not shape the memory
        self.alpha = 0.0     # 'scaffold' variant: weight of the continuous training scaffold (annealed to 0 by the trainer; 0 at test)
        self.pixels = do < 0                                   # do<0 signals pixel input: |do| = number of low-dim channels
        if self.pixels:
            self.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, hid), nn.ReLU())
            self.g_low = nn.Sequential(nn.Linear(-do, hid), nn.ReLU()); self.g_o_head = nn.Linear(2 * hid, d)
            self.g_o = lambda o: self.g_o_head(torch.cat([self.cnn(o[0]), self.g_low(o[1])], -1))    # o = (img (B,3,H,W) float, low (B,k))
        else:
            self.g_o = nn.Sequential(nn.Linear(do, hid), nn.ReLU(), nn.Linear(hid, d))
        self.g_a = nn.Sequential(nn.Linear(da, hid), nn.ReLU(), nn.Linear(hid, d))
        self.E_raw = nn.Parameter(torch.randn(K, d) * 0.5)
        self.f = nn.Sequential(nn.Linear(3 * d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, d))
        self.pi = nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da))
        # matched side information (control): ds > 0 -> the recurrent transition and the conditional prior receive ONLY the
        # symbolic observation s_t = f_t(o_t) (one-hot over the vocabulary); the policy head still reads the raw observation for control.
        if ds > 0: self.g_s = nn.Sequential(nn.Linear(ds, hid), nn.ReLU(), nn.Linear(hid, d))
        self.prior = nn.Sequential(nn.Linear(ds if ds > 0 else abs(do), hid), nn.ReLU(), nn.Linear(hid, K))
        # hierarchical realisation (na > 0, requires ds > 0): the behavioural head pi_high(sym_a | C_t, s_t) chooses the symbolic action
        # from the code and f_t(o_t) ONLY; a memory-free low-level controller pi_low(a | o_t, sym_a) executes it from the raw observation.
        if na > 0:
            self.pi_high = nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(), nn.Linear(hid, na))
            self.pi_low = nn.Sequential(nn.Linear(abs(do) + na, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da))
        # intent realisation (na == -1, requires ds > 0): pi_high(s_t, C_t) -> continuous action target trained by MSE (the memory sees only
        # f_t(o_t) through this head); a memory-free controller pi_low(o_t, stopgrad(intent)) refines it from the raw observation.
        if na == -1:
            self.pi_high = nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da))
            self.pi_low = nn.Sequential(nn.Linear(abs(do) + da, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da))
        self.prior_uncond = nn.Parameter(torch.zeros(K))
        self.gru = nn.GRUCell(2 * d, d); self.readout = nn.Linear(d, d)
        if variant == "transformer":            # full-history baseline: causal Transformer over (o_t, a_{t-1}) tokens; C_t = VQ read-out (memory bypasses the code)
            self.tf_in = nn.Linear(2 * d, hid); self.tf_pos = nn.Parameter(torch.zeros(128, hid))
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(hid, 4, 2 * hid, dropout=0.0, batch_first=True), num_layers=2)
            self.tf_out = nn.Linear(hid, d); self._hist = None
        # BFS decoder: GRU over the continuation window (o_{t+1..t+j}, a_{t..t+j-1}), initialised from (C_t, o_t); predicts a_{t+j}
        self.bfs_init = nn.Linear(2 * d, hid); self.bfs_gru = nn.GRU(2 * d, hid, batch_first=True)
        self.bfs_head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, da))
        # dual-carrier control: a SECOND recurrent hard-VQ code C2_t = Q2(F2(C2_{t-1}, o_t, a_{t-1})) with its own codebook (K2),
        # transition and conditional prior.  The policy head reads only (o_t, C_t); the identification (theta) head reads only (o_t, C2_t).
        # Both carriers pay the same rate penalty.  dual_detach: the second carrier's input embeddings are stop-gradient (no theta gradient
        # reaches the shared observation/action encoders).
        if K2 > 0:
            self.E2_raw = nn.Parameter(torch.randn(K2, d) * 0.5)
            self.f2 = nn.Sequential(nn.Linear(3 * d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, d))
            self.prior2 = nn.Sequential(nn.Linear(ds if ds > 0 else abs(do), hid), nn.ReLU(), nn.Linear(hid, K2))

    @property
    def E(self):
        return F.normalize(self.E_raw, dim=-1) if self.l2 else self.E_raw

    @property
    def E2(self):
        return F.normalize(self.E2_raw, dim=-1) if self.l2 else self.E2_raw

    def quantize(self, et, E=None):
        if E is None: E = self.E
        if self.l2: et = F.normalize(et, dim=-1)
        d2 = (et[:, None, :] - E[None]).pow(2).sum(-1); k = d2.argmin(-1); ek = E[k]
        eq = et + (ek - et).detach(); q = F.softmax(-d2 / self.tau, -1); oh = F.one_hot(k, E.shape[0]).float() + q - q.detach()
        vq = (et.detach() - ek).pow(2).sum(-1) + self.commit * (et - ek.detach()).pow(2).sum(-1)
        return k, eq, oh, vq

    def init_state(self, B, device):
        """(e_0, h_0, a_prev) — e_0 = code 0, no previous action."""
        self._hist = None
        if self.K2 > 0: self._e2 = self.E2[torch.zeros(B, dtype=torch.long, device=device)]
        return self.E[torch.zeros(B, dtype=torch.long, device=device)], torch.zeros(B, self.d, device=device), torch.zeros(B, self.cfg["da"], device=device)

    def step(self, o, e, h, a_prev, s=None, sa=None):
        """One recurrent step. Returns (k, eq, oh, vq, pred, h_new). o, a_prev are *normalised*; s = symbolic one-hot (ds > 0 only);
        sa = one-hot of the recorded symbolic action (hierarchical, teacher forcing); at closed loop sa=None -> argmax of pi_high."""
        eo, ea = self.g_o(o), self.g_a(a_prev); v = self.variant
        epi = eo                                   # the policy head always reads the raw observation
        self._last_eo = epi
        if self.ds > 0: eo = self.g_s(s)           # ... but the carrier update / proposal sees only f_t(o_t)
        if v == "transformer":                    # step-wise use: keep the token history and re-run the causal model (T <= 128)
            tok = torch.cat([eo, ea], -1)[:, None]; self._hist = tok if self._hist is None else torch.cat([self._hist, tok], 1)
            et = self.tf_forward(self._hist)[:, -1]
        elif v == "bypass": h = self.gru(torch.cat([eo, ea], -1), h); et = self.readout(h)
        elif v == "nonpersistent": et = self.f(torch.cat([torch.zeros_like(e), eo, ea], -1))
        elif v == "scaffold":                     # continuous scaffold: a GRU carrier added to the proposal with weight alpha (1 -> 0 during training)
            h = self.gru(torch.cat([eo, ea], -1), h); et = e + self.f(torch.cat([e, eo, ea], -1)) + self.alpha * self.readout(h)
        else:
            et = e + self.f(torch.cat([e, eo, ea], -1))
            if self.bound == "tanh": et = torch.tanh(et)
        self._last_et = (F.normalize(et, dim=-1) if self.l2 else et).detach()
        k, eq, oh, vq = self.quantize(et)
        if self.K2 > 0:                            # second carrier: same inputs (optionally stop-gradient), own transition/codebook
            eo2, ea2 = (eo.detach(), ea.detach()) if self.dual_detach else (eo, ea)
            et2 = self._e2 + self.f2(torch.cat([self._e2, eo2, ea2], -1))
            self._k2, self._eq2, self._oh2, self._vq2 = self.quantize(et2, self.E2); self._e2 = self._eq2
        if self.na == -1:
            self._last_intent = self.pi_high(torch.cat([eo, eq], -1))            # behavioural action target from (s_t, C_t) only
            return k, eq, oh, vq, self._last_intent + self.pi_low(torch.cat([o, self._last_intent.detach()], -1)), h
        if self.na > 0:
            self._last_logits = self.pi_high(torch.cat([eo, eq], -1))            # behavioural decision from (s_t, C_t) only
            if sa is None: sa = F.one_hot(self._last_logits.argmax(-1), self.na).float()
            return k, eq, oh, vq, self.pi_low(torch.cat([o, sa], -1)), h            # execution from the raw observation, no memory
        pin = torch.cat([epi, eq], -1); pin = pin.detach() if self.detach_pi else pin
        return k, eq, oh, vq, self.pi(pin), h

    def tf_forward(self, tokens):
        """tokens (B,T,2d) -> proposals (B,T,d), causal."""
        T = tokens.shape[1]; x = self.tf_in(tokens) + self.tf_pos[:T][None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        return self.tf_out(self.tf(x, mask=mask, is_causal=True))

    def rate(self, o, oh, s=None):
        if self.variant == "uncond": logp = F.log_softmax(self.prior_uncond, -1)[None].expand(oh.shape[0], -1)
        else: logp = F.log_softmax(self.prior(s if self.ds > 0 else (o[1] if self.pixels else o)), -1)
        return -(oh * logp).sum(-1) / math.log(2)

    def rollout(self, O, A, S=None, SA=None):
        """Teacher-forced pass over (B,T,do) / (B,T,da) normalised tensors; S = (B,T,ds) symbolic one-hots when ds > 0;
        SA = (B,T,na) recorded symbolic-action one-hots (hierarchical).  Per-step behavioural logits are stacked in self._logits."""
        if self.pixels: B, T = O[1].shape[:2]; dev = O[1].device; at = lambda t: (O[0][:, t], O[1][:, t])
        else: B, T, _ = O.shape; dev = O.device; at = lambda t: O[:, t]
        e, h, a_prev = self.init_state(B, dev)
        ks, eqs, rates, vqs, preds = [], [], [], [], []
        self._ets = []; self._eos = []; logits = []; ks2, eqs2, rates2, vqs2 = [], [], [], []
        if self.variant == "transformer" and not self.pixels:
            eo = self.g_o(O); ea = self.g_a(torch.cat([torch.zeros_like(A[:, :1]), A[:, :-1]], 1))
            ets = self.tf_forward(torch.cat([eo, ea], -1))                      # (B,T,d)
            for t in range(T):
                k, eq, oh, vq = self.quantize(ets[:, t]); self._ets.append((F.normalize(ets[:, t], dim=-1) if self.l2 else ets[:, t]).detach())
                rates.append(self.rate(O[:, t], oh)); preds.append(self.pi(torch.cat([eo[:, t], eq], -1))); ks.append(k); eqs.append(eq); vqs.append(vq)
            return torch.stack(ks, 1), torch.stack(eqs, 1), torch.stack(rates, 1), torch.stack(vqs, 1), torch.stack(preds, 1)
        for t in range(T):
            st = S[:, t] if S is not None else None
            k, eq, oh, vq, pred, h = self.step(at(t), e, h, a_prev, st, SA[:, t] if SA is not None else None)
            self._ets.append(self._last_et)
            self._eos.append(self._last_eo)
            if self.na > 0: logits.append(self._last_logits)
            if self.na == -1: logits.append(self._last_intent)
            rates.append(self.rate(at(t), oh, st)); preds.append(pred); ks.append(k); eqs.append(eq); vqs.append(vq)
            if self.K2 > 0:
                logp2 = F.log_softmax(self.prior2(st if self.ds > 0 else (at(t)[1] if self.pixels else at(t))), -1)
                rates2.append(-(self._oh2 * logp2).sum(-1) / math.log(2)); ks2.append(self._k2); eqs2.append(self._eq2); vqs2.append(self._vq2)
            e, a_prev = eq, A[:, t]
        self._logits = torch.stack(logits, 1) if logits else None
        if self.K2 > 0: self._ks2, self._eqs2, self._rates2, self._vqs2 = torch.stack(ks2, 1), torch.stack(eqs2, 1), torch.stack(rates2, 1), torch.stack(vqs2, 1)
        return torch.stack(ks, 1), torch.stack(eqs, 1), torch.stack(rates, 1), torch.stack(vqs, 1), torch.stack(preds, 1)

    def bfs_pred(self, eqs, O, A, t, J, noobs=False):
        """Behavioural-future decoder.  noobs=True: decision-centric control -- the decoder receives NO future observation and NO
        future action (open-loop prediction of a_{t+1..t+J} from (C_t, o_t) alone), so everything any future action depends on must
        be carried by the code even when a future observation would re-provide it."""
        T = O.shape[1]; J = min(J, T - 1 - t)
        if J <= 0: return None
        h0 = torch.tanh(self.bfs_init(torch.cat([self.g_o(O[:, t]), eqs[:, t]], -1)))[None]
        tok = torch.cat([self.g_o(O[:, t + 1:t + 1 + J]), self.g_a(A[:, t:t + J])], -1)
        if noobs: tok = torch.zeros_like(tok)
        out, _ = self.bfs_gru(tok, h0)
        return self.bfs_head(out)      # (B,J,da): position j-1 predicts a_{t+j}


def save_model(path, model, o_mu, o_sd, a_mu, a_sd, extra=None):
    torch.save(dict(cfg=model.cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    o_mu=o_mu.cpu(), o_sd=o_sd.cpu(), a_mu=a_mu.cpu(), a_sd=a_sd.cpu(), extra=extra or {}), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = DIACRITIC(**ck["cfg"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


class Policy:
    """Closed-loop wrapper: raw obs in, raw action out; keeps the discrete memory state across steps."""
    def __init__(self, path, device="cpu"):
        self.m, ck = load_model(path, device); self.dev = torch.device(device)
        self.o_mu, self.o_sd, self.a_mu, self.a_sd = (ck[k].to(self.dev) for k in ("o_mu", "o_sd", "a_mu", "a_sd"))
        self.extra = ck["extra"]; self.e = self.h = self.a_prev = None

    def reset(self, B):
        self.e, self.h, self.a_prev = self.m.init_state(B, self.dev); self.t = 0; self.n_unseen = 0

    def symbolic(self, obs_raw):
        """f_t(o_t) computed online from the raw observation with the frozen binner saved at training (extra['sym_binner']):
        phase from the one-hot, probe symbol from the probe channels, visible classes by nearest per-step class mean of the
        end-effector position, optional sag symbol from a per-step height threshold.  Returns (B, ds) one-hot."""
        sb = self.extra["sym_binner"]; t = self.t; o = obs_raw.cpu().numpy(); B = o.shape[0]
        vocab = sb["vocab"]; out = torch.zeros(B, len(vocab), device=self.dev)
        ph_names = sb["phase_names"]; means = sb["means"].get(str(t), {}); sag = sb["sag"].get(str(t))
        for i in range(B):
            phase = ph_names[int(np.argmax(o[i, sb["phase_slice"][0]:sb["phase_slice"][1]]))]
            probe = int(o[i, sb["probe_idx"][1]] > 0.5) if o[i, sb["probe_idx"][0]] > 0.5 else -1
            sym = [phase, probe]
            for which in ("b1", "b2"):
                if which in means:
                    mu = np.asarray(means[which]); sym.append(int(np.argmin(((mu - o[i, :3][None]) ** 2).sum(-1))))
                else: sym.append(-1)
            if sag is not None: sym.append(int((o[i, 2] < sag[0]) == bool(sag[1])))
            key = str(tuple(sym)); j = vocab.get(key)
            if j is None: self.n_unseen += 1; j = 0
            out[i, j] = 1.0
        return out

    LOW_IDX = [7, 17, 18] + list(range(19, 27))     # pixel policies: gripper, probe(2), phase one-hot(8) (same as train_diacritic.py)

    def preprocess_image(self, img_uint8):
        """(B,H,W,3) uint8 camera frame -> (B,3,r,r) float in [0,1] with the crop/resize used at training (from the saved args)."""
        a = self.extra.get("args", {}); im = img_uint8.to(self.dev)
        crop = a.get("px_crop", "40,112,16,112"); res = int(a.get("px_res", 64))
        if crop: y0, y1, x0, x1 = map(int, crop.split(",")); im = im[:, y0:y1, x0:x1]
        im = torch.nn.functional.interpolate(im.permute(0, 3, 1, 2).float(), size=(res, res), mode="area").round()
        return im / 255.0

    @torch.no_grad()
    def act(self, obs_raw, img_uint8=None):
        o = (obs_raw.to(self.dev) - self.o_mu) / self.o_sd
        if self.m.pixels:
            assert img_uint8 is not None, "pixel policy needs the camera frame"
            o = (self.preprocess_image(img_uint8), o[:, self.LOW_IDX])
        s = self.symbolic(obs_raw) if self.m.ds > 0 else None
        k, eq, oh, vq, pred, self.h = self.m.step(o, self.e, self.h, self.a_prev, s)
        self.e, self.a_prev = eq, pred; self.t += 1
        return pred * self.a_sd + self.a_mu, k
