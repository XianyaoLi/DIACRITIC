"""Instrument training on one config: code flip rate between consecutive steps, boundary margin, prior fit."""
import os, sys; os.environ["OMP_NUM_THREADS"]="1"
import torch, numpy as np, torch.nn.functional as F, math; torch.set_num_threads(1)
from toy_env import toy_gap
from toy_diacritic import ToyData, Cfg, DIACRITIC, evaluate
gap2=int(sys.argv[1]); lam=float(sys.argv[2]); seed=int(sys.argv[3]); steps=int(sys.argv[4]) if len(sys.argv)>4 else 4000
data=ToyData(toy_gap(gap2=gap2)); cfg=Cfg(lam=lam, seed=seed, steps=steps)
torch.manual_seed(seed); np.random.seed(seed)
O,A,W=data.O,data.A,data.W; T=O.shape[1]
model=DIACRITIC(data.n_o,data.n_a,cfg.K,cfg.d,cfg.hid,cfg.tau); opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
prev=None; ph=data.env.phases
for it in range(steps):
    ks,eqs,rates,vqs,logits=model.rollout(O,A)
    ce=F.cross_entropy(logits.reshape(-1,data.n_a),A.reshape(-1),reduction="none").view(-1,T)
    beta_t=cfg.beta*min(1.0,(it+1)/cfg.beta_warmup)
    bfs=torch.zeros_like(ce)
    if lam>0:
        for t in range(T-1):
            lg=model.bfs_logits(eqs,O,A,t,cfg.J)
            if lg is None: continue
            Jt=lg.shape[1]; tgt=A[:,t+1:t+1+Jt]
            bfs[:,t]=F.cross_entropy(lg.reshape(-1,data.n_a),tgt.reshape(-1),reduction="none").view(-1,Jt).mean(-1)
    loss=((ce+beta_t*rates+vqs+lam*bfs).mean(1)*W).sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    if cfg.reinit_every and it%cfg.reinit_every==0 and it<int(steps*cfg.reinit_frac):
        with torch.no_grad():
            used=torch.zeros(cfg.K,dtype=torch.bool); used[ks.unique()]=True
            if (~used).any():
                src=eqs.detach().reshape(-1,cfg.d); n=(~used).sum().item()
                model.E_raw.data[~used]=src[torch.randint(0,len(src),(n,))]+0.01*torch.randn(n,cfg.d)
    if it%200==0 or it==steps-1:
        with torch.no_grad():
            flip=float((ks!=prev).float().mean()) if prev is not None else float("nan"); prev=ks.clone()
            # boundary margin: (d2nd - d1st)/d1st at each (episode,t)
            e=eqs.reshape(-1,cfg.d); d2=(e[:,None,:]-model.E[None]).pow(2).sum(-1); s2,_=d2.sort(-1)
            margin=float(((s2[:,1]-s2[:,0])/(s2[:,0]+1e-8)).median()); dmin=float(s2[:,0].mean())
            r=evaluate(model,data)
            g1=ph["gap1"][0]-1
            print(f"it {it:5d} loss {loss.item():.3f} ce {(ce.mean(1)*W).sum():.3f} bfs {(bfs.mean(1)*W).sum():.3f} rate {(rates.mean(1)*W).sum():.2f} flip {flip:.3f} margin {margin:.2f} dmin {dmin:.3f} codes {ks.unique().numel():2d} | HC@gap1 {r['H(C|O)'][g1]:.2f} SΓ@gap1 {r['S_Gam'][g1]:.2f} D1 {r['D_TV'][ph['use1']-1]:.2f} D2 {r['D_TV'][ph['use2']-1]:.2f}", flush=True)
