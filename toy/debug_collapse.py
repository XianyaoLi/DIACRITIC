import os; os.environ["OMP_NUM_THREADS"]="1"
import torch, sys, time, json; torch.set_num_threads(1)
from toy_env import toy_gap
from toy_diacritic import ToyData, Cfg, train, evaluate
gap2 = int(sys.argv[1]); lam = float(sys.argv[2]); tag = sys.argv[3]; only = sys.argv[4] if len(sys.argv) > 4 else None
data = ToyData(toy_gap(gap2=gap2))
configs = {
 "base":        dict(),
 "warmup":      dict(beta_warmup=1000),
 "bfs_mean":    dict(bfs_norm="mean"),
 "reinit_all":  dict(reinit_frac=1.0),
 "lr1e-3":      dict(lr=1e-3),
 "lam0.2":      dict(lam=0.2),
 "warm+mean":   dict(beta_warmup=1000, bfs_norm="mean"),
 "warm+mean+reinit": dict(beta_warmup=1000, bfs_norm="mean", reinit_frac=1.0),
}
ph = data.env.phases
for name, kw in configs.items():
    if only and name != only: continue
    if lam == 0 and name in ("bfs_mean", "lam0.2", "warm+mean", "warm+mean+reinit"): continue
    outs = []
    for seed in range(3):
        cfg = Cfg(beta=0.03, lam=lam, steps=2000, seed=seed, **kw)
        t0 = time.time(); m = train(data, cfg); r = evaluate(m, data)
        g1 = ph["gap1"][0]-1; g2 = ph["gap2"][0]-1
        outs.append((r["S_Gam"][g1], r["S_Gam"][g2], r["D_TV"][ph["use1"]-1], r["D_TV"][ph["use2"]-1], r["H(C|O)"][g1], r["H(C|O)"][g2], r["codes_used"]))
    print(f"[{tag} gap2={gap2} lam={lam}] {name:18s} " + " | ".join(f"SΓg1={a:.2f} SΓg2={b:.2f} D1={c:.2f} D2={d:.2f} H1={e:.2f} H2={f:.2f} K={k}" for a,b,c,d,e,f,k in outs), flush=True)
