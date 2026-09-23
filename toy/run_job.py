"""Run one sweep job given as a JSON string; append result line to results/<name>.jsonl (atomic append)."""
import os, sys, json, time, fcntl
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["KMP_BLOCKTIME"] = "0"
import torch; torch.set_num_threads(1); torch.set_num_interop_threads(1)
from toy_env import toy_reveal, toy_gap
from toy_diacritic import ToyData, Cfg, train, evaluate

job = json.loads(sys.argv[1]); name = sys.argv[2]
env = toy_reveal(p_correct=job["p"], leak=job.get("leak", 0.0)) if job["env"] == "reveal" else toy_gap(gap1=job.get("gap1", 1), gap2=job["gap2"], p_correct=job["p"])
data = ToyData(env)
cfg = Cfg(beta=job["beta"], lam=job["lam"], steps=job["steps"], seed=job["seed"], n_train=job.get("n_train", 0), K=job.get("K", 16), J=job.get("J", 100),
          lr=job.get("lr", 1e-3), l2=job.get("l2", True), commit=job.get("commit", 0.25), beta_warmup=job.get("beta_warmup", 1000), variant=job.get("variant", "diacritic"),
          refine=job.get("refine", ""), refine_every=job.get("refine_every", 200), refine_start=job.get("refine_start", 1000), refine_thresh=job.get("refine_thresh", 1.5), scaffold_hold=job.get("scaffold_hold", 1500), scaffold_end=job.get("scaffold_end", 4000))
t0 = time.time(); model = train(data, cfg); res = evaluate(model, data)
out = dict(job=job, env=env.name, T=env.T, phases=getattr(env, "phases", None), res=res, secs=time.time() - t0)
with open(f"results/{name}.jsonl", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(out) + "\n"); f.flush(); fcntl.flock(f, fcntl.LOCK_UN)
m = res["mean"]
print(f"{env.name} lam={job['lam']} beta={job['beta']} seed={job['seed']} S_Gam={m['S_Gam']:.2f} D={m['D_TV']:.3f} H(C|O)={m['H(C|O)']:.2f} ({time.time()-t0:.0f}s)", flush=True)
