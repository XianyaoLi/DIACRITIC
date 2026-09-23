"""Acceptance (b): leak probe.  Train classifiers on the *raw* observation windows of each phase to predict
beta_1, beta_2, theta and the mass bin.  Requirements (A' spec §4b):
    gap1 / gap2 windows  ->  beta_2 accuracy within CI of chance 1/R2  (nothing re-reveals beta_2 before use2)
    gap2 window          ->  beta_1 accuracy at chance                  (beta_1 leaves no trace after the grasp)
    gap1 / gap2 windows  ->  theta / mass bin at chance                 (kinematic attach hides the mass)
Positive controls: scan window -> (beta_1, beta_2) ~ 100 % (probe channel works); grasp window -> beta_1 ~ 100 %.

Usage:  python isaac/leak_probe.py isaac/data/tier4_gap6 [--seeds 5] [--epochs 300]
"""
import sys, os, glob, json, argparse
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--mass_bins", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(4)

eps = []
for f in sorted(glob.glob(os.path.join(args.data, "ep*_env*.npz"))):
    z = np.load(f, allow_pickle=True)
    eps.append(dict(obs=z["obs"].astype(np.float32), phase=list(z["phase"]), theta=int(z["theta"]), z=int(z["z"]),
                    b1=int(z["b1"]), b2=int(z["b2"]), mass=float(z["mass"])))
meta = json.load(open(os.path.join(args.data, "meta.json")))
R1, R2, M = meta["R1"], meta["R2"], meta["M"]
phases = eps[0]["phase"]
masses = np.array([e["mass"] for e in eps]); mbin = np.digitize(masses, np.quantile(masses, np.linspace(0, 1, args.mass_bins + 1)[1:-1]))
print(f"{len(eps)} episodes, obs dim {eps[0]['obs'].shape[1]}, T={len(phases)}")

def window(ph):
    idx = [i for i, p in enumerate(phases) if p == ph]
    X = np.stack([e["obs"][idx].reshape(-1) for e in eps]); return X, idx

def fit_eval(X, y, n_cls, seed):
    rng = np.random.default_rng(seed); n = len(X); perm = rng.permutation(n); ntr = int(0.75 * n)
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_cls))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad(); loss = nn.functional.cross_entropy(net(Xt[tr]), yt[tr]); loss.backward(); opt.step()
    with torch.no_grad():
        acc_tr = (net(Xt[tr]).argmax(-1) == yt[tr]).float().mean().item()
        acc_te = (net(Xt[te]).argmax(-1) == yt[te]).float().mean().item()
    return acc_tr, acc_te, len(te)

targets = {"beta1": (np.array([e["b1"] for e in eps]), R1), "beta2": (np.array([e["b2"] for e in eps]), R2),
           "theta": (np.array([e["theta"] for e in eps]), M), "mass_bin": (mbin, args.mass_bins), "z": (np.array([e["z"] for e in eps]), meta["N"])}
rows = []
print("\nphase    target    chance | test acc (mean±std over seeds) | train acc | n_test | verdict")
expect_chance = {("gap1", "beta2"), ("gap2", "beta2"), ("gap2", "beta1"), ("gap1", "theta"), ("gap2", "theta"), ("gap1", "mass_bin"), ("gap2", "mass_bin")}
expect_high = {("scan", "beta1"), ("scan", "beta2"), ("grasp", "beta1"), ("place", "beta2")}
all_ok = True
for ph in ["scan", "gap1", "grasp", "gap2", "place"]:
    X, idx = window(ph)
    for name, (y, ncls) in targets.items():
        if name == "z" and ph != "gap2": continue
        res = [fit_eval(X, y, ncls, s) for s in range(args.seeds)]
        te = np.array([r[1] for r in res]); tr = np.array([r[0] for r in res]); nte = res[0][2]
        chance = 1.0 / ncls; ci = 2 * np.sqrt(chance * (1 - chance) / nte)
        verdict = ""
        if (ph, name) in expect_chance:
            verdict = "at chance ✓" if te.mean() <= chance + ci else "LEAK ✗"; all_ok &= te.mean() <= chance + ci
        elif (ph, name) in expect_high:
            verdict = "revealed ✓ (positive control)" if te.mean() > 0.85 else "positive control FAILED ✗"; all_ok &= te.mean() > 0.85
        print(f"{ph:8s} {name:9s} {chance:6.3f} | {te.mean():.3f}±{te.std():.3f}                | {tr.mean():.3f}     | {nte:4d}   | {verdict}")
        rows.append(dict(phase=ph, target=name, chance=chance, test_acc=float(te.mean()), test_std=float(te.std()), train_acc=float(tr.mean()), n_test=nte, verdict=verdict))
print("\nVERDICT:", "no leak; positive controls pass" if all_ok else "FAIL (see rows)")
json.dump(rows, open(os.path.join(args.data, "leak_probe.json"), "w"), indent=1)
sys.exit(0 if all_ok else 1)
