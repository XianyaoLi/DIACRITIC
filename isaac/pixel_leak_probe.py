"""Pixel leak probe: can a small CNN read the hidden class bits / theta / mass from SINGLE camera frames in a given phase?
Chance-level accuracy on held-out episodes = the images hide what the state observation hides; otherwise the pixel
benchmark leaks and no memory is needed.  Usage: python isaac/pixel_leak_probe.py isaac/data/tier4_gap6_px"""
import sys, glob, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
d = sys.argv[1]; res = 64; crop = (40, 112, 16, 112); dev = "cuda"
files = sorted(glob.glob(os.path.join(d, "ep*_env*.npz"))); eps = [np.load(f, allow_pickle=True) for f in files]
def frames(e):
    im = torch.tensor(e["img"][:, crop[0]:crop[1], crop[2]:crop[3]]).permute(0, 3, 1, 2).float()
    return F.interpolate(im, size=(res, res), mode="area").to(torch.uint8)
IM = torch.stack([frames(e) for e in eps]); PH = np.stack([e["phase"] for e in eps])           # (E,T,3,r,r), (E,T)
rng = np.random.RandomState(0); perm = rng.permutation(len(eps)); tr, te = perm[:448], perm[448:]
def probe(label, phase, name):
    y = torch.tensor(np.array([label(e) for e in eps])); K = int(y.max()) + 1
    idx = [(i, t) for i in range(len(eps)) for t in range(PH.shape[1]) if (PH[i, t] == phase if isinstance(phase, str) else t == phase)]
    def batch(ids, epset):
        sel = [(i, t) for i, t in ids if i in epset]; return IM[[i for i, t in sel], [t for i, t in sel]].to(dev).float() / 255, y[[i for i, t in sel]].to(dev)
    trs = set(tr.tolist()); tes = set(te.tolist()); Xtr, ytr = batch(idx, trs); Xte, yte = batch(idx, tes)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, K)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for it in range(1500):
        b = torch.randint(0, len(Xtr), (256,), device=dev); loss = F.cross_entropy(net(Xtr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): acc = (net(Xte).argmax(-1) == yte).float().mean().item(); tr_acc = (net(Xtr[:2000]).argmax(-1) == ytr[:2000]).float().mean().item()
    chance = max(np.bincount(yte.cpu().numpy(), minlength=K)) / len(yte)
    print(f"{name:26s} phase={str(phase):6s} K={K:2d} frames tr/te {len(Xtr)}/{len(Xte)}  train acc {tr_acc:.2f}  TEST acc {acc:.2f}  (majority {chance:.2f})", flush=True)
    return acc
out = {}
for phase in ["gap1", "gap2", "place"]:
    out[f"b1_{phase}"] = probe(lambda e: int(e["b1"]), phase, "b1 (grasp side class)")
    out[f"b2_{phase}"] = probe(lambda e: int(e["b2"]), phase, "b2 (place slot class)")
out["b1_t8"] = probe(lambda e: int(e["b1"]), 8, "b1 at FIRST b1 step t=8")
out["b2_t24"] = probe(lambda e: int(e["b2"]), 24, "b2 at FIRST b2 step t=24")
out["b2_t23"] = probe(lambda e: int(e["b2"]), 23, "b2 at t=23 (last gap2)")
out["theta_gap2"] = probe(lambda e: int(e["theta"]), "gap2", "theta (all hidden modes)")
out["mass_gap2"] = probe(lambda e: int(np.digitize(float(e["mass"]), np.quantile([float(x["mass"]) for x in eps], [0.25, 0.5, 0.75]))), "gap2", "mass quartile")
json.dump(out, open(os.path.join(d, "pixel_leak_probe.json"), "w"), indent=1)
