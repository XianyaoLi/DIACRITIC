"""W6: deployment cost of the memory carriers as a function of history length -- per-step inference latency and carried state size for
(a) the causal Transformer full-history baseline (re-attends over all past tokens each step), (b) the GRU bypass (fixed continuous state),
(c) DIACRITIC (fixed discrete state, one code index + the code vector).  Same widths as the paper's models (d=32, hid=128, 2 layers, 4 heads).
Usage: python isaac/analysis/w6_cost.py  -> markdown table (median of 200 timed steps after 20 warm-up, batch 1 and 64, CPU and CUDA)."""
import sys, os, time, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diacritic_model import DIACRITIC
do, da = 27, 4
def timed(fn, n=200, warm=20, sync=False):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if sync: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))
rows = []
for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    torch.set_num_threads(4)
    for B in (1, 64):
        for T in (33, 64, 128, 256, 512, 1024):
            res = {}
            with torch.no_grad():
                for name, variant in (("Transformer (full history)", "transformer"), ("GRU bypass (continuous state)", "bypass"), ("DIACRITIC (K=16 code)", "diacritic")):
                    m = DIACRITIC(do, da, 16, 32, 128, 1.0, True, 0.25, variant, "none").to(dev).eval()
                    if variant == "transformer":
                        if T > 128: m.tf_pos = torch.nn.Parameter(torch.zeros(T, 128, device=dev))     # extend positions for the measurement
                        hist = torch.randn(B, T - 1, 64, device=dev)       # the T-1 past tokens already embedded; the new step re-attends over all of them
                        o = torch.randn(B, do, device=dev); a = torch.randn(B, da, device=dev)
                        def step():
                            tok = torch.cat([m.g_o(o), m.g_a(a)], -1)[:, None]; H = torch.cat([hist, tok], 1); et = m.tf_forward(H)[:, -1]; return m.quantize(et)[0]
                        state_bytes = (T - 1) * 64 * 4 * B                     # cached token history per episode (no KV cache in this implementation)
                    else:
                        e, h, ap = m.init_state(B, dev); o = torch.randn(B, do, device=dev); s = None
                        def step(): return m.step(o, e, h, ap)
                        state_bytes = (32 * 4) * B if variant == "bypass" else (32 * 4 + 1) * B      # continuous d=32 state | code vector + code index
                    res[name] = (timed(step, sync=(dev == "cuda")), state_bytes / B)
            rows.append((dev, B, T, res))
print("| device | batch | history length T | Transformer step (ms) | GRU step (ms) | DIACRITIC step (ms) | carried state per episode: Transformer / GRU / DIACRITIC (bytes) |")
print("|---|---|---|---|---|---|---|")
for dev, B, T, res in rows:
    tr, gr, di = res["Transformer (full history)"], res["GRU bypass (continuous state)"], res["DIACRITIC (K=16 code)"]
    print(f"| {dev} | {B} | {T} | {tr[0]:.2f} | {gr[0]:.2f} | {di[0]:.2f} | {tr[1]:.0f} / {gr[1]:.0f} / {di[1]:.0f} |")
json.dump([dict(device=d, batch=B, T=T, **{k: v for k, v in r.items()}) for d, B, T, r in rows], open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w6_cost.json"), "w"), indent=1)
