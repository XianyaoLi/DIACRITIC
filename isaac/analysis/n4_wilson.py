"""N4: Wilson 95% score intervals for every x/n sufficiency count the paper compares on A', plus two-sided Fisher
exact p-values for the contrasts the paper draws.  Sufficient := summary.S_Gam_gap2 > 0.9 and summary.S_G_place > 0.9.

Cells (counts are recomputed from the jsonl files by filtering on args):
 (a) fig3.jsonl      tier4_gap6_2k, diacritic, lam in {0,1}, beta in {0, 3e-4, 1e-3, 2e-3, 3e-3}
 (b) gap_refine.jsonl (refine == '' rows; there is no gap.jsonl in results/) + scaffold.jsonl:
                     tier4_gap6/10/20 for (diacritic lam0 b2e-3), (diacritic lam1 b2e-3), (scaffold lam1 b2e-3)
 (c) gap_refine.jsonl refine '' vs 'mse' (lam 0) and '' vs 'bfs' (lam 1) at gap 6/10/20
 (d) curric.jsonl     scaffold lam1 b2e-3 warm-started from the gap-6 scaffold; control = scaffold.jsonl / pI_scaffold.jsonl
 (e) datasize.jsonl   n_train 448/1152/3584, diacritic (b1e-3, lam0) vs scaffold (b2e-3, lam1)

Usage: python3 isaac/analysis/n4_wilson.py   (prints markdown)
"""
import json, os, math
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "cluster_results", "results")
THR = 0.9


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (two-sided 95% by default)."""
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(fn):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        print(f"MISSING file: {p}"); return []
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows: r["_file"] = fn
    return rows


def suff(r):
    s = r["summary"]; return s["S_Gam_gap2"] > THR and s["S_G_place"] > THR


def select(rows, data=None, variant=None, lam=None, beta=None, refine=None, n_train=None, K=None, aux_join=0.0):
    out = []
    for r in rows:
        a = r["args"]
        if data is not None and os.path.basename(a["data"]) != data: continue
        if variant is not None and a.get("variant") != variant: continue
        if lam is not None and float(a.get("lam", 0.0)) != lam: continue
        if beta is not None and float(a["beta"]) != beta: continue
        if refine is not None and a.get("refine", "") != refine: continue
        if n_train is not None and int(a["n_train"]) != n_train: continue
        if K is not None and int(a["K"]) != K: continue
        if aux_join is not None and float(a.get("aux_join", 0.0)) != aux_join: continue
        out.append(r)
    # one row per seed (keep the last written one) and report duplicates
    by_seed = {}
    for r in out: by_seed[r["args"]["seed"]] = r
    dup = len(out) - len(by_seed)
    return list(by_seed.values()), dup


cells = {}   # name -> (k, n, dup)

def cell(name, rows, **kw):
    sel, dup = select(rows, **kw)
    k = sum(suff(r) for r in sel); n = len(sel)
    cells[name] = (k, n, dup)
    return k, n


def fmt(name):
    k, n, dup = cells[name]
    if n == 0: return f"| {name} | -- | -- | -- | missing |"
    lo, hi = wilson(k, n)
    return f"| {name} | {k}/{n} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] |{' dup rows collapsed: %d' % dup if dup else ''} |"


def fisher(a, b):
    ka, na, _ = cells[a]; kb, nb, _ = cells[b]
    if na == 0 or nb == 0: return None
    return fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")[1]


fig3 = load("fig3.jsonl"); gr = load("gap_refine.jsonl"); sc = load("scaffold.jsonl")
cu = load("curric.jsonl"); ds = load("datasize.jsonl"); pisc = load("pI_scaffold.jsonl")

# (a)
BETAS = [0.0, 0.0003, 0.001, 0.002, 0.003]
for lam in (0.0, 1.0):
    for b in BETAS:
        cell(f"(a) fig3 tier4_gap6_2k diacritic lam={int(lam)} beta={b:g}", fig3, data="tier4_gap6_2k", variant="diacritic", lam=lam, beta=b, K=16)
# (b)
for g in (6, 10, 20):
    d = f"tier4_gap{g}"
    cell(f"(b) {d} diacritic lam=0 beta=0.002 (gap_refine, no refine)", gr, data=d, variant="diacritic", lam=0.0, beta=0.002, refine="")
    cell(f"(b) {d} diacritic lam=1 beta=0.002 (gap_refine, no refine)", gr, data=d, variant="diacritic", lam=1.0, beta=0.002, refine="")
    cell(f"(b) {d} scaffold lam=1 beta=0.002 (scaffold)", sc, data=d, variant="scaffold", lam=1.0, beta=0.002, refine="")
    cell(f"(b) {d} scaffold lam=0 beta=0.002 (scaffold)", sc, data=d, variant="scaffold", lam=0.0, beta=0.002, refine="")
# (c)
for g in (6, 10, 20):
    d = f"tier4_gap{g}"
    cell(f"(c) {d} lam=0 refine=mse", gr, data=d, variant="diacritic", lam=0.0, beta=0.002, refine="mse")
    cell(f"(c) {d} lam=1 refine=bfs", gr, data=d, variant="diacritic", lam=1.0, beta=0.002, refine="bfs")
# (d)
for d in ("tier4_gap10", "tier4_gap20", "pI_M16", "pI_M32"):
    cell(f"(d) curriculum {d} scaffold lam=1 beta=0.002 (curric)", cu, data=d, variant="scaffold", lam=1.0, beta=0.002)
for d in ("pI_M16", "pI_M32"):
    cell(f"(d) control {d} scaffold lam=1 beta=0.002 (pI_scaffold)", pisc, data=d, variant="scaffold", lam=1.0, beta=0.002)
# (e)
for n, d in ((448, "tier4_gap6"), (1152, "tier4_gap6_2k"), (3584, "tier4_gap6_4k")):
    cell(f"(e) datasize n_train={n} diacritic beta=0.001 lam=0", ds, data=d, variant="diacritic", lam=0.0, beta=0.001, n_train=n)
    cell(f"(e) datasize n_train={n} scaffold beta=0.002 lam=1", ds, data=d, variant="scaffold", lam=1.0, beta=0.002, n_train=n)

print("| cell | sufficient | p-hat | Wilson 95% | note |")
print("|---|---|---|---|---|")
for name in cells: print(fmt(name))

# contrasts
contrasts = []
for b in BETAS:
    contrasts.append((f"(a) beta={b:g}: lam=1 vs lam=0", f"(a) fig3 tier4_gap6_2k diacritic lam=1 beta={b:g}", f"(a) fig3 tier4_gap6_2k diacritic lam=0 beta={b:g}"))
for lam in (0, 1):
    for b in BETAS[1:]:
        contrasts.append((f"(a) lam={lam}: beta={b:g} vs beta=0", f"(a) fig3 tier4_gap6_2k diacritic lam={lam} beta={b:g}", f"(a) fig3 tier4_gap6_2k diacritic lam={lam} beta=0"))
for g in (6, 10, 20):
    d = f"tier4_gap{g}"
    contrasts.append((f"(b) gap {g}: scaffold lam=1 vs diacritic lam=1", f"(b) {d} scaffold lam=1 beta=0.002 (scaffold)", f"(b) {d} diacritic lam=1 beta=0.002 (gap_refine, no refine)"))
    contrasts.append((f"(b) gap {g}: scaffold lam=1 vs diacritic lam=0", f"(b) {d} scaffold lam=1 beta=0.002 (scaffold)", f"(b) {d} diacritic lam=0 beta=0.002 (gap_refine, no refine)"))
    contrasts.append((f"(b) gap {g}: diacritic lam=1 vs lam=0", f"(b) {d} diacritic lam=1 beta=0.002 (gap_refine, no refine)", f"(b) {d} diacritic lam=0 beta=0.002 (gap_refine, no refine)"))
for who, key in (("diacritic lam=0", "diacritic lam=0 beta=0.002 (gap_refine, no refine)"), ("diacritic lam=1", "diacritic lam=1 beta=0.002 (gap_refine, no refine)"), ("scaffold lam=1", "scaffold lam=1 beta=0.002 (scaffold)")):
    contrasts.append((f"(b) {who}: gap 6 vs gap 20", f"(b) tier4_gap6 {key}", f"(b) tier4_gap20 {key}"))
    contrasts.append((f"(b) {who}: gap 6 vs gap 10", f"(b) tier4_gap6 {key}", f"(b) tier4_gap10 {key}"))
for g in (6, 10, 20):
    d = f"tier4_gap{g}"
    contrasts.append((f"(c) gap {g}: refine=mse vs none (lam=0)", f"(c) {d} lam=0 refine=mse", f"(b) {d} diacritic lam=0 beta=0.002 (gap_refine, no refine)"))
    contrasts.append((f"(c) gap {g}: refine=bfs vs none (lam=1)", f"(c) {d} lam=1 refine=bfs", f"(b) {d} diacritic lam=1 beta=0.002 (gap_refine, no refine)"))
contrasts.append(("(d) gap 10: curriculum vs scaffold from scratch", "(d) curriculum tier4_gap10 scaffold lam=1 beta=0.002 (curric)", "(b) tier4_gap10 scaffold lam=1 beta=0.002 (scaffold)"))
contrasts.append(("(d) gap 20: curriculum vs scaffold from scratch", "(d) curriculum tier4_gap20 scaffold lam=1 beta=0.002 (curric)", "(b) tier4_gap20 scaffold lam=1 beta=0.002 (scaffold)"))
contrasts.append(("(d) pI_M16: curriculum vs scaffold from scratch", "(d) curriculum pI_M16 scaffold lam=1 beta=0.002 (curric)", "(d) control pI_M16 scaffold lam=1 beta=0.002 (pI_scaffold)"))
contrasts.append(("(d) pI_M32: curriculum vs scaffold from scratch", "(d) curriculum pI_M32 scaffold lam=1 beta=0.002 (curric)", "(d) control pI_M32 scaffold lam=1 beta=0.002 (pI_scaffold)"))
for n in (448, 1152, 3584):
    contrasts.append((f"(e) n_train={n}: scaffold vs diacritic", f"(e) datasize n_train={n} scaffold beta=0.002 lam=1", f"(e) datasize n_train={n} diacritic beta=0.001 lam=0"))
for who, key in (("diacritic", "diacritic beta=0.001 lam=0"), ("scaffold", "scaffold beta=0.002 lam=1")):
    contrasts.append((f"(e) {who}: n_train 3584 vs 448", f"(e) datasize n_train=3584 {key}", f"(e) datasize n_train=448 {key}"))
    contrasts.append((f"(e) {who}: n_train 1152 vs 448", f"(e) datasize n_train=1152 {key}", f"(e) datasize n_train=448 {key}"))

print("\n| contrast | A | B | Fisher two-sided p | p<0.05 |")
print("|---|---|---|---|---|")
for name, a, b in contrasts:
    p = fisher(a, b)
    ka, na, _ = cells[a]; kb, nb, _ = cells[b]
    if p is None: print(f"| {name} | {ka}/{na} | {kb}/{nb} | missing cell | -- |"); continue
    print(f"| {name} | {ka}/{na} | {kb}/{nb} | {p:.3f} | {'YES' if p < 0.05 else 'no'} |")
