"""N3: "written but not read" on the 4-bit join (tier16_gap6).

For every tier16_gap6 run in oracleK.jsonl / oracle.jsonl / tier.jsonl, report per seed the code-side sufficiency
S_Gam averaged over gap1 and over gap2 steps (recomputed from per_t + phases; identical to summary[S_Gam_gap*]),
S_Gam at the LAST gap2 step (the step right before the place action reads the join), S_G averaged over the place
steps, H(C|O) in gap1 / gap2, and the sufficiency flag (S_Gam_gap2 > 0.9 and S_G_place > 0.9).
Then per (K, head) group: #stored-but-not-read = S_Gam_gap2 > 0.9 and S_G_place <= 0.9; #not-stored = S_Gam_gap2 <= 0.9.

Usage: python3 isaac/analysis/n3_written_not_read.py   (prints markdown)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "cluster_results", "results")
FILES = ["oracleK.jsonl", "oracle.jsonl", "tier.jsonl"]
THR = 0.9


def load(fn):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        print(f"MISSING file: {p}"); return []
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows: r["_file"] = fn
    return rows


def phase_mean(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]
    return float(np.nanmean([r["per_t"][key][i] for i in idx]))


def phase_last(r, key, ph):
    idx = [i for i, p in enumerate(r["phases"]) if p == ph]
    return float(r["per_t"][key][idx[-1]])


rows = [r for f in FILES for r in load(f) if os.path.basename(r["args"]["data"]) == "tier16_gap6"]
if not rows: sys.exit("no tier16_gap6 rows")

# group key: (K, head?, variant, file)
def gkey(r):
    a = r["args"]; return (int(a["K"]), float(a.get("aux_join", 0.0)) > 0, a["variant"], r["_file"])

groups = {}
for r in rows: groups.setdefault(gkey(r), []).append(r)

theory = rows[0]["theory"]; ph = rows[0]["phases"]
Hgam1 = float(np.mean([theory["H(Gamma|O)"][i] for i, p in enumerate(ph) if p == "gap1"]))
Hgam2 = float(np.mean([theory["H(Gamma|O)"][i] for i, p in enumerate(ph) if p == "gap2"]))
print(f"Solver targets on tier16_gap6: H(Gamma|O) gap1 = {Hgam1:.3f} bits, gap2 = {Hgam2:.3f} bits (from theory[] of the first row).\n")

print("| file | K | head | variant | beta | lam | seed | S_Gam gap1 | S_Gam gap2 | S_Gam last gap2 | S_G place | H(C|O) gap1 | H(C|O) gap2 | class_err | sufficient | status |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
summ = []
for k in sorted(groups, key=lambda k: (k[1], k[0], k[2], k[3])):
    K, head, variant, fn = k
    n_suf = n_snr = n_ns = 0
    for r in sorted(groups[k], key=lambda r: r["args"]["seed"]):
        a, s = r["args"], r["summary"]
        sg1, sg2 = phase_mean(r, "S_Gam", "gap1"), phase_mean(r, "S_Gam", "gap2")
        sgl = phase_last(r, "S_Gam", "gap2")
        sgp = phase_mean(r, "S_G", "place")
        hc1, hc2 = phase_mean(r, "H(C|O)", "gap1"), phase_mean(r, "H(C|O)", "gap2")
        # cross-check with stored summary
        for mine, key in ((sg1, "S_Gam_gap1"), (sg2, "S_Gam_gap2"), (sgp, "S_G_place"), (hc1, "HC_gap1"), (hc2, "HC_gap2")):
            assert abs(mine - s[key]) < 1e-6, (key, mine, s[key])
        suf = s["S_Gam_gap2"] > THR and s["S_G_place"] > THR
        if suf: status = "sufficient"; n_suf += 1
        elif s["S_Gam_gap2"] > THR: status = "stored, not read"; n_snr += 1
        else: status = "not stored"; n_ns += 1
        print(f"| {fn} | {K} | {'yes' if head else 'no'} | {variant} | {a['beta']} | {a['lam']} | {a['seed']} | {sg1:.3f} | {sg2:.3f} | {sgl:.3f} | {sgp:.3f} | {hc1:.2f} | {hc2:.2f} | {s['class_err_test']:.3f} | {'yes' if suf else 'no'} | {status} |")
    summ.append((fn, K, head, variant, len(groups[k]), n_suf, n_snr, n_ns))

print("\nPer-group split (threshold 0.9 on the gap2-mean S_Gam and the place-mean S_G):\n")
print("| file | K | head | variant | n | sufficient | stored but not read (S_Gam_gap2>0.9, S_G_place<=0.9) | not stored (S_Gam_gap2<=0.9) |")
print("|---|---|---|---|---|---|---|---|")
for fn, K, head, variant, n, ns, nsnr, nns in summ:
    print(f"| {fn} | {K} | {'yes' if head else 'no'} | {variant} | {n} | {ns}/{n} | {nsnr}/{n} | {nns}/{n} |")

# ---- per-group means and where the join is lost (gap1 -> gap2 -> place)
print("\nPer-group means over seeds, and the locus of the loss.  'written gap1, lost by gap2' = S_Gam_gap1 > 0.9 and S_Gam_gap2 <= 0.9;")
print("'written gap1' = S_Gam_gap1 > 0.9.  max|S_Gam_gap2 - S_G_place| shows whether whatever survives gap2 is read at place.\n")
print("| file | K | head | variant | n | mean S_Gam gap1 | mean S_Gam gap2 | mean S_G place | mean H(C|O) gap1 | mean H(Gam|O) gap1 (per_t) | written gap1 | written gap1, lost by gap2 | max abs(S_Gam_gap2 - S_G_place) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(groups, key=lambda k: (k[1], k[0], k[2], k[3])):
    K, head, variant, fn = k; rs = groups[k]
    sg1 = np.array([r["summary"]["S_Gam_gap1"] for r in rs]); sg2 = np.array([r["summary"]["S_Gam_gap2"] for r in rs])
    sgp = np.array([r["summary"]["S_G_place"] for r in rs]); hc1 = np.array([r["summary"]["HC_gap1"] for r in rs])
    hg1 = np.array([phase_mean(r, "H(Gam|O)", "gap1") for r in rs])
    print(f"| {fn} | {K} | {'yes' if head else 'no'} | {variant} | {len(rs)} | {sg1.mean():.3f} | {sg2.mean():.3f} | {sgp.mean():.3f} | {hc1.mean():.2f} | {hg1.mean():.2f} | {int((sg1 > THR).sum())}/{len(rs)} | {int(((sg1 > THR) & (sg2 <= THR)).sum())}/{len(rs)} | {np.abs(sg2 - sgp).max():.3f} |")
