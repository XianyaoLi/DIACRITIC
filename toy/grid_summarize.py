"""Summarise grid-corridor runs (toy/cluster results grid_*.jsonl): per (tag, variant, beta, lam, M, W, gap1, gap2): seeds reaching sufficiency
(S_Gam > 0.9 at every gap step and err = 0 at both junctions), rates H(C|O) in hall 1 / hall 2 vs theory, surplus H(C|Gam,O), I(C;theta|O),
I(C;w|O) in the halls, and the sandwich when Gamma is undefined.  Usage: python toy/grid_summarize.py <jsonl...> [--csv out.csv]"""
import sys, json, collections, re
import numpy as np
def read_records(f):
    """jsonl, or concatenated JSON objects without newlines (early shard merge)."""
    txt = open(f).read(); dec = json.JSONDecoder(); i = 0; out = []
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t": i += 1
        if i >= len(txt): break
        try: obj, j = dec.raw_decode(txt, i); out.append(obj); i = j
        except Exception: i = txt.find("\n", i); i = len(txt) if i < 0 else i + 1
    return out


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else ""; files = [f for f in sys.argv[1:] if not f.startswith("--") and f != csv]
    rows = []
    for f in files: rows += read_records(f)
    def rng(r, ph):
        a, b = r["phases"][ph]; return list(range(a - 1, b))
    def key(r):
        a = r["args"]; m = re.search(r"M=(\d+),R1=(\d+),R2=(\d+),W=(\d+),N=\d+,gap1=(\d+),gap2=(\d+)", r["env"])
        return (re.sub(r"\.jsonl.*$", "", a["out"].split("/")[-1]), a["variant"], a["beta"], a["lam"], int(m.group(1)), int(m.group(4)), int(m.group(5)), int(m.group(6)))
    g = collections.defaultdict(list)
    for r in rows: g[key(r)].append(r)
    def suff(r):
        res = r["res"]; g1, g2 = rng(r, "gap1"), rng(r, "gap2"); u1, u2 = r["phases"]["use1"] - 1, r["phases"]["use2"] - 1
        sg = [res["S_Gam"][i] for i in g1 + g2 if not np.isnan(res["S_Gam"][i])]
        exact = all(v > 0.9 for v in sg) if sg else True                          # general regime: no Gamma labels -> use behaviour only
        return exact and res["err"][u1] < 1e-6 and res["err"][u2] < 1e-6
    def m(r, k, ph): return float(np.nanmean([r["res"][k][i] for i in rng(r, ph)]))
    def th(r, k, ph): return float(np.nanmean([r["theory"][k][i] for i in rng(r, ph)]))
    hdr = f"{'tag':13s} {'variant':10s} {'beta':>5s} {'lam':>3s} {'M':>3s} {'W':>2s} {'g1':>3s} {'g2':>3s} n | suff | H(C|O) hall1 [Gam,GamS] hall2 [Gam,GamS] | H(C|Gam,O) | I(C;th|O) h1 h2 | I(C;w|O) h1 h2 | err use2 | H(H|O) h1"
    print(hdr); out = []
    for k, rs in sorted(g.items()):
        ok = [suff(r) for r in rs]; S = [r for r, o in zip(rs, ok) if o] or rs
        f = lambda fn, R=S: float(np.nanmean([fn(r) for r in R]))
        r0 = rs[0]
        line = (f"{k[0]:13s} {k[1]:10s} {k[2]:5.3f} {k[3]:3g} {k[4]:3d} {k[5]:2d} {k[6]:3d} {k[7]:3d} {len(rs)} | {sum(ok)}/{len(rs)} | "
                f"{f(lambda r: m(r,'H(C|O)','gap1')):5.2f} [{th(r0,'H(Gamma|O)','gap1'):4.2f},{th(r0,'H(GammaS|O)','gap1'):4.2f}]  {f(lambda r: m(r,'H(C|O)','gap2')):5.2f} [{th(r0,'H(Gamma|O)','gap2'):4.2f},{th(r0,'H(GammaS|O)','gap2'):4.2f}] | "
                f"{f(lambda r: m(r,'H(C|Gam,O)','gap2')):5.2f} | {f(lambda r: m(r,'I(C;theta|O)','gap1')):4.2f} {f(lambda r: m(r,'I(C;theta|O)','gap2')):4.2f} | "
                f"{f(lambda r: m(r,'I(C;w|O)','gap1')):4.2f} {f(lambda r: m(r,'I(C;w|O)','gap2')):4.2f} | {f(lambda r: r['res']['err'][r['phases']['use2']-1], rs):4.2f} | {th(r0,'H(H|O)','gap1'):4.2f}")
        print(line)
        out.append(dict(tag=k[0], variant=k[1], beta=k[2], lam=k[3], M=k[4], W=k[5], gap1=k[6], gap2=k[7], n=len(rs), n_suff=sum(ok),
                        HC_h1=f(lambda r: m(r,'H(C|O)','gap1')), HC_h2=f(lambda r: m(r,'H(C|O)','gap2')), HGam_h1=th(r0,'H(Gamma|O)','gap1'), HGam_h2=th(r0,'H(Gamma|O)','gap2'),
                        HGamS_h1=th(r0,'H(GammaS|O)','gap1'), HGamS_h2=th(r0,'H(GammaS|O)','gap2'), ICw_h1=f(lambda r: m(r,'I(C;w|O)','gap1')), ICth_h1=f(lambda r: m(r,'I(C;theta|O)','gap1')),
                        err_use2=f(lambda r: r['res']['err'][r['phases']['use2']-1], rs), HH_h1=th(r0,'H(H|O)','gap1')))
    if csv:
        import csv as _csv
        with open(csv, "w") as fh: w = _csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print("saved", csv)


if __name__ == "__main__":
    main()
