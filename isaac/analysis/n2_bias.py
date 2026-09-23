"""N2: estimator bias of the sufficiency score S_Gamma (and S_G).

S_Gamma(t) = 1 - H(Gamma_t | C_t, Obar_t) / H(Gamma_t | Obar_t) is reported with the plug-in estimator.  Here it is
recomputed from the saved hard codes for
    * the 16 fig3 runs (tier4_gap6_2k, diacritic, lam=0, beta in {0, 0.001}),
    * every gap_refine and scaffold run whose codes exist locally (tier4_gap6 / tier4_gap10 / tier4_gap20),
at every gap2 step with (a) plug-in and (b) Miller-Madow applied through joint entropies,
    H_MM(X) = H_plugin(X) + (m - 1) / (2 n ln 2),  m = occupied cells, n = episodes,
    H(Gamma | C, Obar) = H_MM(Gamma, C, Obar) - H_MM(C, Obar),   H(Gamma | Obar) = H_MM(Gamma, Obar) - H_MM(Obar),
and S_G at the place steps analogously from G_lab.  Steps with H_plugin(Gamma|Obar) < 1e-9 are skipped (as in
train_diacritic.py, where they are nan).  Plug-in values are asserted equal to the stored per_t['S_Gam'] / ['S_G'].

Per run: the minimum and the mean over gap2 steps of S_Gamma (both estimators), the same for S_G over the place steps,
the 'sufficient' verdict at thresholds 0.8 / 0.9 / 0.95 for both estimators and for both aggregates (mean = the paper's
summary convention S_Gam_gap2 / S_G_place; min = the stricter worst-step version), and for the fig3 runs an episode
bootstrap (200 resamples) 95 % interval of the min-over-gap2 plug-in S_Gamma.  Then a table of sufficient-seed counts per
(dataset, variant, beta, lam, refine) cell for 2 estimators x 3 thresholds (x 2 aggregates), flagging every cell whose
count differs between the two estimators.

Runs are matched to code files by basename(args.save_codes) (present in every row of the three jsonl files).

Usage: python isaac/analysis/n2_bias.py [--boot 200] [--json OUT.json]
       (prints markdown; ~1 min)
"""
import os, sys, json, time, argparse, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "toy"))
from aprime_data import prepare, solver_labels   # noqa: E402

LN2 = float(np.log(2.0))
DATA_ROOT = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "cluster_results", "results")
CODES = os.path.join(HERE, "..", "cluster_results", "codes")
THRS = (0.8, 0.9, 0.95)


# ----------------------------------------------------------------------------- entropy helpers (dense int keys)
def dense(labels):
    d = {}
    return np.array([d.setdefault(x, len(d)) for x in labels], dtype=np.int64)


def joint(*cols):
    out = np.zeros_like(cols[0])
    for c in cols:
        out = out * (int(c.max()) + 1) + c
    return np.unique(out, return_inverse=True)[1].astype(np.int64)


def H_from_counts(cnt, mm=False):
    n = cnt.sum(); p = cnt[cnt > 0] / n
    h = float(-(p * np.log2(p)).sum())
    if mm:
        h += (len(p) - 1) / (2.0 * n * LN2)
    return h


def H(x, mm=False, idx=None):
    if idx is not None: x = x[idx]
    return H_from_counts(np.bincount(x), mm)


def suff_score(target, c, o, mm=False, idx=None):
    """S = 1 - H(target | c, o) / H(target | o); nan when the plug-in H(target|o) < 1e-9 (both estimators skip that step)."""
    Hto = H(joint(target, o), False, idx) - H(o, False, idx)
    if Hto < 1e-9:
        return float("nan")
    if mm:
        Hto = H(joint(target, o), True, idx) - H(o, True, idx)
    Htco = H(joint(target, c, o), mm, idx) - H(joint(c, o), mm, idx)
    return 1.0 - Htco / Hto


def nanmin(v): v = [x for x in v if not math.isnan(x)]; return min(v) if v else float("nan")
def nanmean(v): v = [x for x in v if not math.isnan(x)]; return float(np.mean(v)) if v else float("nan")


# ----------------------------------------------------------------------------- run selection
def select_runs():
    runs = []
    for tag in ("fig3", "gap_refine", "scaffold"):
        p = os.path.join(RES, f"{tag}.jsonl")
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); a = r["args"]
            if tag == "fig3" and not (os.path.basename(a["data"]) == "tier4_gap6_2k" and a["variant"] == "diacritic"
                                      and float(a["lam"]) == 0.0 and float(a["beta"]) in (0.0, 0.001)):
                continue
            fn = os.path.join(CODES, tag, os.path.basename(a["save_codes"]))
            r["_tag"] = tag; r["_codes"] = fn; r["_dataset"] = os.path.basename(a["data"])
            r["_refine"] = (a.get("refine") or "none")
            runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--json", default="")
    args = ap.parse_args()
    t0 = time.time()
    runs = select_runs()
    missing = [r for r in runs if not os.path.exists(r["_codes"])]
    runs = [r for r in runs if os.path.exists(r["_codes"])]
    print(f"{len(runs)} runs with local codes ({len(missing)} jsonl rows without a local code file" +
          (": " + ", ".join(os.path.basename(r['_codes']) for r in missing) if missing else "") + ")\n")

    labels = {}
    def get_labels(ds):
        if ds not in labels:
            eps, meta, vis, env, solver, key = prepare(os.path.join(DATA_ROOT, ds), augment=True)
            G_lab, Gam_lab, GamS_lab, O_lab = solver_labels(solver, eps, key)
            T = len(eps[0]["phase"]); phases = [str(p) for p in eps[0]["phase"]]
            labels[ds] = dict(E=len(eps), T=T, phases=phases, gap2=[t for t, p in enumerate(phases) if p == "gap2"],
                              place=[t for t, p in enumerate(phases) if p == "place"],
                              G=[dense(G_lab[t]) for t in range(T)], Gam=[dense(Gam_lab[t]) for t in range(T)], O=[dense(O_lab[t]) for t in range(T)])
            print(f"loaded {ds}: E={labels[ds]['E']}, T={T}, gap2 steps (0-based) {labels[ds]['gap2']}, place steps {labels[ds]['place']}")
        return labels[ds]

    rng = np.random.RandomState(0)
    recs = []
    for r in sorted(runs, key=lambda r: (r["_tag"], r["_dataset"], r["args"]["variant"], float(r["args"]["beta"]), float(r["args"]["lam"]), r["_refine"], int(r["args"]["seed"]))):
        L = get_labels(r["_dataset"]); E = L["E"]
        C = np.load(r["_codes"])["codes"]; assert C.shape == (E, L["T"]), (r["_codes"], C.shape)
        a, s = r["args"], r["summary"]
        rec = dict(tag=r["_tag"], dataset=r["_dataset"], variant=a["variant"], beta=float(a["beta"]), lam=float(a["lam"]), refine=r["_refine"],
                   seed=int(a["seed"]), file=os.path.basename(r["_codes"]), stored=dict(S_Gam_gap2=s["S_Gam_gap2"], S_G_place=s["S_G_place"]),
                   HGam_O_gap2=float(np.mean([r["per_t"]["H(Gam|O)"][t] for t in L["gap2"]])))
        for est in ("plug", "mm"):
            mm = est == "mm"
            sg = [suff_score(L["Gam"][t], dense(C[:, t]), L["O"][t], mm) for t in L["gap2"]]
            sp = [suff_score(L["G"][t], dense(C[:, t]), L["O"][t], mm) for t in L["place"]]
            if not mm:      # alignment / definition check against the stored per-step values
                for t, v in zip(L["gap2"], sg):
                    st = r["per_t"]["S_Gam"][t]; assert (math.isnan(v) and st is None or math.isnan(st)) or abs(v - st) < 1e-6, (rec["file"], t, v, st)
                for t, v in zip(L["place"], sp):
                    st = r["per_t"]["S_G"][t]; assert (math.isnan(v) and (st is None or math.isnan(st))) or abs(v - st) < 1e-6, (rec["file"], t, v, st)
            rec[est] = dict(gam_steps=sg, g_steps=sp, gam_min=nanmin(sg), gam_mean=nanmean(sg), g_min=nanmin(sp), g_mean=nanmean(sp))
        assert abs(rec["plug"]["gam_mean"] - s["S_Gam_gap2"]) < 1e-6 and abs(rec["plug"]["g_mean"] - s["S_G_place"]) < 1e-6
        for est in ("plug", "mm"):
            for agg in ("mean", "min"):
                for thr in THRS:
                    rec[f"suf_{est}_{agg}_{thr}"] = bool(rec[est][f"gam_{agg}"] > thr and rec[est][f"g_{agg}"] > thr)
        if r["_tag"] == "fig3":
            mins = []
            for _ in range(args.boot):
                idx = rng.randint(0, E, E)
                mins.append(nanmin([suff_score(L["Gam"][t], dense(C[:, t]), L["O"][t], False, idx) for t in L["gap2"]]))
            rec["boot_min_lo"], rec["boot_min_hi"] = float(np.nanpercentile(mins, 2.5)), float(np.nanpercentile(mins, 97.5))
        recs.append(rec)
    print()

    # per-run table
    print("### Per-run S_Gamma over the gap2 steps and S_G over the place steps (plug-in vs Miller-Madow)\n")
    print("| tag | dataset | variant | beta | lam | refine | seed | H(Gam|Obar) gap2 (stored, mean) | S_Gam gap2 min plug | min MM | mean plug (= stored) | mean MM | S_G place min plug | min MM | mean plug (= stored) | mean MM | sufficient@0.9 (mean agg) plug / MM | sufficient@0.9 (min agg) plug / MM | boot 95% of min-gap2 plug-in S_Gam |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in recs:
        p, m = x["plug"], x["mm"]
        boot = f"[{x['boot_min_lo']:.3f}, {x['boot_min_hi']:.3f}]" if "boot_min_lo" in x else "--"
        print(f"| {x['tag']} | {x['dataset']} | {x['variant']} | {x['beta']} | {x['lam']} | {x['refine']} | {x['seed']} | {x['HGam_O_gap2']:.3f} | {p['gam_min']:.3f} | {m['gam_min']:.3f} | {p['gam_mean']:.3f} | {m['gam_mean']:.3f} | {p['g_min']:.3f} | {m['g_min']:.3f} | {p['g_mean']:.3f} | {m['g_mean']:.3f} | "
              f"{'yes' if x['suf_plug_mean_0.9'] else 'no'} / {'yes' if x['suf_mm_mean_0.9'] else 'no'} | {'yes' if x['suf_plug_min_0.9'] else 'no'} / {'yes' if x['suf_mm_min_0.9'] else 'no'} | {boot} |")
    print()

    # cell tables
    cells = {}
    for x in recs:
        cells.setdefault((x["tag"], x["dataset"], x["variant"], x["beta"], x["lam"], x["refine"]), []).append(x)
    flagged = []
    for agg, label in (("mean", "paper aggregate: mean over gap2 steps of S_Gam and mean over place steps of S_G"), ("min", "strict aggregate: min over gap2 steps of S_Gam and min over place steps of S_G")):
        print(f"### Sufficient-seed counts per cell, {label}\n")
        print("| tag | dataset | variant | beta | lam | refine | n | thr 0.8 plug | thr 0.8 MM | thr 0.9 plug | thr 0.9 MM | thr 0.95 plug | thr 0.95 MM | changed (plug vs MM) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in sorted(cells, key=str):
            xs = cells[k]; cols = []; ch = []
            for thr in THRS:
                np_ = sum(x[f"suf_plug_{agg}_{thr}"] for x in xs); nm = sum(x[f"suf_mm_{agg}_{thr}"] for x in xs)
                cols += [str(np_), str(nm)]
                if np_ != nm: ch.append(f"thr {thr}: {np_}->{nm}")
            if ch: flagged.append((agg, k, ch))
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {len(xs)} | " + " | ".join(cols) + f" | {'; '.join(ch) if ch else '-'} |")
        print()
    print("### Cells whose sufficient count changes between the estimators\n")
    if flagged:
        for agg, k, ch in flagged:
            print(f"- [{agg} aggregate] {k[0]} {k[1]} {k[2]} beta={k[3]} lam={k[4]} refine={k[5]}: " + "; ".join(ch))
    else:
        print("- none")
    # largest per-run differences
    d = sorted(recs, key=lambda x: x["plug"]["gam_min"] - x["mm"]["gam_min"], reverse=True)
    print("\nLargest plug-in minus Miller-Madow differences of the min-over-gap2 S_Gam (top 8):")
    for x in d[:8]:
        print(f"- {x['file']}: plug {x['plug']['gam_min']:.3f}, MM {x['mm']['gam_min']:.3f}, diff {x['plug']['gam_min']-x['mm']['gam_min']:.3f}; mean agg plug {x['plug']['gam_mean']:.3f} MM {x['mm']['gam_mean']:.3f}")
    allsg = [(x["plug"]["gam_mean"] - x["mm"]["gam_mean"]) for x in recs]; allsp = [(x["plug"]["g_mean"] - x["mm"]["g_mean"]) for x in recs]
    print(f"\nOver all {len(recs)} runs: plug-in minus MM of the gap2-mean S_Gam: mean {np.mean(allsg):.4f}, max {np.max(allsg):.4f}; of the place-mean S_G: mean {np.mean(allsp):.4f}, max {np.max(allsp):.4f}.")
    print(f"Runs sufficient at 0.9 (mean agg) under plug-in: {sum(x['suf_plug_mean_0.9'] for x in recs)}, under MM: {sum(x['suf_mm_mean_0.9'] for x in recs)}; verdict flips: {sum(x['suf_plug_mean_0.9'] != x['suf_mm_mean_0.9'] for x in recs)}.")

    if args.json:
        json.dump(recs, open(args.json, "w"), indent=1, default=lambda o: o.item() if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
