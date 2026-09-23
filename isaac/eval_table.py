"""Collect closed-loop evaluations (isaac/eval/<name>/closedloop_eval.jsonl + closed-loop code sufficiency) into one table.
Usage: python isaac/eval_table.py isaac/eval/<glob>   e.g. 'isaac/eval/tier4_gap6_*'"""
import sys, os, glob, json, subprocess, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for d in sorted(glob.glob(sys.argv[1])):
    f = os.path.join(d, "closedloop_eval.jsonl")
    if not os.path.isfile(f): continue
    r = json.loads(open(f).readlines()[-1])
    out = subprocess.run([sys.executable, os.path.join(HERE, "closedloop_codes.py"), d], capture_output=True, text=True).stdout
    g1 = re.search(r"gap1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out); g2 = re.search(r"gap2\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
    rows.append(dict(name=os.path.basename(d), succ=r["success_rate"], side=r["side_acc"], slot=r["slot_acc"], err=r["action_err_cm"],
                     cl_gap1=float(g1.group(1)) if g1 else float("nan"), cl_gap2=float(g2.group(1)) if g2 else float("nan"),
                     cl_suff_b2=float(g2.group(3)) if g2 else float("nan"), n=r["episodes"]))
print(f"{'model':48s} eps  succ  side  slot  err(cm) | CL H(C|O) gap1 gap2 | CL suff(b2)@gap2")
for r in rows:
    print(f"{r['name']:48s} {r['n']:4d} {r['succ']:5.2f} {r['side']:5.2f} {r['slot']:5.2f} {r['err']:6.2f}  |  {r['cl_gap1']:4.2f}  {r['cl_gap2']:4.2f}   |  {r['cl_suff_b2']:4.2f}")
if rows:
    cl_ok = [r for r in rows if r["cl_suff_b2"] > 0.9]; print(f"\nclosed-loop sufficient (suff(b2)@gap2 > 0.9): {len(cl_ok)}/{len(rows)}; their success {np.mean([r['succ'] for r in cl_ok]) if cl_ok else float('nan'):.2f}, gap2 rate {np.mean([r['cl_gap2'] for r in cl_ok]) if cl_ok else float('nan'):.2f}")
