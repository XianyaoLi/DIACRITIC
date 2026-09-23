"""List models on the cluster that have no closed-loop evaluation yet and write eval job lines: '<models subdir> <model.pt> <env args>'.
Run locally: python isaac/cluster/make_eval_jobs.py > isaac/cluster/jobs_eval_rest.txt   (queries cluster over ssh)"""
import subprocess, re, sys
out = subprocess.run(["ssh", "-o", "BatchMode=yes", "cluster", 'cd $WORKSPACE && for d in models/*/; do for f in $d*.pt; do n=$(basename $f .pt); [ -f "eval/$n/closedloop_eval.jsonl" ] || echo "$(basename $d) $(basename $f)"; done; done'], capture_output=True, text=True).stdout
n = 0
for l in out.splitlines():
    sub, f = l.split()
    if sub == "smoke": continue
    ta2 = re.search(r"taskA2_M(\d+)", f); ta1 = re.search(r"taskA_M(\d+)", f); m = re.search(r"pI_M(\d+)", f); t = re.search(r"tier(\d+)_gap(\d+)", f); ro = re.search(r"readout(b?)(\d)_M(\d+)", f)
    wg = re.search(r"weigh_g(\d+)", f); wh = re.search(r"weighH(\d+)_g(\d+)", f); lk = re.search(r"_leak([0-9.]+)_b", f)
    if wh: print(f"{sub} {f} --M 4 --R1 4 --R2 2 --gap1 {wh.group(2)} --gap2 4 --reveal weigh --mass_max 2.5 --stratified --weigh_hold {wh.group(1)}"); n += 1; continue
    if lk: print(f"{sub} {f} --R1 2 --R2 2 --gap2 6 --leak_mm {lk.group(1)}"); n += 1; continue
    if "_nojit_" in f: print(f"{sub} {f} --R1 2 --R2 2 --gap2 20 --obj_jitter 0"); n += 1; continue
    if wg: args = f"--M 4 --R1 4 --R2 2 --gap1 {wg.group(1)} --gap2 4 --reveal weigh --mass_max 2.5 --stratified"
    elif ro: args = f"--M {ro.group(3)} --R1 4 --R2 {2 if ro.group(2) == '2' else 8} --gap2 6 --reveal {'readout_bits' if ro.group(1) else 'readout'} --stratified"
    elif ta2: args = f"--M {ta2.group(1)} --R1 1 --R2 4 --gap2 6 --reveal theta --reveal_at place --split_latent --no_attach --mass_max 2.0"
    elif ta1: args = f"--M {ta1.group(1)} --R1 1 --R2 4 --gap2 6 --reveal_at place --no_attach --mass_max 2.0"
    elif m: args = f"--M {m.group(1)} --R1 2 --R2 2 --gap2 6 --reveal theta"
    elif t: R = {4: (2, 2), 8: (4, 2), 16: (4, 4)}[int(t.group(1))]; args = f"--R1 {R[0]} --R2 {R[1]} --gap2 {t.group(2)}" + (" --pixel_policy --cam_res 128" if "_px_" in f else "")
    else: continue
    print(f"{sub} {f} {args}"); n += 1
print(f"{n} eval jobs", file=sys.stderr)
