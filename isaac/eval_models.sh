#!/bin/bash
# Closed-loop evaluation of saved policies in Isaac (local RTX 4090 or the cluster rtx6000 + container).
# Usage: bash isaac/eval_models.sh <model glob> [ENVS=32] [EPISODES=4] [extra env args...]
# Each model is evaluated on the env config recorded in its checkpoint (R1/R2/gap2/M/reveal read from extra.args.data name).
set -u; GLOB=${1:?model glob}; ENVS=${2:-32}; EPS=${3:-4}; shift $(( $# < 3 ? $# : 3 ))
P=python; mkdir -p isaac/eval isaac/logs/eval
for M in $GLOB; do
  name=$(basename "$M" .pt)
  cfg=$($P - "$M" <<'PY'
import sys, torch, os, re
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); ds = os.path.basename(ck["extra"]["args"]["data"])
m = re.search(r"pI_M(\d+)", ds); tier = re.search(r"tier(\d+)_gap(\d+)", ds)
ta = re.search(r"taskA2_M(\d+)", ds)
if ta: print(f"--M {ta.group(1)} --R1 1 --R2 4 --gap2 6 --reveal theta --reveal_at place --split_latent --no_attach --mass_max 2.0")
elif m: print(f"--M {m.group(1)} --R1 2 --R2 2 --gap2 6 --reveal theta")
else:
    g1 = re.search(r"g1_(\d+)", ds); R = {4: (2, 2), 8: (4, 2), 16: (4, 4)}[int(tier.group(1))] if tier else (2, 2); print(f"--R1 {R[0]} --R2 {R[1]} --gap2 {tier.group(2) if tier else 6}" + (f" --gap1 {g1.group(1)}" if g1 else "") + (" --no_attach" if "noattach" in ds else ""))
PY
)
  [ -f "isaac/eval/$name/closedloop_eval.jsonl" ] && { echo "SKIP $name"; continue; }
  echo "== $name  ($cfg)"
  timeout 3600 ~/IsaacLab/isaaclab.sh -p isaac/aprime_env.py --headless --num_envs "$ENVS" --episodes "$EPS" --seed 900 $cfg --policy "$M" --out "isaac/eval/$name" "$@" > "isaac/logs/eval/$name.log" 2>&1
  grep -E "EVAL|Error|Traceback" "isaac/logs/eval/$name.log" | cut -c1-220
done
