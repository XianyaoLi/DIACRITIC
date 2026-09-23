#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=$LOGDIR/toy_%A_%a.out
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=03:00:00
#SBATCH --requeue
# Grid-corridor (toy) training array on CPU nodes.  One task = one line of $JOBS (toy/cluster/make_toy_jobs.py); finished lines are skipped
# (a marker file per line).  sbatch --array=1-N%K --export=ALL,JOBS=<file> ~/DIACRITIC/toy/cluster/toy.slurm
set -u
# Each array task runs a BLOCK of $BLK consecutive job lines in parallel (one CPU thread each) so that throughput per throttled job is x$BLK.
module purge; module load pytorch/2.7
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1
CODE=$HOME/DIACRITIC; W=$WORKSPACE; BLK=${BLK:-4}
JOBS=${JOBS:?}; mkdir -p "$W/results" "$W/toy_done"; chmod 700 "$W/toy_done" 2>/dev/null
echo "node $(hostname) task ${SLURM_ARRAY_TASK_ID} start $(date)"
run_line() {
  LINE="$1"; [ -n "$LINE" ] || return 0
  read -r TAG M R1 R2 WD GAP1 GAP2 VARIANT BETA LAM SEED STEPS L2 LR MIK AUX BOUND K ENV <<< "$LINE"; BOUND=${BOUND:-none}; K=${K:-16}; ENV=${ENV:-grid}
  NAME="${TAG}_M${M}_R${R1}x${R2}_W${WD}_g${GAP1}-${GAP2}_${VARIANT}_b${BETA}_lam${LAM}_mik${MIK}_aux${AUX}_bd${BOUND}_K${K}_${ENV}_s${SEED}"
  mkdir -p "$W/toy_done/$TAG"; chmod 700 "$W/toy_done/$TAG" 2>/dev/null
  [ -f "$W/toy_done/$TAG/$NAME" ] && { echo "SKIP $NAME"; return 0; }
  echo "spec: $LINE"
  ( cd "$CODE/toy" && timeout --kill-after=30s 170m python toy_diacritic.py --env "$ENV" --M "$M" --R1 "$R1" --R2 "$R2" --W "$WD" --gap1 "$GAP1" --gap2 "$GAP2" --variant "$VARIANT" \
      --beta "$BETA" --lam "$LAM" --seed "$SEED" --steps "$STEPS" --l2 "$L2" --lr "$LR" --mik "$MIK" --aux_theta "$AUX" --bound "$BOUND" --K "$K" --out "$W/results/$TAG.jsonl.$SLURM_ARRAY_TASK_ID.$2" > "$W/toy_done/$TAG/$NAME.log" 2>&1 )
  rc=$?; [ $rc -eq 0 ] && touch "$W/toy_done/$TAG/$NAME"; echo "rc=$rc $NAME $(grep -E "codes_used" "$W/toy_done/$TAG/$NAME.log" | tail -1)"
}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * BLK + 1 ))
for i in $(seq 0 $((BLK - 1))); do run_line "$(sed -n "$((START + i))p" "$JOBS")" "$i" & done; wait
# merge the per-run shards into the tag jsonl (append; shards removed)
for f in "$W"/results/*.jsonl."$SLURM_ARRAY_TASK_ID".*; do [ -f "$f" ] || continue; t=${f%%.jsonl.*}.jsonl; cat "$f" >> "$t" && rm -f "$f"; chmod 600 "$t"; done
echo "done $(date)"; exit 0
