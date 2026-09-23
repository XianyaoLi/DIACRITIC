#!/bin/bash
# Formal local sweep on the 1280-episode tier-4 / gap2=6 dataset (augmented symbols, codes saved).
# Grid: beta x {R (lam=0), RF (lam=1)} x seeds.  Same job list format is reused for the the cluster array (one job per line).
# Usage: bash isaac/sweep_local.sh [PARALLEL=4] [DATA=isaac/data/tier4_gap6_2k] [OUT=isaac/results/tier4_gap6_2k.jsonl]
set -u
PAR=${1:-4}; DATA=${2:-isaac/data/tier4_gap6_2k}; OUT=${3:-isaac/results/tier4_gap6_2k.jsonl}
P=python
TAG=$(basename "${OUT%.jsonl}"); mkdir -p isaac/logs/$TAG isaac/codes/$TAG
jobs() { for beta in 0 0.0003 0.001 0.002 0.003; do for lam in 0 1; do for seed in 0 1 2 3; do echo "$beta $lam $seed"; done; done; done; }
jobs | xargs -P "$PAR" -n 3 sh -c "$P isaac/train_diacritic.py $DATA --beta \$0 --lam \$1 --seed \$2 --steps 10000 --batch 512 --n_train 1152 --sym aug \
    --out $OUT --save_codes isaac/codes/$TAG/b\$0_lam\$1_s\$2.npz > isaac/logs/$TAG/b\$0_lam\$1_s\$2.log 2>&1"
$P isaac/summarize.py "$OUT"
