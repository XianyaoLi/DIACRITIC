#!/bin/bash
# P-I (M-sweep, reveal theta): per M, -RF beta=2e-3 x 4 seeds + beta=0 (-R) control x 4 seeds; batch 512 / 10000 steps.
# Waits for each dataset (collect_pI.log line) before launching its jobs.  Usage: bash isaac/sweep_pI.sh [PARALLEL=4]
set -u
PAR=${1:-4}; P=python; OUT=isaac/results/pI_msweep.jsonl
mkdir -p isaac/logs/pI isaac/codes/pI
for M in 4 8 16 32; do
  until grep -q "pI_M$M exit=0" isaac/logs/collect_pI.log 2>/dev/null; do sleep 60; done
  D=isaac/data/pI_M$M
  $P isaac/accept_solver.py $D > isaac/logs/accept_solver_pI_M$M.log 2>&1
  for seed in 0 1 2 3; do echo "$M 0.002 1 $seed"; echo "$M 0 0 $seed"; done | xargs -P "$PAR" -n 4 sh -c \
    "$P isaac/train_diacritic.py isaac/data/pI_M\$0 --beta \$1 --lam \$2 --seed \$3 --steps 10000 --batch 512 --n_train 448 --sym aug \
     --out $OUT --save_codes isaac/codes/pI/M\$0_b\$1_lam\$2_s\$3.npz > isaac/logs/pI/M\$0_b\$1_lam\$2_s\$3.log 2>&1"
done
echo "pI sweep done"
