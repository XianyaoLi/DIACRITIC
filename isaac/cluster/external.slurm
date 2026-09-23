#!/bin/bash
#SBATCH --job-name=ext
#SBATCH --output=$LOGDIR/%x_%A_%a.out
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=06:00:00
#SBATCH --requeue
# External-benchmark runs on CPU (tiny models, python-loop bound).  One array task = one line of $JOBS: "<script> <args...>" -> results under workspace/diacritic/external/
set -u
umask 077
module purge; module load pytorch/2.7
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONPATH=$EXT_DEPS
W=$WORKSPACE/external; mkdir -p $W; JOBS=${JOBS:?}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS"); [ -n "$LINE" ] || exit 0
read -r SCRIPT OUTNAME ARGS <<< "$LINE"
cd "$HOME/DIACRITIC"; echo "task ${SLURM_ARRAY_TASK_ID}: $LINE  start $(date)"
python external/$SCRIPT $ARGS --device cpu --out "$W/$OUTNAME.${SLURM_ARRAY_TASK_ID}.jsonl"; rc=$?; echo "rc=$rc done $(date)"; exit $rc
