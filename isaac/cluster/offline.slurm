#!/bin/bash
#SBATCH --job-name=offline
#SBATCH --output=$LOGDIR/%x_%A_%a.out
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
# Teacher-forced re-evaluation of saved models on an independent recording (isaac/eval_offline.py); CPU only.
# One array task = one line of $JOBS: "<new dataset> <original dataset> <model glob relative to models/> <output name>" -> results/<output name>.jsonl
set -u
umask 077      # every file a job creates is private from the start (parents are 700 as well)
module purge; module load pytorch/2.7
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
W=$WORKSPACE; JOBS=${JOBS:?}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS"); [ -n "$LINE" ] || exit 0
read -r NEW ORIG GLOB OUT <<< "$LINE"
cd "$HOME/DIACRITIC"; rm -f "$W/results/$OUT.jsonl"
python isaac/eval_offline.py --models $W/models/$GLOB --data "$W/data/$NEW" --orig "$W/data/$ORIG" --sym aug --device cpu --out "$W/results/$OUT.jsonl"
rc=$?; chmod 600 "$W/results/$OUT.jsonl" 2>/dev/null; echo "rc=$rc done $(date)"; exit $rc
