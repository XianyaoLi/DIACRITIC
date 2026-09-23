#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --output=$LOGDIR/%x_rec_%A_%a.out
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --partition=<gpu-partition-a>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --requeue
# Expert-data recording inside the Isaac Sim container (IsaacLab jobs -> rtx6000).
# One array task = one line of $JOBS: "<dataset name> <num_envs> <episodes> <seed> <env args...>"; data -> workspace/diacritic/data/<name>/
#   sbatch --array=1-N%K --export=ALL,JOBS=<file> ~/DIACRITIC/isaac/cluster/record.slurm
set -u
umask 077      # every file a job creates is private from the start (parents are 700 as well)
BLUE=$CLUSTER_HOME; W=$BLUE/workspace/diacritic; SIF=$BLUE/workspace/containers/isaac-sim_5.1.0.sif
JOBS=${JOBS:?}; LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS"); [ -n "$LINE" ] || exit 0
read -r NAME NENV NEP SEED ENVARGS <<< "$LINE"
[ -f "$W/data/$NAME/meta.json" ] && [ "$(ls $W/data/$NAME/*.npz 2>/dev/null | wc -l)" -ge $((NENV * NEP)) ] && { echo "SKIP $NAME"; exit 0; }
mkdir -p "$W/data/$NAME"; chmod 700 "$W/data" "$W/data/$NAME"
export APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_TMPDIR=$TMPDIR
ln -sfn /isaac-sim $HOME/IsaacLab/_isaac_sim
echo "node $(hostname) task ${SLURM_ARRAY_TASK_ID} dataset $NAME envs $NENV x $NEP seed $SEED args: $ENVARGS  start $(date)"
apptainer exec --nv --bind /blue \
  --bind $BLUE/workspace/kit_cache:/isaac-sim/kit/cache --bind $BLUE/workspace/kit_data:/isaac-sim/kit/data --bind $BLUE/workspace/kit_logs:/isaac-sim/kit/logs \
  --bind $HOME/DIACRITIC:$PROJECT \
  "$SIF" bash -c "export PYTHONUSERBASE=$BLUE/workspace/isaac_deps; SRC=\$HOME/IsaacLab/source; export PYTHONPATH=\$SRC/isaaclab:\$SRC/isaaclab_assets:\$SRC/isaaclab_rl:\$SRC/isaaclab_mimic:\$SRC/isaaclab_tasks:\$PYTHONUSERBASE/lib/python3.11/site-packages; \
    cd $PROJECT && timeout --kill-after=60s 3h /isaac-sim/python.sh isaac/aprime_env.py --headless --num_envs $NENV --episodes $NEP --seed $SEED $ENVARGS --out $W/data/$NAME"
rc=$?; chmod -R go-rwx "$W/data/$NAME" 2>/dev/null; echo "rc=$rc done $(date)"; exit $rc
