#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --output=$LOGDIR/%x_eval_%A_%a.out
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
# Closed-loop evaluation of saved DIACRITIC policies inside the Isaac Sim container (IsaacLab jobs -> rtx6000).
# One array task = one line of $JOBS: "<models subdir> <model file> <env args...>"; results -> workspace/diacritic/eval/<model>/
#   sbatch --array=1-N%K --export=ALL,JOBS=<file> ~/DIACRITIC/isaac/cluster/eval.slurm
set -u
umask 077      # every file a job creates is private from the start (parents are 700 as well)
BLUE=$CLUSTER_HOME; W=$BLUE/workspace/diacritic; SIF=$BLUE/workspace/containers/isaac-sim_5.1.0.sif
JOBS=${JOBS:?}; LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS"); [ -n "$LINE" ] || exit 0
read -r SUB MODEL ENVARGS <<< "$LINE"; NAME=${MODEL%.pt}; EV=eval${EVALTAG:+_$EVALTAG}      # EVALTAG=<tag> writes to workspace/diacritic/eval_<tag>/ (hybrid rollouts etc.)
[ -f "$W/$EV/$NAME/closedloop_eval.jsonl" ] && { echo "SKIP $NAME"; exit 0; }
mkdir -p "$W/$EV/$NAME"; chmod 700 "$W/$EV" "$W/$EV/$NAME"
export APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_TMPDIR=$TMPDIR
ln -sfn /isaac-sim $HOME/IsaacLab/_isaac_sim
echo "node $(hostname) task ${SLURM_ARRAY_TASK_ID} model $MODEL args: $ENVARGS  start $(date)"
apptainer exec --nv --bind /blue \
  --bind $BLUE/workspace/kit_cache:/isaac-sim/kit/cache --bind $BLUE/workspace/kit_data:/isaac-sim/kit/data --bind $BLUE/workspace/kit_logs:/isaac-sim/kit/logs \
  --bind $HOME/DIACRITIC:$PROJECT \
  "$SIF" bash -c "export PYTHONUSERBASE=$BLUE/workspace/isaac_deps; SRC=\$HOME/IsaacLab/source; export PYTHONPATH=\$SRC/isaaclab:\$SRC/isaaclab_assets:\$SRC/isaaclab_rl:\$SRC/isaaclab_mimic:\$SRC/isaaclab_tasks:\$PYTHONUSERBASE/lib/python3.11/site-packages; \
    cd $PROJECT && timeout --kill-after=60s 3h /isaac-sim/python.sh isaac/aprime_env.py --headless --num_envs 32 --episodes 4 --seed 900 $ENVARGS --policy $W/models/$SUB/$MODEL --out $W/$EV/$NAME"
rc=$?; chmod -R go-rwx "$W/$EV/$NAME" 2>/dev/null; echo "rc=$rc done $(date)"; exit $rc
