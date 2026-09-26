#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=$LOGDIR/%x_%A_%a.out
#SBATCH --account=<account>
#SBATCH --qos=<account>
#SBATCH --partition=<gpu-partition-b>,<gpu-partition-a>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL,REQUEUE
# DIACRITIC offline training array.  One job = one line of $JOBS (see make_jobs.py); already-finished lines are skipped.
#   sbatch --array=1-N%K --export=ALL,JOBS=<jobs file> isaac/cluster/train.slurm
# Rules: training -> B200 first, RTX6000 as fallback (RTX6000 is reserved for IsaacLab jobs); code in ~/DIACRITIC (700); everything on blue under workspace/diacritic/; logs in logs/diacritic/; <=8 CPU, <=46 GB;
# combined parallelism with TANAV <= 10 (throttle.sh).  Resource caps here (4 CPU / 16 GB) are what one training run needs.
set -u
umask 077      # every file a job creates is private from the start (parents are 700 as well)
module purge; module load pytorch/2.7
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
CODE=$HOME/DIACRITIC; W=$WORKSPACE
JOBS=${JOBS:?set JOBS=<jobs file>}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS"); [ -n "$LINE" ] || { echo "no job line ${SLURM_ARRAY_TASK_ID}"; exit 0; }
read -r DS BETA LAM SEED VARIANT STEPS BATCH NTRAIN TAG AUX REFINE INIT JOIN K MIK MIKD SYMIN DISTILL NOOBS DTGT HIGHW DUALK DUALD DSTRIDE DSRC FCSTEPS FCCORR DW DOFF FCSTART NORMFIT <<< "$LINE"; DOFF=${DOFF:-0}; NORMFIT=${NORMFIT:-all}; FCSTART=${FCSTART:-reveal}; DUALK=${DUALK:-0}; DUALD=${DUALD:-0}; DSTRIDE=${DSTRIDE:-1}; DSRC=${DSRC:-teacher}; FCSTEPS=${FCSTEPS:-3000}; FCCORR=${FCCORR:-0}; DW=${DW:-1}; AUX=${AUX:-0}; REFINE=${REFINE:-none}; INIT=${INIT:-none}; JOIN=${JOIN:-0}; K=${K:-16}; MIK=${MIK:-0}; MIKD=${MIKD:-0}; SYMIN=${SYMIN:-0}; DISTILL=${DISTILL:-none}; NOOBS=${NOOBS:-0}; DTGT=${DTGT:-forecast}; HIGHW=${HIGHW:-1}
DISTARG=""; [ "$DISTILL" != "none" ] && DISTARG="--distill $W/models/$DISTILL"
INITARG=""; [ "$INIT" != "none" ] && INITARG="--init_from $W/models/$INIT"
NAME="${DS}_b${BETA}_lam${LAM}_${VARIANT}_aux${AUX}_ref${REFINE}_s${SEED}"; [ "$INIT" != "none" ] && NAME="${NAME}_curric"; [ "$JOIN" != "0" ] && NAME="${NAME}_oracle"; [ "$K" != "16" ] && NAME="${NAME}_K${K}"; [ "$MIK" != "0" ] && NAME="${NAME}_mik${MIK}$([ "$MIKD" != "0" ] && echo d)"; [ "$SYMIN" != "0" ] && NAME="${NAME}_symin${SYMIN}"; [ "$DUALK" != "0" ] && NAME="${NAME}_dual${DUALK}$([ "$DUALD" != "0" ] && echo d)"; [ "$DSTRIDE" != "1" ] && NAME="${NAME}_ds${DSTRIDE}"; [ "$HIGHW" != "1" ] && NAME="${NAME}_hw${HIGHW}"; [ "$NTRAIN" != "448" ] && [ "$NTRAIN" != "1152" ] && NAME="${NAME}_N${NTRAIN}"; [ "$DISTILL" != "none" ] && NAME="${NAME}_distill${DTGT}"; [ "$NOOBS" != "0" ] && NAME="${NAME}_noobs"; [ "$DSRC" != "teacher" ] && NAME="${NAME}_src${DSRC}"; [ "$FCSTEPS" != "3000" ] && NAME="${NAME}_fcs${FCSTEPS}"; [ "$FCCORR" != "0" ] && NAME="${NAME}_cor${FCCORR}"; [ "$DW" != "1" ] && NAME="${NAME}_dw${DW}"; [ "$DOFF" != "0" ] && NAME="${NAME}_off${DOFF}"; [ "$FCSTART" != "reveal" ] && NAME="${NAME}_from${FCSTART}"
REFARG=""; [ "$REFINE" != "none" ] && REFARG="--refine $REFINE"
PXARG=""; [ "${DS%_px}" != "$DS" ] && PXARG="--pixels"     # datasets named *_px carry camera images -> pixel CNN branch
mkdir -p "$W/results" "$W/codes/$TAG" "$W/models/$TAG"; chmod 700 "$W" "$W/results" "$W/codes" "$W/codes/$TAG" "$W/models" "$W/models/$TAG" 2>/dev/null
echo "node $(hostname) task ${SLURM_ARRAY_TASK_ID} job ${SLURM_JOB_ID} restarts ${SLURM_RESTART_COUNT:-0} start $(date)"; echo "spec: $LINE"
if [ -f "$W/codes/$TAG/$NAME.npz" ]; then echo "SKIP (done): $NAME"; exit 0; fi
cd "$CODE"
timeout --kill-after=60s 5h python isaac/train_diacritic.py "$W/data/$DS" --beta "$BETA" --lam "$LAM" --seed "$SEED" --variant "$VARIANT" --aux_theta "$AUX" --aux_join "$JOIN" --K "$K" --aux_mik "$MIK" --mik_detach "$MIKD" --sym_input "$SYMIN" --bfs_noobs "$NOOBS" --distill_target "$DTGT" --high_w "$HIGHW" --dual_K "$DUALK" --dual_detach "$DUALD" --distill_stride "$DSTRIDE" --distill_source "$DSRC" --fc_steps "$FCSTEPS" --fc_corrupt "$FCCORR" --distill_w "$DW" --distill_off "$DOFF" --fc_start "$FCSTART" --norm_fit "$NORMFIT" $DISTARG $REFARG $INITARG $PXARG \
    --steps "$STEPS" --batch "$BATCH" --n_train "$NTRAIN" --sym $([ "${DS#taskA2}" != "$DS" ] && echo aug_sag || echo aug) --out "$W/results/$TAG.jsonl" --save_codes "$W/codes/$TAG/$NAME.npz" --save_model "$W/models/$TAG/$NAME.pt"
rc=$?; chmod 600 "$W/results/$TAG.jsonl" "$W/codes/$TAG/$NAME.npz" "$W/models/$TAG/$NAME.pt" $LOGDIR/train_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>/dev/null
echo "rc=$rc done $(date)"; exit $rc
