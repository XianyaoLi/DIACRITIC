#!/usr/bin/env bash
# Pooled temporal-distance regression of Section 5.1 / Appendix D.6 (isaac/horizon_law.py), run on the exact row source:
# the A' tier-4 state-observation ledgers of the unsupervised sole-carrier learners. horizon_law.py itself keeps only
# variants diacritic/scaffold and drops bypass / Transformer / oracle heads / pixels / Task A rows, giving 640 (run, class)
# rows from 320 runs in 34 configurations. Requires numpy and scipy; about a minute on a CPU.
cd "$(dirname "$0")/../.."
R=isaac/cluster_results/results
PY=$(command -v python || command -v python3)
$PY isaac/horizon_law.py $R/curric.jsonl $R/datasize.jsonl $R/fig3.jsonl $R/gap_refine.jsonl $R/longbudget.jsonl \
  $R/models.jsonl $R/scaffold.jsonl $R/tier.jsonl $R/taskA.jsonl $R/taskA_short.jsonl $R/oracle.jsonl $R/pI.jsonl \
  $R/pI_refine.jsonl $R/pI_scaffold.jsonl $R/pI_sysid.jsonl $R/fullhist.jsonl $R/px.jsonl $R/sysK.jsonl $R/mik.jsonl $R/oracleK.jsonl
