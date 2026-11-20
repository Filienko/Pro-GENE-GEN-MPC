#!/bin/bash
# Follow-up to copula_eval.sh (whose MeLoMIA-ND step ran without the attack repo's environment): waits for it to end,
# then MeLoMIA-ND on GPU for the BRCA copula runs, the fused attack, and every table.  Resumable.
#   (nohup setsid ./copula_finish.sh >> $OUT/copula_eval.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=8
T=$OUT/TABPFN; V=$HOME/privacy/mia_verify; MIAPY=$HOME/miniconda3/envs/camda-mia/bin/python; MIA=$HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026
RUNS="lrdpc_eps10 lrdpc_eps30 lrdpc_eps100 lrdpc_eps1000"
say() { echo "$(date +%F_%T) [finish] $*"; }
until grep -q COPULA_DONE $OUT/copula_eval.log && ! ps -u $USER -o args= | grep -q "[c]opula_eval.sh"; do sleep 30; done
source $V/env.sh
for r in $RUNS; do
  ls -d $V/results/runs/BRCA__melomia_nd__ours@run=${r}__s5__* > /dev/null 2>&1 && continue
  while :; do G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 16000 {print $1; exit}'); [ -n "$G" ] && break; sleep 120; done
  ( cd $MIA && CUDA_VISIBLE_DEVICES=$G $MIAPY scripts/run_experiment.py $V/configs/verify_brca.yaml --only melomia_nd --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_melomia.log 2>&1 )
  say "MeLoMIA $r on GPU $G:"; grep MEAN $V/attack_${r}_melomia.log | sed 's/^/    /'
done
rm -f $T/attack_lr/fused_lrdpc.csv; python fused_attack.py --runs $RUNS --out_csv $T/attack_lr/fused_lrdpc.csv > /dev/null 2>&1
python tabpfn_report.py $T > $T/report_screen.log 2>&1
python brca_compare_table.py > /dev/null 2>&1; python adult_table.py > /dev/null 2>&1; python rank_table.py > /dev/null 2>&1
say COPULA_FINISHED
