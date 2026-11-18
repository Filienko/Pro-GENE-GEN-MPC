#!/bin/bash
# DP-MVN at further eps (EPSL) through the comparison pipeline: generate -> eval_suite + MAMA-lite -> MahalaMIA +
# MAMA-MIA v2 (CPU) -> table -> MeLoMIA-ND (GPU, waits for room) -> kNN-LR / fused -> report + table.  Resumable.
#   EPSL="100 200 300" (nohup setsid ./dpmvn_eps2.sh >> $OUT/TABPFN/dpmvn_eps2.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
T=$OUT/TABPFN; E=$OUT/EPSWEEP/BRCA; A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; V=$HOME/privacy/mia_verify
MIAPY=$HOME/miniconda3/envs/camda-mia/bin/python; MIA=$HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026
EPSL=${EPSL:-"100 200 300"}; RUNS=$(for e in $EPSL; do printf "lrdp_eps%s " $e; done)
say() { echo "$(date +%F_%T) $*"; }
for e in $EPSL; do r=lrdp_eps$e
  [ -f $E/$r/config.json ] || python run_lrdp.py $A --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v2 --out_dir $E/$r > $E/lrdp_$e.log 2>&1
  for f in 1 2 3 4 5; do
    [ -f $T/BRCA/eval/${r}_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $E/$r --out_csv $T/BRCA/eval/${r}_f$f.csv > $T/BRCA/eval/${r}_f$f.log 2>&1
  done
  [ -f $T/BRCA/mama/$r.csv ] || python mama_lite.py $A --syn_dir $E/$r --out_csv $T/BRCA/mama/$r.csv > /dev/null 2>&1
done
say "scored $RUNS"
source $V/env.sh
for r in $RUNS; do
  $HOME/miniconda3/envs/camda_sota/bin/python mia_import_targets.py --dataset BRCA --run_dir $E/$r > /dev/null 2>&1 || say "IMPORT FAILED $r"
  ( cd $MIA && CUDA_VISIBLE_DEVICES="" $MIAPY scripts/run_experiment.py $V/configs/verify_brca_cpu.yaml --only mahalamia mamamia_v2 \
      --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_fast.log 2>&1 )
  say "fast attacks $r:"; grep -E "^=== |MEAN" $V/attack_${r}_fast.log | sed 's/^/    /'
done
python tabpfn_report.py $T > $T/report_screen.log 2>&1; python brca_compare_table.py > /dev/null 2>&1; say "table (without MeLoMIA-ND)"
for r in $RUNS; do
  if ls -d $V/results/runs/BRCA__melomia_nd__ours@run=${r}__s5__* > /dev/null 2>&1; then continue; fi
  while :; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 16000 {print $1; exit}')
    [ -n "$G" ] && ! ps -eo args | grep -q "[m]elomia_nd\|[r]un_experiment.py.*verify_brca.yaml" && break; sleep 300
  done
  say "MeLoMIA-ND $r on GPU $G"
  ( cd $MIA && CUDA_VISIBLE_DEVICES=$G $MIAPY scripts/run_experiment.py $V/configs/verify_brca.yaml --only melomia_nd \
      --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_melomia.log 2>&1 )
  grep -E "MEAN" $V/attack_${r}_melomia.log | sed "s/^/    $r /"
done
python fused_attack.py --runs $RUNS --out_csv $T/attack_lr/fused_dpmvn_eps2.csv > /dev/null 2>&1
python tabpfn_report.py $T > $T/report_screen.log 2>&1; python brca_compare_table.py > $T/compare_table.log 2>&1
say DPMVN_EPS2_DONE
