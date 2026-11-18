#!/bin/bash
# DP-MVN (LowRankDP v2) at eps 0.5 / 1 / 1000 (+ the MVN baseline's MeLoMIA-ND) through the comparison pipeline:
# eval_suite + MAMA-lite (CPU) -> MahalaMIA + MAMA-MIA v2 (CPU) -> MeLoMIA-ND (GPU, waits until one has room)
# -> kNN-LR / fused -> report + BRCA_compare table.  Resumable.
#   (nohup setsid ./dpmvn_eps_eval.sh >> $OUT/TABPFN/dpmvn_eps.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
T=$OUT/TABPFN; E=$OUT/EPSWEEP/BRCA; A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; V=$HOME/privacy/mia_verify
MIAPY=$HOME/miniconda3/envs/camda-mia/bin/python; MIA=$HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026
RUNS="lrdp_eps1000 lrdp_eps0.5 lrdp_eps1"
say() { echo "$(date +%F_%T) $*"; }
[ -f $E/lrdp_eps1000/config.json ] || python run_lrdp.py $A --splits_yaml $BRCA_SPLITS --dp_epsilon 1000 --v2 --out_dir $E/lrdp_eps1000 > $E/lrdp_1000.log 2>&1
for f in 1 2 3 4 5; do   # eval for eps 1000 (0.5 / 1 are in EPSWEEP/BRCA/eval.csv already)
  [ -f $T/BRCA/eval/lrdp_eps1000_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $E/lrdp_eps1000 --out_csv $T/BRCA/eval/lrdp_eps1000_f$f.csv > $T/BRCA/eval/lrdp_eps1000_f$f.log 2>&1
done
for r in lrdp_eps1000 lrdp_eps0.5; do
  [ -f $T/BRCA/mama/$r.csv ] || python mama_lite.py $A --syn_dir $E/$r --out_csv $T/BRCA/mama/$r.csv > /dev/null 2>&1
done
say "scored"
source $V/env.sh
for r in $RUNS; do
  $HOME/miniconda3/envs/camda_sota/bin/python mia_import_targets.py --dataset BRCA --run_dir $E/$r > /dev/null 2>&1 || say "IMPORT FAILED $r"
  ( cd $MIA && CUDA_VISIBLE_DEVICES="" $MIAPY scripts/run_experiment.py $V/configs/verify_brca_cpu.yaml --only mahalamia mamamia_v2 \
      --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_fast.log 2>&1 )
  say "fast attacks $r:"; grep -E "^=== |MEAN" $V/attack_${r}_fast.log | sed 's/^/    /'
done
python brca_compare_table.py > /dev/null 2>&1; say "table (without MeLoMIA-ND for the new runs)"
for r in mvn $RUNS; do   # MeLoMIA-ND: needs a GPU (hours per fold on CPU); wait for one with >= 16 GB free
  if ls -d $V/results/runs/BRCA__melomia_nd__ours@run=${r}__s5__* > /dev/null 2>&1; then continue; fi
  while :; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 16000 {print $1; exit}')
    [ -n "$G" ] && break; sleep 300
  done
  say "MeLoMIA-ND $r on GPU $G"
  ( cd $MIA && CUDA_VISIBLE_DEVICES=$G $MIAPY scripts/run_experiment.py $V/configs/verify_brca.yaml --only melomia_nd \
      --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_melomia.log 2>&1 )
  grep -E "MEAN" $V/attack_${r}_melomia.log | sed "s/^/    $r /"
done
python fused_attack.py --runs mvn $RUNS --out_csv $T/attack_lr/fused_dpmvn_eps.csv > /dev/null 2>&1
python tabpfn_report.py $T > $T/report_screen.log 2>&1
python brca_compare_table.py > $T/compare_table.log 2>&1
say DPMVN_EPS_DONE
