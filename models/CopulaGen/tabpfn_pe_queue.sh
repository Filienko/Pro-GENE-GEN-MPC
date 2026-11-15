#!/bin/bash
# TabPFN-PE (run_tabpfn_pe.py) on BRCA, on CPU, in parallel with the GPU queue (tabpfn_compare.sh).  No public data.
#   tabpfn_pe10      LowRankDP eps 8  start + noisy votes -> eps 10 total
#   tabpfn_pe30      LowRankDP eps 20 start + noisy votes -> eps 30 total
#   tabpfn_pe_sig0   LowRankDP eps 8  start + NOISELESS votes: non-private ceiling of the method, not DP
# Each fold is scored like the other runs (eval_suite + MAMA-MIA + report).  Resumable.
#   (nohup setsid ./tabpfn_pe_queue.sh >> $OUT/TABPFN/pe_run.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
T=$OUT/TABPFN; PY=$HOME/miniconda3/envs/camda_sota/bin/python; L=$OUT/EPSWEEP/BRCA
A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; PE="--T 10 --r0 0.3 --rT 0.05 --device cpu"
score() {  # $1 run, $2 fold
  [ -f $T/BRCA/eval/$1_f$2.csv ] || OMP_NUM_THREADS=6 python eval_suite.py $A --folds $2 --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/eval/$1_f$2.csv > $T/BRCA/eval/$1_f$2.log 2>&1
  [ -f $T/BRCA/mama/$1_f$2.csv ] || python mama_lite.py $A --folds $2 --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/mama/$1_f$2.csv > /dev/null 2>&1
  python tabpfn_report.py $T > $T/report_pe.log 2>&1
}
run() {  # $1 run name, rest = run_tabpfn_pe.py args; all folds, one config per background lane
  R=$1; shift
  for f in 1 2 3 4 5; do
    if [ ! -f $T/BRCA/$R/synthetic_data_split_$f.csv ]; then
      echo "$(date +%F_%T) START $R fold $f"
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 $PY run_tabpfn_pe.py $A --splits_yaml $BRCA_SPLITS --folds $f --out_dir $T/BRCA/$R $PE "$@" \
        > $T/BRCA/${R}_f$f.log 2>&1 || { echo "$(date +%F_%T) FAILED $R fold $f"; continue; }
    fi
    score $R $f; echo "$(date +%F_%T) DONE $R fold $f"
  done
}
run tabpfn_pe10 --init_dir $L/lrdp_eps8 --init_eps 8 --total_eps 10 &
run tabpfn_pe30 --init_dir $L/lrdp_eps20 --init_eps 20 --total_eps 30 &
run tabpfn_pe_sig0 --init_dir $L/lrdp_eps8 --init_eps 8 --total_eps 10 --sigma0 &
wait; echo "$(date +%F_%T) PE_DONE"
