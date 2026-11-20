#!/bin/bash
# One TabPFN job on the Bioresponse subset (run via gpu_worker.sh, which sets CUDA_VISIBLE_DEVICES), then scoring + table.
#   ./bio_gpu.sh RUN "FOLDS" [run_tabpfn.py args]
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=6
R=$1; FOLDS=$2; shift 2
B=$OUT/BIO; A="--counts $BIO_TSV --annot $BIO_ANN --label_col Subtype"; PY=$HOME/miniconda3/envs/camda_sota/bin/python
CAMDA_SRC=$HOME/privacy/PPML-Huskies-Red-Team/src
for f in $FOLDS; do
  [ -f $B/$R/synthetic_data_split_$f.csv ] || $PY run_tabpfn.py $A --splits_yaml $BIO_SPLITS --folds $f --out_dir $B/$R "$@" > $B/${R}_f$f.log 2>&1 || echo "FAILED $R fold $f"
  [ -f $B/eval/${R}_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $B/$R --out_csv $B/eval/${R}_f$f.csv > $B/eval/${R}_f$f.log 2>&1
  [ -f $B/kl/${R}_f$f.csv ] || python camda_kl_overlap.py $A --camda_src $CAMDA_SRC --syn_dir $B/$R --folds $f --out_csv $B/kl/${R}_f$f.csv > /dev/null 2>&1
  [ -f $B/mama/${R}_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $B/$R --out_csv $B/mama/${R}_f$f.csv > /dev/null 2>&1
  [ -f $B/knn/knn_${R}_${R}_f${f}_scores.csv ] || OMP_NUM_THREADS=16 python knn_lr_attack.py $A --syn_dir $B/$R --folds $f --k 40 --n_feat 150 --same_class 0 --out_csv $B/knn/knn_$R.csv > /dev/null 2>&1
done
python bio_table.py BIO > /dev/null 2>&1; echo "$(date +%F_%T) BIO $R folds $FOLDS done"
