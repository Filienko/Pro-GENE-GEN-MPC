#!/bin/bash
# DP-MVN with gene-screened DP-PCA (--pca_genes m), BRCA 5 folds: generate -> eval_suite -> MAMA-lite.  Resumable.
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
L=$OUT/LRDP/pcag/BRCA; A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; mkdir -p $L/eval $L/mama
for spec in ${SPECS:-"30 100" "30 150" "30 300" "10 150"}; do set -- $spec; e=$1; m=$2; R=pcag_eps${e}_m$m
  [ -f $L/$R/config.json ] || python run_lrdp.py $A --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v2 --pca_genes $m --out_dir $L/$R > $L/$R.log 2>&1
  for f in 1 2 3 4 5; do
    [ -f $L/eval/${R}_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $L/$R --out_csv $L/eval/${R}_f$f.csv > $L/eval/${R}_f$f.log 2>&1 &
  done; wait
  [ -f $L/mama/$R.csv ] || python mama_lite.py $A --syn_dir $L/$R --out_csv $L/mama/$R.csv > /dev/null 2>&1
  echo "$(date +%F_%T) done $R"
done; echo PCAG_DONE
