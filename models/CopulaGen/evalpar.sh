#!/bin/bash
# usage: evalpar.sh <dataset BRCA|COMB> <out_csv> <folds "1 2"> dir1 dir2 ...   (one process per dir, 6 BLAS threads each)
source $(dirname $0)/env.sh
DS=$1; OCSV=$2; FOLDS=$3; shift 3
if [ $DS = BRCA ]; then C=$BRCA_TSV; A=$BRCA_ANN; L=Subtype; else C=$COMB_TSV; A=$COMB_ANN; L=cancer_type; fi
tmp=$(mktemp -d)
for d in "$@"; do
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 python $(dirname $0)/eval_suite.py --counts $C --annot $A --label_col $L --folds $FOLDS --syn_dir $d --out_csv $tmp/$(basename $d).csv > $tmp/$(basename $d).log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 7 ]; do sleep 2; done
done
wait
grep -h " f[0-9] " $tmp/*.log
python - $OCSV $tmp <<'PY'
import sys,glob,pandas as pd
d=pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[2]+'/*.csv')])
d.to_csv(sys.argv[1],index=False)
c=['w1','w1_cls','disc_auc','prec','rec','dens','cov','corr_mae','wcorr_r','coex0.3_tpr','coex0.3_fer','de_tpr','de_fpr','de_rho','tstr_acc','tstr_f1','mahala_auc','dcr_auc','dcrr_auc','domias_auc','marg_auc','dcr_share']
pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(d.groupby('run')[c].mean().round(3).to_string())
PY
