#!/bin/bash
# Ablation: from a naive DP Gaussian to LowRankDP, one component at a time (private-only, exact GDP
# accounting, 5 folds, eps 10 and 30).  Resumable: finished steps are skipped.
#   nohup setsid ./lrdp_ablation.sh > $OUT/LRDP/ablation_run.log 2>&1 < /dev/null &
# Result: $OUT/LRDP/ablation/summary.txt
source $(dirname $0)/env.sh; cd $(dirname $0)
L=$OUT/LRDP/ablation; R=$L/redteam; export OMP_NUM_THREADS=4; mkdir -p $L/BRCA $L/COMB $R
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
declare -A V
V[a0_textbook]="run_dp_gauss.py --mode textbook --cov class"      # public worst-case bounds, full covariance per class
V[a1_adapt_class]="run_dp_gauss.py --mode adaptive --cov class"   # + private standardisation and clipping threshold
V[a2_adapt_pooled]="run_dp_gauss.py --mode adaptive --cov pooled" # + one shared full covariance
V[a3_lowrank]="run_lrdp.py --ablate eb covshrink"                 # + low-rank covariance and per-gene spread, no shrinkage
V[a4_lowrank_eb]="run_lrdp.py --ablate covshrink"                 # + shrinkage of class shifts
V[a5_lowrank_cov]="run_lrdp.py --ablate eb"                       # (shrinkage of spread / covariance only)
V[a6_full]="run_lrdp.py"                                          # LowRankDP v2
for v in "${!V[@]}"; do for e in 10 30; do
  [ -f $L/BRCA/${v}_eps$e/config.json ] || python ${V[$v]} $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --out_dir $L/BRCA/${v}_eps$e > $L/BRCA/${v}_$e.log 2>&1 &
  [ -f $L/COMB/${v}_eps$e/config.json ] || python ${V[$v]} $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --out_dir $L/COMB/${v}_eps$e > $L/COMB/${v}_$e.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 10 ]; do sleep 1; done
done; done; wait; echo GEN_DONE
( for d in $L/BRCA/*_eps*/; do n=$(basename $d)
    [ -f $R/simple_BRCA_$n.csv ] || python redteam_attacks.py simple $BB --mama_bins 10 --target_dir $d --work $R/tmp --out_csv $R/simple_BRCA_$n.csv > $R/simple_BRCA_$n.log 2>&1 &
    while [ $(jobs -r | wc -l) -ge 5 ]; do sleep 2; done
  done; wait; echo MIA_DONE ) &
[ -s $L/BRCA/eval.csv ] || ./evalpar.sh BRCA $L/BRCA/eval.csv "1 2 3 4 5" $L/BRCA/*_eps*/ > $L/BRCA/eval.log 2>&1
echo EVAL_BRCA_DONE
[ -s $L/COMB/eval.csv ] || ./evalpar.sh COMB $L/COMB/eval.csv "1 2 3 4 5" $L/COMB/*_eps*/ > $L/COMB/eval.log 2>&1
echo EVAL_COMB_DONE
wait
python - > $L/summary.txt 2>&1 <<PY
import pandas as pd, glob, os
pd.set_option('display.width', 250)
c = ['w1','w1_cls','mean_err','sd_err','corr_mae','wcorr_r','coex0.3_tpr','coex0.3_fer','de_tpr','de_fpr','tstr_acc','tstr_f1','disc_auc']
for D in ('BRCA', 'COMB'):
    d = pd.read_csv(f"$L/{D}/eval.csv"); d['eps'] = d.run.str.extract(r'eps(\d+)').astype(int); d['v'] = d.run.str.replace(r'_eps\d+', '', regex=True)
    t = d.groupby(['eps', 'v'])[c].mean().round(3)
    if D == 'BRCA':
        m = {}
        for f in glob.glob("$R/simple_BRCA_*.csv"):
            x = pd.read_csv(f); n = os.path.basename(f)[12:-4]
            m[(int(n.split('eps')[1]), n.split('_eps')[0])] = x[x.attack.str.startswith('MAMA-MIA 1+2')].auc.mean()
        t['mama'] = pd.Series(m).round(3)
    print(D); print(t.to_string())
PY
echo ABLATION_DONE
