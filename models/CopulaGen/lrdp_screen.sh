#!/bin/bash
# Screening of LowRankDP improvements (private-only, folds 1-3): exact GDP accounting, between-class basis,
# budget shares, winsorising / clip quantile, marginal reshaping.  Resumable: finished steps are skipped.
#   nohup setsid ./lrdp_screen.sh > $OUT/LRDP/screen_run.log 2>&1 < /dev/null &          (round 1)
#   SCREEN=screen2 nohup setsid ./lrdp_screen.sh > $OUT/LRDP/screen2_run.log 2>&1 < /dev/null &   (round 2)
source $(dirname $0)/env.sh; cd $(dirname $0)
SCREEN=${SCREEN:-screen}; L=$OUT/LRDP/$SCREEN; R=$L/redteam; export OMP_NUM_THREADS=4; mkdir -p $L/BRCA $L/COMB $R
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
FOLDS="1 2 3"
declare -A V
if [ $SCREEN = screen ]; then      # round 1: one change at a time
# (winsorising and marginal reshaping were also screened here and rejected; their code was removed after commit a3122af)
V[base]=""
V[g]="--accounting gdp"
V[gb]="--accounting gdp --between"
V[gbs1]="--accounting gdp --between --shares {\"mean\":0.50,\"ccov\":0.10}"
V[gbs2]="--accounting gdp --between --shares {\"mean\":0.45,\"basis\":0.20,\"ccov\":0.10}"
V[gbq75]="--accounting gdp --between --clip_q 0.75"
V[gbq97]="--accounting gdp --between --clip_q 0.97"
else                               # round 2: combinations around the round-1 winners
V[gq75]="--accounting gdp --clip_q 0.75"
V[gbq60]="--accounting gdp --between --clip_q 0.60"
V[gbq80]="--accounting gdp --between --clip_q 0.80"
V[gbq75s1]="--accounting gdp --between --clip_q 0.75 --shares {\"mean\":0.50,\"ccov\":0.10}"
V[gbq75s2]="--accounting gdp --between --clip_q 0.75 --shares {\"mean\":0.45,\"basis\":0.20,\"ccov\":0.10}"
V[gq75s2]="--accounting gdp --clip_q 0.75 --shares {\"mean\":0.45,\"basis\":0.20,\"ccov\":0.10}"
fi
for v in "${!V[@]}"; do for e in 10 30; do
  [ -f $L/BRCA/${v}_eps$e/config.json ] || python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --folds $FOLDS ${V[$v]} --out_dir $L/BRCA/${v}_eps$e > $L/BRCA/${v}_$e.log 2>&1 &
  [ -f $L/COMB/${v}_eps$e/config.json ] || python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --folds $FOLDS ${V[$v]} --out_dir $L/COMB/${v}_eps$e > $L/COMB/${v}_$e.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 10 ]; do sleep 1; done
done; done; wait; echo GEN_DONE
# MAMA-MIA / MahalaMIA on BRCA for every variant (background, 6 at a time)
( for d in $L/BRCA/*_eps*/; do n=$(basename $d)
    [ -f $R/simple_BRCA_$n.csv ] || python redteam_attacks.py simple $BB --mama_bins 10 --folds $FOLDS --target_dir $d --work $R/tmp --out_csv $R/simple_BRCA_$n.csv > $R/simple_BRCA_$n.log 2>&1 &
    while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 2; done
  done; wait; echo MIA_DONE ) &
[ -s $L/BRCA/eval.csv ] || ./evalpar.sh BRCA $L/BRCA/eval.csv "$FOLDS" $L/BRCA/*_eps*/ > $L/BRCA/eval.log 2>&1
echo EVAL_BRCA_DONE
[ -s $L/COMB/eval.csv ] || ./evalpar.sh COMB $L/COMB/eval.csv "$FOLDS" $L/COMB/*_eps*/ > $L/COMB/eval.log 2>&1
echo EVAL_COMB_DONE
wait
python - > $L/summary.txt 2>&1 <<PY
import pandas as pd, glob, os
pd.set_option('display.width', 250)
c = ['w1','w1_cls','corr_mae','wcorr_r','coex0.3_tpr','coex0.3_fer','de_tpr','de_fpr','tstr_acc','tstr_f1','disc_auc','marg_auc']
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
echo SCREEN_DONE
