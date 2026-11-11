#!/bin/bash
# LowRankDP --between variants: generation, fidelity, simple red-team attacks.  Resumable: every step is
# skipped when its output exists, so rerunning after an interruption continues where it stopped.
#   nohup setsid ./lrdp_between.sh > $OUT/LRDP/between_run.log 2>&1 < /dev/null &
source $(dirname $0)/env.sh; cd $(dirname $0)
REF=$HOME/privacy/SyntheticData_MIA_PGM/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv
L=$OUT/LRDP; R=$L/redteam; export OMP_NUM_THREADS=6; mkdir -p $R
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
for e in 10 30; do for v in btw btw35; do
  [ $v = btw35 ] && X="--basis_share 0.35" || X=""
  [ -f $L/BRCA/lrdp_${v}_eps$e/config.json ] || python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --between $X --out_dir $L/BRCA/lrdp_${v}_eps$e > $L/BRCA/${v}_$e.log 2>&1 &
  [ -f $L/COMB/lrdp_${v}_eps$e/config.json ] || python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --between $X --out_dir $L/COMB/lrdp_${v}_eps$e > $L/COMB/${v}_$e.log 2>&1 &
done; done; wait; echo GEN_DONE
for v in btw btw35; do for e in 10 30; do
  [ -f $R/simple_BRCA_${v}_eps$e.csv ] || python redteam_attacks.py simple $BB --mama_bins 10 --target_dir $L/BRCA/lrdp_${v}_eps$e --work $R/tmp --out_csv $R/simple_BRCA_${v}_eps$e.csv > $R/simple_BRCA_${v}_eps$e.log 2>&1 &
  [ -f $R/simple_COMB_${v}_eps$e.csv ] || python redteam_attacks.py simple $CC --mama_bins 10 --folds 1 2 3 --reference $REF --target_dir $L/COMB/lrdp_${v}_eps$e --work $R/tmp --out_csv $R/simple_COMB_${v}_eps$e.csv > $R/simple_COMB_${v}_eps$e.log 2>&1 &
done; done
[ -s $L/BRCA/eval_btw.csv ] || ./evalpar.sh BRCA $L/BRCA/eval_btw.csv "1 2 3 4 5" $L/BRCA/lrdp_btw*/ > $L/BRCA/eval_btw.log 2>&1
echo EVAL_BRCA_DONE
[ -s $L/COMB/eval_btw.csv ] || ./evalpar.sh COMB $L/COMB/eval_btw.csv "1 2 3 4 5" $L/COMB/lrdp_btw*/ > $L/COMB/eval_btw.log 2>&1
echo EVAL_COMB_DONE
wait
python - > $L/between_summary.txt 2>&1 <<PY
import pandas as pd, glob, os
pd.set_option('display.width', 250)
c = ['w1','w1_cls','corr_mae','wcorr_r','coex0.3_tpr','coex0.3_fer','de_tpr','de_fpr','tstr_acc','tstr_f1','disc_auc','marg_auc','mahala_auc']
for D in ('BRCA', 'COMB'):
    d = pd.concat([pd.read_csv(f"$L/{D}/eval.csv"), pd.read_csv(f"$L/{D}/eval_btw.csv")])
    print(D); print(d.groupby('run')[c].mean().round(3).to_string())
for f in sorted(glob.glob("$R/simple_*_eps*.log")):
    t = [l.split() for l in open(f) if l.startswith(("MahalaMIA", "MAMA-MIA 1+2"))]
    print(os.path.basename(f)[7:-4].ljust(22), "  ".join(f"{' '.join(x[:-6])} {x[-6]}" for x in t))
PY
echo BETWEEN_DONE
