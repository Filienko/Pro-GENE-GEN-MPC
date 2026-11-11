#!/bin/bash
source $(dirname $0)/env.sh; cd $(dirname $0)
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --folds 1 2"
D=$OUT/BRCA_s2; mkdir -p $D
run(){ n=$1; shift; OMP_NUM_THREADS=4 python run_copulagen.py $B --tail 0.02 "$@" --out_dir $D/$n > $D/$n.gen.log 2>&1; }
run ss03 --knots 41 --shape_shrink 0.3 &
run ss06 --knots 41 --shape_shrink 0.6 &
run ss08 --knots 41 --shape_shrink 0.8 &
run n0_300 --knots 41 --n0 300 &
run kn11 --knots 11 &
run kn11_ss05 --knots 11 --shape_shrink 0.5 &
run ss06_k30 --knots 41 --shape_shrink 0.6 --k 30 &
run ss06_k40_sh5 --knots 41 --shape_shrink 0.6 --k 40 --shrink 0.5 &
run ss06_diff --knots 41 --shape_shrink 0.6 --factor diffusion --device cuda:0 &
run ss06_diff_k30 --knots 41 --shape_shrink 0.6 --k 30 --factor diffusion --device cuda:1 &
wait
./evalpar.sh BRCA $D/eval.csv "1 2" $D/*/ $OUT/BRCA_dp1/*/
OMP_NUM_THREADS=8 python eval_suite.py --counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --folds 1 2 --oracle --syn_dir $OUT/BRCA_s1/pgm_star --out_csv $D/oracle.csv > $D/oracle.log 2>&1
echo SWEEP_DONE
