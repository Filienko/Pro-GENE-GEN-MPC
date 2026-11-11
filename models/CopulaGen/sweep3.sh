#!/bin/bash
source $(dirname $0)/env.sh; cd $(dirname $0)
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --folds 1 2"
D=$OUT/BRCA_s3; mkdir -p $D
run(){ n=$1; shift; OMP_NUM_THREADS=4 python run_copulagen.py $B --tail 0.02 --knots 41 "$@" --out_dir $D/$n > $D/$n.gen.log 2>&1; }
for g in 0.3 0.5 0.7; do run gm${g}_k30 --gauss_mix $g --k 30 & done
for g in 0.5 0.7; do run gm${g}_k50 --gauss_mix $g & done
run gm0.5_k30_diff --gauss_mix 0.5 --k 30 --factor diffusion --device cuda:0 &
run gm0.7_k30_diff --gauss_mix 0.7 --k 30 --factor diffusion --device cuda:1 &
run g_k30 --marginal gauss --k 30 &
run g_k30_diff --marginal gauss --k 30 --factor diffusion --device cuda:1 &
wait
./evalpar.sh BRCA $D/eval.csv "1 2" $D/*/ $OUT/BRCA/pgm_star_eps10 $OUT/BRCA/pgm_star_eps1
echo SWEEP_DONE
