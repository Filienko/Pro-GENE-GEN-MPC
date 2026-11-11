#!/bin/bash
source $(dirname $0)/env.sh; cd $(dirname $0)
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS --folds 1 2"
D=$OUT/BRCA_s1; mkdir -p $D
run(){ n=$1; shift; python run_copulagen.py $B "$@" --out_dir $D/$n > $D/$n.gen.log 2>&1; }
run q21t02 --knots 21 --tail 0.02 &
run q41t02 --knots 41 --tail 0.02 &
run q41t05 --knots 41 --tail 0.05 &
run q81t01 --knots 81 --tail 0.01 &
run q41t02_gmm --knots 41 --tail 0.02 --factor gmm &
run q41t02_kde1 --knots 41 --tail 0.02 --factor kde --kde_h 1.0 &
run q41t02_k100 --knots 41 --tail 0.02 --k 100 &
run q41t02_k30 --knots 41 --tail 0.02 --k 30 &
run q41t02_diff --knots 41 --tail 0.02 --factor diffusion --device cuda:0 &
run g_gauss --marginal gauss &
wait
(cd ../FactorGen; for m in gauss diffusion; do python run_factorgen.py --method $m --k 50 --device cuda:1 $B --out_dir $D/fg_$m > $D/fg_$m.gen.log 2>&1; done)
python ../StructPGM/run_structpgm.py --structure star --epsilon inf $B --out_dir $D/pgm_star > $D/pgm.gen.log 2>&1
./evalpar.sh BRCA $D/eval.csv "1 2" $D/*/
echo SWEEP_DONE
