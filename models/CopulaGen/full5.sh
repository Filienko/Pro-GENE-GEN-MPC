#!/bin/bash
# Full 5-fold runs of finalists + baselines on BRCA and COMBINED, then evaluation.
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=6
for DS in BRCA COMB; do
  if [ $DS = BRCA ]; then B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS"; else B="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type --splits_yaml $COMB_SPLITS"; fi
  D=$OUT/FULL_$DS; mkdir -p $D
  run(){ n=$1; shift; [ -f $D/$n/config.json ] || python run_copulagen.py $B --tail 0.02 --knots 41 "$@" --out_dir $D/$n > $D/$n.gen.log 2>&1; }
  run copula_gauss_k30 --k 30 --gauss_mix 0.5 &
  run copula_gauss_k50 --k 50 --gauss_mix 0.5 &
  run copula_diff_k30 --k 30 --gauss_mix 0.7 --factor diffusion --device cuda:0 &
  (cd ../FactorGen
   [ -f $D/factorgauss_k50/config.json ] || python run_factorgen.py --method gauss --k 50 $B --out_dir $D/factorgauss_k50 > $D/fg.gen.log 2>&1
   [ -f $D/factordiff_k50/config.json ] || python run_factorgen.py --method diffusion --k 50 --device cuda:1 $B --out_dir $D/factordiff_k50 > $D/fd.gen.log 2>&1
   [ -f $D/mvn_camda/config.json ] || python run_factorgen.py --method mvn_camda $B --out_dir $D/mvn_camda > $D/mvn.gen.log 2>&1) &
  (cd ../StructPGM; for e in inf 10; do [ -f $D/pgm_star_eps$e/config.json ] || python run_structpgm.py --structure star --epsilon $e $B --out_dir $D/pgm_star_eps$e > $D/pgm$e.gen.log 2>&1; done) &
  wait
  echo "GEN_DONE $DS $(date)"
done
for DS in BRCA COMB; do
  D=$OUT/FULL_$DS
  ./evalpar.sh $DS $D/eval.csv "1 2 3 4 5" $D/*/ > $D/eval.log 2>&1
  if [ $DS = BRCA ]; then C=$BRCA_TSV; A=$BRCA_ANN; L=Subtype; else C=$COMB_TSV; A=$COMB_ANN; L=cancer_type; fi
  OMP_NUM_THREADS=12 python eval_suite.py --counts $C --annot $A --label_col $L --oracle --syn_dir $D/pgm_star_epsinf --out_csv $D/oracle.csv > $D/oracle.log 2>&1
  echo "EVAL_DONE $DS $(date)"
done
echo FULL_DONE
