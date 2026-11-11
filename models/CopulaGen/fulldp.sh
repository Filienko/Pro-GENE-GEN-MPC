#!/bin/bash
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=4
for DS in BRCA COMB; do
  if [ $DS = BRCA ]; then B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype --splits_yaml $BRCA_SPLITS"; else B="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type --splits_yaml $COMB_SPLITS"; fi
  D=$OUT/FULLDP_$DS; mkdir -p $D
  for e in 10 30; do
    python run_copulagen.py $B --dp_epsilon $e --bins 6 --k 20 --dp_split 0.02 0.68 0.30 --out_dir $D/dpcopula_eps$e > $D/dpc$e.log 2>&1 &
    (cd ../StructPGM; python run_structpgm.py --structure star --epsilon $e $B --out_dir $D/pgm_star_eps$e > $D/pgm$e.log 2>&1) &
  done
  wait
done
until grep -q FULL_DONE $OUT/logs/full5.log; do sleep 20; done
for DS in BRCA COMB; do ./evalpar.sh $DS $OUT/FULLDP_$DS/eval.csv "1 2 3 4 5" $OUT/FULLDP_$DS/*/ > $OUT/FULLDP_$DS/eval.log 2>&1; done
echo DP_DONE
