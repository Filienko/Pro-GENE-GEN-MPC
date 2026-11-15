#!/bin/bash
# Epsilon sweep, centralized: LowRankDP (v2 defaults, no public data) vs Private-PGM star, 5 CAMDA folds,
# BRCA + COMBINED, eps in EPS.  Generation -> eval_suite -> MAMA-MIA (only if the red-team repo is present)
# -> one PNG per metric.  Resumable: every step is skipped when its output exists.
#   (nohup setsid ./lrdp_epsweep.sh > $OUT/EPSWEEP/run.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
EPS=${EPS:-"0.25 0.5 1 5 10"}
L=$OUT/EPSWEEP; R=$L/redteam; export OMP_NUM_THREADS=6; mkdir -p $L/BRCA $L/COMB $R
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
MAMA=$HOME/privacy/SyntheticData_MIA_PGM; REF=$MAMA/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv
for e in $EPS; do for D in BRCA COMB; do
  [ $D = BRCA ] && { A=$BB; SP=$BRCA_SPLITS; } || { A=$CC; SP=$COMB_SPLITS; }
  [ -f $L/$D/lrdp_eps$e/config.json ] || python run_lrdp.py $A --splits_yaml $SP --dp_epsilon $e --out_dir $L/$D/lrdp_eps$e > $L/$D/lrdp_$e.log 2>&1 &
  [ -f $L/$D/pgm_eps$e/config.json ] || (cd ../StructPGM; python run_structpgm.py --structure star --epsilon $e $A --splits_yaml $SP --out_dir $L/$D/pgm_eps$e > $L/$D/pgm_$e.log 2>&1) &
  while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 1; done
done; done; wait; echo GEN_DONE
for D in BRCA COMB; do
  [ -s $L/$D/eval.csv ] || ./evalpar.sh $D $L/$D/eval.csv "1 2 3 4 5" $L/$D/*_eps*/ > $L/$D/eval.log 2>&1
  echo EVAL_${D}_DONE
done
if [ -f $MAMA/competition_mia.py ]; then
  for d in $L/BRCA/*_eps*/ $L/COMB/*_eps*/; do n=$(basename $d); D=$(basename $(dirname $d))
    [ $D = BRCA ] && X="$BB" || X="$CC --folds 1 2 3 --reference $REF"
    [ -f $R/simple_${D}_$n.csv ] || python redteam_attacks.py simple $X --mama_bins 10 --target_dir $d --work $R/tmp --out_csv $R/simple_${D}_$n.csv > $R/simple_${D}_$n.log 2>&1 &
    while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 2; done
  done; wait; echo MIA_DONE
else
  echo "MIA_SKIPPED: $MAMA not found (rerun this script once it is back; finished steps are skipped)"
fi
python plot_epsweep.py $L
echo EPSWEEP_DONE
