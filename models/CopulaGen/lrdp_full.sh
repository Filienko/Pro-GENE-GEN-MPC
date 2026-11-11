#!/bin/bash
# Full 5-fold LowRankDP runs (private-only and public-anchored, eps 10 / 30) + fidelity + red-team attacks.
source $(dirname $0)/env.sh; cd $(dirname $0)
REF=$HOME/privacy/SyntheticData_MIA_PGM/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv; L=$OUT/LRDP; R=$L/redteam; export OMP_NUM_THREADS=6
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
for e in 10 30; do
  python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v1 --out_dir $L/BRCA/lrdp_priv_eps$e > $L/BRCA/priv$e.log 2>&1 &
  python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v1 --external_counts $REF --out_dir $L/BRCA/lrdp_pub_eps$e > $L/BRCA/pub$e.log 2>&1 &
  python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --v1 --out_dir $L/COMB/lrdp_priv_eps$e > $L/COMB/priv$e.log 2>&1 &
  python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --v1 --external_counts $REF --out_dir $L/COMB/lrdp_pub_eps$e > $L/COMB/pub$e.log 2>&1 &
done; wait; echo GEN_DONE
# red-team simple attacks (MAMA-MIA 10 bins + MahalaMIA); COMBINED on folds 1-3 (slow python loop)
for v in priv pub; do for e in 10 30; do
  python redteam_attacks.py simple $BB --mama_bins 10 --target_dir $L/BRCA/lrdp_${v}_eps$e --work $R/tmp --out_csv $R/simple_BRCA_${v}_eps$e.csv > $R/simple_BRCA_${v}_eps$e.log 2>&1 &
  python redteam_attacks.py simple $CC --mama_bins 10 --folds 1 2 3 --reference $REF --target_dir $L/COMB/lrdp_${v}_eps$e --work $R/tmp --out_csv $R/simple_COMB_${v}_eps$e.csv > $R/simple_COMB_${v}_eps$e.log 2>&1 &
done; done
# MeLoMIA with synth-shadows against the eps=30 private-only variant on BRCA
(python redteam_attacks.py base $BB --gen lrdp --gen_eps 30 --K 15 --target_dir $L/BRCA/lrdp_priv_eps30 --work $R/w_BRCA_priv30 > $R/base_BRCA_priv30.log 2>&1
 python redteam_attacks.py melomia $BB --gen lrdp --K 15 --target_dir $L/BRCA/lrdp_priv_eps30 --work $R/w_BRCA_priv30 --device cuda:0 --out_csv $R/melomia_BRCA_priv30.csv > $R/melomia_BRCA_priv30.log 2>&1) &
./evalpar.sh BRCA $L/BRCA/eval.csv "1 2 3 4 5" $L/BRCA/*/ > $L/BRCA/eval.log 2>&1
./evalpar.sh COMB $L/COMB/eval.csv "1 2 3 4 5" $L/COMB/*/ > $L/COMB/eval.log 2>&1
for d in $L/BRCA/lrdp_*/; do python camda_kl_overlap.py $BB --syn_dir $d --out_csv $L/BRCA/kl_$(basename $d).csv > /dev/null 2>&1 & done
for d in $L/COMB/lrdp_*/; do python camda_kl_overlap.py $CC --syn_dir $d --out_csv $L/COMB/kl_$(basename $d).csv > /dev/null 2>&1 & done
wait; echo LRDP_DONE
