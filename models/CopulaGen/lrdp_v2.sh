#!/bin/bash
# LowRankDP v2 preset (exact GDP accounting, clip quantile 0.75, re-split budget), with and without --between:
# 5-fold generation, fidelity, KL / overlap, simple red-team attacks, model-aware shadow attack.
# Resumable: finished steps are skipped.
#   nohup setsid ./lrdp_v2.sh > $OUT/LRDP/v2_run.log 2>&1 < /dev/null &
source $(dirname $0)/env.sh; cd $(dirname $0)
REF=$HOME/privacy/SyntheticData_MIA_PGM/RED_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv
L=$OUT/LRDP/v2; R=$L/redteam; export OMP_NUM_THREADS=6; mkdir -p $L/BRCA $L/COMB $R
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
KW='{"accounting":"gdp","clip_q":0.75,"between":true,"shares":{"n":0.02,"clip":0.03,"anchor":0.08,"mean":0.45,"var":0.12,"basis":0.20,"ccov":0.10}}'
for e in 10 30; do
  [ -f $L/BRCA/v2_priv_eps$e/config.json ] || python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v2 --between --out_dir $L/BRCA/v2_priv_eps$e > $L/BRCA/priv$e.log 2>&1 &
  [ -f $L/BRCA/v2_pub_eps$e/config.json ]  || python run_lrdp.py $BB --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v2 --external_counts $REF --out_dir $L/BRCA/v2_pub_eps$e > $L/BRCA/pub$e.log 2>&1 &
  [ -f $L/COMB/v2_priv_eps$e/config.json ] || python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --v2 --between --out_dir $L/COMB/v2_priv_eps$e > $L/COMB/priv$e.log 2>&1 &
  [ -f $L/COMB/v2_pub_eps$e/config.json ]  || python run_lrdp.py $CC --splits_yaml $COMB_SPLITS --dp_epsilon $e --v2 --external_counts $REF --out_dir $L/COMB/v2_pub_eps$e > $L/COMB/pub$e.log 2>&1 &
  # v2p_* = the --v2 preset itself (parameters only); v2_priv_* above = --v2 plus the optional --between basis
  for D in BRCA COMB; do [ $D = BRCA ] && { A=$BB; SP=$BRCA_SPLITS; } || { A=$CC; SP=$COMB_SPLITS; }
    [ -f $L/$D/v2p_priv_eps$e/config.json ] || python run_lrdp.py $A --splits_yaml $SP --dp_epsilon $e --v2 --out_dir $L/$D/v2p_priv_eps$e > $L/$D/p_priv$e.log 2>&1 &
  done
done; wait; echo GEN_DONE
( for v in priv pub p; do for e in 10 30; do T=v2_${v}_eps$e; [ $v = p ] && T=v2p_priv_eps$e
    [ -f $R/simple_BRCA_${v}_eps$e.csv ] || python redteam_attacks.py simple $BB --mama_bins 10 --target_dir $L/BRCA/$T --work $R/tmp --out_csv $R/simple_BRCA_${v}_eps$e.csv > $R/simple_BRCA_${v}_eps$e.log 2>&1 &
    [ -f $R/simple_COMB_${v}_eps$e.csv ] || python redteam_attacks.py simple $CC --mama_bins 10 --folds 1 2 3 --reference $REF --target_dir $L/COMB/$T --work $R/tmp --out_csv $R/simple_COMB_${v}_eps$e.csv > $R/simple_COMB_${v}_eps$e.log 2>&1 &
  done; done; wait; echo SIMPLE_DONE ) &
# model-aware shadow attack: shadows are LowRankDP with the same preset and eps
( for e in 10 30; do
    ( [ -f $R/aware_BRCA_priv$e.csv ] || { python redteam_attacks.py base $BB --gen lrdp --gen_eps $e --gen_kw "$KW" --K 15 --target_dir $L/BRCA/v2_priv_eps$e --work $R/w_BRCA_priv$e > $R/base_BRCA_priv$e.log 2>&1 &&
        python redteam_attacks.py aware $BB --gen lrdp --gen_eps $e --gen_kw "$KW" --K 15 --target_dir $L/BRCA/v2_priv_eps$e --work $R/w_BRCA_priv$e --out_csv $R/aware_BRCA_priv$e.csv > $R/aware_BRCA_priv$e.log 2>&1; } ) &
    ( [ -f $R/aware_COMB_priv$e.csv ] || { python redteam_attacks.py base $CC --gen lrdp --gen_eps $e --gen_kw "$KW" --K 10 --target_dir $L/COMB/v2_priv_eps$e --work $R/w_COMB_priv$e > $R/base_COMB_priv$e.log 2>&1 &&
        python redteam_attacks.py aware $CC --gen lrdp --gen_eps $e --gen_kw "$KW" --K 10 --target_dir $L/COMB/v2_priv_eps$e --work $R/w_COMB_priv$e --out_csv $R/aware_COMB_priv$e.csv > $R/aware_COMB_priv$e.log 2>&1; } ) &
  done; wait; echo AWARE_DONE ) &
[ -s $L/BRCA/eval.csv ] || ./evalpar.sh BRCA $L/BRCA/eval.csv "1 2 3 4 5" $L/BRCA/v2_*/ > $L/BRCA/eval.log 2>&1
echo EVAL_BRCA_DONE
[ -s $L/COMB/eval.csv ] || ./evalpar.sh COMB $L/COMB/eval.csv "1 2 3 4 5" $L/COMB/v2_*/ > $L/COMB/eval.log 2>&1
echo EVAL_COMB_DONE
[ -s $L/BRCA/eval_p.csv ] || ./evalpar.sh BRCA $L/BRCA/eval_p.csv "1 2 3 4 5" $L/BRCA/v2p_*/ > $L/BRCA/eval_p.log 2>&1
[ -s $L/COMB/eval_p.csv ] || ./evalpar.sh COMB $L/COMB/eval_p.csv "1 2 3 4 5" $L/COMB/v2p_*/ > $L/COMB/eval_p.log 2>&1
echo EVAL_P_DONE
for d in $L/BRCA/v2*_*/; do [ -f $L/BRCA/kl_$(basename $d).csv ] || python camda_kl_overlap.py $BB --syn_dir $d --out_csv $L/BRCA/kl_$(basename $d).csv > /dev/null 2>&1 & done
for d in $L/COMB/v2*_*/; do [ -f $L/COMB/kl_$(basename $d).csv ] || python camda_kl_overlap.py $CC --syn_dir $d --out_csv $L/COMB/kl_$(basename $d).csv > /dev/null 2>&1 & done
wait
python - > $L/summary.txt 2>&1 <<PY
import pandas as pd, glob, os
pd.set_option('display.width', 250)
c = ['w1','w1_cls','corr_mae','wcorr_r','coex0.3_tpr','coex0.3_fer','de_tpr','de_fpr','tstr_acc','tstr_f1','disc_auc','marg_auc','mahala_auc']
O = "$OUT/LRDP"
for D in ('BRCA', 'COMB'):
    d = pd.concat([pd.read_csv(f"{O}/{D}/eval.csv"), pd.read_csv(f"$L/{D}/eval.csv"), pd.read_csv(f"$L/{D}/eval_p.csv")])
    print(D); print(d.groupby('run')[c].mean().round(3).to_string())
    k = []
    for f in sorted(glob.glob(f"{O}/{D}/kl_*.csv") + glob.glob(f"$L/{D}/kl_*.csv")):
        x = pd.read_csv(f); x = x[~x.run.str.startswith('REAL')]
        k.append(x.drop(columns='fold').groupby('run').mean())
    print(pd.concat(k).round(3).to_string())
for f in sorted(glob.glob(f"{O}/redteam/simple_*_eps*.log") + glob.glob("$R/simple_*_eps*.log")):
    t = [l.split() for l in open(f) if l.startswith(("MahalaMIA", "MAMA-MIA 1+2"))]
    print(("v2 " if "/v2/" in f else "v1 ") + os.path.basename(f)[7:-4].ljust(20), "  ".join(f"{' '.join(x[:-6])} {x[-6]}+-{x[-5]}" for x in t))
for f in sorted(glob.glob("$R/aware_*.log")):
    t = [l.split() for l in open(f) if "[ensemble]" in l]
    print("v2 aware", os.path.basename(f)[6:-4], t[0][-6] if t else "n/a", [l.strip()[:90] for l in open(f) if l.startswith("[meta]")])
PY
echo V2_DONE
