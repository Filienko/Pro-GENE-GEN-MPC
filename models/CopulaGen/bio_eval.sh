#!/bin/bash
# Second dataset: HF inria-soda/tabular-benchmark clf_num/Bioresponse (1089-row balanced subset, 419 zero-inflated
# skewed features, values mapped [0,1] -> [2,22]; 5 splits 871/218, see ~/privacy/hf_data/BIO).  Same generators and
# metrics as BRCA.  Attacks: kNN-LR, MAMA-lite, and eval_suite's MahalaMIA / DCR / DOMIAS / marginal (MeLoMIA-ND and the
# challenge repo's attacks are wired to BRCA/COMBINED only).  CPU part first; TabPFN runs wait for a free GPU.  Resumable.
#   (nohup setsid ./bio_eval.sh >> $OUT/BIO/run.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
DS=${DS:-BIO}; TSV=${DS}_TSV; ANN=${DS}_ANN; SPL=${DS}_SPLITS     # DS=BIO (1089-row subset) or BIOFULL (all 3434)
B=$OUT/$DS; A="--counts ${!TSV} --annot ${!ANN} --label_col Subtype"; S="--splits_yaml ${!SPL}"
mkdir -p $B/eval $B/mama $B/kl $B/knn; PY=$HOME/miniconda3/envs/camda_sota/bin/python
CAMDA_SRC=$HOME/privacy/PPML-Huskies-Red-Team/src
say() { echo "$(date +%F_%T) $*"; }
score() {  # $1 run name (dir $B/$1), all folds present
  local d=$B/$1
  for f in 1 2 3 4 5; do [ -f $d/synthetic_data_split_$f.csv ] || continue
    [ -f $B/eval/$1_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $d --out_csv $B/eval/$1_f$f.csv > $B/eval/$1_f$f.log 2>&1 &
    [ -f $B/kl/$1_f$f.csv ] || python camda_kl_overlap.py $A --camda_src $CAMDA_SRC --syn_dir $d --folds $f --out_csv $B/kl/$1_f$f.csv > /dev/null 2>&1 &
    wait
    [ -f $B/mama/$1_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $d --out_csv $B/mama/$1_f$f.csv > /dev/null 2>&1
    [ -f $B/knn/knn_$1_$1_f${f}_scores.csv ] || OMP_NUM_THREADS=16 python knn_lr_attack.py $A --syn_dir $d --folds $f --k 40 --n_feat 150 --same_class 0 \
        --out_csv $B/knn/knn_$1.csv > /dev/null 2>&1
  done; say "scored $1"
}
# ---- CPU generators
[ -f $B/mvn/config.json ] || $PY run_mvn.py $A $S --out_dir $B/mvn > $B/mvn.log 2>&1
for e in 10 30 50 100 1000; do
  [ -f $B/lrdp_eps$e/config.json ] || python run_lrdp.py $A $S --dp_epsilon $e --v2 --out_dir $B/lrdp_eps$e > $B/lrdp_$e.log 2>&1
done
[ -f $B/pgm_eps10/config.json ] || (cd ../StructPGM && python run_structpgm.py --structure star --epsilon 10 $A $S --out_dir $B/pgm_eps10 > $B/pgm_10.log 2>&1)
[ -f $B/dpcov_prompt_only/config.json ] || python ablate_no_tabpfn.py $A $S --jitter_dir $B/lrdp_eps30 --out_dir $B/dpcov_prompt_only > $B/prompt_only.log 2>&1
say "CPU generators done"
for r in mvn lrdp_eps10 lrdp_eps30 lrdp_eps50 lrdp_eps100 lrdp_eps1000 pgm_eps10 dpcov_prompt_only; do score $r; done
python bio_table.py $DS > /dev/null 2>&1; say "table (CPU methods)"
[ "$DS" = BIO ] || { say "${DS}_DONE (CPU methods only)"; exit 0; }
# ---- GPU: TabPFN-3.5 and the dpcov 65% hybrid; wait until the BRCA queue (nd_ablation.sh) has started and a GPU is empty
until grep -q "GPUs free" $OUT/TABPFN/nd_ablation.log 2>/dev/null; do sleep 120; done; sleep 600
tab() {  # $1 run, rest run_tabpfn args
  local R=$1; shift
  for f in ${FOLDS:-3 1 2 4 5}; do
    [ -f $B/$R/synthetic_data_split_$f.csv ] && continue
    while :; do
      G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 47000 {print $1; exit}')
      [ -n "$G" ] && break; sleep 120
    done
    say "START $R fold $f on GPU $G"
    CUDA_VISIBLE_DEVICES=$G $PY run_tabpfn.py $A $S --folds $f --out_dir $B/$R "$@" > $B/${R}_f$f.log 2>&1 || say "FAILED $R fold $f"
  done
  score $R; python bio_table.py $DS > /dev/null 2>&1
}
tab tabpfn_35 &
sleep 900
tab tabpfn_35_dpcov65 --microagg 5 --restore dpcov --jitter_dir $B/lrdp_eps30 --add_context_dir $B/lrdp_eps30 --add_context_n 470 &
wait; python bio_table.py $DS > $B/table.log 2>&1; say ${DS}_DONE
