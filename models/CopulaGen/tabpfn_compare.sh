#!/bin/bash
# TabPFN-3.5 (MIA_on_bulkRNAseq_CAMDA2026 generator) and MIAV on TabPFN-3.5 vs LowRankDP at eps 1/5/10/30 vs P-PGM eps 10, on our metrics.
# One TabPFN process at a time on GPU $GPU (a gene-by-gene fit takes ~30 GB and ~4 h per BRCA fold).
# After every generated fold: eval_suite + MAMA-MIA for it, then the report and plots are refreshed, so partial
# results are always readable.  Resumable: finished folds / evals are skipped.
#   (nohup setsid ./tabpfn_compare.sh > $OUT/TABPFN/run.log 2>&1 < /dev/null &)
# Runs TabPFN-3.5 only (needs the Prior Labs licence accepted; token from TABPFN_TOKEN or ~/.cache/tabpfn/auth_token).
source $(dirname $0)/env.sh; cd $(dirname $0)
GPU=${GPU:-1}; T=$OUT/TABPFN; PY=$HOME/miniconda3/envs/camda_sota/bin/python; export OMP_NUM_THREADS=6
BB="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; CC="--counts $COMB_TSV --annot $COMB_ANN --label_col cancer_type"
mkdir -p $T/BRCA/eval $T/BRCA/mama

baselines() {  # MAMA-MIA for the LowRankDP / P-PGM runs (cheap, once)
  for D in BRCA; do [ $D = BRCA ] && A=$BB || A=$CC
    for d in $OUT/EPSWEEP/$D/lrdp_eps1 $OUT/EPSWEEP/$D/lrdp_eps5 $OUT/EPSWEEP/$D/lrdp_eps10 $OUT/LRDP/v2/$D/v2p_priv_eps30 $OUT/EPSWEEP/$D/pgm_eps10; do
      [ -f $T/$D/mama/$(basename $d).csv ] || python mama_lite.py $A --syn_dir $d --out_csv $T/$D/mama/$(basename $d).csv > /dev/null 2>&1
    done; done
}
score() {  # $1 dataset, $2 run name, $3 fold
  D=$1; d=$T/$D/$2; f=$3; [ $D = BRCA ] && A=$BB || A=$CC
  [ -f $T/$D/eval/$2_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $d --out_csv $T/$D/eval/$2_f$f.csv > $T/$D/eval/$2_f$f.log 2>&1
  [ -f $T/$D/mama/$2_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $d --out_csv $T/$D/mama/$2_f$f.csv > /dev/null 2>&1
  python tabpfn_report.py $T > $T/report.log 2>&1
}
gen() {  # $1 dataset, $2 run name, $3 fold, rest = generator args (GEN=run_tabpfn.py | run_miav.py)
  D=$1; R=$2; f=$3; shift 3; G=${GEN:-run_tabpfn.py}; [ $D = BRCA ] && { A=$BB; SP=$BRCA_SPLITS; } || { A=$CC; SP=$COMB_SPLITS; }
  if [ ! -f $T/$D/$R/synthetic_data_split_$f.csv ]; then
    echo "$(date +%F_%T) START $D $R fold $f"
    CUDA_VISIBLE_DEVICES=$GPU $PY $G $A --splits_yaml $SP --folds $f --out_dir $T/$D/$R "$@" > $T/$D/${R}_f$f.log 2>&1 \
      || { echo "$(date +%F_%T) FAILED $D $R fold $f (see $T/$D/${R}_f$f.log)"; return; }
  fi
  score $D $R $f; echo "$(date +%F_%T) DONE $D $R fold $f"
}

# a fold started by the previous queue may still be generating: let it finish before touching the GPU
while pgrep -f "run_tabpfn.py .*--out_dir" > /dev/null; do sleep 60; done
# fail fast if the TabPFN-3.5 licence is not accepted for the cached / supplied token
CUDA_VISIBLE_DEVICES=$GPU TABPFN_DISABLE_TELEMETRY=1 $PY -c "
import numpy as np; from tabpfn import TabPFNRegressor, TabPFNClassifier
X = np.random.randn(40, 2); TabPFNRegressor(device='cuda').fit(X, X[:, 0]); TabPFNClassifier(device='cuda').fit(X, X[:, 0] > 0)" \
  > $T/licence_check.log 2>&1 || { echo "$(date +%F_%T) LICENCE_MISSING: accept the TabPFN-3.5 licence (see $T/licence_check.log), then rerun"; exit 1; }
echo "$(date +%F_%T) LICENCE_OK"
baselines; echo "$(date +%F_%T) BASELINES_DONE"
# MIAV (Chaibub Neto, ICLR 2026) on TabPFN-3.5: stopped by request 2026-11-13 (BRCA 5 folds, COMBINED folds 1-2 done)
# for f in 1 2 3 4 5; do GEN=run_miav.py gen BRCA miav_35 $f; done
# BRCA only for now (2026-11-13); COMBINED later:  for f in 1 2 3 4 5; do gen COMB tabpfn_35 $f; done
# Fold-major so every variant has fold 1 early.  ~2 h per BRCA fold (~3.5 h with the public pool).
#   tabpfn_35            non-private: real training rows are the prompt (the repo's recipe)
#   tabpfn_35_dp10/dp30  DP by post-processing: the prompt is LowRankDP's eps=10 / eps=30 synthetic fold
#   tabpfn_35_dp10pub    as dp10 plus 600 public non-breast TCGA tumours (disjoint patients) as an extra class
LR10=$OUT/EPSWEEP/BRCA/lrdp_eps10; LR30=$OUT/LRDP/v2/BRCA/v2p_priv_eps30; PUB="--public_counts $COMB_TSV --public_annot $COMB_ANN --n_public 600"
# methods without public data first (request 2026-11-13); the public-pool variant afterwards
for f in 1 2 3 4 5; do
  gen BRCA tabpfn_35 $f
  gen BRCA tabpfn_35_dp10 $f --context_dir $LR10
  gen BRCA tabpfn_35_dp30 $f --context_dir $LR30
done
# deferred 2026-11-15 (GPU needed for the MeLoMIA-ND attack verification):
# for f in 1 2 3 4 5; do gen BRCA tabpfn_35_dp10pub $f --context_dir $LR10 $PUB; done
echo "$(date +%F_%T) TABPFN_BRCA_DONE"
