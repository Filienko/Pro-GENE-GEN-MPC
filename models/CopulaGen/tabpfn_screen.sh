#!/bin/bash
# Screening of empirical (not formally DP) MIA defences for the TabPFN-3.5 generator on BRCA.
# Each variant: generate (GPU, ~2 h per fold) -> eval_suite + MAMA-MIA + KL report (CPU, background) ->
# MeLoMIA-ND + MahalaMIA + MAMA-MIA v2 from MIA_on_bulkRNAseq_CAMDA2026 (cached shadow stack, see mia_verify/).
# The attack runs only while no TabPFN generation holds the GPU (they do not fit together).  Resumable.
#   FOLDS="3" VARIANTS="tabpfn_35_ma5" (nohup setsid ./tabpfn_screen.sh >> $OUT/TABPFN/screen.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
GPU=${GPU:-1}; T=$OUT/TABPFN; PY=$HOME/miniconda3/envs/camda_sota/bin/python; FOLDS=${FOLDS:-3}
A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; V=$HOME/privacy/mia_verify
say() { echo "$(date +%F_%T) $*"; }
score() {  # $1 run, $2 fold  (CPU)
  [ -f $T/BRCA/eval/$1_f$2.csv ] || OMP_NUM_THREADS=6 python eval_suite.py $A --folds $2 --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/eval/$1_f$2.csv > $T/BRCA/eval/$1_f$2.log 2>&1
  [ -f $T/BRCA/mama/$1_f$2.csv ] || python mama_lite.py $A --folds $2 --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/mama/$1_f$2.csv > /dev/null 2>&1
  python tabpfn_report.py $T > $T/report_screen.log 2>&1
}
attack() {  # $1 run, rest folds  (GPU)
  local r=$1; shift
  ( source $V/env.sh && $PY mia_import_targets.py --dataset BRCA --run_dir $T/BRCA/$r > /dev/null 2>&1 && \
    cd $HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026 && CUDA_VISIBLE_DEVICES=$GPU $HOME/miniconda3/envs/camda-mia/bin/python \
    scripts/run_experiment.py $V/configs/verify_brca.yaml --generators "ours@run=$r" --splits "$@" > $V/attack_$r.log 2>&1 )
  say "ATTACK $r folds $*:"; grep -E "^=== |split [0-9]: AUC" $V/attack_$r.log | sed 's/^/    /'
}
gen() {  # $1 run name, rest = run_tabpfn.py args
  local R=$1; shift
  for f in $FOLDS; do
    if [ ! -f $T/BRCA/$R/synthetic_data_split_$f.csv ]; then
      say "START $R fold $f"
      CUDA_VISIBLE_DEVICES=$GPU $PY run_tabpfn.py $A --splits_yaml $BRCA_SPLITS --folds $f --out_dir $T/BRCA/$R "$@" \
        > $T/BRCA/${R}_f$f.log 2>&1 || { say "FAILED $R fold $f"; continue; }
    fi
    score $R $f &
  done
  attack $R $FOLDS
}

declare -A ARGS=(
  [tabpfn_35_ma5]="--microagg 5 --restore cov"            # prompt: 5-member cluster means, spread restored (Gaussian)
  [tabpfn_35_launder]="--context_dir $T/BRCA/tabpfn_35"    # prompt: TabPFN-3.5's own synthetic release
  [tabpfn_35_shard4]="--shards 4"                          # 4 disjoint member shards, n/4 rows each
  [tabpfn_35_ma10]="--microagg 10 --restore cov"
  [tabpfn_35_ma5swap]="--microagg 5 --restore swap"        # cluster means + another cluster's real deviations
  [tabpfn_35_ma5mean]="--microagg 5 --restore none"         # cluster means only (~172 prompt rows)
  [tabpfn_35_ma5covlrdp]="--microagg 5 --restore cov --add_context_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30"  # ~50% real-derived rows
  [tabpfn_35_ma5lrdp300]="--microagg 5 --restore none --add_context_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30 --add_context_n 300"  # ~36%
  [tabpfn_35_ma5lrdp]="--microagg 5 --restore none --add_context_dir $OUT/LRDP/v2/BRCA/v2p_priv_eps30"  # means + LowRankDP eps30 rows
)
for v in ${VARIANTS:-tabpfn_35_ma5 tabpfn_35_launder tabpfn_35_shard4 tabpfn_35_ma10}; do gen $v ${ARGS[$v]}; done
wait; say "SCREEN_DONE"
