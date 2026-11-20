#!/bin/bash
# Run via gpu_worker.sh (sets CUDA_VISIBLE_DEVICES).  Two-stage ablation, BRCA: stage 1 (DP-MVN eps30 | ND on members) x stage 2 (none | TabPFN | ND) x real-derived share
# (0% | 65% | 100%).  Waits for the TabPFN ratio series to free the GPUs.  Generate -> score -> attacks (as tabpfn_screen.sh).
#   FOLDS=3 (nohup setsid ./nd_ablation.sh >> $OUT/TABPFN/nd_ablation.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
T=$OUT/TABPFN; L=$OUT/LRDP/v2/BRCA/v2p_priv_eps30; FOLDS=${FOLDS:-3}; V=$HOME/privacy/mia_verify
A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; MPY=$HOME/miniconda3/envs/camda-mia/bin/python
say() { echo "$(date +%F_%T) $*"; }
G=${CUDA_VISIBLE_DEVICES:-1}; say "ND chain on GPU $G"   # started by gpu_worker.sh on a free GPU
score() {  # $1 run
  for f in $FOLDS; do
    [ -f $T/BRCA/eval/$1_f$f.csv ] || OMP_NUM_THREADS=6 python eval_suite.py $A --folds $f --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/eval/$1_f$f.csv > $T/BRCA/eval/$1_f$f.log 2>&1
    [ -f $T/BRCA/mama/$1_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $T/BRCA/$1 --out_csv $T/BRCA/mama/$1_f$f.csv > /dev/null 2>&1
  done
}
attack() {  # $1 run, $2 GPU
  ( source $V/env.sh && $HOME/miniconda3/envs/camda_sota/bin/python mia_import_targets.py --dataset BRCA --run_dir $T/BRCA/$1 > /dev/null 2>&1 && \
    cd $HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026 && CUDA_VISIBLE_DEVICES=$2 $MPY scripts/run_experiment.py $V/configs/verify_brca.yaml \
      --generators "ours@run=$1" --splits $FOLDS > $V/attack_$1.log 2>&1 )
  say "ATTACK $1:"; grep -E "^=== |split [0-9]: AUC" $V/attack_$1.log | sed 's/^/    /'
  OMP_NUM_THREADS=16 python fused_attack.py --runs $1 --folds $FOLDS --out_csv $T/attack_lr/fused_$1.csv 2>/dev/null | grep -E "fused|kNN" | sed "s/^/    /"
}
nd() {  # $1 run, rest: run_nd_stage.py args ; GPU 0
  local R=$1; shift; say "START $R"
  CUDA_VISIBLE_DEVICES=$G $MPY run_nd_stage.py $A --splits_yaml $BRCA_SPLITS --folds $FOLDS --out_dir $T/BRCA/$R "$@" > $T/BRCA/$R.log 2>&1 || { say "FAILED $R"; return; }
  score $R; attack $R $G
}
nd nd_real --train real                                                           # stage 1 only (CAMDA ND baseline)
nd nd_from_lrdp --train release --release_dir $L                                  # DP-MVN -> ND, 0% real (formal eps30)
nd nd_mix65_lrdp --train mix --release_dir $L --jitter_dir $L --n_release 470     # DP-MVN + cluster rows -> ND, 65%
nd nd_cluster --train cluster --jitter_dir $L                                     # cluster rows -> ND, 100%
nd nd_from_nd --train release --release_dir $T/BRCA/nd_real                       # ND -> ND, 0%
nd nd_mix65_nd --train mix --release_dir $T/BRCA/nd_real --jitter_dir $T/BRCA/nd_real --n_release 470   # ND + cluster rows -> ND, 65%
# stage 2 = TabPFN with stage 1 = ND (needs the whole GPU, so after the ND cells)
GPU=$G FOLDS="$FOLDS" VARIANTS=tabpfn_35_mix65_nd ./tabpfn_screen.sh >> $T/screen_tabpfn_35_mix65_nd.log 2>&1
python tabpfn_report.py $T > $T/report_screen.log 2>&1; say ND_ABLATION_DONE
