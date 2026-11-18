#!/bin/bash
# MVN baseline (run_mvn.py) through the same scoring + attacks as the TabPFN variants, on CPU.
source $(dirname $0)/env.sh; cd $(dirname $0)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
T=$OUT/TABPFN; R=${R:-mvn}; A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; V=$HOME/privacy/mia_verify
PY=$HOME/miniconda3/envs/camda_sota/bin/python
say() { echo "$(date +%F_%T) $*"; }
for f in 1 2 3 4 5; do
  [ -f $T/BRCA/eval/${R}_f$f.csv ] || python eval_suite.py $A --folds $f --syn_dir $T/BRCA/$R --out_csv $T/BRCA/eval/${R}_f$f.csv > $T/BRCA/eval/${R}_f$f.log 2>&1
  [ -f $T/BRCA/mama/${R}_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $T/BRCA/$R --out_csv $T/BRCA/mama/${R}_f$f.csv > /dev/null 2>&1
  say "scored fold $f"
done
( source $V/env.sh && $PY mia_import_targets.py --dataset BRCA --run_dir $T/BRCA/$R && \
  cd $HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026 && CUDA_VISIBLE_DEVICES="" $HOME/miniconda3/envs/camda-mia/bin/python \
  scripts/run_experiment.py $V/configs/verify_brca_cpu.yaml --generators "ours@run=$R" --splits 1 2 3 4 5 > $V/attack_$R.log 2>&1 )
say "ATTACK $R:"; grep -E "^=== |split [0-9]: AUC" $V/attack_$R.log | sed 's/^/    /'
python fused_attack.py --runs $R --out_csv $T/attack_lr/fused_$R.csv
python tabpfn_report.py $T > $T/report_screen.log 2>&1
say MVN_DONE
