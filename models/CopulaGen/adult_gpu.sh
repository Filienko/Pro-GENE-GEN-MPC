#!/bin/bash
# One TabPFN job on Adult (run via gpu_worker.sh), then scoring.   ./adult_gpu.sh RUN "FOLDS" [run_tabpfn.py args]
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=6
R=$1; FOLDS=$2; shift 2; AD=$OUT/ADULT; PY=$HOME/miniconda3/envs/camda_sota/bin/python
for f in $FOLDS; do
  [ -f $AD/$R/synthetic_data_split_$f.csv ] || $PY run_tabpfn.py --counts $ADULT_TSV --annot $ADULT_ANN --label_col Subtype --splits_yaml $ADULT_SPLITS \
      --folds $f --out_dir $AD/$R "$@" > $AD/${R}_f$f.log 2>&1 || echo "FAILED $R fold $f"
done
./adult_score.sh $R; echo "$(date +%F_%T) ADULT $R folds $FOLDS done"
