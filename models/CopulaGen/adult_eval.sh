#!/bin/bash
# Adult (adult_prep.py subset): generators -> adult_eval.py (W1 / TVD / TSTR / DCR, DCR-ratio, Mahalanobis, DOMIAS)
# + kNN-LR + MAMA-lite -> adult_table.py.  CPU generators here; the TabPFN runs go through the GPU queue (adult_gpu.sh).
#   (nohup setsid ./adult_eval.sh >> $OUT/ADULT/run.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
AD=$OUT/ADULT; A="--counts $ADULT_TSV --annot $ADULT_ANN --label_col Subtype"; S="--splits_yaml $ADULT_SPLITS"
mkdir -p $AD/eval $AD/knn $AD/mama; PY=$HOME/miniconda3/envs/camda_sota/bin/python
say() { echo "$(date +%F_%T) $*"; }
[ -f $AD/mvn/config.json ] || $PY run_mvn.py $A $S --out_dir $AD/mvn > $AD/mvn.log 2>&1
for e in 10 30 100 1000; do
  [ -f $AD/lrdp_eps$e/config.json ] || python run_lrdp.py $A $S --dp_epsilon $e --v2 --bounds_json $ADULT_META --out_dir $AD/lrdp_eps$e > $AD/lrdp_$e.log 2>&1
done
[ -f $AD/pgm_eps10/config.json ] || (cd ../StructPGM && python run_structpgm.py --structure star --epsilon 10 $A $S --out_dir $AD/pgm_eps10 > $AD/pgm_10.log 2>&1)
[ -f $AD/dpcov_prompt_only/config.json ] || python ablate_no_tabpfn.py $A $S --jitter_dir $AD/lrdp_eps30 --out_dir $AD/dpcov_prompt_only > $AD/prompt_only.log 2>&1
say "CPU generators done"
./adult_score.sh mvn lrdp_eps10 lrdp_eps30 lrdp_eps100 lrdp_eps1000 pgm_eps10 dpcov_prompt_only
say ADULT_CPU_DONE
