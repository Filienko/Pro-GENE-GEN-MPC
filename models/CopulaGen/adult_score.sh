#!/bin/bash
# Score Adult runs (all folds present): adult_eval.py + kNN-LR + MAMA-lite, then the table.   ./adult_score.sh RUN [RUN ...]
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=6
AD=$OUT/ADULT; A="--counts $ADULT_TSV --annot $ADULT_ANN --label_col Subtype"
for r in "$@"; do
  for f in 1 2 3 4 5; do [ -f $AD/$r/synthetic_data_split_$f.csv ] || continue
    [ -f $AD/eval/${r}_f$f.csv ] || python adult_eval.py --syn_dir $AD/$r --folds $f --out_csv $AD/eval/${r}_f$f.csv > /dev/null 2>&1
    [ -f $AD/knn/knn_${r}_${r}_f${f}_scores.csv ] || OMP_NUM_THREADS=16 python knn_lr_attack.py $A --syn_dir $AD/$r --folds $f --k 40 --n_feat 150 --min_pos 2 --same_class 0 --out_csv $AD/knn/knn_$r.csv > /dev/null 2>&1
    [ -f $AD/mama/${r}_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $AD/$r --out_csv $AD/mama/${r}_f$f.csv > /dev/null 2>&1
  done; echo "$(date +%F_%T) scored $r"
done
python adult_table.py > /dev/null 2>&1
