#!/bin/bash
# CPU evaluation of the DP-MVN stage ablation (dpmvn_ablation.py releases in $OUT/ABLATION/BRCA/eps*_*), resumable:
# eval_suite (fidelity, utility, MahalaMIA / DCR / DOMIAS / marginal attacks), CAMDA KL, MAMA-lite, kNN-LR, and the
# tracing attack with the releases' own noise SDs.  Releases with more rows than the training set (*_x<m>) get the
# tracing attack only.
#   nohup setsid ./dpmvn_ablation_eval.sh >> $OUT/ABLATION/BRCA/run_eval.log 2>&1 < /dev/null &
cd "$(dirname "$0")"; source ./env.sh
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
A=${ABL:-$OUT/ABLATION/BRCA}; P=${PAR:-14}
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"
mkdir -p $A/eval $A/kl $A/mama $A/knn $A/attack
jobs_list() {
  for d in $A/eps*_*/; do r=$(basename $d)
    echo "[ -f $A/attack/trace_$r.csv ] || python dpmvn_trace_attack.py $B --syn_dir $d --sigma_from_config --n_member_from_ids --out_csv $A/attack/trace_$r.csv > $A/attack/trace_$r.log 2>&1"
    case $r in *_x*) continue;; esac
    for f in 1 2 3 4 5; do
      echo "[ -f $A/eval/${r}_f$f.csv ] || python eval_suite.py $B --folds $f --syn_dir $d --out_csv $A/eval/${r}_f$f.csv > /dev/null 2>&1"
      echo "[ -f $A/kl/${r}_f$f.csv ] || python camda_kl_overlap.py $B --camda_src $HOME/privacy/PPML-Huskies-Red-Team/src --syn_dir $d --folds $f --out_csv $A/kl/${r}_f$f.csv > /dev/null 2>&1"
    done
    echo "[ -f $A/mama/$r.csv ] || python mama_lite.py $B --syn_dir $d --out_csv $A/mama/$r.csv > /dev/null 2>&1"
    echo "[ -f $A/knn/knn_$r.csv ] || python knn_lr_attack.py $B --syn_dir $d --k 40 --n_feat 150 --same_class 0 --out_csv $A/knn/knn_$r.csv > /dev/null 2>&1"
  done
}
echo "$(date +%F_%T) start ($(jobs_list | wc -l) jobs)"
jobs_list | xargs -P $P -I{} sh -c '{}'
echo "$(date +%F_%T) done"; touch $A/EVAL_DONE
