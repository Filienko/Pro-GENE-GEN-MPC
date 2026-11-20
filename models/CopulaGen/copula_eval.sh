#!/bin/bash
# DP-MVN + Gaussian-copula marginals (run_lrdp.py --copula), 5 folds:
#   BRCA : anchor histogram (no extra budget), --copula_thr 10   -> $OUT/EPSWEEP/BRCA/lrdpc_eps*   (full attack suite)
#   ADULT: own 100-bin histogram (share 0.08), threshold 3        -> $OUT/ADULT/lrdpc_eps*
#   (nohup setsid ./copula_eval.sh >> $OUT/copula_eval.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0); export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
EPSL=${EPSL:-"10 30 100 1000"}; T=$OUT/TABPFN; E=$OUT/EPSWEEP/BRCA; V=$HOME/privacy/mia_verify
B="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype"; MIAPY=$HOME/miniconda3/envs/camda-mia/bin/python; MIA=$HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026
say() { echo "$(date +%F_%T) $*"; }
# ---- Adult (CPU only, minutes)
for e in $EPSL; do r=lrdpc_eps$e
  [ -f $OUT/ADULT/$r/config.json ] || python run_lrdp.py --counts $ADULT_TSV --annot $ADULT_ANN --label_col Subtype --splits_yaml $ADULT_SPLITS \
      --dp_epsilon $e --v2 --bounds_json $ADULT_META --copula --copula_bins 100 --out_dir $OUT/ADULT/$r > $OUT/ADULT/lrdpc_$e.log 2>&1
  ./adult_score.sh $r
done; say "ADULT copula done"
# ---- BRCA generation + CPU scoring + cheap attacks
RUNS=""
for e in $EPSL; do r=lrdpc_eps$e; RUNS="$RUNS $r"
  [ -f $E/$r/config.json ] || python run_lrdp.py $B --splits_yaml $BRCA_SPLITS --dp_epsilon $e --v2 --copula --copula_thr 10 --out_dir $E/$r > $E/lrdpc_$e.log 2>&1
  for f in 1 2 3 4 5; do
    [ -f $T/BRCA/eval/${r}_f$f.csv ] || python eval_suite.py $B --folds $f --syn_dir $E/$r --out_csv $T/BRCA/eval/${r}_f$f.csv > /dev/null 2>&1 &
    [ -f $T/BRCA/kl/${r}_f$f.csv ] || python camda_kl_overlap.py $B --camda_src $HOME/privacy/PPML-Huskies-Red-Team/src --syn_dir $E/$r --folds $f --out_csv $T/BRCA/kl/${r}_f$f.csv > /dev/null 2>&1 &
  done; wait
  [ -f $T/BRCA/mama/$r.csv ] || python mama_lite.py $B --syn_dir $E/$r --out_csv $T/BRCA/mama/$r.csv > /dev/null 2>&1
  ( source $V/env.sh && $HOME/miniconda3/envs/camda_sota/bin/python mia_import_targets.py --dataset BRCA --run_dir $E/$r > /dev/null 2>&1 && \
    cd $MIA && CUDA_VISIBLE_DEVICES="" $MIAPY scripts/run_experiment.py $V/configs/verify_brca_cpu.yaml --only mahalamia mamamia_v2 \
      --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_fast.log 2>&1 )
  say "BRCA $r scored"; grep MEAN $V/attack_${r}_fast.log | sed 's/^/    /'
done
# ---- BRCA MeLoMIA-ND (GPU)
for r in $RUNS; do
  ls -d $V/results/runs/BRCA__melomia_nd__ours@run=${r}__s5__* > /dev/null 2>&1 && continue
  while :; do G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 16000 {print $1; exit}'); [ -n "$G" ] && break; sleep 120; done
  ( cd $MIA && CUDA_VISIBLE_DEVICES=$G $MIAPY scripts/run_experiment.py $V/configs/verify_brca.yaml --only melomia_nd --generators "ours@run=$r" --splits 1 2 3 4 5 > $V/attack_${r}_melomia.log 2>&1 )
  say "MeLoMIA $r:"; grep MEAN $V/attack_${r}_melomia.log | sed 's/^/    /'
done
python fused_attack.py --runs $RUNS --out_csv $T/attack_lr/fused_lrdpc.csv > /dev/null 2>&1
python tabpfn_report.py $T > $T/report_screen.log 2>&1
say COPULA_DONE
