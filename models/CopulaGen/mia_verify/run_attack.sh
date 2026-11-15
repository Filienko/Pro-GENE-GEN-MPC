#!/bin/bash
# MeLoMIA-ND (paper setting, = the reported 0.75 on TabPFN-3.5 BRCA) + MahalaMIA + MAMA-MIA v2 on our BRCA releases.
# Uses ~/privacy/MIA_on_bulkRNAseq_CAMDA2026 unchanged; data / artifacts / results redirected by env.sh.
# GPU 1 is shared with TabPFN generation, which takes ~46 of its 48 GB, so the two never run together: this waits
# for any TabPFN fold to finish, runs the attack, then resumes the TabPFN queue.
# Resumable: every stage of the attack is cached.   (nohup setsid ./run_attack.sh >> run_attack.log 2>&1 < /dev/null &)
V=$HOME/privacy/mia_verify; source $V/env.sh; cd $HOME/privacy/MIA_on_bulkRNAseq_CAMDA2026
P=$HOME/miniconda3/envs/camda-mia/bin/python; CFG=$V/configs/verify_brca.yaml; export CUDA_VISIBLE_DEVICES=${GPU:-1}
R=$HOME/privacy/Pro-GENE-GEN-MPC/models/results_copula; CG=$HOME/privacy/Pro-GENE-GEN-MPC/models/CopulaGen
say() { echo "$(date +%F_%T) $*"; }

say "waiting for the running TabPFN fold to free GPU $CUDA_VISIBLE_DEVICES"
while ps -eo args | grep -q "[r]un_tabpfn.py"; do sleep 60; done
# score that fold (CPU) the way tabpfn_compare.sh would have
( cd $CG && source env.sh && A="--counts $BRCA_TSV --annot $BRCA_ANN --label_col Subtype" && T=$OUT/TABPFN && for f in 5; do
    [ -f $T/BRCA/eval/tabpfn_35_dp10_f$f.csv ] || OMP_NUM_THREADS=6 python eval_suite.py $A --folds $f --syn_dir $T/BRCA/tabpfn_35_dp10 --out_csv $T/BRCA/eval/tabpfn_35_dp10_f$f.csv > $T/BRCA/eval/tabpfn_35_dp10_f$f.log 2>&1
    [ -f $T/BRCA/mama/tabpfn_35_dp10_f$f.csv ] || python mama_lite.py $A --folds $f --syn_dir $T/BRCA/tabpfn_35_dp10 --out_csv $T/BRCA/mama/tabpfn_35_dp10_f$f.csv > /dev/null 2>&1
  done; python tabpfn_report.py $T > $T/report.log 2>&1 ) &
$HOME/miniconda3/envs/camda_sota/bin/python $CG/mia_import_targets.py --dataset BRCA --run_dir \
  $R/TABPFN/BRCA/{tabpfn_35,tabpfn_35_dp10,tabpfn_35_dp30,tabpfn_pe10,tabpfn_pe30,miav_35} \
  $R/EPSWEEP/BRCA/{lrdp_eps10,pgm_eps10} $R/LRDP/v2/BRCA/v2p_priv_eps30 > $V/import.log 2>&1; say "IMPORTED"

say "PREPARE shadow stack (4 workers share the 30 shadows through lock files)"
for w in 1 2 3 4; do $P scripts/run_experiment.py $CFG --only melomia_nd --prepare-only --no-meta > $V/prepare_w$w.log 2>&1 & sleep 120; done
wait
grep -l "Error" $V/prepare_w*.log && { say "PREPARE_FAILED (see prepare_w*.log)"; exit 1; }
say "PREPARE_DONE"

# verification target first; this process also trains (and caches) the meta-classifier
$P scripts/run_experiment.py $CFG --generators "ours@run=tabpfn_35" > $V/attack_tabpfn_35.log 2>&1; say "DONE tabpfn_35"
grep -E "MEAN|split . *: AUC" $V/attack_tabpfn_35.log
lane() { for g in "$@"; do $P scripts/run_experiment.py $CFG --generators "ours@run=$g" > $V/attack_$g.log 2>&1; say "DONE $g"; grep MEAN $V/attack_$g.log; done; }
lane tabpfn_35_dp30 v2p_priv_eps30 tabpfn_pe30 &
lane tabpfn_35_dp10 lrdp_eps10 tabpfn_pe10 &
lane pgm_eps10 miav_35 &
wait; say "ATTACK_DONE"

# give GPU 1 back to the TabPFN queue (remaining: dp30 fold 5)
( cd $CG && source env.sh && echo "$(date +%F_%T) restarted after the attack" >> $OUT/TABPFN/run.log && \
  (nohup setsid ./tabpfn_compare.sh >> $OUT/TABPFN/run.log 2>&1 < /dev/null &) ); say "TABPFN_QUEUE_RESUMED"
