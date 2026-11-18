#!/bin/bash
# Start tabpfn_screen.sh variants one by one, each on a GPU that is (nearly) empty and not used by an MIA run.
#   VARIANTS="a b" FOLDS=3 (nohup setsid ./gpu_queue.sh >> $OUT/TABPFN/gpu_queue.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
for v in $VARIANTS; do
  while :; do
    G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 47000 {print $1; exit}')
    if [ -n "$G" ] && ! ps -eo args | grep -q "[r]un_experiment.py"; then sleep 120
      G2=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$G '$1 == g && $2 >= 47000 {print $1}')
      [ "$G2" = "$G" ] && break
    fi; sleep 60
  done
  echo "$(date +%F_%T) launch $v on GPU $G"
  ( GPU=$G FOLDS="${FOLDS:-3}" VARIANTS=$v nohup setsid ./tabpfn_screen.sh >> $OUT/TABPFN/screen_$v.log 2>&1 < /dev/null & )
  sleep 300
done; echo QUEUE_DONE
