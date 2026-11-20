#!/bin/bash
# GPU job queue.  Each worker owns one GPU; it waits until that GPU has the head job's memory free, then pops the job
# (flock) and runs it with CUDA_VISIBLE_DEVICES set.  Queue lines: "<MiB needed> <command>".
#   GPU=1 (nohup setsid ./gpu_worker.sh $OUT/gpu_jobs.txt >> $OUT/gpu_worker1.log 2>&1 < /dev/null &)
source $(dirname $0)/env.sh; cd $(dirname $0)
Q=$1; G=${GPU:?}; say() { echo "$(date +%F_%T) [GPU $G] $*"; }
while :; do
  job=""
  exec 9> $Q.lock; flock 9
  head=$(head -n 1 $Q)
  if [ -z "$head" ]; then flock -u 9; say "queue empty"; break; fi
  need=${head%% *}; free=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -ge "$need" ]; then sed -i 1d $Q; job=${head#* }; fi
  flock -u 9
  if [ -z "$job" ]; then sleep 60; continue; fi
  say "start: $job"; CUDA_VISIBLE_DEVICES=$G bash -c "$job"; say "done (rc=$?): $job"
  sleep 30
done
