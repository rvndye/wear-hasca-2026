#!/bin/zsh
# every 4 min: push the head of queue.txt on CPU; if the CPU cap is hit and GPU budget remains, push it on GPU instead.
WL=${WEAR_LB:-$HOME/kaggle/wear_lb}; OWNER=${KAGGLE_OWNER:-rvndyerwin}; HERE=${0:A:h}
mkdir -p $WL/kern && cd $WL/kern
GPU_LEFT=${GPU_LEFT:-3}
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
for i in $(seq 1 900); do
  line=$(head -1 queue.txt)
  if [ -n "$line" ]; then
    out=$(python3 $HERE/mkfork.py ${=line} --push 2>&1)
    if echo "$out" | grep -q "successfully pushed"; then
      echo "$(date -u +%H:%M) pushed CPU: $line" >> qrun.log; sed -i '' '1d' queue.txt
    elif echo "$out" | grep -q "Maximum batch CPU session"; then
      if [ "$GPU_LEFT" -gt 0 ]; then
        out2=$(python3 $HERE/mkfork.py ${=line} --gpu --push 2>&1)
        if echo "$out2" | grep -q "successfully pushed"; then
          echo "$(date -u +%H:%M) pushed GPU: $line" >> qrun.log; sed -i '' '1d' queue.txt; GPU_LEFT=$((GPU_LEFT-1))
        fi
      fi
    else
      echo "$(date -u +%H:%M) push error: $line :: $(echo $out | tail -c 300)" >> qrun.log
    fi
  fi
  sleep 240
done
