#!/bin/zsh
# at 00:01 UTC Sep 4 submit each line of sched_list.txt: "<file>|<message>" (list read at fire time)
cd ~/kaggle/wear
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-09-04 00:01:00" +%s)
while [ $(date -u +%s) -lt $target ]; do sleep 30; done
while IFS='|' read -r f msg; do
  [ -z "$f" ] && continue
  for i in $(seq 1 60); do if curl -s -o /dev/null --max-time 10 https://www.kaggle.com/; then ./submit.sh "$f" "$msg" >> logs_sched_submit.txt 2>&1; echo "$(date -u) submitted $f" >> logs_sched_submit.txt; break; fi; sleep 60; done
  sleep 40
done < sched_list.txt
echo "$(date -u) SCHED DONE" >> logs_sched_submit.txt
