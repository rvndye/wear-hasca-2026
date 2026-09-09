#!/bin/zsh
# poll kaggle submissions every 2 min when network is up; append to logs_lb.txt
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
cd ~/kaggle/wear
for i in $(seq 1 400); do
  if curl -s -o /dev/null --max-time 10 https://www.kaggle.com/; then
    out=$(kaggle competitions submissions 3rd-wear-dataset-challenge-hasca-2026 2>/dev/null | awk 'NR>2 {print $1, $NF}')
    if [ -n "$out" ]; then echo "=== $(date -u) ===" > logs_lb.txt; echo "$out" >> logs_lb.txt; fi
  fi
  sleep 120
done
