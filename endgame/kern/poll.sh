#!/bin/zsh
# poll WEAR fork kernels; when one completes/errors, fetch its small outputs once. Writes kern/status.txt
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
WL=${WEAR_LB:-$HOME/kaggle/wear_lb}; OWNER=${KAGGLE_OWNER:-rvndyerwin}; HERE=${0:A:h}
mkdir -p $WL/kern && cd $WL/kern
for i in $(seq 1 1000); do
  : > status.tmp
  for d in wear-p2-*/; do
    k=${d%/}; st=$(kaggle kernels status $OWNER/$k 2>/dev/null | sed -n 's/.*status "KernelWorkerStatus\.\([A-Z_]*\)".*/\1/p')
    echo "$(date -u +%H:%M) $k $st" >> status.tmp
    if [[ "$st" == "COMPLETE" || "$st" == "ERROR" ]] && [ ! -f $WL/kout/$k/.fetched ]; then
      mkdir -p $WL/kout/$k
      (cd $WL/kout/$k && kaggle kernels output $OWNER/$k -p . > fetch.txt 2>&1) && touch $WL/kout/$k/.fetched
    fi
  done
  mv status.tmp status.txt
  sleep 300
done
