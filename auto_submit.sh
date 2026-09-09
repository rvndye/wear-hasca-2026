#!/bin/zsh
# retry until network is up; submit probe, read its score, then submit the appropriate blend
cd ~/kaggle/wear
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
net() { curl -s -o /dev/null --max-time 10 https://www.kaggle.com/; }
sub() { for i in $(seq 1 120); do if net; then ./submit.sh "$1" "$2" >> logs_auto_submit.txt 2>&1; echo "$(date -u) submitted $1" >> logs_auto_submit.txt; return; fi; sleep 30; done; }
score() { for i in $(seq 1 20); do s=$(kaggle competitions submissions 3rd-wear-dataset-challenge-hasca-2026 2>/dev/null | awk -v f="$1" '$1==f {print $NF; exit}'); if [ -n "$s" ] && [ "$s" != "privateScore" ] && [[ "$s" =~ ^0 ]]; then echo $s; return; fi; sleep 30; done; echo 0; }
sub subs/sub_f2_nnHS7_full.csv "probe: NN hand-feats + inorm video + static path (centered mean via dropout MLP) + vid_drop 0.7, foldset2; OOF 0.726 raw / 0.741 tuned"
sleep 60
s=$(score sub_f2_nnHS7_full.csv); echo "$(date -u) probe score $s" >> logs_auto_submit.txt
if (( $(echo "$s >= 0.735" | bc -l) )); then
  sub subs/sub_m_all3.csv "m_all3: HS7 x2 + HCF7 x2 + 6 NN + 5 LGB + stack1/stack3 x2; OOF 0.751 raw / 0.757 honest"
else
  sub subs/sub_m_all2.csv "m_all2: HCF7 x2 + 6 NN + 5 LGB + stack1/stack3 x2 (no static path); OOF 0.746 raw / 0.752 honest"
fi
echo "$(date -u) AUTO DONE" >> logs_auto_submit.txt
