#!/bin/zsh
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
WL=${WEAR_LB:-$HOME/kaggle/wear_lb}; mkdir -p $WL/data/inertial && cd $WL/data
for i in 0 0_2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 14_2 15 16 17 18 19 20 21; do
  f=sbj_$i.csv; [ -s inertial/$f ] && continue
  for a in 1 2 3; do kaggle competitions download -c 3rd-wear-dataset-challenge-hasca-2026 -f train/inertial_feat/$f -p inertial >/dev/null 2>&1; [ -s inertial/$f ] && break; [ -s inertial/$f.zip ] && break; sleep 5; done
  [ -f inertial/$f.zip ] && (cd inertial && unzip -o -q $f.zip && rm $f.zip)
done
for f in test/test_meta_data.csv sample_submission.csv; do kaggle competitions download -c 3rd-wear-dataset-challenge-hasca-2026 -f $f -p . >/dev/null 2>&1; done
echo "DONE $(ls inertial | wc -l) files"
