#!/bin/zsh
# usage: ./submit.sh subs/sub_x.csv "message"
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
kaggle competitions submit -c 3rd-wear-dataset-challenge-hasca-2026 -f "$1" -m "$2" 2>&1 | grep -v -i "warn\|urllib3"
