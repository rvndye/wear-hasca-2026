#!/bin/zsh
# Download all competition files individually (resumable per file).
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
COMP=3rd-wear-dataset-challenge-hasca-2026
cd ~/kaggle/wear/data
FILES=(test/test_videomae_data.npy)
for i in 0 0_2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 14_2 15 16 17 18 19 20 21; do FILES+=(train/inertial_feat/sbj_$i.csv); done
for i in 0 0_2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 14_2 15 16 17 18 19 20 21; do FILES+=(train/videomae_feat/sbj_$i.npy); done
mkdir -p inertial videomae
for f in "${FILES[@]}"; do
  base=$(basename "$f")
  case "$f" in
    train/inertial_feat/*) dest=inertial ;;
    train/videomae_feat/*) dest=videomae ;;
    *) dest=. ;;
  esac
  if [ -s "$dest/$base" ]; then echo "skip $f"; continue; fi
  echo "$(date +%H:%M:%S) downloading $f"
  for attempt in 1 2 3; do
    kaggle competitions download -c $COMP -f "$f" -p "$dest" 2>&1 | grep -v -i "warn\|urllib3" | tr '\r' '\n' | tail -1
    if [ -s "$dest/$base" ] || [ -s "$dest/$base.zip" ]; then break; fi
    echo "retry $attempt"; sleep 5
  done
  if [ -f "$dest/$base.zip" ]; then (cd "$dest" && unzip -o -q "$base.zip" && rm "$base.zip"); fi
done
echo "$(date +%H:%M:%S) ALL DONE"
