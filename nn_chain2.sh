#!/bin/zsh
cd "${0:A:h}"
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs/logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs/logs_$tag.txt 2>&1; }
run f2_nnF --folds 0,1 --epochs 3 --vid_center --vid_drop 0.5
run f2_nnC --folds 0,1 --epochs 3 --vid_inorm
run f2_nnCF --folds 0,1 --epochs 3 --vid_inorm --vid_drop 0.5
run f2_nnD --folds 0,1 --epochs 3 --no_video
run f2_nnB --folds 0,1 --epochs 3 --vid_center
echo CHAIN DONE
