#!/bin/zsh
cd ~/kaggle/wear
while ! grep -q "CHAIN DONE" logs_nn_chain2.txt 2>/dev/null; do sleep 20; done
run() { tag=$1; shift; grep -q "OOF macro" logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs_$tag.txt 2>&1; }
run f2_nnD_full --epochs 3 --no_video
run f2_nnCF_full --epochs 3 --vid_inorm --vid_drop 0.5
run f2_nnF_full --epochs 3 --vid_center --vid_drop 0.5
echo CHAIN3 DONE
