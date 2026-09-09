#!/bin/zsh
cd ~/kaggle/wear
while ! grep -q "CHAIN4 DONE" logs_nn_chain4.txt 2>/dev/null; do sleep 20; done
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs_$tag.txt 2>&1; }
run f2_nnHCF_full --epochs 3 --vid_inorm --vid_drop 0.5 --hand_feats
run f2_nnHD_full --epochs 3 --no_video --hand_feats
run f2_nnCF_full_s1 --epochs 3 --vid_inorm --vid_drop 0.5 --seed 1
run f2_nnHCF_full_s1 --epochs 3 --vid_inorm --vid_drop 0.5 --hand_feats --seed 1
echo CHAIN5 DONE
