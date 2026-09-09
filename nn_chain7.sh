#!/bin/zsh
cd ~/kaggle/wear
while ! grep -q "CHAIN6 DONE" logs_nn_chain6.txt 2>/dev/null; do sleep 30; done
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs_$tag.txt 2>&1; }
run f2_nnH10_static_vd07 --folds 0,1 --epochs 3 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.7
run f2_nnHCF7_full --epochs 3 --hand_feats --vid_inorm --vid_drop 0.7
run f2_nnHS_full --epochs 3 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.5
run f2_nnH8_vd08 --folds 0,1 --epochs 3 --hand_feats --vid_inorm --vid_drop 0.8
run f2_nnHCF7_full_s1 --epochs 3 --hand_feats --vid_inorm --vid_drop 0.7 --seed 1
run f2_nnHS_full_s1 --epochs 3 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.5 --seed 1
run f2_nnHCF7_full_s2 --epochs 3 --hand_feats --vid_inorm --vid_drop 0.7 --seed 2
run f2_nnHD_full_s1 --epochs 3 --no_video --hand_feats --seed 1
echo CHAIN7 DONE
