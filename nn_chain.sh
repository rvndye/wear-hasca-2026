#!/bin/zsh
cd ~/kaggle/wear
run() { tag=$1; shift; [ -f preds/oof_$tag.npy ] || grep -q "PARTIAL\|OOF macro" logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag "$@" > logs_$tag.txt 2>&1; }
run nn_full_F --epochs 3 --vid_center --vid_drop 0.5
run nnN_combo --folds 0,2 --epochs 3 --vid_center --vid_drop 0.5 --mixup 0.4
run nnM_inorm_center --folds 0,2 --epochs 3 --vid_inorm --vid_center --null_weight 0.3
run nnL_inorm_static --folds 0,2 --epochs 3 --vid_inorm --vid_static --vid_center
run nn_full_N --epochs 3 --vid_center --vid_drop 0.5 --mixup 0.4 --seed 1
run nn_full_O --epochs 3 --vid_center --vid_drop 0.5 --mixup 0.4 --null_weight 0.3 --seed 2
echo CHAIN DONE
