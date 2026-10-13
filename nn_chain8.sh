#!/bin/zsh
cd "${0:A:h}"
while ! grep -q "OOF macro" logs/logs_f2_nnHCF7_full.txt 2>/dev/null; do sleep 20; done
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs/logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs/logs_$tag.txt 2>&1; }
S="--epochs 3 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.7"
run f2_nnHS7_full $=S
run f2_nnHS7_full_s1 $=S --seed 1
run f2_nnHCF7_full_s1 --epochs 3 --hand_feats --vid_inorm --vid_drop 0.7 --seed 1
run f2_nnHS7_full_s2 $=S --seed 2
run f2_nnHD_full_s1 --epochs 3 --no_video --hand_feats --seed 1
run f2_nnH8_vd08 --folds 0,1 --epochs 3 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.85
echo CHAIN8 DONE
