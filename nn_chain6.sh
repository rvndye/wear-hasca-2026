#!/bin/zsh
cd "${0:A:h}"
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs/logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs/logs_$tag.txt 2>&1; }
B="--folds 0,1 --hand_feats --vid_inorm"
run f2_nnH1_mixup $=B --epochs 3 --vid_drop 0.5 --mixup 0.4
run f2_nnH2_ep4 $=B --epochs 4 --vid_drop 0.5
run f2_nnH4_vd07 $=B --epochs 3 --vid_drop 0.7
run f2_nnH5_dim256 $=B --epochs 3 --vid_drop 0.5 --dim 256
run f2_nnH3_static $=B --epochs 3 --vid_drop 0.5 --vid_static --vid_center
run f2_nnH6_rot30 $=B --epochs 3 --vid_drop 0.5 --rot_deg 30
run f2_nnH7_nw $=B --epochs 3 --vid_drop 0.5 --null_weight 0.5
echo CHAIN6 DONE
