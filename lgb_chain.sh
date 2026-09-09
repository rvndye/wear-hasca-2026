#!/bin/zsh
cd ~/kaggle/wear
run() { tag=$1; shift; grep -q "OOF macro" logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python run_lgb.py --tag $tag --foldset 2 "$@" > logs_$tag.txt 2>&1; }
run f2_lgb_noVideo_v2 --lr 0.05 --stride 50 --no_video --feat_v2
run f2_lgb_noVideo_v2_s25_nw --lr 0.05 --stride 25 --no_video --feat_v2 --null_weight 0.3
run f2_lgb_best_v2_s25 --lr 0.05 --stride 25 --vid_center --vid_diff --null_weight 0.3 --feat_v2
run f2_lgb_noVideo_v2_l31 --lr 0.03 --stride 25 --no_video --feat_v2 --leaves 31
echo LGBCHAIN DONE
