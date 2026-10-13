#!/bin/zsh
cd "${0:A:h}"
while ! grep -q "CHAIN3 DONE" logs/logs_nn_chain3.txt 2>/dev/null; do sleep 20; done
run() { tag=$1; shift; grep -q "PARTIAL\|OOF macro" logs/logs_$tag.txt 2>/dev/null && { echo "skip $tag"; return; }; .venv/bin/python train_nn.py --tag $tag --foldset 2 "$@" > logs/logs_$tag.txt 2>&1; }
run f2_nnHD --folds 0,1 --epochs 3 --no_video --hand_feats
run f2_nnHCF --folds 0,1 --epochs 3 --vid_inorm --vid_drop 0.5 --hand_feats
echo CHAIN4 DONE
