#!/bin/zsh
# after overnight chains finish, build candidate blends (fast) and log them
cd "${0:A:h}"
while ! grep -q "CHAIN5 DONE" logs/logs_nn_chain5.txt 2>/dev/null || ! grep -q "LGBCHAIN DONE" logs/logs_lgb_chain.txt 2>/dev/null; do sleep 60; done
b() { tag=$1; shift; echo "== $tag $@" >> logs/logs_morning.txt; nice -n 5 .venv/bin/python blend.py --foldset 2 --tag $tag --models "$@" 2>&1 | grep -E "BLEND" >> logs/logs_morning.txt; }
b m_lgbF2 f2_lgb_noVideo_v2_s25_nw:1 f2_lgb_best_v2_s25:1 f2_lgb_noVideo_v2_l31:1 f2_lgb_best_s25:0.5
b m_nnH f2_nnHCF_full:1 f2_nnHCF_full_s1:1 f2_nnHD_full:1
b m_nnH_lgb f2_nnHCF_full:1 f2_nnHCF_full_s1:1 f2_nnHD_full:1 f2_lgb_noVideo_v2_s25_nw:1 f2_lgb_best_v2_s25:1 f2_lgb_noVideo_v2_l31:1
b m_all f2_nnHCF_full:1 f2_nnHCF_full_s1:1 f2_nnHD_full:1 f2_nnCF_full:1 f2_nnCF_full_s1:1 f2_nnD_full:1 f2_lgb_noVideo_v2_s25_nw:1 f2_lgb_best_v2_s25:1 f2_lgb_noVideo_v2_l31:1 lgb_center:1 lgb_noVideo:1
b m_all_lgbheavy f2_nnHCF_full:1 f2_nnHCF_full_s1:1 f2_nnHD_full:1 f2_nnD_full:1 f2_lgb_noVideo_v2_s25_nw:2 f2_lgb_best_v2_s25:2 f2_lgb_noVideo_v2_l31:2 lgb_center:2 lgb_noVideo:2
b m_inertial_only f2_nnHD_full:1 f2_nnD_full:1 f2_lgb_noVideo_v2_s25_nw:1 f2_lgb_noVideo_v2_l31:1 lgb_noVideo:1
echo "MORNING DONE $(date -u)" >> logs/logs_morning.txt
