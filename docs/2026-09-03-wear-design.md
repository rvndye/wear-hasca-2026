# 3rd WEAR Dataset Challenge (HASCA 2026) - design note

Date: 2026-09-03. Work dir `~/kaggle/wear` (venv `.venv`, python 3.12, torch MPS, lightgbm).

## Task
Predict one of 19 classes (0 = null + 18 workout activities) for 12,234 test windows.
Each test window = 1 s of ONE accelerometer (50 x 3, sensor location given in
`test_meta_data.csv`) + 15 VideoMAEv2-Base frame features (768-d, stored (N, 768, 15)).
Test subjects 22-25 are unseen. Metric: macro F1 over the 19 classes. Public/private LB split
unknown. 10 submissions/day.

## Prize eligibility (checked 2026-09-03)
Technical report deadline was July 5 2026 (HASCA); host post "Technical Report Deadline
Passed" (2 months ago) confirms it. Prizes/final ranking are closed to new entrants;
the Kaggle LB stays open until ~early Oct 2026. We compete for LB position only.

## Data facts
- Train: 24 recordings / 22 subjects (`sbj_14_2`, `sbj_0_2` are repeat sessions).
  Inertial 50 Hz, 4 sensors x 3 axes (g units), labels per sample (NaN = null, ~40%).
  Video 30 FPS, 768-d per frame; durations align exactly (frames/30 == samples/50).
- Alignment: inertial start s -> video start v = s*3//5; the 15 test frames are v+8..v+22
  (the only 15 frames whose 16-frame clips stay inside the 1 s window).
- Test window counts per subject equal session seconds -> non-overlapping 1 s tiling,
  one random sensor per second. No duplicate video windows, no shared frames.
- Some recordings have NaN sensor values (per-sensor validity mask needed).
- Segments are long (30-180 s), so windows are almost always pure-label; use majority label.

## Approach
1. `prep.py`: CSV -> `cache/<rec>_inertial.npy` float32 (T,4,3), `cache/<rec>_labels.npy` int8,
   video used via mmap. Window index (stride 25 = 0.5 s) with majority label.
2. Model A (fast baseline, first submission): LightGBM on hand-crafted inertial features
   (per-axis + magnitude stats, FFT bands, jerk, autocorr) + PCA-reduced video mean/std +
   sensor-location id. 5-fold GroupKFold by subject.
3. Model B: PyTorch fusion net (1D-CNN inertial encoder with sensor embedding + attention
   pooling over the 15 video frames), same folds, trained on MPS. Multi-sensor training
   (all 4 sensors per window as separate samples).
4. Blend probabilities (OOF-tuned weights), tune class-prior / null bias on OOF macro-F1.
5. Deliberately NOT used unless the user asks: reconstructing the test timeline by chaining
   windows through video-feature similarity (the host removed boundary frames to prevent
   exactly this; legal under the written rules but against the challenge design).

## Validation
OOF macro-F1 with GroupKFold(5) by subject on stride-25 windows, all 4 sensors.
Log every experiment in `results.log`.

## Status log

### 2026-09-03 (day 1) - LB probes and what they taught us
Public LB (10 subs used): lgb_quick 0.655 | blend_lgb2 0.724 | **blend_lgb3 0.7415 (best)** |
blend_lgb4 0.733 | nn_full_F + lgb 0.701 | nn_full_F alone 0.615 | same with null bias -0.6: 0.703 |
lgb_noVideo alone 0.673 | f2_nnD_full (NN inertial-only) pending.

CV vs LB: inertial-only LightGBM CV 0.589 -> LB 0.673 (+0.08); video-heavy NN CV 0.665 -> LB 0.615 (-0.05).
Cause: training subjects share recording locations (per-subject mean VideoMAE cosine 0.84-0.96 to a
training twin) but test sbj 22 (42% of test windows) has max cosine 0.69 to any training subject.
Subject-wise CV therefore rewards scene memorisation. Fold set 2 (`--foldset 2`, scene-cluster folds)
is harder but still not as harsh as the real test shift.

Decisions: go inertial-heavy; use video only through scene-invariant representations (per-window
instance norm = dynamics only, `--vid_inorm`, or per-subject centering + heavy input dropout).
Null-bias magnitude barely matters on LB. EM prior adaptation hurts (rejected).
Bias tuning on OOF is worth +0.03..0.10 macro F1 and transfers to LB.

Next (day 2 submissions): f2 blends of inertial LightGBM (v2 features) + dynamics-only NN
(f2_nnCF_full) with small video weight; probe NN-inorm alone; hand-feature NN branch (f2_nnHCF).

### 2026-09-03 evening - fold set 2 results (raw / bias-tuned macro F1, screening folds 0,1 unless "full")
NN: base(center+vdrop) 0.631/0.649 | inorm 0.591/0.621 | inorm+vdrop 0.643/0.650 | inertial-only 0.550/0.573 |
inertial-only+hand feats 0.599/0.609 | **hand feats + inorm + vdrop 0.673/0.680; full 5-fold 0.690/0.694 (f2_nnHCF_full)**.
LGB: noVideo 0.597/0.612 | noVideo v2 feats 0.607/0.625 | center+diff+nw (best_s25) 0.620/0.660.
Candidate blends (honest tuned): blend3 + f2_nnHCF_full + f2_nnD_full 0.742; f2_nnHCF_full + lgb_noVideo 0.716.
Scheduled for 00:01 UTC Sep 4 (sched_list.txt): those two plus f2_nnHCF_full alone as a probe.
Overnight: f2_nnHD_full, f2_nnCF_full_s1, f2_nnHCF_full_s1 (chain5); LGB v2-feature variants (lgb_chain.sh); morning.sh blends.

### 2026-09-04 morning - LB results of the scheduled submissions
cand_v3_hcf2 (f2_nnHCF_full x2 + f2_nnHD_full + 2 inertial LGB v2) **0.774 (best, ~rank 11)** |
cand_lgb3_hcf_nnD 0.769 | f2_nnHCF_full alone 0.738 (CV 0.690: the hand-feature + instance-normed-video
NN transfers to test like the inertial models do, unlike the centered-video NN).
Overnight seeds: f2_nnHCF_full_s1 0.691/0.695, f2_nnCF_full_s1 0.659/0.663, f2_nnHD_full 0.624/0.630;
LGB v2 l31 0.613/0.626. Morning blends: m_all (11 models) 0.729 raw / 0.747 honest -> submitted.
Next: screen mixup / 4 epochs / vid_drop 0.7 / dim 256 / static path / rot 30 / null weight on the HCF config
(nn_chain6.sh), then full runs + seeds of the winners.

### 2026-09-04 midday
LB: m_all 0.7807 | m_all_w (OOF-optimised weights) 0.7763 (worse: keep equal weights, broad blends) |
**m_all + stack1 (LightGBM on NN/LGB OOF log-probs, w3) 0.7828 (best)**. stack1 honest 0.740, stack2 (+feats) 0.737.
Screening on HCF config (folds 0,1): mixup 0.669/0.675 (no gain) | 4 epochs 0.671/0.677 (no gain) |
**vid_drop 0.7 0.688/0.700 (+0.015)** -> full runs with vd 0.7 (3 seeds) queued; screening vd 0.8 / 0.9.
Screening (cont., folds 0,1 raw/tuned): dim 256 0.675/0.682 (neutral) | rot 30 deg 0.669/0.677 (no gain) |
null weight 0.5 0.675/0.681 (neutral) | **static path (inorm dynamics + centered per-window mean via dropout MLP) 0.697/0.716 (best)**.
stack3 (leaves 63, +l31 LGB) 0.7415/0.741; m_all + stack1 + stack3 honest 0.7515.
Full runs: HCF7 (vd 0.7) 0.702/0.713 | **HS7 (static path + vd 0.7) 0.726/0.741 (best single)**.
Blends: m_all2 (+HCF7) honest 0.752 | m_all3 (+HS7 x2, HCF7 x2) honest 0.757. auto_submit.sh probes HS7 alone, then m_all3 if probe >= 0.735 else m_all2.
