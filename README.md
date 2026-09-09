# 3rd WEAR Dataset Challenge @HASCA 2026 (Kaggle)

Human activity recognition on the WEAR outdoor-sports dataset: 19 classes (null + 18 workouts) predicted from
1-second windows of a **single** wrist/leg accelerometer (50 x 3) plus 15 frame-wise VideoMAEv2-Base embeddings
(768-d) from the egocentric camera. Test participants are unseen. Metric: macro-F1.
Competition: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026

**Result:** public leaderboard macro-F1 **0.785** (`subs/sub_m_all3.csv`, 11th of 83 teams, Sep 2026) with an equal-weight ensemble of
~16 models plus a LightGBM stacker. Prize eligibility had already closed (HASCA technical-report deadline July 5 2026),
so this was a leaderboard-only effort.

## What mattered (see `docs/2026-09-03-wear-design.md` for the full log)
* **Subject-wise CV was misleading for video.** Training participants share recording locations, so held-out subjects
  always had a "scene twin" in training (per-subject mean-embedding cosine 0.84-0.96) while the largest test subject
  had none (0.69). Clustering subjects by scene similarity gave location-proxy folds (`--foldset 2`) that track the LB.
* **Scene-invariant video representations.** Per-window instance normalisation of the frame embeddings (dynamics
  only), heavy input dropout (0.7), and a small regularised "static path" on the centred per-window mean turned video
  from a liability on new locations into the strongest signal (NN alone: LB 0.615 -> 0.738+).
* **Hybrid inertial modelling.** 1D-CNN on raw signals + MLP over ~300 hand-crafted time/frequency features
  (`common.py`, `features_v2.py`) beat either alone. LightGBM on the same features adds diversity.
* **Macro-F1 decision tuning.** Per-class log-prob offsets tuned on OOF (+0.03..0.10), validated fold-honestly.
  Equal-weight broad blends beat OOF-optimised weights on the LB. EM label-shift adaptation hurt.

## Pipeline
| step | script |
|---|---|
| download competition files (needs Kaggle API token) | `download.sh` |
| CSV -> npy caches + stride-25 window index | `prep.py` |
| hand-crafted features v1 / v2, pooled video stats, video motion feats | `features.py`, `features_v2.py`, `features_vid_diff.py` |
| float16 concatenated video cache for NN training | `build_video_cache.py` |
| LightGBM (5-fold by subject, F1 early stopping) | `run_lgb.py --foldset 2 --stride 25 --feat_v2 ...` |
| PyTorch fusion NN (MPS) | `train_nn.py --foldset 2 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.7` |
| level-2 stacker on OOF log-probs | `stack.py` |
| blend + bias tuning + submission file | `blend.py --foldset 2 --tag X --models a:1 b:2 ...` |
| submit | `./submit.sh subs/sub_X.csv "message"` |

Experiment log: `results.log`; run logs: `logs_*.txt`; submissions: `subs/`; public notebooks studied: `ref/`.
Prediction artefacts (`preds/`: OOF + test probabilities per model) are attached to the GitHub release, not tracked.

## Resume from scratch
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
./download.sh && .venv/bin/python prep.py && .venv/bin/python features.py && .venv/bin/python features_v2.py \
  && .venv/bin/python features_vid_diff.py && .venv/bin/python build_video_cache.py
```
Data licence: WEAR is CC BY-NC-SA 4.0 (Bock et al., IMWUT 2024) and is not redistributed here.
