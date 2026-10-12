# 3rd WEAR Dataset Challenge @HASCA 2026 (Kaggle)

Human activity recognition on the WEAR outdoor-sports dataset: 19 classes (null + 18 workouts) predicted from
1-second windows of a **single** wrist/leg accelerometer (50 x 3) plus 15 frame-wise VideoMAEv2-Base embeddings
(768-d) from the egocentric camera. Test participants are unseen. Metric: macro-F1.
Competition: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026

**Result (Oct 12 2026 endgame):** public leaderboard macro-F1 **0.93459** (`subs_lb/vote_final_nw.csv`, selected as the final submission; 10th place on the public LB at selection time). See *Endgame* below.

**Earlier result (Sep 2026, per-window models only):** public leaderboard macro-F1 **0.785** (`subs/sub_m_all3.csv`, 11th of 83 teams, Sep 2026) with an equal-weight ensemble of
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


## Endgame (Oct 10-12 2026): timeline reconstruction + our window models
By October the leaderboard had moved to *timeline reconstruction*: public notebooks link the shuffled 1-second test
tiles of each participant through the video embeddings, propagate labels over a per-subject kNN graph, enforce
per-subject class counts (each workout once per session) and refine bout boundaries. The host did not forbid it
(prizes had already closed). The strongest public pipeline was goodpjw2008 "Part 2: Ranked Links" (nested OOF 0.9308,
public 0.92990; copies under `endgame/ref_public/`).

What we did (all code in `endgame/`):
1. `work/tiles.py` rebuilds the public pipeline's 69,326 test-like train tiles exactly (sha1 asserts pass) and
   `work/map_ours.py` maps our own September window models onto those tiles. On their tiles our window blend scores
   0.754 macro-F1 vs 0.724 for the public window nets; through label propagation + count calibration
   (`work/proxy.py`) 0.863 vs 0.829.
2. `kern/mkfork.py` builds forks of the Part 2 notebook that probability-average our window log-probs into its
   window input (2:1 with theirs by default), optionally add them as count-prior features (`--counts`), fill the
   stacker's fusion/IMU slots (`--fullrep`), whiten the kNN video embedding (`--whiten`) and offset link seeds
   (`--seedoff`). Our tile log-probs live in the private Kaggle dataset `rvndyerwin/wear-ours-tile-logp`
   (`kds/`). `kern/qrun.sh` keeps Kaggle CPU slots filled from a queue; `kern/poll.sh` fetches finished outputs.
3. 60+ full pipeline runs (about 3 h each on Kaggle CPU) were combined by label voting (`work/evalruns.py vote`),
   scored exactly on the out-of-fold tiles. `work/evalclean.py` re-scores excluding the documented train label
   errors; `results/runs_summary.csv` lists every run.

Findings:
* Our window evidence moves nested OOF only slightly (the pipeline compresses window gains), but single runs with it
  score 0.929-0.934 public vs 0.926-0.930 for public reruns, and votes of our runs 0.933-0.935.
* Whitening the kNN embedding is +0.002 OOF broadly (15/22 subjects) yet about -0.002 on the public LB in every
  paired comparison; the final pick is the vote of 25 unwhitened runs (public 0.93459, OOF 0.9306).
* Rejected: stacking window models before propagation, post-hoc bout swaps, cleaned training labels, votes that mix
  in public-input runs.
* `docs/endgame_decision_doc.md` is the planning document; `docs/public_notebook_maps.txt` maps each public notebook.
