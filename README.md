# 3rd WEAR Dataset Challenge @HASCA 2026 (Kaggle)

Human activity recognition on the WEAR outdoor-sports dataset: 19 classes (null + 18 workouts) predicted from
1-second windows of a **single** wrist/leg accelerometer (50 x 3) plus 15 frame-wise VideoMAEv2-Base embeddings
(768-d) from the egocentric camera. Test participants are unseen. Metric: macro-F1.
Competition: https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026

**Final result: 9th of 236 teams on the private leaderboard** (macro-F1 **0.92718**; public 0.93459) with
`subs_lb/vote_final_nw.csv`, a label vote of 25 timeline-reconstruction pipeline runs that use our window models
(see *Endgame*). Prize eligibility had closed before we joined (HASCA technical-report deadline July 5 2026), so this
was a leaderboard-only effort.

**September version (per-window models only):** public macro-F1 0.785 (private 0.770; 11th of 83 teams on the public
leaderboard at the time) with `subs/sub_m_all3.csv`: 13 base models (neural nets and LightGBM) with hand-set weights of
1-2, plus two LightGBM stackers.

All submissions with their public and private scores: `endgame/results/lb_scores.csv`.

## What mattered in September (full log: `docs/2026-09-03-wear-design.md`)
* **Subject-wise CV was misleading for video.** Training participants share recording locations, so held-out subjects
  always had a "scene twin" in training (per-subject mean-embedding cosine 0.84-0.96) while the largest test subject
  had none (0.69). Clustering subjects by scene similarity gave location-proxy folds (`--foldset 2`) that track the LB.
* **Scene-invariant video representations.** Per-window instance normalisation of the frame embeddings (dynamics
  only), heavy input dropout (0.7), and a small regularised "static path" on the centred per-window mean turned video
  from a liability on new locations into the strongest signal (NN alone: LB 0.615 -> 0.738+).
* **Hybrid inertial modelling.** 1D-CNN on raw signals + MLP over ~300 hand-crafted time/frequency features
  (`common.py`, `features_v2.py`) beat either alone. LightGBM on the same features adds diversity.
* **Macro-F1 decision tuning.** Per-class log-prob offsets tuned on OOF (+0.03..0.10), validated fold-honestly.
  Broad blends with simple weights beat OOF-optimised weights on the LB. EM label-shift adaptation hurt.

## Pipeline (September)
| step | script |
|---|---|
| download competition files (needs a Kaggle API token, see *Setup*) | `download.sh` |
| CSV -> npy caches + stride-25 window index | `prep.py` |
| hand-crafted features v1 / v2, pooled video stats, video motion feats | `features.py`, `features_v2.py`, `features_vid_diff.py` |
| float16 concatenated video cache for NN training | `build_video_cache.py` |
| LightGBM (5-fold by subject, F1 early stopping) | `run_lgb.py --foldset 2 --stride 25 --feat_v2 ...` |
| PyTorch fusion NN (Apple MPS or CPU) | `train_nn.py --foldset 2 --hand_feats --vid_inorm --vid_static --vid_center --vid_drop 0.7` |
| level-2 stacker on OOF log-probs | `stack.py` |
| blend + bias tuning + submission file | `blend.py --foldset 2 --tag X --models a:1 b:2 ...` |
| submit | `./submit.sh subs/sub_X.csv "message"` |

Experiment log: `results.log`; run logs: `logs/`; sequential experiment queues with the exact flags of most models:
`nn_chain*.sh`, `lgb_chain.sh` (the rest, including the stackers, are recorded as JSON in `results.log`); submissions: `subs/`.
Prediction artefacts (OOF + test probabilities per model) are attached to the GitHub release v1.0, not tracked.

## Setup
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# Kaggle auth: the scripts read a Kaggle API token from ~/.kaggle/access_token and pass it as KAGGLE_API_TOKEN;
# the dummy KAGGLE_USERNAME/KAGGLE_KEY values only satisfy the older CLI's legacy credential check.
export KAGGLE_USERNAME=x KAGGLE_KEY=x KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
./download.sh && .venv/bin/python prep.py && .venv/bin/python features.py && .venv/bin/python features_v2.py \
  && .venv/bin/python features_vid_diff.py && .venv/bin/python build_video_cache.py
```
Data and caches go to `data/` and `cache/` in the repo (about 14 GB; set `WEAR_ROOT` to use another directory).

## Endgame (Oct 10-12 2026): timeline reconstruction + our window models
By October the leaderboard had moved to *timeline reconstruction*: public notebooks link the shuffled 1-second test
tiles of each participant through the video embeddings, propagate labels over a per-subject kNN graph, enforce
per-subject class counts (each workout once per session) and refine bout boundaries. The host did not forbid it
(prizes had already closed). The strongest public pipeline was goodpjw2008's "Part 2: Ranked Links" (nested OOF
0.9308, public 0.92990), which we forked.

What we did (code in `endgame/`):
1. `work/tiles.py` rebuilds the public pipeline's 69,326 test-like train tiles exactly (its sha1 asserts pass) and
   `work/map_ours.py` maps our September window models onto those tiles. On their tiles our window blend scores
   0.754 macro-F1 vs 0.724 for the public window nets; after label propagation + count calibration
   (`work/proxy.py`) 0.863 vs 0.829.
2. `kern/mkfork.py` builds forks of the Part 2 notebook that probability-average our window log-probs into its window
   input with weight w (0.67 in most runs; 0.8 and 1.0 in some). Options: our probabilities as count-prior features
   (`--counts`), in the stacker's fusion/IMU slots (`--fullrep`), a whitened kNN video embedding (`--whiten`), and
   offset link-scorer seeds (`--seedoff`). Our tile log-probs were uploaded as a private Kaggle dataset
   (`wear-ours-tile-logp`, layout in `kds/`). `kern/qrun.sh` keeps Kaggle CPU slots filled from a queue;
   `kern/poll.sh` downloads finished outputs.
3. 57 fork runs (32 unwhitened, 25 whitened; about 3 h each on Kaggle CPU, 2.4 h on GPU) plus an unmodified control
   run were combined by label voting (`work/evalruns.py vote`), scored exactly on the out-of-fold tiles.
   `work/evalclean.py` re-scores excluding documented train label errors. `results/runs_summary.csv` lists every run;
   `results/vote_final_nw_members.txt` lists the 25 runs of the final vote (re-running the vote reproduces
   `subs_lb/vote_final_nw.csv` exactly).

Findings (public / private macro-F1):
* Our window evidence moved nested OOF only slightly (the pipeline compresses window gains), but runs with it mostly
  beat the public pipeline (all of them on the private LB; all votes and 3 of 4 single runs on the public LB): single runs 0.9286-0.9334 / 0.9232-0.9278 and votes of our runs 0.9320-0.9346 / 0.9265-0.9290,
  vs 0.92990 / 0.92130 for the public pipeline's own output file.
* Whitening the kNN video embedding was +0.002 OOF (15 of 22 subjects) but 0.0013-0.0022 *lower* on the public LB
  in every paired comparison, so the final pick was the unwhitened vote. The private LB sided with OOF: the whitened
  25-run vote scored 0.92902 private vs 0.92718 (same 9th place). Lesson: on about 75 disputed test tiles, a public
  split this small could not overrule broad OOF evidence.
* Rejected: stacking window models before propagation (hurts propagation), post-hoc bout swaps, cleaned training
  labels, and votes mixing in public-input runs.
* `endgame/docs/endgame_decision_doc.md` is the planning note written before the runs.

### Running the endgame code
It needs the public dataset `goodpjw2008/wear-hasca-part2-gpu-outputs` (about 1.5 GB) attached to the Kaggle forks,
our own tile log-probs re-uploaded under your account (set `KAGGLE_OWNER`), the base notebook pulled with
`endgame/fetch_refs.sh`, and a working directory for data and kernel outputs (`WEAR_LB`, default `~/kaggle/wear_lb`).
Our September prediction arrays come from the v1.0 release (set `WEAR_PREDS`).

## Credits
* WEAR dataset: Marius Bock, Hilde Kuehne, Kristof Van Laerhoven, Michael Moeller. *WEAR: An Outdoor Sports Dataset
  for Wearable and Egocentric Activity Recognition.* Proc. ACM IMWUT 8(4), 2024, article 175.
  https://doi.org/10.1145/3699776. Licensed CC BY-NC-SA 4.0; the data is not redistributed here.
* Public Kaggle notebooks the endgame builds on (fetched by `endgame/fetch_refs.sh`; not in the current tree, though
  commits before Oct 13 2026 contain unmodified copies, see `THIRD_PARTY_NOTICES.md`):
  goodpjw2008, "Part 2: Ranked Links" (forked) and "Learned Links + Counts"; Briano (woominyo), "Timeline + Graph"
  (base pipeline); Hanh Tran (honghanhhh), "WEAR@HASCA LB 0.9" (timeline links); Prabhav Semwal (yeashusemwal),
  "Ranked Timelines + Boundary Refiner"; Jiwei Liu, inference packaging; ItsmeSujan (sujanmajhisuzan), bagged
  boundary refiner; habakan, "What egocentric video embeddings encode"
  (https://github.com/habakan/what-egocentric-embeddings-encode).
* September notebooks studied (copies in early commits only): lakhindarpal/3rd-wear-dataset-challenge-hasca-2026,
  nomannic19/ts-emb-3wdc-temporal-fusion-ensemble, stmugiwara/wear-v7-full5.
* Built with the assistance of an AI coding assistant (Claude, Anthropic).

## License
No license has been chosen yet for this repository's own code, so default copyright applies. The submission and
prediction files are derived from the WEAR data and fall under CC BY-NC-SA 4.0. Third-party notebooks keep their
authors' licenses.
