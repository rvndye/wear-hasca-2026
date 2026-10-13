# WEAR@HASCA endgame: decision document (about 54 h left, deadline Oct 12 21:59 UTC)

> Planning note written on Oct 10 2026, before the runs, with the help of an AI coding assistant; kept as written apart from small edits for publication
> (local paths replaced with repo paths, a local-machine note removed).
> What changed in execution: the dataset was named `wear-ours-tile-logp` (not `wear-ours-tileprobs`), and
> `endgame/kern/mkfork.py` probability-averages our window probabilities with weight w = 0.67 / 0.8 / 1.0 instead
> of the log-linear blend at a = 0.6 / 0.75 sketched below. Paths like `~/kaggle/wear_lb/...` refer to the local
> working directory (`WEAR_LB`). Outcome: README *Endgame* (9th on the private leaderboard, macro-F1 0.92718).

## 0. Decisions at a glance

1. **Submit `submission_lb_0_92990.csv` now.** It costs no compute and takes our public score from 0.78492 (best so far, `sub_m_all3`) to 0.92990. The file is in the public dataset `goodpjw2008/wear-hasca-part2-gpu-outputs` (90 KB). I listed the dataset through the API today; I did not download the file.
2. **Base: goodpjw2008 "Part 2: Ranked Links".** It is a CPU-only kernel that uses the GPU outputs of that dataset. We do not retrain anything on GPU.
3. **Plug-in: our `m_all4` probabilities.** Blend them log-linearly into `d['logp']`/`t['logp']` in cell 13 (Option A), with a = 0.6 and a = 0.75.
   - I checked the alignment locally today: tile hashes, labels, coverage, limb order and test order all match.
   - In a local propagation proxy the gain carries through: +0.034 to +0.040 over their window blend, and every fold improves.
4. **Rank every variant by nested OOF** against a control run on the same platform. Use the leaderboard only for paired comparisons against that control. The final two picks are the best OOF-validated variant or vote, plus a hedge.

---

## 1. Lineage and comparison

### Which notebook builds on which

```
honghanhhh "Hungarian Chain Viterbi LB 0.74"  (L0 links: candidates, 12 pair feats, LGB, Hungarian)
  └─> woominyo/Briano "LB 0.890 Timeline + Graph"  (public 0.90315; tiles, folds, window nets, S3/T, L2 ridge links, graph, Sinkhorn-97)
        ├─> jiweiliu public-fast-gpu-inference  (inference-only repack; reads jiweiliu/wear-timeline-graph-models; ~0.903, "0.9077 on board")
        │     └─> jiweiliu wear-soft-context-private-inference  (5 jiweiliu "*-private-models" datasets; LB unknown)
        └─> goodpjw2008 Part 1 "Learned Links + Counts"  (adds two-tower, SSL ridge, L3, bagging, g1.5, learned counts, adapted T, refiner)
              ├─> sujanmajhisuzan bagged-boundary-refiner  (Part 1 with only cell 38 changed; no measured LB)
              ├─> honghanhhh wear-hasca-lb-0-9 v2  (core stack = Part 1's 0.92718 version; v1 0.92485, v2 unknown)
              └─> goodpjw2008 Part 2 "Ranked Links"  (LambdaRank L3, 60 negatives, 3 two-tower seeds; CPU kernel)
                    consumes dataset goodpjw2008/wear-hasca-part2-gpu-outputs (= GPU stages of the author's own 0.92990 run)
yeashusemwal "Ranked Timelines + Boundary Refiner" (0.92875): thin driver over an unnamed dataset (hasca_pipeline.py + 6 npz); idea source for Part 2's ranker, no shared code
kansukehabano "What egocentric video embeddings encode" (0.894): independent and leak-free (no links), video-kNN propagation + count band; reads kansukehabano/wear-video-graph-inputs
```

No notebook reads another notebook's kernel output. The only artifact passed between notebooks is the Part 2 GPU-outputs dataset.

### Notebook comparison

| Notebook | Public LB | Runs on | Outside dependency | Verdict |
|---|---|---|---|---|
| goodpjw2008 **Part 2** | 0.92990 (author's GB10 run; file is shipped). Kaggle CPU rerun scored 0.92620 with 98.7% identical labels | CPU, about 3 h (markdown claim, no logs) | gpu-outputs dataset (public; checked today) | **BASE** |
| goodpjw2008 Part 1 | 0.92942 (own machine); 0.92444 when rerun on Kaggle T4 | T4x2, about 2 h, retrains everything | none | Fallback if the Part 2 dataset disappears |
| yeashusemwal ranked timelines | 0.92875 | CPU | Unnamed dataset; all logic in hasca_pipeline.py | Skip. Its submission.csv can still be a voting member |
| sujanmajhisuzan | none | T4x2 | none | Skip. Unconditional test island filter and threshold bias |
| honghanhhh lb-0-9 v2 | v1 0.92485, v2 unknown | T4x2 | none | Skip |
| woominyo/Briano | 0.90315 | T4x2 | none | Ancestor only |
| jiweiliu public / soft-context | about 0.903 to 0.9077 / unknown | T4x2 inference | jiweiliu datasets, some private | Skip |
| kansukehabano | 0.894 (CV 0.886) | light | kansukehabano dataset | Skip (score ceiling) |

### Stage scores

Nested OOF is from the scoreboards in Parts 1 and 2. Every number is from markdown; none of the notebooks has logged output.

| Stage | Part 2 OOF | Part 1 OOF | LB evidence |
|---|---|---|---|
| Window nets, 4 fusion + 2 IMU, imu_w 0.3 | 0.7239 (I reproduced exactly 0.7239 from the dataset) | 0.7239 | Briano: single-window baselines 0.60 to 0.69. Ours: 0.785 |
| + label propagation + L0 smoothing + Sinkhorn-97 | 0.8464 | 0.8464 | |
| + S3 and T tabular experts | 0.8776 | 0.8776 | |
| + L2 links + graph self-training (= public pipeline) | 0.9040 | 0.9040 | 0.90624 (Briano 0.90315) |
| + L3 links (two-tower + SSL, ranked in Part 2) | 0.9137 | 0.9138 | |
| + link bagging (8 matchings) | 0.9163 | 0.9152 | |
| + kNN g 0.5 to 1.5 | 0.9174 | 0.9170 | |
| + learned count prior (2 passes) | 0.9235 | 0.9232 | |
| + pseudo-label adapted T | 0.9275 | 0.9271 | |
| + boundary refiner | **0.9308** (folds .9366/.9178/.9360/.9177/.9446) | 0.9300 | **0.92990** / 0.92942 |
| Oracle: true counts | 0.9362 | 0.9359 | |

- **Prabhav's LB path:** 0.9077, then 0.9190 (SSL links, adapted teacher, two-pass counts), 0.9237, 0.9272 (8 ranked timelines), 0.9280 (refiner), 0.9286 (4 refiner seeds), 0.92875 (whitening).
- **Habakan's path:** 0.648 (inertial only), 0.830 (propagation), 0.843 (video pooling), 0.852 (30-PC removal), 0.886 CV / 0.894 LB (count band).
- **Noise:** the same code moves the LB by 0.0037 to 0.005. Six Part 2 variants scored 0.92349 to 0.92990, and their OOF did not rank them in the same order.
- **Field on Oct 8:** the top 20 ranged from 0.9454 down to 0.9274. Expect many teams to submit the 0.92990 file.

---

## 2. Base choice: fork Part 2 as a CPU kernel

**Why Part 2:**
- It is the highest measured public score and ships its own result file.
- It needs no GPU, because the window nets and three two-tower seeds come frozen from the dataset. GPU retraining was the main source of the 0.004 to 0.005 run-to-run drift.
- Every stage reruns on CPU, and each run prints nested OOF for every stage.

### Inputs (kernel-metadata)
- `competition_sources`: 3rd-wear-dataset-challenge-hasca-2026
- `dataset_sources`: `goodpjw2008/wear-hasca-part2-gpu-outputs` (pin the current version) and our new private dataset `rvndyerwin/wear-ours-tileprobs` (about 15 to 30 MB, contents in section 3)
- `enable_gpu=false`, `enable_internet=false`

**Inserted first cell:**
```python
import os, glob
from pathlib import Path
hits = glob.glob("/kaggle/input/**/runs/final_s0/oof_raw.npz", recursive=True); assert hits
os.environ["WEAR_ARTIFACTS"] = str(Path(hits[0]).parents[2])   # the notebook's own glob is only one level deep
os.environ["WEAR_ABLATION"] = "1"   # 0 crashes: cell 34 reads ablation keys unconditionally (verified, txt lines 3453-3455)
os.environ["WEAR_DUMP"] = "1"       # keeps dump_stageB.npz (B, P, Q, 8 matchings, logp) in /kaggle/working
# Do NOT set WEAR_SEED in this CPU fork: it shifts the two-tower seeds and can trigger CPU retraining with 300 steps.
```

### Run order (each variant in its own kernel slug, so they run in parallel)
| Kernel | Variant |
|---|---|
| K0 | Control: unmodified apart from the cell above |
| K1 | Option A, a = 0.6 |
| K2 | Option A, a = 0.75 |
| K3 | Option B, w = 0.15 (links unchanged) |

- **Runtime:** about 3 h claimed, with no logs. LK_NEG=60 makes 3.6 times the scorer rows, so budget 4 to 5 h. The session cap is 12 h. Use 4 vCPUs and the default thread settings.
- **Before submitting, check:**
  - the window OOF print in cell 13 (K0 must show 0.7239);
  - the blend-B diagnostic in cell 22 (K0 about 0.8776);
  - the final SCORES table (K0 about 0.9308; small x86 vs ARM differences are expected).

### Outputs to keep (download only the small files)
- `submission.csv`
- `submission_public_pipeline.csv`
- `submission_our_run_lb_0_92990.csv` (K0 only)
- `final_probabilities.npz`: QB_OOF (69326,19) pre-refiner, QB_TEST, REF_OOF, test_labels
- `dump_stageB.npz`
- The log containing the SCORES table and per-fold values

The scratch directory /tmp/wear is deleted at the end of a run, which is fine because everything above lives in /kaggle/working.

---

## 3. Plug-in plan for our per-window probabilities

### 3.1 Alignment, checked locally today
The tile rebuild (`~/kaggle/wear_lb/work/tiles.py`) follows Briano's cell-8 recipe exactly:
- It reproduces 69,326 tiles with fold sizes [14353, 12995, 14625, 12768, 14585] and both sha1 asserts (f81d90085315 / e4177f2589d7).
- `sim['y']` and `sim['sbj']` equal `y` and `grp` in all 6 public `oof_raw.npz` files.

How the two tilings line up:
- **Starts.** Their tile starts are multiples of 50 samples from sample 0 of each recording, so they are multiples of 25. Our index (`prep.py`) uses `arange(0, T-49, 25)` and keeps a start only if `start*3//5+23 <= F`. That is the same video-end rule as theirs (`+8+15`). Recordings are sorted by stem in both, giving 138,632 windows. The mapping is `gidx = offset[rec] + start//25` (`sim_gidx25.npy`), i.e. our even windows.
- **Limbs.** Both use the order [right_arm, right_leg, left_leg, left_arm]. Both define validity the same way: no NaN in that limb's 3 axes over the 50 samples, before interpolation. Their limb is a uniform draw among valid limbs with `default_rng(1234+fold)`. We scored all 4 limbs, so we simply take `ours[gidx, sim['sensor']]`.
- **Coverage.** `ours/m_all4_oof.npy` covers 100% of tiles, with 0 NaN rows. Our majority labels match `sim['y']` (asserted in `map_ours.py`).
- **Test order.** Our test arrays are in `test_meta_data.csv` row order (`train_nn.py`/`blend.py` write `meta['id']` directly). The ids are ascending, so row order equals id order. Argmax agreement with their test blend is 0.8575, against 0.83 on OOF, which confirms the order.
- **Folds.** Our foldset 2 (scene clusters, `SBJ_FOLD2`) assigns each subject id to exactly one fold, so it is **subject-held-out**; sbj_0/sbj_0_2 and sbj_14/sbj_14_2 stay together. The leakage risk raised in the maps does not apply. What remains is second-order: a different set of training subjects per fold. Our scene-cluster folds are harder than theirs, so our OOF should if anything be pessimistic.
- **Calibration.** Our test probabilities are the mean of 5 fold models; their test log-probs are averaged over 5 folds. Same structure. The stored `m_all4` OOF is before bias tuning (window F1 0.7542 vs 0.7546 logged), which is what we want.

**Still unknown:**
- The labelling rule for the test tiles (majority vs centre sample).
- Whether S3, the L2/L3 link scorers and the count ridge respond well to sharper evidence. They retrain in-notebook, but QN_REF and TEST_CFGS were tuned on the authors' evidence.
- How much of the gain survives past the S3/T and link stages.

### 3.2 Measured evidence on the 69,326 tiles
The proxy is Briano's label_prop (k 10, tau 0.1, use_p 0.5, alpha 0.9, 20 iterations) over the final_s0 embedding, then P², then Sinkhorn-97 per subject. It has no L0 smoothing. The script is `endgame/work/proxy.py`.

| Evidence | Window mF1 | LP+Sinkhorn | Per fold (their folds) |
|---|---|---|---|
| Public 4F+2I blend | 0.7239 | 0.8294 | .820 .799 .861 .805 .856 |
| Public fusion only / IMU only | 0.7181 / 0.5997 | 0.8260 / 0.7688 | |
| Ours m_all3 (the LB 0.785 ensemble) | 0.7510 | 0.8611 | |
| **Ours m_all4** | 0.7542 | **0.8632** | .862 .829 .893 .831 .893 |
| Ours nnHS7 x3 seeds / stack4 | 0.7376 / 0.7527 | 0.8447 / 0.8524 | |
| Ours inertial-only / LGB without video | 0.6309 / 0.6039 | 0.8113 / 0.7759 | |
| Public + m_all4, a = 0.5 | 0.7571 | 0.8647 | |
| **Public + m_all4, a = 0.75** | 0.7655 | **0.8695** | .873 .834 .899 .838 .897 |
| 0.25 public + 0.6 m_all4 + 0.15 inertial-only | | 0.8694 | |

- On OOF, the public blend is wrong on 24.0% of tiles and ours on 21.6%; both are wrong on 15.6%. Agreement is 0.83.
- The improvement holds in every fold and grows slightly through propagation, from +0.030 at the window to +0.034 after propagation.
- The authors' oracle ("fix 30% of window errors at random, +0.003 final") suggests strong compression downstream. My estimate for the final nested OOF is **+0.003 to +0.010**. That is an estimate, not a measurement.

### 3.3 "Video should group, IMU should name" (Peter Mugoya)
**Supported in part:**
- IMU-only evidence gains more from propagation: +0.169 for theirs and +0.180 for ours, against +0.108 and +0.109 for fused evidence.
- With **their** probabilities, the graph affinity works better without the probability term: use_p 0 gives 0.8344, 0.5 gives 0.8294 and 1.0 gives 0.8103. Their fused probabilities mostly duplicate the video grouping.

**Contradicted where it matters for us:**
- Fused evidence ends higher in absolute terms: 0.8632 vs 0.8113 for ours, 0.8260 vs 0.7688 for theirs.
- With **our** probabilities, use_p 0.5 is best: 0.8632, against 0.8476 at 0 and 0.8541 at 1.0.
- Adding our inertial-only model on top of public evidence (0.8538) is worse than adding our fused ensemble (0.8647), and a separate IMU term adds nothing on top of fused m_all4 (0.8694 vs 0.8695).

**Why:** our video path is per-window instance-normalised (dynamics only) with a small static path. It is largely scene-invariant, so it adds naming information rather than repeating what the scene-driven kNN graph already groups.

**Decision:** plug in the fused `m_all4`, keep use_p at 0.5, and do not build an IMU-only path.

### 3.4 Exact modification points
**Our dataset (`rvndyerwin/wear-ours-tileprobs`):**
- `m_all4_oof.npy` (69326,19) log-probs in sim order
- `m_all4_test.npy` (12234,19) log-probs in meta order
- `test_id.npy` (meta ids in row order)
- `ext_meta.npz` (rec, start, sensor, y copied from `~/kaggle/wear_lb/work/sim_meta.npz`)
- Optionally `nnHS7x3_*`, `m_inertial_only_*`

All of these already exist under `~/kaggle/wear_lb/work/ours/`.

**Option A (main).** Insert in cell 13, after `d`/`t` are built and before `del _o, ...`:
```python
EXT_A = float(os.environ.get("WEAR_EXT_A", "0.6")); EXT_NAME = os.environ.get("WEAR_EXT_NAME", "m_all4")
ED = Path(glob.glob("/kaggle/input/**/ext_meta.npz", recursive=True)[0]).parent
em = np.load(ED / "ext_meta.npz")
for k in ("rec", "start", "sensor", "y"): assert (em[k] == sim[k]).all(), k
eo = np.load(ED / f"{EXT_NAME}_oof.npy").astype(np.float32)
et = np.load(ED / f"{EXT_NAME}_test.npy").astype(np.float32)
pos = pd.Index(np.load(ED / "test_id.npy", allow_pickle=True)).get_indexer(np.load(PROC / "test_id.npy", allow_pickle=True))
assert (pos >= 0).all(); et = et[pos]
_l = lambda x: (x - np.logaddexp.reduce(x, 1, keepdims=True)).astype(np.float32)
d["logp"] = _l((1 - EXT_A) * d["logp"] + EXT_A * eo); t["logp"] = _l((1 - EXT_A) * t["logp"] + EXT_A * et)
print("window OOF with ext", macro_f1(d["y"], d["logp"].argmax(1)))   # expect ~0.76 (a=0.6), 0.7655 (a=0.75)
EXT_O, EXT_T = eo, et   # reused by the add-ons below
```
- Leave `fusion_logp` and `imu_logp` unchanged, so the 533/361-column asserts in cell 22 still hold.
- Every downstream consumer picks up the change automatically: PLP, S3, the bc_w/bc_lp/maxp link features (and through them L2/L3), the public graph, counts, adapted-T pseudo-labels and the refiner's `lw` feature.

**Option B (links untouched).** In cell 22, change `TEST_LOGP_B`/`OOF_LOGP_B` to `tab_blend(logp, [(S3,0.45),(T,0.25),(EXT,0.15)])`. Apply the same change to `B2_OOF`/`B2_TEST` in cell 36, using adapted T. The window term then gets weight 0.15. **Both cells** must change; otherwise stage B silently drops the expert.

**Add-ons (round 2):**
- **Counts, cell 34:** `profile_features([P, Bp, np.exp(EXT_O)])` and `([Pt, Btp, np.exp(EXT_T)])`.
- **Refiner, cell 38:** add `EXT[g,cur]-EXT[g,oth]` for the tile and for its chain neighbours at -2, -1, +1, +2.

---

## 4. Ranked improvements

| # | Change | Expected gain | Cost | Risk | How to validate |
|---|---|---|---|---|---|
| 1 | Submit the 0.92990 file | +0.145 LB over our 0.785 | 0 | none | Known LB |
| 2 | **Option A plug-in m_all4, a 0.6 and 0.75** (K1, K2) | +0.003 to +0.010 final OOF (proxy +0.035 to 0.040) | 2 x 4 CPU-h, in parallel | medium: link scorer, QN_REF and TEST_CFGS were tuned on their evidence | Nested SCORES vs K0. Gate: pooled at least +0.002 and at least 4 of 5 folds up. Then a paired LB check against K0 |
| 3 | Control K0 (unmodified, same platform) | none by itself; required baseline and voting member | 4 CPU-h | low | Must reproduce about 0.9308 OOF |
| 4 | Our probabilities as a 3rd count-profile matrix (cell 34) | +0.001 to +0.004. Counts are the largest lever: true counts +0.013, halving count error +0.010 | add-on to a round-2 run | low to medium: the count optimum is sharp (a 3-tile shift costs 0.002) | Printed count error (baseline 7.3 tiles) and OOF |
| 5 | Our probabilities as refiner features (cell 38) | +0.0005 to +0.002 (refiner baseline +0.0033; boundary tiles are about 64% right) | add-on | low (trained leave-one-fold-out) | Refiner OOF delta |
| 6 | Option B, w 0.15 (K3) | +0.002 to +0.006; fallback if A degrades links | 4 CPU-h | low | OOF; compare the L3 linked/exact prints with K0 |
| 7 | **Run-level ensemble** (K0, best A, B, whiten, seed-offset variant): majority vote of `REF_OOF`/test labels, or mean QB, then argmax | +0.0005 to +0.002 OOF (author: 2-run average +0.0006); mainly lower private-LB variance | minutes of CPU | low | Exactly scoreable on OOF, since all runs share the sim row order |
| 8 | Link-bagging diversity: offset the Gumbel seed (`1000*k+sbj` plus a constant) and the L3 scorer seeds by code edit, not WEAR_SEED | about 0.0005, as voting members only | 4 CPU-h each | low | OOF of the vote |
| 9 | WEAR_WHITEN=0.5 on top of the best A | +0.003 OOF; LB -0.0022 to +0.0032 (Prabhav: +0.0001) | 4 CPU-h | medium (poor LB transfer) | Voting member only, not a standalone pick |
| 10 | Count and null handling for small subjects (see note below the table) | 0 to 0.003 | offline from the dump, or an add-on | medium | OOF on train subjects 5, 19, 20, 21 plus pooled OOF |
| 11 | Habakan's complex-variant share prior (pairs 11/12, 13/14, 16/17 kept at 45 to 58%) as a soft per-subject bias before argmax | 0 to 0.002 | tiny | low to medium | OOF from the dump |
| 12 | Cross-pipeline label vote: Prabhav's 0.92875 submission (attach as a kernel source), the 0.92990 file, and our best | ±0.003, cannot be checked on OOF | 1 submission | medium | LB only; spend at most 1 or 2 slots |

**Note on row 10.** Test subjects 24 and 25 have 1995 and 1914 tiles. Both window models predict 31 to 34% null for them, while the final pipeline outputs 0.15 and 0.09. On train, the fixed-97 prior over-forces activity for the small subjects: forced null vs true null is 0.17 vs 0.27 for sbj 19, 0.085 vs 0.14 for sbj 20, and 0.19 vs 0.28 for sbj 5. Things to try: the count clip [70,135] and the null floor.

**Do not spend time on:**
- label cleaning (WEAR_CLEAN=1: OOF up, LB mean 0.925 vs 0.927);
- a 4th two-tower seed on GPU (the 3rd seed moved exact successors by +0.2 points and scored LB 0.92585);
- 16 matchings, a 3rd count pass, a richer refiner, the BiGRU decoder, adapting the window nets (all documented as flat or negative);
- the sujan island filter;
- Jiwei's or Prabhav's code paths;
- habakan as a base.

### Timeline (submission cap 10 per day, resets 00:00 UTC; about 28 submissions left)
- **Oct 10:**
  - Submit S1 = the 0.92990 file.
  - Upload our dataset (about 30 min), then push K0, K1, K2, K3 in parallel.
  - About 5 h later, submit K0 and the two best of K1 to K3 by OOF. This gives paired LB deltas against K0.
- **Oct 11:**
  - Round 2: K4 = best A + counts add-on + refiner add-on; K5 = best A + whiten; K6 = best A with seed offsets.
  - Submit 3 to 4 by OOF.
  - Evening: a small CPU ensemble kernel scores label votes and QB means on OOF; submit the best 2.
- **Oct 12:**
  - Only OOF-validated tweaks (rows 10 and 11). Keep 3 or more submissions in reserve.
  - Select the finals by about 20:00 UTC.
  - Pick 1 = best nested OOF, single run or vote.
  - Pick 2 = the 0.92990 file as a hedge. If our family also beats it on paired LB, take the second-best diverse OOF variant instead.
- **Realistic target:** 0.930 to 0.937 public.

---

## 5. Risks

- **Dataset dependency.** If `goodpjw2008/wear-hasca-part2-gpu-outputs` (about 1.5 GB) is changed or deleted, the fork fails; pin the version. If it mounts deeper than one level, the notebook's own glob finds nothing and falls back without error to CPU_RUNS (4 epochs x 100 steps), which scores far lower. The explicit WEAR_ARTIFACTS cell prevents this. Fallback base: Part 1 on T4x2 (about 2 h, more variance).
- **Hard crashes and asserts:**
  - WEAR_ABLATION=0 raises a KeyError in cell 34.
  - Asserts check 69,326 rows, the fold sizes, the sha1 hashes, 533/361 S3/T columns, and `y`/`grp` equality in cell 13. Adding columns to S3 needs edits to the asserts.
- **Randomness and reproducibility:**
  - The same code on Kaggle CPU scored 0.92620 vs 0.92990 (98.7% identical labels).
  - Never compare a fork's LB against 0.92990; compare OOF against K0, and LB only as K_i minus K0.
- **LB overfitting:**
  - Public noise is ±0.004 to 0.005, and moving 150 labels is worth 0.005.
  - The public/private split over the 4 test subjects is unknown.
  - Rank by nested OOF, which is itself slightly optimistic because TEST_CFGS and QN_REF were chosen on the full OOF.
  - One subject is worth about 0.002 of OOF.
- **Plug-in specific:**
  - Sharper evidence could disturb the fixed QN_REF score scale and TEST_CFGS. Option B and a = 0.6 are the hedges.
  - Our stackers are inside m_all4. They are cross-validated, but they are a second level built on the same folds.
  - Never use our bias-tuned arrays; plug in the raw ones.
- **Count prior:**
  - It assumes one circuit per test subject. That fits the null shares for 22 and 23 (window 0.70/0.46 vs pipeline 0.66/0.43).
  - For 24 and 25 it is the largest unvalidatable private-LB risk.
- **Kernel operations:**
  - One kernel slug runs one version at a time, so use a separate slug per variant.
  - Runtime is unlogged (budget about 5 h).
  - WEAR_SEED in a CPU fork may trigger two-tower retraining on CPU.
  - CLEANUP deletes the scratch, so WEAR_DUMP=1 is required.
  - `final_probabilities.npz` holds pre-refiner QB.
- **Herding:** many teams will submit the same 0.92990 file. Moving up on private needs a real gain, not just that file.

---

## Files
- `~/kaggle/wear_lb/work/`: `tiles.py` and `map_ours.py`; `sim_meta.npz` and `sim_gidx25.npy`; `ours/m_all4_oof.npy` and `ours/m_all4_test.npy`, plus 60 other mapped models.
- `~/kaggle/wear_lb/gpu_out/runs/`: the public window nets and their outputs.
- `endgame/work/proxy.py`: the window and propagation probe.
- `endgame/ref_public/wear-hasca-part-2-ranked-links-lb-0-92990/` (fetched by `endgame/fetch_refs.sh`): the base notebook.