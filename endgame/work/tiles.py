"""Rebuild the public pipeline's 69,326 test-like OOF tiles (exact recipe of woominyo/lb-0-890 cell 8)
from the train inertial CSVs, verify their sha1 asserts, and save sim_meta.npz + per-recording info.
Video length F is taken as T*3//5 (the public notebook asserts F*5 == T*3 for every recording)."""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
D = Path.home() / 'kaggle/wear_lb/data'; W = Path.home() / 'kaggle/wear_lb/work'
N_CLS, WIN, VLEN, STRIDE, FOLDS = 19, 50, 15, 5, 5
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
           "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
           "stretching (hamstrings)", "stretching (lumbar rotation)", "push-ups", "push-ups (complex)",
           "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
LABEL_IDX = {c: i for i, c in enumerate(CLASSES)}
SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
ACC_COLS = [f"{s}_acc_{a}" for s in SENSORS for a in "xyz"]
recs, LAB, VAL = [], {}, {}
for csv in sorted((D / 'inertial').glob('*.csv')):
    df = pd.read_csv(csv, usecols=["sbj_id", *ACC_COLS, "label"], dtype={"label": str})
    acc = df[ACC_COLS].to_numpy(np.float32)
    VAL[csv.stem] = ~np.isnan(acc).reshape(-1, 4, 3).any(2)
    LAB[csv.stem] = df["label"].fillna("null").astype(str).str.strip().map(LABEL_IDX).to_numpy(np.int8)
    T = len(acc); recs.append({"stem": csv.stem, "sbj": int(df["sbj_id"].iloc[0]), "T": T, "F": T * 3 // 5})
assert len(recs) == 24 and sum(r["T"] for r in recs) == 3466400 and sum(r["F"] for r in recs) == 2079840
W_rec, W_start, W_y, W_sbj, W_ok = [], [], [], [], []
for i, r in enumerate(recs):
    lab = LAB[r['stem']].astype(np.int64); val = VAL[r['stem']]; T, Fr = r["T"], r["F"]
    starts = np.arange(0, T - WIN + 1, STRIDE); starts = starts[starts * 3 // 5 + 8 + VLEN <= Fr]
    cs = np.zeros((T + 1, N_CLS), np.int32); np.cumsum(np.eye(N_CLS, dtype=np.int32)[lab], axis=0, out=cs[1:])
    cnt = cs[starts + WIN] - cs[starts]
    miss = np.zeros((T + 1, 4), np.int32); np.cumsum(~val, axis=0, out=miss[1:])
    W_ok.append((miss[starts + WIN] - miss[starts]) == 0); W_rec.append(np.full(len(starts), i)); W_start.append(starts)
    W_y.append(cnt.argmax(1)); W_sbj.append(np.full(len(starts), r["sbj"]))
W_rec, W_start, W_y, W_sbj, W_ok = map(np.concatenate, (W_rec, W_start, W_y, W_sbj, W_ok))
keep = W_ok.any(1); W_rec, W_start, W_y, W_sbj, W_ok = (x[keep] for x in (W_rec, W_start, W_y, W_sbj, W_ok))
sbjs, sbj_n = np.unique(W_sbj, return_counts=True); fold_of, load = {}, np.zeros(FOLDS)
for s in sbjs[np.argsort(-sbj_n)]:
    f = int(load.argmin()); fold_of[s] = f; load[f] += sbj_n[sbjs == s][0]
W_fold = np.array([fold_of[s] for s in W_sbj])
out = {k: [] for k in ("rec", "start", "sensor", "y", "sbj", "fold")}
for fold in range(FOLDS):
    idx = np.flatnonzero(W_fold == fold); idx = idx[W_start[idx] % WIN == 0]
    rng = np.random.default_rng(1234 + fold); cp = np.cumsum(W_ok[idx] / W_ok[idx].sum(1, keepdims=True), 1)
    sensor = (rng.random(len(idx))[:, None] > cp).sum(1)
    for k, v in (("rec", W_rec[idx]), ("start", W_start[idx]), ("sensor", sensor), ("y", W_y[idx]), ("sbj", W_sbj[idx]), ("fold", np.full(len(idx), fold))):
        out[k].append(v)
sim = {k: np.concatenate(v).astype(np.int64) for k, v in out.items()}
assert len(sim["y"]) == 69326, len(sim["y"])
assert np.bincount(sim["fold"]).tolist() == [14353, 12995, 14625, 12768, 14585], np.bincount(sim["fold"]).tolist()
h1 = hashlib.sha1(sim["sensor"].tobytes()).hexdigest()[:12]; h2 = hashlib.sha1(np.stack([sim["rec"], sim["start"]]).tobytes()).hexdigest()[:12]
print('sha1 sensor', h1, 'expect f81d90085315 |', 'rec,start', h2, 'expect e4177f2589d7')
assert h1 == 'f81d90085315' and h2 == 'e4177f2589d7'
np.savez(W / 'sim_meta.npz', **sim); json.dump(recs, open(W / 'recordings.json', 'w'), indent=1)
# our stride-25 window index (prep.py): per recording all starts arange(0,T-49,25) with start*3//5+23 <= F, concatenated in sorted-rec order
n25 = [int(((np.arange(0, r['T'] - WIN + 1, 25) * 3 // 5 + 23) <= r['F']).sum()) for r in recs]
off = np.r_[0, np.cumsum(n25)[:-1]]; assert sum(n25) == 138632, sum(n25)
gidx = off[sim['rec']] + sim['start'] // 25
np.save(W / 'sim_gidx25.npy', gidx)
print('folds', {f: sorted(int(s) for s in fold_of if fold_of[s] == f) for f in range(FOLDS)})
print('label dist', np.bincount(sim['y'], minlength=19).tolist()); print('OK')
