"""Convert train CSVs to compact npy caches and build the window index.

Outputs (in cache/):
  <rec>_inertial.npy  float32 (T, 4, 3)   sensors ordered right_arm, right_leg, left_leg, left_arm
  <rec>_labels.npy    int8 (T,)           0 = null, 1..18 activities
  windows.parquet     one row per (recording, start) window with majority label
"""
import os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

DATA = os.path.expanduser('~/kaggle/wear/data')
CACHE = os.path.expanduser('~/kaggle/wear/cache')
os.makedirs(CACHE, exist_ok=True)
CLASSES = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)",
           "jogging (butt-kicks)", "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
           "stretching (hamstrings)", "stretching (lumbar rotation)", "push-ups", "push-ups (complex)",
           "sit-ups", "sit-ups (complex)", "burpees", "lunges", "lunges (complex)", "bench-dips"]
L2I = {c: i for i, c in enumerate(CLASSES)}
SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
COLS = [f"{s}_acc_{a}" for s in SENSORS for a in "xyz"]
WIN, STRIDE = 50, 25


def convert(rec):
    out_i, out_l = f"{CACHE}/{rec}_inertial.npy", f"{CACHE}/{rec}_labels.npy"
    if os.path.exists(out_i) and os.path.exists(out_l):
        return rec, int(np.load(out_l, mmap_mode='r').shape[0])
    df = pd.read_csv(f"{DATA}/inertial/{rec}.csv", low_memory=False,
                     dtype={c: 'float32' for c in COLS} | {'label': 'string', 'sbj_id': 'int16'})
    x = df[COLS].to_numpy(np.float32).reshape(-1, 4, 3)
    lab = df['label'].fillna('null').map(L2I)
    assert lab.notna().all(), f"unknown label in {rec}: {set(df['label'].dropna()) - set(CLASSES)}"
    np.save(out_i, x); np.save(out_l, lab.to_numpy(np.int8))
    return rec, len(df)


def build_index(rec):
    lab = np.load(f"{CACHE}/{rec}_labels.npy")
    x = np.load(f"{CACHE}/{rec}_inertial.npy", mmap_mode='r')
    nvid = np.load(f"{DATA}/videomae/{rec}.npy", mmap_mode='r').shape[0]
    T = len(lab)
    starts = np.arange(0, T - WIN + 1, STRIDE)
    starts = starts[(starts * 3 // 5 + 23) <= nvid]
    rows = []
    for s in starts:
        w = lab[s:s + WIN]
        cnt = np.bincount(w, minlength=19)
        maj = int(cnt.argmax()); purity = cnt[maj] / WIN
        valid = ~np.isnan(np.asarray(x[s:s + WIN])).any(axis=(0, 2))  # (4,)
        rows.append((rec, int(s), maj, float(purity), *valid.tolist()))
    return rows


if __name__ == '__main__':
    recs = sorted(f[:-4] for f in os.listdir(f"{DATA}/inertial") if f.endswith('.csv'))
    with ProcessPoolExecutor(6) as ex:
        for rec, n in ex.map(convert, recs):
            print(f"{rec:8s} T={n}", flush=True)
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(build_index, recs):
            rows += r
    idx = pd.DataFrame(rows, columns=['rec', 'start', 'label', 'purity'] + [f'valid_{s}' for s in SENSORS])
    idx['sbj'] = idx['rec'].str.extract(r'sbj_(\d+)').astype(int)
    idx.to_parquet(f"{CACHE}/windows.parquet")
    print(idx.shape); print(idx['label'].value_counts().sort_index().to_dict())
    print('purity<1 frac', (idx.purity < 1).mean())
    print('invalid sensor windows', {s: int((~idx[f'valid_{s}']).sum()) for s in SENSORS})
    print(idx.groupby('sbj').size().to_dict())
