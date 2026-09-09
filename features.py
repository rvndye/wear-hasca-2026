"""Build hand-crafted feature matrices for LightGBM.
cache/feat_inertial.npy (n_win, 4, F), cache/feat_video.npy (n_win, 2*768) pooled mean/std,
cache/test_feat_inertial.npy (n_test, F), cache/test_feat_video.npy (n_test, 1536)."""
import numpy as np, pandas as pd, time
from common import *

idx = load_index()
t0 = time.time()
FI, FV = [], []
for rec, g in idx.groupby('rec', sort=False):
    starts = g['start'].to_numpy()
    w = inertial_windows(rec, starts)            # (n,4,50,3)
    w = np.nan_to_num(w)
    fi = np.stack([inertial_features(w[:, s]) for s in range(4)], 1)
    v = video_windows(rec, starts)               # (n,15,768)
    fv = np.concatenate([v.mean(1), v.std(1)], 1)
    FI.append(fi); FV.append(fv)
    print(f'{rec:8s} {len(g):6d} windows  {time.time()-t0:6.0f}s', flush=True)
FI = np.concatenate(FI); FV = np.concatenate(FV)
assert len(FI) == len(idx)
np.save(f'{CACHE}/feat_inertial.npy', FI); np.save(f'{CACHE}/feat_video.npy', FV)
print('train feats', FI.shape, FV.shape)

ti = np.load(f'{DATA}/test_inertial_data.npy').astype(np.float32)   # (N,50,3)
tv = np.load(f'{DATA}/test_videomae_data.npy', mmap_mode='r')          # (N,768,15)
print('test nan inertial', np.isnan(ti).sum())
tfi = inertial_features(np.nan_to_num(ti))
tfv = np.concatenate([np.asarray(tv, dtype=np.float32).mean(2), np.asarray(tv, dtype=np.float32).std(2)], 1)
np.save(f'{CACHE}/test_feat_inertial.npy', tfi); np.save(f'{CACHE}/test_feat_video.npy', tfv)
print('test feats', tfi.shape, tfv.shape, f'{time.time()-t0:.0f}s')
