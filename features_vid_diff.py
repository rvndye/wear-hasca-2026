"""Video motion features per window: mean abs frame-to-frame difference (768) and last-first frame diff (768)."""
import numpy as np, time
from common import *
idx = load_index(); out = np.zeros((len(idx), 1536), np.float32); t0 = time.time()
for rec, g in idx.groupby('rec', sort=False):
    v = video_windows(rec, g['start'].to_numpy())  # (n,15,768)
    out[g.index, :768] = np.abs(np.diff(v, axis=1)).mean(1); out[g.index, 768:] = v[:, -1] - v[:, 0]
    print(rec, f'{time.time()-t0:.0f}s', flush=True)
np.save(f'{CACHE}/feat_video_diff.npy', out)
tv = np.asarray(np.load(f'{DATA}/test_videomae_data.npy', mmap_mode='r'), dtype=np.float32).transpose(0, 2, 1)
tout = np.concatenate([np.abs(np.diff(tv, axis=1)).mean(1), tv[:, -1] - tv[:, 0]], 1)
np.save(f'{CACHE}/test_feat_video_diff.npy', tout); print('done', out.shape, tout.shape)
