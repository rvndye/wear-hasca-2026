"""Concatenate all train video features into one float16 memmap + offsets (for fast NN training)."""
import numpy as np, json, os
from common import *
recs = sorted(f[:-4] for f in os.listdir(f'{DATA}/inertial') if f.endswith('.csv'))
sizes = {r: np.load(f'{DATA}/videomae/{r}.npy', mmap_mode='r').shape[0] for r in recs}
offs, o = {}, 0
for r in recs: offs[r] = o; o += sizes[r]
V = np.lib.format.open_memmap(f'{CACHE}/video_all_f16.npy', mode='w+', dtype=np.float16, shape=(o, 768))
for r in recs:
    v = np.load(f'{DATA}/videomae/{r}.npy', mmap_mode='r')
    for s in range(0, sizes[r], 20000):
        V[offs[r]+s: offs[r]+min(s+20000, sizes[r])] = np.asarray(v[s:s+20000], dtype=np.float16)
    print(r, sizes[r], flush=True)
V.flush(); json.dump(offs, open(f'{CACHE}/video_offsets.json', 'w'))
print('total frames', o)
