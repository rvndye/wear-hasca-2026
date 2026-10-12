"""Map our saved OOF (stride-25 x 4 limbs, or LGB rows + oofmeta) onto the public 69,326 tiles; save tile-level
log-probs per model: ours/<name>_oof.npy (69326,19) float32 log-probs in sim row order, ours/<name>_test.npy (12234,19)."""
import numpy as np, os, glob
from pathlib import Path
W = Path.home() / 'kaggle/wear_lb/work'; O = Path.home() / 'kaggle/wear_lb/oof'; T = Path.home() / 'Desktop/Kaggle/wear/preds'
(W / 'ours').mkdir(exist_ok=True)
sim = dict(np.load(W / 'sim_meta.npz')); g = np.load(W / 'sim_gidx25.npy'); s = sim['sensor']; y = sim['y']
def mf1(y, p, C=19):
    cm = np.bincount(y*C+p, minlength=C*C).reshape(C,C); tp=np.diag(cm).astype(float); d=cm.sum(0)+cm.sum(1); pr=d>0; return (2*tp[pr]/d[pr]).mean()
rows = []
for f in sorted(glob.glob(str(O / 'oof_*.npy'))):
    name = os.path.basename(f)[4:-4]
    if not (T / f'test_{name}.npy').exists(): continue
    a = np.load(f).astype(np.float32)
    if a.ndim == 2:
        mp = O / f'oofmeta_{name}.npy'
        if not mp.exists(): continue
        mt = np.load(mp); full = np.full((138632, 4, 19), np.nan, np.float32); full[mt[:, 0], mt[:, 1]] = a; a = full
        ylab = np.full((138632, 4), -1); ylab[mt[:, 0], mt[:, 1]] = mt[:, 2]
        ok = ylab[g, s] >= 0
        if ok.any(): assert (ylab[g, s][ok] == y[ok]).all(), name
    p = a[g, s]
    cov = ~np.isnan(p[:, 0])
    if cov.mean() < 0.999: rows.append((name, cov.mean(), None)); continue
    p = np.clip(p, 1e-7, 1); p = p / p.sum(1, keepdims=True)
    te = np.load(T / f'test_{name}.npy').astype(np.float32); te = np.clip(te, 1e-7, 1); te = te / te.sum(1, keepdims=True)
    np.save(W / 'ours' / f'{name}_oof.npy', np.log(p)); np.save(W / 'ours' / f'{name}_test.npy', np.log(te))
    rows.append((name, cov.mean(), mf1(y, p.argmax(1))))
for n, c, f1 in sorted(rows, key=lambda r: -(r[2] or 0)): print(f'{n:28s} coverage {c:.3f}  tile mF1 {f1 if f1 is not None else float("nan"):.4f}')
