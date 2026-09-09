"""Richer inertial features (v2): full log-spectrum, quantiles, covariance eigen-structure, more autocorr, downsampled magnitude."""
import numpy as np, time
from common import *

def inertial_features_v2(w):
    n = len(w); m = np.linalg.norm(w, axis=2); feats = []
    sig = [w[:, :, 0], w[:, :, 1], w[:, :, 2], m]
    for a in sig:
        c = a - a.mean(1, keepdims=True)
        f = np.abs(np.fft.rfft(c, axis=1))  # (n,26)
        feats.append(np.log1p(f[:, 1:]))                                  # 25 spectral bins
        feats.append(np.percentile(a, [5, 10, 20, 30, 40, 60, 70, 80, 90, 95], axis=1).T)  # 10 quantiles
        feats.append(np.stack([np.sqrt((a ** 2).mean(1)), np.abs(c).mean(1), a.max(1) - a.min(1)], 1))
        ac = np.stack([(c[:, l:] * c[:, :-l]).mean(1) / (c.var(1) + 1e-6) for l in (2, 3, 4, 6, 8, 12, 16, 20, 24)], 1)
        feats.append(ac)
        d = np.diff(a, axis=1); fd = np.abs(np.fft.rfft(d - d.mean(1, keepdims=True), axis=1)) ** 2
        p = fd[:, 1:] / (fd[:, 1:].sum(1, keepdims=True) + 1e-8)
        feats.append(np.stack([-(p * np.log(p + 1e-12)).sum(1), fd[:, 1:].argmax(1) + 1, np.log1p(fd[:, 1:].sum(1))], 1))
    # covariance eigen-structure (orientation invariant)
    c3 = w - w.mean(1, keepdims=True); cov = np.einsum('nti,ntj->nij', c3, c3) / WIN
    ev = np.linalg.eigvalsh(cov)  # ascending (n,3)
    feats.append(np.concatenate([np.log(ev + 1e-6), (ev / (ev.sum(1, keepdims=True) + 1e-9))], 1))
    # angle between gravity (mean) and dominant motion axis
    evec = np.linalg.eigh(cov)[1][:, :, -1]; g = w.mean(1); g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    feats.append(np.abs((evec * g).sum(1))[:, None])
    # downsampled magnitude (10 points) and sorted magnitude deciles are above; add mean-crossing rate of magnitude
    mc = m - m.mean(1, keepdims=True); feats.append((np.sign(mc[:, 1:]) != np.sign(mc[:, :-1])).mean(1)[:, None])
    feats.append(m.reshape(n, 10, 5).mean(2))
    return np.concatenate(feats, 1).astype(np.float32)

if __name__ == '__main__':
    idx = load_index(); t0 = time.time(); FI = []
    for rec, g in idx.groupby('rec', sort=False):
        w = np.nan_to_num(inertial_windows(rec, g['start'].to_numpy()))
        FI.append(np.stack([inertial_features_v2(w[:, s]) for s in range(4)], 1)); print(rec, f'{time.time()-t0:.0f}s', flush=True)
    FI = np.concatenate(FI); np.save(f'{CACHE}/feat_inertial_v2.npy', FI)
    ti = np.nan_to_num(np.load(f'{DATA}/test_inertial_data.npy').astype(np.float32))
    np.save(f'{CACHE}/test_feat_inertial_v2.npy', inertial_features_v2(ti)); print('done', FI.shape)
