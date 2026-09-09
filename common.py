import os, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.expanduser('~/kaggle/wear')
DATA, CACHE = f'{ROOT}/data', f'{ROOT}/cache'
SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
S2I = {s: i for i, s in enumerate(SENSORS)}
NCLS = 19
WIN = 50
# canonical subject -> fold assignment (GroupKFold(5) on the stride-25 window index)
SBJ_FOLD = {0: 1, 1: 3, 2: 4, 3: 2, 4: 1, 5: 0, 6: 4, 7: 2, 8: 3, 9: 1, 10: 3, 11: 2, 12: 2, 13: 4, 14: 0, 15: 1, 16: 3, 17: 0, 18: 4, 19: 2, 20: 4, 21: 0}

# fold set 2: location-proxy folds from scene-similarity clusters of per-subject mean video features
SBJ_FOLD2 = {1: 0, 4: 0, 5: 0, 9: 0,
             10: 1, 11: 1, 12: 1, 13: 1, 15: 1,
             19: 2, 20: 2, 21: 2, 0: 2,
             3: 3, 14: 3, 16: 3, 7: 3, 8: 3,
             2: 4, 6: 4, 17: 4, 18: 4}
FOLDSETS = {1: SBJ_FOLD, 2: SBJ_FOLD2}


def fold_map(foldset=1):
    return FOLDSETS[int(foldset)]


def fold_splits(groups, foldset=1, nfolds=5):
    fm = fold_map(foldset)
    fo = np.array([fm[int(s)] for s in groups])
    return [(np.where(fo != f)[0], np.where(fo == f)[0]) for f in range(nfolds)]


def load_index():
    return pd.read_parquet(f'{CACHE}/windows.parquet')


def video_windows(rec, starts):
    """(n, 15, 768) float32 of the 15 in-window frames for the given inertial starts."""
    v = np.load(f'{DATA}/videomae/{rec}.npy', mmap_mode='r')
    vs = starts * 3 // 5 + 8
    idx = vs[:, None] + np.arange(15)[None]
    return np.asarray(v[idx], dtype=np.float32)


def inertial_windows(rec, starts):
    """(n, 4, 50, 3) float32."""
    x = np.load(f'{CACHE}/{rec}_inertial.npy', mmap_mode='r')
    idx = starts[:, None] + np.arange(WIN)[None]
    return np.asarray(x[idx], dtype=np.float32).transpose(0, 2, 1, 3)


def _basic(a):  # a: (n, T) -> 9 stats
    q25, med, q75 = np.percentile(a, [25, 50, 75], axis=1)
    return np.stack([a.mean(1), a.std(1), a.min(1), a.max(1), med, q75 - q25,
                     stats.skew(a, axis=1), stats.kurtosis(a, axis=1), np.abs(a).mean(1)], 1)


BANDS = [(1, 3), (3, 6), (6, 11), (11, 18), (18, 26)]


def _spectral(a):  # a: (n, T) mean-removed -> band log-energies (5), dom freq (1), spectral entropy (1)
    f = np.abs(np.fft.rfft(a - a.mean(1, keepdims=True), axis=1)) ** 2  # (n, 26)
    tot = f[:, 1:].sum(1) + 1e-8
    bands = np.stack([np.log1p(f[:, lo:hi].sum(1)) for lo, hi in BANDS], 1)
    dom = f[:, 1:].argmax(1) + 1
    p = f[:, 1:] / tot[:, None]
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return np.concatenate([bands, dom[:, None], ent[:, None], np.log(tot)[:, None]], 1)


def inertial_features(w):
    """w: (n, 50, 3) -> (n, F) hand-crafted features."""
    n = len(w)
    m = np.linalg.norm(w, axis=2)  # (n, 50)
    feats = [_basic(w[:, :, k]) for k in range(3)] + [_basic(m)]
    d = np.diff(w, axis=1); dm = np.diff(m, axis=1)
    feats.append(np.concatenate([d.std(1), np.abs(d).mean(1), dm.std(1)[:, None], np.abs(dm).mean(1)[:, None]], 1))
    feats += [_spectral(w[:, :, k]) for k in range(3)] + [_spectral(m)]
    c = w - w.mean(1, keepdims=True)
    sd = c.std(1) + 1e-6
    corr = np.stack([(c[:, :, i] * c[:, :, j]).mean(1) / (sd[:, i] * sd[:, j]) for i, j in [(0, 1), (0, 2), (1, 2)]], 1)
    zcr = (np.sign(c[:, 1:]) != np.sign(c[:, :-1])).mean(1)  # (n,3)
    mc = m - m.mean(1, keepdims=True)
    ac = np.stack([(mc[:, lag:] * mc[:, :-lag]).mean(1) / (mc.var(1) + 1e-6) for lag in (5, 10, 15, 20, 25)], 1)
    thr = m.mean(1, keepdims=True) + 0.5 * m.std(1, keepdims=True)
    peaks = ((m[:, 1:-1] > m[:, :-2]) & (m[:, 1:-1] >= m[:, 2:]) & (m[:, 1:-1] > thr)).sum(1)
    g = w.mean(1); g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)  # gravity direction
    feats += [corr, zcr, ac, peaks[:, None].astype(np.float32), g]
    return np.concatenate(feats, 1).astype(np.float32)


def macro_f1(y, p, C=NCLS):
    y = np.asarray(y, np.int64); p = np.asarray(p, np.int64)
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    present = (cm.sum(1) + cm.sum(0)) > 0  # sklearn: classes absent in both y and p are skipped
    return f1[present].mean()


def tune_bias(prob, y, iters=3, grid=np.linspace(-2, 2, 41)):
    """Coordinate-ascent per-class log-prob offsets maximizing macro F1 on OOF."""
    lp = np.log(prob + 1e-9); b = np.zeros(prob.shape[1])
    best = macro_f1(y, (lp + b).argmax(1))
    for _ in range(iters):
        for c in range(prob.shape[1]):
            cur = b[c]
            for g in grid:
                b[c] = g
                s = macro_f1(y, (lp + b).argmax(1))
                if s > best + 1e-6: best, cur = s, g
            b[c] = cur
    return b, best


def tune_bias_cv(prob, y, folds, iters=2, grid=np.linspace(-2, 2, 41)):
    """Honest estimate: tune bias on 4 folds' OOF, apply to the 5th; returns macro F1 of the stitched predictions."""
    pred = np.zeros(len(y), int)
    for f in np.unique(folds):
        tr = folds != f
        b, _ = tune_bias(prob[tr], y[tr], iters=iters, grid=grid)
        pred[~tr] = (np.log(prob[~tr] + 1e-9) + b).argmax(1)
    return macro_f1(y, pred)
