"""Post-hoc bout-level label swaps: within each subject, the tiles carrying class c form that class's bout(s).
For every pair of activity classes (a, b), swapping the two groups changes the summed window evidence by
gain = sum_{tiles labelled a} [lp_b - lp_a] + sum_{tiles labelled b} [lp_a - lp_b]. Greedily apply swaps with gain > thr.
usage: boutswap.py <run-or-vote-labels.npy> <evidence names...>"""
import sys, numpy as np, itertools
from pathlib import Path
import os
WL = Path(os.environ.get("WEAR_LB", Path.home() / "kaggle" / "wear_lb"))  # endgame working dir (data, kernel outputs, caches)
W = WL / 'work'; K = WL / 'kout'
sim = dict(np.load(W / 'sim_meta.npz')); y, sbj, fold = sim['y'], sim['sbj'], sim['fold']
def mf1(y, p, C=19):
    cm = np.bincount(y*C+p, minlength=C*C).reshape(C, C); tp = np.diag(cm); d = cm.sum(0)+cm.sum(1); pr = d > 0; return float((2*tp[pr]/d[pr]).mean())
def swaps(lab, lp, groups, thr, norm=False, include_null=False):
    lab = lab.copy(); n_sw = 0
    cls = range(0 if include_null else 1, 19)
    for s in np.unique(groups):
        ii = np.flatnonzero(groups == s)
        while True:
            L = lab[ii]; best = (thr, None)
            S = {c: lp[ii[L == c]].sum(0) for c in cls}   # summed evidence of group c for every class
            N = {c: max((L == c).sum(), 1) for c in cls}
            for a, b in itertools.combinations(cls, 2):
                if (L == a).sum() == 0 or (L == b).sum() == 0: continue
                g = (S[a][b] - S[a][a]) + (S[b][a] - S[b][b])
                if norm: g = g / (N[a] + N[b])
                if g > best[0]: best = (g, (a, b))
            if best[1] is None: break
            a, b = best[1]; ma, mb = ii[L == a], ii[L == b]; lab[ma], lab[mb] = b, a; n_sw += 1
    return lab, n_sw
if __name__ == '__main__':
    src = sys.argv[1]
    lab = np.load(src) if src.endswith('.npy') else np.load(K / src / 'final_probabilities.npz')['oof_labels'].astype(int)
    print(f'{src}: base OOF {mf1(y, lab):.4f}')
    ev = {'ours_m_all5': np.load(W / 'ours/m_all5_oof.npy'), 'ours_m_all4': np.load(W / 'ours/m_all4_oof.npy')}
    R = WL / 'gpu_out/runs'
    th = np.log(np.mean([np.exp(np.load(R / f'final_s{i}/oof_raw.npz')['logp']) for i in range(4)], 0) + 1e-9)
    ev['theirs_fusion'] = th; ev['avg_ours5_theirs'] = np.log(0.5 * np.exp(ev['ours_m_all5']) + 0.5 * np.exp(th) + 1e-9)
    if src in [p.name for p in K.iterdir()]:
        ev['run_QB'] = np.log(np.clip(np.load(K / src / 'final_probabilities.npz')['oof'], 1e-9, 1))
    for name, lp in ev.items():
        for thr in [0.0, 5.0, 20.0]:
            new, k = swaps(lab, lp, sbj, thr)
            print(f'  evidence {name:16s} thr {thr:5.1f}: swaps {k:3d} -> OOF {mf1(y, new):.4f} | folds ' + ' '.join(f'{mf1(y[fold==f], new[fold==f]):.4f}' for f in range(5)))
