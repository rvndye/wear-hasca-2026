"""OOF macro F1 excluding the documented train label errors (goodpjw2008 WEAR_CLEAN spans):
sbj_10 after second 2566 and sbj_2 after second 3452 (activities labelled null), sbj_7 after second 1368 (labels 10 s late)."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
import os
WL = Path(os.environ.get("WEAR_LB", Path.home() / "kaggle" / "wear_lb"))  # endgame working dir (data, kernel outputs, caches)
W = WL / 'work'; K = WL / 'kout'
sim = dict(np.load(W / 'sim_meta.npz')); y, fold, rec, start, sbj = sim['y'], sim['fold'], sim['rec'], sim['start'], sim['sbj']
stems = [r['stem'] for r in json.load(open(W / 'recordings.json'))]
sec = start // 50; stem = np.array(stems)[rec]
bad = ((stem == 'sbj_10') & (sec >= 2566)) | ((stem == 'sbj_2') & (sec >= 3452)) | ((stem == 'sbj_7') & (sec >= 1368))
keep = ~bad
def mf1(y, p, C=19):
    cm = np.bincount(y*C+p, minlength=C*C).reshape(C, C); tp = np.diag(cm); d = cm.sum(0)+cm.sum(1); pr = d > 0; return float((2*tp[pr]/d[pr]).mean())
print(f'excluded tiles: {bad.sum()} ({bad.mean():.2%})')
runs = sys.argv[1:] or sorted(p.name for p in K.iterdir() if (p / 'final_probabilities.npz').exists())
for r in runs:
    lab = np.load(K / r / 'final_probabilities.npz')['oof_labels'].astype(int)
    per_sbj = {int(s): round(mf1(y[(sbj == s) & keep], lab[(sbj == s) & keep]), 4) for s in [1, 2, 7, 8, 10, 16]}
    print(f'{r:28s} all {mf1(y, lab):.4f} | clean {mf1(y[keep], lab[keep]):.4f} | clean folds ' +
          ' '.join(f'{mf1(y[keep & (fold==f)], lab[keep & (fold==f)]):.4f}' for f in range(5)) + f' | {per_sbj}')
