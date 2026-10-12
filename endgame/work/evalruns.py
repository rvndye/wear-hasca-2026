"""Score finished fork runs on OOF and build label votes across runs.
usage: evalruns.py                      -> per-run OOF (post-refiner labels) + per fold + test agreement
       evalruns.py vote a b c [--out f]  -> majority vote of post-refiner labels (ties -> first run), OOF + test csv"""
import sys, glob, re, numpy as np, pandas as pd
from pathlib import Path
W = Path.home() / 'kaggle/wear_lb/work'; K = Path.home() / 'kaggle/wear_lb/kout'
sim = dict(np.load(W / 'sim_meta.npz')); y, fold = sim['y'], sim['fold']
def mf1(y, p, C=19):
    cm = np.bincount(y*C+p, minlength=C*C).reshape(C, C); tp = np.diag(cm); d = cm.sum(0)+cm.sum(1); pr = d > 0; return float((2*tp[pr]/d[pr]).mean())
ids = pd.read_csv(Path.home() / 'kaggle/wear_lb/data/test_meta_data.csv')['id'].to_numpy()
def load(name):
    z = np.load(K / name / 'final_probabilities.npz'); sub = pd.read_csv(K / name / 'submission.csv').set_index('id').loc[ids].iloc[:, 0].to_numpy()
    return z['oof_labels'].astype(int), sub.astype(int), z
def scores_from_log(name):
    logs = glob.glob(str(K / name / '*.log'))
    if not logs: return ''
    txt = open(logs[0], errors='ignore').read()
    m = re.findall(r'(stage [AB]: OOF macro F1[^"\\]{0,600})', txt)
    return m[-1][:400] if m else ''
if len(sys.argv) == 1:
    runs = sorted(p.name for p in K.iterdir() if (p / 'final_probabilities.npz').exists())
    base = load(runs[0])[1] if runs else None
    for r in runs:
        lo, lt, z = load(r)
        print(f'{r:28s} OOF {mf1(y, lo):.4f} | folds ' + ' '.join(f'{mf1(y[fold==f], lo[fold==f]):.4f}' for f in range(5))
              + f' | pre-refiner {mf1(y, z["oof"].argmax(1)):.4f} | test null {np.mean(lt==0):.3f}')
    pub = pd.read_csv(Path.home() / 'kaggle/wear_lb/pub/wear-hasca-part-2-ranked-links-lb-0-92990/submission_our_run_lb_0_92990.csv').set_index('id').loc[ids].iloc[:, 0].to_numpy()
    for r in runs: print(f'  test agreement {r} vs public 0.92990 file: {np.mean(load(r)[1] == pub):.4f}')
elif sys.argv[1] == 'vote':
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    names = [a for a in sys.argv[2:] if not a.startswith('--') and a != out]
    Lo = np.stack([load(n)[0] for n in names]); Lt = np.stack([load(n)[1] for n in names])
    def vote(L):
        cnt = np.zeros((L.shape[1], 19)); 
        for k in range(L.shape[0]): cnt[np.arange(L.shape[1]), L[k]] += 1 + 1e-3 * (L.shape[0] - k)
        return cnt.argmax(1)
    vo, vt = vote(Lo), vote(Lt)
    print('members', names); print(f'VOTE OOF {mf1(y, vo):.4f} | folds ' + ' '.join(f'{mf1(y[fold==f], vo[fold==f]):.4f}' for f in range(5)))
    if out: pd.DataFrame({'id': ids, 'target_feature': vt}).to_csv(out, index=False); print('wrote', out)
