"""Label-shift adaptation (Saerens et al. EM) per subject, validated on OOF.
usage: python adapt.py --tag blend_nnF_lgb [--per_subject] [--iters 20] [--bias]"""
import argparse, numpy as np, pandas as pd
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--per_subject', action='store_true'); ap.add_argument('--bias', action='store_true', help='also apply OOF-tuned bias after EM')
ap.add_argument('--damp', type=float, default=1.0, help='exponent on prior ratio (1 = full EM, 0 = none)')
ap.add_argument('--out', default=None)
args = ap.parse_args()
idx = load_index(); valid = idx[[f'valid_{s}' for s in SENSORS]].to_numpy(); meta = pd.read_csv(f'{DATA}/test_meta_data.csv')
O = np.load(f'{ROOT}/preds/oof_{args.tag}.npy'); T = np.load(f'{ROOT}/preds/test_{args.tag}.npy')
have = valid & ~np.isnan(O[..., 0])
yy = np.repeat(idx.label.to_numpy(), 4).reshape(-1, 4)[have]; sb = np.repeat(idx.sbj.to_numpy(), 4).reshape(-1, 4)[have]; PO = O[have]
pri_train = np.bincount(yy, minlength=NCLS) / len(yy)

def em(P, pri0, iters, damp):
    """P: (n,C) posteriors under training prior pri0. Returns adapted posteriors."""
    pri = pri0.copy()
    for _ in range(iters):
        W = P * (pri / pri0) ** damp; W /= W.sum(1, keepdims=True)
        pri = W.mean(0)
    return W

def adapt(P, groups):
    out = np.zeros_like(P)
    if args.per_subject:
        for g in np.unique(groups): m = groups == g; out[m] = em(P[m], pri_train, args.iters, args.damp)
    else: out = em(P, pri_train, args.iters, args.damp)
    return out

print(f'OOF raw F1 {macro_f1(yy, PO.argmax(1)):.4f}')
PA = adapt(PO, sb)
print(f'OOF EM-adapted F1 {macro_f1(yy, PA.argmax(1)):.4f}  (per_subject={args.per_subject}, damp={args.damp})')
fo = np.array([SBJ_FOLD[int(s)] for s in sb])
if args.bias:
    print(f'OOF EM + CV-honest bias F1 {tune_bias_cv(PA, yy, fo, iters=1):.4f}   (raw + CV-honest bias {tune_bias_cv(PO, yy, fo, iters=1):.4f})')
# per-subject null share: true vs predicted before/after
for g in np.unique(sb)[:22]:
    m = sb == g
    print(f'  sbj {g:2d} true null {np.mean(yy[m]==0):.2f} raw pred null {np.mean(PO[m].argmax(1)==0):.2f} EM pred null {np.mean(PA[m].argmax(1)==0):.2f}  F1 raw {macro_f1(yy[m], PO[m].argmax(1)):.3f} EM {macro_f1(yy[m], PA[m].argmax(1)):.3f}')
TA = adapt(T, meta.sbj_id.to_numpy())
if args.bias:
    b, _ = tune_bias(PA, yy, iters=2); pred = (np.log(TA + 1e-9) + b).argmax(1); print('bias', np.round(b, 2).tolist())
else: pred = TA.argmax(1)
for g in sorted(meta.sbj_id.unique()):
    m = (meta.sbj_id == g).to_numpy(); print(f'  test sbj {g}: raw null {np.mean(T[m].argmax(1)==0):.2f} -> adapted null {np.mean(pred[m]==0):.2f}')
out = args.out or f'{args.tag}_em{"S" if args.per_subject else ""}{"B" if args.bias else ""}'
pd.DataFrame({'id': meta['id'], 'target_feature': pred}).to_csv(f'{ROOT}/subs/sub_{out}.csv', index=False); np.save(f'{ROOT}/preds/test_{out}.npy', TA)
print('wrote', f'subs/sub_{out}.csv', 'dist', np.bincount(pred, minlength=NCLS).tolist())
