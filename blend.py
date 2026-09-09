"""Blend OOF/test probabilities of several models (rows = window x sensor), tune bias, write submission.
usage: python blend.py --tag blend1 --models lgb_v1:1 nn_v1:1"""
import argparse, numpy as np, pandas as pd
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--geo', action='store_true', help='geometric mean instead of arithmetic')
ap.add_argument('--null_bias_override', type=float, default=None)
ap.add_argument('--foldset', type=int, default=1)
ap.add_argument('--opt_weights', type=int, default=0, help='rounds of coordinate ascent on blend weights (raw macro-F1 on rows with all models)')
args = ap.parse_args()
idx = load_index(); valid = idx[[f'valid_{s}' for s in SENSORS]].to_numpy()
meta = pd.read_csv(f'{DATA}/test_meta_data.csv')
oofs, tests, ws = [], [], []
for m in args.models:
    name, w = (m.split(':') + ['1'])[:2]; w = float(w)
    o = np.load(f'{ROOT}/preds/oof_{name}.npy'); t = np.load(f'{ROOT}/preds/test_{name}.npy')
    if o.ndim == 2:  # lgb format: rows = valid (window, sensor) pairs (global window idx) -> (n_win, 4, C), NaN where absent
        mt = np.load(f'{ROOT}/preds/oofmeta_{name}.npy'); full = np.full((len(idx), 4, NCLS), np.nan, np.float32)
        full[mt[:, 0], mt[:, 1]] = o; o = full
    else:
        o = o.copy(); o[~valid] = np.nan
    have = ~np.isnan(o[..., 0])
    yy0 = np.repeat(idx.label.to_numpy(), 4).reshape(-1, 4)
    print(f'{name:12s} w={w} rows={have.sum()} OOF F1={macro_f1(yy0[have], o[have].argmax(1)):.4f}')
    oofs.append(o); tests.append(t); ws.append(w)
ws = np.array(ws) / sum(ws)
if args.opt_weights > 0:
    have0 = np.all([~np.isnan(o[..., 0]) for o in oofs], 0) & valid
    y0 = np.repeat(idx.label.to_numpy(), 4).reshape(-1, 4)[have0]
    O0 = [o[have0] for o in oofs]
    def score(w):
        w = np.asarray(w) / np.sum(w); P = sum(wi * o for wi, o in zip(w, O0)); return macro_f1(y0, P.argmax(1))
    best = score(ws); grid = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
    for r in range(args.opt_weights):
        for i in range(len(ws)):
            cur = ws[i]; bw = cur
            for g in grid:
                w2 = ws.copy(); w2[i] = g
                if w2.sum() == 0: continue
                sc = score(w2)
                if sc > best + 1e-5: best, bw = sc, g
            ws[i] = bw
        print(f'opt round {r}: raw F1 {best:.4f} weights {np.round(ws, 2).tolist()}')
    ws = ws / ws.sum()
if args.geo:
    O = np.exp(sum(w * np.log(o + 1e-9) for w, o in zip(ws, oofs))); T = np.exp(sum(w * np.log(t + 1e-9) for w, t in zip(ws, tests)))
else:
    O = sum(w * o for w, o in zip(ws, oofs)); T = sum(w * t for w, t in zip(ws, tests))
have = np.all([~np.isnan(o[..., 0]) for o in oofs], 0) & valid
print('rows with all models:', have.sum())
yy = np.repeat(idx.label.to_numpy(), 4).reshape(-1, 4)[have]; PO = O[have]; vv = np.ones(len(yy), bool)
f1 = macro_f1(yy[vv], PO[vv].argmax(1)); b, f1b = tune_bias(PO[vv], yy[vv])
sbj_rows = np.repeat(idx.sbj.to_numpy(), 4).reshape(-1, 4)[have]
fo = np.array([fold_map(args.foldset)[int(s)] for s in sbj_rows])
f1cv = tune_bias_cv(PO, yy, fo)
print(f'BLEND OOF F1 {f1:.4f} bias-tuned {f1b:.4f} CV-honest-tuned {f1cv:.4f} bias={np.round(b,2).tolist()}')
if args.null_bias_override is not None: b[0] = args.null_bias_override; print('null bias overridden ->', b[0], 'OOF F1 with override', round(macro_f1(yy, (np.log(PO + 1e-9) + b).argmax(1)), 4))
pred = (np.log(T + 1e-9) + b).argmax(1)
pd.DataFrame({'id': meta['id'], 'target_feature': pred}).to_csv(f'{ROOT}/subs/sub_{args.tag}.csv', index=False)
np.save(f'{ROOT}/preds/oof_{args.tag}.npy', O); np.save(f'{ROOT}/preds/test_{args.tag}.npy', T)
print('test pred dist', np.bincount(pred, minlength=NCLS).tolist())
with open(f'{ROOT}/results.log', 'a') as fh: fh.write(f'{args.tag}\tOOF={f1:.4f}\tbias={f1b:.4f}\tcvbias={f1cv:.4f}\tmodels={args.models}\n')
