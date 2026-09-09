"""Level-2 LightGBM stacker: NN OOF probabilities (fold set 2) + hand-crafted inertial feats + sensor id.
Same fold set 2 -> honest OOF. usage: python stack.py --tag stack1 --nn f2_nnHCF_full f2_nnHCF_full_s1 f2_nnHD_full"""
import argparse, json, numpy as np, pandas as pd, lightgbm as lgb
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='stack1'); ap.add_argument('--nn', nargs='+', required=True)
ap.add_argument('--lgb', nargs='*', default=[], help='lgb OOF tags (rows = valid window/sensor pairs) to include as features')
ap.add_argument('--feats', action='store_true', help='include hand-crafted inertial feats v1+v2')
ap.add_argument('--stride', type=int, default=25); ap.add_argument('--lr', type=float, default=0.05); ap.add_argument('--leaves', type=int, default=31)
args = ap.parse_args()
idx = load_index(); valid = idx[[f'valid_{s}' for s in SENSORS]].to_numpy(); meta = pd.read_csv(f'{DATA}/test_meta_data.csv')
keep = (idx['start'] % args.stride == 0).to_numpy()
F, TF, names = [], [], []
for t in args.nn:
    o = np.load(f'{ROOT}/preds/oof_{t}.npy'); te = np.load(f'{ROOT}/preds/test_{t}.npy')
    F.append(np.log(o + 1e-6)); TF.append(np.log(te + 1e-6)); names.append(t)
for t in args.lgb:
    o = np.load(f'{ROOT}/preds/oof_{t}.npy'); mt = np.load(f'{ROOT}/preds/oofmeta_{t}.npy'); full = np.full((len(idx), 4, NCLS), np.nan, np.float32); full[mt[:, 0], mt[:, 1]] = o
    F.append(np.log(full + 1e-6)); TF.append(np.log(np.load(f'{ROOT}/preds/test_{t}.npy') + 1e-6)); names.append(t)
F = np.concatenate(F, 2); TF = np.concatenate(TF, 1)   # (n_win,4,K*19), (N,K*19)
if args.feats:
    HF = np.concatenate([np.load(f'{CACHE}/feat_inertial.npy'), np.load(f'{CACHE}/feat_inertial_v2.npy')], 2)
    THF = np.concatenate([np.load(f'{CACHE}/test_feat_inertial.npy'), np.load(f'{CACHE}/test_feat_inertial_v2.npy')], 1)
    F = np.concatenate([F, HF], 2); TF = np.concatenate([TF, THF], 1)
wi, si = np.nonzero(valid & keep[:, None])
X = np.concatenate([F[wi, si], si[:, None].astype(np.float32)], 1); y = idx.label.to_numpy()[wi]; groups = idx.sbj.to_numpy()[wi]
Xt = np.concatenate([TF, meta['sensor_location'].map(S2I).to_numpy()[:, None].astype(np.float32)], 1)
ok = ~np.isnan(X).any(1); X, y, groups, wi, si = X[ok], y[ok], groups[ok], wi[ok], si[ok]
print('stack X', X.shape, 'Xt', Xt.shape, 'models', names, flush=True)
params = dict(objective='multiclass', num_class=NCLS, learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127, num_threads=8, verbose=-1, seed=0, metric='None')
oof = np.zeros((len(X), NCLS), np.float32); tp = np.zeros((len(Xt), NCLS), np.float32); sc = X.shape[1] - 1
for f, (tr, va) in enumerate(fold_splits(groups, 2)):
    yv = y[va]
    def feval(pred, data):
        p = pred.reshape(len(yv), NCLS) if pred.ndim == 1 else pred; return 'macro_f1', macro_f1(yv, p.argmax(1)), True
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=[sc]); dva = lgb.Dataset(X[va], y[va], reference=dtr, categorical_feature=[sc])
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], feval=feval, callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[va] = m.predict(X[va], num_iteration=m.best_iteration); tp += m.predict(Xt, num_iteration=m.best_iteration) / 5
    print(f'fold {f} best_it={m.best_iteration} F1={macro_f1(yv, oof[va].argmax(1)):.4f}', flush=True)
fo = np.array([SBJ_FOLD2[int(g)] for g in groups])
f1 = macro_f1(y, oof.argmax(1)); b, f1b = tune_bias(oof, y); f1cv = tune_bias_cv(oof, y, fo, iters=1)
print(f'STACK OOF F1 {f1:.4f} bias-tuned {f1b:.4f} CV-honest {f1cv:.4f}')
np.save(f'{ROOT}/preds/oof_{args.tag}.npy', oof); np.save(f'{ROOT}/preds/test_{args.tag}.npy', tp); np.save(f'{ROOT}/preds/oofmeta_{args.tag}.npy', np.stack([wi, si, y], 1))
pred = (np.log(tp + 1e-9) + b).argmax(1); pd.DataFrame({'id': meta['id'], 'target_feature': pred}).to_csv(f'{ROOT}/subs/sub_{args.tag}.csv', index=False)
print('test dist', np.bincount(pred, minlength=NCLS).tolist())
with open(f'{ROOT}/results.log', 'a') as fh: fh.write(f'{args.tag}\tSTACK OOF={f1:.4f}\tbias={f1b:.4f}\tcv={f1cv:.4f}\t{json.dumps(vars(args))}\n')
