"""LightGBM baseline: hand-crafted inertial feats + PCA video feats + sensor id.
GroupKFold(5) by subject; writes preds/oof_<tag>.npy, preds/test_<tag>.npy, subs/sub_<tag>.csv."""
import argparse, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='lgb_v1')
ap.add_argument('--foldset', type=int, default=1)
ap.add_argument('--feat_v2', action='store_true', help='append inertial feature set v2')
ap.add_argument('--pca_mean', type=int, default=128)
ap.add_argument('--pca_std', type=int, default=64)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--rounds', type=int, default=3000)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--stride', type=int, default=25, help='use only windows with start %% stride == 0')
ap.add_argument('--no_video', action='store_true')
ap.add_argument('--no_inertial', action='store_true')
ap.add_argument('--null_weight', type=float, default=1.0)
ap.add_argument('--es', type=int, default=100, help='early stopping patience on val macro-F1')
ap.add_argument('--vid_center', action='store_true', help='subtract per-subject mean of pooled video features (train: per recording subject; test: per test subject)')
ap.add_argument('--vid_diff', action='store_true', help='add PCA of mean abs frame-to-frame video difference (needs feat_video_diff)')
args = ap.parse_args()

idx = load_index()
FI = np.load(f'{CACHE}/feat_inertial.npy'); FV = np.load(f'{CACHE}/feat_video.npy')
if args.feat_v2:
    FI = np.concatenate([FI, np.load(f'{CACHE}/feat_inertial_v2.npy')], 2)
TFI = np.load(f'{CACHE}/test_feat_inertial.npy'); TFV = np.load(f'{CACHE}/test_feat_video.npy')
if args.feat_v2:
    TFI = np.concatenate([TFI, np.load(f'{CACHE}/test_feat_inertial_v2.npy')], 1)
meta = pd.read_csv(f'{DATA}/test_meta_data.csv')

keep = (idx['start'] % args.stride == 0).to_numpy()
idx = idx[keep].reset_index(drop=True); FI, FV = FI[keep], FV[keep]
print('windows', len(idx))

if args.vid_center:
    for sb in np.unique(idx['sbj']):
        mk = (idx['sbj'] == sb).to_numpy(); FV[mk, :768] -= FV[mk, :768].mean(0)
    for sb in meta['sbj_id'].unique():
        mk = (meta['sbj_id'] == sb).to_numpy(); TFV[mk, :768] -= TFV[mk, :768].mean(0)
# video PCA (fit on train pooled features)
pm = PCA(args.pca_mean, random_state=0).fit(FV[:, :768]); ps = PCA(args.pca_std, random_state=0).fit(FV[:, 768:])
def vid(F): return np.concatenate([pm.transform(F[:, :768]), ps.transform(F[:, 768:])], 1).astype(np.float32)
VW, TV = vid(FV), vid(TFV)
if args.vid_diff:
    FD = np.load(f'{CACHE}/feat_video_diff.npy')[keep]; TFD = np.load(f'{CACHE}/test_feat_video_diff.npy')
    pd1 = PCA(64, random_state=0).fit(FD[:, :768]); pd2 = PCA(32, random_state=0).fit(FD[:, 768:])
    def vdiff(F): return np.concatenate([pd1.transform(F[:, :768]), pd2.transform(F[:, 768:])], 1).astype(np.float32)
    VW = np.concatenate([VW, vdiff(FD)], 1); TV = np.concatenate([TV, vdiff(TFD)], 1)
print('video pca var explained', pm.explained_variance_ratio_.sum().round(3), ps.explained_variance_ratio_.sum().round(3))

# expand windows x valid sensors into rows
valid = idx[[f'valid_{s}' for s in SENSORS]].to_numpy()
wi, si = np.nonzero(valid)
def build(fi_rows, v_rows, sens):
    parts = []
    if not args.no_inertial: parts.append(fi_rows)
    if not args.no_video: parts.append(v_rows)
    parts.append(sens[:, None].astype(np.float32))
    return np.concatenate(parts, 1)
X = build(FI[wi, si], VW[wi], si)
y = idx['label'].to_numpy()[wi]; groups = idx['sbj'].to_numpy()[wi]
Xt = build(TFI, TV, meta['sensor_location'].map(S2I).to_numpy())
sens_col = X.shape[1] - 1
print('X', X.shape, 'Xt', Xt.shape)

params = dict(objective='multiclass', num_class=NCLS, learning_rate=args.lr, num_leaves=args.leaves,
              min_data_in_leaf=50, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, max_bin=127, num_threads=8, verbose=-1, seed=0)
oof = np.zeros((len(X), NCLS), np.float32); tp = np.zeros((len(Xt), NCLS), np.float32)
w = np.where(y == 0, args.null_weight, 1.0)
t0 = time.time(); iters = []
for f, (tr, va) in enumerate(fold_splits(groups, args.foldset)):
    dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], categorical_feature=[sens_col], free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], reference=dtr, categorical_feature=[sens_col])
    yv = y[va]
    def feval(pred, data):
        p = pred.reshape(len(yv), NCLS) if pred.ndim == 1 else pred
        return 'macro_f1', macro_f1(yv, p.argmax(1)), True
    m = lgb.train(dict(params, metric='None'), dtr, args.rounds, valid_sets=[dva], feval=feval,
                  callbacks=[lgb.early_stopping(args.es, verbose=False), lgb.log_evaluation(0)])
    oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
    tp += m.predict(Xt, num_iteration=m.best_iteration) / args.folds
    iters.append(m.best_iteration)
    print(f'fold {f} sbj={sorted(set(groups[va]))} best_it={m.best_iteration} '
          f'F1={macro_f1(y[va], oof[va].argmax(1)):.4f} {time.time()-t0:.0f}s', flush=True)
f1 = macro_f1(y, oof.argmax(1))
b, f1b = tune_bias(oof, y)
print(f'OOF macro F1 {f1:.4f}  bias-tuned {f1b:.4f}  bias={np.round(b,2).tolist()}')
# per-sensor-location breakdown
for s in range(4):
    mk = si == s
    print(f'  {SENSORS[s]:10s} F1={macro_f1(y[mk], oof[mk].argmax(1)):.4f}')
np.save(f'{ROOT}/preds/oof_{args.tag}.npy', oof); np.save(f'{ROOT}/preds/test_{args.tag}.npy', tp)
np.save(f'{ROOT}/preds/oofmeta_{args.tag}.npy', np.stack([np.where(keep)[0][wi], si, y], 1))  # global window idx
pred = (np.log(tp + 1e-9) + b).argmax(1)
sub = pd.DataFrame({'id': meta['id'], 'target_feature': pred}); sub.to_csv(f'{ROOT}/subs/sub_{args.tag}.csv', index=False)
print('test pred dist', np.bincount(pred, minlength=NCLS).tolist())
with open(f'{ROOT}/results.log', 'a') as fh:
    fh.write(f'{args.tag}\tOOF={f1:.4f}\tbias={f1b:.4f}\titers={iters}\t{json.dumps(vars(args))}\n')
