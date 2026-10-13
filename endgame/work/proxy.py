"""Proxy evaluation of window-probability inputs through the public context baseline minus links:
label propagation over the per-subject video graph -> sharpen -> per-subject Sinkhorn count calibration."""
import numpy as np, scipy.sparse as sp, sys, itertools
from pathlib import Path
import os
WL = Path(os.environ.get("WEAR_LB", Path.home() / "kaggle" / "wear_lb"))  # endgame working dir (data, kernel outputs, caches)
W = WL / 'work'; R = WL / 'gpu_out/runs'
N_CLS = 19; TRAIN_SETS = {0: 2, 14: 2}
sim = dict(np.load(W / 'sim_meta.npz')); y = sim['y']; sbj = sim['sbj']
def macro_f1(y, pred):
    cm = np.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).reshape(N_CLS, N_CLS); tp = np.diag(cm); d = cm.sum(0) + cm.sum(1); pr = d > 0
    return float((2 * tp[pr] / d[pr]).mean())
def lsm(x): x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
def mean_logp(ls): return np.log(np.mean([np.exp(x) for x in ls], 0) + 1e-8)
def label_prop(P0, emb, grp, k=10, tau=0.1, use_p=0.5, alpha=0.9, iters=20):
    E = emb.astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-6; F = np.empty_like(P0)
    for g in np.unique(grp):
        ii = np.flatnonzero(grp == g); n = len(ii); Q = np.sqrt(P0[ii]).astype(np.float32)
        S = E[ii] @ E[ii].T + use_p * (Q @ Q.T); np.fill_diagonal(S, -np.inf)
        nb = np.argpartition(-S, k, axis=1)[:, :k]; rows = np.repeat(np.arange(n), k); vals = S[rows, nb.reshape(-1)]
        Wm = sp.csr_matrix((np.exp((vals - vals.max()) / tau), (rows, nb.reshape(-1))), shape=(n, n)); Wm = Wm.maximum(Wm.T)
        d_ = np.asarray(Wm.sum(1)).ravel(); d_[d_ == 0] = 1; Dm = sp.diags(1 / np.sqrt(d_)); Wn = Dm @ Wm @ Dm
        f0 = P0[ii]; f = f0.copy()
        for _ in range(iters): f = alpha * (Wn @ f) + (1 - alpha) * f0
        F[ii] = f
    return F
def calibrate(P, sbj, sets, per_ex=97, null_min=0.05, iters=50):
    out = np.empty_like(P)
    for s in np.unique(sbj):
        ii = np.flatnonzero(sbj == s); n = len(ii); null = max(n - 18 * per_ex * sets.get(int(s), 1), null_min * n)
        t_ = np.full(P.shape[1], (n - null) / 18); t_[0] = null; Q = P[ii].copy()
        for _ in range(iters): Q *= (t_ / np.maximum(Q.sum(0), 1e-9))[None]; Q /= Q.sum(1, keepdims=True)
        out[ii] = Q
    return out
def finish(P, T=0.5):
    Q = np.clip(P, 1e-12, None) ** (1 / T); return calibrate(Q / Q.sum(1, keepdims=True), sbj, TRAIN_SETS).argmax(1)
runs = {r: np.load(R / r / 'oof_raw.npz') for r in ['final_s0', 'final_s1', 'final_s2', 'final_s3', 'imu_s0', 'imu_s1']}
for r, z in runs.items(): assert (z['y'] == y).all() and (z['grp'] == sbj).all(), r
emb = runs['final_s0']['emb'].astype(np.float32)
fus = mean_logp([runs[f'final_s{i}']['logp'] for i in range(4)]); imu = mean_logp([runs[f'imu_s{i}']['logp'] for i in range(2)])
O = lambda n: np.load(W / 'ours' / f'{n}_oof.npy')
def ev(name, logp):
    logp = lsm(logp); w = macro_f1(y, logp.argmax(1)); c = macro_f1(y, finish(np.exp(logp)))
    P = label_prop(np.exp(logp), emb, sbj); P /= P.sum(1, keepdims=True); f = macro_f1(y, finish(P))
    print(f'{name:55s} window {w:.4f} | +counts {c:.4f} | +LP+counts {f:.4f}', flush=True); return f
EXP = sys.argv[1] if len(sys.argv) > 1 else 'base'
if EXP == 'base':
    ev('theirs fus+0.3imu', fus + 0.3 * imu)
    ev('theirs fus only', fus); ev('theirs imu only', imu)
    for n in ['m_all4', 'm_all3', 'f2_nnHS7_full', 'cand_v3_hcf2', 'm_inertial_only', 'f2_nnHD_full']: ev(f'ours {n}', O(n))
    ev('theirs + m_all3 (1:1)', fus + 0.3 * imu + O('m_all3'))
    ev('theirs + m_all4 (1:1)', fus + 0.3 * imu + O('m_all4'))
    ev('theirs + 0.5 m_all3', fus + 0.3 * imu + 0.5 * O('m_all3'))
    ev('theirs + ours IMU (m_inertial_only)', fus + 0.3 * imu + O('m_inertial_only'))
    ev('theirs + ours IMU (HD + lgb nv v2 nw)', fus + 0.3 * imu + 0.5 * O('f2_nnHD_full') + 0.5 * O('f2_lgb_noVideo_v2_s25_nw'))
    ev('fus + 1.0 imu + ours IMU', fus + 1.0 * imu + O('m_inertial_only'))
if EXP == 'sweep':
    th = lsm(fus + 0.3 * imu); m4 = O('m_all4'); m3 = O('m_all3'); ine = O('m_inertial_only'); c3 = O('cand_v3_hcf2')
    pavg = lambda ps, ws: np.log(sum(w * np.exp(p) for p, w in zip(ps, ws)) / sum(ws) + 1e-9)
    ev('prob-avg m_all4 + theirs (1:1)', pavg([m4, th], [1, 1]))
    ev('prob-avg m_all4 + theirs (2:1)', pavg([m4, th], [2, 1]))
    ev('prob-avg m_all4 + theirs (3:1)', pavg([m4, th], [3, 1]))
    ev('PoE m_all4 + 0.3 theirs', m4 + 0.3 * th)
    ev('PoE m_all4 + 0.5 ours-inertial', m4 + 0.5 * ine)
    ev('prob-avg m_all4 + ours-inertial (2:1)', pavg([m4, ine], [2, 1]))
    ev('prob-avg m_all4 + cand_v3_hcf2 (1:1)', pavg([m4, c3], [1, 1]))
    ev('prob-avg m_all4 + theirs imu (3:1)', pavg([m4, lsm(imu)], [3, 1]))
    ev('m_all4 tempered x0.7', 0.7 * m4); ev('m_all4 tempered x1.5', 1.5 * m4)
if EXP == 'stack':
    import lightgbm as lgb
    fold = sim['fold']; sensor = sim['sensor']
    their = {r: runs[r]['logp'] for r in runs}
    ours = {n: O(n) for n in ['f2_nnHS7_full', 'f2_nnHS7_full_s1', 'f2_nnHCF7_full', 'f2_nnHCF_full', 'f2_nnHD_full', 'f2_nnCF_full',
                              'f2_lgb_noVideo_v2_s25_nw', 'f2_lgb_best_v2_s25', 'lgb_center', 'lgb_noVideo', 'stack4']}
    def run_stack(feats, tag, leaves=15, rounds=400):
        X = np.concatenate([feats[k] for k in feats] + [sensor[:, None]], 1).astype(np.float32); oof = np.zeros((len(y), N_CLS))
        prm = dict(objective='multiclass', num_class=N_CLS, learning_rate=0.05, num_leaves=leaves, min_data_in_leaf=200, feature_fraction=0.5,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, max_bin=63, num_threads=10, verbose=-1, seed=0)
        for f in range(5):
            tr, va = fold != f, fold == f
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], categorical_feature=[X.shape[1] - 1]), rounds)
            oof[va] = m.predict(X[va])
        return ev(tag, np.log(oof + 1e-9))
    run_stack(their, 'STACK theirs 6 runs')
    run_stack(ours, 'STACK ours 11 models')
    run_stack({**their, **ours}, 'STACK theirs+ours 17')
if EXP == 'blend5':
    M4 = {'f2_nnHS7_full': 2, 'f2_nnHS7_full_s1': 2, 'f2_nnHCF7_full': 2, 'f2_nnHCF_full': 1, 'f2_nnHCF_full_s1': 1, 'f2_nnHD_full': 1,
          'f2_nnCF_full': 1, 'f2_nnCF_full_s1': 1, 'f2_nnD_full': 1, 'f2_lgb_noVideo_v2_s25_nw': 1, 'f2_lgb_best_v2_s25': 1,
          'f2_lgb_noVideo_v2_l31': 1, 'lgb_center': 1, 'lgb_noVideo': 1, 'stack1': 2, 'stack3': 2}
    def pav(ws, split='oof'):
        tot = sum(ws.values()); return np.log(sum(w * np.exp(np.load(W / 'ours' / f'{n}_{split}.npy')) for n, w in ws.items()) / tot + 1e-9)
    r4 = pav(M4); print('check m_all4 rebuild max abs diff', np.abs(np.exp(r4) - np.exp(O('m_all4'))).max())
    M5 = dict(M4, f2_nnHS7_full_s2=2, f2_nnHCF7_full_s1=2, f2_nnHD_full_s1=1, stack4=2)
    ev('m_all4 (rebuilt)', r4); ev('m_all5 = m_all4 + new seeds + stack4', pav(M5))
    M6 = {k: v for k, v in M5.items() if not k.startswith('stack')}
    ev('m_all5 without stacks', pav(M6))
    th = lsm(fus + 0.3 * imu)
    ev('m_all5 + theirs 2:1', np.log(2/3*np.exp(pav(M5)) + 1/3*np.exp(th) + 1e-9))
    np.save(W / 'ours' / 'm_all5_oof.npy', pav(M5).astype(np.float32)); np.save(W / 'ours' / 'm_all5_test.npy', pav(M5, 'test').astype(np.float32))
