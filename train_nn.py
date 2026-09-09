"""PyTorch multimodal fusion model, GroupKFold(5) by subject (same folds as LightGBM).
Each training sample = one (window, sensor) pair: inertial (50,3) + sensor id + 15x768 video."""
import argparse, json, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='nn_v1')
ap.add_argument('--foldset', type=int, default=1)
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--bs', type=int, default=256, help='windows per batch (x4 sensors)')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-2)
ap.add_argument('--dim', type=int, default=192)
ap.add_argument('--vid_layers', type=int, default=1)
ap.add_argument('--mod_drop', type=float, default=0.15, help='prob of dropping a modality per sample')
ap.add_argument('--smooth', type=float, default=0.05)
ap.add_argument('--stride', type=int, default=25)
ap.add_argument('--folds', type=str, default='0,1,2,3,4')
ap.add_argument('--no_video', action='store_true')
ap.add_argument('--no_inertial', action='store_true')
ap.add_argument('--aug', type=int, default=1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--vid_center', action='store_true', help='subtract per-subject mean video feature (train per subject, test per test subject)')
ap.add_argument('--null_weight', type=float, default=1.0)
ap.add_argument('--vid_inorm', action='store_true', help='per-window instance norm of video frames over time')
ap.add_argument('--eval_every', type=int, default=1)
ap.add_argument('--hand_feats', action='store_true', help='add standardized hand-crafted inertial features (v1+v2) via an MLP branch')
ap.add_argument('--rot_deg', type=float, default=15.0, help='max random rotation (deg) for inertial augmentation')
ap.add_argument('--vid_noise', type=float, default=0.0, help='gaussian noise std (in units of per-dim std ~0.36) added to video input in training')
ap.add_argument('--vid_static', action='store_true', help='with --vid_inorm: also feed the (centered) per-window mean frame through a small MLP')
ap.add_argument('--vid_drop', type=float, default=0.1, help='dropout on raw 768-d video input')
ap.add_argument('--mixup', type=float, default=0.0, help='mixup alpha (0 = off)')
ap.add_argument('--vid_lr_mult', type=float, default=1.0, help='lr multiplier for video encoder')
ap.add_argument('--scene_swap', type=float, default=0.0, help='prob of replacing a sample\'s subject-mean video component with another training subject\'s')
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ---------------- data ----------------
idx = load_index()
idx = idx[idx['start'] % args.stride == 0].reset_index(drop=True)
offs = json.load(open(f'{CACHE}/video_offsets.json'))
VALL = np.load(f'{CACHE}/video_all_f16.npy', mmap_mode='r')
recs = sorted(idx['rec'].unique())
X = np.zeros((len(idx), 4, WIN, 3), np.float32)
VF = np.zeros((len(idx),), np.int64)  # first video frame index into VALL
for rec, g in idx.groupby('rec', sort=False):
    st = g['start'].to_numpy()
    X[g.index] = np.nan_to_num(inertial_windows(rec, st))
    VF[g.index] = offs[rec] + st * 3 // 5 + 8
y = idx['label'].to_numpy(np.int64); groups = idx['sbj'].to_numpy()
valid = idx[[f'valid_{s}' for s in SENSORS]].to_numpy()
print('windows', len(idx), 'loading video to RAM...', flush=True)
t0 = time.time()
# gather all needed frames into RAM as float16 (n, 15, 768)
fr = (VF[:, None] + np.arange(15)[None]).reshape(-1)
V = np.asarray(VALL[fr]).reshape(len(idx), 15, 768)
print(f'video {V.shape} {V.nbytes/1e9:.1f}GB in {time.time()-t0:.0f}s', flush=True)

ti = np.nan_to_num(np.load(f'{DATA}/test_inertial_data.npy').astype(np.float32))          # (N,50,3)
tv = np.ascontiguousarray(np.load(f'{DATA}/test_videomae_data.npy').transpose(0, 2, 1)).astype(np.float16)  # (N,15,768)
meta = pd.read_csv(f'{DATA}/test_meta_data.csv'); tsens = meta['sensor_location'].map(S2I).to_numpy()
HF = THF = None
if args.hand_feats:
    keep = (load_index()['start'] % args.stride == 0).to_numpy()
    HF = np.concatenate([np.load(f'{CACHE}/feat_inertial.npy')[keep], np.load(f'{CACHE}/feat_inertial_v2.npy')[keep]], 2).astype(np.float32)  # (n,4,F)
    THF = np.concatenate([np.load(f'{CACHE}/test_feat_inertial.npy'), np.load(f'{CACHE}/test_feat_inertial_v2.npy')], 1).astype(np.float32)
    HF = np.nan_to_num(HF); THF = np.nan_to_num(THF)
    hmu = HF.reshape(-1, HF.shape[-1]).mean(0); hsd = HF.reshape(-1, HF.shape[-1]).std(0) + 1e-6
    HF = np.clip((HF - hmu) / hsd, -6, 6); THF = np.clip((THF - hmu) / hsd, -6, 6)
    print('hand feats', HF.shape, flush=True)
MU = np.zeros((32, 768), np.float32)
for sb in np.unique(groups):
    MU[sb] = V[groups == sb].astype(np.float32).mean((0, 1))
MU_t = torch.from_numpy(MU).to(dev)
if args.vid_center:
    for sb in np.unique(groups):
        mk = groups == sb; mu = V[mk].astype(np.float32).mean((0, 1)); V[mk] = (V[mk].astype(np.float32) - mu).astype(np.float16)
    for sb in meta['sbj_id'].unique():
        mk = (meta['sbj_id'] == sb).to_numpy(); mu = tv[mk].astype(np.float32).mean((0, 1)); tv[mk] = (tv[mk].astype(np.float32) - mu).astype(np.float16)
    print('video centered per subject', flush=True)

# ---------------- model ----------------
class ConvBlock(nn.Module):
    def __init__(self, ci, co, k, stride=1):
        super().__init__()
        self.c = nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(co)
        self.skip = nn.Conv1d(ci, co, 1, stride=stride, bias=False) if (ci != co or stride != 1) else nn.Identity()
    def forward(self, x): return F.gelu(self.bn(self.c(x)) + self.skip(x))

class InertialEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(4, 64, 5), ConvBlock(64, 128, 5), ConvBlock(128, 128, 3, 2),
                                 ConvBlock(128, 256, 3), ConvBlock(256, 256, 3, 2))
        self.proj = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d), nn.GELU())
    def forward(self, x):  # x (B,50,3)
        x = x.transpose(1, 2)
        x = torch.cat([x, x.norm(dim=1, keepdim=True)], 1)
        h = self.net(x)
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))

class VideoEnc(nn.Module):
    def __init__(self, d, layers):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(768), nn.Dropout(args.vid_drop), nn.Linear(768, d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, 15, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, activation='gelu',
                                        batch_first=True, norm_first=True), layers, enable_nested_tensor=False) if layers > 0 else None
        self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU())
        self.static = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.3), nn.Linear(768, d // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d // 2, d)) if args.vid_static else None
    def forward(self, v):  # (B,15,768)
        stat = None
        if args.vid_inorm:
            mu = v.mean(1, keepdim=True)
            if args.vid_static: stat = self.static(mu[:, 0])
            v = (v - mu) / (v.std(1, keepdim=True) + 1e-3)
        h = self.inp(v) + self.pos
        if self.tf is not None: h = self.tf(h)
        a = torch.softmax(self.att(h), 1)
        out = self.proj(torch.cat([(h * a).sum(1), h.mean(1)], 1))
        return out + stat if stat is not None else out

class Fusion(nn.Module):
    def __init__(self, d, vid_layers):
        super().__init__()
        self.ie, self.ve = InertialEnc(d), VideoEnc(d, vid_layers)
        self.se = nn.Embedding(4, 16)
        nh = HF.shape[-1] if HF is not None else 0
        self.hf = nn.Sequential(nn.Dropout(0.2), nn.Linear(nh, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, d), nn.LayerNorm(d), nn.GELU()) if nh else None
        self.head = nn.Sequential(nn.Linear(2 * d + 16 + (d if nh else 0), 2 * d), nn.GELU(), nn.Dropout(0.3), nn.Linear(2 * d, NCLS))
    def forward(self, x, v, s, mask_i=None, mask_v=None, hf=None):
        hi, hv = self.ie(x), self.ve(v)
        if mask_i is not None: hi = hi * mask_i[:, None]
        if mask_v is not None: hv = hv * mask_v[:, None]
        parts = [hi, hv, self.se(s)]
        if self.hf is not None:
            hh = self.hf(hf)
            if mask_i is not None: hh = hh * mask_i[:, None]
            parts.append(hh)
        return self.head(torch.cat(parts, 1))

def augment(x):  # x (B,50,3) tensor on device
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 1, 1, device=x.device))          # amplitude scaling
    x = x + 0.02 * torch.randn_like(x)                                    # jitter
    # random small rotation about a random axis
    ang = (torch.rand(B, device=x.device) - 0.5) * 2 * (args.rot_deg * math.pi / 180)
    ax = F.normalize(torch.randn(B, 3, device=x.device), dim=1)
    K = torch.zeros(B, 3, 3, device=x.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -ax[:, 2], ax[:, 1], ax[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -ax[:, 0], -ax[:, 1], ax[:, 0]
    R = torch.eye(3, device=x.device)[None] + torch.sin(ang)[:, None, None] * K + (1 - torch.cos(ang))[:, None, None] * (K @ K)
    return torch.einsum('btc,bdc->btd', x, R)

def run_fold(tr, va):
    model = Fusion(args.dim, args.vid_layers).to(dev)
    vid_params = list(model.ve.parameters()); vid_ids = {id(p) for p in vid_params}
    other = [p for p in model.parameters() if id(p) not in vid_ids]
    opt = torch.optim.AdamW([{'params': other, 'lr': args.lr}, {'params': vid_params, 'lr': args.lr * args.vid_lr_mult}], weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr, args.lr * args.vid_lr_mult], total_steps=steps, pct_start=0.1)
    cw = torch.ones(NCLS, device=dev); cw[0] = args.null_weight
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.smooth)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(tr); tl, n = 0, 0; t0 = time.time()
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            xb = torch.from_numpy(X[b]).to(dev)                     # (B,4,50,3)
            vb = torch.from_numpy(V[b].astype(np.float32)).to(dev)  # (B,15,768)
            yb = torch.from_numpy(y[b]).to(dev); vm = torch.from_numpy(valid[b]).to(dev)
            B = len(b)
            xb = xb.reshape(B * 4, WIN, 3); sb = torch.arange(4, device=dev).repeat(B)
            vb = vb[:, None].expand(B, 4, 15, 768).reshape(B * 4, 15, 768)
            yb = yb.repeat_interleave(4); vm = vm.reshape(-1)
            hb = torch.from_numpy(HF[b]).to(dev).reshape(B * 4, -1) if HF is not None else None
            if args.aug: xb = augment(xb)
            if args.vid_noise > 0: vb = vb + args.vid_noise * 0.36 * torch.randn_like(vb)
            if args.scene_swap > 0:
                sbj_b = torch.from_numpy(groups[b]).to(dev).repeat_interleave(4)
                tr_sbjs = torch.from_numpy(np.unique(groups[tr])).to(dev)
                other = tr_sbjs[torch.randint(len(tr_sbjs), (B * 4,), device=dev)]
                delta = (MU_t[other] - MU_t[sbj_b]) * (torch.rand(B * 4, device=dev) < args.scene_swap).float()[:, None]
                vb = vb + delta[:, None, :]
            mi = torch.ones(B * 4, device=dev); mv = torch.ones(B * 4, device=dev)
            if args.mod_drop > 0:
                r = torch.rand(B * 4, device=dev)
                mi = (r > args.mod_drop).float(); mv = ((r < 1 - args.mod_drop) | (r > 1 - args.mod_drop / 2)).float()
                mv = torch.where(mi == 0, torch.ones_like(mv), mv)  # never drop both
            if args.no_video: mv = torch.zeros_like(mv)
            if args.no_inertial: mi = torch.zeros_like(mi)
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup)); perm2 = torch.randperm(B * 4, device=dev)
                xb = lam * xb + (1 - lam) * xb[perm2]; vb = lam * vb + (1 - lam) * vb[perm2]
                if hb is not None: hb = lam * hb + (1 - lam) * hb[perm2]
                out = model(xb, vb, sb, mi, mv, hb)
                loss = lam * crit(out[vm], yb[vm]) + (1 - lam) * crit(out[vm], yb[perm2][vm])
            else:
                out = model(xb, vb, sb, mi, mv, hb)
                loss = crit(out[vm], yb[vm])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(b); n += len(b)
        f1 = None
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            p = predict(model, va); f1 = macro_f1(y[va].repeat(4)[valid[va].reshape(-1)], p.argmax(1)[valid[va].reshape(-1)])
        print(f'  ep {ep+1} loss {tl/n:.4f} ' + (f'valF1 {f1:.4f} ' if f1 else '') + f'{time.time()-t0:.0f}s', flush=True)
    return model

@torch.no_grad()
def predict(model, ids, bs=512):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]; B = len(b)
        xb = torch.from_numpy(X[b]).to(dev).reshape(B * 4, WIN, 3)
        vb = torch.from_numpy(V[b].astype(np.float32)).to(dev)[:, None].expand(B, 4, 15, 768).reshape(B * 4, 15, 768)
        sb = torch.arange(4, device=dev).repeat(B)
        mi = torch.zeros(B * 4, device=dev) if args.no_inertial else None
        mv = torch.zeros(B * 4, device=dev) if args.no_video else None
        hb = torch.from_numpy(HF[b]).to(dev).reshape(B * 4, -1) if HF is not None else None
        out.append(torch.softmax(model(xb, vb, sb, mi, mv, hb), 1).cpu().numpy())
    return np.concatenate(out)  # (len(ids)*4, 19) ordered window-major, sensor-minor

@torch.no_grad()
def predict_test(model, bs=1024):
    model.eval(); out = []
    for i in range(0, len(ti), bs):
        xb = torch.from_numpy(ti[i:i + bs]).to(dev); vb = torch.from_numpy(tv[i:i + bs].astype(np.float32)).to(dev)
        sb = torch.from_numpy(tsens[i:i + bs]).to(dev)
        mi = torch.zeros(len(xb), device=dev) if args.no_inertial else None
        mv = torch.zeros(len(xb), device=dev) if args.no_video else None
        hb = torch.from_numpy(THF[i:i + bs]).to(dev) if THF is not None else None
        out.append(torch.softmax(model(xb, vb, sb, mi, mv, hb), 1).cpu().numpy())
    return np.concatenate(out)

folds = [int(f) for f in args.folds.split(',')]
oof = np.zeros((len(idx), 4, NCLS), np.float32); tp = np.zeros((len(ti), NCLS), np.float32)
for f, (tr, va) in enumerate(fold_splits(groups, args.foldset)):
    if f not in folds: continue
    print(f'fold {f} sbj={sorted(set(groups[va]))} ntr={len(tr)} nva={len(va)}', flush=True)
    model = run_fold(tr, va)
    oof[va] = predict(model, va).reshape(len(va), 4, NCLS)
    tp += predict_test(model) / len(folds)
    torch.save(model.state_dict(), f'{ROOT}/preds/model_{args.tag}_f{f}.pt')
if len(folds) < 5:
    vaidx = np.concatenate([va for f, (tr, va) in enumerate(fold_splits(groups, args.foldset)) if f in folds])
    yy = np.repeat(y[vaidx], 4); vv = valid[vaidx].reshape(-1); po = oof[vaidx].reshape(-1, NCLS)
    f1 = macro_f1(yy[vv], po[vv].argmax(1)); b, f1b = tune_bias(po[vv], yy[vv], iters=1)
    print(f'PARTIAL folds={folds} OOF macro F1 {f1:.4f} bias-tuned {f1b:.4f}')
    with open(f'{ROOT}/results.log', 'a') as fh:
        fh.write(f'{args.tag}\tPARTIAL{folds}\tOOF={f1:.4f}\tbias={f1b:.4f}\t{json.dumps(vars(args))}\n')
if len(folds) == 5:
    yy = np.repeat(y, 4); vv = valid.reshape(-1); po = oof.reshape(-1, NCLS)
    f1 = macro_f1(yy[vv], po[vv].argmax(1)); b, f1b = tune_bias(po[vv], yy[vv])
    print(f'OOF macro F1 {f1:.4f} bias-tuned {f1b:.4f}')
    for s in range(4):
        mk = valid[:, s]; print(f'  {SENSORS[s]:10s} F1={macro_f1(y[mk], oof[mk, s].argmax(1)):.4f}')
    np.save(f'{ROOT}/preds/oof_{args.tag}.npy', oof); np.save(f'{ROOT}/preds/test_{args.tag}.npy', tp)
    pred = (np.log(tp + 1e-9) + b).argmax(1)
    pd.DataFrame({'id': meta['id'], 'target_feature': pred}).to_csv(f'{ROOT}/subs/sub_{args.tag}.csv', index=False)
    with open(f'{ROOT}/results.log', 'a') as fh:
        fh.write(f'{args.tag}\tOOF={f1:.4f}\tbias={f1b:.4f}\t{json.dumps(vars(args))}\n')
