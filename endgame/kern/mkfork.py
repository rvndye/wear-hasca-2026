"""Build a fork of goodpjw2008 Part 2 that blends our window log-probs into d['logp'] / t['logp'].
usage: mkfork.py <slug> <ours_model|none> <ours_w> [--push]"""
import json, sys, copy, subprocess, os
from pathlib import Path
WL = Path(os.environ.get('WEAR_LB', Path.home() / 'kaggle' / 'wear_lb'))   # endgame working dir
OWNER = os.environ.get('KAGGLE_OWNER', 'rvndyerwin')                        # Kaggle user that owns the forks and our dataset
# base notebook: run endgame/fetch_refs.sh first (pulls goodpjw2008's Part 2 notebook into endgame/ref_public/)
SRC = Path(os.environ.get('WEAR_P2_NB', Path(__file__).resolve().parents[1] / 'ref_public/wear-hasca-part-2-ranked-links-lb-0-92990/wear-hasca-part-2-ranked-links-lb-0-92990.ipynb'))
slug, model, w = sys.argv[1], sys.argv[2], float(sys.argv[3])
counts = '--counts' in sys.argv
gpu = '--gpu' in sys.argv
fullrep = '--fullrep' in sys.argv
whiten = '--whiten' in sys.argv
seedoff = int(sys.argv[sys.argv.index('--seedoff') + 1]) if '--seedoff' in sys.argv else 0
nb = json.load(open(SRC))
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
HEADER = f'''# === rvndyerwin fork header: pin input paths, choose window blend ===
import os
from pathlib import Path
OURS_MODEL, OURS_W = {model!r}, {w!r}
COUNTS_ADDON = {counts!r}
FULLREP = {fullrep!r}
SEEDOFF2 = {seedoff!r}
EXT_O = EXT_T = None
if {whiten!r}:
    os.environ["WEAR_WHITEN"] = "0.5"
_art = sorted(Path("/kaggle/input").glob("**/runs/final_s0/oof_raw.npz"))
if _art and not os.environ.get("WEAR_ARTIFACTS"):
    os.environ["WEAR_ARTIFACTS"] = str(_art[0].parent.parent.parent)
_ours = sorted(Path("/kaggle/input").glob(f"**/{{OURS_MODEL}}_oof.npy")) if OURS_MODEL != "none" else []
OURS_DIR = _ours[0].parent if _ours else None
print("WEAR_ARTIFACTS", os.environ.get("WEAR_ARTIFACTS"), "| OURS_DIR", OURS_DIR, "| model", OURS_MODEL, "w", OURS_W)
assert OURS_MODEL == "none" or OURS_DIR is not None, "our dataset not found"
assert os.environ.get("WEAR_ARTIFACTS"), "gpu-outputs dataset not found"
'''
nb['cells'].insert(0, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': HEADER.splitlines(True)})
INJECT = '''
    if OURS_DIR is not None and OURS_W > 0:   # rvndyerwin: probability-average our window log-probs into the input blend
        _oo = np.load(OURS_DIR / f"{OURS_MODEL}_oof.npy").astype(np.float64)
        _ot = np.load(OURS_DIR / f"{OURS_MODEL}_test.npy").astype(np.float64)
        assert _oo.shape == d["logp"].shape and _ot.shape == t["logp"].shape, (_oo.shape, _ot.shape)
        def _mix(ours, theirs, w):
            p = w * np.exp(ours) + (1 - w) * np.exp(theirs.astype(np.float64))
            return np.log(p / p.sum(1, keepdims=True) + 1e-12).astype(np.float32)
        print(f"ours {OURS_MODEL}: window OOF macro F1 {macro_f1(d['y'], _oo.argmax(1)):.4f}; theirs {macro_f1(d['y'], d['logp'].argmax(1)):.4f}")
        EXT_O, EXT_T = np.exp(_oo), np.exp(_ot)
        if FULLREP:   # our evidence also in the S3 fusion / IMU slots (same 19 columns each, so the 533-column assert holds)
            _io = np.load(OURS_DIR / "m_inertial_only_oof.npy").astype(np.float32); _it = np.load(OURS_DIR / "m_inertial_only_test.npy").astype(np.float32)
            d["fusion_logp"], t["fusion_logp"] = _oo.astype(np.float32), _ot.astype(np.float32)
            d["imu_logp"], t["imu_logp"] = _io, _it
            print(f"FULLREP: S3 fusion slot <- {OURS_MODEL}, IMU slot <- m_inertial_only ({macro_f1(d['y'], _io.argmax(1)):.4f})")
        d["logp"], t["logp"] = _mix(_oo, d["logp"], OURS_W), _mix(_ot, t["logp"], OURS_W)
        print(f"mixed (w_ours={OURS_W}): window OOF macro F1 {macro_f1(d['y'], d['logp'].argmax(1)):.4f}")
        del _oo, _ot
'''
done = False
for c in nb['cells']:
    src = ''.join(c['source'])
    if 'with stage("W4 blend"):' in src:
        anchor = '    del _o, _fus, _imu, _fus_t, _imu_t\n'
        assert anchor in src
        c['source'] = src.replace(anchor, INJECT.lstrip('\n') + anchor).splitlines(True); done = True
assert done
if seedoff:
    hits = 0
    for c in nb['cells']:
        src = ''.join(c['source']); new = src
        new = new.replace('{**LK_PARAMS, "seed": seed}', '{**LK_PARAMS, "seed": seed + 10 * SEEDOFF2}')
        new = new.replace('{**LK_PARAMS, **LK_RANK, "seed": seed}', '{**LK_PARAMS, **LK_RANK, "seed": seed + 10 * SEEDOFF2}')
        new = new.replace('BAG_TAU, 1000 * k + s)', 'BAG_TAU, 1000 * k + s + 100000 * SEEDOFF2)')
        if new != src: hits += 1; c['source'] = new.splitlines(True)
    assert hits >= 2, hits
if counts:
    hit = 0
    for c in nb['cells']:
        src = ''.join(c['source'])
        if 'def decode(B_oof, B_test, tag, passes=2):' in src:
            a1 = 'X, key = profile_features([P, Bp], sbj, TRAIN_SETS)'
            a2 = 'Xt, kt = profile_features([Pt, Btp], t["sbj"], {})'
            assert a1 in src and a2 in src
            src = src.replace(a1, 'X, key = profile_features([P, Bp] + ([EXT_O] if (COUNTS_ADDON and EXT_O is not None) else []), sbj, TRAIN_SETS)')
            src = src.replace(a2, 'Xt, kt = profile_features([Pt, Btp] + ([EXT_T] if (COUNTS_ADDON and EXT_T is not None) else []), t["sbj"], {})')
            c['source'] = src.splitlines(True); hit += 1
    assert hit == 1, hit
d = WL / 'kern' / slug; d.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(d / 'nb.ipynb', 'w'), indent=1)
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "nb.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": ["goodpjw2008/wear-hasca-part2-gpu-outputs", f"{OWNER}/wear-ours-tile-logp"],
        "kernel_sources": [], "competition_sources": ["3rd-wear-dataset-challenge-hasca-2026"], "model_sources": []}
json.dump(meta, open(d / 'kernel-metadata.json', 'w'), indent=1)
print('built', d)
if '--push' in sys.argv:
    env = dict(os.environ, KAGGLE_USERNAME='x', KAGGLE_KEY='x', KAGGLE_API_TOKEN=open(Path.home() / '.kaggle/access_token').read().strip())
    r = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], capture_output=True, text=True, env=env)
    print((r.stdout + r.stderr).replace('\n\n', '\n')[-600:])
