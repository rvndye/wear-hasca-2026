"""List / selectively download Kaggle kernel outputs.
usage: kout.py list <owner/slug>           -> prints file name and size (HEAD request)
       kout.py get <owner/slug> <dest> <regex>  -> downloads matching files only"""
import sys, os, re, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = KaggleApi(); api.authenticate()
mode, kernel = sys.argv[1], sys.argv[2]
owner, slug = kernel.split('/')
files, token = [], None
with api.build_kaggle_client() as kc:
    while True:
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug
        if token: req.page_token = token
        resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
        files += list(resp.files); token = resp.next_page_token
        if not token: break
if mode == 'list':
    for f in files:
        try: sz = int(requests.head(f.url, allow_redirects=True, timeout=20).headers.get('Content-Length', -1))
        except Exception: sz = -1
        print(f'{sz:>12d}  {f.file_name}')
    print(f'{len(files)} files')
else:
    dest, pat = sys.argv[3], re.compile(sys.argv[4])
    for f in files:
        if not pat.search(f.file_name): continue
        out = os.path.join(dest, f.file_name); os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        r = requests.get(f.url, timeout=600); r.raise_for_status(); open(out, 'wb').write(r.content)
        print('got', f.file_name, len(r.content))
