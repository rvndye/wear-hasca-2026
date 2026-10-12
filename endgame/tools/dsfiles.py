import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
ds = sys.argv[1]; tok = None; tot = 0
while True:
    r = api.dataset_list_files(ds, page_token=tok, page_size=200)
    for f in r.files: print(f'{f.total_bytes:>12d}  {f.name}'); tot += f.total_bytes
    tok = r.next_page_token
    if not tok: break
print('TOTAL', tot / 1e9, 'GB')
