import os
from pathlib import Path

import requests

# directory the raw .gz files are written to (override with GOODREADS_RAW)
RAW_DIR = Path(os.environ.get('GOODREADS_RAW', Path.home() / 'Desktop' / 'coding'))
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/'
FILES = ['goodreads_books_romance.json.gz', 'goodreads_interactions_romance.json.gz']

RAW_DIR.mkdir(parents=True, exist_ok=True)
for name in FILES:
    out = RAW_DIR / name
    with requests.get(BASE + name, stream=True) as r:
        r.raise_for_status()
        with open(out, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print('downloaded', out)
print('Dataset has been downloaded!')
