import pandas as pd
import requests
import os

DIR = '/Users/pengyuwen/Desktop/nn'
url="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_romance.json.gz"
url1='https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_romance.json.gz'
outdir='/Users/pengyuwen/Desktop/coding'
output_path = os.path.join(outdir, 'goodreads_books_romance.json.gz')
output_path1=os.path.join(outdir, 'goodreads_interactions_romance.json.gz')
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
with requests.get(url1, stream=True) as r:
    r.raise_for_status()
    with open(output_path1, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print('Dataset has been downloaded!')