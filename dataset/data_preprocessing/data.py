import pandas as pd
import gzip
import json

file_path = '/Users/pengyuwen/Desktop/coding/goodreads_interactions_romance.json.gz'
romance_book_ids = set()
with gzip.open('/Users/pengyuwen/Desktop/coding/goodreads_books_romance.json.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        romance_book_ids.add(r['book_id'])
rows = []
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('rating', 0) == 0:
            continue
        if r['book_id'] not in romance_book_ids:
            continue
        rows.append((r['user_id'], r['book_id'], r['rating']))

df_inter = pd.DataFrame(rows, columns=['user_id', 'book_id', 'rating'])
print(len(df_inter))
df_inter.to_parquet('inter_romance_filtered.parquet')
print(df_inter.shape)