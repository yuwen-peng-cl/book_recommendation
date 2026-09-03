import gzip
import json
import pandas as pd

BOOKS_PATH = '/Users/pengyuwen/Desktop/coding/goodreads_books_romance.json.gz'
INTER_PATH = '/Users/pengyuwen/Desktop/coding/goodreads_interactions_romance.json.gz'

# 1) books: keep only those with a description, remember their ids
book_rows = []
books_with_desc = set()
with gzip.open(BOOKS_PATH, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        desc = (r.get('description') or '').strip()
        if not desc:
            continue
        bid = r['book_id']
        books_with_desc.add(bid)
        authors = r.get('authors') or []
        book_rows.append((
            bid,
            r.get('publication_year'),
            desc,
            authors[0]['author_id'] if authors else None,
            authors,
        ))

df_books = pd.DataFrame(book_rows, columns=[
    'book_id', 'publication_year', 'description', 'author_id', 'authors',
])
df_books['publication_year'] = pd.to_numeric(df_books['publication_year'], errors='coerce')

# 2) interactions: keep rated interactions whose book has a description
inter_rows = []
seen_book_ids = set()
with gzip.open(INTER_PATH, 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r.get('rating', 0) == 0:
            continue
        bid = r['book_id']
        if bid not in books_with_desc:
            continue
        inter_rows.append((r['user_id'], bid, r['rating']))
        seen_book_ids.add(bid)

df_inter = pd.DataFrame(inter_rows, columns=['user_id', 'book_id', 'rating'])
df_inter = df_inter.drop_duplicates(['user_id', 'book_id'], keep='last')

# 3) align back: keep only books that actually appear in the interactions
df_books = df_books[df_books['book_id'].isin(seen_book_ids)].reset_index(drop=True)

df_books.to_parquet('books_filtered.parquet')
df_inter.to_parquet('interactions_filtered.parquet')
print('books:', df_books.shape, 'interactions:', df_inter.shape)
