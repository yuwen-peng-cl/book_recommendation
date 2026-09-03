import numpy as np
import pandas as pd

SEED = 42
MIN_RATINGS = 5      # drop users with too little history (CF needs some)
MAX_RATINGS = 200    # drop heavy users so they don't dominate
N_USERS = 10000
DEV_FRAC = 0.15
TEST_FRAC = 0.15

df_inter = pd.read_parquet('interactions_filtered.parquet')
df_books = pd.read_parquet('books_filtered.parquet')

# 1) filter users by activity, then sample N_USERS of them
counts = df_inter.groupby('user_id').size()
eligible = counts[(counts >= MIN_RATINGS) & (counts <= MAX_RATINGS)].index.to_numpy()
rng = np.random.default_rng(SEED)
sampled = rng.choice(eligible, size=min(N_USERS, len(eligible)), replace=False)
df = df_inter[df_inter['user_id'].isin(sampled)].copy()

# 2) per-user train/dev/test split (each user gets at least 1 dev and 1 test)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
rank = df.groupby('user_id').cumcount()
size = df.groupby('user_id')['user_id'].transform('size')
n_test = np.maximum(1, np.round(size * TEST_FRAC)).astype(int)
n_dev = np.maximum(1, np.round(size * DEV_FRAC)).astype(int)
n_train = size - n_dev - n_test
df['split'] = np.where(rank < n_train, 'train',
              np.where(rank < n_train + n_dev, 'dev', 'test'))

# 3) keep only books actually touched by the sampled users (profiles stay full)
kept = set(df['book_id'].unique())
df_books = df_books[df_books['book_id'].isin(kept)].reset_index(drop=True)

# 4) write
cols = ['user_id', 'book_id', 'rating']
for name in ('train', 'dev', 'test'):
    df[df['split'] == name][cols].to_parquet(f'{name}.parquet')
df_books.to_parquet('books_sample.parquet')

print('users:', df['user_id'].nunique(), 'books:', df_books.shape[0])
print(df['split'].value_counts().to_dict())
