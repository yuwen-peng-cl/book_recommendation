from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation import evaluate, print_report

DATA = Path(__file__).resolve().parent / 'dataset' / 'data_preprocessing'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


def main():
    train = pd.read_parquet(DATA / 'train.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    books = pd.read_parquet(DATA / 'books_sample.parquet')

    uids = train['user_id'].unique()
    u2i = {u: n for n, u in enumerate(uids)}
    b2i = {b: n for n, b in enumerate(books['book_id'].to_numpy())}
    n_items = len(b2i)

    tr_u = train['user_id'].map(u2i).to_numpy()
    tr_i = train['book_id'].map(b2i).to_numpy()
    item_freq = np.bincount(tr_i, minlength=n_items)

    train_items = {}
    for u, i in zip(tr_u, tr_i):
        train_items.setdefault(int(u), []).append(int(i))
    train_items = {u: torch.tensor(v, device=device) for u, v in train_items.items()}

    rel = {}
    for u, b, r in zip(test['user_id'], test['book_id'], test['rating']):
        if r >= 4 and u in u2i and b in b2i:
            rel.setdefault(u2i[u], set()).add(b2i[b])

    # every user gets the same ranking: the books with the most training ratings
    pop = torch.tensor(item_freq, dtype=torch.float32, device=device)

    @torch.no_grad()
    def score_all(users):
        return pop.unsqueeze(0).repeat(len(users), 1)

    print('popularity baseline (no training, rank by number of training ratings)')
    print_report(evaluate(score_all, train_items, rel, item_freq, device=device))


if __name__ == '__main__':
    main()
