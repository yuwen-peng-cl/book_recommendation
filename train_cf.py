import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import evaluate, print_report

DATA = Path(__file__).resolve().parent / 'dataset' / 'data_preprocessing'
SEED = int(os.environ.get('SEED', 42))
K = 64            # latent dim
EPOCHS = 20
BATCH = 1024
LR = 5e-3
WD_GRID = [1e-5, 1e-4, 1e-3, 1e-2]   # L2 strengths to try on dev

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(SEED)


class MF(nn.Module):
    def __init__(self, n_users, n_items, k, mu):
        super().__init__()
        self.U = nn.Embedding(n_users, k)
        self.V = nn.Embedding(n_items, k)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        self.mu = mu

    def forward(self, u, i):
        return (self.U(u) * self.V(i)).sum(1) + self.bu(u).squeeze(1) + \
            self.bi(i).squeeze(1) + self.mu

    @torch.no_grad()
    def score_all(self, u):
        return self.U(u) @ self.V.weight.t() + self.bi.weight.t()


def load():
    train = pd.read_parquet(DATA / 'train.parquet')
    dev = pd.read_parquet(DATA / 'dev.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    books = pd.read_parquet(DATA / 'books_sample.parquet')

    uids = train['user_id'].unique()
    u2i = {u: n for n, u in enumerate(uids)}
    b2i = {b: n for n, b in enumerate(books['book_id'].to_numpy())}

    def to_idx(df):
        m = df['user_id'].isin(u2i) & df['book_id'].isin(b2i)
        df = df[m]
        u = torch.tensor(df['user_id'].map(u2i).to_numpy(), device=device)
        i = torch.tensor(df['book_id'].map(b2i).to_numpy(), device=device)
        r = torch.tensor(df['rating'].to_numpy(dtype=np.float32), device=device)
        return u, i, r

    return train, dev, test, u2i, b2i, to_idx


def rel_set(df, u2i, b2i):
    rel = {}
    for u, b, r in zip(df['user_id'], df['book_id'], df['rating']):
        if r >= 4 and u in u2i and b in b2i:
            rel.setdefault(u2i[u], set()).add(b2i[b])
    return rel


def mse(model, u, i, r):
    with torch.no_grad():
        return ((model(u, i) - r) ** 2).mean().item()


def train_once(n_users, n_items, mu, tu, ti, tr, du, di, dr, k, wd, verbose):
    torch.manual_seed(SEED)
    model = MF(n_users, n_items, k, mu).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    loss_fn = nn.MSELoss()
    n = len(tu)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            loss = loss_fn(model(tu[idx], ti[idx]), tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose:
            print(f'  epoch {ep + 1:2d}  train_mse={tot / n:.4f}  '
                  f'dev_mse={mse(model, du, di, dr):.4f}')
    return model


def main():
    train, dev, test, u2i, b2i, to_idx = load()
    n_users, n_items = len(u2i), len(b2i)
    tu, ti, tr = to_idx(train)
    du, di, dr = to_idx(dev)
    mu = float(tr.float().mean())

    item_freq = np.bincount(ti.cpu().numpy(), minlength=n_items)
    train_items = {}
    tu_np, ti_np = tu.cpu().numpy(), ti.cpu().numpy()
    for u, i in zip(tu_np, ti_np):
        train_items.setdefault(int(u), []).append(int(i))
    train_items = {u: torch.tensor(v, device=device) for u, v in train_items.items()}
    ild_emb = None
    cache = DATA / 'books_emb.npy'
    if cache.exists():
        e = np.load(cache)
        if len(e) == n_items:
            ild_emb = torch.tensor(e)
    rel_dev = rel_set(dev, u2i, b2i)
    rel_test = rel_set(test, u2i, b2i)

    # sweep L2 on dev
    print(f'{"wd":>8} {"train_mse":>10} {"dev_mse":>8} {"dev_NDCG@10":>12}')
    results = {}
    for wd in WD_GRID:
        model = train_once(n_users, n_items, mu, tu, ti, tr, du, di, dr,
                           K, wd, verbose=False)
        tm = mse(model, tu, ti, tr)
        dm = mse(model, du, di, dr)
        out = evaluate(model.score_all, train_items, rel_dev, item_freq, device=device)
        ndcg = out['overall'][10]['NDCG']
        results[wd] = ndcg
        print(f'{wd:>8.0e} {tm:>10.4f} {dm:>8.4f} {ndcg:>12.4f}')

    best_wd = max(results, key=results.get)
    print(f'\nbest wd = {best_wd:.0e} (by dev NDCG@10)\n')

    # retrain best with per-epoch train/dev curve (see the gap), report test
    print(f'final model, wd={best_wd:.0e}:')
    model = train_once(n_users, n_items, mu, tu, ti, tr, du, di, dr,
                       K, best_wd, verbose=True)
    print('\nTEST:')
    out = evaluate(model.score_all, train_items, rel_test, item_freq, device=device,
                   item_emb=ild_emb)
    print_report(out)


if __name__ == '__main__':
    main()
