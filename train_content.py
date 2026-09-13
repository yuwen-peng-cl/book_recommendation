import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import evaluate, print_report

DATA = Path(__file__).resolve().parent / 'dataset' / 'data_preprocessing'
CACHE = DATA / 'books_emb.npy'
ENCODER = 'all-MiniLM-L6-v2'
MAX_LEN = 128
SEED = 42
D = 64            # projection dim
EPOCHS = int(os.environ.get('EPOCHS', 20))
BATCH = 1024
N_NEG = 4         # sampled unobserved negatives per positive
LR = 5e-3
WD = 1e-5
SUBSAMPLE_USERS = int(os.environ.get('SUBSAMPLE_USERS', 0))  # 0 = all users

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(SEED)


def encode_books(descs):
    if CACHE.exists():
        emb = np.load(CACHE)
        if len(emb) == len(descs):
            return emb
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER, device=device)
    enc.max_seq_length = MAX_LEN
    emb = enc.encode(list(descs), batch_size=256, show_progress_bar=True,
                     convert_to_numpy=True)
    np.save(CACHE, emb)
    return emb


class Content(nn.Module):
    def __init__(self, emb, d):
        super().__init__()
        self.emb = nn.Parameter(torch.tensor(emb), requires_grad=False)  # frozen
        e = emb.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(e, 128), nn.ReLU(), nn.Linear(128, d),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def item_vecs(self):
        return self.mlp(self.emb)                       # [n_items, d]


def main():
    train = pd.read_parquet(DATA / 'train.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    books = pd.read_parquet(DATA / 'books_sample.parquet')

    emb = encode_books(books['description'].to_numpy())

    uids = train['user_id'].unique()
    u2i = {u: n for n, u in enumerate(uids)}
    b2i = {b: n for n, b in enumerate(books['book_id'].to_numpy())}
    n_users, n_items = len(u2i), len(b2i)

    tr_u = train['user_id'].map(u2i).to_numpy()
    tr_i = train['book_id'].map(b2i).to_numpy()
    tr_r = train['rating'].to_numpy(dtype=np.float32)

    # profile = liked (rating>=4) train books; fall back to all train books
    liked, allb = {}, {}
    for u, i, r in zip(tr_u, tr_i, tr_r):
        allb.setdefault(int(u), []).append(int(i))
        if r >= 4:
            liked.setdefault(int(u), []).append(int(i))
    prof_u, prof_b = [], []
    for u in range(n_users):
        bs = liked.get(u) or allb[u]
        prof_u += [u] * len(bs)
        prof_b += bs
    prof_u = torch.tensor(prof_u, device=device)
    prof_b = torch.tensor(prof_b, device=device)
    count = torch.zeros(n_users, device=device).index_add_(
        0, prof_u, torch.ones_like(prof_u, dtype=torch.float))

    # training positives: liked pairs whose user has >=2 liked (LOO valid)
    train_users = [u for u in liked if len(liked[u]) >= 2]
    if SUBSAMPLE_USERS:  # match a fine-tuning run's user subset for a fair comparison
        random.seed(SEED)
        random.shuffle(train_users)
        train_users = train_users[:SUBSAMPLE_USERS]
    pos_u, pos_b = [], []
    for u in train_users:
        pos_u += [u] * len(liked[u])
        pos_b += liked[u]
    pos_u = torch.tensor(pos_u, device=device)
    pos_b = torch.tensor(pos_b, device=device)

    model = Content(emb, D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss()

    npos = len(pos_u)
    g = torch.Generator(device=device).manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(npos, device=device, generator=g)
        total = 0.0
        for s in range(0, npos, BATCH):
            idx = perm[s:s + BATCH]
            pu, pi = pos_u[idx], pos_b[idx]
            b = len(idx)

            vc = model.item_vecs()                       # [I, d]
            S = torch.zeros(n_users, D, device=device).index_add_(
                0, prof_u, vc[prof_b])
            uagg = S / count.unsqueeze(1)                # full profile

            # leave-one-out profile for the positive
            loo = (S[pu] - vc[pi]) / (count[pu] - 1).unsqueeze(1)
            pos_s = (loo * vc[pi]).sum(1)                # [b]
            nj = torch.randint(0, n_items, (b, N_NEG), device=device, generator=g)
            neg_s = (uagg[pu].unsqueeze(1) * vc[nj]).sum(2)   # [b, N_NEG]

            logits = torch.cat([pos_s.unsqueeze(1), neg_s], 1) * model.scale + model.bias
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1.0
            loss = bce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b
        print(f'epoch {ep + 1:2d}  bce={total / npos:.4f}')

    # eval
    item_freq = np.bincount(tr_i, minlength=n_items)
    train_items = {u: torch.tensor(v, device=device) for u, v in allb.items()}
    rel = {}
    for u, b, r in zip(test['user_id'], test['book_id'], test['rating']):
        if r >= 4 and u in u2i and b in b2i:
            rel.setdefault(u2i[u], set()).add(b2i[b])

    model.eval()
    with torch.no_grad():
        vc = model.item_vecs()
        S = torch.zeros(n_users, D, device=device).index_add_(0, prof_u, vc[prof_b])
        uagg = S / count.unsqueeze(1)

    @torch.no_grad()
    def score_all(users):
        return uagg[users] @ vc.t()

    out = evaluate(score_all, train_items, rel, item_freq, device=device)
    print_report(out)


if __name__ == '__main__':
    main()
