import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation import evaluate, print_report

DATA = Path(__file__).resolve().parent / 'dataset' / 'data_preprocessing'
ENCODER = 'all-MiniLM-L6-v2'
MAX_LEN = 128
SEED = int(os.environ.get('SEED', 42))
D = 64
FT_USERS = int(os.environ.get('FT_USERS', 3000))  # 0 = all eligible users
PROF_CAP = 6          # profile books sampled per positive
N_NEG = 2
BATCH = 64
EPOCHS = int(os.environ.get('EPOCHS', 2))
LR_ENC = 2e-5         # small lr for the unfrozen transformer layers
LR_HEAD = 1e-3        # larger lr for the projection head
UNFREEZE = int(os.environ.get('UNFREEZE', 2))   # top encoder layers to fine-tune
FULL_FT = os.environ.get('FULL_FT', '0') == '1'  # unfreeze the whole encoder + embeddings
USE_META = os.environ.get('USE_META', '0') == '1'  # fuse author + year into the book vector
A_DIM = 32            # author embedding dim
DEBUG = os.environ.get('DEBUG', '0') == '1'  # granular per-op timing for first steps
CKPT_DIR = Path(__file__).resolve().parent / 'checkpoints'
EVAL_ONLY = os.environ.get('EVAL_ONLY', '0') == '1'  # skip training; score saved checkpoints
PATIENCE = int(os.environ.get('PATIENCE', 0))  # stop after this many epochs without a dev gain; 0 = off
RUN_TAG = os.environ.get('RUN_TAG', '')  # suffix for checkpoint names, keeps separate runs apart
EVAL_EPOCH = int(os.environ.get('EVAL_EPOCH', 0))  # with EVAL_ONLY, score just this epoch on test

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(SEED)
random.seed(SEED)


class ContentFT(nn.Module):
    def __init__(self, tf, d, author_idx=None, year_feat=None, n_authors=0):
        super().__init__()
        self.tf = tf
        if FULL_FT:
            for p in self.tf.parameters():
                p.requires_grad = True
        else:
            for p in self.tf.parameters():
                p.requires_grad = False
            for p in self.tf.encoder.layer[-UNFREEZE:].parameters():
                p.requires_grad = True
        h = self.tf.config.hidden_size
        in_dim = h
        if USE_META:
            self.author_emb = nn.Embedding(n_authors, A_DIM)   # learned, index 0 = unknown
            self.register_buffer('author_idx', author_idx)     # long [n_items]
            self.register_buffer('year_feat', year_feat)       # float [n_items, 2]
            in_dim = h + A_DIM + 2
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, d))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, ids, mask, book_idx):
        out = self.tf(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)   # mean pool
        pooled = F.normalize(pooled, dim=1)                    # match Normalize
        if USE_META:
            a = self.author_emb(self.author_idx[book_idx])
            y = self.year_feat[book_idx]
            pooled = torch.cat([pooled, a, y], dim=1)
        return self.mlp(pooled)


def main():
    train = pd.read_parquet(DATA / 'train.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    dev = pd.read_parquet(DATA / 'dev.parquet')
    books = pd.read_parquet(DATA / 'books_sample.parquet')

    b2i = {b: n for n, b in enumerate(books['book_id'].to_numpy())}
    uids = train['user_id'].unique()
    u2i = {u: n for n, u in enumerate(uids)}
    n_users, n_items = len(u2i), len(b2i)

    tr_u = train['user_id'].map(u2i).to_numpy()
    tr_i = train['book_id'].map(b2i).to_numpy()
    tr_r = train['rating'].to_numpy(dtype=np.float32)

    liked, allb = {}, {}
    for u, i, r in zip(tr_u, tr_i, tr_r):
        allb.setdefault(int(u), []).append(int(i))
        if r >= 4:
            liked.setdefault(int(u), []).append(int(i))

    # tokenize every description once (cheap, one-off), keep ids/mask on CPU
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(ENCODER, device=device)
    tok = st.tokenizer
    enc = tok(list(books['description'].to_numpy()), padding='max_length',
              truncation=True, max_length=MAX_LEN, return_tensors='pt')
    ids_all, mask_all = enc['input_ids'], enc['attention_mask']

    # per-book metadata features (author index + normalized year with a missing flag)
    author_idx = year_feat = None
    n_authors = 0
    if USE_META:
        a2i, aidx = {}, np.zeros(n_items, dtype=np.int64)  # 0 = unknown author
        for k, a in enumerate(books['author_id'].to_numpy()):
            if a is None or (isinstance(a, float) and np.isnan(a)):
                continue
            aidx[k] = a2i.setdefault(a, len(a2i) + 1)
        n_authors = len(a2i) + 1
        yr = books['publication_year'].to_numpy(dtype=float)
        miss = np.isnan(yr)
        mean, std = np.nanmean(yr), np.nanstd(yr)
        ynorm = np.where(miss, 0.0, (yr - mean) / (std + 1e-9))
        author_idx = torch.tensor(aidx)
        year_feat = torch.tensor(np.stack([ynorm, miss.astype(np.float32)], 1),
                                 dtype=torch.float32)
        print(f'meta on: {n_authors} authors, {int(miss.sum())} books missing year')

    model = ContentFT(st[0].auto_model, D, author_idx, year_feat, n_authors).to(device)
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith('tf.')]
    enc_p = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith('tf.')]
    opt = torch.optim.Adam([
        {'params': enc_p, 'lr': LR_ENC},
        {'params': head, 'lr': LR_HEAD},
    ])
    bce = nn.BCEWithLogitsLoss()

    def encode_idx(idx):  # idx: 1D long tensor of book indices -> [len, d]
        return model.encode(ids_all[idx.cpu()].to(device),
                            mask_all[idx.cpu()].to(device), idx.to(device))

    # training positives from a user subsample (speed)
    ft_users = [u for u in liked if len(liked[u]) >= 2]
    random.shuffle(ft_users)
    if FT_USERS:
        ft_users = ft_users[:FT_USERS]
    pos = [(u, i) for u in ft_users for i in liked[u]]
    print(f'seed={SEED}  ft users={len(ft_users)}  positives={len(pos)}  '
          f'steps/epoch={len(pos)//BATCH}')

    # eval structures (built once): full-profile aggregation for ALL users, masks, relevance sets
    prof_u, prof_b = [], []
    for u in range(n_users):
        bs = liked.get(u) or allb[u]
        prof_u += [u] * len(bs)
        prof_b += bs
    prof_u = torch.tensor(prof_u, device=device)
    prof_b = torch.tensor(prof_b, device=device)
    prof_cnt = torch.zeros(n_users, device=device).index_add_(
        0, prof_u, torch.ones(len(prof_u), device=device))
    item_freq = np.bincount(tr_i, minlength=n_items)
    train_items = {u: torch.tensor(v, device=device) for u, v in allb.items()}

    def rel_set(df):
        rel = {}
        for u, b, r in zip(df['user_id'], df['book_id'], df['rating']):
            if r >= 4 and u in u2i and b in b2i:
                rel.setdefault(u2i[u], set()).add(b2i[b])
        return rel
    rel_dev, rel_test = rel_set(dev), rel_set(test)

    def run_eval(rel):
        # encode every book with the current encoder, aggregate user profiles, rank
        model.eval()
        with torch.no_grad():
            vc = torch.zeros(n_items, D, device=device)
            for s in range(0, n_items, 256):
                j = torch.arange(s, min(s + 256, n_items))
                vc[s:s + len(j)] = encode_idx(j)
            uagg = torch.zeros(n_users, D, device=device).index_add_(0, prof_u, vc[prof_b])
            uagg = uagg / prof_cnt.unsqueeze(1)

        @torch.no_grad()
        def score_all(users):
            return uagg[users] @ vc.t()
        out = evaluate(score_all, train_items, rel, item_freq, device=device)
        model.train()
        model.tf.eval()
        return out

    tag = ('meta' if USE_META else 'desc') + RUN_TAG
    CKPT_DIR.mkdir(exist_ok=True)
    best_ndcg, best_ep = -1.0, 0

    if EVAL_ONLY and EVAL_EPOCH:
        ck = CKPT_DIR / f'content_ft_{tag}_s{SEED}_ep{EVAL_EPOCH}.pt'
        model.load_state_dict(torch.load(ck, map_location=device))
        print(f'scoring {ck.name} on test')
        print_report(run_eval(rel_test))
        return

    if EVAL_ONLY:
        # recover results from saved checkpoints: dev per epoch, then test on the dev-best
        for ep in range(EPOCHS):
            ck = CKPT_DIR / f'content_ft_{tag}_s{SEED}_ep{ep + 1}.pt'
            model.load_state_dict(torch.load(ck, map_location=device))
            d = run_eval(rel_dev)
            nd = d['overall'][10]['NDCG']
            print(f'epoch {ep + 1}  DEV NDCG@10={nd:.4f}  R@20={d["overall"][20]["R"]:.4f}', flush=True)
            if nd > best_ndcg:
                best_ndcg, best_ep = nd, ep + 1
        print(f'best epoch by dev NDCG@10: {best_ep} ({best_ndcg:.4f})')
        model.load_state_dict(torch.load(
            CKPT_DIR / f'content_ft_{tag}_s{SEED}_ep{best_ep}.pt', map_location=device))
        print('TEST:')
        print_report(run_eval(rel_test))
        return

    model.train()
    model.tf.eval()   # MPS SDPA has no dropout; grads still flow through unfrozen layers
    for ep in range(EPOCHS):
        random.shuffle(pos)
        tot, nb = 0.0, 0
        for s in range(0, len(pos) - BATCH + 1, BATCH):
            batch = pos[s:s + BATCH]
            pu = [u for u, _ in batch]
            pi = [i for _, i in batch]
            # per-positive profile = up to PROF_CAP liked books excluding the pos
            prof_rows, prof_books = [], []
            for r, (u, i) in enumerate(batch):
                cand = [x for x in liked[u] if x != i]
                pick = random.sample(cand, min(PROF_CAP, len(cand)))
                prof_rows += [r] * len(pick)
                prof_books += pick
            negs = [random.randrange(n_items) for _ in range(BATCH * N_NEG)]

            union = list(set(pi) | set(prof_books) | set(negs))
            if DEBUG:
                import time as _t; _t0 = _t.time()
                print(f'[dbg] step{nb} union={len(union)} start encode', flush=True)
            uidx = torch.tensor(union, device=device)
            vu = encode_idx(uidx)                                  # [U, d]
            if DEBUG:
                torch.mps.synchronize(); print(f'[dbg] encode done {_t.time()-_t0:.2f}s', flush=True)
            lookup = torch.full((n_items,), -1, dtype=torch.long, device=device)
            lookup[uidx] = torch.arange(len(union), device=device)

            vpi = vu[lookup[torch.tensor(pi, device=device)]]      # [B, d]
            vneg = vu[lookup[torch.tensor(negs, device=device)]].view(BATCH, N_NEG, D)
            pr = torch.tensor(prof_rows, device=device)
            pb = vu[lookup[torch.tensor(prof_books, device=device)]]
            uagg = torch.zeros(BATCH, D, device=device).index_add_(0, pr, pb)
            cnt = torch.zeros(BATCH, device=device).index_add_(
                0, pr, torch.ones(len(prof_rows), device=device))
            uagg = uagg / cnt.clamp(min=1).unsqueeze(1)

            pos_s = (uagg * vpi).sum(1)
            neg_s = (uagg.unsqueeze(1) * vneg).sum(2)
            logits = torch.cat([pos_s.unsqueeze(1), neg_s], 1) * model.scale + model.bias
            labels = torch.zeros_like(logits)
            labels[:, 0] = 1.0
            loss = bce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if DEBUG:
                torch.mps.synchronize(); print(f'[dbg] step{nb} full done {_t.time()-_t0:.2f}s loss={loss.item():.4f}', flush=True)
                if nb >= 3:
                    print('[dbg] first 3 steps OK, exiting', flush=True); return
            tot += loss.item()
            nb += 1
            if nb % 20 == 0:
                print(f'  ep{ep + 1} step {nb}  bce={tot / nb:.4f}')
        print(f'epoch {ep + 1}  bce={tot / nb:.4f}')
        dev_out = run_eval(rel_dev)
        dev_ndcg = dev_out['overall'][10]['NDCG']
        print(f'epoch {ep + 1}  DEV NDCG@10={dev_ndcg:.4f}  R@20={dev_out["overall"][20]["R"]:.4f}', flush=True)
        ck = CKPT_DIR / f'content_ft_{tag}_s{SEED}_ep{ep + 1}.pt'
        torch.save(model.state_dict(), ck)
        if dev_ndcg > best_ndcg:
            best_ndcg, best_ep = dev_ndcg, ep + 1
        elif PATIENCE and ep + 1 - best_ep >= PATIENCE:
            print(f'early stop: no dev gain for {PATIENCE} epochs, best epoch {best_ep}', flush=True)
            break

    # final: reload the epoch that was best on dev, report on test
    print(f'best epoch by dev NDCG@10: {best_ep} ({best_ndcg:.4f})')
    model.load_state_dict(torch.load(CKPT_DIR / f'content_ft_{tag}_s{SEED}_ep{best_ep}.pt', map_location=device))
    print('TEST:')
    print_report(run_eval(rel_test))


if __name__ == '__main__':
    main()
