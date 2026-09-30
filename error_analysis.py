"""Where each recommender fails, and a few side by side lists."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parent / 'data'
CKPT = Path(__file__).resolve().parent / 'checkpoints'
SEED = int(os.environ.get('SEED', 42))
K = 10
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


def topk_for(score_all, users, train_items, n_items, k):
    out = {}
    for s in range(0, len(users), 512):
        batch = users[s:s + 512]
        sc = score_all(torch.tensor(batch, device=device))
        for r, u in enumerate(batch):
            ti = train_items.get(u)
            if ti is not None and len(ti):
                sc[r, ti] = -1e30
        idx = torch.topk(sc, k, dim=1).indices.cpu().numpy()
        for r, u in enumerate(batch):
            out[u] = idx[r]
    return out


def main():
    train = pd.read_parquet(DATA / 'train.parquet')
    test = pd.read_parquet(DATA / 'test.parquet')
    books = pd.read_parquet(DATA / 'books_sample.parquet')
    titles = pd.read_parquet(DATA / 'titles.parquet').set_index('book_id')['title']

    uids = train['user_id'].unique()
    u2i = {u: n for n, u in enumerate(uids)}
    b2i = {b: n for n, b in enumerate(books['book_id'].to_numpy())}
    i2b = books['book_id'].to_numpy()
    n_users, n_items = len(u2i), len(b2i)
    title_of = np.array([titles.get(b, '?') for b in i2b])
    desc_len = books['description'].str.split().str.len().to_numpy()

    tr_u = train['user_id'].map(u2i).to_numpy()
    tr_i = train['book_id'].map(b2i).to_numpy()
    tr_r = train['rating'].to_numpy(dtype=np.float32)
    item_freq = np.bincount(tr_i, minlength=n_items)

    liked, allb = {}, {}
    for u, i, r in zip(tr_u, tr_i, tr_r):
        allb.setdefault(int(u), []).append(int(i))
        if r >= 4:
            liked.setdefault(int(u), []).append(int(i))
    train_items = {u: torch.tensor(v, device=device) for u, v in allb.items()}

    rel = {}
    for u, b, r in zip(test['user_id'], test['book_id'], test['rating']):
        if r >= 4 and u in u2i and b in b2i:
            rel.setdefault(u2i[u], set()).add(b2i[b])
    users = sorted(rel)

    models = {}

    # matrix factorization, retrained here with the L2 value picked on dev
    from models.train_cf import MF, train_once
    dv = pd.read_parquet(DATA / 'dev.parquet')
    dv = dv[dv['user_id'].isin(u2i) & dv['book_id'].isin(b2i)]
    tu = torch.tensor(tr_u, device=device)
    ti = torch.tensor(tr_i, device=device)
    trr = torch.tensor(tr_r, device=device)
    du = torch.tensor(dv['user_id'].map(u2i).to_numpy(), device=device)
    di = torch.tensor(dv['book_id'].map(b2i).to_numpy(), device=device)
    dr = torch.tensor(dv['rating'].to_numpy(dtype=np.float32), device=device)
    cf = train_once(n_users, n_items, float(tr_r.mean()), tu, ti, trr,
                    du, di, dr, 64, 1e-2, verbose=False)
    cf.eval()
    models['CF'] = cf.score_all

    # content models, from the dev-selected checkpoints
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    tok = st.tokenizer
    enc = tok(list(books['description'].to_numpy()), padding='max_length',
              truncation=True, max_length=128, return_tensors='pt')
    ids_all, mask_all = enc['input_ids'], enc['attention_mask']

    a2i, aidx = {}, np.zeros(n_items, dtype=np.int64)
    for k, a in enumerate(books['author_id'].to_numpy()):
        if a is not None and not (isinstance(a, float) and np.isnan(a)):
            aidx[k] = a2i.setdefault(a, len(a2i) + 1)
    yr = books['publication_year'].to_numpy(dtype=float)
    miss = np.isnan(yr)
    ynorm = np.where(miss, 0.0, (yr - np.nanmean(yr)) / (np.nanstd(yr) + 1e-9))
    author_idx = torch.tensor(aidx, device=device)
    year_feat = torch.tensor(np.stack([ynorm, miss.astype(np.float32)], 1),
                             dtype=torch.float32, device=device)

    prof_u, prof_b = [], []
    for u in range(n_users):
        bs = liked.get(u) or allb[u]
        prof_u += [u] * len(bs)
        prof_b += bs
    prof_u = torch.tensor(prof_u, device=device)
    prof_b = torch.tensor(prof_b, device=device)
    prof_cnt = torch.zeros(n_users, device=device).index_add_(
        0, prof_u, torch.ones(len(prof_u), device=device))

    import torch.nn as nn
    import torch.nn.functional as F

    def load_content(use_meta, ep):
        tag = ('meta' if use_meta else 'desc') + '_es'
        sd = torch.load(CKPT / f'content_ft_{tag}_s{SEED}_ep{ep}.pt', map_location=device)
        tf = SentenceTransformer('all-MiniLM-L6-v2', device=device)[0].auto_model
        h = tf.config.hidden_size
        in_dim = h + 34 if use_meta else h
        mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)
        emb = nn.Embedding(len(a2i) + 1, 32).to(device) if use_meta else None
        tf.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith('tf.')})
        mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith('mlp.')})
        if use_meta:
            emb.load_state_dict({'weight': sd['author_emb.weight']})
        tf.eval(); mlp.eval()
        with torch.no_grad():
            vc = torch.zeros(n_items, 64, device=device)
            for s in range(0, n_items, 256):
                j = torch.arange(s, min(s + 256, n_items))
                out = tf(input_ids=ids_all[j].to(device),
                         attention_mask=mask_all[j].to(device)).last_hidden_state
                m = mask_all[j].to(device).unsqueeze(-1).float()
                pooled = F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)
                if use_meta:
                    jd = j.to(device)
                    pooled = torch.cat([pooled, emb(author_idx[jd]), year_feat[jd]], 1)
                vc[j] = mlp(pooled)
            uagg = torch.zeros(n_users, 64, device=device).index_add_(0, prof_u, vc[prof_b])
            uagg = uagg / prof_cnt.unsqueeze(1)
        return lambda uu: uagg[uu] @ vc.t()

    models['desc-only'] = load_content(False, 5)
    models['desc+meta'] = load_content(True, 5)

    tops = {name: topk_for(fn, users, train_items, n_items, K)
            for name, fn in models.items()}

    # 1) content model recall by description length of the relevant book
    print(f'\n=== hit rate@{K} by description length of the wanted book ===')
    qs = np.quantile(desc_len, [.25, .5, .75])
    def lb(n): return '短(<%d词)' % qs[0] if n < qs[0] else ('中' if n < qs[2] else '长(>%d词)' % qs[2])
    print(f"{'model':12} " + ' '.join(f'{x:>14}' for x in ['短', '中', '长']))
    for name in models:
        hit = {'短': [0, 0], '中': [0, 0], '长': [0, 0]}
        for u in users:
            got = set(tops[name][u].tolist())
            for it in rel[u]:
                b = lb(desc_len[it])
                key ='短' if b.startswith('短') else ('长' if b.startswith('长') else '中')
                hit[key][1] += 1
                if it in got:
                    hit[key][0] += 1
        print(f'{name:12} ' + ' '.join(f'{hit[k][0]/max(hit[k][1],1):>14.4f}' for k in ['短', '中', '长']))

    # 2) hit rate by how many books the user liked in train
    print(f'\n=== hit rate@{K} by size of the user profile ===')
    pf = np.array([len(liked.get(u, allb[u])) for u in users])
    edges = [0, 4, 8, 17, 10 ** 9]
    names = ['<=4', '5-8', '9-17', '18+']
    print(f"{'model':12} " + ' '.join(f'{x:>10}' for x in names))
    for name in models:
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [u for u, p in zip(users, pf) if lo < p <= hi]
            h = sum(len(set(tops[name][u].tolist()) & rel[u]) for u in sel)
            t = sum(len(rel[u]) for u in sel)
            row.append(h / max(t, 1))
        print(f'{name:12} ' + ' '.join(f'{v:>10.4f}' for v in row))

    # 3) side by side lists
    print('\n=== side by side (users where the models disagree most) ===')
    def clean(u):
        # skip users whose lists repeat a title, the catalogue holds several
        # editions of the same book and a repeated title makes a poor example
        if len(set(title_of[i] for i in liked[u])) != len(liked[u]):
            return False
        return all(len(set(title_of[i] for i in tops[m][u][:5])) == 5 for m in models)

    cand = [u for u in users
            if 3 <= len(liked.get(u, [])) <= 8 and len(rel[u]) >= 2 and clean(u)]
    cand.sort(key=lambda u: -len(set(tops['desc+meta'][u].tolist()) - set(tops['CF'][u].tolist())))
    print(f'({len(cand)} users qualify as clean examples)')
    for u in cand[:2]:
        print('\n' + '-' * 70)
        print('likes in train:')
        for i in liked[u][:6]:
            print(f'   {title_of[i][:60]}')
        print('wants in test:')
        for i in sorted(rel[u]):
            print(f'   {title_of[i][:60]}   [{item_freq[i]} train ratings]')
        for name in models:
            print(f'{name} top5:')
            for i in tops[name][u][:5]:
                mark = ' <-- HIT' if i in rel[u] else ''
                print(f'   {title_of[i][:55]:<57}[{item_freq[i]:>4}]{mark}')


if __name__ == '__main__':
    main()
