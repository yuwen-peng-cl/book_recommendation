import math

import numpy as np
import torch


def freq_bucket(f):
    if f == 0:
        return '0'
    if f <= 2:
        return '1-2'
    if f <= 5:
        return '3-5'
    if f <= 10:
        return '6-10'
    return '11+'


BUCKETS = ['0', '1-2', '3-5', '6-10', '11+']


def evaluate(score_all, train_items, rel_items, item_freq,
             ks=(5, 10, 20), user_batch=512, device='cpu'):
    """Rank-based evaluation, model agnostic.

    score_all(user_idx_tensor) -> tensor [B, n_items] of scores.
    train_items[u] : 1D long tensor of items to mask (already interacted).
    rel_items[u]   : set of relevant item indices (rating >= 4 in test).
    item_freq      : 1D np array, train frequency per item, for bucketing.
    """
    kmax = max(ks)
    n_items = len(item_freq)
    covered = {k: set() for k in ks}
    users = [u for u in rel_items if len(rel_items[u]) > 0]
    bucket_of = np.array([freq_bucket(f) for f in item_freq])

    prec = {k: 0.0 for k in ks}
    rec = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    # per bucket recall: hits / total, per k
    b_hit = {k: {b: 0 for b in BUCKETS} for k in ks}
    b_tot = {k: {b: 0 for b in BUCKETS} for k in ks}
    n = 0

    idcg_cache = [0.0] * (kmax + 1)
    for m in range(1, kmax + 1):
        idcg_cache[m] = idcg_cache[m - 1] + 1.0 / math.log2(m + 1)

    for start in range(0, len(users), user_batch):
        batch = users[start:start + user_batch]
        scores = score_all(torch.tensor(batch, device=device))
        for r, u in enumerate(batch):
            ti = train_items.get(u)
            if ti is not None and len(ti) > 0:
                scores[r, ti] = -1e30
        top = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()

        for r, u in enumerate(batch):
            rel = rel_items[u]
            ranked = top[r]
            n += 1
            for k in ks:
                topk = ranked[:k]
                hit = [it for it in topk if it in rel]
                nhit = len(hit)
                prec[k] += nhit / k
                rec[k] += nhit / len(rel)
                dcg = sum(1.0 / math.log2(rank + 2)
                          for rank, it in enumerate(topk) if it in rel)
                ndcg[k] += dcg / idcg_cache[min(len(rel), k)]
                # per bucket
                topk_set = set(topk.tolist())
                covered[k].update(topk_set)
                for it in rel:
                    b = bucket_of[it]
                    b_tot[k][b] += 1
                    if it in topk_set:
                        b_hit[k][b] += 1

    out = {'overall': {}, 'bucket_recall': {}, 'coverage': {}}
    for k in ks:
        out['overall'][k] = {
            'P': prec[k] / n, 'R': rec[k] / n, 'NDCG': ndcg[k] / n,
        }
        out['coverage'][k] = len(covered[k]) / n_items
        out['bucket_recall'][k] = {
            b: (b_hit[k][b] / b_tot[k][b] if b_tot[k][b] else float('nan'))
            for b in BUCKETS
        }
    out['n_users'] = n
    return out


def print_report(out):
    print(f"users evaluated: {out['n_users']}")
    print('overall:')
    for k, m in out['overall'].items():
        print(f"  @{k:<2}  P={m['P']:.4f}  R={m['R']:.4f}  NDCG={m['NDCG']:.4f}")
    print('catalog coverage:')
    for k in out['coverage']:
        print(f"  @{k:<2}  {out['coverage'][k]:.4f}")
    print('recall by book train-frequency bucket:')
    header = '  k   ' + '  '.join(f'{b:>6}' for b in BUCKETS)
    print(header)
    for k, br in out['bucket_recall'].items():
        row = f'  @{k:<2} ' + '  '.join(f'{br[b]:6.3f}' for b in BUCKETS)
        print(row)
