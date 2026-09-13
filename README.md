# Neural Book Recommendation: Interactions vs. Text

Comparing two neural recommenders for books on the Goodreads (romance) data:

- **Interaction based** — matrix factorization (learned user/item embeddings, MSE).
- **Text based** — a network on top of Sentence-BERT description embeddings,
  trained with a logistic ranking objective and negative sampling.

Both are ranked with the same protocol, with a focus on how they behave as a
function of item popularity (the **cold-start** regime).

## Layout

```
download_dataset.py                     download the raw Goodreads romance files
dataset/data_preprocessing/data.py      filter + align books/interactions -> parquet
dataset/data_preprocessing/sample_split.py   sample users, split train/dev/test
train_cf.py                             matrix-factorization model + evaluation
train_content.py                        text model (frozen SBERT + head) + evaluation
evaluation.py                           model-agnostic ranking metrics + popularity buckets
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU. On MPS, export
`PYTORCH_ENABLE_MPS_FALLBACK=1` before running.

## Data

The raw files are large (the romance interactions expand to ~15 GB) and are not
in the repo. Download them (from the UCSD Goodreads dataset, Wan & McAuley):

```
python download_dataset.py
```

By default they go to `~/Desktop/coding`; set `GOODREADS_RAW` to change that:

```
GOODREADS_RAW=/path/to/raw python download_dataset.py
```

The preprocessing scripts read the same `GOODREADS_RAW` directory.

## Run

```
python dataset/data_preprocessing/data.py          # -> books_filtered / interactions_filtered
python dataset/data_preprocessing/sample_split.py   # -> train / dev / test / books_sample
python train_cf.py                                  # trains CF, prints metrics
python train_content.py                             # trains text model, prints metrics
```

`train_content.py` encodes every book once with MiniLM and caches the vectors to
`dataset/data_preprocessing/books_emb.npy`; later runs reuse the cache.

Each `train_*.py` trains in memory and prints overall Precision/Recall/NDCG@K
plus recall broken down by how many training ratings each book has.

## Reusing the evaluation

`evaluation.py` is model agnostic: it only needs a scoring function
`score_all(user_idx) -> [n_users_in_batch, n_items]`. Any recommender that can
score users against all items can be evaluated and bucketed by item popularity
with the same code — see how `train_cf.py` and `train_content.py` call
`evaluate(...)`.
