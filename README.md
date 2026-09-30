# Neural Book Recommendation: Interactions vs. Text

Comparing two neural recommenders for books on the Goodreads (romance) data:

- **Interaction based** — matrix factorization (learned user/item embeddings, MSE).
- **Text based** — Sentence-BERT (MiniLM) on the book descriptions, top two
  encoder layers fine-tuned with a logistic ranking objective and negative
  sampling. A second variant also gets the author and publication year.

Both are ranked with the same protocol, with a focus on how they behave as a
function of item popularity (the **cold-start** regime).

## Layout

```
download_dataset.py                     download the raw Goodreads romance files
data/data.py                            filter + align books/interactions -> parquet
data/sample_split.py                    sample users, split train/dev/test
models/train_cf.py                      matrix-factorization model + evaluation
models/train_content.py                 text model, frozen SBERT + head (dev scaffolding)
models/train_content_ft.py              text model, fine-tuned top layers, optional metadata
evaluation.py                           model-agnostic ranking metrics + popularity buckets
error_analysis/error_analysis.py        where the models fail + side-by-side top-5 lists
error_analysis/README.md                its output, see there for the side-by-side lists
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
python data/data.py                  # -> books_filtered / interactions_filtered
python data/sample_split.py          # -> train / dev / test / books_sample
python -m models.train_cf            # trains CF, prints metrics
python -m models.train_content       # trains text model, prints metrics
```

`models/train_content.py` encodes every book once with MiniLM and caches the vectors to
`data/books_emb.npy`; later runs reuse the cache. It is only
used to get the pipeline working, the reported text models come from
`models/train_content_ft.py`:

```
FT_USERS=0 EPOCHS=10 PATIENCE=2 RUN_TAG=_es SEED=42 python -m models.train_content_ft               # description only
FT_USERS=0 EPOCHS=10 PATIENCE=2 RUN_TAG=_es SEED=42 USE_META=1 python -m models.train_content_ft    # + author, year
```

| variable | default | meaning |
|---|---|---|
| `SEED` | 42 | random seed, also part of the checkpoint name |
| `FT_USERS` | 3000 | users used for training, 0 = all |
| `EPOCHS` | 2 | maximum number of epochs |
| `PATIENCE` | 0 | stop after this many epochs without a dev gain, 0 = off |
| `UNFREEZE` | 2 | number of top encoder layers that are fine-tuned |
| `USE_META` | 0 | add author embedding, publication year and a missing-year flag |
| `RUN_TAG` | | suffix for the checkpoint names in `checkpoints/` |
| `EVAL_ONLY`, `EVAL_EPOCH` | 0 | skip training and score saved checkpoints on test |

Checkpoints are written to `checkpoints/` and are not part of this repository,
so the error analysis needs the models to be trained first. Run it with
`python -m error_analysis.error_analysis`; its output is kept in
`error_analysis/README.md`.

The best epoch is picked on dev (NDCG@10) and scored on test. `models/train_cf.py` also
reads `SEED`. The error analysis loads the saved checkpoints of the same `SEED`.

Each `train_*.py` trains in memory and prints overall Precision/Recall/NDCG@K
plus recall broken down by how many training ratings each book has.

## Results

Test set, mean ± std over seeds 42, 43, 44. A positive is a rating of 4 or more.

| | NDCG@20 | Recall@20 |
|---|---|---|
| CF | .064 ± .001 | .085 ± .001 |
| text, description | .068 ± .011 | .122 ± .007 |
| text, description + metadata | .101 ± .007 | .188 ± .009 |

Recall@20 by how many training ratings the book has (mean over the seeds):

| | 0 | 1-2 | 3-5 | 6-10 | 11+ |
|---|---|---|---|---|---|
| CF | .000 | .000 | .000 | .000 | .095 |
| text, description | .005 | .008 | .007 | .013 | .117 |
| text, description + metadata | .033 | .043 | .059 | .070 | .166 |

With the tuned L2 weight CF is close to a popularity ranking: it only ever
recommends the most rated books, so it gets nothing on books with few ratings.

## Reusing the evaluation

`evaluation.py` is model agnostic: it only needs a scoring function
`score_all(user_idx) -> [n_users_in_batch, n_items]`. Any recommender that can
score users against all items can be evaluated and bucketed by item popularity
with the same code — see how `models/train_cf.py` and `models/train_content.py` call
`evaluate(...)`.
