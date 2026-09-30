# Error analysis

Supplementary material. Produced by `error_analysis.py` (run as
`python -m error_analysis.error_analysis` from the repository root) with seed 42, using the
checkpoints selected on dev. It is not part of the report.

All three models are the ones reported in the paper: matrix factorization,
the fine-tuned Sentence-BERT model on descriptions only, and the same model
with author and publication year.

## Where each model fails

### By the length of the description of the wanted book

Hit rate@10 for the books a user liked in the test split, grouped by how long
the description of that book is (quartiles of the description length).

| model | short | medium | long |
|---|---|---|---|
| matrix factorization | 0.017 | 0.058 | 0.013 |
| Sentence-BERT | 0.035 | 0.061 | 0.029 |
| Sentence-BERT + metadata | 0.057 | 0.079 | 0.060 |

The text models were expected to struggle on books with short or generic
descriptions. They do not. On the shortest quarter of the descriptions they are
two to three times better than matrix factorization, which is itself weakest
there and on the longest descriptions.

### By how many books the user liked in training

Hit rate@10 grouped by the size of the user profile.

| model | <= 4 | 5-8 | 9-17 | 18+ |
|---|---|---|---|---|
| matrix factorization | 0.117 | 0.074 | 0.046 | 0.021 |
| Sentence-BERT | 0.143 | 0.090 | 0.065 | 0.025 |
| Sentence-BERT + metadata | 0.185 | 0.134 | 0.093 | 0.039 |

Every model does worse as the profile grows, since a longer history means a
broader taste and more books to guess. The ordering between the models is the
same in every group.

## Side by side

Two users, picked automatically as the ones where the metadata model and matrix
factorization disagree most. Users whose lists contain two editions of the same
title are skipped; 224 users qualify. The number after each title is how many
ratings that book has in the training data.

### User A

Liked in training:

    Rhapsodic (The Bargainer, #1)
    Darkness Awakened (Order of the Blade #1)
    Ice Planet Barbarians (Ice Planet Barbarians, #1)
    Barbarian Alien (Ice Planet Barbarians, #2)
    Storm Warrior (Grim, #1)
    Barbarian Lover (Ice Planet Barbarians, #3)

Wanted in test:

    Blood and Sin (The Infernari, #1)        [1]
    Tangled Beauty (Tangled, #1)             [3]
    Good Intentions (The Road to Hell, #1)   [0]

Top 5 of each model:

| matrix factorization | Sentence-BERT | + metadata |
|---|---|---|
| Pride and Prejudice [2180] | Huntsman's Prey [1] | Arouse [2] |
| Me Before You [902] | Ice Ice Babies (Ice Planet Barbarians #6.6) [3] | Fire in His Kiss [1] |
| Anna and the French Kiss [567] | Consorts [1] | Barbarian's Prize (Ice Planet Barbarians, #5) [6] |
| Lady Midnight [228] | Born of Ice [23] | Barbarian Mine (Ice Planet Barbarians, #4) [7] |
| Slammed [302] | Bittersweet Seraphim [0] | Barbarian's Choice (Ice Planet Barbarians, #11) [3] |

The user is reading through a series. The metadata model returns three further
volumes of that same series, all of them with single digit rating counts. The
description only model stays in the right corner of the collection but does not
find the series. Matrix factorization returns general bestsellers.

### User B

Liked in training:

    The Nymph's Muse
    Satin & Oak
    A Man for the Season
    An Introduction to Sappho
    A Hero's Return
    Tragedy of the Virgin Bride

Wanted in test:

    The Barista's Heart    [0]
    Returning to Nature    [0]

Top 5 of each model:

| matrix factorization | Sentence-BERT | + metadata |
|---|---|---|
| Pride and Prejudice [2180] | Shadow Haven [1] | Arouse [2] |
| Me Before You [902] | Caramel Flava [2] | Looking for La La [0] |
| Anna and the French Kiss [567] | Brie Bows to Her Master [0] | With No Reservations [1] |
| Lady Midnight [228] | Brie's Russian Fantasy [1] | Always the Vampire [1] |
| Slammed [302] | A Line Crossed [1] | Fight (NOLA Zombie, #2) [0] |

The two users have nothing in common, but matrix factorization returns the same
five books to both of them. This is the coverage result of the report seen from
the level of a single user. Both text models return books with almost no
ratings, which is where the books these users actually wanted are.
