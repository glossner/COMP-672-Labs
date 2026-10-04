# Lab 1 — Tokenization and a smoothed bigram model

Chapters 1, 2, 3

## Implementation tasks

1. Implement NFC-aware tokenization; document how punctuation, underscores, digits, and combining marks behave.

2. Implement weighted pair counting and deterministic BPE merging. Inspect at least six learned merges and segment three held-out words.

3. Implement normalized add-alpha bigram probabilities with a training-only vocabulary and explicit BOS, EOS, and UNK conventions.

4. Implement perplexity with token-weighted log losses; tune alpha on development documents using the provided candidate values.


Functions to complete: `tokenize`, `learn_bpe`, `Bigram.probability`, `Bigram.perplexity`.

## Required experiments

- Report development perplexity for alpha .01, .1, and 1; choose one before reporting the test score.

- Compare word, character, and BPE token counts on the same five sentences. Do not compare their raw per-token perplexities as if the units were identical.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week01_starter.ipynb)
