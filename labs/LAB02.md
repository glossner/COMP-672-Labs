# Lab 2 — Logistic regression embeddings and a neural classifier

Chapters 4, 5, 6

## Implementation tasks

1. Implement stable mean binary cross-entropy and its gradients with the specified L2 convention. Check both weights and bias by finite differences.

2. Implement batch gradient descent and report whether the training objective falls.

3. Implement cosine similarity with similarity defined as 0 when either vector has zero norm. Inspect five nearest document vectors under the SVD representation.

4. Implement a paired document bootstrap for the accuracy difference between two fixed predictions.


Functions to complete: `loss_gradient`, `train_logistic`, `cosine`, `paired_bootstrap`.

## Required experiments

- Run the fixed 2×2 comparison: TF-IDF or 16-dimensional SVD features crossed with linear or MLP classification. Report development and test macro F1 and the two-by-two confusion matrices.

- Analyze at least four negated examples. Add a bigram-feature comparison selected on development data and explain how feature expressiveness affects the linear baseline.

## Brief analysis in your notebook or code comments

1. Why is the generated label rule difficult for an additive unigram classifier?

2. What part of the comparison measures representation and what part measures nonlinearity?

3. What uncertainty does the paired interval capture, and what training variability does it omit?

Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week02_starter.ipynb)

