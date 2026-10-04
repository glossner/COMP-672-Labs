# Lab 3 — A tiny transformer with causal and masked objectives

Chapters 7, 8, 9

## Implementation tasks

1. Implement scaled attention and a causal mask; verify that changing a future value leaves earlier causal outputs unchanged.

2. Implement shifted next-token cross-entropy and explain why the last position is excluded.

3. Implement a selected-position masked loss and contrast it with the causal objective; ensure unselected positions do not contribute.

4. Implement the rank-2 adapter forward pass with a frozen base head and scale alpha/r. Count trainable parameters and check the zero-update initialization.


Functions to complete: `attention`, `causal_loss`, `masked_loss`, `LoRALinear.forward`.

## Required experiments

- Run the tiny causal pretraining and head-only LoRA adaptation experiments. Report initial loss, fixed-training-endpoint test loss, adapter parameter count, and adaptation loss before and after. Use the same adapted-distribution test set for the before/after comparison.

- Run the bidirectional masked model. Perturb right context to demonstrate visibility and explain why its loss is not directly ranked against causal perplexity.

## Brief analysis in your notebook or code comments

1. Which causal-mask bug could create deceptively low training loss?

2. What can adaptation of only an output head change, and what can it not directly change in the frozen backbone?

3. The finite grammar repeats possible sequences across independent draws. Explain why these results measure a small distribution-fitting exercise rather than held-out linguistic compositionality.


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week03_starter.ipynb)
