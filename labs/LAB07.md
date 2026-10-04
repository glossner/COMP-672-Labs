# Lab 7 — Residual audio quantization and a TTS evaluation harness

Chapters 17

## Implementation tasks

1. Implement nearest-neighbor residual quantization; return indices and reconstructed vectors, and check residuals after each stage.

2. Implement fixed-length nominal rate, using ceiling(log2 K) for non-power-of-two codebooks and stating excluded overhead.

3. Implement a paired bootstrap over sentences for simulated ratings. Keep rater identities fixed and state the resulting inferential limitation.


Functions to complete: `residual_quantize`, `nominal_bitrate`, `paired_sentence_bootstrap`.

## Required experiments

- Compare one and two quantization stages for reconstruction error and nominal rate. Plot or tabulate the tradeoff and show at least three reconstructions.

- Analyze the provided simulated paired ratings, reporting means and a sentence-bootstrap interval. Prepare a blank, blinded listening-study protocol for intelligibility, naturalness, and speaker similarity; do not report synthetic values as human judgments.

## Brief analysis in your notebook or code comments

1. Which parts of a neural codec are absent from the scalar laboratory analogue?

2. Why do codec MSE and automatic transcription error not fully measure TTS quality?

3. What changes to the resampling design would allow generalization beyond the particular raters?


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week07_starter.ipynb)

