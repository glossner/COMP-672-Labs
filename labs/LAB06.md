# Lab 6 — Speech features CTC and recognition error analysis

Chapters 15, 16

## Implementation tasks

1. Implement no-padding framing and verify the 16-kHz one-second example yields 98 frames of 400 samples.

2. Implement windowing, power spectrum, supplied Mel-filter projection, and log flooring. Plot or tabulate a feature slice with units.

3. Implement CTC collapse and enumerate compatible paths for a tiny example using the supplied exact enumerator.

4. Implement edit alignment and return S, D, I, N, and WER using the declared tie and tokenization conventions.


Functions to complete: `frame_signal`, `log_mel`, `ctc_collapse`, `word_errors`.

## Required experiments

- Compare clean and noise-corrupted two-tone features, then compare two frame lengths while keeping the hop explicit. Show the resulting dimensions.

- Compute per-utterance and pooled WER for the supplied transcripts. Add a repeated-label CTC example and an insertion-heavy transcript whose WER exceeds one.

## Brief analysis in your notebook or code comments

1. Why are these tone experiments useful for feature debugging but insufficient evidence about phoneme recognition?

2. Why does removing blanks before collapsing repeats produce a wrong CTC transcript?

3. Why can an unweighted average of utterance WER differ from pooled corpus WER?


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week06_starter.ipynb)

