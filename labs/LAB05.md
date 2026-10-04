# Lab 5 — A small recurrent translator and beam search

Chapters 14, 13

## Implementation tasks

1. Implement normalized source attention and verify the output is a weighted combination of encoder states.

2. Implement a beam with cumulative log scores, EOS handling, a finite generation limit, and deterministic tie breaking.

3. Implement next-target loss for the teacher-forced recurrent decoder. Trace a training step and a free-running step.


Functions to complete: `context_attention`, `beam_search`, `sequence_loss`.

## Required experiments

- Train the small GRU model on the fixed training combinations and evaluate greedy versus width-3 beam search on held-out combinations. Report all individual predictions as well as exact match.

- Run the supplied probability-tree counterexample and verify that beam width 2 finds probability .38 while greedy finds .36. State the no-length-normalization convention.

## Brief analysis in your notebook or code comments

1. Why can teacher-forced loss improve without equivalent free-running performance?

2. Is a beam gain evidence about search or about the learned distribution, and what does the controlled counterexample isolate?

3. What would be needed to extend the invented reordering task to real machine translation?


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week05_starter.ipynb)

