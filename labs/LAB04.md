# Lab 4 — Retrieval and a bounded evidence-based controller

Chapters 10, 11, 12; 

## Implementation tasks

1. Implement reciprocal rank with an explicit cutoff and zero credit for misses.

2. Implement the bounded controller using only search and read, validate observations, count every call, and return answer, abstain, budget, or tool_error.


Functions to complete: `reciprocal_rank`, `run_agent`.

## Required experiments

- Report MRR@3, answer success, supporting source, and calls for the four supplied questions.

- Run the supplied unanswerable, forced-failure, budget, and adversarial-text cases; explain each outcome from its trace.

## Brief analysis in your notebook or code comments

1. Why does success of this deterministic controller not establish the performance of an LLM planner?

2. Which observed passage supports one successful answer, and how can you verify that support?

3. Which failure case was most informative, and what one change would you evaluate next?


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission

From Colab: **Share → General access → Anyone with the link → Viewer**. Click **Copy link**. Paste the link into the Canvas assignment submission.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week04_starter.ipynb)
