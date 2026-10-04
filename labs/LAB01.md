# Lab 1 — Tokenization and a smoothed bigram model

**2 points.** Chapters 1, 2, 3; outcomes LO1, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

## Implementation tasks

1. Implement NFC-aware tokenization; document how punctuation, underscores, digits, and combining marks behave.

2. Implement weighted pair counting and deterministic BPE merging. Inspect at least six learned merges and segment three held-out words.

3. Implement normalized add-alpha bigram probabilities with a training-only vocabulary and explicit BOS, EOS, and UNK conventions.

4. Implement perplexity with token-weighted log losses; tune alpha on development documents using the provided candidate values.


Functions to complete: `tokenize`, `learn_bpe`, `Bigram.probability`, `Bigram.perplexity`.

## Required experiments

- Report development perplexity for alpha .01, .1, and 1; choose one before reporting the test score.

- Compare word, character, and BPE token counts on the same five sentences. Do not compare their raw per-token perplexities as if the units were identical.

## Brief analysis in your notebook or code comments

1. Explain why a distribution over successors must sum to one even for an unseen history.

2. Show a tokenizer failure case and propose a task-dependent remedy.

3. Explain what the generated corpus permits you to conclude and what a real-domain evaluation would add.


Answer each question in approximately 2–4 sentences beside the results. This is part of the lab, with no separate report.

## Submission and rubric

Submit the link to your completed, saved Colab copy (or edited notebook/source if working locally), a results JSON file, and an execution command. Follow the save/share steps in the repository README. For Colab, state “Run cells from top to bottom in a fresh CPU runtime.” Include versions, seed, split policy, settings, runtime, and clear labels. No points depend on a positive gain.

| Criterion | Points | Evidence |
| --- | --- | --- |
| Implementation | 0.8 | Required functions implement the intended mathematics and boundary behavior |
| Experiments | 0.6 | Both required comparisons are run and outputs are recorded |
| Comprehension | 0.4 | Brief answers accurately connect results to the three analysis questions |
| Reproducibility | 0.2 | Runnable submission with configuration, seed, and data scope |
| Total | 2.0 | Partial credit is proportional to the evidence supplied |

Split implementation credit equally among required functions; split experiment credit equally between the two comparisons. Assess the three analysis answers together on a 0–0.4 scale; award 0.4 for all three accurate and grounded, 0.2 for partial understanding, 0 for absent or fundamentally incorrect explanations. Intermediate credit is allowed.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week01_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
