# Lab 5 — A small recurrent translator and beam search

**2 points.** Chapters 14, 13; outcomes LO5, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week05_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
