# Lab 4 — Retrieval and a bounded evidence-based controller

**2 points.** Chapters 10, 11, 12; outcomes LO4, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week04_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
