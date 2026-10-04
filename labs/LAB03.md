# Lab 3 — A tiny transformer with causal and masked objectives

**2 points.** Chapters 7, 8, 9; outcomes LO3, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week03_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
