# Lab 2 — Logistic regression embeddings and a neural classifier

**2 points.** Chapters 4, 5, 6; outcomes LO2, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week02_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
