# Lab 7 — Residual audio quantization and a TTS evaluation harness

**2 points.** Chapters 17; outcomes LO7, LO8, LO3, LO4, LO6. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week07_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
