# Lab 6 — Speech features CTC and recognition error analysis

**2 points.** Chapters 15, 16; outcomes LO6, LO8. Allow 4–5 hours including the 60-minute lab meeting. Complete the Python starter or the Jupyter starter, not both. The scaffolding contains intentional `NotImplementedError` gaps.

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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week06_starter.ipynb)

[Save and submit instructions](../README.md#save-and-submit)
