# Classification and scoring

GASBench reports both classification accuracy and a combined score that rewards
correct predictions and accurate probabilities. This page explains the labels,
the result fields, and how the final score is calculated.

## Class labels

Models must use this output order:

| Modality | Class 0 | Class 1 | Class 2 |
| --- | --- | --- | --- |
| Image and video | `real` | `synthetic` | `semisynthetic` |
| Audio | `real` | `synthetic` | — |

- **Real:** captured media without substantial generated or replaced content. This
  class also includes non-AI rendering, such as CGI, charts, and game footage.
- **Synthetic:** fully synthesized output. Generation conditioned on a captured
  image is still synthetic if the model synthesizes the complete output.
- **Semisynthetic:** captured visual content with localized generated or replaced
  regions, such as a face swap. Editing exclusively synthetic or rendered media
  does not make it semisynthetic.

The visual taxonomy is experimental. Audio is binary; audio metadata labeled
`semisynthetic` maps to class 1.

Models return logits. GASBench applies softmax and takes the class with the
highest probability as the prediction. See [model outputs](Safetensors.md#4-inputoutput-specifications)
for the interface.

## Read the results

| Field | Meaning |
| --- | --- |
| `sn34_score` | Final score, including the robustness blend when enabled. Higher is better. |
| `benchmark_score` | Accuracy: the fraction of correct class predictions, with any configured weights. |
| `binary_sn34_score` | Base-pass score for real versus all non-real classes combined. |
| `multiclass_sn34_score` | Base-pass score that distinguishes every class in the table above. |

By default, `sn34_score` uses binary scoring. Set `--multiclass-scoring` (Python:
`multiclass_scoring=True`) to reward distinguishing synthetic from semisynthetic
media. Both variants are reported. For two-class audio, their normalized
calculations are equivalent.

Binary scoring uses `p_not_real = 1 - p_real` and predicts non-real when that
probability exceeds `0.5`. For example, probabilities `[0.4, 0.3, 0.3]` predict
`real` in the three-class task but `not real` in the binary task.

Metrics use successful predictions. Unreadable source samples are recorded as
skips; inference failures invalidate the run. Every selected sample must have a
recorded outcome before a run can complete. The fields above describe the base
pass unless explicitly blended.

## How SN34 is calculated

Each pass combines **MCC** (Matthews correlation coefficient), which measures
classification performance, with **Brier error**, which measures squared
probability error. Higher MCC and lower Brier error improve the score.

| Mode | MCC (`M`) | Brier error (`B`) | Baseline (`B0`) |
| --- | --- | --- | --- |
| Binary | `binary_mcc` | `binary_brier` | `0.25` |
| Multiclass | `gorodkin_mcc` | `multiclass_brier` | `(K - 1) / K` for `K` classes |

Binary Brier error compares `p_not_real` with the 0/1 label. Multiclass Brier
error sums squared errors across all class probabilities. Both are averaged
over samples; the baseline is the error from uniform probabilities.

The default calculation is:

```text
mcc_component   = max(0, min((M + 1) / 2, 1)) ** 1.2
brier_component = max(0, (B0 - B) / B0) ** 1.8
pass_score      = sqrt(max(1e-12, mcc_component * brier_component))
```

A perfect predictor scores approximately `1`. Brier error at or above its
baseline gives the numerical floor, `0.000001`. A run with no successful base
predictions fails instead of publishing a score.

`binary_cross_entropy` (log loss) and `per_class_recall` are additional
diagnostics; they do not enter this formula.

## Dataset weighting

Samples have equal weight by default. `score_composition` can give public,
private holdout, and GAS-Station samples different shares of the total weight.
For example, a 50/50 public/holdout target gives each group equal total weight
even if their sample counts differ. Missing groups are dropped and the remaining
target shares are renormalized.

These weights apply to accuracy, MCC, Brier error, and cross-entropy. The legacy
`holdout_weight` option affects accuracy only; `score_composition` supersedes it.

## Robustness scoring

An augmentation pass scores a transformed subset of the inputs separately,
using the same scoring mode and the base pass's provenance weights. When it
produces successful predictions and `aug_weight` is positive:

```text
sn34_score = (1 - aug_weight) * base_sn34_score + aug_weight * aug_sn34_score
```

`base_sn34_score` preserves the score before blending. `augmentation_robustness`
is the augmented score divided by the base score: `1` means equal scores and
values below `1` indicate degradation. With blending disabled, `sn34_score`
stays at the base score.

See [running robustness evaluations](Running-Benchmarks.md#evaluate-robustness)
for transforms and CLI options. For a Subnet 34 round, use that round's
configuration for the scoring mode, dataset shares, and augmentation settings.
