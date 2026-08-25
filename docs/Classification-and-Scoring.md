# Classification Taxonomy and Scoring

GASBench evaluates the provenance of image, video, and audio media. The visual
taxonomy introduced in v0.9.0 is experimental: it makes a finer distinction
than the historical `real` versus `synthetic` task, and may evolve as the
benchmark is evaluated in practice.

## Classification contract

Class indices are part of the model interface and must use this order:

| Modality | Classes (`num_classes`) |
| --- | --- |
| Image | `0=real`, `1=synthetic`, `2=semisynthetic` (`3`) |
| Video | `0=real`, `1=synthetic`, `2=semisynthetic`, `3=rendered` (`4`) |
| Audio | `0=real`, `1=synthetic` (`2`) |

The classes mean:

- **Real**: captured from the physical world without material generated or
  replaced visual content.
- **Synthetic**: fully synthesized output, including generative-model output.
  It remains synthetic when captured media conditions generation, because the
  output pixels are still synthesized.
- **Semisynthetic**: retains materially captured visual content alongside
  spatially localized generated or replaced visual content.
- **Rendered**: fully produced by a graphics, game-engine, animation, or
  simulation pipeline rather than captured by a camera. This class currently
  applies only to video.

Modifying exclusively synthetic or rendered media does not make it
semisynthetic. Image-to-video generation, for example, is synthetic when the
model synthesizes the complete output, even if a captured image conditions it.

Audio remains a binary task. Any audio dataset metadata using
`semisynthetic` is collapsed onto the synthetic label.

## Probabilities and predictions

Models return one logit per class in the order above. GASBench applies softmax,
uses the argmax as the multiclass prediction, and retains the probabilities for
calibration metrics.

For compatibility metrics, all non-real classes collapse into one class:

\[
p_{\text{not real}} = 1 - p_{\text{real}}.
\]

This binary view is always reported, but it does not reward a visual model for
distinguishing synthetic, semisynthetic, and rendered media.

## Metrics

GASBench reports both binary and multiclass variants on every run:

- `binary_mcc`: MCC after collapsing every non-real class into synthetic.
- `binary_brier`: mean squared error of `p_not real`; `0.25` is the constant
  `p=0.5` baseline.
- `binary_cross_entropy`: binary log loss for the same collapsed probabilities.
- `gorodkin_mcc`: Gorodkin's \(R_K\), the multiclass generalization of MCC.
- `multiclass_brier`: mean of \(\sum_k (p_k-y_k)^2\). Its uniform-prediction
  baseline for \(K\) classes is \((K-1)/K\).
- `per_class_recall`: recall indexed by the class numbers above.
- `binary_sn34_score` and `multiclass_sn34_score`: the two comparable SN34
  score variants.

For either scoring mode, let \(M\) be the relevant MCC, \(B\) the relevant
Brier score, and \(B_0\) its random baseline (`0.25` for binary or
\((K-1)/K\) for multiclass):

\[
M_{norm} = \operatorname{clip}\left(\frac{M+1}{2},0,1\right)^{1.2}
\]

\[
B_{norm} = \max\left(0,\frac{B_0-B}{B_0}\right)^{1.8}
\]

\[
SN34 = \sqrt{M_{norm} B_{norm}}.
\]

`sn34_score` is the variant selected by the benchmark configuration. Subnet 34
currently selects multiclass scoring for image and video. Audio uses binary
scoring; for two classes, the normalized multiclass calculation is
mathematically identical.

## Dataset composition and robustness

A benchmark may assign target score shares to public, private holdout, and
GAS-Station samples. GASBench converts those shares into per-sample weights and
uses them consistently for accuracy, MCC, Brier, cross-entropy, and the derived
SN34 scores. If a configured provenance group is absent, the remaining shares
are renormalized.

When an augmentation pass is enabled, GASBench reports:

- `base_sn34_score`: score on the normal evaluation pass;
- `aug_sn34_score`: score on the augmentation pass;
- `augmentation_robustness`: robustness diagnostics; and
- `sn34_score`: the configured blend
  \((1-w)\,base + w\,aug\).

The Subnet 34 round configuration, rather than the GASBench library, is the
source of truth for the current provenance shares, augmentation sample count,
and blend weight. Local GASBench runs can select multiclass scoring with
`--multiclass-scoring`.
