# Dataset taxonomy audit

Date: 2026-08-21

## Scope and decision rule

This audit covers the active GASBench registry: 190 image, 219 video, and 141
audio entries (550 entries, 525 unique repository paths). Legacy configs are not
active and are outside the counts below.

The audit uses the dataset card, project page, or primary paper where the name
alone does not establish provenance. For `34data/*` mirrors, the upstream family
is authoritative only when the mirror is a documented homogeneous slice. A
mirror must not inherit one label from a mixed upstream archive without a
verifiable filter.

The working visual-media rule is:

- `real`: materially camera-captured, without synthetic visual alteration;
- `synthetic`: fully produced by a neural generator;
- `rendered`: fully produced by a graphics, game-engine, or simulation pipeline;
- `semisynthetic`: retains materially captured visual content alongside
  spatially localized generated or replaced visual content. Fully synthesized
  output remains `synthetic`, even when captured media conditions its
  generation; modifying exclusively synthetic or rendered media does not make
  it semisynthetic.

Audio remains binary (`real`/`synthetic`). Replay and presentation attacks are
not synthetic speech under this definition, even if another benchmark calls
them "spoof".

## Executive findings

### Recommended changes

| Dataset | Current | Recommended | Confidence | Basis |
|---|---|---|---|---|
| `v15-human-vid-digifakeavfvfa_with_audio` | synthetic | semisynthetic | high | DigiFakeAV's FVFA visual stream is produced with face-swap/lip-sync style deepfake processing over source identity/video material. Fake audio does not change the visual provenance label. |

### Must verify the `34data` repack before retaining

| Dataset | Current | Risk | Recommended action |
|---|---|---|---|
| `asvspoof2021-df-eval` | synthetic | The official DF evaluation database contains both bona-fide and spoof trials. | Confirm that the mirror contains spoof-key rows only; otherwise split or exclude. |
| `asvspoof2021-la-eval` | synthetic | The official LA evaluation database contains both bona-fide and spoof trials. | Confirm spoof-only filtering; otherwise split or exclude. |
| `asvspoof2021-pa-eval` | synthetic | The official PA database contains bona-fide speech and replay attacks. Replayed human speech is not generated speech. | Exclude from synthetic detection unless the benchmark intentionally treats replay as synthetic; even then, split bona-fide rows first. |

These are not cosmetic uncertainties. If the mirrors preserve the official
archives, class-zero human recordings are currently injected into class one.

## Confirmed rendered video families

The following active `rendered` assignments are consistent with their published
construction:

- `abot-world-explorer`: action-conditioned game/simulation world-exploration
  episodes, with keyboard actions and COLMAP camera poses;
- `physicalai-part1-split4`, `physicalai-part1-split5`: simulator data;
- `bedlam-closeup-suburb-a`, `bedlam-closeup-suburb-b`: conventionally rendered
  synthetic humans;
- `nvidia-sdg-synhuman-shard-7`, `nvidia-sdg-synhuman-shard-8`: NVIDIA synthetic
  data generation/rendering;
- `cs2-10k-data-ancient-part-01`: game-engine footage;
- `ByteDance_Synthetic_Videos`: explicitly CGI video with rendering-setting tags;
- `scene-decoupled-video-dataset`: 46,816 sequences rendered in Unreal Engine 5;
- `synthetic-vstat-block-counting`, `synthetic-vstat-shuffle-puzzle`: rendered
  synthetic task environments;
- `synwts`: a digital twin implemented with NVIDIA Isaac Sim.

No active image is labeled `rendered`, consistent with the three-class image
head. Known CGI-only image dumps are excluded rather than folded into neural
`synthetic`.

## Confirmed generative versus edited families

### High-confidence fully generative

The registry is consistent for the major explicit generator families: Stable
Diffusion/SDXL/SD3, FLUX, DALL-E, Midjourney, StyleGAN/GAN face sets, Veo, Sora,
Kling, CogVideoX, Wan, HunyuanVideo, LTX-Video, Mochi, OpenSora, VideoCrafter,
T2V-Turbo, and text-to-video preference datasets. DeepAction's six fake splits
are explicitly text-to-video generations; its Pexels split is correctly real.
PoseDreamer is diffusion-generated rather than a conventional 3D render and is
correctly `synthetic`. Open-VFX is also correctly `synthetic`: Pika/PixVerse
fully synthesize its image-conditioned VFX videos from Pexels reference images.

### High-confidence semisynthetic

FaceForensics/FF++, face swap, reenactment, lip-sync, inpainting, object
replacement/removal, and instruction-edit outputs are correctly semisynthetic
when their released target is derived from captured input. This supports the
current labels for FakeClue FF++, AttGAN/StarGAN/STGAN, receipts-i2i, GPT image
edit, DFD, FOMM, FSGAN, Wav2Lip, FaceDancer, FaceVid2Vid, Sim/SwimSwap,
LivePortrait/EchoMimic/Roop MAVOS slices, FakeParts edit tasks, and Señorita edit
tasks.

## Uncertain visual families requiring sample/column lineage checks

These cannot be decided reliably from a method name or top-level card. They
should retain their current label only provisionally.

| Family | Current | Why uncertain | What resolves it |
|---|---|---|---|
| `imagepulsev2-*` | synthetic | The family contains both edit-like and generated pairs. | Verify whether each sampled target retains captured pixels or is fully resynthesized. Fully resynthesized targets remain synthetic regardless of source conditioning. |
| `image_patches_raw` | synthetic | Name and registry metadata do not establish whether patches are generated, edited, or extracted from mixed imagery. | Inspect upstream construction and sampled column. |
| `semisynthetic-video` | semisynthetic | Generic mirror name hides the generating method and source. | Verify that outputs retain captured regions rather than fully resynthesizing every frame. |
| `VAP-data` | synthetic | VAP-Data contains semantic-control/reference-video pairs and serves generation, editing, and VFX tasks. | Determine whether the registered clips are targets, prompts, or a mixture; fully synthesized targets remain synthetic. |
| `lovora-fake` | synthetic | The “fake” split alone does not distinguish localized compositing from full-frame synthesis. | Confirm whether the registered output retains captured pixels. |
| `spoof_png` | real | Presentation-attack imagery is camera-captured but is not authentic/live. It is “real” only under pixel provenance, not liveness semantics. | Document that GASBench classifies pixel production rather than scene authenticity, or exclude presentation attacks. |

## Mixed/ambiguous exclusions reviewed

The current exclusions are justified under the one-label-per-entry model:

- `justweirdimages`: real photos, animation, and composites without a reliable
  split;
- `cg-fake-id`: CGI stills, while image has no rendered class;
- `deepfake-insight`: filenames do not establish manipulated versus real rows;
- `artifact-bench`: mixed real/fake archive without a usable registry filter;
- PICA is correctly not split into “real source” and synthetic target: its source
  frames come from synthetic videos.

## Audio review

The explicit TTS, voice-conversion, vocoder, and speech-generation families are
consistent with `synthetic`, including MLAAD, VCC fake, DFADD, ShiftySpeech,
ElevenLabs, CosyVoice, F5-TTS, Fish Speech, VITS, Glow-TTS, Grad-TTS,
Tacotron/WaveGrad, and SpeechArena generated outputs. Corpus, ASR, emotion, and
bonafide splits are consistent with `real`.

Remaining audio checks:

- `arabic-deepfake`: the card points to a separate real-audio source and an RVC
  fake dataset, but the registered repository must be checked to ensure it is
  fake-only rather than paired/mixed;
- `asvspoof5-spoof`: verify the repack filters spoof rows only (the paired
  `asvspoof5-bonafide` entry suggests it probably does);
- `fakesound2-*`, `audiospoofing-mini-*`, `cvoice-small-*`, and similar paired
  mirrors are structurally plausible, but their mirror-side filtering is not
  externally inspectable from upstream names alone.

## Sources consulted

- ABot-World-0 paper and ABot World Explorer 500h dataset card
- DeepAction dataset card / Human Action CLIPS paper
- PoseDreamer paper and dataset release
- CineScene project and Scene-Decoupled Video Dataset card
- ByteDance Synthetic Videos card and associated paper
- SynWTS card and AI City Challenge documentation
- Open-VFX/VFX Creator paper and dataset card
- DigiFakeAV repository and FakeAVCeleb/FVFA method descriptions
- Señorita-2M project, paper, and repository
- VAP-Data card and Video-As-Prompt paper
- ImagePulseV2 collection cards and project documentation
- IP-Adapter paper/repository
- ASVspoof 2021 official challenge documentation and evaluation plan
- DFADD paper and official repository

## Bottom line

Most named, homogeneous generator/edit/render families agree with the v23
taxonomy. One video entry should be relabeled with high confidence, and the
three ASVspoof 2021 archive entries need immediate mirror-level verification.
The uncertain table should not be silently “resolved” from names: each item
needs sampled-column and source-lineage evidence.
