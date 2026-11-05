# ANPR at Home

This repository contains YOLO plate detection notebooks and an experimental
ResNet18 recognizer. The recognizer's goal is to retain plausible character
alternatives and rank complete plates, including when the input is blurred.
It accepts standalone plate crops and crops from a detector. Recognition
quality has been measured on a small public real-image benchmark; independent
validation on the user's camera and trained detector remains incomplete.

## Findings

- ResNet18 is a usable feature extractor, but global pooling and fixed character
  slots were weak choices for the tested variable-length data. The best measured
  prototype uses ResNet18, a bidirectional LSTM and CTC sequence alignment.
- U-Net preprocessing did not show a consistent benefit in the matched spatial
  ResNet experiment and added processing cost. Keep it optional. The earlier
  small apparent gain used different classifier initialization and is not enough
  evidence to recommend it.
- On the exploratory real-image benchmark, CTC retained the true plate in the
  top 50 for 62 of 107 original crops and 38 of 107 crops with synthetic defocus
  setting 1. Stronger artificial defocus and motion blur still failed on nearly all examples. These are
  candidate-recall results, not exact-reading accuracy or a guarantee of correctness.
- All model training and temperature fitting used synthetic data. The public
  benchmark informed development, so it cannot serve as an untouched final
  camera-domain holdout. Its crops use annotation boxes, not the user's trained
  YOLO detections.
- Naturally blurred vehicle clips have not been labelled or evaluated. The
  corruption experiments apply artificial blur and other changes to source
  crops; they do not replace clear-reference labelling of real video tracks.

The [CTC report](reports/ctc-experiment.json),
[U-Net comparison](reports/spatial-experiment.json) and raw per-condition reports
below preserve the measurements and their limitations. The next evidence gate is
labelled vehicle clips and execution with the previously trained detector.

## Design and experiment contract

The first baseline keeps the original notebook's ResNet18 and independent
character heads. Class zero means absent trailing characters, not a CTC blank.
The default capacity is seven characters; shorter registrations are padded,
and longer labels must use a model trained with a larger capacity. Inputs are
RGB plate crops, letterboxed to 192 by 64 pixels without stretching. This
preprocessing differs from the original notebook's 224 by 224 stretch, so
old recognizer weights cannot be treated as a compatible evaluated checkpoint.
The YOLO detector remains separate and does not need retraining to supply crops.

The next controlled experiment retains this globally pooled model as
`--head-layout global` and adds `--head-layout spatial`. The spatial variant
retains width through the last two ResNet stages and sends each positional head
its own pooled region. It keeps the same parameter shapes, count and initial
weights, so the comparison tests spatial processing rather than adding
a larger classifier. The default stays global until measured results justify
another choice. Neither variant uses CTC or a learned text-language prior.

A separate CTC experiment now tests the alignment limitation directly. It uses
the same width-preserving ResNet18 encoder, a bidirectional sequence layer and
a shared frame classifier. CTC class zero is a blank between characters, rather
than absent trailing output slots. It learns transcription without assigning
each character to a fixed horizontal region. The original positional model
remains available and is not silently replaced.

CTC beam search is bounded and approximate. Retained plate candidates are scored
by summing all valid alignments in log space. Full frame distributions remain
available. Ordered character marginals sum the probability of each character
appearing at each output position across all alignments and possible strings.
Their null class means the string ended before that position. Separate
`alignment_positions` retain best-candidate Viterbi frame snapshots; their null
class means a frame blank. The character marginals do not depend on the beam,
lookup list or chosen best string and are not normalized over returned candidates.
Mass beyond the configured plate capacity is reported explicitly. Character
marginals remain dependent, so multiplying them does not give a sequence score.
Neither these model probabilities nor a sequence score establishes calibrated
real-camera correctness. The experiment must improve measured recall before
it is adopted.

Optional mask initialization is isolated from the shared backbone/head random
generator state. New RGB and mask runs can therefore start with identical shared
weights. Historical mask results below predate that control and remain labelled
as limited evidence. Existing version-one checkpoints still load with global
pooling when layout metadata is absent.

Positional candidate ranking uses all character classes and sums log probabilities across
all output positions, including trailing padding. It enforces contiguous text
followed by padding. Optional position masks or a known-plate list are explicit
constraints, not visual evidence. Scores are independent-head model scores,
not calibrated probabilities that a complete plate is correct. A bounded top-K
search avoids enumerating every character combination. The output preserves
the full distribution at each position, even when it returns few plate candidates.

The optional U-Net experiment supplies a soft character mask alongside the
original RGB image. It must not replace the original image with a hard mask.
Synthetic rendering can provide exact masks; real mask supervision requires
aligned annotations. Segmentation does not rectify perspective or recover
information that blur has destroyed. Retain the extra network only if it improves
held-out recognition at a useful candidate budget and acceptable latency.

Evaluation must keep identical plates and shared source captures in one split.
Connected groups are indivisible, even when this prevents ideal split sizes.
Train on realistic blur, resolution, compression and crop changes. Fit any
temperature only on validation data; report character calibration separately
from whole-plate recall. Test on held-out real detector crops as well as synthetic
ones. Report exact plate accuracy, recall at 5/10/50 candidates, candidate count,
latency and character calibration by corruption. A readable mask or passing
unit tests do not demonstrate useful blurred-plate recognition.

## Data needed

Roboflow project: <https://app.roboflow.com/homeanpr/journey-cam-anpr/train>.
The original notebook uses version 5 of `homeanpr/journey-cam-anpr`.
No trained weights or real datasets are checked in. Detector annotations usually
contain boxes and class names, not registrations. OCR training needs the exact
plate text for each crop, plus a source video/session group to prevent leakage.
Do not use guessed OCR outputs as evaluation ground truth.

### Labelling naturally blurred footage

A blurry crop needs a label established from a clearly readable frame of the
same vehicle track. Label the track once from that reference, then associate its
original blurry crops with the verified text. Confirm that the track follows the
same plate throughout; nearby vehicles, occlusion and tracker identity switches
can otherwise create incorrect labels. A clear frame used as reference must not
be an input to the single-crop recognition test.

For each labelled track, retain the original clip/capture ID, track ID, reference
frame timestamp, verified plate text and each test frame's timestamp and crop
coordinates. Keep the source pixels unchanged. The current training manifest
needs `path,plate,group`; store the reference and frame provenance in accompanying
annotation columns or a separate annotation file. They are dataset evidence, not
implemented video-tracking or automatic label-propagation features.

If no frame gives a trustworthy reading, keep the track's text unknown. Exclude
it from exact-text accuracy and candidate-recall calculations rather than guessing
a label. It can still be retained for a separately reported abstention review.
The current supervised loader rejects empty labels, so unknown tracks must stay
outside its labelled manifest. Report how many tracks were unlabelled and why;
tracks with a clear reference may be easier than tracks that never become readable.

Keep the clear reference and all related blurry frames in the same split. Use a
shared capture/session `group` for related tracks from one recording; repeated
plate identities also stay together through the existing split checks. Do not
randomly split adjacent frames. Select epochs, temperatures and rejection
thresholds using training/validation sources, then evaluate the separate test
captures without using their clear references during prediction.

Artificial blur of labelled clear crops remains useful for controlled experiments,
but does not fully reproduce natural motion, low native resolution, glare,
compression or detector crop errors. Evaluate natural clips separately and report
both crop-level results and track counts so many near-duplicate frames from one
vehicle do not look like independent evidence. No natural-video labelling or
track-level evaluation has been completed yet.

An interim real-image check uses the official
[OpenALPR European benchmark](https://github.com/openalpr/benchmarks/tree/9790ed20d475be1e940a7e2a3f492a97c2ebed9b/endtoend/eu)
at revision `9790ed20d475be1e940a7e2a3f492a97c2ebed9b`. This is a separate
European benchmark, not the user's UK camera domain. Its repository declares
AGPL-3.0; the local source and crop folders retain its license and pinned source
URLs. Image-acquisition and annotation methods are not documented there, so
labels are described as upstream benchmark annotations, not independently
verified human truth.

To reproduce the local source, select `.jpg` and `.txt` blobs under `endtoend/eu/`
from the [pinned Git tree](https://api.github.com/repos/openalpr/benchmarks/git/trees/9790ed20d475be1e940a7e2a3f492a97c2ebed9b?recursive=1).
Download them from `https://raw.githubusercontent.com/openalpr/benchmarks/9790ed20d475be1e940a7e2a3f492a97c2ebed9b/`
followed by each tree path, and download root `LICENSE` from that same revision.
Keep the EU files together in `runs/openalpr-eu/source/`. The local
`UPSTREAM.json` records exact URLs, byte counts and SHA256 values for this run.

```bash
.venv/bin/python -m anpr.openalpr --source runs/openalpr-eu/source --output runs/openalpr-eu/crops --max-length 7
```

This imports 107 crops from 108 source examples. The eight-character label is
retained as an explicit capacity exclusion in `provenance.json`; it is never
truncated. All rows are test-only and share one source group. They do not train
weights, choose epochs or fit temperature. These are annotation-box crops, so
they test recognition without demonstrating the user's YOLO detection quality.
The `clean` condition means no added corruption, not an independently verified
sharp image. Added defocus, motion, downsampling, JPEG and crop changes are
synthetic. No naturally blurred video tracks are included in the evaluation.

Use a CSV manifest with `path,plate,group` columns, and optional `split,mask_path`
columns. Paths are relative to the manifest. Splits are `train`, `val` and `test`.
Use either supplied splits for all rows or deterministic group splitting. Related
plates and capture groups must not cross split boundaries. Masks use white
character pixels and black background and must match the image geometry.

To prepare an extracted Roboflow YOLO export:

```bash
.venv/bin/python -m anpr.prepare --dataset extracted-roboflow-export --output dataset/ocr-crops --plate-class-id 0
```

This writes original RGB crops and `manifest.csv`. Plate text and capture group
remain blank until supplied by a human or an existing trusted annotation source.
The training loader rejects incomplete templates. Box/class annotations cannot
supply registration text. Fill `plate` and `group` per row, then review the split
assignments. `valid` exports map to `val`; an identity or source crossing splits
will fail validation rather than quietly leak into test.

Existing transcriptions can be joined with `--annotations labels.csv`, using
`source,box_index,plate,group` columns. `source` is the dataset-relative image
path, such as `train/images/frame.jpg`; `box_index` is its zero-based YOLO label
line before class filtering. Unmatched crops remain incomplete. Output folders
must be empty so a second export cannot silently overwrite corrected labels.

## Setup

Use Python 3.10 or newer. Development checks here use Python 3.12 and CPU PyTorch.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-recognition.txt
.venv/bin/python -m pytest
```

Install the appropriate PyTorch build instead for GPU training. YOLO integration
also requires `ultralytics`, which is listed in the original `requirements.txt`.
The notebooks retain their original dependencies. The new commands do not rely
on notebook variables or prior cell execution.

## Train and evaluate

For an initial execution check, generate synthetic crops and paired masks:

```bash
.venv/bin/python -m anpr.data --generate-dir runs/synthetic --count 300 --seed 42
.venv/bin/python -m anpr.train --manifest runs/synthetic/manifest.csv --output runs/rgb.pt --epochs 10 --batch-size 16 --pretrained
.venv/bin/python -m anpr.train --manifest runs/synthetic/manifest.csv --output runs/masked.pt --epochs 10 --batch-size 16 --pretrained --use-mask
```

`--pretrained` downloads ImageNet ResNet18 initialization. Omit it for an offline
execution check. By default the generator uses one simplified monospaced font and layout;
its scores do not demonstrate real-camera recognition. Use a labelled real-crop
manifest for the actual experiment. The mask model requires aligned targets for
every train and validation row; the RGB baseline does not need masks. Test crops
do not need masks because inference predicts its own soft mask.

For broader positive label exposure, choose explicit training formats and fonts:

```bash
.venv/bin/python -m anpr.data --generate-dir runs/coverage-study/data --count 1000 --seed 42 --formats A AA AAA AAAA AAAAA AAAAAA AAAAAAA LLDDLLL --font-paths /usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
.venv/bin/python -m anpr.train --manifest runs/coverage-study/data/manifest.csv --output runs/coverage-study/spatial.pt --epochs 15 --batch-size 8 --threads 2 --pretrained --head-layout spatial
```

These masks control synthetic label sampling, not inference constraints. `A`
allows any supported letter or digit, so shorter masks also teach trailing
padding. Font paths must exist and support the required glyphs. `generator.json`
records fonts and hashes, sampling settings and limitations. Output folders must
be empty. The original default preserves its earlier labels and pixel hashes.
Finite samples still need the positional coverage audit; selecting a vocabulary
does not guarantee enough examples for every class at every head.

```bash
.venv/bin/python -m anpr.evaluate --checkpoint runs/rgb.pt --manifest runs/synthetic/manifest.csv --output runs/rgb-evaluation.json --calibrate --calibration-output runs/rgb-calibration.json --data-kind synthetic
.venv/bin/python -m anpr.evaluate --checkpoint runs/masked.pt --manifest runs/synthetic/manifest.csv --output runs/masked-evaluation.json --calibrate --calibration-output runs/masked-calibration.json --data-kind synthetic
```

Evaluation uses test rows only and applies clean/original, defocus, motion,
downsample, JPEG and crop conditions at severities 1 and 2. These severity units
are development settings, not physical camera measurements. All rows count
toward recall, including unreadable predictions. Character ECE uses ten equal
confidence bins and is reported both with and without padding. Temperature is
fitted on original validation crops only, so calibration under blur remains
an empirical question. It does not change the unconstrained ranking of independent
heads, and it does not calibrate whole-plate correctness.

Training checkpoints record split membership, seed, epoch, manifest hash, model
capacity, mask mode, alphabet and preprocessing. Evaluation rejects test rows
whose plate identity, source group or path appeared in checkpoint training or
validation. Changing the split seed cannot manufacture a held-out result.
Reports hash the checkpoint and manifest and state whether the data is real or
synthetic. Keep source images immutable with the manifest. CPU runs default to
two threads; select `--device cuda` and a suitable batch size for GPU runs.
Completed training also writes a `.training.json` file beside the checkpoint
with the loss history and run settings. Generated data and checkpoints under
`runs/` stay local and are ignored by Git.

For a separate test-only manifest, transfer the frozen validation sidecar rather
than fitting confidence scaling on test labels:

```bash
.venv/bin/python -m anpr.evaluate --checkpoint runs/rgb.pt --manifest runs/openalpr-eu/crops/manifest.csv --output runs/real-evaluation.json --calibration runs/rgb-calibration.json --dataset-provenance runs/openalpr-eu/crops/provenance.json --data-kind real
```

`--calibration` verifies the checkpoint hash, validation-only fit, source manifest
hash, actual fitted validation records and finite positive temperature. Evaluation
rejects test rows sharing a fitted plate, capture group or file path, including
calibration performed after checkpoint training on a different source. Older
sidecars without fitted membership must be regenerated using `--calibrate`.
The option is mutually exclusive with `--calibrate`
and `--temperature`. Transferred calibration is still measured on the target
domain; a source-domain fit does not certify real-camera confidence.

## CTC alignment experiment

The CTC alignment comparison uses the same broader dataset and preprocessing:

```bash
.venv/bin/python -m anpr.ctc_train --manifest runs/coverage-study/data/manifest.csv --output runs/ctc-comparison/ctc.pt --epochs 15 --batch-size 8 --threads 2 --pretrained
.venv/bin/python -m anpr.ctc_evaluate --checkpoint runs/ctc-comparison/ctc.pt --manifest runs/coverage-study/data/manifest.csv --output runs/ctc-comparison/synthetic.json --calibrate --calibration-output runs/ctc-comparison/calibration.json --dataset-provenance runs/coverage-study/data/generator.json --data-kind synthetic --beam-width 100
.venv/bin/python -m anpr.ctc_evaluate --checkpoint runs/ctc-comparison/ctc.pt --manifest runs/openalpr-eu/crops/manifest.csv --output runs/ctc-comparison/real.json --calibration runs/ctc-comparison/calibration.json --dataset-provenance runs/openalpr-eu/crops/provenance.json --data-kind real --beam-width 100
```

CTC uses its own evaluator because frames are not prealigned character heads.
It reports plate recall, character edit rate, exact true-sequence NLL and raw
sequence-score ECE. Temperature is selected on validation data using a declared
17-point grid from 0.25 through 4. The objective is not assumed convex, and a grid
boundary selection remains visible. CER can exceed one and must not be compared
directly with positional character accuracy. The sequence layer adds parameters,
so this compares complete architectures rather than isolating alignment alone.
Training rejects impossible repeat alignments instead of silently zeroing loss.

The objective follows the [original CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
and uses [PyTorch CTCLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CTCLoss.html).
The decoder's exact scores are checked against exhaustive short paths and
PyTorch's loss. Ordered character marginals use a separate forward count-state
calculation with O(frames * output positions * alphabet size) work; they do not
depend on which candidate strings survive the beam. Full decoding remains slower
than independent-head ranking, and measured decoder time is reported separately.

One `TODO(refactor)` remains in `anpr/ctc_train.py`: split validation, path guards
and provenance writing are duplicated with the positional trainer. Their losses
stay explicit during the comparison; a shared runner should follow once the
training protocol is settled so those protections cannot drift.

Generic inference dispatches from checkpoint metadata. CTC checkpoints use format
two; existing positional checkpoints retain format one and their original
decoding semantics. Use `--beam-width` only for CTC. The original evaluator rejects
CTC weights rather than treating their frames as fixed character positions.

To use the measured CTC prototype on a standalone crop:

```bash
.venv/bin/python -m anpr.inference --checkpoint runs/ctc-comparison/ctc.pt --calibration runs/ctc-comparison/calibration.json --image plate-crop.jpg --top-k 50 --beam-width 100 --output runs/ctc-comparison/predictions.jsonl
```

## Recognize plate crops and detections

```bash
.venv/bin/python -m anpr.inference --checkpoint runs/rgb.pt --image plate-crop.jpg --top-k 10 --output runs/crop.jsonl
.venv/bin/python -m anpr.inference --checkpoint runs/rgb.pt --yolo-weights best.pt --source video.mp4 --top-k 10 --output runs/video.jsonl
```

YOLO integration crops `orig_img` using its pixel bounding boxes, converts BGR to
RGB, clips a small margin to image bounds, and retains source/frame/box metadata.
It streams videos and keeps detection confidence separate from character scores.
The default plate class is zero; use `--plate-class-id` for a multi-class detector.
The detector adapter has tests using representative Results objects. Running the
actual trained detector still requires its exported weights and ultralytics.

Each JSONL record includes `positions` with every character/absence alternative,
`candidates` with plate text and visual `log_score`, an uncertainty status, and
latency. Absence is JSON null in positional character alternatives. Positional
scores include trailing padding; CTC scores sum complete blank/repeat alignments.
Neither model renormalizes scores over the returned short list.

Apply a validation-derived temperature with `--temperature`, or explicitly
constrain a supported seven-character format using `--pattern LLDDLLL`.
`--known-plates plates.txt` scores one registration per nonempty line without
discarding alternatives that fell below a character cutoff. These constraints
can exclude the true plate, so unconstrained recall is the primary metric.
Direct lookup scoring costs O(database size * head count); generated candidates
use bounded prefix search rather than a Cartesian product. The decoder caps
the requested list at 1,000 candidates.

Standalone and detector inference also accept `--calibration` to verify and apply
the frozen sidecar instead of copying its temperature manually. For example:

```bash
.venv/bin/python -m anpr.inference --checkpoint runs/rgb.pt --calibration runs/rgb-calibration.json --image plate-crop.jpg --top-k 10
```

The same option is available as `PlateRecognizer.from_checkpoint(...,
calibration="runs/rgb-calibration.json")`. Per-call explicit temperature overrides
are identified as overrides in the output rather than presented as the frozen
validation setting.

From Python:

```python
from PIL import Image
from anpr.inference import PlateRecognizer

# Load once and reuse the evaluation-mode model across standalone crops.
recognizer = PlateRecognizer.from_checkpoint("runs/rgb.pt")
with Image.open("plate-crop.jpg") as plate_image:
    # Retain ranked alternatives rather than reducing the output to one argmax string.
    prediction = recognizer.recognize(plate_image.convert("RGB"), top_k=10)
```

Predictions are `uncertain` by default, even when the best candidate is strong.
`unreadable` marks empty, uniform or threshold-failing evidence while retaining
suggestions. The threshold is a heuristic, not an evaluated rejection policy.
Use `--min-position-confidence` to set an explicit floor and evaluate its false
rejection rate on your held-out captures before relying on it.

## Current limits

The [CTC experiment](reports/ctc-experiment.json) compares sequence alignment with
fixed heads on exactly the same broader dataset. On 107 original real crops,
the true plate appears in the top 50 for 62 crops, compared with one for the
fixed-head model. Synthetic defocus setting 1 retains 38 of 107; stronger added
blur remains poor. These conditions do not measure naturally blurred video.
The [real recall comparison](reports/ctc-real-recall.png) and raw reports record
all conditions. Epoch selection and temperature fitting use synthetic validation
only. This supports CTC for continued prototype work, while leaving the original
baseline intact. It does not establish production accuracy on the user's camera.

The [character example](reports/ctc-character-example.json) uses the first public
manifest row, without selecting a successful crop. It shows ordered character
probabilities and candidate ranking, including a case where the top prediction
is wrong. No character marginals are multiplied into a plate score.
The [ordered alternatives table](reports/ctc-character-alternatives.png) shows
the three highest model probabilities at each position for that same crop.

The controlled spatial experiment is recorded in
[reports/spatial-experiment.json](reports/spatial-experiment.json), with a
[synthetic versus real comparison](reports/spatial-comparison.png). It keeps the
original baseline, changes no trainable parameter count, and starts new variants
with matched shared initialization. Spatial features greatly improve synthetic
candidate recall. The matched U-Net does not improve the stronger spatial model
overall in this single run, so there is no evidence to make it the default.

All three fixed-head models in that earlier trial fail complete-plate recall on the public EU source.
The label audit shows that every target plate needs at least one class with no
positive training examples at the corresponding head. The single UK-style
synthetic format is therefore a concrete coverage problem, alongside font and
camera differences. New evaluation reports include this positional coverage
diagnostic. It does not exclude hard examples or replace recognition metrics.
The public source is now an exploratory regression benchmark; future work using
its feedback must not describe it as an untouched final holdout.

The [broader-label study](reports/coverage-experiment.json) subsequently used
1,000 synthetic identities, lengths one through seven and three fonts. It closes
the positive positional-class gaps, but still misses almost every complete real
plate. The selected checkpoint is epoch nine, chosen by synthetic validation
loss rather than real results. This study changes sample count, fonts and label
distribution together, so it does not isolate one cause. Its
[real recall chart](reports/coverage-real-recall.png) records the failure rather
than treating wider vocabulary support as useful recognition accuracy. Real
camera training data, variable-length alignment and actual YOLO crop evaluation
remain necessary next steps.

The initial local experiment trained both variants for 15 epochs on 300 synthetic
identities. The paired test contains 90 crops from 45 held-out identities, one
layout and one font. The mask variant returned the true plate more often in this
run, but both models missed most true plates even at 50 candidates. This is not
useful recognition performance and does not establish a general U-Net benefit.
Head initialization also differs between variants, so repeated matched runs are
needed to isolate the mask contribution.

Measured outputs and the protocol are in [reports/synthetic-experiment.json](reports/synthetic-experiment.json),
with a [candidate-recall chart](reports/synthetic-candidate-recall.png).
Raw per-condition scores are in [RGB results](reports/synthetic-rgb.json) and
[RGB plus mask results](reports/synthetic-rgb-mask.json). They include calibration
and latency measurements, not a claim of production accuracy. The next evidence
gate is labelled real Roboflow crops and the exported trained detector.

Independent heads do not explicitly model character dependencies. Global pooling
also loses explicit left-to-right feature layout. Keep this baseline to establish
whether ranked candidates help before changing the architecture. A spatial
sequence recognizer such as a CNN with CTC or PARSeq should be adopted based on
matched plate-domain measurements, not general OCR benchmark claims.

The original notebook uses synthetic-only data, random row splitting, clean
validation images and per-character argmax metrics. These cannot establish
blurred-plate candidate recall and may leak white/yellow variants of one plate
across splits. New code must not carry those evaluation assumptions forward.

The globally pooled baseline is retained for comparison with the new spatial
layout. Completing that experiment replaces the earlier global-pooling refactor
TODO; there is no outstanding TODO for work already implemented. Spatial heads
still use a fixed positional layout and do not solve variable-length alignment
as CTC would. No claim of useful blurred-plate accuracy follows from CPU smoke
training, synthetic-only results or passing tests.

Training, evaluation and inference reject output paths that alias declared input
assets, including resolved symlinks and hard links. Checkpoints, masks and crop
files referenced by training/evaluation manifests are protected before writes.
Distinct existing report destinations remain valid for reruns. These checks
prevent mistaken paths; they do not provide concurrent filesystem isolation.
