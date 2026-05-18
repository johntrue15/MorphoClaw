# Iterative Segmentation Training

This document describes the methodology behind the
`metadata_to_morphsource.seg_train` package: an iterative pipeline that
trains a custom 3D segmentation **student** model on top of nnInteractive,
graduates it to autonomous operation when validation Dice clears a
threshold, and records every segmentation event in a publication-grade
ledger so the result can be compared head-to-head with human
segmentation.

## Goal

> "Use nnInteractive (and other pretrained models) to bootstrap a
> dataset-specific segmentation model that, after a small number of
> iterations, can segment new MorphoSource specimens *without* a
> human-curated ground truth — and prove it with a fully reproducible
> data trail."

## Conceptual flow

```text
┌────────────────────────────────────────────────────────────────────┐
│ Round 0 (no student yet)                                           │
│   For each MorphoSource (CT, mesh) pair:                           │
│     1. Voxelize curated mesh onto CT grid           → human GT     │
│     2. nnInteractive paint loop (LLM-driven)        → pseudo-label │
│     3. Compare pseudo-label vs human GT             → Dice/IoU/etc │
│     4. Append every event to ExperimentLedger                      │
│   Train Student v0 on accumulated (CT, pseudo-label) pairs         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Round 1..N                                                         │
│   For each new specimen:                                           │
│     1. Student inference (sliding-window + MC dropout)             │
│     2. ConfidenceRouter decides:                                   │
│        ─ ACCEPT_STUDENT  → keep student mask                       │
│        ─ CORRECT_W/_NNI  → run nnInteractive paint loop            │
│        ─ REJECT          → fall back to nnInteractive from scratch │
│     3. Append to manifest with reliability weight per source       │
│   Re-train Student v(n) on the new manifest                        │
│   If val Dice ≥ graduation_threshold for 2 rounds in a row →       │
│     graduate: subsequent rounds skip nnInteractive (still log it). │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ Paper export                                                       │
│   ledger.jsonl + ledger.csv → paper_summary.md, mode_summary.csv,  │
│   round_progression.csv, vs_human.csv, dice_progression.png, …     │
└────────────────────────────────────────────────────────────────────┘
```

## Components

| Module | Responsibility |
|--------|----------------|
| `experiment_ledger.py`    | Append-only JSONL + CSV log of every segmentation event (the paper's source of truth). |
| `dataset.py`              | Manifest of (volume, label) samples + reliability weights + train / val / holdout splits keyed on physical specimen. |
| `pseudo_label_generator.py` | Wrappers that shell out to the existing nnInteractive paint loop and the headless mesh voxelizer. |
| `student_model.py`        | MONAI 3D U-Net trainer + sliding-window inference + Monte-Carlo dropout uncertainty. |
| `confidence_router.py`    | Auditable decision policy: accept student, correct with nnInteractive, or reject. |
| `iterative_trainer.py`    | Round orchestrator (per-specimen processing, training, graduation). |
| `paper_export.py`         | CSVs, plots, and a Markdown summary derived from the ledger. |
| `cli.py` / `__main__.py`  | `python -m metadata_to_morphsource.seg_train discover|round|export|summary` |

## Reliability weights

`dataset.py` assigns a reliability weight ∈ (0, 1] to each sample. The
weight is multiplied into the training loss per sample so curated meshes
dominate even when pseudo-labels are abundant.

| Mode | Default reliability |
|------|---------------------:|
| `human_gt` / `expert_annotation` | 1.00 |
| `morphosource_mesh` (voxelized) | 0.98 |
| `nninteractive_human_confirmed` | 0.95 |
| `student_corrected` | 0.85 |
| `ensemble` | 0.80 |
| `nninteractive` | 0.75 |
| `monai_auto` | 0.70 |
| `student` | 0.60 |

Override per-run via `DatasetManifest.reliability_overrides`.

## Splitting

Splits are deterministic and keyed on `physical_object_id`, so two media
of the same specimen never straddle splits — preventing Dice from being
inflated by within-specimen leakage.

- `human_gt` / `expert_annotation` samples always go into the **holdout**
  pool so validation never leaks the gold standard into training.
- All other samples are hashed into `train`/`val`/`holdout` according to
  configurable ratios (default 70/20/10).

## Confidence router

Three decisions, exposed thresholds:

```python
RouterPolicy(
    min_confidence=0.85,         # mean foreground softmax prob
    max_entropy=0.35,            # mean per-voxel binary entropy
    min_foreground_fraction=1e-5,
    max_foreground_fraction=0.6,
)
```

`router_threshold` and `routed_to_nninteractive` are written to every
ledger row, so the paper can publish ROC-style decision curves at
multiple thresholds without re-running anything.

## Graduation rule

After every round, the trainer checks the freshly trained student's
best validation Dice against `graduation_dice_threshold` (default 0.85).
Two consecutive passes flip `graduated = True`, after which the router
unconditionally accepts the student. nnInteractive is still invoked for
a randomised audit fraction (configurable via `paper_tag`-specific
hooks) so the paper can keep reporting head-to-head agreement.

## Reproducibility

Each ledger row carries:

- `git_sha` (from `$GITHUB_SHA` or `$GIT_SHA`),
- `python_version`, `platform`, `host`,
- `run_id`, `round_index`, `paper_tag`,
- the exact `model_version` and `device` used,
- the input `volume_path` (specimens are staged under
  `runs/<name>/downloads/`).

Combined with the per-round `student_weights/<version>.config.json`,
this is sufficient to re-run any episode bit-for-bit.

## Paper artefacts

`python -m metadata_to_morphsource.seg_train export --run-dir runs/skull_v1`
emits:

| File | What it contains |
|------|-------------------|
| `per_episode.csv` | One row per segmentation event (full ledger) |
| `mode_summary.csv` | Mean Dice, IoU, surface distance, runtime per mode |
| `round_progression.csv` | Student vs nnInteractive Dice per round + mode mix |
| `vs_human.csv` | Per-specimen comparison: human GT vs nnInteractive vs student |
| `dice_progression.png` | Mean Dice across rounds with graduation line |
| `mode_mix.png` | Stacked-bar of segmentation source per round |
| `time_per_mode.png` | Wall-clock time per mode |
| `paper_summary.md` | Human-readable summary that drives the GitHub-issue post |

These are the headline tables and figures for the publication's
"Methods" and "Results" sections.

## Operational guide

### One-time bootstrap

```bash
.github/scripts/install_nninteractive.sh
.github/scripts/install_seg_train_extras.sh
```

### Run a round locally

```bash
# 1. Discover specimens with curated meshes:
python -m metadata_to_morphsource.seg_train discover \
    --query "primate skull mesh" \
    --max-pairs 12 \
    --output runs/skull_v1/specimens.json

# 2. Round 0 (no student yet):
python -m metadata_to_morphsource.seg_train round \
    --specimens runs/skull_v1/specimens.json \
    --run-dir   runs/skull_v1 \
    --paper-tag chameleon_skull_v1 \
    --goal      "Segment the cranial bone"

# 3. Round 1+ uses the previous round's student:
python -m metadata_to_morphsource.seg_train round \
    --specimens runs/skull_v1/specimens.json \
    --run-dir   runs/skull_v1 \
    --paper-tag chameleon_skull_v1 \
    --goal      "Segment the cranial bone" \
    --student   runs/skull_v1/round_000/student_weights/student_r000.artifact.json

# 4. Export paper data after any round:
python -m metadata_to_morphsource.seg_train export \
    --run-dir runs/skull_v1 \
    --paper-tag chameleon_skull_v1 \
    --output  runs/skull_v1/paper
```

### Run on the GitHub Actions runner

Trigger **Actions → Iterative Segmentation Training → Run workflow**;
fill in `run_name`, `paper_tag`, `goal`. After Round 0 completes,
download the artifact, find the `*.artifact.json` path inside, and pass
it as `student_artifact` for Round 1.

## Testing the pipeline

The pipeline ships with a three-tier test ladder so you can verify the
plumbing **without** running a real iteration through nnInteractive,
3D Slicer, or MorphoSource downloads. One command runs the whole
ladder:

```bash
make test-seg-train          # tiers 1 + 2
make test-seg-train-full     # tiers 1 + 2 + numpy-marked unit tests
```

Or invoke the smoke script directly with finer-grained control:

```bash
# Default: pure-Python tiers only (~1s, no heavy deps required)
Tests/smoke_seg_train.sh

# Add the synthetic NIfTI integration round (needs numpy + SimpleITK)
Tests/smoke_seg_train.sh --integration

# Add the numpy-marked router unit tests (entropy / probability volume)
Tests/smoke_seg_train.sh --include-numpy

# Combine, point at a specific Python, and bump the watchdog timeout:
PYTHON=/path/to/venv/bin/python \
SEG_TRAIN_PROBE_TIMEOUT=180 \
Tests/smoke_seg_train.sh --integration --include-numpy
```

What each tier proves:

| Tier | Tests                                                                         | What it verifies                                                                                                                                                       |
| ---- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1   | `test_seg_train_{ledger,router,dataset,paper_export,cli}.py`                  | Pure-Python correctness of the ledger schema, router policy, dataset splitting, paper-export aggregations and Markdown output, and the CLI subparser surface.          |
|  2   | embedded in `smoke_seg_train.sh`                                              | The package is importable, every CLI subcommand prints `--help`, `summary` exits non-zero on missing run dirs, and `export --no-plots` writes all five paper artifacts. |
|  3   | `test_seg_train_integration.py`                                               | The full `IterativeTrainer.run_round` orchestration on a synthetic 32³ NIfTI volume, with `nnInteractive` and the Slicer voxelizer monkey-patched out.                  |
|  4   | `Tests/test_chameleon_stapes_iterative.sh` (live)                             | The same orchestration but on a **real** MorphoSource specimen — actual download, DICOM-to-NIfTI, Slicer/VTK voxelisation, real nnInteractive paint loop, real Dice.   |

Tier 3 is the most important one because it actually runs the
ledger → router → manifest → paper-export pipeline end-to-end. It's
opt-in locally because some macOS Anaconda installs ship a numpy
that segfaults on import; CI always runs it (see
`.github/workflows/seg_train_tests.yml`). To exercise the test
manually:

```bash
python -m pytest Tests/test_seg_train_integration.py -v
```

Beyond the smoke ladder, an end-to-end live test goes:

1. `bash .github/scripts/install_nninteractive.sh` and
   `bash .github/scripts/install_seg_train_extras.sh` — bootstrap MONAI
   on top of the nnInteractive venv.
2. `python -m metadata_to_morphsource.seg_train discover ...` — fetch a
   tiny (1–3 specimen) batch.
3. `python -m metadata_to_morphsource.seg_train round ...` — Round 0
   with nnInteractive only.
4. Inspect `runs/<name>/ledger/ledger.csv` and
   `runs/<name>/round_000/round_000_report.json`. Each row in the CSV
   should carry a `mode` (`human_gt` / `nninteractive`), a `dice` (when
   compared to the voxelized GT), and a non-zero `duration_s`.
5. `python -m metadata_to_morphsource.seg_train export --run-dir runs/<name> ...` —
   confirm `paper_summary.md`, `mode_summary.csv`, `vs_human.csv`, and
   `round_progression.csv` are produced.

### Tier 4 — live test on a real MorphoSource specimen

Tier 4 exercises the full pipeline against the canonical
**chameleon-stapes** pair (Chamaeleo calyptratus, `uf:herp:191369`,
right stapes). The right stapes is one of the smallest bones in the
body, so the whole loop fits in 5–15 minutes on a runner — long enough
to be a real test, short enough to keep OpenAI quota cheap. One
command:

```bash
make test-seg-train-live
# or, with knobs:
MAX_STEPS=8 STUDENT_EPOCHS=2 \
    Tests/test_chameleon_stapes_iterative.sh
```

What it actually does (every step is real; nothing is mocked):

1. **Pre-flight** checks for `MORPHOSOURCE_API_KEY`, `OPENAI_API_KEY`,
   the bootstrapped nnInteractive venv, MONAI, and (if requested)
   Slicer.
2. **`seg_train prepare --preset chameleon_stapes`** — shells out to
   `nninteractive_compare.py --skip-paint-loop` to download the CT
   (DICOM) and the GT mesh (PLY), convert DICOM to NIfTI via
   SimpleITK, crop the CT around the mesh + 4 mm margin, and
   voxelise the GT mesh onto the cropped grid (Slicer or pure-Python
   VTK). Writes a `specimens.json` pointing at `volume.nii.gz`,
   `gt_mesh.ply`, and `gt_voxelized.nii.gz`.
3. **`seg_train round`** — Round 0 with the real nnInteractive paint
   loop (an actual LLM call to GPT-4o), real
   `compare_labelmaps` metric computation against the voxelised GT,
   real ledger row writes, and (unless `SKIP_TRAINING=1`) a real
   1-epoch MONAI student training pass on the resulting
   `(CT, paint-loop label, voxelised GT)` tuple.
4. **Ledger sanity checks** — asserts the `human_gt` and
   `nninteractive` modes both appear, the nnInteractive episode has
   `has_ground_truth=True`, a Dice in `[0, 1]`, a positive
   `duration_s`, and at least one prompt was emitted.
5. **`seg_train export --no-plots`** — writes the five paper artefacts
   (`per_episode.csv`, `mode_summary.csv`, `round_progression.csv`,
   `vs_human.csv`, `paper_summary.md`) to `$RUN_DIR/paper/`.
6. **Optional Round 1** — if Round 0 produced a student artefact, runs
   one more round with `--student` and `--skip-training` to exercise
   the *student inference + ConfidenceRouter* path on real data and
   re-export the paper.

CI dispatch is via **Actions → Seg-Train Live Chameleon Stapes →
Run workflow** (workflow file:
`.github/workflows/seg_train_live_chameleon.yml`). The runner
auto-bootstraps both venvs, runs the same script as `make
test-seg-train-live`, posts the round report and `paper_summary.md`
to the workflow step summary, and uploads `ledger/`, `paper/`, the
round reports, the nnInteractive labelmap + summary, and the
voxelised GT as a workflow artefact.

### CI summary

| Workflow                                    | When it runs                                                | What it covers                                                |
| ------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| **Seg-Train Smoke Tests**                   | Auto on PR/push touching `seg_train/**`                     | Tiers 1 + 2 + 3 (synthetic, no API keys, no GPU)              |
| **Seg-Train Live Chameleon Stapes**         | Manual dispatch (self-hosted runner)                        | Tier 4 (real specimen, real Slicer/VTK + nnInteractive)       |
| **Iterative Segmentation Training**         | Manual dispatch (self-hosted runner)                        | Production multi-round training campaign                      |

### Privacy / human-in-the-loop reviews

The `mode = "nninteractive_human_confirmed"` reliability slot exists so
that any specimen reviewed by a domain expert (manually editing the
labelmap in 3D Slicer) can be promoted in the manifest:

```python
from metadata_to_morphsource.seg_train import DatasetManifest, SegmentationSample

m = DatasetManifest.load("runs/skull_v1/manifest.json")
m.add_sample(SegmentationSample(
    sample_id="000656244__r001__expert",
    media_id="000656244",
    physical_object_id="ms:1234",
    volume_path="…/volume.nii.gz",
    label_path="…/expert_review.nii.gz",
    mode="nninteractive_human_confirmed",
    has_human_review=True,
    notes="2 mins, JD reviewed in Slicer",
))
m.save()
```

This automatically routes the sample into the holdout pool so it never
leaks into training but is used to compute `vs_human.csv`.

## Limitations & open work

1. **Class scope** — the student is currently a single-class binary
   segmenter. Extending to multi-class needs a per-class reliability
   propagation rule.
2. **No active learning over the search space** — `discover` simply
   returns the first viable pairs; a future round could prioritise
   specimens whose router score sits near the graduation boundary.
3. **MONAI / torch on Apple Silicon** — training works on MPS but the
   sliding-window inferer is sometimes slow on macOS Metal; production
   runs should use a Linux + CUDA runner.
4. **Citation** — the paper export currently does not include
   citation-ready BibTeX of the MorphoSource records. `citation_extractor.py`
   could be re-used here.

See the [project README](https://github.com/johntrue15/MorphoClaw/blob/main/README.md) for the wider AutoResearchClaw context.
