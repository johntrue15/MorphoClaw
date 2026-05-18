---
title: CLI reference
description: Standalone command-line entry points for AutoResearchClaw — agent, paint loop, comparison harness, and iterative training.
---

# CLI reference

Most workflows can be exercised directly from a shell. All commands assume
you have run `pip install -r requirements.txt` and exported the keys from
[**Secrets & Env**](secrets.md).

## Research agent

```bash
cd .github/scripts
python research_agent.py "Your research topic" \
    --research-depth 10 \
    --github-issues 1 \
    --media-list 000656244
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--research-depth N` | `10` | Inner-loop cycles |
| `--github-issues N` | `3` | Outer-loop reports |
| `--media-id ID` | &mdash; | Seed a single MorphoSource media |
| `--media-list ID` | &mdash; | Seed from a curated media list |
| `--continue-from N` | &mdash; | Resume from a tracking issue |

## Dashboard

```bash
cd .github/scripts
python dashboard.py
# Open http://localhost:5001
```

## MorphoSource search (no agent)

```bash
cd .github/scripts
python morphosource_api.py "query string" --limit 10
```

## nnInteractive paint loop

```bash
source ~/.autoresearchclaw/nninteractive/bin/activate

python .github/scripts/nninteractive_loop.py \
    --input /path/to/volume.nii.gz \
    --goal  "Segment the heart" \
    --media-id 000656244 \
    --output-dir /tmp/nni_loop \
    --max-steps 10
```

Or via `slicer_tool` (downloads from MorphoSource first):

```bash
python .github/scripts/slicer_tool.py 000769445 \
    --nninteractive --nni-goal "Segment the cranial cavity"
```

## nnInteractive vs MorphoSource GT

```bash
# Discover candidate (CT, mesh) pairs
python .github/scripts/find_segmentation_pairs.py \
    --query "skull mesh" --max-pairs 5 \
    --output candidate_pairs.json

# Run the comparison with explicit IDs
python .github/scripts/nninteractive_compare.py \
    --ct-media-id 000656244 \
    --gt-media-id 000656245 \
    --goal "Segment the cranial bone" \
    --output-dir /tmp/nni_compare
```

Quick smoke-test preset (chameleon stapes &mdash; ~600 KB GT mesh, fast):

```bash
.github/scripts/test_chameleon_stapes.sh
# or
python .github/scripts/nninteractive_compare.py \
    --preset chameleon_stapes \
    --output-dir /tmp/nni_compare_stapes
```

## Iterative segmentation training

```bash
# Discover specimens
python -m metadata_to_morphsource.seg_train discover \
    --query "primate skull mesh" --max-pairs 12 \
    --output runs/skull_v1/specimens.json

# Round 0 (no student yet)
python -m metadata_to_morphsource.seg_train round \
    --specimens runs/skull_v1/specimens.json \
    --run-dir   runs/skull_v1 \
    --paper-tag chameleon_skull_v1 \
    --goal      "Segment the cranial bone"

# Round 1+ (use last round's student)
python -m metadata_to_morphsource.seg_train round \
    --specimens runs/skull_v1/specimens.json \
    --run-dir   runs/skull_v1 \
    --paper-tag chameleon_skull_v1 \
    --goal      "Segment the cranial bone" \
    --student   runs/skull_v1/round_000/student_weights/student_r000.artifact.json

# Paper export at any time
python -m metadata_to_morphsource.seg_train export \
    --run-dir runs/skull_v1 --paper-tag chameleon_skull_v1 \
    --output  runs/skull_v1/paper

# Live download / DICOM → NIfTI / crop / voxelise for any (CT, mesh) pair
python -m metadata_to_morphsource.seg_train prepare \
    --preset chameleon_stapes \
    --prep-dir runs/stapes_live/prep \
    --output  runs/stapes_live/specimens.json
```

## Tests

```bash
make test-seg-train         # Tiers 1 + 2 — pure Python + CLI surface
make test-seg-train-full    # adds numpy-marked unit tests
Tests/smoke_seg_train.sh --integration  # Tier 3 — synthetic NIfTI
make test-seg-train-live    # Tier 4 — real downloads + real Slicer + real nnInteractive
```

For the full ladder and what each tier exercises, see
[**Iterative Segmentation**](../ITERATIVE_SEGMENTATION.md#testing-the-pipeline).
