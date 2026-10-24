# ARC synthetic-training package v1

This local research archive contains the finalized official ARC train/validation
split and two generated-data views for experiments in ARC-style rule inference.
It is for local transfer only. Check source-dataset terms before any publication
or redistribution.

## Extract and verify

    unzip arc-synthetic-training-package-v1.zip
    cd arc-synthetic-training-package-v1
    shasum -a 256 -c manifests/checksums.sha256

## Included datasets

| File | Records | Role |
| --- | ---: | --- |
| data/arc_train.jsonl | 1,008 tasks | Official ARC training tasks: ARC-2 primary plus 8 ARC-1-only tasks. |
| data/arc_validate.jsonl | 138 tasks | Protected official evaluation tasks: 120 ARC-2 plus 18 ARC-1-only tasks. |
| data/google_cleaned.jsonl | 99,532 pairs | Clean generated input/output pairs associated with ARC training tasks. |
| data/synthetic_arc_train.jsonl | 199,899 tasks | ARC-shaped generated episodes: 12,673 Google-derived and 187,226 BARC-derived. |

ARC_Validate is evaluation-only. Do not use it for model fitting, prompt
selection, synthetic-data construction, or hyperparameter selection.

Raw ARC, GoogleDS, and BARC downloads, BARC_cleaned, temporary SQLite indexes,
audit intermediates, and Jupyter checkpoints are intentionally excluded.
Synthetic metadata is provenance-only, not model-prompt content.

manifests/checksums.sha256 covers every archived file except itself.
