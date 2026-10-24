# Dataset schemas

All files use UTF-8 JSON Lines. Every grid is a non-empty rectangular list of
integer color values from 0 through 9.

## Official ARC task

Files: data/arc_train.jsonl and data/arc_validate.jsonl.

Required fields: task_id (string), source (string), task_fingerprint (SHA-256
string), train (non-empty input/output pair list), and test (non-empty
input/output pair list). The task fingerprint hashes train and test pair
multisets separately, so it is stable when demonstration ordering changes.

## Google generated pair

File: data/google_cleaned.jsonl.

Required fields: record_type equal to googleds_generated_pair, source equal to
googleds, googleds_task_id, linked_arc_task_id, pair_index_in_source_file,
pair_fingerprint, input, and output. This is pair-level auxiliary data, not an
official ARC task.

## Synthetic ARC-style task

File: data/synthetic_arc_train.jsonl.

Required fields: task_id, source (google_aux or barc_aux), task_fingerprint,
train, test, and metadata. Train/test have the same episode shape as official
ARC tasks. Metadata is provenance-only: it must not enter model prompts, model
targets, or evaluation inputs. No executable BARC source code is included in
these model-facing synthetic task records.
