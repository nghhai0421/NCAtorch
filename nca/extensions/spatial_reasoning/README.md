# Spatial reasoning with Adaptive NCAs

Train and evaluate Neural Cellular Automata on 2D spatial reasoning puzzles,
as in *2D Spatial Reasoning with Adaptive Neural Cellular Automata*:

| Task | What the NCA does |
| --- | --- |
| **Sudoku** | Fills in a 9x9 Sudoku from its clues. |
| **MNIST Sudoku** | Solves the same puzzles drawn as handwritten digits. Uses a Sudoku model plus a separately trained image encoder/decoder. |
| **Maze** | Marks the shortest path between start and goal in a 30x30 maze. |
| **Color balance** | Turns a random black/white image into one with exactly 50% white pixels. |
| **ARC adaptation** | Predicts output colors and categorical grid sizes from padded input/output NPZ pairs. |

The proposed model, the *adaptive NCA* (aNCA), lets every cell learn where to
look: its perception uses deformable convolutions whose sampling positions
change with the cell's state. The grid is treated as a torus, like the surface
of a donut, so a cell at the left border can look at cells at the right
border. Fixed-kernel NCAs are included as baselines.

All commands run from the repository root.

## Framework integration

The host factories call explicit hooks from [registration.py](registration.py)
to add the following components to their registries. The host config schema
imports the extension settings and calls its validator directly.

| Component | Registered selectors |
| --- | --- |
| Perceptions | `anca_deform`, `anca_deform_v1`, `row_conv`, `column_conv` |
| Datasets | `sudoku`, `maze`, `color_balance`, `arc_pairs` |
| Trainer | `spatial_reasoning` |
| Settings | `EXTENSIONS.SPATIAL_REASONING`, validated by [config.py](config.py) |

These datasets automatically select the spatial trainer when `TRAINER_TYPE`
is omitted or null; explicitly choosing an incompatible trainer is rejected.
The trainer reuses the host rollout, sample pool, logging and checkpointing,
and supplies its own task loss and AdamW optimizer. Its
`USES_FRAMEWORK_LOSS = False` setting prevents construction of an unused host
loss from `TRAINING.LOSS_FN`.

See the general [extension guide](../../../docs/extensions_guide.md) for the
registration hooks and instructions for adding an independent extension.

## Quick start

```bash
uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/sudoku.yaml
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_sudoku --checkpoint train_log/<run>/ca_final.pt
```

Training writes to `train_log/<TRAIN_NAME>_<timestamp>/`. That folder holds
the checkpoints (`ca_final.pt`), the exact config used and sample images.
Source datasets are downloaded on first use through their data providers.
`DATASET.DATAROOT` controls the extension's local caches, defaulting to
`datasets/`; Hugging Face source downloads use its own cache.

## Training

| Task | Command |
| --- | --- |
| Sudoku | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/sudoku.yaml` |
| Maze | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/maze.yaml` |
| Color balance | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/color_balance.yaml` |
| ARC | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/arc_baseline.yaml` |

During training, the console (and W&B, if `LOGGING.WANDB: true`) shows the
loss terms every `LOG_INTERVAL` steps. To measure how well a model solves
the task, use the evaluation scripts below.

**Repeated runs:** change the seed, e.g. `-o SEED=43`. Color-balance training
data follows this seed, including when using DataLoader workers; repeat runs
with the same worker count to reproduce the data sequence. The paper reports four
Sudoku runs, five maze runs and three color-balance runs.

**Other models:** the paper configs train the aNCA by default. The baselines of
the paper's tables are listed in the config as commented `PERCEPTIONS` blocks:
3x3/5x5/7x7/9x9 convolution, row/column/box (R/C/B), dilated convolution and
DCNv1. Replace the active block with one of them.

**Any setting** can be overridden on the command line, e.g.
`-o TRAINING.STEPS=50000` or `-o EXTENSIONS.SPATIAL_REASONING.SUDOKU.DIFFICULTY=[0.5,0.8]`.

For CPU training with the Sudoku or maze example, also override
`-o TRAINING.MIXED_PRECISION=false` when using `--device cpu`: the deformable
perception currently does not support CPU autocast.

### ARC softmax baseline

[configs/arc_baseline.yaml](configs/arc_baseline.yaml) follows the maze config's
AdamW, warmup/stable/decay schedule, gradient clipping and 18–24 update rollouts.
It uses three 3x3 DCNv2 heads, 44 mutable channels, synchronous updates and
fresh seeds. Fixed input coordinates are condition channels. The complete
canvas evolves with unconstrained color and size logits.

Place the archives in `datasets/arc_generated/`, or set `DATASET.DATAROOT`.
The training input is NPZ. Each split contains exactly these five fields:

| Field (replace `split` with `train` or `test`) | Shape | Meaning |
| --- | --- | --- |
| `split_input`, `split_output` | `[N, 30, 30]` | Top-left-aligned, zero-padded color IDs |
| `split_input_size`, `split_output_size` | `[N, 2]` | Original height and width |
| `split_task_id` | `[N]` | Unique generated episode ID, shared across its pairs |

`synthetic_arc_train.npz` supplies demonstrations; `synthetic_arc_test.npz`
supplies queries from the same source episodes. These filenames describe
pair roles, not an unseen-task partition. Plain testing trains on demonstration
pairs and scores all query pairs with fixed weights. This workflow can keep
`ARC.HOLDOUT_TASK_IDS` and `ARC.EVAL_TASK_IDS` empty.

For the separate unseen-task adaptation experiment, set `ARC.HOLDOUT_TASK_IDS`
before pretraining, excluding all generated tasks belonging to the chosen
validation and test source families. `ARC.EVAL_TASK_IDS` selects a subset of
those held-out tasks. Resolve family membership from the source package when
making this partition; NPZ task IDs identify individual generated episodes.

The earlier archives used linked source IDs and blank IDs for some records.
Regenerate them with the corrected `scripts/arc-sin_preprocess.py` before
task-level adaptation. That script saves the five fields above, preserving the
top-level source `task_id` and omitting metadata. Existing archives remain
usable for per-pair training and plain full-test prediction. Preprocessing writes
the two archive filenames in the data directory when explicitly run.

#### Complete train-then-test experiment

The ARC baseline enables `ARC.TEST_AFTER_TRAIN`. The existing training command
first saves `ca_final.pt`, then scores **every** row of the test NPZ without
adaptation or task-ID filtering. The model receives only the seed and input
condition; targets are used for scoring and display. Test inference uses full
precision, `ARC.TEST_STEPS` updates (24), and `ARC.TEST_BATCH_SIZE` pairs (16).
Model weights are unchanged. Other datasets retain their original workflow.

The run folder contains:

```text
ca_final.pt
test/
    metrics.json
    predictions.npz
    imgs/step_rows_0_15.png
    imgs/step_rows_16_31.png
```

`metrics.json` records the checkpoint, source archive, pair count, rollout
length, and pair-averaged metrics. Accuracy is in percent; losses retain their
native scale. The final short batch is included and weighted by its pair count.

`predictions.npz` contains the original `test_input`, `test_output`, their sizes
and `test_task_id`, plus `test_row_index`, `test_prediction`,
`test_prediction_size`, and `test_exact_match`. Predicted grids are top-left
aligned and zero-padded; their saved sizes define the returned rectangle.
Archive row indices identify pairs even when legacy task IDs are blank or repeated.

PNG pages show input (`Seed`), prediction, and target for the first
`ARC.TEST_IMAGE_LIMIT` pairs (32 by default) in archive order. Every pair is
still predicted and scored. Set the limit to 0 to skip PNGs, or to the test-pair
count to render all pairs. Rendering every pair can create many files.

For an existing checkpoint, run plain testing separately:

```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.evaluate_arc --checkpoint train_log/<run>/ca_final.pt --mode test
```

The default mode is `test`. Exports go into a new timestamped directory beside
the checkpoint. Override with `--output-dir <new-directory>`, `--n-steps`,
`--batch-size`, `--image-limit`, and `--device`. Existing output directories are
protected from overwriting. A completed export has `metrics.json`; failed
exports may leave a partial image directory. If testing fails after training,
the final checkpoint remains available for a separate test run.

For training-only smoke runs, override
`-o EXTENSIONS.SPATIAL_REASONING.ARC.TEST_AFTER_TRAIN=false` so the short check
does not start a complete test-archive pass afterward.

#### Held-out task adaptation

Use the saved pretraining config and verified held-out IDs:

```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.evaluate_arc --mode adapt --checkpoint train_log/<run>/ca_final.pt --task-ids <heldout_task_id>
```

Each task restores the checkpoint weights, creates a fresh AdamW optimizer,
and adapts only on its demonstrations. Query answers are used for scoring
after prediction. The table reports before/after exact-grid, size, height,
width and true-rectangle color accuracy, averaged over query pairs. Adaptation
uses full precision, `ARC.ADAPT_STEPS` updates and `ARC.ADAPT_LR`; override these
with `--adapt-steps` and `--adapt-lr`. The rollout length is `--n-steps` (24 by
default). Adaptation settings can be fixed in advance; if tuning them, use
validation tasks. This mode requires the held-out partition recorded with
the checkpoint and reports metrics without the plain-test NPZ/PNG exports.

For an explanation from equations to tensors and hooks, see the workspace's
`anca_arc_hooks_implementation_guide.md`. CPU ARC training also requires
`--device cpu -o TRAINING.MIXED_PRECISION=false`.

### MNIST Sudoku

The Sudoku model itself is not retrained. Instead, an encoder reads the
handwritten clues into a Sudoku board, the Sudoku model solves it and a
decoder draws the result as handwriting. Train the three helper networks once:

```bash
M=nca.extensions.spatial_reasoning.scripts
C=nca/extensions/spatial_reasoning/configs/codec.yaml
# 1. digit classifier, used to train the decoder and to read the final image
uv run python -m $M.train_digit_classifier --out checkpoints/mnist_cnn.pt
# 2. encoder and 3. decoder
uv run python -m $M.train_codec --config $C -o EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=encoder
uv run python -m $M.train_codec --config $C -o EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=decoder
```

The encoder and decoder end up in `train_log/sudoku_codec_<mode>_<timestamp>/`.

## Evaluation (paper tables)

Each script prints a table. Without `--config`, the `config.yaml` next to the
checkpoint is used.

**Sudoku (Tab. 1):** solve rate per difficulty, 1000 puzzles each.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_sudoku --checkpoint train_log/<sudoku run>/ca_final.pt
```

**MNIST Sudoku (Tab. 2):** the Sudoku model together with the encoder and decoder.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_mnist_sudoku \
    --checkpoint train_log/<sudoku run>/ca_final.pt \
    --codec-config nca/extensions/spatial_reasoning/configs/codec.yaml \
    --encoder train_log/<encoder run>/encoder_final.pt \
    --decoder train_log/<decoder run>/decoder_final.pt
```

**Maze (Tab. 3 and 9):** the 1000 test mazes, plus 1000 new mazes of a
different style (Prim's algorithm) that the model has never seen.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_maze --checkpoint train_log/<maze run>/ca_final.pt
```

**Color balance (Tab. 4):** 1000 random start images.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_color_balance --checkpoint train_log/<color balance run>/ca_final.pt
```

Common options: `--n-steps` (rollout length; the defaults are the paper's
1500 / 100 / 200 steps), `--batch-size`, `--device`.

### What the numbers mean

| Task | Metric | Meaning |
| --- | --- | --- |
| Sudoku | **ASR** | % of valid completed boards after restoring the original clues. Puzzles can have several solutions, and any valid completion counts. |
| Sudoku | **ACP** | % of filled-in cells that break no row, column or box rule. |
| Sudoku | *ASR(exact)*, *ACP(exact)* | The same, but compared against the one stored solution. |
| Sudoku | Buckets | **Easy**, **Medium**, **Hard**, and **Full** use the removal-fraction ranges in `scripts/eval_sudoku.py`; removed-cell counts are rounded down. |
| Maze | **Valid Path** | % of mazes where the marked path connects start and goal without crossing walls. |
| Maze | **Valid Optimal** | % where that path is also as short as possible. |
| Color balance | **Imbalance** | Average number of pixels away from an exact 50/50 split. |
| Color balance | **Accuracy** | % of images with exactly 50/50. |

For Sudoku, evaluation takes predictions only at originally empty cells and
restores the supplied clues before checking validity. It therefore measures
completion of the missing cells, without separately measuring whether the raw
model output preserved every clue. This postprocessing does not supply the
missing solution digits to the model.

## Configuration reference

Task settings live under `EXTENSIONS.SPATIAL_REASONING`. Model, learning-rate
and rollout settings remain in the usual host sections (`MODEL`, `TRAINING`,
`PATTERN_POOL`). For example, the task selection in a Sudoku config is:

```yaml
DATASET:
  NAME: sudoku
TRAINING:
  TRAINER_TYPE: spatial_reasoning  # Optional: inferred from the dataset.
EXTENSIONS:
  SPATIAL_REASONING:
    LOSS: sudoku_constraint
    WEIGHT_DECAY: 0.0
    SUDOKU:
      DIFFICULTY: [0.01, 0.81]
```

This is a fragment; use [configs/sudoku.yaml](configs/sudoku.yaml) for a complete
model and training configuration. The extension section supplies settings;
`DATASET.NAME` and the model selectors determine which components are used.

The spatial trainer requires `LOSS` to match the selected dataset. The codec
uses the separate `CODEC` settings and its own training entry point.
Unknown settings and unsupported task combinations are validated when the
configuration is loaded. All settings below are relative to
`EXTENSIONS.SPATIAL_REASONING`.

| Setting | Meaning |
| --- | --- |
| `LOSS` | Training objective: `sudoku_constraint`, `maze_path`, `color_balance` or `arc_supervised` (must match `DATASET.NAME`) |
| `WEIGHT_DECAY` | AdamW weight decay |
| `CHECK_BACKPROP` | Optional smoke-run check: finite loss, present/finite gradients, nonzero total gradient norm, and actual weight updates |
| `SUDOKU.DIFFICULTY` | Fraction of cells removed from training puzzles, a value or a `[min, max]` range |
| `SUDOKU.*_WEIGHT`, `SOFTMAX_TEMPERATURE` | Loss terms: Sudoku rules, given clues, decisiveness |
| `MAZE.HF_DATASET_ID` | Maze dataset on Hugging Face |
| `MAZE.*_WEIGHT` | Loss terms: path overlap, binary decisions, path length |
| `COLOR_BALANCE.BIAS_MIN/MAX` | Range of the white share of start images |
| `COLOR_BALANCE.WHITE_RATIO` | Target white share (0.5 = 50/50) |
| `COLOR_BALANCE.*_WEIGHT` | Loss terms: global balance, pure black/white pixels |
| `ARC.TRAIN_FILE`, `TEST_FILE` | Demonstration/query NPZ filenames relative to `DATASET.DATAROOT` |
| `ARC.TEST_AFTER_TRAIN` | Automatically score/export the entire test archive after the final checkpoint; enabled in the ARC baseline |
| `ARC.TEST_STEPS`, `TEST_BATCH_SIZE` | Fixed plain-test rollout length and inference batch size |
| `ARC.TEST_IMAGE_LIMIT` | Number of leading test pairs rendered as PNGs; 0 disables images, without limiting prediction/scoring |
| `ARC.HOLDOUT_TASK_IDS`, `EVAL_TASK_IDS` | Tasks excluded from pretraining and the held-out subset to evaluate |
| `ARC.SIZE_LOSS_WEIGHT` | Weight on the sum of height and width cross-entropy |
| `ARC.ADAPT_STEPS`, `ADAPT_LR` | Per-task demonstration updates and learning rate |
| `CODEC.*` | Encoder/decoder training (only in `codec.yaml`) |

Set `DATASET.DATAROOT` to relocate parsed maze data and generated Sudoku test
puzzles from `datasets/`. The image size for color balance is
`DATASET.TARGET_SIZE`.

### Accessing settings from Python

Python code should access the validated settings through the extension helper:

```python
from nca.extensions.spatial_reasoning.config import settings

task_config = settings(config)
loss_name = task_config.LOSS
```

For an explicitly supplied section, `config.EXTENSIONS.SPATIAL_REASONING`
returns the same typed settings. `settings(config)` returns default settings when the
section is omitted; those defaults do not select a task loss.
