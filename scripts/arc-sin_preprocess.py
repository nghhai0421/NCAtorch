"""Convert synthetic ARC episodes to demonstration and query NPZ archives.

Read the source JSONL and write to ``datasets/arc_generated/``. Each archive
contains padded input/output grids, their original (height, width), and the
generated task ID shared by all demonstrations and queries of one episode.
Grids are top-left aligned on a 30 by 30 canvas with zero padding.
"""

import json
from pathlib import Path

import numpy as np


def main():
    # Settings
    project_dir = Path(__file__).resolve().parents[1]
    source_path = project_dir / "arc-synthetic-training-package-v1/data/synthetic_arc_train.jsonl"
    output_dir = project_dir / "datasets/arc_generated"
    grid_size = 30
    splits = ("train", "test")

    # Count Pairs
    pair_counts = dict.fromkeys(splits, 0)
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            task = json.loads(line)
            for split in splits:
                pair_counts[split] += len(task[split])

    # Prepare Arrays
    data = {}
    saved_counts = dict.fromkeys(splits, 0)
    skipped_counts = dict.fromkeys(splits, 0)
    for split in splits:
        count = pair_counts[split]
        for name in ("input", "output"):
            data[f"{split}_{name}"] = np.zeros(
                (count, grid_size, grid_size), dtype=np.uint8
            )
            # Sizes are (height, width), before padding.
            data[f"{split}_{name}_size"] = np.zeros((count, 2), dtype=np.uint8)
        data[f"{split}_task_id"] = []

    # Process Data
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            task = json.loads(line)
            task_id = task["task_id"]

            for split in splits:
                for pair in task[split]:
                    # Skip the pair if either grid cannot fit; keep other pairs.
                    if any(
                        len(pair[name]) > grid_size
                        or any(len(row) > grid_size for row in pair[name])
                        for name in ("input", "output")
                    ):
                        skipped_counts[split] += 1
                        continue

                    index = saved_counts[split]
                    for name in ("input", "output"):
                        grid = np.asarray(pair[name])
                        if (
                            grid.ndim != 2
                            or not grid.size
                            or not np.issubdtype(grid.dtype, np.integer)
                            or np.any((grid < 0) | (grid > 9))
                        ):
                            raise ValueError(
                                f"Invalid {name} grid in {split} task {task['task_id']}"
                            )
                        height, width = grid.shape
                        data[f"{split}_{name}"][index, :height, :width] = grid
                        data[f"{split}_{name}_size"][index] = (height, width)

                    data[f"{split}_task_id"].append(task_id)
                    saved_counts[split] += 1

    # Save Result
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        count = saved_counts[split]
        for name in ("input", "output", "input_size", "output_size"):
            data[f"{split}_{name}"] = data[f"{split}_{name}"][:count]
        data[f"{split}_task_id"] = np.asarray(data[f"{split}_task_id"], dtype=str)

        split_data = {
            key: value for key, value in data.items() if key.startswith(f"{split}_")
        }
        output_path = output_dir / f"synthetic_arc_{split}.npz"
        np.savez_compressed(output_path, **split_data)
        print(f"Saved {output_path}")
        print(
            f"{split}: saved {saved_counts[split]:,} pairs, "
            f"skipped {skipped_counts[split]:,} oversized pairs"
        )


if __name__ == "__main__":
    main()
