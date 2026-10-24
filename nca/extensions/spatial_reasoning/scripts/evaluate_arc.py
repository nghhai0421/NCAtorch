"""Export full-test ARC predictions, or evaluate per-task adaptation.

The default test mode scores every query pair without changing model weights.
Adaptation is a separate ``--mode adapt`` workflow for held-out tasks.

    python -m nca.extensions.spatial_reasoning.scripts.evaluate_arc --checkpoint train_log/<run>/ca_final.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..data.arc import ArcDataset
from ..data.factories import dataroot
from ..evaluation import evaluate_arc, evaluate_arc_test
from .common import load_run, print_table, seed_everything


def main():
    # Load Settings and Model
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("test", "adapt"), default="test")
    parser.add_argument("--config", default=None, help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-steps", type=int, default=None, help="Test rollout length (ARC.TEST_STEPS by default).")
    parser.add_argument("--batch-size", type=int, default=None, help="Test batch size; defaults to ARC.TEST_BATCH_SIZE.")
    parser.add_argument("--image-limit", type=int, default=None, help="Pairs rendered as PNGs; 0 disables images.")
    parser.add_argument("--output-dir", default=None, help="New directory for test exports; existing directories are protected.")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Defaults to ARC.EVAL_TASK_IDS.")
    parser.add_argument("--adapt-steps", type=int, default=None, help="Defaults to ARC.ADAPT_STEPS.")
    parser.add_argument("--adapt-lr", type=float, default=None, help="Defaults to ARC.ADAPT_LR.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    config, model = load_run(args, cond_dim=13)
    arc = settings(config).ARC
    if config.DATASET.NAME != "arc_pairs":
        parser.error("Choose an ARC run configuration.")

    # Predict the Complete Test Archive
    if args.mode == "test":
        if args.task_ids is not None or args.adapt_steps is not None or args.adapt_lr is not None:
            parser.error("Task selection and adaptation overrides require --mode adapt.")
        checkpoint = Path(args.checkpoint)
        output_dir = args.output_dir or checkpoint.parent / (
            f"test_{checkpoint.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        evaluate_arc_test(
            model, config, output_dir, checkpoint=checkpoint,
            n_steps=args.n_steps, batch_size=args.batch_size, image_limit=args.image_limit,
        )
        return

    if args.batch_size is not None or args.image_limit is not None or args.output_dir is not None:
        parser.error("Batch size, image limit, and prediction exports apply to --mode test.")
    n_steps = arc.TEST_STEPS if args.n_steps is None else args.n_steps
    task_ids = arc.EVAL_TASK_IDS if args.task_ids is None else args.task_ids

    # Create Datasets
    demos = ArcDataset(dataroot(config) / arc.TRAIN_FILE, "train", config.MODEL.CHANNEL_N)
    queries = ArcDataset(dataroot(config) / arc.TEST_FILE, "test", config.MODEL.CHANNEL_N)

    # Adapt and Score Held-Out Tasks
    scores = evaluate_arc(
        model, demos, queries, config, task_ids, n_steps,
        adapt_steps=args.adapt_steps, adapt_lr=args.adapt_lr,
    )
    metric_names = ("exact_accuracy", "size_accuracy", "height_accuracy", "width_accuracy", "color_accuracy")
    rows = [
        (phase, *(f"{scores[phase][name]:.2f}" for name in metric_names))
        for phase in ("before", "after")
    ]

    # Report Results
    print_table(
        f"{args.checkpoint}: {scores['tasks']} tasks, {scores['queries']} queries ({n_steps} updates)",
        ("Adaptation", "Exact %", "Size %", "Height %", "Width %", "Color % (true rectangle)"),
        rows,
    )


if __name__ == "__main__":
    main()
