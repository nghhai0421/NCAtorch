"""``DATASET_REGISTRY`` constructors: ``(config, train) -> (dataset, cond_dim, height, width)``."""

from pathlib import Path

import numpy as np

from ..config import settings


def dataroot(config) -> Path:
    return Path(config.DATASET.DATAROOT or "datasets")


def create_sudoku(config, train):
    from .sudoku import GRID_SIZE, SudokuDataset

    sudoku = settings(config).SUDOKU
    dataset = SudokuDataset(
        channel_n=config.MODEL.CHANNEL_N,
        difficulty=sudoku.DIFFICULTY,
        total_samples=sudoku.TRAIN_SAMPLES if train else 1000,
        train=train,
        seed=config.SEED,
        dataroot=dataroot(config),
    )
    return dataset, 1, GRID_SIZE, GRID_SIZE


def create_maze(config, train):
    from .maze import MazeDataset

    dataset = MazeDataset(
        channel_n=config.MODEL.CHANNEL_N,
        train=train,
        hf_dataset_id=settings(config).MAZE.HF_DATASET_ID,
        dataroot=dataroot(config),
    )
    return dataset, 2, dataset.grid_size, dataset.grid_size


def create_color_balance(config, train):
    from .color_balance import ColorBalanceDataset

    balance, size = settings(config).COLOR_BALANCE, config.DATASET.TARGET_SIZE
    dataset = ColorBalanceDataset(
        channel_n=config.MODEL.CHANNEL_N,
        size=size,
        train=train,
        bias_min=balance.BIAS_MIN,
        bias_max=balance.BIAS_MAX,
        white_ratio=balance.WHITE_RATIO,
    )
    return dataset, 0, size, size


def create_arc(config, train):
    """Load demonstration or query pairs, applying the configured task selection."""
    from .arc import ArcDataset

    arc = settings(config).ARC
    split = "train" if train else "test"
    filename = arc.TRAIN_FILE if train else arc.TEST_FILE
    dataset = ArcDataset(dataroot(config) / filename, split, config.MODEL.CHANNEL_N)

    if train and arc.HOLDOUT_TASK_IDS:
        if not np.isin(arc.HOLDOUT_TASK_IDS, dataset.task_ids).all():
            raise ValueError("Some held-out ARC task IDs are absent from the demonstration archive.")
        dataset.rows = np.flatnonzero(~np.isin(dataset.task_ids, arc.HOLDOUT_TASK_IDS))
    elif not train and arc.EVAL_TASK_IDS:
        if not np.isin(arc.EVAL_TASK_IDS, dataset.task_ids).all():
            raise ValueError("Some ARC evaluation task IDs are absent from the query archive.")
        dataset.rows = np.flatnonzero(np.isin(dataset.task_ids, arc.EVAL_TASK_IDS))

    if len(dataset) == 0:
        raise ValueError("The ARC task selection contains no pairs.")
    return dataset, 13, dataset.grid_size, dataset.grid_size
