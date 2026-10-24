"""Paper metrics and rollout helpers shared by trainers and evaluation scripts."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.sudoku import GRID_SIZE, sudoku_violations


@torch.no_grad()
def rollout(model, state, cond, n_steps):
    """Apply ``n_steps`` CA updates; the host model returns ``(state, dx)``."""
    for _ in range(n_steps):
        state, _ = model(state, cond)
    return state


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def sudoku_metrics(pred_digits, target_digits, given_mask):
    """Per-puzzle Sudoku metrics; digits are ``(B, 9, 9)`` in 0..8.

    Clue cells are taken from the target, so only the solved cells are judged.
        solved:  every non-clue cell equals the dataset solution (exact match)
        valid:   the completed grid is a valid Sudoku (ASR in the paper)
        acp:     % of non-clue cells equal to the dataset solution
        acp_valid: % of non-clue cells without a row/column/box conflict (ACP in the paper)

    Puzzles without empty cells count as solved and are left out of the ACP means.
    """
    to_solve = given_mask < 0.5
    grid = torch.where(to_solve, pred_digits, target_digits)
    conflict = sudoku_violations(grid)
    n_to_solve = to_solve.sum(dim=(1, 2)).float()
    correct = ((pred_digits == target_digits) & to_solve).sum(dim=(1, 2)).float()
    no_conflict = (to_solve & ~conflict).sum(dim=(1, 2)).float()
    return {
        "solved": (correct == n_to_solve).float(),
        "valid": (~conflict.flatten(1).any(dim=1)).float(),
        "acp": 100.0 * correct / n_to_solve,  # NaN for puzzles without empty cells
        "acp_valid": 100.0 * no_conflict / n_to_solve,
    }


def summarize(per_sample: dict) -> dict:
    """Concatenate per-batch metric tensors and average (ignoring NaN); rates in percent."""
    out = {}
    for key, values in per_sample.items():
        mean = torch.cat(values).float().nanmean().item()
        out[key] = mean * 100.0 if key in ("solved", "valid", "valid_path", "valid_optimal", "exact") else mean
    return out


@torch.no_grad()
def evaluate_sudoku(model, dataset, n_puzzles, n_steps, batch_size, device):
    """ASR/ACP of the array NCA on the first ``n_puzzles`` of ``dataset``."""
    results = {}
    for start in range(0, n_puzzles, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, n_puzzles))]
        seed, cond, target = (torch.stack(parts).to(device) for parts in zip(*batch))
        state = rollout(model, seed, cond, n_steps)
        metrics = sudoku_metrics(state[:, :GRID_SIZE].argmax(dim=1), target.argmax(dim=1), cond[:, 0])
        for key, value in metrics.items():
            results.setdefault(key, []).append(value.cpu())
    return summarize(results)


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

def _connected(path, start, goal):
    """4-connectivity of ``start`` and ``goal`` through the boolean ``path`` grid."""
    if not (path[start] and path[goal]):
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nxt[0] < path.shape[0] and 0 <= nxt[1] < path.shape[1] and path[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def maze_metrics(state, cond, target):
    """Per-maze ``valid_path`` (start and goal connected through predicted
    non-wall path cells) and ``valid_optimal`` (additionally the shortest-path length)."""
    free = (cond[:, 0] <= 0.5).cpu()
    pred = (state[:, 0].clamp(0, 1) > 0.5).cpu() & free
    true = (target[:, 0] > 0.5).cpu() & free
    endpoints = cond[:, 1].cpu()
    valid, optimal = [], []
    for b in range(pred.shape[0]):
        starts, goals = (endpoints[b] > 0.5).nonzero(), (endpoints[b] < -0.5).nonzero()
        # A maze without both endpoints cannot be solved.
        ok = len(starts) > 0 and len(goals) > 0 and _connected(
            pred[b].numpy(), tuple(starts[0].tolist()), tuple(goals[0].tolist()),
        )
        valid.append(ok)
        optimal.append(ok and int(pred[b].sum()) == int(true[b].sum()))
    return {"valid_path": torch.tensor(valid).float(), "valid_optimal": torch.tensor(optimal).float()}


@torch.no_grad()
def evaluate_maze(model, seeds, conds, targets, n_steps, batch_size, device):
    """Valid Path / Valid Optimal over stacked maze tensors."""
    results = {}
    for start in range(0, seeds.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        cond = conds[sl].to(device)
        state = rollout(model, seeds[sl].to(device), cond, n_steps)
        for key, value in maze_metrics(state, cond, targets[sl]).items():
            results.setdefault(key, []).append(value)
    return summarize(results)


@torch.no_grad()
def evaluate_arc_test(model, config, output_dir, checkpoint=None, n_steps=None, batch_size=None, image_limit=None):
    """Predict every test-NPZ pair in archive order, without adapting weights.

    Save a self-contained comparison NPZ, aggregate losses/accuracy, and a
    bounded number of PNG comparisons. Accuracy is reported in percent; losses
    retain their native cross-entropy scale. Task IDs are labels, not filters.
    """
    from nca.utils.visualization import visualize_batch

    from .arc_readout import decode_grids
    from .config import settings
    from .data.arc import ArcDataset
    from .data.factories import dataroot
    from .losses import create_loss

    arc = settings(config).ARC
    n_steps = arc.TEST_STEPS if n_steps is None else n_steps
    batch_size = arc.TEST_BATCH_SIZE if batch_size is None else batch_size
    image_limit = arc.TEST_IMAGE_LIMIT if image_limit is None else image_limit
    if n_steps < 1 or batch_size < 1 or image_limit < 0:
        raise ValueError("Use positive test steps/batch size and a nonnegative image limit.")

    # Create Test Dataset
    source_path = dataroot(config) / arc.TEST_FILE
    dataset = ArcDataset(source_path, "test", config.MODEL.CHANNEL_N)
    if len(dataset) == 0:
        raise ValueError("The ARC test archive contains no pairs.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.device(config.DEVICE).type == "cuda",
    )
    loss_fn = create_loss(config).to(config.DEVICE)
    predictions = np.zeros_like(dataset.outputs)
    prediction_sizes = np.zeros_like(dataset.output_sizes)
    exact_matches = np.zeros(len(dataset), dtype=bool)
    totals = {}

    # Prepare Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "imgs"
    if image_limit:
        image_dir.mkdir()
    was_training = model.training
    model.eval()
    offset = 0
    print(f"Testing all {len(dataset)} ARC pairs: {n_steps} updates, batch size {batch_size}.")
    try:
        # Predict and Score Test Pairs
        for seed, condition, target in tqdm(loader, desc="ARC test", unit="batch"):
            seed, condition, target = (
                tensor.to(config.DEVICE) for tensor in (seed, condition, target)
            )
            state = rollout(model, seed, condition, n_steps)
            metrics = loss_fn(state, target, condition)
            if not torch.isfinite(state).all() or any(not torch.isfinite(value) for value in metrics.values()):
                raise RuntimeError("ARC test produced non-finite states, losses, or metrics.")
            count = seed.shape[0]
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value.item() * count

            # Store Cropped Predictions on Padded Canvases
            for index, grid in enumerate(decode_grids(state)):
                row = offset + index
                predicted = np.asarray(grid, dtype=np.uint8)
                height, width = predicted.shape
                predictions[row, :height, :width] = predicted
                prediction_sizes[row] = (height, width)
                true_height, true_width = map(int, dataset.output_sizes[row])
                exact_matches[row] = np.array_equal(
                    predicted, dataset.outputs[row, :true_height, :true_width]
                )

            # Render a Fixed Prefix for Repeatable Comparisons
            shown = min(count, max(0, image_limit - offset))
            if shown:
                input_rgb, prediction_rgb, target_rgb = dataset.batch_to_rgb(
                    seed[:shown], state[:shown], target[:shown], condition[:shown]
                )
                visualize_batch(
                    input_rgb, prediction_rgb, target_rgb, {},
                    step_i=f"rows_{offset}_{offset + shown - 1}", save_path=str(image_dir),
                )
            offset += count
    finally:
        model.train(was_training)

    # Save All Predictions and Aggregate Metrics
    np.savez_compressed(
        output_dir / "predictions.npz",
        test_row_index=dataset.rows,
        test_task_id=dataset.task_ids,
        test_input=dataset.inputs,
        test_input_size=dataset.input_sizes,
        test_output=dataset.outputs,
        test_output_size=dataset.output_sizes,
        test_prediction=predictions,
        test_prediction_size=prediction_sizes,
        test_exact_match=exact_matches,
    )
    report = {
        "mode": "test", "pairs": len(dataset), "n_steps": n_steps,
        "batch_size": batch_size, "images_saved_for_pairs": min(image_limit, len(dataset)),
        "source_archive": str(source_path.resolve()),
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        "accuracy_unit": "percent", "loss_unit": "native cross-entropy",
        "metrics": {
            name: value / len(dataset) * (100.0 if name.endswith("_accuracy") else 1.0)
            for name, value in totals.items()
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(f"ARC test complete: {len(dataset)} pairs; outputs saved to {output_dir}")
    print(json.dumps(report["metrics"], indent=2, allow_nan=False))
    return report


def evaluate_arc(model, demos, queries, config, task_ids, n_steps, adapt_steps=None, adapt_lr=None):
    """Adapt independently on held-out tasks and report query accuracy percentages.

    Each task restores the pretrained weights and creates a fresh optimizer.
    Only demonstration targets enter gradient updates; query targets are encoded
    after prediction for scoring. Metrics are averaged over query pairs.
    """
    from nca.training.evolve_factory import create_evolver

    from .config import settings
    from .losses import create_loss

    spatial = settings(config)
    arc = spatial.ARC
    adapt_steps = arc.ADAPT_STEPS if adapt_steps is None else adapt_steps
    adapt_lr = arc.ADAPT_LR if adapt_lr is None else adapt_lr
    if not task_ids or any(not task_id.strip() for task_id in task_ids):
        raise ValueError("Select nonempty ARC task IDs for evaluation.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Select each ARC evaluation task once.")
    if not set(task_ids).issubset(arc.HOLDOUT_TASK_IDS):
        raise ValueError("Evaluation tasks must be excluded in the pretrained run's HOLDOUT_TASK_IDS.")
    if n_steps < 1 or adapt_steps < 0 or not 0 < adapt_lr < float("inf"):
        raise ValueError("Use positive rollout length and learning rate, and nonnegative adaptation steps.")

    # Select Tasks
    task_rows = []
    for task_id in task_ids:
        demo_rows = np.flatnonzero(demos.task_ids == task_id)
        query_rows = np.flatnonzero(queries.task_ids == task_id)
        if len(demo_rows) == 0 or len(query_rows) == 0:
            raise ValueError(f"Task {task_id} needs demonstrations and queries in the NPZ archives.")
        task_rows.append((demo_rows, query_rows))

    evolver = create_evolver(config)
    loss_fn = create_loss(config).to(config.DEVICE)
    pretrained = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    was_training = model.training
    metric_names = ("exact_accuracy", "size_accuracy", "height_accuracy", "width_accuracy", "color_accuracy")
    totals = {phase: dict.fromkeys(metric_names, 0.0) for phase in ("before", "after")}
    query_count = 0

    try:
        for demo_rows, query_rows in task_rows:
            # Encode Demonstrations and Query Inputs
            demo_samples = [demos[int(row)] for row in demo_rows]
            demo_seed, demo_cond, demo_target = (
                torch.stack(parts).to(config.DEVICE) for parts in zip(*demo_samples)
            )
            query_samples = [
                queries._encode_input(queries.inputs[row], queries.input_sizes[row])
                for row in query_rows
            ]
            query_seed, query_cond = (
                torch.stack(parts).to(config.DEVICE) for parts in zip(*query_samples)
            )

            # Predict Before Adaptation
            model.load_state_dict(pretrained)
            model.eval()
            with torch.no_grad():
                before_state = evolver(model, query_seed.clone(), query_cond, iter_n=n_steps)

            # Adapt on Demonstrations
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=adapt_lr,
                betas=config.TRAINING.OPTIMIZER_BETAS,
                weight_decay=spatial.WEIGHT_DECAY,
            )
            for _ in range(adapt_steps):
                optimizer.zero_grad(set_to_none=True)
                state = evolver(model, demo_seed.clone(), demo_cond, iter_n=n_steps)
                losses = loss_fn(state, demo_target, demo_cond)
                losses["total_loss"].backward()
                if config.TRAINING.GRADIENT_CLIPPING_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAINING.GRADIENT_CLIPPING_NORM
                    )
                optimizer.step()

            # Predict After Adaptation
            model.eval()
            with torch.no_grad():
                after_state = evolver(model, query_seed.clone(), query_cond, iter_n=n_steps)

                # Score Queries
                query_targets = []
                for row in query_rows:
                    colors, mask = queries._padded_colors(queries.outputs[row], queries.output_sizes[row])
                    query_targets.append(torch.cat([colors, mask], dim=0))
                query_target = torch.stack(query_targets).to(config.DEVICE)
                for phase, state in (("before", before_state), ("after", after_state)):
                    metrics = loss_fn(state, query_target, query_cond)
                    for name in metric_names:
                        totals[phase][name] += metrics[name].item() * len(query_rows)
            query_count += len(query_rows)
    finally:
        # Leave the caller's pretrained model ready for reuse.
        model.load_state_dict(pretrained)
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return {
        "tasks": len(task_ids),
        "queries": query_count,
        **{
            phase: {name: 100.0 * value / query_count for name, value in metrics.items()}
            for phase, metrics in totals.items()
        },
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """Load a CA state dict, dropping a ``torch.compile`` prefix if present."""
    state = torch.load(Path(path), map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    getattr(model, "_orig_mod", model).load_state_dict(state)
    return model.eval().to(device)
