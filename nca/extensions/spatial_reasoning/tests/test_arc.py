"""ARC tensor encoding, categorical objectives, task selection and adaptation.

Use tiny local NPZ fixtures; the generated ARC package is not required.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import DATASET_REGISTRY, create_dataset
from nca.extensions.spatial_reasoning.arc_readout import decode_grids, size_logits
from nca.extensions.spatial_reasoning.data.arc import ArcDataset
from nca.extensions.spatial_reasoning.data.factories import create_arc
from nca.extensions.spatial_reasoning.evaluation import evaluate_arc, evaluate_arc_test
from nca.extensions.spatial_reasoning.scripts import evaluate_arc as arc_cli
from nca.extensions.spatial_reasoning.losses import ArcLoss, create_loss
from nca.extensions.spatial_reasoning.trainer import SpatialReasoningTrainer
from nca.training.evolve_factory import create_evolver
from nca.training.trainer_factory import TRAINER_REGISTRY
from nca.utils.config import Config
from scripts import train_ca


@pytest.fixture
def arc_archives(tmp_path):
    """Three pairs from two tasks, including real black cells and padding."""
    inputs = np.zeros((3, 30, 30), dtype=np.uint8)
    inputs[:, 0, 1] = 2
    inputs[2, 0, 1] = 3
    outputs = np.zeros_like(inputs)
    outputs[:, :2, :3] = 4
    for split in ("train", "test"):
        np.savez_compressed(tmp_path / f"synthetic_arc_{split}.npz", **{
            f"{split}_input": inputs,
            f"{split}_input_size": np.array([[2, 2]] * 3, dtype=np.uint8),
            f"{split}_output": outputs,
            f"{split}_output_size": np.array([[2, 3]] * 3, dtype=np.uint8),
            f"{split}_task_id": np.array(["task_a", "task_a", "task_b"]),
        })
    return tmp_path


@pytest.fixture
def arc_config_dict(arc_archives):
    path = Path(__file__).resolve().parents[1] / "configs/arc_baseline.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["DEVICE"] = "cpu"
    raw["DATASET"].update(DATAROOT=str(arc_archives), DROP_LAST_BATCH=False)
    raw["TRAINING"].update(BATCH_SIZE=2, MIXED_PRECISION=False, ITER_N_MIN=1, ITER_N_MAX=1)
    raw["LOGGING"]["INTERMEDIATE_LOGGING_STEPS"] = []
    raw["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"].update(
        HOLDOUT_TASK_IDS=["task_b"], EVAL_TASK_IDS=["task_b"]
    )
    return raw


def test_arc_encoding_and_rendering(arc_archives):
    dataset = ArcDataset(arc_archives / "synthetic_arc_train.npz", "train", 44)
    seed, cond, target = dataset[0]
    assert seed.shape == (44, 30, 30)
    assert cond.shape == (13, 30, 30)
    assert target.shape == (11, 30, 30)
    assert seed[0, 0, 0] == 1  # Real black activates its one-hot channel.
    assert seed[:, 2:, :].sum() == 0
    assert seed[10:].sum() == 0
    assert cond[10].sum() == 4 and target[10].sum() == 6
    assert cond[11:, 0, 0].tolist() == [0.0, 0.0]
    assert cond[11:, -1, -1].tolist() == [1.0, 1.0]
    images = dataset.batch_to_rgb(seed[None], seed[None], target[None], cond[None])
    assert all(image.shape == (1, 3, 30, 30) for image in images)
    assert images[0][0, :, 0, 0].sum() == 0
    assert torch.all(images[0][0, :, 2, 0] == 0.95)


def test_size_readout_uses_full_rows_and_columns():
    state = torch.zeros(1, 44, 30, 30)
    state[:, 10, 3, :] = 10
    state[:, 11, :, 6] = 10
    state[:, 4] = 5
    height_logits, width_logits = size_logits(state)
    assert torch.equal(height_logits, state[:, 10].mean(dim=2))
    assert torch.equal(width_logits, state[:, 11].mean(dim=1))
    assert decode_grids(state) == [[[4] * 7 for _ in range(4)]]


def test_arc_loss_masks_padding_and_backpropagates_size_scores(arc_archives):
    dataset = ArcDataset(arc_archives / "synthetic_arc_train.npz", "train", 44)
    _, _, target = dataset[0]
    state = torch.zeros(1, 44, 30, 30, requires_grad=True)
    losses = ArcLoss()(state, target[None])
    expected = torch.log(torch.tensor(10.0)) + 2 * torch.log(torch.tensor(30.0))
    assert torch.allclose(losses["total_loss"], expected)
    losses["total_loss"].backward()
    assert state.grad is not None
    assert state.grad[:, :10, 2:, :].count_nonzero() == 0
    assert state.grad[:, :10, :, 3:].count_nonzero() == 0
    assert state.grad[:, 10:12].count_nonzero() == 1800
    assert state.grad[:, 12:].count_nonzero() == 0


def test_arc_exact_accuracy_requires_correct_size_and_colors(arc_archives):
    dataset = ArcDataset(arc_archives / "synthetic_arc_test.npz", "test", 44)
    _, _, target = dataset[0]
    state = torch.zeros(1, 44, 30, 30)
    state[:, 4] = 20
    state[:, 10, 1, :] = 20
    state[:, 11, :, 2] = 20
    loss = ArcLoss()
    assert loss(state, target[None])["exact_accuracy"] == 1
    state[:, 10, 4, :] = 30
    metrics = loss(state, target[None])
    assert metrics["color_accuracy"] == 1
    assert metrics["size_accuracy"] == 0 and metrics["exact_accuracy"] == 0


def test_color_loss_gives_each_example_equal_weight():
    targets = torch.zeros(2, 11, 30, 30)
    targets[0, 4, :1, :1] = 1
    targets[0, 10, :1, :1] = 1
    targets[1, 4, :2, :3] = 1
    targets[1, 10, :2, :3] = 1
    state = torch.zeros(2, 44, 30, 30)
    state[1, 4] = 5
    logits = state[:, :10, 0, 0]
    reference = torch.nn.functional.cross_entropy(logits, torch.tensor([4, 4]))
    assert torch.allclose(ArcLoss()(state, targets)["color_loss"], reference)


def test_arc_factory_excludes_holdouts_and_uses_host_pipeline(arc_config_dict):
    config = Config(**arc_config_dict)
    train, cond_dim, height, width = create_arc(config, train=True)
    test, _, _, _ = create_arc(config, train=False)
    assert train.rows.tolist() == [0, 1] and test.rows.tolist() == [2]
    assert (cond_dim, height, width) == (13, 30, 30)
    assert DATASET_REGISTRY["arc_pairs"] is create_arc
    assert TRAINER_REGISTRY[config.TRAINING.TRAINER_TYPE] is SpatialReasoningTrainer
    loader, _, _, _ = create_dataset(config, train=True)
    seed, cond, target = next(iter(loader))
    model = create_model(config, cond_dim, height, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    state = create_evolver(config)(model, seed, cond, iter_n=1)
    losses = create_loss(config)(state, target, cond)
    losses["total_loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.count_nonzero() > 0 for gradient in gradients)
    optimizer.step()
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, model.parameters()))


@pytest.fixture
def backprop_check_trainer(arc_config_dict):
    """Initialize optimizer-hook state without creating training logs or files."""
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["CHECK_BACKPROP"] = True
    trainer = SpatialReasoningTrainer.__new__(SpatialReasoningTrainer)
    trainer.config = Config(**arc_config_dict)
    trainer.ca_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        for parameter in trainer.ca_model.parameters():
            parameter.fill_(1)
    trainer.optimizer = torch.optim.SGD(trainer.ca_model.parameters(), lr=0.1)
    trainer.lr_scheduler = None
    trainer.scaler = None
    trainer.accumulation_steps = 1
    trainer._initialize_additional_components()
    return trainer


def test_backprop_check_verifies_real_updates_and_allows_zero_lr_warmup(backprop_check_trainer, capsys):
    trainer = backprop_check_trainer
    before = [parameter.detach().clone() for parameter in trainer.ca_model.parameters()]
    trainer.optimizer.param_groups[0]["lr"] = 0
    trainer.ca_model(torch.ones(1, 2)).square().mean().backward()
    trainer._step_generator(1, 1)
    assert trainer._checked_optimizer_steps == 0
    assert all(torch.equal(old, parameter) for old, parameter in zip(before, trainer.ca_model.parameters()))

    trainer.optimizer.param_groups[0]["lr"] = 0.1
    trainer.ca_model(torch.ones(1, 2)).square().mean().backward()
    trainer._step_generator(1, 2)
    assert trainer._checked_optimizer_steps == 1
    assert trainer._backprop_gradient_norm > 0
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, trainer.ca_model.parameters()))
    trainer._on_train_end()
    assert "Backprop check PASSED" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "nan", "inf", "zero"])
def test_backprop_check_rejects_broken_gradients(backprop_check_trainer, problem):
    trainer = backprop_check_trainer
    parameters = list(trainer.ca_model.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    if problem == "missing":
        parameters[0].grad = None
    elif problem == "zero":
        for parameter in parameters:
            parameter.grad.zero_()
    else:
        parameters[0].grad.fill_(float(problem))
    with pytest.raises(RuntimeError, match="Backprop check failed"):
        trainer._clip_gradients(parameters)


def test_backprop_check_rejects_noop_optimizer(backprop_check_trainer, monkeypatch):
    trainer = backprop_check_trainer
    trainer.ca_model(torch.ones(1, 2)).square().mean().backward()
    monkeypatch.setattr("torch.optim.SGD.step", lambda self: None)
    with pytest.raises(RuntimeError, match="did not change any"):
        trainer._step_generator(1, 1)


def test_backprop_check_requires_a_verified_update(backprop_check_trainer):
    with pytest.raises(RuntimeError, match="no positive-learning-rate"):
        backprop_check_trainer._on_train_end()


@pytest.mark.parametrize("section, key, value, message", [
    ("MODEL", "CHANNEL_N", 11, "at least 12 mutable channels"),
    ("MODEL", "CLAMP_OUTPUT", True, "unconstrained logits"),
    ("MODEL", "LIVING_MASK", True, "unconstrained logits"),
    ("MODEL", "USE_POSITIONAL_EMBEDDINGS", True, "unconstrained logits"),
    ("DATASET", "TARGET_SIZE", 32, "TARGET_SIZE=30"),
    ("PATTERN_POOL", "ENABLED", True, "unconstrained logits"),
])
def test_arc_config_rejects_incompatible_baseline_settings(arc_config_dict, section, key, value, message):
    arc_config_dict[section][key] = value
    with pytest.raises(ValidationError, match=message):
        Config(**arc_config_dict)


def test_arc_adaptation_resets_weights_and_optimizer_per_task(arc_archives, arc_config_dict, monkeypatch):
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"].update(
        HOLDOUT_TASK_IDS=["task_a", "task_b"], EVAL_TASK_IDS=["task_a", "task_b"]
    )
    config = Config(**arc_config_dict)
    demos = ArcDataset(arc_archives / "synthetic_arc_train.npz", "train", 44)
    queries = ArcDataset(arc_archives / "synthetic_arc_test.npz", "test", 44)
    model = create_model(config, 13, 30, 30).eval()
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer_initials = []
    adamw = torch.optim.AdamW

    def record_optimizer(parameters, **kwargs):
        optimizer_initials.append({name: tensor.clone() for name, tensor in model.state_dict().items()})
        return adamw(parameters, **kwargs)

    monkeypatch.setattr("torch.optim.AdamW", record_optimizer)
    scores = evaluate_arc(model, demos, queries, config, ["task_a", "task_b"], 1, adapt_steps=1)
    assert scores["tasks"] == 2 and scores["queries"] == 3
    assert len(optimizer_initials) == 2
    assert all(torch.equal(snapshot[name], value) for snapshot in optimizer_initials for name, value in initial.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())
    assert not model.training
    assert all(0 <= value <= 100 for phase in ("before", "after") for value in scores[phase].values())
    with pytest.raises(ValueError, match="excluded"):
        evaluate_arc(model, demos, queries, config, ["unheld_task"], 1)


class FixedArcPrediction(torch.nn.Module):
    """Correct colors/width; predict the wrong height for the color-3-marked pair."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, state, condition):
        prediction = torch.zeros_like(state) + self.anchor * 0
        prediction[:, 4] = 20
        prediction[:, 10, 1, :] = 20
        prediction[:, 11, :, 2] = 20
        wrong_height = condition[:, 3, 0, 1].bool()
        prediction[wrong_height, 10, 1, :] = 0
        prediction[wrong_height, 10, 3, :] = 20
        return prediction, prediction - state


def test_full_arc_test_exports_all_rows_and_weights_the_partial_batch(arc_config_dict, tmp_path):
    # The fixture selects only task_b for adaptation; plain testing must still score all three rows.
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"]["TRAIN_FILE"] = "unused_train_file.npz"
    config = Config(**arc_config_dict)
    model = FixedArcPrediction().train()
    output_dir = tmp_path / "test_results"
    report = evaluate_arc_test(model, config, output_dir, n_steps=1, batch_size=2, image_limit=1)
    assert report["pairs"] == 3 and report["images_saved_for_pairs"] == 1
    assert report["accuracy_unit"] == "percent"
    assert report["metrics"]["exact_accuracy"] == pytest.approx(200 / 3)
    assert report["metrics"]["size_accuracy"] == pytest.approx(200 / 3)
    assert report["metrics"]["height_accuracy"] == pytest.approx(200 / 3)
    assert report["metrics"]["width_accuracy"] == 100
    assert report["metrics"]["color_accuracy"] == 100
    assert model.training and model.anchor.grad is None and model.anchor.item() == 0
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")) == report
    assert (output_dir / "imgs/step_rows_0_0.png").is_file()
    assert len(list((output_dir / "imgs").glob("*.png"))) == 1
    with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
        assert archive["test_row_index"].tolist() == [0, 1, 2]
        assert archive["test_task_id"].tolist() == ["task_a", "task_a", "task_b"]
        assert archive["test_prediction"].shape == (3, 30, 30)
        assert archive["test_prediction"].dtype == np.uint8
        assert archive["test_prediction_size"].tolist() == [[2, 3], [2, 3], [4, 3]]
        assert archive["test_exact_match"].tolist() == [True, True, False]
        assert archive["test_prediction"][0, 2:, :].sum() == 0
        assert archive["test_prediction"][2, 4:, :].sum() == 0
        assert archive["test_input_size"].tolist() == [[2, 2]] * 3
        assert archive["test_output_size"].tolist() == [[2, 3]] * 3
        assert archive["test_output"][0, :2, :3].tolist() == [[4] * 3] * 2


def test_full_arc_test_protects_existing_outputs(arc_config_dict, tmp_path):
    output_dir = tmp_path / "existing_test"
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        evaluate_arc_test(FixedArcPrediction(), Config(**arc_config_dict), output_dir, image_limit=0)
    assert list(output_dir.iterdir()) == []


def test_full_arc_test_rejects_nonfinite_predictions_and_restores_mode(arc_config_dict, tmp_path):
    model = FixedArcPrediction().train()
    with torch.no_grad():
        model.anchor.fill_(float("nan"))
    output_dir = tmp_path / "failed_test"
    with pytest.raises(RuntimeError, match="non-finite"):
        evaluate_arc_test(model, Config(**arc_config_dict), output_dir, n_steps=1, image_limit=0)
    assert model.training
    assert not (output_dir / "metrics.json").exists()


def test_arc_cli_defaults_to_plain_test_without_holdout_ids(arc_config_dict, tmp_path, monkeypatch):
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"].update(HOLDOUT_TASK_IDS=[], EVAL_TASK_IDS=[])
    config = Config(**arc_config_dict)
    model = FixedArcPrediction().eval()
    output_dir = tmp_path / "cli_test"
    monkeypatch.setattr(
        "nca.extensions.spatial_reasoning.scripts.evaluate_arc.load_run",
        lambda args, cond_dim: (config, model),
    )
    monkeypatch.setattr("sys.argv", [
        "evaluate_arc", "--checkpoint", str(tmp_path / "ca_final.pt"),
        "--output-dir", str(output_dir), "--n-steps", "1", "--batch-size", "2", "--image-limit", "0",
    ])
    arc_cli.main()
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["pairs"] == 3 and report["mode"] == "test"
    assert not model.training
    assert not (output_dir / "imgs").exists()


def test_arc_training_saves_checkpoint_then_runs_the_full_test(arc_config_dict, tmp_path, monkeypatch):
    from nca.extensions.spatial_reasoning import evaluation

    run_test = evaluation.evaluate_arc_test

    def verify_saved_weights(model, config, output_dir, checkpoint=None):
        assert Path(checkpoint).is_file()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert all(torch.equal(model.state_dict()[name], value) for name, value in saved.items())
        report = run_test(model, config, output_dir, checkpoint=checkpoint)
        assert all(torch.equal(model.state_dict()[name], value) for name, value in saved.items())
        return report

    monkeypatch.setattr(
        "nca.extensions.spatial_reasoning.evaluation.evaluate_arc_test", verify_saved_weights
    )
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"].update(
        TEST_AFTER_TRAIN=True, TEST_STEPS=1, TEST_BATCH_SIZE=2, TEST_IMAGE_LIMIT=1,
    )
    config_path = tmp_path / "arc.yaml"
    config_path.write_text(yaml.safe_dump(arc_config_dict), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_SWEEP", raising=False)
    monkeypatch.setattr("sys.argv", [
        "train_ca.py", "--config", str(config_path),
        "-o", "TRAINING.STEPS=2", "-o", "TRAINING.WARMUP_STEPS=1",
        "-o", "LOGGING.LOG_INTERVAL=2", "-o", "LOGGING.SAVE_INTERVAL=2",
    ])
    with pytest.raises(SystemExit) as exit_info:
        train_ca.main()
    assert exit_info.value.code == 0
    (run,) = (tmp_path / "train_log").iterdir()
    checkpoint = run / "ca_final.pt"
    assert checkpoint.is_file()
    report = json.loads((run / "test/metrics.json").read_text(encoding="utf-8"))
    assert report["pairs"] == 3
    assert report["checkpoint"] == str(checkpoint.resolve())
    assert (run / "test/predictions.npz").is_file()
    assert (run / "test/imgs/step_rows_0_0.png").is_file()

    # Reload the checkpoint through the standalone CLI and reproduce the predictions.
    separate_output = tmp_path / "reloaded_test"
    monkeypatch.setattr("sys.argv", [
        "evaluate_arc", "--checkpoint", str(checkpoint), "--device", "cpu",
        "--output-dir", str(separate_output), "--n-steps", "1", "--batch-size", "2", "--image-limit", "0",
    ])
    arc_cli.main()
    with np.load(run / "test/predictions.npz", allow_pickle=False) as automatic:
        with np.load(separate_output / "predictions.npz", allow_pickle=False) as standalone:
            assert np.array_equal(automatic["test_prediction"], standalone["test_prediction"])
            assert np.array_equal(automatic["test_prediction_size"], standalone["test_prediction_size"])


@pytest.mark.parametrize("key, value", [("TEST_STEPS", 0), ("TEST_BATCH_SIZE", 0), ("TEST_IMAGE_LIMIT", -1)])
def test_invalid_arc_test_settings_are_rejected(arc_config_dict, key, value):
    arc_config_dict["EXTENSIONS"]["SPATIAL_REASONING"]["ARC"][key] = value
    with pytest.raises(ValidationError, match=key):
        Config(**arc_config_dict)
