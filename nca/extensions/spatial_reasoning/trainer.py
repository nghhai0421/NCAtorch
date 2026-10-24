"""Trainer of the spatial reasoning tasks.

The host image trainer with two changes: the loss is chosen with
``EXTENSIONS.SPATIAL_REASONING.LOSS`` and also receives the condition (Sudoku clues, maze
walls and endpoints), and the optimizer is AdamW with
``EXTENSIONS.SPATIAL_REASONING.WEIGHT_DECAY``. ``TRAINING.LOSS_FN`` is not used.
Optional ``CHECK_BACKPROP`` diagnostics verify gradients and optimizer updates
during smoke runs without changing the host training loop.
"""

import torch
from torch import optim

from nca.training.trainers.image_gen_trainer import ImageGenTrainer
from nca.training.training_utils import create_scheduler

from .config import settings
from .losses import create_loss


class SpatialReasoningTrainer(ImageGenTrainer):
    USES_FRAMEWORK_LOSS = False
    MAX_LOG_SAMPLES = 8

    def _initialize_additional_components(self):
        self.loss_fn = create_loss(self.config)
        self.check_backprop = settings(self.config).CHECK_BACKPROP
        self._backprop_gradient_norm = 0.0
        self._checked_optimizer_steps = 0

    def _initialize_base_optimizers(self):
        weight_decay = settings(self.config).WEIGHT_DECAY
        training = self.config.TRAINING
        self.optimizer = optim.AdamW(
            self.ca_model.parameters(), lr=training.LEARNING_RATE, betas=training.OPTIMIZER_BETAS,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        print(f"Using AdamW (weight decay {weight_decay}), LR schedule {training.LR_SCHEDULE_MODE}")

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)
        prediction, final_state = self.forward(initial_state, cond, target, logging=logging)
        losses = self.loss_fn(prediction, target, cond)
        if self.check_backprop and not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Backprop check failed: total loss is not finite.")
        return prediction, final_state, losses

    def _clip_gradients(self, parameters):
        """Inspect unscaled, accumulated gradients before the host clips them."""
        if self.check_backprop:
            norms = []
            for name, parameter in self.ca_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                gradient = parameter.grad
                if gradient is None:
                    raise RuntimeError(f"Backprop check failed: missing gradient for {name}.")
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Backprop check failed: non-finite gradient for {name}.")
                norms.append(gradient.detach().float().norm())
            gradient_norm = torch.stack(norms).norm()
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError("Backprop check failed: total gradient norm must be finite and nonzero.")
            self._backprop_gradient_norm = gradient_norm.item()
        return super()._clip_gradients(parameters)

    def _step_generator(self, accumulated_steps, completed_batches):
        """Verify weight changes around the host's real optimizer step."""
        if not self.check_backprop:
            return super()._step_generator(accumulated_steps, completed_batches)

        # A zero learning rate during warmup intentionally leaves weights unchanged.
        positive_lr = any(group["lr"] > 0 for group in self.optimizer.param_groups)
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.ca_model.named_parameters()
            if parameter.requires_grad
        } if positive_lr else {}
        stepped = super()._step_generator(accumulated_steps, completed_batches)

        if positive_lr:
            if not stepped:
                raise RuntimeError("Backprop check failed: optimizer step was skipped.")
            changed = 0
            for name, parameter in self.ca_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if not torch.isfinite(parameter).all():
                    raise RuntimeError(f"Backprop check failed: non-finite parameter {name} after updating.")
                changed += int(not torch.equal(before[name], parameter.detach()))
            if changed == 0:
                raise RuntimeError("Backprop check failed: optimizer did not change any trainable parameters.")
            self._checked_optimizer_steps += 1
            print(
                f"Backprop check: batch {completed_batches}, "
                f"gradient norm={self._backprop_gradient_norm:.6g}, "
                f"updated parameter tensors={changed}/{len(before)}"
            )
        return stepped

    def _on_train_end(self):
        """A successful smoke run must include a verified optimizer update."""
        if self.check_backprop:
            if self._checked_optimizer_steps == 0:
                raise RuntimeError("Backprop check failed: no positive-learning-rate optimizer updates were verified.")
            print(f"Backprop check PASSED: {self._checked_optimizer_steps} optimizer updates verified.")

    def add_img_logs(self, x0, x, target, cond=None):
        """Render only a few samples; Sudoku boards are drawn cell by cell."""
        n = self.MAX_LOG_SAMPLES
        self.logger.set_state_logs({k: v[:n] for k, v in self.logger.get_state_logs().items()})
        super().add_img_logs(x0[:n], x[:n], target[:n], cond[:n] if cond is not None else None)
