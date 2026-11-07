"""Tests for the trainer audit findings (style of vesuvius/tests/ink_detection)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.config import (
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.training.metrics import Confusion
from vesuvius.ink_detection.training.optimizers import create_training_optimizer
import vesuvius.ink_detection.training.train as train_module
from vesuvius.ink_detection.training.train import (
    create_training_scheduler,
    prepare_loss_inputs,
    stage_training_request,
)
from vesuvius.ink_detection.types import MetricBatch

from .helpers import config_mapping


def _training_mapping() -> dict:
    authored = config_mapping(depth=1, side=4)
    authored.update(
        {
            "num_iterations": 6,
            "out_dir": "/data/tmp/ink-output",
            "seed": 9,
            "scheduler": {"name": "cosine_annealing"},
        }
    )
    return authored


def _training_config(authored: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


# 01: EMA starts from the weights of start_step


class _SyntheticTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, config, *, do_augmentations, patches=None, segments=None, **kwargs):
        del config, do_augmentations, kwargs
        self.training_patches = [0, 1]
        self.validation_patches = []
        self.segments = []
        self._patches = self.training_patches if patches is None else list(patches)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {
            "image": torch.ones(1, 1, 4, 4),
            "inklabels": torch.zeros(1, 1, 4, 4),
            "supervision_mask": torch.ones(1, 1, 4, 4),
        }


class _SyntheticTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"ink": image[:, :1, 0] * self.value}


class _SyntheticTrainingLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction - target[:, :1]) ** 2)


class _RecordingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}


def _run_synthetic_training(tmp_path: Path, monkeypatch, *, ema: dict, num_iterations: int, grad_acc_steps: int = 1):
    import vesuvius.ink_detection.data.dataset as dataset_module
    import vesuvius.ink_detection.models.model as model_module
    import vesuvius.ink_detection.training.losses as losses_module
    import vesuvius.ink_detection.training.optimizers as optimizers_module
    import vesuvius.ink_detection.training.samplers as samplers_module

    authored = _training_mapping()
    authored.update(
        {
            "num_iterations": num_iterations,
            "out_dir": str(tmp_path / "output"),
            "grad_acc_steps": grad_acc_steps,
            "mixed_precision": "no",
            "dataloader_workers": 0,
            "pin_memory": False,
            "val_every": 99,
            "save_every": num_iterations,
            "log_every": 99,
            "ema": ema,
        }
    )
    authored["model_config"]["pretrained_backbone"] = "synthetic"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")

    model = _SyntheticTrainingModel()
    values = []

    def record_forward(image):
        values.append(float(model.value.detach()))
        return {"ink": image[:, :1, 0] * model.value}

    monkeypatch.setattr(model, "forward", record_forward)
    monkeypatch.setattr(dataset_module, "InkDataset", _SyntheticTrainingDataset)
    monkeypatch.setattr(model_module, "make_model", lambda config: model)
    monkeypatch.setattr(losses_module, "create_loss", lambda config: _SyntheticTrainingLoss())
    monkeypatch.setattr(
        optimizers_module,
        "create_training_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        samplers_module,
        "build_sampling_policy",
        lambda patches, config, batch_size: SimpleNamespace(
            batch_sampler=None, shuffle=False, sampler=None, generator=None, audit={}
        ),
    )
    scheduler = _RecordingScheduler()
    monkeypatch.setattr(train_module, "create_training_scheduler", lambda optimizer, config: scheduler)
    assert train_module._run_training(stage_training_request(config_path)) == 0
    checkpoint = torch.load(
        tmp_path / "output" / f"ckpt_{num_iterations:06}.pth", map_location="cpu", weights_only=False
    )
    return checkpoint, values, scheduler


def test_ema_starts_from_the_weights_of_start_step(tmp_path: Path, monkeypatch):
    # Loss (v - 0)^2 with lr 0.1: v -> 0.8 v each step, so v = 1.0, 0.8, 0.64 after 0, 1, 2 steps.
    checkpoint, _, scheduler = _run_synthetic_training(
        tmp_path,
        monkeypatch,
        num_iterations=3,
        ema={
            "enabled": True,
            "decay": 0.5,
            "start_step": 2,
            "update_every_steps": 1,
            "validate": False,
            "save_in_checkpoint": True,
        },
    )
    assert scheduler.steps == 3
    torch.testing.assert_close(checkpoint["model"]["value"], torch.tensor(0.512))
    # Average of the weights seen from start_step on: copy 0.64 at step 2, then
    # 0.5 * 0.64 + 0.5 * 0.512 = 0.576 at step 3. The step 0 weights (1.0) must not enter.
    torch.testing.assert_close(checkpoint["ema_model"]["value"], torch.tensor(0.576))


# 02: full_3d coarse logits are upsampled on block centres
