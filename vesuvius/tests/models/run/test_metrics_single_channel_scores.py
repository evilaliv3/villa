"""VOI and connected components score a single channel head like iou_dice does.

A head with out_channels 1 emits a score per voxel (a logit with activation none, a probability
with sigmoid). voi.py and connected_components.py squeezed it and compared it with the class id,
so a perfect prediction counted as empty: connected components difference 2 for 2 ground truth
components, VOI 0.92 instead of 0 for probabilities. iou_dice thresholds it (0 for logits, 0.5
for probabilities); the same cut is applied here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vesuvius.models.evaluation.connected_components import ConnectedComponentsMetric
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.evaluation.voi import VOIMetric


def _scene():
    gt = np.zeros((1, 1, 24, 24, 24), dtype=np.float32)
    gt[0, 0, 2:8, 2:8, 2:8] = 1
    gt[0, 0, 12:20, 10:18, 14:22] = 1
    return gt


@pytest.mark.parametrize(("fg", "bg"), [(10.0, -10.0), (0.98, 0.02), (0.9, -0.9)])
def test_perfect_single_channel_prediction_scores_perfectly(fg, bg):
    gt_np = _scene()
    pred = torch.from_numpy(np.where(gt_np > 0, fg, bg).astype(np.float32))
    gt = torch.from_numpy(gt_np)
    assert IOUDiceMetric(num_classes=2).compute(pred, gt)["dice_class_1"] == pytest.approx(1.0)
    voi = VOIMetric().compute(pred, gt)
    assert voi["voi_total"] == pytest.approx(0.0)
    cc = ConnectedComponentsMetric(num_classes=2).compute(pred, gt)
    assert cc["connected_components_difference_class_1"] == 0.0
    assert cc["connected_components_difference_class_0"] == 0.0
