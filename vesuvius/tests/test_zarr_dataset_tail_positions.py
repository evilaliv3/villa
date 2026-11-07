"""Tests for the modelsdata draft patches (style of vesuvius/tests/models/datasets).

Each test fails on upstream/main 7e6b82164 and passes with the matching patch applied:
  test_grid_positions_cover_the_tail            patch01_zarr_dataset_tail_positions.patch
  test_cache_key_distinguishes_ignore_label     patch02_cache_ignore_label.patch
  test_cross_frame_full_scan_applies_thresholds patch03_crossframe_full_scan_thresholds.patch
  test_ds_downsampling_keeps_ignore_sentinel    patch04_ds_ignore_sentinel.patch
  test_mutex_mask_is_padded_like_affinities     patch05_mutex_mask_padding.patch
  test_cross_frame_skeleton_ignore_zero         patch06_crossframe_skeleton_ignore_chain.patch
  test_intensity_sampler_reads_chunks           patch08_intensity_chunk_grouped_sampling.patch
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, create_group

from vesuvius.data import affine
from vesuvius.models.datasets.cross_frame_dataset import CrossFrameZarrDataset
from vesuvius.models.datasets.mutex_affinity_dataset import MutexAffinityDataset, PatchInfo
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches.cache import build_cache_params, cache_filename


def test_grid_positions_cover_the_tail():
    ds = ZarrDataset.__new__(ZarrDataset)
    ds.patch_size = (64, 64, 64)
    positions = ds._iter_3d_positions((100, 128, 150), (64, 64, 64))
    covered = np.zeros((100, 128, 150), dtype=bool)
    for z, y, x in positions:
        covered[z:z + 64, y:y + 64, x:x + 64] = True
    assert covered.all(), f"{(~covered).mean():.3f} of the volume is never in a patch"
    assert all(z + 64 <= 100 and y + 64 <= 128 and x + 64 <= 150 for z, y, x in positions)
    ds.patch_size = (64, 64)
    positions_2d = ds._iter_2d_positions((100, 128), (64, 64))
    assert (36, 0) in positions_2d and (36, 64) in positions_2d
