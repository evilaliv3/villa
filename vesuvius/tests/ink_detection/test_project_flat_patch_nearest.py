"""project_flat_patch puts a surface point in the voxel whose centre is nearest, the rule of
vc_reader.resident.project_mask (bandVoxel, nearest_index) for the labels of the same native sample.
"""
import numpy as np

from vesuvius.ink_detection.data.geometry import project_flat_patch


def test_project_flat_patch_places_points_like_project_mask():
    positions = np.array([[[10.5, 10.5, 10.5], [3.49, 7.5, 2.0]]], np.float32)
    valid = np.ones((1, 2), bool)
    flat = np.ones((1, 2), np.uint8)
    crop = (0, 0, 0, 16, 16, 16)
    scattered = project_flat_patch(flat, positions, valid, crop)
    assert set(map(tuple, np.argwhere(scattered > 0))) == {(11, 11, 11), (3, 8, 2)}
