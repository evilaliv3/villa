"""For vc_reader/tests/test_resident_sampling.py: a 2 x 2 cell whose corners are farther apart than the
sampler can cover (kSubdivisions * 0.5 = 32 voxels between neighbours) is a tear of the flat grid, not a
sheet; project_mask must not draw a bilinear patch across it. Cells within the contract are unchanged."""
import numpy as np
from vc_reader.resident import project_mask


def _band(spacing, half_thickness=1.0):
    # two columns of pixels `spacing` voxels apart in x, one voxel apart in y
    positions = np.array([[[10.0, 10.0, 10.0], [10.0, 10.0, 10.0 + spacing]],
                          [[10.0, 11.0, 10.0], [10.0, 11.0, 10.0 + spacing]]], np.float32)
    normals = np.zeros_like(positions); normals[..., 0] = 1.0
    valid = np.ones((2, 2), bool)
    return project_mask(valid, positions, normals, valid, (0, 0, 0), (20, 20, 120), half_thickness=half_thickness) > 0


def test_cell_within_the_sampling_contract_is_filled():
    band = _band(30.0)
    assert band[9:12, 10:12, 10:41].all()  # 31 voxels of x, 3 of z: the sheet between the columns


def test_cell_beyond_the_sampling_contract_is_a_tear():
    band = _band(40.0)
    assert not band[:, :, 15:46].any(), 'no bilinear bridge across 40 voxels'
