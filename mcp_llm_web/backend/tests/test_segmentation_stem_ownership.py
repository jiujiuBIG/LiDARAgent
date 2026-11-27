import numpy as np
from affine import Affine

from tools.Segmentation.point_instances import label_crown_and_stem_instances, label_traced_instances


def test_verified_stem_spans_both_sides_of_breast_height():
    cloud = np.array([
        [0.55, 0.45, 0.7],
        [0.55, 0.45, 0.9],
        [0.45, 0.45, 1.3],
        [0.55, 0.45, 1.7],
        [1.5, 0.45, 0.95],
    ])
    labels = label_crown_and_stem_instances(
        cloud, np.array([[7, 7]]), Affine.identity(),
        {7: (0.5, 0.5, 15)}, 2.0, stem_positions=[(0.5, 0.5)],
    )
    np.testing.assert_array_equal(labels, [0, 7, 7, 7, 0])


def test_stem_trace_extends_below_breast_height_only_with_same_column_support():
    cloud = np.array([[0.1, 0.1, 0.9], [1.1, 0.1, 0.9], [0.1, 0.1, 0.7]])
    owners = np.full((1, 2, 18), -1, dtype=np.int32)
    owners[0, :, 3:5] = [[0, 0], [0, 1]]
    labels = label_traced_instances(
        cloud, owners, np.array([0.0, 0.0]), 1.0, 0.5,
        {0: (0.1, 0.1, 8), 1: (4.1, 0.1, 8)},
        {7: (0.1, 0.1, 8), 8: (4.1, 0.1, 8)},
    )
    np.testing.assert_array_equal(labels, [7, 0, 0])


def test_short_crown_does_not_take_upper_canopy_from_neighbour():
    cloud = np.array([
        [0.75, 0.5, 15.0],  # Within the short seed's crown in XY, above its apex.
        [0.75, 0.5, 15.2],  # Same geometry previously claimed by the stem trace.
        [1.25, 0.5, 15.0],  # Nearby return already owned by the taller crown.
        [0.75, 0.5, 6.0],   # Correct lower-crown return in the same XY column.
        [0.75, 0.5, 23.0],  # No nearby, height-supported crown at this elevation.
    ])
    labels = label_crown_and_stem_instances(
        cloud, np.array([[7, 8]]), Affine.identity(),
        {7: (0.5, 0.5, 7), 8: (1.5, 0.5, 17)}, 2.0,
        traced_labels=np.array([0, 7, 0, 7, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(labels, [8, 8, 8, 7, 0])
