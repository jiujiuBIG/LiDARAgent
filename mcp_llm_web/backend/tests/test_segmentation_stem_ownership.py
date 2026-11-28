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


def test_shared_crown_stems_recover_only_forced_links_to_existing_instances():
    cloud = np.array([
        [1.6, 0.5, 1.3], [3.8, 0.5, 1.3], [1.6, 0.5, 0.9], [3.8, 0.5, 1.7],
        [0.5, 0.5, 15.0], [2.5, 0.5, 15.0], [3.8, 0.5, 0.3],
    ])
    raster = np.array([[7, 8, 8, 8, 8]])
    stems = [(1.6, 0.5), (3.8, 0.5)]
    unique = {7: (0.5, 0.5, 16), 8: (2.5, 0.5, 16)}
    labels = label_crown_and_stem_instances(
        cloud, raster, Affine.identity(), unique, 2.0, stem_positions=stems,
    )
    np.testing.assert_array_equal(labels, [7, 8, 7, 8, 7, 8, 0])

    ambiguous = {7: (2.0, 0.5, 16), 8: (2.5, 0.5, 16)}
    labels = label_crown_and_stem_instances(
        cloud, raster, Affine.identity(), ambiguous, 2.0, stem_positions=stems,
    )
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 7, 8, 0])


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


def test_lower_crown_recovery_requires_independent_stems_and_a_visible_cap():
    from tools.Segmentation.point_instances import select_separated_crown_tops

    raster = np.full((4, 7), 7, dtype=np.int32)
    heights = np.full(raster.shape, 22.0)
    heights[1, 3] = 17.5
    stems = [(1.5, 1.5), (3.5, 1.5), (5.5, 1.5)]
    traces = {0: (1.5, 1.5, 21), 1: (3.5, 1.5, 17), 2: (5.5, 1.5, 16)}
    crowns = {7: (1.5, 1.5, 22)}

    assert select_separated_crown_tops(stems, traces, raster, heights, Affine.identity(), crowns) == [
        (3.5, 1.5, 17),
    ]
    # A low cap within the only stem's crown can be a branch, even if visible.
    assert not select_separated_crown_tops(
        stems[1:2], {0: traces[1]}, raster, heights, Affine.identity(), crowns,
    )
    # Two low traces provide no independent support for the incumbent crown.
    assert not select_separated_crown_tops(
        stems, {**traces, 0: (1.5, 1.5, 18)}, raster, heights, Affine.identity(), crowns,
    )
def test_occluded_crown_has_a_point_only_id_and_preserves_the_upper_crown(tmp_path):
    from affine import Affine
    import laspy
    from tools.Segmentation import point_instances as instances

    rng = np.random.default_rng(12)
    stem = np.column_stack((rng.normal(2.5, 0.04, 24), rng.normal(2.5, 0.04, 24), np.full(24, 1.3)))
    lower = np.column_stack((rng.normal(2.5, 0.2, 140), rng.normal(2.5, 0.2, 140), rng.uniform(9, 12, 140)))
    upper = np.column_stack((lower[:, :2], np.full(len(lower), 19.0)))
    cloud = np.vstack((stem, lower, upper))
    raster = np.ones((6, 6), dtype=np.int32)
    original = np.r_[np.zeros(len(stem), dtype=np.int32), np.ones(len(lower) + len(upper), dtype=np.int32)]
    caps = {0: (2.5, 2.5, 12.0), 1: (4.5, 2.5, 19.0)}
    labels, parents = instances.recover_occluded_instances(
        cloud, original, raster, Affine.identity(), raster * 19.0,
        [(2.5, 2.5), (4.5, 2.5)], caps, {1: caps[1]},
    )
    assert parents == {2: 1}
    np.testing.assert_array_equal(labels[: len(stem) + len(lower)], 2)
    np.testing.assert_array_equal(labels[-len(upper):], 1)
    assert np.all(raster == 1)

    las = laspy.create()
    las.x, las.y, las.z = cloud.T
    source = tmp_path / "input.las"
    las.write(source)
    table = tmp_path / "tree_summary.csv"
    export = instances.write_instance_cloud(
        source, tmp_path / "tree_instances.las", cloud[:, 2], labels,
        summary_path=table, occluded_parent_ids=parents,
    )
    assert export["summary_filename"] == table.name
    assert export["occluded_tree_count"] == 1
    assert table.read_text().splitlines()[-1].split(",") == ["2", "164", str(lower[:, 2].max()), "24", "1"]


def test_codominant_crown_recovery_splits_independent_stems_with_horizontal_separation():
    from tools.Segmentation.point_instances import select_separated_crown_tops

    raster = np.full((4, 7), 7, dtype=np.int32)
    heights = np.full(raster.shape, 25.0)
    heights[1, 3] = 24.2
    crowns = {7: (1.5, 1.5, 25.0)}

    # Co-dominant second stem >= 1.0 m away with visible CHM cap is recovered.
    stems_far = [(1.5, 1.5), (3.5, 1.5)]
    traces_far = {0: (1.5, 1.5, 24.5), 1: (3.5, 1.5, 23.8)}
    assert select_separated_crown_tops(stems_far, traces_far, raster, heights, Affine.identity(), crowns) == [
        (3.5, 1.5, 23.8),
    ]

    # Co-dominant second stem < 1.0 m away is rejected to avoid splitting a single crown.
    stems_close = [(1.5, 1.5), (2.2, 1.5)]
    traces_close = {0: (1.5, 1.5, 24.5), 1: (2.2, 1.5, 23.8)}
    assert not select_separated_crown_tops(stems_close, traces_close, raster, heights, Affine.identity(), crowns)


def test_shared_crown_apex_centred_stem_recovers_unassigned_breast_height():
    cloud = np.array([
        [1.55, 0.5, 1.3],   # Primary stem directly beneath crown 7 apex (d=0.05 m)
        [1.55, 0.5, 0.9],
        [2.35, 0.5, 1.3],   # Secondary stem in the same crown (d=0.85 m)
        [1.5, 0.5, 20.0],   # Canopy return for crown 7
    ])
    raster = np.array([[7, 7, 7, 7]])
    stems = [(1.55, 0.5), (2.35, 0.5)]
    treetops = {7: (1.5, 0.5, 20.0)}
    labels = label_crown_and_stem_instances(
        cloud, raster, Affine.identity(), treetops, 2.0, stem_positions=stems,
    )
    np.testing.assert_array_equal(labels, [7, 7, 0, 7])
