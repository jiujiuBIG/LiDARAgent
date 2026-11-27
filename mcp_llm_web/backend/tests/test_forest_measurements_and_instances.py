import asyncio
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import laspy
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS


def load_measurement(filename):
    path = Path(__file__).resolve().parents[1] / "tools" / "ForestParams" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dbh_uses_hag_and_keeps_unmeasured_trees(tmp_path, monkeypatch):
    tool = load_measurement("3.DHB.py")
    frame = pd.DataFrame({
        "x": np.arange(44), "y": np.zeros(44), "z": 101.3,
        "treeid": [1] * 20 + [2, 3, 4, 4] + [5] * 20,
        "heightaboveground": [1.3] * 20 + [1.3, np.nan, 2.1, 2.2] + [1.3] * 20,
    })
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    received = []

    def fit(xyz, treeid, **kwargs):
        received.extend(xyz[:, 2])
        return pd.DataFrame({"treeid": [1], "DBH": [0.30], "vote": [0.9]})

    monkeypatch.setattr(tool, "fit_dbh_sections", fit)
    result = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path)))
    assert result["status"] == "success"
    assert 101.3 not in received and 1.3 in received
    rows = pd.read_csv(result["output_path"]).set_index("treeid")
    assert rows["measurement_status"].tolist() == [
        "measured", "insufficient_breast_height_points", "missing_height_above_ground",
        "no_breast_height_points", "fit_failed",
    ]
    assert rows.loc[1, "DBH"] == pytest.approx(0.30)


def test_dbh_requires_explicit_confirmation_to_use_z(tmp_path, monkeypatch):
    tool = load_measurement("3.DHB.py")
    frame = pd.DataFrame({"x": [0], "y": [0], "z": [1.3], "treeid": [1]})
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    monkeypatch.setattr(tool, "fit_dbh_sections", lambda **kw: pd.DataFrame())
    rejected = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path)))
    assert rejected["status"] == "error"
    accepted = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path), z_is_normalized=True))
    assert accepted["status"] == "warning"
    assert accepted["data"]["height_reference"] == "normalized_z"


def test_dbh_fits_known_diameter_with_absolute_z_and_explicit_height(tmp_path, monkeypatch):
    tool = load_measurement("3.DHB.py")
    angle = np.tile(np.linspace(0, 2 * np.pi, 360, endpoint=False), 3)
    radii = np.repeat([0.198, 0.2, 0.202], 360)
    frame = pd.DataFrame({
        "x": radii * np.cos(angle), "y": radii * np.sin(angle),
        "z": 101.3, "treeid": 1, "heightaboveground": 1.3,
    })
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    result = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path)))
    assert result["status"] == "success", result
    assert pd.read_csv(result["output_path"])["DBH"].item() == pytest.approx(0.4, abs=0.02)


def test_dbh_fits_sparse_partial_stem_and_rejects_a_line(tmp_path, monkeypatch):
    tool = load_measurement("3.DHB.py")
    angle = np.linspace(-1.1, 1.1, 26)
    radius = 0.16 + 0.003 * np.sin(np.arange(len(angle)))
    stem = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    stem = np.vstack((stem, [[0.33, 0.1], [0.2, -0.31], [0.1, 0.2]]))
    line = np.column_stack((np.linspace(-0.2, 0.2, 26), np.zeros(26)))
    points = np.vstack((stem, line))
    frame = pd.DataFrame({
        "x": 480000 + points[:, 0], "y": 5600000 + points[:, 1], "z": 101.3,
        "treeid": [1] * len(stem) + [2] * len(line), "heightaboveground": 1.3,
    })
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    result = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path)))
    rows = pd.read_csv(result["output_path"]).set_index("treeid")
    assert rows.loc[1, "DBH"] == pytest.approx(0.32, abs=0.03)
    assert rows.loc[2, "measurement_status"] == "fit_failed"


def test_dbh_interpolates_at_breast_height_with_two_sided_tilted_stem_evidence(tmp_path, monkeypatch):
    tool = load_measurement("3.DHB.py")
    heights = np.repeat([0.9, 1.05, 1.3, 1.55, 1.7], 16)
    angles = np.tile(np.linspace(0.1, 4.7, 16), 5)
    tilt = 0.3
    radius = 0.18
    x = tilt * (heights - 1.3) + radius * np.sqrt(1 + tilt**2) * np.cos(angles)
    y = radius * np.sin(angles)
    stem = pd.DataFrame({
        "x": 480000 + x, "y": 5600000 + y, "z": 100 + heights,
        "treeid": 1, "heightaboveground": heights,
    })
    one_sided = stem.loc[stem.heightaboveground >= 1.3].assign(treeid=2)
    distant = stem.loc[stem.heightaboveground.isin([0.9, 1.7])].assign(treeid=3)
    frame = pd.concat([stem, one_sided, distant], ignore_index=True)
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    result = asyncio.run(tool.extract_forest_dbh("test.las", str(tmp_path)))
    rows = pd.read_csv(result["output_path"]).set_index("treeid")
    assert rows.loc[1, "measurement_status"] == "measured"
    assert rows.loc[1, "DBH"] == pytest.approx(0.36, abs=0.02)
    assert rows.loc[1, "fit_method"] == "bracketed_stem_cylinder"
    assert rows.loc[2, "measurement_status"] == "insufficient_breast_height_points"
    assert rows.loc[3, "measurement_status"] == "no_breast_height_points"


def test_crown_threshold_and_fit_share_the_same_points(tmp_path, monkeypatch):
    tool = load_measurement("2.TreeCrown.py")
    z = np.arange(5000, dtype=np.float64)
    frame = pd.DataFrame({"x": z, "y": z % 20, "z": z, "treeid": 1})
    monkeypatch.setattr(tool, "read_point_cloud", lambda *args, **kwargs: frame)
    monkeypatch.setattr(tool, "denoise_point_cloud", lambda points: points[:-200])
    fitted = []

    def fit(points):
        fitted.append(points)
        return {"major_axis_length": 8, "minor_axis_length": 5, "center_x": 0, "center_y": 0}

    monkeypatch.setattr(tool, "calculate_crown_ellipse_dimensions", fit)
    result = asyncio.run(tool.extract_tree_crown("test.las", str(tmp_path)))
    expected = z[:-200]
    expected = expected[expected >= np.percentile(expected, 30)]
    np.testing.assert_array_equal(fitted[0][:, 2], expected)
    assert pd.read_csv(result["output_path"])["points_length"].item() == len(expected)


def test_crown_exports_full_sorted_axes_in_projected_coordinates(tmp_path, monkeypatch):
    tool = load_measurement("2.TreeCrown.py")
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    rotation = np.radians(37)
    u, v = 4 * np.cos(angles), 2.5 * np.sin(angles)
    center = np.array([480612.345, 5601258.789])
    x = center[0] + u * np.cos(rotation) - v * np.sin(rotation)
    y = center[1] + u * np.sin(rotation) + v * np.cos(rotation)
    z = np.full(len(angles), 20.0)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [*center, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    header.add_crs(CRS.from_epsg(25833))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = x, y, z
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud.treeID = np.full(len(angles), 12, dtype=np.int32)
    source = tmp_path / "known_crown.las"
    cloud.write(source)

    monkeypatch.setattr(tool, "denoise_point_cloud", lambda points: points)
    result = asyncio.run(tool.extract_tree_crown(str(source), str(tmp_path)))
    assert result["status"] == "success", result
    row = pd.read_csv(result["output_path"]).iloc[0]
    assert row.treeid == 12
    assert row.major_axis_length == pytest.approx(8.0, abs=0.02)
    assert row.minor_axis_length == pytest.approx(5.0, abs=0.02)
    np.testing.assert_allclose([row.center_x, row.center_y], center, rtol=0, atol=0.01)

    fit = tool.calculate_crown_ellipse_dimensions(np.column_stack((x, y, z)))
    assert fit["convex_hull_area"] == pytest.approx(np.pi * 4 * 2.5, rel=0.001)
    assert fit["orientation"] == pytest.approx(rotation, abs=0.01)


def test_instance_trace_preserves_points_and_leaves_ambiguous_unassigned(tmp_path):
    from tools.Segmentation.point_instances import label_traced_instances, write_instance_cloud
    from tools.Preprocessing.read import read_point_cloud

    source = tmp_path / "input.las"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(32650))
    las = laspy.LasData(header)
    las.x = [0.1, 0.1, 1.1, 1.1, 2.1, 0.1]
    las.y = [0.1] * 6
    las.z = [101.3, 108, 101.3, 108, 101.3, 100]
    las.intensity = [1, 2, 3, 4, 5, 6]
    las.classification = [1, 1, 1, 1, 1, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="treeid", type=np.int32))
    las.treeid = [40, 40, 41, 41, 42, 0]
    las.write(source)

    hag = np.array([1.3, 8, 1.3, 8, 1.3, np.nan])
    cloud = np.column_stack((np.asarray(las.x), np.asarray(las.y), hag))
    owners = np.full((1, 3, 18), -1, dtype=np.int32)
    owners[0, 0, 3:17] = 0
    owners[0, 1, 3] = 1
    owners[0, 1, 4] = 2
    owners[0, 1, 16] = 1
    owners[0, 2, 3:17] = 2
    labels = label_traced_instances(
        cloud, owners, np.array([0.0, 0.0]), 1, 0.5,
        {0: (0.1, 0.1, 8), 1: (3.1, 0.1, 12), 2: (3.2, 0.1, 12)},
        {7: (0.1, 0.1, 8), 8: (3.1, 0.1, 12)},
    )
    np.testing.assert_array_equal(labels, [7, 7, 0, 0, 0, 0])
    path = tmp_path / "tree_instances.las"
    summary = write_instance_cloud(source, path, hag, labels)
    saved = laspy.read(path)
    assert saved.header.parse_crs() == las.header.parse_crs()
    for name in ("X", "Y", "Z", "intensity", "classification", "treeid"):
        np.testing.assert_array_equal(np.asarray(saved[name]), np.asarray(las[name]))
    np.testing.assert_array_equal(saved.lidaragent_tree_id, labels)
    np.testing.assert_allclose(saved.lidaragent_hag, hag, equal_nan=True)
    assert read_point_cloud(str(path)).treeid.tolist() == labels.tolist()
    assert summary["labelled_points"] == 2 and summary["unassigned_points"] == 4


def test_instance_labels_use_crown_and_verified_stem_without_projecting_the_whole_column():
    from affine import Affine
    from tools.Segmentation import point_instances

    cloud = np.array([
        [0.45, 0.5, 12], [1.5, 0.5, 11], [2.5, 0.5, 12],
        [0.45, 0.5, 1.3], [1.5, 0.5, 1.3], [0.45, 0.5, 0.1],
        [2.5, 0.5, np.nan],
    ])
    labels = point_instances.label_crown_and_stem_instances(
        cloud, np.array([[7, 7, 8]]), Affine.identity(),
        {7: (0.5, 0.5, 15), 8: (2.5, 0.5, 15)}, 2.0,
        stem_positions=[(0.5, 0.5)],
    )
    np.testing.assert_array_equal(labels, [7, 7, 8, 7, 0, 0, 0])


@pytest.mark.parametrize("tool_name", ("pycrown", "watershed"))
def test_segmentation_preserves_unassigned_points_in_full_las(tmp_path, tool_name):
    from tools.Segmentation.pycrown_segmentation import segment_trees_pycrown
    from tools.Segmentation.watershed_segmentation import segment_trees_watershed

    source = tmp_path / "cloud.las"
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(CRS.from_epsg(32650))
    las = laspy.LasData(header)
    las.x, las.y, las.z = [500000.0, 500000.5], [6000000.0, 6000000.5], [100.0, 108.0]
    las.intensity = [10, 20]
    las.classification = [2, 1]
    las.return_number = [1, 1]
    las.write(source)
    tool = {"pycrown": segment_trees_pycrown, "watershed": segment_trees_watershed}[tool_name]
    out = tmp_path / "result"
    result = asyncio.run(tool(str(source), use_pointcloud_supplement=False, output_path=str(out)))
    assert result["status"] == "success", result
    assert result["data"]["point_instance_export"]["unassigned_points"] == 2
    with ZipFile(result["output_path"]) as archive:
        assert "tree_instances.las" in archive.namelist()
    exported = laspy.read(out / "tree_instances.las")
    np.testing.assert_array_equal(exported.intensity, las.intensity)
    np.testing.assert_array_equal(exported.lidaragent_tree_id, [0, 0])
    np.testing.assert_allclose(exported.lidaragent_hag, [0, 8])
