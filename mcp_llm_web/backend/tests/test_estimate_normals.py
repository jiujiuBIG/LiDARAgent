import asyncio

import laspy
import numpy as np
import pytest

from tools.Preprocessing.estimate_normals import estimate_point_cloud_normals


def _write_las(path, xyz):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([100.0, 200.0, 3.0])
    header.add_extra_dim(laspy.ExtraBytesParams(name="custom", type=np.uint16))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz.T
    las.intensity = np.arange(len(xyz), dtype=np.uint16) + 10
    las.classification = np.arange(len(xyz), dtype=np.uint8) % 5
    las.custom = np.arange(len(xyz), dtype=np.uint16) + 20
    las.write(path)


def test_plane_normals_and_las_metadata_are_preserved(tmp_path):
    xy = np.array([(x, y) for x in range(5) for y in range(5)], dtype=float)
    xyz = np.column_stack((xy, 2 * xy[:, 0] + 3 * xy[:, 1] + 1))
    source = tmp_path / "plane.las"
    _write_las(source, xyz)

    result = asyncio.run(estimate_point_cloud_normals(str(source), 20.0, 24, "up", str(tmp_path)))
    assert result["status"] == "success"
    with laspy.open(result["output_path"]) as reader:
        out = reader.read()
    expected = np.array([-2.0, -3.0, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.all(out.normal_valid == 1)
    assert np.allclose(np.column_stack((out.normal_x, out.normal_y, out.normal_z)), expected, atol=1e-5)
    assert np.array_equal(out.X, laspy.read(source).X)
    assert np.array_equal(out.intensity, laspy.read(source).intensity)
    assert np.array_equal(out.custom, laspy.read(source).custom)
    assert np.array_equal(out.header.scales, laspy.read(source).header.scales)
    assert np.array_equal(out.header.offsets, laspy.read(source).header.offsets)


def test_duplicate_and_linear_neighborhoods_are_invalid(tmp_path):
    line = np.column_stack((np.arange(15), np.zeros(15), np.zeros(15))).astype(float)
    xyz = np.vstack((np.zeros((2, 3)), line))
    source = tmp_path / "line.las"
    _write_las(source, xyz)
    result = asyncio.run(estimate_point_cloud_normals(str(source), 100.0, 32, "none", str(tmp_path)))
    out = laspy.read(result["output_path"])
    assert np.all(out.normal_valid == 0)
    assert result["data"]["invalid_count"] == len(xyz)


def test_eleven_distinct_neighbors_are_insufficient(tmp_path):
    grid = np.array([(x, y, 0) for x in range(4) for y in range(3)], dtype=float)
    source = tmp_path / "twelve.las"
    _write_las(source, np.vstack((grid, grid)))

    result = asyncio.run(estimate_point_cloud_normals(str(source), 20, 32, "up", str(tmp_path)))

    assert result["data"]["neighbor_shortage_count"] == 24
    assert result["data"]["valid_count"] == 0


def test_output_collision_is_reported_without_overwrite(tmp_path):
    xyz = np.array([(x, y, 0) for x in range(5) for y in range(5)], dtype=float)
    source = tmp_path / "collision.las"
    _write_las(source, xyz)
    existing = tmp_path / "collision_normals.las"
    existing.write_bytes(b"keep")
    result = asyncio.run(estimate_point_cloud_normals(str(source), 10, 24, "up", str(tmp_path)))
    assert result["status"] == "error"
    assert existing.read_bytes() == b"keep"


def test_stem_outward_requires_explicit_schema(tmp_path):
    xyz = np.array([(x, y, 0) for x in range(5) for y in range(5)], dtype=float)
    source = tmp_path / "schema.las"
    _write_las(source, xyz)
    result = asyncio.run(estimate_point_cloud_normals(str(source), 10, 24, "stem_outward", str(tmp_path)))
    assert result["status"] == "error"
    assert "label_schema" in result["message"]


def test_up_adds_orientation_valid_only_when_requested(tmp_path):
    xyz = np.array([(x, y, 0) for x in range(5) for y in range(5)], dtype=float)
    source = tmp_path / "marked.las"
    _write_las(source, xyz)
    result = asyncio.run(estimate_point_cloud_normals(str(source), 10, 24, "up", str(tmp_path)))
    assert result["status"] == "success"
    out = laspy.read(result["output_path"])
    assert "orientation_valid" in out.point_format.dimension_names
    assert np.all(out.orientation_valid == 1)


@pytest.mark.parametrize("schema,label_field", [
    ("forinstance", "classification"),
    ("nibio_mls", "label"),
])
@pytest.mark.parametrize("tile_size", [None, 0.11])
def test_stem_normals_orient_outward_with_explicit_labels(tmp_path, schema, label_field, tile_size):
    azimuth, z = np.meshgrid(np.arange(32) * (2 * np.pi / 32), np.arange(16) * 0.04)
    stem_xyz = np.column_stack((0.14 * np.cos(azimuth.ravel()), 0.14 * np.sin(azimuth.ravel()), z.ravel()))
    branch_xyz = np.array([(x / 20 + 2, y / 20, 0.3) for x in range(6) for y in range(6)])
    source = tmp_path / "labelled.las"
    _write_las(source, np.vstack((stem_xyz, branch_xyz)))
    las = laspy.read(source)
    las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint16))
    las.treeID = np.full(len(las.points), 12, dtype=np.uint16)
    labels = np.repeat([4, 5], [len(stem_xyz), len(branch_xyz)]).astype(np.uint8)
    if label_field == "label":
        las.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8))
        las.label = labels
        las.classification = np.zeros(len(las.points), dtype=np.uint8)
    else:
        las.classification = labels
    las.write(source)

    result = asyncio.run(estimate_point_cloud_normals(
        str(source), 0.13, 32, "stem_outward", str(tmp_path), label_schema=schema,
        tile_size=tile_size,
    ))

    assert result["status"] == "success", result
    out = laspy.read(result["output_path"])
    directed = np.asarray(out.orientation_valid, dtype=bool)
    assert directed.sum() >= 0.9 * len(stem_xyz)
    assert result["data"]["oriented_count"] == directed.sum()
    assert result["data"]["label_schema"] == schema
    assert not np.any(directed[len(stem_xyz):])
    radial = np.column_stack((np.cos(azimuth.ravel()), np.sin(azimuth.ravel())))
    normals = np.column_stack((out.normal_x, out.normal_y))[:len(stem_xyz)]
    assert np.all(np.sum(normals * radial, axis=1)[directed[:len(stem_xyz)]] > 0.95)
    assert np.array_equal(out.treeID, las.treeID)
    assert np.array_equal(np.asarray(out[label_field]), np.asarray(las[label_field]))


def test_outward_request_on_input_without_labels_keeps_basic_normals(tmp_path):
    xyz = np.array([(x, y, 0) for x in range(5) for y in range(5)], dtype=float)
    source = tmp_path / "unlabelled.las"
    _write_las(source, xyz)

    basic = asyncio.run(estimate_point_cloud_normals(
        str(source), 10, 24, output_path=str(tmp_path / "basic"),
    ))
    result = asyncio.run(estimate_point_cloud_normals(
        str(source), 10, 24, "stem_outward", str(tmp_path / "outward"), label_schema="forinstance",
    ))

    assert basic["status"] == result["status"] == "success"
    base = laspy.read(basic["output_path"])
    out = laspy.read(result["output_path"])
    assert "orientation_valid" not in base.point_format.dimension_names
    assert result["data"]["oriented_count"] == 0
    assert "缺少" in result["message"]
    assert not np.any(out.orientation_valid)
    for name in ("normal_x", "normal_y", "normal_z", "normal_valid"):
        assert np.array_equal(np.asarray(out[name]), np.asarray(base[name]))


def test_tiled_normals_match_whole_file_across_tile_boundaries(tmp_path):
    rng = np.random.default_rng(74)
    xy = rng.uniform(0, 1.2, size=(1800, 2))
    xyz = np.column_stack((xy, 0.2 * xy[:, 0] - 0.1 * xy[:, 1]))
    source = tmp_path / "sloped.las"
    _write_las(source, xyz)

    def run(name, **kwargs):
        response = asyncio.run(estimate_point_cloud_normals(
            str(source), 0.17, 32, "up", str(tmp_path / name), **kwargs,
        ))
        assert response["status"] == "success", response
        return laspy.read(response["output_path"]), response["data"]

    whole, whole_data = run("whole")
    tiled, tiled_data = run("tiled", tile_size=0.10)

    assert tiled_data["processing"] == "tiled"
    assert tiled_data["neighbor_shortage_count"] == whole_data["neighbor_shortage_count"]
    assert np.array_equal(tiled.normal_valid, whole.normal_valid)
    assert np.array_equal(tiled.orientation_valid, whole.orientation_valid)
    assert np.count_nonzero(tiled.normal_valid) > 1500
    for field in ("X", "Y", "Z", "intensity", "custom", "classification"):
        assert np.array_equal(np.asarray(tiled[field]), np.asarray(whole[field])), field
    for field in ("normal_x", "normal_y", "normal_z"):
        np.testing.assert_allclose(tiled[field], whole[field], atol=1e-6)
