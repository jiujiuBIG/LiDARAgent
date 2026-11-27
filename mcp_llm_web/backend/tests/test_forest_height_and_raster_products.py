import asyncio
from pathlib import Path

import laspy
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from tools.Product import chm, dsm, dtm
from tools.Segmentation.pycrown_segmentation import _load_grid_from_raster, segment_trees_pycrown
from tools.Segmentation.pycrown_v1 import _load_point_cloud_arrays, build_grid_from_las


def write_las(path, points, classification, hag=None, include_crs=True):
    header = laspy.LasHeader(point_format=0, version="1.2")
    if include_crs:
        header.add_crs(CRS.from_epsg(32650))
    las = laspy.LasData(header)
    las.x, las.y, las.z = np.asarray(points).T
    las.classification = classification
    las.return_number = np.ones(len(points), dtype=np.uint8)
    if hag is not None:
        las.add_extra_dim(laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32))
        las["HeightAboveGround"] = hag
    las.write(path)


def write_grid(path, values):
    with rasterio.open(
        path, "w", driver="GTiff", width=3, height=3, count=1,
        dtype="float32", crs="EPSG:32650", nodata=np.nan,
        transform=from_origin(500000, 6000003, 1, 1),
    ) as target:
        target.write(np.asarray(values, dtype=np.float32), 1)


def test_pycrown_chm_without_ground_does_not_treat_absolute_z_as_height(tmp_path):
    canopy = tmp_path / "site_CHM.tif"
    cloud = tmp_path / "site.las"
    write_grid(canopy, np.full((3, 3), 12))
    write_las(cloud, [(500001.5, 6000001.5, 100), (500001.5, 6000001.5, 112)], [2, 1])
    grid = _load_grid_from_raster(canopy, None, None, cloud)
    assert np.isnan(_load_point_cloud_arrays(grid)[-1]).all()
    result = asyncio.run(segment_trees_pycrown(
        str(canopy), point_cloud_path=str(cloud), output_path=str(tmp_path / "output")
    ))
    assert result["status"] == "success", result
    assert result["data"]["pointcloud_supplement_used"] is False
    assert result["data"]["pointcloud_height_reference"] is None


def test_pycrown_dtm_alone_as_auxiliary_and_nodata(tmp_path):
    canopy, terrain, cloud = (tmp_path / name for name in ("site_CHM.tif", "site_DTM.tif", "site.las"))
    write_grid(canopy, np.full((3, 3), 12))
    heights = np.full((3, 3), 100.0)
    heights[1, 2] = np.nan
    write_grid(terrain, heights)
    write_las(cloud, [
        (500001.5, 6000001.5, 100), (500001.5, 6000001.5, 112), (500002.5, 6000001.5, 112),
    ], [2, 1, 1])
    grid = _load_grid_from_raster(canopy, None, str(terrain), cloud)
    np.testing.assert_allclose(_load_point_cloud_arrays(grid)[-1], [0, 12, np.nan], equal_nan=True)
    raw_grid = build_grid_from_las(cloud, resolution_m=1)
    assert raw_grid.ground_reference == "classified_ground"
    np.testing.assert_allclose(_load_point_cloud_arrays(raw_grid)[-1][:2], [0, 12])


def test_pycrown_accepts_explicit_height_and_discards_heuristic_reference(tmp_path):
    canopy, cloud = tmp_path / "site_CHM.tif", tmp_path / "site.las"
    write_grid(canopy, np.full((3, 3), 12))
    points = [(500000.5, 6000001.5, 100), (500001.5, 6000001.5, 112)]
    write_las(cloud, points, [1, 1], hag=[0, 12])
    grid = _load_grid_from_raster(canopy, None, None, cloud)
    np.testing.assert_allclose(_load_point_cloud_arrays(grid)[-1], [0, 12])
    unclassified = tmp_path / "raw.las"
    write_las(unclassified, points, [1, 1])
    assert np.isnan(_load_point_cloud_arrays(build_grid_from_las(unclassified))[-1]).all()


def test_three_products_are_aligned_and_have_north_up_values(tmp_path, monkeypatch):
    cloud = tmp_path / "plot.las"
    write_las(cloud, [
        (500000, 6000000, 100), (500002, 6000000, 100),
        (500000, 6000002, 104), (500002, 6000002, 104),
        (500000.5, 6000001.5, 112),
    ], [2, 2, 2, 2, 1])
    monkeypatch.setattr(chm, "_run_csf_in_subprocess", lambda *_: ([0, 1, 2, 3], [4]))
    monkeypatch.setattr(dtm, "_run_csf_in_subprocess", lambda *_: [0, 1, 2, 3])
    results = [
        asyncio.run(tool(str(cloud), str(tmp_path), resolution=1.0))
        for tool in (chm.generate_chm, dsm.generate_dsm, dtm.generate_dtm)
    ]
    assert all(result["status"] == "success" for result in results), results
    arrays = []
    for result in results:
        with rasterio.open(result["output_path"]) as source:
            assert source.crs.to_epsg() == 32650
            assert source.shape == (3, 3)
            assert source.transform == from_origin(500000, 6000002, 1, 1)
            assert np.isnan(source.nodata)
            arrays.append(source.read(1, masked=True))
    canopy, surface, terrain = arrays
    assert terrain[0, 0] == pytest.approx(103)
    assert terrain[1, 0] == pytest.approx(101)
    assert surface[0, 0] == pytest.approx(112)
    assert canopy[0, 0] == pytest.approx(9)
    assert canopy.mask[2, 2]
    assert all(Path(item["output_path"]).is_file() for item in results)
    canopy_path, surface_path, terrain_path = [item["output_path"] for item in results]
    grid = _load_grid_from_raster(Path(canopy_path), surface_path, terrain_path, cloud)
    assert _load_point_cloud_arrays(grid)[-1][-1] == pytest.approx(9)


def test_product_without_known_crs_does_not_write_tiff(tmp_path):
    cloud = tmp_path / "missing_crs.las"
    write_las(cloud, [(0, 0, 1), (0, 2, 1), (2, 0, 1)], [2, 2, 2], include_crs=False)
    result = asyncio.run(dsm.generate_dsm(str(cloud), str(tmp_path)))
    assert result["status"] == "error"
    assert "CRS" in result["message"]
    assert not list(tmp_path.glob("*.tif"))

    explicit = asyncio.run(dsm.generate_dsm(str(cloud), str(tmp_path), input_crs="EPSG:32650"))
    assert explicit["status"] == "success", explicit
    with rasterio.open(explicit["output_path"]) as source:
        assert source.crs.to_epsg() == 32650


def test_pycrown_zip_excludes_unrelated_files_in_output_dir(tmp_path):
    from zipfile import ZipFile

    canopy = tmp_path / "site_CHM.tif"
    write_grid(canopy, np.full((3, 3), 12))
    out_dir = tmp_path / "shared_out"
    out_dir.mkdir()
    unrelated = out_dir / "unrelated_note.txt"
    unrelated.write_text("do not archive", encoding="utf-8")

    result = asyncio.run(segment_trees_pycrown(str(canopy), output_path=str(out_dir)))
    assert result["status"] == "success", result
    with ZipFile(result["output_path"]) as zf:
        assert "unrelated_note.txt" not in zf.namelist()
        assert "summary.json" in zf.namelist()
