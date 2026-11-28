import laspy
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin

from tools.Segmentation.spatial_inputs import validate_chm, validate_point_cloud_alignment


def _attachments(tmp_path, *names):
    files = []
    for index, filename in enumerate(names):
        path = tmp_path / f"{index}_{filename}"
        path.touch()
        files.append({"filename": filename, "file_path": str(path)})
    return files


@pytest.mark.parametrize("parser_name", ("parse_watershed_args", "parse_pycrown_args"))
def test_chat_selects_chm_by_role(tmp_path, parser_name):
    from app.services.mcp_service import MCPService

    parse = getattr(MCPService(), parser_name)
    files = _attachments(tmp_path, "plot_DSM.tif", "plot_DTM.tif", "plot_CHM.tif", "plot.las")
    arguments = parse(files, {})
    assert arguments["file_path"] == files[2]["file_path"]
    assert arguments["point_cloud_path"] == files[3]["file_path"]
    if parser_name == "parse_pycrown_args":
        assert arguments["dsm_path"] == files[0]["file_path"]
        assert arguments["dtm_path"] == files[1]["file_path"]


@pytest.mark.parametrize("parser_name", ("parse_watershed_args", "parse_pycrown_args"))
@pytest.mark.parametrize("names", (("plot_DSM.tif", "plot_DTM.tif"), ("plot_CHM.tif", "other_CHM.tif")))
def test_chat_rejects_missing_or_duplicate_chm(tmp_path, parser_name, names):
    from app.services.mcp_service import MCPService

    parse = getattr(MCPService(), parser_name)
    with pytest.raises(ValueError, match="CHM"):
        parse(_attachments(tmp_path, *names), {})


@pytest.mark.parametrize("parser_name", ("parse_watershed_args", "parse_pycrown_args"))
@pytest.mark.parametrize("flag", (False, True))
def test_chat_forwards_explicit_occluded_crown_setting(tmp_path, parser_name, flag):
    from app.services.mcp_service import MCPService

    parse = getattr(MCPService(), parser_name)
    files = _attachments(tmp_path, "plot.las")
    key = "recover_occluded_crowns"
    assert key not in parse(files, {})
    assert parse(files, {key: flag})[key] is flag
    with pytest.raises(ValueError, match=key):
        parse(files, {key: "false"})


@pytest.mark.parametrize(
    "epsg, east, expected_error",
    [
        (25832, 500001.0, None),
        (25833, 500001.0, "CRS 不一致"),
        (None, 500001.0, "缺少 CRS"),
        (25832, 501000.0, "空间范围不重叠"),
    ],
)
def test_cloud_alignment_checks_actual_las_header(tmp_path, epsg, east, expected_error):
    chm_path = tmp_path / "plot_CHM.tif"
    transform = from_origin(500000.0, 6000005.0, 0.5, 0.5)
    with rasterio.open(
        chm_path, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32",
        crs="EPSG:25832", transform=transform,
    ) as dataset:
        dataset.write(np.zeros((10, 10), dtype=np.float32), 1)
    las_path = tmp_path / "plot.las"
    header = laspy.LasHeader(point_format=0, version="1.2")
    if epsg is not None:
        header.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    las.x = [east, east + 1]
    las.y = [6000001.0, 6000002.0]
    las.z = [0.0, 10.0]
    las.write(las_path)

    with rasterio.open(chm_path) as dataset:
        crs = validate_chm(dataset, square_pixels=True)
        if expected_error is None:
            validate_point_cloud_alignment(las_path, crs, dataset.bounds)
        else:
            with pytest.raises(ValueError, match=expected_error):
                validate_point_cloud_alignment(las_path, crs, dataset.bounds)
