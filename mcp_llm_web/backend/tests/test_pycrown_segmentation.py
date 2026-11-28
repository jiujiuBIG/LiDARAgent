import asyncio
import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest
import rasterio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pyproj import CRS
from rasterio.transform import from_origin
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.chat import chat_router
from app.core.config import settings
from app.models import database
from app.services.llm_service import llm_service
from app.services.mcp_service import PYCROWN_TOOL, mcp_service
from tools.Segmentation.pycrown_segmentation import segment_trees_pycrown
from tools.Segmentation.watershed_segmentation import segment_trees_watershed


def _write_synthetic_las(path: Path):
    """Create a synthetic LAS file with ground points and two distinct conical crowns."""
    rng = np.random.default_rng(7)
    gx = rng.uniform(500000.0, 500015.0, 400)
    gy = rng.uniform(6000000.0, 6000015.0, 400)
    gz = np.zeros_like(gx)
    gcls = np.full(len(gx), 2, dtype=np.uint8)

    dx, dy = np.meshgrid(np.arange(-1.5, 1.6, 0.2), np.arange(-1.5, 1.6, 0.2))
    dx, dy = dx.ravel(), dy.ravel()
    radii = np.hypot(dx, dy)
    keep = radii <= 1.5
    dx, dy, radii = dx[keep], dy[keep], radii[keep]
    t1x, t1y, t1z = 500004.5 + dx, 6000004.5 + dy, 18.0 - 2.0 * radii
    t2x, t2y, t2z = 500010.5 + dx, 6000010.5 + dy, 15.0 - 2.0 * radii

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.add_crs(CRS.from_epsg(25832))
    header.offsets = [500000.0, 6000000.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = np.concatenate([gx, t1x, t2x])
    las.y = np.concatenate([gy, t1y, t2y])
    las.z = np.concatenate([gz, t1z, t2z])
    las.classification = np.concatenate([gcls, np.ones(len(t1x) + len(t2x), dtype=np.uint8)])
    las.return_number = np.ones(len(las.x), dtype=np.uint8)
    las.write(path)


def test_pycrown_tool_segments_synthetic_las_and_packs_outputs(tmp_path):
    las_path = tmp_path / "synthetic_plot.las"
    _write_synthetic_las(las_path)
    out_dir = tmp_path / "out_job"

    result = asyncio.run(
        segment_trees_pycrown(
            str(las_path),
            sigma_m=0.35,
            min_distance_m=0.75,
            min_height_m=4.0,
            min_canopy_height_m=2.0,
            crown_radius_m=6.0,
            use_pointcloud_supplement=False,
            export_las=True,
            output_path=str(out_dir),
        )
    )
    assert result["status"] == "success", result
    assert result["data"]["tree_count"] == 2
    assert result["data"]["smooth_crowns_exported"] is True
    assert result["data"]["classified_lidar_points"] > 0
    zip_path = Path(result["output_path"])
    assert zip_path.is_file() and zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert {
        "crown_labels.tif",
        "crowns.gpkg",
        "crowns_smooth.gpkg",
        "treetops.gpkg",
        "treetops.csv",
        "trees.las",
        "summary.json",
    }.issubset(names)
    with laspy.open(out_dir / "trees.las") as output:
        assert output.header.parse_crs().to_epsg() == 25832


@pytest.mark.parametrize("segment", (segment_trees_pycrown, segment_trees_watershed))
@pytest.mark.parametrize("flag", (None, False, True))
def test_occluded_recovery_switch_keeps_the_instance_summary(tmp_path, segment, flag):
    source = tmp_path / "plot.las"
    _write_synthetic_las(source)
    kwargs = {} if flag is None else {"recover_occluded_crowns": flag}
    result = asyncio.run(segment(str(source), output_path=str(tmp_path / "result"), **kwargs))

    assert result["status"] == "success", result
    data = result["data"]
    assert data.get("settings", data).get("recover_occluded_crowns") is (flag is not False)
    instance = data["point_instance_export"]
    assert instance["summary_filename"] == "tree_summary.csv"
    assert instance["occluded_tree_count"] == 0
    with zipfile.ZipFile(result["output_path"]) as archive:
        rows = list(csv.DictReader(archive.read("tree_summary.csv").decode("utf-8").splitlines()))
        assert instance["filename"] in archive.namelist()
    assert len(rows) == instance["labelled_trees"] == 2
    assert sum(int(row["point_count"]) for row in rows) == instance["labelled_points"]
    assert all(not row["parent_2d_crown_id"] for row in rows)


@pytest.mark.skipif(
    not Path("D:/code/testdata_chm_seg/c02_forinst_nibio_p1/chm_0p50m.tif").is_file(),
    reason="c02 test fixture is not present on this machine",
)
def test_pycrown_tool_recovers_understory_trees_on_c02(tmp_path):
    c02_chm = Path("D:/code/testdata_chm_seg/c02_forinst_nibio_p1/chm_0p50m.tif")
    out_dir = tmp_path / "c02_pycrown_job"

    result = asyncio.run(
        segment_trees_pycrown(
            str(c02_chm),
            use_pointcloud_supplement=True,
            export_las=False,
            output_path=str(out_dir),
        )
    )
    assert result["status"] == "success", result
    data = result["data"]
    assert data["pointcloud_supplement_used"] is True
    assert data["smooth_crowns_exported"] is True
    assert data["tree_count"] >= 27


class RecordingPyCrownSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        out_dir = Path(arguments["output_path"])
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_file = out_dir / "plot_pycrown_crowns.zip"
        zip_file.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        result = {
            "status": "success",
            "output_path": str(zip_file),
            "message": "PyCrown 单木分割完成，共分割出 29 棵单木。",
            "data": {
                "tree_count": 29,
                "moved_tops": 0,
                "sigma_m": arguments["sigma_m"],
                "min_distance_m": arguments["min_distance_m"],
                "min_height_m": arguments["min_height_m"],
                "min_canopy_height_m": arguments["min_canopy_height_m"],
                "crown_radius_m": arguments["crown_radius_m"],
                "chm_resolution_m": arguments["chm_resolution_m"],
                "pointcloud_supplement_used": arguments["use_pointcloud_supplement"],
            },
        }
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(result))])


def test_agent_routes_pycrown_request_and_enables_download(tmp_path, monkeypatch):
    source = tmp_path / "plot.laz"
    source.touch()
    results = tmp_path / "results"
    session = RecordingPyCrownSession()

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'pycrown_chat.db'}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "RESULTS_DIR", results)
    monkeypatch.setattr(mcp_service, "all_tools", {PYCROWN_TOOL: {"session": session}})

    async def complete(messages):
        if len(messages) == 2 and messages[0]["content"].startswith("You are a router"):
            return {
                "content": json.dumps(
                    {
                        "tool_name": PYCROWN_TOOL,
                        "reason": "用户要求 PyCrown 单木分割",
                        "arguments": {
                            "sigma_m": 0.35,
                            "min_distance_m": 0.75,
                            "min_height_m": 4.0,
                            "min_canopy_height_m": 2.0,
                            "crown_radius_m": 6.0,
                            "use_pointcloud_supplement": True,
                        },
                    }
                )
            }
        return {"content": "已完成 PyCrown 单木分割，共识别出 29 棵单木。"}

    monkeypatch.setattr(llm_service, "chat_complete", complete)

    app = FastAPI()
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
    asyncio.run(database.init_db())
    file_id = "test-pycrown-file"
    asyncio.run(
        database.db_service.add_file(
            file_id=file_id,
            filename=source.name,
            file_type="laz",
            file_size=100,
            file_path=str(source),
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/send",
            json={"message": "请使用 PyCrown 算法对这份点云进行单木树冠分割", "file_ids": [file_id]},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["tool_data"]["tree_count"] == 29
        assert payload["visualizations"][0]["title"] == "🌲 PyCrown 单木分割结果"
        assert payload["download_url"] is not None
        dl = client.get(payload["download_url"])
        assert dl.status_code == 200
        assert "plot_pycrown_crowns.zip" in dl.headers["content-disposition"]

    asyncio.run(engine.dispose())
