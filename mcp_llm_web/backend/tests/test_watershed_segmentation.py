import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.chat import chat_router
from app.core.config import settings
from app.models import database
from app.services.agent_service import agent_service
from app.services.llm_service import llm_service
from app.services.mcp_service import MCPService, WATERSHED_TOOL, mcp_service
from tools.Segmentation.watershed_segmentation import segment_trees_watershed


def _write_synthetic_chm(path: Path):
    """Create a 0.5 m synthetic CHM with two distinct tree crowns."""
    yy, xx = np.mgrid[0:30, 0:30]
    tree1 = 18.0 * np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 10.0)
    tree2 = 14.0 * np.exp(-((xx - 21) ** 2 + (yy - 21) ** 2) / 8.0)
    chm = np.maximum(tree1, tree2).astype(np.float32)
    chm[chm < 1.5] = 0.0
    transform = from_origin(500000.0, 6000015.0, 0.5, 0.5)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=chm.shape[0],
        width=chm.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:25832",
        transform=transform,
    ) as dst:
        dst.write(chm, 1)


def test_watershed_tool_segments_synthetic_chm_and_packs_outputs(tmp_path):
    chm_path = tmp_path / "synthetic_chm.tif"
    _write_synthetic_chm(chm_path)
    out_dir = tmp_path / "out_job"

    result = asyncio.run(
        segment_trees_watershed(
            str(chm_path),
            sigma_m=0.35,
            min_distance_m=1.0,
            min_canopy_height_m=2.0,
            use_pointcloud_supplement=False,
            output_path=str(out_dir),
        )
    )
    assert result["status"] == "success", result
    assert result["data"]["tree_count"] == 2
    assert result["data"]["chm_seed_count"] == 2
    assert result["data"]["stem_supplemented_count"] == 0
    zip_path = Path(result["output_path"])
    assert zip_path.is_file() and zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert {"chm.tif", "markers.tif", "crowns.tif", "tree_crowns.shp", "tree_crowns.geojson", "treetops.csv", "run.json"}.issubset(names)


@pytest.mark.skipif(
    not Path("D:/code/testdata_chm_seg/c02_forinst_nibio_p1/chm_0p50m.tif").is_file(),
    reason="c02 test fixture is not present on this machine",
)
def test_watershed_tool_recovers_understory_trees_on_c02(tmp_path):
    c02_chm = Path("D:/code/testdata_chm_seg/c02_forinst_nibio_p1/chm_0p50m.tif")
    out_dir = tmp_path / "c02_job"

    result = asyncio.run(
        segment_trees_watershed(
            str(c02_chm),
            sigma_m=0.35,
            min_distance_m=0.75,
            min_canopy_height_m=2.0,
            use_pointcloud_supplement=True,
            output_path=str(out_dir),
        )
    )
    assert result["status"] == "success", result
    data = result["data"]
    assert data["pointcloud_supplement_used"] is True
    assert data["chm_seed_count"] == 13
    assert data["stem_supplemented_count"] == 14
    assert data["tree_count"] == 27


class RecordingWatershedSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        out_dir = Path(arguments["output_path"])
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_file = out_dir / "plot_watershed_crowns.zip"
        zip_file.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        result = {
            "status": "success",
            "output_path": str(zip_file),
            "message": "分水岭单木分割完成，共分割出 26 棵单木。",
            "data": {
                "tree_count": 26,
                "chm_seed_count": 13,
                "stem_supplemented_count": 13,
                "sigma_m": arguments["sigma_m"],
                "min_distance_m": arguments["min_distance_m"],
                "min_canopy_height_m": arguments["min_canopy_height_m"],
                "chm_resolution_m": arguments["chm_resolution_m"],
                "pointcloud_supplement_used": arguments["use_pointcloud_supplement"],
            },
        }
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(result))])


def test_agent_routes_watershed_request_and_enables_download(tmp_path, monkeypatch):
    source = tmp_path / "plot.laz"
    source.touch()
    results = tmp_path / "results"
    session = RecordingWatershedSession()

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'watershed_chat.db'}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "RESULTS_DIR", results)
    monkeypatch.setattr(mcp_service, "all_tools", {WATERSHED_TOOL: {"session": session}})

    async def complete(messages):
        if len(messages) == 2 and messages[0]["content"].startswith("You are a router"):
            return {
                "content": json.dumps(
                    {
                        "tool_name": WATERSHED_TOOL,
                        "reason": "用户要求分水岭单木分割",
                        "arguments": {
                            "sigma_m": 0.35,
                            "min_distance_m": 0.75,
                            "min_canopy_height_m": 2.0,
                            "use_pointcloud_supplement": True,
                        },
                    }
                )
            }
        return {"content": "已完成分水岭单木分割，共识别出 26 棵单木。"}

    monkeypatch.setattr(llm_service, "chat_complete", complete)

    app = FastAPI()
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
    asyncio.run(database.init_db())
    file_id = "test-watershed-file"
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
            json={"message": "请使用分水岭算法对这份点云进行单木树冠分割", "file_ids": [file_id]},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["tool_data"]["tree_count"] == 26
        assert payload["visualizations"][0]["title"] == "🌲 分水岭单木分割结果"
        assert payload["download_url"] is not None
        dl = client.get(payload["download_url"])
        assert dl.status_code == 200
        assert "plot_watershed_crowns.zip" in dl.headers["content-disposition"]

    asyncio.run(engine.dispose())
