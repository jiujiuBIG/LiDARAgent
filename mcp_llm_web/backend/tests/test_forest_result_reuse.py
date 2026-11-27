import asyncio
import json
from zipfile import ZipFile

import laspy
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.chat import chat_router, reusable_result_file
from app.core.config import settings
from app.models import database
from app.services.agent_service import agent_service
from app.services.mcp_service import NORMALS_TOOL, PYCROWN_TOOL, WATERSHED_TOOL, mcp_service


@pytest.fixture
def result_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    job_dir = results_dir / "job"
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(settings, "RESULTS_DIR", results_dir)
    return job_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'reuse.db'}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    app = FastAPI()
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
    asyncio.run(database.init_db())
    with TestClient(app) as test_client:
        yield test_client
    asyncio.run(engine.dispose())


def make_result(path, **fields):
    return {"status": "success", "output_path": str(path), **fields}


def tool_message(tool, result):
    return database.MessageModel(role="assistant", tool_results=json.dumps([{"tool": tool, "result": result}]))


@pytest.mark.parametrize("tool", [PYCROWN_TOOL, WATERSHED_TOOL])
def test_forest_instances_are_reusable_in_response_and_history(client, result_dir, monkeypatch, tool):
    archive_path = result_dir / "canopy.zip"
    instance_path = result_dir / "tree_instances.las"
    cloud = laspy.create(point_format=3, file_version="1.2")
    cloud.add_extra_dims([
        laspy.ExtraBytesParams(name="lidaragent_tree_id", type="int32"),
        laspy.ExtraBytesParams(name="lidaragent_hag", type="float64"),
    ])
    cloud.x = [1, 2]
    cloud.y = [1, 2]
    cloud.z = [100, 102]
    cloud["lidaragent_tree_id"] = [1, 1]
    cloud["lidaragent_hag"] = [1.3, 3.3]
    cloud.write(instance_path)
    (result_dir / "trees.las").write_bytes(b"legacy crown points")
    with ZipFile(archive_path, "w") as archive:
        archive.write(instance_path, instance_path.name)
    result = make_result(archive_path, data={"point_instance_export": {"filename": instance_path.name}})
    seen = []

    async def process_message(message, history, file_context):
        if file_context is None:
            return {"content": "segmentation result", "tool_results": [{"tool": tool, "result": result}]}
        seen.extend(file_context)
        args = mcp_service.parse_tool_args("extract_forest_dbh", message, {}, file_context)
        assert args["file_path"] == str(instance_path)
        return {"content": "measurement result"}

    monkeypatch.setattr(agent_service, "process_message", process_message)
    base = f"{settings.API_V1_PREFIX}/chat"
    send = f"{base}/send"
    creation = client.post(send, json={"message": "segment trees"})
    assert creation.status_code == 200
    body = creation.json()
    session_id = body["session_id"]
    session_url = f"{base}/sessions/{session_id}"
    history = client.get(session_url).json()["messages"]
    assert history[0]["reusable_result"] is None
    message_id = history[1]["id"]
    metadata = {"message_id": message_id, "filename": "tree_instances.las", "kind": "tree_instances"}
    assert body["reusable_result"] == history[1]["reusable_result"] == metadata
    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert download.content == archive_path.read_bytes()
    assert history[1]["download_url"] == body["download_url"]
    selection = {"message": "DBH", "session_id": session_id, "result_message_id": message_id}
    wrong_session = client.post(send, json={**selection, "session_id": "other"})
    assert wrong_session.status_code == 404
    assert client.get(body["download_url"].replace(session_id, "other")).status_code == 404
    mixed = client.post(send, json={**selection, "file_ids": ["upload"]})
    assert mixed.status_code == 400
    continued = client.post(send, json=selection)
    assert continued.status_code == 200
    assert continued.json()["used_files"] == [f"result:{message_id}"]
    assert seen == [{
        "id": f"result:{message_id}",
        "filename": "tree_instances.las",
        "file_type": "las",
        "file_path": str(instance_path),
        "forest_input_kind": "tree_instances",
    }]
    instance_path.unlink()
    assert client.post(send, json=selection).status_code == 404
    stored_result = client.get(session_url).json()["messages"][1]
    assert stored_result["reusable_result"] is None
    assert stored_result["download_url"] == body["download_url"]


@pytest.mark.parametrize("direct", [False, True])
def test_historical_pycrown_points_allow_crown_but_reject_dbh(result_dir, direct):
    crown_path = result_dir / "trees.las"
    crown_path.touch()
    output = crown_path if direct else result_dir / "history.zip"
    output.touch()
    assert reusable_result_file(tool_message(PYCROWN_TOOL, make_result(output))) == (crown_path, "crown_points")
    file_context = [{"file_path": str(crown_path), "forest_input_kind": "crown_points"}]
    assert mcp_service.parse_tool_args("extract_tree_crown", "冠幅", {}, file_context)["file_path"] == str(crown_path)
    with pytest.raises(ValueError, match="树干点"):
        mcp_service.parse_tool_args("extract_forest_dbh", "胸径", {}, file_context)


def test_normals_remain_reusable_as_point_cloud(result_dir):
    point_cloud = result_dir / "normals.las"
    point_cloud.touch()
    assert reusable_result_file(tool_message(NORMALS_TOOL, make_result(point_cloud))) == (point_cloud, "point_cloud")


@pytest.mark.parametrize("issue", [
    "missing_zip", "outside_zip", "failed", "missing_instance", "instance_is_directory", "chm_only", "raster_only",
])
@pytest.mark.parametrize("tool", [PYCROWN_TOOL, WATERSHED_TOOL])
def test_unavailable_segmentations_offer_no_reusable_result(tmp_path, result_dir, tool, issue):
    directory = tmp_path if issue == "outside_zip" else result_dir
    archive = directory / ("canopy.tif" if issue == "raster_only" else "canopy.zip")
    if issue != "missing_zip":
        archive.touch()
    instance = directory / "tree_instances.las"
    if issue == "instance_is_directory":
        instance.mkdir()
    elif issue != "missing_instance":
        instance.touch()
    if issue in {"chm_only", "missing_instance"}:
        (directory / "trees.las").touch()
    result = make_result(archive, status="error" if issue == "failed" else "success", data={
        "point_instance_export": None if issue == "chm_only" else {"filename": "tree_instances.las"},
    })
    assert reusable_result_file(tool_message(tool, result)) is None


def test_agent_forwards_raster_product_resolution_and_crs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.services.llm_service import llm_service

    cloud = tmp_path / "plot.las"
    cloud.touch()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(settings, "RESULTS_DIR", results_dir)
    calls = []

    class RecordingSession:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({
                "status": "success",
                "output_path": str(tmp_path / "plot_CHM.tif"),
                "message": "ok",
                "data": {"resolution": arguments.get("resolution", 0.5), "crs": arguments.get("input_crs")},
            }))])

    monkeypatch.setattr(mcp_service, "all_tools", {"generate_chm": {"session": RecordingSession()}})

    async def fake_complete(messages):
        if len(messages) == 2 and messages[0]["content"].startswith("You are a router"):
            return {"content": json.dumps({
                "tool_name": "generate_chm",
                "reason": "生成 CHM",
                "arguments": {
                    "resolution": 1.0,
                    "input_crs": "EPSG:32650",
                    "cloth_resolution": 0.2,
                    "file_path": "evil.las",
                    "output_path": "evil",
                },
            })}
        return {"content": "完成"}

    monkeypatch.setattr(llm_service, "chat_complete", fake_complete)
    response = asyncio.run(agent_service.process_message(
        "按 1 米分辨率生成 CHM，坐标系 EPSG:32650",
        [],
        [{"id": "u1", "file_path": str(cloud), "filename": cloud.name}],
    ))
    assert response["tool_results"][0]["result"]["status"] == "success"
    (tool_name, actual), = calls
    assert tool_name == "generate_chm"
    assert actual["file_path"] == str(cloud)
    assert actual["resolution"] == 1.0
    assert actual["input_crs"] == "EPSG:32650"
    assert actual["cloth_resolution"] == 0.2
    assert results_dir in (results_dir, *tmp_path.parents)
    assert str(actual["output_path"]).startswith(str(results_dir))

    with pytest.raises(ValueError, match="resolution"):
        mcp_service.parse_raster_product_args(
            "generate_dsm", [{"file_path": str(cloud)}], {"resolution": -0.5}
        )
    with pytest.raises(ValueError, match="input_crs"):
        mcp_service.parse_raster_product_args(
            "generate_dsm", [{"file_path": str(cloud)}], {"input_crs": "  "}
        )
