import asyncio
import io

import laspy
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.chat import chat_router
from app.core.config import settings
from app.models import database
from app.services.agent_service import agent_service
from app.services.mcp_service import NORMALS_TOOL
from tools.Preprocessing.estimate_normals import estimate_point_cloud_normals


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'download.db'}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    app = FastAPI()
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
    asyncio.run(database.init_db())
    with TestClient(app) as test_client:
        yield test_client
    asyncio.run(engine.dispose())


def send_chat(client, monkeypatch, tool_result, tool_name=NORMALS_TOOL):
    async def process_message(**kwargs):
        return {
            "content": tool_result["message"],
            "tool_results": [{"tool": tool_name, "result": tool_result}],
            "output_path": tool_result.get("output_path"),
            "tool_data": tool_result.get("data"),
        }

    monkeypatch.setattr(agent_service, "process_message", process_message)
    response = client.post("/api/v1/chat/send", json={"message": "估计法向量"})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("upload_prefix", ["", "01234567-89ab-4cde-8fab-0123456789ab_"])
def test_chat_and_history_download_actual_las(client, tmp_path, monkeypatch, upload_prefix):
    source = tmp_path / f"{upload_prefix}测试.las"
    las = laspy.create(point_format=3, file_version="1.2")
    las.x = np.tile(np.arange(5), 5)
    las.y = np.repeat(np.arange(5), 5)
    las.z = np.zeros(25)
    las.write(source)
    result = asyncio.run(estimate_point_cloud_normals(
        str(source), 10, 24, output_path=str(settings.RESULTS_DIR / "job"),
    ))
    assert result["status"] == "success"

    chat = send_chat(client, monkeypatch, result)
    url = chat["download_url"]
    assert url.startswith("/api/v1/chat/sessions/")
    assert chat["output_path"] == result["output_path"]
    history = client.get(f"/api/v1/chat/sessions/{chat['session_id']}").json()
    assert history["messages"][0]["download_url"] is None
    assert history["messages"][1]["download_url"] == url

    download = client.get(url)
    assert download.status_code == 200
    assert download.headers["content-disposition"] == "attachment; filename*=utf-8''%E6%B5%8B%E8%AF%95_normals.las"
    restored = laspy.read(io.BytesIO(download.content))
    assert np.array_equal(restored.X, las.X)
    assert np.array_equal(restored.Y, las.Y)
    assert np.array_equal(restored.Z, las.Z)
    assert np.count_nonzero(restored.normal_valid) == result["data"]["valid_count"] > 0

    wrong_session = url.replace(chat["session_id"], "other-session")
    assert client.get(wrong_session).status_code == 404
    client.delete(f"/api/v1/chat/sessions/{chat['session_id']}")
    assert client.get(url).status_code == 404


@pytest.mark.parametrize("kind", ["failed", "outside", "missing", "other_tool"])
def test_unavailable_results_have_no_download(client, tmp_path, monkeypatch, kind):
    file = (tmp_path if kind == "outside" else settings.RESULTS_DIR) / "a_normals.las"
    if kind != "missing":
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    chat = send_chat(client, monkeypatch, {
        "status": "error" if kind == "failed" else "success",
        "message": "tool result",
        "output_path": str(file),
    }, tool_name="extract_tree_height" if kind == "other_tool" else NORMALS_TOOL)

    assert chat["download_url"] is None
    base_url = f"/api/v1/chat/sessions/{chat['session_id']}"
    message = client.get(base_url).json()["messages"][1]
    assert message["download_url"] is None
    assert client.get(f"{base_url}/messages/{message['id']}/download").status_code == 404
