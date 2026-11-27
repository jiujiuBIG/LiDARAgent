import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.agent_service import agent_service
from app.services.llm_service import llm_service
from app.services.mcp_service import MCPService, NORMALS_TOOL, mcp_service, settings


class RecordingSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        result = {
            "status": "success",
            "output_path": str(Path(arguments["output_path"]) / "a_normals.las"),
            "message": "ok",
            "data": {
                key: arguments[key]
                for key in ("radius", "max_nn", "orientation", "label_schema", "tile_size")
                if key in arguments
            },
        }
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(result))])


@pytest.mark.parametrize("message,requested,expected_parameters", [
    ("半径 0.35，最多 20 个邻居，计算法向量。", {}, {"orientation": "none"}),
    ("半径 0.35，最多 20 个邻居，朝上。", {"orientation": "up"}, {"orientation": "up"}),
    (
        "半径 0.35，最多 20 个邻居，分块宽度 16 米。",
        {"tile_size": 16},
        {"orientation": "none", "tile_size": 16.0},
    ),
    (
        "半径 0.35，最多 20 个邻居，用 FORinstance 标签让树干朝外。",
        {"orientation": "stem_outward", "label_schema": "forinstance"},
        {"orientation": "stem_outward", "label_schema": "forinstance"},
    ),
])
def test_chat_passes_normal_parameters_and_controls_paths(
    tmp_path, monkeypatch, message, requested, expected_parameters
):
    source = tmp_path / "a.las"
    source.touch()
    results = tmp_path / "results"
    session = RecordingSession()
    monkeypatch.setattr(settings, "RESULTS_DIR", results)
    monkeypatch.setattr(mcp_service, "all_tools", {NORMALS_TOOL: {"session": session}})

    async def complete(messages):
        if len(messages) == 2 and messages[0]["content"].startswith("You are a router"):
            return {"content": json.dumps({
                "tool_name": NORMALS_TOOL,
                "reason": "用户要求估计法向量",
                "arguments": {
                    "radius": 0.35, "max_nn": 20, **requested,
                    "file_path": "wrong.las", "output_path": "wrong",
                },
            })}
        return {"content": "处理完成"}

    monkeypatch.setattr(llm_service, "chat_complete", complete)
    result = asyncio.run(agent_service.process_message(
        message,
        [],
        [{"id": "upload", "file_path": str(source), "filename": source.name}],
    ))
    (tool_name, actual), = session.calls
    assert tool_name == NORMALS_TOOL
    assert actual["file_path"] == str(source)
    assert Path(actual["output_path"]).parent == results
    expected = {"radius": 0.35, "max_nn": 20, **expected_parameters}
    assert result["tool_data"] == expected
    assert result["tool_results"][0]["arguments"] == expected
    assert Path(result["output_path"]).parent == Path(actual["output_path"])


def test_chat_suggests_normal_parameters_before_an_explicit_followup(tmp_path, monkeypatch):
    source = tmp_path / "a.las"
    source.touch()
    files = [{"id": "upload", "file_path": str(source), "filename": source.name}]
    session = RecordingSession()
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(mcp_service, "all_tools", {NORMALS_TOOL: {"session": session}})
    first_message = "请计算这份点云的法向量。"
    confirmed_message = "确认坐标单位是米，用 0.25 米半径、最多 64 个邻居计算法向量。"

    async def complete(messages):
        if messages[0]["content"].startswith("You are a router"):
            request = json.loads(messages[1]["content"])
            return {"content": json.dumps({
                "tool_name": NORMALS_TOOL,
                "reason": "用户要求估计法向量",
                "arguments": (
                    {"radius": 0.25, "max_nn": 64} if request["user_message"] == confirmed_message else {}
                ),
            })}
        return {"content": "处理完成"}

    monkeypatch.setattr(llm_service, "chat_complete", complete)
    first = asyncio.run(agent_service.process_message(first_message, [], files))
    assert not session.calls
    assert first["output_path"] is None
    assert "请确认搜索半径" in first["content"]
    assert "坐标单位是米" in first["content"]
    for hint in ("0.25 米", "混合点云 32", "平整地面 128", "移动扫描树干 64", "细枝与树冠试跑 16"):
        assert hint in first["content"]

    history = [{"role": "user", "content": first_message}, {"role": "assistant", "content": first["content"]}]
    result = asyncio.run(agent_service.process_message(confirmed_message, history, files))
    (_, actual), = session.calls
    assert actual["radius"] == 0.25
    assert actual["max_nn"] == 64
    assert actual["file_path"] == str(source)
    assert result["output_path"]


@pytest.mark.parametrize("arguments", [
    {}, {"radius": 0}, {"radius": float("nan")}, {"radius": True},
    {"radius": 1, "max_nn": 11}, {"radius": 1, "max_nn": 12.5},
    {"radius": 1, "orientation": "side"},
    {"radius": 1, "orientation": "stem_outward"},
    {"radius": 1, "orientation": "stem_outward", "label_schema": "unknown"},
    {"radius": 1, "tile_size": 0}, {"radius": 1, "tile_size": True},
    {"radius": 1, "tile_size": float("inf")},
])
def test_normal_tool_rejects_invalid_parameters_without_call(tmp_path, arguments):
    service = MCPService()
    session = RecordingSession()
    service.all_tools[NORMALS_TOOL] = {"session": session}
    source = tmp_path / "source.laz"
    source.touch()
    result = asyncio.run(service.call_tool(
        NORMALS_TOOL, "估计法向量", [{"file_path": str(source)}], arguments,
    ))
    assert result["status"] == "error"
    assert not session.calls
