"""Unit tests for SegmentAnyTree MCP segmentation tool and MCPService wiring."""

from __future__ import annotations

import asyncio
import json
import sys
import types
import zipfile
from pathlib import Path

import laspy
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import mcp.server.fastmcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    # Lightweight stub so pure-logic unit tests run even without full backend MCP deps
    mcp_pkg = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    mcp_fast = types.ModuleType("mcp.server.fastmcp")

    class _DummyFastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self, *args, **kwargs):
            def _dec(fn):
                return fn
            return _dec

        def run(self, *args, **kwargs):
            pass

    mcp_fast.FastMCP = _DummyFastMCP
    sys.modules["mcp"] = mcp_pkg
    sys.modules["mcp.server"] = mcp_server
    sys.modules["mcp.server.fastmcp"] = mcp_fast

try:
    from app.services.mcp_service import MCPService
    HAS_APP = True
except ImportError:
    MCPService = None
    HAS_APP = False

from tools.Segmentation import segmentanytree_segmentation as sat_mod


def _write_synthetic_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500000.0, 6000000.0, 100.0])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(path)


def test_stage_and_remap_roundtrip_with_duplicates_and_min_points(tmp_path: Path):
    # 2 trees (30 pts, 25 pts), 1 tiny fragment (3 pts), plus ground (10 pts) + 2 exact duplicate points
    rng = np.random.default_rng(42)
    t1 = np.column_stack([500001 + rng.normal(0, 0.2, 30), 6000002 + rng.normal(0, 0.2, 30), 102 + rng.uniform(0, 12, 30)])
    t2 = np.column_stack([500009 + rng.normal(0, 0.2, 25), 6000008 + rng.normal(0, 0.2, 25), 101 + rng.uniform(0, 9, 25)])
    frag = np.column_stack([500005 + rng.normal(0, 0.05, 3), 6000005 + rng.normal(0, 0.05, 3), 101 + rng.uniform(0, 2, 3)])
    gnd = np.column_stack([500000 + np.linspace(0.0, 10.0, 10), 6000000 + np.linspace(0.0, 10.0, 10), np.full(10, 100.0)])
    xyz = np.vstack([t1, t2, frag, gnd, t1[:2]])  # 70 pts total, 2 exact duplicates

    src = tmp_path / "forest.las"
    _write_synthetic_las(src, xyz)

    staged_las = tmp_path / "staged.las"
    inverse, stage_stats = sat_mod._stage_point_cloud(src, staged_las, voxel_dedup_m=0.0)
    assert stage_stats["source_points"] == 70
    assert stage_stats["staged_points"] == 68

    # Simulate SegmentAnyTree output with shuffled point order
    staged = laspy.read(staged_las)
    n_staged = len(staged.points)
    # Assign tree labels based on x coordinate in staged cloud
    inst = np.zeros(n_staged, dtype=np.int32)
    sem = np.zeros(n_staged, dtype=np.uint8)
    sx, sz = np.asarray(staged.x), np.asarray(staged.z)
    inst[(sx < 4.0) & (sz > 0.5)] = 10
    inst[(sx > 6.5) & (sz > 0.5)] = 20
    inst[(sx >= 4.0) & (sx <= 6.5) & (sz > 0.5)] = 30
    sem[inst > 0] = 1

    perm = rng.permutation(n_staged)
    sat_out = tmp_path / "sat_out.las"
    out_hdr = laspy.LasHeader(point_format=6, version="1.4")
    out_hdr.scales = staged.header.scales
    out_cloud = laspy.LasData(out_hdr)
    out_cloud.X = staged.X[perm]
    out_cloud.Y = staged.Y[perm]
    out_cloud.Z = staged.Z[perm]
    out_cloud.add_extra_dim(laspy.ExtraBytesParams(name="source_index", type="u4"))
    out_cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type="i4"))
    out_cloud.add_extra_dim(laspy.ExtraBytesParams(name="PredSemantic", type="u1"))
    out_cloud["source_index"] = np.asarray(staged["source_index"])[perm]
    out_cloud["PredInstance"] = inst[perm]
    out_cloud["PredSemantic"] = sem[perm]
    out_cloud.write(sat_out)

    out_las = tmp_path / "final.las"
    out_csv = tmp_path / "trees.csv"
    export_stats = sat_mod._remap_and_export(
        src, sat_out, inverse, n_staged, out_las, out_csv, min_instance_points=5
    )

    assert export_stats["tree_count"] == 2
    assert export_stats["filtered_small_instances"] == 1
    final_cloud = laspy.read(out_las)
    assert len(final_cloud.points) == 70
    assert "treeID" in final_cloud.point_format.dimension_names
    assert set(np.unique(final_cloud["treeID"]).tolist()) == {0, 1, 2}
    # Duplicate points must receive identical treeID
    assert int(final_cloud["treeID"][0]) == int(final_cloud["treeID"][-2])
    assert out_csv.is_file()


def test_mcp_tool_end_to_end_mocked_and_low_density_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Sparse cloud: 20 points spread over 100m x 100m -> 0.002 pts/m^2 (< 15 pts/m^2 warning)
    xyz = np.column_stack([
        np.linspace(500000.0, 500100.0, 20),
        np.linspace(6000000.0, 6000100.0, 20),
        np.linspace(100.0, 115.0, 20),
    ])
    src = tmp_path / "sparse.las"
    _write_synthetic_las(src, xyz)

    def fake_remote(local_staged_las: Path, local_output_laz: Path, **kwargs):
        staged = laspy.read(local_staged_las)
        staged.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type="i4"))
        staged.add_extra_dim(laspy.ExtraBytesParams(name="PredSemantic", type="u1"))
        staged["PredInstance"] = np.where(np.arange(len(staged.points)) < 10, 1, 2).astype(np.int32)
        staged["PredSemantic"] = np.ones(len(staged.points), dtype=np.uint8)
        staged.write(local_output_laz)
        return {
            "job_id": "mock_job",
            "remote_host": "mock",
            "docker_image": "segment-any-tree:fixed",
            "used_lowram_utm2local": False,
            "remote_inference_sec": 1,
        }

    monkeypatch.setattr(sat_mod, "_run_remote_segmentanytree", fake_remote)
    out_dir = tmp_path / "out"
    res = asyncio.run(sat_mod.segment_trees_segmentanytree(str(src), output_path=str(out_dir)))

    assert res["status"] == "success"
    assert res["data"]["tree_count"] == 2
    assert res["data"]["density_warning"] is not None
    assert "segment_trees_pycrown" in res["message"]
    zip_path = Path(res["output_path"])
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert {"sparse_segmentanytree.las", "tree_summary.csv", "summary.json"} <= names
        summary = json.loads(zf.read("summary.json").decode("utf-8"))
        assert summary["tree_count"] == 2


@pytest.mark.skipif(not HAS_APP, reason="Full FastAPI/MCP backend environment not installed in this Python")
def test_mcp_service_parse_segmentanytree_args(tmp_path: Path):
    svc = MCPService()
    las_file = tmp_path / "sample.las"
    las_file.write_bytes(b"placeholder")

    args = svc.parse_segmentanytree_args(
        [{"file_path": str(las_file), "filename": "sample.las"}],
        {"min_instance_points": 50, "voxel_dedup_m": 0.05},
    )
    assert args["file_path"] == str(las_file)
    assert args["min_instance_points"] == 50
    assert args["voxel_dedup_m"] == 0.05

    with pytest.raises(ValueError):
        svc.parse_segmentanytree_args([], {})
    with pytest.raises(ValueError):
        svc.parse_segmentanytree_args([{"file_path": str(las_file)}], {"min_instance_points": -1})
    with pytest.raises(ValueError):
        svc.parse_segmentanytree_args([{"file_path": str(las_file)}], {"voxel_dedup_m": -0.1})
