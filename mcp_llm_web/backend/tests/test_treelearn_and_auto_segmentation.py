"""Unit tests for TreeLearn MCP segmentation tool and auto_segment_trees router."""

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
except ImportError:
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

from tools.Segmentation import auto_segmentation as auto_mod
from tools.Segmentation import treelearn_segmentation as tl_mod


def _write_synthetic_las(path: Path, xyz: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500000.0, 6000000.0, 100.0])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(path)


def test_treelearn_stage_and_remap_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(7)
    t1 = np.column_stack([500001 + rng.normal(0, 0.2, 30), 6000002 + rng.normal(0, 0.2, 30), 102 + rng.uniform(0, 12, 30)])
    t2 = np.column_stack([500009 + rng.normal(0, 0.2, 25), 6000008 + rng.normal(0, 0.2, 25), 101 + rng.uniform(0, 9, 25)])
    frag = np.column_stack([500005 + rng.normal(0, 0.05, 3), 6000005 + rng.normal(0, 0.05, 3), 101 + rng.uniform(0, 2, 3)])
    gnd = np.column_stack([500000 + np.linspace(0.0, 10.0, 10), 6000000 + np.linspace(0.0, 10.0, 10), np.full(10, 100.0)])
    xyz = np.vstack([t1, t2, frag, gnd, t1[:2]])  # 70 pts, 2 exact duplicates

    src = tmp_path / "forest.las"
    _write_synthetic_las(src, xyz)

    staged_las = tmp_path / "plot.las"
    inverse, stage_stats = tl_mod._stage_point_cloud(src, staged_las, voxel_dedup_m=0.0)
    assert stage_stats["source_points"] == 70
    assert stage_stats["staged_points"] == 68

    staged = laspy.read(staged_las)
    staged_xyz = np.column_stack((np.asarray(staged.x), np.asarray(staged.y), np.asarray(staged.z)))
    labels = np.zeros(len(staged_xyz), dtype=np.int32)
    sx, sz = staged_xyz[:, 0], staged_xyz[:, 2]
    labels[(sx < 4.0) & (sz > 0.5)] = 5
    labels[(sx > 6.5) & (sz > 0.5)] = 9
    labels[(sx >= 4.0) & (sx <= 6.5) & (sz > 0.5)] = 12

    npz_path = tmp_path / "plot.npz"
    np.savez_compressed(npz_path, points=staged_xyz, labels=labels)

    out_las = tmp_path / "forest_treelearn.las"
    out_csv = tmp_path / "tree_summary.csv"
    export_stats = tl_mod._remap_and_export(
        src, staged_las, npz_path, inverse, out_las, out_csv, min_instance_points=5
    )

    assert export_stats["tree_count"] == 2
    assert export_stats["filtered_small_instances"] == 1
    final_cloud = laspy.read(out_las)
    assert len(final_cloud.points) == 70
    assert set(np.unique(final_cloud["treeID"]).tolist()) == {0, 1, 2}
    assert int(final_cloud["treeID"][0]) == int(final_cloud["treeID"][-2])
    assert out_csv.is_file()


def test_treelearn_mcp_tool_mocked_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    xyz = np.column_stack([
        np.linspace(500000.0, 500010.0, 40),
        np.linspace(6000000.0, 6000010.0, 40),
        np.linspace(100.0, 115.0, 40),
    ])
    src = tmp_path / "plot.las"
    _write_synthetic_las(src, xyz)

    def fake_remote(local_staged_las: Path, local_output_npz: Path, **kwargs):
        staged = laspy.read(local_staged_las)
        staged_xyz = np.column_stack((np.asarray(staged.x), np.asarray(staged.y), np.asarray(staged.z)))
        labels = np.where(np.arange(len(staged_xyz)) < 20, 1, 2).astype(np.int32)
        np.savez_compressed(local_output_npz, points=staged_xyz, labels=labels)
        return {
            "job_id": "tl_mock",
            "remote_host": "mock",
            "weight_variant": kwargs.get("weight_variant", "default"),
            "weight_file": "model_weights_20241213_5df2f928.pth",
            "tau_min": kwargs.get("tau_min", 75),
            "remote_inference_sec": 2,
        }

    monkeypatch.setattr(tl_mod, "_run_remote_treelearn", fake_remote)
    out_dir = tmp_path / "out"
    res = asyncio.run(
        tl_mod.segment_trees_treelearn(str(src), output_path=str(out_dir), weight_variant="small", tau_min=60)
    )

    assert res["status"] == "success"
    assert res["data"]["tree_count"] == 2
    assert res["data"]["weight_variant"] == "small"
    assert res["data"]["tau_min"] == 60
    zip_path = Path(res["output_path"])
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        assert {"plot_treelearn.las", "tree_summary.csv", "summary.json"} <= set(zf.namelist())

    bad = asyncio.run(tl_mod.segment_trees_treelearn(str(src), weight_variant="invalid"))
    assert bad["status"] == "error"


def test_auto_router_decision_rules(tmp_path: Path):
    # 1. Dense cloud (>= 15 pts/m^2) -> SegmentAnyTree
    dense_profile = {"input_type": "point_cloud", "point_count": 200_000, "bbox_density_pts_m2": 80.0}
    r1 = auto_mod.choose_segmentation_route(dense_profile, preference="auto")
    assert r1["action"] == "run"
    assert r1["selected_tool"] == "segment_trees_segmentanytree"

    # 2. Sparse ALS (< 15 pts/m^2) -> PyCrown
    sparse_profile = {"input_type": "point_cloud", "point_count": 20_000, "bbox_density_pts_m2": 4.5}
    r2 = auto_mod.choose_segmentation_route(sparse_profile, preference="auto")
    assert r2["action"] == "run"
    assert r2["selected_tool"] == "segment_trees_pycrown"

    # 3. High purity preference -> TreeLearn (with small trees flag)
    r3 = auto_mod.choose_segmentation_route(dense_profile, preference="high_purity", include_small_trees=True)
    assert r3["action"] == "run"
    assert r3["selected_tool"] == "segment_trees_treelearn"
    assert r3["tool_kwargs"]["weight_variant"] == "small"

    # 4. Ultra-massive cloud (> 100M pts) -> needs_confirmation unless allow_large_cloud=True
    huge_profile = {"input_type": "point_cloud", "point_count": 464_000_000, "bbox_density_pts_m2": 5000.0}
    r4 = auto_mod.choose_segmentation_route(huge_profile, preference="auto", allow_large_cloud=False)
    assert r4["action"] == "needs_confirmation"
    r5 = auto_mod.choose_segmentation_route(huge_profile, preference="auto", allow_large_cloud=True)
    assert r5["action"] == "run"
    assert r5["selected_tool"] == "segment_trees_segmentanytree"
    assert r5["tool_kwargs"]["voxel_dedup_m"] == 0.05
    assert auto_mod.choose_segmentation_route(
        huge_profile, preference="fast",
    )["action"] == "needs_confirmation"

    # 5. CHM raster -> PyCrown (or Watershed when fast)
    chm_profile = {"input_type": "chm_raster", "point_count": None, "bbox_density_pts_m2": None}
    assert auto_mod.choose_segmentation_route(chm_profile)["selected_tool"] == "segment_trees_pycrown"
    assert auto_mod.choose_segmentation_route(chm_profile, preference="fast")["selected_tool"] == "segment_trees_watershed"


def test_auto_router_end_to_end_dispatch_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Dense synthetic LAS: 500 pts in 2m x 2m -> 125 pts/m^2 -> should route to SegmentAnyTree
    rng = np.random.default_rng(1)
    xyz = np.column_stack([
        500000 + rng.uniform(0, 2, 500),
        6000000 + rng.uniform(0, 2, 500),
        100 + rng.uniform(0, 15, 500),
    ])
    src = tmp_path / "dense.las"
    _write_synthetic_las(src, xyz)

    called = {}

    async def fake_dispatch(selected_tool, file_path, output_path, tool_kwargs):
        called["tool"] = selected_tool
        called["kwargs"] = tool_kwargs
        return {
            "status": "success",
            "output_path": str(tmp_path / "mock.zip"),
            "message": "mock ok",
            "data": {"tree_count": 3},
        }

    monkeypatch.setattr(auto_mod, "_dispatch_tool", fake_dispatch)
    res = asyncio.run(auto_mod.auto_segment_trees(str(src), output_path=str(tmp_path / "out")))
    assert res["status"] == "success"
    assert called["tool"] == "segment_trees_segmentanytree"
    assert res["data"]["routing"]["selected_tool"] == "segment_trees_segmentanytree"
    assert "[自动路由：" in res["message"]


def test_downstream_contract_hag_ground_mask_preserve_gt_and_dbh_height(tmp_path: Path):
    """Verify SegmentAnyTree & TreeLearn exports satisfy DEVELOPMENT_STANDARDS.md contract
    and feed directly into 1.TreeHight.py and 3.DHB.py without errors."""
    import importlib.util
    from tools.Segmentation import segmentanytree_segmentation as sat_mod

    # Build a sloped plot (z_ground = 100 + 0.5 * x_local, 10 m slope!) with 1 clean trunk + crown
    # Tree centered at x_local=10 (z_ground=105.0), radius=0.15m (DBH=0.30m), height=12.0m
    gx, gy = np.meshgrid(np.linspace(0.0, 20.0, 21), np.linspace(0.0, 10.0, 11))
    gnd_xyz = np.column_stack([500000 + gx.ravel(), 6000000 + gy.ravel(), 100.0 + 0.5 * gx.ravel()])

    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    z_rings = np.linspace(0.8, 2.0, 13)  # HAG from 0.8m to 2.0m (includes 1.2-1.4m breast height)
    stem_pts = []
    for zr in z_rings:
        sx = 500010.0 + 0.15 * np.cos(theta)
        sy = 6000005.0 + 0.15 * np.sin(theta)
        sz = np.full_like(theta, 105.0 + zr)
        stem_pts.append(np.column_stack([sx, sy, sz]))
    stem_xyz = np.vstack(stem_pts)
    crown_xyz = np.column_stack([
        500010.0 + np.cos(theta) * 1.2,
        6000005.0 + np.sin(theta) * 1.2,
        np.full_like(theta, 105.0 + 12.0),
    ])

    all_xyz = np.vstack([gnd_xyz, stem_xyz, crown_xyz])
    cls = np.zeros(len(all_xyz), dtype=np.uint8)
    cls[: len(gnd_xyz)] = 2  # ASPRS ground
    gt_tree_id = np.zeros(len(all_xyz), dtype=np.int32)
    gt_tree_id[len(gnd_xyz) :] = 99  # Existing ground-truth treeID must NOT be overwritten

    src = tmp_path / "sloped_gt.las"
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([500000.0, 6000000.0, 100.0])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
    las.classification = cls
    las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type="i4"))
    las["treeID"] = gt_tree_id
    las.write(src)

    # Simulate TreeLearn accidentally labelling ALL points (including ground!) as instance 7
    staged_las = tmp_path / "staged.las"
    inverse, _ = tl_mod._stage_point_cloud(src, staged_las)
    staged = laspy.read(staged_las)
    staged_xyz = np.column_stack((np.asarray(staged.x), np.asarray(staged.y), np.asarray(staged.z)))
    npz_path = tmp_path / "pred.npz"
    np.savez_compressed(npz_path, points=staged_xyz, labels=np.full(len(staged_xyz), 7, dtype=np.int32))

    tl_out = tmp_path / "sloped_tl.las"
    tl_csv = tmp_path / "tl_summary.csv"
    stats = tl_mod._remap_and_export(src, staged_las, npz_path, inverse, tl_out, tl_csv)

    # Contract assertions: crown ring extends 1.2 m downhill on 0.5 slope -> true max HAG ~ 12.6 m (NOT 17 m from global z min!)
    assert stats["hag_method"] == "ground_class_idw"
    assert 12.0 <= stats["height_max_m"] <= 12.8
    assert stats["trees_with_breast_height_points"] == 1
    assert stats["breast_height_labelled_points"] > 30

    exported = laspy.read(tl_out)
    dims = set(exported.point_format.dimension_names)
    assert {"lidaragent_tree_id", "lidaragent_hag", "treeID"} <= dims
    # Existing GT treeID preserved untouched
    assert np.array_equal(np.asarray(exported["treeID"]), gt_tree_id)
    # Ground points forced to 0 in lidaragent_tree_id
    assert np.all(np.asarray(exported["lidaragent_tree_id"])[: len(gnd_xyz)] == 0)
    assert np.all(np.asarray(exported["lidaragent_tree_id"])[len(gnd_xyz) :] == 1)

    # Feed directly into real 1.TreeHight.py and 3.DHB.py (if pandas is available)
    try:
        import pandas  # noqa: F401
    except ImportError:
        return

    def _load_mod(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    height_mod = _load_mod(BACKEND_DIR / "tools" / "ForestParams" / "1.TreeHight.py", "tree_height_mod")
    dbh_mod = _load_mod(BACKEND_DIR / "tools" / "ForestParams" / "3.DHB.py", "forest_dbh_mod")

    h_res = asyncio.run(height_mod.extract_tree_height(str(tl_out), output_path=str(tmp_path / "h_out")))
    assert h_res["status"] == "success"
    assert h_res["data"]["height_above_ground_field"] == "lidaragent_hag"
    assert 12.0 <= h_res["data"]["tree_height_mean_m"] <= 12.8

    d_res = asyncio.run(dbh_mod.extract_forest_dbh(str(tl_out), output_path=str(tmp_path / "d_out")))
    assert d_res["status"] == "success"
    assert d_res["data"]["tree_count"] == 1
    assert d_res["data"]["height_reference"] == "lidaragent_hag"
    assert abs(d_res["data"]["dbh_mean"] - 0.30) < 0.03

