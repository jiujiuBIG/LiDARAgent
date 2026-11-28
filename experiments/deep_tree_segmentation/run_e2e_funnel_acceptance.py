"""End-to-end funnel acceptance for SegmentAnyTree & TreeLearn MCP tools.

Follows DEVELOPMENT_STANDARDS.md (Standard 1-4 & Section 4.2):
- Runs real remote GPU inference via segment_trees_segmentanytree & segment_trees_treelearn
  (or reuses existing segmented LAS when --reuse-existing is passed)
- Verifies input/output point alignment (count, XYZ coordinates, CRS, original treeID preserved)
  and ground/non-finite mask contract (classification == 2 -> lidaragent_tree_id == 0)
- Evaluates point-level instance matching (IoU >= 0.50, negative reference IDs excluded)
  against original source ground-truth treeID
- Feeds exported LAS directly into all 3 downstream ForestParams tools:
    1.TreeHight.py, 2.TreeCrown.py, 3.DHB.py
- Records per-reference-tree matching + 3 independent measurement branches (height, crown, DBH),
  including valid HAG count, 1.2-1.4 m breast-height point count, fit_method, all required
  failure categories, tool failures, missing output rows, and explicit denominators.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "mcp_llm_web" / "backend"
EXP_DIR = REPO_ROOT / "experiments" / "deep_tree_segmentation"
for p in (str(BACKEND_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

SHIMMED_IMPORTS: list[str] = []

# Lightweight FastMCP stub when running outside full MCP server env
try:
    import mcp.server.fastmcp  # noqa: F401
except ImportError:
    SHIMMED_IMPORTS.append("mcp.server.fastmcp")
    mcp_pkg = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    mcp_fast = types.ModuleType("mcp.server.fastmcp")

    class _DummyFastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self, *args, **kwargs):
            return lambda fn: fn

        def run(self, *args, **kwargs):
            pass

    mcp_fast.FastMCP = _DummyFastMCP
    sys.modules["mcp"] = mcp_pkg
    sys.modules["mcp.server"] = mcp_server
    sys.modules["mcp.server.fastmcp"] = mcp_fast

from deep_trees import instance_scores
from tools.Segmentation.segmentanytree_segmentation import segment_trees_segmentanytree
from tools.Segmentation.treelearn_segmentation import segment_trees_treelearn


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


height_mod = _load_module(BACKEND_DIR / "tools" / "ForestParams" / "1.TreeHight.py", "tree_height_mod")
crown_mod = _load_module(BACKEND_DIR / "tools" / "ForestParams" / "2.TreeCrown.py", "tree_crown_mod")
dbh_mod = _load_module(BACKEND_DIR / "tools" / "ForestParams" / "3.DHB.py", "forest_dbh_mod")

PLOTS = {
    "c03": {
        "key": "c03",
        "name": "CULS 3",
        "category": "clean_baseline",
        "source_laz": Path(r"D:\code\testdata_chm_seg\c03_forinst_culs_p3\cloud.laz"),
        "field_csv": REPO_ROOT / "experiments/.demo_data/real_plot_acceptance_20261127_separated_crowns/c03/pycrown_field_dbh.csv",
    },
    "p13": {
        "key": "p13",
        "name": "NIBIO 13",
        "category": "occluded_tilted_hard",
        "source_laz": Path(r"D:\code\testdata_chm_seg\batch\fi_nibio_p13\cloud.laz"),
        "field_csv": REPO_ROOT / "experiments/.demo_data/real_plot_acceptance_20261127_separated_crowns/p13/pycrown_field_dbh.csv",
    },
    "p19": {
        "key": "p19",
        "name": "NIBIO 19",
        "category": "occluded_tilted_hard",
        "source_laz": Path(r"D:\code\testdata_chm_seg\batch\fi_nibio_p19\cloud.laz"),
        "field_csv": REPO_ROOT / "experiments/.demo_data/real_plot_acceptance_20261127_separated_crowns/p19/pycrown_field_dbh.csv",
    },
}


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _get_gt_tree_id(las: laspy.LasData) -> np.ndarray:
    dims = {d.lower(): d for d in las.point_format.dimension_names}
    for cand in ("treeid", "tree_id"):
        if cand in dims:
            return np.asarray(las[dims[cand]], dtype=np.int64)
    raise ValueError(f"No ground-truth treeID field found in {list(las.point_format.dimension_names)}")


def _verify_alignment_and_contract(src_las: laspy.LasData, seg_las: laspy.LasData) -> dict:
    src_gt = _get_gt_tree_id(src_las)
    seg_dims = set(seg_las.point_format.dimension_names)

    count_match = len(src_las.x) == len(seg_las.x)
    xyz_aligned = bool(
        count_match
        and np.allclose(src_las.x, seg_las.x, rtol=0, atol=1e-4)
        and np.allclose(src_las.y, seg_las.y, rtol=0, atol=1e-4)
        and np.allclose(src_las.z, seg_las.z, rtol=0, atol=1e-4)
    )
    try:
        seg_gt = _get_gt_tree_id(seg_las)
        gt_preserved = bool(count_match and np.array_equal(src_gt, seg_gt))
    except ValueError:
        gt_preserved = False

    has_tree_id = "lidaragent_tree_id" in seg_dims
    has_hag = "lidaragent_hag" in seg_dims
    pred_ids = np.asarray(seg_las["lidaragent_tree_id"], dtype=np.int64) if has_tree_id else np.array([], dtype=np.int64)
    hag = np.asarray(seg_las["lidaragent_hag"], dtype=np.float64) if has_hag else np.array([], dtype=np.float64)

    ground_masked = True
    if has_tree_id and "classification" in seg_dims:
        ground_mask = np.asarray(seg_las.classification) == 2
        ground_masked = bool(np.all(pred_ids[ground_mask] == 0))

    non_finite_hag_masked = True
    if has_tree_id and has_hag:
        non_finite = ~np.isfinite(hag)
        non_finite_hag_masked = bool(np.all(pred_ids[non_finite] == 0))

    src_crs = str(src_las.header.parse_crs()) if src_las.header.parse_crs() is not None else None
    seg_crs = str(seg_las.header.parse_crs()) if seg_las.header.parse_crs() is not None else None
    crs_preserved = src_crs == seg_crs

    passed = all([
        count_match,
        xyz_aligned,
        gt_preserved,
        has_tree_id,
        has_hag,
        ground_masked,
        non_finite_hag_masked,
        crs_preserved,
    ])
    return {
        "passed": passed,
        "source_points": int(len(src_las.x)),
        "output_points": int(len(seg_las.x)),
        "count_match": count_match,
        "xyz_aligned": xyz_aligned,
        "gt_treeid_preserved": gt_preserved,
        "has_lidaragent_tree_id": has_tree_id,
        "has_lidaragent_hag": has_hag,
        "ground_class2_masked_zero": ground_masked,
        "non_finite_hag_masked_zero": non_finite_hag_masked,
        "crs_preserved": crs_preserved,
    }


def _find_existing_segmented_las(out_dir: Path, model: str) -> Path | None:
    cand = out_dir / f"cloud_{model}.las"
    if cand.is_file():
        return cand
    matches = sorted(out_dir.glob(f"*_{model}.la*"))
    return matches[0] if matches else None


def _error_stats(errors: np.ndarray) -> dict:
    if len(errors) == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    return {
        "n": int(len(errors)),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 3),
        "bias": round(float(np.mean(errors)), 3),
    }


async def eval_one(
    plot: dict,
    model: str,
    out_root: Path,
    *,
    reuse_existing: bool = False,
    seg_root: Path | None = None,
) -> dict:
    out_dir = out_root / plot["key"] / model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Evaluating {plot['key']} / {model} (reuse_existing={reuse_existing}) ...", flush=True)

    search_dir = (seg_root / plot["key"] / model) if seg_root is not None else out_dir
    existing_las = _find_existing_segmented_las(search_dir, model) if reuse_existing else None
    seg_meta: dict = {}
    if existing_las is not None:
        seg_las_path = existing_las
        seg_meta = {"reused_segmented_cloud": True, "segmented_cloud": str(seg_las_path)}
        print(f"    Reusing existing segmented cloud: {seg_las_path}", flush=True)
    else:
        if model == "segmentanytree":
            seg_res = await segment_trees_segmentanytree(str(plot["source_laz"]), output_path=str(out_dir))
        else:
            seg_res = await segment_trees_treelearn(str(plot["source_laz"]), output_path=str(out_dir))

        if seg_res["status"] != "success":
            print(f"    ERROR: {seg_res['message']}", flush=True)
            return {"status": "error", "message": seg_res["message"]}
        seg_las_path = Path(seg_res["data"]["segmented_cloud"])
        seg_meta = {
            "reused_segmented_cloud": False,
            "segmented_cloud": str(seg_las_path),
            "elapsed_sec": seg_res["data"].get("total_elapsed_sec"),
            "hag_method": seg_res["data"].get("hag_method"),
            "breast_height_labelled_points": seg_res["data"].get("breast_height_labelled_points"),
            "trees_with_breast_height_points": seg_res["data"].get("trees_with_breast_height_points"),
        }

    src_las = laspy.read(plot["source_laz"])
    seg_las = laspy.read(seg_las_path)

    # 0. Alignment & contract verification before scoring
    contract = _verify_alignment_and_contract(src_las, seg_las)
    if not contract["passed"]:
        raise RuntimeError(f"Contract check failed for {plot['key']}/{model}: {contract}")

    gt_ids = _get_gt_tree_id(src_las)
    pred_ids = np.asarray(seg_las["lidaragent_tree_id"], dtype=np.int64)
    hag_vals = np.asarray(seg_las["lidaragent_hag"], dtype=np.float64)

    # Reference tree heights from source GT + lidaragent_hag (point-cloud geometric reference)
    ref_tree_ids = sorted(int(x) for x in np.unique(gt_ids) if x > 0)
    ref_height_map: dict[int, float | None] = {}
    for rid in ref_tree_ids:
        rh = hag_vals[gt_ids == rid]
        rh = rh[np.isfinite(rh) & (rh >= 0)]
        ref_height_map[rid] = float(rh.max()) if len(rh) else None

    # 1. Point-level matching (IoU >= 0.50, negative reference IDs excluded)
    pm = instance_scores(gt_ids, pred_ids)
    match_map = {int(m["reference_id"]): (int(m["prediction_id"]), float(m["iou"])) for m in pm["matches"]}

    # 2. Downstream 1.TreeHight.py
    h_res = await height_mod.extract_tree_height(str(seg_las_path), output_path=str(out_dir / "height"))
    h_ok = h_res.get("status") in ("success", "warning") and h_res.get("output_path") and Path(h_res["output_path"]).is_file()
    h_df = pd.read_csv(h_res["output_path"]) if h_ok else pd.DataFrame()
    h_by_pred = h_df.set_index("treeid").to_dict(orient="index") if not h_df.empty else {}

    # 3. Downstream 2.TreeCrown.py
    c_res = await crown_mod.extract_tree_crown(str(seg_las_path), output_path=str(out_dir / "crown"))
    c_ok = c_res.get("status") in ("success", "warning") and c_res.get("output_path") and Path(c_res["output_path"]).is_file()
    c_df = pd.read_csv(c_res["output_path"]) if c_ok else pd.DataFrame()
    c_by_pred = c_df.set_index("treeid").to_dict(orient="index") if not c_df.empty else {}

    # 4. Downstream 3.DHB.py (must run directly via lidaragent_hag without z_is_normalized=True)
    d_res = await dbh_mod.extract_forest_dbh(str(seg_las_path), output_path=str(out_dir / "dbh"))
    d_ok = d_res.get("status") in ("success", "warning") and d_res.get("output_path") and Path(d_res["output_path"]).is_file()
    d_df = pd.read_csv(d_res["output_path"]) if d_ok else pd.DataFrame()
    d_by_pred = d_df.set_index("treeid").to_dict(orient="index") if not d_df.empty else {}

    # Field DBH lookup
    field_df = pd.read_csv(plot["field_csv"])[["reference_id", "field_dbh_cm"]].drop_duplicates("reference_id")
    field_dbh_map = {int(r["reference_id"]): float(r["field_dbh_cm"]) for _, r in field_df.iterrows()}

    # 5. Per-reference-tree 3-branch funnel table
    per_tree_rows = []
    for rid in ref_tree_ids:
        matched = rid in match_map
        pred_id, iou = match_map[rid] if matched else (None, None)
        ref_h = ref_height_map.get(rid)
        has_field = rid in field_dbh_map
        field_cm = field_dbh_map.get(rid)

        # Height branch
        if not matched:
            h_status, est_h, h_err = "unmatched_instance", None, None
        elif not h_ok:
            h_status, est_h, h_err = "tool_failed", None, None
        elif pred_id not in h_by_pred:
            h_status, est_h, h_err = "missing_output_row", None, None
        else:
            val = h_by_pred[pred_id].get("tree_height_m")
            if pd.notna(val) and np.isfinite(float(val)):
                est_h = float(val)
                h_status = "measured"
                h_err = (est_h - ref_h) if ref_h is not None else None
            else:
                h_status, est_h, h_err = "missing_height_above_ground", None, None

        # Crown branch
        if not matched:
            c_status, c_maj, c_min, c_pts = "unmatched_instance", None, None, None
        elif not c_ok:
            c_status, c_maj, c_min, c_pts = "tool_failed", None, None, None
        elif pred_id not in c_by_pred:
            c_status, c_maj, c_min, c_pts = "missing_output_row", None, None, None
        else:
            rec_c = c_by_pred[pred_id]
            c_maj = float(rec_c["major_axis_length"]) if pd.notna(rec_c.get("major_axis_length")) else None
            c_min = float(rec_c["minor_axis_length"]) if pd.notna(rec_c.get("minor_axis_length")) else None
            c_pts = int(rec_c["points_length"]) if pd.notna(rec_c.get("points_length")) else None
            c_status = "measured" if (c_maj is not None and c_maj > 0) else "fit_failed"

        # DBH branch
        if not matched:
            d_status = "unmatched_instance"
            d_valid_hag, d_bh_pts, d_method, est_dbh_cm, d_err = None, None, None, None, None
        elif not d_ok:
            d_status = "tool_failed"
            d_valid_hag, d_bh_pts, d_method, est_dbh_cm, d_err = None, None, None, None, None
        elif pred_id not in d_by_pred:
            d_status = "missing_output_row"
            d_valid_hag, d_bh_pts, d_method, est_dbh_cm, d_err = None, None, None, None, None
        else:
            rec_d = d_by_pred[pred_id]
            d_valid_hag = int(rec_d["valid_height_points"]) if pd.notna(rec_d.get("valid_height_points")) else 0
            d_bh_pts = int(rec_d["breast_height_points"]) if pd.notna(rec_d.get("breast_height_points")) else 0
            d_method = str(rec_d["fit_method"]) if pd.notna(rec_d.get("fit_method")) else None
            d_status = str(rec_d.get("measurement_status", "missing_output_row"))
            dbh_m = rec_d.get("DBH")
            est_dbh_cm = float(dbh_m) * 100.0 if pd.notna(dbh_m) and float(dbh_m) > 0 else None
            d_err = (est_dbh_cm - field_cm) if (est_dbh_cm is not None and field_cm is not None) else None

        per_tree_rows.append({
            "reference_id": rid,
            "matched": matched,
            "prediction_id": pred_id,
            "iou": round(iou, 4) if iou is not None else None,
            "ref_tree_height_m": round(ref_h, 3) if ref_h is not None else None,
            "height_status": h_status,
            "estimated_height_m": round(est_h, 3) if est_h is not None else None,
            "height_error_m": round(h_err, 3) if h_err is not None else None,
            "crown_status": c_status,
            "crown_major_m": round(c_maj, 3) if c_maj is not None else None,
            "crown_minor_m": round(c_min, 3) if c_min is not None else None,
            "crown_points": c_pts,
            "has_field_dbh": has_field,
            "field_dbh_cm": field_cm,
            "valid_height_points": d_valid_hag,
            "breast_height_points": d_bh_pts,
            "dbh_fit_method": d_method,
            "dbh_status": d_status,
            "estimated_dbh_cm": round(est_dbh_cm, 2) if est_dbh_cm is not None else None,
            "dbh_error_cm": round(d_err, 2) if d_err is not None else None,
        })

    per_tree_df = pd.DataFrame(per_tree_rows)
    per_tree_csv = out_dir / "per_tree_funnel.csv"
    per_tree_df.to_csv(per_tree_csv, index=False)

    # Also write field_dbh_funnel.csv for backward compatibility with previous inspection
    field_funnel_df = per_tree_df[per_tree_df["has_field_dbh"]].copy()
    field_funnel_df.to_csv(out_dir / "field_dbh_funnel.csv", index=False)

    n_ref = len(per_tree_df)
    n_field = len(field_funnel_df)

    # Height branch summary
    h_measured = per_tree_df[per_tree_df["height_status"] == "measured"]
    h_errs = h_measured["height_error_m"].dropna().to_numpy(dtype=np.float64)
    h_stats = _error_stats(h_errs)

    # Crown branch summary
    c_measured = per_tree_df[per_tree_df["crown_status"] == "measured"]

    # DBH branch summary (all reference trees + field DBH subset)
    d_all_measured = per_tree_df[per_tree_df["dbh_status"] == "measured"]
    d_field_measured = field_funnel_df[field_funnel_df["dbh_status"] == "measured"]
    d_errs = d_field_measured["dbh_error_cm"].dropna().to_numpy(dtype=np.float64)
    d_stats = _error_stats(d_errs)

    result = {
        "status": "success",
        **seg_meta,
        "contract_verification": contract,
        "point_matching": {k: v for k, v in pm.items() if k != "matches"},
        "height_branch": {
            "tool_status": h_res.get("status"),
            "height_above_ground_field": h_res["data"].get("height_above_ground_field") if h_res.get("data") else None,
            "tool_output_trees": int(h_res["data"]["tree_count"]) if h_res.get("data") else 0,
            "matched_measured": int(len(h_measured)),
            "denominator_reference_trees": n_ref,
            "coverage_rate": round(len(h_measured) / n_ref, 4) if n_ref else None,
            "status_counts": {k: int(v) for k, v in per_tree_df["height_status"].value_counts().to_dict().items()},
            "cloud_hag_error_m": h_stats,
        },
        "crown_branch": {
            "tool_status": c_res.get("status"),
            "tool_output_trees": int(c_res["data"]["tree_count"]) if c_res.get("data") else 0,
            "tool_unmeasured_trees": int(c_res["data"].get("unmeasured_tree_count", 0)) if c_res.get("data") else 0,
            "matched_measured": int(len(c_measured)),
            "denominator_reference_trees": n_ref,
            "coverage_rate": round(len(c_measured) / n_ref, 4) if n_ref else None,
            "status_counts": {k: int(v) for k, v in per_tree_df["crown_status"].value_counts().to_dict().items()},
            "field_truth_note": "No per-tree independent field crown width truth available; geometric coverage only.",
        },
        "dbh_branch": {
            "tool_status": d_res.get("status"),
            "height_reference": d_res["data"].get("height_reference") if d_res.get("data") else None,
            "tool_output_trees": int(d_res["data"]["tree_count"]) if d_res.get("data") else 0,
            "all_reference_trees": {
                "measured": int(len(d_all_measured)),
                "denominator_reference_trees": n_ref,
                "measurable_rate": round(len(d_all_measured) / n_ref, 4) if n_ref else None,
                "status_counts": {k: int(v) for k, v in per_tree_df["dbh_status"].value_counts().to_dict().items()},
                "fit_method_counts": {k: int(v) for k, v in d_all_measured["dbh_fit_method"].value_counts().to_dict().items()},
            },
            "field_truth_subset": {
                "measured": int(len(d_field_measured)),
                "denominator_field_trees": n_field,
                "measurable_rate": round(len(d_field_measured) / n_field, 4) if n_field else None,
                "status_counts": {k: int(v) for k, v in field_funnel_df["dbh_status"].value_counts().to_dict().items()},
                "fit_method_counts": {k: int(v) for k, v in d_field_measured["dbh_fit_method"].value_counts().to_dict().items()},
                "error_cm": d_stats,
            },
        },
        "per_tree_funnel_csv": str(per_tree_csv),
    }
    print(
        f"    DONE {plot['key']}/{model}: "
        f"TP={pm['tp']}/{pm['reference_trees']} (F1={pm['f1']:.4f}, IoU={pm['mean_matched_iou']:.4f}) | "
        f"H={len(h_measured)}/{n_ref} | Crown={len(c_measured)}/{n_ref} | "
        f"DBH(field)={len(d_field_measured)}/{n_field} {result['dbh_branch']['field_truth_subset']['status_counts']} | "
        f"MAE={d_stats['mae']} cm RMSE={d_stats['rmse']} cm",
        flush=True,
    )
    return result


async def main():
    parser = argparse.ArgumentParser(description="End-to-end 3-branch funnel acceptance")
    parser.add_argument("--plots", default="c03,p13,p19", help="Comma-separated plot keys")
    parser.add_argument("--models", default="segmentanytree,treelearn", help="Comma-separated models")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing segmented LAS when present")
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "experiments/.demo_data/deep_e2e_funnel_20261127"),
        help="Output root directory",
    )
    parser.add_argument(
        "--seg-root",
        default=None,
        help="Optional root directory to read existing segmented LAS files from when --reuse-existing is set",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    seg_root = Path(args.seg_root) if args.seg_root else None
    out_root.mkdir(parents=True, exist_ok=True)
    plot_keys = [k.strip() for k in args.plots.split(",") if k.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary_path = out_root / "summary_v2_3branch.json"
    summary: dict = {
        "meta": {
            "git_head": _git_head(),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "shimmed_imports": SHIMMED_IMPORTS,
            "point_matching_scope": "point-level IoU >= 0.50, negative reference IDs excluded",
        },
        "plots": {},
    }
    for pk in plot_keys:
        plot = PLOTS[pk]
        summary["plots"][pk] = {
            "name": plot["name"],
            "category": plot["category"],
            "source_laz": str(plot["source_laz"]),
        }
        for model in models:
            summary["plots"][pk][model] = await eval_one(
                plot, model, out_root, reuse_existing=args.reuse_existing, seg_root=seg_root
            )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full 3-branch summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
