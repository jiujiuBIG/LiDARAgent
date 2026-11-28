"""Prepare and score matched SegmentAnyTree and TreeLearn pilot runs."""

import argparse
import json
import shutil
from pathlib import Path

import laspy
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


def coordinates(cloud):
    return np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)))


def prepare(source, stage, label_field):
    cloud = laspy.read(source)
    if label_field not in cloud.point_format.extra_dimension_names:
        raise ValueError(f"{source} has no instance dimension {label_field!r}")
    truth = np.asarray(cloud[label_field], dtype=np.int64)
    if not np.any(truth > 0):
        raise ValueError(f"{source} has no positively labelled reference trees")

    xyz = np.column_stack((cloud.X, cloud.Y, cloud.Z)).astype(np.int64)
    raw_origin = xyz.min(axis=0)
    distinct, inverse = np.unique(xyz - raw_origin, axis=0, return_inverse=True)
    if len(distinct) > 2**24 or distinct.max() > np.iinfo(np.int32).max:
        raise ValueError("Split the cloud into smaller spatial plots before pilot inference")
    inverse = inverse.reshape(-1)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = cloud.header.scales.copy()
    staged = laspy.LasData(header)
    staged.X, staged.Y, staged.Z = distinct.T
    # SegmentAnyTree round-trips these indices through a float32 PLY.
    staged.add_extra_dim(laspy.ExtraBytesParams(name="source_index", type="u4"))
    staged.source_index = np.arange(len(distinct), dtype=np.uint32)

    stage.mkdir(parents=True, exist_ok=False)
    (stage / "forest").mkdir()
    staged.write(stage / "forest" / "plot.las")
    np.savez_compressed(stage / "reference.npz", labels=truth, inverse=inverse)
    origin = raw_origin * cloud.header.scales + cloud.header.offsets
    summary = {
        "source": str(source.resolve()),
        "instance_field": label_field,
        "source_points": len(truth),
        "inference_points": len(distinct),
        "reference_trees": len(np.unique(truth[truth > 0])),
        "origin_xyz": origin.tolist(),
        "extent_xyz_m": (distinct.max(axis=0) * header.scales).tolist(),
    }
    (stage / "input.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def instance_scores(truth, prediction):
    # Negative reference IDs mark points outside evaluation. Zero is background.
    valid = truth >= 0
    truth = truth[valid]
    prediction = prediction[valid]
    gt_ids, gt_index, gt_count = np.unique(truth, return_inverse=True, return_counts=True)
    pred_ids, pred_index, pred_count = np.unique(prediction, return_inverse=True, return_counts=True)
    gt_tree, pred_tree = gt_ids > 0, pred_ids > 0
    both = (truth > 0) & (prediction > 0)
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    np.add.at(intersections, (gt_index[both], pred_index[both]), 1)
    union = gt_count[:, None] + pred_count[None, :] - intersections
    iou = (intersections / union)[np.ix_(gt_tree, pred_tree)]
    eligible = iou >= 0.5
    # Maximize match count first, then total IoU among eligible matches.
    reward = np.where(eligible, 1 + iou / (min(iou.shape) + 1), 0)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    matched = eligible[rows, columns]
    matched_ious = iou[rows[matched], columns[matched]]
    tp = int(matched.sum())
    n_gt, n_pred = iou.shape
    return {
        "reference_trees": n_gt,
        "predicted_trees": n_pred,
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
        "f1": 2 * tp / (n_gt + n_pred) if n_gt + n_pred else 0.0,
        "mean_matched_iou": float(matched_ious.mean()) if tp else None,
        "panoptic_quality": float(2 * matched_ious.sum() / (n_gt + n_pred)) if n_gt + n_pred else 0.0,
        "matches": [
            {"reference_id": int(a), "prediction_id": int(b), "iou": float(value)}
            for a, b, value in zip(gt_ids[gt_tree][rows[matched]], pred_ids[pred_tree][columns[matched]], matched_ious)
        ],
    }


def evaluate(stage, result, model, minimums):
    staged = laspy.read(stage / "forest" / "plot.las")
    xyz = coordinates(staged)
    if model == "segmentanytree":
        cloud = laspy.read(result)
        index = np.asarray(cloud["source_index"])
        if (
            len(index) != len(xyz)
            or not np.isfinite(index).all()
            or not np.array_equal(index, index.astype(np.int64))
            or not np.array_equal(np.sort(index), np.arange(len(xyz)))
        ):
            raise ValueError("SegmentAnyTree source_index must cover each staged point exactly once")
        order = np.argsort(index)
        # Upstream LAS converts through float32 PLY and truncates at 1 mm.
        expected, returned, tolerance = xyz, coordinates(cloud)[order], 0.0011
        labels = np.asarray(cloud["PredInstance"])[order]
    else:
        with np.load(result) as output:
            expected, returned, tolerance = xyz, output["points"], 0.0001
            labels = output["labels"]
    if returned.shape != expected.shape or not np.allclose(returned, expected, rtol=0, atol=tolerance):
        raise ValueError(f"{model} coordinates do not align with the staged input")
    if (
        labels.shape != (len(xyz),)
        or not np.isfinite(labels).all()
        or not np.array_equal(labels, labels.astype(np.int64))
        or np.any(labels < 0)
    ):
        raise ValueError(f"{model} returned invalid instance IDs")

    with np.load(stage / "reference.npz") as reference:
        truth = reference["labels"]
        prediction = labels.astype(np.int64)[reference["inverse"]]
    ids, index, counts = np.unique(prediction, return_inverse=True, return_counts=True)
    scores = []
    for minimum in minimums:
        filtered = np.where(counts[index] >= minimum, ids[index], 0)
        score = instance_scores(truth, filtered)
        score["min_instance_points"] = minimum
        score["points_evaluated"] = int(np.count_nonzero(truth >= 0))
        scores.append(score)
    return scores


def treelearn_config(upstream, stage, run, weights, tau_min, workers):
    source_config = upstream / "configs" / "pipeline" / "pipeline.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    source_las = stage / "forest" / "plot.las"
    if not source_las.is_file() or not weights.is_file():
        raise FileNotFoundError("Expected a staged plot.las and a local TreeLearn checkpoint")
    run.mkdir(parents=True, exist_ok=False)
    (run / "forest").mkdir()
    forest = run / "forest" / "plot.las"
    shutil.copyfile(source_las, forest)
    config["forest_path"] = forest.resolve().as_posix()
    config["pretrain"] = weights.resolve().as_posix()
    config["dataloader"]["num_workers"] = workers
    config["save_cfg"].update(save_formats=["npz"], save_treewise=False, save_pointwise=False, return_type="original")
    if tau_min is not None:
        config.setdefault("grouping", {})["tau_min"] = tau_min
    output = run / "pipeline.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--stage", type=Path, required=True)
    preparation.add_argument("--label-field", default="treeID")
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--model", choices=("segmentanytree", "treelearn"), required=True)
    scoring.add_argument("--stage", type=Path, required=True)
    scoring.add_argument("--result", type=Path, required=True)
    scoring.add_argument("--min-instance-points", type=int, nargs="+", default=[0])
    scoring.add_argument("--output", type=Path, required=True)
    config = commands.add_parser("treelearn-config")
    for name in ("upstream", "stage", "run", "weights"):
        config.add_argument(f"--{name}", type=Path, required=True)
    config.add_argument("--tau-min", type=int)
    config.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare(args.source, args.stage, args.label_field), indent=2))
    elif args.command == "evaluate":
        if any(value < 0 for value in args.min_instance_points):
            parser.error("--min-instance-points must be nonnegative")
        scores = evaluate(args.stage, args.result, args.model, args.min_instance_points)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump({"model": args.model, "scores": scores}, handle, indent=2)
            handle.write("\n")
        print(args.output)
    else:
        if args.workers < 0 or (args.tau_min is not None and args.tau_min < 2):
            parser.error("--workers must be nonnegative and --tau-min must be at least 2")
        print(treelearn_config(args.upstream, args.stage, args.run, args.weights, args.tau_min, args.workers))


if __name__ == "__main__":
    main()
