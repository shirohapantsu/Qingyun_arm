#!/usr/bin/env python3
"""Validate a multi-capsule URDF mesh fit against representative poses.

The input is the JSON emitted by ``fit_urdf_link_capsules.py``.  This helper
builds a temporary in-memory candidate profile, removes unconfirmed workcell
obstacles, expands collision exemptions only for pieces of the same link and
for directly adjacent URDF links, and runs the normal collision checker.  It
does not modify the calibrated profile or command hardware.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import load_motion_params  # noqa: E402
from qingyun.grabbing.kinematics_ext import ArmModel  # noqa: E402
from qingyun.grabbing.safety import (  # noqa: E402
    CollisionChecker,
    CollisionViolation,
    JointLimits,
    build_error_envelope,
    segment_halfspace_distance,
    segment_segment_distance,
    validate_segment,
)
from qingyun.grabbing.trajectory import make_joint_segment  # noqa: E402


def _adjacent_mesh_links(urdf_path: Path, mesh_links: set[str]) -> set[frozenset[str]]:
    root = ET.fromstring(urdf_path.read_bytes())
    pairs: set[frozenset[str]] = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        a = parent.attrib["link"]
        b = child.attrib["link"]
        if a in mesh_links and b in mesh_links:
            pairs.add(frozenset((a, b)))
    return pairs


def _selected_candidate(row: dict, count: int) -> dict:
    if count == 1:
        candidates = [row["candidate"]]
    else:
        candidates = [
            item for item in row["clustered_candidates"]
            if int(item["cluster_count"]) == count
        ]
    if len(candidates) != 1:
        raise ValueError(f"{row['link']}: no unique {count}-capsule candidate")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--cluster-count", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--per-mesh-segments", type=int, default=0)
    parser.add_argument("--per-mesh-clusters", type=int, default=0)
    parser.add_argument("--per-mesh-parallel-clusters", type=int, default=0)
    parser.add_argument(
        "--per-mesh-cluster-override",
        action="append",
        default=[],
        metavar="LINK:COLLISION_INDEX:COUNT",
        help="指定单个URDF collision mesh的聚类数；未指定者保持单胶囊",
    )
    parser.add_argument("--extra-ignore-from", type=Path)
    parser.add_argument(
        "--extra-ignore-pair",
        action="append",
        nargs=2,
        default=[],
        metavar=("CAPSULE_A", "CAPSULE_B"),
        help="追加具名候选豁免对，仅用于离线审计；可重复传入",
    )
    parser.add_argument(
        "--use-profile-ignore-pairs",
        action="store_true",
        help="候选ID兼容时合并输入profile中的既有CAL-034豁免，仅用于离线复审",
    )
    parser.add_argument(
        "--pose-deg",
        action="append",
        nargs=5,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="追加一个五关节实体参考姿态；可重复传入",
    )
    parser.add_argument(
        "--pose-gripper-pct",
        type=float,
        default=2.0028612303290414,
        help="追加实体参考姿态使用的夹爪开度百分比",
    )
    parser.add_argument(
        "--path-json",
        action="append",
        type=json.loads,
        default=[],
        help=(
            "追加路径JSON：name/start_deg/end_deg/gripper_pct/duration_s；"
            "整段按生产validate_segment检查"
        ),
    )
    args = parser.parse_args()
    per_mesh_cluster_overrides: dict[tuple[str, int], int] = {}
    for text in args.per_mesh_cluster_override:
        try:
            link, collision_index_text, count_text = text.rsplit(":", 2)
            key = (link, int(collision_index_text))
            count = int(count_text)
        except ValueError as exc:
            raise SystemExit(
                "per-mesh-cluster-override格式应为LINK:COLLISION_INDEX:COUNT"
            ) from exc
        if count < 1:
            raise SystemExit("per-mesh-cluster-override的COUNT必须>=1")
        if key in per_mesh_cluster_overrides:
            raise SystemExit(f"重复的per-mesh-cluster-override：{key}")
        per_mesh_cluster_overrides[key] = count

    fit = json.loads(args.fit.resolve(strict=True).read_text(encoding="utf-8"))
    profile_path = args.profile.resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    urdf_path = (profile_path.parent / profile["model"]["urdf_path"]).resolve(strict=True)
    calibration_path = (
        profile_path.parent / profile["model"]["motor_calibration_path"]
    ).resolve(strict=True)

    candidate = copy.deepcopy(profile)
    candidate["status"] = "simulation"
    candidate["model"]["urdf_path"] = str(urdf_path)
    candidate["model"]["motor_calibration_path"] = str(calibration_path)
    # CAL-032 has explicitly been deferred.  Its two legacy AABBs are simulation
    # placeholders and must not contaminate this CAL-033 geometry-only check.
    candidate["workspace"]["obstacles"] = []

    capsules: list[dict] = []
    ids_by_link: dict[str, list[str]] = {}
    selected_summary = []
    for link_row in fit["links"]:
        link = link_row["link"]
        uniform_per_mesh_modes = sum(
            bool(value)
            for value in (
                args.per_mesh_segments,
                args.per_mesh_clusters,
                args.per_mesh_parallel_clusters,
            )
        )
        if uniform_per_mesh_modes > 1:
            raise ValueError("统一逐mesh分段/聚类模式不能同时使用")
        if per_mesh_cluster_overrides and uniform_per_mesh_modes:
            raise ValueError("逐mesh聚类覆盖不能与统一逐mesh分段/聚类同时使用")
        if uniform_per_mesh_modes or per_mesh_cluster_overrides:
            selected_capsules = []
            for mesh_row in link_row["per_mesh_candidates"]:
                if per_mesh_cluster_overrides:
                    requested = per_mesh_cluster_overrides.get(
                        (link, int(mesh_row["collision_index"])), 1
                    )
                    source = (
                        mesh_row["by_segment_count"]
                        if requested == 1
                        else mesh_row["by_cluster_count"]
                    )
                    field = "segment_count" if requested == 1 else "cluster_count"
                else:
                    if args.per_mesh_segments:
                        source = mesh_row["by_segment_count"]
                        field = "segment_count"
                        requested = args.per_mesh_segments
                    elif args.per_mesh_clusters:
                        source = mesh_row["by_cluster_count"]
                        field = "cluster_count"
                        requested = args.per_mesh_clusters
                    else:
                        source = mesh_row["by_parallel_cluster_count"]
                        field = "cluster_count"
                        requested = args.per_mesh_parallel_clusters
                match = [
                    item for item in source
                    if int(item[field]) == requested
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"{link}/{mesh_row['mesh']}: no unique "
                        f"{requested}-{field} candidate"
                    )
                selected_capsules.extend(match[0]["capsules"])
            selected = {"capsules": selected_capsules}
        else:
            selected = (
                link_row["adaptive_candidate"]
                if args.adaptive
                else _selected_candidate(link_row, args.cluster_count)
            )
            if selected is None:
                raise ValueError(f"{link}: fit JSON has no adaptive candidate")
        ids_by_link[link] = []
        for index, capsule in enumerate(selected["capsules"], start=1):
            capsule_id = f"{link.removesuffix('_link')}_{index:02d}"
            ids_by_link[link].append(capsule_id)
            capsules.append(
                {
                    "id": capsule_id,
                    "link": link,
                    "p0_m": capsule["p0_m"],
                    "p1_m": capsule["p1_m"],
                    "radius_m": capsule["demo_radius_with_physical_extra_m"],
                }
            )
        selected_summary.append(
            {
                "link": link,
                "capsule_count": len(selected["capsules"]),
                "maximum_demo_radius_m": max(
                    item["demo_radius_with_physical_extra_m"]
                    for item in selected["capsules"]
                ),
            }
        )
    candidate["collision"]["link_capsules"] = capsules

    mesh_links = set(ids_by_link)
    adjacent = _adjacent_mesh_links(urdf_path, mesh_links)
    ignored: list[list[str]] = []
    for i, first in enumerate(capsules):
        for second in capsules[i + 1:]:
            links = frozenset((first["link"], second["link"]))
            if first["link"] == second["link"] or links in adjacent:
                ignored.append([first["id"], second["id"]])
    candidate["collision"]["ignore_self_pairs"] = ignored
    if args.use_profile_ignore_pairs:
        existing = {frozenset(pair) for pair in ignored}
        valid_ids = {item["id"] for item in capsules}
        for pair in profile["collision"]["ignore_self_pairs"]:
            if not set(pair) <= valid_ids:
                raise ValueError(
                    "profile ignore pair与候选胶囊ID不兼容：" + json.dumps(pair)
                )
            if frozenset(pair) not in existing:
                ignored.append(pair)
                existing.add(frozenset(pair))
        candidate["collision"]["ignore_self_pairs"] = ignored
    if args.extra_ignore_from is not None:
        prior = json.loads(
            args.extra_ignore_from.resolve(strict=True).read_text(encoding="utf-8")
        )
        existing = {frozenset(pair) for pair in ignored}
        valid_ids = {item["id"] for item in capsules}
        for label in prior["nonignored_capsule_pair_maximum_penetration_m"]:
            pair = label.split(" x ")
            if len(pair) != 2 or not set(pair) <= valid_ids:
                raise ValueError(f"invalid prior capsule pair: {label}")
            if frozenset(pair) not in existing:
                ignored.append(pair)
                existing.add(frozenset(pair))
        candidate["collision"]["ignore_self_pairs"] = ignored
    if args.extra_ignore_pair:
        existing = {frozenset(pair) for pair in ignored}
        valid_ids = {item["id"] for item in capsules}
        for pair in args.extra_ignore_pair:
            if pair[0] == pair[1] or not set(pair) <= valid_ids:
                raise ValueError(f"invalid explicit capsule pair: {pair}")
            if frozenset(pair) not in existing:
                ignored.append(pair)
                existing.add(frozenset(pair))
        candidate["collision"]["ignore_self_pairs"] = ignored

    with tempfile.TemporaryDirectory(prefix="cal033-") as tmp:
        temporary_profile = Path(tmp) / "profile.json"
        temporary_profile.write_text(
            json.dumps(candidate, ensure_ascii=False), encoding="utf-8"
        )
        params = load_motion_params(temporary_profile, mode="mock")
        model = ArmModel(params)
        envelope = build_error_envelope(params, params.collision.max_joint_substep_deg)
        checker = CollisionChecker(params, model, envelope)
        poses = [
            ("home", params.workspace.home_joints_deg),
            *(
                (f"safe_waypoint_{index}", pose)
                for index, pose in enumerate(params.workspace.safe_waypoints_deg, start=1)
            ),
            ("urdf_orthogonal", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        physical_pose_names = set()
        for index, pose in enumerate(args.pose_deg or (), start=1):
            name = f"physical_reference_{index}"
            physical_pose_names.add(name)
            poses.append((name, np.asarray(pose, dtype=float)))
        results = []
        path_results = []
        exact_mesh_distances: dict[tuple[str, str], float] = {}
        capsule_pair_penetrations: dict[tuple[str, str], float] = {}
        pose_capsule_pair_penetrations: dict[str, dict[str, float]] = {}
        table_penetrations: dict[str, float] = {}
        for pose_name, pose in poses:
            gripper_samples = (
                (float(args.pose_gripper_pct),)
                if pose_name in physical_pose_names
                else (0.0, 25.0, 50.0, 75.0, 100.0)
            )
            for gripper_pct in gripper_samples:
                pose_key = f"{pose_name}@{gripper_pct:.6g}%"
                pose_pair_penetrations: dict[str, float] = {}
                # The vendor collision model is used only as a diagnostic here:
                # it can distinguish a coarse-capsule false positive from actual
                # intersection of the nominal URDF meshes.
                model.frames_for(list(ids_by_link), pose, gripper_pct)
                geometry = model.kin.robot.collision_model.geometryObjects
                for distance in model.kin.robot.distances():
                    link_a = geometry[distance.objA].name.rsplit("_", 1)[0]
                    link_b = geometry[distance.objB].name.rsplit("_", 1)[0]
                    if link_a == link_b:
                        continue
                    key = tuple(sorted((link_a, link_b)))
                    value = float(distance.min_distance)
                    exact_mesh_distances[key] = min(exact_mesh_distances.get(key, value), value)
                axes = checker._axes(pose, gripper_pct, [0.0] * 5)
                for capsule_id, start, end, radius in axes:
                    if capsule_id in checker._table_exempt:
                        continue
                    penetration = (
                        radius
                        + checker._table_inflation
                        - segment_halfspace_distance(
                            start, end, checker._table_origin, [0.0, 0.0, 1.0]
                        )
                    )
                    if penetration > 0:
                        table_penetrations[capsule_id] = max(
                            table_penetrations.get(capsule_id, 0.0), penetration
                        )
                for first_index, first in enumerate(axes):
                    for second in axes[first_index + 1:]:
                        pair = tuple(sorted((first[0], second[0])))
                        if frozenset(pair) in checker._ignored:
                            continue
                        penetration = (
                            first[3]
                            + second[3]
                            - segment_segment_distance(first[1], first[2], second[1], second[2])
                        )
                        if penetration > 0:
                            capsule_pair_penetrations[pair] = max(
                                capsule_pair_penetrations.get(pair, 0.0), penetration
                            )
                            pose_pair_penetrations[" x ".join(pair)] = float(penetration)
                pose_capsule_pair_penetrations[pose_key] = pose_pair_penetrations
                try:
                    checker.check_pose(
                        pose,
                        gripper_pct,
                        pair_delta_deg=[0.0] * 5,
                        stage=f"{pose_name}@{gripper_pct:.0f}% ",
                    )
                except CollisionViolation as exc:
                    results.append(
                        {
                            "pose": pose_name,
                            "gripper_pct": gripper_pct,
                            "ok": False,
                            "reason": str(exc),
                        }
                    )
                else:
                    results.append(
                        {"pose": pose_name, "gripper_pct": gripper_pct, "ok": True}
                    )

        limits = JointLimits.from_params(params)
        for index, path_payload in enumerate(args.path_json, start=1):
            required = {"name", "start_deg", "end_deg", "gripper_pct", "duration_s"}
            if set(path_payload) != required:
                raise ValueError(
                    f"path-json[{index}]键必须精确为{sorted(required)}"
                )
            segment = make_joint_segment(
                str(path_payload["name"]),
                np.asarray(path_payload["start_deg"], dtype=float),
                np.asarray(path_payload["end_deg"], dtype=float),
                float(path_payload["gripper_pct"]),
                params,
                min_duration_s=float(path_payload["duration_s"]),
            )
            try:
                checked_path = validate_segment(segment, model, params, limits, checker)
            except (CollisionViolation, RuntimeError) as exc:
                path_results.append({
                    "name": str(path_payload["name"]),
                    "gripper_pct": float(path_payload["gripper_pct"]),
                    "nodes": int(segment.nodes),
                    "ok": False,
                    "reason": str(exc),
                })
            else:
                path_results.append({
                    "name": checked_path.name,
                    "gripper_pct": float(path_payload["gripper_pct"]),
                    "nodes": checked_path.nodes,
                    "ok": True,
                    "peak_joint_velocity_deg_s": [
                        float(v) for v in checked_path.peak_joint_velocity_deg_s
                    ],
                    "peak_joint_acceleration_deg_s2": [
                        float(v) for v in checked_path.peak_joint_acceleration_deg_s2
                    ],
                    "peak_tcp_velocity_m_s": float(checked_path.peak_tcp_velocity_m_s),
                    "peak_tcp_acceleration_m_s2": float(
                        checked_path.peak_tcp_acceleration_m_s2
                    ),
                })

    output = {
        "schema_version": 1,
        "source_fit": str(args.fit.resolve()),
        "profile_unchanged": True,
        "unconfirmed_obstacles_excluded": True,
        "same_link_pairs_ignored": True,
        "only_direct_urdf_adjacent_link_pairs_ignored": True,
        "extra_ignore_source": (
            None if args.extra_ignore_from is None else str(args.extra_ignore_from.resolve())
        ),
        "explicit_extra_ignore_pairs": args.extra_ignore_pair,
        "profile_ignore_pairs_merged": bool(args.use_profile_ignore_pairs),
        "per_mesh_cluster_overrides": [
            {"link": link, "collision_index": index, "cluster_count": count}
            for (link, index), count in sorted(per_mesh_cluster_overrides.items())
        ],
        "runtime_table_contact_exemption_tested": True,
        "additional_physical_reference_poses_deg": args.pose_deg or [],
        "additional_physical_reference_gripper_pct": float(args.pose_gripper_pct),
        "capsule_count": len(capsules),
        "ignore_pair_count": len(ignored),
        "candidate_collision": {
            "link_capsules": capsules,
            "ignore_self_pairs": ignored,
        },
        "candidate_collision_matches_input_profile": (
            profile["collision"]["link_capsules"] == capsules
            and profile["collision"]["ignore_self_pairs"] == ignored
        ),
        "selected": selected_summary,
        "minimum_nominal_mesh_distance_by_link_pair_m": {
            " x ".join(pair): distance
            for pair, distance in sorted(exact_mesh_distances.items())
        },
        "nonignored_capsule_pair_maximum_penetration_m": {
            " x ".join(pair): penetration
            for pair, penetration in sorted(capsule_pair_penetrations.items())
        },
        "nonignored_capsule_pair_penetration_by_pose_m": pose_capsule_pair_penetrations,
        "nonexempt_table_maximum_penetration_m": table_penetrations,
        "pose_results": results,
        "path_results": path_results,
        "all_representative_poses_clear": all(item["ok"] for item in results),
        "all_requested_paths_clear": all(item["ok"] for item in path_results),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if (
        output["all_representative_poses_clear"]
        and output["all_requested_paths_clear"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
