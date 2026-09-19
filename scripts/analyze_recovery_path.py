#!/usr/bin/env python3
"""离线分析从保守碰撞包络内现姿态退出到安全中转位的单调净距。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qingyun.grabbing.kinematics_ext import ArmModel  # noqa: E402
from qingyun.grabbing.safety import (  # noqa: E402
    CollisionChecker,
    JointLimits,
    build_error_envelope,
    segment_segment_distance,
    validate_segment,
)
from qingyun.grabbing.trajectory import make_joint_segment  # noqa: E402
from scripts.validate_home_waypoint_motion import _load_calibration_motion_params  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--start-deg", required=True, type=json.loads)
    parser.add_argument("--gripper-pct", required=True, type=float)
    parser.add_argument("--duration-s", type=float, default=20.0)
    args = parser.parse_args()

    params, temporary = _load_calibration_motion_params(args.profile.resolve(strict=True))
    try:
        q0 = np.asarray(args.start_deg, dtype=float)
        q1 = np.asarray(params.workspace.safe_waypoints_deg[0], dtype=float)
        model = ArmModel(params)
        checker = CollisionChecker(
            params,
            model,
            build_error_envelope(params, params.collision.max_joint_substep_deg),
        )
        segment = make_joint_segment(
            "serial_recovery_to_waypoint",
            q0,
            q1,
            args.gripper_pct,
            params,
            min_duration_s=args.duration_s,
        )

        def axes_at(index: int) -> dict:
            q = segment.joints_deg[index]
            if index + 1 < segment.nodes:
                delta = segment.joints_deg[index + 1] - q
            elif index:
                delta = q - segment.joints_deg[index - 1]
            else:
                delta = np.zeros(5)
            return {
                cid: (a, b, radius)
                for cid, a, b, radius in checker._axes(  # noqa: SLF001 - evidence metric
                    q, args.gripper_pct, delta
                )
            }

        # 当前位置已经由实体机械臂证明可存在。自动找出保守胶囊在起点的全部既有
        # 重叠；只临时豁免这些对，完整validate_segment仍会拒绝桌面、障碍与新碰撞。
        start_axes = axes_at(0)
        ids = list(start_axes)
        initial_pairs: list[tuple[str, str]] = []
        for i, first in enumerate(ids):
            for second in ids[i + 1 :]:
                if frozenset((first, second)) in checker._ignored:  # noqa: SLF001
                    continue
                a0, a1, ar = start_axes[first]
                b0, b1, br = start_axes[second]
                if segment_segment_distance(a0, a1, b0, b1) - ar - br < 0.0:
                    initial_pairs.append((first, second))
        for pair in initial_pairs:
            checker._ignored.add(frozenset(pair))  # noqa: SLF001 - offline analysis only
        checked = validate_segment(
            segment, model, params, JointLimits.from_params(params), checker
        )

        axes_by_node = [axes_at(i) for i in range(segment.nodes)]
        pair_results = []
        all_monotonic_and_clear = bool(initial_pairs)
        for pair in initial_pairs:
            clearances = []
            for axes in axes_by_node:
                a0, a1, ar = axes[pair[0]]
                b0, b1, br = axes[pair[1]]
                clearances.append(
                    segment_segment_distance(a0, a1, b0, b1) - ar - br
                )
            clearance = np.asarray(clearances)
            changes = np.diff(clearance)
            first_clear = np.flatnonzero(clearance >= 0.0)
            monotonic = bool(np.all(changes >= -1e-5))
            clears = bool(first_clear.size and clearance[-1] >= 0.0)
            all_monotonic_and_clear &= monotonic and clears
            pair_results.append({
                "pair": list(pair),
                "start_mm": float(clearance[0] * 1000.0),
                "minimum_mm": float(np.min(clearance) * 1000.0),
                "end_mm": float(clearance[-1] * 1000.0),
                "largest_step_decrease_mm": float(min(0.0, np.min(changes)) * 1000.0),
                "first_nonnegative_node": int(first_clear[0]) if first_clear.size else None,
                "monotonic_nondecreasing_with_0_01mm_tolerance": monotonic,
                "clears_by_target": clears,
            })
        result = {
            "nodes": checked.nodes,
            "duration_s": float(segment.duration_s),
            "initial_conservative_overlap_pairs": pair_results,
            "all_initial_overlaps_monotonic_and_clear_by_target": all_monotonic_and_clear,
            "all_other_limit_motion_and_collision_checks": "PASS",
            "peak_joint_velocity_deg_s": [
                float(v) for v in checked.peak_joint_velocity_deg_s
            ],
            "peak_joint_acceleration_deg_s2": [
                float(v) for v in checked.peak_joint_acceleration_deg_s2
            ],
            "peak_tcp_velocity_m_s": float(checked.peak_tcp_velocity_m_s),
            "peak_tcp_acceleration_m_s2": float(checked.peak_tcp_acceleration_m_s2),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if all_monotonic_and_clear else 2
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
