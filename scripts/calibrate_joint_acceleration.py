#!/usr/bin/env python3
"""CAL-059：当前姿态恢复到中转位后，完成五轴加速度阶梯并卸力。"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.executor import SyncExecutor  # noqa: E402
from qingyun.grabbing.kinematics_ext import ArmModel  # noqa: E402
from qingyun.grabbing.motor_control import (  # noqa: E402
    CalibrationReader,
    Sts3215MotorController,
    StsProtocol,
)
from qingyun.grabbing.safety import (  # noqa: E402
    CollisionChecker,
    JointLimits,
    build_error_envelope,
    validate_segment,
)
from qingyun.grabbing.trajectory import make_joint_segment  # noqa: E402
from scripts.calibrate import (  # noqa: E402
    _peak_command_acceleration,
    new_run_id,
    new_sample_id,
)
from scripts.move_gripper_raw import _read_raw, _write_raw  # noqa: E402
from scripts.read_joint_direction import (  # noqa: E402
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402
from scripts.validate_home_waypoint_motion import (  # noqa: E402
    _feedback_dict,
    _load_calibration_motion_params,
)


OFFSETS_DEG = np.array([14.0, 14.0, 12.0, 12.0, 14.0])
ACCELERATION_LEVELS_DEG_S2 = (120.0, 190.0, 260.0)


class CaptureExecutor(SyncExecutor):
    """复用生产执行器，并按轨迹保存真实反馈与本次加速度条件。"""

    def __init__(self, *args, load_label: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.load_label = load_label
        self.context: dict[str, Any] | None = None
        self.context_rows: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []

    def read_feedback(self):
        feedback = super().read_feedback()
        if self.context is not None:
            command = (
                np.asarray(self.state.last_submitted_deg, float)
                if self.state.last_submitted_deg is not None
                else np.asarray(feedback.angles_deg, float)
            )
            self.context_rows.append({
                "record_type": "sample",
                "trajectory_id": self.context["trajectory_id"],
                "segment": self.context["trajectory_id"],
                "command_deg": [float(v) for v in command],
                "feedback_deg": [float(v) for v in feedback.angles_deg],
                "speed_deg_s": [float(v) for v in feedback.speeds_deg_s],
                "current_ma": [float(v) for v in feedback.currents_ma],
                "tcp_reference_B_m": None,
                "time_ns": int(feedback.sample_start_ns),
                "gripper_pct": float(feedback.gripper_pct),
                "load_label": self.load_label,
                "stop_reason": None,
                "passed": None,
                "direction": int(self.context["direction"]),
                "active_axis": self.context["active_axis"],
                "requested_peak_acceleration_deg_s2": float(
                    self.context["requested_peak_acceleration_deg_s2"]
                ),
            })
        return feedback

    def play_captured(self, segment, *, axis_index: int, requested_acceleration: float) -> None:
        delta = float(segment.end_joints[axis_index] - segment.start_joints[axis_index])
        if abs(delta) <= 1e-9:
            raise ValueError("CAL-059 采集段不能是零位移")
        self.context = {
            "trajectory_id": segment.name,
            "direction": 1 if delta > 0.0 else -1,
            "active_axis": JOINT_NAMES[axis_index],
            "requested_peak_acceleration_deg_s2": requested_acceleration,
        }
        self.context_rows = []
        try:
            self.play_segment(segment)
        except BaseException as exc:
            for row in self.context_rows:
                row["passed"] = False
                row["stop_reason"] = f"{type(exc).__name__}: {exc}"
            self.rows.extend(self.context_rows)
            raise
        else:
            for row in self.context_rows:
                row["passed"] = True
            self.rows.extend(self.context_rows)
        finally:
            self.context = None
            self.context_rows = []


def _candidate_params(params, axis_index: int, acceleration_deg_s2: float):
    acceleration = np.asarray(params.motion.max_acceleration_deg_s2, float).copy()
    acceleration[axis_index] = acceleration_deg_s2
    return replace(
        params,
        motion=replace(params.motion, max_acceleration_deg_s2=acceleration),
    )


def _build_test_segments(params, waypoint: np.ndarray, gripper_pct: float):
    """构建30段：三档 × 五轴 × 去/回，并对每段做完整碰撞/动态校验。"""
    model = ArmModel(params)
    limits = JointLimits.from_params(params)
    checker = CollisionChecker(
        params,
        model,
        build_error_envelope(params, params.collision.max_joint_substep_deg),
    )
    segments = []
    preflight = []
    for level_index, requested_acceleration in enumerate(
        ACCELERATION_LEVELS_DEG_S2, start=1
    ):
        side = 1.0 if level_index % 2 else -1.0
        for axis_index in range(5):
            target = waypoint.copy()
            target[axis_index] += side * OFFSETS_DEG[axis_index]
            planning_params = _candidate_params(
                params, axis_index, requested_acceleration
            )
            for leg_index, (start, end) in enumerate(
                ((waypoint, target), (target, waypoint)), start=1
            ):
                name = f"cal059_L{level_index}_J{axis_index + 1}_leg{leg_index}"
                segment = make_joint_segment(
                    name, start, end, gripper_pct, planning_params
                )
                checked = validate_segment(
                    segment, model, planning_params, limits, checker
                )
                planned_acceleration = np.asarray(
                    checked.peak_joint_acceleration_deg_s2, float
                )
                if planned_acceleration[axis_index] > requested_acceleration + 1e-6:
                    raise RuntimeError(
                        f"{name} 离散命令加速度超过请求值："
                        f"{planned_acceleration[axis_index]:.3f}>{requested_acceleration:.3f}"
                    )
                segments.append(
                    (segment, axis_index, float(requested_acceleration))
                )
                preflight.append({
                    "trajectory_id": name,
                    "active_axis": JOINT_NAMES[axis_index],
                    "direction": 1 if end[axis_index] > start[axis_index] else -1,
                    "start_deg": [float(v) for v in start],
                    "target_deg": [float(v) for v in end],
                    "duration_s": float(segment.duration_s),
                    "requested_peak_acceleration_deg_s2": float(
                        requested_acceleration
                    ),
                    "planned_peak_joint_velocity_deg_s": [
                        float(v) for v in checked.peak_joint_velocity_deg_s
                    ],
                    "planned_peak_joint_acceleration_deg_s2": [
                        float(v) for v in planned_acceleration
                    ],
                    "planned_peak_tcp_velocity_m_s": float(
                        checked.peak_tcp_velocity_m_s
                    ),
                    "planned_peak_tcp_acceleration_m_s2": float(
                        checked.peak_tcp_acceleration_m_s2
                    ),
                    "full_limit_motion_collision_check": "PASS",
                })
    return segments, preflight


def _write_jsonl(path: Path, run_id: str, metadata: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for payload in [metadata, *rows]:
            handle.write(json.dumps({
                "stage": "motion",
                "run_id": run_id,
                "sample_id": new_sample_id(),
                "captured_at_ns": int(payload.get("time_ns", time.monotonic_ns())),
                "payload": payload,
            }, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--recovery-duration-s", type=float, default=20.0)
    parser.add_argument("--settle-timeout-s", type=float, default=8.0)
    parser.add_argument("--settle-position-tolerance-deg", type=float, default=2.8)
    parser.add_argument("--start-segment-index", type=int, default=1)
    parser.add_argument("--operator-recovery-collision-override", action="store_true")
    parser.add_argument("--operator-confirmation")
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--load-label", required=True)
    args = parser.parse_args()

    if args.raw_output.exists() or args.report_output.exists():
        raise SystemExit("拒绝覆盖既有 CAL-059 证据")
    if args.recovery_duration_s < 10.0:
        raise SystemExit("恢复段至少10秒")
    if args.settle_timeout_s <= 0.0:
        raise SystemExit("到位等待时间必须为正")
    if not 1 <= args.start_segment_index <= 30:
        raise SystemExit("start-segment-index 必须在[1,30]")
    if not args.operator_recovery_collision_override or not args.operator_confirmation:
        raise SystemExit("当前姿态恢复段需要具名操作者碰撞例外确认")

    profile_path = args.profile.resolve(strict=True)
    calibration = load_calibration_profile(profile_path)
    stable_port = Path(calibration.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"稳定串口不是本机已确认地址：{stable_port}")
    resolved_port = stable_port.resolve(strict=True)
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"稳定地址未解析到 ttyACM：{resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit(f"USB 身份不匹配：{identity}")

    reader = CalibrationReader(calibration)
    try:
        torque_before = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        snapshot = reader.read_snapshot()
    finally:
        reader.close()
    if torque_before != [0] * len(MOTOR_NAMES):
        raise SystemExit(f"六轴并非全部卸力，拒绝运动：{torque_before}")
    if ids != list(range(1, 7)) or models != [777] * 6:
        raise SystemExit(f"舵机身份不匹配：ids={ids}, models={models}")

    params, temporary_profile = _load_calibration_motion_params(profile_path)
    atexit.register(temporary_profile.unlink, missing_ok=True)
    params = replace(
        params,
        motion=replace(
            params.motion,
            settle_position_tol_deg=np.full(
                5, float(args.settle_position_tolerance_deg)
            ),
            settle_timeout_s=float(args.settle_timeout_s),
        ),
    )
    waypoint = np.asarray(params.workspace.safe_waypoints_deg[0], float)
    q_start = np.asarray(snapshot["q_urdf_deg"], float)
    gripper_pct = float(snapshot["gripper_pct"])
    model = ArmModel(params)
    limits = JointLimits.from_params(params)
    recovery = make_joint_segment(
        "cal059_current_to_waypoint",
        q_start,
        waypoint,
        gripper_pct,
        params,
        min_duration_s=args.recovery_duration_s,
    )
    recovery_checked = validate_segment(recovery, model, params, limits, None)
    normal_recovery_rejection = None
    normal_checker = CollisionChecker(
        params,
        model,
        build_error_envelope(params, params.collision.max_joint_substep_deg),
    )
    try:
        validate_segment(recovery, model, params, limits, normal_checker)
    except Exception as exc:
        normal_recovery_rejection = f"{type(exc).__name__}: {exc}"
    test_segments, test_preflight = _build_test_segments(
        params, waypoint, gripper_pct
    )
    selected_test_segments = test_segments[args.start_segment_index - 1:]

    run_id = new_run_id()
    started_at = datetime.now().astimezone().isoformat()
    report: dict[str, Any] = {
        "schema_version": 1,
        "parameter_id": "CAL-059",
        "test": "current_to_waypoint_then_joint_acceleration_sweep_then_torque_off",
        "started_at": started_at,
        "status": "RUNNING",
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "stable_port": str(stable_port),
        "resolved_port": str(resolved_port),
        "usb_identity": identity,
        "environment": {
            "power": args.power,
            "ambient_temperature_c": args.ambient_temperature_c,
            "base_mount": args.base_mount,
            "load_label": args.load_label,
        },
        "readonly_preflight": {
            "torque_enabled": torque_before,
            "ids": ids,
            "model_numbers": models,
            "snapshot": snapshot,
        },
        "recovery": {
            "start_deg": [float(v) for v in q_start],
            "target_deg": [float(v) for v in waypoint],
            "duration_s": float(recovery.duration_s),
            "operator_collision_override": True,
            "operator_confirmation": args.operator_confirmation,
            "normal_collision_result": (
                "PASS" if normal_recovery_rejection is None
                else normal_recovery_rejection
            ),
            "limit_and_motion_check": "PASS",
            "peak_joint_velocity_deg_s": [
                float(v) for v in recovery_checked.peak_joint_velocity_deg_s
            ],
        },
        "temporary_settle_position_tolerance_deg": [
            float(args.settle_position_tolerance_deg)
        ] * 5,
        "temporary_settle_timeout_s": float(args.settle_timeout_s),
        "settle_tolerance_profile_changed": False,
        "test_preflight": test_preflight,
        "start_segment_index": int(args.start_segment_index),
        "selected_test_segment_count": len(selected_test_segments),
        "eeprom_written": False,
        "profile_written": False,
        "torque_enabled_after": None,
    }

    motor = None
    torque_may_be_enabled = False
    executor = None
    failure: BaseException | None = None
    try:
        motor = Sts3215MotorController(params, initialize=False)
        motor.initialize(verify_identity=True, enable_torque=False)
        feedback_before = motor.get_feedback()
        report["feedback_before_goal_seed"] = _feedback_dict(feedback_before)
        raw_targets = motor._validate_and_encode(  # noqa: SLF001
            feedback_before.angles_deg, feedback_before.gripper_pct
        )
        values = [StsProtocol.split_u16(raw) for raw in raw_targets]
        torque_may_be_enabled = True
        motor._protocol.sync_write_targets(  # noqa: SLF001
            [axis.servo_id for axis in motor.mapping.axes],
            motor._goal_addr,  # noqa: SLF001
            motor._goal_len,  # noqa: SLF001
            values,
            time.monotonic() + max(0.1, params.timing.write_timeout_s),
        )
        goal_readback = [
            _read_raw(motor._protocol, motor.mapping, axis, "Goal_Position")  # noqa: SLF001
            for axis in motor.mapping.axes
        ]
        if goal_readback != raw_targets:
            raise RuntimeError("当前位置目标预置读回不一致")
        torque_on = [
            _read_raw(motor._protocol, motor.mapping, axis, "Torque_Enable")  # noqa: SLF001
            for axis in motor.mapping.axes
        ]
        if torque_on == [1] * 6:
            motor._torque_enabled = True  # noqa: SLF001
            report["torque_enable_source"] = "goal_seed_implicit_verified"
        else:
            motor.configure_torque(True, deadline_s=time.monotonic() + 1.0)
            time.sleep(0.1)
            motor._protocol.reset_input()  # noqa: SLF001
            torque_on = [
                _read_raw(motor._protocol, motor.mapping, axis, "Torque_Enable")  # noqa: SLF001
                for axis in motor.mapping.axes
            ]
            report["torque_enable_source"] = "explicit_write"
        if torque_on != [1] * 6:
            raise RuntimeError(f"六轴力矩未全部使能：{torque_on}")
        report["torque_enabled_during"] = torque_on
        time.sleep(0.2)
        motor._protocol.reset_input()  # noqa: SLF001

        executor = CaptureExecutor(motor, params, load_label=args.load_label)
        executor.play_segment(recovery)
        recovery_feedback = motor.get_feedback()
        report["recovery_feedback"] = _feedback_dict(recovery_feedback)
        report["recovery_endpoint_abs_error_deg"] = [
            float(v) for v in np.abs(recovery_feedback.angles_deg - waypoint)
        ]
        for index, (segment, axis_index, requested_acceleration) in enumerate(
            selected_test_segments, start=args.start_segment_index
        ):
            print(
                f"CAL-059 {index}/{len(test_segments)} {segment.name} "
                f"target={requested_acceleration:.0f}deg/s^2",
                flush=True,
            )
            executor.play_captured(
                segment,
                axis_index=axis_index,
                requested_acceleration=requested_acceleration,
            )

        passed_rows = [row for row in executor.rows if row["passed"]]
        peak_acceleration, acceleration_groups = _peak_command_acceleration(
            passed_rows, params.timing.fps
        )
        current = np.abs(np.asarray([row["current_ma"] for row in passed_rows], float))
        error = np.abs(np.asarray([
            np.asarray(row["command_deg"], float)
            - np.asarray(row["feedback_deg"], float)
            for row in passed_rows
        ]))
        report["sample_rows"] = len(executor.rows)
        report["acceleration_trajectory_groups"] = acceleration_groups
        report["measured_peak_command_acceleration_deg_s2"] = [
            float(v) for v in peak_acceleration
        ]
        if np.any(peak_acceleration > max(ACCELERATION_LEVELS_DEG_S2) * 1.01):
            raise RuntimeError(
                "命令加速度峰值超过请求量级，拒绝生成候选："
                + json.dumps([float(v) for v in peak_acceleration])
            )
        report["proposed_max_acceleration_deg_s2"] = [
            round(max(float(v) * 0.85, 50.0), 2) for v in peak_acceleration
        ]
        report["tracking_p99_deg"] = [
            float(v) for v in np.percentile(error, 99, axis=0)
        ]
        peak_current = np.max(current, axis=0)
        report["peak_joint_current_ma"] = [float(v) for v in peak_current]
        report["software_hard_current_ma"] = [
            float(v) for v in params.joints.hard_current_ma
        ]
        report["current_threshold_result"] = (
            "PASS" if np.all(peak_current < params.joints.hard_current_ma) else "FAIL"
        )
        report["status"] = "PASS"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if executor is not None:
            if executor.state.last_feedback is not None:
                report["failure_feedback_before_torque_disable"] = _feedback_dict(
                    executor.state.last_feedback
                )
                report["failure_endpoint_abs_error_from_waypoint_deg"] = [
                    float(v) for v in np.abs(
                        executor.state.last_feedback.angles_deg - waypoint
                    )
                ]
            report["failure_peak_following_error_deg"] = [
                float(v) for v in executor.state.peak_following_error_deg
            ]
            report["failure_peak_joint_current_ma"] = [
                float(v) for v in executor.state.peak_joint_current_ma
            ]
    finally:
        if motor is not None:
            if torque_may_be_enabled:
                try:
                    for axis in motor.mapping.axes:
                        _write_raw(  # noqa: SLF001
                            motor._protocol, motor.mapping, axis, "Torque_Enable", 0
                        )
                        time.sleep(0.03)
                    torque_after = []
                    for axis in motor.mapping.axes:
                        last_error = None
                        for _attempt in range(3):
                            try:
                                torque_after.append(_read_raw(  # noqa: SLF001
                                    motor._protocol,
                                    motor.mapping,
                                    axis,
                                    "Torque_Enable",
                                ))
                                break
                            except Exception as exc:
                                last_error = exc
                                time.sleep(0.05)
                        else:
                            assert last_error is not None
                            raise last_error
                    report["torque_enabled_after"] = torque_after
                    motor._torque_enabled = False  # noqa: SLF001
                except Exception as exc:
                    report["torque_disable_failure"] = f"{type(exc).__name__}: {exc}"
            motor.close()
        temporary_profile.unlink(missing_ok=True)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        rows = executor.rows if executor is not None else []
        metadata = {
            "record_type": "metadata",
            "source": "hardware",
            "parameter_id": "CAL-059",
            "captured_at": started_at,
            "status": report["status"],
            "profile": report["profile"],
            "profile_sha256": report["profile_sha256"],
            "stable_port": report["stable_port"],
            "resolved_port": report["resolved_port"],
            "usb_identity": identity,
            "environment": report["environment"],
            "sample_rows": len(rows),
            "torque_enabled_before": torque_before,
            "torque_enabled_after": report.get("torque_enabled_after"),
            "eeprom_written": False,
            "failure": report.get("failure"),
        }
        _write_jsonl(args.raw_output, run_id, metadata, rows)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if failure is not None:
        raise SystemExit(f"CAL-059 失败：{failure}; evidence={args.report_output}")
    if report.get("torque_enabled_after") != [0] * 6:
        raise SystemExit("CAL-059 运动完成，但未确认六轴全部卸力")
    print(
        "CAL-059 采集通过并已卸力；候选 max_acceleration="
        + json.dumps(report["proposed_max_acceleration_deg_s2"])
        + f"; evidence={args.report_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
