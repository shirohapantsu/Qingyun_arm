#!/usr/bin/env python3
"""CAL-046/047 空载低速 waypoint→home→waypoint 真机往返验证。"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import (  # noqa: E402
    load_calibration_profile,
    load_motion_params,
)
from qingyun.grabbing.arm_control import ArmController  # noqa: E402
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
    segment_segment_distance,
    validate_segment,
)
from qingyun.grabbing.trajectory import make_joint_segment  # noqa: E402
from scripts.read_joint_direction import (  # noqa: E402
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
)
from scripts.move_gripper_raw import _read_raw, _write_raw  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def _feedback_dict(feedback) -> dict:
    return {
        "angles_deg": [float(v) for v in feedback.angles_deg],
        "speeds_deg_s": [float(v) for v in feedback.speeds_deg_s],
        "currents_ma": [float(v) for v in feedback.currents_ma],
        "gripper_pct": float(feedback.gripper_pct),
        "gripper_speed_pct_s": float(feedback.gripper_speed_pct_s),
        "gripper_current_ma": float(feedback.gripper_current_ma),
        "sample_start_ns": int(feedback.sample_start_ns),
        "sample_end_ns": int(feedback.sample_end_ns),
        "sequence": int(feedback.sequence),
    }


def _load_calibration_motion_params(profile_path: Path):
    """以临时 simulation 状态加载完整校验结构，不修改真实 draft profile。

    标定期 profile 必须保持 draft，生产 real loader 因此会拒绝它；这里仍复用完整
    MotionParams 结构、限位、轨迹和碰撞检查，仅把流程状态写到同目录临时副本。
    临时副本不会被保存为标定结果，也不会绕过任何数值/资源校验。
    """
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["status"] = "simulation"
    payload["verification"] = None
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=".cal047-motion-",
        dir=profile_path.parent, delete=False, encoding="utf-8",
    )
    try:
        json.dump(payload, handle)
        handle.close()
        temporary = Path(handle.name)
        return load_motion_params(temporary, mode="mock"), temporary
    except Exception:
        Path(handle.name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segment-duration-s", type=float, default=8.0)
    parser.add_argument("--start-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        help="仅本次标定运行覆盖motion.settle_timeout_s，不写回profile",
    )
    parser.add_argument(
        "--settle-position-tolerance-deg",
        type=float,
        help="仅本次标定运行将五轴到位容差临时设为该值，不写回profile",
    )
    parser.add_argument(
        "--mode",
        choices=("roundtrip", "recover-to-waypoint"),
        default="roundtrip",
        help="roundtrip要求起点已在中转位；recover模式先预检再低速串口恢复到中转位",
    )
    parser.add_argument("--operator-collision-override", action="store_true")
    parser.add_argument(
        "--allow-margin-intrusion-recovery",
        action="store_true",
        help="显式允许恢复模式从侵入运行margin、但仍在绝对限位内的起点单调退出",
    )
    parser.add_argument("--operator-confirmation")
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--load-label", required=True)
    args = parser.parse_args()
    if args.segment_duration_s < 5.0:
        raise SystemExit("首次往返每段至少5秒")
    if args.start_tolerance_deg <= 0.0:
        raise SystemExit("start-tolerance-deg 必须为正")
    if args.settle_timeout_s is not None and args.settle_timeout_s <= 0.0:
        raise SystemExit("settle-timeout-s必须为正")
    if (
        args.settle_position_tolerance_deg is not None
        and args.settle_position_tolerance_deg <= 0.0
    ):
        raise SystemExit("settle-position-tolerance-deg必须为正")
    if args.operator_collision_override:
        if args.mode != "recover-to-waypoint":
            raise SystemExit("碰撞模型操作者覆盖只允许用于recover-to-waypoint")
        if not args.operator_confirmation:
            raise SystemExit("碰撞模型操作者覆盖必须保存现场确认原文")
    if args.allow_margin_intrusion_recovery and args.mode != "recover-to-waypoint":
        raise SystemExit("margin侵入恢复只允许用于recover-to-waypoint")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有证据：{args.output}")

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

    # 第一重前置使用严格只读入口；在确认起点和六轴卸力前，不构造有写能力的驱动。
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
    if ids != list(range(1, len(MOTOR_NAMES) + 1)) or models != [777] * len(MOTOR_NAMES):
        raise SystemExit(f"舵机身份不匹配：ids={ids}, models={models}")

    params, temporary_profile = _load_calibration_motion_params(profile_path)
    atexit.register(temporary_profile.unlink, missing_ok=True)
    configured_settle_timeout_s = float(params.motion.settle_timeout_s)
    configured_settle_position_tol_deg = np.asarray(
        params.motion.settle_position_tol_deg, dtype=float
    ).copy()
    if (
        args.settle_timeout_s is not None
        or args.settle_position_tolerance_deg is not None
    ):
        effective_settle_position_tol_deg = (
            np.full(5, float(args.settle_position_tolerance_deg), dtype=float)
            if args.settle_position_tolerance_deg is not None
            else configured_settle_position_tol_deg
        )
        params = replace(
            params,
            motion=replace(
                params.motion,
                settle_timeout_s=(
                    float(args.settle_timeout_s)
                    if args.settle_timeout_s is not None
                    else params.motion.settle_timeout_s
                ),
                settle_position_tol_deg=effective_settle_position_tol_deg,
            ),
        )
    waypoint = np.asarray(params.workspace.safe_waypoints_deg[0], dtype=float)
    home = np.asarray(params.workspace.home_joints_deg, dtype=float)
    q_start = np.asarray(snapshot["q_urdf_deg"], dtype=float)
    start_error = np.abs(q_start - waypoint)
    configured_joint_limits = JointLimits.from_params(params)
    configured_joint_margin_deg = np.asarray(params.joints.margin_deg, dtype=float).copy()
    initial_limit_return = []
    if args.mode == "roundtrip" and np.any(start_error > args.start_tolerance_deg):
        temporary_profile.unlink(missing_ok=True)
        raise SystemExit(
            "当前姿态不是第1中转位，拒绝上力：当前角度="
            + json.dumps([float(v) for v in q_start])
            + "；目标角度="
            + json.dumps([float(v) for v in waypoint])
            + "；逐轴偏差="
            + json.dumps([float(v) for v in start_error])
        )

    preflight_recovery = None
    if args.mode == "recover-to-waypoint":
        if snapshot["gripper_pct"] is None:
            temporary_profile.unlink(missing_ok=True)
            raise SystemExit("夹爪开度尚未标定，拒绝恢复轨迹")
        # 卸力后的重力下落可能把关节带入“绝对行程内、但侵入运行 margin”的区域。
        # 恢复模式只对这样的轴临时取消 margin，并在下方证明整段从第0节点开始
        # 单调返回原有效限位。绝不扩张 URDF/实测/应用三重交集，也不写回 profile。
        violating_indices = [
            i
            for i in range(len(JOINT_NAMES))
            if (
                q_start[i] < configured_joint_limits.lower[i] - 1e-6
                or q_start[i] > configured_joint_limits.upper[i] + 1e-6
            )
        ]
        if violating_indices:
            if not args.allow_margin_intrusion_recovery:
                temporary_profile.unlink(missing_ok=True)
                raise SystemExit(
                    "当前姿态侵入关节运行margin；未获显式限位余量例外授权，拒绝上力："
                    + json.dumps(
                        configured_joint_limits.violations(q_start), ensure_ascii=False
                    )
                )
            recovery_margin_deg = np.asarray(params.joints.margin_deg, dtype=float).copy()
            recovery_margin_deg[violating_indices] = 0.0
            recovery_params = replace(
                params,
                joints=replace(params.joints, margin_deg=recovery_margin_deg),
            )
            recovery_limits = JointLimits.from_params(recovery_params)
            if not recovery_limits.inside(q_start):
                temporary_profile.unlink(missing_ok=True)
                raise SystemExit(
                    "当前姿态已超出取消margin后的绝对限位，拒绝上力："
                    + json.dumps(recovery_limits.violations(q_start), ensure_ascii=False)
                )
            params = recovery_params
        # 必须在构造有写能力的驱动前完成整段校验；失败时全程只有READ。
        model = ArmModel(params)
        limits = JointLimits.from_params(params)
        checker = CollisionChecker(
            params,
            model,
            build_error_envelope(params, params.collision.max_joint_substep_deg),
        )
        segment = make_joint_segment(
            "serial_recovery_to_waypoint",
            q_start,
            waypoint,
            float(snapshot["gripper_pct"]),
            params,
            min_duration_s=args.segment_duration_s,
        )
        for i in violating_indices:
            values = np.asarray(segment.joints_deg[:, i], dtype=float)
            changes = np.diff(values)
            if q_start[i] > configured_joint_limits.upper[i]:
                monotonic_return = bool(
                    waypoint[i] <= configured_joint_limits.upper[i] + 1e-6
                    and np.all(changes <= 1e-9)
                )
                side = "above_upper"
            else:
                monotonic_return = bool(
                    waypoint[i] >= configured_joint_limits.lower[i] - 1e-6
                    and np.all(changes >= -1e-9)
                )
                side = "below_lower"
            if not monotonic_return:
                temporary_profile.unlink(missing_ok=True)
                raise SystemExit(
                    f"{JOINT_NAMES[i]}未从margin侵入区单调返回有效限位，拒绝上力"
                )
            inside_nodes = np.flatnonzero(
                (values >= configured_joint_limits.lower[i] - 1e-6)
                & (values <= configured_joint_limits.upper[i] + 1e-6)
            )
            if not inside_nodes.size:
                temporary_profile.unlink(missing_ok=True)
                raise SystemExit(f"{JOINT_NAMES[i]}恢复段末仍未进入原有效限位，拒绝上力")
            initial_limit_return.append({
                "axis": JOINT_NAMES[i],
                "side": side,
                "start_deg": float(q_start[i]),
                "configured_effective_lower_deg": float(configured_joint_limits.lower[i]),
                "configured_effective_upper_deg": float(configured_joint_limits.upper[i]),
                "configured_margin_deg": float(configured_joint_margin_deg[i]),
                "temporary_margin_deg": float(params.joints.margin_deg[i]),
                "first_node_inside_configured_limit": int(inside_nodes[0]),
                "monotonic_return": True,
            })
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
                for cid, a, b, radius in checker._axes(  # noqa: SLF001 - recovery proof
                    q, float(snapshot["gripper_pct"]), delta
                )
            }

        start_axes = axes_at(0)
        capsule_ids = list(start_axes)
        initial_pairs: list[tuple[str, str]] = []
        for i, first in enumerate(capsule_ids):
            for second in capsule_ids[i + 1 :]:
                if frozenset((first, second)) in checker._ignored:  # noqa: SLF001
                    continue
                a0, a1, ar = start_axes[first]
                b0, b1, br = start_axes[second]
                if segment_segment_distance(a0, a1, b0, b1) - ar - br < 0.0:
                    initial_pairs.append((first, second))
        for pair in initial_pairs:
            checker._ignored.add(frozenset(pair))  # noqa: SLF001 - recovery proof only
        # 操作者已由实体路径确认胶囊模型误报时，只对本次恢复路线跳过胶囊判定；
        # validate_segment仍保留限位、步长、关节/TCP速度与加速度检查。默认路径则
        # 继续只豁免起点已有且后续必须单调退出的保守重叠。
        checked = validate_segment(
            segment,
            model,
            params,
            limits,
            None if args.operator_collision_override else checker,
        )
        overlap_evidence = []
        axes_by_node = [axes_at(i) for i in range(segment.nodes)]
        for pair in initial_pairs:
            clearance = []
            for axes in axes_by_node:
                a0, a1, ar = axes[pair[0]]
                b0, b1, br = axes[pair[1]]
                clearance.append(
                    segment_segment_distance(a0, a1, b0, b1) - ar - br
                )
            values = np.asarray(clearance)
            changes = np.diff(values)
            first_clear = np.flatnonzero(values >= 0.0)
            monotonic = bool(np.all(changes >= -1e-5))
            clears = bool(first_clear.size and values[-1] >= 0.0)
            if (not monotonic or not clears) and not args.operator_collision_override:
                temporary_profile.unlink(missing_ok=True)
                raise SystemExit(
                    f"既有保守重叠{pair}未沿恢复路径单调退出，拒绝上力"
                )
            overlap_evidence.append({
                "pair": list(pair),
                "start_mm": float(values[0] * 1000.0),
                "minimum_mm": float(np.min(values) * 1000.0),
                "end_mm": float(values[-1] * 1000.0),
                "first_nonnegative_node": int(first_clear[0]),
                "monotonic_nondecreasing_with_0_01mm_tolerance": monotonic,
            })
        preflight_recovery = {
            "nodes": checked.nodes,
            "duration_s": float(segment.duration_s),
            "start_deg": [float(v) for v in q_start],
            "target_deg": [float(v) for v in waypoint],
            "gripper_pct": float(snapshot["gripper_pct"]),
            "peak_joint_velocity_deg_s": [
                float(v) for v in checked.peak_joint_velocity_deg_s
            ],
            "peak_joint_acceleration_deg_s2": [
                float(v) for v in checked.peak_joint_acceleration_deg_s2
            ],
            "peak_tcp_velocity_m_s": float(checked.peak_tcp_velocity_m_s),
            "peak_tcp_acceleration_m_s2": float(checked.peak_tcp_acceleration_m_s2),
            "initial_conservative_overlap_escape": overlap_evidence,
            "initial_margin_intrusion_return": initial_limit_return,
            "joint_margin_profile_changed": False,
            "joint_limit_and_motion_checks": "PASS",
            "collision_check": (
                "OPERATOR_OVERRIDE_FOR_THIS_RECOVERY_ROUTE_ONLY"
                if args.operator_collision_override
                else "PASS_WITH_MONOTONIC_INITIAL_OVERLAP_ESCAPE"
            ),
            "operator_confirmation": args.operator_confirmation,
        }

    report = {
        "schema_version": 1,
        "parameter_ids": ["CAL-046", "CAL-047"],
        "test": (
            "unloaded_low_speed_waypoint_home_waypoint_round_trip"
            if args.mode == "roundtrip"
            else "unloaded_low_speed_serial_recovery_to_waypoint"
        ),
        "started_at": datetime.now().astimezone().isoformat(),
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
        "segment_duration_s": args.segment_duration_s,
        "configured_settle_timeout_s": configured_settle_timeout_s,
        "effective_settle_timeout_s": float(params.motion.settle_timeout_s),
        "settle_timeout_profile_changed": False,
        "configured_settle_position_tol_deg": [
            float(v) for v in configured_settle_position_tol_deg
        ],
        "effective_settle_position_tol_deg": [
            float(v) for v in params.motion.settle_position_tol_deg
        ],
        "settle_position_tolerance_profile_changed": False,
        "waypoint_deg": [float(v) for v in waypoint],
        "home_deg": [float(v) for v in home],
        "readonly_preflight": {
            "torque_enabled": torque_before,
            "ids": ids,
            "model_numbers": models,
            "snapshot": snapshot,
            "start_error_from_waypoint_deg": [float(v) for v in start_error],
        },
        "goal_seeded_to_present_before_torque_enable": False,
        "preflight_recovery_segment": preflight_recovery,
        "eeprom_written": False,
        "operator_collision_override": bool(args.operator_collision_override),
        "operator_confirmation": args.operator_confirmation,
        "legs": [],
    }

    motor = None
    torque_may_be_enabled = False
    failure = None
    try:
        motor = Sts3215MotorController(params, initialize=False)
        motor.initialize(verify_identity=True, enable_torque=False)
        controller_torque_before = [
            _read_raw(motor._protocol, motor.mapping, axis, "Torque_Enable")  # noqa: SLF001
            for axis in motor.mapping.axes
        ]
        if controller_torque_before != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"驱动初始化后六轴并非全部卸力：{controller_torque_before}")
        feedback_before = motor.get_feedback()
        report["feedback_before_goal_seed"] = _feedback_dict(feedback_before)

        # CAL-052实测本机在Torque_Enable=0时广播Goal_Position会自动令六路变为1。
        # 因而目标虽然等于实测当前位置、不会要求位姿跳变，但从发送前起就必须按“可能
        # 已上力”清理，不能把它称为卸力状态下的无害预置。这里只写SRAM；
        # Homing/Min/Max等EEPROM均无写路径。
        raw_targets = motor._validate_and_encode(  # noqa: SLF001 - calibration safety preseed
            feedback_before.angles_deg, feedback_before.gripper_pct
        )
        servo_ids = [axis.servo_id for axis in motor.mapping.axes]
        values = [StsProtocol.split_u16(raw) for raw in raw_targets]
        torque_may_be_enabled = True
        motor._protocol.sync_write_targets(  # noqa: SLF001 - see safety comment above
            servo_ids, motor._goal_addr, motor._goal_len, values,  # noqa: SLF001
            time.monotonic() + max(0.1, params.timing.write_timeout_s),
        )
        goal_readback = [
            _read_raw(motor._protocol, motor.mapping, axis, "Goal_Position")  # noqa: SLF001
            for axis in motor.mapping.axes
        ]
        if goal_readback != raw_targets:
            raise RuntimeError(
                f"当前位置目标预置读回不一致：target={raw_targets}, readback={goal_readback}"
            )
        report["goal_seeded_to_present_before_torque_enable"] = True
        report["goal_seed_raw"] = raw_targets
        goal_seed_torque_readback = [
            _read_raw(motor._protocol, motor.mapping, axis, "Torque_Enable")  # noqa: SLF001
            for axis in motor.mapping.axes
        ]
        report["goal_seed_torque_readback"] = goal_seed_torque_readback

        # CAL-052及本次读回已经证明Goal_Position使六轴隐式上力。此处若再逐台冗余
        # 写Torque_Enable=1，真机连续两次在紧随其后的READ收到损坏/异常ID状态帧。
        # 已有六路硬件读回全1时直接同步控制器内部状态，等待迟到帧并清空输入；只有
        # 固件没有全部自动上力时才走显式逐台写1。生产通用初始化仍使用显式配置。
        if goal_seed_torque_readback == [1] * len(MOTOR_NAMES):
            motor._torque_enabled = True  # noqa: SLF001 - six-register readback is authoritative
            report["torque_enable_source"] = "goal_seed_implicit_enable_verified_by_six_reads"
            torque_on_readback = goal_seed_torque_readback
        else:
            motor.configure_torque(True, deadline_s=time.monotonic() + 1.0)
            time.sleep(0.10)
            motor._protocol.reset_input()  # noqa: SLF001 - drain delayed write-side frames
            torque_on_readback = [
                _read_raw(motor._protocol, motor.mapping, axis, "Torque_Enable")  # noqa: SLF001
                for axis in motor.mapping.axes
            ]
            report["torque_enable_source"] = "explicit_torque_enable_write"
        report["torque_enabled_readback"] = torque_on_readback
        if torque_on_readback != [1] * len(MOTOR_NAMES):
            raise RuntimeError(f"六轴力矩未全部使能：{torque_on_readback}")
        time.sleep(0.2)
        motor._protocol.reset_input()  # noqa: SLF001 - separate write/feedback phases
        feedback_after_enable = motor.get_feedback()
        enable_jump = np.abs(feedback_after_enable.angles_deg - feedback_before.angles_deg)
        report["feedback_after_torque_enable"] = _feedback_dict(feedback_after_enable)
        report["torque_enable_position_change_deg"] = [float(v) for v in enable_jump]
        if np.any(enable_jump > args.start_tolerance_deg):
            raise RuntimeError(
                "使能力矩后位置变化超过阈值：" + json.dumps([float(v) for v in enable_jump])
            )

        arm = ArmController(motor, params)
        legs = (("waypoint_to_home", home), ("home_to_waypoint", waypoint))
        if args.mode == "recover-to-waypoint":
            ticks_before = arm.executor.state.ticks_played
            started = time.monotonic()
            # 这条segment已经在上力前按“既有重叠只许单调退出”的专用规则验证；
            # 常规move_joints会在第0节点重新把既有保守重叠判成新碰撞，无法执行退出。
            arm.executor.play_segment(segment)
            feedback = motor.get_feedback()
            error = np.abs(feedback.angles_deg - waypoint)
            report["legs"].append({
                "name": "current_to_waypoint",
                "elapsed_s": time.monotonic() - started,
                "ticks_played": arm.executor.state.ticks_played - ticks_before,
                "target_deg": [float(v) for v in waypoint],
                "feedback": _feedback_dict(feedback),
                "endpoint_abs_error_deg": [float(v) for v in error],
            })
        else:
            for name, target in legs:
                ticks_before = arm.executor.state.ticks_played
                started = time.monotonic()
                arm.move_joints(target, min_duration_s=args.segment_duration_s)
                feedback = motor.get_feedback()
                error = np.abs(feedback.angles_deg - target)
                report["legs"].append({
                    "name": name,
                    "elapsed_s": time.monotonic() - started,
                    "ticks_played": arm.executor.state.ticks_played - ticks_before,
                    "target_deg": [float(v) for v in target],
                    "feedback": _feedback_dict(feedback),
                    "endpoint_abs_error_deg": [float(v) for v in error],
                })
        report["peak_following_error_deg"] = [
            float(v) for v in arm.executor.state.peak_following_error_deg
        ]
        report["peak_joint_current_ma"] = [
            float(v) for v in arm.executor.state.peak_joint_current_ma
        ]
        report["status"] = "PASS"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if motor is not None and torque_may_be_enabled:
            try:
                # 对“未在窗口内到位”保存卸力前反馈，避免重力下落后的姿态被误当成
                # 上力末端误差；若通信本身已失败则不为取证延误清理超过一次读取。
                report["failure_feedback_before_torque_disable"] = _feedback_dict(
                    motor.get_feedback()
                )
            except Exception as feedback_exc:
                report["failure_feedback_capture_error"] = (
                    f"{type(feedback_exc).__name__}: {feedback_exc}"
                )
    finally:
        if motor is not None:
            if torque_may_be_enabled:
                try:
                    # 失败清理不能依赖尚未标定的20 ms运行期时限。复用真机夹爪定位
                    # 已验证的逐台长预算读写，并在每次READ前清理迟到残帧。
                    for axis in motor.mapping.axes:
                        _write_raw(motor._protocol, motor.mapping, axis, "Torque_Enable", 0)  # noqa: SLF001
                        time.sleep(0.03)
                    torque_after = []
                    torque_read_attempts = []
                    for axis in motor.mapping.axes:
                        last_error = None
                        for attempt in range(1, 4):
                            try:
                                torque_after.append(
                                    _read_raw(
                                        motor._protocol, motor.mapping, axis,  # noqa: SLF001
                                        "Torque_Enable",
                                    )
                                )
                                torque_read_attempts.append(attempt)
                                break
                            except Exception as exc:
                                last_error = exc
                                time.sleep(0.05)
                        else:
                            assert last_error is not None
                            raise last_error
                    report["torque_enabled_after"] = torque_after
                    report["torque_disable_read_attempts"] = torque_read_attempts
                    motor._torque_enabled = False  # noqa: SLF001 - hardware readback is authoritative
                except Exception as exc:
                    report["torque_disable_failure"] = f"{type(exc).__name__}: {exc}"
            motor.close()
        temporary_profile.unlink(missing_ok=True)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if failure is not None:
        raise SystemExit(f"CAL-046/047 真机路径失败：{failure}; evidence={args.output}")
    if report.get("torque_enabled_after") != [0] * len(MOTOR_NAMES):
        raise SystemExit(f"往返完成但未确认六轴卸力：{report.get('torque_enabled_after')}")
    action = "空载低速往返" if args.mode == "roundtrip" else "串口恢复到第1中转位"
    print(f"CAL-046/047 {action}通过；六轴已卸力；evidence={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
