#!/usr/bin/env python3
"""CAL-082：按运行期小步逻辑采集空夹爪张开速度。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import MotorMapping, SerialTransport, StsProtocol  # noqa: E402
from scripts.move_gripper_raw import _read_raw, _sync_goal, _write_raw  # noqa: E402
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


TRANSACTION_TIMEOUT_S = 5.0
POSITION_TOLERANCE_RAW = 5
PREPOSITION_TIMEOUT_S = 10.0
SETTLE_TIMEOUT_S = 3.0


def _pct_to_raw(closed_raw: int, open_raw: int, pct: float) -> int:
    return int(round(closed_raw + (open_raw - closed_raw) * pct / 100.0))


def _raw_to_pct(closed_raw: int, open_raw: int, raw: int) -> float:
    return (float(raw) - closed_raw) * 100.0 / (open_raw - closed_raw)


def _sample(protocol: StsProtocol, mapping: MotorMapping, gripper) -> dict:
    position = _read_raw(protocol, mapping, gripper, "Present_Position")
    velocity = _read_raw(protocol, mapping, gripper, "Present_Velocity")
    current = _read_raw(protocol, mapping, gripper, "Present_Current")
    moving = _read_raw(protocol, mapping, gripper, "Moving")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "position_raw": position,
        "gripper_pct": _raw_to_pct(
            mapping.gripper_closed_raw, mapping.gripper_open_raw, position
        ),
        "velocity_register_raw": velocity,
        "velocity_pct_s": mapping.raw_velocity_to_deg_s("gripper", velocity),
        "current_register_raw": current,
        "current_ma": mapping.raw_current_to_ma("gripper", current),
        "moving": moving,
    }


def _linear_slope(rows: list[dict]) -> float:
    t0 = rows[0]["monotonic_ns"]
    x = [(row["monotonic_ns"] - t0) / 1e9 for row in rows]
    y = [float(row["gripper_pct"]) for row in rows]
    x_bar = statistics.fmean(x)
    y_bar = statistics.fmean(y)
    denominator = sum((v - x_bar) ** 2 for v in x)
    if denominator <= 0.0:
        raise RuntimeError("采样时间跨度为零，无法拟合速度")
    return sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y)) / denominator


def main(*, parameter_id: str = "CAL-082", opening: bool = True) -> int:
    if (parameter_id, opening) not in (("CAL-082", True), ("CAL-083", False)):
        raise ValueError("只允许 CAL-082 张开或 CAL-083 空爪闭合采集")
    parser = argparse.ArgumentParser(description=f"{parameter_id}：空夹爪小步速度采集")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--start-pct", type=float, default=25.0 if opening else 75.0)
    parser.add_argument("--end-pct", type=float, default=75.0 if opening else 25.0)
    parser.add_argument("--candidate-speed-pct-s", type=float, default=25.0 if opening else 15.0)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有证据：{args.output}")
    if args.operator_confirmation != f"{parameter_id}就绪":
        raise SystemExit(f"缺少精确的 {parameter_id}就绪 操作者确认")
    if not (0.0 <= args.start_pct <= 100.0 and 0.0 <= args.end_pct <= 100.0):
        raise SystemExit("开度必须在 [0,100] 内")
    if not (args.start_pct < args.end_pct if opening else args.start_pct > args.end_pct):
        raise SystemExit("起止开度与已确认的采集方向不一致")
    if args.candidate_speed_pct_s <= 0.0:
        raise SystemExit("candidate-speed-pct-s 必须为正")

    profile_path = args.profile.resolve(strict=True)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = load_calibration_profile(profile_path)
    stable_port = Path(profile.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"稳定串口不匹配：{stable_port}")
    resolved_port = stable_port.resolve(strict=True)
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"稳定串口未解析到 ttyACM：{resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit(f"USB 身份不匹配：{identity}")
    if profile.joints is None:
        raise SystemExit("关节标定未完成，拒绝访问")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    gripper = mapping.axis("gripper")
    closed_raw = int(profile.motor.gripper_closed_raw)
    open_raw = int(profile.motor.gripper_open_raw)
    start_raw = _pct_to_raw(closed_raw, open_raw, args.start_pct)
    end_raw = _pct_to_raw(closed_raw, open_raw, args.end_pct)
    for raw in (start_raw, end_raw):
        if not gripper.range_min <= raw <= gripper.range_max:
            raise SystemExit(f"目标 raw={raw} 越出 EEPROM 范围")

    fps = float(raw_profile["timing"]["fps"])
    max_tick_lateness_s = float(raw_profile["timing"]["max_tick_lateness_s"])
    hard_current_ma = float(raw_profile["gripper"]["hard_current_ma"])
    tick_s = 1.0 / fps
    steps = int(round(abs(args.end_pct - args.start_pct) / args.candidate_speed_pct_s * fps))
    if steps < 2:
        raise SystemExit("试验步数不足")
    commanded_speed = abs(args.end_pct - args.start_pct) / (steps * tick_s)
    direction = 1 if opening else -1
    report: dict = {
        "schema_version": 1,
        "parameter_id": parameter_id,
        "test": "empty_gripper_runtime_style_incremental_open" if opening else "empty_gripper_runtime_style_incremental_close",
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
            "load": "empty_gripper",
        },
        "operator_confirmation": args.operator_confirmation,
        "start_pct": args.start_pct,
        "end_pct": args.end_pct,
        "start_raw": start_raw,
        "end_raw": end_raw,
        "fps": fps,
        "tick_s": tick_s,
        "steps": steps,
        "candidate_speed_pct_s": args.candidate_speed_pct_s,
        "commanded_speed_pct_s": commanded_speed,
        "commanded_signed_rate_pct_s": direction * commanded_speed,
        "eeprom_written": False,
        "joint_servo_motion_commands_sent": 0,
        "gripper_motion_commands_sent": 0,
        "samples": [],
    }

    transport = None
    protocol = None
    torque_may_be_enabled = False
    originals: dict[str, int] = {}
    failure: BaseException | None = None
    try:
        transport = SerialTransport(str(stable_port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        levels = []
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError(f"{axis.name} Response_Status_Level={level}")
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
            levels.append(level)
        ids = [_read_raw(protocol, mapping, axis, "ID") for axis in mapping.axes]
        models = [_read_raw(protocol, mapping, axis, "Model_Number") for axis in mapping.axes]
        torques = [_read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes]
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise RuntimeError(f"ID 不匹配：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"型号不匹配：{models}")
        if torques != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"试验前六轴必须全部卸力：{torques}")
        report["readonly_preflight"] = {
            "response_status_level": levels,
            "ids": ids,
            "model_numbers": models,
            "torque_enabled": torques,
        }

        initial = _sample(protocol, mapping, gripper)
        report["initial_sample"] = initial
        if not min(closed_raw, open_raw) - POSITION_TOLERANCE_RAW <= initial["position_raw"] <= max(closed_raw, open_raw) + POSITION_TOLERANCE_RAW:
            raise RuntimeError(f"夹爪起始位置越出标定端点：{initial['position_raw']}")
        for register in ("Acceleration", "Goal_Time", "Goal_Velocity", "Torque_Limit"):
            originals[register] = _read_raw(protocol, mapping, gripper, register)
        report["original_sram_settings"] = originals

        # 定位阶段沿用此前 CAL-026 已验证的温和 SRAM 设置；这些值不写 EEPROM，
        # 进入正式速度段前会恢复运行期原值，避免把定位速度混入 CAL-082 结论。
        _write_raw(protocol, mapping, gripper, "Acceleration", 5)
        _write_raw(protocol, mapping, gripper, "Goal_Time", 0)
        _write_raw(protocol, mapping, gripper, "Goal_Velocity", 10)
        _write_raw(protocol, mapping, gripper, "Torque_Limit", 600)

        # 本机 CAL-052 证明：Torque_Enable=0 时写 Goal_Position 也可能隐式使能。
        # 因此在第一条目标帧之前就把状态标成“可能已使能”，finally 必须失能六轴。
        torque_may_be_enabled = True
        _sync_goal(protocol, mapping, gripper, initial["position_raw"])
        _write_raw(protocol, mapping, gripper, "Torque_Enable", 1)
        time.sleep(0.05)
        _sync_goal(protocol, mapping, gripper, start_raw)
        report["gripper_motion_commands_sent"] += 1
        deadline = time.monotonic() + PREPOSITION_TIMEOUT_S
        while True:
            row = _sample(protocol, mapping, gripper)
            report.setdefault("preposition_samples", []).append(row)
            if abs(row["current_ma"]) > hard_current_ma:
                raise RuntimeError(f"定位阶段夹爪电流超限：{row['current_ma']}mA")
            if abs(row["position_raw"] - start_raw) <= POSITION_TOLERANCE_RAW and row["moving"] == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"夹爪未在时限内到达 {args.start_pct}% 起点")
            time.sleep(0.05)

        for register, value in originals.items():
            _write_raw(protocol, mapping, gripper, register, value)
            # 等级1 WRITE无ACK；逐项留出处理时间并读回，不以提交成功代替生效。
            time.sleep(0.03)
            if _read_raw(protocol, mapping, gripper, register) != value:
                raise RuntimeError(f"正式速度段前 {register} 恢复读回不一致")
        time.sleep(0.2)
        report["test_sram_settings"] = {
            register: _read_raw(protocol, mapping, gripper, register)
            for register in originals
        }
        if report["test_sram_settings"] != originals:
            raise RuntimeError("正式速度段前 SRAM 未恢复到运行期原值")

        base = time.monotonic()
        for index in range(1, steps + 1):
            due = base + (index - 1) * tick_s
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            lateness = time.monotonic() - due
            if lateness > max_tick_lateness_s:
                raise RuntimeError(f"第{index}步迟到 {lateness:.6f}s，超过时序上限")
            command_pct = args.start_pct + (args.end_pct - args.start_pct) * index / steps
            command_raw = _pct_to_raw(closed_raw, open_raw, command_pct)
            _sync_goal(protocol, mapping, gripper, command_raw)
            report["gripper_motion_commands_sent"] += 1
            sample_due = base + index * tick_s
            remaining = sample_due - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            row = _sample(protocol, mapping, gripper)
            row.update({
                "index": index,
                "command_pct": command_pct,
                "command_raw": command_raw,
                "command_tracking_error_pct": command_pct - row["gripper_pct"],
            })
            report["samples"].append(row)
            if abs(row["current_ma"]) > hard_current_ma:
                raise RuntimeError(f"速度段夹爪电流超限：{row['current_ma']}mA")

        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        while True:
            row = _sample(protocol, mapping, gripper)
            report.setdefault("settle_samples", []).append(row)
            if abs(row["current_ma"]) > hard_current_ma:
                raise RuntimeError(f"到位阶段夹爪电流超限：{row['current_ma']}mA")
            if abs(row["position_raw"] - end_raw) <= POSITION_TOLERANCE_RAW and row["moving"] == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"夹爪未在时限内到达 {args.end_pct}% 终点")
            time.sleep(0.05)

        rows = report["samples"]
        measured_speed = _linear_slope(rows)
        abs_currents = [abs(float(row["current_ma"])) for row in rows]
        tracking_errors = [abs(float(row["command_tracking_error_pct"])) for row in rows]
        final_pct = float(report["settle_samples"][-1]["gripper_pct"])
        report["result"] = {
            "measured_linear_speed_pct_s": measured_speed,
            "measured_directional_rate_pct_s": direction * measured_speed,
            "speed_ratio_to_command": direction * measured_speed / commanded_speed,
            "peak_current_ma": max(abs_currents),
            "median_current_ma": statistics.median(abs_currents),
            "max_command_tracking_error_pct": max(tracking_errors),
            "final_settled_pct": final_pct,
            "final_error_pct": final_pct - args.end_pct,
        }
        if not 0.85 * commanded_speed <= direction * measured_speed <= 1.15 * commanded_speed:
            raise RuntimeError(f"实测线性速度 {measured_speed:.3f}%/s 未跟随候选值")
        if abs(final_pct - args.end_pct) > 100.0 * POSITION_TOLERANCE_RAW / abs(open_raw - closed_raw):
            raise RuntimeError(f"终点误差过大：{final_pct - args.end_pct:.3f}%")
        report["status"] = "PASS_PENDING_OPERATOR_OBSERVATION"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            if torque_may_be_enabled:
                for axis in mapping.axes:
                    try:
                        _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                        time.sleep(0.03)
                    except BaseException as exc:
                        report.setdefault("torque_disable_errors", []).append(
                            f"{axis.name}: {type(exc).__name__}: {exc}"
                        )
            for register, value in originals.items():
                try:
                    _write_raw(protocol, mapping, gripper, register, value)
                    time.sleep(0.03)
                except BaseException as exc:
                    report.setdefault("sram_restore_errors", []).append(
                        f"{register}: {type(exc).__name__}: {exc}"
                    )
            try:
                final_torques = []
                for axis in mapping.axes:
                    for attempt in range(3):
                        try:
                            final_torques.append(_read_raw(protocol, mapping, axis, "Torque_Enable"))
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(0.05)
                report["torque_enabled_after"] = final_torques
                report["sram_settings_after_cleanup"] = {
                    register: _read_raw(protocol, mapping, gripper, register)
                    for register in originals
                }
                if report["sram_settings_after_cleanup"] != originals:
                    report["sram_restore_errors"] = ["清理后SRAM读回不一致"]
            except BaseException as exc:
                report["torque_final_check_error"] = f"{type(exc).__name__}: {exc}"
            transport.close()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if report.get("torque_enabled_after") != [0] * len(MOTOR_NAMES):
        raise SystemExit(f"{parameter_id} 后未确认六轴卸力；evidence={args.output}")
    if report.get("sram_restore_errors"):
        raise SystemExit(f"{parameter_id} 后SRAM恢复未确认；evidence={args.output}")
    if failure is not None:
        raise SystemExit(f"{parameter_id} 失败：{failure}; evidence={args.output}")
    result = report["result"]
    print(
        f"{parameter_id} 自动采集通过并已卸力；"
        f"实测={result['measured_linear_speed_pct_s']:.3f}%/s，"
        f"峰值电流={result['peak_current_ma']:.1f}mA；evidence={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
