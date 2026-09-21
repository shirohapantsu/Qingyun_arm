#!/usr/bin/env python3
"""六轴卸力时以等长 SRAM 原值回写测量 CAL-052 同步写时序。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import (  # noqa: E402
    MotorMapping,
    SerialTransport,
    StsProtocol,
)
from scripts.move_gripper_raw import _read_raw, _write_raw  # noqa: E402
from scripts.read_joint_direction import (  # noqa: E402
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--measurement-timeout-s", type=float, default=1.0)
    parser.add_argument("--safety-check-every", type=int, default=100)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--load-label", required=True)
    args = parser.parse_args()
    if args.ticks < 1000:
        raise SystemExit("CAL-052 至少需要1000个tick")
    if args.fps <= 0.0 or args.measurement_timeout_s <= 0.0:
        raise SystemExit("fps和measurement-timeout-s必须为正")
    if args.safety_check_every < 1:
        raise SystemExit("safety-check-every必须为正")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有证据：{args.output}")

    profile_path = args.profile.resolve(strict=True)
    profile = load_calibration_profile(profile_path)
    stable_port = Path(profile.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"稳定串口不是本机已确认地址：{stable_port}")
    resolved_port = stable_port.resolve(strict=True)
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"稳定地址未解析到ttyACM：{resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit(f"USB身份不匹配：{identity}")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    transport = SerialTransport(str(stable_port), profile.motor.baudrate)
    protocol = StsProtocol(transport)
    records: list[dict] = []
    safety_checks: list[dict] = []
    failure = None
    cleanup_torque_disable_writes = 0
    cleanup_torque_readback = None
    torque_after = None
    raw_position_after = None
    register_readback_after = None
    try:
        response_levels = []
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError(
                    f"{axis.name} Response_Status_Level={level}，无法确定写应答语义"
                )
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
            response_levels.append(level)
        ids = [_read_raw(protocol, mapping, axis, "ID") for axis in mapping.axes]
        models = [_read_raw(protocol, mapping, axis, "Model_Number") for axis in mapping.axes]
        torque_before = [
            _read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes
        ]
        raw_position_before = [
            _read_raw(protocol, mapping, axis, "Present_Position") for axis in mapping.axes
        ]
        if response_levels != [1] * len(MOTOR_NAMES):
            raise RuntimeError(f"当前测试只接受已确认的level-1设备：{response_levels}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)) or models != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"舵机身份不匹配：ids={ids}, models={models}")
        if torque_before != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"六轴并非全部卸力：{torque_before}")
        for axis, raw in zip(mapping.axes, raw_position_before):
            if not axis.range_min <= raw <= axis.range_max:
                raise RuntimeError(
                    f"{axis.name}当前位置{raw}越出EEPROM行程"
                    f"[{axis.range_min},{axis.range_max}]"
                )

        # Goal_Position 在本机实测会令六台舵机自动上力，不能用于卸力静态测试。
        # Goal_Time 同为2字节 SRAM，SYNC_WRITE 帧长与生产 Goal_Position 帧完全相同；
        # 逐轴读出后原值回写，不改变持久配置，也不提交任何位置目标。
        write_register = "Goal_Time"
        write_addr, write_len = mapping.common_register(write_register)
        if write_len != 2:
            raise RuntimeError(f"{write_register}不是预期的2字节寄存器：{write_len}")
        register_values_before = [
            _read_raw(protocol, mapping, axis, write_register) for axis in mapping.axes
        ]
        servo_ids = [axis.servo_id for axis in mapping.axes]
        write_values = [
            int(value).to_bytes(write_len, byteorder="little", signed=False)
            for value in register_values_before
        ]
        tick_ns = int(round(1e9 / args.fps))
        schedule_start_ns = time.monotonic_ns() + tick_ns
        for index in range(args.ticks):
            deadline_ns = schedule_start_ns + index * tick_ns
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.005))
            write_start_ns = time.monotonic_ns()
            protocol.sync_write_targets(
                servo_ids,
                write_addr,
                write_len,
                write_values,
                time.monotonic() + args.measurement_timeout_s,
            )
            write_end_ns = time.monotonic_ns()
            records.append({
                "record_type": "sample",
                "tick_index": index,
                "deadline_ns": deadline_ns,
                "write_start_ns": write_start_ns,
                "write_end_ns": write_end_ns,
                "tick_lateness_ns": max(0, write_start_ns - deadline_ns),
                "register": write_register,
                "register_values": register_values_before,
            })
            if index == 0 or (index + 1) % args.safety_check_every == 0:
                torque = [
                    _read_raw(protocol, mapping, axis, "Torque_Enable")
                    for axis in mapping.axes
                ]
                raw_position = [
                    _read_raw(protocol, mapping, axis, "Present_Position")
                    for axis in mapping.axes
                ]
                safety_checks.append({
                    "after_tick": index + 1,
                    "torque_enabled": torque,
                    "raw_position": raw_position,
                })
                if torque != [0] * len(MOTOR_NAMES):
                    raise RuntimeError(
                        f"第{index + 1}个tick后发现力矩已开启：{torque}"
                    )
        torque_after = [
            _read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes
        ]
        raw_position_after = [
            _read_raw(protocol, mapping, axis, "Present_Position") for axis in mapping.axes
        ]
        register_readback_after = [
            _read_raw(protocol, mapping, axis, write_register) for axis in mapping.axes
        ]
        if torque_after != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"采集后六轴不再全部卸力：{torque_after}")
        if register_readback_after != register_values_before:
            raise RuntimeError(
                f"最终{write_register}读回不一致："
                f"{register_readback_after} != {register_values_before}"
            )
    except BaseException as exc:
        failure = exc
        # 任一异常都按“力矩状态未知”处理，只写 SRAM Torque_Enable=0，并读回确认。
        try:
            for axis in mapping.axes:
                _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                cleanup_torque_disable_writes += 1
                time.sleep(0.03)
            cleanup_torque_readback = [
                _read_raw(protocol, mapping, axis, "Torque_Enable")
                for axis in mapping.axes
            ]
        except Exception as cleanup_exc:
            cleanup_torque_readback = (
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
    finally:
        transport.close()

    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        "parameter_id": "CAL-052",
        "captured_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if failure is None else "FAIL",
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
        "requested_ticks": args.ticks,
        "completed_ticks": len(records),
        "fps": args.fps,
        "measurement_timeout_s": args.measurement_timeout_s,
        "response_status_level_by_motor": (
            response_levels if "response_levels" in locals() else None
        ),
        "torque_enabled_before": torque_before if "torque_before" in locals() else None,
        "torque_enabled_after": torque_after,
        "raw_position_before": (
            raw_position_before if "raw_position_before" in locals() else None
        ),
        "raw_position_after": raw_position_after,
        "no_op_write_register": write_register if "write_register" in locals() else None,
        "register_values_before": (
            register_values_before if "register_values_before" in locals() else None
        ),
        "register_readback_after": register_readback_after,
        "periodic_safety_checks": safety_checks,
        "sync_write_instructions_sent": len(records),
        "goal_position_write_instructions_sent": 0,
        "torque_enable_write_instructions_sent": cleanup_torque_disable_writes,
        "cleanup_torque_readback": cleanup_torque_readback,
        "position_target_commands_sent": 0,
        "motion_commanded": False,
        "eeprom_written": False,
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failure is not None:
        raise SystemExit(f"CAL-052采集失败：{failure}; completed={len(records)}")
    durations_ms = [
        (row["write_end_ns"] - row["write_start_ns"]) / 1e6 for row in records
    ]
    print(json.dumps({
        "ticks": len(records),
        "write_min_ms": min(durations_ms),
        "write_max_ms": max(durations_ms),
        "torque_enabled_after": torque_after,
        "raw_position_stable": raw_position_before == raw_position_after,
        "register_readback_matches": register_readback_after == register_values_before,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
