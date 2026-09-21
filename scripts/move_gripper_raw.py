#!/usr/bin/env python3
"""Safely move only servo 6 to a calibrated raw position for manual measurement."""

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
from scripts.read_joint_direction import (  # noqa: E402
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


TRANSACTION_TIMEOUT_S = 5.0
# 真机在低载静态定位时连续观察到约 4 raw 的内部到位死区。demo 手工测量按
# 端点跨度的 0.36%（5 / 1398）验收，并在证据中始终保留目标与实际反馈值。
POSITION_TOLERANCE_RAW = 5
DEFAULT_TIMEOUT_S = 8.0


def _read_raw(protocol: StsProtocol, mapping: MotorMapping, axis, register: str) -> int:
    # Level-1 devices should not acknowledge writes, but a delayed status frame must
    # never be mistaken for the following read response. Drain before every read.
    protocol.reset_input()
    address, length = mapping.register(axis, register)
    data = protocol.read_registers(
        axis.servo_id, address, length, time.monotonic() + TRANSACTION_TIMEOUT_S
    )
    return int.from_bytes(data, byteorder="little", signed=False)


def _write_raw(
    protocol: StsProtocol, mapping: MotorMapping, axis, register: str, value: int
) -> None:
    address, length = mapping.register(axis, register)
    if not 0 <= int(value) < 1 << (8 * length):
        raise ValueError(f"{register}={value} cannot fit in {length} bytes")
    protocol.write_registers(
        axis.servo_id,
        address,
        int(value).to_bytes(length, byteorder="little", signed=False),
        time.monotonic() + TRANSACTION_TIMEOUT_S,
    )


def _sync_goal(protocol: StsProtocol, mapping: MotorMapping, axis, raw: int) -> None:
    """Submit one Goal_Position through broadcast SYNC_WRITE, which has no ACK."""
    address, length = mapping.register(axis, "Goal_Position")
    if not 0 <= int(raw) < 1 << (8 * length):
        raise ValueError(f"Goal_Position={raw} cannot fit in {length} bytes")
    protocol.sync_write_targets(
        [axis.servo_id],
        address,
        length,
        [int(raw).to_bytes(length, byteorder="little", signed=False)],
        time.monotonic() + TRANSACTION_TIMEOUT_S,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--target-raw", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", type=float, required=True)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--velocity-raw", type=int, default=5)
    parser.add_argument("--acceleration-raw", type=int, default=5)
    parser.add_argument("--torque-limit-raw", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite evidence: {args.output}")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")

    profile_path = args.profile.resolve(strict=True)
    profile = load_calibration_profile(profile_path)
    stable_port = Path(profile.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"Unexpected stable serial path: {stable_port}")
    try:
        resolved_port = stable_port.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Stable serial path is currently absent: {stable_port}") from exc
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"Stable serial path did not resolve to ttyACM: {resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit(f"USB identity mismatch: {identity}")
    if profile.joints is None:
        raise SystemExit("CAL-018/019 are incomplete; refusing actuation")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    gripper = mapping.axis("gripper")
    safe_lower = min(profile.motor.gripper_closed_raw, profile.motor.gripper_open_raw)
    safe_upper = max(profile.motor.gripper_closed_raw, profile.motor.gripper_open_raw)
    if not safe_lower <= args.target_raw <= safe_upper:
        raise SystemExit(
            f"target raw {args.target_raw} outside calibrated safe endpoints "
            f"[{safe_lower},{safe_upper}]"
        )
    if not gripper.range_min <= args.target_raw <= gripper.range_max:
        raise SystemExit(
            f"target raw {args.target_raw} outside EEPROM range "
            f"[{gripper.range_min},{gripper.range_max}]"
        )

    report = {
        "schema_version": 1,
        "parameter_id": "CAL-026",
        "operation": "position_gripper_for_manual_gap_measurement",
        "started_at": datetime.now().astimezone().isoformat(),
        "success": False,
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "stable_port": str(stable_port),
        "resolved_port": str(resolved_port),
        "usb_identity": identity,
        "environment": {
            "power": args.power,
            "ambient_temperature_c": args.ambient_temperature_c,
            "base_mount": args.base_mount,
        },
        "target_raw": args.target_raw,
        "calibrated_safe_endpoints_raw": [safe_lower, safe_upper],
        "official_gripper_range_raw": [gripper.range_min, gripper.range_max],
        "temporary_sram_settings": {
            "Acceleration": args.acceleration_raw,
            "Goal_Time": 0,
            "Goal_Velocity": args.velocity_raw,
            "Torque_Limit": args.torque_limit_raw,
        },
        "eeprom_written": False,
        "joint_servo_motion_commands_sent": 0,
        "gripper_motion_commands_sent": 0,
        "samples": [],
    }

    transport = None
    protocol = None
    originals: dict[str, int] = {}
    torque_was_enabled = False
    final_torque: list[int] | None = None
    failure: BaseException | None = None
    try:
        transport = SerialTransport(str(stable_port), profile.motor.baudrate)
        protocol = StsProtocol(transport)

        response_levels = []
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError(
                    f"{axis.name} Response_Status_Level={level}; cannot safely write"
                )
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
            response_levels.append(level)
        report["response_status_level_by_motor"] = response_levels

        ids = [_read_raw(protocol, mapping, axis, "ID") for axis in mapping.axes]
        models = [_read_raw(protocol, mapping, axis, "Model_Number") for axis in mapping.axes]
        torques = [_read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes]
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise RuntimeError(f"servo ID mismatch: {ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"servo model mismatch: {models}")
        if torques != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"all six torques must initially be off: {torques}")
        report["ids"] = ids
        report["model_numbers"] = models
        report["torque_enabled_before"] = torques

        start_raw = _read_raw(protocol, mapping, gripper, "Present_Position")
        if not gripper.range_min <= start_raw <= gripper.range_max:
            raise RuntimeError(
                f"gripper present position {start_raw} outside EEPROM range "
                f"[{gripper.range_min},{gripper.range_max}]"
            )
        report["start_raw"] = start_raw

        for register in ("Acceleration", "Goal_Time", "Goal_Velocity", "Torque_Limit"):
            originals[register] = _read_raw(protocol, mapping, gripper, register)
        report["original_sram_settings"] = originals

        _write_raw(protocol, mapping, gripper, "Acceleration", args.acceleration_raw)
        _write_raw(protocol, mapping, gripper, "Goal_Time", 0)
        _write_raw(protocol, mapping, gripper, "Goal_Velocity", args.velocity_raw)
        _write_raw(protocol, mapping, gripper, "Torque_Limit", args.torque_limit_raw)
        # Some STS firmware rewrites Goal_Position to Present_Position when torque is
        # enabled. First seed the current position to prevent a jump, enable and verify
        # torque, then submit the actual target.
        _sync_goal(protocol, mapping, gripper, start_raw)
        _write_raw(protocol, mapping, gripper, "Torque_Enable", 1)
        torque_was_enabled = True
        time.sleep(0.05)
        torque_readback = _read_raw(protocol, mapping, gripper, "Torque_Enable")
        report["gripper_torque_enable_readback"] = torque_readback
        if torque_readback != 1:
            raise RuntimeError(f"gripper torque enable did not latch: {torque_readback}")
        _sync_goal(protocol, mapping, gripper, args.target_raw)
        report["gripper_motion_commands_sent"] = 1

        deadline = time.monotonic() + args.timeout_s
        settled = 0
        while time.monotonic() < deadline:
            position = _read_raw(protocol, mapping, gripper, "Present_Position")
            current_raw = _read_raw(protocol, mapping, gripper, "Present_Current")
            current_ma = float(mapping.raw_current_to_ma("gripper", current_raw))
            moving = _read_raw(protocol, mapping, gripper, "Moving")
            report["samples"].append(
                {
                    "time": datetime.now().astimezone().isoformat(),
                    "position_raw": position,
                    "current_register_raw": current_raw,
                    "current_ma": current_ma,
                    "moving": moving,
                }
            )
            if abs(current_ma) > 450.0:
                raise RuntimeError(f"gripper current exceeded demo hard limit: {current_ma} mA")
            if abs(position - args.target_raw) <= POSITION_TOLERANCE_RAW and moving == 0:
                settled += 1
                if settled >= 3:
                    break
            else:
                settled = 0
            time.sleep(0.05)
        else:
            raise RuntimeError(f"gripper did not settle at raw {args.target_raw} in time")

        report["final_raw_before_torque_off"] = report["samples"][-1]["position_raw"]
        report["success"] = True
    except BaseException as exc:
        failure = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            if torque_was_enabled:
                try:
                    _write_raw(protocol, mapping, gripper, "Torque_Enable", 0)
                except BaseException as exc:
                    report["torque_disable_error"] = f"{type(exc).__name__}: {exc}"
                    if failure is None:
                        failure = exc
                        report["success"] = False
            # Restore only temporary SRAM speed/torque settings, after torque is off.
            for register, value in originals.items():
                try:
                    _write_raw(protocol, mapping, gripper, register, value)
                except BaseException as exc:
                    report.setdefault("sram_restore_errors", []).append(
                        f"{register}: {type(exc).__name__}: {exc}"
                    )
                    if failure is None:
                        failure = exc
                        report["success"] = False
            try:
                final_torque = []
                final_torque_read_attempts = []
                for axis in mapping.axes:
                    last_read_error: Exception | None = None
                    for attempt in range(1, 4):
                        try:
                            final_torque.append(
                                _read_raw(protocol, mapping, axis, "Torque_Enable")
                            )
                            final_torque_read_attempts.append(attempt)
                            break
                        except Exception as exc:
                            last_read_error = exc
                            time.sleep(0.03)
                    else:
                        assert last_read_error is not None
                        raise last_read_error
                report["final_torque_read_attempts_by_motor"] = final_torque_read_attempts
                report["torque_enabled_after"] = final_torque
                if final_torque != [0] * len(MOTOR_NAMES):
                    raise RuntimeError(f"torque did not return to all-off state: {final_torque}")
            except BaseException as exc:
                report["final_torque_check_error"] = f"{type(exc).__name__}: {exc}"
                if failure is None:
                    failure = exc
                    report["success"] = False
        if transport is not None:
            transport.close()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failure is not None:
        raise SystemExit(str(failure))
    print(
        f"gripper reached raw={report['final_raw_before_torque_off']}; "
        f"all torques off; evidence={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
