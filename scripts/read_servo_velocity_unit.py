#!/usr/bin/env python3
"""只读六台 STS3215 的速度单位因子，为 CAL-016 保存原始证据。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


EXPECTED_STABLE_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14114904-if00"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    args = parser.parse_args()

    profile_path = args.profile.resolve(strict=True)
    profile = load_calibration_profile(profile_path)
    stable_port = Path(profile.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"稳定串口不是本机已确认地址，拒绝访问：{stable_port}")
    try:
        resolved_port = stable_port.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"稳定串口当前不存在：{stable_port}") from exc
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"稳定地址未解析到 ttyACM，拒绝访问：{resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != {"vid": "1a86", "pid": "55d3", "serial": "5B14114904"}:
        raise SystemExit(f"USB 身份不匹配，拒绝访问：{identity}")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有证据：{args.output}")

    reader = CalibrationReader(profile)
    try:
        torque = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        model_numbers = reader.read_common_register("Model_Number")
        firmware_major = reader.read_common_register("Firmware_Major_Version")
        firmware_minor = reader.read_common_register("Firmware_Minor_Version")
        angular_resolution = reader.read_common_register("Angular_Resolution")
        velocity_unit_factor = reader.read_common_register("Velocity_Unit_factor")
    finally:
        reader.close()

    expected_ids = list(range(1, len(MOTOR_NAMES) + 1))
    expected_firmware = [str(v) for v in profile.motor.firmware]
    firmware = [f"{major}.{minor}" for major, minor in zip(
        firmware_major, firmware_minor, strict=True)]
    checks = {
        "ids_match": ids == expected_ids,
        "models_match_sts3215": model_numbers == [777] * len(MOTOR_NAMES),
        "firmware_matches_profile": firmware == expected_firmware,
        "torque_disabled": torque == [0] * len(MOTOR_NAMES),
    }
    if not all(checks.values()):
        raise SystemExit(f"身份或安全前置不匹配，拒绝保存：{checks}")

    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-016",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "hardware_read_only",
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
        "motor_names": list(MOTOR_NAMES),
        "ids": ids,
        "model_numbers": model_numbers,
        "firmware": firmware,
        "torque_enabled_by_motor": torque,
        "angular_resolution_raw": angular_resolution,
        "velocity_unit_factor_raw": velocity_unit_factor,
        "checks": checks,
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"CAL-016 速度单位只读证据已保存：{args.output}")
    print(f"Velocity_Unit_factor={velocity_unit_factor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
