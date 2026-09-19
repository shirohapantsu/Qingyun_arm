#!/usr/bin/env python3
"""通过稳定串口逐轴失能并读回确认六台舵机。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import MotorMapping, SerialTransport, StsProtocol  # noqa: E402
from scripts.move_gripper_raw import _read_raw, _write_raw  # noqa: E402
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY  # noqa: E402
from scripts.read_servo_eeprom import usb_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    profile = load_calibration_profile(args.profile.resolve(strict=True))
    stable_port = Path(profile.motor.port)
    if str(stable_port) != EXPECTED_STABLE_PORT:
        raise SystemExit(f"稳定串口不匹配：{stable_port}")
    resolved_port = stable_port.resolve(strict=True)
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"稳定串口未解析到ttyACM：{resolved_port}")
    if usb_identity(resolved_port) != EXPECTED_USB_IDENTITY:
        raise SystemExit("USB身份不匹配")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    transport = SerialTransport(str(stable_port), profile.motor.baudrate)
    protocol = StsProtocol(transport)
    try:
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
        for axis in mapping.axes:
            _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
            time.sleep(0.03)
        torque = []
        for axis in mapping.axes:
            last_error: Exception | None = None
            for _ in range(3):
                try:
                    torque.append(_read_raw(protocol, mapping, axis, "Torque_Enable"))
                    break
                except Exception as exc:  # retry only for cleanup verification
                    last_error = exc
                    time.sleep(0.05)
            else:
                assert last_error is not None
                raise last_error
    finally:
        transport.close()
    if torque != [0] * len(MOTOR_NAMES):
        raise SystemExit(f"失能读回失败：{torque}")
    print(f"六轴已失能，读回={torque}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
