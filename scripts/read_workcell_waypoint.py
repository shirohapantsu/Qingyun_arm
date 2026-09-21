#!/usr/bin/env python3
"""只读采集 CAL-046/047 的关节空间待机位或安全中转位。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.read_joint_direction import (  # noqa: E402
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
    _capture_burst,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("home", "waypoint"))
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    args = parser.parse_args()

    if args.samples < 3:
        raise SystemExit("至少读取3个静态样本")
    if not 0.0 <= args.sample_period_s <= 1.0:
        raise SystemExit("sample-period-s 必须在 [0,1] 秒")
    if args.role == "home" and args.order != 0:
        raise SystemExit("home 的 order 必须为0")
    if args.role == "waypoint" and args.order < 1:
        raise SystemExit("waypoint 的 order 必须至少为1")

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
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit(f"USB 身份不匹配，拒绝访问：{identity}")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有证据：{args.output}")

    reader = CalibrationReader(profile)
    try:
        torque = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        if torque != [0] * len(MOTOR_NAMES):
            raise SystemExit(f"六轴并非全部卸力，拒绝示教采样：{torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采样：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采样：{models}")
        pose = _capture_burst(reader, args.samples, args.sample_period_s)
        raw_joint = pose["raw_position_representative"][: len(JOINT_NAMES)]
        q_urdf_deg = [
            float(reader.mapping.raw_to_degrees(index, raw))
            for index, raw in enumerate(raw_joint)
        ]
        lower, upper = reader.mapping.clamp_free_limits_deg()
    finally:
        reader.close()

    violations = [
        f"{name}={q:.3f}deg 不在 [{lo:.3f},{hi:.3f}]"
        for name, q, lo, hi in zip(JOINT_NAMES, q_urdf_deg, lower, upper)
        if q < lo - 1e-6 or q > hi + 1e-6
    ]
    if violations:
        raise SystemExit("示教姿态越出有效限位，拒绝保存：" + "; ".join(violations))

    parameter_id = "CAL-046" if args.role == "home" else "CAL-047"
    payload = {
        "schema_version": 1,
        "parameter_id": parameter_id,
        "status": "CALIBRATING",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "manual_joint_space_waypoint_hardware_read_only",
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
        "operator_confirmation": args.operator_confirmation,
        "motor_names": list(MOTOR_NAMES),
        "ids": ids,
        "model_numbers": models,
        "torque_enabled_by_motor": torque,
        "samples": args.samples,
        "pose": pose,
        "waypoint_role": args.role,
        "order": args.order,
        "q_urdf_deg": q_urdf_deg,
        "effective_limits_deg": [
            [float(lo), float(hi)] for lo, hi in zip(lower, upper)
        ],
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"q_urdf_deg": q_urdf_deg,
                      "raw_static_span": pose["raw_position_static_span"]},
                     ensure_ascii=False))
    print(f"已保存 {parameter_id} {args.role} 只读示教样本：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
