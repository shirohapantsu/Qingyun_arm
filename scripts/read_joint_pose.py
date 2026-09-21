#!/usr/bin/env python3
"""一次性只读保存一个 CAL-018 人工关节摆位，供非交互采集。"""

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
    DIRECTION_REFERENCE_POSE,
    JOINT_REFERENCE,
    _capture_burst,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--joint", required=True, choices=tuple(JOINT_REFERENCE))
    parser.add_argument(
        "--pose", required=True,
        choices=("baseline", "urdf_positive", "urdf_negative"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--samples-per-pose", type=int, default=10)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples_per_pose < 3:
        raise SystemExit("每个姿态至少读取3个静态样本")
    if not 0.0 <= args.sample_period_s <= 1.0:
        raise SystemExit("sample-period-s 必须在 [0,1] 秒")

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

    joint_index = JOINT_NAMES.index(args.joint)
    reference = JOINT_REFERENCE[args.joint]
    reader = CalibrationReader(profile)
    try:
        torque = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        if torque != [0] * len(MOTOR_NAMES):
            raise SystemExit(f"六轴并非全部卸力，拒绝采样：{torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采样：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采样：{models}")
        pose = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        axis = reader.mapping.axis(args.joint)
    finally:
        reader.close()

    raw = pose["raw_position_representative"][joint_index]
    if not axis.range_min <= raw <= axis.range_max:
        raise SystemExit(
            f"{args.joint}={raw} 越出官方行程 [{axis.range_min},{axis.range_max}]，拒绝保存"
        )
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-018",
        "source": "manual_pose_hardware_read_only",
        "captured_at": datetime.now().astimezone().isoformat(),
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
        "ids": ids,
        "model_numbers": models,
        "torque_enabled_by_motor": torque,
        "joint": args.joint,
        "joint_index": joint_index,
        "servo_id": reference["servo_id"],
        "parent_link": reference["parent_link"],
        "child_link": reference["child_link"],
        "direction_reference_pose": DIRECTION_REFERENCE_POSE,
        "pose_label": args.pose,
        "official_raw_range": [axis.range_min, axis.range_max],
        "pose": pose,
        "joint_raw_representative": raw,
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"已保存 {args.joint} {args.pose}：{raw} raw → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
