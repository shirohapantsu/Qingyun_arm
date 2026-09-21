#!/usr/bin/env python3
"""人工摆位、程序只读地采集 CAL-021 单轴应用限位验证点。"""

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
    JOINT_REFERENCE,
    _capture_burst,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def effective_test_target_deg(side: str, nominal_limit_deg: float, margin_deg: float) -> float:
    """应用边界内缩 margin 后的实际验证目标。"""
    if side not in ("lower", "upper"):
        raise ValueError("side 必须为 lower/upper")
    if margin_deg < 0:
        raise ValueError("margin_deg 不得为负")
    return nominal_limit_deg + margin_deg if side == "lower" else nominal_limit_deg - margin_deg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--joint", required=True, choices=tuple(JOINT_NAMES))
    parser.add_argument("--side", required=True, choices=("lower", "upper"))
    parser.add_argument("--nominal-application-limit-deg", required=True, type=float)
    parser.add_argument("--margin-deg", required=True, type=float)
    parser.add_argument("--operator-observation", required=True)
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
    if profile.joints is None:
        raise SystemExit("CAL-018/019 未完成，拒绝把总线角当成应用限位角")
    joint_index = JOINT_NAMES.index(args.joint)
    configured_limit = float(profile.joints.application_limits_deg[
        joint_index, 0 if args.side == "lower" else 1
    ])
    if abs(configured_limit - args.nominal_application_limit_deg) > 1e-9:
        raise SystemExit(
            f"命令行名义边界 {args.nominal_application_limit_deg}° 与当前 profile "
            f"{configured_limit}° 不同，拒绝采集错目标"
        )
    configured_margin = float(profile.joints.margin_deg[joint_index])
    if abs(configured_margin - args.margin_deg) > 1e-9:
        raise SystemExit(
            f"命令行 margin {args.margin_deg}° 与当前 profile {configured_margin}° 不同"
        )
    target = effective_test_target_deg(args.side, configured_limit, configured_margin)

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
            raise SystemExit(f"六轴并非全部卸力，拒绝采样：{torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采样：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采样：{models}")
        pose = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        raw_all = pose["raw_position_representative"]
        q_bus_all = [
            reader.mapping.raw_to_bus_deg(name, raw_all[index])
            for index, name in enumerate(JOINT_NAMES)
        ]
        q_urdf_all = [
            reader.mapping.raw_to_degrees(name, raw_all[index])
            for index, name in enumerate(JOINT_NAMES)
        ]
        axis = reader.mapping.axis(args.joint)
    finally:
        reader.close()

    raw = raw_all[joint_index]
    if not axis.range_min <= raw <= axis.range_max:
        raise SystemExit(
            f"{args.joint}={raw} 越出官方行程 [{axis.range_min},{axis.range_max}]，拒绝保存"
        )
    actual = q_urdf_all[joint_index]
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-021",
        "status": "CALIBRATING",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "manual_application_limit_pose_hardware_read_only",
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
        "servo_id": JOINT_REFERENCE[args.joint]["servo_id"],
        "parent_link": JOINT_REFERENCE[args.joint]["parent_link"],
        "child_link": JOINT_REFERENCE[args.joint]["child_link"],
        "endpoint_side": args.side,
        "nominal_application_limit_deg": configured_limit,
        "configured_margin_deg": configured_margin,
        "effective_test_target_deg": target,
        "operator_observation": args.operator_observation,
        "official_raw_range": [axis.range_min, axis.range_max],
        "pose": pose,
        "raw_position_representative": raw,
        "q_bus_deg": q_bus_all[joint_index],
        "q_urdf_deg": actual,
        "target_error_deg": actual - target,
        "all_joint_q_urdf_deg": q_urdf_all,
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(
        f"已保存 {args.joint} {args.side}: q_urdf={actual:.6f}°, "
        f"有效目标={target:.6f}°，误差={actual-target:+.6f}° → {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
