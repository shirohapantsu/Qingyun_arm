#!/usr/bin/env python3
"""只读采集 SO-101 的 URDF 已知几何姿态，计算 CAL-019 五轴零位偏置。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.read_joint_direction import (  # noqa: E402
    DIRECTION_REFERENCE_POSE,
    EXPECTED_STABLE_PORT,
    EXPECTED_USB_IDENTITY,
    _capture_burst,
)
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


REFERENCE_DEFINITION = (
    "repository-bound so101_new_calib.urdf orthogonal joint-center reference; "
    "this is a known nonzero URDF pose, not q=[0]*5 and not the LeRobot "
    "follower_zero.webp manual-calibration pose"
)

# 由本仓库绑定的 so101_new_calib.urdf 在 q=[0,0,0,0,0] 时直接派生。
# 数值描述的是相邻舵机轴心在 base_link 中的相对位置，供实体摆位复核；
# 零位偏置仍由真机 Present_Position 读取，不能由这些尺寸替代。
ZERO_POSE_GEOMETRY_MM = {
    "joint_centers_in_base": {
        "servo_1": [38.8353, 0.0, 62.4],
        "servo_2": [69.2345, -18.2778, 116.6],
        "servo_3": [97.2344, -18.2778, 229.17],
        "servo_4": [232.1344, -18.2773, 234.37],
        "servo_5": [293.2344, -0.1773, 234.37],
    },
    "servo_2_to_3": {
        "forward": 28.0,
        "vertical_up": 112.57,
        "center_distance": 116.000021,
        "forward_from_vertical_deg": 13.969,
        "exactly_vertical_up_requires_q2_deg": -13.969,
    },
    "servo_3_to_4": {"forward": 134.9, "vertical_up": 5.2},
    "servo_4_to_5": {
        "forward": 61.1,
        "lateral": 18.1,
        "vertical_up": 0.0,
        "horizontal_distance": 63.724,
    },
}


def infer_zero_offsets_deg(
    signs: Sequence[int],
    q_bus_deg: Sequence[float],
    q_reference_deg: Sequence[float],
) -> list[float]:
    """由 q_ref = sign*q_bus + offset 反算 offset。"""
    if not (len(signs) == len(q_bus_deg) == len(q_reference_deg) == len(JOINT_NAMES)):
        raise ValueError(f"sign/q_bus/q_reference 均须为长度 {len(JOINT_NAMES)}")
    if any(int(value) not in (-1, 1) for value in signs):
        raise ValueError(f"sign 只能为 ±1，实际 {list(signs)}")
    return [
        float(reference) - int(sign) * float(bus)
        for sign, bus, reference in zip(signs, q_bus_deg, q_reference_deg)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--fixture-id", default="urdf_orthogonal_joint_center_reference")
    parser.add_argument("--samples-per-pose", type=int, default=10)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.repeat_index < 1:
        raise SystemExit("repeat-index 必须从 1 开始")
    if args.samples_per_pose < 3:
        raise SystemExit("每个姿态至少读取3个静态样本")
    if not 0.0 <= args.sample_period_s <= 1.0:
        raise SystemExit("sample-period-s 必须在 [0,1] 秒")

    profile_path = args.profile.resolve(strict=True)
    profile_json = json.loads(profile_path.read_text(encoding="utf-8"))
    signs = profile_json.get("joints", {}).get("sign")
    if not isinstance(signs, list) or len(signs) != len(JOINT_NAMES):
        raise SystemExit("CAL-018 joints.sign 尚未完整写入，拒绝计算 CAL-019")
    try:
        signs = [int(value) for value in signs]
        infer_zero_offsets_deg(signs, [0.0] * len(JOINT_NAMES), [0.0] * len(JOINT_NAMES))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"CAL-018 joints.sign 无效：{exc}") from exc
    if profile_json.get("joints", {}).get("zero_offset_deg") is not None:
        raise SystemExit("profile 已有 zero_offset_deg；拒绝把新读数伪装成首次零位标定")

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
            raise SystemExit(f"六轴并非全部卸力，拒绝采样：{torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采样：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采样：{models}")
        pose = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        raw_joint = pose["raw_position_representative"][: len(JOINT_NAMES)]
        q_bus_deg = [
            reader.mapping.raw_to_bus_deg(name, raw_joint[index])
            for index, name in enumerate(JOINT_NAMES)
        ]
        official_ranges = [
            [reader.mapping.axis(name).range_min, reader.mapping.axis(name).range_max]
            for name in JOINT_NAMES
        ]
    finally:
        reader.close()

    for name, raw, bounds in zip(JOINT_NAMES, raw_joint, official_ranges):
        if not bounds[0] <= raw <= bounds[1]:
            raise SystemExit(f"{name}={raw} 越出官方行程 {bounds}，拒绝保存")
    q_reference_deg = list(DIRECTION_REFERENCE_POSE["q_reference_deg"])
    offsets = infer_zero_offsets_deg(signs, q_bus_deg, q_reference_deg)
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-019",
        "status": "CALIBRATING",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "urdf_known_geometry_pose_hardware_read_only",
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
        "fixture_id": args.fixture_id,
        "repeat_index": args.repeat_index,
        "reference_definition": REFERENCE_DEFINITION,
        "reference_warning": (
            "Known nonzero URDF reference pose; do not use follower_zero.webp "
            "or relabel this pose as q=[0]*5."
        ),
        "orthogonal_joint_center_reference": DIRECTION_REFERENCE_POSE,
        "urdf_zero_pose_geometry_mm": ZERO_POSE_GEOMETRY_MM,
        "motor_names": list(MOTOR_NAMES),
        "ids": ids,
        "model_numbers": models,
        "torque_enabled_by_motor": torque,
        "samples_per_pose": args.samples_per_pose,
        "pose": pose,
        "joint_raw_representative": raw_joint,
        "official_raw_ranges": official_ranges,
        "q_bus_deg": q_bus_deg,
        "q_reference_deg": q_reference_deg,
        "joint_sign": signs,
        "zero_offset_deg_candidate": offsets,
        "formula": "zero_offset_deg = q_reference_deg - joint_sign * q_bus_deg",
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"CAL-019 URDF已知几何姿态第 {args.repeat_index} 轮已保存：{args.output}")
    print("zero_offset_deg_candidate=" + json.dumps(offsets), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
