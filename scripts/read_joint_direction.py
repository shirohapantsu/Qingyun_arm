#!/usr/bin/env python3
"""人工正反向摆位、程序只读地确认 CAL-018 单个关节方向。"""

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

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


EXPECTED_STABLE_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14114904-if00"
EXPECTED_USB_IDENTITY = {"vid": "1a86", "pid": "55d3", "serial": "5B14114904"}
MIN_DIRECTION_DELTA_COUNTS = 30

# 为了让人工摆位有清晰的几何判据，不直接使用 q=[0]×5 的
# “近竖直/近水平”外观。下列角度由本仓库绑定 URDF 正向运动学
# 直接求得：2→3 精确竖直，3→4 水平，4→5 同高。它是已知非零角
# 参考姿态，不是 follower_zero.webp，也不是 URDF q=[0]×5。
DIRECTION_REFERENCE_POSE = {
    "name": "urdf_orthogonal_joint_center_reference",
    "q_reference_deg": [0.0, -13.968, 16.175, -2.207, 0.0],
    "geometry_mm": {
        "servo_2_to_3": {"forward": 0.0, "vertical_up": 116.000},
        "servo_3_to_4": {"forward": 135.000, "vertical_up": 0.0},
        "servo_4_to_5": {
            "forward": 61.100,
            "lateral_positive_y": 18.100,
            "vertical_up": 0.0,
        },
    },
    "warning": (
        "Known nonzero URDF reference pose; do not relabel as q=[0]*5 or as "
        "the LeRobot follower_zero.webp pose."
    ),
}

JOINT_REFERENCE = {
    "shoulder_pan": {
        "servo_id": 1,
        "parent_link": "base_link",
        "child_link": "shoulder_link",
        "urdf_axis_joint_frame": [0.0, 0.0, 1.0],
        "urdf_axis_parent_frame_at_zero": [0.0, 0.0, -1.0],
        "positive_instruction": (
            "从基座正上方向下看，让肩部/整条手臂绕底座顺时针转动约10～20°"
        ),
        "negative_instruction": (
            "缓慢反向越过起始位置，让肩部/整条手臂从起始位置逆时针约10～20°"
        ),
    },
    "shoulder_lift": {
        "servo_id": 2,
        "parent_link": "shoulder_link",
        "child_link": "upper_arm_link",
        "urdf_axis_joint_frame": [0.0, 0.0, 1.0],
        "urdf_axis_parent_frame_at_zero": [0.0, 1.0, 0.0],
        "positive_instruction": (
            "从URDF正交轴心参考姿态开始（2→3竖直），保持1号底座角不动，"
            "使3号肘轴心向机械臂前方移动、同时下降约10～20°；"
            "这是绕parent-frame +Y轴的右手正向"
        ),
        "negative_instruction": (
            "从同一参考姿态缓慢反向，使3号肘轴心向后移动、"
            "同时下降约10～20°"
        ),
    },
    "elbow_flex": {
        "servo_id": 3,
        "parent_link": "upper_arm_link",
        "child_link": "lower_arm_link",
        "urdf_axis_joint_frame": [0.0, 0.0, 1.0],
        "urdf_axis_parent_frame_at_zero": [0.0, 0.0, 1.0],
        "positive_instruction": (
            "从URDF正交轴心参考姿态开始（3→4水平），保持2号上臂不动，使4号腕轴心"
            "向下偏转约10～20°"
        ),
        "negative_instruction": (
            "从同一参考姿态缓慢反向，使4号腕轴心向上偏转约10～20°"
        ),
    },
    "wrist_flex": {
        "servo_id": 4,
        "parent_link": "lower_arm_link",
        "child_link": "wrist_link",
        "urdf_axis_joint_frame": [0.0, 0.0, 1.0],
        "urdf_axis_parent_frame_at_zero": [0.0, 0.0, 1.0],
        "positive_instruction": (
            "从URDF正交轴心参考姿态开始（4→5同高），保持3号前臂不动，"
            "使5号腕转轴心和夹爪"
            "向下偏转约10～20°"
        ),
        "negative_instruction": (
            "从同一参考姿态缓慢反向，使5号腕转轴心和夹爪向上偏转约10～20°"
        ),
    },
    "wrist_roll": {
        "servo_id": 5,
        "parent_link": "wrist_link",
        "child_link": "gripper_link",
        "urdf_axis_joint_frame": [0.0, 0.0, 1.0],
        "urdf_axis_parent_frame_at_zero": [0.0, 1.0, 0.0],
        "positive_instruction": (
            "保持4号腕部俯仰不动，从4号腕关节沿前臂朝夹爪方向观察，"
            "将整个夹爪总成顺时针滚转约10～20°"
        ),
        "negative_instruction": (
            "从同一观察方向将夹爪总成逆时针转回，越过起始姿态后再约10～20°"
        ),
    },
}


def infer_joint_sign(
    baseline_raw: int,
    positive_raw: int,
    negative_raw: int,
    *,
    min_delta_counts: int = MIN_DIRECTION_DELTA_COUNTS,
) -> tuple[int, dict]:
    """由操作者确认的 URDF 正/负摆位推断 q_urdf = sign*q_bus + offset。"""
    pos_delta = int(positive_raw) - int(baseline_raw)
    neg_delta = int(negative_raw) - int(baseline_raw)
    if abs(pos_delta) < min_delta_counts or abs(neg_delta) < min_delta_counts:
        raise ValueError(
            f"正反摆位相对起点都须至少 {min_delta_counts} count，"
            f"实际 positive={pos_delta}, negative={neg_delta}"
        )
    if pos_delta * neg_delta >= 0:
        raise ValueError(
            f"正负参考没有落在起点两侧：positive={pos_delta}, negative={neg_delta}"
        )
    sign = 1 if pos_delta > 0 else -1
    return sign, {
        "positive_delta_counts": pos_delta,
        "negative_delta_counts": neg_delta,
        "opposite_sides_of_baseline": True,
        "minimum_delta_counts": int(min_delta_counts),
    }


def _capture_burst(reader: CalibrationReader, samples: int, period_s: float) -> dict:
    torque_before = reader.read_torque_enabled()
    if torque_before != [0] * len(MOTOR_NAMES):
        raise RuntimeError(f"采样前发现力矩已开启：{torque_before}")
    captured_at = datetime.now().astimezone().isoformat()
    rows: list[list[int]] = []
    for sample_index in range(samples):
        rows.append(reader.read_common_register("Present_Position"))
        if sample_index + 1 < samples:
            time.sleep(period_s)
    torque_after = reader.read_torque_enabled()
    if torque_after != [0] * len(MOTOR_NAMES):
        raise RuntimeError(f"采样后发现力矩已开启：{torque_after}")
    representatives = [
        int(round(statistics.median(row[index] for row in rows)))
        for index in range(len(MOTOR_NAMES))
    ]
    spans = [
        max(row[index] for row in rows) - min(row[index] for row in rows)
        for index in range(len(MOTOR_NAMES))
    ]
    return {
        "captured_at": captured_at,
        "raw_position_by_sample": rows,
        "raw_position_representative": representatives,
        "raw_position_static_span": spans,
        "torque_enabled_before": torque_before,
        "torque_enabled_after": torque_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--joint", required=True, choices=tuple(JOINT_REFERENCE))
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

    reference = JOINT_REFERENCE[args.joint]
    joint_index = JOINT_NAMES.index(args.joint)
    reader = CalibrationReader(profile)
    try:
        initial_torque = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        if initial_torque != [0] * len(MOTOR_NAMES):
            raise SystemExit(f"六轴并非全部卸力，拒绝人工摆位：{initial_torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采集：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采集：{models}")

        print(
            f"安全前置通过：正在测 ID {reference['servo_id']} {args.joint}，"
            f"连接 {reference['parent_link']} → {reference['child_link']}。",
            flush=True,
        )
        print("自动读取起始姿态；请托住其余连杆并保持不动。", flush=True)
        baseline = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        baseline_raw = baseline["raw_position_representative"][joint_index]
        print(f"起始位置：{baseline_raw} raw。", flush=True)

        confirmation = input(
            f"URDF 正向：{reference['positive_instruction']}；不要接近硬止挡。"
            "保持不动后输入 READY："
        ).strip()
        if confirmation != "READY":
            raise SystemExit("未收到精确的 READY，终止且不保存结果")
        positive = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        positive_raw = positive["raw_position_representative"][joint_index]
        print(f"正向位置：{positive_raw} raw。", flush=True)

        confirmation = input(
            f"URDF 反向：{reference['negative_instruction']}；不要接近硬止挡。"
            "保持不动后输入 READY："
        ).strip()
        if confirmation != "READY":
            raise SystemExit("未收到精确的 READY，终止且不保存结果")
        negative = _capture_burst(reader, args.samples_per_pose, args.sample_period_s)
        negative_raw = negative["raw_position_representative"][joint_index]
        print(f"反向位置：{negative_raw} raw。", flush=True)
    finally:
        reader.close()

    sign, direction_check = infer_joint_sign(baseline_raw, positive_raw, negative_raw)
    axis = reader.mapping.axis(args.joint)
    for label, raw in (("baseline", baseline_raw), ("positive", positive_raw),
                       ("negative", negative_raw)):
        if not axis.range_min <= raw <= axis.range_max:
            raise SystemExit(
                f"{label}={raw} 越出 {args.joint} 官方行程 "
                f"[{axis.range_min},{axis.range_max}]，拒绝保存"
            )
    count_to_deg = axis.count_to_deg
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-018",
        "status": "VERIFIED_FOR_DEMO",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "manual_direction_reference_hardware_read_only",
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
        "model_numbers": models,
        "joint": args.joint,
        "joint_index": joint_index,
        "reference": reference,
        "direction_reference_pose": DIRECTION_REFERENCE_POSE,
        "official_raw_range": [axis.range_min, axis.range_max],
        "count_to_deg": count_to_deg,
        "samples_per_pose": args.samples_per_pose,
        "poses": {
            "baseline": baseline,
            "urdf_positive": positive,
            "urdf_negative": negative,
        },
        "direction_check": direction_check,
        "result": {
            "joint_sign": sign,
            "formula": "q_urdf_deg = joint_sign * q_bus_deg + zero_offset_deg",
            "positive_delta_bus_deg": direction_check["positive_delta_counts"] * count_to_deg,
            "negative_delta_bus_deg": direction_check["negative_delta_counts"] * count_to_deg,
        },
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"CAL-018 单轴方向证据已保存：{args.output}")
    print(f"{args.joint}.sign={sign:+d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
