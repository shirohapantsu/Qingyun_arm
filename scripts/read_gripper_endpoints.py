#!/usr/bin/env python3
"""人工摆位、程序只读地采集 CAL-017 夹爪安全开闭端点。"""

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
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


EXPECTED_STABLE_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14114904-if00"
EXPECTED_USB_IDENTITY = {"vid": "1a86", "pid": "55d3", "serial": "5B14114904"}
ENDPOINT_SEQUENCE = ("closed", "open") * 5


def summarize_placement(
    endpoint: str,
    repeat: int,
    raw_positions: list[list[int]],
    *,
    gripper_index: int,
    range_min: int,
    range_max: int,
) -> dict:
    """校验一次静态 burst，并给出不受单点抖动影响的代表值。"""
    if endpoint not in {"closed", "open"}:
        raise ValueError(f"未知端点：{endpoint}")
    if not raw_positions:
        raise ValueError("端点 burst 不能为空")
    expected_width = len(MOTOR_NAMES)
    if any(len(row) != expected_width for row in raw_positions):
        raise ValueError(f"每行 raw_position 必须恰有 {expected_width} 项")
    gripper_raw = [int(row[gripper_index]) for row in raw_positions]
    outside = [v for v in gripper_raw if not range_min <= v <= range_max]
    if outside:
        raise ValueError(
            f"夹爪读数越出官方行程 [{range_min}, {range_max}]：{outside}"
        )
    representative = int(round(statistics.median(gripper_raw)))
    return {
        "endpoint": endpoint,
        "repeat": int(repeat),
        "raw_position_by_sample": raw_positions,
        "gripper_raw_by_sample": gripper_raw,
        "gripper_raw_min": min(gripper_raw),
        "gripper_raw_max": max(gripper_raw),
        "gripper_raw_span": max(gripper_raw) - min(gripper_raw),
        "gripper_raw_representative": representative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--samples-per-placement", type=int, default=10)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples_per_placement < 3:
        raise SystemExit("每次摆位至少读取 3 个静态样本")
    if not 0.0 <= args.sample_period_s <= 1.0:
        raise SystemExit("sample-period-s 必须在 [0, 1] 秒")

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
        gripper_index = MOTOR_NAMES.index("gripper")
        gripper_axis = reader.mapping.axis("gripper")
        initial_torque = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        if initial_torque != [0] * len(MOTOR_NAMES):
            raise SystemExit(f"六轴并非全部卸力，拒绝人工摆位：{initial_torque}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)):
            raise SystemExit(f"舵机 ID 顺序不匹配，拒绝采集：{ids}")
        if models != [777] * len(MOTOR_NAMES):
            raise SystemExit(f"舵机型号不是六台 STS3215，拒绝采集：{models}")

        print("安全前置通过：USB 身份正确，ID/型号正确，六轴 Torque_Enable 均为 0。")
        print("全过程只发 READ；请始终托住手臂，只缓慢手动移动 ID 6 夹爪。", flush=True)
        placements: list[dict] = []
        for sequence_index, endpoint in enumerate(ENDPOINT_SEQUENCE):
            repeat = sequence_index // 2 + 1
            if endpoint == "closed":
                instruction = (
                    "移走物体，从张开侧缓慢合拢，至两侧软垫刚好完整接触；"
                    "不要继续压缩软垫或顶住硬止挡"
                )
            else:
                instruction = (
                    "从闭合侧缓慢张开至最大可用位置；若感觉硬止挡，"
                    "立即回退到没有持续预载的位置"
                )
            confirmation = input(
                f"CAL-017 第 {repeat}/5 轮 {endpoint.upper()}：{instruction}。"
                "保持不动后输入 READY："
            ).strip()
            if confirmation != "READY":
                raise SystemExit("未收到精确的 READY，终止且不保存结果")
            torque_before = reader.read_torque_enabled()
            if torque_before != [0] * len(MOTOR_NAMES):
                raise SystemExit(f"采样前发现力矩已开启，立即终止：{torque_before}")
            captured_at = datetime.now().astimezone().isoformat()
            rows: list[list[int]] = []
            for sample_index in range(args.samples_per_placement):
                rows.append(reader.read_common_register("Present_Position"))
                if sample_index + 1 < args.samples_per_placement:
                    time.sleep(args.sample_period_s)
            torque_after = reader.read_torque_enabled()
            if torque_after != [0] * len(MOTOR_NAMES):
                raise SystemExit(f"采样后发现力矩已开启，立即终止：{torque_after}")
            placement = summarize_placement(
                endpoint,
                repeat,
                rows,
                gripper_index=gripper_index,
                range_min=gripper_axis.range_min,
                range_max=gripper_axis.range_max,
            )
            placement.update({
                "captured_at": captured_at,
                "torque_enabled_before": torque_before,
                "torque_enabled_after": torque_after,
            })
            placements.append(placement)
            print(
                f"已读取 {endpoint} 第 {repeat} 轮："
                f"代表值={placement['gripper_raw_representative']}，"
                f"静态范围={placement['gripper_raw_min']}..{placement['gripper_raw_max']}",
                flush=True,
            )
    finally:
        reader.close()

    closed = [p["gripper_raw_representative"] for p in placements
              if p["endpoint"] == "closed"]
    opened = [p["gripper_raw_representative"] for p in placements
              if p["endpoint"] == "open"]
    closed_candidate = int(statistics.median(closed))
    open_candidate = int(statistics.median(opened))
    if closed_candidate == open_candidate:
        raise SystemExit("闭合与张开候选端点相同，拒绝保存")
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-017",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "manual_positioning_hardware_read_only",
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
        "official_gripper_range": [gripper_axis.range_min, gripper_axis.range_max],
        "sequence": list(ENDPOINT_SEQUENCE),
        "samples_per_placement": args.samples_per_placement,
        "placements": placements,
        "closed_representatives": closed,
        "open_representatives": opened,
        "closed_repeat_span": max(closed) - min(closed),
        "open_repeat_span": max(opened) - min(opened),
        "candidate": {
            "gripper_closed_raw": closed_candidate,
            "gripper_open_raw": open_candidate,
            "raw_span": open_candidate - closed_candidate,
        },
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"CAL-017 端点只读证据已保存：{args.output}")
    print(json.dumps(payload["candidate"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
