#!/usr/bin/env python3
"""只读测量 CAL-051/053 的1000次整臂反馈块读取时序。"""

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
    CalibrationReader,
    StsProtocol,
)
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
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--load-label", required=True)
    args = parser.parse_args()
    if args.ticks < 1000:
        raise SystemExit("CAL-051 至少需要1000个tick")
    if args.fps <= 0.0 or args.measurement_timeout_s <= 0.0:
        raise SystemExit("fps和measurement-timeout-s必须为正")
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

    reader = CalibrationReader(profile, read_timeout_s=args.measurement_timeout_s)
    records: list[dict] = []
    failure = None
    try:
        torque_before = reader.read_torque_enabled()
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        raw_position_before = reader.read_common_register("Present_Position")
        if torque_before != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"六轴并非全部卸力：{torque_before}")
        if ids != list(range(1, len(MOTOR_NAMES) + 1)) or models != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"舵机身份不匹配：ids={ids}, models={models}")

        servo_ids = [axis.servo_id for axis in reader.mapping.axes]
        tick_ns = int(round(1e9 / args.fps))
        schedule_start_ns = time.monotonic_ns() + tick_ns
        for index in range(args.ticks):
            deadline_ns = schedule_start_ns + index * tick_ns
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.005))
            read_start_ns = time.monotonic_ns()
            # 直接调用CalibrationReader已推导好的反馈块布局，避免夹爪百分比端点
            # 换算影响通信计时；总线事务与生产get_feedback完全相同。
            block = reader._protocol.bulk_read(  # noqa: SLF001 - timing instrument
                servo_ids,
                reader._fb_addr,  # noqa: SLF001
                reader._fb_len,  # noqa: SLF001
                time.monotonic() + args.measurement_timeout_s,
            )
            read_end_ns = time.monotonic_ns()
            if set(block) != set(servo_ids):
                raise RuntimeError(f"第{index}个tick反馈不完整：{sorted(block)}")
            records.append({
                "record_type": "sample",
                "tick_index": index,
                "deadline_ns": deadline_ns,
                "read_start_ns": read_start_ns,
                "read_end_ns": read_end_ns,
                "sample_span_ns": read_end_ns - read_start_ns,
                "tick_lateness_ns": max(0, read_start_ns - deadline_ns),
                "feedback_raw_hex_by_id": {
                    str(sid): block[sid].hex() for sid in servo_ids
                },
            })
        torque_after = reader.read_torque_enabled()
        raw_position_after = reader.read_common_register("Present_Position")
        if torque_after != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"采集后六轴不再全部卸力：{torque_after}")
    except BaseException as exc:
        failure = exc
        torque_after = None
        raw_position_after = None
    finally:
        reader.close()

    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        "parameter_ids": ["CAL-051", "CAL-053"],
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
        "torque_enabled_before": torque_before if "torque_before" in locals() else None,
        "torque_enabled_after": torque_after,
        "ids": ids if "ids" in locals() else None,
        "model_numbers": models if "models" in locals() else None,
        "raw_position_before": (
            raw_position_before if "raw_position_before" in locals() else None
        ),
        "raw_position_after": raw_position_after,
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failure is not None:
        raise SystemExit(f"CAL-051采集失败：{failure}; completed={len(records)}")
    durations_ms = [
        (row["read_end_ns"] - row["read_start_ns"]) / 1e6 for row in records
    ]
    print(json.dumps({
        "ticks": len(records),
        "read_min_ms": min(durations_ms),
        "read_max_ms": max(durations_ms),
        "torque_enabled_after": torque_after,
        "raw_position_stable": raw_position_before == raw_position_after,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
