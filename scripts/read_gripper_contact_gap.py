#!/usr/bin/env python3
"""CAL-084：只读保存首次接触摆位及尺测间隙，不上力或闭爪。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import _parse_gripper, load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from qingyun.grabbing.kinematics_ext import gripper_pct_to_gap  # noqa: E402
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY, _capture_burst  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-role", required=True, choices=("minimum", "maximum"))
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--contact-band-mm", required=True, type=float)
    parser.add_argument("--measured-gap-mm", required=True, type=float)
    parser.add_argument("--pose-held-confirmed", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖证据：{args.output}")
    if args.repeat < 1 or not all(math.isfinite(v) and v > 0 for v in (args.contact_band_mm, args.measured_gap_mm)):
        raise SystemExit("轮次和测量值必须有效且为正")
    profile_path = args.profile.resolve(strict=True)
    profile = load_calibration_profile(profile_path)
    port = Path(profile.motor.port)
    if str(port) != EXPECTED_STABLE_PORT:
        raise SystemExit("稳定串口地址不匹配")
    resolved = port.resolve(strict=True)
    if not resolved.name.startswith("ttyACM"):
        raise SystemExit("稳定地址未解析为ttyACM")
    identity = usb_identity(resolved)
    if identity != EXPECTED_USB_IDENTITY:
        raise SystemExit("USB身份不匹配")
    reader = CalibrationReader(profile)
    try:
        ids = reader.read_common_register("ID")
        models = reader.read_common_register("Model_Number")
        if ids != [1, 2, 3, 4, 5, 6] or models != [777] * 6:
            raise SystemExit("ID或型号不匹配")
        burst = _capture_burst(reader, 10, 0.05)
        raw = burst["raw_position_representative"][5]
        pct = reader.mapping.raw_to_gripper_pct(raw)
    finally:
        reader.close()
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    gripper = _parse_gripper("gripper", raw_profile["gripper"])
    gap_mm = gripper_pct_to_gap(pct, gripper) * 1000.0
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-084",
        "source": "manual_contact_pose_hardware_read_only",
        "status": "CAPTURED_PENDING_REVIEW",
        "contact_label": "first_contact",
        "accepted_for_profile_fit": False,
        "profile_sha256": sha256_file(profile_path),
        "stable_port": str(port),
        "resolved_port": str(resolved),
        "usb_identity": identity,
        "ids": ids,
        "models": models,
        "sample_role": args.sample_role,
        "repeat": args.repeat,
        "contact_band_distance_from_each_fingertip_mm": args.contact_band_mm,
        "operator_measured_gap_mm": args.measured_gap_mm,
        # 真实接触带的尺测量与gap表参考坐标分列；变接触带时绝不能混填。
        "actual_gap_m": args.measured_gap_mm / 1000.0,
        "reference_gap_m": gap_mm / 1000.0,
        "pose_held_through_capture_confirmed": args.pose_held_confirmed,
        "burst": burst,
        "gripper_raw": raw,
        "gripper_pct": pct,
        "gap_table_interpolated_mm": gap_mm,
        "direct_minus_table_gap_mm": args.measured_gap_mm - gap_mm,
        "environment_basis": "CAL-050 current demo baseline; no changed conditions reported",
        "write_instructions_sent": 0,
        "motion_instructions_sent": 0,
        "eeprom_written": False,
        "profile_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"raw={raw}, 开度={pct:.3f}%, gap表={gap_mm:.3f}mm, 尺测={args.measured_gap_mm:.3f}mm；六轴卸力；evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
