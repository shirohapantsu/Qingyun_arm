#!/usr/bin/env python3
"""通过标定 profile 的稳定串口地址生成六台舵机 EEPROM 只读快照。"""

from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usb_identity(resolved_port: Path) -> dict[str, str | None]:
    current = Path("/sys/class/tty") / resolved_port.name / "device"
    try:
        current = current.resolve(strict=True)
    except FileNotFoundError:
        return {"vid": None, "pid": None, "serial": None}
    for parent in (current, *current.parents):
        vendor = parent / "idVendor"
        product = parent / "idProduct"
        if vendor.is_file() and product.is_file():
            serial_file = parent / "serial"
            return {
                "vid": vendor.read_text(encoding="ascii").strip(),
                "pid": product.read_text(encoding="ascii").strip(),
                "serial": (serial_file.read_text(encoding="utf-8").strip()
                           if serial_file.is_file() else None),
            }
    return {"vid": None, "pid": None, "serial": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    args = parser.parse_args()

    profile_path = args.profile.resolve(strict=True)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = load_calibration_profile(profile_path)
    stable_port = Path(profile.motor.port)
    if not str(stable_port).startswith("/dev/serial/by-id/"):
        raise SystemExit(f"拒绝非稳定 by-id 地址：{stable_port}")
    try:
        resolved_port = stable_port.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"稳定串口当前不存在：{stable_port}") from exc
    identity = usb_identity(resolved_port)
    if identity["vid"] != "1a86" or identity["pid"] != "55d3":
        raise SystemExit(f"USB 身份不匹配，拒绝访问：{identity}")

    reader = CalibrationReader(profile)
    try:
        torque = reader.read_torque_enabled()
        motors = reader.read_eeprom_snapshot()
    finally:
        reader.close()

    expected_firmware = list(profile.motor.firmware)
    checks = []
    for index, row in enumerate(motors):
        registers = row["registers"]
        firmware = f"{registers['Firmware_Major_Version']['raw']}." \
                   f"{registers['Firmware_Minor_Version']['raw']}"
        model_number = registers["Model_Number"]["raw"]
        actual_id = registers["ID"]["raw"]
        checks.append({
            "name": MOTOR_NAMES[index],
            "configured_id": index + 1,
            "id_matches": actual_id == index + 1,
            "model_number": model_number,
            "model_matches_sts3215": model_number == 777,
            "firmware": firmware,
            "firmware_matches_profile": firmware == expected_firmware[index],
        })
    if not all(row["id_matches"] and row["model_matches_sts3215"]
               and row["firmware_matches_profile"] for row in checks):
        raise SystemExit(f"身份核验失败，拒绝保存快照：{checks}")

    table_path = ROOT / "libs/so_arm_core/motors/feetech/tables.py"
    calibration_path = (
        profile_path.parent / raw_profile["model"]["motor_calibration_path"]
    ).resolve(strict=True)
    payload = {
        "schema_version": 1,
        "parameter_id": "CAL-108",
        "captured_at": datetime.now().astimezone().isoformat(),
        "source": "hardware_read_only",
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "motor_calibration": str(calibration_path.relative_to(ROOT)),
        "motor_calibration_sha256": sha256_file(calibration_path),
        "control_table": str(table_path.relative_to(ROOT)),
        "control_table_sha256": sha256_file(table_path),
        "stable_port": str(stable_port),
        "resolved_port": str(resolved_port),
        "usb_identity": identity,
        "environment": {
            "power": args.power,
            "ambient_temperature_c": args.ambient_temperature_c,
            "base_mount": args.base_mount,
        },
        "torque_enabled_by_motor": torque,
        "identity_checks": checks,
        "motors": motors,
    }
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有快照：{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"CAL-108 只读快照已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
