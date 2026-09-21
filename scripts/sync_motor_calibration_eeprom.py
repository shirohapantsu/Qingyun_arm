#!/usr/bin/env python3
"""将权威 motor_calibration.json 受控同步到六台 STS3215 EEPROM。

默认只读并输出计划；只有同时给出 ``--execute`` 和文件的完整 SHA256 才会写。
写入范围严格限定为 Homing_Offset、Min_Position_Limit、Max_Position_Limit，
每写一个字段立即读回精确比较。脚本不启用力矩、不发送位置命令，也不修改
Response_Status_Level；若任一舵机力矩不为 0、身份不符或串口不是稳定 by-id
地址，写入前即拒绝。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import JointsParams, load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import (  # noqa: E402
    MotorMapping,
    SerialTransport,
    StsProtocol,
)
from libs.so_arm_core.motors.feetech.tables import MODEL_NUMBER_TABLE  # noqa: E402


CALIBRATION_REGISTERS = (
    ("Homing_Offset", "homing_offset"),
    ("Min_Position_Limit", "range_min"),
    ("Max_Position_Limit", "range_max"),
)
EXPECTED_USB = {"vid": "1a86", "pid": "55d3", "serial": "5B14114904"}
TRANSACTION_TIMEOUT_S = 5.0
EEPROM_SETTLE_S = 0.05


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
                "serial": (
                    serial_file.read_text(encoding="utf-8").strip()
                    if serial_file.is_file()
                    else None
                ),
            }
    return {"vid": None, "pid": None, "serial": None}


def _placeholder_joints() -> JointsParams:
    return JointsParams(
        sign=np.ones(5, dtype=np.int64),
        zero_offset_deg=np.zeros(5, dtype=np.float64),
        measured_limits_deg=np.array([[-180.0, 180.0]] * 5),
        application_limits_deg=np.array([[-180.0, 180.0]] * 5),
        margin_deg=np.full(5, 0.1),
        hard_current_ma=np.full(5, 900.0),
    )


def _read_value(block: bytes, address: int, length: int) -> int:
    chunk = block[address : address + length]
    if len(chunk) != length:
        raise RuntimeError(
            f"EEPROM 块不足：address={address}, length={length}, block={len(block)}"
        )
    return int.from_bytes(chunk, byteorder="little", signed=False)


def _read_bus_state(
    protocol: StsProtocol, mapping: MotorMapping
) -> dict[str, Any]:
    deadline = time.monotonic() + TRANSACTION_TIMEOUT_S
    ids = [axis.servo_id for axis in mapping.axes]
    blocks = protocol.bulk_read(ids, 0, 40, deadline)

    response_levels: list[int] = []
    motors: list[dict[str, Any]] = []
    for axis in mapping.axes:
        block = blocks[axis.servo_id]

        def raw(register: str) -> int:
            address, length = mapping.register(axis, register)
            return _read_value(block, address, length)

        response_level = raw("Response_Status_Level")
        if response_level not in (0, 1):
            raise RuntimeError(
                f"{axis.name} Response_Status_Level={response_level}，无法安全确定写应答策略"
            )
        response_levels.append(response_level)
        protocol.set_write_response(axis.servo_id, expects_ack=(response_level == 0))

        firmware = f"{raw('Firmware_Major_Version')}.{raw('Firmware_Minor_Version')}"
        model_number = raw("Model_Number")
        actual_id = raw("ID")
        expected_model_number = int(MODEL_NUMBER_TABLE[axis.model])
        if actual_id != axis.servo_id:
            raise RuntimeError(
                f"{axis.name} ID 不匹配：配置 {axis.servo_id}，读回 {actual_id}"
            )
        if model_number != expected_model_number:
            raise RuntimeError(
                f"{axis.name} 型号不匹配：期望 {expected_model_number}，读回 {model_number}"
            )
        if firmware != axis.firmware:
            raise RuntimeError(
                f"{axis.name} 固件不匹配：profile {axis.firmware}，读回 {firmware}"
            )
        motors.append(
            {
                "name": axis.name,
                "id": axis.servo_id,
                "model_number": model_number,
                "firmware": firmware,
                "response_status_level": response_level,
                "raw_hex_0_39": block.hex(),
                "homing_offset": mapping.decode_signed(
                    axis, "Homing_Offset", raw("Homing_Offset")
                ),
                "range_min": raw("Min_Position_Limit"),
                "range_max": raw("Max_Position_Limit"),
            }
        )

    torque_address, torque_length = mapping.common_register("Torque_Enable")
    lock_address, lock_length = mapping.common_register("Lock")
    deadline = time.monotonic() + TRANSACTION_TIMEOUT_S
    torque_blocks = protocol.bulk_read(ids, torque_address, torque_length, deadline)
    deadline = time.monotonic() + TRANSACTION_TIMEOUT_S
    lock_blocks = protocol.bulk_read(ids, lock_address, lock_length, deadline)
    torque = [int(torque_blocks[axis.servo_id][0]) for axis in mapping.axes]
    locks = [int(lock_blocks[axis.servo_id][0]) for axis in mapping.axes]
    if any(value != 0 for value in torque):
        raise RuntimeError(f"六轴力矩未全部关闭，拒绝写入：{torque}")
    if any(value not in (0, 1) for value in locks):
        raise RuntimeError(f"Lock 寄存器不是 0/1，拒绝写入：{locks}")
    return {
        "motors": motors,
        "torque_enabled_by_motor": torque,
        "lock_by_motor": locks,
        "response_status_level_by_motor": response_levels,
    }


def _expected_state(mapping: MotorMapping) -> list[dict[str, Any]]:
    return [
        {
            "name": axis.name,
            "id": axis.servo_id,
            "drive_mode": axis.drive_mode,
            "homing_offset": axis.homing_offset,
            "range_min": axis.range_min,
            "range_max": axis.range_max,
        }
        for axis in mapping.axes
    ]


def _write_and_verify(
    protocol: StsProtocol,
    mapping: MotorMapping,
    axis,
    register: str,
    expected_decoded: int,
) -> dict[str, Any]:
    address, length = mapping.register(axis, register)
    encoded = mapping.encode_signed(axis, register, expected_decoded)
    if encoded >= 1 << (8 * length):
        raise RuntimeError(
            f"{axis.name}.{register} 编码值 {encoded} 超出 {length} 字节"
        )
    data = list(int(encoded).to_bytes(length, byteorder="little", signed=False))
    protocol.write_registers(
        axis.servo_id,
        address,
        data,
        time.monotonic() + TRANSACTION_TIMEOUT_S,
    )
    time.sleep(EEPROM_SETTLE_S)
    readback_raw = int.from_bytes(
        protocol.read_registers(
            axis.servo_id,
            address,
            length,
            time.monotonic() + TRANSACTION_TIMEOUT_S,
        ),
        byteorder="little",
        signed=False,
    )
    readback_decoded = mapping.decode_signed(axis, register, readback_raw)
    if readback_decoded != expected_decoded:
        raise RuntimeError(
            f"{axis.name}.{register} 写后读回不一致：期望 {expected_decoded}，"
            f"读回 {readback_decoded}（raw={readback_raw}）"
        )
    return {
        "name": axis.name,
        "id": axis.servo_id,
        "register": register,
        "address": address,
        "length": length,
        "expected_decoded": expected_decoded,
        "encoded_raw": encoded,
        "readback_raw": readback_raw,
        "readback_decoded": readback_decoded,
        "verified": True,
    }


def _set_and_verify_locks(
    protocol: StsProtocol, mapping: MotorMapping, target_locks: list[int]
) -> list[int]:
    address, length = mapping.common_register("Lock")
    current: list[int] | None = None
    # Lock 是幂等的 RAM 写保护位，不是 EEPROM 标定值。等级1的 WRITE 没有状态
    # 应答，因此允许最多三轮、且每轮只重发读回仍不一致的通道；校准字段本身仍
    # 坚持单次写后立即读回，不在这里重试。
    for _ in range(3):
        blocks = protocol.bulk_read(
            [axis.servo_id for axis in mapping.axes],
            address,
            length,
            time.monotonic() + TRANSACTION_TIMEOUT_S,
        )
        current = [int(blocks[axis.servo_id][0]) for axis in mapping.axes]
        if current == target_locks:
            return current
        for axis, actual, target in zip(mapping.axes, current, target_locks, strict=True):
            if actual == target:
                continue
            protocol.write_registers(
                axis.servo_id,
                address,
                [target],
                time.monotonic() + TRANSACTION_TIMEOUT_S,
            )
        time.sleep(EEPROM_SETTLE_S)
    blocks = protocol.bulk_read(
        [axis.servo_id for axis in mapping.axes],
        address,
        length,
        time.monotonic() + TRANSACTION_TIMEOUT_S,
    )
    current = [int(blocks[axis.servo_id][0]) for axis in mapping.axes]
    if current != target_locks:
        raise RuntimeError(f"Lock 状态恢复失败：期望 {target_locks}，读回 {current}")
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-calibration-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--restore-locks-only",
        action="store_true",
        help="不写校准字段；仅在校准值已经与权威文件一致时把六台 Lock 恢复为1",
    )
    parser.add_argument("--power", required=True)
    parser.add_argument("--ambient-temperature-c", required=True, type=float)
    parser.add_argument("--base-mount", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"拒绝覆盖既有报告：{args.output}")
    profile_path = args.profile.resolve(strict=True)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = load_calibration_profile(profile_path)
    calibration_path = (
        profile_path.parent / raw_profile["model"]["motor_calibration_path"]
    ).resolve(strict=True)
    actual_sha256 = sha256_file(calibration_path)
    profile_sha256 = raw_profile["model"]["motor_calibration_sha256"]
    if actual_sha256 != args.expected_calibration_sha256:
        raise SystemExit(
            "拒绝：当前校准文件 SHA256 与命令确认值不一致："
            f"{actual_sha256} != {args.expected_calibration_sha256}"
        )
    if actual_sha256 != profile_sha256:
        raise SystemExit(
            f"拒绝：当前校准文件 SHA256 与 profile 绑定值不一致：{profile_sha256}"
        )

    stable_port = Path(profile.motor.port)
    if not str(stable_port).startswith("/dev/serial/by-id/"):
        raise SystemExit(f"拒绝非稳定 by-id 地址：{stable_port}")
    try:
        resolved_port = stable_port.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"稳定串口当前不存在：{stable_port}") from exc
    if not resolved_port.name.startswith("ttyACM"):
        raise SystemExit(f"拒绝：稳定地址未解析到 ttyACM：{resolved_port}")
    identity = usb_identity(resolved_port)
    if identity != EXPECTED_USB:
        raise SystemExit(f"USB 身份不匹配，拒绝访问：{identity} != {EXPECTED_USB}")

    mapping = MotorMapping(
        profile.motor,
        profile.joints if profile.joints is not None else _placeholder_joints(),
        profile.motor_calibration,
        None,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "parameter_id": "CAL-012",
        "mode": (
            "execute"
            if args.execute
            else "restore_locks_only"
            if args.restore_locks_only
            else "dry_run"
        ),
        "started_at": datetime.now().astimezone().isoformat(),
        "success": False,
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "motor_calibration": str(calibration_path.relative_to(ROOT)),
        "motor_calibration_sha256": actual_sha256,
        "stable_port": str(stable_port),
        "resolved_port": str(resolved_port),
        "usb_identity": identity,
        "environment": {
            "power": args.power,
            "ambient_temperature_c": args.ambient_temperature_c,
            "base_mount": args.base_mount,
        },
        "allowed_eeprom_registers": [name for name, _ in CALIBRATION_REGISTERS],
        "expected": _expected_state(mapping),
        "operations": [],
    }

    transport = None
    original_locks: list[int] | None = None
    error: BaseException | None = None
    try:
        transport = SerialTransport(str(stable_port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        before = _read_bus_state(protocol, mapping)
        report["before"] = before
        original_locks = list(before["lock_by_motor"])

        current_by_name = {row["name"]: row for row in before["motors"]}
        plan: list[dict[str, Any]] = []
        for axis in mapping.axes:
            current = current_by_name[axis.name]
            for register, key in CALIBRATION_REGISTERS:
                expected = int(getattr(axis, key))
                if int(current[key]) != expected:
                    address, length = mapping.register(axis, register)
                    plan.append(
                        {
                            "name": axis.name,
                            "id": axis.servo_id,
                            "register": register,
                            "address": address,
                            "length": length,
                            "before_decoded": int(current[key]),
                            "expected_decoded": expected,
                        }
                    )
        report["plan"] = plan

        if args.restore_locks_only:
            if plan:
                raise RuntimeError(
                    "Lock-only 模式要求校准字段已经与权威文件一致；当前仍有差异："
                    f"{plan}"
                )
            report["lock_after_restore"] = _set_and_verify_locks(
                protocol, mapping, [1] * len(mapping.axes)
            )
            restored_state = _read_bus_state(protocol, mapping)
            report["after_lock_only_restore"] = restored_state
            if restored_state["torque_enabled_by_motor"] != [0] * len(mapping.axes):
                raise RuntimeError(
                    "恢复 Lock 后力矩状态改变："
                    f"{restored_state['torque_enabled_by_motor']}"
                )
        elif args.execute:
            # 串口上的等级1 WRITE 没有应答，偶发单个 Lock 写入未落地时不能据此
            # 推断 EEPROM 已经解锁。复用锁状态校验器，仅重发读回仍为1的通道。
            report["lock_after_unlock"] = _set_and_verify_locks(
                protocol, mapping, [0] * len(mapping.axes)
            )

            plan_keys = {(row["name"], row["register"]) for row in plan}
            for axis in mapping.axes:
                for register, key in CALIBRATION_REGISTERS:
                    if (axis.name, register) not in plan_keys:
                        continue
                    operation = _write_and_verify(
                        protocol,
                        mapping,
                        axis,
                        register,
                        int(getattr(axis, key)),
                    )
                    report["operations"].append(operation)

            after = _read_bus_state(protocol, mapping)
            report["after_before_lock_restore"] = after
            actual_after = {
                row["name"]: (
                    row["homing_offset"], row["range_min"], row["range_max"]
                )
                for row in after["motors"]
            }
            expected_after = {
                axis.name: (axis.homing_offset, axis.range_min, axis.range_max)
                for axis in mapping.axes
            }
            if actual_after != expected_after:
                raise RuntimeError(
                    f"六轴最终校准读回不一致：{actual_after} != {expected_after}"
                )
            if after["torque_enabled_by_motor"] != [0] * len(mapping.axes):
                raise RuntimeError(
                    f"写入后力矩状态改变：{after['torque_enabled_by_motor']}"
                )
        report["success"] = True
    except BaseException as exc:
        error = exc
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if transport is not None:
            if args.execute and original_locks is not None:
                try:
                    report["lock_after_restore"] = _set_and_verify_locks(
                        protocol, mapping, original_locks
                    )
                except BaseException as restore_exc:
                    report["lock_restore_error"] = {
                        "type": type(restore_exc).__name__,
                        "message": str(restore_exc),
                    }
                    report["success"] = False
                    if error is None:
                        error = restore_exc
            transport.close()
        report["completed_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if error is not None:
        raise SystemExit(f"CAL-012 同步失败；证据已保存到 {args.output}：{error}")
    print(
        (
            "CAL-012 写回并逐字段读回成功"
            if args.execute
            else "CAL-012 Lock 恢复成功"
            if args.restore_locks_only
            else "CAL-012 dry-run 通过"
        )
        + f"：{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
