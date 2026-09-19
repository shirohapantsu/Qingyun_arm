#!/usr/bin/env python3
"""CAL-085 完全重测：生产 close_speed 下的空爪/带物单轮接触电流采集。

2026-09-18 操作者下令忽略既有记录、完全重测 CAL-085。本工具每次执行只跑
一轮（--mode loaded 或 --mode empty），逐轮批准、逐轮取证：
  * 仅 6 号（gripper）电机运动，1~5 号全程卸力、零目标、零写目标；
  * 定位段用临时 SRAM，正式闭合/张开段恢复运行期原值，逐项读回核验；
  * 闭合命令按生产方式 30 Hz 小步 ramp（close_speed_pct_s / FPS）；
  * 带物轮：受阻冻结目标（反馈停滞或滞后超限）→ 保持 0.3 s（9 帧）→ 张开；
  * 空爪轮：闭合至空夹门（empty_closed_pct + empty_tol_pct）即停 → 张开；
  * 诊断保护（非生产参数）：|电流|>65 mA、|跟踪差|>150 raw、闭合+保持
    5 s 预算、张开 5 s 预算、tick 迟到 > max_tick_lateness_s 即中止。

在线停止逻辑说明：CAL-091/083 实测空爪 15%/s 正常跟踪滞后可达 59 raw，
故不能用固定 25 raw 滞后门判受阻；改为反馈停滞（4 帧位移 ≤2 raw）或
滞后 ≥75 raw（空爪最大 59 + 余量），且仅在 pct>空夹门+5% 的接触相关区
生效。本脚本不拟合阈值、不写 profile/EEPROM；生产判据重放由离线分析完成。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import MOTOR_NAMES  # noqa: E402
from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.kinematics_ext import MotionError, gripper_pct_to_gap  # noqa: E402
from qingyun.grabbing.motor_control import (  # noqa: E402
    MotorMapping,
    SerialTransport,
    StsProtocol,
    _feedback_block_layout,
)
import scripts.move_gripper_raw as io_mod  # noqa: E402
from scripts.move_gripper_raw import _read_raw, _sync_goal, _write_raw  # noqa: E402
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY  # noqa: E402
from scripts.read_servo_eeprom import sha256_file, usb_identity  # noqa: E402

APPROVALS = {
    "loaded": "批准，CAL-085重测带物就绪",
    "empty": "批准，CAL-085重测空爪就绪",
}
LOAD_LABELS = {
    "loaded": "independently supported plastic fixture, 20mm contact band",
    "empty": "empty_gripper",
}
# 诊断保护（非生产参数；生产判据离线重放）
ABORT_CURRENT_MA = 65.0
ABORT_TRACKING_RAW = 150
FREEZE_LAG_RAW = 75
STALL_WINDOW_SAMPLES = 4
STALL_MAX_MOVE_RAW = 2
# 电流爬升预触发（仅 loaded 轮；2026-09-18 阶段1证据：空爪基线峰13mA，
# 接触后爬升至71.5mA，26mA=13×2 在压缩~1.5mm 处先于位置类条件触发）。
FREEZE_RAMP_CURRENT_MA = 26.0
FREEZE_RAMP_FRAMES = 2
FREEZE_ZONE_MARGIN_PCT = 5.0
CLOSING_BUDGET_S = 5.0
OPENING_BUDGET_S = 5.0
HOLD_FRAMES = 9
POSITION_TOLERANCE_RAW = 5
# 定位/到位容差：STS3215 低载内部到位死区实测约4~6 raw（CAL-026记载~4；
# 本轮2026-09-18实测289帧静止于-6raw、0~6.5mA），容差须覆盖该死区。
PREPOSITION_TOLERANCE_RAW = 8
PREPOSITION_TIMEOUT_S = 15.0
SETTLE_TIMEOUT_S = 3.0
PREFLIGHT_TIMEOUT_S = 1.0
MOTION_TIMEOUT_S = 0.25
CLEANUP_TIMEOUT_S = 0.5
SRAM_REGISTERS = ("Acceleration", "Goal_Time", "Goal_Velocity", "Torque_Limit")
EXPECTED_RUNTIME_SRAM = {"Acceleration": 0, "Goal_Time": 0, "Goal_Velocity": 0, "Torque_Limit": 500}
PREPOSITION_SRAM = {"Acceleration": 5, "Goal_Time": 0, "Goal_Velocity": 10, "Torque_Limit": 600}
EXPECTED_CONTROLLERS = {
    "P_Coefficient": 16,
    "D_Coefficient": 32,
    "I_Coefficient": 0,
    "Minimum_Startup_Force": 16,
    "CW_Dead_Zone": 1,
    "CCW_Dead_Zone": 1,
}
CONTROLLER_REGISTERS = tuple(EXPECTED_CONTROLLERS)


def pct_to_raw(closed_raw: int, open_raw: int, pct: float) -> int:
    return int(round(closed_raw + (open_raw - closed_raw) * pct / 100.0))


def raw_to_pct(closed_raw: int, open_raw: int, raw: int) -> float:
    return (float(raw) - closed_raw) * 100.0 / (open_raw - closed_raw)


def rolling_mean_max(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return max(
        statistics.fmean(values[i - window + 1 : i + 1]) for i in range(window - 1, len(values))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(APPROVALS))
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--ambient-temperature-c", type=float, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("拒绝覆盖既有证据")
    if args.operator_confirmation != APPROVALS[args.mode]:
        raise SystemExit(f"缺少精确的 {APPROVALS[args.mode]} 操作者确认")

    profile_path = args.profile.resolve(strict=True)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = load_calibration_profile(profile_path)
    if profile.joints is None:
        raise SystemExit("关节标定未完成，拒绝访问")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    jaw = mapping.axis("gripper")
    # load_calibrationProfile 只含 motor/joints 最小集；gripper 组按 CAL-082 先例
    # 从原始 JSON 读取，gap 换算复用生产插值函数（含禁止外插保护）。
    raw_gripper = raw_profile["gripper"]
    preopen_pct = float(raw_gripper["preopen_pct"])
    close_speed = float(raw_gripper["close_speed_pct_s"])
    open_speed = float(raw_gripper["open_speed_pct_s"])
    empty_gate_pct = float(raw_gripper["empty_closed_pct"]) + float(raw_gripper["empty_tol_pct"])
    contact_gap_range = [float(v) for v in raw_gripper["contact_gap_range_m"]]
    window_samples = int(raw_gripper["window_samples"])
    gap_tables = SimpleNamespace(
        gap_table=[
            SimpleNamespace(gripper_pct=float(e["gripper_pct"]), gap_m=float(e["gap_m"]))
            for e in raw_gripper["gap_table"]
        ]
    )
    if min(close_speed, open_speed) <= 0.0:
        raise SystemExit("开合速度必须为正")
    if not 0.0 < empty_gate_pct < preopen_pct <= 100.0:
        raise SystemExit(f"空夹门/preopen 关系异常：gate={empty_gate_pct} preopen={preopen_pct}")
    if not (len(contact_gap_range) == 2 and contact_gap_range[0] < contact_gap_range[1]):
        raise SystemExit(f"contact_gap_range_m 异常：{contact_gap_range}")
    if window_samples < 1:
        raise SystemExit("window_samples 必须 ≥1")
    closed_raw = int(profile.motor.gripper_closed_raw)
    open_raw = int(profile.motor.gripper_open_raw)
    preopen_raw = pct_to_raw(closed_raw, open_raw, preopen_pct)
    fps = float(raw_profile["timing"]["fps"])
    tick_s = 1.0 / fps
    max_tick_lateness_s = float(raw_profile["timing"]["max_tick_lateness_s"])
    if not jaw.range_min <= preopen_raw <= jaw.range_max:
        raise SystemExit(f"preopen raw={preopen_raw} 越出 EEPROM 范围 [{jaw.range_min},{jaw.range_max}]")

    report: dict = {
        "schema_version": 2,
        "parameter_id": "CAL-085",
        "test": f"production_speed_single_run_{args.mode}",
        "mode": args.mode,
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "operator_confirmation": args.operator_confirmation,
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "motor_calibration_sha256": sha256_file(profile_path.parent / "motor_calibration.json"),
        "environment": {
            "power": args.power,
            "ambient_temperature_c": args.ambient_temperature_c,
            "base_mount": args.base_mount,
            "load": LOAD_LABELS[args.mode],
        },
        "trajectory_from_profile": {
            "preopen_pct": preopen_pct,
            "preopen_raw": preopen_raw,
            "close_speed_pct_s": close_speed,
            "open_speed_pct_s": open_speed,
            "empty_gate_pct": empty_gate_pct,
            "contact_gap_range_m": contact_gap_range,
            "window_samples": window_samples,
            "fps": fps,
            "tick_s": tick_s,
            "max_tick_lateness_s": max_tick_lateness_s,
            "closed_raw": closed_raw,
            "open_raw": open_raw,
        },
        "diagnostic_bounds": {
            "abort_current_ma": ABORT_CURRENT_MA,
            "abort_tracking_raw": ABORT_TRACKING_RAW,
            "freeze_lag_raw": FREEZE_LAG_RAW,
            "freeze_ramp_current_ma": FREEZE_RAMP_CURRENT_MA,
            "freeze_ramp_frames": FREEZE_RAMP_FRAMES,
            "stall_window_samples": STALL_WINDOW_SAMPLES,
            "stall_max_move_raw": STALL_MAX_MOVE_RAW,
            "freeze_zone_min_pct": empty_gate_pct + FREEZE_ZONE_MARGIN_PCT,
            "closing_budget_s": CLOSING_BUDGET_S,
            "opening_budget_s": OPENING_BUDGET_S,
            "hold_frames": HOLD_FRAMES,
            "note": "全部为采集诊断保护，非生产参数；生产判据离线重放",
        },
        "eeprom_written": False,
        "profile_written": False,
        "joint_servo_motion_commands_sent": 0,
        "gripper_motion_commands_sent": 0,
        "preposition_samples": [],
        "samples": [],
        "post_torque_off_samples": [],
    }

    transport = None
    protocol: StsProtocol | None = None
    torque_may_be_enabled = False
    originals: dict[str, int] = {}
    failure: BaseException | None = None
    stop_reason: str | None = None
    frozen_cmd_raw: int | None = None

    # 单事务反馈块（位置/速度/电流，块起止由 vendor 表推导，同生产 get_feedback）
    pos_addr, pos_len = mapping.common_register("Present_Position")
    vel_addr, vel_len = mapping.common_register("Present_Velocity")
    cur_addr, cur_len = mapping.common_register("Present_Current")
    fb_addr, fb_len, fb_off = _feedback_block_layout(
        (pos_addr, pos_len), (vel_addr, vel_len), (cur_addr, cur_len)
    )

    def sample_block(phase: str, index: int, command_raw: int | None, command_pct: float | None):
        protocol.reset_input()
        data = protocol.read_registers(
            jaw.servo_id, fb_addr, fb_len, time.monotonic() + MOTION_TIMEOUT_S
        )
        pos = StsProtocol.join_u16(data[fb_off["Present_Position"] : fb_off["Present_Position"] + 2])
        vel = StsProtocol.join_u16(data[fb_off["Present_Velocity"] : fb_off["Present_Velocity"] + 2])
        cur = StsProtocol.join_u16(data[fb_off["Present_Current"] : fb_off["Present_Current"] + 2])
        pct = raw_to_pct(closed_raw, open_raw, pos)
        try:
            gap = float(gripper_pct_to_gap(pct, gap_tables))
        except MotionError:
            gap = None
        row = {
            "phase": phase,
            "index": index,
            "monotonic_ns": time.monotonic_ns(),
            "command_raw": command_raw,
            "command_pct": command_pct,
            "position_raw": pos,
            "gripper_pct": pct,
            "gap_m": gap,
            "velocity_register_raw": vel,
            "velocity_deg_s": mapping.raw_velocity_to_deg_s("gripper", vel),
            "current_register_raw": cur,
            "current_ma": mapping.raw_current_to_ma("gripper", cur),
            "tracking_error_raw": (pos - command_raw) if command_raw is not None else None,
        }
        return row

    try:
        # --- 阶段 0：只读前置 ---
        stable_port = Path(profile.motor.port)
        if str(stable_port) != EXPECTED_STABLE_PORT:
            raise RuntimeError(f"稳定串口不匹配：{stable_port}")
        resolved_port = stable_port.resolve(strict=True)
        if not resolved_port.name.startswith("ttyACM"):
            raise RuntimeError(f"稳定串口未解析到 ttyACM：{resolved_port}")
        identity = usb_identity(resolved_port)
        if identity != EXPECTED_USB_IDENTITY:
            raise RuntimeError(f"USB 身份不匹配：{identity}")
        occupancy = subprocess.run(["fuser", str(stable_port)], capture_output=True, text=True)
        if occupancy.returncode != 1 or occupancy.stdout.strip():
            raise RuntimeError("串口被占用或占用检查失败")
        report.update(stable_port=str(stable_port), resolved_port=str(resolved_port), usb_identity=identity)

        transport = SerialTransport(str(stable_port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        io_mod.TRANSACTION_TIMEOUT_S = PREFLIGHT_TIMEOUT_S
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError(f"{axis.name} Response_Status_Level={level}")
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
        preflight = {
            key: [_read_raw(protocol, mapping, a, reg) for a in mapping.axes]
            for key, reg in (
                ("ids", "ID"),
                ("models", "Model_Number"),
                ("firmware_major", "Firmware_Major_Version"),
                ("firmware_minor", "Firmware_Minor_Version"),
                ("torques", "Torque_Enable"),
            )
        }
        report["readonly_preflight"] = preflight
        if preflight["ids"] != list(range(1, len(MOTOR_NAMES) + 1)):
            raise RuntimeError(f"ID 不匹配：{preflight['ids']}")
        if preflight["models"] != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"型号不匹配：{preflight['models']}")
        expected_firmware = []
        for fw in profile.motor.firmware:
            major, minor = str(fw).split(".")
            expected_firmware.append((int(major), int(minor)))
        actual_firmware = list(
            zip(preflight["firmware_major"], preflight["firmware_minor"])
        )
        if actual_firmware != expected_firmware:
            raise RuntimeError(
                f"固件不匹配：{actual_firmware}，profile 期望 {expected_firmware}"
            )
        if preflight["torques"] != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"试验前六轴必须全部卸力：{preflight['torques']}")

        report["readonly_eeprom"] = {}
        for axis in mapping.axes:
            lo = _read_raw(protocol, mapping, axis, "Min_Position_Limit")
            hi = _read_raw(protocol, mapping, axis, "Max_Position_Limit")
            offset = mapping.decode_signed(axis, "Homing_Offset", _read_raw(protocol, mapping, axis, "Homing_Offset"))
            report["readonly_eeprom"][axis.name] = {
                "range_min": lo,
                "range_max": hi,
                "homing_offset": offset,
            }
            if (lo, hi, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f"{axis.name} EEPROM 与权威校准不一致；未做任何写")

        controllers = {
            reg: _read_raw(protocol, mapping, jaw, reg) for reg in CONTROLLER_REGISTERS
        }
        report["readonly_gripper_controller_settings"] = controllers
        if controllers != EXPECTED_CONTROLLERS:
            raise RuntimeError(f"夹爪控制器字段与权威快照不一致：{controllers}")

        frames = []
        for index in range(10):
            frames.append([_read_raw(protocol, mapping, a, "Present_Position") for a in mapping.axes])
            if index < 9:
                time.sleep(0.05)
        spans = [max(f[i] for f in frames) - min(f[i] for f in frames) for i in range(len(MOTOR_NAMES))]
        torques = [_read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes]
        report["fresh_readonly_baseline"] = {
            "raw_position_by_sample": frames,
            "raw_position_static_span": spans,
            "torque_enabled_after": torques,
        }
        report["arm_positions_outside_calibration_domain_diagnostic"] = [
            {
                "servo_id": a.servo_id,
                "positions_raw": [f[i] for f in frames],
                "range_raw": [a.range_min, a.range_max],
            }
            for i, a in enumerate(mapping.axes[:5])
            if any(not a.range_min <= f[i] <= a.range_max for f in frames)
        ]
        if torques != [0] * len(MOTOR_NAMES) or any(s > 2 for s in spans):
            raise RuntimeError(f"六轴静态基线不满足（span={spans} torque={torques}）")
        start_pos = frames[-1][5]
        if not closed_raw - POSITION_TOLERANCE_RAW <= start_pos <= open_raw + POSITION_TOLERANCE_RAW:
            raise RuntimeError(f"夹爪起始位置越出标定端点：{start_pos}")

        originals = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in SRAM_REGISTERS}
        report["original_sram_settings"] = originals.copy()
        if originals != EXPECTED_RUNTIME_SRAM:
            raise RuntimeError(f"运行期 SRAM 非预期：{originals}，期望 {EXPECTED_RUNTIME_SRAM}")

        # --- 定位段：临时 SRAM 温和移动到 preopen，随后恢复运行期原值 ---
        io_mod.TRANSACTION_TIMEOUT_S = MOTION_TIMEOUT_S
        for reg, value in PREPOSITION_SRAM.items():
            _write_raw(protocol, mapping, jaw, reg, value)
        torque_may_be_enabled = True
        _sync_goal(protocol, mapping, jaw, start_pos)
        _write_raw(protocol, mapping, jaw, "Torque_Enable", 1)
        time.sleep(0.05)
        if _read_raw(protocol, mapping, jaw, "Torque_Enable") != 1:
            raise RuntimeError("夹爪上力未生效")
        _sync_goal(protocol, mapping, jaw, preopen_raw)
        report["gripper_motion_commands_sent"] += 1
        deadline = time.monotonic() + PREPOSITION_TIMEOUT_S
        preposition_settled: list[int] = []
        while True:
            row = sample_block("preposition", len(preposition_settled) + 1, preopen_raw, preopen_pct)
            report["preposition_samples"].append(row)
            if abs(row["current_ma"]) > ABORT_CURRENT_MA:
                raise RuntimeError(f"定位段夹爪电流超限：{row['current_ma']}mA")
            if preposition_settled and preposition_settled[-1] == row["position_raw"]:
                preposition_settled.append(row["position_raw"])
            else:
                preposition_settled = [row["position_raw"]]
            if (
                len(preposition_settled) >= 3
                and abs(row["position_raw"] - preopen_raw) <= PREPOSITION_TOLERANCE_RAW
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"夹爪未在时限内到达 preopen {preopen_raw}")
            time.sleep(0.05)

        for reg, value in originals.items():
            _write_raw(protocol, mapping, jaw, reg, value)
            time.sleep(0.03)
            if _read_raw(protocol, mapping, jaw, reg) != value:
                raise RuntimeError(f"正式段前 {reg} 恢复读回不一致")
        time.sleep(0.2)
        report["test_sram_settings"] = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in SRAM_REGISTERS}
        if report["test_sram_settings"] != originals:
            raise RuntimeError("正式段前 SRAM 未恢复到运行期原值")

        # --- 闭合段：生产方式 30 Hz 小步 ramp ---
        freeze_zone_min_pct = empty_gate_pct + FREEZE_ZONE_MARGIN_PCT
        base = time.monotonic()
        closing_deadline_total = base + CLOSING_BUDGET_S
        cmd_pct = preopen_pct
        index = 0
        closing_positions: list[int] = []
        ramp_high_frames = 0
        while True:
            index += 1
            due = base + (index - 1) * tick_s
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            if time.monotonic() - due > max_tick_lateness_s:
                raise RuntimeError(f"闭合段第{index}步迟到超过时序上限")
            cmd_pct = max(0.0, cmd_pct - close_speed / fps)
            cmd_raw = pct_to_raw(closed_raw, open_raw, cmd_pct)
            _sync_goal(protocol, mapping, jaw, cmd_raw)
            report["gripper_motion_commands_sent"] += 1
            sample_due = base + index * tick_s
            remaining = sample_due - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            row = sample_block("closing", index, cmd_raw, cmd_pct)
            report["samples"].append(row)
            closing_positions.append(row["position_raw"])
            lag = row["tracking_error_raw"]
            if abs(row["current_ma"]) > ABORT_CURRENT_MA:
                raise RuntimeError(f"闭合段夹爪电流超限：{row['current_ma']}mA")
            if lag > ABORT_TRACKING_RAW:
                raise RuntimeError(f"闭合段跟踪差超限：{lag}raw")
            in_freeze_zone = row["gripper_pct"] > freeze_zone_min_pct
            if args.mode == "loaded" and in_freeze_zone:
                ramp_high_frames = (
                    ramp_high_frames + 1 if abs(row["current_ma"]) >= FREEZE_RAMP_CURRENT_MA else 0
                )
                if ramp_high_frames >= FREEZE_RAMP_FRAMES:
                    stop_reason = "FREEZE_RAMP"
                    frozen_cmd_raw = cmd_raw
                elif (
                    len(closing_positions) >= STALL_WINDOW_SAMPLES
                    and closing_positions[-STALL_WINDOW_SAMPLES] - row["position_raw"]
                    <= STALL_MAX_MOVE_RAW
                ):
                    stop_reason = "FREEZE_STALL"
                    frozen_cmd_raw = cmd_raw
                elif lag >= FREEZE_LAG_RAW:
                    stop_reason = "FREEZE_LAG"
                    frozen_cmd_raw = cmd_raw
            if stop_reason is not None:
                report["freeze"] = {
                    "stop_reason": stop_reason,
                    "index": index,
                    "lag_raw": lag,
                    "feedback_raw": row["position_raw"],
                    "feedback_pct": row["gripper_pct"],
                    "gap_m": row["gap_m"],
                    "velocity_deg_s": row["velocity_deg_s"],
                    "current_ma": row["current_ma"],
                    "frozen_cmd_raw": frozen_cmd_raw,
                    "frozen_cmd_pct": cmd_pct,
                }
                break
            if row["gripper_pct"] <= empty_gate_pct:
                stop_reason = "NO_CONTACT" if args.mode == "loaded" else "EMPTY_REACHED"
                break
            if time.monotonic() >= closing_deadline_total:
                raise RuntimeError("闭合段预算耗尽仍未停止")

        # --- 保持段（仅带物受阻冻结）：固定目标 9 帧 ---
        if args.mode == "loaded" and frozen_cmd_raw is not None:
            frozen_pct = cmd_pct
            hold_base = time.monotonic()
            for hold_index in range(1, HOLD_FRAMES + 1):
                due = hold_base + (hold_index - 1) * tick_s
                now = time.monotonic()
                if now < due:
                    time.sleep(due - now)
                if time.monotonic() - due > max_tick_lateness_s:
                    raise RuntimeError(f"保持段第{hold_index}帧迟到超过时序上限")
                _sync_goal(protocol, mapping, jaw, frozen_cmd_raw)
                report["gripper_motion_commands_sent"] += 1
                sample_due = hold_base + hold_index * tick_s
                remaining = sample_due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                row = sample_block("hold", hold_index, frozen_cmd_raw, frozen_pct)
                report["samples"].append(row)
                if abs(row["current_ma"]) > ABORT_CURRENT_MA:
                    raise RuntimeError(f"保持段夹爪电流超限：{row['current_ma']}mA")
                if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                    raise RuntimeError(f"保持段跟踪差超限：{row['tracking_error_raw']}raw")
                if time.monotonic() >= closing_deadline_total:
                    raise RuntimeError("闭合+保持总预算耗尽")

        # --- 张开段：回 preopen 并确认到位 ---
        open_base = time.monotonic()
        open_deadline = open_base + OPENING_BUDGET_S
        open_index = 0
        while cmd_pct < preopen_pct:
            open_index += 1
            due = open_base + (open_index - 1) * tick_s
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            if time.monotonic() - due > max_tick_lateness_s:
                raise RuntimeError(f"张开段第{open_index}步迟到超过时序上限")
            cmd_pct = min(preopen_pct, cmd_pct + open_speed / fps)
            cmd_raw = pct_to_raw(closed_raw, open_raw, cmd_pct)
            _sync_goal(protocol, mapping, jaw, cmd_raw)
            report["gripper_motion_commands_sent"] += 1
            sample_due = open_base + open_index * tick_s
            remaining = sample_due - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            row = sample_block("opening", open_index, cmd_raw, cmd_pct)
            report["samples"].append(row)
            if abs(row["current_ma"]) > ABORT_CURRENT_MA:
                raise RuntimeError(f"张开段夹爪电流超限：{row['current_ma']}mA")
            if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                raise RuntimeError(f"张开段跟踪差超限：{row['tracking_error_raw']}raw")
            if time.monotonic() >= open_deadline:
                raise RuntimeError("张开段预算耗尽")

        settle_deadline = time.monotonic() + SETTLE_TIMEOUT_S
        # STS3215 低载静态定位有约4 raw内部到位死区（CAL-026记载，本轮实测
        # 稳定停在-6raw），故到位判据用“连续3帧位置不变”，不用单帧Moving标志。
        settled_frames: list[int] = []
        while True:
            row = sample_block("settle", len(settled_frames) + 1, preopen_raw, preopen_pct)
            report["samples"].append(row)
            if abs(row["current_ma"]) > ABORT_CURRENT_MA:
                raise RuntimeError(f"到位段夹爪电流超限：{row['current_ma']}mA")
            if settled_frames and settled_frames[-1] == row["position_raw"]:
                settled_frames.append(row["position_raw"])
            else:
                settled_frames = [row["position_raw"]]
            if len(settled_frames) >= 3:
                break
            if time.monotonic() >= settle_deadline:
                raise RuntimeError(f"夹爪未在时限内回到 preopen {preopen_raw}")
            time.sleep(0.05)

        # --- 结果摘要（生产判据离线重放另行执行） ---
        def phase_rows(name: str) -> list[dict]:
            return [r for r in report["samples"] if r["phase"] == name]

        result: dict = {
            "stop_reason": stop_reason,
            "first_index_in_contact_gap_zone": next(
                (
                    r["index"]
                    for r in phase_rows("closing")
                    if r["gap_m"] is not None
                    and contact_gap_range[0] <= r["gap_m"] <= contact_gap_range[1]
                ),
                None,
            ),
        }
        for name in ("closing", "hold", "opening"):
            rows = phase_rows(name)
            if not rows:
                continue
            currents = [abs(float(r["current_ma"])) for r in rows]
            result[name] = {
                "frames": len(rows),
                "peak_abs_current_ma": max(currents),
                f"max_{window_samples}frame_mean_abs_current_ma": rolling_mean_max(
                    currents, window_samples
                ),
                "max_abs_tracking_error_raw": max(abs(r["tracking_error_raw"]) for r in rows),
                "final_feedback_raw": rows[-1]["position_raw"],
                "final_gap_m": rows[-1]["gap_m"],
            }
        hold_rows = phase_rows("hold")
        if hold_rows:
            result["hold_feedback_drift_raw"] = hold_rows[-1]["position_raw"] - hold_rows[0]["position_raw"]
        report["result"] = result
        report["status"] = (
            "PASS_PENDING_OPERATOR_OBSERVATION"
            if stop_reason in ("FREEZE_RAMP", "FREEZE_STALL", "FREEZE_LAG", "EMPTY_REACHED")
            else "NO_CONTACT_PENDING_OPERATOR_OBSERVATION"
        )
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL" if protocol is not None else "FAIL_PRE_ACCESS"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report.setdefault("result", {})["stop_reason"] = stop_reason
    finally:
        if protocol is not None:
            # 夹爪先卸力，再处理其余五轴；任何失败都只重试卸力，不重试运动。
            io_mod.TRANSACTION_TIMEOUT_S = CLEANUP_TIMEOUT_S
            for axis in [jaw] + [a for a in mapping.axes if a.servo_id != jaw.servo_id]:
                try:
                    _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                    time.sleep(0.03)
                except BaseException as exc:
                    report.setdefault("cleanup_errors", []).append(
                        f"disable {axis.name}: {type(exc).__name__}: {exc}"
                    )
                    try:
                        _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                    except BaseException as retry:
                        report.setdefault("cleanup_errors", []).append(
                            f"disable retry {axis.name}: {type(retry).__name__}: {retry}"
                        )
            for reg, value in originals.items():
                try:
                    _write_raw(protocol, mapping, jaw, reg, value)
                    time.sleep(0.03)
                    if _read_raw(protocol, mapping, jaw, reg) != value:
                        raise RuntimeError("SRAM 恢复读回不一致")
                except BaseException as exc:
                    report.setdefault("cleanup_errors", []).append(
                        f"restore {reg}: {type(exc).__name__}: {exc}"
                    )
            try:
                report["torque_enabled_after"] = [
                    _read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes
                ]
                report["sram_settings_after"] = {
                    reg: _read_raw(protocol, mapping, jaw, reg) for reg in originals
                }
                report["gripper_position_after_torque_off_raw"] = _read_raw(
                    protocol, mapping, jaw, "Present_Position"
                )
                for post_index in range(5):
                    row = sample_block("post_torque_off", post_index + 1, None, None)
                    report["post_torque_off_samples"].append(row)
                    if post_index < 4:
                        time.sleep(0.05)
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(
                    f"final readback: {type(exc).__name__}: {exc}"
                )
            if report.get("torque_enabled_after") != [0] * len(MOTOR_NAMES) or report.get("cleanup_errors"):
                report["status"] = "FAIL_CLEANUP_UNCONFIRMED"
        if transport is not None:
            transport.close()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary_keys = (
        "status",
        "stop_reason",
        "freeze",
        "result",
        "failure",
        "cleanup_errors",
        "torque_enabled_after",
        "sram_settings_after",
        "gripper_motion_commands_sent",
        "gripper_position_after_torque_off_raw",
    )
    print(json.dumps({k: report.get(k) for k in summary_keys if k in report}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in ("PASS_PENDING_OPERATOR_OBSERVATION", "NO_CONTACT_PENDING_OPERATOR_OBSERVATION") else 1


if __name__ == "__main__":
    raise SystemExit(main())
