#!/usr/bin/env python3
"""CAL-093~095 稳定持物单轮采集：生产接触判据 → 3.0s 反馈开度保持 → 55% 释放。

2026-09-18 操作者批准本方案第 1 轮。与 CAL-085 重测工具的区别：闭合段在线
复现生产判据（arm_control._close_until_contact：3 帧均值严格 > contact、
gap∈contact带、pct>空夹门、连续 contact_dwell_s 时刻判定），命中后按生产
_confirm_contact 语义冻结"确认时刻实际反馈开度"为重复目标，保持 3.0 s
（大于 hold_dwell 候选，观测漂移/电流衰减），再以生产 20%/s 张开至
release_pct=55（占位值，正式值待 CAL-081），静止后附加 1.0 s 落定观察窗。
仅 6 号运动；1~5 号全程卸力零目标；不写 profile/EEPROM；本脚本不派生
参数、不验收，数据离线分析。

诊断保护（非生产参数，不与运行值混同）：
  * 保持段 |电流|>hard_current_ma（动态读 profile；2026-09-18 起 150mA）、
    |Δpct|>10%、跟踪差>150raw 即停；
  * 闭合预算 5s、张开预算 5s、到位 3s、tick 迟到 > max_tick_lateness_s；
  * 任何 SRAM 读回不一致/通信异常立即停止并进入卸力恢复。
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

APPROVAL = "批准，CAL-093~095 稳定持物试验就绪"
APPROVAL_RETRY_R1 = "批准，CAL-093~095 稳定持物第1轮重测就绪"
APPROVAL_R1_A3 = "批准，CAL-093~095 稳定持物第1轮attempt3就绪"
APPROVAL_R3 = "批准，CAL-093~095 round3就绪"
APPROVALS = {APPROVAL, APPROVAL_RETRY_R1, APPROVAL_R1_A3, APPROVAL_R3}
# 经批准的固定试验预算（区别于运行参数本身）。
HOLD_OBS_S = 3.0
RELEASE_DWELL_OBS_S = 1.0
# 电流停止线 = profile gripper.hard_current_ma（与生产 executor.monitor 镜像，
# 2026-09-18 随 CAL-086 从 78 上调为 150——操作者消除闭合确认竞态的策略决定）。
ABORT_DRIFT_PCT = 10.0           # 诊断门；生产 slip 判据本身是受测对象
ABORT_TRACKING_RAW = 150
CLOSING_BUDGET_S = 5.0
OPENING_BUDGET_S = 5.0
SETTLE_TIMEOUT_S = 3.0
DRIFT_SETTLE_IGNORE_S = 0.1      # 漂移统计前排除的沉降段
PREFLIGHT_TIMEOUT_S = 1.0
MOTION_TIMEOUT_S = 0.25
CLEANUP_TIMEOUT_S = 0.5
PREPOSITION_TOLERANCE_RAW = 8
PREPOSITION_TIMEOUT_S = 15.0
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--ambient-temperature-c", type=float, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("拒绝覆盖既有证据")
    if args.operator_confirmation not in APPROVALS:
        raise SystemExit(f"缺少精确确认，须为：{' / '.join(sorted(APPROVALS))}")
    if args.round < 1:
        raise SystemExit("轮次必须 ≥1")

    profile_path = args.profile.resolve(strict=True)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = load_calibration_profile(profile_path)
    if profile.joints is None:
        raise SystemExit("关节标定未完成，拒绝访问")

    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    jaw = mapping.axis("gripper")
    g = raw_profile["gripper"]
    preopen_pct = float(g["preopen_pct"])
    release_pct = float(g["release_pct"])
    close_speed = float(g["close_speed_pct_s"])
    open_speed = float(g["open_speed_pct_s"])
    contact_current_ma = float(g["contact_current_ma"])
    window_samples = int(g["window_samples"])
    contact_dwell_s = float(g["contact_dwell_s"])
    empty_gate_pct = float(g["empty_closed_pct"]) + float(g["empty_tol_pct"])
    contact_gap_range = [float(v) for v in g["contact_gap_range_m"]]
    gap_tables = SimpleNamespace(
        gap_table=[
            SimpleNamespace(gripper_pct=float(e["gripper_pct"]), gap_m=float(e["gap_m"]))
            for e in g["gap_table"]
        ]
    )
    closed_raw = int(profile.motor.gripper_closed_raw)
    open_raw = int(profile.motor.gripper_open_raw)
    preopen_raw = pct_to_raw(closed_raw, open_raw, preopen_pct)
    fps = float(raw_profile["timing"]["fps"])
    tick_s = 1.0 / fps
    max_tick_lateness_s = float(raw_profile["timing"]["max_tick_lateness_s"])
    if not 0.0 < empty_gate_pct < preopen_pct < release_pct <= 100.0:
        raise SystemExit(f"开度关系异常：gate={empty_gate_pct} preopen={preopen_pct} release={release_pct}")
    abort_current_ma = float(g["hard_current_ma"])  # 与生产 monitor 同值同语义
    if not jaw.range_min <= preopen_raw <= jaw.range_max:
        raise SystemExit(f"preopen raw={preopen_raw} 越出 EEPROM 范围")

    report: dict = {
        "schema_version": 2,
        "parameter_id": "CAL-093~095",
        "test": f"production_contact_hold_release_round{args.round}",
        "round": args.round,
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
            "load": "independently supported plastic fixture, 20mm contact band",
        },
        "criteria_from_profile": {
            "preopen_pct": preopen_pct,
            "preopen_raw": preopen_raw,
            "release_pct": release_pct,
            "close_speed_pct_s": close_speed,
            "open_speed_pct_s": open_speed,
            "contact_current_ma": contact_current_ma,
            "window_samples": window_samples,
            "contact_dwell_s": contact_dwell_s,
            "empty_gate_pct": empty_gate_pct,
            "contact_gap_range_m": contact_gap_range,
            "hard_current_ma": float(g["hard_current_ma"]),
            "fps": fps,
            "tick_s": tick_s,
            "max_tick_lateness_s": max_tick_lateness_s,
            "closed_raw": closed_raw,
            "open_raw": open_raw,
        },
        "diagnostic_bounds": {
            "hold_abort_current_ma": abort_current_ma,
            "hold_abort_drift_pct": ABORT_DRIFT_PCT,
            "abort_tracking_raw": ABORT_TRACKING_RAW,
            "hold_obs_s": HOLD_OBS_S,
            "release_dwell_obs_s": RELEASE_DWELL_OBS_S,
            "closing_budget_s": CLOSING_BUDGET_S,
            "opening_budget_s": OPENING_BUDGET_S,
            "drift_settle_ignore_s": DRIFT_SETTLE_IGNORE_S,
            "note": "全部为采集诊断保护；hold_obs 3.0s 是批准的观测预算，非 CAL-093 结论",
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
    originals: dict[str, int] = {}
    failure: BaseException | None = None
    outcome: str | None = None

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
        return {
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

    try:
        # --- 阶段 0：只读前置（与 CAL-085 重测工具一致）---
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
        actual_firmware = list(zip(preflight["firmware_major"], preflight["firmware_minor"]))
        if actual_firmware != expected_firmware:
            raise RuntimeError(f"固件不匹配：{actual_firmware}，期望 {expected_firmware}")
        if preflight["torques"] != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"试验前六轴必须全部卸力：{preflight['torques']}")

        report["readonly_eeprom"] = {}
        for axis in mapping.axes:
            lo = _read_raw(protocol, mapping, axis, "Min_Position_Limit")
            hi = _read_raw(protocol, mapping, axis, "Max_Position_Limit")
            offset = mapping.decode_signed(axis, "Homing_Offset", _read_raw(protocol, mapping, axis, "Homing_Offset"))
            report["readonly_eeprom"][axis.name] = {"range_min": lo, "range_max": hi, "homing_offset": offset}
            if (lo, hi, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f"{axis.name} EEPROM 与权威校准不一致；未做任何写")

        controllers = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in CONTROLLER_REGISTERS}
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
        release_raw_check = pct_to_raw(closed_raw, open_raw, release_pct)
        START_MIN_RAW = preopen_raw - 15   # 覆盖滑行/死区停靠（CAL-085带物轮停靠2019~2021）；压物位≤1858，裕度≥152raw
        START_MAX_RAW = release_raw_check + 8
        near_preopen = START_MIN_RAW <= start_pos <= START_MAX_RAW
        report["start_position_gate"] = {
            "start_raw": start_pos,
            "preopen_raw": preopen_raw,
            "release_park_raw": release_raw_check,
            "allowed_range_raw": [START_MIN_RAW, START_MAX_RAW],
            "passed": near_preopen,
        }
        if not near_preopen:
            raise RuntimeError(
                f"夹爪起点{start_pos}不在[{START_MIN_RAW},{START_MAX_RAW}]内"
                "（低于下限疑似手指压在假果上），未上力、零目标"
            )

        originals = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in SRAM_REGISTERS}
        report["original_sram_settings"] = originals.copy()
        if originals != EXPECTED_RUNTIME_SRAM:
            raise RuntimeError(f"运行期 SRAM 非预期：{originals}")

        # --- 定位段 ---
        io_mod.TRANSACTION_TIMEOUT_S = MOTION_TIMEOUT_S
        for reg, value in PREPOSITION_SRAM.items():
            _write_raw(protocol, mapping, jaw, reg, value)
        _sync_goal(protocol, mapping, jaw, start_pos)
        _write_raw(protocol, mapping, jaw, "Torque_Enable", 1)
        time.sleep(0.05)
        if _read_raw(protocol, mapping, jaw, "Torque_Enable") != 1:
            raise RuntimeError("夹爪上力未生效")
        _sync_goal(protocol, mapping, jaw, preopen_raw)
        report["gripper_motion_commands_sent"] += 1
        deadline = time.monotonic() + PREPOSITION_TIMEOUT_S
        settled: list[int] = []
        while True:
            row = sample_block("preposition", len(settled) + 1, preopen_raw, preopen_pct)
            report["preposition_samples"].append(row)
            if abs(row["current_ma"]) > abort_current_ma:
                raise RuntimeError(f"定位段夹爪电流超限：{row['current_ma']}mA")
            if settled and settled[-1] == row["position_raw"]:
                settled.append(row["position_raw"])
            else:
                settled = [row["position_raw"]]
            if len(settled) >= 3 and abs(row["position_raw"] - preopen_raw) <= PREPOSITION_TOLERANCE_RAW:
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

        # --- 闭合段：在线生产判据（复现 _close_until_contact）---
        base = time.monotonic()
        closing_deadline = base + CLOSING_BUDGET_S
        cmd_pct = preopen_pct
        index = 0
        currents: list[float] = []
        contact_since: float | None = None
        hold_pct: float | None = None
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
            now = time.monotonic()

            if abs(row["current_ma"]) > abort_current_ma:
                raise RuntimeError(f"闭合段夹爪电流超限：{row['current_ma']}mA")
            if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                raise RuntimeError(f"闭合段跟踪差超限：{row['tracking_error_raw']}raw")

            currents.append(abs(row["current_ma"]))
            if len(currents) > window_samples:
                currents.pop(0)
            filtered = statistics.fmean(currents) if len(currents) >= window_samples else 0.0
            in_gap = bool(
                row["gap_m"] is not None
                and contact_gap_range[0] <= row["gap_m"] <= contact_gap_range[1]
            )
            above_empty = row["gripper_pct"] > empty_gate_pct
            if filtered > contact_current_ma and in_gap and above_empty:
                if contact_since is None:
                    contact_since = now
                elif now - contact_since >= contact_dwell_s:
                    hold_pct = row["gripper_pct"]
                    report["contact_confirmation"] = {
                        "index": index,
                        "confirm_elapsed_s": now - base,
                        "first_qualify_elapsed_s": contact_since - base,
                        "filtered_ma": filtered,
                        "feedback_raw": row["position_raw"],
                        "feedback_pct": hold_pct,
                        "gap_m": row["gap_m"],
                        "current_ma": row["current_ma"],
                        "command_raw": cmd_raw,
                        "command_pct": cmd_pct,
                    }
                    outcome = "CONTACT_CONFIRMED"
                    break
            else:
                contact_since = None
            if row["gripper_pct"] <= empty_gate_pct:
                outcome = "NO_CONTACT_EMPTY_GATE"
                break
            if time.monotonic() >= closing_deadline:
                raise RuntimeError("闭合段预算耗尽仍未判定")

        # --- 保持段：冻结确认时刻实际反馈开度 3.0s（批准观测预算）---
        if hold_pct is not None:
            hold_cmd_raw = pct_to_raw(closed_raw, open_raw, hold_pct)
            hold_base = time.monotonic()
            hold_deadline = hold_base + HOLD_OBS_S
            hold_index = 0
            while True:
                hold_index += 1
                due = hold_base + (hold_index - 1) * tick_s
                now = time.monotonic()
                if now < due:
                    time.sleep(due - now)
                if time.monotonic() - due > max_tick_lateness_s:
                    raise RuntimeError(f"保持段第{hold_index}帧迟到超过时序上限")
                _sync_goal(protocol, mapping, jaw, hold_cmd_raw)
                report["gripper_motion_commands_sent"] += 1
                sample_due = hold_base + hold_index * tick_s
                remaining = sample_due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                row = sample_block("hold", hold_index, hold_cmd_raw, hold_pct)
                report["samples"].append(row)
                if abs(row["current_ma"]) > abort_current_ma:
                    raise RuntimeError(f"保持段夹爪电流超限：{row['current_ma']}mA")
                if abs(row["gripper_pct"] - hold_pct) > ABORT_DRIFT_PCT:
                    raise RuntimeError(f"保持段开度漂移超限：{row['gripper_pct'] - hold_pct:.3f}%")
                if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                    raise RuntimeError(f"保持段跟踪差超限：{row['tracking_error_raw']}raw")
                if time.monotonic() >= hold_deadline:
                    report["hold_observed_s"] = time.monotonic() - hold_base
                    break

        # --- 释放段：20%/s 张开到 release_pct=55，静止后附 1.0s 落定观察 ---
        open_base = time.monotonic()
        open_deadline = open_base + OPENING_BUDGET_S
        open_index = 0
        while cmd_pct < release_pct:
            open_index += 1
            due = open_base + (open_index - 1) * tick_s
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            if time.monotonic() - due > max_tick_lateness_s:
                raise RuntimeError(f"张开段第{open_index}步迟到超过时序上限")
            cmd_pct = min(release_pct, cmd_pct + open_speed / fps)
            cmd_raw = pct_to_raw(closed_raw, open_raw, cmd_pct)
            _sync_goal(protocol, mapping, jaw, cmd_raw)
            report["gripper_motion_commands_sent"] += 1
            sample_due = open_base + open_index * tick_s
            remaining = sample_due - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            row = sample_block("opening", open_index, cmd_raw, cmd_pct)
            report["samples"].append(row)
            if abs(row["current_ma"]) > abort_current_ma:
                raise RuntimeError(f"张开段夹爪电流超限：{row['current_ma']}mA")
            if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                raise RuntimeError(f"张开段跟踪差超限：{row['tracking_error_raw']}raw")
            if time.monotonic() >= open_deadline:
                raise RuntimeError("张开段预算耗尽")

        release_target_raw = pct_to_raw(closed_raw, open_raw, release_pct)
        settle_deadline = time.monotonic() + SETTLE_TIMEOUT_S
        settled = []
        while True:
            row = sample_block("release_settle", len(settled) + 1, release_target_raw, release_pct)
            report["samples"].append(row)
            if abs(row["current_ma"]) > abort_current_ma:
                raise RuntimeError(f"到位段夹爪电流超限：{row['current_ma']}mA")
            if settled and settled[-1] == row["position_raw"]:
                settled.append(row["position_raw"])
            else:
                settled = [row["position_raw"]]
            if len(settled) >= 3:
                report["release_settle_s"] = (time.monotonic() - open_base)
                break
            if time.monotonic() >= settle_deadline:
                raise RuntimeError(f"夹爪未在时限内静止于 release {release_pct}%")
            time.sleep(0.05)

        dwell_base = time.monotonic()
        dwell_index = 0
        while time.monotonic() - dwell_base < RELEASE_DWELL_OBS_S:
            dwell_index += 1
            _sync_goal(protocol, mapping, jaw, release_target_raw)
            report["gripper_motion_commands_sent"] += 1
            row = sample_block("release_dwell", dwell_index, release_target_raw, release_pct)
            report["samples"].append(row)
            if abs(row["current_ma"]) > abort_current_ma:
                raise RuntimeError(f"落定观察段电流超限：{row['current_ma']}mA")
            time.sleep(max(0.0, tick_s - 0.005))

        # --- 结果摘要 ---
        def phase_rows(name: str) -> list[dict]:
            return [r for r in report["samples"] if r["phase"] == name]

        result: dict = {"stop_reason": outcome}
        for name in ("closing", "hold", "opening", "release_settle", "release_dwell"):
            rows = phase_rows(name)
            if not rows:
                continue
            currents_ = [abs(float(r["current_ma"])) for r in rows]
            entry = {
                "frames": len(rows),
                "elapsed_s": (rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]) / 1e9,
                "peak_abs_current_ma": max(currents_),
                "median_abs_current_ma": statistics.median(currents_),
                "max_abs_tracking_error_raw": max(
                    abs(r["tracking_error_raw"]) for r in rows if r["tracking_error_raw"] is not None
                ),
                "final_feedback_raw": rows[-1]["position_raw"],
            }
            result[name] = entry
        hold_rows = phase_rows("hold")
        if hold_rows:
            first_ns = hold_rows[0]["monotonic_ns"]
            stat_rows = [
                r for r in hold_rows
                if (r["monotonic_ns"] - first_ns) / 1e9 >= DRIFT_SETTLE_IGNORE_S
            ]
            currents_h = [abs(float(r["current_ma"])) for r in hold_rows]
            drifts = [abs(float(r["gripper_pct"]) - hold_pct) for r in (stat_rows or hold_rows)]
            pos_first = (stat_rows or hold_rows)[0]["position_raw"]
            result["hold"] = {
                **result.get("hold", {}),
                "drift_stats": {
                    "ignore_first_s": DRIFT_SETTLE_IGNORE_S,
                    "max_abs_drift_pct": max(drifts),
                    "final_drift_pct": float(hold_rows[-1]["gripper_pct"]) - hold_pct,
                    "max_abs_position_drift_raw": max(
                        abs(r["position_raw"] - pos_first) for r in (stat_rows or hold_rows)
                    ),
                },
                "current_quarter_s_max_ma": [
                    max(currents_h[i:i + max(1, int(fps / 4))])
                    for i in range(0, len(currents_h), max(1, int(fps / 4)))
                ],
            }
        report["result"] = result
        report["status"] = (
            "PASS_PENDING_OPERATOR_OBSERVATION"
            if outcome == "CONTACT_CONFIRMED"
            else "NO_CONTACT_PENDING_OPERATOR_OBSERVATION"
        )
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL" if protocol is not None else "FAIL_PRE_ACCESS"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report.setdefault("result", {})["stop_reason"] = outcome
    finally:
        if protocol is not None:
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
                report["sram_settings_after"] = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in originals}
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
        "status", "failure", "contact_confirmation", "hold_observed_s",
        "release_settle_s", "result", "cleanup_errors", "torque_enabled_after",
        "sram_settings_after", "gripper_motion_commands_sent",
        "gripper_position_after_torque_off_raw",
    )
    print(json.dumps({k: report.get(k) for k in summary_keys if k in report}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in ("PASS_PENDING_OPERATOR_OBSERVATION", "NO_CONTACT_PENDING_OPERATOR_OBSERVATION") else 1


if __name__ == "__main__":
    raise SystemExit(main())
