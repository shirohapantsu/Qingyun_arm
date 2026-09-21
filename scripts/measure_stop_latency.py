#!/usr/bin/env python3
"""CAL-102 停止延迟实测：空爪生产闭合中异步注入停止，5 次取最坏延迟。

2026-09-18 操作者批准需求值 0.20s 与本方案。方法与生产停止链一致：
每 tick 提交前检查停止标志（executor.check_stop 语义）；后台线程在随机
1.2~1.8s 时刻置位（threading.Timer，进程内）。延迟定义：
  command_halt_ms = 主循环观测到标志（此后不再有任何新目标）− T0置位时刻；
物理证据另记 velocity_settle_ms = |速度反馈|<1 deg/s 首次时刻 − T0。
5 次独立注入，逐次含停止后 0.5s 只读反馈证据段；轮间以受控重新提交
张回 preopen（标定台架语义）。电流停止线 = profile hard_current_ma（与
生产 monitor 镜像），跟踪 >150raw、闭合注入前预算 3s、tick 迟到超限即
中止；任何失败路径先卸力。仅 6 号运动；零 EEPROM/profile 写；不派生。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import threading
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

APPROVAL = "批准，CAL-102 需求 0.20s 确认并同意空爪停测方案"
ITERATIONS = 5
TRIGGER_RANGE_S = (1.2, 1.8)
PRE_TRIGGER_BUDGET_S = 3.0   # 注入时刻上限（空爪全程≈2.5s到空夹门）
EVIDENCE_S = 0.5
ABORT_DRIFT_RAW = 40
ABORT_TRACKING_RAW = 150
OPEN_BACK_BUDGET_S = 5.0
PREPOSITION_TOLERANCE_RAW = 8
PREPOSITION_TIMEOUT_S = 15.0
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


def latency_stats(latencies_ms: list[float]) -> dict:
    return {
        "n": len(latencies_ms),
        "max_ms": max(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "all_ms": [round(v, 3) for v in latencies_ms],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--base-mount", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit("拒绝覆盖既有证据")
    if args.operator_confirmation != APPROVAL:
        raise SystemExit(f"缺少精确确认：{APPROVAL}")

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
    close_speed = float(g["close_speed_pct_s"])
    open_speed = float(g["open_speed_pct_s"])
    empty_gate_pct = float(g["empty_closed_pct"]) + float(g["empty_tol_pct"])
    abort_current_ma = float(g["hard_current_ma"])
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
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    trigger_delays = [rng.uniform(*TRIGGER_RANGE_S) for _ in range(ITERATIONS)]

    report: dict = {
        "schema_version": 2,
        "parameter_id": "CAL-102",
        "test": "gripper_stop_latency_5x",
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "operator_confirmation": args.operator_confirmation,
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256_file(profile_path),
        "motor_calibration_sha256": sha256_file(profile_path.parent / "motor_calibration.json"),
        "environment": {
            "power": args.power, "base_mount": args.base_mount,
            "load": "empty_gripper", "seed": args.seed,
        },
        "config": {
            "iterations": ITERATIONS,
            "trigger_range_s": list(TRIGGER_RANGE_S),
            "trigger_delays_s": [round(v, 4) for v in trigger_delays],
            "preopen_pct": preopen_pct,
            "close_speed_pct_s": close_speed,
            "open_speed_pct_s": open_speed,
            "empty_gate_pct": empty_gate_pct,
            "abort_current_ma_from_profile_hard": abort_current_ma,
            "evidence_s": EVIDENCE_S,
            "fps": fps,
            "tick_s": tick_s,
            "latency_definition": "command_halt=置位→主循环tick前检查观测（此后零新目标）；velocity_settle=置位→|速度|<1deg/s首次",
        },
        "eeprom_written": False,
        "profile_written": False,
        "joint_servo_motion_commands_sent": 0,
        "gripper_motion_commands_sent": 0,
        "iterations": [],
    }

    transport = None
    protocol: StsProtocol | None = None
    originals: dict[str, int] = {}
    stop_flag = threading.Event()

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
            "phase": phase, "index": index, "monotonic_ns": time.monotonic_ns(),
            "command_raw": command_raw, "command_pct": command_pct,
            "position_raw": pos, "gripper_pct": pct, "gap_m": gap,
            "velocity_register_raw": vel,
            "velocity_deg_s": mapping.raw_velocity_to_deg_s("gripper", vel),
            "current_ma": mapping.raw_current_to_ma("gripper", cur),
            "tracking_error_raw": (pos - command_raw) if command_raw is not None else None,
        }

    try:
        # --- 只读前置 ---
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
        transport = SerialTransport(str(stable_port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        io_mod.TRANSACTION_TIMEOUT_S = PREFLIGHT_TIMEOUT_S
        for axis in mapping.axes:
            level = _read_raw(protocol, mapping, axis, "Response_Status_Level")
            protocol.set_write_response(axis.servo_id, expects_ack=(level == 0))
        preflight = {
            key: [_read_raw(protocol, mapping, a, reg) for a in mapping.axes]
            for key, reg in (("ids", "ID"), ("models", "Model_Number"), ("torques", "Torque_Enable"))
        }
        report["readonly_preflight"] = preflight
        if preflight["ids"] != list(range(1, len(MOTOR_NAMES) + 1)):
            raise RuntimeError(f"ID 不匹配：{preflight['ids']}")
        if preflight["models"] != [777] * len(MOTOR_NAMES):
            raise RuntimeError(f"型号不匹配：{preflight['models']}")
        if preflight["torques"] != [0] * len(MOTOR_NAMES):
            raise RuntimeError(f"六轴必须全部卸力：{preflight['torques']}")
        for axis in mapping.axes:
            lo = _read_raw(protocol, mapping, axis, "Min_Position_Limit")
            hi = _read_raw(protocol, mapping, axis, "Max_Position_Limit")
            off = mapping.decode_signed(axis, "Homing_Offset", _read_raw(protocol, mapping, axis, "Homing_Offset"))
            if (lo, hi, off) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f"{axis.name} EEPROM 与权威校准不一致；未做任何写")
        controllers = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in CONTROLLER_REGISTERS}
        if controllers != EXPECTED_CONTROLLERS:
            raise RuntimeError(f"夹爪控制器字段不一致：{controllers}")
        frames = []
        for index in range(10):
            frames.append([_read_raw(protocol, mapping, a, "Present_Position") for a in mapping.axes])
            if index < 9:
                time.sleep(0.05)
        spans = [max(f[i] for f in frames) - min(f[i] for f in frames) for i in range(len(MOTOR_NAMES))]
        if any(s > 2 for s in spans):
            raise RuntimeError(f"六轴静态基线不满足：span={spans}")
        start_pos = frames[-1][5]
        if not (closed_raw - 5 <= start_pos <= open_raw + 5):
            raise RuntimeError(f"起点{start_pos}越出标定端点")
        report["start_position_raw"] = start_pos
        originals = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in SRAM_REGISTERS}
        report["original_sram_settings"] = originals.copy()
        if originals != EXPECTED_RUNTIME_SRAM:
            raise RuntimeError(f"运行期 SRAM 非预期：{originals}")

        # --- 上力并温和定位到 preopen（一次性准备）---
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
        settled: list[int] = []
        deadline = time.monotonic() + PREPOSITION_TIMEOUT_S
        while True:
            row = sample_block("preposition", len(settled) + 1, preopen_raw, preopen_pct)
            if settled and settled[-1] == row["position_raw"]:
                settled.append(row["position_raw"])
            else:
                settled = [row["position_raw"]]
            if len(settled) >= 3 and abs(row["position_raw"] - preopen_raw) <= PREPOSITION_TOLERANCE_RAW:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("未能定位到 preopen")
            time.sleep(0.05)
        for reg, value in originals.items():
            _write_raw(protocol, mapping, jaw, reg, value)
            time.sleep(0.03)
            if _read_raw(protocol, mapping, jaw, reg) != value:
                raise RuntimeError(f"正式段前 {reg} 恢复读回不一致")

        # --- 5 次注入 ---
        for iteration in range(1, ITERATIONS + 1):
            stop_flag.clear()
            t0_holder: list[int] = []

            def fire() -> None:
                t0_holder.append(time.monotonic_ns())
                stop_flag.set()

            delay = trigger_delays[iteration - 1]
            timer = threading.Timer(delay, fire)
            entry: dict = {"iteration": iteration, "trigger_delay_s": round(delay, 4),
                           "closing_samples": []}
            base = time.monotonic()
            timer.start()
            cmd_pct = preopen_pct
            index = 0
            halted_ns: int | None = None
            t0_ns: int | None = None
            last_goal_ns: int | None = None
            last_goal_raw: int | None = None
            try:
                while True:
                    index += 1
                    due = base + (index - 1) * tick_s
                    now = time.monotonic()
                    if now < due:
                        time.sleep(due - now)
                    if time.monotonic() - due > max_tick_lateness_s:
                        raise RuntimeError(f"第{iteration}次闭合第{index}步迟到超限")
                    if stop_flag.is_set():
                        # 检查点语义：本 tick 不再提交任何新目标
                        halted_ns = time.monotonic_ns()
                        t0_ns = t0_holder[0]
                        break
                    cmd_pct = max(0.0, cmd_pct - close_speed / fps)
                    cmd_raw = pct_to_raw(closed_raw, open_raw, cmd_pct)
                    _sync_goal(protocol, mapping, jaw, cmd_raw)
                    report["gripper_motion_commands_sent"] += 1
                    last_goal_ns = time.monotonic_ns()
                    last_goal_raw = cmd_raw
                    sample_due = base + index * tick_s
                    remaining = sample_due - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    row = sample_block("closing", index, cmd_raw, cmd_pct)
                    entry["closing_samples"].append(row)
                    if abs(row["current_ma"]) > abort_current_ma:
                        raise RuntimeError(f"第{iteration}次闭合电流超限：{row['current_ma']}mA")
                    if abs(row["tracking_error_raw"]) > ABORT_TRACKING_RAW:
                        raise RuntimeError(f"第{iteration}次跟踪差超限")
                    if row["gripper_pct"] <= empty_gate_pct:
                        raise RuntimeError(f"第{iteration}次在注入前到达空夹门（触发窗失效）")
                    if time.monotonic() - base > PRE_TRIGGER_BUDGET_S + delay:
                        raise RuntimeError(f"第{iteration}次注入后预算耗尽仍未见停止标志")
            finally:
                timer.cancel()
            entry["last_goal_raw"] = last_goal_raw
            entry["last_goal_monotonic_ns"] = last_goal_ns
            entry["halt_monotonic_ns"] = halted_ns
            entry["command_halt_ms"] = (halted_ns - t0_ns) / 1e6

            # 停止后 0.5s 只读证据段（零提交）
            evidence: list[dict] = []
            ev_base = time.monotonic()
            velocity_settle_ms: float | None = None
            drift_raw_max = 0
            first_pos: int | None = None
            ev_index = 0
            while time.monotonic() - ev_base < EVIDENCE_S:
                ev_index += 1
                row = sample_block("post_stop_evidence", ev_index, None, None)
                evidence.append(row)
                if first_pos is None:
                    first_pos = row["position_raw"]
                drift_raw_max = max(drift_raw_max, abs(row["position_raw"] - first_pos))
                if (velocity_settle_ms is None
                        and abs(row["velocity_deg_s"]) < 1.0):
                    velocity_settle_ms = (row["monotonic_ns"] - t0_ns) / 1e6
                time.sleep(max(0.0, tick_s - 0.004))
            entry["post_stop_evidence"] = evidence
            entry["post_stop_drift_raw_max"] = drift_raw_max
            entry["velocity_settle_ms"] = velocity_settle_ms
            entry["drift_gate_raw"] = ABORT_DRIFT_RAW
            report["iterations"].append(entry)
            if drift_raw_max > ABORT_DRIFT_RAW:
                raise RuntimeError(
                    f"第{iteration}次停止后位置漂移{drift_raw_max}raw>40——超出在途目标完成量")
            if velocity_settle_ms is None:
                raise RuntimeError(
                    f"第{iteration}次停止后{EVIDENCE_S}s内速度未归零——停止链异常")

            # 受控恢复：张回 preopen 供下一轮（不计入延迟统计）
            open_base = time.monotonic()
            open_deadline = open_base + OPEN_BACK_BUDGET_S
            oi = 0
            while cmd_pct < preopen_pct:
                oi += 1
                due = open_base + (oi - 1) * tick_s
                now = time.monotonic()
                if now < due:
                    time.sleep(due - now)
                cmd_pct = min(preopen_pct, cmd_pct + open_speed / fps)
                cmd_raw = pct_to_raw(closed_raw, open_raw, cmd_pct)
                _sync_goal(protocol, mapping, jaw, cmd_raw)
                report["gripper_motion_commands_sent"] += 1
                sample_due = open_base + oi * tick_s
                remaining = sample_due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                row = sample_block("recover_open", oi, cmd_raw, cmd_pct)
                if abs(row["current_ma"]) > abort_current_ma:
                    raise RuntimeError(f"第{iteration}次恢复张开电流超限")
                if time.monotonic() >= open_deadline:
                    raise RuntimeError(f"第{iteration}次恢复张开预算耗尽")
            time.sleep(0.3)

        lat = [float(e["command_halt_ms"]) for e in report["iterations"]]
        vs = [float(e["velocity_settle_ms"]) for e in report["iterations"]
              if e["velocity_settle_ms"] is not None]
        report["result"] = {
            "command_halt": latency_stats(lat),
            "velocity_settle": latency_stats(vs) if vs else None,
            "limit_ms": float(raw_profile["acceptance"]["max_stop_latency_s"]) * 1000.0,
            "verdict": "PASS" if max(lat) <= float(raw_profile["acceptance"]["max_stop_latency_s"]) * 1000.0 else "FAIL",
        }
        report["status"] = "PASS_PENDING_OPERATOR_OBSERVATION"
    except BaseException as exc:
        report["status"] = "FAIL" if protocol is not None else "FAIL_PRE_ACCESS"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            io_mod.TRANSACTION_TIMEOUT_S = CLEANUP_TIMEOUT_S
            for axis in [jaw] + [a for a in mapping.axes if a.servo_id != jaw.servo_id]:
                try:
                    _write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                    time.sleep(0.03)
                except BaseException as exc:
                    report.setdefault("cleanup_errors", []).append(f"disable {axis.name}: {exc}")
            for reg, value in originals.items():
                try:
                    _write_raw(protocol, mapping, jaw, reg, value)
                    time.sleep(0.03)
                    if _read_raw(protocol, mapping, jaw, reg) != value:
                        raise RuntimeError("SRAM 恢复读回不一致")
                except BaseException as exc:
                    report.setdefault("cleanup_errors", []).append(f"restore {reg}: {exc}")
            try:
                report["torque_enabled_after"] = [
                    _read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes
                ]
                report["sram_settings_after"] = {reg: _read_raw(protocol, mapping, jaw, reg) for reg in originals}
                report["gripper_position_after_torque_off_raw"] = _read_raw(
                    protocol, mapping, jaw, "Present_Position")
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(f"final readback: {exc}")
            if report.get("torque_enabled_after") != [0] * len(MOTOR_NAMES) or report.get("cleanup_errors"):
                report["status"] = "FAIL_CLEANUP_UNCONFIRMED"
        if transport is not None:
            transport.close()
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    print(json.dumps({k: report.get(k) for k in
                      ("status", "failure", "result", "cleanup_errors", "torque_enabled_after",
                       "sram_settings_after", "gripper_motion_commands_sent",
                       "gripper_position_after_torque_off_raw")},
                     ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS_PENDING_OPERATOR_OBSERVATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
