#!/usr/bin/env python3
"""CAL-089/090: exactly one approved empty-close diagnostic cycle.

Trial20mA/10raw guards are not calibrated force or production safety. No arm
goals, endpoint/EEPROM/profile changes, or automatic retry. Repeats2/3 require
the explicit two-cycle approval and each previous successful unload/restore.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import load_calibration_profile
from qingyun.grabbing.motor_control import MotorMapping, SerialTransport, StsProtocol
from scripts import move_gripper_raw as io
from scripts.calibrate_gripper_empty_current import SETTINGS, segment_targets
from scripts.calibrate_gripper_open_speed import _sample
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY
from scripts.read_servo_eeprom import sha256_file, usb_identity

CONFIRMATION = "批准，CAL-089/090空爪首轮就绪"
BATCH_CONFIRMATION = "第二轮批准，第三轮无需申请直接执行即可"
POSE_CHANGE_CONFIRMATION = "是我主动调整的，无视即可"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    session = json.loads(args.session.read_text(encoding="utf-8"))
    plan = session if args.repeat == 1 else session[f"proposed_repeat{args.repeat}"]
    if args.attempt == 2 and args.repeat != 2:
        raise SystemExit("Only the explicitly resumed repeat2 may have attempt2")
    confirmation = CONFIRMATION if args.repeat == 1 else plan["required_confirmation"]
    if args.output.exists() or args.operator_confirmation != confirmation:
        raise SystemExit("Evidence exists or exact approval missing")
    pose_override = None
    if confirmation == POSE_CHANGE_CONFIRMATION:
        pose_override = session["operator_confirmed_arm2_pose_change"]
        if (pose_override["operator_confirmation"] != POSE_CHANGE_CONFIRMATION
                or pose_override["servo_id"] != 2 or pose_override["new_reference_raw"] != 921
                or pose_override["max_deviation_raw"] != 10):
            raise SystemExit("Intentional axis2 pose change approval differs")
    elif args.repeat > 1 and confirmation != BATCH_CONFIRMATION:
        raise SystemExit("Unknown repeat approval")
    expected = {
        "parameter_ids": ["CAL-089", "CAL-090"], "required_confirmation": CONFIRMATION,
        "servo_id": 6, "expected_initial_position_raw": 1806,
        "initial_position_max_deviation_raw": 10, "fixed_open_target_raw": 1569,
        "fixed_close_target_raw": 1499, "incremental_command_speed_pct_s": 1,
        "sample_fps": 30, "gripper_static_span_max_raw": 2, "arm_static_span_max_raw": 2,
        "arm_reference_max_deviation_raw": 10, "sram_settings": SETTINGS,
        "max_tracking_error_raw": 10, "feedback_bounds_raw": [1499, 2897],
        "all_phase_trial_abort_current_ma": 20, "trial_timeout_s": 30,
        "maximum_final_hold_s": .3, "final_hold_frames": 9,
        "final_hold_static_span_max_raw": 2, "accepted_final_feedback_range_raw": [1499, 1509],
        "first_approved_cycles": 1, "between_cycle_motion_authorized": False,
        "arm_joint_goal_commands": 0,
        "reference_arm_pose_report": "CAL-085_fixture_stronger_contact_attempt1_20260917.json",
        "proposed_report": "CAL-089_090_empty_close_repeat1_attempt1_20260917.json",
    }
    if args.repeat > 1:
        approval = session["repeat2_and_repeat3_approval"]
        if approval["operator_confirmation"] != BATCH_CONFIRMATION or approval["approved_repeats"] != [2, 3]:
            raise SystemExit("Missing explicit repeat2/3 approval")
        for key in ("parameter_ids", "first_approved_cycles", "between_cycle_motion_authorized", "proposed_report"):
            expected.pop(key)
        previous = next((r for r in reversed(session["powered_cycles"])
                         if r["repeat"] == args.repeat-1 and r["status"] in
                         ("CONFIRMED_NO_ABNORMALITY", "PASS_PENDING_OPERATOR_OBSERVATION")), None)
        if previous is None:
            raise SystemExit("Previous successful cycle missing")
        expected.update({
            "required_confirmation": confirmation, "repeat": args.repeat,
            "initial_position_max_deviation_raw": 5, "approved_cycle_count": 1,
            "reference_trial_report": previous["report"],
            "reference_arm_pose_report": previous["report"],
            "report": f"CAL-089_090_empty_close_repeat{args.repeat}_attempt{args.attempt}_20260917.json",
        })
        expected["expected_initial_position_raw"] = 1502 if args.repeat == 2 else plan["expected_initial_position_raw"]
        if not 1499 <= expected["expected_initial_position_raw"] <= 1509:
            raise SystemExit("Previous unloaded position outside approved closed-domain range")
        wanted_initial_range = [max(1499, expected["expected_initial_position_raw"] - 5),
                                min(2897, expected["expected_initial_position_raw"] + 5)]
        if plan["accepted_initial_feedback_range_raw"] != wanted_initial_range:
            raise SystemExit("Repeat initial range differs from approved tolerance/domain")
        if previous is None or previous["report"] != expected["reference_trial_report"]:
            raise SystemExit("Previous successful cycle missing")
        if args.repeat == 2 and previous.get("operator_observation") != "无异常":
            raise SystemExit("First-cycle observation missing")
        if args.repeat == 3 and previous["status"] != "PASS_PENDING_OPERATOR_OBSERVATION":
            raise SystemExit("Second cycle did not pass; third cycle not permitted")
    output_key = "proposed_report" if args.repeat == 1 else "report"
    if any(plan[k] != v for k, v in expected.items()) or args.output.name != plan[output_key]:
        raise SystemExit("Plan/output differs from exact first-cycle approval")
    expected_initial = expected["expected_initial_position_raw"]
    initial_tolerance = expected["initial_position_max_deviation_raw"]
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != plan["profile_sha256"]:
        raise SystemExit("Profile fingerprint changed")
    profile = load_calibration_profile(profile_path)
    if profile.joints is None:
        raise SystemExit("Missing joint mapping")
    port = Path(profile.motor.port)
    if str(port) != EXPECTED_STABLE_PORT:
        raise SystemExit("Unexpected stable serial path")
    resolved = port.resolve(strict=True)
    if not resolved.name.startswith("ttyACM") or usb_identity(resolved) != EXPECTED_USB_IDENTITY:
        raise SystemExit("USB/ttyACM identity mismatch")
    occupancy = subprocess.run(["fuser", str(port)], capture_output=True, text=True)
    if occupancy.returncode != 1 or occupancy.stdout.strip():
        raise SystemExit("Serial occupancy check failed or device busy")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if mapping.gripper_closed_raw != 1499 or mapping.gripper_open_raw != 2897:
        raise SystemExit("Calibrated endpoints changed")
    for target in (1569, 1499):
        if not gripper.range_min <= target <= gripper.range_max:
            raise SystemExit("Goal outside authoritative calibration range")
    source = json.loads((profile_path.parent / "reports" / plan["reference_arm_pose_report"]).read_text(encoding="utf-8"))
    if (source["profile_sha256"] != plan["profile_sha256"]
            or source["status"] != "PASS_PENDING_OPERATOR_OBSERVATION"
            or source["torque_enabled_after"] != [0] * 6
            or source["sram_settings_after"] != source["original_sram_settings"]
            or source["test_sram_settings"] != SETTINGS):
        raise SystemExit("Reference evidence changed")
    if args.repeat > 1 and (
            source["actual_command_path_raw"][1:] != [1569, 1499]
            or source["result"]["terminal_span_raw"] > 2
            or source["gripper_position_after_torque_off_raw"] != expected_initial
            or source["result"]["peak_abs_current_ma"] > 20
            or source["result"]["max_tracking_error_raw"] > 10
            or source["powered_duration_s"] >= 30
            or source["final_hold_observed_duration_s"] >= .3):
        raise SystemExit("Previous cycle data differs from successful bounded source")
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if raw_profile["timing"]["fps"] != 30:
        raise SystemExit("Sampling rate changed")
    io.TRANSACTION_TIMEOUT_S = .25
    report = {
        "schema_version": 1, "parameter_ids": ["CAL-089", "CAL-090"],
        "test": "single_powered_empty_close_endpoint_diagnostic", "repeat": args.repeat, "attempt": args.attempt,
        "status": "RUNNING", "started_at": datetime.now().astimezone().isoformat(),
        "profile_sha256": sha256_file(profile_path), "approved_plan": plan,
        "operator_confirmation": args.operator_confirmation,
        "stable_port": str(port), "resolved_port": str(resolved), "usb_identity": usb_identity(resolved),
        "environment": {"power_v": 12, "ambient_temperature_c": None,
                        "base_mount": "fixed to wooden board", "load": "empty gripper",
                        "temperature_note": "today not remeasured"},
        "joint_servo_motion_commands_sent": 0, "gripper_motion_commands_sent": 0,
        "eeprom_written": False, "profile_written": False, "samples": [],
        "transaction_timeout_s": .25,
    }
    transport = protocol = None
    originals = {}
    torque_may_be_on = sram_may_be_changed = False
    motion_start = failure = None

    def checked_sample(phase: str, command: int) -> dict:
        # Preserve raw/out-of-domain diagnostic evidence BEFORE strict mapping.
        row = _sample(protocol, mapping, gripper)
        row.update(phase=phase, command_raw=command, tracking_error_raw=command - row["position_raw"])
        report["samples"].append(row)
        if abs(row["current_ma"]) > 20:
            raise RuntimeError(f"20mA trial current stop: {row['current_ma']}mA")
        row["gripper_pct"] = mapping.raw_to_gripper_pct(row["position_raw"])
        if abs(row["tracking_error_raw"]) > 10:
            raise RuntimeError("Trial tracking difference exceeded10raw")
        if motion_start is not None and time.monotonic() - motion_start > 30:
            raise RuntimeError("30s powered trial timeout")
        return row

    try:
        transport = SerialTransport(str(port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        for axis in mapping.axes:
            level = io._read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError("Unexpected write response level")
            protocol.set_write_response(axis.servo_id, expects_ack=level == 0)
        ids = [io._read_raw(protocol, mapping, a, "ID") for a in mapping.axes]
        models = [io._read_raw(protocol, mapping, a, "Model_Number") for a in mapping.axes]
        torques = [io._read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes]
        report["readonly_preflight"] = {"ids": ids, "models": models, "torques": torques}
        if ids != [1, 2, 3, 4, 5, 6] or models != [777] * 6 or torques != [0] * 6:
            raise RuntimeError("ID/model/initial torque mismatch; no writes")
        report["readonly_eeprom"] = {}
        for axis in mapping.axes:
            minimum = io._read_raw(protocol, mapping, axis, "Min_Position_Limit")
            maximum = io._read_raw(protocol, mapping, axis, "Max_Position_Limit")
            offset = mapping.decode_signed(axis, "Homing_Offset", io._read_raw(protocol, mapping, axis, "Homing_Offset"))
            report["readonly_eeprom"][axis.name] = {"range_min": minimum, "range_max": maximum, "homing_offset": offset}
            if (minimum, maximum, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f"EEPROM mismatch {axis.name}; no writes")
        report["readonly_gripper_controller_settings"] = {
            r: io._read_raw(protocol, mapping, gripper, r)
            for r in ("P_Coefficient", "D_Coefficient", "I_Coefficient", "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone")
        }
        if report["readonly_gripper_controller_settings"] != source["readonly_gripper_controller_settings"]:
            raise RuntimeError("Controller differs from reference; no writes")
        before = [io._read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes]
        frames = []
        for index in range(10):
            frames.append([io._read_raw(protocol, mapping, a, "Present_Position") for a in mapping.axes])
            if index < 9:
                time.sleep(.05)
        after = [io._read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes]
        spans = [max(f[i] for f in frames) - min(f[i] for f in frames) for i in range(6)]
        reference_arm = source["fresh_readonly_baseline"]["raw_position_by_sample"][0][:5]
        report["arm_deviation_from_original_source_raw"] = [max(abs(f[i] - reference_arm[i]) for f in frames) for i in range(5)]
        if pose_override is not None:
            # Explicit operator-confirmed axis2 repositioning ONLY. Other axes,
            # static-span, valid-domain, and all jaw protection gates remain.
            reference_arm = reference_arm.copy()
            reference_arm[1] = 921
            report["operator_confirmed_arm2_pose_change"] = pose_override
        deviations = [max(abs(f[i] - reference_arm[i]) for f in frames) for i in range(5)]
        fresh = int(statistics.median(f[5] for f in frames))
        report["fresh_readonly_baseline"] = {
            "raw_position_by_sample": frames, "raw_position_static_span": spans,
            "gripper_raw": fresh, "torque_enabled_before": before, "torque_enabled_after": after,
        }
        report["arm_deviation_from_reference_raw"] = deviations
        # Jaw-only supported, torque-off arm pose consistency is not an arm
        # travel-limit acceptance. Keep any raw deviation visible; no arm goal
        # is authorized here. Arm motion requires its normal strict limit gates.
        report["readonly_arm_position_outside_eeprom_range"] = [
            {"servo_id": axis.servo_id, "axis": axis.name,
             "position_raw_range": [min(f[i] for f in frames), max(f[i] for f in frames)],
             "calibration_range_raw": [axis.range_min, axis.range_max]}
            for i, axis in enumerate(mapping.axes[:5])
            if any(not axis.range_min <= f[i] <= axis.range_max for f in frames)
        ]
        if pose_override is not None and report["readonly_arm_position_outside_eeprom_range"]:
            raise RuntimeError("Resumed supported arm pose outside calibration domain; no writes")
        if (before != [0] * 6 or after != [0] * 6 or any(s > 2 for s in spans)
                or any(d > 10 for d in deviations) or abs(fresh - expected_initial) > initial_tolerance):
            raise RuntimeError("Initial/source pose not stable/approved; no writes")
        initial = checked_sample("readonly_initial", fresh)
        start = initial["position_raw"]
        if abs(start - fresh) > 2 or abs(start - expected_initial) > initial_tolerance:
            raise RuntimeError("Fresh jaw position moved before writes")
        report["actual_command_path_raw"] = [start, 1569, 1499]
        for register in SETTINGS:
            originals[register] = io._read_raw(protocol, mapping, gripper, register)
        report["original_sram_settings"] = originals.copy()
        if originals != SETTINGS:
            if originals != {**SETTINGS, "Acceleration": 0}:
                raise RuntimeError("Unexpected SRAM settings; no motion")
            sram_may_be_changed = True
            io._write_raw(protocol, mapping, gripper, "Acceleration", 5)
            time.sleep(.03)
        report["test_sram_settings"] = {r: io._read_raw(protocol, mapping, gripper, r) for r in SETTINGS}
        if report["test_sram_settings"] != SETTINGS:
            raise RuntimeError("Test SRAM readback mismatch")
        tick = 1 / 30
        lateness_limit = min(.02, float(raw_profile["timing"]["max_tick_lateness_s"]))
        print("CAL-089/090 preflight passed; one approved empty-close cycle", flush=True)
        torque_may_be_on = True  # first goal may implicitly enable torque
        motion_start = time.monotonic()
        io._sync_goal(protocol, mapping, gripper, start)
        report["gripper_motion_commands_sent"] += 1
        io._write_raw(protocol, mapping, gripper, "Torque_Enable", 1)
        time.sleep(.03)
        checked_sample("initial_hold", start)
        for phase, first, last in (("prepare_5pct", start, 1569), ("empty_closing", 1569, 1499)):
            base = time.monotonic()
            for index, target in enumerate(segment_targets(first, last, 1398, 30)):
                due = base + index * tick
                time.sleep(max(0, due - time.monotonic()))
                if time.monotonic() - due > lateness_limit or time.monotonic() - motion_start > 30:
                    raise RuntimeError("Trial loop lateness or powered timeout before goal")
                io._sync_goal(protocol, mapping, gripper, target)
                report["gripper_motion_commands_sent"] += 1
                time.sleep(max(0, due + tick - time.monotonic()))
                checked_sample(phase, target)
            print(f"CAL-089/090 {phase} sampled", flush=True)
        hold_start = time.monotonic()
        for index in range(9):
            due = hold_start + index * tick
            time.sleep(max(0, due - time.monotonic()))
            if time.monotonic() - due > lateness_limit or time.monotonic() - hold_start >= .3:
                raise RuntimeError("Final hold exceeded timing bound")
            checked_sample("final_hold", 1499)
            if time.monotonic() - hold_start >= .3:
                raise RuntimeError("Final hold sample exceeded0.3s")
        io._write_raw(protocol, mapping, gripper, "Torque_Enable", 0)
        report["final_hold_observed_duration_s"] = time.monotonic() - hold_start
        report["powered_duration_s"] = time.monotonic() - motion_start
        report["termination_reason"] = "ONE_EMPTY_CLOSE_COMPLETE_IMMEDIATE_UNLOAD"
        hold = [r for r in report["samples"] if r["phase"] == "final_hold"]
        positions = [r["position_raw"] for r in hold]
        if min(positions) < 1499 or max(positions) > 1509 or max(positions) - min(positions) > 2:
            raise RuntimeError("Closed terminal range/span outside approved acceptance")
        representative = statistics.median(positions)
        report["result"] = {
            "terminal_representative_raw": representative,
            "terminal_representative_pct": mapping.raw_to_gripper_pct(representative),
            "terminal_span_raw": max(positions) - min(positions), "independent_cycles": 1,
            "peak_abs_current_ma": max(abs(r["current_ma"]) for r in report["samples"]),
            "max_tracking_error_raw": max(abs(r["tracking_error_raw"]) for r in report["samples"]),
            "terminal_current_counts_ma": {str(v): sum(r["current_ma"] == v for r in hold)
                                           for v in sorted({r["current_ma"] for r in hold})},
            "empty_parameters_fitted": False, "repeatability_verified": False,
        }
        report["status"] = "PASS_PENDING_OPERATOR_OBSERVATION"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            if torque_may_be_on:
                for axis in [gripper] + [a for a in mapping.axes if a.servo_id != gripper.servo_id]:
                    try:
                        io._write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                        if axis.servo_id == 6 and "powered_duration_s" not in report:
                            report["powered_duration_s"] = time.monotonic() - motion_start
                        time.sleep(.03)
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(f"disable {axis.name}: {exc}")
            if torque_may_be_on or sram_may_be_changed:
                for register, value in originals.items():
                    try:
                        io._write_raw(protocol, mapping, gripper, register, value)
                        time.sleep(.03)
                        if io._read_raw(protocol, mapping, gripper, register) != value:
                            raise RuntimeError("Restore readback mismatch")
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(f"restore {register}: {exc}")
            try:
                report["torque_enabled_after"] = [io._read_raw(protocol, mapping, a, "Torque_Enable") for a in mapping.axes]
                report["sram_settings_after"] = {r: io._read_raw(protocol, mapping, gripper, r) for r in originals}
                report["gripper_position_after_torque_off_raw"] = io._read_raw(protocol, mapping, gripper, "Present_Position")
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(f"final readback: {exc}")
            transport.close()
        if report.get("torque_enabled_after") != [0] * 6 or report.get("cleanup_errors"):
            report["status"] = "FAIL_CLEANUP_UNCONFIRMED"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "result", "failure", "cleanup_errors", "torque_enabled_after", "gripper_motion_commands_sent") if k in report}, ensure_ascii=False), flush=True)
    return 0 if failure is None and report["status"] == "PASS_PENDING_OPERATOR_OBSERVATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
