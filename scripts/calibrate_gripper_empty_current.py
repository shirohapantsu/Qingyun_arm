#!/usr/bin/env python3
"""CAL-085: one explicitly approved, bounded empty-gripper current capture.

This captures diagnostic data only. Neither the 65mA software stop nor the
plastic-fixture experiment establishes fruit damage or production safety.
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.calibrate_gripper_open_speed import _sample
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY
from scripts.read_servo_eeprom import sha256_file, usb_identity

CONFIRMATION = "批准，CAL-085空爪初测就绪"
EXPANDED_CONFIRMATION = "批准，CAL-085扩展空爪对照就绪"
UPDATED_BASELINE_CONFIRMATION = "批准，CAL-085新基准空爪对照就绪"
REPOSITION_CONFIRMATION = "批准，CAL-085空爪归位并重测就绪"
CONTACT_DOMAIN_CONFIRMATION = "批准，CAL-085接触范围空爪基线就绪"
SUSTAINED_CONFIRMATION = "批准，CAL-085持续电流空爪短保持就绪"
STRONGER_MATCHED_CONFIRMATION = "批准，CAL-085增强夹持匹配空爪短保持就绪"
SETTINGS = {"Acceleration": 5, "Goal_Time": 0, "Goal_Velocity": 0, "Torque_Limit": 500}


def segment_targets(start: int, end: int, span: int, fps: float) -> list[int]:
    """Round small goals, with command slope bounded to 1%/s on average."""
    if span <= 0 or fps <= 0:
        raise ValueError("positive calibration span and fps required")
    steps = math.ceil(abs(end - start) * 100.0 / span * fps)
    return [round(start + (end - start) * i / steps) for i in range(1, steps + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--repeat", type=int, choices=(1,2), default=1)
    parser.add_argument("--reposition-before-repeat", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--expanded-control", action="store_true")
    mode.add_argument("--updated-baseline-control", action="store_true")
    mode.add_argument("--contact-domain-control", action="store_true")
    mode.add_argument("--sustained-current-control", action="store_true")
    mode.add_argument("--stronger-matched-control", action="store_true")
    args = parser.parse_args()
    short_hold_mode = args.sustained_current_control or args.stronger_matched_control
    candidate_stop_ma = 39 if args.stronger_matched_control else 26
    plan_confirmation = (STRONGER_MATCHED_CONFIRMATION if args.stronger_matched_control
                         else SUSTAINED_CONFIRMATION if args.sustained_current_control
                         else CONTACT_DOMAIN_CONFIRMATION if args.contact_domain_control
                         else UPDATED_BASELINE_CONFIRMATION if args.updated_baseline_control
                         else EXPANDED_CONFIRMATION if args.expanded_control else CONFIRMATION)
    if args.repeat == 2 and not args.updated_baseline_control:
        raise SystemExit("Repeat2 is limited to the currently requested new-baseline empty control")
    if args.reposition_before_repeat and not (args.repeat == 2 and args.updated_baseline_control):
        raise SystemExit("Reposition is limited to the explicitly approved updated-baseline repeat2")
    confirmation = (REPOSITION_CONFIRMATION if args.reposition_before_repeat
                    else "重新测量" if args.repeat == 2 else plan_confirmation)
    if args.output.exists() or args.operator_confirmation != confirmation:
        raise SystemExit("Evidence already exists or exact approval is missing")
    session = json.loads(args.session.read_text(encoding="utf-8"))
    if session["parameter_id"] != "CAL-085" or session["manual_first_contact_raw"] != 1855:
        raise SystemExit("Unexpected manual contact baseline")
    plan_key = ("proposed_stronger_empty_control" if args.stronger_matched_control
                else "proposed_sustained_empty_control" if args.sustained_current_control
                else "proposed_complete_contact_domain_empty_control" if args.contact_domain_control
                else "proposed_updated_baseline_empty_control" if args.updated_baseline_control
                else "proposed_expanded_empty_control" if args.expanded_control else "proposed_empty_trial")
    plan = session[plan_key]
    expected_path = ([1815,1871,1801] if args.stronger_matched_control
                     else [1825,1878,1815] if args.sustained_current_control
                     else [1837,1908,1625] if args.contact_domain_control
                     else [1843,1885,1829] if args.updated_baseline_control
                     else [1854,1896,1840] if args.expanded_control else [1855,1897,1848])
    trial_timeout_s = 30.0 if args.contact_domain_control else 10.0
    timeout_key = "total_powered_timeout_s" if args.contact_domain_control else "trial_timeout_s"
    source_evidence = None
    if short_hold_mode:
        expected_sustained = {
            "required_confirmation": plan_confirmation, "servo_id":6,
            "expected_initial_position_raw":expected_path[0],"initial_position_max_deviation_raw":5,
            "gripper_static_span_max_raw":2,"arm_static_span_max_raw":2,
            "arm_reference_max_deviation_raw":10 if args.stronger_matched_control else 5,"fixed_operational_reference_raw":1836,
            "fixed_open_target_raw":expected_path[1],"fixed_close_target_raw":expected_path[2],
            "illustrative_command_path_raw":expected_path,
            "closure_budget_from_fixed_reference_raw":1836-expected_path[2],
            "extra_closure_from_last_command_endpoint_raw":0,
            "incremental_command_speed_pct_s":1,"sample_fps":30,
            "maximum_final_hold_s":.3,"final_hold_frames":9,"final_position_tolerance_raw":5,
            "closing_and_hold_candidate_current_stop_ma":candidate_stop_ma,
            "all_phase_software_current_abort_ma":65,"max_tracking_error_raw":10,
            "trial_timeout_s":10,"sram_settings":SETTINGS,"arm_joint_goal_commands":0,
            "reference_trial_report":("CAL-085_fixture_stronger_contact_attempt1_20260917.json"
                                      if args.stronger_matched_control else "CAL-085_fixture_increased_closure_attempt1_20260917.json"),
        }
        if any(plan[k] != v for k,v in expected_sustained.items()):
            raise SystemExit("Short-hold plan differs from explicit approval")
        if args.stronger_matched_control and args.output.name != plan["report"]:
            raise SystemExit("Stronger matched control evidence filename differs from approval")
        source_row = next((r for r in session["powered_current_trials"]
                           if r["report"]==expected_sustained["reference_trial_report"]),None)
        if source_row is None or (source_row.get("operator_observation") != "无异常"
                                 if args.stronger_matched_control else not source_row.get("operator_confirmed_grip")):
            raise SystemExit("Missing required source operator observation")
        source_evidence = json.loads((args.profile.resolve(strict=True).parent/"reports"/source_row["report"]).read_text(encoding="utf-8"))
        source_path = [1825,1871,1801] if args.stronger_matched_control else [1830,1878,1815]
        source_after = 1815 if args.stronger_matched_control else 1825
        source_spans = source_evidence["fresh_readonly_baseline"]["raw_position_static_span"]
        if (source_evidence["profile_sha256"] != session["profile_sha256"]
                or source_evidence["actual_command_path_raw"] != source_path
                or source_evidence["gripper_position_after_torque_off_raw"] != source_after
                or len(source_spans) != 6 or any(s>2 for s in source_spans)
                or (not args.stronger_matched_control and source_spans != [0]*6)
                or source_evidence["status"] != "PASS_PENDING_OPERATOR_OBSERVATION"
                or source_evidence["test_sram_settings"] != SETTINGS
                or source_evidence["sram_settings_after"] != source_evidence["original_sram_settings"]
                or source_evidence["torque_enabled_after"] != [0]*6):
            raise SystemExit("Fixed-reference source changed; refusing motion")
    elif (plan["path_raw"] != expected_path
            or plan["incremental_command_speed_pct_s"] != 1.0
            or plan["software_current_abort_ma"] != 65.0
            or plan[timeout_key] != trial_timeout_s
            or plan["sample_fps"] != 30.0 or plan["final_hold_s"] != 0.2
            or plan["sram_settings"] != SETTINGS):
        raise SystemExit("Plan differs from the approved empty trial")
    if (args.expanded_control or args.updated_baseline_control) and (
            plan["required_confirmation"] != plan_confirmation
            or plan["servo_id"] != 6
            or plan["expected_initial_baseline_raw"] != expected_path[0]
            or plan["initial_tolerance_raw"] != 5
            or plan["max_tracking_error_raw"] != 10
            or plan["additional_commanded_closure_budget_from_contact_raw"] != 14):
        raise SystemExit("Expanded plan differs from explicit approval")
    if args.contact_domain_control:
        expected_domain = {"required_confirmation":CONTACT_DOMAIN_CONFIRMATION,
                           "servo_id":6,"expected_initial_raw":1837,
                           "initial_tolerance_raw":5,"max_tracking_error_raw":10}
        if any(plan[k] != v for k,v in expected_domain.items()):
            raise SystemExit("Contact-domain control differs from explicit approval")
    if args.updated_baseline_control:
        contact = session["confirmed_updated_contact_baseline"]
        if contact["gripper_raw"] != 1843 or not contact["uncompressed_first_contact_confirmed"]:
            raise SystemExit("Updated contact reference not confirmed")
    if args.repeat == 2:
        previous = next((r for r in session["powered_current_trials"]
                         if r["role"]=="updated_baseline_empty_range_control" and r["repeat"]==1),None)
        if previous is None or previous["actual_command_path_raw"] != expected_path or previous["torque_enabled_after"] != [0]*6:
            raise SystemExit("No completed same-path new-baseline control to repeat")
    if args.reposition_before_repeat:
        reposition_plan = session["proposed_updated_baseline_empty_reposition_repeat"]
        expected_reposition = {
            "required_confirmation":REPOSITION_CONFIRMATION,"servo_id":6,
            "expected_initial_raw":1833,"initial_tolerance_raw":5,
            "control_path_raw":[1843,1885,1829],
            "incremental_command_speed_pct_s":1,"sample_fps":30,
            "software_current_abort_ma":65,"max_tracking_error_raw":10,
            "total_powered_timeout_s":10,"final_hold_s":.2,"sram_settings":SETTINGS,
        }
        if any(reposition_plan[k] != v for k,v in expected_reposition.items()):
            raise SystemExit("Reposition plan differs from explicit approval")
    start_raw, middle_raw, end_raw = expected_path
    feedback_lower, feedback_upper = min(expected_path)-5, max(expected_path)+5
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != session["profile_sha256"]:
        raise SystemExit("Profile changed after preparation; refusing motion")
    profile = load_calibration_profile(profile_path)
    port = Path(profile.motor.port)
    if str(port) != EXPECTED_STABLE_PORT:
        raise SystemExit("Unexpected stable serial path")
    resolved = port.resolve(strict=True)
    if not resolved.name.startswith("ttyACM") or usb_identity(resolved) != EXPECTED_USB_IDENTITY:
        raise SystemExit("Serial USB identity mismatch")
    occupancy = subprocess.run(["fuser", str(port)], capture_output=True, text=True)
    if occupancy.returncode != 1 or occupancy.stdout.strip():
        raise SystemExit(f"Serial busy or occupancy check failed: {occupancy.stdout} {occupancy.stderr}")
    if profile.joints is None:
        raise SystemExit("Missing motor mapping calibration")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if mapping.gripper_closed_raw != 1499 or mapping.gripper_open_raw != 2897:
        raise SystemExit("Unexpected gripper endpoints")
    for target in expected_path:
        if not max(1499, gripper.range_min) <= target <= min(2897, gripper.range_max):
            raise SystemExit("Target outside calibration bounds")
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if args.contact_domain_control:
        from configs.motion_params import _parse_gripper
        from qingyun.grabbing.kinematics_ext import gap_to_gripper_pct
        tables = _parse_gripper("gripper",raw_profile["gripper"])
        bounds = [1499+1398*gap_to_gripper_pct(g,tables)/100
                  for g in raw_profile["gripper"]["contact_gap_range_m"]]
        derived = [math.floor(bounds[0])-14,math.ceil(bounds[1])+42]
        if derived != [1625,1908] or plan["derivation"]["resulting_empty_command_coverage"] != derived:
            raise SystemExit("Contact range or derived sweep changed; refusing motion")
    if raw_profile["timing"]["fps"] != 30:
        raise SystemExit("Unexpected runtime sample rate")
    # Bound each communication wait rather than retaining the older 5s wait
    # during a powered test. This remains software-only protection.
    io.TRANSACTION_TIMEOUT_S = 0.25
    report = {
        "schema_version": 1, "parameter_id": "CAL-085",
        "test": ("powered_empty_gripper_stronger_contact_matched_short_hold_control" if args.stronger_matched_control
                 else "powered_empty_gripper_fixed_reference_sustained_current_control" if args.sustained_current_control
                 else "powered_empty_gripper_complete_contact_domain_control" if args.contact_domain_control
                 else "powered_empty_gripper_updated_baseline_control" if args.updated_baseline_control
                 else "powered_empty_gripper_expanded_range_control" if args.expanded_control
                 else "powered_empty_gripper_current_baseline"),
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "RUNNING", "operator_confirmation": confirmation,
        "repeat":args.repeat, "plan_authorization_confirmation":plan_confirmation,
        "reposition_before_repeat":args.reposition_before_repeat,
        "actual_command_path_raw": expected_path,
        "profile_sha256": sha256_file(profile_path), "stable_port": str(port),
        "resolved_port": str(resolved), "usb_identity": usb_identity(resolved),
        "approved_plan": plan, "environment": {"power_v": 12, "ambient_temperature_c": None if (args.contact_domain_control or short_hold_mode) else 26.8, "base_mount": "fixed to wooden board", "load": "empty gripper", "temperature_note": "today not remeasured; historical26.8C is not a new measurement" if (args.contact_domain_control or short_hold_mode) else "historical operator baseline"},
        "transaction_timeout_s": 0.25, "joint_servo_motion_commands_sent": 0,
        "gripper_motion_commands_sent": 0, "eeprom_written": False,
        "profile_written": False, "samples": [],
    }
    transport = protocol = None
    originals = {}
    torque_may_be_on = False
    sram_may_be_changed = False
    motion_start = None
    failure = None

    def checked_sample(phase: str, command: int) -> dict:
        row = _sample(protocol, mapping, gripper)
        row.update({"phase": phase, "command_raw": command,
                    "tracking_error_raw": command - row["position_raw"]})
        report["samples"].append(row)
        if abs(row["current_ma"]) > 65.0:
            raise RuntimeError(f"Current stop exceeded: {row['current_ma']}mA")
        if not feedback_lower <= row["position_raw"] <= feedback_upper:
            raise RuntimeError(f"Feedback outside approved path tolerance: {row['position_raw']}")
        if abs(row["tracking_error_raw"]) > 10:
            raise RuntimeError("Empty gripper failed to track within 10 raw")
        if motion_start is not None and time.monotonic() - motion_start > trial_timeout_s:
            raise RuntimeError(f"{trial_timeout_s:g}s powered trial timeout")
        if short_hold_mode and phase in ("empty_closing","final_hold") and abs(row["current_ma"]) >= candidate_stop_ma:
            report["termination_reason"] = f"CANDIDATE_CURRENT_{candidate_stop_ma}MA_DIAGNOSTIC_STOP"
            report["candidate_stop_sample"] = row.copy()
            raise RuntimeError(f"{candidate_stop_ma}mA diagnostic stop before complete empty short-hold control")
        return row

    try:
        transport = SerialTransport(str(port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        for axis in mapping.axes:
            level = io._read_raw(protocol, mapping, axis, "Response_Status_Level")
            if level not in (0, 1):
                raise RuntimeError("Unexpected write response configuration")
            protocol.set_write_response(axis.servo_id, expects_ack=level == 0)
        ids = [io._read_raw(protocol, mapping, axis, "ID") for axis in mapping.axes]
        models = [io._read_raw(protocol, mapping, axis, "Model_Number") for axis in mapping.axes]
        torques = [io._read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes]
        report["readonly_preflight"] = {"ids": ids, "models": models, "torques": torques}
        if ids != [1, 2, 3, 4, 5, 6] or models != [777] * 6 or torques != [0] * 6:
            raise RuntimeError("Preflight ID/model/torque mismatch")
        eeprom = {}
        for axis in mapping.axes:
            minimum = io._read_raw(protocol,mapping,axis,"Min_Position_Limit")
            maximum = io._read_raw(protocol,mapping,axis,"Max_Position_Limit")
            offset = mapping.decode_signed(axis,"Homing_Offset",io._read_raw(protocol,mapping,axis,"Homing_Offset"))
            eeprom[axis.name] = {"range_min":minimum,"range_max":maximum,"homing_offset":offset}
            if (minimum,maximum,offset) != (axis.range_min,axis.range_max,axis.homing_offset):
                report["readonly_eeprom"] = eeprom
                raise RuntimeError(f"EEPROM differs from authoritative file: {axis.name}; no motion or EEPROM write")
        report["readonly_eeprom"] = eeprom
        report["readonly_gripper_controller_settings"] = {
            r:io._read_raw(protocol,mapping,gripper,r)
            for r in ("P_Coefficient","D_Coefficient","I_Coefficient","Minimum_Startup_Force","CW_Dead_Zone","CCW_Dead_Zone")
        }
        report["initial_sample"] = _sample(protocol, mapping, gripper)
        if short_hold_mode:
            # Jaw-only stability missed79raw elbow drift in an earlier fixture
            # capture. Check all six axes BEFORE any SRAM/goal writes; retain
            # the fixed1836 reference, not the unloaded initial jaw position.
            before = [io._read_raw(protocol,mapping,a,"Torque_Enable") for a in mapping.axes]
            frames = []
            for _ in range(10):
                frames.append([io._read_raw(protocol,mapping,a,"Present_Position") for a in mapping.axes])
                time.sleep(.05)
            after = [io._read_raw(protocol,mapping,a,"Torque_Enable") for a in mapping.axes]
            spans = [max(f[i] for f in frames)-min(f[i] for f in frames) for i in range(6)]
            fresh = int(statistics.median(f[5] for f in frames))
            reference_arm = source_evidence["fresh_readonly_baseline"]["raw_position_by_sample"][0][:5]
            deviations = [max(abs(f[i]-reference_arm[i]) for f in frames) for i in range(5)]
            report["fresh_readonly_baseline"] = {"captured_at":datetime.now().astimezone().isoformat(),"raw_position_by_sample":frames,"raw_position_static_span":spans,"gripper_raw":fresh,"torque_enabled_before":before,"torque_enabled_after":after}
            report["arm_deviation_from_reference_raw"] = deviations
            if (before != [0]*6 or after != [0]*6 or any(s>2 for s in spans)
                    or any(d>plan["arm_reference_max_deviation_raw"] for d in deviations)
                    or abs(fresh-expected_path[0])>5):
                raise RuntimeError("Short-hold initial/source pose not stable/approved; no writes")
            if report["readonly_gripper_controller_settings"] != source_evidence["readonly_gripper_controller_settings"]:
                raise RuntimeError("Gripper controller changed since fixed-reference source; no writes")
            start_raw = fresh
            report["actual_command_path_raw"] = [start_raw,middle_raw,end_raw]
            report["fixed_operational_reference_raw"] = 1836
            report["fresh_position_is_not_new_contact_reference"] = True
            feedback_lower,feedback_upper = min(start_raw,end_raw)-5,max(start_raw,middle_raw)+5
        required_initial = 1833 if args.reposition_before_repeat else start_raw
        if abs(report["initial_sample"]["position_raw"] - required_initial) > 5:
            raise RuntimeError(f"Starting gripper raw={report['initial_sample']['position_raw']} differs from approved baseline")
        for register in SETTINGS:
            originals[register] = io._read_raw(protocol, mapping, gripper, register)
        report["original_sram_settings"] = originals.copy()
        # Match previous empty-test runtime settings, do not silently alter output.
        if originals != SETTINGS:
            # After reconnection SRAM Acceleration may be0. The authorized
            # contact-domain plan explicitly specifies5; prepare ONLY this
            # register, retain output/time/velocity, and restore0 even if the
            # preparation fails before any goal. This is not an EEPROM write.
            allowed_boot_settings = {**SETTINGS,"Acceleration":0}
            if not (args.contact_domain_control or short_hold_mode) or originals != allowed_boot_settings:
                raise RuntimeError("Current SRAM differs from proposed settings; no motion")
            sram_may_be_changed = True
            io._write_raw(protocol,mapping,gripper,"Acceleration",5)
            time.sleep(.03)
        report["test_sram_settings"] = {r:io._read_raw(protocol,mapping,gripper,r) for r in SETTINGS}
        if report["test_sram_settings"] != SETTINGS:
            raise RuntimeError("Approved test SRAM readback mismatch; no motion")
        print("CAL-085 preflight passed; starting approved empty gripper path", flush=True)
        # CAL-052: first Goal_Position may implicitly enable torque. Mark before
        # sending so interruption/errors still trigger six-axis torque cleanup.
        torque_may_be_on = True
        motion_start = time.monotonic()
        first_goal = report["initial_sample"]["position_raw"] if args.reposition_before_repeat else start_raw
        io._sync_goal(protocol, mapping, gripper, first_goal)
        report["gripper_motion_commands_sent"] += 1
        io._write_raw(protocol, mapping, gripper, "Torque_Enable", 1)
        time.sleep(0.03)
        checked_sample("initial_hold", first_goal)
        tick = 1.0 / 30.0
        lateness_limit = min(0.02, float(raw_profile["timing"]["max_tick_lateness_s"]))
        def run_segment(phase: str, start: int, end: int) -> None:
            if start == end:
                return
            targets = segment_targets(start, end, 1398, 30.0)
            base = time.monotonic()
            for index, target in enumerate(targets):
                due = base + index * tick
                time.sleep(max(0.0, due - time.monotonic()))
                if time.monotonic() - due > lateness_limit:
                    raise RuntimeError("Sample loop exceeded approved tick lateness")
                if time.monotonic() - motion_start > trial_timeout_s:
                    raise RuntimeError(f"{trial_timeout_s:g}s powered trial timeout before goal")
                io._sync_goal(protocol, mapping, gripper, target)
                report["gripper_motion_commands_sent"] += 1
                time.sleep(max(0.0, due + tick - time.monotonic()))
                checked_sample(phase, target)
            print(f"CAL-085 {phase} sampled", flush=True)
        if args.reposition_before_repeat:
            report["approved_reposition_plan"] = reposition_plan
            run_segment("preposition_opening",first_goal,start_raw)
            for _ in range(6):
                time.sleep(tick)
                reset_sample = checked_sample("preposition_settle",start_raw)
            if abs(reset_sample["position_raw"]-start_raw)>5:
                raise RuntimeError("Serial reposition did not meet original 5raw control-start check; no control segment")
            report["control_start_readback_raw"] = reset_sample["position_raw"]
        for phase, start, end in (("opening", start_raw, middle_raw), ("empty_closing", middle_raw, end_raw)):
            run_segment(phase,start,end)
        hold_frames = 9 if short_hold_mode else 6
        hold_start = time.monotonic()
        for index in range(hold_frames):
            if short_hold_mode:
                due = hold_start+index*tick
                time.sleep(max(0,due-time.monotonic()))
                if time.monotonic()-due > lateness_limit or time.monotonic()-hold_start >= .3:
                    raise RuntimeError("Short hold exceeded approved timing bound")
            else:
                time.sleep(tick)
            final = checked_sample("final_hold", end_raw)
        if short_hold_mode:
            # Sample9 frames at0..8ticks and disable immediately, before stats
            # or readback, rather than extending the approved hold for cleanup.
            io._write_raw(protocol,mapping,gripper,"Torque_Enable",0)
            report["final_hold_observed_duration_s"] = time.monotonic()-hold_start
            report["powered_duration_s"] = time.monotonic()-motion_start
            report["termination_reason"] = "EMPTY_SHORT_HOLD_COMPLETE_IMMEDIATE_UNLOAD"
        if abs(final["position_raw"] - end_raw) > 5:
            raise RuntimeError("Final empty position outside 5 raw tolerance")
        report["powered_duration_s"] = time.monotonic() - motion_start
        report["result"] = {
            "final_position_raw": final["position_raw"],
            "by_phase": {
                phase: {"samples": len(rows),
                        "peak_abs_current_ma": max(abs(r["current_ma"]) for r in rows),
                        "median_abs_current_ma": statistics.median(abs(r["current_ma"]) for r in rows),
                        "observed_current_raw": sorted(set(r["current_register_raw"] for r in rows)),
                        "max_tracking_error_raw": max(abs(r["tracking_error_raw"]) for r in rows)}
                for phase in ("initial_hold", "preposition_opening", "preposition_settle", "opening", "empty_closing", "final_hold")
                if (rows := [r for r in report["samples"] if r["phase"] == phase])
            },
            "contact_threshold_fitted": False,
        }
        report["status"] = "PASS_PENDING_OPERATOR_OBSERVATION"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            if torque_may_be_on:
                # No arm goals; only torque-disable writes to IDs 1..5.
                # Disable the only powered servo first on abort, then verify all
                # six axes in the normal motor order after cleanup.
                cleanup_axes = [gripper]+[a for a in mapping.axes if a.servo_id != gripper.servo_id]
                for axis in cleanup_axes:
                    try:
                        io._write_raw(protocol, mapping, axis, "Torque_Enable", 0)
                        time.sleep(0.03)
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(f"disable {axis.name}: {exc}")
            if torque_may_be_on or sram_may_be_changed:
                for register, value in originals.items():
                    try:
                        io._write_raw(protocol, mapping, gripper, register, value)
                        time.sleep(0.03)
                        if io._read_raw(protocol, mapping, gripper, register) != value:
                            raise RuntimeError("restore readback mismatch")
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(f"restore {register}: {exc}")
            try:
                report["torque_enabled_after"] = [io._read_raw(protocol, mapping, axis, "Torque_Enable") for axis in mapping.axes]
                report["sram_settings_after"] = {r: io._read_raw(protocol, mapping, gripper, r) for r in originals}
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(f"final readback: {exc}")
            transport.close()
        if report.get("torque_enabled_after") != [0] * 6 or report.get("cleanup_errors"):
            report["status"] = "FAIL_CLEANUP_UNCONFIRMED"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "result", "failure", "cleanup_errors", "torque_enabled_after") if k in report}, ensure_ascii=False), flush=True)
    return 1 if failure is not None or report["status"] != "PASS_PENDING_OPERATOR_OBSERVATION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
