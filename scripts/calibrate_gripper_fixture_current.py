#!/usr/bin/env python3
"""CAL-085 explicitly approved, bounded plastic-fixture diagnostic only."""
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

CONFIRMATION = "批准，CAL-085假果接触初测就绪"
REPEAT2_CONFIRMATION = "批准，CAL-085假果接触重测就绪"
EXPANDED_CONFIRMATION = "批准，CAL-085假果扩展接触初测就绪"
UPDATED_BASELINE_CONFIRMATION = "批准，CAL-085新基准假果接触初测就绪"
INCREASED_CLOSURE_CONFIRMATION = "无异常，可以增加闭合程度再次尝试"
SUSTAINED_CONFIRMATION = "批准，CAL-085假果持续电流短保持就绪"
RELAXED_SUSTAINED_CONFIRMATION = "放宽容差完成标定"
STRONGER_CONFIRMATION = "可以大胆增大夹持力度，假果不易损坏"


def covered_by_intervals(lower: int, upper: int, intervals: list[list[int]]) -> bool:
    """Only accept a continuous union of already observed empty command ranges."""
    cursor = lower
    for start,end in sorted(intervals):
        if end < cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor,end)
        if cursor >= upper:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "session", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--repeat", type=int, choices=(1,2), default=1)
    parser.add_argument("--relaxed-placement", action="store_true")
    parser.add_argument("--stronger-contact", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--expanded-contact", action="store_true")
    mode.add_argument("--updated-baseline-contact", action="store_true")
    mode.add_argument("--increased-closure-contact", action="store_true")
    mode.add_argument("--sustained-current-contact", action="store_true")
    args = parser.parse_args()
    if args.relaxed_placement and not args.sustained_current_contact:
        raise SystemExit("Relaxed placement applies only to the authorized fixture short hold")
    if args.stronger_contact and (not args.sustained_current_contact or args.relaxed_placement):
        raise SystemExit("Stronger plastic-fixture diagnostic is a separate sustained-contact approval")
    relaxed_placement = args.relaxed_placement or args.stronger_contact
    opening_offset_raw = 35 if args.stronger_contact else 42
    candidate_current_stop_ma = 39 if args.stronger_contact else 26
    fixed_reference_mode = args.increased_closure_contact or args.sustained_current_contact
    updated_mode = args.updated_baseline_contact or fixed_reference_mode
    expanded = args.expanded_contact or updated_mode
    if expanded and args.repeat != 1:
        raise SystemExit("Expanded contact permits one separately approved first trial only")
    confirmation = (STRONGER_CONFIRMATION if args.stronger_contact
                    else RELAXED_SUSTAINED_CONFIRMATION if args.relaxed_placement
                    else SUSTAINED_CONFIRMATION if args.sustained_current_contact
                    else INCREASED_CLOSURE_CONFIRMATION if args.increased_closure_contact
                    else UPDATED_BASELINE_CONFIRMATION if args.updated_baseline_contact
                    else EXPANDED_CONFIRMATION if args.expanded_contact
                    else CONFIRMATION if args.repeat == 1 else REPEAT2_CONFIRMATION)
    if args.output.exists() or args.operator_confirmation != confirmation:
        raise SystemExit("Exact approval missing or evidence already exists")
    session = json.loads(args.session.read_text(encoding="utf-8"))
    plan = (session["proposed_stronger_fixture_current_trial"] if args.stronger_contact
            else session["proposed_relaxed_sustained_fixture_current_trial"] if args.relaxed_placement
            else session["proposed_sustained_fixture_current_trial"] if args.sustained_current_contact
            else session["proposed_increased_closure_fixture_trial"] if args.increased_closure_contact
            else session["proposed_updated_baseline_fixture_trial"] if args.updated_baseline_contact
            else session["proposed_expanded_fixture_trial"] if args.expanded_contact
            else session["proposed_fixture_trial"])
    closure_budget = 35 if args.stronger_contact else 21 if fixed_reference_mode else 14 if expanded else 7
    expected_baseline = (1836 if args.sustained_current_contact else 1831 if args.increased_closure_contact else 1843 if args.updated_baseline_contact
                         else 1854 if args.expanded_contact else 1855)
    expected = {
        "servo_id": 6,
        "fresh_gripper_static_span_max_raw": 2,
        ("path_offsets_from_reference_raw" if args.increased_closure_contact
         else "path_offsets_from_fresh_baseline_raw"): [0,42,-closure_budget],
        "incremental_command_speed_pct_s": 1.0, "sample_fps": 30.0,
        "additional_commanded_closure_budget_raw": closure_budget,
        "closing_candidate_current_stop_ma": 26.0,
        "all_phase_software_current_abort_ma": 65.0,
        "trial_timeout_s": 10.0, "final_hold_s": 0.0,
        "sram_settings": SETTINGS, "arm_joint_goal_commands": 0,
    }
    if expanded:
        expected.update({"required_confirmation":confirmation,
                         "expected_baseline_raw":expected_baseline,
                         "fresh_baseline_max_deviation_raw":15 if args.updated_baseline_contact else 5,
                         "max_tracking_error_raw":10})
    else:
        expected["fresh_baseline_max_deviation_from_1855_raw"] = 5
    if args.sustained_current_contact:
        expected = {
            "required_confirmation":confirmation,"parameter_id":"CAL-085","servo_id":6,
            "fixed_operational_reference_raw":1836,"fixed_open_target_raw":1836+opening_offset_raw,
            "fixed_close_target_raw":1836-closure_budget,"expected_initial_position_raw":1836,
            "initial_position_max_deviation_raw":20 if relaxed_placement else 5,"gripper_static_span_max_raw":2,
            "arm_static_span_max_raw":2,"arm_reference_max_deviation_raw":10 if relaxed_placement else 5,
            "illustrative_command_path_raw":[1836,1836+opening_offset_raw,1836-closure_budget],
            "closure_budget_from_fixed_reference_raw":closure_budget,"extra_closure_from_last_command_endpoint_raw":14 if args.stronger_contact else 0,
            "maximum_final_hold_s":.3,"final_hold_frames":9,"sample_fps":30,
            "incremental_command_speed_pct_s":1,"closing_and_hold_candidate_current_stop_ma":candidate_current_stop_ma,
            "all_phase_software_current_abort_ma":65,"max_tracking_error_raw":25 if args.stronger_contact else 10,
            "trial_timeout_s":10,"sram_settings":SETTINGS,"arm_joint_goal_commands":0,
            "paired_control_report":"CAL-085_empty_sustained_current_control_attempt1_20260917.json",
            "reference_grip_report":"CAL-085_fixture_increased_closure_attempt1_20260917.json",
        }
        if args.stronger_contact:
            expected["opening_max_tracking_error_raw"] = 10
            expected["initial_hold_max_tracking_error_raw"] = 10
    if session["parameter_id"] != "CAL-085" or any(plan[k] != v for k, v in expected.items()):
        raise SystemExit("Plan differs from approved fixture trial")
    source_evidence = None
    if args.sustained_current_contact:
        reports = args.profile.resolve(strict=True).parent/"reports"
        source_evidence = json.loads((reports/plan["paired_control_report"]).read_text(encoding="utf-8"))
        grip_row = next((r for r in session["powered_current_trials"]
                         if r["report"]==plan["reference_grip_report"]),None)
        grip_evidence = json.loads((reports/plan["reference_grip_report"]).read_text(encoding="utf-8"))
        if (grip_row is None or not grip_row.get("operator_confirmed_grip")
                or grip_evidence["profile_sha256"] != session["profile_sha256"]
                or grip_evidence["fixed_contact_reference_raw"] != 1836
                or grip_evidence["actual_command_path_raw"][-2:] != [1878,1815]
                or source_evidence.get("operator_observation") != "无异常"
                or source_evidence["status"] != "PASS_PENDING_OPERATOR_OBSERVATION"
                or source_evidence["profile_sha256"] != session["profile_sha256"]
                or source_evidence["actual_command_path_raw"][-2:] != [1878,1815]
                or source_evidence["approved_plan"]["incremental_command_speed_pct_s"] != 1
                or source_evidence["approved_plan"]["maximum_final_hold_s"] != .3
                or source_evidence["test_sram_settings"] != SETTINGS
                or source_evidence["fresh_readonly_baseline"]["raw_position_static_span"] != [0]*6
                or source_evidence["torque_enabled_after"] != [0]*6
                or source_evidence["final_hold_observed_duration_s"] > .3
                or len([r for r in source_evidence["samples"] if r["phase"]=="final_hold"]) != 9):
            raise SystemExit("Missing confirmed matched empty short-hold or grip source")
    if args.increased_closure_contact:
        # Never compound the extra closure by treating the unloaded/partly
        # closed feedback as a new first-contact zero. Extend the previous
        # stable trial's absolute endpoint1822 by only7raw, to1815.
        source_name = "CAL-085_fixture_updated_baseline_contact_repeat1_attempt3_20260917.json"
        source_row = next((row for row in session["powered_current_trials"]
                           if row["report"] == source_name), None)
        if (plan["reference_trial_report"] != source_name
                or plan["fixed_contact_reference_raw"] != 1836
                or plan["fixed_open_target_raw"] != 1878
                or plan["fixed_close_target_raw"] != 1815
                or plan["arm_reference_max_deviation_raw"] != 5
                or source_row is None or source_row.get("operator_observation") != "无异常"):
            raise SystemExit("Missing confirmed stable source trial or fixed extension plan")
        source_evidence = json.loads((args.profile.resolve(strict=True).parent/"reports"/source_name).read_text(encoding="utf-8"))
        if (source_evidence["profile_sha256"] != session["profile_sha256"]
                or source_evidence["actual_command_path_raw"] != [1836,1878,1822]
                or source_evidence["fresh_readonly_baseline"]["raw_position_static_span"] != [0]*6
                or source_evidence["gripper_position_after_torque_off_raw"] != 1831
                or source_evidence["test_sram_settings"] != SETTINGS
                or source_evidence["torque_enabled_after"] != [0]*6):
            raise SystemExit("Source trial changed or was not a stable bounded capture")
    if args.repeat == 2:
        repeat_plan = session["proposed_fixture_repeat2"]
        repeat_expected = {
            "required_confirmation": REPEAT2_CONFIRMATION,
            "path_raw_if_baseline_unchanged": [1854,1896,1847],
            "expected_preflight_baseline_raw": 1854,
            "incremental_command_speed_pct_s": 1.0,
            "closing_candidate_current_stop_ma": 26.0,
            "all_phase_software_current_abort_ma": 65.0,
            "trial_timeout_s": 10.0, "final_hold_s": 0.0,
        }
        if any(repeat_plan[k] != v for k,v in repeat_expected.items()):
            raise SystemExit("Repeat2 differs from explicitly confirmed plan")
    if session["powered_current_trials"][0].get("operator_observation") != "无异常":
        raise SystemExit("Missing empty-trial observation")
    if expanded:
        if args.sustained_current_contact:
            control = next((r for r in session["powered_current_trials"]
                            if r["role"]=="sustained_empty_current_control"
                            and r["report"]==plan["paired_control_report"]),None)
        elif updated_mode:
            control = next((r for r in reversed(session["powered_current_trials"])
                            if r["role"] in ("complete_contact_domain_empty_control","updated_baseline_empty_range_control")
                            and r.get("operator_observation")=="无异常"),None)
        else:
            control = next((r for r in session["powered_current_trials"]
                            if r["role"]=="expanded_empty_range_control"),None)
        if control is None or control.get("operator_observation") != "无异常":
            raise SystemExit("Missing expanded empty control observation")
        if updated_mode and control["report"] != plan["paired_control_report"]:
            raise SystemExit("New-baseline paired control differs from approved report")
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != session["profile_sha256"]:
        raise SystemExit("Profile changed since preparation")
    profile = load_calibration_profile(profile_path)
    port = Path(profile.motor.port)
    if str(port) != EXPECTED_STABLE_PORT:
        raise SystemExit("Unexpected stable serial path")
    resolved = port.resolve(strict=True)
    if not resolved.name.startswith("ttyACM") or usb_identity(resolved) != EXPECTED_USB_IDENTITY:
        raise SystemExit("USB identity mismatch")
    occupied = subprocess.run(["fuser", str(port)], capture_output=True, text=True)
    if occupied.returncode != 1 or occupied.stdout.strip():
        raise SystemExit("Serial occupied or fuser check failed")
    if profile.joints is None:
        raise SystemExit("Missing joint calibration")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if (mapping.gripper_closed_raw, mapping.gripper_open_raw) != (1499,2897):
        raise SystemExit("Unexpected gripper calibration endpoints")
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if raw_profile["timing"]["fps"] != 30:
        raise SystemExit("Unexpected runtime sample rate")
    io.TRANSACTION_TIMEOUT_S = 0.25
    report = {
        "schema_version": 1, "parameter_id": "CAL-085",
        "test": ("supported_plastic_fixture_bounded_stronger_contact_diagnostic" if args.stronger_contact
                 else "supported_plastic_fixture_fixed_reference_sustained_current_diagnostic" if args.sustained_current_contact
                 else "supported_plastic_fixture_fixed_reference_increased_closure_diagnostic" if args.increased_closure_contact
                 else "supported_plastic_fixture_updated_baseline_contact_diagnostic" if args.updated_baseline_contact
                 else "supported_plastic_fixture_expanded_low_speed_contact_diagnostic" if args.expanded_contact
                 else "supported_plastic_fixture_bounded_low_speed_contact_diagnostic"),
        "started_at": datetime.now().astimezone().isoformat(), "status": "RUNNING",
        "operator_confirmation": confirmation, "repeat":args.repeat, "approved_plan": plan,
        "relaxed_placement":relaxed_placement,"stronger_contact":args.stronger_contact,
        "matched_static_hold_control":not args.stronger_contact if args.sustained_current_contact else False,
        "profile_sha256": sha256_file(profile_path), "stable_port": str(port),
        "resolved_port": str(resolved), "usb_identity": usb_identity(resolved),
        "environment": {"power_v":12,"ambient_temperature_c":None,
                        "ambient_temperature_note":"Not remeasured for this trial; historical measurement was 26.8C",
                        "base_mount":"fixed to wooden board",
                        "fixture":"supported approximately 42mm plastic imitation fruit at actual 20mm band"},
        "transaction_timeout_s":0.25, "eeprom_written":False, "profile_written":False,
        "joint_servo_motion_commands_sent":0, "gripper_motion_commands_sent":0,
        "samples":[], "contact_threshold_fitted":False,
    }
    transport = protocol = None
    originals = {}
    torque_may_be_on = False
    sram_may_be_changed = False
    failure = None
    motion_start = None
    baseline = None
    stop_reason = None

    def capture(phase: str, command: int) -> dict:
        row = _sample(protocol, mapping, gripper)
        row.update({"phase":phase,"command_raw":command,"tracking_error_raw":command-row["position_raw"]})
        report["samples"].append(row)
        if abs(row["current_ma"]) > 65:
            raise RuntimeError(f"65mA software abort: {row['current_ma']}mA")
        if not baseline-closure_budget <= row["position_raw"] <= baseline+47:
            raise RuntimeError("Feedback outside bounded path")
        # A stiff supported fixture intentionally resists the closing goal.
        # Only explicit stronger-contact approval permits a25raw error ceiling
        # during closing/holding. Opening and first-goal checks stay at10raw;
        # never reuse this tolerance as a general motor-tracking limit.
        tracking_limit_raw = 25 if args.stronger_contact and phase in ("fixture_closing","fixture_hold") else 10
        if abs(row["tracking_error_raw"]) > tracking_limit_raw:
            raise RuntimeError(f"Tracking error exceeded {tracking_limit_raw} raw during {phase}")
        if time.monotonic()-motion_start > 10:
            raise RuntimeError("10s powered timeout")
        return row

    try:
        transport = SerialTransport(str(port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        for axis in mapping.axes:
            level = io._read_raw(protocol,mapping,axis,"Response_Status_Level")
            if level not in (0,1):
                raise RuntimeError("Unexpected response configuration")
            protocol.set_write_response(axis.servo_id,expects_ack=level==0)
        ids = [io._read_raw(protocol,mapping,a,"ID") for a in mapping.axes]
        models = [io._read_raw(protocol,mapping,a,"Model_Number") for a in mapping.axes]
        before = [io._read_raw(protocol,mapping,a,"Torque_Enable") for a in mapping.axes]
        report["readonly_preflight"] = {"ids":ids,"models":models,"torques":before}
        if ids != [1,2,3,4,5,6] or models != [777]*6 or before != [0]*6:
            raise RuntimeError("ID/model/torque preflight mismatch")
        # Read-only EEPROM comparison; never silently write EEPROM as part of a
        # jaw-motion authorization. A mismatch terminates before any goals.
        eeprom = {}
        for axis in mapping.axes:
            minimum = io._read_raw(protocol,mapping,axis,"Min_Position_Limit")
            maximum = io._read_raw(protocol,mapping,axis,"Max_Position_Limit")
            offset = mapping.decode_signed(axis,"Homing_Offset",io._read_raw(protocol,mapping,axis,"Homing_Offset"))
            eeprom[axis.name] = {"range_min":minimum,"range_max":maximum,"homing_offset":offset}
            if (minimum,maximum,offset) != (axis.range_min,axis.range_max,axis.homing_offset):
                report["readonly_eeprom"] = eeprom
                raise RuntimeError(f"EEPROM differs from authoritative file: {axis.name}; no motion/write")
        report["readonly_eeprom"] = eeprom
        if expanded:
            controller = {r:io._read_raw(protocol,mapping,gripper,r)
                          for r in control["readonly_gripper_controller_settings"]}
            report["readonly_gripper_controller_settings"] = controller
            if controller != control["readonly_gripper_controller_settings"]:
                raise RuntimeError("Controller settings changed since paired empty control; no motion")
        frames = []
        for _ in range(10):
            frames.append([io._read_raw(protocol,mapping,a,"Present_Position") for a in mapping.axes])
            time.sleep(0.05)
        after = [io._read_raw(protocol,mapping,a,"Torque_Enable") for a in mapping.axes]
        spans = [max(f[i] for f in frames)-min(f[i] for f in frames) for i in range(6)]
        baseline = int(statistics.median(f[5] for f in frames))
        report["fresh_readonly_baseline"] = {"captured_at":datetime.now().astimezone().isoformat(),"raw_position_by_sample":frames,"raw_position_static_span":spans,"gripper_raw":baseline,"torque_enabled_before":before,"torque_enabled_after":after}
        # This user-authorized demo placement tolerance is NOT the powered
        # tracking tolerance: that independent abort remains 10raw above.
        placement_tolerance_raw = 20 if relaxed_placement else 15 if args.updated_baseline_contact else 5
        if after != [0]*6 or spans[5]>2 or abs(baseline-expected_baseline)>placement_tolerance_raw:
            raise RuntimeError(f"Fresh baseline not approved/stable: R={baseline}, spans={spans}")
        # Independent arm support is a physical prerequisite for a paired
        # contact-current capture. A stable jaw alone cannot establish that
        # the20mm contact band stayed put while unpowered joints drifted.
        # Gate all five arm axes before any SRAM/goal writes; never compensate
        # this drift by enabling arm torque or relaxing powered protections.
        if any(span > 2 for span in spans[:5]):
            raise RuntimeError(f"Supported arm not static during preflight: spans={spans}; no motion")
        if args.repeat == 2 and baseline != 1854:
            raise RuntimeError(f"Repeat2 baseline shifted from approved 1854 to {baseline}; no motion")
        initial_command_raw = baseline
        if fixed_reference_mode:
            reference_arm = source_evidence["fresh_readonly_baseline"]["raw_position_by_sample"][0][:5]
            deviations = [max(abs(frame[i]-reference_arm[i]) for frame in frames) for i in range(5)]
            report["arm_deviation_from_reference_raw"] = deviations
            # Only user-authorized STATIC placement consistency is relaxed.
            # Powered tracking/current/static-span/timeout bounds stay intact.
            reference_pose_tolerance_raw = 10 if relaxed_placement else 5
            if any(deviation > reference_pose_tolerance_raw for deviation in deviations):
                raise RuntimeError("Arm pose changed from fixed contact reference; no motion")
            baseline = 1836
            report["fixed_contact_reference_raw"] = baseline
            report["fresh_position_is_not_new_contact_reference"] = True
        targets_path = [initial_command_raw,baseline+opening_offset_raw,baseline-closure_budget]
        report["actual_command_path_raw"] = targets_path
        if updated_mode:
            intervals = []
            sources = []
            for row in session["powered_current_trials"]:
                if (row["role"] not in ("expanded_empty_range_control","updated_baseline_empty_range_control","complete_contact_domain_empty_control")
                        or row.get("operator_observation") != "无异常"
                        or row.get("readonly_gripper_controller_settings",controller) != controller):
                    continue
                evidence = json.loads((profile_path.parent/"reports"/row["report"]).read_text(encoding="utf-8"))
                if (evidence["profile_sha256"] != session["profile_sha256"]
                        or evidence["approved_plan"]["incremental_command_speed_pct_s"] != 1
                        or evidence.get("test_sram_settings",evidence["original_sram_settings"]) != SETTINGS
                        or evidence["torque_enabled_after"] != [0]*6):
                    continue
                intervals.append([min(row["actual_command_path_raw"]),max(row["actual_command_path_raw"])])
                sources.append(row["report"])
            report["paired_empty_coverage"] = {"command_intervals_raw":intervals,"source_reports":sources}
            if not covered_by_intervals(min(targets_path),max(targets_path),intervals):
                raise RuntimeError("Fresh baseline shifts path outside confirmed same-speed empty coverage; no motion")
        for target in targets_path:
            if not max(1499,gripper.range_min)<=target<=min(2897,gripper.range_max):
                raise RuntimeError("Derived target outside calibration limits")
        originals = {r:io._read_raw(protocol,mapping,gripper,r) for r in SETTINGS}
        report["original_sram_settings"] = originals.copy()
        if originals != SETTINGS:
            allowed_boot_settings = {**SETTINGS,"Acceleration":0}
            if not updated_mode or originals != allowed_boot_settings:
                raise RuntimeError("SRAM differs from proposed settings; no motion")
            # Prepare only the approved acceleration5, retain time/velocity/
            # output, and restore original0 even if setup fails before a goal.
            sram_may_be_changed = True
            io._write_raw(protocol,mapping,gripper,"Acceleration",5)
            time.sleep(.03)
        report["test_sram_settings"] = {r:io._read_raw(protocol,mapping,gripper,r) for r in SETTINGS}
        if report["test_sram_settings"] != SETTINGS:
            raise RuntimeError("Approved test SRAM readback mismatch; no motion")
        print(f"CAL-085 readonly checks passed: R={baseline}, path={targets_path}; starting fixture trial",flush=True)
        # CAL-052: the goal write can implicitly enable torque. Cleanup must be
        # armed before the very first goal; never send arm-joint goal packets.
        torque_may_be_on = True
        motion_start = time.monotonic()
        io._sync_goal(protocol,mapping,gripper,initial_command_raw)
        report["gripper_motion_commands_sent"] += 1
        io._write_raw(protocol,mapping,gripper,"Torque_Enable",1)
        time.sleep(0.03)
        capture("initial_hold",initial_command_raw)
        tick = 1/30
        late_limit = min(0.02,float(raw_profile["timing"]["max_tick_lateness_s"]))
        for phase,start,end in (("opening",initial_command_raw,baseline+opening_offset_raw),("fixture_closing",baseline+opening_offset_raw,baseline-closure_budget)):
            base = time.monotonic()
            for index,target in enumerate(segment_targets(start,end,1398,30)):
                due = base+index*tick
                time.sleep(max(0,due-time.monotonic()))
                if time.monotonic()-due>late_limit:
                    raise RuntimeError("Tick lateness exceeded bound")
                if time.monotonic()-motion_start>10:
                    raise RuntimeError("10s powered timeout before goal")
                io._sync_goal(protocol,mapping,gripper,target)
                report["gripper_motion_commands_sent"] += 1
                time.sleep(max(0,due+tick-time.monotonic()))
                row = capture(phase,target)
                if phase=="fixture_closing" and abs(row["current_ma"])>=candidate_current_stop_ma:
                    stop_reason = f"CANDIDATE_CURRENT_{candidate_current_stop_ma}MA_DIAGNOSTIC_STOP"
                    report["candidate_stop_sample"] = row.copy()
                    break
            if stop_reason:
                break
        if args.sustained_current_contact and not stop_reason:
            # Only this separately approved mode may retain the last1815 goal
            # for9 samples. No deeper goals, output increase, or recovery move;
            # keep all current/tracking/timeout checks active during holding.
            hold_start = time.monotonic()
            for index in range(9):
                due = hold_start+index*tick
                time.sleep(max(0,due-time.monotonic()))
                if time.monotonic()-due > late_limit or time.monotonic()-hold_start >= .3:
                    raise RuntimeError("Fixture short hold exceeded approved timing bound")
                row = capture("fixture_hold",baseline-closure_budget)
                if abs(row["current_ma"]) >= candidate_current_stop_ma:
                    stop_reason = f"CANDIDATE_CURRENT_{candidate_current_stop_ma}MA_DIAGNOSTIC_STOP"
                    report["candidate_stop_sample"] = row.copy()
                    break
            report["fixture_hold_sample_duration_s"] = time.monotonic()-hold_start
        # Stop torque immediately, before aggregation/reporting. Only the
        # explicit short-hold mode above may dwell; never add recovery motion.
        io._write_raw(protocol,mapping,gripper,"Torque_Enable",0)
        if args.sustained_current_contact and "fixture_hold_sample_duration_s" in report:
            report["fixture_hold_observed_duration_s"] = time.monotonic()-hold_start
        report["powered_duration_s"] = time.monotonic()-motion_start
        stop_reason = stop_reason or ("FIXTURE_SHORT_HOLD_COMPLETE_IMMEDIATE_UNLOAD" if args.sustained_current_contact
                                     else "COMMAND_BUDGET_EXHAUSTED_WITHOUT_CANDIDATE_CURRENT")
        report["termination_reason"] = stop_reason
        report["result"] = {"by_phase":{
            phase:{"samples":len(rows),"peak_abs_current_ma":max(abs(r["current_ma"]) for r in rows),"median_abs_current_ma":statistics.median(abs(r["current_ma"]) for r in rows),"position_raw_range":[min(r["position_raw"] for r in rows),max(r["position_raw"] for r in rows)],"max_tracking_error_raw":max(abs(r["tracking_error_raw"]) for r in rows)}
            for phase in ("initial_hold","opening","fixture_closing","fixture_hold")
            if (rows:=[r for r in report["samples"] if r["phase"]==phase])
        },"last_sample":report["samples"][-1],"confirmed_physical_contact":False,
            "runtime_threshold_validated":False,"contact_threshold_fitted":False}
        report["status"] = "PASS_PENDING_OPERATOR_OBSERVATION"
    except BaseException as exc:
        failure = exc
        report["status"] = "FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if protocol is not None:
            if torque_may_be_on:
                # On error, disable the powered jaw first rather than holding it
                # while five already-unpowered arm axes receive cleanup writes.
                cleanup_axes = [gripper]+[a for a in mapping.axes if a.servo_id != gripper.servo_id]
                for axis in cleanup_axes:
                    try:
                        io._write_raw(protocol,mapping,axis,"Torque_Enable",0)
                        time.sleep(0.03)
                    except BaseException as exc:
                        report.setdefault("cleanup_errors",[]).append(f"disable {axis.name}: {exc}")
            if torque_may_be_on or sram_may_be_changed:
                for register,value in originals.items():
                    try:
                        io._write_raw(protocol,mapping,gripper,register,value)
                        time.sleep(0.03)
                        if io._read_raw(protocol,mapping,gripper,register)!=value:
                            raise RuntimeError("SRAM restore mismatch")
                    except BaseException as exc:
                        report.setdefault("cleanup_errors",[]).append(f"restore {register}: {exc}")
            try:
                report["torque_enabled_after"] = [io._read_raw(protocol,mapping,a,"Torque_Enable") for a in mapping.axes]
                report["sram_settings_after"] = {r:io._read_raw(protocol,mapping,gripper,r) for r in originals}
                report["gripper_position_after_torque_off_raw"] = io._read_raw(protocol,mapping,gripper,"Present_Position")
            except BaseException as exc:
                report.setdefault("cleanup_errors",[]).append(f"final readback: {exc}")
            transport.close()
        if report.get("torque_enabled_after") != [0]*6 or report.get("cleanup_errors"):
            report["status"] = "FAIL_CLEANUP_UNCONFIRMED"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","actual_command_path_raw","termination_reason","result","failure","cleanup_errors","torque_enabled_after") if k in report},ensure_ascii=False),flush=True)
    return 0 if failure is None and report["status"]=="PASS_PENDING_OPERATOR_OBSERVATION" else 1


if __name__=="__main__":
    raise SystemExit(main())
