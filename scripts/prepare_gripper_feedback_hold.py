#!/usr/bin/env python3
"""One explicitly approved CAL-085 EMPTY opening preparation, never closure.

Only6 goals,20pct/s to55pct, current20mA/tracking120raw/6s diagnostic bounds.
No fixture/arm goals/EEPROM/profile writes, no retries or feedback-force hold.
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
from scripts.calibrate_gripper_empty_current import SETTINGS
from scripts.calibrate_gripper_open_speed import _sample
from scripts.calibrate_gripper_timing import ramp_target
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY
from scripts.read_servo_eeprom import sha256_file, usb_identity

APPROVAL = "批准，CAL-085反馈保持基准空爪张开就绪"
PROFILE_SHA = "fe6aaf95e8418975d6eb966fe20ad55a8743084f63c443474aa838c21beffb2f"
TIMING_PROFILE_SHA = "aabb213608b99f707433f1c7add925fcd7b0972c53aafbb7acdacee83a6af85d"
INTERMEDIATE_SHA = "4a1782e01bab930fd19e4b557a831dd9b406ca3fddc889480b1fc6a1dd30f7a2"
OUTPUT_NAME = "CAL-085_feedback_hold_empty_preparation_attempt1_20260917.json"


def validate_plan(plan: dict) -> None:
    expected = {
        "parameter_id": "CAL-085", "dependent_next_parameter": "CAL-093",
        "status": "APPROVED_ONE_EMPTY_OPENING_PREPARATION", "required_confirmation": APPROVAL,
        "operator_confirmation": APPROVAL, "profile_sha256_current": PROFILE_SHA,
        "reference_report": "CAL-091_092_empty_timing_repeat3_attempt1_20260917.json",
        "reference_capture_profile_sha256": TIMING_PROFILE_SHA,
        "intentional_delta_reports": ["CAL-091_092_demo_integration_final_20260917.json", "CAL-096_097_empty_settle_derivation_20260917.json"],
        "serial_stable_path": EXPECTED_STABLE_PORT, "expected_usb": EXPECTED_USB_IDENTITY,
        "servo_id": 6, "approved_cycles": 1, "proposed_cycles": 1,
        "expected_start_raw": 1511, "start_max_deviation_raw": 5, "start_accepted_raw": [1506, 1515],
        "static_read_frames": 10, "all_six_static_span_max_raw": 2, "source_arm_max_deviation_raw": 10,
        "all_axis_raw_domain_required": True, "open_target_pct": 55, "open_target_raw": 2268,
        "opening_command_speed_pct_s": 20, "sample_fps": 30,
        "opening_completion": {"position_tol_pct": .6, "speed_tol_pct_s": .3, "continuous_dwell_s": .15},
        "sram_settings": SETTINGS, "feedback_bounds_raw": [1499, 2897], "max_tracking_error_raw": 120,
        "all_phase_abs_current_abort_ma": 20, "whole_powered_timeout_s": 6,
        "closing_goals_authorized": False, "fixture_present_during_motion": False,
        "arm_goals_authorized": 0, "eeprom_writes_authorized": False, "profile_writes_authorized": False,
        "automatic_retry_authorized": False,
    }
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Plan differs from exact approved empty opening")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.name != OUTPUT_NAME or args.operator_confirmation != APPROVAL:
        raise SystemExit("Evidence exists, output differs or exact approval missing")
    plan = json.loads(args.session.read_text())
    validate_plan(plan)
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != PROFILE_SHA:
        raise SystemExit("Current profile fingerprint changed")
    raw_profile = json.loads(profile_path.read_text())
    profile = load_calibration_profile(profile_path)
    if profile.joints is None:
        raise SystemExit("Joint mapping missing")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if (mapping.gripper_closed_raw, mapping.gripper_open_raw) != (1499, 2897):
        raise SystemExit("Endpoint mapping changed")
    if not gripper.range_min <= 2268 <= gripper.range_max:
        raise SystemExit("Goal outside authoritative calibration domain")
    if (raw_profile['timing']['fps'] != 30 or raw_profile['gripper']['settle_tol_pct'] != .6
            or raw_profile['gripper']['settle_speed_pct_s'] != .3 or raw_profile['motion']['settle_dwell_s'] != .15):
        raise SystemExit("Completion criterion/rate changed")
    reports_dir = profile_path.parent / "reports"
    timing = json.loads((reports_dir / plan['intentional_delta_reports'][0]).read_text())
    settle = json.loads((reports_dir / plan['intentional_delta_reports'][1]).read_text())
    source_path = reports_dir / plan['reference_report']
    source = json.loads(source_path.read_text())
    manifest = next(row for row in timing['sources'] if row['report'] == source_path.name)
    if (timing['status'] != 'CONFIRMED_DEMO_INTEGRATED'
            or timing['profile_sha256_before'] != TIMING_PROFILE_SHA or timing['profile_sha256_after'] != INTERMEDIATE_SHA
            or timing['only_changed_fields'] != {'gripper.close_timeout_s': {'before': 6, 'after': 4.7}, 'gripper.open_timeout_s': {'before': 6, 'after': 4.3}}
            or settle['status'] != 'CONFIRMED_DEMO_INTEGRATED'
            or settle['profile_sha256_before'] != INTERMEDIATE_SHA or settle['profile_sha256_after'] != PROFILE_SHA
            or settle['only_changed_fields'] != {'gripper.settle_tol_pct': {'before': 2, 'after': .6}, 'gripper.settle_speed_pct_s': {'before': 3, 'after': .3}}
            or sha256_file(source_path) != manifest['source_sha256_after_operator_review']
            or source['profile_sha256'] != TIMING_PROFILE_SHA
            or source['status'] != 'PASS_PENDING_OPERATOR_OBSERVATION'
            or source['post_capture_review']['operator_observation'] != '无异常'
            or source['torque_enabled_after'] != [0]*6 or source['sram_settings_after'] != source['original_sram_settings']
            or source['test_sram_settings'] != SETTINGS):
        raise SystemExit("Source evidence/configuration review lineage differs")
    report = {
        'schema_version': 1, 'parameter_id': 'CAL-085', 'dependent_parameter_id': 'CAL-093',
        'test': 'single_empty_opening_preparation_no_closure', 'status': 'RUNNING',
        'started_at': datetime.now().astimezone().isoformat(), 'profile_sha256': PROFILE_SHA,
        'approved_plan': plan, 'operator_confirmation': APPROVAL, 'source_report_sha256': sha256_file(source_path),
        'joint_servo_motion_commands_sent': 0, 'gripper_motion_commands_sent': 0,
        'eeprom_written': False, 'profile_written': False, 'closing_goals_sent': 0, 'samples': [],
        'transaction_timeout_s': .25,
        'environment': {'power_v': 12, 'ambient_temperature_c': None, 'base_mount': 'fixed to wooden board', 'load': 'empty gripper'},
    }
    io.TRANSACTION_TIMEOUT_S = .25
    protocol = transport = None
    originals = {}
    torque_may_be_on = sram_may_be_changed = False
    motion_start = None
    failure = None

    def deadline_check() -> None:
        if motion_start is not None and time.monotonic()-motion_start >= 6:
            raise RuntimeError('6s powered opening deadline')

    def checked_sample(phase: str, command: int, cap: int = 120) -> dict:
        row = _sample(protocol, mapping, gripper)
        row.update(phase=phase, command_raw=command, tracking_error_raw=command-row['position_raw'])
        report['samples'].append(row)  # retain raw failure evidence before strict domain check
        if abs(row['current_ma']) > 20:
            raise RuntimeError(f"20mA diagnostic current abort: {row['current_ma']}mA")
        row['gripper_pct'] = mapping.raw_to_gripper_pct(row['position_raw'])
        if abs(row['tracking_error_raw']) > cap:
            raise RuntimeError(f'Tracking exceeded{cap}raw')
        deadline_check()
        return row

    try:
        # Recheck stable identity and occupancy immediately before this access.
        port = Path(profile.motor.port)
        if str(port) != EXPECTED_STABLE_PORT:
            raise RuntimeError('Stable port differs')
        resolved = port.resolve(strict=True)
        identity = usb_identity(resolved)
        if not resolved.name.startswith('ttyACM') or identity != EXPECTED_USB_IDENTITY:
            raise RuntimeError('USB/ttyACM identity differs')
        report.update(stable_port=str(port), resolved_port=str(resolved), usb_identity=identity)
        occupancy = subprocess.run(['fuser', str(port)], capture_output=True, text=True)
        if occupancy.returncode != 1 or occupancy.stdout.strip():
            raise RuntimeError('Serial occupancy check failed or device busy')
        transport = SerialTransport(str(port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        for axis in mapping.axes:
            level = io._read_raw(protocol, mapping, axis, 'Response_Status_Level')
            if level not in (0, 1):
                raise RuntimeError('Unexpected response level')
            protocol.set_write_response(axis.servo_id, expects_ack=level == 0)
        ids = [io._read_raw(protocol, mapping, a, 'ID') for a in mapping.axes]
        models = [io._read_raw(protocol, mapping, a, 'Model_Number') for a in mapping.axes]
        torques = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
        report['readonly_preflight'] = {'ids': ids, 'models': models, 'torques': torques}
        if ids != [1,2,3,4,5,6] or models != [777]*6 or torques != [0]*6:
            raise RuntimeError('ID/model/torque differs; no writes')
        report['readonly_eeprom'] = {}
        for axis in mapping.axes:
            minimum = io._read_raw(protocol, mapping, axis, 'Min_Position_Limit')
            maximum = io._read_raw(protocol, mapping, axis, 'Max_Position_Limit')
            offset = mapping.decode_signed(axis, 'Homing_Offset', io._read_raw(protocol, mapping, axis, 'Homing_Offset'))
            report['readonly_eeprom'][axis.name] = {'range_min': minimum, 'range_max': maximum, 'homing_offset': offset}
            if (minimum, maximum, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f'EEPROM mismatch {axis.name}; no writes')
        report['readonly_gripper_controller_settings'] = {r: io._read_raw(protocol, mapping, gripper, r) for r in
            ('P_Coefficient','D_Coefficient','I_Coefficient','Minimum_Startup_Force','CW_Dead_Zone','CCW_Dead_Zone')}
        if report['readonly_gripper_controller_settings'] != source['readonly_gripper_controller_settings']:
            raise RuntimeError('Controller differs; no writes')
        frames = []
        for index in range(10):
            frames.append([io._read_raw(protocol, mapping, a, 'Present_Position') for a in mapping.axes])
            if index < 9:
                time.sleep(.05)
        spans = [max(f[i] for f in frames)-min(f[i] for f in frames) for i in range(6)]
        reference = source['fresh_readonly_baseline']['raw_position_by_sample'][0][:5]
        deviations = [max(abs(f[i]-reference[i]) for f in frames) for i in range(5)]
        torques_after = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
        report['fresh_readonly_baseline'] = {'raw_position_by_sample': frames, 'raw_position_static_span': spans, 'torque_enabled_after': torques_after}
        report['arm_deviation_from_reference_raw'] = deviations
        if (any(s>2 for s in spans) or any(d>10 for d in deviations) or torques_after != [0]*6
                or any(not a.range_min<=f[i]<=a.range_max for f in frames for i,a in enumerate(mapping.axes))
                or any(not 1506<=f[5]<=1515 for f in frames)):
            raise RuntimeError('Fresh start/static/source/domain differs; no writes')
        fresh = int(statistics.median(f[5] for f in frames))
        start = checked_sample('readonly_initial', fresh, 5)['position_raw']
        if abs(start-fresh)>2 or not 1506<=start<=1515:
            raise RuntimeError('Fresh jaw moved; no writes')
        originals = {r: io._read_raw(protocol, mapping, gripper, r) for r in SETTINGS}
        report['original_sram_settings'] = originals.copy()
        if originals not in (SETTINGS, {**SETTINGS,'Acceleration':0}):
            raise RuntimeError('Unexpected SRAM; no motion')
        if originals['Acceleration'] != 5:
            sram_may_be_changed = True
            io._write_raw(protocol, mapping, gripper, 'Acceleration', 5)
            time.sleep(.03)
        report['test_sram_settings'] = {r: io._read_raw(protocol, mapping, gripper, r) for r in SETTINGS}
        if report['test_sram_settings'] != SETTINGS:
            raise RuntimeError('SRAM preparation mismatch')
        print('CAL-085 preparation preflight passed; only EMPTY opening to55pct, no closing', flush=True)
        torque_may_be_on = True  # first goal may implicitly enable torque
        motion_start = time.monotonic()
        io._sync_goal(protocol, mapping, gripper, start)
        report['gripper_motion_commands_sent'] += 1
        io._write_raw(protocol, mapping, gripper, 'Torque_Enable', 1)
        time.sleep(.03)
        checked_sample('initial_hold', start)
        tick = 1/30
        lateness = min(.02,float(raw_profile['timing']['max_tick_lateness_s']))
        base = time.monotonic()
        settled_since = None
        previous = start
        index = 0
        while True:
            index += 1
            due = base+(index-1)*tick
            time.sleep(max(0,due-time.monotonic()))
            deadline_check()
            if time.monotonic()-due>lateness:
                raise RuntimeError('Opening tick lateness')
            target = ramp_target(start,2268,20,index)
            if not start<=previous<=target<=2268:
                raise RuntimeError('Refuse closing/out-of-path goal')
            io._sync_goal(protocol,mapping,gripper,target)
            report['gripper_motion_commands_sent'] += 1
            previous = target
            time.sleep(max(0,due+tick-time.monotonic()))
            row = checked_sample('empty_opening',target)
            now = row['monotonic_ns']/1e9
            good = abs(row['gripper_pct']-55)<=.6 and abs(row['velocity_pct_s'])<=.3
            settled_since = (now if settled_since is None else settled_since) if good else None
            if target==2268 and settled_since is not None and now-settled_since>=.15:
                report['result'] = {'final_command_raw':2268,'final_feedback_raw':row['position_raw'],'final_feedback_pct':row['gripper_pct'],
                    'final_velocity_pct_s':row['velocity_pct_s'],'position_error_pct':abs(row['gripper_pct']-55),
                    'continuous_settle_span_s':now-settled_since,'completion_elapsed_s':now-motion_start,
                    'peak_abs_current_ma':max(abs(r['current_ma']) for r in report['samples']),
                    'max_tracking_error_raw':max(abs(r['tracking_error_raw']) for r in report['samples'])}
                break
        io._write_raw(protocol,mapping,gripper,'Torque_Enable',0)
        report['powered_duration_s'] = time.monotonic()-motion_start
        report['actual_command_path_raw'] = [start,2268]
        report['termination_reason'] = 'OPENING_SETTLED_IMMEDIATE_UNLOAD_NO_CLOSURE'
        report['status'] = 'PASS_PENDING_OPERATOR_OBSERVATION'
    except BaseException as exc:
        failure = exc
        report['status'] = 'FAIL' if protocol is not None else 'FAIL_PRE_ACCESS'
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        if protocol is not None:
            if torque_may_be_on:
                for axis in [gripper]+[a for a in mapping.axes if a.servo_id!=6]:
                    try:
                        io._write_raw(protocol,mapping,axis,'Torque_Enable',0)
                        if axis.servo_id==6 and 'powered_duration_s' not in report:
                            report['powered_duration_s'] = time.monotonic()-motion_start
                        time.sleep(.03)
                    except BaseException as exc:
                        report.setdefault('cleanup_errors',[]).append(f'disable {axis.name}: {exc}')
            if torque_may_be_on or sram_may_be_changed:
                for register,value in originals.items():
                    try:
                        io._write_raw(protocol,mapping,gripper,register,value)
                        time.sleep(.03)
                        if io._read_raw(protocol,mapping,gripper,register)!=value:
                            raise RuntimeError('SRAM restore readback mismatch')
                    except BaseException as exc:
                        report.setdefault('cleanup_errors',[]).append(f'restore {register}: {exc}')
            try:
                report['torque_enabled_after'] = [io._read_raw(protocol,mapping,a,'Torque_Enable') for a in mapping.axes]
                report['sram_settings_after'] = {r:io._read_raw(protocol,mapping,gripper,r) for r in originals}
                report['gripper_position_after_torque_off_raw'] = io._read_raw(protocol,mapping,gripper,'Present_Position')
            except BaseException as exc:
                report.setdefault('cleanup_errors',[]).append(f'final readback: {exc}')
            if report.get('torque_enabled_after') != [0]*6 or report.get('cleanup_errors'):
                report['status'] = 'FAIL_CLEANUP_UNCONFIRMED'
        if transport is not None:
            transport.close()
        report['finished_at'] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','result','failure','cleanup_errors','torque_enabled_after','gripper_motion_commands_sent','gripper_position_after_torque_off_raw') if k in report},ensure_ascii=False),flush=True)
    return 0 if failure is None and report['status']=='PASS_PENDING_OPERATOR_OBSERVATION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
