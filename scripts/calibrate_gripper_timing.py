#!/usr/bin/env python3
"""CAL-091/092: one cycle from an explicitly approved three-cycle EMPTY batch.

Only servo6 goals. No EEPROM/profile edits, retries or contact-force claims.
Abort at any current/range/tracking/timing anomaly and unload/restore first.
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
from scripts.calibrate_gripper_empty_current import SETTINGS
from scripts.calibrate_gripper_open_speed import _sample
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY
from scripts.read_servo_eeprom import sha256_file, usb_identity

APPROVAL = "批准，CAL-091/092空爪时序首轮就绪，后续2、3轮无需申请直接执行即可"
PROFILE_SHA = "aabb213608b99f707433f1c7add925fcd7b0972c53aafbb7acdacee83a6af85d"
OLD_PROFILE_SHA = "1ea15d62d288bc02f56ccabe09593f60d68ad3fd6f5f37632589995dc7c18bb4"


def ramp_target(start: int, end: int, speed: float, tick_index: int) -> int:
    """Integer quantization of the approved per-tick average command slope."""
    distance = min(abs(end - start), speed * 1398 / 100 / 30 * tick_index)
    return round(start + (distance if end >= start else -distance))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.operator_confirmation != APPROVAL:
        raise SystemExit("Evidence exists or approved batch text missing")
    plan = json.loads(args.session.read_text())
    expected = {
        "parameter_ids": ["CAL-091", "CAL-092"], "servo_id": 6,
        "profile_sha256_current": PROFILE_SHA, "reference_capture_profile_sha256": OLD_PROFILE_SHA,
        "reference_report": "CAL-089_090_empty_close_repeat3_attempt1_20260917.json",
        "intentional_profile_delta_report": "CAL-089_090_demo_integration_final_20260917.json",
        "operator_confirmation": APPROVAL, "approved_cycles": 3, "approved_repeats": [1, 2, 3],
        "expected_start_raw": 1502, "start_max_deviation_raw": 5, "start_accepted_raw": [1499, 1507],
        "static_read_frames": 10, "all_six_static_span_max_raw": 2, "source_arm_max_deviation_raw": 10,
        "all_axis_raw_domain_required": True, "open_target_raw": 2268, "open_target_pct": 55,
        "open_speed_pct_s": 20, "close_speed_pct_s": 15, "close_lower_command_limit_raw": 1499,
        "empty_boundary_pct": 1.21, "sample_fps": 30, "sram_settings": SETTINGS,
        "feedback_raw_bounds": [1499, 2897], "all_phase_abort_abs_current_ma": 20,
        "open_tracking_abort_raw": 120, "close_tracking_abort_raw": 80,
        "leg_timeout_s": {"opening": 6, "closing": 6}, "whole_powered_timeout_s": 15,
        "arm_goal_commands_authorized": 0, "eeprom_writes_authorized": False, "profile_writes_authorized": False,
        "opening_completion_runtime_criterion": {"position_error_max_pct": 2, "feedback_speed_max_pct_s": 3, "continuous_dwell_s": .15},
        "opening_completion_additional_diagnostic_criterion": {"position_error_max_raw": 5, "feedback_raw_speed_required": 0, "continuous_dwell_s": .15},
    }
    if any(plan.get(k) != v for k, v in expected.items()):
        raise SystemExit("Plan differs from approved bounded empty batch")
    expected_output = f"CAL-091_092_empty_timing_repeat{args.repeat}_attempt1_20260917.json"
    if args.output.name != expected_output:
        raise SystemExit("Unexpected output evidence name")
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != PROFILE_SHA:
        raise SystemExit("Profile fingerprint changed")
    raw_profile = json.loads(profile_path.read_text())
    profile = load_calibration_profile(profile_path)
    if profile.joints is None:
        raise SystemExit("Missing joint mapping")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if (mapping.gripper_closed_raw, mapping.gripper_open_raw) != (1499, 2897):
        raise SystemExit("Endpoint mapping changed")
    for target in (1499, 2268):
        if not gripper.range_min <= target <= gripper.range_max:
            raise SystemExit("Goal outside calibration limits")
    if (raw_profile['timing']['fps'] != 30 or raw_profile['gripper']['empty_closed_pct'] != .21
            or raw_profile['gripper']['empty_tol_pct'] != 1):
        raise SystemExit("Timing/empty gate changed")
    reports_dir = profile_path.parent / "reports"
    integration = json.loads((reports_dir / plan['intentional_profile_delta_report']).read_text())
    source_path = reports_dir / plan['reference_report']
    source = json.loads(source_path.read_text())
    source_manifest = next(r for r in integration['sources'] if r['report'] == source_path.name)
    if (integration['profile_sha256_before'] != OLD_PROFILE_SHA
            or integration['profile_sha256_after'] != PROFILE_SHA
            or integration['only_changed_fields'] != {
                'gripper.empty_closed_pct': {'before': 3, 'after': .21},
                'gripper.empty_tol_pct': {'before': 2, 'after': 1}}
            or not integration['all_other_profile_fields_preserved']
            or sha256_file(source_path) != source_manifest['reviewed_report_sha256']
            or source['profile_sha256'] != OLD_PROFILE_SHA
            or source['post_capture_review']['operator_observation'] != '无异常'
            or source['torque_enabled_after'] != [0] * 6
            or source['sram_settings_after'] != source['original_sram_settings']):
        raise SystemExit("Reference or intentional profile delta evidence changed")
    expected_start = 1502
    start_range = [1499, 1507]
    previous = None
    if args.repeat > 1:
        previous_name = f"CAL-091_092_empty_timing_repeat{args.repeat-1}_attempt1_20260917.json"
        previous = json.loads((reports_dir / previous_name).read_text())
        if (previous['status'] != 'PASS_PENDING_OPERATOR_OBSERVATION'
                or previous['profile_sha256'] != PROFILE_SHA
                or previous['torque_enabled_after'] != [0] * 6
                or previous['sram_settings_after'] != previous['original_sram_settings']
                or previous.get('cleanup_errors')):
            raise SystemExit("Previous cycle failed; do not continue/retry")
        expected_start = previous['gripper_position_after_torque_off_raw']
        if not 1499 <= expected_start <= 1515:
            raise SystemExit("Previous unloaded jaw outside empty gate; no reposition/retry")
        start_range = [max(1499, expected_start - 5), min(1515, expected_start + 5)]
    # Before every access independently verify stable path, USB identity and occupancy.
    port = Path(profile.motor.port)
    if str(port) != EXPECTED_STABLE_PORT or str(port) != plan['serial_stable_path']:
        raise SystemExit("Stable port differs")
    resolved = port.resolve(strict=True)
    if not resolved.name.startswith('ttyACM') or usb_identity(resolved) != EXPECTED_USB_IDENTITY:
        raise SystemExit("USB/ttyACM identity mismatch")
    occupied = subprocess.run(['fuser', str(port)], capture_output=True, text=True)
    if occupied.returncode != 1 or occupied.stdout.strip():
        raise SystemExit("Port busy or occupancy check failed")
    io.TRANSACTION_TIMEOUT_S = .25
    report = {
        'schema_version': 1, 'parameter_ids': ['CAL-091', 'CAL-092'], 'repeat': args.repeat, 'attempt': 1,
        'status': 'RUNNING', 'started_at': datetime.now().astimezone().isoformat(),
        'profile_sha256': PROFILE_SHA, 'approved_plan': plan, 'operator_confirmation': APPROVAL,
        'source_report_sha256': sha256_file(source_path), 'expected_start_raw': expected_start,
        'start_accepted_raw': start_range, 'stable_port': str(port), 'resolved_port': str(resolved),
        'usb_identity': usb_identity(resolved), 'transaction_timeout_s': .25,
        'environment': {'power_v': 12, 'ambient_temperature_c': None, 'base_mount': 'fixed to wooden board', 'load': 'empty'},
        'joint_servo_motion_commands_sent': 0, 'gripper_motion_commands_sent': 0,
        'eeprom_written': False, 'profile_written': False, 'samples': [],
    }
    protocol = transport = None
    originals = {}
    torque_may_be_on = sram_may_be_changed = False
    motion_start = phase_start = None
    failure = None

    def ensure_deadlines(leg: bool = True) -> None:
        now = time.monotonic()
        if motion_start is not None and now - motion_start >= 15:
            raise RuntimeError('15s total powered deadline')
        if leg and phase_start is not None and now - phase_start >= 6:
            raise RuntimeError('6s leg deadline')

    def checked_sample(phase: str, command: int, tracking_cap: int) -> dict:
        row = _sample(protocol, mapping, gripper)
        row.update(phase=phase, command_raw=command, tracking_error_raw=command-row['position_raw'])
        report['samples'].append(row)  # preserve raw failures, never clip
        if abs(row['current_ma']) > 20:
            raise RuntimeError(f"20mA diagnostic current abort: {row['current_ma']}mA")
        row['gripper_pct'] = mapping.raw_to_gripper_pct(row['position_raw'])
        if abs(row['tracking_error_raw']) > tracking_cap:
            raise RuntimeError(f'Tracking exceeded{tracking_cap}raw')
        ensure_deadlines()
        return row

    def submit(raw: int) -> None:
        ensure_deadlines()
        if not 1499 <= raw <= 2268:
            raise RuntimeError('Goal outside approved diagnostic path')
        io._sync_goal(protocol, mapping, gripper, raw)
        report['gripper_motion_commands_sent'] += 1

    try:
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
        if ids != [1, 2, 3, 4, 5, 6] or models != [777] * 6 or torques != [0] * 6:
            raise RuntimeError('Identity/torque mismatch; no writes')
        report['readonly_eeprom'] = {}
        for axis in mapping.axes:
            minimum = io._read_raw(protocol, mapping, axis, 'Min_Position_Limit')
            maximum = io._read_raw(protocol, mapping, axis, 'Max_Position_Limit')
            offset = mapping.decode_signed(axis, 'Homing_Offset', io._read_raw(protocol, mapping, axis, 'Homing_Offset'))
            report['readonly_eeprom'][axis.name] = {'range_min': minimum, 'range_max': maximum, 'homing_offset': offset}
            if (minimum, maximum, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError(f'EEPROM differs {axis.name}; no writes')
        report['readonly_gripper_controller_settings'] = {
            r: io._read_raw(protocol, mapping, gripper, r) for r in
            ('P_Coefficient', 'D_Coefficient', 'I_Coefficient', 'Minimum_Startup_Force', 'CW_Dead_Zone', 'CCW_Dead_Zone')}
        if report['readonly_gripper_controller_settings'] != source['readonly_gripper_controller_settings']:
            raise RuntimeError('Controller changed; no writes')
        frames = []
        for index in range(10):
            frames.append([io._read_raw(protocol, mapping, a, 'Present_Position') for a in mapping.axes])
            if index < 9:
                time.sleep(.05)
        spans = [max(f[i] for f in frames)-min(f[i] for f in frames) for i in range(6)]
        reference = source['fresh_readonly_baseline']['raw_position_by_sample'][0][:5]
        deviations = [max(abs(f[i]-reference[i]) for f in frames) for i in range(5)]
        after = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
        report['fresh_readonly_baseline'] = {'raw_position_by_sample': frames, 'raw_position_static_span': spans, 'torque_enabled_after': after}
        report['arm_deviation_from_reference_raw'] = deviations
        if (any(s > 2 for s in spans) or any(d > 10 for d in deviations) or after != [0] * 6
                or any(not a.range_min <= f[i] <= a.range_max for f in frames for i, a in enumerate(mapping.axes))
                or any(not start_range[0] <= f[5] <= start_range[1] for f in frames)):
            raise RuntimeError('Fresh source/static/domain/start mismatch; no writes')
        fresh = int(statistics.median(f[5] for f in frames))
        initial = checked_sample('readonly_initial', fresh, 5)
        start = initial['position_raw']
        if abs(start-fresh) > 2 or not start_range[0] <= start <= start_range[1]:
            raise RuntimeError('Fresh jaw moved; no writes')
        originals = {r: io._read_raw(protocol, mapping, gripper, r) for r in SETTINGS}
        report['original_sram_settings'] = originals.copy()
        if originals not in (SETTINGS, {**SETTINGS, 'Acceleration': 0}):
            raise RuntimeError('Unexpected SRAM; no writes')
        if originals['Acceleration'] != 5:
            sram_may_be_changed = True
            io._write_raw(protocol, mapping, gripper, 'Acceleration', 5)
            time.sleep(.03)
        report['test_sram_settings'] = {r: io._read_raw(protocol, mapping, gripper, r) for r in SETTINGS}
        if report['test_sram_settings'] != SETTINGS:
            raise RuntimeError('SRAM preparation mismatch')
        print(f'CAL-091/092 repeat{args.repeat}: preflight passed, only gripper opening then empty closing', flush=True)
        tick = 1/30
        lateness = min(.02, float(raw_profile['timing']['max_tick_lateness_s']))
        torque_may_be_on = True  # MUST precede first goal (can implicitly enable)
        motion_start = phase_start = time.monotonic()
        submit(start)
        io._write_raw(protocol, mapping, gripper, 'Torque_Enable', 1)
        time.sleep(.03)
        checked_sample('initial_hold', start, 120)
        base = time.monotonic()
        runtime_since = strict_since = None
        runtime_elapsed = None
        index = 0
        while True:
            index += 1
            due = base+(index-1)*tick
            time.sleep(max(0, due-time.monotonic()))
            if time.monotonic()-due > lateness:
                raise RuntimeError('Opening tick lateness')
            target = ramp_target(start, 2268, 20, index)
            submit(target)
            time.sleep(max(0, due+tick-time.monotonic()))
            row = checked_sample('empty_opening', target, 120)
            now = row['monotonic_ns']/1e9
            runtime_ok = abs(row['gripper_pct']-55) <= 2 and abs(row['velocity_pct_s']) <= 3
            strict_ok = abs(row['position_raw']-2268) <= 5 and row['velocity_register_raw'] == 0
            runtime_since = (now if runtime_since is None else runtime_since) if runtime_ok else None
            strict_since = (now if strict_since is None else strict_since) if strict_ok else None
            if runtime_elapsed is None and runtime_since is not None and now-runtime_since >= .15:
                runtime_elapsed = now-phase_start
            if target == 2268 and runtime_ok and strict_since is not None and now-strict_since >= .15:
                if runtime_elapsed is None:
                    raise RuntimeError('Strict completion unexpectedly before runtime criterion')
                report['opening_result'] = {'runtime_completion_elapsed_s': runtime_elapsed, 'strict_completion_elapsed_s': now-phase_start,
                    'final_position_raw': row['position_raw'], 'final_feedback_pct': row['gripper_pct'], 'final_velocity_pct_s': row['velocity_pct_s']}
                break
        print(json.dumps({'repeat': args.repeat, 'opening': report['opening_result']}, ensure_ascii=False), flush=True)
        # Begin closing from CURRENT measured feedback, like _close_until_contact.
        # Hold no additional pressure at the empty gate; first crossing exits.
        close_start = row['position_raw']
        phase_start = time.monotonic()
        base = phase_start
        index = 0
        while True:
            index += 1
            due = base+(index-1)*tick
            time.sleep(max(0, due-time.monotonic()))
            if time.monotonic()-due > lateness:
                raise RuntimeError('Closing tick lateness')
            target = ramp_target(close_start, 1499, 15, index)
            submit(target)
            time.sleep(max(0, due+tick-time.monotonic()))
            row = checked_sample('empty_closing', target, 80)
            if row['gripper_pct'] <= 1.21:
                report['closing_result'] = {'empty_gate_elapsed_s': row['monotonic_ns']/1e9-phase_start,
                    'final_position_raw': row['position_raw'], 'final_feedback_pct': row['gripper_pct'], 'last_command_raw': target,
                    'closed_command_floor_reached': target == 1499, 'hold_added': False}
                break
        io._write_raw(protocol, mapping, gripper, 'Torque_Enable', 0)
        report['powered_duration_s'] = time.monotonic()-motion_start
        report['termination_reason'] = 'EMPTY_GATE_FIRST_CROSSING_IMMEDIATE_UNLOAD'
        report['actual_command_path_raw'] = [start, 2268, report['closing_result']['last_command_raw']]
        report['result'] = {'opening_s': report['opening_result']['strict_completion_elapsed_s'], 'closing_s': report['closing_result']['empty_gate_elapsed_s'],
            'peak_abs_current_ma': max(abs(r['current_ma']) for r in report['samples']),
            'max_open_tracking_error_raw': max(abs(r['tracking_error_raw']) for r in report['samples'] if r['phase']=='empty_opening'),
            'max_close_tracking_error_raw': max(abs(r['tracking_error_raw']) for r in report['samples'] if r['phase']=='empty_closing')}
        report['status'] = 'PASS_PENDING_OPERATOR_OBSERVATION'
    except BaseException as exc:
        failure = exc
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['status'] = 'FAIL'
    finally:
        if protocol is not None:
            if torque_may_be_on:
                for axis in [gripper]+[a for a in mapping.axes if a.servo_id != 6]:
                    try:
                        io._write_raw(protocol, mapping, axis, 'Torque_Enable', 0)
                        if axis.servo_id == 6 and 'powered_duration_s' not in report:
                            report['powered_duration_s'] = time.monotonic()-motion_start
                        time.sleep(.03)
                    except BaseException as exc:
                        report.setdefault('cleanup_errors', []).append(f'disable {axis.name}: {exc}')
            if torque_may_be_on or sram_may_be_changed:
                for register, value in originals.items():
                    try:
                        io._write_raw(protocol, mapping, gripper, register, value)
                        time.sleep(.03)
                        if io._read_raw(protocol, mapping, gripper, register) != value:
                            raise RuntimeError('Restore mismatch')
                    except BaseException as exc:
                        report.setdefault('cleanup_errors', []).append(f'restore {register}: {exc}')
            try:
                report['torque_enabled_after'] = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
                report['sram_settings_after'] = {r: io._read_raw(protocol, mapping, gripper, r) for r in originals}
                report['gripper_position_after_torque_off_raw'] = io._read_raw(protocol, mapping, gripper, 'Present_Position')
            except BaseException as exc:
                report.setdefault('cleanup_errors', []).append(f'final readback: {exc}')
            transport.close()
        if report.get('torque_enabled_after') != [0]*6 or report.get('cleanup_errors'):
            report['status'] = 'FAIL_CLEANUP_UNCONFIRMED'
        report['finished_at'] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'result', 'failure', 'cleanup_errors', 'torque_enabled_after', 'gripper_motion_commands_sent', 'gripper_position_after_torque_off_raw') if k in report}, ensure_ascii=False), flush=True)
    return 0 if failure is None and report['status'] == 'PASS_PENDING_OPERATOR_OBSERVATION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
