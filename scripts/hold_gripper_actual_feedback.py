#!/usr/bin/env python3
"""CAL-085: one approved fixed actual-feedback hold, no approach or retry.

This isolated calibration primitive does NOT qualify the production contact
candidate or bypass the real-mode draft guard. Only servo6 goals, <=0.3s powered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

APPROVAL = '批准，CAL-085实测反馈短保持就绪'
PROFILE_SHA = 'fe6aaf95e8418975d6eb966fe20ad55a8743084f63c443474aa838c21beffb2f'
SOURCE_SHA = '85dfdc9405aa5cf8257fed26a1016e8ad96e8196c9ed7beb2e7223622d861c4b'
SAMPLES_SHA = 'c9008a22f8c37b849b2ceeb1153f38b74adaa0231d7a6da8464df4dfc106a361'
MOTOR_SHA = 'ad315322ad4d19cc322c8f21d86d9667d9df9eca2de325bebbd72fa62deefff5'
OUTPUT = 'CAL-085_direct_feedback_hold_fixture_attempt1_20260917.json'


def validate_plan(plan: dict) -> None:
    expected = {
        'parameter_id': 'CAL-085', 'dependent_parameter_id': 'CAL-093',
        'status': 'APPROVED_ONE_DIRECT_FEEDBACK_HOLD',
        'required_confirmation': APPROVAL, 'operator_confirmation': APPROVAL,
        'reference_report': 'CAL-085_floor1830_endpoint_confirmation_attempt1_20260917.json',
        'reference_report_sha256': SOURCE_SHA, 'reference_raw_samples_sha256': SAMPLES_SHA,
        'profile_sha256': PROFILE_SHA, 'motor_calibration_sha256': MOTOR_SHA,
        'source_operator_observation': '已夹紧', 'operator_confirmed_grip': True,
        'servo_id': 6, 'stable_port': EXPECTED_STABLE_PORT,
        'expected_resolved_port': '/dev/ttyACM0', 'expected_usb_identity': EXPECTED_USB_IDENTITY,
        'expected_start_raw': 1839, 'start_max_deviation_raw': 5,
        'start_accepted_raw': [1834, 1844], 'hold_target_bounds_raw': [1834, 1844],
        'all_six_static_read_frames': 10, 'gripper_static_span_max_raw': 2,
        'feedback_bounds_raw': [1499, 2897], 'sample_fps': 30,
        'maximum_feedback_hold_s': .3, 'whole_powered_timeout_s': .3,
        'max_tracking_error_raw': 10, 'all_phase_abs_current_abort_ma': 20,
        'max_tick_lateness_s': .02, 'transaction_timeout_ceiling_s': .25,
        'sram_settings': SETTINGS, 'proposed_cycles': 1, 'approved_cycles': 1,
        'authorized_physical_trials': 1, 'remaining_authorized_physical_trials': 1,
        'new_motion_commands': 0, 'cumulative_gripper_goals': 6768,
        'all_six_identity_eeprom_torque_off_checks_required': True,
        'gripper_controller_settings_check_required': True,
        'arm_static_span_is_diagnostic_only': True,
        'arm_source_deviation_is_diagnostic_only': True,
        'arm_raw_domain_is_diagnostic_only': True,
        'arm_goals_authorized': 0, 'closing_approach_goals_authorized': 0,
        'opening_preparation_goals_authorized': 0,
        'eeprom_writes_authorized': False, 'profile_writes_authorized': False,
        'candidate_contact_qualification_forced': False, 'production_grasp_authorized': False,
        'fixture_contact_band_mm': 20, 'fixture_independently_supported': True,
    }
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Plan differs from exact approved one-shot actual-feedback hold')


def check_row(row: dict, target: int) -> None:
    if not 1499 <= row['position_raw'] <= 2897:
        raise RuntimeError('Actual jaw position outside authoritative domain')
    if abs(row['current_ma']) > 20:
        raise RuntimeError('20mA diagnostic current abort')
    if abs(target-row['position_raw']) > 10:
        raise RuntimeError('Tracking exceeded10raw')


def powered_timeout(start: float, now: float) -> float:
    remaining = .3-(now-start)
    # Six transactions at most (goal, explicit enable, four sample reads),
    # reserving15ms for immediate jaw torque-off. Each stage recomputes this.
    if remaining <= .015:
        raise RuntimeError('0.3s powered hold deadline/cleanup reserve')
    return min(.25, (remaining-.015)/6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'session', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--operator-confirmation', required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.name != OUTPUT or args.operator_confirmation != APPROVAL:
        raise SystemExit('Existing evidence, wrong output or missing exact approval')
    plan = json.loads(args.session.read_text())
    validate_plan(plan)
    if sha256_file(args.profile) != PROFILE_SHA or sha256_file(args.profile.parent/'motor_calibration.json') != MOTOR_SHA:
        raise SystemExit('Authoritative configuration fingerprint changed')
    source_path = args.profile.parent/'reports'/plan['reference_report']
    source = json.loads(source_path.read_text())
    sample_sha = hashlib.sha256(json.dumps(source['samples'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (sha256_file(source_path) != SOURCE_SHA or sample_sha != SAMPLES_SHA
            or source['profile_sha256'] != PROFILE_SHA
            or not source['post_capture_review']['operator_confirmed_grip']
            or source['candidate_contact_qualified'] or source['feedback_hold_completed']
            or source['torque_enabled_after'] != [0]*6
            or source['sram_settings_after'] != source['original_sram_settings']):
        raise SystemExit('Reviewed grip source changed')
    profile = load_calibration_profile(args.profile)
    if profile.joints is None:
        raise SystemExit('Joint mapping missing')
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    jaw = mapping.axis('gripper')
    if (jaw.servo_id, mapping.gripper_closed_raw, mapping.gripper_open_raw) != (6, 1499, 2897):
        raise SystemExit('Jaw mapping changed')
    report = {
        'schema_version': 1, 'parameter_id': 'CAL-085', 'dependent_parameter_id': 'CAL-093',
        'test': 'single_fixed_actual_feedback_hold_no_approach', 'status': 'RUNNING',
        'started_at': datetime.now().astimezone().isoformat(), 'approved_plan': plan,
        'operator_confirmation': APPROVAL, 'profile_sha256': PROFILE_SHA,
        'source_report_sha256': SOURCE_SHA, 'source_raw_samples_sha256': SAMPLES_SHA,
        'joint_servo_motion_commands_sent': 0, 'gripper_motion_commands_sent': 0,
        'closing_goal_updates_sent': 0, 'opening_preparation_goal_updates_sent': 0,
        'feedback_hold_goal_updates_sent': 0, 'eeprom_written': False, 'profile_written': False,
        'candidate_contact_qualified': False, 'candidate_contact_qualification_forced': False,
        'feedback_hold_completed': False, 'samples': [],
        'environment': {'power_v': 12, 'ambient_temperature_c': None,
                        'base_mount': 'fixed to wooden board',
                        'load': 'independently supported plastic fixture,20mm contact band'},
    }
    protocol = transport = None
    originals = {}
    torque_may_be_on = sram_may_be_changed = False
    start = target = None
    failure = None
    io.TRANSACTION_TIMEOUT_S = .25
    try:
        port = Path(profile.motor.port)
        if str(port) != EXPECTED_STABLE_PORT:
            raise RuntimeError('Stable port differs')
        resolved = port.resolve(strict=True)
        identity = usb_identity(resolved)
        if str(resolved) != '/dev/ttyACM0' or identity != EXPECTED_USB_IDENTITY:
            raise RuntimeError('USB/ttyACM identity differs')
        occupancy = subprocess.run(['fuser', str(port)], capture_output=True, text=True)
        if occupancy.returncode != 1 or occupancy.stdout.strip():
            raise RuntimeError('Serial busy or occupancy check failed')
        report.update(stable_port=str(port), resolved_port=str(resolved), usb_identity=identity)
        transport = SerialTransport(str(port), profile.motor.baudrate)
        protocol = StsProtocol(transport)
        for axis in mapping.axes:
            level = io._read_raw(protocol, mapping, axis, 'Response_Status_Level')
            if level not in (0, 1):
                raise RuntimeError('Unexpected response level')
            protocol.set_write_response(axis.servo_id, expects_ack=level == 0)
        preflight = {key: [io._read_raw(protocol, mapping, a, reg) for a in mapping.axes]
                     for key, reg in [('ids', 'ID'), ('models', 'Model_Number'), ('torques', 'Torque_Enable')]}
        report['readonly_preflight'] = preflight
        if preflight != {'ids': [1,2,3,4,5,6], 'models': [777]*6, 'torques': [0]*6}:
            raise RuntimeError('ID/model/torque differs; no writes')
        report['readonly_eeprom'] = {}
        for axis in mapping.axes:
            lo = io._read_raw(protocol, mapping, axis, 'Min_Position_Limit')
            hi = io._read_raw(protocol, mapping, axis, 'Max_Position_Limit')
            offset = mapping.decode_signed(axis, 'Homing_Offset', io._read_raw(protocol, mapping, axis, 'Homing_Offset'))
            report['readonly_eeprom'][axis.name] = {'range_min': lo, 'range_max': hi, 'homing_offset': offset}
            if (lo, hi, offset) != (axis.range_min, axis.range_max, axis.homing_offset):
                raise RuntimeError('EEPROM mismatch; no writes')
        controllers = {r: io._read_raw(protocol, mapping, jaw, r) for r in
                       ('P_Coefficient','D_Coefficient','I_Coefficient','Minimum_Startup_Force','CW_Dead_Zone','CCW_Dead_Zone')}
        report['readonly_gripper_controller_settings'] = controllers
        if controllers != source['readonly_gripper_controller_settings']:
            raise RuntimeError('Jaw controller changed; no writes')
        frames = []
        for index in range(10):
            frames.append([io._read_raw(protocol, mapping, a, 'Present_Position') for a in mapping.axes])
            if index < 9:
                time.sleep(.05)
        spans = [max(f[i] for f in frames)-min(f[i] for f in frames) for i in range(6)]
        torques = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
        report['fresh_readonly_baseline'] = {'raw_position_by_sample': frames, 'raw_position_static_span': spans, 'torque_enabled_after': torques}
        report['arm_positions_outside_calibration_domain_diagnostic'] = [
            {'servo_id': a.servo_id, 'positions_raw': [f[i] for f in frames], 'range_raw': [a.range_min, a.range_max]}
            for i, a in enumerate(mapping.axes[:5]) if any(not a.range_min <= f[i] <= a.range_max for f in frames)]
        if torques != [0]*6 or spans[5] > 2 or any(not 1834 <= f[5] <= 1844 for f in frames):
            raise RuntimeError('Fresh jaw start/static differs; no writes')
        originals = {r: io._read_raw(protocol, mapping, jaw, r) for r in SETTINGS}
        report['original_sram_settings'] = originals.copy()
        if originals not in (SETTINGS, {**SETTINGS, 'Acceleration': 0}):
            raise RuntimeError('Unexpected SRAM; no motion')
        if originals['Acceleration'] != 5:
            sram_may_be_changed = True  # set before write: partial failures restore
            io._write_raw(protocol, mapping, jaw, 'Acceleration', 5)
            time.sleep(.03)
        report['test_sram_settings'] = {r: io._read_raw(protocol, mapping, jaw, r) for r in SETTINGS}
        if report['test_sram_settings'] != SETTINGS:
            raise RuntimeError('SRAM preparation mismatch')
        row = _sample(protocol, mapping, jaw)
        report['fresh_readonly_hold_reference'] = row.copy()
        target = row['position_raw']  # freeze ONCE, never rebase or use old1830
        check_row(row, target)
        if not 1834 <= target <= 1844 or abs(target-frames[-1][5]) > 2:
            raise RuntimeError('Fresh jaw moved before hold; no goals')
        report['hold_target_raw'] = target
        print(f'Preflight passed; only6 fixed actual-feedback target{target}, total powered <=0.3s', flush=True)
        torque_may_be_on = True  # SYNC goal can implicitly enable torque
        start = time.monotonic()  # includes FIRST goal and explicit enable
        report['powered_started_monotonic_ns'] = time.monotonic_ns()
        for index in range(9):
            due = start+index/30
            time.sleep(max(0, due-time.monotonic()))
            if time.monotonic()-due > .02:
                raise RuntimeError('Hold tick lateness')
            io.TRANSACTION_TIMEOUT_S = powered_timeout(start, time.monotonic())
            io._sync_goal(protocol, mapping, jaw, target)
            report['gripper_motion_commands_sent'] += 1
            report['feedback_hold_goal_updates_sent'] += 1
            if index == 0:
                io._write_raw(protocol, mapping, jaw, 'Torque_Enable', 1)
            io.TRANSACTION_TIMEOUT_S = powered_timeout(start, time.monotonic())
            row = _sample(protocol, mapping, jaw)
            row.update(phase='fixed_actual_feedback_hold', command_raw=target,
                       tracking_error_raw=target-row['position_raw'])
            report['samples'].append(row)  # preserve guard-failure evidence
            check_row(row, target)
            powered_timeout(start, time.monotonic())
        # Jaw first, within reserved budget; do not leave it powered for statistics.
        io.TRANSACTION_TIMEOUT_S = min(.01, max(.0001, .3-(time.monotonic()-start)))
        io._write_raw(protocol, mapping, jaw, 'Torque_Enable', 0)
        report['powered_duration_s'] = time.monotonic()-start
        if report['powered_duration_s'] >= .3:
            raise RuntimeError('Powered duration exceeded0.3s')
        report['feedback_hold_completed'] = True
        report['status'] = 'PASS_PENDING_OPERATOR_OBSERVATION'
        report['result'] = {'hold_target_raw': target, 'final_feedback_raw': row['position_raw'],
                            'final_velocity_pct_s': row['velocity_pct_s'],
                            'peak_abs_current_ma': max(abs(r['current_ma']) for r in report['samples']),
                            'max_tracking_error_raw': max(abs(r['tracking_error_raw']) for r in report['samples']),
                            'sample_count': len(report['samples']),
                            'physical_holding_observation_pending': True,
                            'stable_grip_confirmed': False, 'unsupported_load_bearing_confirmed': False}
    except BaseException as exc:
        failure = exc
        report['status'] = 'FAIL' if protocol is not None else 'FAIL_PRE_ACCESS'
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        if protocol is not None:
            if torque_may_be_on or sram_may_be_changed:
                for axis in [jaw]+[a for a in mapping.axes if a.servo_id != 6]:
                    try:
                        io.TRANSACTION_TIMEOUT_S = .01 if axis.servo_id == 6 and 'powered_duration_s' not in report else .25
                        io._write_raw(protocol, mapping, axis, 'Torque_Enable', 0)
                        if axis.servo_id == 6 and start is not None and 'powered_duration_s' not in report:
                            report['powered_duration_s'] = time.monotonic()-start
                        time.sleep(.03)
                    except BaseException as exc:
                        report.setdefault('cleanup_errors', []).append(f'disable {axis.name}: {exc}')
                        # Immediate safety-only retry of torque OFF, never a motion retry.
                        try:
                            io.TRANSACTION_TIMEOUT_S = .25
                            io._write_raw(protocol, mapping, axis, 'Torque_Enable', 0)
                        except BaseException as retry:
                            report.setdefault('cleanup_errors', []).append(f'disable retry {axis.name}: {retry}')
                io.TRANSACTION_TIMEOUT_S = .25
                for register, value in originals.items():
                    try:
                        io._write_raw(protocol, mapping, jaw, register, value)
                        time.sleep(.03)
                        if io._read_raw(protocol, mapping, jaw, register) != value:
                            raise RuntimeError('SRAM restore readback mismatch')
                    except BaseException as exc:
                        report.setdefault('cleanup_errors', []).append(f'restore {register}: {exc}')
            try:
                io.TRANSACTION_TIMEOUT_S = .25
                report['torque_enabled_after'] = [io._read_raw(protocol, mapping, a, 'Torque_Enable') for a in mapping.axes]
                report['sram_settings_after'] = {r: io._read_raw(protocol, mapping, jaw, r) for r in originals}
                report['gripper_position_after_torque_off_raw'] = io._read_raw(protocol, mapping, jaw, 'Present_Position')
            except BaseException as exc:
                report.setdefault('cleanup_errors', []).append(f'final readback: {exc}')
            if report.get('torque_enabled_after') != [0]*6 or report.get('cleanup_errors'):
                report['status'] = 'FAIL_CLEANUP_UNCONFIRMED'
        if transport is not None:
            transport.close()
        report['finished_at'] = datetime.now().astimezone().isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status','result','failure','cleanup_errors','powered_duration_s','torque_enabled_after','gripper_motion_commands_sent','gripper_position_after_torque_off_raw') if k in report}, ensure_ascii=False), flush=True)
    return 0 if failure is None and report['status'] == 'PASS_PENDING_OPERATOR_OBSERVATION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
