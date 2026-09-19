#!/usr/bin/env python3
"""One approved CAL-085 PLASTIC-only nominal-contact slow approach.

Only6,1pct/s monotonic closing to1866 (nominal),1852 (small step), or1837
(candidate-triggered feedback hold).20mA/10raw bounds,30s/3s deadlines.
Only explicit feedback-hold mode allows up to0.3s at the first hit feedback.
Explicit endpoint-confirmation mode adds at most0.2s at the same1837goal.
No arm goals, EEPROM/profile writes, retries or deeper goals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import _parse_gripper, load_calibration_profile
from qingyun.grabbing.motor_control import MotorMapping, SerialTransport, StsProtocol
from scripts import move_gripper_raw as io
from scripts.calibrate_gripper_empty_current import SETTINGS, segment_targets
from scripts.calibrate_gripper_open_speed import _sample
from scripts.read_joint_direction import EXPECTED_STABLE_PORT, EXPECTED_USB_IDENTITY
from scripts.read_servo_eeprom import sha256_file, usb_identity
from qingyun.grabbing.kinematics_ext import gripper_pct_to_gap

APPROVAL = "批准，CAL-085假果轻触接近首轮就绪"
SCOPE_CLARIFICATION = "只需要关注六轴即可，其他轴是我调整的"
PROFILE_SHA = "fe6aaf95e8418975d6eb966fe20ad55a8743084f63c443474aa838c21beffb2f"
OUTPUT_NAME = "CAL-085_fixture_just_touch_approach_attempt1_20260917.json"
SMALL_STEP_APPROVAL = "批准，CAL-085小步接近就绪"
SMALL_STEP_OUTPUT = "CAL-085_fixture_small_step_approach_attempt1_20260917.json"
FEEDBACK_HOLD_APPROVAL = "批准，CAL-085接触后反馈短保持就绪"
FEEDBACK_HOLD_OUTPUT = "CAL-085_feedback_hold_fixture_trial_attempt1_20260917.json"
CURRENT_BASELINE_APPROVAL = "依旧轻触，可能是我维持假果位置时发生了位移，可以在此基础上直接开始测试"
CURRENT_BASELINE_OUTPUT = "CAL-085_feedback_hold_fixture_trial_attempt2_20260917.json"
ENDPOINT_APPROVAL = "批准，CAL-085同目标停步确认就绪"
ENDPOINT_OUTPUT = "CAL-085_endpoint_confirmation_feedback_hold_attempt1_20260917.json"
FLOOR1830_APPROVAL = "批准，CAL-085最深1830停步确认就绪"
FLOOR1830_OUTPUT = "CAL-085_floor1830_endpoint_confirmation_attempt1_20260917.json"


def validate_endpoint_plan(plan: dict, floor1830: bool = False) -> None:
    expected = {
        "parameter_id": "CAL-085", "dependent_parameter_id": "CAL-093",
        "status": "APPROVED_ONE_ENDPOINT_CONFIRMATION_TRIAL",
        "required_confirmation": ENDPOINT_APPROVAL, "operator_confirmation": ENDPOINT_APPROVAL,
        "profile_sha256": PROFILE_SHA,
        "reference_report": "CAL-085_feedback_hold_fixture_trial_attempt2_20260917.json",
        "reference_report_sha256": "ebd43328f6b36f6f0c470af19101843d2f288450eb1597c87b27e546582d8494",
        "reference_raw_samples_sha256": "0ac3f218cbf886df5a073b15a0d6fa08ae8437cb752646d91131433e6a542071",
        "source_operator_observation": "仍轻触", "physical_light_touch_confirmed": True,
        "stable_grip_confirmed": False, "compression_zero_confirmed": False,
        "fixture_contact_band_mm": 20, "fixture_independently_supported": True,
        "servo_id": 6, "expected_start_raw": 1844, "start_max_deviation_raw": 5,
        "start_accepted_raw": [1839,1849], "all_six_static_read_frames": 10,
        "gripper_static_span_max_raw": 2, "all_axis_raw_domain_required": False,
        "arm_static_span_is_diagnostic_only": True, "arm_source_deviation_is_diagnostic_only": True,
        "arm_raw_domain_is_diagnostic_only": True, "operator_scope_clarification": SCOPE_CLARIFICATION,
        "all_six_identity_eeprom_torque_off_checks_required": True,
        "close_target_raw": 1837, "previous_absolute_goal_floor_raw": 1837,
        "closing_command_speed_pct_s": 1, "sample_fps": 30,
        "candidate_contact_current_ma": 10, "candidate_window_samples": 3,
        "candidate_contact_dwell_s": .1, "candidate_gap_range_m": [.022,.042],
        "candidate_empty_boundary_pct": 1.21, "maximum_endpoint_confirmation_s": .2,
        "endpoint_confirmation_goal_raw": 1837, "confirmation_is_new_powered_command_dwell": True,
        "hold_target_bounds_raw": [1837,1849], "maximum_feedback_hold_s": .3,
        "feedback_hold_switch_may_be_more_open_than_last_command": True,
        "whole_powered_timeout_s": 3, "feedback_bounds_raw": [1499,2897],
        "max_tracking_error_raw": 10, "all_phase_abs_current_abort_ma": 20,
        "transaction_timeout_s": .25, "max_tick_lateness_s": .02, "sram_settings": SETTINGS,
        "proposed_cycles": 1, "approved_cycles": 1, "authorized_physical_trials": 1,
        "remaining_authorized_physical_trials": 1, "arm_goals_authorized": 0,
        "opening_preparation_goals_authorized": 0, "eeprom_writes_authorized": False,
        "profile_writes_authorized": False, "new_motion_commands": 0, "cumulative_gripper_goals": 6712,
    }
    if floor1830:
        expected.update({
            "status": "APPROVED_ONE_FLOOR1830_CONFIRMATION_TRIAL",
            "required_confirmation": FLOOR1830_APPROVAL, "operator_confirmation": FLOOR1830_APPROVAL,
            "reference_report": "CAL-085_endpoint_confirmation_feedback_hold_attempt1_20260917.json",
            "reference_report_sha256": "e97841f3696603ccf7b29cdfd5835dc5971aefec83408189f199a040ac4b68a7",
            "reference_raw_samples_sha256": "2093b3eadc224af75052b7170f024921e5ebd537d71ef71707c6d38bd7d5eba4",
            "source_operator_observation": None, "source_trial_physical_observation_pending": True,
            "physical_light_touch_confirmed": False, "expected_start_raw": 1842,
            "start_accepted_raw": [1837,1847], "close_target_raw": 1830,
            "endpoint_confirmation_goal_raw": 1830, "hold_target_bounds_raw": [1830,1847],
            "cumulative_gripper_goals": 6735,
        })
    if any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Plan differs from approved same-goal endpoint confirmation')


def validate_feedback_hold_plan(plan: dict, current_baseline: bool = False) -> None:
    expected = {
        "parameter_id": "CAL-085", "dependent_parameter_id": "CAL-093",
        "status": "APPROVED_ONE_FEEDBACK_HOLD_TRIAL",
        "required_confirmation": FEEDBACK_HOLD_APPROVAL, "operator_confirmation": FEEDBACK_HOLD_APPROVAL,
        "profile_sha256": PROFILE_SHA,
        "reference_report": "CAL-085_fixture_small_step_approach_attempt1_20260917.json",
        "reference_report_sha256": "c4a4b17ffdadc21f9461a36e9ff6bd7f76ef5c474ec2db35adfaf7d468504f3b",
        "reference_raw_samples_sha256": "2b2289fa1dc83ada9dd35d9c2d9d7c84810d332c0b315c3bdd17ce344cf47d9b",
        "source_operator_observation": "刚轻触且无明显压缩",
        "physical_light_touch_confirmed": True, "no_visible_compression_confirmed": True,
        "first_contact_confirmed": False, "compression_zero_confirmed": False, "stable_grip_confirmed": False,
        "fixture_contact_band_mm": 20, "fixture_independently_supported": True,
        "servo_id": 6, "light_touch_reference_feedback_raw": 1858, "expected_start_raw": 1858,
        "start_max_deviation_raw": 5, "start_accepted_raw": [1853,1863],
        "all_six_static_read_frames": 10, "gripper_static_span_max_raw": 2,
        "all_axis_raw_domain_required": False, "arm_static_span_is_diagnostic_only": True,
        "arm_source_deviation_is_diagnostic_only": True, "arm_raw_domain_is_diagnostic_only": True,
        "operator_scope_clarification": SCOPE_CLARIFICATION,
        "all_six_identity_eeprom_torque_off_checks_required": True,
        "close_target_raw": 1837, "maximum_commanded_closure_from_reference_raw": 21,
        "closing_command_speed_pct_s": 1, "sample_fps": 30,
        "candidate_contact_current_ma": 10, "candidate_window_samples": 3, "candidate_contact_dwell_s": .1,
        "candidate_gap_range_m": [.022,.042], "candidate_empty_boundary_pct": 1.21,
        "hold_target_bounds_raw": [1837,1863],
        "feedback_hold_switch_may_be_more_open_than_last_command": True,
        "maximum_feedback_hold_s": .3, "holding_is_not_force_or_load_bearing_confirmation": True,
        "whole_powered_timeout_s": 3, "feedback_bounds_raw": [1499,2897],
        "max_tracking_error_raw": 10, "all_phase_abs_current_abort_ma": 20,
        "transaction_timeout_s": .25, "max_tick_lateness_s": .02, "sram_settings": SETTINGS,
        "proposed_cycles": 1, "approved_cycles": 1, "authorized_physical_trials": 1,
        "remaining_authorized_physical_trials": 1, "arm_goals_authorized": 0,
        "opening_preparation_goals_authorized": 0, "eeprom_writes_authorized": False,
        "profile_writes_authorized": False, "new_motion_commands": 0, "cumulative_gripper_goals": 6645,
    }
    if current_baseline:
        expected.update({
            "status": "APPROVED_ONE_CURRENT_BASELINE_FEEDBACK_HOLD_TRIAL",
            "required_confirmation": CURRENT_BASELINE_APPROVAL, "operator_confirmation": CURRENT_BASELINE_APPROVAL,
            "light_touch_reference_feedback_raw": 1855, "expected_start_raw": 1855,
            "start_accepted_raw": [1853,1860], "maximum_commanded_closure_from_reference_raw": 18,
            "cumulative_gripper_goals": 6676,
            "previous_failure_report": "CAL-085_feedback_hold_fixture_trial_attempt1_20260917.json",
            "previous_failure_report_sha256": "fc93294b407b2998890990805183633eb354f8703a346267aa794b8a3c38a6af",
            "previous_failure_raw_samples_sha256": "dbd1ec9491317782d5227b6590537b3ce7ed1a8366241de3e9ba89ccaf5c4cf9",
            "current_operator_observation": CURRENT_BASELINE_APPROVAL,
            "current_physical_light_touch_confirmed": True,
            "possible_operator_fixture_shift": True,
            "fixture_shift_proven": False,
            "current_no_visible_compression_reconfirmed": False,
            "retry_number": 2,
        })
    if any(plan.get(k) != v for k,v in expected.items()):
        raise ValueError("Plan differs from approved candidate-triggered feedback hold")


class ContactCandidate:
    """Exploratory10mA only; preserves production gap/empty/window/time semantics."""

    def __init__(self, gp):
        self.gp = gp
        self.currents = deque(maxlen=3)
        self.since_ns = None
        self.previous_ns = None

    def update(self, row: dict) -> bool:
        timestamp = row['monotonic_ns']
        if self.previous_ns is not None and timestamp <= self.previous_ns:
            raise ValueError('Candidate samples must be strictly chronological')
        self.previous_ns = timestamp
        self.currents.append(abs(row['current_ma']))
        mean = statistics.fmean(self.currents) if len(self.currents)==3 else 0.0
        gap = gripper_pct_to_gap(row['gripper_pct'], self.gp)
        gate = (.022<=gap<=.042 and row['gripper_pct']>1.21)
        eligible = len(self.currents)==3 and mean>10 and gate
        if eligible:
            if self.since_ns is None:
                self.since_ns = timestamp
        else:
            self.since_ns = None
        duration = 0.0 if self.since_ns is None else (timestamp-self.since_ns)/1e9
        qualified = eligible and duration>=.1
        row.update(candidate_mean_ma=mean, candidate_table_gap_m=gap,
                   candidate_gap_empty_gate=gate, candidate_eligible=eligible,
                   candidate_continuous_above_s=duration, candidate_qualified=qualified)
        return qualified


def validate_small_step_plan(plan: dict) -> None:
    expected = {
        "parameter_id": "CAL-085", "dependent_parameter_id": "CAL-093",
        "status": "APPROVED_ONE_SMALL_STEP_APPROACH",
        "required_confirmation": SMALL_STEP_APPROVAL, "operator_confirmation": SMALL_STEP_APPROVAL,
        "profile_sha256": PROFILE_SHA,
        "reference_report": "CAL-085_fixture_just_touch_approach_attempt2_20260917.json",
        "reference_report_sha256": "1a2a0d55435135b44c5a3f41e705b1a49d4226582e2b9b5bff924703ce66d7ad",
        "reference_raw_samples_sha256": "dd69e9bec6cfd97e89ef2e0699d37ca6569bc6a1213fa93640d85cd7d86a3458",
        "source_operator_observation": "仍有间隙",
        "fixture_contact_band_mm": 20, "fixture_independently_supported": True,
        "first_contact_confirmed": False, "compression_zero_confirmed": False,
        "servo_id": 6, "approved_cycles": 1, "proposed_cycles": 1,
        "authorized_physical_approaches": 1, "remaining_authorized_physical_approaches": 1,
        "expected_start_raw": 1870, "start_max_deviation_raw": 5,
        "start_accepted_raw": [1865,1875], "all_six_static_read_frames": 10,
        "gripper_static_span_max_raw": 2, "all_axis_raw_domain_required": False,
        "arm_static_span_is_diagnostic_only": True, "arm_source_deviation_is_diagnostic_only": True,
        "arm_raw_domain_is_diagnostic_only": True,
        "operator_scope_clarification": SCOPE_CLARIFICATION,
        "all_six_identity_eeprom_torque_off_checks_required": True,
        "previous_command_raw": 1866, "close_target_raw": 1852, "target_decrement_raw": 14,
        "closing_command_speed_pct_s": 1, "sample_fps": 30,
        "sram_settings": SETTINGS, "feedback_bounds_raw": [1499,2897],
        "max_tracking_error_raw": 10, "all_phase_abs_current_abort_ma": 20,
        "whole_powered_timeout_s": 3, "transaction_timeout_s": .25,
        "max_tick_lateness_s": .02, "final_hold_s": 0, "settle_dwell_added": False,
        "arm_goals_authorized": 0, "eeprom_writes_authorized": False, "profile_writes_authorized": False,
        "new_motion_commands": 0, "cumulative_gripper_goals": 6605,
    }
    if any(plan.get(k) != v for k,v in expected.items()):
        raise ValueError("Plan differs from approved single small-step approach")


def validate_plan(plan: dict, gripper_only: bool = False) -> None:
    expected = {
        "parameter_id": "CAL-085", "dependent_parameter_id": "CAL-093",
        "status": "APPROVED_ONE_NOMINAL_CONTACT_APPROACH", "required_confirmation": APPROVAL,
        "operator_confirmation": APPROVAL, "profile_sha256": PROFILE_SHA,
        "reference_report": "CAL-085_feedback_hold_empty_preparation_attempt1_20260917.json",
        "reference_report_sha256": "c267e68451de2e33b18e386dd1e69023bc77289c8bd8c7bd04839924ad067346",
        "reference_raw_samples_sha256": "1728f235acf7c85074077e1df63e5eb0164d0ac75b5885c812b8e3a7ccb1fda0",
        "source_operator_observation": "无异常，假果基准摆放就绪，接触带20mm",
        "fixture_contact_band_mm": 20, "fixture_independently_supported": True,
        "fixture_nominal_width_mm": 42, "width_is_approximate_not_new_measurement": True,
        "servo_id": 6, "approved_cycles": 1, "proposed_cycles": 1,
        "expected_start_raw": 2264, "start_max_deviation_raw": 5, "start_accepted_raw": [2259, 2269],
        "all_six_static_read_frames": 10, "all_six_static_span_max_raw": 2, "source_arm_max_deviation_raw": 10,
        "all_axis_raw_domain_required": True, "close_target_raw": 1866,
        "closing_command_speed_pct_s": 1, "sample_fps": 30,
        "sram_settings": SETTINGS, "feedback_bounds_raw": [1499, 2897], "max_tracking_error_raw": 10,
        "all_phase_abs_current_abort_ma": 20, "whole_powered_timeout_s": 30,
        "final_hold_s": 0, "settle_dwell_added": False,
        "arm_goals_authorized": 0, "eeprom_writes_authorized": False, "profile_writes_authorized": False,
    }
    if gripper_only:
        expected.update({
            "status": "APPROVED_GRIPPER_ONLY_RESUMED_APPROACH",
            "all_axis_raw_domain_required": False,
            "operator_scope_clarification": SCOPE_CLARIFICATION,
            "gripper_static_span_max_raw": 2,
            "arm_static_span_is_diagnostic_only": True,
            "arm_source_deviation_is_diagnostic_only": True,
            "arm_raw_domain_is_diagnostic_only": True,
            "resumed_attempt": 2,
            "resumed_report": "CAL-085_fixture_just_touch_approach_attempt2_20260917.json",
            "authorized_physical_approaches": 1,
        })
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Plan differs from approved bounded slow approach")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-confirmation", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gripper-only-rebaseline", action="store_true")
    mode.add_argument("--small-step", action="store_true")
    mode.add_argument("--feedback-hold", action="store_true")
    mode.add_argument("--feedback-hold-current-baseline", action="store_true")
    mode.add_argument("--endpoint-confirmation", action="store_true")
    mode.add_argument("--floor1830-confirmation", action="store_true")
    args = parser.parse_args()
    endpoint_mode = args.endpoint_confirmation or args.floor1830_confirmation
    feedback_hold_mode = args.feedback_hold or args.feedback_hold_current_baseline or endpoint_mode
    expected_output = OUTPUT_NAME.replace("attempt1", "attempt2") if args.gripper_only_rebaseline else OUTPUT_NAME
    expected_confirmation = SCOPE_CLARIFICATION if args.gripper_only_rebaseline else APPROVAL
    if args.small_step:
        expected_output, expected_confirmation = SMALL_STEP_OUTPUT, SMALL_STEP_APPROVAL
    if args.feedback_hold:
        expected_output, expected_confirmation = FEEDBACK_HOLD_OUTPUT, FEEDBACK_HOLD_APPROVAL
    if args.feedback_hold_current_baseline:
        expected_output, expected_confirmation = CURRENT_BASELINE_OUTPUT, CURRENT_BASELINE_APPROVAL
    if args.endpoint_confirmation:
        expected_output,expected_confirmation = ENDPOINT_OUTPUT,ENDPOINT_APPROVAL
    if args.floor1830_confirmation:
        expected_output,expected_confirmation = FLOOR1830_OUTPUT,FLOOR1830_APPROVAL
    if args.output.exists() or args.output.name != expected_output or args.operator_confirmation != expected_confirmation:
        raise SystemExit("Evidence exists, output differs or exact approval missing")
    plan = json.loads(args.session.read_text())
    if endpoint_mode:
        validate_endpoint_plan(plan,args.floor1830_confirmation)
    elif feedback_hold_mode:
        validate_feedback_hold_plan(plan,args.feedback_hold_current_baseline)
    elif args.small_step:
        validate_small_step_plan(plan)
    else:
        validate_plan(plan, args.gripper_only_rebaseline)
    target_raw = 1852 if args.small_step else 1866
    powered_timeout_s = 3 if args.small_step else 30
    start_lower, start_upper = (1865,1875) if args.small_step else (2259,2269)
    gripper_only = args.small_step or args.gripper_only_rebaseline
    if feedback_hold_mode:
        target_raw, powered_timeout_s = 1837, 3
        start_lower, start_upper = 1853,1863
        gripper_only = True
        if args.feedback_hold_current_baseline:
            start_lower,start_upper = 1853,1860
        if args.endpoint_confirmation:
            start_lower,start_upper = 1839,1849
        if args.floor1830_confirmation:
            target_raw,start_lower,start_upper = 1830,1837,1847
    profile_path = args.profile.resolve(strict=True)
    if sha256_file(profile_path) != PROFILE_SHA:
        raise SystemExit("Current profile fingerprint changed")
    raw_profile = json.loads(profile_path.read_text())
    profile = load_calibration_profile(profile_path)
    gp = _parse_gripper('gripper',raw_profile['gripper'])
    if profile.joints is None:
        raise SystemExit("Joint mapping missing")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    gripper = mapping.axis("gripper")
    if (mapping.gripper_closed_raw, mapping.gripper_open_raw) != (1499, 2897):
        raise SystemExit("Endpoint mapping changed")
    if not gripper.range_min <= target_raw <= gripper.range_max:
        raise SystemExit("Goal outside authoritative calibration domain")
    if (raw_profile['timing']['fps'] != 30 or raw_profile['gripper']['settle_tol_pct'] != .6
            or raw_profile['gripper']['settle_speed_pct_s'] != .3 or raw_profile['motion']['settle_dwell_s'] != .15):
        raise SystemExit("Completion criterion/rate changed")
    if feedback_hold_mode and (gp.window_samples!=3 or gp.contact_dwell_s!=.1
            or list(gp.contact_gap_range_m)!=[.022,.042]
            or abs(gp.empty_closed_pct+gp.empty_tol_pct-1.21)>1e-10):
        raise SystemExit('Production candidate dependency semantics changed')
    reports_dir = profile_path.parent / "reports"
    previous_failure_hash = None
    if args.feedback_hold_current_baseline:
        previous_path = reports_dir/plan['previous_failure_report']
        previous = json.loads(previous_path.read_text())
        previous_failure_hash = sha256_file(previous_path)
        previous_samples_hash = hashlib.sha256(json.dumps(previous['samples'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (previous_failure_hash!=plan['previous_failure_report_sha256']
                or previous_samples_hash!=plan['previous_failure_raw_samples_sha256']
                or previous['status']!='FAIL' or previous['failure']!='RuntimeError: Tracking exceeded10raw'
                or previous['gripper_motion_commands_sent']!=31 or previous['joint_servo_motion_commands_sent']!=0
                or previous['feedback_hold_goal_updates_sent']!=0
                or previous['torque_enabled_after']!=[0]*6
                or previous['sram_settings_after']!=previous['original_sram_settings']
                or previous['gripper_position_after_torque_off_raw']!=1855
                or previous['profile_sha256']!=PROFILE_SHA
                or previous['post_capture_review']['operator_observation']!=CURRENT_BASELINE_APPROVAL
                or not previous['post_capture_review']['physical_light_touch_confirmed']):
            raise SystemExit('Reviewed previous failure/current operator request changed')
    if args.gripper_only_rebaseline:
        rejected = json.loads((reports_dir / "CAL-085_fixture_just_touch_approach_attempt1_20260917.json").read_text())
        if (rejected['status'] != 'FAIL' or rejected['gripper_motion_commands_sent'] != 0
                or rejected['joint_servo_motion_commands_sent'] != 0
                or rejected['torque_enabled_after'] != [0]*6
                or rejected['sram_settings_after'] != {} or rejected.get('original_sram_settings')
                or rejected['profile_sha256'] != PROFILE_SHA):
            raise SystemExit('Only original readonly rejected attempt can be resumed')
    source_path = reports_dir / plan['reference_report']
    source = json.loads(source_path.read_text())
    samples_hash = hashlib.sha256(json.dumps(source['samples'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (sha256_file(source_path) != plan['reference_report_sha256']
            or samples_hash != plan['reference_raw_samples_sha256']
            or source['profile_sha256'] != PROFILE_SHA
            or source['status'] != 'PASS_PENDING_OPERATOR_OBSERVATION'
            or (not args.floor1830_confirmation and source['post_capture_review']['operator_observation'] != plan['source_operator_observation'])
            or source['torque_enabled_after'] != [0]*6 or source['sram_settings_after'] != source['original_sram_settings']
            or source['test_sram_settings'] != SETTINGS):
        raise SystemExit("Preparation/setup/source evidence changed")
    if args.floor1830_confirmation:
        if (source['result']['final_command_raw']!=1837
                or source['gripper_position_after_torque_off_raw']!=1842
                or source['gripper_motion_commands_sent']!=23 or source['joint_servo_motion_commands_sent']!=0
                or source['endpoint_confirmation_goal_updates_sent']!=6
                or source['feedback_hold_goal_updates_sent']!=0 or source['candidate_contact_qualified']
                or source['approved_plan']['fixture_contact_band_mm']!=20
                or not source['approved_plan']['fixture_independently_supported']):
            raise SystemExit('Prior1837confirmation/current-state source changed')
    elif args.endpoint_confirmation:
        if (source['result']['final_command_raw']!=1837
                or source['gripper_position_after_torque_off_raw']!=1844
                or source['gripper_motion_commands_sent']!=36 or source['joint_servo_motion_commands_sent']!=0
                or source['feedback_hold_goal_updates_sent']!=0 or source['candidate_contact_qualified']
                or source['approved_plan']['fixture_contact_band_mm']!=20
                or not source['approved_plan']['fixture_independently_supported']
                or not source['post_capture_review']['physical_light_touch_confirmed']):
            raise SystemExit('Reviewed no-candidate/light-touch source changed')
    elif feedback_hold_mode:
        if (source['result']['final_command_raw']!=1852
                or source['gripper_position_after_torque_off_raw']!=1858
                or source['gripper_motion_commands_sent']!=40 or source['joint_servo_motion_commands_sent']!=0
                or source['approved_plan']['fixture_contact_band_mm']!=20
                or not source['approved_plan']['fixture_independently_supported']
                or not source['post_capture_review']['physical_light_touch_confirmed']
                or not source['post_capture_review']['no_visible_compression_confirmed']
                or source['post_capture_review']['stable_grip_confirmed']):
            raise SystemExit('Physical light-touch operation reference changed')
    elif args.small_step:
        if (source['result']['final_command_raw'] != 1866
                or source['gripper_position_after_torque_off_raw'] != 1870
                or source['gripper_motion_commands_sent'] != 856
                or source['joint_servo_motion_commands_sent'] != 0
                or source['approved_plan']['fixture_contact_band_mm'] != 20
                or not source['approved_plan']['fixture_independently_supported']
                or source['post_capture_review']['first_contact_confirmed']
                or source['post_capture_review']['compression_zero_confirmed']):
            raise SystemExit('Previous approach/clearance observation changed')
    elif (not source['post_capture_review']['fixture_setup_ready']
            or source['post_capture_review']['contact_band_from_fingertip_mm'] != 20):
        raise SystemExit('Fixture setup changed')
    report = {
        'schema_version': 1, 'parameter_id': 'CAL-085', 'dependent_parameter_id': 'CAL-093',
        'test': ('single_plastic_fixture_candidate_triggered_feedback_hold' if feedback_hold_mode
                 else 'single_plastic_fixture_small_step_approach_no_hold' if args.small_step
                 else 'single_plastic_fixture_nominal_contact_approach_no_hold'), 'status': 'RUNNING',
        'started_at': datetime.now().astimezone().isoformat(), 'profile_sha256': PROFILE_SHA,
        'approved_plan': plan, 'operator_confirmation': args.operator_confirmation,
        'original_motion_approval': expected_confirmation,
        'gripper_only_rebaseline': args.gripper_only_rebaseline, 'small_step': args.small_step,
        'feedback_hold_mode': feedback_hold_mode,
        'current_baseline_mode': args.feedback_hold_current_baseline,
        'endpoint_confirmation_mode': endpoint_mode,
        'floor1830_mode': args.floor1830_confirmation,
        'previous_failure_report_sha256': previous_failure_hash,
        'source_report_sha256': sha256_file(source_path),
        'joint_servo_motion_commands_sent': 0, 'gripper_motion_commands_sent': 0,
        'eeprom_written': False, 'profile_written': False, 'closing_goal_updates_sent': 0, 'opening_goal_updates_sent': 0,
        'opening_preparation_goal_updates_sent': 0, 'feedback_hold_goal_updates_sent': 0,
        'feedback_hold_switch_opening_goal_updates_sent': 0, 'final_hold_s': 0, 'samples': [],
        'endpoint_confirmation_goal_updates_sent': 0,
        'transaction_timeout_s': .25,
        'environment': {'power_v': 12, 'ambient_temperature_c': None, 'base_mount': 'fixed to wooden board', 'load': 'independently supported plastic fixture,20mm contact band'},
    }
    io.TRANSACTION_TIMEOUT_S = .25
    protocol = transport = None
    originals = {}
    torque_may_be_on = sram_may_be_changed = False
    motion_start = None
    failure = None
    detector = ContactCandidate(gp) if feedback_hold_mode else None
    candidate_hit = None
    hold_start = hold_target = None
    hold_completed = False
    last_command = None
    confirmation_start = None

    def deadline_check() -> None:
        if motion_start is not None and time.monotonic()-motion_start >= powered_timeout_s:
            raise RuntimeError(f'{powered_timeout_s}s powered approach deadline')

    def checked_sample(phase: str, command: int, cap: int = 10) -> dict:
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

    def record_candidate(row: dict) -> None:
        nonlocal candidate_hit,hold_target
        candidate_hit = row.copy()
        report['candidate_first_hit'] = candidate_hit
        hold_target = row['position_raw']
        lower,upper = plan['hold_target_bounds_raw']
        if not lower<=hold_target<=upper:
            raise RuntimeError(f'Candidate feedback hold target outside approved{lower}..{upper}')
        report['hold_target_raw'] = hold_target
        report['closing_command_at_candidate_hit_raw'] = row['command_raw']
        print(f"CAL-085 candidate hit: command={row['command_raw']}, actual={hold_target}; freeze actual feedback only",flush=True)

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
        report['arm_positions_outside_calibration_domain_diagnostic'] = [
            {'servo_id':a.servo_id,'axis':a.name,'positions_raw':[f[i] for f in frames],
             'range_raw':[a.range_min,a.range_max]}
            for i,a in enumerate(mapping.axes[:5]) if any(not a.range_min<=f[i]<=a.range_max for f in frames)
        ]
        # User-authorized jaw-only supported fixture experiment. Other-axis
        # positions are retained, NOT clipped or accepted for future arm motion.
        if gripper_only:
            rejected_pose = spans[5]>2 or any(not gripper.range_min<=f[5]<=gripper.range_max for f in frames)
        else:
            rejected_pose = (any(s>2 for s in spans) or any(d>10 for d in deviations)
                or any(not a.range_min<=f[i]<=a.range_max for f in frames for i,a in enumerate(mapping.axes)))
        if (rejected_pose or torques_after != [0]*6
                or any(not start_lower<=f[5]<=start_upper for f in frames)):
            raise RuntimeError('Fresh start/static/source/domain differs; no writes')
        fresh = int(statistics.median(f[5] for f in frames))
        start = checked_sample('readonly_initial', fresh, 5)['position_raw']
        if abs(start-fresh)>2 or not start_lower<=start<=start_upper:
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
        print(f'CAL-085 preflight passed; only6 slow closing to{target_raw}; '
              + ('candidate hit permits actual-feedback hold up to0.3s' if feedback_hold_mode else 'no final hold'), flush=True)
        torque_may_be_on = True  # first goal may implicitly enable torque
        motion_start = time.monotonic()
        io._sync_goal(protocol, mapping, gripper, start)
        last_command = start
        report['gripper_motion_commands_sent'] += 1
        io._write_raw(protocol, mapping, gripper, 'Torque_Enable', 1)
        time.sleep(.03)
        checked_sample('initial_hold', start)
        tick = 1/30
        lateness = min(.02,float(raw_profile['timing']['max_tick_lateness_s']))
        base = time.monotonic()
        previous = start
        targets = segment_targets(start, target_raw, 1398, 30)
        for index, target in enumerate(targets):
            due = base + index*tick
            time.sleep(max(0, due-time.monotonic()))
            deadline_check()
            if time.monotonic()-due > lateness:
                raise RuntimeError('Approach tick lateness')
            if not target_raw <= target <= previous <= start:
                raise RuntimeError('Refuse opening/deeper-than-approved goal')
            io._sync_goal(protocol, mapping, gripper, target)
            report['gripper_motion_commands_sent'] += 1
            report['closing_goal_updates_sent'] += 1
            previous = target
            time.sleep(max(0, due+tick-time.monotonic()))
            last_command = target
            row = checked_sample('fixture_feedback_hold_closing' if feedback_hold_mode
                                 else 'fixture_small_step_approach' if args.small_step else 'fixture_nominal_approach', target)
            if detector is not None and detector.update(row):
                record_candidate(row)
                break
            if index and index % 300 == 0:
                print(f"CAL-085 approach {index+1}/{len(targets)} target={target} actual={row['position_raw']}", flush=True)
        if endpoint_mode and candidate_hit is None:
            # The existing detector/window/since timestamp cross this boundary.
            # Budget starts at endpoint feedback, not after setup or printing.
            confirmation_start = row['monotonic_ns']/1e9
            report['endpoint_confirmation_started_monotonic_ns'] = row['monotonic_ns']
            for index in range(6):
                due = confirmation_start+index*tick
                time.sleep(max(0,due-time.monotonic()))
                deadline_check()
                remaining = .2-(time.monotonic()-confirmation_start)
                if remaining<=.005 or time.monotonic()-due>lateness:
                    raise RuntimeError('Endpoint confirmation deadline/tick lateness')
                io.TRANSACTION_TIMEOUT_S = min(.25,(remaining-.005)/6)
                io._sync_goal(protocol,mapping,gripper,target_raw)
                last_command = target_raw
                report['gripper_motion_commands_sent'] += 1
                report['endpoint_confirmation_goal_updates_sent'] += 1
                row = checked_sample('fixture_endpoint_confirmation',target_raw)
                if time.monotonic()-confirmation_start>=.2:
                    raise RuntimeError('Endpoint confirmation exceeded0.2s')
                if detector.update(row):
                    record_candidate(row)
                    break
        if candidate_hit is not None:
            # Fixed target ONCE from first hit. Never ratchet to newer feedback.
            hold_start = time.monotonic()
            if confirmation_start is not None:
                report['endpoint_confirmation_powered_duration_s'] = hold_start-confirmation_start
                if report['endpoint_confirmation_powered_duration_s']>=.2:
                    raise RuntimeError('Endpoint confirmation exceeded0.2s before switch')
            report['feedback_hold_started_monotonic_ns'] = time.monotonic_ns()
            for index in range(9):
                due = hold_start+index*tick
                time.sleep(max(0,due-time.monotonic()))
                deadline_check()
                remaining = .3-(time.monotonic()-hold_start)
                if remaining<=.005 or time.monotonic()-due>lateness:
                    raise RuntimeError('Feedback hold deadline/tick lateness')
                # Bound all five write/read transactions within remaining hold budget.
                io.TRANSACTION_TIMEOUT_S = min(.25,(remaining-.005)/6)
                io._sync_goal(protocol,mapping,gripper,hold_target)
                report['gripper_motion_commands_sent'] += 1
                report['feedback_hold_goal_updates_sent'] = report.get('feedback_hold_goal_updates_sent',0)+1
                if hold_target>last_command:
                    report['opening_goal_updates_sent'] += 1
                    report['feedback_hold_switch_opening_goal_updates_sent'] += 1
                last_command = hold_target
                row = checked_sample('fixture_feedback_hold',hold_target)
                if time.monotonic()-hold_start>=.3:
                    raise RuntimeError('Feedback hold exceeded0.3s')
            hold_completed = True
        # No endpoint squeeze if no candidate; unload before computing statistics.
        io.TRANSACTION_TIMEOUT_S = .25
        io._write_raw(protocol,mapping,gripper,'Torque_Enable',0)
        if confirmation_start is not None and 'endpoint_confirmation_powered_duration_s' not in report:
            report['endpoint_confirmation_powered_duration_s'] = time.monotonic()-confirmation_start
        if hold_start is not None:
            report['feedback_hold_powered_duration_s'] = time.monotonic()-hold_start
            report['final_hold_s'] = report['feedback_hold_powered_duration_s']
            report['feedback_hold_completed'] = hold_completed
        report['powered_duration_s'] = time.monotonic()-motion_start
        report['result'] = {
            'final_command_raw': last_command, 'final_feedback_raw': row['position_raw'],
            'final_feedback_pct': row['gripper_pct'], 'final_velocity_pct_s': row['velocity_pct_s'],
            'tracking_error_raw': row['tracking_error_raw'],
            'completion_elapsed_s': row['monotonic_ns']/1e9-motion_start,
            'peak_abs_current_ma': max(abs(r['current_ma']) for r in report['samples']),
            'max_tracking_error_raw': max(abs(r['tracking_error_raw']) for r in report['samples']),
            'first_contact_confirmed': False, 'compression_zero_confirmed': False,
            'physical_contact_observation_pending': True, 'final_hold_added': hold_start is not None,
        }
        report['actual_command_path_raw'] = ([start,candidate_hit['command_raw'],hold_target]
                                             if candidate_hit is not None else [start,target_raw])
        if feedback_hold_mode:
            report['candidate_contact_qualified'] = candidate_hit is not None
            report['feedback_hold_completed'] = hold_completed
            report['result'].update(candidate_contact_qualified=candidate_hit is not None,
                                    feedback_hold_completed=hold_completed,hold_target_raw=hold_target)
        report['termination_reason'] = ('CANDIDATE_FEEDBACK_HOLD_COMPLETE_IMMEDIATE_UNLOAD' if hold_completed
                                        else 'CANDIDATE_NOT_OBSERVED_AFTER_ENDPOINT_CONFIRMATION_IMMEDIATE_UNLOAD' if endpoint_mode
                                        else 'CANDIDATE_NOT_OBSERVED_AT_CLOSURE_LIMIT_IMMEDIATE_UNLOAD' if feedback_hold_mode
                                        else 'SMALL_STEP_LIMIT_COMPLETE_IMMEDIATE_UNLOAD_NO_HOLD' if args.small_step
                                        else 'NOMINAL_APPROACH_LIMIT_COMPLETE_IMMEDIATE_UNLOAD_NO_HOLD')
        report['status'] = 'PASS_PENDING_OPERATOR_OBSERVATION'
    except BaseException as exc:
        failure = exc
        report['status'] = 'FAIL' if protocol is not None else 'FAIL_PRE_ACCESS'
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        io.TRANSACTION_TIMEOUT_S = .25
        if protocol is not None:
            if torque_may_be_on:
                for axis in [gripper]+[a for a in mapping.axes if a.servo_id!=6]:
                    try:
                        io._write_raw(protocol,mapping,axis,'Torque_Enable',0)
                        if axis.servo_id==6 and 'powered_duration_s' not in report:
                            report['powered_duration_s'] = time.monotonic()-motion_start
                        if axis.servo_id==6 and hold_start is not None and 'feedback_hold_powered_duration_s' not in report:
                            report['feedback_hold_powered_duration_s'] = time.monotonic()-hold_start
                            report['final_hold_s'] = report['feedback_hold_powered_duration_s']
                            report['feedback_hold_completed'] = False
                        if axis.servo_id==6 and confirmation_start is not None and 'endpoint_confirmation_powered_duration_s' not in report:
                            report['endpoint_confirmation_powered_duration_s'] = time.monotonic()-confirmation_start
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
