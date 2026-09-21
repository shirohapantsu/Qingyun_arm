#!/usr/bin/env python3
"""CAL-096/097: read-only derivation from reviewed EMPTY jaw timing traces.

Print a proposed report only; no serial, evidence/profile writes or full tests.
Three-count margins are engineering allowances, not physical safety accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import load_calibration_profile
from qingyun.grabbing.motor_control import MotorMapping

BASE = ROOT / "calibration/REAL_ARM/PC"
EXPECTED_PROFILE_SHA = "4a1782e01bab930fd19e4b557a831dd9b406ca3fddc889480b1fc6a1dd30f7a2"


def ceil_tenth(value: float) -> float:
    return math.ceil(value * 10) / 10


def first_completion(rows: list[dict], phase_start_s: float,
                     position_tol_pct: float, speed_tol_pct_s: float, dwell_s: float) -> dict | None:
    """Replay actual runtime position/speed conjunction with continuous time."""
    since = None
    for row in rows:
        now = row["monotonic_ns"] / 1e9
        if abs(row["gripper_pct"] - 55) <= position_tol_pct and abs(row["velocity_pct_s"]) <= speed_tol_pct_s:
            if since is None:
                since = now
            elif now - since >= dwell_s:
                return {"elapsed_s": now - phase_start_s, "first_qualifying_monotonic_ns": round(since * 1e9),
                        "completion_monotonic_ns": row["monotonic_ns"], "continuous_span_s": now - since,
                        "position_raw": row["position_raw"], "position_error_pct": abs(row["gripper_pct"] - 55),
                        "velocity_pct_s": row["velocity_pct_s"], "command_raw": row["command_raw"]}
        else:
            since = None
    return None


def main() -> None:
    profile_bytes = (BASE / "profile.json").read_bytes()
    digest = hashlib.sha256(profile_bytes).hexdigest()
    if digest != EXPECTED_PROFILE_SHA:
        # After our two-field integration, allow an exact in-memory rollback
        # fingerprint check. Never alter active profile or recast old captures.
        reviewed = json.loads((BASE / "reports/CAL-096_097_empty_settle_derivation_20260917.json").read_text())
        restored = profile_bytes.replace(b'"settle_tol_pct": 0.6', b'"settle_tol_pct": 2', 1)
        restored = restored.replace(b'"settle_speed_pct_s": 0.3', b'"settle_speed_pct_s": 3', 1)
        if (reviewed["status"] != "CONFIRMED_DEMO_INTEGRATED"
                or reviewed["profile_sha256_before"] != EXPECTED_PROFILE_SHA
                or reviewed["profile_sha256_after"] != digest
                or hashlib.sha256(restored).hexdigest() != EXPECTED_PROFILE_SHA):
            raise ValueError("Frozen profile changed beyond reviewed two-field integration")
    raw_profile = json.loads(profile_bytes)
    profile = load_calibration_profile(BASE / "profile.json")
    mapping = MotorMapping(profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg)
    integration = json.loads((BASE / "reports/CAL-091_092_demo_integration_final_20260917.json").read_text())
    if (integration["status"] != "CONFIRMED_DEMO_INTEGRATED"
            or integration["profile_sha256_after"] != EXPECTED_PROFILE_SHA
            or integration["operator_observation"] != "无异常"
            or integration["only_changed_fields"] != {
                "gripper.close_timeout_s": {"before": 6, "after": 4.7},
                "gripper.open_timeout_s": {"before": 6, "after": 4.3}}):
        raise ValueError("Independent timing review/configuration lineage differs")
    if (mapping.gripper_closed_raw, mapping.gripper_open_raw) != (1499, 2897):
        raise ValueError("Feedback mapping endpoints changed")
    dwell = raw_profile["motion"]["settle_dwell_s"]
    if dwell != .15 or raw_profile["timing"]["fps"] != 30:
        raise ValueError("Dwell or sample rate changed")
    sources = []
    controller = None
    for manifest in integration["sources"]:
        name = manifest["report"]
        if Path(name).name != name:
            raise ValueError("Evidence name must remain local basename")
        source_bytes = (BASE / "reports" / name).read_bytes()
        report = json.loads(source_bytes)
        samples_hash = hashlib.sha256(json.dumps(report["samples"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256_after_operator_review"]
                or samples_hash != manifest["raw_samples_sha256"]
                or report["post_capture_review"]["operator_observation"] != "无异常"
                or report["profile_sha256"] != integration["profile_sha256_before"]
                or report["status"] != "PASS_PENDING_OPERATOR_OBSERVATION"
                or report["torque_enabled_after"] != [0] * 6
                or report["sram_settings_after"] != report["original_sram_settings"]
                or report["test_sram_settings"] != {"Acceleration": 5, "Goal_Time": 0, "Goal_Velocity": 0, "Torque_Limit": 500}
                or report["joint_servo_motion_commands_sent"] != 0):
            raise ValueError("Source trace/review/mapping/cleanup changed")
        current_controller = report["readonly_gripper_controller_settings"]
        if controller is not None and current_controller != controller:
            raise ValueError("Controller settings differ across cycles")
        controller = current_controller
        rows = [row for row in report["samples"] if row["phase"] == "empty_opening"]
        for previous, row in zip(rows, rows[1:]):
            if row["monotonic_ns"] <= previous["monotonic_ns"]:
                raise ValueError("Nonmonotonic trace")
        for row in rows:
            if (abs(mapping.raw_to_gripper_pct(row["position_raw"]) - row["gripper_pct"]) > 1e-9
                    or abs(mapping.raw_velocity_to_deg_s("gripper", row["velocity_register_raw"]) - row["velocity_pct_s"]) > 1e-9):
                raise ValueError("Recorded feedback units differ from current mapping")
        window = []
        # Shortest observed trailing window spanning the existing dwell, not
        # a guessed frame count and not the full moving ramp.
        for row in reversed(rows):
            if abs(row["position_raw"] - 2268) > 5 or row["velocity_register_raw"] != 0:
                break
            window.append(row)
            if (window[0]["monotonic_ns"] - window[-1]["monotonic_ns"]) / 1e9 >= dwell:
                break
        window.reverse()
        if not window or (window[-1]["monotonic_ns"] - window[0]["monotonic_ns"]) / 1e9 < dwell:
            raise ValueError("No independent observed terminal dwell window")
        # Report's completion elapsed uses monotonic_ns converted to seconds;
        # reconstruct its common phase clock without inventing a wall timestamp.
        phase_start = rows[-1]["monotonic_ns"] / 1e9 - report["opening_result"]["strict_completion_elapsed_s"]
        sources.append({"report": name, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "raw_samples_sha256": samples_hash, "rows": rows, "phase_start_s": phase_start,
                        "window": {"frames": len(window), "span_s": (window[-1]["monotonic_ns"] - window[0]["monotonic_ns"]) / 1e9,
                                   "first_monotonic_ns": window[0]["monotonic_ns"], "last_monotonic_ns": window[-1]["monotonic_ns"],
                                   "max_position_error_pct": max(abs(r["gripper_pct"] - 55) for r in window),
                                   "max_abs_velocity_pct_s": max(abs(r["velocity_pct_s"]) for r in window),
                                   "position_raw": [r["position_raw"] for r in window],
                                   "velocity_register_raw": [r["velocity_register_raw"] for r in window]}})
    if len(sources) != 3:
        raise ValueError("Exactly three reviewed independent demo cycles required")
    position_unit = 100 / 1398
    velocity_unit = abs(mapping.raw_velocity_to_deg_s("gripper", 1))
    position_max = max(source["window"]["max_position_error_pct"] for source in sources)
    velocity_max = max(source["window"]["max_abs_velocity_pct_s"] for source in sources)
    position_tol = ceil_tenth(position_max + 3 * position_unit)
    speed_tol = ceil_tenth(velocity_max + 3 * velocity_unit)
    for source in sources:
        rows = source.pop("rows")
        phase_start = source.pop("phase_start_s")
        source["replay_with_derived_thresholds"] = first_completion(rows, phase_start, position_tol, speed_tol, dwell)
        completion = source["replay_with_derived_thresholds"]
        if completion is None or completion["elapsed_s"] >= raw_profile["gripper"]["open_timeout_s"]:
            raise ValueError("Derived conjunction invalidates reviewed opening timeout")
    output = {"schema_version": 1, "parameter_ids": ["CAL-096", "CAL-097"],
              "status": "DERIVED_DEMO_FROM_REVIEWED_EMPTY_TRACES", "profile_sha256_before": EXPECTED_PROFILE_SHA,
              "profile_sha256_used": digest,
              "source": "offline_existing_hardware_data_no_motion", "independent_cycles": 3,
              "dwell_s_unchanged": dwell, "sample_fps": 30, "target_pct": 55, "target_raw": 2268,
              "readonly_gripper_controller_settings": controller, "sources": sources,
              "position_error_max_pct": position_max, "velocity_max_pct_s": velocity_max,
              "position_pct_per_feedback_count": position_unit, "velocity_pct_s_per_feedback_count": velocity_unit,
              "engineering_margin_feedback_counts": 3, "rounding": "ceil to0.1 interface unit",
              "derived_settle_tol_pct": position_tol, "derived_settle_speed_pct_s": speed_tol,
              "existing_open_timeout_s": raw_profile["gripper"]["open_timeout_s"],
              "existing_close_timeout_s": raw_profile["gripper"]["close_timeout_s"],
              "timeout_replay_decision": "All three actual traces complete the new conjunction before4.3s; preserve both reviewed timeout fields. Closing empty branch does not consume these opening-settle thresholds.",
              "new_motion_commands": 0, "eeprom_written": False, "profile_written": False,
              "limitations": ["Three independent EMPTY-only cycles at55pct;18terminal frames are not18independent repetitions.",
                              "Observed zero raw velocity does not prove physical zero motion or arbitrarily precise speed.",
                              "Three-count margin is an explicit engineering allowance like prior joint-settle calibration, not a measured confidence/safety bound.",
                              "Terminal windows alone cannot certify no premature opening completion on every path/target; replay checks current actual traces only.",
                              "No full100pct/fruit/TCP/placement/force certification; profile remains draft and contact group unresolved.",
                              "Changes to mapping, dwell, target domain, speed, SRAM/controller require re-review; full tests deferred until complete claw subgroup."]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
