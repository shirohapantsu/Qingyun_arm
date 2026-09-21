#!/usr/bin/env python3
"""CAL-087 bounded offline replay; no serial access or source/profile writes.

Recorded diagnostic motion is open-loop. After the first counterfactual contact
hit, later hits do not simulate _confirm_contact's changed holding command.
Signal-onset delay is NOT delay from a measured physical first contact.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import _parse_gripper
from qingyun.grabbing.kinematics_ext import gripper_pct_to_gap

BASE = ROOT / "calibration/REAL_ARM/PC"
FIXTURE = "CAL-085_fixture_stronger_contact_attempt1_20260917.json"
EMPTY = "CAL-085_empty_stronger_contact_matched_control_attempt1_20260917.json"


def read_report(name: str) -> dict:
    if Path(name).name != name:
        raise ValueError("Report must be a local evidence basename")
    return json.loads((BASE / "reports" / name).read_text(encoding="utf-8"))


def replay(report: dict, window: int, threshold: float, gp) -> dict:
    currents: deque[float] = deque(maxlen=window)
    since = first_signal = previous_ns = first_hit = None
    peak = longest = 0.0
    hits = frames = 0
    for row in report["samples"]:
        if row["phase"] not in ("empty_closing", "final_hold", "fixture_closing", "fixture_hold"):
            continue
        timestamp = row["monotonic_ns"]
        if previous_ns is not None and timestamp <= previous_ns:
            raise ValueError("Nonmonotonic source timestamps")
        previous_ns = timestamp
        frames += 1
        now = timestamp / 1e9
        current = abs(row["current_ma"])
        currents.append(current)
        filtered = float(np.mean(currents)) if len(currents) == window else 0.0
        peak = max(peak, filtered)
        gap = gripper_pct_to_gap(row["gripper_pct"], gp)
        gate = (gp.contact_gap_range_m[0] <= gap <= gp.contact_gap_range_m[1]
                and row["gripper_pct"] > gp.empty_closed_pct + gp.empty_tol_pct)
        if first_signal is None and current > threshold and gate:
            first_signal = now
        if filtered > threshold and gate:
            if since is None:
                since = now
            else:
                elapsed = now - since
                longest = max(longest, elapsed)
                if elapsed >= gp.contact_dwell_s:
                    hits += 1
                    if first_hit is None:
                        first_hit = {
                            "phase": row["phase"], "position_raw": row["position_raw"],
                            "command_raw": row["command_raw"], "mean_ma": filtered,
                            "continuous_above_s": elapsed,
                            "delay_from_first_in_gate_raw_current_above_threshold_s": now - first_signal,
                        }
        else:
            since = None
    return {"included_frames": frames, "peak_filtered_mean_ma": peak,
            "longest_qualifying_run_s": longest, "hit_count": hits, "first_hit": first_hit}


def main() -> None:
    profile_bytes = (BASE / "profile.json").read_bytes()
    profile = json.loads(profile_bytes)
    gp = _parse_gripper("gripper", profile["gripper"])
    fixture, empty = read_report(FIXTURE), read_report(EMPTY)
    digest = hashlib.sha256(profile_bytes).hexdigest()
    for report in (fixture, empty):
        if (report["profile_sha256"] != digest
                or report["status"] != "PASS_PENDING_OPERATOR_OBSERVATION"
                or report["torque_enabled_after"] != [0] * 6
                or report["sram_settings_after"] != report["original_sram_settings"]
                or report["actual_command_path_raw"][1:] != [1871, 1801]):
            raise ValueError("Matched source evidence/configuration not intact")
    if (fixture["test_sram_settings"] != empty["test_sram_settings"]
            or fixture["readonly_gripper_controller_settings"] != empty["readonly_gripper_controller_settings"]):
        raise ValueError("Matched control settings differ")
    session = read_report("CAL-085_contact_current_session_20260916.json")
    if session["profile_sha256"] != digest:
        raise ValueError("Session profile fingerprint differs")
    source_row = next(r for r in session["powered_current_trials"] if r["report"] == FIXTURE)
    empty_row = next(r for r in session["powered_current_trials"] if r["report"] == EMPTY)
    if source_row["operator_observation"] != "无异常" or empty_row["operator_observation"] != "无异常":
        raise ValueError("Matched operator observations missing")
    history = read_report("CAL-085_stronger_contact_threshold_replay_20260917.json")
    controls = [{"report": r["report"], "operator_observation": r["operator_observation"],
                 "evidence_kind": "historical_observational_not_same_endpoint",
                 "data": read_report(r["report"])}
                for r in history["historical_empty_threshold_10_ma_replays"]]
    controls.append({"report": EMPTY, "operator_observation": "无异常",
                     "evidence_kind": "same_endpoint_and_settings_matched_control", "data": empty})
    fps = profile["timing"]["fps"]
    comparisons = []
    for window in (1, 2, 3, 4, 5, 7):
        controls_replayed = [{k: c[k] for k in ("report", "operator_observation", "evidence_kind")}
                             | {"replay": replay(c["data"], window, 10.0, gp)} for c in controls]
        budget_s = window / fps + gp.contact_dwell_s + 2.0 / fps
        comparisons.append({
            "window_samples": window,
            "fixture_replay": replay(fixture, window, 10.0, gp),
            "empty_controls": controls_replayed,
            "empty_reports_with_false_contact_hits": sum(c["replay"]["hit_count"] > 0 for c in controls_replayed),
            "nominal_fitter_delay_budget_s": budget_s,
            "illustrative_extra_command_pct_at_trial1pct_s": budget_s,
            "illustrative_extra_command_pct_at_profile15pct_s": budget_s * profile["gripper"]["close_speed_pct_s"],
        })
    current = next(c for c in comparisons if c["window_samples"] == gp.window_samples)
    if current["empty_reports_with_false_contact_hits"] or not current["fixture_replay"]["first_hit"]:
        raise ValueError("Current window not supported even conditionally by these sources")
    output = {
        "schema_version": 1, "parameter_id": "CAL-087", "status": "UNVERIFIED",
        "source": "offline_causal_replay_existing_hardware_samples_no_motion",
        "profile_sha256": digest, "contact_threshold_used_ma": 10,
        "source_report_sha256": {
            name: hashlib.sha256((BASE / "reports" / name).read_bytes()).hexdigest()
            for name in [FIXTURE] + [c["report"] for c in controls]
        },
        "contact_threshold_is_demo_candidate_not_integrated": True,
        "contact_dwell_s_fixed_not_newly_calibrated": gp.contact_dwell_s,
        "sample_fps": fps, "current_and_conditionally_retained_window_samples": gp.window_samples,
        "comparison": "strict abs rolling mean > threshold with continuous dwell, gap and empty-exclusion gates",
        "comparisons": comparisons,
        "current_profile220_threshold_fixture_replay": replay(fixture, gp.window_samples, gp.contact_current_ma, gp),
        "decision": "Retain3 conditionally; recorded evidence does not establish an optimum or require a larger window. No production integration/verification while threshold and physical evidence remain unresolved.",
        "profile_written": False, "eeprom_written": False, "new_motion_commands": 0,
        "limitations": [
            "First above-threshold raw current is an electrical signal marker, not measured first physical contact.",
            "Counterfactual replay cannot verify closed-loop contact hold because real commands change after first hit.",
            "One stronger plastic trial at1pct/s; no true-fruit, production15pct/s, force, compression or stop-latency certification.",
            "Fitter window/FPS+dwell+2/FPS budget is a nominal engineering formula, not a measured worst-case bound.",
            "Historical controls include one without operator observation and are auxiliary, not seven equivalent matched repetitions.",
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
