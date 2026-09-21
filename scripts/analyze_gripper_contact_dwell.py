#!/usr/bin/env python3
"""CAL-088 read-only conditional replay of existing CAL-087 evidence.

No serial, goals, evidence rewrites, profile changes, or force/safety claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import _parse_gripper
from scripts.analyze_gripper_current_window import BASE, FIXTURE, EMPTY, read_report, replay


def main() -> None:
    profile_bytes = (BASE / "profile.json").read_bytes()
    profile = json.loads(profile_bytes)
    digest = hashlib.sha256(profile_bytes).hexdigest()
    gp = _parse_gripper("gripper", profile["gripper"])
    prior_name = "CAL-087_current_window_causal_comparison_20260917.json"
    prior = read_report(prior_name)
    if (prior["profile_sha256"] != digest
            or prior["current_and_conditionally_retained_window_samples"] != gp.window_samples
            or prior["contact_threshold_used_ma"] != 10
            or prior["sample_fps"] != profile["timing"]["fps"]):
        raise ValueError("Window comparison no longer matches current profile")
    for name, expected in prior["source_report_sha256"].items():
        if hashlib.sha256((BASE / "reports" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since window comparison: {name}")
    fixed_window = next(c for c in prior["comparisons"] if c["window_samples"] == gp.window_samples)
    controls = fixed_window["empty_controls"]
    fixture = read_report(FIXTURE)
    fps = profile["timing"]["fps"]
    comparisons = []
    for dwell in (1 / fps, 2 / fps, .1, .15, .2, .3, .5):
        trial_gp = replace(gp, contact_dwell_s=dwell)
        empties = [{k: c[k] for k in ("report", "operator_observation", "evidence_kind")}
                   | {"replay": replay(read_report(c["report"]), gp.window_samples, 10.0, trial_gp)}
                   for c in controls]
        budget = gp.window_samples / fps + dwell + 2.0 / fps
        comparisons.append({
            "contact_dwell_s": dwell,
            "fixture_replay": replay(fixture, gp.window_samples, 10.0, trial_gp),
            "empty_controls": empties,
            "empty_reports_with_false_contact_hits": sum(c["replay"]["hit_count"] > 0 for c in empties),
            "nominal_fitter_delay_budget_s": budget,
            "illustrative_extra_command_pct_at_trial1pct_s": budget,
            "illustrative_extra_command_pct_at_profile15pct_s": budget * gp.close_speed_pct_s,
        })
    retained = next(c for c in comparisons if math.isclose(c["contact_dwell_s"], gp.contact_dwell_s))
    if retained["empty_reports_with_false_contact_hits"] or not retained["fixture_replay"]["first_hit"]:
        raise ValueError("Current dwell unsupported even conditionally by existing data")
    output = {
        "schema_version": 1, "parameter_id": "CAL-088", "status": "UNVERIFIED",
        "source": "offline_causal_replay_existing_hardware_samples_no_motion",
        "profile_sha256": digest, "window_samples_fixed": gp.window_samples,
        "sample_fps": fps, "contact_threshold_used_ma": 10,
        "contact_threshold_is_demo_candidate_not_integrated": True,
        "current_and_conditionally_retained_contact_dwell_s": gp.contact_dwell_s,
        "source_window_comparison_report": prior_name,
        "source_window_comparison_sha256": hashlib.sha256((BASE / "reports" / prior_name).read_bytes()).hexdigest(),
        "source_report_sha256": prior["source_report_sha256"],
        "matched_control_report": EMPTY, "fixture_report": FIXTURE,
        "comparison": "strict rolling absolute mean >10mA continuously >=dwell; same gap and empty-exclusion gates",
        "comparisons": comparisons,
        "current_profile220_threshold_fixture_replay": replay(fixture, gp.window_samples, gp.contact_current_ma, gp),
        "decision": "Retain current0.1s conditionally; no recorded empty qualifying runs establish a necessary minimum dwell. Do not shorten solely because these traces have zero false positives, or lengthen without physical compression budget.",
        "cal084_to_cal088_group_status": "NOT_ACCEPTED_FOR_PRODUCTION_OR_FULL_GRASP",
        "profile_written": False, "eeprom_written": False, "new_motion_commands": 0,
        "limitations": [
            "Electrical first-above-threshold signal is not measured first physical contact.",
            "Sampling jitter and strict dwell comparison can defer a hit by a tick;0.1s is not a fixed frame count.",
            "Only one enhanced plastic fixture trace at1pct/s and one same-endpoint empty match; six historical controls are auxiliary, including one without operator observation.",
            "Changing the dwell changes the counterfactual first-hit time; later open-loop samples do not reproduce real feedback-opening hold commands.",
            "Nominal fitter budget is not a measured worst-case latency; extra command percentage is not physical compression or pressure.",
            "No real-fruit/production15pct_s/damage/stop-response/holding validation; current220mA configuration still misses the fixture signal.",
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
