#!/usr/bin/env python3
"""CAL-085 重测离线分析：以生产判据重放空爪/带物迹线并派生接触阈值。

复现 qingyun/grabbing/arm_control.py::_close_until_contact 的精确判据：
  filtered = 最近 window_samples 帧 |电流| 的均值（窗口未满时不判）
  in_gap   = contact_gap_range_m[0] <= gap(actual_pct) <= contact_gap_range_m[1]
  above_empty = actual_pct > empty_closed_pct + empty_tol_pct
  连续满足 filtered > threshold ∧ in_gap ∧ above_empty 达 contact_dwell_s → 确认接触
  actual_pct <= 空夹门 → EMPTY；超 close_timeout_s → UNKNOWN

通过条件：全部空爪迹线零确认（零误报）∧ 全部带物迹线在 gap 区内确认。
带物迹线在 FREEZE_RAMP 后命令冻结；生产中命令会继续推进，电流只会更高，
故本重放的确认时刻是保守上界。压缩预算按 close_speed×确认延迟估算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE_THRESHOLDS_MA = [13.0, 15.0, 17.3, 19.5, 26.0]
WINDOW_SCAN = [2, 3, 4, 5]
DWELL_SCAN_S = [1 / 30, 2 / 30, 0.1, 0.15, 0.2, 0.3]


def _samples(report: dict) -> list[dict]:
    return [s for s in report["samples"] if s["phase"] in ("closing", "hold")]


def _gap_slope_mm_per_pct(samples: list[dict]) -> float:
    """接触附近 gap(mm) 对 pct 的局部斜率，用于估算延迟压缩量。"""
    gaps = [(s["gripper_pct"], s["gap_m"] * 1000.0) for s in samples if s["gap_m"] is not None]
    if len(gaps) < 2:
        return 0.0
    (p0, g0), (p1, g1) = gaps[0], gaps[-1]
    return (g1 - g0) / (p1 - p0) if p1 != p0 else 0.0


def replay(samples: list[dict], threshold_ma: float, window: int, dwell_s: float,
           gap_range_m: list[float], empty_gate_pct: float, close_speed_pct_s: float,
           tick_s: float) -> dict:
    """按生产语义重放一条迹线，返回判定结果与关键帧。"""
    currents: list[float] = []
    contact_since: float | None = None
    first_cross: dict | None = None
    confirmed: dict | None = None
    max_filtered_in_gate: float = 0.0
    t0 = samples[0]["monotonic_ns"]
    for idx, s in enumerate(samples):
        now = (s["monotonic_ns"] - t0) / 1e9
        currents.append(abs(float(s["current_ma"])))
        if len(currents) > window:
            currents.pop(0)
        filtered = statistics.fmean(currents) if len(currents) >= window else 0.0
        gap = s["gap_m"]
        in_gap = gap is not None and gap_range_m[0] <= gap <= gap_range_m[1]
        above_empty = s["gripper_pct"] > empty_gate_pct
        if in_gap and above_empty:
            max_filtered_in_gate = max(max_filtered_in_gate, filtered)
        if filtered > threshold_ma and in_gap and above_empty:
            if contact_since is None:
                contact_since = now
                if first_cross is None:
                    first_cross = {
                        "index": idx, "t_s": now, "gap_mm": gap * 1000.0,
                        "filtered_ma": filtered, "pct": s["gripper_pct"],
                    }
            elif now - contact_since >= dwell_s and confirmed is None:
                confirmed = {
                    "index": idx, "t_s": now, "gap_mm": gap * 1000.0,
                    "filtered_ma": filtered, "pct": s["gripper_pct"],
                    "delay_from_cross_s": now - first_cross["t_s"],
                }
        else:
            contact_since = None
        if s["gripper_pct"] <= empty_gate_pct:
            return {"verdict": "EMPTY", "max_filtered_in_gate_ma": max_filtered_in_gate,
                    "first_cross": first_cross, "confirmed": confirmed}
    return {"verdict": "CONFIRMED" if confirmed else "NO_CONFIRM",
            "max_filtered_in_gate_ma": max_filtered_in_gate,
            "first_cross": first_cross, "confirmed": confirmed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--empty-report", action="append", required=True)
    parser.add_argument("--loaded-report", action="append", required=True)
    parser.add_argument("--extra-loaded-report", action="append", default=[],
                        help="参考带物迹线（如阶段1试探），不参与通过判定")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("拒绝覆盖既有证据")

    raw_profile = json.loads(args.profile.read_text(encoding="utf-8"))
    gp = raw_profile["gripper"]
    gap_range = [float(v) for v in gp["contact_gap_range_m"]]
    empty_gate = float(gp["empty_closed_pct"]) + float(gp["empty_tol_pct"])
    close_speed = float(gp["close_speed_pct_s"])
    tick_s = 1.0 / float(raw_profile["timing"]["fps"])
    production_window = int(gp["window_samples"])
    production_dwell = float(gp["contact_dwell_s"])

    def load(path_str: str) -> tuple[str, dict, str]:
        path = Path(path_str)
        report = json.loads(path.read_text(encoding="utf-8"))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.name, report, sha

    empty = [load(p) for p in args.empty_report]
    loaded = [load(p) for p in args.loaded_report]
    extra = [load(p) for p in args.extra_loaded_report]

    out: dict = {
        "schema_version": 1,
        "parameter_id": "CAL-085",
        "test": "production_criteria_replay_threshold_derivation",
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "production_criteria": {
            "window_samples": production_window,
            "contact_dwell_s": production_dwell,
            "contact_gap_range_m": gap_range,
            "empty_gate_pct": empty_gate,
            "close_speed_pct_s": close_speed,
            "tick_s": tick_s,
            "source": "qingyun/grabbing/arm_control.py::_close_until_contact",
        },
        "traces": {
            "empty": [{"file": n, "sha256": s} for n, _, s in empty],
            "loaded": [{"file": n, "sha256": s} for n, _, s in loaded],
            "extra_reference": [{"file": n, "sha256": s} for n, _, s in extra],
        },
        "candidates": {},
    }

    # 1) 每条迹线的空夹门内滤波上界（误报边界）
    bounds = []
    for name, report, _ in empty:
        rows = _samples(report)
        r = replay(rows, float("inf"), production_window, production_dwell,
                   gap_range, empty_gate, close_speed, tick_s)
        bounds.append({"file": name, "max_filtered_in_gate_ma": r["max_filtered_in_gate_ma"]})
    out["empty_filtered_upper_bound_ma"] = {
        "per_trace": bounds,
        "max": max(b["max_filtered_in_gate_ma"] for b in bounds),
    }

    # 2) 候选阈值逐个重放
    for threshold in CANDIDATE_THRESHOLDS_MA:
        entry: dict = {"threshold_ma": threshold, "empty_false_positives": [],
                       "loaded_results": []}
        for name, report, _ in empty:
            rows = _samples(report)
            r = replay(rows, threshold, production_window, production_dwell,
                       gap_range, empty_gate, close_speed, tick_s)
            if r["verdict"] == "CONFIRMED":
                entry["empty_false_positives"].append(
                    {"file": name, "confirmed": r["confirmed"]})
        all_loaded_confirmed = True
        for name, report, _ in loaded:
            rows = _samples(report)
            r = replay(rows, threshold, production_window, production_dwell,
                       gap_range, empty_gate, close_speed, tick_s)
            entry["loaded_results"].append({"file": name, **r})
            if r["verdict"] != "CONFIRMED":
                all_loaded_confirmed = False
        entry["pass"] = not entry["empty_false_positives"] and all_loaded_confirmed
        entry["extra_reference"] = []
        for name, report, _ in extra:
            rows = _samples(report)
            r = replay(rows, threshold, production_window, production_dwell,
                       gap_range, empty_gate, close_speed, tick_s)
            entry["extra_reference"].append({"file": name, "verdict": r["verdict"],
                                             "confirmed": r["confirmed"]})
        out["candidates"][f"{threshold:g}"] = entry

    # 3) 通过候选中的最小值（含参考信息）
    passing = [t for t in CANDIDATE_THRESHOLDS_MA
               if out["candidates"][f"{t:g}"]["pass"]]
    out["passing_thresholds_ma"] = passing
    out["recommended_threshold_ma"] = min(passing) if passing else None

    # 4) 推荐阈值下的窗口/驻留敏感性扫描
    if out["recommended_threshold_ma"] is not None:
        thr = out["recommended_threshold_ma"]
        scan = []
        for window in WINDOW_SCAN:
            for dwell in DWELL_SCAN_S:
                fp = 0
                miss = 0
                for _, report, _ in empty:
                    r = replay(_samples(report), thr, window, dwell,
                               gap_range, empty_gate, close_speed, tick_s)
                    fp += 1 if r["verdict"] == "CONFIRMED" else 0
                for _, report, _ in loaded:
                    r = replay(_samples(report), thr, window, dwell,
                               gap_range, empty_gate, close_speed, tick_s)
                    miss += 0 if r["verdict"] == "CONFIRMED" else 1
                scan.append({"window_samples": window, "dwell_s": dwell,
                             "empty_false_positives": fp, "loaded_misses": miss,
                             "pass": fp == 0 and miss == 0})
        out["window_dwell_scan"] = scan

        # 5) 压缩预算与硬阈值数据
        hold_stats = []
        for name, report, _ in loaded:
            hold = [s for s in report["samples"] if s["phase"] == "hold"]
            closing = [s for s in report["samples"] if s["phase"] == "closing"]
            if hold:
                hold_stats.append({
                    "file": name,
                    "hold_peak_abs_current_ma": max(abs(s["current_ma"]) for s in hold),
                    "hold_max_filtered_ma": max(
                        statistics.fmean(abs(s["current_ma"]) for s in hold[max(0, i - production_window + 1): i + 1])
                        for i in range(len(hold))),
                    "hold_feedback_drift_raw": hold[-1]["position_raw"] - hold[0]["position_raw"],
                    "freeze_gap_mm": closing[-1]["gap_m"] * 1000.0,
                })
        out["loaded_hold_stats"] = hold_stats
        out["hold_peak_abs_current_ma_max"] = max(
            h["hold_peak_abs_current_ma"] for h in hold_stats) if hold_stats else None
        # 确认延迟期间生产命令继续闭合的估算压缩量
        compress = []
        for name, report, _ in loaded:
            rows = _samples(report)
            r = replay(rows, thr, production_window, production_dwell,
                       gap_range, empty_gate, close_speed, tick_s)
            if r["confirmed"]:
                slope = _gap_slope_mm_per_pct(rows)
                extra_pct = close_speed * (r["confirmed"]["delay_from_cross_s"] + tick_s)
                compress.append({
                    "file": name,
                    "first_cross_gap_mm": r["first_cross"]["gap_mm"],
                    "confirmed_gap_mm_on_frozen_trace": r["confirmed"]["gap_mm"],
                    "delay_from_cross_s": r["confirmed"]["delay_from_cross_s"],
                    "estimated_extra_closure_mm_in_production": abs(slope) * extra_pct,
                })
        out["production_compression_budget_estimate"] = {
            "method": "确认延迟×close_speed×局部gap斜率（生产命令在延迟期间继续推进；"
                      "冻结迹线电流为下界，实际确认不晚于重放结果）",
            "per_trace": compress,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "empty_max_filtered_ma": out["empty_filtered_upper_bound_ma"]["max"],
        "passing_thresholds_ma": out.get("passing_thresholds_ma"),
        "recommended_threshold_ma": out.get("recommended_threshold_ma"),
        "hold_peak_abs_current_ma_max": out.get("hold_peak_abs_current_ma_max"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
