#!/usr/bin/env python3
"""tool 采集：单样本一步完成——读总线→建参考行→capture 追加一行。

流程（姿态已力矩锁定、固定指尖触针后调用）：
1. CalibrationReader 读 q_urdf_deg / gripper_pct（核对用）；
2. 用 scripts/tool_tcp_ref.build_row 以**标称开度档位**生成参考行
   （gripper_pct 覆盖总线抖动值，tcp_reference_B_m / T_B_TCP_reference /
   gap_m / gripper_reference_angle_deg 一并算出）；
3. 写临时参考 JSON，subprocess 调 scripts/calibrate.py capture
   --stage tool --hardware（capture 自己再读一次总线拿权威 q_urdf_deg）。

用法：
  python scripts/tool_capture_sample.py --pct 75 --pose-label pose1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import CalibrationReader  # noqa: E402
from scripts.tool_tcp_ref import build_row  # noqa: E402

PIN_B = [0.350, -0.025, 0.039]  # 针尖基座坐标（一次性实测，2026-09-21）
FIXTURE_ID = "tool_tip_fixture_20260920"
PROFILE = ROOT / "calibration/REAL_ARM/PC/profile.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pct", required=True, type=float, help="标称开度档位")
    ap.add_argument("--pose-label", required=True)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "calibration/REAL_ARM/PC/raw/tool_stage_fit_20260921.jsonl")
    ap.add_argument("--tol-pct", type=float, default=4.0,
                    help="总线开度与标称档位允许偏差（百分点），超出拒绝采集")
    args = ap.parse_args()

    profile = load_calibration_profile(PROFILE)
    reader = CalibrationReader(profile)
    try:
        snap = reader.read_snapshot()
    finally:
        reader.close()
    q = [float(v) for v in snap["q_urdf_deg"]]
    pct_bus = float(snap["gripper_pct"])
    if abs(pct_bus - args.pct) > args.tol_pct:
        raise SystemExit(f"开度核对失败：总线 {pct_bus:.1f}% vs 标称 {args.pct}%")

    row, c = build_row(q, args.pct, PIN_B, FIXTURE_ID)
    ref = {"pose_label": args.pose_label,
           "verification": (f"固定指尖端点接触针尖（操作者确认）；针尖基座坐标 {PIN_B}；"
                            f"闭合轴 ĉ_B={list(map(lambda v: round(v, 3), c))}；"
                            "TCP=针尖+gap/2·ĉ；旋转参考=CAD 名义（CAL-024 先例）；"
                            f"总线核对 q={[round(v,2) for v in q]} pct={pct_bus:.1f}"),
           "samples": [row]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)
        ref_path = Path(f.name)

    cmd = [sys.executable, str(ROOT / "scripts/calibrate.py"), "capture",
           "--stage", "tool", "--hardware", "--profile", str(PROFILE),
           "--output", str(args.output), "--reference", str(ref_path),
           "--operator", "w", "--power", "12V",
           "--base-mount", "fixed_to_wooden_work_surface_no_gap",
           "--pad", "grid5mm_a4x2_20260921", "--load", "unloaded"]
    print("q_bus:", [round(v, 2) for v in q], " pct_bus:", round(pct_bus, 1))
    print("row  :", json.dumps(row, ensure_ascii=False))
    r = subprocess.run(cmd, cwd=ROOT)
    ref_path.unlink(missing_ok=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
