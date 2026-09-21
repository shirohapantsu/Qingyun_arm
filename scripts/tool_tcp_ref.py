#!/usr/bin/env python3
"""tool 阶段：由固定指尖触针的针尖坐标 + 开度，换算 _fit_tool 需要的参考行。

约定（CAD 名义旋转，CAL-024/109 先例）：
- 针尖固定在基座系 P_pin（一次性实测），固定指尖轻触针尖 => 固定指尖 = P_pin。
- TCP = 夹持中心 = 固定指垫面 + (gap/2) 沿闭合轴（固定指 -> 活动指方向）。
- 闭合轴 ĉ_B = R_B_E · R_e_tcp · e_x（R_e_tcp 第 0 列 = TCP +X = 闭合代表方向，指南 6.3）。
- gap_m / angle_deg 复用 CAL-026 五点实测表按开度线性插值（项目约束）。
- T_B_TCP_reference 用名义 FK 构建：R_B_TCP = R_B_E · R_e_tcp，t = TCP。

用法：
  python scripts/tool_tcp_ref.py --q 1.26,-14.78,16.99,-3.14,0.02 --pct 25 \
      --pin 0.350,-0.025,0.039 --fixture-id tool_tip_fixture_20260920
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.so_arm_core.kinematics import RobotKinematics  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF = str(ROOT / "libs/so_arm_core/so101_new_calib.urdf")

# CAL-026 五点实测净间隙（真实）；CAL-027 URDF 名义角表。
GAP_PCT = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
GAP_M = np.array([0.0096, 0.0405, 0.0712, 0.0985, 0.1207])
ANGLE_DEG = np.array([-10.0, 17.5, 45.0, 72.5, 100.0])

# profile.tool.rotation_e_tcp（CAD 名义，CAL-024）。
R_E_TCP = np.array([
    [0.0, -0.0486886, 0.998814],
    [0.0, -0.998814, -0.0486886],
    [1.0, 0.0, 0.0],
])


def _kin():
    return RobotKinematics(URDF, "gripper_frame_link", JOINT_NAMES)


def closing_dir_B(q_deg: np.ndarray) -> np.ndarray:
    """固定指 -> 活动指的单位闭合方向（基座系），由名义 FK 得到。"""
    T = np.asarray(_kin().forward_kinematics(np.deg2rad(q_deg)), dtype=float)
    R_B_E = T[:3, :3]
    return R_B_E @ R_E_TCP[:, 0]


def build_row(q_deg, pct, pin_m, fixture_id, load_label="unloaded"):
    q = np.asarray(q_deg, float)
    pin = np.asarray(pin_m, float)
    gap = float(np.interp(pct, GAP_PCT, GAP_M))
    angle = float(np.interp(pct, GAP_PCT, ANGLE_DEG))
    c = closing_dir_B(q)
    c = c / np.linalg.norm(c)
    tcp = pin + (gap / 2.0) * c
    T = np.asarray(_kin().forward_kinematics(np.deg2rad(q)), dtype=float)
    R_B_E = T[:3, :3]
    R_B_TCP = R_B_E @ R_E_TCP
    T_B_TCP = np.eye(4)
    T_B_TCP[:3, :3] = R_B_TCP
    T_B_TCP[:3, 3] = tcp
    return {
        "gripper_pct": float(pct),
        "tcp_reference_B_m": [round(float(v), 6) for v in tcp],
        "T_B_TCP_reference": [[round(float(v), 6) for v in row] for row in T_B_TCP],
        "gap_m": round(gap, 6),
        "gripper_reference_angle_deg": round(angle, 4),
        "fixture_id": fixture_id,
        "load_label": load_label,
    }, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", required=True, help="q_urdf_deg 逗号分隔 5 值")
    ap.add_argument("--pct", required=True, type=float, help="标称开度档位 %")
    ap.add_argument("--pin", required=True, help="针尖基座坐标 x,y,z 逗号分隔 m")
    ap.add_argument("--fixture-id", required=True)
    ap.add_argument("--load-label", default="unloaded")
    ap.add_argument("--out", type=Path, help="写出参考 JSON（含 samples 列表）")
    args = ap.parse_args()

    q = [float(v) for v in args.q.split(",")]
    pin = [float(v) for v in args.pin.split(",")]
    if len(q) != 5 or len(pin) != 3:
        raise SystemExit("--q 需 5 值，--pin 需 3 值")
    row, c = build_row(q, args.pct, pin, args.fixture_id, args.load_label)
    print("closing_dir_B:", np.round(c, 4))
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"samples": [row]}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
