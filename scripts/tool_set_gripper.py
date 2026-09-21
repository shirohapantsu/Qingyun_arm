#!/usr/bin/env python3
"""tool 采集：只把夹爪（id=6）单播到目标开度档位，等待到位并读回。

与 move_gripper_raw.py 的区别：那个是 CAL-026 卡尺量测专用（要求六台全卸力、
结束全卸力）；本脚本用于 tool 阶段逐开度采集，**保持 1~5 力矩锁定不动**，
只对夹爪单播 Goal_Position（规避 CAL-052 广播会令六台自动上力）。

端点来自 profile.motor.gripper_closed_raw / gripper_open_raw；raw 线性映射：
  raw = closed + pct/100 * (open - closed)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import load_calibration_profile  # noqa: E402
from qingyun.grabbing.motor_control import (  # noqa: E402
    MotorMapping,
    SerialTransport,
    StsProtocol,
)

SETTLE_TOL_RAW = 8
TIMEOUT_S = 8.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, default=ROOT / "calibration/REAL_ARM/PC/profile.json")
    ap.add_argument("--pct", required=True, type=float, help="目标开度 0~100")
    args = ap.parse_args()

    profile = load_calibration_profile(args.profile)
    mapping = MotorMapping(
        profile.motor, profile.joints, profile.motor_calibration, profile.urdf_limits_deg
    )
    grip = mapping.axis("gripper")
    closed, open_ = profile.motor.gripper_closed_raw, profile.motor.gripper_open_raw
    lo, hi = min(closed, open_), max(closed, open_)
    target = int(round(closed + args.pct / 100.0 * (open_ - closed)))
    if not lo <= target <= hi:
        raise SystemExit(f"target raw {target} outside endpoints [{lo},{hi}]")

    transport = SerialTransport(profile.motor.port, profile.motor.baudrate)
    proto = StsProtocol(transport)
    try:
        addr, length = mapping.register(grip, "Response_Status_Level")
        level = int.from_bytes(proto.read_registers(grip.servo_id, addr, length,
                              time.monotonic() + 5.0), "little", signed=False)
        proto.set_write_response(grip.servo_id, expects_ack=(level == 0))
        gaddr, glen = mapping.register(grip, "Goal_Position")
        proto.write_registers(grip.servo_id, gaddr,
                              (target & 0xFFFF).to_bytes(glen, "little", signed=False),
                              time.monotonic() + 5.0)
        paddr, plen = mapping.register(grip, "Present_Position")
        deadline = time.monotonic() + TIMEOUT_S
        settled = 0
        pos = None
        while time.monotonic() < deadline:
            pos = int.from_bytes(proto.read_registers(grip.servo_id, paddr, plen,
                                 time.monotonic() + 5.0), "little", signed=False)
            if abs(pos - target) <= SETTLE_TOL_RAW:
                settled += 1
                if settled >= 3:
                    break
            else:
                settled = 0
            time.sleep(0.05)
        pct_read = (pos - closed) / (open_ - closed) * 100.0
        print(f"target_pct={args.pct} target_raw={target} present_raw={pos} "
              f"read_pct={pct_read:.2f}")
        if settled < 3:
            return 2
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
