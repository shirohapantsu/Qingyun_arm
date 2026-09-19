#!/usr/bin/env python3
"""CAL-083：复用已验证的小步采集逻辑，只做空夹爪闭合速度试验。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_gripper_open_speed import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(parameter_id="CAL-083", opening=False))
