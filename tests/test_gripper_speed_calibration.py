"""开合速度小组闭合后检查采集器的符号与运行期开度换算一致性。"""

import pytest

from qingyun.grabbing.motor_control import gripper_pct_to_raw, raw_to_gripper_pct
from scripts.calibrate_gripper_open_speed import _linear_slope, _pct_to_raw, _raw_to_pct


@pytest.mark.parametrize("rate", [20.0, -15.0])
def test_速度拟合保留开合方向(rate):
    times = [0.0, 0.034, 0.069, 0.103, 0.138]
    rows = [{"monotonic_ns": 1000000000 + round(t * 1e9),
             "gripper_pct": 50.0 + rate * t} for t in times]
    assert _linear_slope(rows) == pytest.approx(rate)


@pytest.mark.parametrize("endpoints", [(1499, 2897), (2897, 1499)])
def test_采集开度换算与生产函数一致(endpoints):
    closed, opened = endpoints
    for pct in [0.0, 25.0, 37.65, 75.0, 100.0]:
        raw = _pct_to_raw(closed, opened, pct)
        assert raw == gripper_pct_to_raw(pct, closed, opened)
        assert _raw_to_pct(closed, opened, raw) == pytest.approx(
            raw_to_gripper_pct(raw, closed, opened)
        )


def test_采集时间零跨度拒绝拟合():
    with pytest.raises(RuntimeError, match="时间跨度为零"):
        _linear_slope([{"monotonic_ns": 1, "gripper_pct": 25.0}] * 2)
