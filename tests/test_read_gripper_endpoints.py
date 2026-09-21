from __future__ import annotations

import pytest

from scripts.read_gripper_endpoints import ENDPOINT_SEQUENCE, summarize_placement


def test_端点序列严格交替五轮():
    assert ENDPOINT_SEQUENCE == ("closed", "open") * 5


def test_静态端点burst取中位数并保留抖动范围():
    rows = [[100, 200, 300, 400, 500, value]
            for value in [2000, 2001, 2000, 2002, 1999]]
    out = summarize_placement(
        "closed", 1, rows, gripper_index=5, range_min=1458, range_max=2980,
    )
    assert out["gripper_raw_representative"] == 2000
    assert out["gripper_raw_min"] == 1999
    assert out["gripper_raw_max"] == 2002
    assert out["gripper_raw_span"] == 3


def test_越出官方行程或错误行宽拒绝():
    with pytest.raises(ValueError, match="越出官方行程"):
        summarize_placement(
            "open", 1, [[1, 2, 3, 4, 5, 3000]],
            gripper_index=5, range_min=1458, range_max=2980,
        )
    with pytest.raises(ValueError, match="恰有 6 项"):
        summarize_placement(
            "open", 1, [[1, 2]],
            gripper_index=5, range_min=1458, range_max=2980,
        )
