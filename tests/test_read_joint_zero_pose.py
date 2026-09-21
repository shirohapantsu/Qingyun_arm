from __future__ import annotations

import pytest

from scripts.read_joint_zero_pose import infer_zero_offsets_deg


def test_零位偏置按已确认方向反算():
    result = infer_zero_offsets_deg(
        [1, -1, 1, -1, 1],
        [10.0, 20.0, -30.0, -40.0, 50.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert result == pytest.approx([-10.0, 20.0, 30.0, -40.0, -50.0])


def test_零位偏置输入维度和方向受约束():
    with pytest.raises(ValueError, match="长度 5"):
        infer_zero_offsets_deg([1], [0.0], [0.0])
    with pytest.raises(ValueError, match="只能为 ±1"):
        infer_zero_offsets_deg([1, 1, 0, 1, 1], [0.0] * 5, [0.0] * 5)
