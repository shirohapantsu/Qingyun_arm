from __future__ import annotations

import pytest

from scripts.read_joint_direction import infer_joint_sign


def test_urdf正向对应raw增加则sign为正():
    sign, check = infer_joint_sign(2000, 2150, 1840)
    assert sign == 1
    assert check["positive_delta_counts"] == 150
    assert check["negative_delta_counts"] == -160


def test_urdf正向对应raw减少则sign为负():
    sign, check = infer_joint_sign(2000, 1800, 2190)
    assert sign == -1
    assert check["opposite_sides_of_baseline"] is True


def test_摆幅不足或没有跨过起点拒绝():
    with pytest.raises(ValueError, match="至少 30 count"):
        infer_joint_sign(2000, 2020, 1900)
    with pytest.raises(ValueError, match="没有落在起点两侧"):
        infer_joint_sign(2000, 2100, 2200)
