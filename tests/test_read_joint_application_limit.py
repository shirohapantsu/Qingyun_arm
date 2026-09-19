from __future__ import annotations

import pytest

from scripts.read_joint_application_limit import effective_test_target_deg


def test_应用边界按方向内缩余量():
    assert effective_test_target_deg("lower", -110.0, 2.0) == -108.0
    assert effective_test_target_deg("upper", 110.0, 2.0) == 108.0


def test_应用边界拒绝错误方向或负余量():
    with pytest.raises(ValueError, match="lower/upper"):
        effective_test_target_deg("middle", 0.0, 2.0)
    with pytest.raises(ValueError, match="不得为负"):
        effective_test_target_deg("upper", 0.0, -1.0)
