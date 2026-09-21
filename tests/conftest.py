"""tests/ 的共享 fixture。

放在 conftest.py 里而不是 tests/support.py 里，是因为 pytest 只会自动收集
conftest 中定义的 fixture；写在 support.py 里的 @pytest.fixture 对各测试模块
是不可见的，会以"fixture 'params' not found"报错。
"""

from __future__ import annotations

import pytest

from configs.motion_params import MotionParams
from tests.support import load_sim


@pytest.fixture(scope="session")
def params() -> MotionParams:
    """整个测试会话共享一份已加载的仿真配置。

    会话级共享是为了省掉反复构造 placo 模型的开销；测试一律只读它，需要改动时
    请先 deepcopy 或走 write_profile 生成独立副本。
    """
    return load_sim()
