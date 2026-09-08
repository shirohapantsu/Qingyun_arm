"""pytest 根配置：把项目根目录加入 sys.path。

本项目不是 pip 安装的包，模块之间用 "configs.*"、"qingyun.*"、"libs.*" 这种
从项目根算起的绝对导入。把根目录插到 sys.path 最前面，pytest 直接
`python -m pytest` 就能收集 tests/ 下的用例，不需要安装成本地包。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
