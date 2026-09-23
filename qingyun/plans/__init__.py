"""草莓分拣任务的方案注册表（P1-04）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2 文件责任表（``qingyun/plans/__init__.py`` = 固定注册表
        ``PLAN_REGISTRY = {"strawberry": StrawberryPlan}``，**新增**）
        §3.5（plans 的公开签名）
        §3.2 第 1 条（注册表键集合必须恰为 ``{strawberry}``，D06/D07：本期不注册蓝莓）
        §4 末段（生产入口传 PLAN_REGISTRY；mock 入口传 P5 的 MOCK_REGISTRY，
        **不覆盖生产全局变量**——所以本模块只暴露这一个字典，不提供任何
        "注册/反注册"函数，也不接受运行期改写）
        §8 步骤 4

设计要点：
    * 注册表是**模块级常量**，键与值都在 import 时定死；``main`` 与预检接收
      **同一个 registry 对象**（P1 §4 末段），因此这里不能用函数返回新字典
      （返回新对象会让"预检过的就是会话里用的那份"这条不变量失去静态依据）。
    * 蓝莓（blueberry）**不注册**：D06/D07 本期只实例化草莓，蓝莓口令在 P2 侧
      按 ``invalid`` 处理；若模型真的输出 ``blueberry``，按 P1 §5 映射
      ``terminate("LLM_CONTRACT_VIOLATION")``，属会话层而不是本表要兜的底。
    * 本包是 ``qingyun`` 命名空间包下**唯一**的常规包（有 ``__init__.py``）：
      这是 §2 文件责任表点名的交付物，不是对既有包结构的改动。
"""

from __future__ import annotations

from qingyun.plans.base import BasePlan
from qingyun.plans.strawberry import StrawberryPlan

__all__ = ["PLAN_REGISTRY", "BasePlan", "StrawberryPlan"]

# 固定注册表（P1 §2/§3.5 逐字）：键必须与类的 TASK_ID 一致（预检第 1 条会核）。
PLAN_REGISTRY: dict[str, type[BasePlan]] = {"strawberry": StrawberryPlan}
