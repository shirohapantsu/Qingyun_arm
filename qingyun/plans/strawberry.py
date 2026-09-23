"""草莓方案实例（P1-04）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2 文件责任表：``qingyun/plans/strawberry.py`` = 草莓落点 ID 映射与面积
           阈值 K 占位（P4 回填），**新增**
        §3.5：类定义逐字（TASK_ID / AREA_THRESHOLD_M2 / A_PLACE_IDS / B_PLACE_ID /
           C_PLACE_ID 五个类属性）
        §3.2 第 1/2/3 条（预检读的正是这些类属性：A 为非空无重复元组、A/B/C 互不
           重叠、每个 place_id 必须存在于 ``params.places``、K 未回填时 None → 拒绝运行）
        §1/D06/D07：本期只实例化草莓，不实现 blueberry.py，也不做运动参数切换
    docs/实施文档/README.md 决策 D04（A 槽位数量随 P4 落点标定确定，分配即消耗）、
    D05（K 按果品独立标定，本期只需草莓这一个值）

设计要点：
    * 本文件**只有类属性**，没有任何行为：分级、计数、分发、恢复全部在
      ``base.BasePlan`` 里，按果品差异的部分全部通过类属性注入。
    * ``AREA_THRESHOLD_M2 = None`` 是有意的"未标定"声明（P1 §3.5 注释原文）。
      生产预检（§3.2 第 3 条）会因此拒绝运行并提示"K 未标定"；mock 入口用注明
      "非实测"的名义值注入（P5 §2）。**本文件不填任何猜测数值**，也不为此放宽
      ``base.classify`` 的判据。
    * 坐标不复制进 plan：``grasp_and_place`` 按 place_id 从 ``params.places`` 读坐标，
      plan 只持有 ID 序列（§3.5 末段）。落点命名与坐标标定归 P4。
    * A 槽位数量 3 是 §3.5 给出的当前值（"数量 P4 落点标定后定"）；改动它需要同步
      改预检用例与 P4/P5 的落点资产，不在本任务里自由发挥。
"""

from __future__ import annotations

from typing import ClassVar

from qingyun.plans.base import BasePlan

__all__ = ["StrawberryPlan"]


class StrawberryPlan(BasePlan):
    """草莓分拣：不成熟 → C 料箱；成熟且面积 ≥ K → 优品槽位；其余成熟 → B 料箱。"""

    TASK_ID: ClassVar[str] = "strawberry"
    AREA_THRESHOLD_M2: ClassVar[float | None] = None    # P4 标定后回填实测 m² 值；预检拒绝 None
    A_PLACE_IDS: ClassVar[tuple[str, ...]] = (
        "strawberry_A_01",
        "strawberry_A_02",
        "strawberry_A_03",
    )   # 数量 P4 落点标定后定（D04：按顺序消耗，不回滚、不降级）
    B_PLACE_ID: ClassVar[str] = "strawberry_B_bin"
    C_PLACE_ID: ClassVar[str] = "strawberry_C_bin"
