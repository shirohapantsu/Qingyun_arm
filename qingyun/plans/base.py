"""plans 业务核心：计数器矩阵、分级、落点分配、结果分发顺序、恢复矩阵与完成出口（P1-04）。

规范来源（本文逐条实现，编号即落点）：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §3.5  plans 的公开签名（RETRYABLE_STATUSES / MAX_CONSECUTIVE_FAILURES /
              BasePlan 的四个 ClassVar 与 __init__/run/classify/next_place_id）
        §6.1  计数器矩阵（no_target_streak / ignore / consecutive_failures / a_count
              在"新建 Plan / 视觉成功 / NoTarget / SUCCESS / 重试类失败"五种事件下的
              唯一合法变化；D02/D03/D04）
        §6.2  run() 分发顺序（固定不可重排：0 视觉 → 契约复核 → 分级 → 分配 →
              attempt+1 → assign → 抓取 → **先审计 grasp_result** →
              1 失控阶段 → 2 SUCCESS → 3 直死类 → 4 重试类计数 → 5 ignore 封顶 →
              6 失败预算 → 7 恢复）
        §6.3  next_place_id（D04：分配即消耗、不回滚、不降级）
        §6.4  视觉返回值契约复核（plans 侧只做浅校验，深层几何归运动）
        §6.5  恢复矩阵（D08 定稿；D14 后仅服务"继续下一轮"路径）
        §6.6  _finish（任务完成出口；D14 封顶**不**调它）
        §7    日志事件最低字段（assign / grasp_result / recovery / task_done）
        §3.4  runlog 上下文：scan_id 每轮递增并包住"扫描→分配→抓取→恢复"
        §3.6  模块桩引用方式（``from qingyun.grabbing import vision``；
              NoTarget/VisionHardError 始终从桩/真实现导入）
    docs/实施文档/README.md 决策：
        D02（连续失败预算 N=10）、D03（失败后临时跳选，成功或 NoTarget 即归零，
        无永久排除名单）、D04（优品槽位分配即消耗，失败不回退，耗尽即终止）、
        D05（分级阈值 K 按果品独立，本期只有草莓；None 表示 P4 未标定）、
        D08（运动后 IK_FAILED/PLAN_INVALID 一律 terminate，不区分 holding、
        不设自动/人工恢复分支）、D13（人工确认"回车即确认"，EOF/Ctrl-C → terminate；
        JSONL 日志纳入）、D14（ignore 封顶 → terminate("NO_GRASPABLE_TARGET")，
        不恢复、不确认、不回 home，正常完成出口只剩 NoTarget×3）。

注入边界（本文件的一切外部依赖都可被测试整体替换，P1 §2 末段/§3.6 末段）：
    * ``vision``：**模块对象**引用，plan 只经 ``vision.get_target(...)`` 调用；
      测试把 ``qingyun.plans.base.vision`` 换成 ``tests.support_upper.FakeVision``
      实例即可驱动整条业务链（真实 vision 桩保持不动）。
    * ``NoTarget`` / ``VisionHardError``：**按名字**从桩/真实现导入后用于 except。
      这两者是 P1 §3.6/P3 §3 冻结的常驻契约（替换桩体时类名不变），按名导入可以让
      "整体替换 vision 名字"不至于把 except 子句里的异常类一起换掉——替身必须抛
      真实类，这里也只认这两个真实类（分层由 P1-03 的用例钉死）。
    * ``shutdown`` / ``runlog``：同样按模块名引用，测试可注入替身模块。
    * ``_input``：模块级可替换引用（D13 的"回车即确认"唯一人工交互点）。

不做（范围外）：任何视觉/几何计算（P3）、任何舵机或运动原语（运动模块）、
配置装配与冷启动、主循环与 task_instance_id 分配（P1-05 的 main）。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import numpy as np
from configs.common_interface import GraspResult, GraspStatus, HoldingState, VisionInterface
from qingyun import runlog, shutdown
from qingyun.grabbing import vision
from qingyun.grabbing.vision import NoTarget, VisionHardError

if TYPE_CHECKING:  # pragma: no cover — 仅注解，避免 plans 拉起 placo/URDF
    from qingyun.grabbing.arm_control import ArmController

__all__ = [
    "RETRYABLE_STATUSES",
    "MAX_CONSECUTIVE_FAILURES",
    "NO_TARGET_STREAK_LIMIT",
    "STAGE_FAULT_UNCONTROLLED",
    "BasePlan",
]

# --- §3.5 逐字常量 -----------------------------------------------------------

# 可重试状态集合（P1 §3.5 逐字）：只有这五种失败允许"跳选下一个目标"。
# 配置/通信/碰撞/放置/中止等一律直死（D02 明确"原直死错误不因本决定改为忽略"）。
RETRYABLE_STATUSES = frozenset({
    GraspStatus.IK_FAILED,
    GraspStatus.PLAN_INVALID,
    GraspStatus.GRASP_MISS,
    GraspStatus.GRASP_UNCERTAIN,
    GraspStatus.GRIP_SLIP,
})

# D02：连续失败预算 N=10（成功清零；NoTarget **不**清零，见 §6.1 与 T08）。
MAX_CONSECUTIVE_FAILURES = 10

# §6.2/§9 T03/T19：连续 NoTarget 三连是唯一的正常完成出口（D14 后无第二个）。
NO_TARGET_STREAK_LIMIT = 3

# 运动侧阶段名（qingyun/grabbing/arm_control.py 的 STAGE_FAULT_UNCONTROLLED）。
# 不 import arm_control：那会把 placo/URDF/IK 栈拉进 plans 的 import 链，
# 而 plan 只需要这个字符串常量；两侧的一致性由 tests/test_plan_flow.py 钉死。
STAGE_FAULT_UNCONTROLLED = "FAULT_UNCONTROLLED"

# D13：人工确认的唯一输入口。经模块级名字引用，测试可整体替换（回车即确认）。
_input: Callable[[str], str] = input


class _ContractViolation(Exception):
    """内部短路信号：§6.4 复核里任一条不满足时抛出，由统一出口转 terminate。

    它**不**外泄给调用方——只在 ``_validate_target_contract`` 的 try 块内被捕获，
    作用是让多条判据共用一段"违约 → terminate"的收尾，避免每条判据各写一次
    终止调用（那种写法最容易在中间漏计数字段）。
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _is_plain_number(value: Any) -> bool:
    """「数值」判定：排除 bool（True 当 1 用是最常见的静默坑），
    并接受 numpy 标量（P3 真实现完全可能下发 np.float64）。"""
    return isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)


def _is_integral(value: Any) -> bool:
    """「整数」判定：同上排除 bool，并接受 np.integer。"""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


class BasePlan:
    """一个任务实例的完整业务循环（P1 §3.5）。

    实例即任务：``main`` 为每个有效任务新建一个实例（§5），计数器从 0 起，
    因此本类**不**提供任何类级/全局计数状态，也不允许实例间复用。
    """

    # --- §3.5 四个 ClassVar（预检 §3.2 第 1/3 条按这些名字读） ---
    TASK_ID: ClassVar[str] = ""
    AREA_THRESHOLD_M2: ClassVar[float | None] = None      # D05；None = P4 未回填
    A_PLACE_IDS: ClassVar[tuple[str, ...]] = ()           # D04：分配即消耗
    B_PLACE_ID: ClassVar[str] = ""
    C_PLACE_ID: ClassVar[str] = ""

    def __init__(self, arm: ArmController) -> None:
        """全部计数器归零（§6.1 第一行"新建 Plan"）。"""
        self.arm = arm
        # §6.1 四个业务计数器
        self.no_target_streak = 0
        self.ignore = 0
        self.consecutive_failures = 0
        self.a_count = 0
        # §6.6 完成报告所需
        self.success_count = 0
        self.failure_counts: dict[str, int] = {}
        # §3.4 日志关联计数器（每任务从 0 起）
        self.scan_id = 0
        self.attempt_id = 0

    # =====================================================================
    # 1. 主循环（§6.1 计数器 + §6.2 分发顺序 + §3.4 上下文）
    # =====================================================================

    def run(self) -> None:
        """执行任务循环，直到 NoTarget 三连（唯一正常出口）或任一 terminate 分支。

        §6.2 的分发顺序是**固定**的，本文逐行对应伪代码，不得重排：
        其中"封顶先于预算"（第 5 步 vs 第 6 步）与"审计先于一切终止分支"
        是 D14/§6.2 的两条硬要求，各有专门用例咬合（T05/T06/T26）。

        每轮开头 ``scan_id += 1`` 并用 ``runlog.context(scan_id=...)`` 包住
        "本轮扫描 → 分配 → 抓取 → 恢复"（§6.2 注）：``with`` 保证 continue、
        return 与异常退出（含 terminate 抛的 SystemExit）三种离开方式都释放上下文。
        """
        while True:
            self.scan_id += 1
            with runlog.context(scan_id=self.scan_id):
                # ---- 0. 一次完整视觉扫描（§6.2 的 try 块） ----
                # ignore 是"本次请求跳过前 ignore 个"的临时跳选游标（D03）。
                requested_ignore = self.ignore
                try:
                    target = vision.get_target(self.ignore)
                except NoTarget:
                    # §6.1：NoTarget → ignore 归零、no_target_streak +1；
                    # 三连即任务正常完成（§6.6/D14 唯一出口）。三连之间不加 sleep，
                    # 每次都是真实新扫描（§6.2 末段，T03）。
                    self.ignore = 0
                    self.no_target_streak += 1
                    if self.no_target_streak >= NO_TARGET_STREAK_LIMIT:
                        self._finish("NO_TARGET_STREAK")
                        return
                    continue
                except VisionHardError as exc:
                    # §5/§6.2：视觉管线故障不参与 NoTarget 计数（T23）。
                    self._terminate("VISION_HARD_ERROR", f"视觉管线故障：{exc}")

                # ---- 视觉成功返回：streak 归零（§6.1），ignore 保持（跳选继续） ----
                self.no_target_streak = 0
                self._validate_target_contract(target)          # §6.4

                grade = self.classify(target)
                place_id = self.next_place_id(grade)            # A 越界 → terminate（§6.3）
                self.attempt_id += 1
                runlog.event(
                    "assign",
                    grade=grade,
                    place_id=place_id,
                    a_count=self.a_count,
                    valid_count=target.valid_count,
                    ignore_used=self.ignore,
                    attempt_id=self.attempt_id,
                )

                result = self.arm.grasp_and_place(target, place_id)

                # §6.2：先记录完整 grasp_result，非 SUCCESS 累计 failure_counts；
                # 任何直死/封顶分支之前都必须完成这次审计（否则最后一次失败无据可查）。
                self._audit_grasp_result(
                    result,
                    place_id=place_id,
                    grade=grade,
                    ignore_used=requested_ignore,
                    valid_count=target.valid_count,
                )

                # ---- 1. 失控阶段最先判（§6.2/T11）：臂未被确认保持，不谈成功也不谈恢复 ----
                if result.stage == STAGE_FAULT_UNCONTROLLED:
                    self._terminate(
                        "FAULT_UNCONTROLLED",
                        f"运动侧报告失控阶段（未被确认保持）：status={result.status.value} "
                        f"stage={result.stage} "
                        f"holding={result.holding.value} reason={result.reason}",
                    )

                # ---- 2. SUCCESS：ignore/连续失败计数归零，a_count 不回滚（D04） ----
                if result.status is GraspStatus.SUCCESS:
                    self.ignore = 0
                    self.consecutive_failures = 0
                    self.success_count += 1
                    continue

                # ---- 3. 非重试类：以状态名直死（§6.2/D02） ----
                if result.status not in RETRYABLE_STATUSES:
                    self._terminate(
                        result.status.value,
                        f"不可重试的运动结果：stage={result.stage} "
                        f"holding={result.holding.value} reason={result.reason}",
                    )

                # ---- 4. 重试类：跳选游标 +1、连续失败预算 +1 ----
                self.ignore += 1
                self.consecutive_failures += 1

                # ---- 5. D14 封顶：本次递增后 ignore ≥ 当次返回的 valid_count ----
                # 必须先于预算判定（T06 用同一次失败区分两者）；封顶后不恢复、
                # 不人工确认、不回 home、不发任何新运动命令（§6.2 末段）。
                if self.ignore >= target.valid_count:
                    self._terminate(
                        "NO_GRASPABLE_TARGET",
                        f"ignore={self.ignore} ≥ valid_count={target.valid_count}，"
                        f"末次 {result.status.value}/{result.stage}",
                    )

                # ---- 6. D02 预算：连续失败 ≥ 10，同样不恢复、臂冻结 ----
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self._terminate(
                        "FAILURE_BUDGET_EXHAUSTED",
                        f"连续失败预算耗尽：consecutive_failures={self.consecutive_failures} "
                        f"≥ {MAX_CONSECUTIVE_FAILURES}，"
                        f"末次 {result.status.value}/{result.stage}，"
                        f"ignore={self.ignore}/valid_count={target.valid_count}，"
                        f"失败码计数={self._failure_counts_text()}",
                    )

                # ---- 7. 恢复矩阵（§6.5）：只有这里可能搬臂/等人确认 ----
                self._recover(result)

    # =====================================================================
    # 2. 分级与落点分配（§3.5 classify / §6.3 next_place_id）
    # =====================================================================

    def classify(self, target: VisionInterface) -> str:
        """固定分级：不成熟 → C；否则面积 ≥ K → A，余下 B（§3.5 末段）。

        ``ripe is False`` 用身份比较而不是 ``not``：调用前 §6.4 复核已保证它是严格
        bool，这里再按字面判一次，万一上游放松也不会把 ``None``/``0`` 误读成 C。
        面积单位是 m²（D05），**等于阈值归 A**。
        """
        if target.ripe is False:
            return "C"
        k = self.AREA_THRESHOLD_M2
        if k is None:
            self._terminate(
                "PLAN_CONFIG_INVALID",
                f"{type(self).__name__}.AREA_THRESHOLD_M2 为 None：K 未标定（D05/P4 回填），"
                "预检应已拒绝运行，绝不能靠 classify 默认填值",
            )
        if isinstance(k, bool) or not _is_plain_number(k):
            self._terminate(
                "PLAN_CONFIG_INVALID",
                f"AREA_THRESHOLD_M2 应为数值，实际 {type(k).__name__}",
            )
        kf = float(k)
        if not math.isfinite(kf) or kf <= 0.0:
            self._terminate(
                "PLAN_CONFIG_INVALID",
                f"AREA_THRESHOLD_M2 必须是有限正数（单位 m²），实际 {kf}",
            )
        area = float(target.length_m) * float(target.width_m)
        return "A" if area >= kf else "B"

    def next_place_id(self, grade: str) -> str:
        """按品级取落点 ID（§6.3）。

        D04：A 名额**分配即消耗**——不回滚、不降级（耗尽不降 B、不循环覆盖 A_01），
        任何运动结果（含运动前拒绝）都不回调 ``a_count``。
        """
        if grade == "A":
            if self.a_count >= len(self.A_PLACE_IDS):
                self._terminate(
                    "A_SLOTS_EXHAUSTED",
                    "优品槽位耗尽（GRASP_MISS 空耗捅穿摆桌保证）",
                )
            pid = self.A_PLACE_IDS[self.a_count]
            self.a_count += 1
            return pid
        if grade == "B":
            return self.B_PLACE_ID
        if grade == "C":
            return self.C_PLACE_ID
        self._terminate("PLAN_CONFIG_INVALID", f"未知 grade {grade!r}")
        raise AssertionError("unreachable")  # _terminate 是 NoReturn，显式收尾供类型检查

    # =====================================================================
    # 3. 视觉返回值契约复核（§6.4，plans 侧浅校验）
    # =====================================================================

    def _validate_target_contract(self, target: VisionInterface) -> None:
        """复核 §6.4 的五条判据；违约 → ``terminate("VISION_CONTRACT_VIOLATION")``。

        判据（逐条对应规格文字）：
            1. ``length_m >= width_m > 0`` 且两者为有限数值（bool 不算数值）；
               规格是一条链式判据，这里拆成"length 为正""width 为正""length 不小于
               width"三句，只为让 terminate detail 指到具体坏字段，判定的集合不变；
            2. ``ripe`` 是严格 bool；
            3. ``valid_count`` 是正整数且 **> 本次请求的 ignore**；
            4. ``yaw_deg`` 是有限数值且落在 ``[-90, 90)``；
            5. 六字段全部存在（``position`` 只做存在性检查）。
        违约时**不触发抓取、不计 NoTarget、不消耗 A 槽位**：本函数排在 classify
        与 next_place_id 之前，天然满足（T24 断言零抓取调用）。
        ``position``/``yaw`` 的深层校验（shape、dtype、有限性细节、可达性）归运动
        （INVALID_INPUT 直死），这里不重复实现；但 yaw 的角度区间是 §6.4 点名的
        plans 侧判据，必须在此检查。
        """
        try:
            self._require_contract(target)
        except _ContractViolation as exc:
            self._terminate("VISION_CONTRACT_VIOLATION", exc.detail)

    def _require_contract(self, target: VisionInterface) -> None:
        fields = ("position", "yaw_deg", "length_m", "width_m", "ripe", "valid_count")
        missing = [name for name in fields if not hasattr(target, name)]
        if missing:
            raise _ContractViolation(f"目标缺少 VisionInterface 字段 {missing}")

        length = target.length_m
        width = target.width_m
        for name, value in (("length_m", length), ("width_m", width)):
            if not _is_plain_number(value):
                raise _ContractViolation(
                    f"{name} 应为非 bool 有限数值，实际 {type(value).__name__}={value!r}"
                )
            if not math.isfinite(float(value)):
                raise _ContractViolation(f"{name} 非有限：{value!r}")
        if not float(length) > 0.0:
            raise _ContractViolation(f"length_m 必须为正，实际 {length!r}")
        if not float(width) > 0.0:
            raise _ContractViolation(f"width_m 必须为正，实际 {width!r}")
        if float(length) < float(width):
            raise _ContractViolation(
                f"length_m 不得小于 width_m（视觉侧须按 OBB 长短轴给出）："
                f"length_m={length!r} width_m={width!r}"
            )

        ripe = target.ripe
        if not isinstance(ripe, bool):
            raise _ContractViolation(f"ripe 必须是严格 bool，实际 {type(ripe).__name__}={ripe!r}")

        valid_count = target.valid_count
        if not _is_integral(valid_count):
            raise _ContractViolation(
                f"valid_count 必须是整数，实际 {type(valid_count).__name__}={valid_count!r}"
            )
        if valid_count <= 0:
            raise _ContractViolation(f"valid_count 必须是正整数，实际 {valid_count!r}")
        if valid_count <= self.ignore:
            raise _ContractViolation(
                f"valid_count={valid_count} 未大于本次请求的 ignore={self.ignore}"
                "（视觉对 ignore ≥ valid_count 应抛 NoTarget，返回它是契约违约）"
            )

        yaw = target.yaw_deg
        if not _is_plain_number(yaw):
            raise _ContractViolation(f"yaw_deg 应为非 bool 数值，实际 {type(yaw).__name__}")
        if not math.isfinite(float(yaw)):
            raise _ContractViolation(f"yaw_deg 非有限：{yaw!r}")
        if not (-90.0 <= float(yaw) < 90.0):
            raise _ContractViolation(f"yaw_deg={yaw!r} 不在 [-90,90) 区间内")

    # =====================================================================
    # 4. 恢复矩阵（§6.5，D08 定稿；D14 封顶已在 §6.2 第 5 步终止，不进这里）
    # =====================================================================

    def _recover(self, result: GraspResult) -> None:
        """按 §6.5 矩阵处置一次重试类失败。

        | 情形 | 处置 |
        |---|---|
        | IK_FAILED / PLAN_INVALID 且 ``recovery_required=False`` | 运动前失败（臂未动、
          在 home、空爪）：不 reset、不搬臂，直接进入下一轮视觉（ignore 已 +1）|
        | IK_FAILED / PLAN_INVALID 且 ``recovery_required=True`` | **terminate**
          （D08：一律终止，不区分 holding、不设自动/人工分支），臂冻结于失败位 |
        | GRASP_MISS 且 True 且 ``holding=EMPTY`` | 自动：reset_fault → 按
          ``params.workspace.safe_waypoints_deg`` 列表顺序逐个 move_joints →
          move_joints(home) → 下一轮 |
        | GRASP_UNCERTAIN / GRIP_SLIP 且 True | 人工：打印现场状态 → input 回车确认
          （D13）→ reset_fault → 中转 → home → 下一轮 |
        | 其余状态/标志组合 | ``RESULT_CONTRACT_VIOLATION``（保留原 result 信息）|
        | reset 或任一回程原语异常 | ``RECOVERY_FAILED``，不追加 ignore 再试恢复 |

        回程路线冻结为 profile ``safe_waypoints_deg`` 列表顺序 + home，与冷启动复位
        一致（§6.5 末段）；正常 SUCCESS 已由运动 RETURN 保证回 home，不额外回零。
        """
        status = result.status

        if status in (GraspStatus.IK_FAILED, GraspStatus.PLAN_INVALID):
            if result.recovery_required:
                # D08：运动后规划失败一律终止。terminate 码沿用状态名（与 §6.2
                # 第 3 步"以状态名直死"同一族语义），detail 保留 stage/holding/reason。
                self._terminate(
                    status.value,
                    f"D08：运动后 {status.value}（recovery_required=True）一律终止，"
                    f"不自动恢复、不人工确认，臂冻结于失败位："
                    f"stage={result.stage} holding={result.holding.value} "
                    f"reason={result.reason}",
                )
            # 运动前失败：无 reset、无搬臂，直接下一轮。仍记一条 recovery（方式=skip）
            # 让日志能说清"这一轮为什么没有回程动作"（§7 recovery 行）。
            runlog.event(
                "recovery",
                mode="skip",
                status=status.value,
                stage=result.stage,
                holding=result.holding.value,
                reason=result.reason,
                reset="not_needed",
                waypoint_moves=0,
                home="not_needed",
                attempt_id=self.attempt_id,
            )
            return

        # 以下三种（MISS/UNCERTAIN/SLIP）都是"运动已经开始后"的可重试失败。
        if not result.recovery_required:
            self._result_contract_violation(
                result, f"重试类 {status.value} 却 recovery_required=False"
            )
        if status is GraspStatus.GRASP_MISS and result.holding is not HoldingState.EMPTY:
            # MISS 的定义就是"确定闭到空爪"；报 HOLDING/UNKNOWN 的 MISS 无从判定
            # 现场是否还带着物体，绝不能自动搬臂，也不能当 UNCERTAIN 走人工。
            self._result_contract_violation(
                result,
                f"GRASP_MISS 却 holding={result.holding.value}（空爪确定才允许自动回程）",
            )

        manual = status in (GraspStatus.GRASP_UNCERTAIN, GraspStatus.GRIP_SLIP)
        mode = "manual" if manual else "auto"
        if manual:
            self._confirm_scene(result)
        self._return_route(result, mode=mode)

    def _confirm_scene(self, result: GraspResult) -> None:
        """打印现场状态并阻塞等待人工确认（§6.5 人工行；D13）。

        "回车即确认"：不要求特定文本，空回车同样算确认——所以这里只看"input 正常
        返回"这一事件，不解释返回值。EOFError / KeyboardInterrupt（Ctrl-C）一律
        ``MANUAL_CONFIRM_ABORTED``：确认没完成就**不** reset、**不**搬臂（T16）。
        确认期间不后台请求视觉、不发任何指令（同步阻塞天然保证）。
        """
        runlog.console(
            f"[需要人工确认] status={result.status.value} stage={result.stage} "
            f"holding={result.holding.value}"
        )
        runlog.console(f"[需要人工确认] reason={result.reason}")
        try:
            _input("确认物体已处置、可继续后回车...")
        except (EOFError, KeyboardInterrupt):
            self._terminate(
                "MANUAL_CONFIRM_ABORTED",
                f"人工确认被中断（EOF/Ctrl-C），未执行任何复位或回程："
                f"status={result.status.value} stage={result.stage} "
                f"holding={result.holding.value}",
            )

    def _return_route(self, result: GraspResult, *, mode: str) -> None:
        """复位 + 回程：reset_fault → 逐个 safe_waypoints → home（§6.5 第三/四行）。

        任一步抛异常（含 move 未到位——运动原语到位失败本身就以异常形式上报）
        立即 ``RECOVERY_FAILED``，不再追加 ignore、不再试恢复（§6.5 末行）。
        ``wait_settled`` 不在 plan 的使用范围：到位保证由 ``move_joints`` 原语内部
        负责，规格的文字是"任一回程原语异常"。
        """
        try:
            route = [
                [float(v) for v in list(wp)]
                for wp in self.arm.params.workspace.safe_waypoints_deg
            ]
            home = [float(v) for v in list(self.arm.params.workspace.home_joints_deg)]
        except Exception as exc:  # pragma: no cover — 回程路线读自已校验 profile
            self._recovery_failed(result, mode, "读取 params.workspace 回程路线", exc)

        try:
            self.arm.reset_fault()
        except Exception as exc:
            self._recovery_failed(result, mode, "reset_fault", exc)

        moved = 0
        for index, waypoint in enumerate(route):
            try:
                self.arm.move_joints(waypoint)
            except Exception as exc:
                self._recovery_failed(
                    result, mode, f"move_joints(safe_waypoints_deg[{index}])", exc
                )
            moved += 1
        try:
            self.arm.move_joints(home)
        except Exception as exc:
            self._recovery_failed(result, mode, "move_joints(home_joints_deg)", exc)

        runlog.event(
            "recovery",
            mode=mode,
            status=result.status.value,
            stage=result.stage,
            holding=result.holding.value,
            reason=result.reason,
            reset="ok",
            waypoint_moves=moved,
            home="ok",
            confirm="manual_confirmed" if mode == "manual" else "not_required",
            attempt_id=self.attempt_id,
        )

    def _recovery_failed(
        self, result: GraspResult, mode: str, step: str, exc: BaseException
    ) -> NoReturn:
        """§6.5 末行：回程任一步失败 → 立即统一退出（不追加 ignore、不重试恢复）。"""
        runlog.event(
            "recovery",
            mode=mode,
            status=result.status.value,
            stage=result.stage,
            holding=result.holding.value,
            reason=result.reason,
            failed_step=step,
            error=f"{type(exc).__name__}: {exc}",
            attempt_id=self.attempt_id,
        )
        self._terminate(
            "RECOVERY_FAILED",
            f"恢复失败于 {step}（方式={mode}，status={result.status.value}，"
            f"stage={result.stage}，holding={result.holding.value}）："
            f"{type(exc).__name__}: {exc}；不追加 ignore 再试恢复，臂冻结于当前位",
        )

    def _result_contract_violation(self, result: GraspResult, detail: str) -> NoReturn:
        """§6.5 组合违约行：保留原 result 的全部字段供定位。"""
        self._terminate(
            "RESULT_CONTRACT_VIOLATION",
            f"{detail}：运动返回组合违反 §6.5 矩阵，未执行任何恢复动作。"
            f"原 result：status={result.status.value} stage={result.stage} "
            f"holding={result.holding.value} recovery_required={result.recovery_required} "
            f"place_id={result.place_id!r} reason={result.reason}",
        )

    # =====================================================================
    # 5. 审计与完成出口（§6.2 第 0 步之后 / §6.6）
    # =====================================================================

    def _audit_grasp_result(
        self,
        result: GraspResult,
        *,
        place_id: str,
        grade: str,
        ignore_used: int,
        valid_count: int,
    ) -> None:
        """记录本次运动的**完整**返回，并累计失败码计数（§6.2/§7 grasp_result 行）。

        排在分发顺序 1–7 之前，因此"直死/封顶"的最后一次失败也一定有审计行；
        ``failure_counts`` 只对非 SUCCESS 累计，键为状态名（§6.6 完成报告与
        终止后的日志复盘都靠它）。
        """
        runlog.event(
            "grasp_result",
            status=result.status.value,
            stage=result.stage,
            reason=result.reason,
            holding=result.holding.value,
            recovery_required=result.recovery_required,
            attempt_id=self.attempt_id,
            place_id=place_id,
            grade=grade,
            ignore_used=ignore_used,
            valid_count=valid_count,
        )
        if result.status is not GraspStatus.SUCCESS:
            key = result.status.value
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

    def _finish(self, reason: str) -> None:
        """任务完成出口（§6.6）：一条 ``task_done`` 事件 + 一行终端"任务完成"。

        只由 NoTarget 三连触发（D14 后唯一正常出口）；D14 封顶走 terminate，
        **不**调本方法。终端行必须含"任务完成"字样，同时把成功数、失败码计数与
        已分配优品槽位和盘托出，不掩盖"槽位已消耗但没抓走"的未抓走目标。
        """
        runlog.event(
            "task_done",
            reason=reason,
            success=self.success_count,
            failure_counts=dict(self.failure_counts),
            a_slots_used=self.a_count,
            no_target_streak=self.no_target_streak,
            consecutive_failures=self.consecutive_failures,
            ignore=self.ignore,
            attempts=self.attempt_id,
        )
        runlog.console(
            f"任务完成：结束原因={reason}，成功={self.success_count}，"
            f"失败码计数={self._failure_counts_text()}，"
            f"已分配优品槽位={self.a_count}"
        )

    def _failure_counts_text(self) -> str:
        """失败码计数的紧凑文本（终端与 terminate detail 共用同一份渲染）。"""
        return json.dumps(self.failure_counts, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    # =====================================================================
    # 6. 终止出口
    # =====================================================================

    @staticmethod
    def _terminate(status: str, detail: str) -> NoReturn:
        """唯一失败出口：交 ``qingyun.shutdown.terminate``（P1 §3.3）。

        经模块名引用，测试可整体替换 ``qingyun.plans.base.shutdown``；默认实现
        打印 status/detail/所处阶段/现场处置指引并 ``sys.exit(1)``，不发任何舵机
        指令、不回 home、不关扭矩（CAL-052 底线）。
        """
        shutdown.terminate(status, detail)
