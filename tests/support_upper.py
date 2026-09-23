"""upper 层离线替身：FakeVision / FakeArm / FakeASR / FakeLLM（P1-04）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2 文件责任表：``tests/support_upper.py`` = FakeVision/FakeArm/FakeASR/
           FakeLLM + 事件记录（**新增**）
        §1/§3.6：真实 vision/asr/cloud_model 在本阶段是**契约桩**（调用即抛硬错误），
           plan/main 的业务分支只能靠替身驱动；"替身必须抛真实异常类"
        §2 末段：业务替身通过测试内 monkeypatch 替换 ``qingyun.plans.base``
           命名空间里的模块引用（本模块只提供替身类，不做任何全局替换）
        §6.2/§9 末段："FakeArm 验证调用顺序与命令数量"
        §4 步骤 7：冷启动复位读 ``arm.motor`` / ``arm.params``（FakeArm 因此暴露
           这两个属性）
    docs/实施文档/P3_视觉检测与目标选择技术文档.md §3（get_target 契约面同名，
        替身只模仿"签名 + 抛真实异常类"这一层，不模仿算法）

边界（**替身不是实现**）：
    * FakeVision 不做候选筛选、不做几何/深度计算、不做 replay、不做排序，
      也不生成目标对象——它只把测试预先构造好的 ``VisionInterface`` 对象或异常实例
      按脚本顺序交出去。真实 ``qingyun/grabbing/vision.py`` 桩保持原样不动。
    * FakeArm 不做运动学/碰撞/夹爪物理仿真：它按脚本顺序返回 ``GraspResult``，
      并把每一次公开原语调用记成有序事件，供"顺序 + 数量"断言。真实运动回归
      （ArmController + MockMotor）不在本阶段重复建设（P5 覆盖）。
    * 异常类一律复用真实类对象（``NoTarget``/``VisionHardError``/``ASRHardError``/
      ``LLMHardError``）——替身自定义同名类会让 plan 的 except 子句认不出来，
      等于把故障洗白成业务分支。
    * 脚本耗尽时的默认行为是**抛错**，不是"重复最后一个成功值"：静默重复会让
      "循环没有多跑一轮"这类断言假通过（T05/T06 恰好依赖轮次精确）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from configs.common_interface import (
    GraspResult,
    GraspStatus,
    HoldingState,
    PlacePose,
    VisionInterface,
)
from qingyun.asr import ASRHardError
from qingyun.cloud_model import LLMHardError
from qingyun.grabbing.vision import NoTarget, VisionHardError

__all__ = [
    "Call",
    "EXHAUST_HARD_ERROR",
    "EXHAUST_REPEAT_LAST",
    "EventScript",
    "resolve_on_exhausted",
    "FakeArm",
    "FakeASR",
    "FakeLLM",
    "FakeVision",
    "FAKE_HOME_JOINTS_DEG",
    "FAKE_SAFE_WAYPOINTS_DEG",
    "make_arm_params",
    "make_result",
]

# ---------------------------------------------------------------------------
# 事件记录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """一次公开原语调用的完整台账（顺序 = 出现在替身 ``events`` 里的次序）。"""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def arg(self, index: int) -> Any:
        return self.args[index]


_MISS = object()

# 耗尽策略的两个内建行为用哨兵表示，而不是保留字符串：替身脚本的值本身可能就是
# 任意字符串（ASR 的 ""、LLM 的 "invalid"），只有哨兵才不会与之混淆。
EXHAUST_HARD_ERROR = object()
EXHAUST_REPEAT_LAST = object()

_BUILTIN_ALIASES: dict[str, Any] = {
    "hard_error": EXHAUST_HARD_ERROR,
    "repeat_last": EXHAUST_REPEAT_LAST,
}


def resolve_on_exhausted(mode: Any, aliases: dict[str, Any]) -> Any:
    """把替身各自的字符串别名换成事件脚本能安全消费的值/哨兵。

    非字符串（目标对象、异常实例、可调用）原样透传——这是"脚本耗尽后返回某个固定
    值/抛某个固定异常"的通用逃生口。
    """
    if isinstance(mode, str):
        table = {**_BUILTIN_ALIASES, **aliases}
        if mode not in table:
            raise ValueError(
                f"未知 on_exhausted 策略 {mode!r}，可用：{sorted(table)}，"
                "或直接给出值/异常实例/无参可调用"
            )
        return table[mode]
    return mode


class EventScript:
    """按顺序消费的事件脚本：元素是"返回值"或"要抛出的异常实例"。

    ``on_exhausted``（脚本耗尽后的行为）经 :func:`resolve_on_exhausted` 归一后是：

    * ``EXHAUST_HARD_ERROR``（字符串别名 ``"hard_error"``，默认）：抛
      ``error_factory`` 造的异常——替身各自的硬错误类或 ``AssertionError``，目的是
      让"循环比规格多跑了一轮"立刻可见；
    * ``EXHAUST_REPEAT_LAST``（别名 ``"repeat_last"``）：反复返回最后一个非异常事件；
    * 一个具体对象：值或异常实例，之后每次调用都用它；
    * 一个无参可调用：由它产出下一次的值或异常。
    """

    def __init__(
        self,
        events: Iterable[Any] = (),
        *,
        label: str,
        error_factory: Callable[[str], BaseException] = RuntimeError,
        on_exhausted: Any = EXHAUST_HARD_ERROR,
    ) -> None:
        self.events: list[Any] = list(events)
        self.label = label
        self.error_factory = error_factory
        self.on_exhausted = on_exhausted
        self.consumed = 0
        self.last_value: Any = _MISS

    def __len__(self) -> int:
        return len(self.events)

    @property
    def remaining(self) -> int:
        return len(self.events)

    def next(self) -> Any:
        """取出下一个事件；异常实例就地抛出。"""
        self.consumed += 1
        if self.events:
            item = self.events.pop(0)
        else:
            item = self._exhausted()
        if isinstance(item, BaseException):
            raise item
        self.last_value = item
        return item

    def _exhausted(self) -> Any:
        mode = self.on_exhausted
        message = (
            f"{self.label} 事件脚本已耗尽（第 {self.consumed} 次调用没有预设事件）："
            "用例的脚本/断言与业务循环的实际轮次不一致，拒绝静默放行"
        )
        if mode is EXHAUST_HARD_ERROR:
            raise self.error_factory(message)
        if mode is EXHAUST_REPEAT_LAST:
            if self.last_value is _MISS:
                raise self.error_factory(f"{self.label} 无历史值可重复（脚本一开始就空）")
            return self.last_value
        if callable(mode):
            return mode()
        return mode


# ---------------------------------------------------------------------------
# FakeVision
# ---------------------------------------------------------------------------


class FakeVision:
    """``qingyun.grabbing.vision`` 契约面的替身：configure / init / get_target。

    典型用法（``tests/test_plan_flow.py``）：
    ``monkeypatch.setattr(qingyun.plans.base, "vision", FakeVision([...]))``。
    脚本元素是 ``VisionInterface`` 对象（直接返回）或异常实例（就地抛出，
    必须是**真实**的 ``NoTarget`` / ``VisionHardError``）。

    本类**不含**任何图像处理、几何计算、候选筛选或 replay 逻辑。
    """

    # 契约面同名异常类：替身抛真实类（P1 §3.6 末段）。
    NoTarget = NoTarget
    VisionHardError = VisionHardError

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        on_exhausted: Any = "hard_error",
    ) -> None:
        self._script = EventScript(
            script,
            label="FakeVision.get_target",
            error_factory=VisionHardError,
            on_exhausted=resolve_on_exhausted(
                on_exhausted,
                {"no_target": NoTarget("FakeVision 脚本耗尽：按 NoTarget 处理（测试专用）")},
            ),
        )
        self.configure_calls: list[dict[str, Any]] = []
        self.init_calls: list[dict[str, Any]] = []
        self.get_target_calls: list[dict[str, Any]] = []

    # --- 断言辅助 ---

    @property
    def ignore_values(self) -> list[int]:
        """每次 get_target 收到的 ignore 实参序列（D03 跳选游标的行为证据）。"""
        return [call["ignore"] for call in self.get_target_calls]

    @property
    def scan_count(self) -> int:
        return len(self.get_target_calls)

    @property
    def remaining(self) -> int:
        return self._script.remaining

    # --- 契约面 ---

    def configure(self, thresholds: Any, config_path: Any) -> None:
        self.configure_calls.append({"thresholds": thresholds, "config_path": config_path})

    def init(self) -> None:
        self.init_calls.append({})

    def get_target(self, ignore: int = 0) -> VisionInterface:
        self.get_target_calls.append(
            {
                "ignore": ignore,
                "call_index": len(self.get_target_calls) + 1,
                "script_remaining": self._script.remaining,
            }
        )
        return self._script.next()


# ---------------------------------------------------------------------------
# FakeArm
# ---------------------------------------------------------------------------

# 回程路线的名义值：只为让 plan 有东西可转发、让用例能按值断言顺序与数量。
# **不是标定数据**，也不写进任何 profile（真机路线由 P4 标定、由运动模块消费）。
FAKE_HOME_JOINTS_DEG: tuple[float, ...] = (0.0, -60.0, 60.0, 0.0, 0.0)
FAKE_SAFE_WAYPOINTS_DEG: tuple[tuple[float, ...], ...] = (
    (12.0, -24.0, 30.0, -15.0, 5.0),
    (6.0, -36.0, 42.0, -9.0, 0.0),
)


def make_arm_params(
    *,
    safe_waypoints_deg: Sequence[Sequence[float]] = FAKE_SAFE_WAYPOINTS_DEG,
    home_joints_deg: Sequence[float] = FAKE_HOME_JOINTS_DEG,
    place_ids: Sequence[str] = (),
) -> SimpleNamespace:
    """plan 侧唯一读取的 params 形状：``workspace.safe_waypoints_deg`` +
    ``workspace.home_joints_deg``（§6.5 回程路线）。

    真实 ``MotionParams`` 里这两个字段是 float64 ndarray，所以替身同样给 ndarray，
    连"按列表顺序逐行迭代"这个语义一起复现。``places`` 一并给出（plan 不读，
    运动读），避免任何误用变成 AttributeError 之外还看不出来。
    """
    return SimpleNamespace(
        workspace=SimpleNamespace(
            safe_waypoints_deg=np.asarray(safe_waypoints_deg, dtype=np.float64),
            home_joints_deg=np.asarray(home_joints_deg, dtype=np.float64),
        ),
        places={
            pid: PlacePose(
                position=np.asarray([0.3, 0.0, 0.05], dtype=np.float64), yaw_deg=0.0
            )
            for pid in place_ids
        },
    )


@dataclass(frozen=True)
class _FailureRule:
    exc: BaseException
    at_call: int | None


class FakeArm:
    """``ArmController`` 公开原语面的替身：脚本化返回 + 全量事件记录。

    记录的东西与规格关心的东西一一对应（P1 §9 末段"验证调用顺序与命令数量"）：
    ``events`` 是按调用先后排列的 ``Call`` 列表，``call_counts`` 是逐原语次数，
    ``place_ids()`` / ``moved_positions()`` 是两条最常断言的序列。

    用 ``fail(name, exc, at_call=k)`` 注入"回程某原语抛异常"（T12）。
    ``grasp_and_place`` 脚本耗尽默认抛 ``AssertionError``（测试基建错误，
    不是业务状态码），绝不自造 SUCCESS。
    """

    def __init__(
        self,
        results: Sequence[Any] = (),
        params: Any = None,
        *,
        motor: Any = None,
        on_exhausted: Any = "hard_error",
    ) -> None:
        self.params = params if params is not None else make_arm_params()
        # cold_start_recover 等会话层代码会读 arm.motor；plan 不读，但替身保持同形。
        self.motor = motor
        self._results = EventScript(
            results,
            label="FakeArm.grasp_and_place",
            error_factory=AssertionError,
            on_exhausted=resolve_on_exhausted(on_exhausted, {}),
        )
        self.events: list[Call] = []
        self.call_counts: dict[str, int] = {}
        self._failures: dict[str, _FailureRule] = {}

    # --- 故障注入 ---

    def fail(self, name: str, exc: BaseException, *, at_call: int | None = None) -> None:
        """让 ``name`` 这个原语抛 ``exc``；``at_call=None`` 表示每次调用都抛。"""
        self._failures[name] = _FailureRule(exc, at_call)

    # --- 断言辅助 ---

    def command_sequence(self) -> list[str]:
        return [call.name for call in self.events]

    def calls_of(self, name: str) -> list[Call]:
        return [call for call in self.events if call.name == name]

    def count_of(self, name: str) -> int:
        return self.call_counts.get(name, 0)

    def place_ids(self) -> list[str]:
        return [call.args[1] for call in self.calls_of("grasp_and_place")]

    def grasp_targets(self) -> list[Any]:
        return [call.args[0] for call in self.calls_of("grasp_and_place")]

    def moved_positions(self) -> list[tuple[float, ...]]:
        """每次 move_joints 的目标关节角（按调用顺序）。"""
        return [call.args[0] for call in self.calls_of("move_joints")]

    def motion_command_count(self) -> int:
        """除 ``grasp_and_place`` 之外的运动/复位原语调用总数。

        封顶（D14）、失控（T11）、预算耗尽（T05）、契约违约（T24）这些分支的
        "零新运动命令 / 零追加串口调用"就靠这个数断言。
        """
        return sum(
            count for name, count in self.call_counts.items() if name != "grasp_and_place"
        )

    def reset_ledger(self) -> None:
        """清空事件与计数（保留脚本与故障注入）。"""
        self.events.clear()
        self.call_counts.clear()

    # --- 公开原语面（与 ArmController 同名同形） ---

    def _begin(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.events.append(Call(name, tuple(args), dict(kwargs)))
        call_index = self.call_counts[name] = self.call_counts.get(name, 0) + 1
        rule = self._failures.get(name)
        if rule is not None and (rule.at_call is None or rule.at_call == call_index):
            raise rule.exc

    def grasp_and_place(self, target: VisionInterface, place_id: str = "default") -> GraspResult:
        self._begin("grasp_and_place", target, place_id)
        return self._results.next()

    def move_joints(self, q_target_deg: Any, min_duration_s: float | None = None) -> None:
        # 归一成 tuple[float]：数组比较要按值，且事件台账必须可打印、可 == 断言。
        q = tuple(float(v) for v in np.asarray(q_target_deg, dtype=np.float64).ravel())
        self._begin("move_joints", q)
        if min_duration_s is not None:
            self.events[-1].kwargs["min_duration_s"] = min_duration_s

    def move_cartesian_top_down(
        self, tcp_xyz: Any, tcp_yaw_deg: float, min_duration_s: float | None = None
    ) -> None:
        p = tuple(float(v) for v in np.asarray(tcp_xyz, dtype=np.float64).ravel())
        self._begin("move_cartesian_top_down", p, float(tcp_yaw_deg))

    def open_gripper(self) -> None:
        self._begin("open_gripper")

    def close_gripper(self) -> HoldingState:
        self._begin("close_gripper")
        return HoldingState.EMPTY

    def wait_settled(self, target_deg: Any, timeout_s: float) -> bool:
        q = tuple(float(v) for v in np.asarray(target_deg, dtype=np.float64).ravel())
        self._begin("wait_settled", q, float(timeout_s))
        return True

    def reset_fault(self) -> None:
        self._begin("reset_fault")


# ---------------------------------------------------------------------------
# 抓取结果构造
# ---------------------------------------------------------------------------

# 各状态在真实运动侧的"典型组合"（stage/holding/recovery_required）。
# 依据：arm_control._refuse（臂未动 → recovery_required=False、holding=UNKNOWN、
# stage=CHECK）、_early_fail（预规划失败同上但 holding 保留）、
# _fault_result（运动后 → stage=FAULT_HOLD 或 FAULT_UNCONTROLLED、recovery_required=True）。
_TYPICAL: dict[GraspStatus, tuple[str, HoldingState, bool]] = {
    GraspStatus.SUCCESS: ("DONE", HoldingState.EMPTY, False),
    GraspStatus.INVALID_INPUT: ("CHECK", HoldingState.UNKNOWN, False),
    GraspStatus.CONFIG_INVALID: ("CHECK", HoldingState.UNKNOWN, False),
    GraspStatus.IK_FAILED: ("CHECK", HoldingState.UNKNOWN, False),
    GraspStatus.PLAN_INVALID: ("CHECK", HoldingState.UNKNOWN, False),
    GraspStatus.COLLISION: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.GRASP_MISS: ("FAULT_HOLD", HoldingState.EMPTY, True),
    GraspStatus.GRASP_UNCERTAIN: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.GRIP_SLIP: ("FAULT_HOLD", HoldingState.HOLDING, True),
    GraspStatus.PLACE_FAIL: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.COMM_ERROR: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.TRACKING_ERROR: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.TIMEOUT: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
    GraspStatus.ABORTED: ("FAULT_HOLD", HoldingState.UNKNOWN, True),
}


def make_result(
    status: GraspStatus,
    *,
    stage: str | None = None,
    holding: HoldingState | None = None,
    recovery_required: bool | None = None,
    reason: str | None = None,
    place_id: str = "",
) -> GraspResult:
    """构造一条 ``GraspResult``；未显式给出的字段取该状态的**典型组合**。

    "典型"只服务于可读性：要演练 §6.5 的组合违约行（T13/T17），用例显式覆盖
    ``recovery_required`` / ``holding`` / ``stage`` 即可，本工厂不做任何合法性判断
    ——判"合法与否"是 plan 的职责，替身没有资格替它圆场。
    """
    typical_stage, typical_holding, typical_recovery = _TYPICAL[status]
    return GraspResult(
        status=status,
        stage=typical_stage if stage is None else stage,
        reason=(f"测试注入：{status.value}@{typical_stage}" if reason is None else reason),
        place_id=place_id,
        holding=typical_holding if holding is None else holding,
        recovery_required=(
            typical_recovery if recovery_required is None else recovery_required
        ),
    )


# ---------------------------------------------------------------------------
# FakeASR / FakeLLM
# ---------------------------------------------------------------------------


class FakeASR:
    """``qingyun.asr`` 契约面替身：init(cfg) / listen_and_transcribe()。

    脚本元素是转写文本字符串或 ``ASRHardError`` 实例（真实类）。耗尽默认抛
    ``ASRHardError``；``on_exhausted="empty"`` 可让后续每轮都返回空串（§5 的
    "空转写 continue" 分支）。本类不含录音/VAD/云请求。
    """

    HardError = ASRHardError

    def __init__(self, script: Sequence[Any] = (), *, on_exhausted: Any = "hard_error") -> None:
        self._script = EventScript(
            script,
            label="FakeASR.listen_and_transcribe",
            error_factory=ASRHardError,
            on_exhausted=resolve_on_exhausted(on_exhausted, {"empty": ""}),
        )
        self.init_calls: list[dict[str, Any]] = []
        self.listen_calls: list[dict[str, Any]] = []
        self.results: list[str] = []

    def init(self, cfg: Any) -> None:
        self.init_calls.append({"cfg": cfg})

    def listen_and_transcribe(self) -> str:
        self.listen_calls.append({"call_index": len(self.listen_calls) + 1})
        text = self._script.next()
        self.results.append(text)
        return text


class FakeLLM:
    """``qingyun.cloud_model`` 契约面替身：init(cfg) / select_plan(text)。

    脚本元素是 task 字符串（``"strawberry"`` / ``"invalid"`` / 违约值）或
    ``LLMHardError`` 实例。耗尽默认抛 ``LLMHardError``；
    ``on_exhausted="invalid"`` 让后续每轮返回 invalid。本类不发起任何云请求。
    """

    HardError = LLMHardError

    def __init__(self, script: Sequence[Any] = (), *, on_exhausted: Any = "hard_error") -> None:
        self._script = EventScript(
            script,
            label="FakeLLM.select_plan",
            error_factory=LLMHardError,
            on_exhausted=resolve_on_exhausted(on_exhausted, {"invalid": "invalid"}),
        )
        self.init_calls: list[dict[str, Any]] = []
        self.select_calls: list[dict[str, Any]] = []

    @property
    def texts(self) -> list[str]:
        return [call["text"] for call in self.select_calls]

    def init(self, cfg: Any) -> None:
        self.init_calls.append({"cfg": cfg})

    def select_plan(self, text: str) -> str:
        self.select_calls.append({"text": text, "call_index": len(self.select_calls) + 1})
        return self._script.next()
