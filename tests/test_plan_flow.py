"""plans 业务循环的离线场景测试（P1-04，覆盖 P1 §9 中 plan 层可测的条目）。

驱动方式（P1 §2 末段 / §3.6 末段）：
    * ``qingyun.plans.base.vision`` 整体替换为 ``tests.support_upper.FakeVision``；
    * ``BasePlan`` 的 ``arm`` 形参接收 ``FakeArm``（按脚本返回 ``GraspResult`` 并
      记录完整事件序列）；
    * ``qingyun.plans.base._input`` 替换为可记录/可抛异常的人为输入替身；
    * 日志经 ``runlog.init`` + ``runlog.set_sink`` 注入内存接收器（P1 §3.4），
      断言事件序列、字段与 context 关联；``task_instance_id`` 由本文件的**外层**
      ``runlog.context`` 注入，模拟 P1-05 的 main（§5）。

真实 vision/asr/cloud_model 桩与运动模块在本文件全程未被调用（P1-03 的用例独立
把关那两侧）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from configs.common_interface import GraspStatus, HoldingState, MotorCommunicationError
from qingyun import runlog, shutdown
from qingyun import plans as plans_package
from qingyun.asr import ASRHardError
from qingyun.cloud_model import LLMHardError
from qingyun.grabbing import vision as vision_stub
from qingyun.grabbing.vision import NoTarget, VisionHardError
from qingyun.plans import PLAN_REGISTRY, StrawberryPlan
from qingyun.plans import base as base_module
from qingyun.plans.base import (
    MAX_CONSECUTIVE_FAILURES,
    NO_TARGET_STREAK_LIMIT,
    RETRYABLE_STATUSES,
    BasePlan,
)
from tests.support import make_target
from tests.support_upper import (
    FAKE_HOME_JOINTS_DEG,
    FAKE_SAFE_WAYPOINTS_DEG,
    FakeArm,
    FakeASR,
    FakeLLM,
    FakeVision,
    make_result,
)

# ---------------------------------------------------------------------------
# 测试资产（全部是**名义值**，只用于让分级/计数分支可判定；非标定数据）
# ---------------------------------------------------------------------------

# K 用乘式定义，使"面积等于阈值"这一用例是**逐位相等**而不是浮点近似（§3.5：等于归 A）。
K_AREA_M2: float = 0.05 * 0.024

# 分级用的四组尺寸（都满足 §6.4 的 length ≥ width > 0）。
_ABOVE_L, _ABOVE_W = 0.06, 0.05     # 0.0030  > K → A
_EQUAL_L, _EQUAL_W = 0.05, 0.024    # == K        → A
_BELOW_L, _BELOW_W = 0.05, 0.023    # 0.00115 < K → B
_UNRIPE_L, _UNRIPE_W = 0.08, 0.06   # 面积最大，但 ripe is False → C（不成熟优先）

_WAYPOINT_TUPLES = [tuple(float(v) for v in wp) for wp in FAKE_SAFE_WAYPOINTS_DEG]
_HOME_TUPLE = tuple(float(v) for v in FAKE_HOME_JOINTS_DEG)
# 一次自动/人工回程 = 逐个中转位 + home。
_RETURN_ROUTE = [*_WAYPOINT_TUPLES, _HOME_TUPLE]
_RETURN_COMMANDS = ["move_joints"] * len(_RETURN_ROUTE)


def _no_confirm(prompt: str = "") -> str:
    """默认的人工输入替身：用例走到 input 却没显式注入替身 → 立刻炸给读者看。"""
    raise AssertionError(
        "本用例未注入人工输入替身，但 plan 调用了 input()："
        "D13 的人工确认只允许出现在 GRASP_UNCERTAIN/GRIP_SLIP 恢复路径"
    )


def _no_target(reason: str = "no_detection") -> NoTarget:
    return NoTarget(reason)


def _a_above(v: int = 5) -> Any:
    return make_target(length_m=_ABOVE_L, width_m=_ABOVE_W, ripe=True, valid_count=v)


def _a_equal(v: int = 5) -> Any:
    return make_target(length_m=_EQUAL_L, width_m=_EQUAL_W, ripe=True, valid_count=v)


def _b(v: int = 5) -> Any:
    return make_target(length_m=_BELOW_L, width_m=_BELOW_W, ripe=True, valid_count=v)


def _c(v: int = 5) -> Any:
    return make_target(length_m=_UNRIPE_L, width_m=_UNRIPE_W, ripe=False, valid_count=v)


def _miss() -> Any:
    """空爪确定（可自动回程）的标准重试类结果。"""
    return make_result(GraspStatus.GRASP_MISS)


def _uncontrolled_miss() -> Any:
    return make_result(GraspStatus.GRASP_MISS, stage="FAULT_UNCONTROLLED")


def _ik_pre_motion() -> Any:
    """运动前规划失败（臂未动、recovery_required=False，§6.5 第一行）。"""
    return make_result(GraspStatus.IK_FAILED, recovery_required=False)


def _success() -> Any:
    return make_result(GraspStatus.SUCCESS)


# ---------------------------------------------------------------------------
# 内存日志接收器
# ---------------------------------------------------------------------------


class Sink:
    """``runlog.set_sink`` 的接收器：收下行 JSON，提供按 kind 取用的视图。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)

    @property
    def records(self) -> list[dict[str, Any]]:
        # 逐行 loads：任何非法 JSON 在这里炸，不会静默少一行。
        return [json.loads(line) for line in self.lines]

    @property
    def kinds(self) -> list[str]:
        return [record["kind"] for record in self.records]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["kind"] == kind]

    def field_series(self, kind: str, field_name: str) -> list[Any]:
        return [record[field_name] for record in self.of(kind)]

    def sole(self, kind: str) -> dict[str, Any]:
        found = self.of(kind)
        assert len(found) == 1, f"{kind} 事件应恰好一条，实际 {len(found)} 条：{found}"
        return found[0]


@dataclass
class Rig:
    """一次演练的四件套：plan + 视觉替身 + 运动替身 + 日志接收器。"""

    plan: BasePlan
    vision: FakeVision
    arm: FakeArm
    sink: Sink


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    """复位 runlog / shutdown 的单进程状态，并默认封死人工确认。"""
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    monkeypatch.setattr(base_module, "_input", _no_confirm)
    yield
    runlog.reset_for_tests()
    shutdown.reset_for_tests()


@pytest.fixture
def sink(tmp_path) -> Sink:
    """建正式 runlog（tmp_path 下）并把事件改道到内存接收器（P1 §3.4）。"""
    runlog.init(tmp_path / "logs")
    receiver = Sink()
    runlog.set_sink(receiver)
    return receiver


@pytest.fixture
def build_plan(monkeypatch, sink: Sink):
    """组装 plan + 两个替身：替换 ``base.vision`` 名字、注入名义 K。"""

    def make(
        script: Any = (),
        results: Any = (),
        *,
        k: float | None = K_AREA_M2,
        arm: FakeArm | None = None,
        plan_cls: type[BasePlan] = StrawberryPlan,
    ) -> Rig:
        fake_vision = FakeVision(script)
        monkeypatch.setattr(base_module, "vision", fake_vision)
        monkeypatch.setattr(plan_cls, "AREA_THRESHOLD_M2", k, raising=False)
        if arm is None:
            arm = FakeArm(results)
        return Rig(plan=plan_cls(arm), vision=fake_vision, arm=arm, sink=sink)

    return make


def expect_terminate(rig: Rig, status: str) -> dict[str, Any]:
    """跑完循环，断言"恰好一次 terminate、status 为期望值、退出码 1"，返回该事件。"""
    with pytest.raises(SystemExit) as excinfo:
        rig.plan.run()
    assert excinfo.value.code == 1, "P1 §3.3：terminate 固定 sys.exit(1)"
    events = rig.sink.of("terminate")
    assert len(events) == 1, f"应恰好一条 terminate 事件，实际 {len(events)} 条：{rig.sink.lines}"
    assert events[0]["status"] == status, (
        f"期望 status={status}，实际 {events[0]['status']}，detail={events[0]['detail']!r}"
    )
    assert events[0]["session_id"] and events[0]["ts"]
    return events[0]


def expect_finish(rig: Rig) -> dict[str, Any]:
    """断言"正常返回 + 恰好一条 task_done + 无 terminate"，返回 task_done 事件。"""
    rig.plan.run()  # 唯一正常出口：NoTarget 三连（D14）
    assert rig.sink.of("terminate") == [], f"不该终止：{rig.sink.lines}"
    return rig.sink.sole("task_done")


# ===========================================================================
# A 组：注册表与契约常量（P1 §2 / §3.5）
# ===========================================================================


def test_A01_注册表恰为strawberry且指向StrawberryPlan():
    assert PLAN_REGISTRY == {"strawberry": StrawberryPlan}
    assert plans_package.PLAN_REGISTRY is PLAN_REGISTRY, "注册表必须是模块级同一对象（§4 末段）"
    assert issubclass(StrawberryPlan, BasePlan)


def test_A02_StrawberryPlan类属性符合3_5逐字():
    assert StrawberryPlan.TASK_ID == "strawberry"
    assert StrawberryPlan.AREA_THRESHOLD_M2 is None, "P4 未回填前必须是 None（D05）"
    assert StrawberryPlan.A_PLACE_IDS == (
        "strawberry_A_01",
        "strawberry_A_02",
        "strawberry_A_03",
    )
    assert StrawberryPlan.B_PLACE_ID == "strawberry_B_bin"
    assert StrawberryPlan.C_PLACE_ID == "strawberry_C_bin"
    assert isinstance(StrawberryPlan.A_PLACE_IDS, tuple)
    assert len(set(StrawberryPlan.A_PLACE_IDS)) == len(StrawberryPlan.A_PLACE_IDS)


def test_A03_RETRYABLE集合与预算常量符合3_5():
    assert RETRYABLE_STATUSES == frozenset({
        GraspStatus.IK_FAILED,
        GraspStatus.PLAN_INVALID,
        GraspStatus.GRASP_MISS,
        GraspStatus.GRASP_UNCERTAIN,
        GraspStatus.GRIP_SLIP,
    })
    assert MAX_CONSECUTIVE_FAILURES == 10, "D02：N=10"
    assert NO_TARGET_STREAK_LIMIT == 3
    # 状态全集 = 可重试 5 + SUCCESS + 直死 8：没有任何状态被漏掉分类。
    assert set(GraspStatus) == RETRYABLE_STATUSES | {GraspStatus.SUCCESS} | {
        GraspStatus.INVALID_INPUT, GraspStatus.CONFIG_INVALID, GraspStatus.COLLISION,
        GraspStatus.PLACE_FAIL, GraspStatus.COMM_ERROR, GraspStatus.TRACKING_ERROR,
        GraspStatus.TIMEOUT, GraspStatus.ABORTED,
    }


def test_A04_新建实例计数器全部归零():
    plan = StrawberryPlan(FakeArm())
    assert (plan.no_target_streak, plan.ignore, plan.consecutive_failures, plan.a_count) == (
        0, 0, 0, 0
    ), "§6.1 第一行"
    assert (plan.success_count, plan.scan_id, plan.attempt_id) == (0, 0, 0)
    assert plan.failure_counts == {}
    # 实例之间不共享计数状态（§5：每任务新实例）。
    other = StrawberryPlan(FakeArm())
    other.a_count = 9
    assert plan.a_count == 0


def test_A05_外部依赖都是可整体替换的模块名引用():
    """§3.6 末段 / §2 末段：注入缝隙本身也是契约。"""
    assert base_module.vision is vision_stub, "默认真引用视觉桩模块"
    assert base_module.NoTarget is NoTarget and base_module.VisionHardError is VisionHardError, (
        "except 子句用的异常类必须是从桩/真实现按名导入的真实类，不随 vision 名字被替换"
    )
    assert base_module.shutdown is shutdown and base_module.runlog is runlog
    assert base_module._input is _no_confirm, "autouse 缝隙自检：_input 确实按模块名被替换"


def test_A06_失控阶段常量与运动侧同名同值():
    """不 import arm_control（避免把 placo 拉进 plans 的 import 链），故两侧文本比对钉死。"""
    from qingyun.grabbing import arm_control

    assert base_module.STAGE_FAULT_UNCONTROLLED == arm_control.STAGE_FAULT_UNCONTROLLED
    assert arm_control.STAGE_FAULT_UNCONTROLLED == "FAULT_UNCONTROLLED"


# ===========================================================================
# B 组：分级 classify（§3.5 末段；P1-T32）
# ===========================================================================


@pytest.mark.parametrize(
    "target_factory, expected",
    [
        pytest.param(_c, "C", id="不成熟→C"),
        pytest.param(_b, "B", id="成熟且面积小于K→B"),
        pytest.param(_a_equal, "A", id="成熟且面积等于K→A"),
        pytest.param(_a_above, "A", id="成熟且面积大于K→A"),
    ],
)
def test_T32_classify逐条分级(build_plan, target_factory, expected):
    rig = build_plan()
    assert rig.plan.classify(target_factory(1)) == expected


def test_T32_不成熟优先于面积阈值(build_plan):
    """ripe is False 时根本不比面积：C 级唯一来源是 ripe（§3.5）。"""
    rig = build_plan()
    assert rig.plan.classify(make_target(length_m=0.9, width_m=0.8, ripe=False, valid_count=1)) == "C"
    assert rig.plan.classify(make_target(length_m=0.9, width_m=0.8, ripe=True, valid_count=1)) == "A"


@pytest.mark.parametrize("bad_k", [None, 0.0, -1.0, float("nan"), float("inf"), True])
def test_B06_K非法时classify拒绝运行而不是默认填值(build_plan, bad_k):
    """§3.2 第 3 条的运行期兜底（§3.5："不能靠 classify 默认填值"）。"""
    rig = build_plan([_a_above(3)], [], k=bad_k)

    expect_terminate(rig, "PLAN_CONFIG_INVALID")

    assert rig.arm.events == [], "分级失败不得触发抓取"
    assert rig.plan.a_count == 0, "分级失败不得消耗 A 槽位"
    assert rig.sink.of("assign") == []


# ===========================================================================
# C 组：NoTarget 三连、完成出口与日志上下文（§3.4 / §6.2 / §6.6；P1-T03、T19）
# ===========================================================================


def test_T03_连续NoTarget三连恰好三次新扫描后完成且调用间无sleep(build_plan, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))
    rig = build_plan([_no_target() for _ in range(NO_TARGET_STREAK_LIMIT)])

    done = expect_finish(rig)

    assert rig.vision.scan_count == 3, "三连调每次都是真实新扫描（§6.2 末段）"
    assert rig.vision.ignore_values == [0, 0, 0], "NoTarget → ignore 归零（§6.1/D03）"
    assert [call["call_index"] for call in rig.vision.get_target_calls] == [1, 2, 3]
    assert sleeps == [], "NoTarget 之间不得插入 sleep"
    assert rig.arm.events == [], "纯 NoTarget 循环不得发任何运动命令"
    assert done["reason"] == "NO_TARGET_STREAK" and done["success"] == 0
    assert rig.sink.kinds == ["task_done"], "整场循环里不得出现别的事件（无分配、无抓取）"


def test_T19_唯一正常完成出口终端一行任务完成含四类统计(build_plan, capsys):
    rig = build_plan([_a_above(3), _no_target(), _no_target(), _no_target()], [_success()])

    done = expect_finish(rig)

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "任务完成" in line]
    assert len(lines) == 1, f"终端应恰好一行任务完成（初审 L9），实际 {lines}"
    assert lines[0] == (
        "任务完成：结束原因=NO_TARGET_STREAK，成功=1，失败码计数={}，已分配优品槽位=1"
    )
    assert done["reason"] == "NO_TARGET_STREAK"
    assert done["success"] == 1
    assert done["failure_counts"] == {}
    assert done["a_slots_used"] == 1


def test_T19_任务完成行同时暴露失败码计数与已消耗槽位(build_plan, capsys):
    """D04 的"空耗"必须可见：A_01 已消耗但没抓走。"""
    rig = build_plan(
        [_a_above(5), _a_above(5), _no_target(), _no_target(), _no_target()],
        [_miss(), _success()],
    )

    done = expect_finish(rig)

    line = [line for line in capsys.readouterr().out.splitlines() if "任务完成" in line][0]
    assert '失败码计数={"GRASP_MISS":1}' in line, line
    assert "已分配优品槽位=2" in line, line
    assert done["failure_counts"] == {"GRASP_MISS": 1}
    assert done["a_slots_used"] == 2


def test_C06_scan_id逐轮递增且事件带外层task_instance_id(build_plan):
    """§3.4：日志关联键 (session_id, task_instance_id, scan_id, attempt_id)。"""
    rig = build_plan(
        [_a_above(5), _b(5), _no_target(), _no_target(), _no_target()],
        [_ik_pre_motion(), _success()],
    )

    with runlog.context(task_id="strawberry", task_instance_id=7):
        expect_finish(rig)
        # 循环自己每一轮的 with 都已退出：外层只剩 main 注入的两个键。
        assert runlog.snapshot_context() == {
            "task_id": "strawberry",
            "task_instance_id": 7,
        }, "退出每轮 with 后 scan_id 必须已释放"

    for record in rig.sink.records:
        assert record["task_instance_id"] == 7, record
        assert record["task_id"] == "strawberry", record
    assert rig.sink.field_series("assign", "scan_id") == [1, 2]
    assert rig.sink.field_series("grasp_result", "scan_id") == [1, 2]
    assert rig.sink.field_series("recovery", "scan_id") == [1]
    assert rig.sink.sole("task_done")["scan_id"] == 5, "task_done 属于第三连 NoTarget 那一轮"
    assert runlog.snapshot_context() == {}, "外层 with 退出后连 task_instance_id 也要释放"


def test_C07_terminate异常退出也释放scan_id上下文(build_plan):
    rig = build_plan([_b(5)], [_uncontrolled_miss()])

    with runlog.context(task_instance_id=3):
        terminate = expect_terminate(rig, "FAULT_UNCONTROLLED")
        assert runlog.snapshot_context() == {"task_instance_id": 3}, (
            "SystemExit 穿过每轮 with 时也要释放 scan_id"
        )

    assert terminate["scan_id"] == 1, "terminate 事件必须带当轮 scan_id（§7 所处阶段）"
    assert terminate["task_instance_id"] == 3
    assert terminate["stage"]["scan_id"] == 1


def test_C08_attempt_id每任务从1起且三事件同号(build_plan):
    rig = build_plan(
        [_b(5), _b(5), _b(5), _no_target(), _no_target(), _no_target()],
        [_ik_pre_motion(), _ik_pre_motion(), _success()],
    )

    expect_finish(rig)

    assert rig.sink.field_series("assign", "attempt_id") == [1, 2, 3]
    assert rig.sink.field_series("grasp_result", "attempt_id") == [1, 2, 3]
    assert rig.sink.field_series("recovery", "attempt_id") == [1, 2]
    assert rig.sink.field_series("assign", "scan_id") == [1, 2, 3]
    assert rig.plan.attempt_id == 3, "attempt_id 只在真的发起抓取时递增"
    assert rig.plan.scan_id == 6


# ===========================================================================
# D 组：计数器矩阵与分发顺序（§6.1 / §6.2；P1-T04…T10、T26）
# ===========================================================================


def test_T04_中间成功目标打断NoTarget连续计数(build_plan):
    """若成功返回不把 no_target_streak 归零，第 4 次调用就会提前"正常完成"。"""
    rig = build_plan(
        [_no_target(), _b(5), _no_target(), _no_target(), _no_target()],
        [_ik_pre_motion()],
    )

    done = expect_finish(rig)

    assert rig.vision.scan_count == 5, "第 4 次不得收尾"
    assert rig.vision.ignore_values == [0, 0, 1, 0, 0], (
        "第 3 次调用仍带 ignore=1（视觉成功不清跳选游标），"
        "第 4/5 次已归零（NoTarget 清游标）——§6.1 两行同时成立"
    )
    assert done["reason"] == "NO_TARGET_STREAK"
    assert rig.plan.no_target_streak == 3
    assert rig.plan.consecutive_failures == 1, "D02：NoTarget 不清零连续失败预算"


def test_T07_若干失败后SUCCESS归零ignore与失败计数并可再次入选(build_plan):
    rig = build_plan(
        [_b(5), _b(5), _no_target(), _no_target(), _no_target()],
        [_miss(), _success()],
    )

    expect_finish(rig)

    assert rig.vision.ignore_values == [0, 1, 0, 0, 0], "SUCCESS → ignore=0（D03：无永久排除名单）"
    assert rig.plan.ignore == 0 and rig.plan.consecutive_failures == 0
    assert rig.plan.success_count == 1
    assert rig.plan.failure_counts == {"GRASP_MISS": 1}, "历史失败仍留档（§6.6 不掩盖）"


def test_T08_失败后NoTarget清零ignore但不清零连续失败预算(build_plan):
    """§6.1 表：NoTarget 只动 ignore / no_target_streak，不碰 consecutive_failures（D02/D03）。"""
    script = [_b(99) for _ in range(5)] + [_no_target("all_filtered")] + [_b(99) for _ in range(5)]
    rig = build_plan(script, [_miss()] * 10)

    expect_terminate(rig, "FAILURE_BUDGET_EXHAUSTED")

    assert rig.vision.scan_count == 11
    assert rig.vision.ignore_values == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4], (
        "第 6 次调用带 ignore=5；NoTarget 后跳选从 0 重新开始（D03）"
    )
    assert rig.plan.consecutive_failures == MAX_CONSECUTIVE_FAILURES
    assert rig.arm.count_of("reset_fault") == 9


def test_T05_V12连续十次可恢复失败第十次判预算终止且无第十次恢复(build_plan):
    rig = build_plan([_b(12) for _ in range(10)], [_miss()] * 10)

    terminate = expect_terminate(rig, "FAILURE_BUDGET_EXHAUSTED")

    assert rig.vision.ignore_values == list(range(10))
    assert rig.plan.ignore == 10 < 12, "第 10 次未触发 D14 封顶，才轮到预算"
    assert rig.arm.count_of("grasp_and_place") == 10
    assert rig.arm.count_of("reset_fault") == 9, "无第 10 次恢复"
    assert rig.arm.count_of("move_joints") == 9 * len(_RETURN_ROUTE)
    assert rig.arm.events[-1].name == "grasp_and_place", "第 10 次抓取之后不得再有任何命令"
    assert len(rig.sink.of("recovery")) == 9
    assert rig.sink.of("task_done") == []
    assert f"consecutive_failures={MAX_CONSECUTIVE_FAILURES}" in terminate["detail"]


def test_T06_V10第十次失败同轮先判封顶D14(build_plan):
    """同一轮里 ignore 与预算同时到 10：必须先判封顶（§6.2 第 5 步排在第 6 步之前）。"""
    rig = build_plan([_b(10) for _ in range(10)], [_miss()] * 10)

    terminate = expect_terminate(rig, "NO_GRASPABLE_TARGET")

    assert terminate["detail"] == "ignore=10 ≥ valid_count=10，末次 GRASP_MISS/FAULT_HOLD"
    assert rig.plan.ignore == 10 and rig.plan.consecutive_failures == 10
    assert rig.arm.count_of("grasp_and_place") == 10
    assert rig.arm.motion_command_count() == 9 * (1 + len(_RETURN_ROUTE)), "第 10 次不恢复"
    assert rig.arm.events[-1].name == "grasp_and_place"
    assert len(rig.sink.of("recovery")) == 9
    assert rig.sink.of("task_done") == []


def test_T26_V1单次重试类失败即封顶且不恢复不确认不回home不task_done(build_plan):
    rig = build_plan([_b(1)], [_miss()])

    terminate = expect_terminate(rig, "NO_GRASPABLE_TARGET")

    assert terminate["detail"] == "ignore=1 ≥ valid_count=1，末次 GRASP_MISS/FAULT_HOLD"
    assert rig.arm.events[-1].name == "grasp_and_place"
    assert rig.arm.motion_command_count() == 0, "零新运动命令：不 reset、不回程、不回 home"
    assert rig.sink.kinds == ["assign", "grasp_result", "terminate"]
    # autouse 已把 _input 换成"调用即炸"，所以"无 input()"这一条是硬断言。
    assert rig.vision.scan_count == 1


def test_T26_同场景FAULT_UNCONTROLLED仍优先于封顶(build_plan):
    rig = build_plan([_b(1)], [_uncontrolled_miss()])

    expect_terminate(rig, "FAULT_UNCONTROLLED")

    assert rig.plan.ignore == 0, "第 4 步未执行：封顶与计数都排在失控判定之后"
    assert rig.arm.motion_command_count() == 0


def test_T09_A第一次抓空下一个A仍用新槽位不回收A_01(build_plan):
    """D04：分配即消耗；A_01 被 GRASP_MISS 空耗后不退还、不降级。"""
    rig = build_plan([_a_above(5), _a_above(5), _no_target(), _no_target(), _no_target()],
                     [_miss(), _success()])

    expect_finish(rig)

    assert rig.arm.place_ids() == ["strawberry_A_01", "strawberry_A_02"]
    assert rig.plan.a_count == 2
    assert rig.sink.field_series("assign", "a_count") == [1, 2], "事件里的 a_count 是分配后的值"
    assert rig.sink.field_series("assign", "place_id") == [
        "strawberry_A_01", "strawberry_A_02"
    ]
    assert rig.sink.field_series("assign", "grade") == ["A", "A"]


def test_T10_A名额耗尽仍有目标时下一次运动调用前终止(build_plan):
    rig = build_plan([_a_above(5) for _ in range(4)], [_success()] * 3)

    terminate = expect_terminate(rig, "A_SLOTS_EXHAUSTED")

    assert rig.arm.place_ids() == ["strawberry_A_01", "strawberry_A_02", "strawberry_A_03"]
    assert rig.arm.count_of("grasp_and_place") == 3, "第 4 次分配失败前不得调用运动"
    assert len(rig.sink.of("assign")) == 3, "第 4 轮没有 assign"
    assert "优品槽位耗尽" in terminate["detail"]
    assert rig.sink.of("task_done") == [], "D14 后这不是正常完成"
    assert rig.plan.a_count == 3, "不回滚也不循环覆盖 A_01"


@pytest.mark.parametrize(
    "status",
    [
        GraspStatus.INVALID_INPUT, GraspStatus.CONFIG_INVALID, GraspStatus.COLLISION,
        GraspStatus.PLACE_FAIL, GraspStatus.COMM_ERROR, GraspStatus.TRACKING_ERROR,
        GraspStatus.TIMEOUT, GraspStatus.ABORTED,
    ],
    ids=lambda value: value.value,
)
def test_DP01_非重试类以状态名直死且不跳选不恢复(build_plan, status):
    """§6.2 第 3 步；D02 明确"配置/通信/碰撞等原直死错误不因预算改为忽略"。"""
    rig = build_plan([_b(3)], [make_result(status)])

    terminate = expect_terminate(rig, status.value)

    assert rig.plan.ignore == 0 and rig.plan.consecutive_failures == 0
    assert rig.vision.scan_count == 1
    assert rig.arm.motion_command_count() == 0
    assert rig.sink.of("recovery") == []
    assert terminate["detail"], "直死必须带上运动侧 reason"


def test_DP02_grasp_result审计先于一切终止分支(build_plan):
    """§6.2：任何直死/封顶分支之前都必须完成该次结果审计。"""
    rig = build_plan([_b(3)], [make_result(GraspStatus.COLLISION, reason="自碰撞")])

    expect_terminate(rig, "COLLISION")

    assert rig.sink.kinds == ["assign", "grasp_result", "terminate"]
    result = rig.sink.sole("grasp_result")
    assert result["status"] == "COLLISION" and result["reason"] == "自碰撞"
    assert result["grade"] == "B" and result["place_id"] == "strawberry_B_bin"
    assert result["recovery_required"] is True and result["holding"] == "UNKNOWN"
    assert result["ignore_used"] == 0 and result["valid_count"] == 3


def test_DP03_assign事件带齐规格最低字段(build_plan):
    rig = build_plan([_a_above(7), _no_target(), _no_target(), _no_target()], [_success()])

    expect_finish(rig)

    assign = rig.sink.sole("assign")
    for key in ("grade", "place_id", "a_count", "valid_count", "ignore_used", "attempt_id"):
        assert key in assign, f"§7 assign 最低字段缺 {key}：{assign}"
    assert assign["grade"] == "A" and assign["valid_count"] == 7 and assign["ignore_used"] == 0
    assert assign["place_id"] == "strawberry_A_01" and assign["a_count"] == 1



# ===========================================================================
# E 组：视觉返回值契约复核与硬错误（§6.2 / §6.4；P1-T23、T24）
# ===========================================================================


def _broken_target(**overrides) -> Any:
    return replace(_b(3), **overrides)


@pytest.mark.parametrize(
    "factory, hint",
    [
        pytest.param(lambda: _broken_target(length_m=-0.05), "length_m 必须为正", id="length为负"),
        pytest.param(lambda: _broken_target(width_m=0.0), "width_m 必须为正", id="width为零"),
        pytest.param(lambda: _broken_target(width_m=0.06), "length_m 不得小于 width_m", id="长小于宽"),
        pytest.param(lambda: _broken_target(length_m=float("nan")), "非有限", id="length为NaN"),
        pytest.param(lambda: _broken_target(width_m=float("inf")), "非有限", id="width为Inf"),
        pytest.param(lambda: _broken_target(length_m="0.05"), "length_m", id="length是字符串"),
        pytest.param(lambda: _broken_target(width_m=None), "width_m", id="width是None"),
        pytest.param(lambda: _broken_target(ripe=1), "严格 bool", id="ripe是整数1"),
        pytest.param(lambda: _broken_target(ripe=None), "严格 bool", id="ripe是None"),
        pytest.param(lambda: _broken_target(ripe="true"), "严格 bool", id="ripe是字符串"),
        pytest.param(lambda: _broken_target(valid_count=0), "正整数", id="valid_count为零"),
        pytest.param(lambda: _broken_target(valid_count=-3), "正整数", id="valid_count为负"),
        pytest.param(lambda: _broken_target(valid_count=True), "整数", id="valid_count是bool"),
        pytest.param(lambda: _broken_target(valid_count=2.0), "整数", id="valid_count是浮点"),
        pytest.param(lambda: _broken_target(valid_count="3"), "整数", id="valid_count是字符串"),
        pytest.param(lambda: _broken_target(yaw_deg=90.0), "[-90,90)", id="yaw等于90"),
        pytest.param(lambda: _broken_target(yaw_deg=-90.5), "[-90,90)", id="yaw小于负90"),
        pytest.param(lambda: _broken_target(yaw_deg=179.0), "[-90,90)", id="yaw未归一化"),
        pytest.param(lambda: _broken_target(yaw_deg=float("nan")), "非有限", id="yaw是NaN"),
        pytest.param(lambda: _broken_target(yaw_deg="0"), "yaw_deg", id="yaw是字符串"),
        pytest.param(
            lambda: SimpleNamespace(position=None, yaw_deg=0.0, length_m=_BELOW_L,
                                    width_m=_BELOW_W, ripe=True),
            "缺少", id="缺valid_count字段",
        ),
        pytest.param(
            lambda: SimpleNamespace(yaw_deg=0.0, length_m=_BELOW_L, width_m=_BELOW_W,
                                    ripe=True, valid_count=3),
            "缺少", id="缺position字段",
        ),
    ],
)
def test_T24_视觉契约违约逐项终止且零抓取调用(build_plan, factory, hint):
    rig = build_plan([factory()])

    terminate = expect_terminate(rig, "VISION_CONTRACT_VIOLATION")

    assert hint in terminate["detail"], terminate["detail"]
    assert rig.arm.events == [], "违约不得触发抓取"
    assert rig.plan.a_count == 0, "违约不得消耗 A 槽位"
    assert rig.sink.of("assign") == [] and rig.sink.of("grasp_result") == []
    assert rig.plan.no_target_streak == 0, "契约违约不计 NoTarget（§6.4）"
    assert rig.plan.ignore == 0, "契约违约不得追加跳选"


def test_T24_valid_count未大于当次请求的ignore时按契约违约终止(build_plan):
    """第一轮合法（V=3 > ignore=0）；第二轮 ignore=1 却返回 V=1 → 违约（§6.4 第 3 条）。"""
    rig = build_plan([_b(3), _broken_target(valid_count=1)], [_ik_pre_motion()])

    terminate = expect_terminate(rig, "VISION_CONTRACT_VIOLATION")

    assert "valid_count=1" in terminate["detail"] and "ignore=1" in terminate["detail"]
    assert rig.vision.ignore_values == [0, 1]
    assert rig.arm.count_of("grasp_and_place") == 1, "第二轮不得触发抓取"


def test_E05_valid_count恰好大于ignore时合法放行(build_plan):
    """反向对照：V = ignore + 1 是合法的，不能因判据实现过头而误死。"""
    rig = build_plan([_b(3), _b(2), _no_target(), _no_target(), _no_target()],
                     [_ik_pre_motion(), _success()])

    expect_finish(rig)

    assert rig.vision.ignore_values == [0, 1, 0, 0, 0]
    assert rig.arm.count_of("grasp_and_place") == 2


def test_T23_get_target抛VisionHardError时终止且不混入NoTarget计数(build_plan):
    rig = build_plan([_no_target(), _no_target(), VisionHardError("相机断开")])

    terminate = expect_terminate(rig, "VISION_HARD_ERROR")

    assert terminate["detail"].startswith("视觉管线故障")
    assert "相机断开" in terminate["detail"]
    assert rig.vision.scan_count == 3
    assert rig.plan.no_target_streak == 2, "硬错误不参与三连计数"
    assert rig.sink.of("task_done") == [], "绝不能把故障洗白成『场地是空的』式正常完成"
    assert rig.sink.kinds == ["terminate"]


def test_E06_硬错误在首轮立即终止(build_plan):
    rig = build_plan([VisionHardError("深度帧全无效")])

    expect_terminate(rig, "VISION_HARD_ERROR")

    assert rig.vision.scan_count == 1
    assert rig.plan.no_target_streak == 0
    assert rig.arm.events == []


# ===========================================================================
# F 组：恢复矩阵（§6.5；P1-T11、T12、T13、T14、T15、T16、T17）
# ===========================================================================


def test_T11_失控阶段最先判定且零追加串口调用(build_plan):
    """§6.2 第 1 步排在 SUCCESS 与重试类之前。"""
    rig = build_plan([_b(3)], [_uncontrolled_miss()])

    terminate = expect_terminate(rig, "FAULT_UNCONTROLLED")

    assert "GRASP_MISS" in terminate["detail"] and "FAULT_UNCONTROLLED" in terminate["detail"]
    assert rig.arm.motion_command_count() == 0, "零追加串口调用（T11）"
    assert rig.vision.scan_count == 1
    assert rig.sink.of("recovery") == []
    assert rig.plan.ignore == 0 and rig.plan.consecutive_failures == 0


def test_T11_失控阶段优先于SUCCESS判定(build_plan):
    """注入"SUCCESS 但 stage=FAULT_UNCONTROLLED"的组合：第 1 步必须先命中。"""
    rig = build_plan([_b(3)], [make_result(GraspStatus.SUCCESS, stage="FAULT_UNCONTROLLED")])

    expect_terminate(rig, "FAULT_UNCONTROLLED")

    assert rig.plan.success_count == 0, "失控不得被记成成功"
    assert rig.arm.motion_command_count() == 0
    assert rig.sink.kinds == ["assign", "grasp_result", "terminate"], "审计仍先于终止"


def test_T13_IK_FAILED带recovery_required时D08一律终止(build_plan):
    rig = build_plan([_b(3)], [make_result(GraspStatus.IK_FAILED, recovery_required=True,
                                           stage="FAULT_HOLD", reason="IK 中途失败")])

    terminate = expect_terminate(rig, "IK_FAILED")

    assert "D08" in terminate["detail"]
    assert "IK 中途失败" in terminate["detail"], "臂冻结于失败位时 detail 必须带上原始 reason"
    assert rig.arm.motion_command_count() == 0, "无 reset、无搬臂"
    assert rig.vision.scan_count == 1, "不进下一轮视觉"
    assert rig.sink.of("recovery") == [], "D08 不是一条『恢复』，不该记 recovery 事件"


def test_T13_PLAN_INVALID带recovery_required时同样终止且不区分holding(build_plan):
    for holding in (HoldingState.EMPTY, HoldingState.HOLDING, HoldingState.UNKNOWN):
        rig = build_plan([_b(3)], [make_result(GraspStatus.PLAN_INVALID,
                                               recovery_required=True, holding=holding,
                                               stage="FAULT_HOLD")])
        expect_terminate(rig, "PLAN_INVALID")
        assert rig.arm.motion_command_count() == 0
        shutdown.reset_for_tests()
        rig.sink.lines.clear()


def test_T14_IK_FAILED不带recovery_required时直接下一轮视觉(build_plan):
    """§6.5 第一行：运动前失败（臂在 home、空爪）→ 不 reset、不搬臂。"""
    rig = build_plan([_b(5), _b(5), _no_target(), _no_target(), _no_target()],
                     [_ik_pre_motion(), _success()])

    expect_finish(rig)

    assert rig.arm.count_of("reset_fault") == 0
    assert rig.arm.count_of("move_joints") == 0
    assert rig.arm.motion_command_count() == 0
    assert rig.vision.ignore_values == [0, 1, 0, 0, 0], "ignore 已 +1 后直接下次视觉"
    recovery = rig.sink.sole("recovery")
    assert recovery["mode"] == "skip" and recovery["status"] == "IK_FAILED"
    assert recovery["reset"] == "not_needed" and recovery["waypoint_moves"] == 0


def test_T14_PLAN_INVALID同样走skip路径(build_plan):
    rig = build_plan([_b(5), _no_target(), _no_target(), _no_target()],
                     [make_result(GraspStatus.PLAN_INVALID, recovery_required=False)])

    expect_finish(rig)

    assert rig.arm.motion_command_count() == 0
    assert rig.sink.sole("recovery")["mode"] == "skip"


@pytest.mark.parametrize("status", [GraspStatus.GRASP_UNCERTAIN, GraspStatus.GRIP_SLIP],
                         ids=lambda value: value.value)
def test_T15_人工确认回车后reset中转home并继续跳选(build_plan, monkeypatch, capsys, status):
    prompts: list[str] = []
    monkeypatch.setattr(base_module, "_input", lambda prompt="": prompts.append(prompt) or "")
    rig = build_plan(
        [_b(5), _b(5), _no_target(), _no_target(), _no_target()],
        [make_result(status), _success()],
    )

    expect_finish(rig)

    assert len(prompts) == 1, "只在 UNCERTAIN/SLIP 恢复处确认一次"
    assert "回车" in prompts[0], "D13：提示语必须说明回车即确认"
    out = capsys.readouterr().out
    assert "status=" + status.value in out, out
    assert "stage=FAULT_HOLD" in out and "holding=" in out and "reason=" in out, (
        "确认前必须打印现场状态四要素"
    )
    assert rig.arm.command_sequence() == [
        "grasp_and_place", "reset_fault", *_RETURN_COMMANDS, "grasp_and_place"
    ]
    assert rig.arm.moved_positions() == _RETURN_ROUTE, "回程路线 = waypoints 列表顺序 + home"
    assert rig.vision.ignore_values == [0, 1, 0, 0, 0], "确认后继续跳选（ignore 保持 +1）"
    recovery = rig.sink.sole("recovery")
    assert recovery["mode"] == "manual" and recovery["confirm"] == "manual_confirmed"
    assert recovery["waypoint_moves"] == len(_WAYPOINT_TUPLES)
    assert recovery["status"] == status.value


def test_T15_空回车同样算确认并继续循环(build_plan, monkeypatch):
    """D13：不要求特定文本——返回空串也算确认。"""
    monkeypatch.setattr(base_module, "_input", lambda prompt="": "")
    rig = build_plan([_b(5), _b(5), _no_target(), _no_target(), _no_target()],
                     [make_result(GraspStatus.GRASP_UNCERTAIN), _success()])

    expect_finish(rig)

    assert rig.plan.success_count == 1
    assert rig.arm.count_of("reset_fault") == 1


def test_T15_自动回程与人工回程的路线完全相同(build_plan):
    """GRASP_MISS(EMPTY) 走 auto，与 manual 只差"确认"一步（§6.5 第三/四行）。"""
    rig = build_plan([_b(5), _no_target(), _no_target(), _no_target()], [_miss()])

    expect_finish(rig)

    assert rig.arm.command_sequence() == ["grasp_and_place", "reset_fault", *_RETURN_COMMANDS]
    assert rig.arm.moved_positions() == _RETURN_ROUTE
    recovery = rig.sink.sole("recovery")
    assert recovery["mode"] == "auto" and recovery["confirm"] == "not_required"


@pytest.mark.parametrize("exc_factory", [EOFError, KeyboardInterrupt],
                         ids=["EOF即Ctrl-C之外", "Ctrl-C"])
def test_T16_人工确认被中断时终止且不复位不搬臂(build_plan, monkeypatch, exc_factory):
    def raiser(prompt: str = "") -> str:
        raise exc_factory()

    monkeypatch.setattr(base_module, "_input", raiser)
    rig = build_plan([_b(5)], [make_result(GraspStatus.GRIP_SLIP)])

    terminate = expect_terminate(rig, "MANUAL_CONFIRM_ABORTED")

    assert "GRIP_SLIP" in terminate["detail"]
    assert rig.arm.motion_command_count() == 0, "确认未完成就不 reset、不回程"
    assert rig.vision.scan_count == 1
    assert rig.sink.of("recovery") == []
    assert rig.sink.of("task_done") == []


def test_T16_EOF与Ctrl_C两条分支走同一出口(build_plan, monkeypatch, sink: Sink):
    """两条分支各自独立跑一遍（terminate 幂等标志复位后），断言处置逐字相同。"""
    seen: list[str] = []
    for exc in (EOFError(), KeyboardInterrupt()):
        monkeypatch.setattr(
            base_module, "_input", lambda prompt="", e=exc: (_ for _ in ()).throw(e)
        )
        rig = build_plan([_b(5)], [make_result(GraspStatus.GRASP_UNCERTAIN)])
        sink.lines.clear()
        expect_terminate(rig, "MANUAL_CONFIRM_ABORTED")
        seen.append(sink.sole("terminate")["detail"])
        assert rig.plan.ignore == 1, "只保留 §6.2 第 4 步的一次 +1，不为恢复再追加"
        shutdown.reset_for_tests()  # 让下一条分支也能真正走一次 terminate
    assert seen[0] == seen[1], "EOF 与 Ctrl-C 的处置必须逐字相同"


@pytest.mark.parametrize("holding", [HoldingState.HOLDING, HoldingState.UNKNOWN],
                         ids=lambda value: value.value)
def test_T17_GRASP_MISS但holding非EMPTY是组合违约(build_plan, holding):
    rig = build_plan([_b(5)], [make_result(GraspStatus.GRASP_MISS, holding=holding)])

    terminate = expect_terminate(rig, "RESULT_CONTRACT_VIOLATION")

    assert "GRASP_MISS" in terminate["detail"] and holding.value in terminate["detail"]
    assert rig.arm.motion_command_count() == 0, "组合违约既不自动回程也不人工确认"
    assert rig.sink.of("recovery") == []
    assert rig.vision.scan_count == 1


@pytest.mark.parametrize("status", [GraspStatus.GRASP_MISS, GraspStatus.GRASP_UNCERTAIN,
                                    GraspStatus.GRIP_SLIP], ids=lambda value: value.value)
def test_T17_抓取后重试类却recovery_requiredFalse是组合违约(build_plan, status):
    rig = build_plan([_b(5)], [make_result(status, recovery_required=False)])

    terminate = expect_terminate(rig, "RESULT_CONTRACT_VIOLATION")

    assert "recovery_required=False" in terminate["detail"]
    assert rig.arm.motion_command_count() == 0


def test_T17_组合违约事件保留原result全部字段(build_plan):
    rig = build_plan([_b(5)], [make_result(GraspStatus.GRASP_MISS, holding=HoldingState.HOLDING,
                                           reason="闭爪判据互相矛盾",
                                           place_id="strawberry_B_bin")])

    terminate = expect_terminate(rig, "RESULT_CONTRACT_VIOLATION")

    detail = terminate["detail"]
    for fragment in ("status=GRASP_MISS", "stage=FAULT_HOLD", "holding=HOLDING",
                     "recovery_required=True", "strawberry_B_bin", "闭爪判据互相矛盾"):
        assert fragment in detail, f"{fragment!r} 必须留在 terminate detail 里供定位"


@pytest.mark.parametrize("at_call, step_hint", [
    (1, "safe_waypoints_deg[0]"),
    (2, "safe_waypoints_deg[1]"),
    (3, "home_joints_deg"),
])
def test_T12_回程某段move未到位立即终止RECOVERY_FAILED(build_plan, at_call, step_hint):
    """§6.5 末行：任一回程原语异常（含 move 未到位）→ 就地统一退出。"""
    arm = FakeArm([_miss()])
    arm.fail("move_joints", MotorCommunicationError("总线在回程中掉线"), at_call=at_call)
    rig = build_plan([_b(5), _b(5)], [], arm=arm)

    terminate = expect_terminate(rig, "RECOVERY_FAILED")

    assert step_hint in terminate["detail"]
    assert "总线在回程中掉线" in terminate["detail"]
    assert rig.vision.scan_count == 1, "不进视觉"
    assert rig.sink.of("task_done") == [], "不打印任务完成"
    assert rig.sink.field_series("assign", "place_id") == ["strawberry_B_bin"]
    assert arm.count_of("move_joints") == at_call, "失败之后不再追加回程命令"


def test_T12_不打印任务完成统一退出并留recovery失败步骤(build_plan, capsys):
    arm = FakeArm([_miss()])
    arm.fail("move_joints", MotorCommunicationError("未到位"))
    rig = build_plan([_b(5), _b(5)], [], arm=arm)

    expect_terminate(rig, "RECOVERY_FAILED")

    assert rig.sink.kinds == ["assign", "grasp_result", "recovery", "terminate"]
    assert capsys.readouterr().out.count("任务完成") == 0, "T12：不得打印任务完成"


def test_T12_reset_fault抛异常时同样RECOVERY_FAILED且不搬臂(build_plan, capsys):
    arm = FakeArm([_miss()])
    arm.fail("reset_fault", MotorCommunicationError("复位被拒绝：当前位置与保持目标偏差超限"))
    rig = build_plan([_b(5), _b(5)], [], arm=arm)

    terminate = expect_terminate(rig, "RECOVERY_FAILED")

    assert "reset_fault" in terminate["detail"]
    assert rig.plan.ignore == 1, "只保留 §6.2 第 4 步的一次 +1，不追加重试恢复"
    assert rig.arm.count_of("move_joints") == 0, "复位都没成功，绝不能继续搬臂"
    assert "任务完成" not in capsys.readouterr().out
    recovery = rig.sink.sole("recovery")
    assert recovery["mode"] == "auto" and recovery["failed_step"] == "reset_fault"


def test_F13_人工确认之后的回程失败同样按RECOVERY_FAILED处理(build_plan, monkeypatch):
    monkeypatch.setattr(base_module, "_input", lambda prompt="": "")
    arm = FakeArm([make_result(GraspStatus.GRASP_UNCERTAIN)])
    arm.fail("move_joints", MotorCommunicationError("中转位写入失败"))
    rig = build_plan([_b(5)], [], arm=arm)

    expect_terminate(rig, "RECOVERY_FAILED")

    assert rig.arm.count_of("reset_fault") == 1
    assert rig.sink.sole("recovery")["mode"] == "manual"


def test_F14_预算耗尽那一轮不进入恢复矩阵(build_plan):
    """§6.2 第 6 步排在第 7 步之前：预算到点就冻结现场，不再搬臂（D02/D14 同一取向）。
    """
    arm = FakeArm([_miss()] * 10)
    arm.fail("move_joints", MotorCommunicationError("回程掉线"), at_call=28)
    rig = build_plan([_b(99) for _ in range(10)], [], arm=arm)

    expect_terminate(rig, "FAILURE_BUDGET_EXHAUSTED")

    assert arm.count_of("move_joints") == 9 * len(_RETURN_ROUTE) == 27
    assert arm.count_of("reset_fault") == 9



# ===========================================================================
# G 组：替身自证（防止"替身太宽容"造成的假通过）
# ===========================================================================


def test_G01_FakeVision脚本耗尽默认抛真实VisionHardError():
    fake = FakeVision([])
    with pytest.raises(VisionHardError):
        fake.get_target(0)
    assert isinstance(fake.get_target_calls[0]["ignore"], int)

    fake2 = FakeVision([], on_exhausted="no_target")
    with pytest.raises(NoTarget):
        fake2.get_target(3)
    assert fake2.ignore_values == [3], "记录参数不影响抛错路径"
    assert fake2.remaining == 0


def test_G02_FakeVision按脚本顺序行动并记录ignore序列():
    target = _b(4)
    fake = FakeVision([target, _no_target()])
    assert fake.get_target(0) is target
    with pytest.raises(NoTarget):
        fake.get_target(1)
    assert fake.ignore_values == [0, 1]
    fake.configure(thresholds=SimpleNamespace(), config_path="vision.json")
    fake.init()
    assert fake.configure_calls[0]["config_path"] == "vision.json"
    assert len(fake.init_calls) == 1


def test_G03_FakeArm事件台账记录顺序与数量并可注入异常():
    first, second = _b(1), _b(2)
    arm = FakeArm([_miss(), _success()])
    assert arm.grasp_and_place(first, "strawberry_B_bin").status is GraspStatus.GRASP_MISS
    assert arm.grasp_and_place(second, "strawberry_A_01").status is GraspStatus.SUCCESS
    arm.reset_fault()
    arm.move_joints([1.0, 2.0, 3.0, 4.0, 5.0])
    assert arm.command_sequence() == [
        "grasp_and_place", "grasp_and_place", "reset_fault", "move_joints"
    ]
    assert arm.place_ids() == ["strawberry_B_bin", "strawberry_A_01"]
    assert arm.grasp_targets() == [first, second], "目标对象本体入台账，可按 id 断言"
    assert arm.moved_positions() == [(1.0, 2.0, 3.0, 4.0, 5.0)]
    assert arm.count_of("move_joints") == 1 and arm.motion_command_count() == 2
    with pytest.raises(AssertionError):
        arm.grasp_and_place(_b(1), "strawberry_B_bin")


    arm2 = FakeArm([_miss()])
    arm2.fail("open_gripper", RuntimeError("夹具卡住"))
    with pytest.raises(RuntimeError):
        arm2.open_gripper()
    assert arm2.count_of("open_gripper") == 1, "异常调用也要计入台账（顺序审计）"


def test_G04_FakeArm脚本耗尽抛AssertionError绝不自造成功():
    arm = FakeArm([])
    with pytest.raises(AssertionError):
        arm.grasp_and_place(_b(1), "strawberry_B_bin")
    assert arm.params.workspace.home_joints_deg is not None
    assert arm.motor is None, "motor 由会话层注入（plan 不读）"


def test_G05_FakeASR与FakeLLM按脚本行动并记录调用():
    asr = FakeASR(["帮我分拣草莓", ""])
    asr.init(cfg=SimpleNamespace())
    assert asr.listen_and_transcribe() == "帮我分拣草莓"
    assert asr.listen_and_transcribe() == ""
    with pytest.raises(ASRHardError):
        asr.listen_and_transcribe()
    assert len(asr.init_calls) == 1 and len(asr.listen_calls) == 3
    assert asr.results == ["帮我分拣草莓", ""]

    llm = FakeLLM(["strawberry", "invalid"])
    assert llm.select_plan("帮我分拣草莓") == "strawberry"
    assert llm.select_plan("今天天气如何") == "invalid"
    assert llm.texts == ["帮我分拣草莓", "今天天气如何"]
    with pytest.raises(LLMHardError):
        llm.select_plan("再来一次")
    assert FakeLLM([], on_exhausted="invalid").select_plan("分拣蓝莓") == "invalid"
    repeat = FakeLLM(["strawberry"], on_exhausted="repeat_last")
    assert repeat.select_plan("a") == "strawberry"
    assert repeat.select_plan("b") == "strawberry", "repeat_last 只在显式要求时才复用"


def test_G06_make_result的典型组合与运动侧一致():
    """典型组合抄自 arm_control 的 _refuse/_fault_result/_result，供读代码时对照。"""
    miss = _miss()
    assert (miss.status, miss.holding, miss.recovery_required, miss.stage) == (
        GraspStatus.GRASP_MISS, HoldingState.EMPTY, True, "FAULT_HOLD"
    )
    pre = _ik_pre_motion()
    assert (pre.stage, pre.holding, pre.recovery_required) == ("CHECK", HoldingState.UNKNOWN, False)
    ok = _success()
    assert (ok.stage, ok.holding, ok.recovery_required) == ("DONE", HoldingState.EMPTY, False)


def test_G07_分级阈值K与目标尺寸自洽():
    """用例资产自证：K 与四组尺寸的相对关系必须如注释所述。"""
    assert _a_above(1).length_m * _a_above(1).width_m > K_AREA_M2
    assert _a_equal(1).length_m * _a_equal(1).width_m == K_AREA_M2
    assert _b(1).length_m * _b(1).width_m < K_AREA_M2
    assert _c(1).ripe is False and _c(1).length_m * _c(1).width_m > K_AREA_M2
