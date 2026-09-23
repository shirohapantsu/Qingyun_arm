"""P4 §2.8.2 内存有界运动事件 trace 的测试（P4-T10 + 运动不变性断言）。

规格逐句对应（docs/实施文档/P4_标定工具与参数发布技术文档.md §2.8.2、§4 P4-T10）：

* ``MotionTraceEvent`` 定义在 arm_control.py，字段 ``sequence/time_ns/stage/kind/outcome``；
  时间取控制器注入的 ``clock_ns``，不引入第二个时间源。
* 公开 ``ArmController.get_last_motion_trace()``：每次 grasp_and_place 开始清空、返回后
  保留到下一次调用；返回不可变快照；事件仅内存追加且有界（每阶段首次进入/首次下降
  发送/结果），不逐 tick 写盘、不新增线程/舵机调用。
* ``stage_enter`` 在阶段变更处记录；``failure`` 在故障保持改写 stage **之前**记录，因此
  事件带的是**原始阶段**；``result`` 每次调用一条，outcome=最终 status。
* ``descent_submit`` 由执行器在 DESCEND 段**首个非零下降节点**的提交边界记录，三态
  submitted/rejected/unknown；预规划、APPROACH 与"首个非零下降节点之前"的失败都不产生
  submitted 事件。
* P4-T10（下降前/后失败、写前拒绝、部分写中断）→ trace 保存原阶段；began_descend 由
  submitted 事件承担，rejected/无事件为 False、unknown 为 null，未知不会被悄悄丢出分母。
* "对原运动测试增加 trace 开关前后运动指令/返回值不变的断言，验证模拟时钟不额外推进"
  由本文件 §7 与 tests/test_grasp_flow.py 末节的追加用例共同承担。

时序全部走注入的虚拟时钟（tests.mock_motor.SimTime），与既有运动测试同一套 Mock 脚本。
本文件只读地观察运动层公开接口与 trace 钩子，不修改任何运动判据、不触碰 vision。
"""

from __future__ import annotations

import dataclasses
import inspect
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from configs.common_interface import (
    GraspResult,
    GraspStatus,
    MotorCommunicationError,
    MotorLimitError,
    MotorStateError,
)
from qingyun.grabbing import arm_control as AC
from qingyun.grabbing import executor as EX
from qingyun.grabbing.arm_control import ArmController, MotionTraceEvent
from qingyun.grabbing.trajectory import MotionSegment
from tests.mock_motor import MockMotorController, SimTime
from tests.support import graspable_target_at


# ---------------------------------------------------------------------------
# 1. 场景脚本与运行缓存
#
# 一次完整抓取—放置约 10~25 秒 CPU 时间，因此同一 (场景, 开关) 的运行结果被缓存复用：
# Mock 故障脚本、虚拟时钟与注入参数完全固定，运动是确定性的，两次运行必然得到同样的
# 指令数与返回值。缓存只为省时间，不跨用例共享任何会被改写的生活状态。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """一次运行的全部可观测结果（写次数、读次数、时钟、返回值、trace 快照）。"""

    scenario: str
    enable_trace: bool
    result: GraspResult
    trace: tuple[MotionTraceEvent, ...]
    command_count: int
    read_count: int
    now_s: float
    threads_before: int
    threads_after: int


# target/object_gap 决定"抓得到/抓空"，fault 是下降边界的注入配置，其余是 Mock 开关。
_SCENARIOS: dict[str, dict] = {
    "success": dict(target=(0.34, 0.04), object_gap=0.022),
    "grasp_miss": dict(target=(0.34, 0.04), object_gap=None),
    "preplan_reject": dict(target=(0.05, 0.0), object_gap=0.022),
    "approach_fail": dict(target=(0.34, 0.04), object_gap=0.022, lag_tau_s=1.2),
    "feedback_expired": dict(target=(0.34, 0.04), object_gap=0.022,
                             expire_feedback=True),
    "before_boundary_rejected": dict(
        target=(0.34, 0.04), object_gap=0.022,
        fault=("before_boundary", "limit_before_write")),
    "descent_rejected": dict(
        target=(0.34, 0.04), object_gap=0.022,
        fault=("boundary", "limit_before_write")),
    "descent_unknown_no_write": dict(
        target=(0.34, 0.04), object_gap=0.022,
        fault=("boundary", "comm_before_write")),
    "descent_unknown_after_write": dict(
        target=(0.34, 0.04), object_gap=0.022,
        fault=("boundary", "comm_after_write")),
    "descent_fail_after_submitted": dict(
        target=(0.34, 0.04), object_gap=0.022,
        fault=("after_boundary", "stick_after_write")),
}

_RUNS: dict[tuple[str, bool], RunRecord] = {}


def _bare_arm(params, *, enable_trace: bool = True) -> tuple[ArmController, SimTime,
                                                             MockMotorController]:
    """一套"停在 home、夹爪 50%、虚拟时钟"的干净装置（与 test_grasp_flow 同构）。"""
    st = SimTime()
    motor = MockMotorController(params, st, joints_deg=params.workspace.home_joints_deg,
                                gripper_pct=50.0)
    motor.settle_instantly()
    arm = ArmController(motor, params, None, clock=st.monotonic, sleep=st.sleep,
                        clock_ns=st.monotonic_ns, enable_motion_trace=enable_trace)
    return arm, st, motor


def _run(params, scenario: str, enable_trace: bool) -> RunRecord:
    spec = _SCENARIOS[scenario]
    st = SimTime()
    motor = MockMotorController(params, st, joints_deg=params.workspace.home_joints_deg,
                                gripper_pct=50.0,
                                lag_tau_s=float(spec.get("lag_tau_s", 0.05)))
    motor.settle_instantly()
    if spec.get("object_gap") is not None:
        motor.place_object(float(spec["object_gap"]))
    if spec.get("expire_feedback"):
        motor.expire_feedback = True
    arm = ArmController(motor, params, None, clock=st.monotonic, sleep=st.sleep,
                        clock_ns=st.monotonic_ns, enable_motion_trace=enable_trace)
    fault = spec.get("fault")
    if fault is not None:
        _install_descent_fault(motor, arm, at=fault[0], action=fault[1])
    before = threading.active_count()
    result = arm.grasp_and_place(graspable_target_at(*spec["target"]), place_id="default")
    return RunRecord(
        scenario=scenario, enable_trace=enable_trace, result=result,
        trace=arm.get_last_motion_trace(), command_count=motor.command_count,
        read_count=motor.read_count, now_s=st.now_s,
        threads_before=before, threads_after=threading.active_count(),
    )


def run_scenario(params, scenario: str, *, enable_trace: bool = True) -> RunRecord:
    key = (scenario, enable_trace)
    if key not in _RUNS:
        _RUNS[key] = _run(params, scenario, enable_trace)
    return _RUNS[key]


def events(record: RunRecord, kind: str) -> tuple[MotionTraceEvent, ...]:
    return tuple(e for e in record.trace if e.kind == kind)


def entered_stages(record: RunRecord) -> tuple[str, ...]:
    return tuple(e.stage for e in events(record, AC.TRACE_STAGE_ENTER))


# ---------------------------------------------------------------------------
# 2. 下降边界的"独立判定"与驱动侧故障注入
# ---------------------------------------------------------------------------


def _descend_boundary(segment) -> int | None:
    """独立实现的"首个非零下降节点"判定，只用于定位故障注入点。

    刻意与生产实现（``executor.first_descent_node_index``）分开写：这里按 tcp_targets 的
    基座 Z 逐点扫描"相对段起点节点 0 的累计降幅 > 1e-9 m"，不复用被测代码的返回值，避免
    "用被测实现验证被测实现"。判定规则本身在 §4 用合成段单独钉住。
    """
    tcp = segment.tcp_targets
    if tcp is None:
        return None
    z0 = float(np.asarray(tcp[0], float)[2, 3])
    for k in range(int(len(tcp))):
        if z0 - float(np.asarray(tcp[k], float)[2, 3]) > 1e-9:
            return k
    return None


def _install_descent_fault(motor: MockMotorController, arm: ArmController, *,
                           at: str, action: str) -> None:
    """在 DESCEND 段的指定节点上注入一次驱动侧故障。

    接线方式满足 P4 §2.8.2 对标定工具的同一约束——不读控制器私有阶段、不替换规划器、
    不改任何运动判据：

    1. 包一层 ``executor.play_segment``，只记录"当前正在回放哪一段"并把段内提交序号归零；
    2. 包一层 ``motor.send_action``，按段名（下降段名为 ``descend``）与段内提交序号定位
       节点，再按 ``action`` 执行"写前拒绝 / 写前通信异常 / 写后通信中断 / 写后卡死"。

    ``at`` 取 ``before_boundary``（该段起点，零位移节点）、``boundary``（首个非零下降节点）、
    ``after_boundary``（已经开始下降之后的下一个节点）。
    """
    state = {"segment": None, "submitted": 0}
    original_play = arm.executor.play_segment

    def play(segment):
        state["segment"] = segment
        state["submitted"] = 0
        return original_play(segment)

    arm.executor.play_segment = play

    original_send = motor.send_action

    def send(joints_deg, gripper_pct):
        segment = state["segment"]
        target_node = None
        if segment is not None and segment.name == "descend":
            boundary = _descend_boundary(segment)
            if boundary is not None:
                target_node = {"before_boundary": 0, "boundary": boundary,
                               "after_boundary": boundary + 1}[at]
        hit = target_node is not None and state["submitted"] == target_node
        if hit and action == "limit_before_write":
            state["submitted"] += 1
            raise MotorLimitError("测试注入：下降指令在写入前被驱动限位整条拒绝")
        if hit and action == "comm_before_write":
            state["submitted"] += 1
            raise MotorCommunicationError("测试注入：下降指令通信异常，无法证明是否写入")
        original_send(joints_deg, gripper_pct)
        state["submitted"] += 1
        if hit and action == "comm_after_write":
            raise MotorCommunicationError("测试注入：下降指令部分写入后总线中断")
        if hit and action == "stick_after_write":
            motor.stuck_joints = True         # 已经开始下降，之后舵机不再跟进

    motor.send_action = send


# ---------------------------------------------------------------------------
# 3. 事件类型与公开快照接口
# ---------------------------------------------------------------------------


def test_事件类型定义与字段顺序符合规格():
    """字段 sequence:int, time_ns:int, stage:str, kind:str, outcome:str|None，冻结。"""
    fields = dataclasses.fields(MotionTraceEvent)
    assert [f.name for f in fields] == ["sequence", "time_ns", "stage", "kind", "outcome"]
    assert [f.type for f in fields] == ["int", "int", "str", "str", "str | None"]
    event = MotionTraceEvent(sequence=1, time_ns=2, stage="DESCEND",
                             kind="descent_submit", outcome="submitted")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.sequence = 7                                    # type: ignore[misc]
    # 冻结因此可哈希：工具能安全地把事件放进 set/dict 做去重与关联。
    assert hash(event) == hash(MotionTraceEvent(1, 2, "DESCEND", "descent_submit",
                                                "submitted"))


def test_trace词表在两层保持一致():
    """kind 与三态 outcome 的取值由一层定义、另一层引用，不允许各自写字面量。"""
    assert (AC.TRACE_STAGE_ENTER, AC.TRACE_DESCENT_SUBMIT, AC.TRACE_FAILURE,
            AC.TRACE_RESULT) == ("stage_enter", "descent_submit", "failure", "result")
    assert AC.TRACE_DESCENT_OUTCOMES == (EX.DESCENT_SUBMITTED, EX.DESCENT_REJECTED,
                                         EX.DESCENT_UNKNOWN)
    assert AC.TRACE_DESCENT_OUTCOMES == ("submitted", "rejected", "unknown")
    assert set(AC.ALL_STAGES) >= {AC.STAGE_CHECK, AC.STAGE_DESCEND, AC.STAGE_DONE,
                                  AC.STAGE_FAULT_HOLD, AC.STAGE_FAULT_UNCONTROLLED}


def test_公开trace出口与开关参数形态():
    signature = inspect.signature(ArmController.__init__)
    assert "enable_motion_trace" in signature.parameters
    assert signature.parameters["enable_motion_trace"].default is True, \
        "trace 默认开启，标定工具无需特殊装配"
    assert inspect.signature(ArmController.get_last_motion_trace).return_annotation \
        == "tuple[MotionTraceEvent, ...]"


def test_未调用任何动作时快照为空(params):
    arm, st, motor = _bare_arm(params)
    assert arm.get_last_motion_trace() == ()
    # 一次真实的小运动（公开原语）：trace 只由技能入口 grasp_and_place 管理，
    # 原语既不清空、不自造事件，上一次调用留下的快照因此保持为空。
    arm.move_joints(np.asarray(params.workspace.home_joints_deg, float)
                    + np.array([5.0, 0, 0, 0, 0]))
    assert motor.command_count > 0, "前置条件：这次原语确实动了"
    assert arm.get_last_motion_trace() == (), \
        "公开原语不改写上一次技能调用的快照（也不自造事件）"


# ---------------------------------------------------------------------------
# 4. 下降边界判定与三态分类（纯计算单元）
# ---------------------------------------------------------------------------


def _vertical_segment(zs: list[float], *, joints: np.ndarray | None = None,
                      name: str = "descend") -> MotionSegment:
    """按给定的逐节点基座 Z 造一段笛卡尔竖直段（joint 行不参与 Z 判定）。"""
    n = len(zs)
    tcp = np.tile(np.eye(4), (n, 1, 1))
    tcp[:, 2, 3] = np.asarray(zs, dtype=np.float64)
    q = np.zeros((n, 5), dtype=np.float64) if joints is None else np.asarray(joints, float)
    return MotionSegment(name, "CARTESIAN", np.arange(n, dtype=np.float64) / 30.0,
                         q, 50.0, tcp)


@pytest.mark.parametrize("zs, expected, why", [
    ([0.0], None, "零位移段只有一个节点（起点已在目标高度）：没有下降节点"),
    ([0.0, 0.0], None, "多节点但全部与起点同高：没有下降节点"),
    ([0.0, 0.0, 0.0, -0.05], 3, "前导零位移节点被跳过，第一个真正下降的是第 3 个"),
    ([0.05, 0.05, 0.049], 2, "基准是段起点节点 0 的高度，不是上一节点"),
    ([0.0, 1e-10, 2e-10], None, "降幅不超过 1e-9 m 的节点不计为下降（幅值判定）"),
    ([0.0, -1e-10, -0.004], 2, "不足阈值的第 1 节点被跳过，超过阈值的第 2 节点是边界"),
    ([0.0, 0.01, 0.02], None, "竖直向上段没有下降节点，绝不为它伪造 submitted"),
    ([0.0, -0.001, -0.002], 1, "常规下降段：节点 0 位移为零，第 1 个节点即边界"),
    ([-0.02, -0.02, -0.03], 2, "起点高度为负也按相对降幅判定"),
])
def test_首个非零下降节点按符号与幅值判定(zs, expected, why):
    assert EX.first_descent_node_index(_vertical_segment(zs)) == expected, why


def test_真实规划输出与测试侧独立判定同结论(params):
    """规划器真实生成的下降段：两套判定必须给出同一个节点号（或同为 None）。"""
    arm, st, motor = _bare_arm(params)
    q = np.asarray(params.workspace.home_joints_deg, float)
    g = float(params.gripper.preopen_pct)
    for dz, expect_descend in ((-0.02, True), (-0.05, True), (0.0, False)):
        segment = AC.make_cartesian_vertical_segment("descend", arm.model, arm.ik,
                                                     arm.limits, q, g, dz, params)
        produced = EX.first_descent_node_index(segment)
        assert (produced is not None) == expect_descend, f"dz={dz}"
        assert produced == _descend_boundary(segment), f"dz={dz} 两套判定不一致"


def test_关节段无TCP目标时按关节位移兜底判定():
    base = np.zeros(5)
    joints = np.vstack([base, base, base + 1e-12, base + np.array([0.5, 0, 0, 0, 0])])
    segment = MotionSegment("x", "JOINT", np.arange(4, dtype=np.float64) / 30.0,
                            joints, 50.0, None)
    assert EX.first_descent_node_index(segment) == 3
    single = MotionSegment("x", "JOINT", np.array([0.0]), joints[:1], 50.0, None)
    assert EX.first_descent_node_index(single) is None, "单节点零位移段不算"


@pytest.mark.parametrize("exc, expected", [
    (MotorLimitError("限位拒绝"), EX.DESCENT_REJECTED),
    (EX.ExecutionError("PLAN_INVALID", "指令被驱动限位拒绝"), EX.DESCENT_REJECTED),
    (MotorCommunicationError("写超时"), EX.DESCENT_UNKNOWN),
    (MotorStateError("设备未就绪"), EX.DESCENT_UNKNOWN),
    (RuntimeError("未知驱动异常"), EX.DESCENT_UNKNOWN),
    (KeyboardInterrupt(), EX.DESCENT_UNKNOWN),
])
def test_下降三态分类只有明确写前拒绝才算rejected(exc, expected):
    assert EX.classify_descent_submit_error(exc) == expected


def test_被翻译成控制层异常的写前拒绝仍判rejected():
    """执行器把 MotorLimitError 翻译成 ExecutionError(PLAN_INVALID)，cause 链要认得出。"""
    wrapped = EX.ExecutionError("PLAN_INVALID", "指令被驱动限位拒绝")
    wrapped.__cause__ = MotorLimitError("整条拒绝")
    assert EX.classify_descent_submit_error(wrapped) == EX.DESCENT_REJECTED
    # 通信类被翻译成 COMM_ERROR：不能证明是否写入，必须 unknown，不能当成没发过。
    comm = EX.ExecutionError("COMM_ERROR", "电机通信/状态异常")
    comm.__cause__ = MotorCommunicationError("部分写入")
    assert EX.classify_descent_submit_error(comm) == EX.DESCENT_UNKNOWN


def test_执行器接入观察钩子后写入次数等于节点数且只报一次边界(params):
    """钩子只做旁路观察：一个段的提交次数不变、边界上报次数恰好为 1。"""
    class FakeSink:
        def __init__(self) -> None:
            self.outcomes: list[str] = []
            self.asks = 0

        def watches_descent(self) -> bool:
            self.asks += 1
            return len(self.outcomes) == 0

        def note_descent_submit(self, outcome: str) -> None:
            self.outcomes.append(outcome)

    arm, st, motor = _bare_arm(params, enable_trace=False)
    q = np.asarray(params.workspace.home_joints_deg, float)
    g = float(params.gripper.preopen_pct)
    segment = AC.make_cartesian_vertical_segment("descend", arm.model, arm.ik, arm.limits,
                                                 q, g, -0.02, params)
    sink = FakeSink()
    executor = EX.SyncExecutor(motor, params, clock=st.monotonic, sleep=st.sleep,
                              clock_ns=st.monotonic_ns, motion_trace=sink)
    commands_before = motor.command_count
    executor.play_segment(segment)
    assert motor.command_count - commands_before == segment.nodes, \
        "回放一段就是逐节点提交，钩子不增减写入"
    assert sink.outcomes == [EX.DESCENT_SUBMITTED]
    assert sink.asks == 1, "play_segment 每次调用只在段首问一次要不要观察"


def test_不接钩子时执行器一次都不询问观察(params):
    arm, st, motor = _bare_arm(params, enable_trace=False)
    assert motor.send_action is not None
    q = np.asarray(params.workspace.home_joints_deg, float)
    segment = AC.make_cartesian_vertical_segment("descend", arm.model, arm.ik, arm.limits,
                                                 q, float(params.gripper.preopen_pct),
                                                 -0.02, params)
    executor = EX.SyncExecutor(motor, params, clock=st.monotonic, sleep=st.sleep,
                              clock_ns=st.monotonic_ns)
    assert executor.motion_trace is None
    executor.play_segment(segment)
    assert motor.command_count == segment.nodes


# ---------------------------------------------------------------------------
# 5. 收集器语义（有界、去重、武装门）
# ---------------------------------------------------------------------------


def test_收集器只记每阶段首次进入且条数有界():
    collector = AC._MotionTraceCollector(lambda: 0)
    collector.begin_call()
    for stage in [AC.STAGE_CHECK, AC.STAGE_OPEN, AC.STAGE_APPROACH, AC.STAGE_DESCEND,
                  AC.STAGE_CLOSE, AC.STAGE_LIFT, AC.STAGE_CHECK]:
        collector.note_stage(stage)
    assert [e.kind for e in collector.snapshot()] == [AC.TRACE_STAGE_ENTER] * 6, \
        "重复进入同一阶段不再追加事件，但阶段名仍被跟踪"
    collector.note_stage(AC.STAGE_DESCEND)
    assert collector.watches_descent() is True, "去重不影响对『当前阶段』的跟踪"
    assert len(collector.snapshot()) <= len(AC.ALL_STAGES) + 3


def test_收集器未武装时零收集且下降钩子只在DESCEND观察一次():
    collector = AC._MotionTraceCollector(lambda: 0)
    collector.note_stage(AC.STAGE_DESCEND)
    assert collector.snapshot() == (), "调用之外不得污染上一次调用的快照"
    assert collector.watches_descent() is False
    collector.begin_call()
    assert collector.watches_descent() is False, "当前阶段不是 DESCEND"
    collector.note_stage(AC.STAGE_DESCEND)
    assert collector.watches_descent() is True
    collector.note_descent_submit(EX.DESCENT_SUBMITTED)
    assert collector.watches_descent() is False, "每次调用只观察一次下降边界"
    collector.note_descent_submit(EX.DESCENT_UNKNOWN)
    collector.note_failure("TIMEOUT")
    collector.note_failure("COMM_ERROR")
    collector.note_stage(AC.STAGE_DESCEND)
    snapshot = collector.snapshot()
    assert [e.outcome for e in snapshot if e.kind == AC.TRACE_DESCENT_SUBMIT] == \
        [EX.DESCENT_SUBMITTED]
    failures = [e for e in snapshot if e.kind == AC.TRACE_FAILURE]
    assert len(failures) == 1 and failures[0].stage == AC.STAGE_DESCEND
    assert len(snapshot) == 3


def test_收集器result之后停收且快照是新建元组():
    ticks = iter(range(1000, 100_000, 1000))
    collector = AC._MotionTraceCollector(lambda: next(ticks))
    collector.begin_call()
    collector.note_stage(AC.STAGE_CHECK)
    snapshot = collector.snapshot()
    collector.note_failure("COLLISION")
    assert snapshot == (MotionTraceEvent(0, 1000, AC.STAGE_CHECK,
                                         AC.TRACE_STAGE_ENTER, None),)
    assert isinstance(snapshot, tuple) and len(collector.snapshot()) == 2, \
        "已交出的快照不受后续追加影响"
    collector.note_result("COLLISION")
    collector.note_stage(AC.STAGE_DONE)
    assert [e.sequence for e in collector.snapshot()] == [0, 1, 2], "result 之后不再收集"


# ---------------------------------------------------------------------------
# 6. P4-T10：下降前/后失败、写前拒绝、部分写中断
# ---------------------------------------------------------------------------


def test_T10_成功_一条descent提交_无failure_结果为SUCCESS(params):
    record = run_scenario(params, "success")
    assert record.result.status is GraspStatus.SUCCESS
    assert [e.outcome for e in events(record, AC.TRACE_DESCENT_SUBMIT)] == \
        [EX.DESCENT_SUBMITTED]
    assert events(record, AC.TRACE_FAILURE) == ()
    results = events(record, AC.TRACE_RESULT)
    assert len(results) == 1 and results[0].outcome == "SUCCESS"
    assert results[0].stage == AC.STAGE_DONE
    assert entered_stages(record) == (
        AC.STAGE_CHECK, AC.STAGE_OPEN, AC.STAGE_APPROACH, AC.STAGE_DESCEND,
        AC.STAGE_CLOSE, AC.STAGE_LIFT, AC.STAGE_TRANSFER, AC.STAGE_LOWER,
        AC.STAGE_RELEASE, AC.STAGE_RETREAT, AC.STAGE_RETURN, AC.STAGE_DONE)
    _assert_common_trace_invariants(record)


def test_T10_预规划失败_无descent事件_failure携带CHECK原阶段(params):
    record = run_scenario(params, "preplan_reject")
    assert record.result.status in (GraspStatus.PLAN_INVALID, GraspStatus.IK_FAILED)
    assert record.command_count == 0, "预规划失败臂未动，不该发过任何指令"
    assert events(record, AC.TRACE_DESCENT_SUBMIT) == ()
    assert entered_stages(record) == (AC.STAGE_CHECK,)
    failures = events(record, AC.TRACE_FAILURE)
    assert len(failures) == 1 and failures[0].stage == AC.STAGE_CHECK
    assert failures[0].outcome == record.result.status.value
    assert AC.STAGE_FAULT_HOLD not in entered_stages(record), "臂未动的拒绝不改写阶段"
    _assert_common_trace_invariants(record)


def test_T10_APPROACH阶段失败不产生任何下降事件(params):
    record = run_scenario(params, "approach_fail")
    assert AC.TRACE_DESCENT_SUBMIT not in [e.kind for e in record.trace]
    assert AC.STAGE_DESCEND not in entered_stages(record)
    assert events(record, AC.TRACE_FAILURE)[-1].stage == AC.STAGE_APPROACH
    assert record.result.status is GraspStatus.TRACKING_ERROR
    _assert_common_trace_invariants(record)


def test_T10_下降首个非零节点之前失败不产生descent事件(params):
    """下降段起点是零位移节点：在它上面被拒绝，等于"还没下发过下降指令"。"""
    record = run_scenario(params, "before_boundary_rejected")
    assert record.result.status is GraspStatus.PLAN_INVALID
    assert events(record, AC.TRACE_DESCENT_SUBMIT) == (), \
        "首个非零下降节点之前的失败不得产生任何 descent_submit 事件"
    assert AC.STAGE_DESCEND in entered_stages(record)
    assert events(record, AC.TRACE_FAILURE)[-1].stage == AC.STAGE_DESCEND
    _assert_failure_before_fault_hold(record)


def test_T10_首个下降节点写前被拒记rejected且没有submitted(params):
    record = run_scenario(params, "descent_rejected")
    submissions = events(record, AC.TRACE_DESCENT_SUBMIT)
    assert [e.outcome for e in submissions] == [EX.DESCENT_REJECTED]
    assert submissions[0].stage == AC.STAGE_DESCEND
    assert EX.DESCENT_SUBMITTED not in [e.outcome for e in submissions], \
        "驱动写前明确拒绝＝可证明没下发，绝不能记成 submitted"
    assert record.result.status is GraspStatus.PLAN_INVALID
    assert record.result.recovery_required is True
    _assert_failure_before_fault_hold(record)


def test_T10_下降边界通信异常两种情形都记unknown(params):
    """未写就断与写了一半才断都无法证明结果，一律 unknown（不冒充 False）。"""
    for scenario in ("descent_unknown_no_write", "descent_unknown_after_write"):
        record = run_scenario(params, scenario)
        submissions = events(record, AC.TRACE_DESCENT_SUBMIT)
        assert [e.outcome for e in submissions] == [EX.DESCENT_UNKNOWN], scenario
        assert record.result.status is GraspStatus.COMM_ERROR, scenario
        _assert_failure_before_fault_hold(record)


def test_T10_写前拒绝与写后中断的差别由驱动侧写入次数证明(params):
    """rejected 与 unknown(写前) 都没有落笔，unknown(写后) 多出一条真实写入。"""
    rejected = run_scenario(params, "descent_rejected")
    after_write = run_scenario(params, "descent_unknown_after_write")
    no_write = run_scenario(params, "descent_unknown_no_write")
    assert after_write.command_count == rejected.command_count + 1
    assert no_write.command_count == rejected.command_count


def test_T10_下降中失败_submitted在场且failure先于FAULT_HOLD改写(params):
    record = run_scenario(params, "descent_fail_after_submitted")
    assert [e.outcome for e in events(record, AC.TRACE_DESCENT_SUBMIT)] == \
        [EX.DESCENT_SUBMITTED]
    _assert_failure_before_fault_hold(record)
    assert events(record, AC.TRACE_FAILURE)[-1].stage == AC.STAGE_DESCEND
    assert record.result.stage in (AC.STAGE_FAULT_HOLD, AC.STAGE_FAULT_UNCONTROLLED)


def test_T10_下降后闭爪失败_failure原阶段是CLOSE(params):
    record = run_scenario(params, "grasp_miss")
    assert record.result.status is GraspStatus.GRASP_MISS
    assert [e.outcome for e in events(record, AC.TRACE_DESCENT_SUBMIT)] == \
        [EX.DESCENT_SUBMITTED]
    assert AC.STAGE_DESCEND in entered_stages(record)
    _assert_failure_before_fault_hold(record)
    failure = events(record, AC.TRACE_FAILURE)[-1]
    assert failure.stage == AC.STAGE_CLOSE, "原始阶段是 CLOSE，不是被改写后的 FAULT_HOLD"
    assert failure.outcome == "GRASP_MISS"


def test_T10_执行期反馈失败同样保留原始阶段与三态缺失(params):
    record = run_scenario(params, "feedback_expired")
    assert record.result.status is GraspStatus.TIMEOUT
    assert events(record, AC.TRACE_DESCENT_SUBMIT) == ()
    assert events(record, AC.TRACE_FAILURE)[-1].stage == AC.STAGE_CHECK
    _assert_failure_before_fault_hold(record)


def began_descend_of(record: RunRecord) -> bool | None:
    """按 §2.8.2 的推导规则从 trace 得出 began_descend（工具侧口径的镜像）。

    submitted → True；rejected / 无事件 → False；unknown → null（保留整条试验，交给
    操作者补现场/视频证据后定值并记修订），不得直接算 False 或丢弃。
    """
    outcomes = [e.outcome for e in events(record, AC.TRACE_DESCENT_SUBMIT)]
    if EX.DESCENT_SUBMITTED in outcomes:
        return True
    if EX.DESCENT_UNKNOWN in outcomes:
        return None
    return False


@pytest.mark.parametrize("scenario, outcome, began", [
    ("success", "submitted", True),
    ("grasp_miss", "submitted", True),
    ("descent_fail_after_submitted", "submitted", True),
    ("descent_rejected", "rejected", False),
    ("descent_unknown_no_write", "unknown", None),
    ("descent_unknown_after_write", "unknown", None),
    ("before_boundary_rejected", None, False),
    ("preplan_reject", None, False),
    ("approach_fail", None, False),
    ("feedback_expired", None, False),
])
def test_T10_began_descend只由submitted事件承担(params, scenario, outcome, began):
    record = run_scenario(params, scenario)
    submissions = [e.outcome for e in events(record, AC.TRACE_DESCENT_SUBMIT)]
    assert (submissions[0] if submissions else None) == outcome, scenario
    assert began_descend_of(record) is began, scenario


def test_T10_trace关闭时零事件收集且不创建收集器(params):
    record = run_scenario(params, "success", enable_trace=False)
    assert record.trace == ()
    arm, st, motor = _bare_arm(params, enable_trace=False)
    assert arm.motion_trace_enabled is False
    assert arm._motion_trace is None, "关闭时不持有事件列表，不占内存"
    assert arm.executor.motion_trace is None, "关闭时执行器拿不到钩子"
    assert arm.get_last_motion_trace() == ()


def test_T10_每次调用清空且返回后保留到下一次调用(params):
    arm, st, motor = _bare_arm(params)
    motor.place_object(0.022)
    motor.expire_feedback = True
    first = arm.grasp_and_place(graspable_target_at(0.34, 0.04))
    assert first.status is GraspStatus.TIMEOUT
    snapshot_first = arm.get_last_motion_trace()
    assert [e.kind for e in snapshot_first] == [
        AC.TRACE_STAGE_ENTER, AC.TRACE_FAILURE, AC.TRACE_STAGE_ENTER, AC.TRACE_RESULT]
    assert arm.get_last_motion_trace() == snapshot_first, "返回后保留到下一次调用"
    # 复位后换一个在 CHECK 就被拒绝的目标：新快照不得含上一次调用的任何事件。
    motor.expire_feedback = False
    arm.reset_fault()
    second = arm.grasp_and_place(graspable_target_at(0.05, 0.0))
    snapshot_second = arm.get_last_motion_trace()
    assert [e.sequence for e in snapshot_second] == list(range(len(snapshot_second)))
    assert second.status.value not in [e.outcome for e in snapshot_first]
    assert [e.outcome for e in snapshot_second][-1] == second.status.value
    assert snapshot_first[-1].outcome == "TIMEOUT", "上一次交出的快照不被改写"


def test_T10_快照不可变外部改动影响不了控制器内部事件(params):
    arm, st, motor = _bare_arm(params)
    motor.expire_feedback = True
    arm.grasp_and_place(graspable_target_at(0.34, 0.04))
    snapshot = arm.get_last_motion_trace()
    assert isinstance(snapshot, tuple)
    with pytest.raises(TypeError):
        snapshot[0] = None                                    # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot[0].stage = "HACKED"                           # type: ignore[misc]
    internal = arm._motion_trace._events                       # noqa: SLF001
    assert isinstance(internal, list) and snapshot == tuple(internal)
    internal.clear()                                           # 只有内部列表可变
    assert snapshot != arm._motion_trace.snapshot()            # noqa: SLF001
    assert arm._motion_trace.snapshot() == ()                   # noqa: SLF001


# ---------------------------------------------------------------------------
# 7. 运动不变性：trace 开关前后指令/返回值/模拟时钟完全一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", [
    "success", "grasp_miss", "descent_rejected", "before_boundary_rejected",
    "preplan_reject", "feedback_expired",
])
def test_开关前后运动指令读回与返回值完全一致(params, scenario):
    """P4 §2.8.2 末段：trace 开关前后运动指令/返回值不变、模拟时钟不额外推进。"""
    on = run_scenario(params, scenario)
    off = run_scenario(params, scenario, enable_trace=False)
    assert on.command_count == off.command_count, "trace 不得新增舵机写入"
    assert on.read_count == off.read_count, "trace 不得新增反馈读取"
    assert on.now_s == off.now_s, "trace 不得额外推进模拟时钟"
    assert on.result == off.result
    for field in ("status", "stage", "reason", "place_id", "holding", "recovery_required"):
        assert getattr(on.result, field) == getattr(off.result, field), (scenario, field)
    assert off.trace == ()
    assert on.trace != ()


def test_开关前后逐条发送的关节角与开度序列一致(params):
    """更细一层的时序不变量：每一次发送的载荷与顺序也必须逐条相同。"""
    on = _command_sequence(params, "descent_rejected", enable_trace=True)
    off = _command_sequence(params, "descent_rejected", enable_trace=False)
    assert on == off
    assert len(on) > 20, "这条断言只在真的走完 OPEN/APPROACH/DESCEND 时才有意义"


def _command_sequence(params, scenario: str, *, enable_trace: bool) -> list[tuple]:
    spec = _SCENARIOS[scenario]
    st = SimTime()
    motor = MockMotorController(params, st, joints_deg=params.workspace.home_joints_deg,
                                gripper_pct=50.0)
    motor.settle_instantly()
    motor.place_object(float(spec["object_gap"]))
    arm = ArmController(motor, params, None, clock=st.monotonic, sleep=st.sleep,
                        clock_ns=st.monotonic_ns, enable_motion_trace=enable_trace)
    _install_descent_fault(motor, arm, at=spec["fault"][0], action=spec["fault"][1])
    log: list[tuple] = []
    original_send = motor.send_action

    def spy(joints_deg, gripper_pct):
        log.append((np.round(np.asarray(joints_deg, float), 9).tolist(),
                    round(float(gripper_pct), 9)))
        return original_send(joints_deg, gripper_pct)

    motor.send_action = spy
    arm.grasp_and_place(graspable_target_at(*spec["target"]), place_id="default")
    return log


def test_开关前后事件条数上界与线程数不变(params):
    on = run_scenario(params, "grasp_miss")
    off = run_scenario(params, "grasp_miss", enable_trace=False)
    assert on.threads_before == on.threads_after
    assert off.threads_after == on.threads_after == threading.active_count()
    assert len(on.trace) <= len(AC.ALL_STAGES) + 3
    assert all(a.time_ns <= b.time_ns for a, b in zip(on.trace, on.trace[1:])), \
        "事件时间戳取自注入时钟，单调不回退"


# ---------------------------------------------------------------------------
# 8. 公共断言小工具
# ---------------------------------------------------------------------------


def _assert_common_trace_invariants(record: RunRecord) -> None:
    trace = record.trace
    assert trace, "开启 trace 的运行必须留下事件"
    assert [e.sequence for e in trace] == list(range(len(trace))), "sequence 从 0 单调递增"
    assert trace[-1].kind == AC.TRACE_RESULT, "每次调用以一条 result 收尾"
    assert [e.kind for e in trace].count(AC.TRACE_RESULT) == 1
    assert len(trace) <= len(AC.ALL_STAGES) + 3, "事件条数必须有界"
    counts = Counter(e.stage for e in events(record, AC.TRACE_STAGE_ENTER))
    assert set(counts.values()) == {1}, "每阶段首次进入只记一条"
    assert all(isinstance(e.time_ns, int) for e in trace)
    assert all(e.kind in (AC.TRACE_STAGE_ENTER, AC.TRACE_DESCENT_SUBMIT,
                          AC.TRACE_FAILURE, AC.TRACE_RESULT) for e in trace)
    for e in trace:
        if e.kind == AC.TRACE_STAGE_ENTER:
            assert e.outcome is None
        else:
            assert isinstance(e.outcome, str) and e.outcome


def _assert_failure_before_fault_hold(record: RunRecord) -> None:
    """failure 事件必须排在把 stage 改写成 FAULT_* 的那条 stage_enter 之前。"""
    pairs = [(e.kind, e.stage) for e in record.trace]
    index_failure = next(i for i, (kind, _) in enumerate(pairs) if kind == AC.TRACE_FAILURE)
    index_hold = next((i for i, (kind, stage) in enumerate(pairs)
                       if kind == AC.TRACE_STAGE_ENTER
                       and stage in (AC.STAGE_FAULT_HOLD, AC.STAGE_FAULT_UNCONTROLLED)),
                      None)
    assert index_hold is not None, "运动后失败必须留下最终阶段的 stage_enter"
    assert index_failure < index_hold
    assert record.trace[index_failure].stage not in (AC.STAGE_FAULT_HOLD,
                                                    AC.STAGE_FAULT_UNCONTROLLED)
    _assert_common_trace_invariants(record)
