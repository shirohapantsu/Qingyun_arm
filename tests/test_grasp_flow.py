"""同步执行器与抓取—放置流程的测试。

覆盖技术文档第八节必测项：
    1 一次视觉下发对应一次同步调用，原始目标在整个调用内不被改写；place_id 只读取
      配置，不根据 grade 选择
    2 输入格式错误、缺失放置配置和越限命令在运动前拒绝
    5 Mock 覆盖滞后、过流、空闭合端电流、抓空、开合超时、通信失败、过期反馈和 CPU 迟到
    6 各动作、sleep 和规划检查点可响应 should_stop；函数返回后无隐藏执行任务

时序全部走注入的虚拟时钟（tests/mock_motor.SimTime），所以几十秒的真实流程
在这台机器上几秒内就能跑完，而且完全可复现。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from configs.common_interface import (
    GraspStatus,
    HoldingState,
    MotorCommunicationError,
    MotorLimitError,
    VisionInterface,
)
from qingyun.grabbing import arm_control as AC
from qingyun.grabbing.arm_control import ArmController
from qingyun.grabbing.executor import ExecutionError, SyncExecutor
from qingyun.grabbing.kinematics_ext import StoppedByRequest
from qingyun.grabbing.trajectory import make_joint_segment
from tests.mock_motor import MockMotorController, SimTime


def make_arm(params, *, joints=None, gripper_pct=50.0, should_stop=None, **mock_kwargs):
    """建一套 (控制器, 虚拟时钟, Mock 驱动)。Mock 初始已停在给定的静止姿态。"""
    st = SimTime()
    motor = MockMotorController(
        params, st,
        joints_deg=joints if joints is not None else params.workspace.home_joints_deg,
        gripper_pct=gripper_pct, **mock_kwargs,
    )
    motor.settle_instantly()
    arm = ArmController(motor, params, should_stop,
                        clock=st.monotonic, sleep=st.sleep, clock_ns=st.monotonic_ns)
    return arm, st, motor


def graspable(params, x, y, z=0.022, grade="A"):
    yaw = math.degrees(math.atan2(y, x))
    return VisionInterface(position=np.array([x, y, z]), yaw_deg=yaw, grade=grade)


# ---------------------------------------------------------------------------
# 1. 一次调用一个目标，原始输入不被改写（必测项 1）
# ---------------------------------------------------------------------------


def test_一次下发对应一次完整同步调用(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    result = arm.grasp_and_place(graspable(params, 0.34, 0.04), place_id="default")
    assert result.status is GraspStatus.SUCCESS
    assert result.stage == AC.STAGE_DONE
    # 状态表里的每个阶段都被经过（stage 单调推进到最后一次记录）。
    assert result.recovery_required is False
    assert result.holding is HoldingState.EMPTY
    # 调用返回后机械臂停在配置的待机位上（RETURN 完成）。
    assert np.allclose(motor.q, params.workspace.home_joints_deg, atol=2.5)
    assert motor.g == pytest.approx(params.gripper.release_pct, abs=3.0)


def test_原始目标数组在调用期间不被改写(params):
    """文档 3.2：数据类冻结不等于 ndarray 内存只读，入口必须自己复制一份。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    pos = np.array([0.34, 0.04, 0.022])
    target = VisionInterface(position=pos, yaw_deg=math.degrees(math.atan2(0.04, 0.34)),
                             grade="A+")
    snapshot = pos.copy()
    arm.grasp_and_place(target, place_id="default")
    assert np.array_equal(pos, snapshot), "入口把调用方的数组改了或没有复制"


def test_grade不参与放置点选择(params):
    """文档 3.2/4.1：grade 只是记录用元数据，place_id 只从参数读取。"""
    for grade in ("A+", "A", "B", "C", "不合格"):
        arm, st, motor = make_arm(params)
        motor.place_object(0.022)
        r = arm.grasp_and_place(graspable(params, 0.35, 0.0, grade=grade), place_id="bin")
        assert r.place_id == "bin"
        # 同一下发位置、不同品级，提交的指令序列长度一致（没有走不同分支）。
        if grade == "A+":
            baseline = motor.command_count
        else:
            assert abs(motor.command_count - baseline) <= 40, grade


def test_同一控制器连续两次调用都成功(params):
    """一次调用只完成一个动作，返回后才进入下一次（文档 1.2）。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    r1 = arm.grasp_and_place(graspable(params, 0.34, 0.03), place_id="default")
    assert r1.status is GraspStatus.SUCCESS
    motor.place_object(0.026)
    r2 = arm.grasp_and_place(graspable(params, 0.36, -0.03), place_id="bin")
    assert r2.status is GraspStatus.SUCCESS, r2


def test_调用进行中拒绝重入(params):
    """文档 1.2：调用过程中禁止重入。"""
    holder = {"arm": None}
    calls = {"n": 0}

    def reenter():
        calls["n"] += 1
        if calls["n"] == 1:
            with pytest.raises(RuntimeError):
                holder["arm"].grasp_and_place(graspable(params, 0.34, 0.0))
        return False

    arm, st, motor = make_arm(params, should_stop=reenter)
    holder["arm"] = arm
    motor.place_object(0.022)
    r = arm.grasp_and_place(graspable(params, 0.34, 0.0))
    assert calls["n"] >= 1
    assert r.status in (GraspStatus.ABORTED, GraspStatus.SUCCESS)


# ---------------------------------------------------------------------------
# 2. 运动前拒绝（必测项 2）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    np.array([0.34, 0.04]),                       # shape 不对
    np.array([0.34, 0.04, float("nan")]),         # 非有限
    np.array([0.34, 0.04, 0.022], dtype=np.float32),   # dtype 不对
])
def test_非法position在运动前被拒绝(params, bad):
    arm, st, motor = make_arm(params)
    target = VisionInterface(position=bad, yaw_deg=0.0, grade="A")
    r = arm.grasp_and_place(target)
    assert r.status is GraspStatus.INVALID_INPUT
    assert r.recovery_required is False
    assert motor.command_count == 0, "输入不合法却已经给总线发过指令"


def test_yaw超出约定区间被拒绝(params):
    """文档 3.2：[-90,90)；越界说明视觉侧没做归一化。"""
    arm, st, motor = make_arm(params)
    for yaw in (90.0, 120.0, -91.0):
        r = arm.grasp_and_place(VisionInterface(
            position=np.array([0.34, 0.0, 0.022]), yaw_deg=yaw, grade="A"))
        assert r.status is GraspStatus.INVALID_INPUT, yaw
    assert motor.command_count == 0


def test_grade类型不对被拒绝(params):
    arm, st, motor = make_arm(params)
    r = arm.grasp_and_place(VisionInterface(
        position=np.array([0.34, 0.0, 0.022]), yaw_deg=0.0, grade=3))
    assert r.status is GraspStatus.INVALID_INPUT


def test_未知place_id时臂不动(params):
    """文档 3.4：未知 place_id 返回 CONFIG_INVALID，臂不动。"""
    arm, st, motor = make_arm(params)
    r = arm.grasp_and_place(graspable(params, 0.34, 0.0), place_id="ghost")
    assert r.status is GraspStatus.CONFIG_INVALID
    assert r.recovery_required is False
    assert motor.command_count == 0


def test_越限的关节目标在发送前被拒绝(params):
    """move_joints 给一个明显越出有效限位的目标，必须在运动前被拒。"""
    arm, st, motor = make_arm(params)
    bad = np.array(params.workspace.home_joints_deg, float)
    bad[0] = float(arm.limits.upper[0] + 10.0)
    with pytest.raises(Exception) as exc:
        arm.move_joints(bad)
    assert exc.value.status == "PLAN_INVALID"
    assert motor.command_count == 0


def test_越限目标锁存后需要复位才能再用(params):
    arm, st, motor = make_arm(params)
    r = arm.grasp_and_place(graspable(params, 0.05, 0.0))    # 绝对不可达
    assert r.status in (GraspStatus.PLAN_INVALID, GraspStatus.IK_FAILED)
    assert r.recovery_required is False, "臂还没动，不该锁存"
    # 未锁存时下一次调用仍可进
    r2 = arm.grasp_and_place(graspable(params, 0.05, 0.0))
    assert r2.status == r.status


def test_放置位姿不可达时在CHECK就被拒绝(params):
    """P1-4 复现：文档 6.3 要求 CHECK "预规划、校验整条名义路线"。

    旧实现只预规划 APPROACH+DESCEND，放置段在闭爪后才首次规划——抓住并提起
    物体之后才报不可达，返回 recovery_required=True 且臂持物悬停。修复后必须
    在臂未动、没发过任何指令时就拒绝。
    """
    from configs.common_interface import PlacePose
    original = params.places["default"]
    params.places["default"] = PlacePose(position=np.array([0.05, 0.0, 0.02]),
                                         yaw_deg=0.0)   # 绝对放不到的位置
    try:
        arm, st, motor = make_arm(params)
        motor.place_object(0.022)
        r = arm.grasp_and_place(graspable(params, 0.34, 0.04), place_id="default")
        assert r.status in (GraspStatus.PLAN_INVALID, GraspStatus.IK_FAILED), r
        assert r.stage == AC.STAGE_CHECK, r
        assert r.recovery_required is False
        assert motor.command_count == 0, "CHECK 拒绝前不应提交任何指令"
        assert arm._fault_latched is False
    finally:
        params.places["default"] = original


def test_入参校验失败不锁存公开原语(params):
    """P2-4：shape/限位/竖直性校验失败发生在臂未动时，不得要求 reset_fault。"""
    arm, st, motor = make_arm(params)
    with pytest.raises(Exception) as exc:
        arm.move_joints(np.array([1.0, 2.0, 3.0]))          # shape 错
    assert exc.value.status == "INVALID_INPUT"
    assert arm._fault_latched is False, "臂未动的入参错误不该锁存"

    bad = np.array(params.workspace.home_joints_deg, float)
    bad[0] = float(arm.limits.upper[0] + 10.0)
    with pytest.raises(Exception) as exc2:
        arm.move_joints(bad)
    assert exc2.value.status == "PLAN_INVALID"
    assert arm._fault_latched is False

    with pytest.raises(Exception) as exc3:
        arm.move_cartesian_top_down(np.array([0.10, 0.20, 0.02]), 0.0)   # 水平偏移
    assert arm._fault_latched is False
    assert motor.command_count == 0
    # 不 reset_fault，直接继续用：一次正常小运动应当成功
    q_ok = np.array(params.workspace.home_joints_deg, float) + np.array([5.0, 0, 0, 0, 0])
    arm.move_joints(q_ok)
    assert motor.command_count > 0


def test_段起点偏差时重新规划到同一目标(params):
    """P2-7：文档 6.3 末段——可重建段超差必须从实测起点重建、目标不变。"""
    q_plan = np.array(params.workspace.home_joints_deg, float)
    q_meas = q_plan + np.array([6.0, 0.0, 0.0, 0.0, 0.0])   # 超出 start_position_tol
    arm, st, motor = make_arm(params, joints=q_meas)
    target = q_plan + np.array([0.0, 8.0, 0.0, 0.0, 0.0])
    step = AC.PlanStep(
        name="guard", stage=AC.STAGE_APPROACH,
        build=lambda q_from: [make_joint_segment("guard", q_from, target, 50.0, params)],
    )
    stale = step.build(q_plan)[0]          # 以"规划起点"生成的段（已过期）
    played: list = []
    original_play = arm.executor.play_segment
    arm.executor.play_segment = lambda s: (played.append(s), original_play(s))[0]

    arm._play_with_offset_guard(stale, step)

    assert len(played) == 1, "偏差路径没有走重新规划分支"
    assert np.allclose(played[0].start_joints, q_meas, atol=1e-6), \
        "重新规划没有从实测起点出发"
    assert np.allclose(played[0].end_joints, target, atol=1e-6), "目标被改掉了"


def test_笛卡尔段起点偏差必须报错不改斜线(params):
    """P2-7：固定竖直线段无法从新起点保持同一直线，只能报错（文档 6.3）。"""
    q_plan = np.array(params.workspace.home_joints_deg, float)
    q_meas = q_plan + np.array([6.0, 0.0, 0.0, 0.0, 0.0])
    arm, st, motor = make_arm(params, joints=q_meas)
    g = 50.0
    step = AC.PlanStep(
        name="vert", stage=AC.STAGE_DESCEND, rebuildable=False,
        build=lambda q_from: [make_joint_segment("vert", q_from, q_from + 0.001,
                                                 g, params)],
    )
    stale = step.build(q_plan)[0]
    played: list = []
    original_play = arm.executor.play_segment
    arm.executor.play_segment = lambda s: (played.append(s), original_play(s))[0]
    with pytest.raises(AC.PlanViolation) as exc:
        arm._play_with_offset_guard(stale, step)
    assert "竖直" in str(exc.value) or "直线" in str(exc.value)
    assert played == [], "报错路径不得提交任何段"


def test_锁存期间拒绝新的动作调用(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.stuck_joints = True                     # 一动就必然失败
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.recovery_required is True
    with pytest.raises(Exception) as exc:
        arm.move_joints(np.array(params.workspace.home_joints_deg))
    assert exc.value.status == "CONFIG_INVALID"
    # 复位后恢复可用
    motor.stuck_joints = False
    arm.reset_fault()
    assert arm._fault_latched is False


def test_reset_fault在臂未停稳时拒绝解锁(params):
    """文档 6.4：reset_fault 只在确认静止后解锁。

    失败收尾时 hold_current 会把"实测位置"重新提交为保持目标，所以刚失败那一刻
    位置确实是跟得上的。要验证拒绝路径，必须模拟"保持提交之后又被人推动过"：
    这里保持目标不变、把实际位置掰开，解锁就必须被拒绝。
    """
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.stuck_joints = True
    arm.grasp_and_place(graspable(params, 0.34, 0.04))
    motor.stuck_joints = False
    assert arm._fault_latched is True
    held = arm.executor.state.last_submitted_deg
    assert held is not None, "失败收尾没有提交保持目标"
    # 保持目标之后被人推动：仍然 stuck，读回来才是"停在别处"而不是弹回保持目标。
    motor.q = np.asarray(held, float) + np.array([0.0, 40.0, 0.0, 0.0, 0.0])
    motor.stuck_joints = True
    with pytest.raises(Exception) as exc:
        arm.reset_fault()
    assert "复位被拒绝" in str(exc.value)
    assert arm._fault_latched is True, "被拒绝的复位不能顺手把锁存清掉"
    motor.stuck_joints = False
    motor.settle_instantly()
    arm.reset_fault()
    assert arm._fault_latched is False


# ---------------------------------------------------------------------------
# 3. 停止响应（必测项 6）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["early", "mid", "late"])
def test_should_stop在各动作中被响应(params, stage):
    """should_stop 按当前虚拟时间触发，停止后必须不再提交轨迹节点。"""
    thresholds = {"early": 0.1, "mid": 6.0, "late": 300.0}
    limit = thresholds[stage]
    armed = {"stopped": False}

    def should_stop():
        return armed["stopped"]

    arm, st, motor = make_arm(params, should_stop=should_stop)
    motor.place_object(0.022)
    original_submit = motor.send_action

    def spy(joints_deg, gripper_pct):
        original_submit(joints_deg, gripper_pct)
        if st.monotonic() >= limit:
            armed["stopped"] = True

    motor.send_action = spy
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.ABORTED, r
    assert r.stage in (AC.STAGE_FAULT_HOLD, AC.STAGE_FAULT_UNCONTROLLED)
    assert r.recovery_required is True
    # 停止之后再没有提交过节点
    n_after = motor.command_count
    st.sleep(1.0)
    assert motor.command_count == n_after, "返回后仍有后台任务在提交指令"


def test_规划检查点也能响应停止(params):
    """预规划阶段（臂还静止）触发 should_stop。"""
    fire = {"n": 0}

    def should_stop():
        fire["n"] += 1
        return fire["n"] > 30        # 前几次检查点放行，之后停止

    arm, st, motor = make_arm(params, should_stop=should_stop)
    motor.place_object(0.022)
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.ABORTED
    # 停止发生在规划期：没有提交任何轨迹节点。唯一允许的一次提交是文档 6.4 要求的
    # 收尾保持（hold_current 一读一写），它把当前位置原样写回去，不是运动指令。
    assert motor.command_count <= 1, motor.command_count
    assert np.allclose(motor.q, np.array(params.workspace.home_joints_deg), atol=1e-6)


def test_返回后没有隐藏执行任务(params):
    """文档 6.4：同步函数返回后不继续运行软件监控循环。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    arm.grasp_and_place(graspable(params, 0.34, 0.04))
    before_cmd, before_read = motor.command_count, motor.read_count
    # 多推几次虚拟时间：如果还有隐藏的循环在读写总线，计数会变。
    for _ in range(50):
        st.sleep(0.033)
    assert (motor.command_count, motor.read_count) == (before_cmd, before_read)


# ---------------------------------------------------------------------------
# 4. 故障注入（必测项 5）
# ---------------------------------------------------------------------------


def test_通信失败返回COMM_ERROR并报告未保持(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    calls = {"n": 0}
    original = motor.send_action

    def fail_after_a_bit(j, g):
        calls["n"] += 1
        if calls["n"] > 20:
            raise MotorCommunicationError("模拟串口断开")
        original(j, g)

    motor.send_action = fail_after_a_bit
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.COMM_ERROR
    assert r.recovery_required is True
    # hold_current 也走同一条坏链路 → 必须如实报告"保持未提交"
    assert r.stage == AC.STAGE_FAULT_UNCONTROLLED, r.stage


def test_通信正常时故障后提交一次保持(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    arm.executor.state.last_submitted_deg = np.array(params.workspace.home_joints_deg, float)
    # 直接触发一次到位失败，通信全程正常
    motor.stuck_joints = True
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status in (GraspStatus.TIMEOUT, GraspStatus.TRACKING_ERROR)
    assert r.stage == AC.STAGE_FAULT_HOLD


def test_关节过流立即停止(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.hard_overcurrent_joints = True
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TRACKING_ERROR
    assert "电流" in r.reason


def test_夹爪过流立即停止(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.gripper_overcurrent = True
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TRACKING_ERROR
    assert "夹爪电流" in r.reason


def test_空闭合端电流不算目标接触(params):
    """文档第七节：空闭合端的受阻电流不计为目标接触，应报 GRASP_MISS。"""
    arm, st, motor = make_arm(params)
    motor.blocked_current_ma = 2000.0     # 空夹到底时电流很大
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.GRASP_MISS
    assert r.holding is HoldingState.EMPTY


def test_闭爪超时返回GRASP_UNCERTAIN(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    # 让实际开度卡在接触区间之上：永远不会判接触，也不会走到空闭合端
    motor.lag_tau_s = 0.0
    motor.gripper_lag_tau_s = 0.0
    original = motor._advance

    def frozen(dt):
        # 夹爪完全不跟进，关节照常
        q_cmd, g_cmd = motor.q_cmd, motor.g_cmd
        original(dt)
        motor.g_cmd = g_cmd

    import qingyun.grabbing.trajectory as TR
    short = 0.05
    object.__setattr__(params.gripper, "close_timeout_s", short)
    try:
        r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
        assert r.status is GraspStatus.GRASP_UNCERTAIN, r
    finally:
        object.__setattr__(params.gripper, "close_timeout_s", 6.0)


def test_张开超时在OPEN阶段报错(params):
    arm, st, motor = make_arm(params)
    motor.gripper_lag_tau_s = 1e9        # 夹爪永远不跟进
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TIMEOUT
    assert "张开" in r.reason


def test_松爪失败返回PLACE_FAIL(params):
    """闭爪成功后再让夹爪卡住，RELEASE 阶段应报 PLACE_FAIL。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    r_ok = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r_ok.status is GraspStatus.SUCCESS
    arm.reset_fault() if arm._fault_latched else None
    # 第二次：走到 LOWER 之后再冻结夹爪
    motor.place_object(0.024)
    original = motor._advance
    stage_box = {"arm": arm}

    def freeze_after_lower(dt):
        if stage_box["arm"]._stage in (AC.STAGE_RELEASE,):
            return
        original(dt)

    motor._advance = freeze_after_lower
    r = arm.grasp_and_place(graspable(params, 0.36, 0.03))
    assert r.status is GraspStatus.PLACE_FAIL, r


def test_持物开度漂移返回GRIP_SLIP(params):
    """文档第七节：持物期间开度相对接触保持值变化超阈值 → GRIP_SLIP。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    original = motor._advance

    def slip_while_holding(dt):
        original(dt)
        # 只在闭爪完成之后开始滑：接触判据本身也看开度，提前滑会串到别的分支。
        motor.grip_slip_rate_pct_s = 8.0 if arm._stage in (
            AC.STAGE_LIFT, AC.STAGE_TRANSFER, AC.STAGE_LOWER) else 0.0

    motor._advance = slip_while_holding
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.GRIP_SLIP, r


def test_过期反馈被拒绝(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.expire_feedback = True
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TIMEOUT
    assert "反馈过期" in r.reason


def test_sequence不递增视为读到缓存(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    arm.executor.read_feedback()          # 先建立基准
    motor.freeze_feedback = True
    with pytest.raises(ExecutionError) as exc:
        arm.executor.read_feedback()
    assert exc.value.status == "COMM_ERROR"
    assert "sequence" in exc.value.reason


def test_CPU迟到超过阈值停止且不补发(params):
    """文档 6.1：迟到超过 max_tick_lateness_s 则停止。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    # 一次跨越阈值的 CPU 抖动：单个 sleep 片段的迟到量超过 max_tick_lateness_s。
    # 注意"轻微迟到"会被顺延吸收而不会累积（见下一条测试），所以这里必须给足量。
    st.extra_lateness_s = params.timing.max_tick_lateness_s * 3.0
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TIMEOUT
    assert "迟到" in r.reason
    n_at_stop = motor.command_count
    st.sleep(5.0)
    assert motor.command_count == n_at_stop, "迟到停止后还在补发积压节点"


def test_轻微迟到只顺延不报错(params):
    """迟到在阈值内时应顺延本段后续时刻并继续。"""
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    st.extra_lateness_s = params.timing.max_tick_lateness_s * 0.2
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.SUCCESS, r


def test_滞后过大导致跟踪误差停止(params):
    arm, st, motor = make_arm(params)
    motor.place_object(0.022)
    motor.lag_tau_s = 1.2         # 严重滞后，跟踪误差会超过持续阈值
    r = arm.grasp_and_place(graspable(params, 0.34, 0.04))
    assert r.status is GraspStatus.TRACKING_ERROR, r
    assert "跟踪误差" in r.reason


# ---------------------------------------------------------------------------
# 5. 执行器原语
# ---------------------------------------------------------------------------


def test_到位判断要求连续满足dwell(params):
    """文档 6.2 与第八节必测项 5："停住但未到目标"必须判失败。

    用 stuck_joints 把舵机钉在原地：速度读数确实是 0、也确实不再运动，唯一不满足
    的是位置误差，所以判据必须落在位置项上。
    """
    arm, st, motor = make_arm(params)
    target = np.array(params.workspace.home_joints_deg, float) + 8.0
    motor.send_action(target, 50.0)
    motor.settle_instantly()                # 先让内部基准对齐
    motor.q = np.array(params.workspace.home_joints_deg, float)
    motor.stuck_joints = True               # 之后完全不跟进
    assert arm.wait_settled(target, 0.5) is False
    # 真的跟上之后同样调用必须返回 True，证明失败不是因为超时而是位置判据
    motor.stuck_joints = False
    assert arm.wait_settled(target, 3.0) is True


def test_到位判断在跟上后返回True(params):
    arm, st, motor = make_arm(params)
    target = np.array(params.workspace.home_joints_deg, float)
    motor.send_action(target, 50.0)
    motor.settle_instantly()
    assert arm.wait_settled(target, 2.0) is True


def test_发送前二次检查步长(params):
    """执行器在发送前会再查一次实际步长，防止规划与提交不一致。"""
    arm, st, motor = make_arm(params)
    q0 = np.array(params.workspace.home_joints_deg, float)
    q1 = q0 + np.array([50.0, 0, 0, 0, 0])
    seg = MotionSegmentLike(params, q0, q1)
    with pytest.raises(ExecutionError) as exc:
        arm.executor.play_segment(seg)
    assert exc.value.status == "PLAN_INVALID"


class MotionSegmentLike:
    """一个"节点过稀"的假段：用来验证执行器的发送前复查，而不是规划器的检查。"""

    def __init__(self, params, q0, q1):
        tick = 1.0 / params.timing.fps
        self.name = "bad"
        self.kind = "JOINT"
        self.time_s = np.array([0.0, tick])
        self.joints_deg = np.vstack([q0, q1])
        self.gripper_pct = 50.0
        self.tcp_targets = None

    @property
    def nodes(self):
        return len(self.time_s)

    @property
    def start_joints(self):
        return self.joints_deg[0]

    @property
    def end_joints(self):
        return self.joints_deg[-1]


def test_保持提交恰好一读一写(params):
    arm, st, motor = make_arm(params)
    q = np.array(params.workspace.home_joints_deg, float)
    arm.executor.submit(q, 60.0)
    before = (motor.command_count, motor.read_count)
    fb = arm.executor.hold_position()
    assert fb is not None
    assert motor.command_count == before[0] + 1
    assert motor.read_count == before[1] + 1
    assert np.allclose(fb.angles_deg, q, atol=1e-6)


def test_保持失败时不谎报已静止(params):
    arm, st, motor = make_arm(params)
    q = np.array(params.workspace.home_joints_deg, float)
    arm.executor.submit(q, 60.0)
    motor.comm_fail_on_read = True
    assert arm.executor.hold_position() is None


def test_原语失败抛出带status的内部错误(params):
    """文档 4.1：原语抛带 GraspStatus 的 MotionError，由技能入口转成 GraspResult。"""
    arm, st, motor = make_arm(params)
    from qingyun.grabbing.kinematics_ext import MotionError

    with pytest.raises(MotionError) as exc:
        arm.move_cartesian_top_down(np.array([0.10, 0.20, 0.02]), 0.0)
    assert exc.value.status in {"PLAN_INVALID", "IK_FAILED", "COLLISION", "CONFIG_INVALID"}
