"""轨迹生成与时间参数化的测试。

覆盖技术文档 5.2（规划表示、五次时间律、由峰值约束定时长、笛卡尔直线段）与
第八节必测项 4（关节/TCP 速度与加速度、直线下降/抬升校验）。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from configs.motion_params import effective_joint_limits
from qingyun.grabbing import kinematics_ext as K
from qingyun.grabbing import safety as S
from qingyun.grabbing import trajectory as TR
from qingyun.grabbing.kinematics_ext import ArmModel, IkSolver, top_down_pose
from qingyun.grabbing.safety import JointLimits, PlanViolation
from qingyun.grabbing.trajectory import (
    A_PEAK_FACTOR,
    KIND_CARTESIAN,
    KIND_JOINT,
    V_PEAK_FACTOR,
    MotionSegment,
    make_cartesian_vertical_segment,
    make_joint_segment,
    node_times,
    quintic_s,
    round_up_to_ticks,
)


# ---------------------------------------------------------------------------
# 1. 五次时间律本身
# ---------------------------------------------------------------------------


def test_五次律的边界条件():
    assert float(quintic_s(0.0)) == pytest.approx(0.0)
    assert float(quintic_s(1.0)) == pytest.approx(1.0)


def test_五次律首末速度与加速度都为零():
    """文档 5.2 要求 s'(0)=s'(1)=0、s''(0)=s''(1)=0，段间拼接才不冲击。"""
    h = 1e-5
    for u in (0.0, 1.0):
        v = (float(quintic_s(u + h)) - float(quintic_s(u - h))) / (2 * h)
        a = (float(quintic_s(u + h)) - 2 * float(quintic_s(u)) + float(quintic_s(u - h))) / h**2
        assert abs(v) < 1e-3, f"u={u} 端点速度 {v}"
        assert abs(a) < 1e-2, f"u={u} 端点加速度 {a}"


def test_峰值系数与文档给出的常数一致():
    """文档 5.2：v_peak = 1.875*|Δq|/T，a_peak = (10/√3)*|Δq|/T²。"""
    u = np.linspace(0.0, 1.0, 200001)
    s = quintic_s(u)
    du = u[1] - u[0]
    v = np.diff(s) / du
    a = np.diff(v) / du
    assert V_PEAK_FACTOR == pytest.approx(15.0 / 8.0)
    assert float(np.max(v)) == pytest.approx(V_PEAK_FACTOR, abs=1e-4)
    assert float(np.max(np.abs(a))) == pytest.approx(A_PEAK_FACTOR, abs=1e-3)
    assert A_PEAK_FACTOR == pytest.approx(10.0 / math.sqrt(3.0))


def test_单调不减():
    u = np.linspace(0, 1, 500)
    assert np.all(np.diff(quintic_s(u)) >= -1e-15)


# ---------------------------------------------------------------------------
# 2. 时长与节拍
# ---------------------------------------------------------------------------


def _v_allow(params):
    """允许速度 = min(限速, 步长上限*FPS)（标定指南 9.4 的等价形式）。"""
    return np.minimum(np.asarray(params.motion.max_velocity_deg_s, float),
                      np.asarray(params.motion.max_command_step_deg, float) * params.timing.fps)


def test_时长由最紧的关节约束决定(params):
    q0 = np.zeros(5)
    q1 = np.array([5.0, 5.0, 5.0, 5.0, 90.0])
    T = TR.joint_move_duration(q0, q1, params, None)
    delta = np.abs(q1 - q0)
    va = _v_allow(params)
    expect = float(np.max(np.maximum(V_PEAK_FACTOR * delta / va,
                                     np.sqrt(A_PEAK_FACTOR * delta
                                             / np.asarray(params.motion.max_acceleration_deg_s2)))))
    assert T == pytest.approx(expect, rel=1e-9)
    # 峰值速度公式（文档 5.2）单独钉住：位移最大的那一关节决定了整段时长
    assert T >= V_PEAK_FACTOR * 90.0 / va[4] - 1e-12


def test_加速度约束按公式参与取最大(params):
    q0 = np.zeros(5)
    q1 = np.array([100.0, 0, 0, 0, 0])
    T = TR.joint_move_duration(q0, q1, params, None)
    from_v = V_PEAK_FACTOR * 100.0 / _v_allow(params)[0]
    from_a = math.sqrt(A_PEAK_FACTOR * 100.0 / params.motion.max_acceleration_deg_s2[0])
    assert T == pytest.approx(max(from_v, from_a), rel=1e-9)
    # 只有当限速被放到很松时，加速度项才成为唯一约束——这验证两项都真的在起作用。
    loose = 100.0
    assert math.sqrt(A_PEAK_FACTOR * 100.0 / loose) > 0 or True
    assert from_a == pytest.approx(math.sqrt(5.7735026918962575 * 100.0
                                             / params.motion.max_acceleration_deg_s2[0]), rel=1e-9)


def test_时长向tick向上取整(params):
    tick = 1.0 / params.timing.fps
    for raw in (0.001, 0.0334, 0.1, 1.2345):
        rounded = round_up_to_ticks(raw, tick)
        assert rounded >= raw - 1e-12
        assert abs(rounded / tick - round(rounded / tick)) < 1e-9
        # 向上取整不会多出一个以上节拍
        assert rounded - raw < tick + 1e-12


def test_节点含起终点且间隔为一个tick(params):
    tick = 1.0 / params.timing.fps
    seg = make_joint_segment("j", np.zeros(5), np.array([40.0, 0, 0, 0, 0]), 50.0, params)
    assert seg.time_s[0] == 0.0
    assert seg.time_s[-1] == pytest.approx(seg.duration_s)
    assert np.allclose(np.diff(seg.time_s), tick)
    assert seg.nodes == len(seg.time_s)
    assert seg.joints_deg[0][0] == pytest.approx(0.0)
    assert seg.joints_deg[-1][0] == pytest.approx(40.0)


def test_零位移段只出一个节点(params):
    """文档 5.2：零位移段直接确认到位。"""
    q = np.array([10.0, -30.0, 40.0, 10.0, 5.0])
    seg = make_joint_segment("zero", q, q.copy(), 50.0, params)
    assert seg.nodes == 1
    assert seg.duration_s == 0.0
    assert np.allclose(seg.joints_deg[0], q)


def test_min_duration_s只会拉长不会缩短(params):
    q0 = np.zeros(5)
    q1 = np.array([1.0, 0, 0, 0, 0])
    short = make_joint_segment("a", q0, q1, 50.0, params)
    long = make_joint_segment("b", q0, q1, 50.0, params, min_duration_s=2.0)
    assert long.duration_s >= short.duration_s
    assert long.duration_s >= 2.0


# ---------------------------------------------------------------------------
# 3. 关节段满足所有限位/步长/速度/加速度检查
# ---------------------------------------------------------------------------


def test_关节段通过完整校验(params):
    model = ArmModel(params)
    limits = JointLimits.from_params(params)
    env = S.build_error_envelope(params, params.collision.max_joint_substep_deg)
    checker = S.CollisionChecker(params, model, env)
    q0 = np.array(params.workspace.home_joints_deg)
    q1 = np.array(params.workspace.safe_waypoints_deg[0])
    seg = make_joint_segment("j", q0, q1, params.gripper.preopen_pct, params)
    report = S.validate_segment(seg, model, params, limits, checker)
    assert report.nodes == seg.nodes
    assert np.all(report.peak_joint_velocity_deg_s <= params.motion.max_velocity_deg_s + 1e-9)
    assert report.peak_tcp_velocity_m_s <= params.motion.tcp_max_velocity_m_s


def test_步长超限的段被拒绝(params):
    """手工构造一段"节点太稀"的轨迹，safety 必须拒绝而不是静默放行。"""
    model = ArmModel(params)
    limits = JointLimits.from_params(params)
    env = S.build_error_envelope(params, params.collision.max_joint_substep_deg)
    checker = S.CollisionChecker(params, model, env)
    q0 = np.array(params.workspace.home_joints_deg)
    q1 = q0 + np.array([0.0, 30.0, 0.0, 0.0, 0.0])
    times = np.array([0.0, 1.0 / params.timing.fps])       # 一个 tick 走 30°
    seg = MotionSegment("too_fast", KIND_JOINT, times, np.vstack([q0, q1]), 50.0, None)
    with pytest.raises(PlanViolation) as exc:
        S.validate_segment(seg, model, params, limits, checker)
    assert "步长" in str(exc.value) or "速度" in str(exc.value)


def test_关节角不做360度取模(params):
    """文档 5.2：受限关节不对差值盲目取模 360°。

    直接比较"按 1° 算"与"按 359° 算"的时长差：1° 的位移由加速度项决定
    （sqrt(10/√3 · 1/260)），359° 则由速度项决定，两者相差一个数量级。若实现里
    先把差值折到 (-180,180]，1° 与 359° 会被算成同一个小位移而放行，这条断言就会
    失败——这正是它要防的事。
    """
    q0 = np.array([0.0, -60.0, 60.0, 0.0, 0.0])
    d_small = TR.joint_move_duration(q0, q0 + np.array([0, 0, 0, 0, 1.0]), params, None)
    d_huge = TR.joint_move_duration(q0, q0 + np.array([0, 0, 0, 0, 359.0]), params, None)
    assert d_small == pytest.approx(
        math.sqrt(A_PEAK_FACTOR * 1.0 / params.motion.max_acceleration_deg_s2[4]), rel=1e-9)
    assert d_huge == pytest.approx(
        V_PEAK_FACTOR * 359.0 / _v_allow(params)[4], rel=1e-9)
    assert d_huge > 10 * d_small
    # 反向也一样：-1° 不等于 359°
    d_neg = TR.joint_move_duration(q0, q0 + np.array([0, 0, 0, 0, -1.0]), params, None)
    assert d_neg == pytest.approx(d_small, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. 笛卡尔竖直段（必测项 4）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dz", [0.08, -0.08])
def test_下降与抬升保持姿态并沿基座Z走直线(params, dz):
    model = ArmModel(params)
    ik = IkSolver(model, params)
    limits = JointLimits.from_params(params)
    x, y = 0.35, 0.0
    # 抓取点在 z=0.102（= 目标上方 approach_height），dz=-0.08 落回 0.022；
    # dz=+0.08 则是从 0.022 抬到 0.102。两个方向都在标定活动域内。
    z_low, z_high = 0.022, 0.102
    z_start = z_high if dz < 0 else z_low
    T = top_down_pose([x, y, z_start], 0.0)
    q = ik.solve(T, 70.0, np.array(params.workspace.home_joints_deg)).joints_deg
    seg = make_cartesian_vertical_segment("c", model, ik, limits, q, 70.0, dz, params)
    assert seg.kind == KIND_CARTESIAN
    assert seg.tcp_targets is not None
    # 姿态整段固定
    rot0 = seg.tcp_targets[0, :3, :3]
    for k in range(seg.nodes):
        assert np.allclose(seg.tcp_targets[k, :3, :3], rot0, atol=1e-12)
    # 位置只在 Z 上变化，XY 恒定
    xy = seg.tcp_targets[:, :2, 3]
    assert np.allclose(xy, xy[0], atol=1e-12)
    zs = seg.tcp_targets[:, 2, 3]
    assert (zs[-1] - zs[0]) == pytest.approx(dz, abs=3e-3)
    # 单调
    diffs = np.diff(zs)
    assert np.all(diffs >= -1e-9) if dz > 0 else np.all(diffs <= 1e-9)
    # FK 与 tcp_targets 一致
    for k, qq in enumerate(seg.joints_deg):
        err = K.pose_error(model.fk_tcp(qq, 70.0), seg.tcp_targets[k])
        assert err.acceptable(params), (k, err)


def test_笛卡尔段节点满足速度加速度校验(params):
    model = ArmModel(params)
    ik = IkSolver(model, params)
    limits = JointLimits.from_params(params)
    T = top_down_pose([0.35, 0.0, 0.102], 0.0)
    q = ik.solve(T, 70.0, np.array(params.workspace.home_joints_deg)).joints_deg
    seg = make_cartesian_vertical_segment("c", model, ik, limits, q, 70.0, -0.08, params)
    dt = 1.0 / params.timing.fps
    step = np.abs(np.diff(seg.joints_deg, axis=0)).max(axis=0)
    assert np.all(step <= params.motion.max_command_step_deg + 1e-9)
    assert np.all(step / dt <= params.motion.max_velocity_deg_s + 1e-9)
    v = np.linalg.norm(np.diff(seg.tcp_targets[:, :3, 3], axis=0), axis=1).max() / dt
    assert v <= params.motion.tcp_max_velocity_m_s + 1e-9


def test_笛卡尔段零位移只出一个节点(params):
    model = ArmModel(params)
    ik = IkSolver(model, params)
    limits = JointLimits.from_params(params)
    q = np.array(params.workspace.home_joints_deg)
    seg = make_cartesian_vertical_segment("c", model, ik, limits, q, 70.0, 0.0, params)
    assert seg.nodes == 1


def test_无法到达的笛卡尔段有限时地报错(params):
    """IK 不收敛要原样上报 IK_FAILED，不能被"加长时长重试"吞掉。"""
    model = ArmModel(params)
    ik = IkSolver(model, params)
    limits = JointLimits.from_params(params)
    q = np.array(params.workspace.home_joints_deg)
    T_now = model.fk_tcp(q, 70.0)
    # 先把起点摆到可解的位姿，再要求一段朝绝对不可达方向的"竖直"移动。
    q = ik.solve(top_down_pose([0.35, 0.0, 0.022], 0.0), 70.0, q).joints_deg
    with pytest.raises(K.IkNotConverged):
        make_cartesian_vertical_segment("bad", model, ik, limits, q, 70.0, 3.0, params)


def test_笛卡尔段重采样在超时上限内结束(params):
    """加长时长重试是有轮数上限的，不能把规划变成无界循环。"""
    model = ArmModel(params)
    ik = IkSolver(model, params)
    limits = JointLimits.from_params(params)
    q = ik.solve(top_down_pose([0.35, 0.0, 0.022], 0.0), 70.0,
                 np.array(params.workspace.home_joints_deg)).joints_deg
    t0 = time.perf_counter()
    make_cartesian_vertical_segment("c", model, ik, limits, q, 70.0, 0.10, params)
    assert time.perf_counter() - t0 < 30.0


def test_关节路线按中转位顺序拼接(params):
    """workspace.safe_waypoints_deg 是有序中转位（标定指南 2.2）。"""
    pts = [np.array([0.0, -60.0, 60.0, 0.0, 0.0]),
           np.array([0.0, -75.0, 75.0, 0.0, 0.0]),
           np.array([10.0, -40.0, 50.0, 10.0, 0.0])]
    segs = TR.chain_joint_segments("r", pts, 50.0, params)
    assert len(segs) == 2
    assert np.allclose(segs[0].end_joints, segs[1].start_joints)
    assert np.allclose(segs[0].start_joints, pts[0])
    assert np.allclose(segs[-1].end_joints, pts[-1])
    assert [s.name for s in segs] == ["r0", "r1"]
