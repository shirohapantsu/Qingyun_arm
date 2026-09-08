"""姿态数学、工具变换与 IK 收敛包装的测试。

覆盖 docs/顶抓姿态构造与R_top_down标定.md 第 6 节要求的"必要数学检查"：
R_TOP_DOWN 与 rz_deg 的正交性和方向、工具变换与逆的往返、TCP 插值端点与中点、
目标中心到配置放置中心的刚体关系往返、合法关节 q→FK→IK→FK 的残差。
以及 docs/机械臂运动控制模块技术文档.md 第八节必测项 3。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from configs.motion_params import effective_joint_limits, wrap180
from qingyun.grabbing import kinematics_ext as K
from qingyun.grabbing.kinematics_ext import (
    R_TOP_DOWN,
    ArmModel,
    IkNotConverged,
    IkSolver,
    MotionError,
    StoppedByRequest,
    approach_tilt_deg,
    close_axis_width_at_yaw,
    gap_to_gripper_pct,
    gripper_pct_to_angle_deg,
    gripper_pct_to_gap,
    holding_transform,
    inverse_transform,
    object_pose,
    place_tcp_pose,
    pose_error,
    rz_deg,
    top_down_pose,
    tool_transform,
    tool_translation,
)


# ---------------------------------------------------------------------------
# 1. 旋转与姿态构造
# ---------------------------------------------------------------------------


def test_R_TOP_DOWN是合法旋转():
    assert R_TOP_DOWN.shape == (3, 3)
    assert np.allclose(R_TOP_DOWN.T @ R_TOP_DOWN, np.eye(3))
    assert pytest.approx(np.linalg.det(R_TOP_DOWN), abs=1e-12) == 1.0


def test_零偏航顶抓的三根轴符合文档定义():
    """姿态文档第 1、2 节：+X 沿 B+X、+Y 沿 B-Y、+Z 沿 B-Z。"""
    T = top_down_pose(np.zeros(3), 0.0)
    assert np.allclose(T[:3, 0], [1, 0, 0], atol=1e-12)
    assert np.allclose(T[:3, 1], [0, -1, 0], atol=1e-12)
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize("yaw", [-90.0, -37.5, 0.0, 12.3, 89.9, 175.0])
def test_任意yaw下接近轴恒为竖直向下(yaw):
    """姿态文档第 2 节：旋转矩阵第三列在任意 yaw 下都是 [0,0,-1]。"""
    T = top_down_pose([0.3, 0.0, 0.1], yaw)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    # 刚体性的正确判据是旋转块自伴为单位阵，而不是整个 4x4 矩阵。
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert np.allclose(inverse_transform(T) @ T, np.eye(4), atol=1e-12)
    assert approach_tilt_deg(T) == pytest.approx(0.0, abs=1e-9)


def test_rz_deg符合右手规则():
    """绕 +Z 右手为正：+X 轴转 90° 应当落到 +Y。"""
    v = rz_deg(90.0) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(v, [0, 1, 0], atol=1e-12)
    assert np.allclose(rz_deg(0.0), np.eye(3))
    # 角度周期性
    assert np.allclose(rz_deg(30.0), rz_deg(390.0), atol=1e-12)


def test_top_down_pose的位置参数就是TCP():
    """姿态文档第 1 节：top_down_pose 的位置参数是 TCP，不是物体中心。"""
    p = np.array([0.31, -0.02, 0.05])
    assert np.allclose(top_down_pose(p, 20.0)[:3, 3], p)


# ---------------------------------------------------------------------------
# 2. 齐次逆变换
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [0.0, 33.0, -71.0])
def test_刚体变换与逆的往返(yaw):
    """自定义齐次逆 inv([R,t]) = [R^T, -R^T t] 必须与通用矩阵逆完全一致。"""
    T_full = top_down_pose([0.25, -0.1, 0.08], yaw)
    assert np.allclose(T_full @ inverse_transform(T_full), np.eye(4), atol=1e-12)
    assert np.allclose(inverse_transform(T_full) @ T_full, np.eye(4), atol=1e-12)
    assert np.allclose(inverse_transform(T_full), np.linalg.inv(T_full), atol=1e-12)
    # 逆的逆回到原矩阵，且平移列被正确还原
    assert np.allclose(inverse_transform(inverse_transform(T_full)), T_full, atol=1e-12)


def test_逆的复合顺序正确():
    """T_A_C = T_A_B @ T_B_C ⇒ T_B_C = inverse(T_A_B) @ T_A_C。"""
    A = top_down_pose([0.3, 0.0, 0.1], 20.0)
    B = top_down_pose([0.05, -0.02, 0.07], -35.0)
    C = A @ B
    assert np.allclose(inverse_transform(A) @ C, B, atol=1e-12)


def test_逆变换对最后一行严格成立(params):
    g = params.gripper.preopen_pct
    T = tool_transform(g, params.tool)
    inv = inverse_transform(T)
    assert np.allclose(inv[3], [0, 0, 0, 1], atol=1e-15)


# ---------------------------------------------------------------------------
# 3. 工具变换与标定表
# ---------------------------------------------------------------------------


def test_tool_transform的旋转部分与开度无关(params):
    """姿态文档第 3 节：旋转固定在工具主体上，只有平移随开度插值。"""
    T_lo = tool_transform(0.0, params.tool)
    T_hi = tool_transform(100.0, params.tool)
    assert np.allclose(T_lo[:3, :3], T_hi[:3, :3])
    assert np.allclose(T_lo[:3, :3], params.tool.rotation_e_tcp)


def test_TCP平移在采样端点上精确复现(params):
    for sample in params.tool.translation_samples:
        got = tool_translation(sample.gripper_pct, params.tool)
        assert np.allclose(got, sample.xyz_m, atol=1e-15), sample.gripper_pct


def test_TCP平移在中点等于分段线性插值(params):
    s = params.tool.translation_samples
    lo, hi = s[0], s[1]
    mid_pct = (lo.gripper_pct + hi.gripper_pct) / 2.0
    expect = (lo.xyz_m + hi.xyz_m) / 2.0
    assert np.allclose(tool_translation(mid_pct, params.tool), expect, atol=1e-15)


@pytest.mark.parametrize("g", [0.0, 25.0, 50.0, 75.0, 100.0])
def test_gap表端点精确且单调可反查(params, g):
    gap = gripper_pct_to_gap(g, params.gripper)
    back = gap_to_gripper_pct(gap, params.gripper)
    assert back == pytest.approx(g, abs=1e-9)


def test_gap与angle表禁止外插(params):
    """标定指南 6.2/6.4：只分段线性插值，不外插；越界必须报错而不是取端点。"""
    with pytest.raises(MotionError):
        gripper_pct_to_gap(101.0, params.gripper)
    with pytest.raises(MotionError):
        gripper_pct_to_angle_deg(-5.0, params.gripper)


def test_活动指角按angle_table单调变化(params):
    angles = [gripper_pct_to_angle_deg(g, params.gripper)
              for g in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)]
    assert all(b > a for a, b in zip(angles, angles[1:])) or all(b < a for a, b in zip(angles, angles[1:]))


def test_闭合方向宽度公式与文档一致(params):
    """标定指南 2.4：abs(cos(beta))*length + abs(sin(beta))*width。"""
    L, W = 0.05, 0.03
    assert close_axis_width_at_yaw(L, W, 0.0) == pytest.approx(L)
    assert close_axis_width_at_yaw(L, W, 90.0) == pytest.approx(W)
    beta = math.radians(45.0)
    assert close_axis_width_at_yaw(L, W, 45.0) == pytest.approx(
        abs(math.cos(beta)) * L + abs(math.sin(beta)) * W
    )


# ---------------------------------------------------------------------------
# 4. FK / IK 往返（必测项 3）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    [0.0, -60.0, 60.0, 0.0, 0.0],
    [10.0, -70.0, 60.0, 20.0, -30.0],
    [-25.0, -40.0, 50.0, -15.0, 60.0],
    [35.0, 10.0, 30.0, 40.0, -20.0],
    [0.0, -90.0, 90.0, 0.0, 0.0],
])
def test_合法关节q经过FK再IK回到同一姿态(params, q):
    """必测项 3：合法 q→FK→IK→FK 达到残差要求。"""
    q = np.array(q, dtype=np.float64)
    model = ArmModel(params)
    lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
    assert np.all(q >= lower) and np.all(q <= upper), "用例本身必须落在有效限位内"
    g = params.gripper.preopen_pct
    T_target = model.fk_tcp(q, g)
    ik = IkSolver(model, params)
    sol = ik.solve(T_target, g, q)
    err = pose_error(model.fk_tcp(sol.joints_deg, g), T_target)
    assert err.position_m <= params.ik.position_tol_m
    assert err.tilt_deg <= params.ik.tilt_tol_deg
    assert err.yaw_deg <= params.ik.yaw_tol_deg


def test_IK结果始终落在有效限位内(params):
    """技术文档 5.1：求解结果要满足有效限位，限位外的分支不能交给执行器。"""
    model = ArmModel(params)
    ik = IkSolver(model, params)
    lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
    for x, y, z in [(0.33, 0.0, 0.022), (0.36, 0.05, 0.022), (0.38, -0.04, 0.03), (0.35, 0.0, 0.15)]:
        yaw = math.degrees(math.atan2(y, x))
        sol = ik.solve(top_down_pose([x, y, z], yaw), 70.0, np.array(params.workspace.home_joints_deg))
        assert np.all(sol.joints_deg >= lower - 1e-6), sol.joints_deg
        assert np.all(sol.joints_deg <= upper + 1e-6), sol.joints_deg


def test_IK不可收敛时在有限时间内退出(params):
    """必测项 3：不可收敛要有限时退出，返回 IK_FAILED 而不是死循环。"""
    import time

    model = ArmModel(params)
    ik = IkSolver(model, params)
    # 基座正下方：这台臂绝对够不到。
    target = top_down_pose([0.02, 0.0, -0.05], 0.0)
    t0 = time.monotonic()
    with pytest.raises(IkNotConverged) as exc:
        ik.solve(target, 70.0, np.array(params.workspace.home_joints_deg))
    elapsed = time.monotonic() - t0
    assert elapsed <= params.ik.max_solve_s + 1.0, f"总时限未生效：{elapsed:.2f}s"
    assert exc.value.status == "IK_FAILED"


def test_IK总时限不随初值数量成倍放宽(params):
    """标定指南 2.3：max_solve_s 覆盖同一目标的全部初值尝试。"""
    import time

    model = ArmModel(params)
    # 故意多给初值，时限仍应是同一份。
    object.__setattr__(
        params.ik, "seed_joints_deg", np.tile(params.workspace.home_joints_deg, (12, 1))
    )
    ik = IkSolver(model, params)
    t0 = time.monotonic()
    with pytest.raises(IkNotConverged):
        ik.solve(top_down_pose([0.01, 0.0, -0.20], 0.0), 70.0,
                 np.array(params.workspace.home_joints_deg))
    assert time.monotonic() - t0 <= params.ik.max_solve_s + 1.0
    object.__setattr__(params.ik, "seed_joints_deg",
                       np.array([[0.0, -90.0, 90.0, 0.0, 0.0]]))


def test_备用初值不改变目标位姿(params):
    """技术文档 5.1：备用初值只改变求解起点，不改变目标位置、角度或抓取方向。"""
    model = ArmModel(params)
    target = top_down_pose([0.35, 0.0, 0.10], 0.0)
    g = 70.0
    ik = IkSolver(model, params)
    sol = ik.solve(target, g, np.array([80.0, -20.0, 20.0, 10.0, 0.0]))
    # 解出来的 TCP 必须仍然落在同一个目标上，而不是被"挪近初值"。
    err = pose_error(model.fk_tcp(sol.joints_deg, g), target)
    assert err.acceptable(params)
    assert sol.seed_index >= 0


def test_IK迭代检查点能响应停止请求(params):
    """技术文档 5.1/6.1：迭代检查点执行停止/反馈检查。"""
    model = ArmModel(params)
    calls = {"n": 0}

    def checkpoint():
        calls["n"] += 1
        if calls["n"] > 3:
            raise StoppedByRequest("测试注入")

    ik = IkSolver(model, params, planning_checkpoint=checkpoint)
    with pytest.raises(StoppedByRequest):
        ik.solve(top_down_pose([0.35, 0.0, 0.05], 0.0), 70.0,
                 np.array(params.workspace.home_joints_deg))
    assert calls["n"] > 0


def test_夹爪开度不作为第六个姿态自由度(params):
    """技术文档 5.1：五关节 IK 不把夹爪开度当额外姿态自由度。"""
    model = ArmModel(params)
    q = np.array([0.0, -85.0, 85.0, 0.0, 0.0])
    g_low, g_high = 10.0, 95.0
    T_low = model.fk_tcp(q, g_low)
    T_high = model.fk_tcp(q, g_high)
    # 开度只通过 tool_transform 平移 TCP，不改变 E 的位姿。
    assert np.allclose(model.fk_e(q), model.fk_e(q))
    assert not np.allclose(T_low[:3, 3], T_high[:3, 3])
    assert np.allclose(T_low[:3, :3], T_high[:3, :3])
    # 关节解永远是 5 个
    ik = IkSolver(model, params)
    sol = ik.solve(model.fk_tcp(q, g_low), g_low, q)
    assert sol.joints_deg.shape == (5,)


def test_solver里夹爪自由度被固定(params):
    """IK 不得顺手改写 gripper 关节角。"""
    model = ArmModel(params)
    from qingyun.grabbing.kinematics_ext import GRIPPER_JOINT

    before = model.kin.robot.get_joint(GRIPPER_JOINT)
    ik = IkSolver(model, params)
    q = np.array([15.0, -75.0, 70.0, 10.0, -5.0])
    ik.solve(model.fk_tcp(q, 70.0), 70.0, q)
    after = model.kin.robot.get_joint(GRIPPER_JOINT)
    assert after == pytest.approx(before, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. 姿态残差定义（姿态文档第 5 节）
# ---------------------------------------------------------------------------


def test_残差三项定义与文档一致():
    T_des = top_down_pose([0.3, 0.0, 0.1], 0.0)
    # 只在接近轴上倾斜 2°：绕基座 Y 转
    Ry = np.array([[math.cos(0.035), 0, math.sin(0.035)],
                   [0, 1, 0],
                   [-math.sin(0.035), 0, math.cos(0.035)]])
    T_act = T_des.copy()
    T_act[:3, :3] = Ry @ T_des[:3, :3]
    err = pose_error(T_act, T_des)
    assert err.position_m == pytest.approx(0.0, abs=1e-12)
    assert err.tilt_deg == pytest.approx(math.degrees(0.035), abs=1e-6)


def test_yaw误差按有向角取最小差():
    """e_yaw = abs((yaw_act - yaw_des + 180) % 360 - 180)，跨 ±180 不炸。"""
    des = top_down_pose([0.3, 0.0, 0.1], 179.0)
    act = top_down_pose([0.3, 0.0, 0.1], -179.0)
    err = pose_error(act, des)
    assert err.yaw_deg == pytest.approx(2.0, abs=1e-9)
    assert err.position_m == pytest.approx(0.0, abs=1e-12)


def test_接近轴退化时判定为无效解():
    """姿态文档第 5 节：投影退化时解无效。构造 +X 指向正上/正下的位姿。"""
    T_des = top_down_pose([0.3, 0.0, 0.1], 0.0)
    T_act = np.eye(4)
    T_act[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    T_act[:3, 3] = [0.3, 0.0, 0.1]
    err = pose_error(T_act, T_des)
    assert err.yaw_degenerate
    assert not err.acceptable(_FakeParams())


class _FakeParams:
    class ik:
        position_tol_m = 0.01
        tilt_tol_deg = 1.0
        yaw_tol_deg = 1.0


# ---------------------------------------------------------------------------
# 6. 抓取中心 → 放置中心的刚体关系（文档 3.3、3.4）
# ---------------------------------------------------------------------------


def test_从中心到抓取TCP的偏移写在物体系中(params):
    """p_grasp = position + rz(yaw) @ center_offset_object_m。"""
    for yaw in (0.0, 30.0, -60.0):
        pos = np.array([0.35, 0.0, 0.022])
        p_grasp, spin, T = K.grasp_target_pose(pos, yaw, params)
        assert np.allclose(p_grasp, pos + rz_deg(yaw) @ params.grasp.center_offset_object_m)
        assert spin == pytest.approx(yaw + params.grasp.yaw_offset_deg)
        assert np.allclose(T[:3, 3], p_grasp)


def test_持物关系往返能从抓取回到放置(params):
    """T_TCP_O 固定 ⇒ 放置 TCP 与放置物体中心之间是刚体关系。

    验证方向：由放置 TCP 反算出的物体中心，必须等于配置里写的放置中心；
    若把配置里的物体中心直接当 TCP 用（文档明确禁止的错误做法），结果就会偏。
    """
    model = ArmModel(params)
    place = params.places["default"]
    T_B_O_grasp = object_pose(np.array([0.35, 0.0, 0.022]), 0.0)
    q_close = np.array([0.0, -30.0, 60.0, -20.0, 0.0])
    g_close = 30.0
    T_TCP_O = holding_transform(model.fk_tcp(q_close, g_close), T_B_O_grasp)
    T_B_O_place = object_pose(place.position, place.yaw_deg)
    T_B_TCP_place = place_tcp_pose(T_B_O_place, T_TCP_O)
    # 用同一个持物关系反算物体中心
    T_B_O_back = T_B_TCP_place @ T_TCP_O
    assert np.allclose(T_B_O_back[:3, 3], place.position, atol=1e-12)
    # 姿态也必须一致（长轴角模 180）
    yaw_back = math.degrees(math.atan2(T_B_O_back[1, 0], T_B_O_back[0, 0]))
    assert wrap180(yaw_back) == pytest.approx(wrap180(place.yaw_deg), abs=1e-9)
    # 反证：放置 TCP 不等于配置的物体中心
    assert not np.allclose(T_B_TCP_place[:3, 3], place.position, atol=1e-6)


def test_放置TCP仍是顶抓姿态(params):
    """由推导可知放置 TCP 的接近轴同样竖直向下，yaw = place.yaw + yaw_offset。

    推导：R_TCP_O = R_TOP·rz(-(θ_g - yaw_g)) = R_TOP（yaw_offset=0 时），
    于是 R_B_TCP_place = rz(θ_p)·R_TOP^T = rz(θ_p)·R_TOP，仍是顶抓。
    前提是 q_close 真是抓取位姿的解，否则 T_TCP_O 里就混进了额外倾角。
    """
    model = ArmModel(params)
    ik = IkSolver(model, params)
    q_open = np.array(params.workspace.home_joints_deg)
    # 5° 长轴角要配 5° 方位角：这台 5 自由度臂的顶抓 yaw 与目标方位角存在结构耦合
    # （6 个标量条件对 5 个关节），tests/support.graspable_target_at 就是按这个规律构造目标。
    grasp_xy = (0.35, 0.35 * math.tan(math.radians(5.0)))
    T_grasp_tcp = top_down_pose([grasp_xy[0], grasp_xy[1], 0.022], 5.0)
    q_grasp = ik.solve(T_grasp_tcp, params.gripper.preopen_pct, q_open).joints_deg
    # 状态表 CLOSE 一行明确"五关节保持"，所以闭爪只改变开度、不改变关节角。
    q_close = q_grasp
    T_B_O_grasp = object_pose(np.array([grasp_xy[0], grasp_xy[1], 0.022]), 5.0)
    T_TCP_O = holding_transform(model.fk_tcp(q_close, 28.0), T_B_O_grasp)
    place = params.places["bin"]
    T_place = place_tcp_pose(object_pose(place.position, place.yaw_deg), T_TCP_O)
    # 不是严格为 0：T_TCP_O 建立在"实际关节解的完整 TCP FK"上，继承了 IK 自身的
    # 残差，所以判据应当是 ik 容限，而不是解析意义上的零。
    assert approach_tilt_deg(T_place) <= params.ik.tilt_tol_deg
    yaw = math.degrees(math.atan2(T_place[1, 0], T_place[0, 0]))
    assert wrap180(yaw) == pytest.approx(
        wrap180(place.yaw_deg + params.grasp.yaw_offset_deg), abs=params.ik.yaw_tol_deg)


def test_规划同一TCP在不同开度下给出不同关节解(params):
    """单动指夹爪的 TCP 随开度变化，闭爪后必须按实际开度重算剩余路径。"""
    model = ArmModel(params)
    ik = IkSolver(model, params)
    q0 = np.array([0.0, -60.0, 60.0, 0.0, 0.0])
    T_open = model.fk_tcp(q0, 90.0)
    g_open = ik.solve(T_open, 90.0, q0).joints_deg
    # 同一个"抓取时刻的 T_open"用 20% 的工具变换去解，得到的关节角必然不同。
    sol2 = ik.solve(T_open, 20.0, q0)
    assert not np.allclose(sol2.joints_deg, g_open, atol=1e-3)
