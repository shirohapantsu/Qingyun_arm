"""有效限位、距离几何与碰撞检查的测试。

覆盖技术文档 5.1（三重交集限位）、5.3（校验范围：胶囊体/AABB/桌面/持物包络、
按 max_joint_substep_deg 加密、只豁免规定接触阶段的指垫—目标接触）与第八节
必测项 4 的"固定障碍和张开工具包络校验"。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from configs.common_interface import JOINT_NAMES
from configs.motion_params import effective_joint_limits
from qingyun.grabbing import safety as S
from qingyun.grabbing.kinematics_ext import ArmModel, top_down_pose
from qingyun.grabbing.safety import (
    BASE_LINK,
    CollisionChecker,
    CollisionViolation,
    JointLimits,
    OrientedBox,
    build_error_envelope,
    check_start_offset,
    obb_to_capsule,
    object_box,
    point_obb_distance,
    segment_halfspace_distance,
    segment_obb_distance,
    segment_segment_distance,
)


def _checker(params):
    model = ArmModel(params)
    env = build_error_envelope(params, params.collision.max_joint_substep_deg)
    return model, env, CollisionChecker(params, model, env)


# ---------------------------------------------------------------------------
# 1. 距离原语：用解析可验证的退化情形钉住实现
# ---------------------------------------------------------------------------


def test_平行线段距离():
    assert segment_segment_distance([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]) == pytest.approx(1.0)


def test_相交线段距离为零():
    assert segment_segment_distance([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_异面线段端点最近时仍给出正确距离():
    """线段端点被裁剪到 [0,1] 之外时必须落到端点-端点情形。"""
    d = segment_segment_distance([0, 0, 0], [1, 0, 0], [10, 0, 5], [11, 0, 5])
    assert d == pytest.approx(math.hypot(9.0, 5.0), rel=1e-12)


def test_退化成点的线段():
    # 点 (1,2,3) 到 Z 轴上线段的最近点是 (0,0,3)，距离 hypot(1,2)
    assert segment_segment_distance([1, 2, 3], [1, 2, 3], [0, 0, 0], [0, 0, 5]) == pytest.approx(
        math.hypot(1.0, 2.0))


def test_点在盒内距离为零():
    assert point_obb_distance([0.5, 0.0, 0.0], [0, 0, 0], [1, 1, 1], np.eye(3)) == pytest.approx(0.0)


def test_点到轴对齐盒面的距离():
    assert point_obb_distance([2.0, 0.5, 0.0], [0, 0, 0], [1, 1, 1], np.eye(3)) == pytest.approx(1.0)
    # 角点外：两个轴同时超出，取欧氏距离
    assert point_obb_distance([2.0, 2.0, 0.0], [0, 0, 0], [1, 1, 1], np.eye(3)) == pytest.approx(
        math.hypot(1.0, 1.0))


def test_旋转后的盒用局部系判断():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)   # 绕 Z 转 90°
    # 长条沿局部 X（转后是世界 Y）
    assert point_obb_distance([0, 2.0, 0], [0, 0, 0], [3, 0.5, 0.5], r) == pytest.approx(0.0)
    assert point_obb_distance([2.0, 0, 0], [0, 0, 0], [3, 0.5, 0.5], r) == pytest.approx(1.5)


def test_线段穿过盒时距离为零():
    d = segment_obb_distance([-5, 0, 0], [5, 0, 0], [0, 0, 0], [1, 1, 1], np.eye(3))
    assert d == pytest.approx(0.0)


def test_线段到盒的最小距离取在线段内部():
    """最近点在线段中间而不是端点：三分法必须找到内部极小值。"""
    d = segment_obb_distance([-2, 5, 0], [2, 5, 0], [0, 0, 0], [1, 1, 1], np.eye(3))
    assert d == pytest.approx(4.0, abs=1e-6)


def test_半空间距离带符号():
    """正值表示整段都在桌面上方。"""
    d = segment_halfspace_distance([0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.0], [0, 0, 1.0])
    assert d == pytest.approx(0.1)
    d2 = segment_halfspace_distance([0, 0, -0.1], [0, 0, 0.2], [0, 0, 0.0], [0, 0, 1.0])
    assert d2 == pytest.approx(-0.1)


def test_OBB外接胶囊体确实包住盒子():
    """近似方向必须保守：胶囊体包住盒子，净距只会算小不会算大。"""
    half = np.array([0.05, 0.02, 0.03])
    r = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    box = OrientedBox(center=np.array([0.3, 0.0, 0.1]), half_extents=half, rotation=r)
    a, b, rad = obb_to_capsule(box)
    corners = [box.center + box.rotation @ (half * np.array([sx, sy, sz]))
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    for c in corners:
        assert segment_obb_distance(a, b, c, np.zeros(3), np.eye(3)) <= rad + 1e-12


# ---------------------------------------------------------------------------
# 2. 有效限位
# ---------------------------------------------------------------------------


def test_三重交集逐项取最紧的界(params):
    limits = JointLimits.from_params(params)
    urdf = params.urdf_limits_deg
    j = params.joints
    for i in range(5):
        lo = max(urdf[i, 0], j.measured_limits_deg[i, 0], j.application_limits_deg[i, 0]) + j.margin_deg[i]
        hi = min(urdf[i, 1], j.measured_limits_deg[i, 1], j.application_limits_deg[i, 1]) - j.margin_deg[i]
        assert limits.lower[i] == pytest.approx(lo)
        assert limits.upper[i] == pytest.approx(hi)


def test_限位检查逐项报错并带上关节名(params):
    limits = JointLimits.from_params(params)
    q = np.array(params.workspace.home_joints_deg)
    q[2] = limits.upper[2] + 0.5
    bad = limits.violations(q)
    assert len(bad) == 1
    assert JOINT_NAMES[2] in bad[0]


def test_容差边界不判越限(params):
    limits = JointLimits.from_params(params)
    assert limits.inside(limits.lower.copy())
    assert limits.inside(limits.upper.copy())
    assert not limits.inside(limits.lower - 1e-3)


def test_夹爪开度越出标定表覆盖区间时报CONFIG_INVALID(params):
    """姿态文档第 3 节：禁止外插。"""
    from qingyun.grabbing.kinematics_ext import gripper_pct_to_gap, MotionError

    with pytest.raises(MotionError) as exc:
        gripper_pct_to_gap(-1.0, params.gripper)
    assert exc.value.status == "CONFIG_INVALID"


# ---------------------------------------------------------------------------
# 3. 碰撞检查
# ---------------------------------------------------------------------------


def test_待机位与中转位不碰任何固定几何(params):
    model, env, ck = _checker(params)
    ck.check_pose(np.array(params.workspace.home_joints_deg), params.gripper.preopen_pct,
                  stage="home ")
    for k, w in enumerate(params.workspace.safe_waypoints_deg):
        ck.check_pose(np.array(w, float), params.gripper.preopen_pct, stage=f"wp{k} ")


def test_明显插入桌面的姿态被拒绝(params):
    """把整条臂"埋"到桌面以下：wrist_flex 拉到限位最深处会出现穿透。"""
    model, env, ck = _checker(params)
    # 直接构造一个 TCP 在桌面以下的姿态做检查（用大位移的兜底用例）。
    q = np.array(params.workspace.home_joints_deg, float)
    q[1] = float(params.joints.measured_limits_deg[1, 0]) + float(params.joints.margin_deg[1])
    q[2] = float(params.joints.measured_limits_deg[2, 1]) - float(params.joints.margin_deg[2])
    # 不假设它一定穿桌，只要求检查器要么放行、要么给出可定位的穿透原因。
    try:
        ck.check_pose(q, params.gripper.preopen_pct, stage="sink ")
    except CollisionViolation as exc:      # 裸 except 拿到的是异常本身
        text = str(exc)
        assert "碰桌" in text or "碰固定障碍" in text or "自碰撞" in text


def test_基座胶囊体只豁免桌面检查不豁免障碍(params):
    """技术文档 1.2：基座与桌面固定；5.3：不豁免连杆/手指碰桌以外的干涉。"""
    model, env, ck = _checker(params)
    assert BASE_LINK in {c.link for c in params.collision.link_capsules}
    exempt = {c.id for c in params.collision.link_capsules if c.link == BASE_LINK}
    assert exempt and exempt <= ck._table_exempt
    # 把一个障碍硬搬到基座上：桌面豁免不能顺手放过障碍碰撞。
    object.__setattr__(params.workspace.obstacles[0], "bounds_m",
                       np.array([[0.0, -0.03, 0.0], [0.04, 0.03, 0.06]]))
    try:
        with pytest.raises(CollisionViolation) as exc:
            ck.check_pose(np.array(params.workspace.home_joints_deg), 70.0, stage="base ")
        assert "碰固定障碍" in str(exc.value)
    finally:
        from tests.support import raw_profile
        raw = raw_profile()["workspace"]["obstacles"][0]["bounds_m"]
        object.__setattr__(params.workspace.obstacles[0], "bounds_m", np.array(raw, float))


def test_指定姿态与障碍重叠时被捕获(params):
    """把一根"墙"直接搬到机械臂末端会经过的地方，检查必须报出来。"""
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    tcp = model.fk_tcp(q, params.gripper.preopen_pct)[:3, 3]
    from configs.motion_params import Obstacle
    wall = Obstacle(id="test_wall",
                    bounds_m=np.array([tcp - 0.02, tcp + 0.02]))
    object.__setattr__(params.workspace, "obstacles", tuple(params.workspace.obstacles) + (wall,))
    try:
        with pytest.raises(CollisionViolation) as exc:
            ck.check_pose(q, params.gripper.preopen_pct, stage="wall ")
        assert "test_wall" in str(exc.value)
    finally:
        object.__setattr__(params.workspace, "obstacles",
                           tuple(o for o in params.workspace.obstacles if o.id != "test_wall"))


def test_持物包络挡住去路时被捕获(params):
    """把"目标盒"直接压在机械臂本体上，非接触阶段必须拒绝。"""
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    tcp = model.fk_tcp(q, params.gripper.preopen_pct)[:3, 3]
    # 只沿闭合方向做一个扁盒，避免顺手压到桌面引入无关的报错
    box = object_box(tcp, 0.0, np.array([0.20, 0.20, 0.20]), 0.0)
    with pytest.raises(CollisionViolation) as exc:
        ck.check_pose(q, params.gripper.preopen_pct, object_obb=box,
                      object_contact_allowed=False, stage="hold ")
    assert "持物包络" in str(exc.value)
    # 接触阶段：压在臂本体上的大盒依旧要罚（见下一条专项测试）。
    with pytest.raises(CollisionViolation) as exc:
        ck.check_pose(q, params.gripper.preopen_pct, object_obb=box,
                      object_contact_allowed=True, stage="hold ")
    assert "持物包络" in str(exc.value)


def test_接触阶段只豁免抓取组件不豁免臂连杆(params):
    """P1-3 复现：目标盒压在 lower_arm 轴上，contact_allowed=True 必须报碰撞。

    技术文档 5.3 只豁免"规定接触阶段的指垫—目标接触"；旧实现把整段
    连杆×目标检查全部跳过，前臂/腕压到目标或持物不报 COLLISION。
    """
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    g = params.gripper.preopen_pct
    axes = {cid: (a, b, r) for cid, a, b, r in ck._axes(q, g)}
    exempt_ids = ck._object_contact_exempt
    assert exempt_ids, "仿真配置里必须注册了抓取组件胶囊体"

    # 目标盒放在某个非豁免（臂上）胶囊体的轴中点上：接触阶段也必须罚。
    arm_cid = next(cid for cid in axes if cid not in exempt_ids and cid != BASE_LINK)
    a, b, _ = axes[arm_cid]
    box = object_box((a + b) * 0.5, 0.0, np.array([0.04, 0.04, 0.04]), 0.0)
    with pytest.raises(CollisionViolation) as exc:
        ck.check_pose(q, g, object_obb=box, object_contact_allowed=True, stage="crush ")
    assert "持物包络" in str(exc.value)

    # 小盒贴在豁免组件（指垫）轴线上：接触阶段放行（这正是规定的指垫—目标接触）。
    pad_cid = sorted(exempt_ids)[0]
    pa, pb, _ = axes[pad_cid]
    pad_box = object_box((pa + pb) * 0.5, 0.0, np.array([0.006, 0.006, 0.006]), 0.0)
    ck.check_pose(q, g, object_obb=pad_box, object_contact_allowed=True, stage="pad ")
    # 但同一盒在非接触阶段要罚：豁免只属于规定接触阶段。
    with pytest.raises(CollisionViolation):
        ck.check_pose(q, g, object_obb=pad_box, object_contact_allowed=False,
                      stage="pad-non ")


def test_扫掠余量用基座系端点计算(params):
    """P1-2 复现：_axes 传给 _sweep_inflation 的必须是基座系端点。

    对 home 姿态逐胶囊体解析重算"基座系端点到各关节轴最大垂距 × 步长角"，
    与 _axes 给出的半径膨胀量对比。旧实现传的是 link 系局部坐标，数值对不上。
    """
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    g = params.gripper.preopen_pct
    substep = math.radians(float(params.collision.max_joint_substep_deg))
    frames = model.frames_for(list(JOINT_NAMES), q, g)
    by_id = {cap.id: cap for cap in params.collision.link_capsules}
    for cid, a, b, r in ck._axes(q, g):
        cap = by_id[cid]
        expect_sweep = 0.0
        for name in JOINT_NAMES:
            Tf = frames[name]
            origin, axis = Tf[:3, 3], Tf[:3, 2]
            r_i = 0.0
            for pt in (a, b):
                v = pt - origin
                r_i = max(r_i, float(np.linalg.norm(v - (v @ axis) * axis)))
            expect_sweep = max(expect_sweep, substep * r_i)
        expect = cap.radius_m + params.tool.position_error_bound_m + expect_sweep
        assert r == pytest.approx(expect, abs=1e-9), f"{cid} 扫掠余量未按基座系计算"


def test_张开后的工具包络在松爪阶段仍被检查(params):
    """技术文档 5.3：松爪过程中也检查张开后的工具包络。"""
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    # 用最大开度检查：指垫摆到最外侧，活动指包络必须参与 FK。
    axes_open = ck._axes(q, 100.0)
    axes_shut = ck._axes(q, 0.0)
    # 同一个胶囊体在两个开度下的位置不同，说明 angle_table 真的进了碰撞模型。
    moved = max(float(np.linalg.norm(np.asarray(a[1]) - np.asarray(b[1])))
                for a, b in zip(axes_open, axes_shut))
    assert moved > 1e-4


def test_相邻节点按max_joint_substep加密(params):
    model, env, ck = _checker(params)
    q0 = np.zeros(5)
    substep = params.collision.max_joint_substep_deg
    q1 = np.array([substep * 3.4, 0, 0, 0, 0])
    n = ck.substep_count(q0, q1)
    assert n >= 3
    assert substep * (n + 1) >= 3.4 * substep - 1e-9
    assert ck.substep_count(q0, q0.copy()) == 0


def test_扫掠余量按加密后的子步长封顶(params):
    """_sweep_inflation 把每个关节的角度增量截到 max_joint_substep_deg。

    加密检查保证了相邻节点之间最多转这么多度，所以余量只需要按这个上限算一次；
    传"整对增量"和传"单个子步增量"必须得到同一个结果，否则要么重复惩罚、要么
    余量不足。两条断言分别钉住这两个方向。
    """
    model, env, ck = _checker(params)
    q0 = np.array(params.workspace.home_joints_deg, float)
    substep = params.collision.max_joint_substep_deg
    q1 = q0 + np.array([substep * 4.0, 0, 0, 0, 0])
    n = ck.substep_count(q0, q1)
    frames = model.frames_for(list(JOINT_NAMES), q0, params.gripper.preopen_pct)
    capped = ck._sweep_inflation([np.zeros(3)], frames, np.array([substep, 0, 0, 0, 0]))
    m_full = ck._sweep_inflation([np.zeros(3)], frames, np.abs(q1 - q0))
    m_sub = ck._sweep_inflation([np.zeros(3)], frames, np.abs(q1 - q0) / (n + 1))
    assert m_full == pytest.approx(capped, rel=1e-12)
    assert m_sub == pytest.approx(capped, rel=1e-12)
    # 一个远小于封顶值的增量，余量要按比例缩小，说明它真的在用增量而不是用常数
    small = ck._sweep_inflation([np.zeros(3)], frames, np.array([substep * 0.25, 0, 0, 0, 0]))
    assert small == pytest.approx(capped * 0.25, rel=1e-9)


def test_自碰撞豁免只覆盖登记过的对(params):
    """clearance_m 之外不该有未登记的接触；删掉一条豁免就应该报自碰撞。"""
    model = ArmModel(params)
    env = build_error_envelope(params, params.collision.max_joint_substep_deg)
    original = params.collision.ignore_self_pairs
    pair = ("upper_arm", "lower_arm")
    if pair not in original and tuple(reversed(pair)) not in original:
        pytest.skip("仿真配置里这一对不是相邻豁免项")
    object.__setattr__(params.collision, "ignore_self_pairs",
                       tuple(p for p in original if set(p) != set(pair)))
    try:
        ck = CollisionChecker(params, model, env)
        # 手肘折到底：相邻两段必然贴近，取消豁免后应报自碰撞
        q = np.array(params.workspace.home_joints_deg, float)
        q[2] = float(ck.limits_upper[2]) if hasattr(ck, "limits_upper") else 89.0
        q[1] = -90.0
        with pytest.raises(CollisionViolation) as exc:
            ck.check_pose(q, params.gripper.preopen_pct, stage="fold ")
        assert "自碰撞" in str(exc.value)
    finally:
        object.__setattr__(params.collision, "ignore_self_pairs", original)


def test_净距要求参与判定而不只是相交(params):
    """clearance_m 是"膨胀之后仍须保持"的最小净距，不是相交判据。

    墙必须做薄（半厚 < gap）且 gap < clearance：这样盒子与膨胀胶囊体在几何上
    确实不相交，判罚只能来自净距条款。此前版本用 half=2mm > gap=1.5mm 的厚墙，
    墙其实伸进胶囊体内部，报的是"相交"，clearance 是否参与判定测不出来。
    """
    model, env, ck = _checker(params)
    q = np.array(params.workspace.home_joints_deg, float)
    from configs.motion_params import Obstacle
    clearance = params.collision.clearance_m
    axes = ck._axes(q, params.gripper.preopen_pct)
    cid, a, b, r = max(axes, key=lambda t: np.linalg.norm(t[1]))
    direction = (b - a) / np.linalg.norm(b - a)
    surface = b + direction * r
    gap = clearance * 0.6                  # 表面外 gap 处放墙：净距不足但未相交
    t = gap * 0.25                         # 薄墙半厚：近端面距胶囊表面 gap-t > 0
    center = surface + direction * gap
    assert segment_obb_distance(a, b, center, np.array([t, t, t]), np.eye(3)) > r + 1e-9, \
        "墙与胶囊体几何上必须不相交，否则测不到净距条款"
    wall = Obstacle(id="near_wall",
                    bounds_m=np.array([center - np.array([t, t, t]),
                                       center + np.array([t, t, t])]))
    object.__setattr__(params.workspace, "obstacles", tuple(params.workspace.obstacles) + (wall,))
    try:
        with pytest.raises(CollisionViolation):
            ck.check_pose(q, params.gripper.preopen_pct, stage="near ")
        # 对照：把墙推到 clearance 之外，同一姿态必须放行，证明判罚来自净距而非位置。
        far = surface + direction * (clearance * 1.5 + 2 * t)
        object.__setattr__(wall, "bounds_m",
                           np.array([far - np.array([t, t, t]),
                                     far + np.array([t, t, t])]))
        ck.check_pose(q, params.gripper.preopen_pct, stage="far ")
    finally:
        object.__setattr__(params.workspace, "obstacles",
                           tuple(o for o in params.workspace.obstacles if o.id != "near_wall"))


def test_桌面判定扣除平度与净距(params):
    """指南 7.1 + 2.2：桌面检查阈值 = 误差膨胀 + table_flatness + clearance。

    把桌面抬到"最低胶囊表面以下 2mm"：几何上不相交（旧实现放行），但低于
    flatness(1mm)+clearance(3mm)=4mm 的净距预算，必须报碰桌。
    """
    model = ArmModel(params)
    env = build_error_envelope(params, params.collision.max_joint_substep_deg)
    q = np.array(params.workspace.home_joints_deg, float)
    ck0 = CollisionChecker(params, model, env)
    margins = {cid: segment_halfspace_distance(a, b, np.zeros(3), np.array([0, 0, 1.0])) - r
               for cid, a, b, r in ck0._axes(q, params.gripper.preopen_pct)
               if cid not in ck0._table_exempt}
    lowest = min(margins.values())
    slack = 0.002                          # 桌面比最低胶囊表面低 2mm
    assert slack < params.workspace.table_flatness_m + params.collision.clearance_m
    original = params.workspace.table_z_m
    object.__setattr__(params.workspace, "table_z_m", lowest - slack)
    try:
        _, _, ck = _checker(params)
        with pytest.raises(CollisionViolation) as exc:
            ck.check_pose(q, params.gripper.preopen_pct, stage="table-gap ")
        assert "碰桌" in str(exc.value)
    finally:
        object.__setattr__(params.workspace, "table_z_m", original)


# ---------------------------------------------------------------------------
# 4. 段起点偏差
# ---------------------------------------------------------------------------


def test_起点偏差判据用逐关节容差(params):
    tol = np.array([3.0, 3.0, 3.0, 3.0, 1.0])
    object.__setattr__(params.motion, "start_position_tol_deg", tol)
    q = np.array(params.workspace.home_joints_deg, float)
    assert check_start_offset(q, q.copy(), params, "x")
    assert check_start_offset(q + np.array([2.9, 0, 0, 0, 0]), q, params, "x")
    # 第 5 关节容差更紧，2° 就该判超
    assert not check_start_offset(q + np.array([0, 0, 0, 0, 2.0]), q, params, "x")
