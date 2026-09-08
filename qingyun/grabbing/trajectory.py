"""轨迹生成与时间参数化。

规范来源：docs/机械臂运动控制模块技术文档.md 第 5.2 节（规划表示与时间律）。

一段动作 = 顺序 MotionSegment。每段带 name/kind/time_s/joints_deg/gripper_pct/
tcp_targets；关节段的 tcp_targets 为 None。节点间隔固定 1/FPS，时间从 0 开始，
包含起终点。

时间律是五次多项式 s(u) = 10u³ − 15u⁴ + 6u⁵（u = t/T）：
    s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
所以段首段尾速度与加速度都为零，相邻段拼接时不会冲击。它的峰值
    max s' = 15/8 = 1.875        （u = 1/2）
    max s'' = 10/sqrt(3)         （u = 1/2 ∓ 1/(2√3)）
就是文档 5.2 里确定时长用的两个系数。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import FloatArray
from configs.motion_params import MotionParams
from qingyun.grabbing.kinematics_ext import (
    ArmModel,
    IkSolver,
    MotionError,
    inverse_transform,
    pose_error,
    top_down_pose,
)
from qingyun.grabbing.safety import JointLimits, PlanViolation

# 关节段/Cartesian 段两种规划表示（技术文档 5.2）。
KIND_JOINT = "JOINT"
KIND_CARTESIAN = "CARTESIAN"

# 五次时间律的峰值系数，见模块头注释的推导。
V_PEAK_FACTOR = 1.875
A_PEAK_FACTOR = 10.0 / math.sqrt(3.0)
# 位移小于这个值就当作零位移段，只出一个节点（文档 5.2"零位移段直接确认到位"）。
_ZERO_DISPLACEMENT_DEG = 1e-9
_ZERO_DISPLACEMENT_M = 1e-9
# 笛卡尔段"检查—调整时长—重新采样"的最大轮数，超出说明该段在当前约束下无解。
_RESAMPLE_ATTEMPTS = 8


@dataclass(frozen=True)
class MotionSegment:
    """一段可独立回放的轨迹。

    time_s 从 0 开始、节点间隔 1/FPS、含起终点；joints_deg 是 (N,5) 的 URDF deg；
    gripper_pct 在段内固定（夹爪自己的开合阶段不走这里）；关节段 tcp_targets=None。
    """

    name: str
    kind: str
    time_s: FloatArray
    joints_deg: FloatArray
    gripper_pct: float
    tcp_targets: FloatArray | None

    @property
    def nodes(self) -> int:
        return int(self.joints_deg.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1]) if self.nodes else 0.0

    @property
    def start_joints(self) -> FloatArray:
        return self.joints_deg[0]

    @property
    def end_joints(self) -> FloatArray:
        return self.joints_deg[-1]


# ---------------------------------------------------------------------------
# 1. 五次时间律
# ---------------------------------------------------------------------------


def quintic_s(u: NDArray | float) -> NDArray | float:
    """归一化位置律 s(u)=10u³−15u⁴+6u⁵。"""
    u = np.asarray(u, dtype=np.float64)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def joint_positions(q0: NDArray, q1: NDArray, u: NDArray) -> FloatArray:
    """按五次律插值一组关节角，u 是归一化时间数组。

    直接取差值、不做 360 度取模：被线缆限制的关节（wrist_roll）取模会把
    "绕一大圈"误判成"走一小步"（技术文档 5.2 末段）。
    """
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    return q0 + (q1 - q0) * quintic_s(np.asarray(u, float))[..., None]


def round_up_to_ticks(duration_s: float, tick_s: float) -> float:
    """把时长向上取整到整数个 tick，保证节点严格落在节拍上。"""
    if duration_s <= 0.0:
        return 0.0
    return math.ceil(duration_s / tick_s - 1e-12) * tick_s


def node_times(duration_s: float, tick_s: float) -> FloatArray:
    """生成 [0, tick, 2*tick, ..., duration] 的时间轴（含两端）。"""
    n = int(round(duration_s / tick_s)) + 1
    return np.arange(n, dtype=np.float64) * tick_s


# ---------------------------------------------------------------------------
# 2. 关节段
# ---------------------------------------------------------------------------


def joint_move_duration(q0: NDArray, q1: NDArray, params: MotionParams,
                        min_duration_s: float | None) -> float:
    """由峰值速度与峰值加速度约束确定一段关节运动所需的最短时长。

    技术文档 5.2：
        v_peak = 1.875*|Δq|/T      → T >= 1.875*|Δq|/v_max
        a_peak = (10/√3)*|Δq|/T²   → T >= sqrt((10/√3)*|Δq|/a_max)
    取所有关节里最大的那个，再与调用方给的 min_duration_s 比较。

    标定指南 9.4 还有第三条上限：实际相邻发送步长不得超过 max_command_step_deg。
    五次律下节点最大步长 = 1.875*|Δq|*tick/T，把它解出 T 会得到和"允许速度 =
    max_command_step_deg*FPS"完全等价的约束，所以这里直接把速度上限压到两者较小值，
    让段按构造就不会步长超限，而不是等 safety 阶段报错重来。
    """
    delta = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    if np.max(delta) <= _ZERO_DISPLACEMENT_DEG:
        return 0.0 if not min_duration_s else max(0.0, float(min_duration_s))
    v_allow = np.minimum(
        np.asarray(params.motion.max_velocity_deg_s, float),
        np.asarray(params.motion.max_command_step_deg, float) * params.timing.fps,
    )
    from_vel = V_PEAK_FACTOR * delta / v_allow
    from_acc = np.sqrt(A_PEAK_FACTOR * delta / np.asarray(params.motion.max_acceleration_deg_s2, float))
    need = float(np.max(np.maximum(from_vel, from_acc)))
    if min_duration_s is not None:
        need = max(need, float(min_duration_s))
    return need


def make_joint_segment(name: str, q0: NDArray, q1: NDArray, gripper_pct: float,
                       params: MotionParams, min_duration_s: float | None = None) -> MotionSegment:
    """生成一个关节空间五次多项式段。"""
    tick = 1.0 / params.timing.fps
    duration = round_up_to_ticks(joint_move_duration(q0, q1, params, min_duration_s), tick)
    if duration <= 0.0:
        # 零位移段：只发一个节点，由执行器确认到位（文档 5.2）。
        return MotionSegment(name, KIND_JOINT, np.array([0.0]), np.array([q1], float),
                             gripper_pct, None)
    times = node_times(duration, tick)
    u = times / duration
    q = joint_positions(q0, q1, u)
    # 五次律在 u=1 处浮点误差不严格为 1，强制钉住端点，避免终点差 1e-16deg。
    q[0] = q0
    q[-1] = q1
    return MotionSegment(name, KIND_JOINT, times, q, gripper_pct, None)


# ---------------------------------------------------------------------------
# 3. 笛卡尔竖直段
# ---------------------------------------------------------------------------


def make_cartesian_vertical_segment(
    name: str,
    model: ArmModel,
    ik: IkSolver,
    limits: JointLimits,
    q_start: NDArray,
    gripper_pct: float,
    dz_m: float,
    params: MotionParams,
    min_duration_s: float | None = None,
) -> MotionSegment:
    """生成"固定 TCP 姿态、沿基座 +Z 直线移动 dz"的笛卡尔段。

    流程按技术文档 5.2：先按时长粗估采样，逐路点链式 warm-start 求 IK，再检查
    关节与 TCP 的速度/加速度；不满足就加大时长重新采样，最多 _RESAMPLE_ATTEMPTS 轮。

    起始 TCP 用实测关节角的 FK 算，不用"规划值"：段执行前控制器会重新读取起点，
    这样即便上一段有到位残差，本段也是从真实位置开始的（文档 6.3 末段）。
    """
    tick = 1.0 / params.timing.fps
    T_start = model.fk_tcp(q_start, gripper_pct)
    p_start = T_start[:3, 3].copy()
    R_fixed = T_start[:3, :3].copy()
    p_end = p_start + np.array([0.0, 0.0, float(dz_m)])
    span = float(np.linalg.norm(p_end - p_start))
    if span <= _ZERO_DISPLACEMENT_M:
        return MotionSegment(name, KIND_CARTESIAN, np.array([0.0]), np.array([q_start], float),
                             gripper_pct, np.array([T_start]))

    # 初值：TCP 直线运动的五次律峰值系数与关节段相同，所以直接按 TCP 位移估时长。
    duration = max(
        V_PEAK_FACTOR * span / params.motion.tcp_max_velocity_m_s,
        math.sqrt(A_PEAK_FACTOR * span / params.motion.tcp_max_acceleration_m_s2),
    )
    if min_duration_s is not None:
        duration = max(duration, float(min_duration_s))

    last_reason = ""
    for _attempt in range(_RESAMPLE_ATTEMPTS):
        duration = round_up_to_ticks(duration, tick)
        times = node_times(duration, tick)
        u = times / duration
        # 位置沿基座 Z 直线插值，姿态整段固定（文档 5.2）。
        offsets = quintic_s(u) * span
        sign = 1.0 if dz_m >= 0.0 else -1.0
        pts = p_start[None, :] + np.outer(offsets, np.array([0.0, 0.0, sign]))
        q_path, tcp_targets, need_longer = _solve_chain(
            name, model, ik, limits, q_start, gripper_pct, pts, R_fixed, times, tick, params
        )
        if need_longer is None:
            return MotionSegment(name, KIND_CARTESIAN, times, q_path, gripper_pct, tcp_targets)
        last_reason = need_longer
        # 时长至少翻倍再试：超限倍数最大也就是约 2，翻倍一次就能满足。
        duration = duration * 2.0 + tick

    raise PlanViolation(f"{name} 笛卡尔段在 {_RESAMPLE_ATTEMPTS} 轮加长时长后仍超限：{last_reason}")


def _solve_chain(name: str, model: ArmModel, ik: IkSolver, limits: JointLimits,
                 q_start: NDArray, gripper_pct: float, points: NDArray, rotation: NDArray,
                 times: NDArray, tick: float, params: MotionParams):
    """对一串 TCP 位置逐个求 IK（链式 warm start），并返回超限原因。

    返回 (joints, tcp_targets, need_longer)。need_longer 为 None 表示这段可用，
    否则是需要更长时长的原因字符串。
    """
    q_path = np.empty((points.shape[0], 5), dtype=np.float64)
    tcp_targets = np.empty((points.shape[0], 4, 4), dtype=np.float64)
    q_prev = np.asarray(q_start, float)
    for k in range(points.shape[0]):
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = points[k]
        tcp_targets[k] = T
        try:
            sol = ik.solve(T, gripper_pct, q_prev)
        except MotionError as exc:
            # IK 不收敛不是"时长不够"，直接原样上报，让入口返回 IK_FAILED。
            raise
        q_path[k] = sol.joints_deg
        q_prev = sol.joints_deg

    bad = limits.violations(q_path.max(axis=0)) + limits.violations(q_path.min(axis=0))
    if bad:
        # IK 已经在自己的限位盒里求解，这里再兜一层，把越限当成段不可用。
        return None, None, "；".join(bad)

    # 关节差分速度/加速度。
    if q_path.shape[0] >= 2:
        step = np.max(np.abs(np.diff(q_path, axis=0)), axis=0)
        if np.any(step > params.motion.max_command_step_deg + 1e-9):
            return None, None, f"相邻发送步长超限 {np.round(step, 3).tolist()}"
        if np.any(step / tick > params.motion.max_velocity_deg_s + 1e-9):
            return None, None, f"关节差分速度超限 {np.round(step / tick, 2).tolist()}"
    if q_path.shape[0] >= 3:
        dd = np.max(np.abs(np.diff(q_path, n=2, axis=0)), axis=0)
        if np.any(dd / (tick * tick) > params.motion.max_acceleration_deg_s2 + 1e-9):
            return None, None, f"关节差分加速度超限 {np.round(dd / tick ** 2, 1).tolist()}"

    # TCP 差分速度/加速度。
    v = float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) / tick) if points.shape[0] >= 2 else 0.0
    if v > params.motion.tcp_max_velocity_m_s + 1e-9:
        return None, None, f"TCP 差分速度超限 {v:.4f}m/s"
    if points.shape[0] >= 3:
        a = float(np.max(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1)) / (tick * tick))
        if a > params.motion.tcp_max_acceleration_m_s2 + 1e-9:
            return None, None, f"TCP 差分加速度超限 {a:.4f}m/s²"

    # 回代 FK 检查每个节点的残差（文档 5.3 的"FK 路点残差"）。
    for k in range(q_path.shape[0]):
        err = pose_error(model.fk_tcp(q_path[k], gripper_pct), tcp_targets[k])
        if not err.acceptable(params):
            return None, None, f"第 {k} 节点 IK 残差未达标 {err}"
    return q_path, tcp_targets, None


# ---------------------------------------------------------------------------
# 4. 路线拼装
# ---------------------------------------------------------------------------


def chain_joint_segments(prefix: str, waypoints: Sequence[NDArray], gripper_pct: float,
                         params: MotionParams, *, start_from: NDArray | None = None,
                         min_duration_s: float | None = None) -> list[MotionSegment]:
    """把一串关节中转位拼成顺序关节段。

    start_from=None 时从 waypoints[0] 开始（用于"当前已经在第一个中转位"的场合），
    否则先走 start_from → waypoints[0]。
    """
    pts = [np.asarray(w, float) for w in waypoints]
    if start_from is not None:
        pts = [np.asarray(start_from, float), *pts]
    segments: list[MotionSegment] = []
    for i in range(len(pts) - 1):
        segments.append(
            make_joint_segment(f"{prefix}{i}", pts[i], pts[i + 1], gripper_pct, params,
                               min_duration_s=min_duration_s)
        )
    return segments
