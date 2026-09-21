"""TCP 变换、姿态构造与 IK 收敛包装。

规范来源：
    docs/顶抓姿态构造与R_top_down标定.md   —— 坐标系约定、R_TOP_DOWN、工具变换、残差定义
    docs/机械臂运动控制模块技术文档.md 5.1  —— IK 迭代、停滞判据、备用初值、检查点
    docs/真机参数测量与标定指南.md 第 12 节  —— 与标定工具共用的函数签名

本模块只做"位姿数学 + 求解器包装"，不做轨迹时间参数化（trajectory.py）、
不做碰撞与限位校验（safety.py）、也不做串口与节拍（executor.py）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import FloatArray, JOINT_NAMES
from configs.motion_params import (
    MotionParams,
    ToolParams,
    effective_joint_limits,
)
from libs.so_arm_core.kinematics import RobotKinematics

# 顶抓目标系的名称，只用于注释与错误信息。
BASE_FRAME = "base_link"
GRIPPER_FRAME = "gripper_frame_link"
GRIPPER_JOINT = "gripper"

# 零偏航顶抓时的工具姿态：TCP +X 沿 B +X、+Y 沿 B -Y、+Z 沿 B -Z。
# 这是工具坐标约定产生的常数，正交且 det=+1（姿态文档第 2 节）。
R_TOP_DOWN: FloatArray = np.diag([1.0, -1.0, -1.0])

# 姿态文档 5 节：TCP +X 投影到基座 XY 平面退化时 yaw 无定义，判定解无效。
_YAW_PROJECTION_EPS_M = 1e-9


class MotionError(RuntimeError):
    """控制内部的运动错误，携带要返回给调用方的状态码。

    技术文档 4.1：公开原语失败抛带 GraspStatus 的内部 MotionError，
    技能入口 grasp_and_place 负责把它转换成 GraspResult。
    """

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class IkNotConverged(MotionError):
    """所有初值都用完仍未同时满足位置/倾角/yaw 三项容限。"""

    def __init__(self, reason: str) -> None:
        super().__init__("IK_FAILED", reason)


class StoppedByRequest(MotionError):
    """should_stop 在检查点返回 True，或收到 KeyboardInterrupt。"""

    def __init__(self, reason: str = "外部停止请求") -> None:
        super().__init__("ABORTED", reason)


# ---------------------------------------------------------------------------
# 1. 基础姿态构造
# ---------------------------------------------------------------------------


def rz_deg(angle_deg: float) -> FloatArray:
    """绕 +Z 的右手旋转矩阵，输入 deg。"""
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def homogeneous(rotation: NDArray, translation: NDArray) -> FloatArray:
    """由 3x3 旋转与 3 平移拼出 4x4 齐次矩阵。"""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def inverse_transform(transform: FloatArray) -> FloatArray:
    """刚体齐次矩阵的逆：inv([R,t]) = [R^T, -R^T t]。

    比通用 np.linalg.inv 更快也更容易核对，并且避免把数值误差扩散到
    最后一行 [0,0,0,1]。
    """
    r = transform[:3, :3]
    t = transform[:3, 3]
    return homogeneous(r.T, -r.T @ t)


def top_down_pose(tcp_xyz: NDArray, tcp_yaw_deg: float) -> FloatArray:
    """构造顶抓 TCP 位姿 T_B_TCP。

    姿态文档第 2 节的定义：旋转是 rz_deg(yaw) @ R_TOP_DOWN，因此任意 yaw 下
    旋转矩阵第三列（接近方向）恒为 [0,0,-1]。位置参数是 TCP，不是物体中心。
    """
    return homogeneous(rz_deg(tcp_yaw_deg) @ R_TOP_DOWN, np.asarray(tcp_xyz, dtype=np.float64))


def object_pose(position: NDArray, yaw_deg: float) -> FloatArray:
    """构造物体中心系相对基座的 T_B_O。

    O 系 +X 沿水平长轴代表方向、+Z 与基座 +Z 平行，所以旋转只有绕 Z 一项。
    """
    return homogeneous(rz_deg(yaw_deg), np.asarray(position, dtype=np.float64))


def close_axis_width_at_yaw(length: float, width: float, yaw_offset_deg: float) -> float:
    """目标在固定 yaw_offset 下沿夹爪闭合方向的最大投影宽度。

    标定指南 2.4 给出的保守式：abs(cos(beta))*length + abs(sin(beta))*width，
    beta 为偏移角转 rad。用于检查 preopen_pct 的 gap 是否真的够大。
    """
    beta = math.radians(float(yaw_offset_deg))
    return abs(math.cos(beta)) * float(length) + abs(math.sin(beta)) * float(width)


# ---------------------------------------------------------------------------
# 2. 开度相关的工具与夹爪标定表
# ---------------------------------------------------------------------------


def _table_axes(pcts: NDArray, values: NDArray) -> tuple[NDArray, NDArray]:
    """返回按开度升序排列的 (开度, 数值) 两列，供插值使用。"""
    order = np.argsort(pcts, kind="stable")
    return pcts[order], values[order]


def interp_table(gripper_pct: float, xs: NDArray, ys: NDArray, table_name: str) -> float:
    """标定表的分段线性插值，禁止外插。

    配置加载已经保证 xs 落在 [0,100] 且严格递增，所以这里只需要判断请求的开度
    是否落在表的覆盖范围内。超出表范围时抛错而不是悄悄取端点值：外插会给出
    物理上不存在的 gap（标定指南 6.2/6.4 明确要求不得外插）。
    """
    g = float(gripper_pct)
    if g < float(xs[0]) - 1e-9 or g > float(xs[-1]) + 1e-9:
        raise MotionError(
            "CONFIG_INVALID",
            f"开度 {g:.3f}% 超出 {table_name} 覆盖区间 [{float(xs[0]):.3f}, {float(xs[-1]):.3f}]",
        )
    # np.interp 在端点处返回端点值，上面已经排除了外插请求。
    return float(np.interp(g, xs, ys))


def gap_table_axes(gripper_tables) -> tuple[NDArray, NDArray]:
    """从 GripperParams 取 gap_table 的两列。"""
    pcts = np.array([s.gripper_pct for s in gripper_tables.gap_table], dtype=np.float64)
    gaps = np.array([s.gap_m for s in gripper_tables.gap_table], dtype=np.float64)
    return _table_axes(pcts, gaps)


def angle_table_axes(gripper_tables) -> tuple[NDArray, NDArray]:
    """从 GripperParams 取 angle_table 的两列。"""
    pcts = np.array([s.gripper_pct for s in gripper_tables.angle_table], dtype=np.float64)
    angles = np.array([s.angle_deg for s in gripper_tables.angle_table], dtype=np.float64)
    return _table_axes(pcts, angles)


def gripper_pct_to_gap(gripper_pct: float, gap_table) -> float:
    """开度 -> 指垫内表面有效开口，单位 m。

    共享函数（标定指南第 12 节）。gap_table 是严格递增的标定表。
    """
    xs, ys = gap_table_axes(gap_table)
    return interp_table(gripper_pct, xs, ys, "gripper.gap_table")


def gripper_pct_to_angle_deg(gripper_pct: float, angle_table) -> float:
    """开度 -> 活动指 URDF 角，单位 deg。只服务于碰撞模型，不参与五关节 IK。"""
    xs, ys = angle_table_axes(angle_table)
    return interp_table(gripper_pct, xs, ys, "gripper.angle_table")


def gap_to_gripper_pct(gap_m: float, gap_table) -> float:
    """有效开口反查开度。

    gap_table 的 gap 严格递增，所以反插值唯一。用于接触判据里"实测开口是否落在
    contact_gap_range_m 内"的开度换算，以及加载期的覆盖预检。
    """
    xs, ys = gap_table_axes(gap_table)
    return interp_table_safe(float(gap_m), ys, xs, "gripper.gap_table")


def interp_table_safe(x: float, xs: NDArray, ys: NDArray, table_name: str) -> float:
    """与 interp_table 相同，但自变量是 xs 上任意一列（用于反查）。

    单独拆出来是因为反查 gap -> pct 时自变量是 gap 而不是开度，而配置里 gap
    保证严格递增，所以同一套代码可用，只是错误信息里的"区间"含义不同。
    """
    if x < float(xs[0]) - 1e-12 or x > float(xs[-1]) + 1e-12:
        raise MotionError(
            "CONFIG_INVALID",
            f"{table_name} 反查值 {x:.6f} 超出覆盖区间 [{float(xs[0]):.6f}, {float(xs[-1]):.6f}]",
        )
    return float(np.interp(x, xs, ys))


def tool_translation(gripper_pct: float, tool_params: ToolParams) -> FloatArray:
    """按开度分段线性插值 TCP 在 E 系中的平移分量。

    姿态文档第 3 节：只对平移做插值，旋转固定在工具主体上，不随活动指转角变化。
    """
    # 拼成 (n,4)：第 0 列是开度，后 3 列是 xyz，一起按开度排序后逐列插值。
    rows = np.array(
        [[s.gripper_pct, *s.xyz_m] for s in tool_params.translation_samples], dtype=np.float64
    )
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    pcts = rows[:, 0]
    out = np.empty(3, dtype=np.float64)
    for axis in range(3):
        out[axis] = interp_table(gripper_pct, pcts, rows[:, axis + 1], "tool.translation_samples")
    return out


def tool_transform(gripper_pct: float, tool_params: ToolParams) -> FloatArray:
    """开度 g 下的 T_E_TCP（标定指南第 12 节的共享函数）。

    旋转部分直接取 tool.rotation_e_tcp；平移按开度插值。单动指夹爪的 TCP 会随
    开度移动，所以闭爪后必须用实际开度重新计算，不能沿用下降时的值。
    """
    return homogeneous(tool_params.rotation_e_tcp, tool_translation(gripper_pct, tool_params))


# ---------------------------------------------------------------------------
# 3. 姿态残差与验收判据
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseError:
    """姿态文档第 5 节定义的三项残差。"""

    position_m: float
    tilt_deg: float
    yaw_deg: float
    yaw_degenerate: bool

    def composite(self, position_tol_m: float, tilt_tol_deg: float, yaw_tol_deg: float) -> float:
        """归一化综合残差 E，用于停滞判据。

        标定指南 2.3：E = e_pos/position_tol_m + e_tilt/tilt_tol_deg + e_yaw/yaw_tol_deg。
        yaw 退化时把 E 置成一个明显未收敛的大值，让停滞检查继续推进而不是卡住。
        """
        if self.yaw_degenerate:
            return float("inf")
        return (
            self.position_m / position_tol_m
            + self.tilt_deg / tilt_tol_deg
            + self.yaw_deg / yaw_tol_deg
        )

    def acceptable(self, params: MotionParams) -> bool:
        """三项误差同时达标才算收敛（姿态文档第 5 节）。"""
        if self.yaw_degenerate:
            return False
        ik = params.ik
        return (
            self.position_m <= ik.position_tol_m
            and self.tilt_deg <= ik.tilt_tol_deg
            and self.yaw_deg <= ik.yaw_tol_deg
        )

    def __str__(self) -> str:  # 便于把失败原因直接写进 GraspResult.reason
        return (
            f"e_pos={self.position_m * 1000:.2f}mm "
            f"e_tilt={self.tilt_deg:.3f}deg "
            f"e_yaw={self.yaw_deg:.3f}deg"
            + (" (yaw 投影退化)" if self.yaw_degenerate else "")
        )


def wrap180deg(angle_deg: float) -> float:
    """把角度差折算到 [-180,180)，用于有向 yaw 误差。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def pose_error(T_act: FloatArray, T_des: FloatArray) -> PoseError:
    """比较两个完整 TCP 位姿，返回姿态文档第 5 节定义的三项残差。"""
    e_pos = float(np.linalg.norm(T_act[:3, 3] - T_des[:3, 3]))

    # 接近轴夹角：把点积裁到 [-1,1] 再取 acos，避免 acos 因数值溢出返回 NaN。
    za = T_act[:3, 2]
    zd = T_des[:3, 2]
    cos_tilt = float(np.clip(np.dot(za, zd), -1.0, 1.0))
    e_tilt = math.degrees(math.acos(cos_tilt))

    # 工具 +X 投影到基座 XY 后用 atan2 求有向 yaw；投影长度接近 0 时 yaw 无定义。
    xa = T_act[:3, 0]
    xd = T_des[:3, 0]
    la = math.hypot(float(xa[0]), float(xa[1]))
    ld = math.hypot(float(xd[0]), float(xd[1]))
    if la < _YAW_PROJECTION_EPS_M or ld < _YAW_PROJECTION_EPS_M:
        return PoseError(e_pos, e_tilt, float("inf"), True)
    yaw_act = math.degrees(math.atan2(float(xa[1]), float(xa[0])))
    yaw_des = math.degrees(math.atan2(float(xd[1]), float(xd[0])))
    e_yaw = abs(wrap180deg(yaw_act - yaw_des))
    return PoseError(e_pos, e_tilt, e_yaw, False)


def approach_tilt_deg(T: FloatArray) -> float:
    """接近轴与竖直向下 [0,0,-1] 的夹角，单位 deg。

    姿态文档 5 节：夹持后更新出的目标姿态可能带允许范围内的工具倾角，
    需要额外检查它的接近轴偏离竖直方向不超过 ik.tilt_tol_deg。
    """
    return math.degrees(math.acos(float(np.clip(-T[:3, 2][2], -1.0, 1.0))))


# ---------------------------------------------------------------------------
# 4. 机器人模型（FK）
# ---------------------------------------------------------------------------


class ArmModel:
    """在 vendor 的 RobotKinematics 之上补充本项目需要的多 link FK 与 TCP FK。

    为什么要包一层：
      * vendor 类只知道一个 target_frame_name，而碰撞检查需要每个 link 的位姿；
      * 碰撞模型的活动指角度要按 gripper.angle_table 单独设置（技术文档 5.1），
        不能让五关节 IK 把夹爪开度当成第 6 个姿态自由度；
      * solver 里的夹爪自由度必须固定，否则 IK 会顺手改动 gripper 关节。
    """

    def __init__(self, params: MotionParams) -> None:
        self.params = params
        self.kin = RobotKinematics(
            str(params.urdf_path_resolved()),
            GRIPPER_FRAME,
            # 显式传 JOINT_NAMES：不依赖上游"取全部关节"的默认行为。
            list(JOINT_NAMES),
        )
        # 固定夹爪自由度：mask_dof 把该关节从决策变量里剔除，IK 不再改它。
        self.kin.solver.mask_dof(GRIPPER_JOINT)
        # 把有效限位（URDF ∩ 实测 ∩ 应用 − margin）交给求解器当搜索盒。
        # vendor 的 solver 默认只受 URDF 限位约束，不这样写的话实测行程和
        # wrist_roll 的线缆限制对 IK 完全不可见，会给出位姿达标但舵机到不了
        # 的分支。验收检查仍然在 IkSolver 里独立做一次，不依赖这里的约束。
        lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
        for i, name in enumerate(JOINT_NAMES):
            self.kin.robot.set_joint_limits(name, math.radians(lower[i]), math.radians(upper[i]))
        self.kin.solver.enable_joint_limits(True)
        self.limit_lower = lower
        self.limit_upper = upper
        # 缓存 link 名字到 placo frame 的映射，避免每节点重复查表。
        self._link_names = set(self.kin.robot.frame_names())

    # --- 底层状态写入 ---

    def _write_joints(self, q_urdf_deg: NDArray, gripper_angle_deg: float) -> None:
        """把 5 个姿态关节（deg, URDF 坐标）与活动指角写进 placo 模型并更新。"""
        rad = np.deg2rad(np.asarray(q_urdf_deg, dtype=np.float64))
        for i, name in enumerate(JOINT_NAMES):
            self.kin.robot.set_joint(name, float(rad[i]))
        self.kin.robot.set_joint(GRIPPER_JOINT, math.radians(float(gripper_angle_deg)))
        self.kin.robot.update_kinematics()

    def frames_for(self, link_names: Sequence[str], q_urdf_deg: NDArray,
                   gripper_pct: float) -> dict[str, FloatArray]:
        """一次写入关节状态，取回多个 link 的 T_B_link。

        碰撞检查每个位姿都要读全部胶囊体所在的 link；如果逐个调 fk_link，
        set_joint + update_kinematics 会被重复几十遍，按 max_joint_substep_deg
        加密之后会慢到不可用。
        """
        angle = gripper_pct_to_angle_deg(gripper_pct, self.params.gripper)
        self._write_joints(q_urdf_deg, angle)
        return {name: np.array(self.kin.robot.get_T_world_frame(name), dtype=np.float64)
                for name in link_names}

    def fk_link(self, link_name: str, q_urdf_deg: NDArray, gripper_pct: float) -> FloatArray:
        """返回 T_B_<link>。活动指所在 link 的角度按 gripper.angle_table 同步。"""
        if link_name not in self._link_names:
            raise MotionError(
                "CONFIG_INVALID", f"URDF 里找不到 link {link_name!r}（collision.link_capsules 引用它）"
            )
        return self.frames_for([link_name], q_urdf_deg, gripper_pct)[link_name]

    def fk_e(self, q_urdf_deg: NDArray) -> FloatArray:
        """T_B_E：只由 5 个姿态关节决定，与夹爪开度无关。"""
        # 走 vendor 的 forward_kinematics，保证与 IK 使用同一套模型代码。
        return np.array(self.kin.forward_kinematics(np.asarray(q_urdf_deg, dtype=np.float64)),
                        dtype=np.float64)

    def fk_tcp(self, q_urdf_deg: NDArray, gripper_pct: float) -> FloatArray:
        """T_B_TCP = FK_E(q) @ T_E_TCP(g)（姿态文档第 3 节）。"""
        return self.fk_e(q_urdf_deg) @ tool_transform(gripper_pct, self.params.tool)

    def tcp_from_e(self, T_B_E: FloatArray, gripper_pct: float) -> FloatArray:
        """已有 E 位姿时换算 TCP，避免多余的一次 FK。"""
        return T_B_E @ tool_transform(gripper_pct, self.params.tool)

    def e_from_tcp(self, T_B_TCP: FloatArray, gripper_pct: float) -> FloatArray:
        """IK 的实际目标：T_B_E = T_B_TCP_target @ inv(T_E_TCP(g))。

        注意用的是同一个开度 g 的工具变换；开度一变 TCP 就变，所以闭爪后必须
        用实测开度重新算剩余路径（技术文档 3.3/5.3）。
        """
        return T_B_TCP @ inverse_transform(tool_transform(gripper_pct, self.params.tool))

    def single_solver_call(self, q_seed_deg: NDArray, T_B_E_target: FloatArray,
                           position_weight: float, orientation_weight: float) -> FloatArray:
        """暴露 vendor 的一次 solve 调用，供 IkSolver 反复迭代。"""
        return np.asarray(
            self.kin.inverse_kinematics(
                np.asarray(q_seed_deg, dtype=np.float64),
                T_B_E_target,
                position_weight,
                orientation_weight,
            ),
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# 5. IK 收敛包装
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IkSolution:
    """一次成功求解的结果。"""

    joints_deg: FloatArray   # (5,) URDF 坐标
    error: PoseError
    iterations: int          # 实际消耗的 solver.solve 次数（含本次求解）
    seed_index: int          # 用的是第几个初值（0 = 传入的 warm start）


class IkSolver:
    """按当前结果继续迭代求解，并施加时限、停滞与停止检查。

    技术文档 5.1：vendor 单次 inverse_kinematics 只执行一次 solver.solve(True)，
    所以收敛循环必须由本包装器掌握，同时受 ik.max_iterations、ik.max_solve_s 与
    停滞判据约束，并在迭代检查点执行停止/反馈检查。
    """

    def __init__(
        self,
        model: ArmModel,
        params: MotionParams,
        planning_checkpoint: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.params = params
        # 缺省检查点什么都不做；ArmController 会注入 should_stop + 反馈巡检。
        self.checkpoint = planning_checkpoint or (lambda: None)
        # 限位直接复用 ArmModel 算好并下发给求解器的那一份，避免同一个交集
        # 在两处独立计算而给出不一致的结果。
        self.limit_lower = model.limit_lower
        self.limit_upper = model.limit_upper

    # --- 内部工具 ---

    def _within_limits(self, q: NDArray) -> bool:
        return bool(np.all(q >= self.limit_lower - 1e-9) and np.all(q <= self.limit_upper + 1e-9))

    def _evaluate(self, q: NDArray, gripper_pct: float, T_des_TCP: FloatArray) -> PoseError:
        """把关节解代回完整 TCP 链算残差（姿态文档：T_act 是"实际关节解的完整 TCP FK"）。"""
        return pose_error(self.model.fk_tcp(q, gripper_pct), T_des_TCP)

    def solve(
        self,
        T_B_TCP_target: FloatArray,
        gripper_pct: float,
        q_seed_deg: NDArray,
    ) -> IkSolution:
        """求一个 TCP 位姿对应的五关节解。

        初值顺序是 [q_seed_deg, *ik.seed_joints_deg]：备用初值只改变求解起点，
        不改变目标位置、角度或抓取方向（技术文档 5.1）。总时限 max_solve_s 不随
        初值切换而重置。

        为什么"迭代到停滞"而不是"第一次达标就返回"：placo 的 KinematicsSolver 在
        多次 solve() 之间会保留 QP 的近端/滞后乘子，同一个目标在不同调用历史下第一
        步的步长并不一样。早返回会让解的质量依赖调用历史——实测同一条笛卡尔链在两
        次运行里会给出形状不同的关节路径，差分加速度是否超限变成随机结果。规划必须
        可复现，所以这里每个初值都一路迭代到停滞或迭代上限，取综合残差最小且落在有效
        限位内的解，最后再按三项容限统一验收（权重只是优化偏好，不替代残差验收）。
        """
        ik = self.params.ik
        T_B_E_target = self.model.e_from_tcp(T_B_TCP_target, gripper_pct)

        seeds = [np.asarray(q_seed_deg, dtype=np.float64)]
        if ik.seed_joints_deg.size:
            seeds.extend(np.asarray(row, dtype=np.float64) for row in ik.seed_joints_deg)

        deadline = time.monotonic() + ik.max_solve_s
        best: IkSolution | None = None
        best_E = float("inf")
        iterations_total = 0

        for seed_index, seed in enumerate(seeds):
            if time.monotonic() >= deadline:
                break
            q = seed
            prev_E = float("inf")
            stagnation = 0
            seed_best: IkSolution | None = None
            seed_best_E = float("inf")
            for _ in range(ik.max_iterations):
                # 迭代检查点：停止请求可以在这里退出，同时保证不会超出 max_solve_s。
                self.checkpoint()
                if time.monotonic() >= deadline:
                    break
                q_next = self.model.single_solver_call(
                    q, T_B_E_target, ik.position_weight, ik.orientation_weight
                )
                iterations_total += 1
                if not np.all(np.isfinite(q_next)):
                    # 数值发散：换下一个初值，别把 NaN 传给执行器。
                    break
                err = self._evaluate(q_next, gripper_pct, T_B_TCP_target)
                e_now = err.composite(ik.position_tol_m, ik.tilt_tol_deg, ik.yaw_tol_deg)
                # 只接受落在有效限位内的候选；限位外的中间迭代值不作为最终解。
                if e_now < seed_best_E and self._within_limits(q_next):
                    seed_best = IkSolution(joints_deg=q_next, error=err,
                                           iterations=iterations_total, seed_index=seed_index)
                    seed_best_E = e_now
                # 停滞判据：连续 stagnation_iterations 次 E 下降不足 min_progress。
                if prev_E - e_now < ik.min_progress:
                    stagnation += 1
                else:
                    stagnation = 0
                prev_E = e_now
                q = q_next
                if stagnation >= ik.stagnation_iterations:
                    break
            if seed_best is not None and seed_best_E < best_E:
                best, best_E = seed_best, seed_best_E

        if best is not None and best.error.acceptable(self.params):
            return best
        if best is None:
            raise IkNotConverged(
                f"经 {iterations_total} 次求解、{len(seeds)} 个初值：所有迭代点都越出有效限位"
            )
        raise IkNotConverged(
            f"经 {iterations_total} 次求解、{len(seeds)} 个初值仍未收敛：残差未达标 {best.error}"
        )


def grasp_target_pose(target_position: NDArray, target_yaw_deg: float, params: MotionParams) -> tuple[
    FloatArray, float, FloatArray
]:
    """从视觉下发的物体中心算出抓取 TCP 位姿（技术文档 3.3）。

    返回 (p_grasp, spin_grasp_deg, T_B_TCP_grasp)。p_grasp 是物体中心加上"物体系中
    表达的固定抓取偏移"旋转后的抓取点；抓取方向是物体长轴加已标定的 yaw_offset_deg。
    """
    p_grasp = (
        np.asarray(target_position, dtype=np.float64)
        + rz_deg(target_yaw_deg) @ params.grasp.center_offset_object_m
    )
    spin = float(target_yaw_deg) + params.grasp.yaw_offset_deg
    return p_grasp, spin, top_down_pose(p_grasp, spin)


def holding_transform(T_B_TCP_close: FloatArray, T_B_O_grasp: FloatArray) -> FloatArray:
    """闭爪后建立的名义物体相对 TCP 变换 T_TCP_O（技术文档 3.4）。"""
    return inverse_transform(T_B_TCP_close) @ T_B_O_grasp


def place_tcp_pose(T_B_O_place: FloatArray, T_TCP_O: FloatArray) -> FloatArray:
    """由放置物体中心与持物关系算实际放置 TCP。

    配置里 places.position 是"释放时物体中心"，不是 TCP，所以必须经过
    T_B_TCP_place = T_B_O_place @ inv(T_TCP_O) 换算（技术文档 3.4）。
    """
    return T_B_O_place @ inverse_transform(T_TCP_O)
