"""同步抓取—放置技能入口与动作状态编排。

规范来源：docs/机械臂运动控制模块技术文档.md 第 3.2、3.3、3.4、4.1、4.3、6.3、7 节。

调用形态是单线程、同步、阻塞的一次调用：

    result = arm.grasp_and_place(target, place_id="default")

一次调用只完成一个抓取—放置动作。目标在调用期间固定不变，控制不向视觉回查数据，
也不内置目标重选、失败重试或品级到放置点的映射（文档 1.2）。

文档 6.3 状态表在本文件里体现为 grasp_and_place 里一段顺序执行的阶段，每进入
一个阶段就把名字写进 self._stage；任何失败都随 GraspResult.stage 返回，不抛异常。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import (
    FloatArray,
    GraspResult,
    GraspStatus,
    HoldingState,
    MotorController,
    PlacePose,
    VisionInterface,
)
from configs.motion_params import MotionParams, ParamsError, wrap180
from qingyun.grabbing.executor import ExecutionError, SyncExecutor
from qingyun.grabbing.kinematics_ext import (
    ArmModel,
    IkNotConverged,
    IkSolver,
    MotionError,
    StoppedByRequest,
    approach_tilt_deg,
    gap_to_gripper_pct,
    grasp_target_pose,
    holding_transform,
    object_pose,
    place_tcp_pose,
    pose_error,
    gripper_pct_to_gap,
    top_down_pose,
)
from qingyun.grabbing.safety import (
    CollisionChecker,
    CollisionViolation,
    ErrorEnvelope,
    JointLimits,
    OrientedBox,
    PlanViolation,
    build_error_envelope,
    check_start_offset,
    object_box,
    validate_segment,
)
from qingyun.grabbing.trajectory import (
    MotionSegment,
    chain_joint_segments,
    make_cartesian_vertical_segment,
    make_joint_segment,
)

# 文档 6.3 状态表里的阶段名。GraspResult.stage 只取这些值。
STAGE_CHECK = "CHECK"
STAGE_OPEN = "OPEN"
STAGE_APPROACH = "APPROACH"
STAGE_DESCEND = "DESCEND"
STAGE_CLOSE = "CLOSE"
STAGE_LIFT = "LIFT"
STAGE_TRANSFER = "TRANSFER"
STAGE_LOWER = "LOWER"
STAGE_RELEASE = "RELEASE"
STAGE_RETREAT = "RETREAT"
STAGE_RETURN = "RETURN"
STAGE_DONE = "DONE"
STAGE_FAULT_HOLD = "FAULT_HOLD"
STAGE_FAULT_UNCONTROLLED = "FAULT_UNCONTROLLED"


@dataclass
class PlanStep:
    """一个"可由实测起点重建"的规划步骤。

    build(q_from) 用实测起点生成段，而不是沿用上一段的规划终点：文档 6.3 末段要求
    不同路段执行前读取实测起点，超出 start_position_tol_deg 时重新规划到同一目标。
    rebuildable=False 的步骤是固定竖直线的笛卡尔段——起点一偏就再也保持不住那条
    直线，只能报错，不能偷偷改成斜线。
    """

    name: str
    stage: str
    build: Callable[[NDArray], list[MotionSegment]]
    obb_for_node: Callable[[int, NDArray], OrientedBox | None] | None = None
    contact_allowed: bool = False
    rebuildable: bool = True


@dataclass
class _PostGraspPlan:
    """闭爪后用实际开度重建出来的余下路径（文档 5.3）。"""

    lift: PlanStep
    transfer: PlanStep
    lower: PlanStep
    retreat: PlanStep
    return_home: PlanStep


class ArmController:
    """SO-ARM101 抓取—放置控制器。"""

    def __init__(
        self,
        motor: MotorController,
        params: MotionParams,
        should_stop: Callable[[], bool] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.motor = motor
        self.params = params
        # 可选的快速、无阻塞同步检查函数，缺省永不停止（文档 4.1）。
        self.should_stop = should_stop or (lambda: False)
        self.clock = clock
        self.sleep = sleep
        self.clock_ns = clock_ns

        self.model = ArmModel(params)
        self.limits = JointLimits.from_params(params)
        self.envelope: ErrorEnvelope = build_error_envelope(
            params, params.collision.max_joint_substep_deg
        )
        self.checker = CollisionChecker(params, self.model, self.envelope)
        self.executor = SyncExecutor(motor, params, should_stop, clock=clock, sleep=sleep,
                                     clock_ns=clock_ns)
        # IK 在规划检查点里复用同一个执行器：停止与反馈巡检逻辑只有一份实现。
        self.ik = IkSolver(self.model, params, self.executor.planning_checkpoint)

        # 实例锁存与重入保护（文档 1.2、4.3）。
        self._fault_latched = False
        self._in_call = False
        self._motion_started = False
        self._stage = STAGE_CHECK
        self._holding = HoldingState.UNKNOWN
        self._hold_pct: float | None = None

    # ==================================================================
    # 1. 公开原语（文档 4.1）
    #
    # 公开原语用于标定和调试，也通过同一套校验与同步执行器；失败抛带
    # GraspStatus 的内部 MotionError，由技能入口统一转成 GraspResult。
    # ==================================================================

    def move_joints(self, q_target_deg: NDArray,
                    min_duration_s: float | None = None) -> None:
        """单段关节运动到目标关节角。"""
        self._enter("move_joints")
        try:
            q_target = self._as_joint_vector("q_target_deg", q_target_deg)
            self._check_limits_or_raise(q_target)
            step = PlanStep(
                name="move_joints", stage=STAGE_APPROACH,
                build=lambda q_from: [make_joint_segment(
                    "move_joints", q_from, q_target, self._current_gripper_pct(),
                    self.params, min_duration_s=min_duration_s)],
            )
            self._motion_started = True
            self._execute_step(step)
        except (MotionError, KeyboardInterrupt) as exc:
            # 只有"运动已开始后"的错误才锁存；入参/限位校验失败时臂没动过（文档 4.3）。
            if self._motion_started:
                self._latch()
            if isinstance(exc, KeyboardInterrupt):
                raise StoppedByRequest("KeyboardInterrupt") from None
            raise
        finally:
            self._exit()

    def move_cartesian_top_down(self, tcp_xyz: NDArray, tcp_yaw_deg: float,
                                min_duration_s: float | None = None) -> None:
        """把当前 TCP 沿竖直直线移动到给定 TCP 位姿。

        参数明确是 TCP 而不是物体中心。只执行竖直直线：要求 XY 与姿态在容差内与
        当前一致；需要横向接近或改变朝向时用已校验的关节段（文档 4.1）。
        """
        self._enter("move_cartesian_top_down")
        try:
            p_target = self._as_position("tcp_xyz", tcp_xyz)
            if not np.isfinite(float(tcp_yaw_deg)):
                raise MotionError("INVALID_INPUT", f"tcp_yaw_deg 非有限：{tcp_yaw_deg}")
            T_target = top_down_pose(p_target, float(tcp_yaw_deg))
            q_from = self._measured_joints()
            g = self._current_gripper_pct()
            self._require_vertical_only(self.model.fk_tcp(q_from, g), T_target)
            step = PlanStep(
                name="move_cartesian", stage=STAGE_APPROACH, rebuildable=False,
                build=lambda q: [make_cartesian_vertical_segment(
                    "move_cartesian", self.model, self.ik, self.limits, q, g,
                    float(T_target[2, 3] - self.model.fk_tcp(q, g)[2, 3]),
                    self.params, min_duration_s=min_duration_s)],
            )
            self._motion_started = True
            self._execute_step(step)
        except (MotionError, KeyboardInterrupt) as exc:
            if self._motion_started:
                self._latch()
            if isinstance(exc, KeyboardInterrupt):
                raise StoppedByRequest("KeyboardInterrupt") from None
            raise
        finally:
            self._exit()

    def open_gripper(self) -> None:
        """小步张开到 gripper.preopen_pct 并确认到位。"""
        self._enter("open_gripper")
        try:
            self._motion_started = True
            self._stage = STAGE_OPEN
            self._step_gripper_open(self.params.gripper.preopen_pct, STAGE_OPEN,
                                    self.params.gripper.open_timeout_s)
        except (MotionError, KeyboardInterrupt) as exc:
            self._latch()
            if isinstance(exc, KeyboardInterrupt):
                raise StoppedByRequest("KeyboardInterrupt") from None
            raise
        finally:
            self._exit()

    def close_gripper(self) -> HoldingState:
        """小步闭合并做接触判据，返回依据电流与开度得到的夹持状态估计。

        HOLDING 只表示"电流与开度判据成立"，不是视觉确认（文档 4.3）。
        """
        self._enter("close_gripper")
        try:
            self._stage = STAGE_CLOSE
            self._motion_started = True
            return self._close_until_contact()
        except (MotionError, KeyboardInterrupt) as exc:
            if self._motion_started:
                self._latch()
            if isinstance(exc, KeyboardInterrupt):
                raise StoppedByRequest("KeyboardInterrupt") from None
            raise
        finally:
            self._exit()

    def wait_settled(self, target_deg: NDArray, timeout_s: float) -> bool:
        """到位判断原语：位置误差、速度与连续 dwell 同时满足才返回 True。"""
        q = self._as_joint_vector("target_deg", target_deg)
        if not timeout_s > 0.0:
            raise MotionError("INVALID_INPUT", f"timeout_s 必须为正，实际 {timeout_s}")
        return self.executor.wait_settled(q, float(timeout_s))

    def reset_fault(self) -> None:
        """解除实例锁存。

        文档 6.4：只在确认机械臂空爪、静止且设备正常后调用。这里能机器检查的是
        "通信正常 + 已经停在最后一次提交的目标附近"；是否真空爪、现场是否可继续
        由操作者确认。本函数不自动续接动作，也不移动回零。
        """
        if not self._fault_latched:
            return
        fb = self.executor.read_feedback()
        last = self.executor.state.last_submitted_deg
        if last is not None:
            drift = np.abs(np.asarray(fb.angles_deg, float) - last)
            hard = np.asarray(self.params.motion.following_error_hard_deg, float)
            if np.any(drift > hard):
                raise MotionError(
                    "CONFIG_INVALID",
                    f"复位被拒绝：当前位置与保持目标偏差 {np.round(drift, 2).tolist()}deg "
                    f"超过 {np.round(hard, 2).tolist()}deg，请先确认机械臂已经停稳",
                )
        self._fault_latched = False
        self._holding = HoldingState.EMPTY
        self._hold_pct = None
        self._motion_started = False
        self._stage = STAGE_CHECK
        self.executor.slip_reference_pct = None

    # ==================================================================
    # 2. 技能入口
    # ==================================================================

    def grasp_and_place(self, target: VisionInterface,
                        place_id: str = "default") -> GraspResult:
        """执行一次完整的抓取—放置。任何失败都以 GraspResult 返回，不抛异常。"""
        # 重入保护：一次调用没有返回之前不接受第二次（文档 1.2）。
        if self._in_call:
            raise RuntimeError("grasp_and_place 不接受重入调用；上一次调用尚未返回")
        self._in_call = True
        self._stage = STAGE_CHECK
        self._motion_started = False
        self._holding = HoldingState.UNKNOWN
        self._hold_pct = None
        self.executor.slip_reference_pct = None

        try:
            if not isinstance(place_id, str):
                return self._refuse(GraspStatus.INVALID_INPUT,
                                    f"place_id 必须是字符串，实际 {type(place_id).__name__}",
                                    str(place_id))
            if self._fault_latched:
                return self._refuse(
                    GraspStatus.CONFIG_INVALID,
                    "实例处于故障锁存状态；确认空爪、静止且现场可继续后调用 reset_fault",
                    place_id,
                )

            # ---- CHECK：输入格式、放置配置、读反馈、预规划并校验整条名义路线 ----
            try:
                position, yaw_deg = self._validate_target(target)
            except MotionError as exc:
                return self._refuse(GraspStatus(exc.status), exc.reason, place_id)
            try:
                place = self.params.place(place_id)
            except ParamsError as exc:
                # 未知 place_id：CONFIG_INVALID，臂不动（文档 3.4）。
                return self._refuse(GraspStatus.CONFIG_INVALID, str(exc), place_id)

            try:
                pre = self._preplan(position, yaw_deg, place)
            except (IkNotConverged, PlanViolation, CollisionViolation) as exc:
                # 预规划失败且臂未动：不锁存（文档 4.3）。
                return self._early_fail(exc, place_id)

            self._stage = STAGE_OPEN
            self._motion_started = True
            self._step_gripper_open(self.params.gripper.preopen_pct, STAGE_OPEN,
                                    self.params.gripper.open_timeout_s)

            self._stage = STAGE_APPROACH
            self._execute_step(pre.approach)
            self._stage = STAGE_DESCEND
            self._execute_step(pre.descend)

            # ---- CLOSE：五关节保持，小步闭合，接触判据通过后保持实际开度 ----
            self._stage = STAGE_CLOSE
            holding = self._close_until_contact()
            if holding is HoldingState.EMPTY:
                self._latch()
                return self._fault_result(GraspStatus.GRASP_MISS,
                                          "闭合到达空闭合区仍未建立有效接触", place_id)
            if holding is HoldingState.UNKNOWN:
                self._latch()
                return self._fault_result(GraspStatus.GRASP_UNCERTAIN,
                                          "闭合状态无法确定", place_id)

            # 闭爪后在静止状态用实测开度更新 TCP/持物关系与余下路径，并在执行前校验。
            q_close = self._measured_joints()
            post = self._replan_after_close(position, yaw_deg, place, q_close,
                                            float(self._hold_pct))

            self._stage = STAGE_LIFT
            self._execute_step(post.lift)
            self._stage = STAGE_TRANSFER
            self._execute_step(post.transfer)
            self._stage = STAGE_LOWER
            self._execute_step(post.lower)

            self._stage = STAGE_RELEASE
            self._release()
            self._stage = STAGE_RETREAT
            self._execute_step(post.retreat)
            self._stage = STAGE_RETURN
            self._execute_step(post.return_home)

            self._stage = STAGE_DONE
            self._holding = HoldingState.EMPTY
            return self._result(GraspStatus.SUCCESS, "抓取—放置完成", place_id,
                               HoldingState.EMPTY)

        except StoppedByRequest as exc:
            self._latch()
            return self._fault_result(GraspStatus.ABORTED, exc.reason, place_id)
        except KeyboardInterrupt:                       # 与 should_stop 走同一条停止路径
            self._latch()
            return self._fault_result(GraspStatus.ABORTED, "KeyboardInterrupt", place_id)
        except CollisionViolation as exc:
            return self._fault_result(GraspStatus.COLLISION, exc.reason, place_id)
        except ExecutionError as exc:
            return self._fault_result(self._status_of(exc), exc.reason, place_id)
        except (IkNotConverged, PlanViolation) as exc:
            return self._early_fail(exc, place_id)
        except MotionError as exc:
            return self._fault_result(self._status_of(exc), exc.reason, place_id)
        finally:
            self._in_call = False

    # ------------------------------------------------------------------
    # 2.1 输入检查（文档 3.2）
    # ------------------------------------------------------------------

    def _validate_target(self, target: VisionInterface) -> tuple[FloatArray, float]:
        """检查 shape (3,)、float64、有限值、角度范围与字段类型，然后复制 position。

        数据类冻结不等于 ndarray 内存只读，所以必须复制一份作为本次调用的固定输入，
        否则调用方可以在阻塞期间改写数组（文档 3.2）。
        """
        for name in ("position", "yaw_deg", "grade"):
            if not hasattr(target, name):
                raise MotionError("INVALID_INPUT", f"目标缺少字段 {name}")
        raw_pos = getattr(target, "position")
        if not isinstance(raw_pos, np.ndarray) or raw_pos.dtype != np.float64:
            raise MotionError(
                "INVALID_INPUT",
                f"position 必须是 np.float64 数组，实际 {type(raw_pos).__name__}"
                f"{getattr(raw_pos, 'dtype', '')}",
            )
        pos = np.asarray(raw_pos, dtype=np.float64)
        if pos.shape != (3,):
            raise MotionError("INVALID_INPUT", f"position shape 应为 (3,)，实际 {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise MotionError("INVALID_INPUT", f"position 含非有限值 {pos.tolist()}")
        yaw = getattr(target, "yaw_deg")
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float, np.floating)):
            raise MotionError("INVALID_INPUT", f"yaw_deg 应为数值，实际 {type(yaw).__name__}")
        if not np.isfinite(float(yaw)):
            raise MotionError("INVALID_INPUT", f"yaw_deg 非有限：{yaw}")
        # 长轴不区分头尾，约定区间 [-90,90)。越界说明视觉侧没有做归一化。
        if not (-90.0 <= float(yaw) < 90.0):
            raise MotionError(
                "INVALID_INPUT",
                f"yaw_deg={yaw} 不在 [-90,90)，请先用 wrap180() 归一化（文档 3.2）",
            )
        if not isinstance(getattr(target, "grade"), str):
            raise MotionError("INVALID_INPUT",
                              f"grade 应为字符串，实际 {type(getattr(target, 'grade')).__name__}")
        return pos.copy(), float(yaw)

    def _as_joint_vector(self, name: str, value: Any) -> FloatArray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (5,):
            raise MotionError("INVALID_INPUT", f"{name} shape 应为 (5,)，实际 {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise MotionError("INVALID_INPUT", f"{name} 含非有限值")
        return arr.copy()

    def _as_position(self, name: str, value: Any) -> FloatArray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise MotionError("INVALID_INPUT", f"{name} 应为 3 个有限浮点数，实际 {value}")
        return arr.copy()

    def _check_limits_or_raise(self, q_deg: NDArray) -> None:
        bad = self.limits.violations(q_deg)
        if bad:
            raise PlanViolation("目标关节角越限：" + "；".join(bad))

    # ------------------------------------------------------------------
    # 2.2 预规划（文档 5.3：开始动作前预规划完整名义路线）
    # ------------------------------------------------------------------

    def _preplan(self, position: FloatArray, yaw_deg: float,
                 place: PlacePose):
        """构造并校验整条名义路线。此时臂还没动，失败不锁存。

        文档 5.3 与 6.3 状态表 CHECK："开始动作前预规划完整名义路线……载入放置
        位姿并预规划、校验整条名义路线"。闭爪开度要到 CLOSE 才知道，文档同时
        要求预规划"包括已标定接触开度区间内的 TCP/持物变化包络"——这里按
        contact_gap_range_m 区间两端各规划并校验一遍抓取段与全部放置段，
        放置位姿不可达/越界/碰撞在臂未动时就会被拒绝。
        """
        g_open = self.params.gripper.preopen_pct
        # 抓取点：中心 + 物体系中表达的固定偏移，方向 = 长轴 + 已标定 yaw 偏移。
        _, spin, T_grasp = grasp_target_pose(position, yaw_deg, self.params)
        self._check_grasp_pose(T_grasp)
        T_above = T_grasp.copy()
        T_above[2, 3] += self.params.grasp.approach_height_m

        q_now = self._measured_joints()
        q_above = self.ik.solve(T_above, g_open, q_now).joints_deg
        # 对抓取位姿做 IK 预检：不可达在臂未动时就报 IK_FAILED。
        self.ik.solve(T_grasp, g_open, q_above)
        target_obb = self._target_obb(position, yaw_deg)

        approach = PlanStep(
            name="approach", stage=STAGE_APPROACH, contact_allowed=False,
            obb_for_node=lambda k, T, box=target_obb: box,
            build=lambda q_from: self._joint_route("approach", q_from, q_above, g_open),
        )
        descend = PlanStep(
            name="descend", stage=STAGE_DESCEND, contact_allowed=True, rebuildable=False,
            obb_for_node=lambda k, T, box=target_obb: box,
            build=lambda q_from: [make_cartesian_vertical_segment(
                "descend", self.model, self.ik, self.limits, q_from, g_open,
                float(T_grasp[2, 3] - self.model.fk_tcp(q_from, g_open)[2, 3]),
                self.params)],
        )
        # 预规划阶段就把这两段全部校验掉（含碰撞与 FK 残差）。
        for step in (approach, descend):
            for seg in step.build(q_now):
                self._validate([seg], step.stage, obb_for_node=step.obb_for_node,
                               contact_allowed=step.contact_allowed)

        # 抓取之后的整条路线：按接触开度区间两端各校验一遍。
        gp = self.params.gripper
        for gap in (gp.contact_gap_range_m[0], gp.contact_gap_range_m[1]):
            g_contact = gap_to_gripper_pct(float(gap), gp)
            q_contact = self.ik.solve(T_grasp, g_contact, q_above).joints_deg
            self._build_post_grasp_steps(position, yaw_deg, place, q_contact, g_contact)
        return _PrePlan(approach=approach, descend=descend, T_grasp=T_grasp,
                        q_above=q_above)

    def _check_grasp_pose(self, T_B_TCP: FloatArray) -> None:
        """抓取/放置 TCP 必须落在标定活动域内，且接近轴基本竖直。

        工作域是视觉判定可抓取条件与控制模型范围的共同依据（标定指南 2.2）。
        控制不重选目标也不改目标位置，所以越界只能拒绝。
        """
        bounds = np.asarray(self.params.workspace.tcp_bounds_m, float)
        p = np.asarray(T_B_TCP[:3, 3], float)
        if np.any(p < bounds[0] - 1e-9) or np.any(p > bounds[1] + 1e-9):
            raise PlanViolation(
                f"TCP {np.round(p, 4).tolist()} 超出 workspace.tcp_bounds_m "
                f"{np.round(bounds, 4).tolist()}"
            )
        tilt = approach_tilt_deg(T_B_TCP)
        if tilt > self.params.ik.tilt_tol_deg:
            raise PlanViolation(
                f"TCP 接近轴偏离竖直 {tilt:.3f}deg > ik.tilt_tol_deg {self.params.ik.tilt_tol_deg}deg"
            )

    def _target_obb(self, position: NDArray, yaw_deg: float) -> OrientedBox:
        return object_box(position, yaw_deg, self.params.grasp.object_envelope_m,
                          self.envelope.object_inflation_m)

    # ------------------------------------------------------------------
    # 2.3 闭爪后的路径更新（文档 3.4）
    # ------------------------------------------------------------------

    def _replan_after_close(self, position: FloatArray, yaw_deg: float, place: PlacePose,
                            q_close: NDArray, g_close: float) -> _PostGraspPlan:
        """闭爪后用实测开度重建并校验余下路径（文档 5.3 后半句）。

        CHECK 已按接触开度区间预规划校验过整条路线；这里按"实际 g"再做一次，
        两者共用同一构造函数。
        """
        return self._build_post_grasp_steps(position, yaw_deg, place, q_close, g_close)

    def _build_post_grasp_steps(self, position: FloatArray, yaw_deg: float,
                                place: PlacePose, q_ref: NDArray,
                                g_close: float) -> _PostGraspPlan:
        """以开度 g_close、参考姿态 q_ref 构造抓取—放置段并全部校验。"""
        # T_TCP_O 在一次持物过程中固定。
        T_TCP_O = holding_transform(self.model.fk_tcp(q_ref, g_close),
                                    object_pose(position, yaw_deg))
        # 放置 TCP 由放置物体中心与持物关系换算，不能直接把配置里的物体中心当 TCP。
        T_place = place_tcp_pose(object_pose(place.position, place.yaw_deg), T_TCP_O)
        self._check_grasp_pose(T_place)

        def held_obb(k: int, T_B_TCP: NDArray) -> OrientedBox:
            """持物包络：按该节点的 TCP 位姿与 T_TCP_O 换算出的物体中心与姿态。"""
            T_B_O = np.asarray(T_B_TCP, float) @ T_TCP_O
            yaw = float(np.degrees(np.arctan2(T_B_O[1, 0], T_B_O[0, 0])))
            return object_box(T_B_O[:3, 3], yaw, self.params.grasp.object_envelope_m,
                              self.envelope.object_inflation_m)

        g_release = self.params.gripper.release_pct
        above_place = T_place.copy()
        above_place[2, 3] += self.params.grasp.approach_height_m
        q_above_place = self.ik.solve(above_place, g_close, q_ref).joints_deg

        lift = PlanStep(
            name="lift", stage=STAGE_LIFT, contact_allowed=True, rebuildable=False,
            obb_for_node=held_obb,
            build=lambda q_from: [make_cartesian_vertical_segment(
                "lift", self.model, self.ik, self.limits, q_from, g_close,
                self.params.grasp.lift_height_m, self.params)],
        )
        transfer = PlanStep(
            name="transfer", stage=STAGE_TRANSFER, contact_allowed=True,
            obb_for_node=held_obb,
            build=lambda q_from: self._joint_route("transfer", q_from, q_above_place, g_close),
        )
        lower = PlanStep(
            name="lower", stage=STAGE_LOWER, contact_allowed=True, rebuildable=False,
            obb_for_node=held_obb,
            build=lambda q_from: [make_cartesian_vertical_segment(
                "lower", self.model, self.ik, self.limits, q_from, g_close,
                float(T_place[2, 3] - self.model.fk_tcp(q_from, g_close)[2, 3]), self.params)],
        )
        # RETREAT 用释放后的开度，并按"仍可能附着物体"的包络检查离开路径（文档 6.3）。
        retreat = PlanStep(
            name="retreat", stage=STAGE_RETREAT, contact_allowed=True, rebuildable=False,
            obb_for_node=held_obb,
            build=lambda q_from: [make_cartesian_vertical_segment(
                "retreat", self.model, self.ik, self.limits, q_from, g_release,
                self.params.grasp.retreat_height_m, self.params)],
        )
        return_home = PlanStep(
            name="return", stage=STAGE_RETURN, contact_allowed=False, obb_for_node=None,
            build=lambda q_from: self._joint_route("return", q_from,
                                                   self.params.workspace.home_joints_deg,
                                                   g_release),
        )
        for step in (lift, transfer, lower, retreat, return_home):
            for seg in step.build(q_ref):
                self._validate([seg], step.stage, obb_for_node=step.obb_for_node,
                               contact_allowed=step.contact_allowed)
        return _PostGraspPlan(lift=lift, transfer=transfer, lower=lower,
                             retreat=retreat, return_home=return_home)

    # ------------------------------------------------------------------
    # 2.4 步骤执行
    # ------------------------------------------------------------------

    def _execute_step(self, step: PlanStep) -> None:
        """读实测起点 → 生成该步 → 校验 → 逐段回放。"""
        q_from = self._measured_joints()
        segments = step.build(q_from)
        for seg in segments:
            self._validate([seg], step.stage, obb_for_node=step.obb_for_node,
                           contact_allowed=step.contact_allowed)
            self._play_with_offset_guard(seg, step)

    def _play_with_offset_guard(self, segment: MotionSegment, step: PlanStep) -> None:
        """起点偏差处理（文档 6.3 末段）。

        超出 start_position_tol_deg 时重新规划到同一目标，不直接跳到规划起点；
        竖直笛卡尔段做不到这件事，只能报错。规划与校验期间五关节与夹爪保持已有
        目标，所以这里不提交任何新指令。
        """
        q_measured = self._measured_joints()
        if check_start_offset(q_measured, segment.start_joints, self.params, segment.name):
            self.executor.play_segment(segment)
            return
        if not step.rebuildable:
            raise PlanViolation(
                f"{step.stage} 实测起点与规划起点偏差超过 "
                f"{np.round(self.params.motion.start_position_tol_deg, 2).tolist()}deg，"
                "而该段是固定竖直线段，无法从新起点保持同一条直线"
            )
        for seg in step.build(q_measured):
            self._validate([seg], step.stage, obb_for_node=step.obb_for_node,
                           contact_allowed=step.contact_allowed)
            self.executor.play_segment(seg)

    def _joint_route(self, prefix: str, q_from: NDArray, q_goal: NDArray,
                     gripper_pct: float) -> list[MotionSegment]:
        """q_from → 全部有序安全中转位 → q_goal（文档 6.3 APPROACH/TRANSFER/RETURN）。"""
        pts: list[FloatArray] = [np.asarray(q_from, float)]
        for w in self.params.workspace.safe_waypoints_deg:
            w = np.asarray(w, float)
            # 已经在某个中转位上就跳过，避免生成一串零位移段。
            if np.max(np.abs(w - pts[-1])) > 1e-9:
                pts.append(w)
        goal = np.asarray(q_goal, float)
        if np.max(np.abs(goal - pts[-1])) > 1e-9:
            pts.append(goal)
        return chain_joint_segments(prefix, pts, gripper_pct, self.params)

    def _validate(self, segments: Sequence[MotionSegment], stage: str, *,
                  obb_for_node: Callable[[int, NDArray], OrientedBox | None] | None = None,
                  contact_allowed: bool = False) -> None:
        """对段做文档 5.3 要求的完整校验。"""
        for seg in segments:
            validate_segment(seg, self.model, self.params, self.limits, self.checker,
                             object_obb_for_node=obb_for_node,
                             object_contact_allowed=contact_allowed)

    # ------------------------------------------------------------------
    # 2.5 夹爪阶段（技术文档第七节）
    # ------------------------------------------------------------------

    def _step_gripper_open(self, target_pct: float, stage: str, timeout_s: float) -> float:
        """按 open_speed_pct_s 小步张开，返回到位后的实测开度。"""
        gp = self.params.gripper
        q_hold = self._held_joint_target()
        cmd = float(self._current_gripper_pct())
        deadline = self.clock() + timeout_s
        step_pct = gp.open_speed_pct_s * self.executor.tick_s
        settled_since: float | None = None
        while True:
            self.executor.check_stop()
            # 每 tick 的变化量不超过速度上限，不一步跳到位（文档第七节"小步张开"）。
            cmd = min(target_pct, cmd + step_pct)
            self.executor.submit(q_hold, cmd)
            self.sleep(self.executor.tick_s)
            fb = self.executor.read_feedback()
            self.executor.monitor(fb)
            if (abs(fb.gripper_pct - target_pct) <= gp.settle_tol_pct
                    and abs(fb.gripper_speed_pct_s) <= gp.settle_speed_pct_s):
                if settled_since is None:
                    settled_since = self.clock()
                elif self.clock() - settled_since >= self.params.motion.settle_dwell_s:
                    return float(fb.gripper_pct)
            else:
                settled_since = None
            if self.clock() >= deadline:
                raise ExecutionError(
                    "TIMEOUT", f"{stage} 夹爪未能在 {timeout_s}s 内张开到 {target_pct:.1f}%"
                )

    def _close_until_contact(self) -> HoldingState:
        """闭爪 + 接触判据。

        每 tick 最多减少 close_speed_pct_s/FPS，五关节目标固定。判据全部成立才算
        接触：滤波后的夹爪电流均值连续 contact_dwell_s 超过 contact_current_ma、
        gap(actual_pct) 落在 contact_gap_range_m 内、actual_pct 大于
        empty_closed_pct + empty_tol_pct（空闭合端的受阻电流不计为目标接触）。
        """
        gp = self.params.gripper
        q_hold = self._held_joint_target()
        cmd = float(self._current_gripper_pct())
        deadline = self.clock() + gp.close_timeout_s
        step_pct = gp.close_speed_pct_s * self.executor.tick_s
        empty_region_pct = gp.empty_closed_pct + gp.empty_tol_pct
        currents: list[float] = []
        contact_since: float | None = None

        while True:
            self.executor.check_stop()
            # 允许略微越过空闭合端去"读到"空闭合状态，由下面的 empty 判定收尾。
            cmd = max(0.0, cmd - step_pct)
            self.executor.submit(q_hold, cmd)
            self.sleep(self.executor.tick_s)
            fb = self.executor.read_feedback()
            self.executor.monitor(fb)
            now = self.clock()

            # 均值窗口：只有凑满 window_samples 个采样才做滤波判断。
            currents.append(abs(fb.gripper_current_ma))
            if len(currents) > gp.window_samples:
                currents.pop(0)
            filtered = float(np.mean(currents)) if len(currents) >= gp.window_samples else 0.0

            try:
                gap = gripper_pct_to_gap(fb.gripper_pct, gp)
            except MotionError:
                # 实测开度落到标定表之外，说明闭合状态无法确定，不能瞎猜接触。
                self._holding = HoldingState.UNKNOWN
                return HoldingState.UNKNOWN

            in_gap = bool(gp.contact_gap_range_m[0] <= gap <= gp.contact_gap_range_m[1])
            above_empty = fb.gripper_pct > empty_region_pct
            if filtered > gp.contact_current_ma and in_gap and above_empty:
                if contact_since is None:
                    contact_since = now
                elif now - contact_since >= gp.contact_dwell_s:
                    return self._confirm_contact(fb.gripper_pct, q_hold)
            else:
                contact_since = None

            if fb.gripper_pct <= empty_region_pct:
                self._holding = HoldingState.EMPTY
                return HoldingState.EMPTY
            if now >= deadline:
                self._holding = HoldingState.UNKNOWN
                return HoldingState.UNKNOWN

    def _confirm_contact(self, hold_pct: float, q_hold: NDArray) -> HoldingState:
        """接触成立后保持本次实测开度，不继续加压；等待 hold_dwell_s 检查漂移与电流。"""
        gp = self.params.gripper
        self._hold_pct = float(hold_pct)
        self.executor.slip_reference_pct = float(hold_pct)
        start = self.clock()
        deadline = start + gp.hold_dwell_s + self.params.motion.settle_timeout_s
        while True:
            self.executor.check_stop()
            self.executor.submit(q_hold, self._hold_pct)
            fb = self.executor.read_feedback()
            self.executor.monitor(fb)
            if abs(fb.gripper_pct - self._hold_pct) > gp.slip_opening_change_pct:
                self._holding = HoldingState.UNKNOWN
                return HoldingState.UNKNOWN
            if abs(fb.gripper_current_ma) > gp.hard_current_ma:
                self._holding = HoldingState.UNKNOWN
                return HoldingState.UNKNOWN
            elapsed = self.clock() - start
            if elapsed >= gp.hold_dwell_s:
                self._holding = HoldingState.HOLDING
                return HoldingState.HOLDING
            if self.clock() >= deadline:
                self._holding = HoldingState.UNKNOWN
                return HoldingState.UNKNOWN
            self.sleep(min(self.params.timing.stop_poll_s, max(0.0, deadline - self.clock())))

    def _release(self) -> None:
        """松爪：小步张开到 release_pct，未到位或超时都算 PLACE_FAIL。

        之后等 release_dwell_s 让物体自行落定；只检查开度/速度与时间，不做视觉
        确认（文档 4.3、第七节）。
        """
        gp = self.params.gripper
        # 松爪后物体已不属于夹爪，滑移判据失去意义，先清掉基准。
        self.executor.slip_reference_pct = None
        try:
            self._step_gripper_open(gp.release_pct, STAGE_RELEASE, gp.open_timeout_s)
        except ExecutionError as exc:
            raise ExecutionError("PLACE_FAIL", f"松爪失败：{exc.reason}") from exc
        self._sleep_bounded(gp.release_dwell_s)
        self._holding = HoldingState.EMPTY

    # ------------------------------------------------------------------
    # 3. 共享小工具
    # ------------------------------------------------------------------

    def _enter(self, name: str) -> None:
        """公开原语的入口检查：不重入、不锁存。"""
        if self._in_call:
            raise RuntimeError(f"{name} 不能在另一次调用进行中重入")
        if self._fault_latched:
            raise MotionError("CONFIG_INVALID",
                              "实例处于故障锁存状态，需先调用 reset_fault（文档 4.3）")
        self._in_call = True
        self._motion_started = False
        self.executor.slip_reference_pct = None

    def _exit(self) -> None:
        self._in_call = False

    def _measured_joints(self) -> FloatArray:
        fb = self.executor.read_feedback()
        return np.asarray(fb.angles_deg, float).copy()

    def _current_gripper_pct(self) -> float:
        fb = self.executor.state.last_feedback
        if fb is None:
            fb = self.executor.read_feedback()
        return float(fb.gripper_pct)

    def _held_joint_target(self) -> FloatArray:
        """夹爪阶段要保持的五关节目标。

        用"已提交目标"而不是实测角：文档第七节要求五关节目标固定，重复提交同一
        条指令才能让舵机位置指令保持不变、不引入抖动。
        """
        last = self.executor.state.last_submitted_deg
        if last is not None:
            return np.asarray(last, float)
        return self._measured_joints()

    def _sleep_bounded(self, duration_s: float) -> None:
        """把等待拆成不大于 stop_poll_s 的片段，逐段检查停止（文档 6.1）。"""
        deadline = self.clock() + duration_s
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0.0:
                return
            self.executor.check_stop()
            self.sleep(min(self.params.timing.stop_poll_s, remaining))

    def _latch(self) -> None:
        """运动已开始后出错就锁存实例，需 reset_fault 才能再接调用（文档 4.3）。"""
        self._fault_latched = True

    def _require_vertical_only(self, T_now: FloatArray, T_target: FloatArray) -> None:
        """move_cartesian_top_down 的前置检查：XY 与姿态必须在容差内一致。"""
        tol = self.params.ik.position_tol_m
        dxy = float(np.hypot(*(np.asarray(T_target[:2, 3], float)
                               - np.asarray(T_now[:2, 3], float))))
        if dxy > tol:
            raise PlanViolation(
                f"move_cartesian_top_down 只允许竖直移动：水平偏移 {dxy * 1000:.2f}mm > "
                f"{tol * 1000:.2f}mm"
            )
        err = pose_error(T_now, T_target)
        if err.tilt_deg > self.params.ik.tilt_tol_deg or err.yaw_deg > self.params.ik.yaw_tol_deg:
            raise PlanViolation(f"move_cartesian_top_down 要求姿态保持不变：{err}")

    def _status_of(self, exc: MotionError) -> GraspStatus:
        """把内部 status 字符串映射回枚举；没登记的一律按执行错误处理。"""
        try:
            return GraspStatus(exc.status)
        except ValueError:
            return GraspStatus.TRACKING_ERROR

    # ---- 结果构造 ----

    def _refuse(self, status: GraspStatus, reason: str, place_id: str) -> GraspResult:
        """臂完全没动时的拒绝返回：recovery_required=False（文档 4.3）。"""
        return GraspResult(
            status=status, stage=self._stage, reason=reason, place_id=place_id,
            holding=HoldingState.UNKNOWN, recovery_required=False,
        )

    def _result(self, status: GraspStatus, reason: str, place_id: str,
                holding: HoldingState) -> GraspResult:
        return GraspResult(
            status=status, stage=self._stage, reason=reason, place_id=place_id,
            holding=holding, recovery_required=False,
        )

    def _early_fail(self, exc: MotionError, place_id: str) -> GraspResult:
        """配置/格式/预规划失败：臂没动就不锁存，动了才锁存（文档 4.3）。"""
        if self._motion_started:
            self._latch()
            return self._fault_result(self._status_of(exc), exc.reason, place_id)
        return GraspResult(
            status=self._status_of(exc), stage=self._stage, reason=exc.reason,
            place_id=place_id, holding=self._holding, recovery_required=False,
        )

    def _fault_result(self, status: GraspStatus, reason: str, place_id: str) -> GraspResult:
        """运动已开始后的失败：锁存实例、做一次保持提交，再返回结果（文档 4.3/6.4）。

        锁存必须发生在这里而不是各调用点：文档 4.3 规定"运动已开始后发生错误时锁存
        故障，recovery_required=True"，只要走到本函数就已经是运动开始后的错误。漏掉
        任何一条错误分支都会让实例带着未确认的现场状态接受下一次调用。

        通信正常时 hold_current 提交固定保持目标 → stage=FAULT_HOLD；
        拿不到可靠反馈或提交失败 → FAULT_UNCONTROLLED，明确告诉调用方"没有保持"，
        不把返回错误当成已经物理静止（标定指南 8.7）。
        """
        self._latch()
        held = self.executor.hold_position()
        self._stage = STAGE_FAULT_HOLD if held is not None else STAGE_FAULT_UNCONTROLLED
        return GraspResult(
            status=status, stage=self._stage, reason=reason, place_id=place_id,
            holding=self._holding, recovery_required=True,
        )


@dataclass
class _PrePlan:
    """预规划出的抓取前两步与关键位姿。"""

    approach: PlanStep
    descend: PlanStep
    T_grasp: FloatArray
    q_above: FloatArray


def normalize_object_yaw(angle_deg: float) -> float:
    """把长轴角归一化到 [-90,90)：wrap180(a)=(a+90)%180-90。供视觉侧与测试复用。"""
    return wrap180(angle_deg)
