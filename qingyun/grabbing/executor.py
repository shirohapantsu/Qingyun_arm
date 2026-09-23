"""同步分段回放、到位判断与反馈监控。

规范来源：docs/机械臂运动控制模块技术文档.md 第 6.1、6.2 节。

本模块是"节拍的所有者"：30Hz 逐节点提交、每 tick 的停止检查/反馈读取/监控/
发送前复查、到位判定、跟踪误差与过流监控都在这里，且全部在调用线程内顺序执行，
不启动任何后台线程或定时器（文档 6.4：同步函数返回后不继续运行软件监控循环）。

时钟与 sleep 都可注入，这样单元测试可以在不真等 30 秒的情况下驱动整条
时序逻辑（包括"CPU 迟到"这类故障注入）。

本模块另有一个只读旁路：运动层内存 trace 的下降指令下发边界（P4 §2.8.2）。它通过
``MotionTraceSink`` 窄协议由上层注入，不新增线程、不新增舵机调用、不改变发送时序。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import (
    FloatArray,
    JointFeedback,
    MotorCommunicationError,
    MotorController,
    MotorLimitError,
    MotorStateError,
)
from configs.motion_params import MotionParams
from qingyun.grabbing.kinematics_ext import MotionError, StoppedByRequest
from qingyun.grabbing.trajectory import MotionSegment

# 单调时钟与睡眠的默认实现，测试时替换成假时钟。
_Clock = Callable[[], float]
_Sleep = Callable[[float], None]


class ExecutionError(MotionError):
    """执行期失败。status 直接对应 GraspStatus 的取值。"""


def _translate_motor_error(exc: Exception) -> ExecutionError:
    """把驱动层异常翻译成控制层状态码。

    技术文档 4.2/4.3：通信与设备状态问题归 COMM_ERROR；写入前被限限位拒绝属于
    我们提交了不该提交的指令，归 PLAN_INVALID（配置或规划错误，不是现场故障）。
    """
    if isinstance(exc, MotorLimitError):
        return ExecutionError("PLAN_INVALID", f"指令被驱动限位拒绝：{exc}")
    if isinstance(exc, (MotorCommunicationError, MotorStateError)):
        return ExecutionError("COMM_ERROR", f"电机通信/状态异常：{exc}")
    raise exc


# ---------------------------------------------------------------------------
# 0. 下降指令下发边界（P4 §2.8.2）
#
# 标定工具要区分"下降指令到底有没有下发"。这件事只有执行器知道：段节点是逐条
# 提交给驱动的，驱动可能正常返回（已下发）、可能在写入前整条拒绝（明确没下发）、
# 也可能抛通信异常（无法证明是否已写入）。控制层的事件类型 MotionTraceEvent 定义
# 在 arm_control.py，本模块只负责"在正确的边界上把三态事实如实交出去"，
# 不持有事件列表、不认识阶段名。
# ---------------------------------------------------------------------------

# descent_submit 事件的三态取值（与 arm_control.MotionTraceEvent.outcome 一致）。
DESCENT_SUBMITTED = "submitted"    # 驱动调用正常返回：下降指令已下发
DESCENT_REJECTED = "rejected"      # 驱动在写入前明确拒绝：可以证明没有下发
DESCENT_UNKNOWN = "unknown"        # 通信异常/中断：不能证明是否写入

# "非零下降"的位移阈值：与轨迹层"零位移段"的判定同源（trajectory 的
# _ZERO_DISPLACEMENT_M/_DEG 是模块私有常量，不 import 私有名，这里同值并注明出处）。
_DESCENT_ZERO_TOL_M = 1e-9
_DESCENT_ZERO_TOL_DEG = 1e-9


@runtime_checkable
class MotionTraceSink(Protocol):
    """执行器向运动层内存 trace 写入下降边界的窄协议（由 arm_control 的收集器实现）。

    协议只有"要不要观察"和"把这一条提交的结果记下来"两个方法：执行器不认识阶段名、
    不构造事件、不持有事件列表，标定工具也不通过本协议访问控制器的内部状态。
    """

    def watches_descent(self) -> bool:
        """当前是否正在观察"首个非零下降节点"（不需要观察时执行器零额外计算）。"""
        ...

    def note_descent_submit(self, outcome: str) -> None:
        """登记一次下降指令下发结果，取值为上面三个常量之一。"""
        ...


def first_descent_node_index(segment: MotionSegment) -> int | None:
    """返回该段中"首个非零下降节点"的下标；该段不含下降节点时返回 None。

    判定口径（P4 §2.8.2"首个非零下降节点"的实现，测试逐条钉住）：

    1. **符号看 TCP 高度**：笛卡尔段的 ``tcp_targets[k][2,3]`` 是第 k 个节点命令的
       基座 Z。以段起点节点 0 的高度为基准，首个满足 ``z0 - z_k > 1e-9 m`` 的节点就是
       "开始下降"的那个节点。节点 0 自身位移为零，永远不算；因此"起点已经在目标高度
       上"的段（轨迹层按零位移只出一个节点，``nodes <= 1``）没有下降节点。
    2. **幅值看阈值**：位移不超过 ``_DESCENT_ZERO_TOL_M`` 的节点不计为下降。因为终点
       相对起点的累计位移等于全长，任何真实下降段必然存在这样的节点（最迟是最后一个
       节点），所以"找不到下降节点"只发生在零位移段与竖直向上段。
    3. **上升段不算下降**：``dz > 0``（比如反常地把抓取点算到当前高度之上）时全部节点
       都在基准之上，返回 None——不为"从未下发下降指令"伪造 submitted 事件。
    4. **兜底（无 tcp_targets 的关节段）**：DESCEND 在控制层只由竖直笛卡尔段构造；若将来
       某段没有 TCP 目标可看，退化为"相对段起点关节角变化首次超过 1e-9 deg 的节点"，
       零位移段同样返回 None。
    """
    joints = np.asarray(segment.joints_deg, dtype=np.float64)
    nodes = int(joints.shape[0])
    if nodes <= 1:
        return None
    tcp = getattr(segment, "tcp_targets", None)
    if tcp is not None:
        arr = np.asarray(tcp, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == nodes and arr.shape[1:] == (4, 4):
            drop = arr[0, 2, 3] - arr[:, 2, 3]
            below = np.nonzero(drop > _DESCENT_ZERO_TOL_M)[0]
            return int(below[0]) if below.size else None
    moved = np.max(np.abs(joints - joints[0]), axis=1)
    over = np.nonzero(moved > _DESCENT_ZERO_TOL_DEG)[0]
    return int(over[0]) if over.size else None


def classify_descent_submit_error(exc: BaseException) -> str:
    """把"下发边界抛出的异常"翻译成 rejected / unknown 两态之一。

    规格只允许两种非成功结局：驱动在**写入前**明确拒绝（``MotorLimitError``：整条命令
    被限位/合法性检查挡下，可以证明总线上没有下发过这条下降指令）记 rejected；其余一律
    unknown——包括 ``MotorCommunicationError``（超时/部分写入）、``MotorStateError``
    （设备未就绪，无法核对是否已经落笔）、以及任何未被翻译的异常与 KeyboardInterrupt
    （中断点位置未知）。宁可把"不确定"记成 unknown 交给操作者补证，也不能把它当成
    rejected/False，那等于悄悄把这条试验从下降分母里抹掉（P4 §2.8.2）。
    """
    cause = exc.__cause__
    if isinstance(exc, MotorLimitError) or isinstance(cause, MotorLimitError):
        return DESCENT_REJECTED
    # _translate_motor_error 已把 MotorLimitError 归入 PLAN_INVALID，据此认出同一事实。
    if isinstance(exc, ExecutionError) and exc.status == "PLAN_INVALID":
        return DESCENT_REJECTED
    return DESCENT_UNKNOWN


@dataclass
class ExecutorState:
    """执行器的可观测运行状态，供 arm_control 与标定工具记录。"""

    last_submitted_deg: FloatArray | None = None
    last_submitted_pct: float | None = None
    last_feedback: JointFeedback | None = None
    sequence: int = 0                       # 已接受的最新反馈 sequence
    ticks_played: int = 0
    peak_following_error_deg: FloatArray = field(default_factory=lambda: np.zeros(5))
    peak_joint_current_ma: FloatArray = field(default_factory=lambda: np.zeros(5))


class SyncExecutor:
    """按 1/FPS 逐节点提交轨迹段，并在每个 tick 做停止、反馈与监控检查。"""

    def __init__(
        self,
        motor: MotorController,
        params: MotionParams,
        should_stop: Callable[[], bool] | None = None,
        *,
        clock: _Clock = time.monotonic,
        sleep: _Sleep = time.sleep,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        motion_trace: MotionTraceSink | None = None,
    ) -> None:
        self.motor = motor
        self.params = params
        # 缺省永不停止（文档 4.1：should_stop 缺省返回 False）。
        self.should_stop = should_stop or (lambda: False)
        self.clock = clock
        self.sleep = sleep
        self.clock_ns = clock_ns
        # 运动层内存 trace 的写入端（P4 §2.8.2）。None 表示不观察：回放路径一次额外的
        # 时钟/总线调用都不会发生，运动指令与返回值与未接入 trace 时逐字节相同。
        self.motion_trace = motion_trace
        self.tick_s = 1.0 / params.timing.fps
        self.state = ExecutorState()
        # 持物滑移基准：由上层在闭爪成功后写入实际保持开度，None 表示当前不持物。
        # 技术文档第七节要求持物期间监控相对该基准的开度变化。
        self.slip_reference_pct: float | None = None
        # 跟踪误差"持续超限"的起始时刻；None 表示当前未超限。
        self._follow_violation_since: float | None = None
        # 规划期反馈巡检的上一次时刻（文档 6.1 planning_poll_s）。
        self._last_planning_poll_s: float = clock()
        self._first_feedback = True

    # ------------------------------------------------------------------
    # 1. 停止与反馈读取
    # ------------------------------------------------------------------

    def check_stop(self) -> None:
        """单个停止检查点。should_stop 必须是快速、无阻塞的同步函数。"""
        if self.should_stop():
            raise StoppedByRequest("should_stop 在检查点返回 True")

    def read_feedback(self) -> JointFeedback:
        """读一次反馈并做时效校验（文档 6.1：反馈跨度、年龄与 sequence 必须有效）。

        年龄从 sample_start_ns 算起而不是读取结束时刻（标定指南 8.3），否则一次
        很慢的读回会被自己的结束时间戳"洗白"。
        """
        try:
            fb = self.motor.get_feedback()
        except Exception as exc:              # 翻译成状态码，不吞原始异常
            raise _translate_motor_error(exc) from exc

        now_ns = self.clock_ns()
        age_s = (now_ns - fb.sample_start_ns) / 1e9
        span_s = (fb.sample_end_ns - fb.sample_start_ns) / 1e9
        if age_s > self.params.timing.max_feedback_age_s:
            raise ExecutionError(
                "TIMEOUT", f"反馈过期：年龄 {age_s * 1000:.1f}ms > {self.params.timing.max_feedback_age_s * 1000:.1f}ms"
            )
        if span_s > self.params.timing.max_feedback_span_s:
            raise ExecutionError(
                "TIMEOUT", f"单次采样跨度过大：{span_s * 1000:.1f}ms > {self.params.timing.max_feedback_span_s * 1000:.1f}ms"
            )
        if span_s < 0.0:
            raise ExecutionError("COMM_ERROR", f"采样时间戳倒置：start={fb.sample_start_ns} end={fb.sample_end_ns}")
        # 第一条反馈只建立基准，之后要求 sequence 严格递增（重复/回退说明读到了缓存）。
        if not self._first_feedback and fb.sequence <= self.state.sequence:
            raise ExecutionError(
                "COMM_ERROR", f"反馈 sequence 未递增：{fb.sequence} <= {self.state.sequence}"
            )
        if not (
            np.all(np.isfinite(fb.angles_deg))
            and np.all(np.isfinite(fb.speeds_deg_s))
            and np.all(np.isfinite(fb.currents_ma))
            and np.isfinite(fb.gripper_pct)
        ):
            raise ExecutionError("COMM_ERROR", "反馈含非有限数值")
        self._first_feedback = False
        self.state.sequence = fb.sequence
        self.state.last_feedback = fb
        return fb

    def sleep_until(self, deadline_s: float) -> None:
        """分段 sleep 到 deadline，逐段检查停止（文档 6.1：等待函数分成不大于 stop_poll_s 的片段）。"""
        poll = self.params.timing.stop_poll_s
        while True:
            remaining = deadline_s - self.clock()
            if remaining <= 0.0:
                return
            self.check_stop()
            self.sleep(min(poll, remaining))

    # ------------------------------------------------------------------
    # 2. 监控
    # ------------------------------------------------------------------

    def monitor(self, fb: JointFeedback) -> None:
        """把反馈与"上一条已提交目标"比较（文档 6.2），避免把新目标的自然滞后误判成故障。"""
        if self.state.last_submitted_deg is None:
            return
        target = np.asarray(self.state.last_submitted_deg, float)
        err = np.abs(np.asarray(fb.angles_deg, float) - target)
        self.state.peak_following_error_deg = np.maximum(self.state.peak_following_error_deg, err)
        self.state.peak_joint_current_ma = np.maximum(
            self.state.peak_joint_current_ma, np.abs(np.asarray(fb.currents_ma, float))
        )
        now = self.clock()

        hard = np.asarray(self.params.motion.following_error_hard_deg, float)
        over_hard = np.where(err > hard)[0]
        if over_hard.size:
            raise ExecutionError(
                "TRACKING_ERROR",
                "跟踪误差立即超限：" + "，".join(
                    f"关节{i}={err[i]:.2f}>{hard[i]:.2f}deg" for i in over_hard),
            )
        cur = np.abs(np.asarray(fb.currents_ma, float))
        hard_cur = np.asarray(self.params.joints.hard_current_ma, float)
        over_cur = np.where(cur > hard_cur)[0]
        if over_cur.size:
            raise ExecutionError(
                "TRACKING_ERROR",
                "关节电流立即超限：" + "，".join(
                    f"关节{i}={cur[i]:.0f}>{hard_cur[i]:.0f}mA" for i in over_cur),
            )
        if abs(fb.gripper_current_ma) > self.params.gripper.hard_current_ma:
            raise ExecutionError(
                "TRACKING_ERROR",
                f"夹爪电流立即超限：{abs(fb.gripper_current_ma):.0f}>"
                f"{self.params.gripper.hard_current_ma:.0f}mA",
            )
        # 持物滑移：开度相对接触保持值漂移过大。这是异常指示，不是物体运动的直接
        # 测量（技术文档第七节），所以只用于停止，不用于估计物体有没有掉。
        if self.slip_reference_pct is not None:
            drift = abs(fb.gripper_pct - self.slip_reference_pct)
            if drift > self.params.gripper.slip_opening_change_pct:
                raise ExecutionError(
                    "GRIP_SLIP",
                    f"持物开度漂移 {drift:.2f}% > "
                    f"{self.params.gripper.slip_opening_change_pct:.2f}%"
                    f"（保持基准 {self.slip_reference_pct:.2f}%）",
                )

        soft = np.asarray(self.params.motion.following_error_deg, float)
        if np.any(err > soft):
            if self._follow_violation_since is None:
                self._follow_violation_since = now
            elif now - self._follow_violation_since >= self.params.motion.following_error_dwell_s:
                raise ExecutionError(
                    "TRACKING_ERROR",
                    f"跟踪误差持续超限超过 {self.params.motion.following_error_dwell_s}s",
                )
        else:
            self._follow_violation_since = None

    def check_command_step(self, prev_deg: NDArray, next_deg: NDArray,
                           dt_s: float, name: str) -> None:
        """发送前复查本次实际步长与速度（文档 6.1 第 4 步）。

        预规划已经查过一遍，这里是防止"实际发送的序列"和"被校验的序列"不一致，
        例如重新规划后忘了重新校验。夹爪开合速度由小步开合逻辑单独约束，不在这里查。
        """
        step = np.abs(np.asarray(next_deg, float) - np.asarray(prev_deg, float))
        if np.any(step > np.asarray(self.params.motion.max_command_step_deg, float) + 1e-9):
            raise ExecutionError("PLAN_INVALID", f"{name} 实际发送步长超限 {np.round(step, 3).tolist()}")
        if dt_s > 0.0:
            vel = step / dt_s
            if np.any(vel > np.asarray(self.params.motion.max_velocity_deg_s, float) + 1e-9):
                raise ExecutionError("PLAN_INVALID", f"{name} 实际发送速度超限 {np.round(vel, 2).tolist()}")

    # ------------------------------------------------------------------
    # 3. 规划检查点
    # ------------------------------------------------------------------

    def planning_checkpoint(self) -> None:
        """规划期间的检查点：每路点、每次 IK 迭代之间调用（文档 6.1）。

        做两件事：
          1. 检查停止请求；
          2. 距上次反馈检查达到 timing.planning_poll_s 时读一次反馈，检查机械臂在
             保持姿态下有没有被外力推动、有没有过流、设备是否还正常。
        单次不可中断的底层求解调用无法在这里抢占，所以规划耗时必须实测后计入
        停止延迟预算（文档 6.1 末段、标定指南 8.5）。
        """
        self.check_stop()
        now = self.clock()
        if now - self._last_planning_poll_s < self.params.timing.planning_poll_s:
            return
        self._last_planning_poll_s = now
        fb = self.read_feedback()
        if self.state.last_submitted_deg is None:
            return
        drift = np.abs(np.asarray(fb.angles_deg, float) - self.state.last_submitted_deg)
        hard = np.asarray(self.params.motion.following_error_hard_deg, float)
        over = np.where(drift > hard)[0]
        if over.size:
            raise ExecutionError(
                "TRACKING_ERROR",
                "规划期间位置保持偏差超限：" + "，".join(
                    f"关节{i}={drift[i]:.2f}>{hard[i]:.2f}deg" for i in over),
            )
        cur = np.abs(np.asarray(fb.currents_ma, float))
        hard_cur = np.asarray(self.params.joints.hard_current_ma, float)
        if np.any(cur > hard_cur):
            raise ExecutionError("TRACKING_ERROR", f"规划期间电流超限 {np.round(cur, 0).tolist()}mA")

    # ------------------------------------------------------------------
    # 4. 回放
    # ------------------------------------------------------------------

    def submit(self, joints_deg: NDArray, gripper_pct: float) -> None:
        """提交一条指令并记录为"已提交目标"。"""
        try:
            self.motor.send_action(np.asarray(joints_deg, float), float(gripper_pct))
        except Exception as exc:
            raise _translate_motor_error(exc) from exc
        self.state.last_submitted_deg = np.array(joints_deg, float)
        self.state.last_submitted_pct = float(gripper_pct)

    def play_segment(self, segment: MotionSegment) -> None:
        """按节拍回放一段轨迹，段末显式发终点并等待到位。

        时序规则（文档 6.1）：
          * 按顺序逐节点提交，不按墙钟跳索引；
          * 每 tick：等待计划时刻 → 停止检查 → 读反馈并校验 → 监控上一条已发目标
            → 检查本次实际步长/速度 → 发送；
          * 轻微迟到就把本段后续时刻整体顺延，保证两次发送间隔不小于一个 tick；
          * 迟到超过 timing.max_tick_lateness_s 直接停止，不补发积压节点。

        trace 钩子（P4 §2.8.2）：只有当上层收集器正在观察下降边界时才多做一次纯计算
        的节点定位（``first_descent_node_index``），并在**那一个节点**的提交处记录
        submitted/rejected/unknown。提交次数、发送时机、读回次数与不接 trace 时完全一致。
        """
        base = self.clock()
        prev_deg = segment.joints_deg[0]
        sink = self.motion_trace
        descent_node = -1
        if sink is not None and sink.watches_descent():
            idx = first_descent_node_index(segment)
            if idx is not None:
                descent_node = idx
        for k in range(segment.nodes):
            due = base + float(segment.time_s[k])
            self.sleep_until(due)
            late = self.clock() - due
            if late > self.params.timing.max_tick_lateness_s:
                raise ExecutionError(
                    "TIMEOUT",
                    f"{segment.name} 第 {k} 节点迟到 {late * 1000:.1f}ms > "
                    f"{self.params.timing.max_tick_lateness_s * 1000:.1f}ms，停止且不补发积压节点",
                )
            if late > 0.0:
                # 顺延：把时间基准整体推后，后续 due 相应变晚，发送间隔保持 >= 1 tick。
                base += late
            self.check_stop()
            fb = self.read_feedback()
            self.monitor(fb)
            if k > 0:
                self.check_command_step(
                    prev_deg, segment.joints_deg[k], self.tick_s, segment.name
                )
            # 发送前再次检查停止与超时（文档 6.1 末段）。
            self.check_stop()
            if k == descent_node:
                self._submit_descent_node(sink, segment, k)
            else:
                self.submit(segment.joints_deg[k], segment.gripper_pct)
            prev_deg = segment.joints_deg[k]
            self.state.ticks_played += 1

        settled = self.wait_settled(segment.end_joints, self.params.motion.settle_timeout_s)
        if not settled:
            raise ExecutionError(
                "TIMEOUT",
                f"{segment.name} 段末未到位（容差 "
                f"{np.round(self.params.motion.settle_position_tol_deg, 2).tolist()}deg，"
                f"时限 {self.params.motion.settle_timeout_s}s）",
            )

    def _submit_descent_node(self, sink: MotionTraceSink, segment: MotionSegment,
                             k: int) -> None:
        """在下降边界的提交处如实登记三态，然后把结果原样交回原调用路径。

        成功与异常都不改变控制层看到的行为：异常照旧向上抛，由技能入口按原契约转成
        GraspResult；这里只是"顺路看一眼驱动的答复"。不重试、不改写、不吞异常，
        因此 trace 开/关两种情况下的运动指令序列与返回值逐字段相同。
        """
        try:
            self.submit(segment.joints_deg[k], segment.gripper_pct)
        except BaseException as exc:                     # 记录后原样抛出，不吞任何异常
            sink.note_descent_submit(classify_descent_submit_error(exc))
            raise
        sink.note_descent_submit(DESCENT_SUBMITTED)

    # ------------------------------------------------------------------
    # 5. 到位判断
    # ------------------------------------------------------------------

    def wait_settled(self, target_deg: NDArray, timeout_s: float) -> bool:
        """五关节到位判断（文档 6.2）。

        条件：误差不超过 motion.settle_position_tol_deg、速度不超过
        motion.settle_velocity_tol_deg_s，并且连续满足 motion.settle_dwell_s。
        返回 False 表示 timeout_s 内没凑齐 dwell；调用方决定是报错还是重新规划。
        期间仍然做停止检查、反馈校验与跟踪/过流监控。
        """
        target = np.asarray(target_deg, float)
        deadline = self.clock() + timeout_s
        satisfied_since: float | None = None
        while True:
            self.check_stop()
            fb = self.read_feedback()
            self.monitor(fb)
            now = self.clock()
            in_pos = np.all(np.abs(np.asarray(fb.angles_deg, float) - target)
                            <= np.asarray(self.params.motion.settle_position_tol_deg, float))
            in_vel = np.all(np.abs(np.asarray(fb.speeds_deg_s, float))
                            <= np.asarray(self.params.motion.settle_velocity_tol_deg_s, float))
            if in_pos and in_vel:
                if satisfied_since is None:
                    satisfied_since = now
                elif now - satisfied_since >= self.params.motion.settle_dwell_s:
                    return True
            else:
                satisfied_since = None      # 连续性被打断，重新累计 dwell
            if now >= deadline:
                return False
            self.sleep(min(self.params.timing.stop_poll_s, deadline - now))

    def hold_position(self) -> JointFeedback | None:
        """停止后的一次保持提交（文档 6.4）。

        通信正常时调用 hold_current 一次，捕获并提交固定保持目标；通信失败则返回
        None 让调用方报告"保持未提交"，不无限重试。函数返回后本执行器不再有任何
        循环，位置保持依赖已提交的舵机位置目标。
        """
        try:
            return self.motor.hold_current()
        except Exception:
            return None
