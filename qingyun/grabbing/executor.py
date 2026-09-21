"""同步分段回放、到位判断与反馈监控。

规范来源：docs/机械臂运动控制模块技术文档.md 第 6.1、6.2 节。

本模块是"节拍的所有者"：30Hz 逐节点提交、每 tick 的停止检查/反馈读取/监控/
发送前复查、到位判定、跟踪误差与过流监控都在这里，且全部在调用线程内顺序执行，
不启动任何后台线程或定时器（文档 6.4：同步函数返回后不继续运行软件监控循环）。

时钟与 sleep 都可注入，这样单元测试可以在不真等 30 秒的情况下驱动整条
时序逻辑（包括"CPU 迟到"这类故障注入）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
    ) -> None:
        self.motor = motor
        self.params = params
        # 缺省永不停止（文档 4.1：should_stop 缺省返回 False）。
        self.should_stop = should_stop or (lambda: False)
        self.clock = clock
        self.sleep = sleep
        self.clock_ns = clock_ns
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
        """
        base = self.clock()
        prev_deg = segment.joints_deg[0]
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
