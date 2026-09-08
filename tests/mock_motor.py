"""单元测试用的机械臂时间源与 Mock 电机驱动。

规范来源：docs/机械臂运动控制模块技术文档.md 第八节必测项 5：
    "Mock 覆盖滞后、过流、空闭合端电流、抓空、开合超时、通信失败、过期反馈和 CPU 迟到。"

设计要点：
    * SimTime 提供可注入的单调时钟与 sleep，sleep 直接推进虚拟时间。这样
      30Hz、几十秒的完整抓取流程能在毫秒级真实时间内跑完，并且完全可复现。
    * MockMotorController 实现同一个 MotorController Protocol，用一阶滞后模型
      模拟舵机跟随，用"闭到物体标定开口就停住 + 电流上升"模拟接触，并且可以
      逐项注入故障。
    * 所有故障都是显式开关，绝不在正常路径里偷偷放宽判据。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from configs.common_interface import (
    JointFeedback,
    MotorCommunicationError,
    MotorLimitError,
    MotorStateError,
)
from configs.motion_params import MotionParams, effective_joint_limits


class SimTime:
    """可推进的虚拟单调时钟。"""

    def __init__(self, start_s: float = 1000.0) -> None:
        self.now_s = float(start_s)
        # 每次 sleep 额外附加的 CPU 迟到量，用于注入"迟到超过 max_tick_lateness_s"。
        self.extra_lateness_s = 0.0

    def sleep(self, dt: float) -> None:
        self.now_s += max(0.0, float(dt)) + self.extra_lateness_s

    def monotonic(self) -> float:
        return self.now_s

    def monotonic_ns(self) -> int:
        return int(self.now_s * 1e9)


class MockMotorController:
    """带故障注入的 SO-ARM101 总线仿真。

    物理量语义严格照 common_interface：angles_deg 是 URDF 坐标 deg，夹爪
    0%=标定闭合端、100%=标定张开端，速度正负分别对应张开/闭合。
    """

    def __init__(
        self,
        params: MotionParams,
        time: SimTime,
        *,
        joints_deg: np.ndarray | None = None,
        gripper_pct: float = 50.0,
        lag_tau_s: float = 0.05,
        gripper_lag_tau_s: float = 0.05,
        noise_deg: float = 0.0,
        current_per_deg_error_ma: float = 25.0,
        blocked_current_ma: float = 900.0,
        current_per_pct_shortfall_ma: float = 35.0,
    ) -> None:
        self.params = params
        self.time = time
        lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
        self._lower = lower
        self._upper = upper

        q0 = np.array(params.workspace.home_joints_deg if joints_deg is None else joints_deg,
                      dtype=np.float64)
        self.q = q0.copy()                      # 实际位置
        self.q_cmd = q0.copy()                  # 已提交目标
        self.g = float(gripper_pct)
        self.g_cmd = float(gripper_pct)
        self._g_vel = 0.0                       # %/s，正为张开
        self._q_prev = q0.copy()

        # --- 仿真参数 ---
        self.lag_tau_s = lag_tau_s
        self.gripper_lag_tau_s = gripper_lag_tau_s
        self.noise_deg = noise_deg
        self.current_per_deg_error_ma = current_per_deg_error_ma
        # 夹爪电流：shortfall * current_per_pct_shortfall_ma，上限截到 blocked_current_ma。
        self.blocked_current_ma = blocked_current_ma
        self.current_per_pct_shortfall_ma = current_per_pct_shortfall_ma

        # --- 故障注入开关（默认全部关闭）---
        self.object_gap_m: float | None = None      # None=夹爪空夹可一路闭合；有值=有物体
        self.comm_fail_on_read: bool = False
        self.comm_fail_on_write: bool = False
        self.state_fault: bool = False
        self.freeze_feedback: bool = False          # sequence 不再递增，模拟读到缓存
        self.expire_feedback: bool = False          # 采样时间戳人为变旧
        self.stuck_joints: bool = False             # 完全不跟随指令：停住但未到位
        self.hard_overcurrent_joints: bool = False  # 关节电流恒定超硬限
        self.gripper_overcurrent: bool = False
        # 持物滑移注入：物体往外滑时，夹爪的堵转点跟着往外移，所以这里累加的是
        # "堵转开度偏移"而不是直接加在实际开度上。后者会被一阶滞后拉回命令值，
        # 稳态漂移只有 rate*tau 那么大，永远碰不到 slip_opening_change_pct 阈值。
        self.grip_slip_rate_pct_s: float = 0.0
        self._slip_offset_pct: float = 0.0
        self.command_count = 0
        self.read_count = 0
        self._sequence = 0
        self._last_write_s: float | None = None

    # ------------------------------------------------------------------
    # MotorController 协议
    # ------------------------------------------------------------------

    def send_action(self, joints_deg: np.ndarray, gripper_pct: float) -> None:
        """校验后单次提交。整条拒绝，不静默裁剪（文档 4.2）。"""
        if self.state_fault:
            raise MotorStateError("Mock：设备未就绪/报告硬件故障")
        if self.comm_fail_on_write:
            raise MotorCommunicationError("Mock：写超时")
        q = np.asarray(joints_deg, dtype=np.float64)
        if q.shape != (5,):
            raise MotorLimitError(f"Mock：joints_deg shape 应为 (5,)，实际 {q.shape}")
        if not np.all(np.isfinite(q)) or not np.isfinite(gripper_pct):
            raise MotorLimitError("Mock：指令含非有限值")
        if np.any(q < self._lower - 1e-9) or np.any(q > self._upper + 1e-9):
            raise MotorLimitError(f"Mock：关节指令越限 {np.round(q, 2).tolist()}")
        # 可用开度区间取 gap_table 的覆盖范围；三张表共同覆盖区间已由配置加载器校验。
        gp = self.params.gripper
        lo_pct = float(gp.gap_table[0].gripper_pct)
        hi_pct = float(gp.gap_table[-1].gripper_pct)
        if not (lo_pct - 1e-9 <= float(gripper_pct) <= hi_pct + 1e-9):
            raise MotorLimitError(f"Mock：夹爪开度 {gripper_pct} 超出标定表覆盖区间")
        self.q_cmd = q.copy()
        self.g_cmd = float(gripper_pct)
        self.command_count += 1
        self._advance(self.time.monotonic() - (self._last_write_s or self.time.monotonic()))
        self._last_write_s = self.time.monotonic()

    def get_feedback(self) -> JointFeedback:
        if self.state_fault:
            raise MotorStateError("Mock：设备未就绪/报告硬件故障")
        if self.comm_fail_on_read:
            raise MotorCommunicationError("Mock：读超时/部分读回")
        self.read_count += 1
        # 读一次也要推进物理仿真：sleep 只在控制器侧发生。
        self._advance(1.0 / self.params.timing.fps)
        start_ns = self.time.monotonic_ns()
        end_ns = start_ns + int(self.params.timing.max_feedback_span_s * 1e6 * 0.5)
        if self.expire_feedback:
            # 人为把采样时刻推旧，触发 timing.max_feedback_age_s 检查。
            start_ns -= int(3.0 * self.params.timing.max_feedback_age_s * 1e9)
        if self.freeze_feedback:
            seq = self._sequence
        else:
            self._sequence += 1
            seq = self._sequence
        angles = self.q.copy()
        if self.noise_deg:
            angles = angles + self._deterministic_noise()
        speeds = (angles - self._q_prev) * self.params.timing.fps
        self._q_prev = angles.copy()
        currents = np.abs(angles - self.q_cmd) * self.current_per_deg_error_ma
        if self.hard_overcurrent_joints:
            currents = np.full(5, max(self.params.joints.hard_current_ma) * 2.0)
        g_current = self._gripper_current_ma()
        return JointFeedback(
            angles_deg=angles,
            speeds_deg_s=speeds,
            currents_ma=currents,
            gripper_pct=self.g,
            gripper_speed_pct_s=self._g_vel,
            gripper_current_ma=g_current,
            sample_start_ns=start_ns,
            sample_end_ns=end_ns,
            sequence=seq,
        )

    def hold_current(self) -> JointFeedback:
        """有界地读一次并写一次相同目标（文档 4.2）。"""
        fb = self.get_feedback()
        self.send_action(fb.angles_deg, fb.gripper_pct)
        return fb

    # ------------------------------------------------------------------
    # 物理仿真
    # ------------------------------------------------------------------

    def _advance(self, dt: float) -> None:
        """把实际位置向已提交目标推进 dt 秒（一阶滞后）。"""
        if dt <= 0.0:
            return
        if self.stuck_joints:
            return                              # 停住但不跟进，专门测"未到位"分支
        alpha = 1.0 - math.exp(-dt / max(1e-9, self.lag_tau_s))
        self.q = self.q + (self.q_cmd - self.q) * alpha
        # 夹爪：向目标靠拢，但闭到物体开口就停住。
        g_alpha = 1.0 - math.exp(-dt / max(1e-9, self.gripper_lag_tau_s))
        target_g = self.g_cmd
        if self.object_gap_m is not None:
            # 滑移让有效开口变大：堵转点向外偏移，夹爪被物体顶开并停在那里。
            target_g = max(target_g, self._pct_for_gap(self.object_gap_m) + self._slip_offset_pct)
        prev = self.g
        self.g = prev + (target_g - prev) * g_alpha
        if self.grip_slip_rate_pct_s:
            self._slip_offset_pct = min(100.0, self._slip_offset_pct
                                        + self.grip_slip_rate_pct_s * dt)
        self._g_vel = (self.g - prev) / dt

    def _pct_for_gap(self, gap_m: float) -> float:
        """gap_table 反查开度：gap 严格递增所以唯一。"""
        gp = self.params.gripper
        gaps = np.array([s.gap_m for s in gp.gap_table])
        pcts = np.array([s.gripper_pct for s in gp.gap_table])
        return float(np.interp(gap_m, gaps, pcts))

    def _gripper_current_ma(self) -> float:
        """夹爪电流模型：与"命令继续压而实际不再动"的差额成正比。

        自由跟随时滞后造成的差额很小（≈ speed*tau），电流只有几十 mA；被物体或被
        空闭合端挡住后，命令按 close_speed_pct_s 继续下降而实际开度不动，差额线性
        增长，电流随之上升。注意空闭合端同样会堵转出大电流——把它和目标接触区分开
        的是判据里的开度下限 empty_closed_pct+empty_tol_pct，不是电流本身（文档第七节）。
        """
        if self.gripper_overcurrent:
            return self.params.gripper.hard_current_ma * 2.0
        if self.stuck_joints:
            # 夹爪也完全不跟进时，用固定大电流模拟卡阻。
            return self.blocked_current_ma
        # 闭爪时命令开度低于实际开度，堵转量 = 实际 - 命令。自由跟随时这个差额
        # 只有一阶滞后的量级（≈ speed*tau），被挡住后随命令继续下降而线性增长。
        shortfall = max(0.0, self.g - self.g_cmd)
        return min(self.blocked_current_ma, shortfall * self.current_per_pct_shortfall_ma)

    def _deterministic_noise(self) -> np.ndarray:
        """不引入随机源：用读回次数的确定性三角波，保证测试可复现。"""
        k = self.read_count
        return self.noise_deg * np.sin(np.arange(5) * 1.7 + k * 0.9)

    # ------------------------------------------------------------------
    # 测试辅助
    # ------------------------------------------------------------------

    def settle_instantly(self) -> None:
        """把实际位置直接对齐到已提交目标，用于跳过收敛过程。"""
        self.q = self.q_cmd.copy()
        self.g = self.g_cmd
        self._g_vel = 0.0
        self._q_prev = self.q.copy()

    def place_object(self, gap_m: float) -> None:
        """放一个有效开口为 gap_m 的物体（决定闭爪停在哪里）。"""
        self.object_gap_m = float(gap_m)

    def clear_object(self) -> None:
        self.object_gap_m = None

    def all_faults_off(self) -> None:
        """复位全部故障注入，保留物理状态。"""
        self.comm_fail_on_read = False
        self.comm_fail_on_write = False
        self.state_fault = False
        self.freeze_feedback = False
        self.expire_feedback = False
        self.stuck_joints = False
        self.hard_overcurrent_joints = False
        self.gripper_overcurrent = False
        self.grip_slip_rate_pct_s = 0.0
        self._slip_offset_pct = 0.0
        self.time.extra_lateness_s = 0.0
