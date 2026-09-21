"""机械臂抓取—放置公共接口。

类型与字段声明；运行时校验、驱动和运动控制由对应模块实现。
规范见 docs/机械臂运动控制模块技术文档.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
)
MOTOR_NAMES = (*JOINT_NAMES, "gripper")


@dataclass(frozen=True)
class VisionInterface:
    """视觉下发的已确定可抓取目标；一次调用对应一个目标。"""

    position: FloatArray  # (3,) 目标几何中心，base_link，m
    yaw_deg: float        # 目标长轴相对基座 +X，绕 +Z 为正，[-90, 90)，模 180°
    grade: str            # 品级元数据，控制仅记录，不用于选择放置点


@dataclass(frozen=True)
class PlacePose:
    """从配置 places[place_id] 加载的释放位姿。"""

    position: FloatArray  # (3,) 释放时目标中心，base_link，m
    yaw_deg: float        # 释放时目标长轴相对基座 +X 的角度，deg，[-90, 90)


@dataclass(frozen=True)
class JointFeedback:
    angles_deg: FloatArray       # (5,) 已转换方向与零位的 URDF 关节角
    speeds_deg_s: FloatArray     # (5,) URDF 关节正方向
    currents_ma: FloatArray      # (5,) 已换算的物理电流
    gripper_pct: float           # 0=标定闭合端，100=标定张开端
    gripper_speed_pct_s: float   # 正值表示张开
    gripper_current_ma: float
    sample_start_ns: int         # time.monotonic_ns()，整次采样开始/结束
    sample_end_ns: int
    sequence: int                # 每次完整成功读回递增


class MotorError(RuntimeError):
    """驱动错误基类。"""


class MotorLimitError(MotorError):
    """命令在写入前整条拒绝。"""


class MotorCommunicationError(MotorError):
    """超时、总线错误或不完整读回。"""


class MotorStateError(MotorError):
    """设备未就绪或报告硬件故障。"""


@runtime_checkable
class MotorController(Protocol):
    def send_action(self, joints_deg: FloatArray, gripper_pct: float) -> None:
        """单次有写超时的同步提交，不等待到位，不持有定时循环。

        全部参数及有效限位检查通过后发送；超限整条拒绝，不静默裁剪。
        返回仅表示提交完成；部分写入或通信失败抛 MotorCommunicationError。
        """

    def get_feedback(self) -> JointFeedback:
        """有总时限地完整读回并换算单位；失败抛异常，不返回缓存或残缺数据。"""

    def hold_current(self) -> JointFeedback:
        """有界地读一次有效反馈，并写一次相同五关节角和夹爪开度。

        返回所用反馈，不持有循环。失败上报，不等同于物理急停。
        """


class GraspStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    CONFIG_INVALID = "CONFIG_INVALID"
    IK_FAILED = "IK_FAILED"
    PLAN_INVALID = "PLAN_INVALID"
    COLLISION = "COLLISION"
    GRASP_MISS = "GRASP_MISS"
    GRASP_UNCERTAIN = "GRASP_UNCERTAIN"
    GRIP_SLIP = "GRIP_SLIP"
    PLACE_FAIL = "PLACE_FAIL"
    COMM_ERROR = "COMM_ERROR"
    TRACKING_ERROR = "TRACKING_ERROR"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"


class HoldingState(str, Enum):
    """基于夹爪开度和电流的夹持状态估计。"""

    EMPTY = "EMPTY"
    HOLDING = "HOLDING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GraspResult:
    status: GraspStatus
    stage: str              # 运动控制文档状态表中的名称
    reason: str
    place_id: str
    holding: HoldingState
    recovery_required: bool  # True 时实例锁定，需完成复位才可接受下一次调用
