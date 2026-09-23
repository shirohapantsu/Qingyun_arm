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
    position: FloatArray  # (3,) 目标几何中心，base_link，m
    yaw_deg: float        # 目标长轴相对基座 +X，绕 +Z 为正，[-90, 90)，模 180°
    length_m: float       # 长轴长度
    width_m: float        # 短轴宽度
    ripe: bool            # YOLO 成熟度判定，C 级唯一来源
    valid_count: int      # 当次视野合法候选总数（ignore 封顶判据）


@dataclass(frozen=True)
class VisionThresholds:
    """从已加载 motion profile 提取的视觉可抓性阈值快照。

    这是 P1 组装注入、P3 消费的共享边界契约（决策 D13）：P1 的 main 在加载并校验
    profile 后组装本类型，经 vision.configure() 注入；视觉模块只导入本类型，不导入
    motion_params/plans/main，从而解耦 P1/P3 的并行开发。

    字段逐一从 ``MotionParams`` 复制，不做单位换算，与运动 profile 同源同值：
        target_bounds_m   <- workspace.target_bounds_m
        object_envelope_m <- grasp.object_envelope_m
        clearance_m       <- collision.clearance_m
        approach_height_m <- grasp.approach_height_m
        table_z_m         <- workspace.table_z_m
        table_flatness_m  <- workspace.table_flatness_m

    数组字段在构造时统一转为 float64 并复制为独立缓冲，避免与来源 MotionParams 的
    数组共享内存（本类冻结，归一化通过 object.__setattr__ 完成）。
    """

    target_bounds_m: FloatArray    # (2,3) [min,max]，合法候选中心判定域，base_link，m
    object_envelope_m: FloatArray  # (3,)   物体长、宽、高，m；与运动 profile 同源同值
    clearance_m: float             # 邻物最小净距，m
    approach_height_m: float       # 接近高度，m（头顶净空检查推导用）
    table_z_m: float               # 桌面平面高度，base_link，m（几何中心估计用）
    table_flatness_m: float        # 桌面平整度容差，m

    def __post_init__(self) -> None:
        # 冻结类不能直接赋值，用 object.__setattr__ 完成 dtype 归一与深拷贝，
        # 使快照与来源 profile 的数组彻底解耦（调用方后续改写来源不影响本快照）。
        object.__setattr__(
            self, "target_bounds_m", np.asarray(self.target_bounds_m, dtype=np.float64).copy()
        )
        object.__setattr__(
            self, "object_envelope_m", np.asarray(self.object_envelope_m, dtype=np.float64).copy()
        )


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
