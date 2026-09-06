# common_interface.py
# 接口规范配置文件

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np

# 视觉接口
@dataclass
class VisionInterface:
    position : np.typing.NDArray[np.float64]    # 三维空间坐标
    yaw_deg : float                             # 目标旋转角度:水平面内相对目标摆放姿态的偏航角
    grade : str                                 # 目标品级

# 电机回读数据类
@dataclass
class JointFeedback:
    angles_deg: np.ndarray      # shape (6,)，关节角，含 gripper
    speeds_deg_s: np.ndarray    # shape (6,)
    currents_ma: np.ndarray     # shape (6,)

# 电机控制类
@runtime_checkable
class MotorController(Protocol):
    def send_action(self, joints_deg: np.ndarray, gripper_pct: float) -> None:
        """非阻塞：写入串口缓冲立即返回。内部先做关节行程软限位钳位（最后一道闸，源自
        lerobot motors_bus 标定范围钳位），禁止内部 sleep。joints_deg 为 5 个姿态关节角(deg)。"""

    def get_feedback(self) -> JointFeedback:
        """同步读 6 舵机角度/速度/电流。总线耗时约 10ms（1M 波特率同步读写），30Hz 节拍内富余。"""

    def wait_until_settled(self, timeout_s: float, tol_deg: float) -> bool:
        """阻塞等待所有关节静止（角度变化率 < tol），超时返回 False。"""

    def emergency_stop(self) -> None:
        """freeze：以当前实测角度持续下发，锁定输出。"""