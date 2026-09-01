# common_interface.py
# 接口规范配置文件

from dataclasses import dataclass
from typing import List,Dict
import numpy as np

# 视觉接口
@dataclass
class VisionInterface:
    position : np.typing.NDArray[np.float64]    # 三维空间坐标
    yaw_deg : float                             # 目标旋转角度
    grade : str                                 # 目标品级

