# servo_params.py
# 电机侧（3号 / C）真机参数集中地：总线参数 + 舵机标定模型 + 换算因子。
# 约定：每个数值附来源注释；带「标定 TODO」的值待真机零位/行程标定后回填。
# 冻结契约见 configs/common_interface.py；关节定义与 readme.md「官方标准关节索引」一致。

from __future__ import annotations

import os
from dataclasses import dataclass

# ================================================================ 总线参数
# 香橙派 5 上 USB2Serial 转换板枚举为 /dev/ttyACM0（见 docs/机械臂运动控制模块技术文档.md）。
# 开发机 Windows 上请用设备管理器查出 COM 口（如 COM5），
# 也可用环境变量 QINGYUN_SERVO_PORT 覆盖，例如：
#   Linux:   QINGYUN_SERVO_PORT=/dev/ttyACM0
#   Windows: set QINGYUN_SERVO_PORT=COM5
SERVO_PORT: str = os.environ.get("QINGYUN_SERVO_PORT", "/dev/ttyACM0")

# STS3215 出厂默认 1Mbps（寄存器 Baud_Rate=0）。SO-ARM100 总线即 1M，勿改。
SERVO_BAUDRATE: int = 1_000_000

# 单包应答总超时。技术文档预算：6 舵机同步读写 ≈ 10ms@1Mbps，
# 顺序读整块反馈 6×~2.5ms ≈ 15ms，仍在 30Hz 节拍(33ms)富余内。
SERVO_RESPONSE_TIMEOUT_S: float = 0.05

# 坏包/超时重试次数。lerobot so_follower 同款策略：Feetech 总线偶发坏状态包，重试即可。
SERVO_READ_RETRIES: int = 2

# 写指令应答探测窗口：STS 出厂默认“只回读指令”(Response_Status_Level=1)，写无应答，
# 因此 send_action 走无探测快速路径；仅当显式 expect_ack=True 校验写结果时，
# 在此窗口内探测应答（未收到即按超时处理）。
SERVO_ACK_PEEK_S: float = 0.008

# ================================================================ 换算因子（STS3215 数据手册）
# 位置：磁编码 4096 步/圈（0~4095 ⇔ 0~360°），即每步 360/4096 ≈ 0.08789°
POS_DEG_PER_STEP: float = 360.0 / 4096.0
# 速度：Present_Speed 寄存器单位为“步/秒”，与位置同尺度
# （来源：beam-bots/feetech 驱动 sts3215.ex 注释 + Feetech 手册；真机联调复核 TODO）
SPEED_DEG_S_PER_BIT: float = 360.0 / 4096.0
# 电流：Present_Current(0x45, 2B 只读) 1 bit = 6.5 mA，满量程 500×6.5≈3250mA
# （官方寄存器表 STS3215_Memory_Table_EN.xlsx 0x45 行；官方产品说明“电流反馈 1=6.5mA”；
#  飞特官网 https://www.feetech.cn/news/2020-05-13_56655 ；交叉核对
#  commanderfun/STS3215 REGISTER_REFERENCE、lerobot tables.py Present_Current=(69,2)）
CURRENT_MA_PER_BIT: float = 6.5

# ================================================================ 关节 ↔ 舵机标定模型
@dataclass(frozen=True)
class ServoJointConfig:
    """单个关节的舵机侧线性标定模型（3号零位标定的落点）。

    正向：joint_deg = direction * (raw - zero_raw) * POS_DEG_PER_STEP + offset_deg
    逆向：raw = zero_raw + (joint_deg - offset_deg) / (direction * POS_DEG_PER_STEP)

    raw      舵机磁编码原始读数 0..4095（Present_Position，地址 0x38）
    zero_raw 当关节角 = offset_deg 时舵机应处的原始读数（零位标定得出）
    """
    name: str          # 关节名（对齐 readme 官方索引与 URDF）
    servo_id: int      # 总线舵机 ID（烧录后与上位机约定一致）
    min_deg: float     # 软限位下界（来源：so101_new_calib.urdf 关节限位，
                       # 见 docs §4.2；真机与 URDF 对齐由官方校准流程保证）
    max_deg: float     # 软限位上界
    zero_raw: int = 2048      # 零位原始读数（标定 TODO：假定 2048≈0°，标定后回填）
    direction: int = 1        # +1/-1：舵机正向与关节正向一致否（标定 TODO）
    offset_deg: float = 0.0   # 关节零位偏置 deg（标定 TODO）

    def deg_to_raw(self, deg: float) -> float:
        return self.zero_raw + (deg - self.offset_deg) / (self.direction * POS_DEG_PER_STEP)

    def raw_to_deg(self, raw: float) -> float:
        return self.direction * (raw - self.zero_raw) * POS_DEG_PER_STEP + self.offset_deg


# 顺序即 readme 官方标准关节索引 0..5；舵机 ID 默认 1..6 连续烧录
# （若实际烧录顺序不同，改 servo_id 即可，其余代码零改动）。
MOTOR_CONFIGS: tuple[ServoJointConfig, ...] = (
    ServoJointConfig(name="shoulder_pan",  servo_id=1, min_deg=-110.0,   max_deg=110.0),
    ServoJointConfig(name="shoulder_lift", servo_id=2, min_deg=-100.0,   max_deg=100.0),
    ServoJointConfig(name="elbow_flex",    servo_id=3, min_deg=-96.83,   max_deg=96.83),
    ServoJointConfig(name="wrist_flex",    servo_id=4, min_deg=-95.0,    max_deg=95.0),
    ServoJointConfig(name="wrist_roll",    servo_id=5, min_deg=-157.21,  max_deg=162.79),
    ServoJointConfig(name="gripper",       servo_id=6, min_deg=-10.0,    max_deg=100.0),
)

# 夹爪百分比 → 关节角线性映射端点（0% 全闭 / 100% 全开，标定 TODO 复核方向与端点；
# URDF gripper 关节量程 -10°~100°，默认按 0% → -10°、100% → 100° 线性）
GRIPPER_DEG_AT_0_PCT: float = -10.0
GRIPPER_DEG_AT_100_PCT: float = 100.0

# 反馈整块读取区间：Present_Position..Present_Current 连续寄存器 (0x38..0x46)，
# 一次读回 15 字节避免逐寄存器 3 次往返。
# 布局（相对起始地址 0x38 的偏移）：
#   0: 2B Present_Position   2: 2B Present_Speed(符号位 bit15)
#   4: 2B Present_Load       6: 1B Present_Voltage  7: 1B Present_Temperature
#   10: 1B Moving            13: 2B Present_Current(×6.5mA)
FEEDBACK_START_ADDR: int = 0x38
FEEDBACK_LEN: int = 15
