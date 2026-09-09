"""SO-ARM101 电机驱动：物理量换算、STS 串口协议与 MotorController 实现。

本文件实现技术文档第二节文件结构里的 ``qingyun/grabbing/motor_control.py``。
它在系统里承担技术文档 1.1 职责表中"电机驱动"那一行的全部内容：
"物理量换算、零位和方向映射、最终指令限位、串口读写、硬件状态反馈"。

规范来源（正文注释会逐处引用条款号，这里先给总览）：
    docs/机械臂运动控制模块技术文档.md
        第二节   文件结构与模块边界
        1.1      模块职责（驱动负责最终指令限位与硬件状态反馈）
        3.1      关节顺序（ID 1~5 为姿态关节、ID 6 为夹爪）与接口单位
        4.2      MotorController 三个方法的语义、整条校验在写入前完成、
                 总时限不因内部寄存器/电机数量成倍增加、不宣称多电机写入的
                 物理事务原子性、错误分类（Limit/Communication/State）
        6.1/6.4  单线程同步执行、返回后不持有循环、停止后调用 hold_current 一次
    docs/真机参数测量与标定指南.md
        4.1   原始记录；"不要重复在本项目里再加一次官方 homing_offset"
        4.2   关节读数的严格换算四行公式；整数取整策略必须固定并用往返测试证明；
              读写时限限制整次方法
        4.3   电流与速度单位换算（禁止把 Present_Load 当 Present_Current）
        4.4   夹爪开闭两端、百分比公式、越界观测属于异常而不是裁剪
        8.2   整次读写的时限与跨度（max_feedback_span_s / age 的算法）
        第 12 节 与标定工具共用的换算函数签名（本文件第 2 节照抄）

四层结构（自下而上）：
    第 1~2 节  共享换算函数          纯数学，标定工具与生产代码调用同一份实现
    第 3~4 节  官方校准文件 + MotorMapping   "物理量 ↔ 总线原始值"双向换算器
    第 5~6 节  ByteTransport + StsProtocol   STS protocol_version=0 组包/解包与有界收发
    第 7 节    Sts3215MotorController        实现 configs.common_interface.MotorController

依赖边界（硬性约束）：
    * 寄存器地址、编码分辨率、符号位定义只从
      ``libs/so_arm_core/motors/feetech/tables.py`` 查，本文件不再抄第二份地址表。
      libs/so_arm_core/SOURCE.md 第 1 节说明了为什么不 vendor LeRobot 的
      motors_bus.py（依赖链成本远高于收益），因此协议层是自己实现的，但表是
      同一份 vendor 表。
    * 只用标准库 + numpy + vendor 表 + 延迟导入的 pyserial，无新增依赖。
      ``import serial`` 写在 SerialTransport 构造函数里（第 6 节），所以没装
      pyserial 的机器依然能 import 本模块并测试第 2/4 节的换算层
      （SOURCE.md 第 2 节：pyserial "仅真机 MotorController 需要；未安装时本
      模块仍可 import"）。
    * 时钟（clock / monotonic_ns）、sleep 与 transport 全部可注入，所以三个协议
      方法都能用假串口和假时钟在单元测试里驱动，包括超时与部分读回路径。
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from configs.common_interface import (
    JOINT_NAMES,
    MOTOR_NAMES,
    FloatArray,
    JointFeedback,
    MotorCommunicationError,
    MotorError,
    MotorLimitError,
    MotorStateError,
)
from configs.motion_params import (
    JointsParams,
    MotionParams,
    MotorParams,
    effective_joint_limits,
)
from libs.so_arm_core.motors.encoding_utils import (
    decode_sign_magnitude,
    encode_sign_magnitude,
)
from libs.so_arm_core.motors.feetech.tables import (
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_PROTOCOL,
)

__all__ = [
    "bus_deg_to_urdf_deg",
    "urdf_deg_to_bus_deg",
    "raw_to_gripper_pct",
    "gripper_pct_to_raw",
    "load_motor_calibration",
    "ServoAxis",
    "MotorMapping",
    "ByteTransport",
    "StsProtocol",
    "SerialTransport",
    "Sts3215MotorController",
    "CalibrationReader",
]

# ---------------------------------------------------------------------------
# 1. 模块常数
# ---------------------------------------------------------------------------

# 标定指南 4.2 第一行公式里的 360/(resolution-1)：分辨率减 1 是因为官方
# DEGREES 归一化把 [0, resolution-1] 当成闭区间（两端都算读数）。
DEG_PER_COUNT_BASE = 360.0

# 浮点比较容差。这些容差只吸收 JSON/浮点表示误差，不给真实的物理越限开门：
# 量化误差本身（sts3215 约 0.088deg/计数）由 joints.margin_deg 承担
# （标定指南 5 节第 5 条："从端点测量误差、编码量化和运行停止余量确定 margin_deg"）。
LIMIT_TOL_DEG = 1e-9
PCT_TOL = 1e-9
# 最终指令限位允许的编码余量：1 个计数。往返换算本身最多差半个到一个步长
# （标定指南 4.2 要求"用往返测试证明误差不超过…一个编码步长"），不给余量会在
# 限位端点附近因取整而误拒合法命令。
FINAL_RAW_TOL_COUNT = 1

# 夹爪开度的接口定义域：0% = 标定闭合端，100% = 标定张开端（技术文档 4.2）。
GRIPPER_PCT_MIN = 0.0
GRIPPER_PCT_MAX = 100.0

# 判定"读数是否落在两个标定端点之间"的容差。注意这里刻意不叫 clamp：
# 越界一律抛异常（标定指南 4.4）。
_CALIB_ENDPOINT_TOL = 1e-9

# 接收轮询：连续这么多次 read_available() 为空后，改交给一次有界的阻塞 read()
# 让出 CPU。它不是超时机制，超时只由整次调用的绝对 deadline 决定。
_POLLS_BEFORE_BLOCK = 4
_POLL_SLEEP_S = 0.0002

# 初始化阶段的整次调用预算（秒）。为什么不用 timing.write_timeout_s：
# 初始化要做 6 台 ×（应答配置读回 + PING + 型号 + 固件×2 + 力矩）≈ 36 次事务
# （verify_identity=False 时少 18 次），而 write_timeout_s 是按"每个控制 tick
# 一次同步写"测出来的周期预算，拿它套初始化只会得到必然超时。但它仍然是
# **一个绝对 deadline 覆盖全部内部事务**，不会退化成"每台一个超时"
# （技术文档 4.2 的总时限原则）。真值要在 timing 阶段实测（标定指南 8.2），
# 这里给一个明显宽松的保守值。
_INIT_TIMEOUT_S = 2.0

# pyserial 适配器的阻塞读分片（秒）。为什么用一个很小的常数而不是
# timing.read_timeout_s：总时限由协议层的"绝对 deadline + 每轮重新检查"保证，
# 适配器只需要保证单次阻塞不会太久。取 1ms 时最坏情况是 deadline 到期后再等
# 1ms 退出，远小于一个 tick（33ms）。真机 USB 转串口的实际延迟必须在 timing
# 阶段实测（标定指南 8.2 第 2 条"实际 SDK 超时行为必须测量"）。
_SERIAL_READ_SLICE_S = 0.001

# pyserial 写超时（秒）。上一版根本没设 write_timeout（pyserial 默认 None =
# 无限阻塞），"整次预算"在慢写路径上形同虚设（审阅指引 3.3）。运行期每次写
# 都带 timeout_s=剩余预算 下来，本常数只是"调用方没给预算"时的兜底上限。
# 注意它限制的是把帧拷进操作系统发送缓冲的时间，不含 USB 实际排空；
# 真实排空延迟无法由软件承诺，列为真机待测项（标定指南 8.2）。
_SERIAL_WRITE_TIMEOUT_S = 0.1


# ---------------------------------------------------------------------------
# 2. 标定指南第 12 节规定的共享换算函数
#
# 签名与 docs/真机参数测量与标定指南.md 第 12 节逐项一致。标定工具
# （scripts/calibrate.py）与生产代码必须调用同一份实现，否则"用同一份原始日志
# 通过生产换算函数重建 JointFeedback，并与采集时保存的物理量一致"（第 12 节
# 必要离线检查第 6 条）就无从谈起。
# ---------------------------------------------------------------------------


def bus_deg_to_urdf_deg(q_bus_deg, sign, zero_offset_deg):
    """总线角 → URDF 关节角（支持标量与 (5,) 向量）。

    标定指南 4.2 第二行公式，原样实现：

        q_urdf = joints.sign * q_bus_deg + joints.zero_offset_deg

    为什么需要这一步：舵机自己的零点和正方向与 URDF 的关节定义无关，方向与
    几何零位是在 joints 阶段实测出来的（标定指南 5 节第 2 条）。技术文档 4.2
    规定这个补偿由 C（驱动）完成，B（规划/运动学）不重复补偿。

    sign / zero_offset_deg 允许是标量（单关节）或长度 5 的向量（整条臂）；
    numpy 的广播规则天然覆盖两种用法，调用方保证形状自洽。
    """
    q = np.asarray(q_bus_deg, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    o = np.asarray(zero_offset_deg, dtype=np.float64)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(o)):
        raise MotorLimitError(f"bus_deg_to_urdf_deg 输入含非有限值：{q!r} / {o!r}")
    _check_sign(s)
    out = s * q + o
    # 标量输入返回 float、向量输入返回 float64 数组：JointFeedback 里
    # angles_deg 是 (5,) float64，而 gripper_pct 这类字段是 Python float。
    return float(out) if out.ndim == 0 else out.astype(np.float64, copy=False)


def urdf_deg_to_bus_deg(q_urdf_deg, sign, zero_offset_deg):
    """URDF 关节角 → 总线角（支持标量与 (5,) 向量）。

    标定指南 4.2 第三行公式：

        q_bus_command = (q_urdf_command - joints.zero_offset_deg) / joints.sign

    这里写成除法而不是乘 sign，是为了让"它是第二行公式的反变换"这件事在代码
    上可见。配置加载已保证 sign 只能是 ±1（configs.motion_params._parse_joints），
    所以除法既不会除零也不引入误差；仍然显式检查一次，防止有人绕过加载器
    直接构造 JointsParams。
    """
    q = np.asarray(q_urdf_deg, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    o = np.asarray(zero_offset_deg, dtype=np.float64)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(o)):
        raise MotorLimitError(f"urdf_deg_to_bus_deg 输入含非有限值：{q!r} / {o!r}")
    _check_sign(s)
    out = (q - o) / s
    return float(out) if out.ndim == 0 else out.astype(np.float64, copy=False)


def _check_sign(sign: NDArray) -> None:
    """sign 必须逐项为 ±1（标定指南 2.1 对 joints.sign 的约束）。"""
    flat = np.atleast_1d(np.asarray(sign, dtype=np.float64))
    bad = np.where((flat != 1.0) & (flat != -1.0))[0]
    if bad.size:
        raise MotorLimitError(
            f"joints.sign 只能取 ±1，实际 {flat.tolist()} 在第 {bad.tolist()} 项不合法"
        )


def raw_to_gripper_pct(raw, closed_raw, open_raw):
    """总线原始位置读数 → 夹爪开度百分比。

    标定指南 4.4 第一行公式：

        g = 100 * (raw_position - closed_raw) / (open_raw - closed_raw)

    三个实现要点：
      1. open_raw 允许小于 closed_raw（张开端读数不保证更大）。分母取
         ``open_raw - closed_raw`` 的原始符号，方向就自动正确：读数朝 open_raw
         移动时 g 一定增大，所以"正 g、正 g 速率 = 张开"始终成立。
      2. 分母为 0 禁止使用（4.4"分母为 0 禁止使用"）。配置加载器已经拦过一次
         （configs/motion_params._parse_motor），这里再兜一层，因为本函数也被
         标定工具直接调用，那条路径不经过加载器。
      3. 观测超出 [closed_raw, open_raw] 时抛 MotorLimitError，**不裁剪成
         0/100**。4.4 明确："超出范围的观测属于异常而不是先裁剪成 0/100 后
         掩盖故障"。裁剪会把"标定端点漂移""homing_offset 被写错""机构被外力
         掰过头"这三类故障伪装成正常的满开/满闭读数。
    """
    closed = int(closed_raw)
    opened = int(open_raw)
    denom = float(opened - closed)
    if denom == 0.0:
        raise MotorLimitError(
            f"gripper_open_raw({opened}) 与 gripper_closed_raw({closed}) 相同，"
            "百分比换算分母为 0（标定指南 4.4）"
        )
    value = float(raw)
    if not math.isfinite(value):
        raise MotorLimitError(f"夹爪原始读数非有限值：{raw!r}")
    lo, hi = (closed, opened) if opened > closed else (opened, closed)
    if value < lo - _CALIB_ENDPOINT_TOL or value > hi + _CALIB_ENDPOINT_TOL:
        raise MotorLimitError(
            f"夹爪原始读数 {value:.1f} 落在标定端点 [{lo}, {hi}] 之外，按异常处理"
            "（标定指南 4.4：不做裁剪掩盖故障）"
        )
    return 100.0 * (value - closed) / denom


def gripper_pct_to_raw(pct, closed_raw, open_raw):
    """夹爪开度百分比 → 总线原始位置读数（整数）。

    标定指南 4.4 第二行公式：

        raw = closed_raw + g / 100 * (open_raw - closed_raw)

    取整策略固定为 Python 内置 ``round()``（对恰好 .5 取偶数，即"四舍六入五成
    双"），理由是 4.2 要求"整数取整策略必须固定并用往返测试证明"：
      * round() 是确定性函数，同一输入永远同一输出，往返测试的界限可复现；
      * int(x + 0.5) 这类写法在负方向上不对称，截断会引入固定的半计数偏置；
      * 取偶数的最大偏差是 0.5 个计数（sts3215 上约 0.044deg），满足 4.2 允许
        的"不超过一个编码步长"界限。

    命令侧同样拒绝越过标定端点：0~100% 之外属于异常（与 4.4 对观测值的要求
    对称；越界命令意味着要求舵机走到没有标定过的地方）。
    """
    closed = int(closed_raw)
    opened = int(open_raw)
    denom = float(opened - closed)
    if denom == 0.0:
        raise MotorLimitError(
            f"gripper_open_raw({opened}) 与 gripper_closed_raw({closed}) 相同，"
            "百分比换算分母为 0（标定指南 4.4）"
        )
    g = float(pct)
    if not math.isfinite(g):
        raise MotorLimitError(f"夹爪开度命令非有限值：{pct!r}")
    if g < GRIPPER_PCT_MIN - _CALIB_ENDPOINT_TOL or g > GRIPPER_PCT_MAX + _CALIB_ENDPOINT_TOL:
        raise MotorLimitError(
            f"夹爪开度命令 {g:.3f}% 超出 [{GRIPPER_PCT_MIN:.0f}, {GRIPPER_PCT_MAX:.0f}]"
            "（0%=标定闭合端、100%=标定张开端，技术文档 4.2）"
        )
    return int(round(closed + g / 100.0 * denom))


# ---------------------------------------------------------------------------
# 3. 官方电机校准文件
# ---------------------------------------------------------------------------

# 官方 calibration 每项的必需键，格式见
# calibration/SIM_ROBOT/simulation/motor_calibration.json。
CALIBRATION_KEYS = {"id", "homing_offset", "range_min", "range_max", "drive_mode"}


def load_motor_calibration(path: Path) -> dict[str, dict[str, int]]:
    """读取官方电机校准 JSON，并核对它与 MOTOR_NAMES/ID 的顺序一致。

    为什么驱动层要读这个文件而不是自己发明零位：标定指南 4.1 第 2 条要求
    "保留官方 LeRobot 校准原始文件"，而 4.2 第一行公式要用的
    ``mid = (range_min + range_max) / 2`` 只能从它来。同时 4.1 第 3 条明确
    "不要重复在本项目里再加一次官方 homing_offset"，所以本函数把
    homing_offset 原样读进来只作登记，换算里完全不使用它。

    这里抛 ValueError 而不是 MotorError 子类：读不到文件属于启动期配置错误，
    入口（arm_control）应映射成 GraspStatus.CONFIG_INVALID，而不是电机故障。
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取电机校准文件 {path}：{exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"电机校准文件 {path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"电机校准文件 {path} 应为以关节名为键的对象")

    missing = [name for name in MOTOR_NAMES if name not in payload]
    if missing:
        raise ValueError(f"电机校准文件缺少条目 {missing}（需要 {list(MOTOR_NAMES)}）")
    unknown = sorted(set(payload) - set(MOTOR_NAMES))
    if unknown:
        raise ValueError(f"电机校准文件出现未知条目 {unknown}")

    out: dict[str, dict[str, int]] = {}
    for name in MOTOR_NAMES:
        item = payload[name]
        if not isinstance(item, dict):
            raise ValueError(f"电机校准条目 {name} 应为对象，实际是 {type(item).__name__}")
        keys = set(item)
        if keys - CALIBRATION_KEYS:
            raise ValueError(f"电机校准条目 {name} 出现未知键 {sorted(keys - CALIBRATION_KEYS)}")
        if CALIBRATION_KEYS - keys:
            raise ValueError(f"电机校准条目 {name} 缺失键 {sorted(CALIBRATION_KEYS - keys)}")
        try:
            conv = {k: int(item[k]) for k in CALIBRATION_KEYS}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"电机校准条目 {name} 的字段应为整数：{exc}") from exc
        # 行程退化（min >= max）会让 mid 失去意义，也会让驱动层"最终指令限位"
        # 的行程包含检查变成永假或永真。
        if conv["range_min"] >= conv["range_max"]:
            raise ValueError(
                f"电机校准条目 {name} 的 range_min({conv['range_min']}) 必须小于 "
                f"range_max({conv['range_max']})"
            )
        if conv["drive_mode"] not in (0, 1):
            raise ValueError(f"电机校准条目 {name} 的 drive_mode 只能是 0/1")
        out[name] = conv
    return out


# ---------------------------------------------------------------------------
# 4. MotorMapping：物理量 ↔ 总线原始值的双向换算器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServoAxis:
    """单个舵机通道的换算规格（profile 的 motor+joints 与官方校准的合并结果）。"""

    name: str
    index: int              # MOTOR_NAMES 里的位置：0~4 姿态关节，5 夹爪
    servo_id: int           # 总线 ID（技术文档 3.1：关节 ID 1~5，夹爪 ID 6）
    model: str              # 型号键，用于查 vendor 表
    firmware: str
    resolution: int         # 编码分辨率（motor.position_resolution）
    mid_raw: float          # 官方 calibration 的 (range_min + range_max)/2
    range_min: int          # 官方行程端点，驱动层"最终指令限位"用
    range_max: int
    homing_offset: int      # 只登记、绝不参与换算（标定指南 4.1 第 3 条）
    drive_mode: int         # 只登记，见 MotorMapping 类注释
    is_gripper: bool        # 夹爪通道的位置量用开度百分比而不是 URDF 角表达

    @property
    def count_to_deg(self) -> float:
        """一个编码计数对应的角度（deg），即 4.2 的 360/(resolution-1)。"""
        return DEG_PER_COUNT_BASE / float(self.resolution - 1)


class MotorMapping:
    """把 profile.json 的 motor + joints 两组参数封装成双向换算器。

    设计取舍：
      * 寄存器地址、分辨率、符号位一律查 vendor 表（模块头的依赖边界）。
      * 位置换算严格走标定指南 4.2 的四行公式 ``raw ↔ q_bus_deg ↔ q_urdf_deg``。
        其中的 raw 是设备上报的 Present_Position，官方 homing 已经写进舵机
        EEPROM，读数本身就含 homing 结果，所以**整条链里没有任何地方再加一次
        homing_offset**（4.1 第 3 条禁止）。
      * drive_mode 同样只登记不使用：官方把方向翻转写进 EEPROM（Phase 位），
        设备上报的读数已经带上它；本项目额外的"总线系→URDF 系"方向由 joints
        阶段实测的 sign 承担（标定指南 5 节第 2 条）。若在换算里再乘一次
        drive_mode，就出现两份方向真值，正是 4.1 第 3 条警告的重复补偿。
      * 本类是纯换算与查询，不碰串口，所以可以离线单元测试（标定指南 12 节
        离线检查第 2、6 条）。
    """

    def __init__(
        self,
        motor: MotorParams,
        joints: JointsParams,
        calibration: Mapping[str, Mapping[str, int]],
        urdf_limits_deg: NDArray | None = None,
    ) -> None:
        if len(motor.ids) != len(MOTOR_NAMES):
            raise ValueError(f"motor.ids 长度应为 {len(MOTOR_NAMES)}，实际 {len(motor.ids)}")
        if len(joints.sign) != len(JOINT_NAMES):
            raise ValueError(f"joints.sign 长度应为 {len(JOINT_NAMES)}")
        if len(joints.zero_offset_deg) != len(JOINT_NAMES):
            raise ValueError(f"joints.zero_offset_deg 长度应为 {len(JOINT_NAMES)}")

        axes: list[ServoAxis] = []
        for i, name in enumerate(MOTOR_NAMES):
            entry = calibration.get(name)
            if entry is None:  # load_motor_calibration 已经查过，这里兜住手工传入的 dict
                raise ValueError(f"官方电机校准缺少条目 {name!r}")
            model = str(motor.models[i])
            if model not in MODEL_CONTROL_TABLE:
                raise ValueError(
                    f"{name} 的型号 {model!r} 不在 vendor 的 MODEL_CONTROL_TABLE 里，"
                    "寄存器地址与符号位就没有出处"
                )
            resolution = int(motor.position_resolution[i])
            if resolution <= 1:
                raise ValueError(f"{name} 的 position_resolution 必须大于 1，实际 {resolution}")
            servo_id = int(motor.ids[i])
            # 技术文档 3.1 固定 ID 1~5 为姿态关节、6 为夹爪。校准文件里的 id 与
            # profile 的 motor.ids 不一致说明装配或配置错位：后面所有换算都会
            # 张冠李戴，必须在这里拒绝。
            if int(entry["id"]) != servo_id:
                raise ValueError(
                    f"{name} 的 motor.ids={servo_id} 与电机校准 id={int(entry['id'])} 不一致"
                )
            axes.append(
                ServoAxis(
                    name=name,
                    index=i,
                    servo_id=servo_id,
                    model=model,
                    firmware=str(motor.firmware[i]),
                    resolution=resolution,
                    # 4.2 第一行公式的 mid：官方 calibration 的行程中点。
                    mid_raw=(int(entry["range_min"]) + int(entry["range_max"])) / 2.0,
                    range_min=int(entry["range_min"]),
                    range_max=int(entry["range_max"]),
                    homing_offset=int(entry["homing_offset"]),
                    drive_mode=int(entry["drive_mode"]),
                    is_gripper=i >= len(JOINT_NAMES),
                )
            )

        self.axes: tuple[ServoAxis, ...] = tuple(axes)
        self.by_name: Mapping[str, ServoAxis] = {a.name: a for a in axes}
        self.by_id: Mapping[int, ServoAxis] = {a.servo_id: a for a in axes}
        self.motor = motor
        self.joints = joints
        self.ids: NDArray[np.int64] = np.asarray(motor.ids, dtype=np.int64)
        self._urdf_limits_deg = (
            None if urdf_limits_deg is None else np.asarray(urdf_limits_deg, dtype=np.float64)
        )
        # 夹爪两端读数缓存，供百分比与速度换算共用（标定指南 4.4）。
        self.gripper_closed_raw = int(motor.gripper_closed_raw)
        self.gripper_open_raw = int(motor.gripper_open_raw)
        self._limits_cache: tuple[FloatArray, FloatArray] | None = None

    # --- 通道与寄存器查询 ---

    def axis(self, name_or_index: str | int | ServoAxis) -> ServoAxis:
        """按名字（MOTOR_NAMES）、按 MOTOR_NAMES 下标或已经解析好的通道取通道。

        允许直接传 ServoAxis 是给内部助手（decode_signed / encode_signed /
        sign_bit / register）复用的：它们已经拿到通道对象，再查一次表既浪费也
        容易在名字与下标混用时出错。
        """
        if isinstance(name_or_index, ServoAxis):
            return name_or_index
        if isinstance(name_or_index, str):
            try:
                return self.by_name[name_or_index]
            except KeyError:
                raise ValueError(
                    f"未知电机名 {name_or_index!r}，可选 {list(MOTOR_NAMES)}"
                ) from None
        idx = int(name_or_index)
        if not 0 <= idx < len(self.axes):
            raise ValueError(f"电机下标 {idx} 超出 [0, {len(self.axes) - 1}]")
        return self.axes[idx]

    def joint_sign_offset(self, name_or_index: str | int) -> tuple[float, float]:
        """取该通道的 (sign, zero_offset_deg)。

        joints 组只有 5 个姿态关节（标定指南 2.1），夹爪没有 sign/offset 条目，
        因此夹爪按 (+1, 0) 处理：它的角度通道只做"原始量 ↔ 总线角"，真正的
        命令量是开度百分比，走 4.4 的百分比公式。
        """
        ax = self.axis(name_or_index)
        if ax.is_gripper:
            return 1.0, 0.0
        return float(self.joints.sign[ax.index]), float(self.joints.zero_offset_deg[ax.index])

    def register(self, name_or_index: str | int, reg_name: str) -> tuple[int, int]:
        """查该通道型号的寄存器表，返回 (地址, 字节数)。

        查不到说明配置里的型号与该寄存器不匹配（例如 SCS 系列没有
        Present_Current），属于配置错误而不是运行期故障。
        """
        ax = self.axis(name_or_index)
        table = MODEL_CONTROL_TABLE[ax.model]
        entry = table.get(reg_name)
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(
                f"寄存器 {reg_name!r} 在型号 {ax.model!r} 的 vendor 控制表里没有登记"
            )
        return int(entry[0]), int(entry[1])

    def common_register(self, reg_name: str) -> tuple[int, int]:
        """要求 6 个通道给出同一 (地址, 字节数)，供整条总线的批量读写使用。

        为什么必须一致：sync_write 与 bulk_read 只能带一个起始地址。如果某个
        通道地址不同，"一次同步写"就会把数据写进另一台舵机的别的寄存器，
        后果比通信失败更糟。
        """
        specs = {self.register(a, reg_name) for a in self.axes}
        if len(specs) != 1:
            raise ValueError(
                f"寄存器 {reg_name!r} 在 6 个通道上的地址/长度不一致：{sorted(specs)}，"
                "不能用单条批量报文访问"
            )
        return specs.pop()

    def sign_bit(self, name_or_index: str | int, reg_name: str) -> int | None:
        """该寄存器是否需要按符号量解码；None 表示按无符号处理。

        沿用 vendor 表的型号定义（STS_SMS_SERIES_ENCODINGS_TABLE /
        MODEL_ENCODING_TABLE），不在本文件另抄一份位定义。
        需要核对的点：表里没有登记 Present_Current，所以本实现按无符号读它。
        这与标定指南 2.1 末段"原始电流的符号由驱动按型号解码"并不矛盾——按
        型号解码的结果就是"该型号该寄存器无方向位"。但这是从 vendor 表推出的，
        不是从 STS3215 当前固件手册确认的，因此必须在 motor 标定阶段核对
        （4.1 第 1 条要求读回型号与固件）；如需修正只改本方法即可，上层不变。
        """
        ax = self.axis(name_or_index)
        enc = MODEL_ENCODING_TABLE.get(ax.model, {})
        bit = enc.get(reg_name)
        return None if bit is None else int(bit)

    # --- 原始量符号编解码 ---

    def decode_signed(self, name_or_index: str | int, reg_name: str, raw: int) -> int:
        """把寄存器读数按型号定义解成有符号整数。

        标定指南 4.1："设备上报的符号位先按型号/固件解码，再进行数值换算"，
        所以这一步必须在 raw→deg/ma 之前，不能反过来。
        """
        ax = self.axis(name_or_index)
        value = int(raw)
        bit = self.sign_bit(ax, reg_name)
        if bit is None:
            # 无符号：原样返回，但仍检查位宽，避免把明显坏掉的字节当成数据。
            if not 0 <= value < (1 << 16):
                raise MotorCommunicationError(
                    f"{ax.name}.{reg_name} 原始读数 {value} 超出 16 位无符号范围"
                )
            return value
        # 符号量编码的方向位是第 bit 位，所以可表示范围是 [0, 2^(bit+1)-1]。
        if not 0 <= value < (1 << (bit + 1)):
            raise MotorCommunicationError(
                f"{ax.name}.{reg_name} 原始读数 {value} 超出符号量编码"
                f"（方向位 {bit}）可表示范围"
            )
        return int(decode_sign_magnitude(value, bit))

    def encode_signed(self, name_or_index: str | int, reg_name: str, value: int) -> int:
        """把有符号整数按型号定义编成寄存器原始值（非负）。"""
        ax = self.axis(name_or_index)
        bit = self.sign_bit(ax, reg_name)
        v = int(value)
        if bit is None:
            if v < 0:
                raise MotorLimitError(
                    f"{ax.name}.{reg_name} 在 vendor 表里按无符号登记，无法编码负值 {v}"
                )
            return v
        try:
            return int(encode_sign_magnitude(v, bit))
        except ValueError as exc:
            # 幅度超限属于命令越限，不是通信故障。
            raise MotorLimitError(f"{ax.name}.{reg_name} 编码越限：{exc}") from exc

    # --- 位置：raw ↔ q_bus_deg ↔ q_urdf_deg（标定指南 4.2 四行公式）---

    def raw_to_bus_deg(self, name_or_index: str | int, raw: int) -> float:
        """4.2 第一行：q_bus_deg = (raw_after_homing - mid) * 360/(resolution-1)。

        raw 直接取设备上报的 Present_Position，**不加 homing_offset**
        （标定指南 4.1 第 3 条）。
        """
        ax = self.axis(name_or_index)
        decoded = self.decode_signed(ax, "Present_Position", raw)
        return (float(decoded) - ax.mid_raw) * ax.count_to_deg

    def bus_deg_to_raw(self, name_or_index: str | int, q_bus_deg: float) -> int:
        """4.2 第四行：raw_command = round(q_bus_command*(resolution-1)/360 + mid)。

        取整固定 round()，与 gripper_pct_to_raw 同一策略（4.2 要求取整策略
        固定并由往返测试证明）。最后按 Goal_Position 的符号位定义编码。
        """
        ax = self.axis(name_or_index)
        encoded_count = int(round(float(q_bus_deg) / ax.count_to_deg + ax.mid_raw))
        return self.encode_signed(ax, "Goal_Position", encoded_count)

    def raw_to_degrees(self, name_or_index: str | int, raw: int) -> float:
        """总线原始读数 → URDF 关节角（deg）。四行公式的前两行合成。"""
        sign, offset = self.joint_sign_offset(name_or_index)
        return bus_deg_to_urdf_deg(self.raw_to_bus_deg(name_or_index, raw), sign, offset)

    def degrees_to_raw(self, name_or_index: str | int, deg: float) -> int:
        """URDF 关节角（deg）→ 总线原始读数。四行公式的后两行合成。"""
        sign, offset = self.joint_sign_offset(name_or_index)
        return self.bus_deg_to_raw(name_or_index, urdf_deg_to_bus_deg(float(deg), sign, offset))

    # --- 电流与速度（标定指南 4.3）---

    def raw_current_to_ma(self, name_or_index: str | int, raw: int) -> float:
        """4.3：current_ma = (decoded_raw_current - current_zero_raw)*current_ma_per_raw。

        读的是 Present_Current 而不是 Present_Load —— 4.3 明确"禁止把
        Present_Load 当 Present_Current"。运行期过流比较用绝对值（2.1 末段），
        那是 executor/夹爪状态机的职责，本函数只负责给出带符号的物理电流。
        """
        ax = self.axis(name_or_index)
        decoded = self.decode_signed(ax, "Present_Current", raw)
        i = ax.index
        return (float(decoded) - float(self.motor.current_zero_raw[i])) * float(
            self.motor.current_ma_per_raw[i]
        )

    def raw_velocity_to_deg_s(self, name_or_index: str | int, raw: int) -> float:
        """速度读数 → 接口单位（姿态关节 deg/s、夹爪 %/s）。

        标定指南 4.3/4.4 的两条路径：
          * 姿态关节：先按厂商比例 velocity_deg_s_per_raw 转成角速度，再乘
            joints.sign，使速度正方向与 URDF 关节正方向一致（4.3"速度先按厂商
            比例转换，关节再乘 joints.sign"；技术文档 4.2 也要求 speeds_deg_s
            用 URDF 正方向）。
          * 夹爪：接口的 gripper_speed_pct_s 单位是 %/s，4.4 给的是
                pct_s = raw_count_speed * 100 / (open_raw - closed_raw)
            其中 raw_count_speed 是"由已换算的轴角速度换回对应编码计数/秒"，
            即 axis_deg_s / count_to_deg。分母带符号，所以"正值表示张开"在
            open_raw < closed_raw 时同样成立。
        """
        ax = self.axis(name_or_index)
        decoded = self.decode_signed(ax, "Present_Velocity", raw)
        axis_deg_s = float(decoded) * float(self.motor.velocity_deg_s_per_raw[ax.index])
        if not ax.is_gripper:
            return axis_deg_s * float(self.joints.sign[ax.index])
        raw_count_speed = axis_deg_s / ax.count_to_deg
        return raw_count_speed * 100.0 / float(self.gripper_open_raw - self.gripper_closed_raw)

    # --- 夹爪百分比（标定指南 4.4）---

    def raw_to_gripper_pct(self, raw: int) -> float:
        """夹爪原始读数 → 开度百分比（端点取自本通道的标定值）。"""
        return raw_to_gripper_pct(raw, self.gripper_closed_raw, self.gripper_open_raw)

    def gripper_pct_to_raw(self, pct: float) -> int:
        """开度百分比 → 夹爪原始读数。"""
        return gripper_pct_to_raw(pct, self.gripper_closed_raw, self.gripper_open_raw)

    # --- 限位查询（只查询、不裁剪：clamp-free）---

    def clamp_free_limits_deg(self) -> tuple[FloatArray, FloatArray]:
        """返回 5 个姿态关节的有效限位 (lower, upper)，单位 deg。

        直接复用 configs.motion_params.effective_joint_limits（技术文档 5.1 的
        公式），本模块不重新实现第二份交集逻辑；B/C 共用同一计算结果正是
        标定指南 5 节第 5 条的要求。

        "clamp_free" 的含义：这个查询只用来**拒绝**越限命令，代码里不存在拿它
        去 np.clip 命令值的路径（技术文档 4.2："越限抛 MotorLimitError，
        不静默裁剪"）。
        """
        if self._limits_cache is None:
            if self._urdf_limits_deg is None:
                raise ValueError(
                    "MotorMapping 构造时没有提供 urdf_limits_deg，无法计算有效限位；"
                    "请改用 MotorMapping.from_params(...)"
                )
            self._limits_cache = effective_joint_limits(self._urdf_limits_deg, self.joints)
        return self._limits_cache

    # --- 便捷构造 ---

    @classmethod
    def from_params(
        cls, params: MotionParams, calibration: Mapping[str, Mapping[str, int]] | None = None
    ) -> "MotorMapping":
        """从 MotionParams 构造；calibration 缺省时按 model.motor_calibration_path 读盘。

        路径解析走 params.motor_calibration_path_resolved()，相对路径以
        profile.json 所在目录为基准（标定指南 1.1）。
        """
        if calibration is None:
            calibration = load_motor_calibration(params.motor_calibration_path_resolved())
        return cls(params.motor, params.joints, calibration, params.urdf_limits_deg)

    @classmethod
    def from_profile(
        cls, profile_path: Path, calibration_path: Path | None = None
    ) -> "MotorMapping":
        """从 profile.json + 官方校准文件构造（标定工具与单元测试的离线入口）。

        用 mode="mock" 加载：这条入口只做离线核对，不要求实机 verified 状态与
        验收报告，但结构、shape、限位交集等校验全部照做。
        """
        from configs.motion_params import load_motion_params

        params = load_motion_params(Path(profile_path), mode="mock")
        calib = load_motor_calibration(
            Path(calibration_path)
            if calibration_path is not None
            else params.motor_calibration_path_resolved()
        )
        return cls(params.motor, params.joints, calib, params.urdf_limits_deg)


# ---------------------------------------------------------------------------
# 5. STS/SCS 串口协议层（protocol_version = 0）
# ---------------------------------------------------------------------------


def _feedback_block_layout(
    pos_spec: tuple[int, int],
    vel_spec: tuple[int, int],
    cur_spec: tuple[int, int],
) -> tuple[int, int, dict[str, int]]:
    """从 vendor 寄存器定义推导"每台一次读全三个反馈量"的连续块。

    STS3215 的 Present_Position / Present_Velocity / Present_Current 是
    56/58/69（各 2 字节）：取最小地址为块起点、最大"地址+宽度-1"为块终点，
    得到 56..70 共 15 字节。一台一次 READ 事务即可拿齐位置/速度/电流，
    六台共 6 次（审阅指引 3.6：比旧的 12 次少一半事务，同一台各量的读取
    间隔也更接近）。块中间的 Present_Load / 电压 / 温度 / 运动标志按原样
    透传、一律不解码——尤其禁止把 Present_Load 当电流（标定指南 4.3）。
    起止地址由表推导而不是抄云端的硬编码 56/15，换型号时自动跟随 vendor 表。

    返回 (块起始地址, 块长度, {寄存器名: 块内偏移})。
    """
    specs = {
        "Present_Position": tuple(pos_spec),
        "Present_Velocity": tuple(vel_spec),
        "Present_Current": tuple(cur_spec),
    }
    start = min(a for a, _ in specs.values())
    end = max(a + s - 1 for a, s in specs.values())
    offsets = {name: a - start for name, (a, _) in specs.items()}
    return start, end - start + 1, offsets


class ByteTransport(Protocol):
    """协议层需要的最小字节管道抽象。

    为什么用 Protocol 而不是继承：单元测试要能塞进一个假管道（标定指南 11 节
    要求故障试验覆盖"通信失败"），真机用 pyserial 适配器，两者没有也不需要
    共同基类。实现约定（协议层的时限语义建立在这几条之上）：
      * write(data, timeout_s) 返回实际写出的字节数；短写由协议层判为通信错误。
        timeout_s 是**整次调用剩下的预算**，实现必须把阻塞时长限制在它以内
        （pyserial 用 write_timeout；假总线推进注入的时钟），而不是收下不用。
      * read(n, timeout_s) **至多**返回 n 字节；允许阻塞，但不得长于
        min(实现自身的阻塞分片, timeout_s)。协议层每轮循环还会重新检查绝对
        deadline，两层一起保证总时限不会被拉长。
      * read_available() 必须非阻塞，返回当前已到达的字节（可能为空）。
      * flush_input() 丢弃接收缓冲里的残留，避免上一次超时留下的迟到字节污染
        下一帧。
      * close() 释放底层资源，允许重复调用。
    """

    def write(self, data: bytes, timeout_s: float | None = None) -> int: ...

    def read(self, n: int, timeout_s: float | None = None) -> bytes: ...

    def read_available(self) -> bytes: ...

    def flush_input(self) -> None: ...

    def close(self) -> None: ...


class StsProtocol:
    """STS 系列 protocol_version=0 的组包、解包与有界收发。

    报文格式（Feetech 串行手册的原始格式；技术文档第五节禁止"改变协议格式"）：

        指令帧  FF FF | ID | LENGTH | INSTRUCTION | PARAM... | CHECKSUM
        状态帧  FF FF | ID | LENGTH | ERROR | PARAM... | CHECKSUM
        LENGTH   = 参数个数 + 2
        CHECKSUM = ~(从 ID 到最后一个参数所有字节之和) & 0xFF

    寄存器值按小端 2 字节（1 字节寄存器按 1 字节）。

    protocol 0 的参数字段里寄存器地址只占 1 字节；protocol 1（SCS 系列）才是
    2 字节地址 + CRC16。本类只支持 protocol 0（MODEL_PROTOCOL["sts3215"] == 0），
    构造驱动的 mapping 时会核对型号，这里也拒绝超出 1 字节的地址。
    指令码与报文格式已按手册示例报文（"读 ID 1 地址 56 长度 2 =
    FF FF 01 04 02 38 02 BE"）由**独立于本类的**测试断言钉死
    （tests/test_motor_control.py 第 3 节）；但地址宽度、应答延迟等
    仍无法从 vendor 的 Python 表里推出来，必须在 motor 标定阶段
    用真机核对（标定指南 4.1 第 1 条、8.2 第 2 条）。
    """

    HEADER = (0xFF, 0xFF)
    BROADCAST_ID = 0xFE

    # 指令码（Feetech 串行总线舵机通信协议手册，protocol_version=0）。
    # 上一版把 READ 写成 0x04、WRITE 写成 0x05、REG_WRITE 写成 0x06，
    # "读 ID 1 地址 56 长度 2"会生成 FF FF 01 04 04 38 02 BC 而不是厂商示例的
    # FF FF 01 04 02 38 02 BE（审阅指引 3.1）。正确性以 tests/test_motor_control.py
    # 里**独立于本类常量**构造的标准报文断言为准，不能拿本表当自查依据。
    INST_RESET = 0x00       # 复位：型号/固件相关，本驱动不提供封装、不调用（见下）
    INST_PING = 0x01
    INST_READ_DATA = 0x02
    INST_WRITE_DATA = 0x03
    INST_REG_WRITE = 0x04   # 异步写入：锁存参数，等 INST_ACTION 触发
    INST_ACTION = 0x05      # 执行此前 REG_WRITE 锁存的内容
    INST_SYNC_WRITE = 0x83

    # 复位(0x00)与"REG_WRITE+ACTION 两段式提交"(0x04/0x05)在这里只登记常量、
    # 不提供指令封装：复位按型号/固件差异大且不可逆，本任务没有需求（prompt
    # §5.2），旧版 reg_write_action() 把 0x04 和 0x05 混成一条 0x06 的误导性
    # 封装已随本表一并移除（审阅指引 3.1）。生产路径只有一条写运动目标的
    # SYNC_WRITE（写完立即执行），不需要两段式提交。

    # 状态帧 ERROR 字节的位定义（STS3215 手册 RETURN_ERROR 位）。bit6/7 的含义
    # 随固件版本有差异，所以报错消息里永远附带原始掩码；"哪些位必须停机"要在
    # motor 标定阶段实测核对（标定指南 4.1）。
    ERROR_BIT_NAMES = {
        0: "电压异常",
        1: "行程末端超时/过载",
        2: "位置偏差（跟随误差）超限",
        3: "目标位置超出行程",
        4: "无效指令",
        5: "编程数据异常",
        6: "电流保护",
        7: "自动回零未完成/状态位",
    }

    def __init__(
        self,
        transport: ByteTransport,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        # 接收帧的分段读取会一次多拿一些字节，未消费的部分暂存在这里（属于
        # 接收分帧状态，不是应用数据缓存）。
        self._pending = bytearray()
        # clock 与 sleep 可注入：单元测试需要一个"每调用一次就前进固定步长"的
        # 假时钟来验证超时，以及一个不真正睡觉的 sleep。
        self._clock = clock
        self._sleep = sleep
        # 运行期 I/O 重试次数为 0（技术文档 4.2 末段）。把它做成可见的属性，
        # 是为了让"不重试"这件事写在代码里，而不是靠读者信任循环结构。
        self.num_retry = 0
        # 每台舵机 WRITE 是否有应答：{servo_id: bool}。由控制器 initialize()
        # 逐台读回 Response_Status_Level（vendor 表地址 8）后配置，见
        # set_write_response()。未配置时默认 True（等应答）——那是单元测试
        # 直连路径的保守值；真机运行前一定先过 initialize()，所以生产语义是
        # "按设备实际应答配置"，不是按注释里的出厂默认猜（prompt §5.2）。
        self._write_ack: dict[int, bool] = {}

    def set_write_response(self, servo_id: int, expects_ack: bool) -> None:
        """登记某台舵机 WRITE 指令的**实际**应答配置（来自寄存器读回）。"""
        self._write_ack[int(servo_id)] = bool(expects_ack)

    def write_expects_ack(self, servo_id: int) -> bool:
        """查询当前 WRITE 应答策略（诊断与测试用）。"""
        return self._write_ack.get(int(servo_id), True)

    # --- 纯函数：组包与解包（不需要 transport，便于单元测试）---

    @staticmethod
    def checksum(body: Sequence[int]) -> int:
        """CHECKSUM = ~(从 ID 到最后一个参数所有字节之和) & 0xFF。"""
        return (~sum(int(b) & 0xFF for b in body)) & 0xFF

    @classmethod
    def build_packet(cls, servo_id: int, instruction: int, params: Sequence[int]) -> bytes:
        """组一条指令帧：FF FF ID (len(params)+2) INSTR PARAM... CHECKSUM。"""
        body = [int(servo_id) & 0xFF, (len(params) + 2) & 0xFF, int(instruction) & 0xFF]
        body.extend(int(p) & 0xFF for p in params)
        return bytes(bytearray(cls.HEADER) + bytearray(body) + bytearray([cls.checksum(body)]))

    @classmethod
    def parse_status_packet(
        cls, raw: bytes, expect_id: int | None = None
    ) -> tuple[int, int, bytes]:
        """解一条状态帧，返回 (ID, ERROR, PARAMS)。

        这里只判格式与校验和，不判故障：ERROR 字节原样返回，让调用方决定
        "非 0 = 硬件故障 → MotorStateError"（技术文档 4.2 的错误分类）。
        帧头/长度/校验和/条数问题一律是 MotorCommunicationError。
        """
        if len(raw) < 6:
            raise MotorCommunicationError(f"状态帧过短（{len(raw)} 字节）：{raw.hex(' ')!r}")
        if (raw[0], raw[1]) != tuple(cls.HEADER):
            raise MotorCommunicationError(f"状态帧帧头错误：{raw.hex(' ')!r}")
        length = raw[3]
        # LENGTH = 参数个数 + 2，参数含 ERROR，所以最短是 2（ERROR + CHECKSUM）。
        if length < 2:
            raise MotorCommunicationError(f"状态帧 LENGTH={length} 非法（应 >= 2）")
        total = 4 + length  # 帧头2 + ID + LENGTH + LENGTH(=ERROR..CHECKSUM)
        if len(raw) < total:
            raise MotorCommunicationError(
                f"状态帧不完整：应有 {total} 字节，实际 {len(raw)}：{raw.hex(' ')!r}"
            )
        servo_id = raw[2]
        # 校验和覆盖"从 ID 到最后一个参数"，即不含帧头、不含校验和自身。
        body = [int(b) for b in raw[2 : total - 1]]
        checksum = raw[total - 1]
        if cls.checksum(body) != checksum:
            raise MotorCommunicationError(
                f"舵机 {servo_id} 状态帧校验和不匹配（收到 0x{checksum:02X}）："
                f"{bytes(raw[:total]).hex(' ')!r}"
            )
        if expect_id is not None and servo_id != int(expect_id):
            raise MotorCommunicationError(
                f"应答 ID 与请求 ID 不符：请求 {expect_id}，收到 {servo_id}"
            )
        error = raw[4]
        params = bytes(raw[5 : total - 1])
        return servo_id, error, params

    @staticmethod
    def split_u16(value: int) -> list[int]:
        """寄存器值按小端 2 字节拆分。"""
        v = int(value) & 0xFFFF
        return [v & 0xFF, (v >> 8) & 0xFF]

    @staticmethod
    def join_u16(data: bytes) -> int:
        """小端 2 字节拼成无符号整数。"""
        if len(data) < 2:
            raise MotorCommunicationError(f"拼 16 位寄存器值需要 2 字节，实际 {len(data)}")
        return int(data[0]) | (int(data[1]) << 8)

    @classmethod
    def describe_error(cls, error: int) -> str:
        """把 ERROR 字节翻成可读的位说明，用于异常消息。"""
        hits = [
            f"bit{bit}={name}" for bit, name in cls.ERROR_BIT_NAMES.items() if error & (1 << bit)
        ]
        return "; ".join(hits) if hits else f"未定义故障位（掩码 0x{error:02X}）"

    # --- 有界收发（所有公开方法共用调用方传入的绝对 deadline）---

    def reset_input(self) -> None:
        """清空接收方向的全部缓冲（transport + 协议内部 pushback）。

        初始化与故障恢复后调用一次，把上一阶段残留的字节丢掉；否则第一个请求
        会去解析上一帧的尾巴，表现为随机的校验和错误。
        """
        self._drain_input()

    def _require_time(self, deadline_s: float, what: str) -> None:
        """整次调用总时限检查：过期立刻抛错，不重试（技术文档 4.2）。"""
        if self._clock() >= deadline_s:
            raise MotorCommunicationError(
                f"{what} 超出整次调用总时限（deadline 已到；运行期 I/O 不重试）"
            )

    def _send(self, packet: bytes, deadline_s: float, what: str) -> None:
        remaining = deadline_s - self._clock()
        if remaining <= 0.0:
            raise MotorCommunicationError(
                f"发送 {what} 超出整次调用总时限（deadline 已到；运行期 I/O 不重试）"
            )
        try:
            # 剩余预算真正交给底层：SerialTransport 会先设 pyserial write_timeout
            # 再写，慢设备/坏驱动不会把整次方法吊死（审阅指引 3.3）。
            written = self._transport.write(packet, timeout_s=remaining)
        except MotorError:
            raise
        except Exception as exc:
            # 串口层的 OSError / 断开等一律归为通信错误，且保留原始原因，
            # 不允许裸异常漏到上层（技术文档 4.2 的错误分类）。
            raise MotorCommunicationError(f"发送 {what} 时底层写异常：{exc}") from exc
        # 阻塞调用返回后再次检查时间：慢写即使最终写完了，也不能宣称满足预算。
        self._require_time(deadline_s, f"发送 {what}")
        if written != len(packet):
            # 短写说明整帧没进 TX 缓冲，后面的应答一定会错配，只能当通信失败。
            raise MotorCommunicationError(
                f"发送 {what} 短写：{written}/{len(packet)} 字节（帧 {packet.hex(' ')}）"
            )

    def _drain_input(self) -> None:
        """丢弃接收缓冲与协议内部的剩余字节。

        pushback 缓冲里留着的一定是上一次调用没读完的字节（超时或被吞掉的应答），
        下一次请求前必须清掉，否则会污染下一帧的解析。
        """
        self._pending.clear()
        self._transport.flush_input()

    def _read_bytes(self, n: int, deadline_s: float, what: str) -> bytes:
        """读满 n 字节，或总时限到期抛 MotorCommunicationError。

        先非阻塞轮询 read_available()；连续几次为空就交给一次有界阻塞 read()
        让出 CPU —— 控制线程每个 tick 还要做规划与监控（技术文档 6.1），不能
        在这里忙等。deadline 才是唯一的超时机制。

        read_available() 可能一次就把整帧都倒出来，而本方法只要 n 字节，所以
        多出来的部分必须放进 pushback 缓冲留给同一帧的下一个字段。早先的版本
        直接截断丢弃，表现为"读帧头时把后面 5 个字节扔掉，然后永远等不到
        LENGTH"——这种挂死比报错更难查，所以这里写成显式的两段缓冲。
        """
        buf = bytearray(self._pending)
        del self._pending[:]
        polls = 0
        while len(buf) < n:
            self._require_time(deadline_s, f"接收 {what}")
            try:
                chunk = self._transport.read_available()
            except MotorError:
                raise
            except Exception as exc:
                raise MotorCommunicationError(f"接收 {what} 时底层读异常：{exc}") from exc
            if not chunk:
                polls += 1
                if polls > _POLLS_BEFORE_BLOCK:
                    polls = 0
                    # 阻塞分片也不得越过剩余预算：把 deadline 换算成 timeout
                    # 交给 transport.read()，让底层调用自己有界。
                    remaining = max(0.0, deadline_s - self._clock())
                    try:
                        chunk = self._transport.read(
                            max(1, n - len(buf)), timeout_s=remaining
                        )
                    except MotorError:
                        raise
                    except Exception as exc:
                        raise MotorCommunicationError(
                            f"接收 {what} 时底层读异常：{exc}"
                        ) from exc
            if chunk:
                buf += chunk
                continue
            self._sleep(_POLL_SLEEP_S)
        out = bytes(buf[:n])
        self._pending.extend(buf[n:])
        return out

    def _recv_status(
        self, expect_id: int | None, deadline_s: float, what: str
    ) -> tuple[int, int, bytes]:
        """收一条状态帧：逐字节找帧头，再按 LENGTH 收完整帧。

        为什么要先扫帧头：上一次超时留下的迟到字节可能还躺在接收缓冲里，把它们
        当成当前应答会让校验和在错误的位置失败，表现为"偶发通信错误"这种最难
        查的故障。发单播请求前调用方会 flush_input()，这里再兜一层。
        """
        b0 = self._read_bytes(1, deadline_s, f"{what} 帧头")
        while b0[0] != self.HEADER[0]:
            b0 = self._read_bytes(1, deadline_s, f"{what} 帧头")
        b1 = self._read_bytes(1, deadline_s, f"{what} 帧头")
        if b1[0] != self.HEADER[1]:
            raise MotorCommunicationError(f"{what} 帧头不完整：FF {b1[0]:02X}")
        # b0/b1 就是帧头的两个 FF，不能再重复拼进去，否则后面每个字段都会错位
        # 一字节（错位后的 LENGTH 会变成上一个 FF，导致一次无意义的长读）。
        head = bytearray(self.HEADER)
        rest = self._read_bytes(2, deadline_s, f"{what} ID/LENGTH")
        length = rest[1]
        # LENGTH 是应答帧的 ERROR+参数+校验和 总长。STS 单次读最多几十字节，
        # 超过这个上限的一定是错位或噪声帧；早点报错，不要拿着坏长度去阻塞读。
        if length > 200:
            raise MotorCommunicationError(
                f"{what} LENGTH={length} 不合理（帧头错位或噪声字节）：{(head + rest).hex(' ')}"
            )
        tail = self._read_bytes(length, deadline_s, f"{what} 负载")
        return self.parse_status_packet(bytes(head + rest + tail), expect_id)

    def _request(
        self,
        servo_id: int,
        instruction: int,
        params: Sequence[int],
        deadline_s: float,
        what: str,
    ) -> tuple[int, bytes]:
        """单播请求 + 等应答，并把非 0 的 ERROR 字节翻译成 MotorStateError。"""
        self._drain_input()
        self._send(self.build_packet(servo_id, instruction, params), deadline_s, what)
        rid, error, rparams = self._recv_status(servo_id, deadline_s, what)
        if error:
            # 技术文档 4.2：设备报告硬件故障 → MotorStateError。
            raise MotorStateError(
                f"舵机 {rid} 报告硬件故障 ERROR=0x{error:02X}"
                f"（{self.describe_error(error)}），请求 {what}"
            )
        return rid, rparams

    # --- 指令封装 ---

    def ping(self, servo_id: int, deadline_s: float) -> int:
        """PING（0x01，无参数）。初始化时用它确认"这个 ID 在总线上活着"。"""
        rid, _ = self._request(servo_id, self.INST_PING, [], deadline_s, f"PING {servo_id}")
        return rid

    def read_registers(self, servo_id: int, addr: int, length: int, deadline_s: float) -> bytes:
        """READ_DATA（0x02）：params = [地址, 字节数]，返回原始参数字节。"""
        if not 0 <= int(addr) <= 0xFF:
            raise ValueError(f"protocol 0 的参数地址只占 1 字节，{addr} 超出范围")
        _, params = self._request(
            servo_id,
            self.INST_READ_DATA,
            [int(addr), int(length)],
            deadline_s,
            f"READ id={servo_id} addr={addr} n={length}",
        )
        if len(params) != int(length):
            raise MotorCommunicationError(
                f"舵机 {servo_id} 从地址 {addr} 读回 {len(params)} 字节，应为 {length} 字节"
            )
        return params

    def write_registers(
        self,
        servo_id: int,
        addr: int,
        data: Sequence[int],
        deadline_s: float,
        *,
        expect_ack: bool | None = None,
    ) -> None:
        """WRITE_DATA（0x03）：单舵机写若干**字节**；是否等应答按设备实际配置。

        data 是字节序列而不是整数，因为寄存器宽度不统一（Torque_Enable 1 字节、
        Goal_Position 2 字节）；2 字节值请用 StsProtocol.split_u16 生成。

        expect_ack=None 时用 set_write_response() 登记的策略（默认 True）。
        该策略来自逐台读回的 Response_Status_Level：出厂"只回读指令"时 WRITE
        本来就不应答，硬等应答会把整次预算耗光；反过来设备全应答时不读状态帧
        就会让迟到应答污染下一次请求——所以必须按**实际配置**处理，而不是按
        注释里的默认值假设（prompt §5.2）。应答配置未知的单元保持默认等待。

        语义边界：等待并校验应答 → 能确认"这台收到了、且未报告故障"；不等
        应答 → 只承诺"整帧已提交给串口"，不承诺设备已执行。两种情况都与
        "设备已到位"无关（技术文档 4.2 要求区分这三层）。
        """
        if not 0 <= int(addr) <= 0xFF:
            raise ValueError(f"protocol 0 的参数地址只占 1 字节，{addr} 超出范围")
        params: list[int] = [int(addr)]
        params.extend(int(b) & 0xFF for b in data)
        what = f"WRITE id={servo_id} addr={addr}"
        wait = self.write_expects_ack(servo_id) if expect_ack is None else bool(expect_ack)
        if wait:
            self._request(servo_id, self.INST_WRITE_DATA, params, deadline_s, what)
        else:
            self._drain_input()
            self._send(
                self.build_packet(servo_id, self.INST_WRITE_DATA, params), deadline_s, what
            )

    def bulk_read(
        self, servo_ids: Sequence[int], addr: int, length: int, deadline_s: float
    ) -> dict[int, bytes]:
        """逐个 READ_DATA 读回多台舵机的同一寄存器块，返回 {id: 原始字节}。

        为什么不用同步读回包：protocol_version=0 没有可靠的 SYNC_READ（Feetech
        的 SYNC_READ 属于 protocol 1 的扩展指令，且多台同时应答会总线冲突），
        所以只能一台一条 READ_DATA。但**所有台共用同一个绝对 deadline**，因此
        总时限不会像朴素实现那样变成 N 倍（技术文档 4.2 第 3 段、标定指南 8.2
        第 2 条）。

        任何一台失败（总时限到期、校验和不对、条数不对）都抛
        MotorCommunicationError，并且绝不把已读到的部分当结果返回——这就是
        "部分读回"的处理方式。
        """
        unique_ids = {int(s) for s in servo_ids}
        out: dict[int, bytes] = {}
        for sid in servo_ids:
            # 每台请求前检查总时限：deadline 一到就抛，不开始下一台。
            self._require_time(deadline_s, f"bulk_read 第 {len(out) + 1}/{len(unique_ids)} 台")
            out[int(sid)] = self.read_registers(int(sid), addr, length, deadline_s)
        # 条数显式复查：正常路径下循环结束就一定够，但"返回 dict"这件事必须被
        # 证明完整（请求里出现重复 ID 时字典会静默合并，那也要报错）。
        if len(out) != len(unique_ids):
            raise MotorCommunicationError(
                f"bulk_read 只读回 {len(out)} 条，请求 {len(unique_ids)} 条（部分读回）"
            )
        return out

    def sync_write_targets(
        self,
        servo_ids: Sequence[int],
        addr: int,
        length: int,
        values: Sequence[Sequence[int] | int],
        deadline_s: float,
    ) -> None:
        """SYNC_WRITE（0x83）：一帧给多台舵机的同一寄存器写值。

        帧格式（protocol 0，1 字节寄存器地址，广播 ID 0xFE）：
            FF FF FE (LEN=参数数+2) 83 | ADDR | SIZE_PER_SERVO | (ID + DATA)*N | CHK
        每台的 DATA 占 length 字节（2 字节寄存器用小端）。

        语义边界（技术文档 4.2 明文要求）：同步写是**广播、没有应答帧**，所以
        本方法只能校验"整帧已提交给操作系统发送缓冲"，不能宣称 6 台舵机都收到
        了、更不能宣称它们会同时开始运动——总线上没有回包就没有任何证据。
        因此这里不读状态帧、不返回成功标志，只在短写或总时限到期时抛
        MotorCommunicationError。
        """
        ids = [int(s) for s in servo_ids]
        if len(ids) != len(values):
            raise ValueError(f"sync_write：{len(ids)} 个 ID 对 {len(values)} 组数据")
        if not 0 <= int(addr) <= 0xFF:
            raise ValueError(f"protocol 0 的参数地址只占 1 字节，{addr} 超出范围")
        payload: list[int] = [int(addr) & 0xFF, int(length) & 0xFF]
        for sid, value in zip(ids, values):
            payload.append(sid & 0xFF)
            if isinstance(value, int):
                payload.extend(self.split_u16(value))
            else:
                payload.extend(int(b) & 0xFF for b in value)
        self._send(
            self.build_packet(self.BROADCAST_ID, self.INST_SYNC_WRITE, payload),
            deadline_s,
            f"SYNC_WRITE addr={addr} ids={ids}",
        )

    # 旧版这里的 reg_write_action()/reset_servo() 已删除：前者把 REG_WRITE(0x04)
    # 与 ACTION(0x05) 混成一条不存在的 0x06，后者是不可逆的恢复出厂指令，
    # 型号/固件相关且本任务无需求（审阅指引 3.1、prompt §5.2）。本项目
    # 的 send_action 只走单帧 0x83 立即执行模式，不需要两段式提交；
    # 复位/EEPROM 操作属于官方标定工具的职责，不进入驱动。


# ---------------------------------------------------------------------------
# 6. pyserial 适配器（延迟导入）
# ---------------------------------------------------------------------------


class SerialTransport:
    """把 pyserial 的 Serial 对象适配成 ByteTransport。

    ``import serial`` 故意写在构造函数里（见模块 docstring 的依赖边界与
    libs/so_arm_core/SOURCE.md 第 2 节）：pyserial 只有真机驱动路径需要，
    开发机/CI/单元测试跑换算层时不应该因为没装 pyserial 而连模块都 import 不了。

    时限落实（审阅指引 3.3）：上一版只设读 timeout、不设 write_timeout，
    pyserial 默认写阻塞无上限，"整次预算"在慢写路径上不成立。现在打开时
    就带一个有限的写超时，并且每次 write/read 都接受协议层传来的
    timeout_s=剩余预算，在调用前落到 pyserial 的 write_timeout / timeout 上。
    仍然不调用 ``flush()``：它要等发送缓冲真正排空，会把"提交完成"变成
    "排空完成"并破坏预算；USB 排空延迟无法由软件承诺，列为真机待测项。
    串口异常统一包成 MotorCommunicationError（保留原因）；只有"设备未就绪"
    （未打开/已关闭）按 MotorStateError 分类。
    """

    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        read_slice_s: float = _SERIAL_READ_SLICE_S,
        write_timeout_s: float = _SERIAL_WRITE_TIMEOUT_S,
    ) -> None:
        try:
            import serial  # 延迟导入：全项目只有这一处使用 pyserial
        except ImportError as exc:  # pragma: no cover - 取决于本机环境
            raise MotorStateError(
                "真机电机驱动需要 pyserial（本机未安装）。换算层（MotorMapping 与"
                "第 2 节的共享函数）不需要它，可以照常做离线单元测试。"
            ) from exc
        self._read_slice_s = float(read_slice_s)
        try:
            # timeout 是"单次 read() 的阻塞分片"，不是整次调用的时限（见常量注释）；
            # write_timeout 有限，运行期还会按剩余预算逐次收紧。
            self._serial = serial.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=read_slice_s,
                write_timeout=write_timeout_s,
            )
        except Exception as exc:  # pyserial 的错误类型很多，统一归为"设备未就绪"
            raise MotorStateError(f"无法打开串口 {port}@{baudrate}：{exc}") from exc
        self._closed = False

    @classmethod
    def wrap(cls, serial_like: object) -> "SerialTransport":
        """包装一个已经打开的 pyserial 对象（便于标定工具复用同一句柄）。"""
        obj = cls.__new__(cls)
        obj._serial = serial_like  # type: ignore[assignment]
        obj._read_slice_s = _SERIAL_READ_SLICE_S
        obj._closed = False
        return obj

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        self._check_open()
        try:
            if timeout_s is not None:
                # 剩余预算落实到本次写：pyserial 的 write_timeout 限制的是
                # 把字节拷进操作系统发送缓冲的时间（非硬实时，见类注释）。
                self._serial.write_timeout = max(float(timeout_s), 1e-4)  # type: ignore[attr-defined]
            return int(self._serial.write(data))  # type: ignore[attr-defined]
        except Exception as exc:
            raise MotorCommunicationError(f"串口写失败：{exc}") from exc

    def read(self, n: int, timeout_s: float | None = None) -> bytes:
        self._check_open()
        try:
            if timeout_s is not None:
                # 阻塞分片不超过 min(常规分片, 剩余预算)，两者取小。
                self._serial.timeout = min(  # type: ignore[attr-defined]
                    self._read_slice_s, max(float(timeout_s), 0.0)
                )
            return bytes(self._serial.read(int(n)))  # type: ignore[attr-defined]
        except Exception as exc:
            raise MotorCommunicationError(f"串口读失败：{exc}") from exc

    def read_available(self) -> bytes:
        self._check_open()
        try:
            waiting = int(getattr(self._serial, "in_waiting", 0) or 0)
            if waiting <= 0:
                return b""
            return bytes(self._serial.read(waiting))  # type: ignore[attr-defined]
        except Exception as exc:
            raise MotorCommunicationError(f"串口读失败：{exc}") from exc

    def flush_input(self) -> None:
        self._check_open()
        try:
            self._serial.reset_input_buffer()  # type: ignore[attr-defined]
        except Exception as exc:
            raise MotorCommunicationError(f"清接收缓冲失败：{exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._serial.close()  # type: ignore[attr-defined]

    def _check_open(self) -> None:
        if self._closed:
            raise MotorStateError("串口已关闭，设备未连接")


# ---------------------------------------------------------------------------
# 7. MotorController 实现
# ---------------------------------------------------------------------------


class Sts3215MotorController:
    """实现 configs.common_interface.MotorController 协议的 STS3215 驱动。

    与技术文档 4.2 那张表的逐条对应：
      send_action   整条校验通过后一次 sync_write；受 write_timeout_s；不等到位
      get_feedback  一次完整读回 6 台 ×(位置/速度/电流)；受 read_timeout_s；跨度
                    超 max_feedback_span_s 判为无效采样并抛错，不返回缓存
      hold_current  读一次 + 写一次，不循环

    错误分类严格按技术文档 4.2：命令越限 MotorLimitError / 超时与不完整读回
    MotorCommunicationError / 未连接、未使能力矩、硬件故障 MotorStateError。

    本类不持有线程、定时器或周期任务（技术文档 6.1：全部在当前调用线程顺序
    执行；4.2 末段："三个方法均不持有周期循环"）。力矩与身份核对只在
    initialize() 里显式做一次，没有任何隐藏的自动回零（4.2 末段、6.4）。
    """

    def __init__(
        self,
        params: MotionParams,
        mapping: MotorMapping | None = None,
        *,
        transport: ByteTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        initialize: bool = True,
    ) -> None:
        self.params = params
        self.timing = params.timing
        self._clock = clock
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        # 没给 mapping 就从 profile 的 motor+joints 与官方校准文件构造。
        self.mapping = mapping if mapping is not None else MotorMapping.from_params(params)

        # 只支持 STS 系列（protocol 0）。MODEL_PROTOCOL 来自 vendor 表，
        # 不在这里另写一份型号→协议映射。
        for ax in self.mapping.axes:
            if MODEL_PROTOCOL.get(ax.model, 1) != 0:
                raise ValueError(
                    f"{ax.name} 的型号 {ax.model!r} 不是 protocol_version=0 的 STS 系列，"
                    "本驱动不支持"
                )

        if transport is None:
            # 真机路径：延迟导入 pyserial（见 SerialTransport 的说明）。
            transport = SerialTransport(params.motor.port, params.motor.baudrate)
        self._transport = transport
        self._protocol = StsProtocol(transport, clock=clock, sleep=sleep)
        self._connected = True
        self._torque_enabled = False
        self._sequence = 0

        # 批量读写用的寄存器地址：按 vendor 表解析一次，并要求 6 个通道一致
        # （MotorMapping.common_register 的理由）。
        self._goal_addr, self._goal_len = self.mapping.common_register("Goal_Position")
        self._pos_addr, self._pos_len = self.mapping.common_register("Present_Position")
        self._vel_addr, self._vel_len = self.mapping.common_register("Present_Velocity")
        self._cur_addr, self._cur_len = self.mapping.common_register("Present_Current")
        for reg, size in (
            ("Goal_Position", self._goal_len),
            ("Present_Position", self._pos_len),
            ("Present_Velocity", self._vel_len),
            ("Present_Current", self._cur_len),
        ):
            # 本文件的编解码都按 2 字节小端写；宽度不同就必须改代码，而不是
            # 悄悄按 2 字节读出错位的数据。
            if size != 2:
                raise ValueError(f"寄存器 {reg} 在 vendor 表里的宽度是 {size} 字节，本驱动只实现 2")
        # 一次事务读全每台的位置/速度/电流：块起止从 vendor 寄存器定义推导
        # （STS3215 → 56..70，15 字节），不复制云端硬编码。中间的 Present_Load
        # 等字节透传但不解码（标定指南 4.3）。
        (self._fb_addr, self._fb_len, self._fb_off) = _feedback_block_layout(
            (self._pos_addr, self._pos_len),
            (self._vel_addr, self._vel_len),
            (self._cur_addr, self._cur_len),
        )

        if initialize:
            self.initialize()

    # --- 生命周期 ---

    def initialize(
        self,
        *,
        verify_identity: bool = True,
        enable_torque: bool = True,
        timeout_s: float = _INIT_TIMEOUT_S,
    ) -> None:
        """连接与 Torque_Enable 配置。**不做**回零、复位或写行程限位。

        技术文档 4.2 末段："连接和力矩配置由驱动的初始化流程完成，不能隐藏自动
        回零动作"。官方 homing 与范围采集属于标定流程（标定指南 4.1 第 3 条），
        由操作者在官方工具里完成，本项目只消费其结果，因此这里没有任何写
        Homing_Offset / Min_Max_Position_Limit / RESET 的路径。

        verify_identity=True 时逐台 PING 并读回 Model_Number 与固件版本，与
        profile 的 motor.models / motor.firmware 比对（标定指南 2.1：这两项的
        测量来源就是"实际读回"）。比对的意义在于换算：current_ma_per_raw 与
        velocity_deg_s_per_raw 是按型号/固件从厂商资料得到的（4.3），型号或固件
        对不上就说明换算比例可能张冠李戴，宁可拒绝启动。
        单元测试与离线只读核对可以用 verify_identity=False 跳过。

        timeout_s：初始化不是周期调用，内部约有 36 次事务（每台应答配置读回 +
        PING + 型号 + 固件×2 + 力矩），所以不用面向每个 tick 的 write_timeout_s，
        而是用一个单独的
        显式预算。原则不变——**一个绝对 deadline 覆盖本次调用里的全部事务**，
        不会退化成"每台一个超时"。真值需要 timing 阶段实测（标定指南 8.2）。
        """
        self._require_connected()
        # 一个绝对 deadline 覆盖本次初始化里的全部寄存器访问（不按台数倍增）。
        deadline_s = self._clock() + timeout_s
        self._protocol.reset_input()
        # 先读回每台的**实际**应答配置，再做任何 WRITE（包括力矩）：
        # WRITE 有没有状态帧由 Response_Status_Level 决定，等错方向要么耗光
        # 预算、要么把迟到应答当成当前结果（prompt §5.2）。这是一次纯 READ
        # 批量（6 事务），不做任何 EEPROM 写入。
        self._configure_write_ack(deadline_s)
        if verify_identity:
            for ax in self.mapping.axes:
                self._protocol.ping(ax.servo_id, deadline_s)
                self._verify_identity(ax, deadline_s)
        if enable_torque:
            self.configure_torque(True, deadline_s=deadline_s)

    def _configure_write_ack(self, deadline_s: float) -> None:
        """逐台读 Response_Status_Level（vendor 表地址 8），配置 WRITE 应答策略。

        STS 手册取值：0=所有指令应答；1=只回读指令（WRITE 不应答）；2=全静默。
        等级 2 连 READ 都不应答，下面的读会在预算内以 MotorCommunicationError
        失败——那正是"设备配置与驱动不匹配"该有的显式报错，不去猜它的含义。
        只读不写：本页不改任何 EEPROM 配置（prompt §5.1：初始化不做隐藏的
        回零/恢复出厂/写 EEPROM）。
        """
        addr, length = self.mapping.common_register("Response_Status_Level")
        if length != 1:
            raise ValueError(f"Response_Status_Level 应为 1 字节寄存器，vendor 表给出 {length}")
        for ax in self.mapping.axes:
            raw = int(self._protocol.read_registers(ax.servo_id, addr, 1, deadline_s)[0])
            if raw not in (0, 1, 2):
                raise MotorStateError(
                    f"{ax.name}（id={ax.servo_id}）的 Response_Status_Level 读回 {raw}，"
                    "不在协议表登记的 0/1/2 内，无法确定写应答语义"
                )
            self._protocol.set_write_response(ax.servo_id, expects_ack=(raw == 0))

    def configure_torque(self, enable: bool, *, deadline_s: float | None = None) -> None:
        """逐台写 Torque_Enable；初始化里调用一次，标定阶段卸力时也会用。

        为什么用单播 WRITE_DATA 而不是 sync_write：力矩开关不是同步运动命令，
        逐台写才能定位"哪一台没配好"；卸力示教与急停时这个信息很重要。
        能否真的"确认收到"取决于该台的实际应答配置（write_registers 按
        Response_Status_Level 决定是否校验状态帧），设备全静默时只承诺提交。
        """
        self._require_ready()
        end = (
            deadline_s
            if deadline_s is not None
            else self._clock() + self.timing.write_timeout_s
        )
        addr, length = self.mapping.common_register("Torque_Enable")
        if length != 1:
            raise ValueError(f"Torque_Enable 应为 1 字节寄存器，vendor 表给出 {length}")
        for ax in self.mapping.axes:
            self._protocol.write_registers(ax.servo_id, addr, [1 if enable else 0], end)
        self._torque_enabled = bool(enable)

    def read_torque_enabled(self, name: str = "shoulder_pan") -> int:
        """读回某个舵机的 Torque_Enable，供标定工具与故障核对使用。

        运行期不做这个检查：它要多花一次总线事务，而力矩状态在 initialize()
        之后不由本驱动改变（卸力只能通过 configure_torque 显式做）。
        """
        self._require_connected()
        addr, length = self.mapping.register(name, "Torque_Enable")
        end = self._clock() + self.timing.read_timeout_s
        raw = self._protocol.read_registers(self.mapping.axis(name).servo_id, addr, length, end)
        return int(raw[0])

    def close(self) -> None:
        """释放串口。可重复调用；调用后三个协议方法都抛 MotorStateError。"""
        if not self._connected:
            return
        self._connected = False
        self._torque_enabled = False
        self._transport.close()

    def __enter__(self) -> "Sts3215MotorController":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- MotorController 协议三方法（技术文档 4.2）---

    def send_action(self, joints_deg: FloatArray, gripper_pct: float) -> None:
        """校验后单次同步提交，受写超时约束，不等待到位，不起线程。

        步骤刻意是"先整条校验、全部换算完、再一次性写"（技术文档 4.2
        "整条参数/限位校验在写入前完成"）：
            1) shape / 有限性 / 六个量的有效限位检查 —— 任何一项不过，整条拒绝
            2) 六个量全部换算成总线原始值（含最终指令限位检查）
            3) 一条 SYNC_WRITE 广播帧提交
        返回只表示"提交完成"。同步写没有应答，所以本方法不宣称 6 台都收到了，
        也不宣称物理事务原子性（技术文档 4.2 明文禁止这种说法）。
        """
        self._require_actuating_ready()
        deadline_s = self._clock() + self.timing.write_timeout_s
        raws = self._validate_and_encode(joints_deg, gripper_pct)
        ids = [ax.servo_id for ax in self.mapping.axes]
        values = [StsProtocol.split_u16(raw) for raw in raws]
        # 一条广播帧：这就是"一次性"的全部含义。不逐台写、不重试、不等到位。
        self._protocol.sync_write_targets(ids, self._goal_addr, self._goal_len, values, deadline_s)

    def get_feedback(self) -> JointFeedback:
        """受总读时限约束地完整读回 6 台并换算单位。

        技术文档 4.2 与标定指南 8.2 第 3 条的要点：
          * 每台一次 READ 事务读回 56..70 连续块（位置/速度/电流；块起止由
            vendor 寄存器定义推导，见 _feedback_block_layout），六台共 6 次，
            全部共用同一个绝对 deadline（read_timeout_s），总时限不随台数倍增；
            块中间的 Present_Load 等字节透传但不解码（标定指南 4.3）；
          * 采样起止时间来自本机 monotonic_ns（可注入），sequence 只在整次
            完整成功读回后递增；
          * 跨度超过 timing.max_feedback_span_s 的采样无效 —— 6 台的读数不再
            对应同一物理时刻，拿它做跟踪误差判断会把"读得慢"误判成"关节在动"，
            所以直接抛 MotorCommunicationError，**不返回缓存也不返回残缺**；
          * 电流与速度必须是换算后的物理量（4.3：禁止把寄存器整数原样填进带
            mA/deg_s 名称的接口）。
        """
        self._require_connected()
        sample_start_ns = self._monotonic_ns()
        # 整次方法只有一个 deadline；六台块读共用它，所以 6 次单播事务加起来
        # 也不能超过 read_timeout_s。
        deadline_s = self._clock() + self.timing.read_timeout_s
        ids = [ax.servo_id for ax in self.mapping.axes]

        block = self._protocol.bulk_read(ids, self._fb_addr, self._fb_len, deadline_s)
        op = self._fb_off["Present_Position"]
        ov = self._fb_off["Present_Velocity"]
        oc = self._fb_off["Present_Current"]
        pos_raw = {
            sid: StsProtocol.join_u16(b[op : op + self._pos_len]) for sid, b in block.items()
        }
        vel_raw = {
            sid: StsProtocol.join_u16(b[ov : ov + self._vel_len]) for sid, b in block.items()
        }
        cur_raw = {
            sid: StsProtocol.join_u16(b[oc : oc + self._cur_len]) for sid, b in block.items()
        }
        # 结束时间戳贴着最后一次 I/O 取：后面的单位换算是纯 CPU 计算，不计入
        # 采样跨度；而 age 从 sample_start_ns 起算（8.2 第 3 条），所以这里不能
        # 把结束时间往前挪来"美化"跨度。
        sample_end_ns = self._monotonic_ns()

        span_s = (sample_end_ns - sample_start_ns) / 1e9
        if span_s > self.timing.max_feedback_span_s:
            raise MotorCommunicationError(
                f"整次读回跨度 {span_s * 1000:.2f}ms 超过 max_feedback_span_s"
                f"（{self.timing.max_feedback_span_s * 1000:.2f}ms），判为无效采样；"
                "不使用缓存也不返回残缺数据（技术文档 4.2）"
            )
        # 应答完整性复查：bulk_read 已经保证 6 条都在，但 JointFeedback 是
        # (5,)+夹爪的定长数组，缺任何一个都会 IndexError 而不是清晰报错。
        missing = [ax.servo_id for ax in self.mapping.axes if ax.servo_id not in pos_raw]
        if missing:
            raise MotorCommunicationError(f"读回结果缺少舵机 {missing}（部分读回）")

        m = self.mapping
        angles = np.empty(len(JOINT_NAMES), dtype=np.float64)
        speeds = np.empty(len(JOINT_NAMES), dtype=np.float64)
        currents = np.empty(len(JOINT_NAMES), dtype=np.float64)
        for i, name in enumerate(JOINT_NAMES):
            sid = int(m.axes[i].servo_id)
            angles[i] = m.raw_to_degrees(i, pos_raw[sid])
            speeds[i] = m.raw_velocity_to_deg_s(i, vel_raw[sid])
            currents[i] = m.raw_current_to_ma(i, cur_raw[sid])
        grip_sid = int(m.axes[len(JOINT_NAMES)].servo_id)
        # 夹爪读数越出标定端点时这里抛 MotorLimitError，而不是裁成 0/100
        # （标定指南 4.4）。
        gripper_pct = m.raw_to_gripper_pct(pos_raw[grip_sid])

        self._sequence += 1  # 只有整次完整成功读回才递增（common_interface 注释）
        return JointFeedback(
            angles_deg=angles,
            speeds_deg_s=speeds,
            currents_ma=currents,
            gripper_pct=float(gripper_pct),
            gripper_speed_pct_s=float(m.raw_velocity_to_deg_s(len(JOINT_NAMES), vel_raw[grip_sid])),
            gripper_current_ma=float(m.raw_current_to_ma(len(JOINT_NAMES), cur_raw[grip_sid])),
            sample_start_ns=int(sample_start_ns),
            sample_end_ns=int(sample_end_ns),
            sequence=int(self._sequence),
        )

    def hold_current(self) -> JointFeedback:
        """有界地读一次有效反馈，再把同样的五关节角与夹爪开度写回去。

        一次读、一次写，不循环（技术文档 4.2）。写回的目标就是当前位置，所以
        正常情况下不产生运动，只是把"已提交的位置目标"重新固定住。

        预算组合（标定指南 8.2；不新增配置字段）：读段受 timing.read_timeout_s
        约束、写段受 timing.write_timeout_s 约束，各自是独立的整次方法预算，
        本方法最坏合计两者之和；文档与实现以此为准。

        失败语义：读失败直接抛出，**不会**发送一个猜出来的保持目标；写失败
        同样抛出，不伪报"保持已提交"。返回后没有任何监控或续写循环。

        保持目标走的是与 send_action 完全相同的整条校验：如果实测位置已经在
        有效限位之外，那是机构/标定/外力造成的异常，本方法按"失败上报"处理，
        让上层进入 FAULT_UNCONTROLLED（技术文档 6.3 状态表、6.4），而不是把
        一个越限目标再次写进舵机。它也不等同于物理急停（4.2 明示）。
        """
        feedback = self.get_feedback()
        self.send_action(feedback.angles_deg, feedback.gripper_pct)
        return feedback

    # --- 内部：整条校验与编码 ---

    def _validate_and_encode(self, joints_deg: NDArray, gripper_pct: float) -> list[int]:
        """整条校验 + 全部换算；任何一项不过就整条拒绝（技术文档 4.2）。

        返回按 MOTOR_NAMES 顺序（前 5 个姿态关节、最后夹爪）的 Goal_Position
        原始值。这里不发送任何东西，所以失败时总线上一帧都没出去过——这正是
        "整条校验在写入前完成"的可验证形式。
        """
        # --- 1) 形状与有限性 ---
        try:
            q = np.asarray(joints_deg, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MotorLimitError(f"joints_deg 无法转成 float64：{exc}") from exc
        if q.shape != (len(JOINT_NAMES),):
            raise MotorLimitError(
                f"joints_deg shape 应为 ({len(JOINT_NAMES)},)，实际 {q.shape}"
            )
        if not np.all(np.isfinite(q)):
            raise MotorLimitError(f"joints_deg 含非有限值：{q.tolist()}")
        # bool/str/complex 都能被 float() 接受或给出难懂的错误，显式挡掉。
        if isinstance(gripper_pct, (bool, np.bool_, str, bytes, complex)):
            raise MotorLimitError(
                f"gripper_pct 应为实数标量，实际是 {type(gripper_pct).__name__}"
            )
        g = float(gripper_pct)
        if not math.isfinite(g):
            raise MotorLimitError(f"gripper_pct 含非有限值：{gripper_pct!r}")

        # --- 2) 六个量全部在有效限位内（越限整条拒绝，不裁剪）---
        lower, upper = self.mapping.clamp_free_limits_deg()
        bad = np.where((q < lower - LIMIT_TOL_DEG) | (q > upper + LIMIT_TOL_DEG))[0]
        if bad.size:
            detail = "; ".join(
                f"{JOINT_NAMES[i]}={q[i]:.3f}deg 限位[{lower[i]:.3f},{upper[i]:.3f}]"
                for i in bad.tolist()
            )
            raise MotorLimitError(f"关节角越限，整条命令拒绝（不静默裁剪）：{detail}")
        if g < GRIPPER_PCT_MIN - PCT_TOL or g > GRIPPER_PCT_MAX + PCT_TOL:
            raise MotorLimitError(
                f"夹爪开度 {g:.3f}% 越出 [{GRIPPER_PCT_MIN:.0f}, {GRIPPER_PCT_MAX:.0f}]"
                "（0%=标定闭合端、100%=标定张开端），整条命令拒绝"
            )

        # --- 3) 全部换算（校验通过后才会走到这里）---
        m = self.mapping
        raws = [m.degrees_to_raw(i, float(q[i])) for i in range(len(JOINT_NAMES))]
        raws.append(m.gripper_pct_to_raw(g))

        # --- 4) 最终指令限位（技术文档 1.1：驱动负责这一层）---
        # 官方校准的 [range_min, range_max] 是舵机自己的行程端点。越出它说明
        # sign / zero_offset_deg / 限位三者之中有东西错了，写下去就是拿舵机去撞
        # 硬端点，比通信失败更糟，所以宁可拒绝。
        for ax, raw in zip(m.axes, raws):
            lo = ax.range_min - FINAL_RAW_TOL_COUNT
            hi = ax.range_max + FINAL_RAW_TOL_COUNT
            if not lo <= raw <= hi:
                raise MotorLimitError(
                    f"{ax.name} 编码后的目标 {raw} 越出官方行程 [{ax.range_min}, "
                    f"{ax.range_max}]（含 {FINAL_RAW_TOL_COUNT} 计数取整余量），"
                    "说明方向/零位/限位配置互相矛盾，整条命令拒绝"
                )
        return raws

    def _verify_identity(self, ax: ServoAxis, deadline_s: float) -> None:
        """读回 Model_Number 与固件版本，与 profile 登记值比对。

        型号与固件是换算比例的出处（标定指南 4.3："current_ma_per_raw 先根据
        当前型号/固件的厂商资料确定"），对不上就没有理由相信 mA 和 deg/s。
        """
        expect_model = MODEL_NUMBER_TABLE.get(ax.model)
        addr, length = self.mapping.register(ax, "Model_Number")
        got = StsProtocol.join_u16(self._protocol.read_registers(ax.servo_id, addr, length, deadline_s))
        if expect_model is not None and got != int(expect_model):
            raise MotorStateError(
                f"{ax.name}（id={ax.servo_id}）型号读回 {got}，配置登记 "
                f"{ax.model}={expect_model}；说明电机顺序或 profile 的 motor.models 错了"
            )
        major_addr, _ = self.mapping.register(ax, "Firmware_Major_Version")
        minor_addr, _ = self.mapping.register(ax, "Firmware_Minor_Version")
        major = self._protocol.read_registers(ax.servo_id, major_addr, 1, deadline_s)[0]
        minor = self._protocol.read_registers(ax.servo_id, minor_addr, 1, deadline_s)[0]
        got_fw = f"{major}.{minor}"
        want_fw = str(ax.firmware).strip()
        if _fw_tuple(want_fw) != _fw_tuple(got_fw):
            raise MotorStateError(
                f"{ax.name}（id={ax.servo_id}）固件读回 {got_fw}，配置登记 {want_fw}；"
                "电流与速度换算比例按固件版本取厂商资料（标定指南 4.3），"
                "不一致时拒绝运行"
            )

    # --- 内部：状态检查 ---

    def _require_connected(self) -> None:
        if not self._connected:
            raise MotorStateError("驱动未连接（串口未打开或已关闭）")

    def _require_ready(self) -> None:
        """允许读写寄存器：必须已连接。

        单独留一个方法名，是为了让 get_feedback（只需要连接）与
        send_action（还需要力矩）使用不同的前置条件，读代码时不会混淆。
        """
        self._require_connected()

    def _require_actuating_ready(self) -> None:
        """允许提交运动命令：必须已连接且力矩已使能。

        力矩没开时写 Goal_Position 不会产生任何运动，静默"成功"返回是最坏结果
        （上层会以为动作已提交，但机械臂根本不动），所以按"设备未就绪"抛
        MotorStateError（技术文档 4.2）。
        """
        self._require_ready()
        if not self._torque_enabled:
            raise MotorStateError("舵机未使能力矩，拒绝提交运动命令（设备未就绪）")


def _fw_tuple(text: str) -> tuple[int, ...]:
    """把 "2.54" / "2.54.0" 这类固件串拆成可比较的整数元组。

    用元组而不是字符串比较，是为了让 "2.5" 与 "2.5.0" 视为同一版本，不至于
    因为 profile 里少写一个 .0 就拒绝启动真机。无法解析时返回哨兵值，比对
    必然失败 —— 宁可报"固件不符"，也不要静默跳过这项检查。
    """
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            return (-1,)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


# ---------------------------------------------------------------------------
# 8. CalibrationReader：标定采集专用的只读总线入口
# ---------------------------------------------------------------------------

# 采集读预算（秒）。控制 tick 的 timing.read_timeout_s 是给 30Hz 闭环用的，
# 而 capture 是操作者节奏的静态读回（指南 3.3），两者量级不同。它仍然是一个
# 绝对 deadline 覆盖本次快照里的全部事务，与运行期原则一致（指南 4.2）。
_CALIB_READ_TIMEOUT_S = 5.0


def _placeholder_joints() -> JointsParams:
    """joints 尚未拟合时让 MotorMapping 构造通过的占位值。

    采集层只用到 raw→q_bus_deg 换算（那只依赖官方校准的行程中点），占位的
    sign/zero_offset 不参与任何采集数据的输出；URDF 角只有在 profile.joints
    真正标定后才由 CalibrationReader 暴露。
    """
    return JointsParams(
        sign=np.ones(5, dtype=np.int64),
        zero_offset_deg=np.zeros(5, dtype=np.float64),
        measured_limits_deg=np.array([[-180.0, 180.0]] * 5),
        application_limits_deg=np.array([[-180.0, 180.0]] * 5),
        margin_deg=np.full(5, 0.1),
        hard_current_ma=np.full(5, 900.0),
    )


class CalibrationReader:
    """标定采集 --hardware 链路的只读总线（标定指南 3.3/4.1/4.2）。

    为什么不复用 Sts3215MotorController：那条构造链要求 profile 能通过
    load_motion_params(mode="real")——status=verified、全部运行期字段非 null、
    哈希与报告匹配——而采集恰恰发生在这些字段还没测出来的 draft 期，实机采集
    因此结构性死锁（P1-5 缺陷 1）。本类只依赖 load_calibration_profile 给出的
    最小字段集，并且**只有读能力**：没有任何提交运动命令的方法，动臂与卸力
    点动属于指南第 3 节规定的人工操作。
    """

    def __init__(
        self,
        profile,                      # configs.motion_params.CalibrationProfile
        *,
        transport: ByteTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        read_timeout_s: float = _CALIB_READ_TIMEOUT_S,
    ) -> None:
        self.profile = profile
        # 夹爪两端读数在 motor 阶段拟合前是 null（它们本来就是该阶段的产物）。
        # MotorMapping 构造会把端点缓存成 int，所以给它一对仅占位的哨兵值；
        # 开度换算只在真实端点存在时才执行（见 read_snapshot）。
        self._gripper_calibrated = profile.motor.gripper_closed_raw is not None
        motor_for_map = profile.motor
        if not self._gripper_calibrated:
            from dataclasses import replace

            motor_for_map = replace(profile.motor, gripper_closed_raw=0,
                                    gripper_open_raw=1)
        self.mapping = MotorMapping(
            motor_for_map,
            profile.joints if profile.joints is not None else _placeholder_joints(),
            profile.motor_calibration,
            profile.urdf_limits_deg,
        )
        if transport is None:
            # 真机路径：延迟导入 pyserial（SerialTransport 的依赖边界说明）。
            transport = SerialTransport(profile.motor.port, profile.motor.baudrate)
        self._transport = transport
        self._protocol = StsProtocol(transport, clock=clock, sleep=sleep)
        self._clock = clock
        self._monotonic_ns = monotonic_ns
        self._read_timeout_s = float(read_timeout_s)
        self._pos_addr, self._pos_len = self.mapping.common_register("Present_Position")
        self._vel_addr, self._vel_len = self.mapping.common_register("Present_Velocity")
        self._cur_addr, self._cur_len = self.mapping.common_register("Present_Current")
        if self._pos_len != 2 or self._vel_len != 2 or self._cur_len != 2:
            raise ValueError("CalibrationReader 按 2 字节小端寄存器编写，vendor 表宽度不符")
        # 与 Sts3215MotorController 同一份块布局推导（vendor 表来源，不硬编码）。
        (self._fb_addr, self._fb_len, self._fb_off) = _feedback_block_layout(
            (self._pos_addr, self._pos_len),
            (self._vel_addr, self._vel_len),
            (self._cur_addr, self._cur_len),
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._transport.close()

    def read_snapshot(self) -> dict:
        """一次有界事务读回 6 台的位置/速度/电流寄存器。

        每台一次 READ 连续块（六台共 6 次事务，共用同一个绝对 deadline），
        原始寄存器值按 vendor 表偏移拆出；Present_Load 等中间字节透传不解码
        （标定指南 4.3）。返回 dict（键名与指南 4.1/5 的 JSONL payload 对齐）：
          raw_position / raw_velocity / raw_current：按 MOTOR_NAMES 顺序的
            寄存器原始无符号值（未解码，供 fit 与离线复算）；
          sample_start_ns / sample_end_ns：本次采样的起止时间戳；
          q_bus_deg：5 个姿态关节的总线角（4.2 第一行公式，只依赖官方校准）；
          q_urdf_deg：joints 已拟合时的 URDF 角，否则 None；
          gripper_pct：按 4.4 百分比公式换算的开度，读数越出标定端点抛
            MotorLimitError（异常而不是裁剪）。
        """
        if self._closed:
            raise MotorStateError("CalibrationReader 已关闭")
        deadline = self._clock() + self._read_timeout_s
        ids = [ax.servo_id for ax in self.mapping.axes]
        sample_start_ns = self._monotonic_ns()
        block = self._protocol.bulk_read(ids, self._fb_addr, self._fb_len, deadline)
        op = self._fb_off["Present_Position"]
        ov = self._fb_off["Present_Velocity"]
        oc = self._fb_off["Present_Current"]
        pos = {
            sid: StsProtocol.join_u16(b[op : op + self._pos_len]) for sid, b in block.items()
        }
        vel = {
            sid: StsProtocol.join_u16(b[ov : ov + self._vel_len]) for sid, b in block.items()
        }
        cur = {
            sid: StsProtocol.join_u16(b[oc : oc + self._cur_len]) for sid, b in block.items()
        }
        sample_end_ns = self._monotonic_ns()

        m = self.mapping
        bus_deg = [m.raw_to_bus_deg(ax, pos[ax.servo_id]) for ax in m.axes[:len(JOINT_NAMES)]]
        urdf_deg: list[float] | None = None
        if self.profile.joints is not None:
            sign = np.asarray(self.profile.joints.sign, dtype=np.float64)
            offset = np.asarray(self.profile.joints.zero_offset_deg, dtype=np.float64)
            urdf_deg = [
                float(bus_deg_to_urdf_deg(b, float(sign[i]), float(offset[i])))
                for i, b in enumerate(bus_deg)
            ]
        grip_sid = int(m.axes[len(JOINT_NAMES)].servo_id)
        return {
            "raw_position": [int(pos[ax.servo_id]) for ax in m.axes],
            "raw_velocity": [int(vel[ax.servo_id]) for ax in m.axes],
            "raw_current": [int(cur[ax.servo_id]) for ax in m.axes],
            "sample_start_ns": int(sample_start_ns),
            "sample_end_ns": int(sample_end_ns),
            "q_bus_deg": [float(v) for v in bus_deg],
            "q_urdf_deg": urdf_deg,
            "gripper_pct": (float(m.raw_to_gripper_pct(pos[grip_sid]))
                            if self._gripper_calibrated else None),
        }
