"""STS3215 驱动层测试（模拟设备的协议真值独立于被测实现）。

覆盖 docs/真机参数测量与标定指南.md 第 4 节（关节换算、电流/速度单位、夹爪两端）、
第 12 节共享函数与"必要离线检查"第 2、6 条，docs/机械臂运动控制模块技术文档.md
第 4.2 节（三个协议方法的语义），以及根目录电机驱动修改 Prompt §7 的验收项。

为什么本文件必须自己实现一份"厂商协议"而不是从被测类导入常量：
    上一版假总线用 ``StsProtocol.INST_READ_DATA`` 等常量判断请求，与被测实现
    共享同一组**错误**指令码，导致 37 项测试全绿而真实报文完全不符（审阅指引
    3.2）。本文件的模拟设备按《飞特串行总线舵机通信协议手册》protocol_version=0
    独立实现（见下方 VENDOR_* 常量与 vendor_* 函数），标准报文断言使用手册
    示例的字面字节；实现若再改错指令码，测试会立刻失败。

真机串口不参与测试：全部走注入的假 transport 与假时钟。没有装 pyserial 的机器
也必须能 import 驱动模块并跑完换算层与协议层用例（见 test_没有pyserial时仍可import）。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import qingyun.grabbing.motor_control as MC
from configs.common_interface import (
    MOTOR_NAMES,
    MotorCommunicationError,
    MotorLimitError,
    MotorStateError,
)
from configs.motion_params import effective_joint_limits
from qingyun.grabbing.motor_control import (
    StsProtocol,
    Sts3215MotorController,
    bus_deg_to_urdf_deg,
    gripper_pct_to_raw,
    load_motor_calibration,
    raw_to_gripper_pct,
    urdf_deg_to_bus_deg,
)
from tests.support import SIM_PROFILE, raw_profile


# ---------------------------------------------------------------------------
# 0. 独立的厂商协议真值（Feetech 串行总线舵机协议手册，protocol_version=0）
#
# 这里的数值抄自手册本身，绝不 import 被测驱动的任何指令常量。
# 手册示例：读 ID 1、起始地址 56、长度 2 的完整请求帧为
#     FF FF 01 04 02 38 02 BE
# ---------------------------------------------------------------------------

VENDOR_INST_RESET = 0x00
VENDOR_INST_PING = 0x01
VENDOR_INST_READ = 0x02
VENDOR_INST_WRITE = 0x03
VENDOR_INST_REG_WRITE = 0x04
VENDOR_INST_ACTION = 0x05
VENDOR_INST_SYNC_WRITE = 0x83
VENDOR_BROADCAST_ID = 0xFE

# STS3215 vendor 控制表里、启动握手要碰的寄存器地址（独立于被测实现，按厂商手册
# 硬编码，供假总线判断"隐式上力/清理读回"发生在哪个寄存器上）。
STS_TORQUE_ENABLE = 40
STS_GOAL_POSITION = 42
STS_PRESENT_POSITION = 56
STS_RESPONSE_STATUS_LEVEL = 8
# EEPROM/配置类寄存器：初始化任何路径都不得写入（prompt §5.1、标定指南 4.1）。
STS_EEPROM_ADDRS = frozenset({5, 6, 8, 9, 11, 31})  # ID/CW_Ang/RLS_Ang/RRL/RLR/Homing


def vendor_checksum(body) -> int:
    """CHECKSUM = ~(ID..最后参数之和) & 0xFF（手册第 3 节）。"""
    return (~sum(int(b) & 0xFF for b in body)) & 0xFF


def vendor_packet(servo_id: int, instruction: int, params) -> bytes:
    """独立构造一条指令帧（模拟主机侧的"正确答案"，不依赖被测实现）。"""
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes(bytearray([0xFF, 0xFF, *body, vendor_checksum(body)]))


def vendor_status(servo_id: int, payload, error: int = 0) -> bytes:
    """独立构造一条状态帧：FF FF ID (len+2) ERROR 数据... CHK。"""
    body = [servo_id, len(payload) + 2, error, *payload]
    return bytes(bytearray([0xFF, 0xFF, *body, vendor_checksum(body)]))


def vendor_split_u16(value: int) -> list[int]:
    v = int(value) & 0xFFFF
    return [v & 0xFF, (v >> 8) & 0xFF]


# ---------------------------------------------------------------------------
# 假总线：按上面的厂商真值应答的 STS 从机集合，允许注入各种故障
# ---------------------------------------------------------------------------


class FakeBus:
    """一个足够真实的 STS 从机集合，用于在无硬件条件下验证协议层。

    实现三件事：READ 回状态帧、WRITE 按应答配置回帧、SYNC_WRITE 落寄存器。
    故障开关：不应答（全局或指定台）、坏校验和、错 ID、ERROR 位、短应答。
    设备行为完全按 ``inst == VENDOR_*`` 判断——如果被测驱动发出未知指令码，
    本类直接 AssertionError 指名道姓，绝不用空应答把它糊弄过去。

    Response_Status_Level（地址 8，每台可不同，默认未写 → 按 0 处理）：
        0 = 所有指令都应答（含 WRITE）；
        1 = 只回应 READ/PING，WRITE 不应答；
    新从机的 EPROM 区（固件 0/1、型号 3~4）由构造函数预置，供初始化链路核对。
    """

    def __init__(
        self,
        ids=(1, 2, 3, 4, 5, 6),
        model_number=777,
        firmware=(2, 54),
    ) -> None:
        self.registers: dict[int, dict[int, int]] = {i: {} for i in ids}
        self.model_number = model_number
        for sid in ids:
            lo, hi = vendor_split_u16(model_number)
            # 只读区：Firmware_Major(0) / Firmware_Minor(1) / Model_Number(3,2B)
            self.registers[sid][0] = firmware[0] & 0xFF
            self.registers[sid][1] = firmware[1] & 0xFF
            self.registers[sid][3] = lo
            self.registers[sid][4] = hi
        self.tx_log: list[bytes] = []
        self.rx_queue = bytearray()
        self.fail_no_response = False
        self.fail_bad_checksum = False
        self.fail_wrong_id = False
        self.fail_error_bit = 0
        self.fail_short_reply = False
        self.fail_ids: set[int] = set()          # 这些台完全不应答
        self.deliver_after_polls = 0             # >0：应答延迟到第 N 次空轮询后才出现
        self._staged = bytearray()
        self._polls = 0
        self.write_count = 0
        self.sync_write_count = 0
        self.read_request_count = 0
        self.write_timeouts: list[float | None] = []
        self.read_timeouts: list[float | None] = []

    # --- 设备配置辅助 ---

    def set_write_level(self, servo_id: int, level: int) -> None:
        """设置该台的 Response_Status_Level（0=全应答，1=只回应读）。"""
        self.registers.setdefault(servo_id, {})[8] = level & 0xFF

    def response_level(self, servo_id: int) -> int:
        return self.registers.get(servo_id, {}).get(8, 0)

    def inject_stale(self, frame: bytes) -> None:
        """往接收缓冲塞一条"上一次超时留下的迟到应答"，测试污染防护。"""
        self.rx_queue += frame

    # --- ByteTransport 协议（timeout_s 是整次调用剩下的预算，模拟设备会用到） ---

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        data = bytes(data)
        self.tx_log.append(data)
        self.write_count += 1
        self.write_timeouts.append(timeout_s)
        assert data[:2] == b"\xff\xff", f"帧头不对：{data[:2].hex()}"
        servo_id, length, inst = data[2], data[3], data[4]
        params = list(data[5 : 5 + length - 2])
        silent = self.fail_no_response or servo_id in self.fail_ids
        if inst == VENDOR_INST_SYNC_WRITE:
            self.sync_write_count += 1
            addr, size = params[0], params[1]
            for k in range(2, len(params), size + 1):
                sid = params[k]
                for off in range(size):
                    self.registers.setdefault(sid, {})[addr + off] = params[k + 1 + off]
            return len(data)                    # 广播永远不回包
        if inst == VENDOR_INST_PING:
            if not silent:
                self._reply(servo_id, [])
            return len(data)
        if inst == VENDOR_INST_READ:
            addr, size = params[0], params[1]
            self.read_request_count += 1
            if silent:
                return len(data)
            payload = [
                self.registers.get(servo_id, {}).get(addr + off, 0) & 0xFF
                for off in range(size)
            ]
            if self.fail_short_reply:
                payload = payload[:1]
            self._reply(servo_id, payload)
            return len(data)
        if inst == VENDOR_INST_WRITE:
            addr = params[0]
            for off in range(len(params) - 1):
                self.registers.setdefault(servo_id, {})[addr + off] = params[1 + off]
            # 手册：等级 1 时 WRITE 不应答。等错应答方向会耗光预算或漏掉故障，
            # 所以这里必须按寄存器值走，而不是无脑回帧。
            if not silent and self.response_level(servo_id) == 0:
                self._reply(servo_id, [])
            return len(data)
        raise AssertionError(
            f"模拟舵机收到未知指令 0x{inst:02X}（帧 {data.hex(' ')}）——"
            "被测实现的指令码与厂商手册不符"
        )

    def _reply(self, servo_id: int, payload: list[int]) -> None:
        sid = servo_id
        if self.fail_wrong_id:
            sid = (servo_id + 7) & 0xFF
        err = self.fail_error_bit
        body = [sid, len(payload) + 2, err, *payload]
        chk = vendor_checksum(body)
        if self.fail_bad_checksum:
            chk = (chk + 1) & 0xFF
        frame = bytes([0xFF, 0xFF, *body, chk])
        if self.deliver_after_polls > 0:
            self._staged += frame
        else:
            self.rx_queue += frame

    def read(self, n: int, timeout_s: float | None = None) -> bytes:
        self.read_timeouts.append(timeout_s)
        src = self.rx_queue if not self._staged else self._staged
        out = bytes(src[:n])
        del src[:n]
        self._polls = 0
        return out

    def read_available(self) -> bytes:
        if self._staged:
            self._polls += 1
            if self._polls > self.deliver_after_polls:
                out = bytes(self._staged)
                self._staged.clear()
                self._polls = 0
                return out
            return b""
        out = bytes(self.rx_queue)
        del self.rx_queue[:]
        return out

    def flush_input(self) -> None:
        self.rx_queue.clear()
        self._staged.clear()
        self._polls = 0

    def close(self) -> None:
        pass

    # --- 便捷：把寄存器按 2 字节小端写进从机 ---

    def set_u16(self, servo_id: int, addr: int, value: int) -> None:
        lo, hi = vendor_split_u16(int(value))
        regs = self.registers.setdefault(servo_id, {})
        regs[addr] = lo
        regs[addr + 1] = hi


class SlowWriteBus(FakeBus):
    """慢设备：每次写花 delay_s（推进共享假时钟），模拟拔线前的小卡死。"""

    def __init__(self, clock: "Clock", delay_s: float, **kw) -> None:
        super().__init__(**kw)
        self._clock = clock
        self._delay_s = delay_s

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        self._clock.t += self._delay_s
        return super().write(data, timeout_s=timeout_s)


class SlowReadBus(FakeBus):
    """慢从机：每次完整应答消耗 delay_s（推进共享假时钟）。"""

    def __init__(self, clock: "Clock", delay_s: float, **kw) -> None:
        super().__init__(**kw)
        self._clock = clock
        self._delay_s = delay_s

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        n = self.read_request_count
        out = super().write(data, timeout_s=timeout_s)
        if self.read_request_count > n:
            self._clock.t += self._delay_s
        return out


class ExplodingSyncWriteBus(FakeBus):
    """同步写在串口层抛 OSError：模拟"读到一半还好、提交保持目标时掉线"。"""

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        if bytes(data)[4] == VENDOR_INST_SYNC_WRITE:
            raise OSError("模拟：TX 失败/设备被拔掉")
        return super().write(data, timeout_s=timeout_s)


class Clock:
    """每次调用前进固定步长的假单调时钟。

    sleep 必须真的推进本时钟：协议层的轮询循环靠 `_clock()` 逼近绝对 deadline
    来退出（motor_control._read_bytes），注入一个"什么都不做"的 sleep 会让
    等不到应答的循环永远转下去。所以时钟与 sleep 成对注入，二者共享同一条时间轴。
    """

    def __init__(self, start: float = 0.0, step_s: float = 0.0) -> None:
        self.t = start
        self.step_s = step_s

    def __call__(self) -> float:
        self.t += self.step_s
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)

    def pair(self):
        return self, self.sleep


class StartupBus(FakeBus):
    """FakeBus + P1 §4.1 启动握手专用故障/行为钩子（CAL-052 隐式上力等）。

    所有钩子默认关闭，绝不改变基类（既有 60 余条用例）的行为。只在握手事务里
    触发对应现象，让 T28–T30 能对"事务序列 + 有界清理"逐项断言：

      goal_write_torque_effect  None | 'all' | {servo_id,...}：广播 Goal_Position
          （CAL-052）后把这些轴的 Torque_Enable 置 1，模拟"写目标即可能上力"。
      goal_write_present_shift  {servo_id: delta_raw}：广播目标时把该轴
          Present_Position(56) 平移 delta_raw，模拟上力后的重力下落/位姿跳变。
      read_error_by_addr        {addr: error_byte}：读该地址回 ERROR 位（设备故障）。
      corrupt_goal_read_ids     {servo_id}：读回 Goal_Position 时翻转字节（读回不匹配）。
      cleanup_write_fail_ids    {servo_id}：清理阶段失能写(Torque_Enable=0)失败。
      cleanup_read_fail_ids     {servo_id}：清理阶段读回 Torque_Enable 失败（仅在该轴
          已发生过一次 Torque_Enable=0 写之后生效，用于区分握手期的正常力矩读）。
      explode_sync_write        True：广播 Goal_Position 时串口层直接抛错（写失败）。
      cleanup_write_delay_s + clock：只有进入清理（已出现 Torque_Enable=0 写）后每次
          事务推进假时钟，用于验证清理预算独立且有界（握手期不被拖慢）。
    """

    def __init__(
        self,
        *,
        clock=None,
        goal_write_torque_effect=None,
        goal_write_present_shift=None,
        cleanup_write_delay_s: float = 0.0,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self._clock = clock
        self.goal_write_torque_effect = goal_write_torque_effect
        self.goal_write_present_shift = dict(goal_write_present_shift or {})
        self.read_error_by_addr: dict[int, int] = {}
        self.corrupt_goal_read_ids: set[int] = set()
        self.cleanup_write_fail_ids: set[int] = set()
        self.cleanup_read_fail_ids: set[int] = set()
        self.explode_sync_write = False
        self._cleanup_write_delay_s = cleanup_write_delay_s
        self._disable_attempt: set[int] = set()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def set_torque(self, servo_id: int, value: int) -> None:
        self.registers.setdefault(servo_id, {})[STS_TORQUE_ENABLE] = value & 0xFF

    def set_goal(self, servo_id: int, raw: int) -> None:
        self.set_u16(servo_id, STS_GOAL_POSITION, raw)

    def _apply_goal_effect(self) -> None:
        if self.goal_write_torque_effect is not None:
            affected = (
                list(self.registers)
                if self.goal_write_torque_effect == "all"
                else list(self.goal_write_torque_effect)
            )
            for sid in affected:
                self.registers.setdefault(sid, {})[STS_TORQUE_ENABLE] = 1
        for sid, delta in self.goal_write_present_shift.items():
            regs = self.registers.setdefault(sid, {})
            cur = (regs.get(STS_PRESENT_POSITION, 0) | (regs.get(STS_PRESENT_POSITION + 1, 0) << 8))
            new = (cur + int(delta)) & 0xFFFF
            regs[STS_PRESENT_POSITION] = new & 0xFF
            regs[STS_PRESENT_POSITION + 1] = (new >> 8) & 0xFF

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        data = bytes(data)
        servo_id, length, inst = data[2], data[3], data[4]
        params = list(data[5 : 5 + length - 2])

        # 清理阶段才推进时钟（模拟只有卸力事务变慢），验证清理预算独立且有界。
        if self._clock is not None and self._disable_attempt and self._cleanup_write_delay_s:
            self._clock.t += self._cleanup_write_delay_s

        # 广播目标失败：串口层直接抛错（不写寄存器）。
        if inst == VENDOR_INST_SYNC_WRITE and self.explode_sync_write:
            raise OSError("模拟：目标广播 SYNC_WRITE TX 失败")

        # 读事务的三种注入：按地址回 ERROR、目标读回破坏、清理读回失败。
        if inst == VENDOR_INST_READ and params:
            addr = params[0]
            if addr in self.read_error_by_addr:
                saved = self.fail_error_bit
                self.fail_error_bit = self.read_error_by_addr[addr]
                try:
                    return super().write(data, timeout_s=timeout_s)
                finally:
                    self.fail_error_bit = saved
            if addr == STS_GOAL_POSITION and servo_id in self.corrupt_goal_read_ids:
                regs = self.registers.setdefault(servo_id, {})
                regs[STS_GOAL_POSITION] = (regs.get(STS_GOAL_POSITION, 0) ^ 0xFF) & 0xFF
                return super().write(data, timeout_s=timeout_s)
            if (
                addr == STS_TORQUE_ENABLE
                and servo_id in self.cleanup_read_fail_ids
                and servo_id in self._disable_attempt
            ):
                raise OSError("模拟：清理阶段 Torque_Enable 读回失败")

        # 清理失能写：记录尝试、可选失败。
        if inst == VENDOR_INST_WRITE and params:
            addr = params[0]
            value = params[1] if len(params) > 1 else 0
            if addr == STS_TORQUE_ENABLE and value == 0:
                self._disable_attempt.add(servo_id)
                if servo_id in self.cleanup_write_fail_ids:
                    raise OSError("模拟：清理阶段 Torque_Enable=0 写失败")

        n = super().write(data, timeout_s=timeout_s)
        if inst == VENDOR_INST_SYNC_WRITE and params and params[0] == STS_GOAL_POSITION:
            self._apply_goal_effect()
        return n


# ---------------------------------------------------------------------------
# 启动握手事务分析辅助（供 T28–T30 断言"序列 + 零写入 + 清理调用顺序"）
# ---------------------------------------------------------------------------

# 事务分类：(kind, addr_or_None, value_or_None, servo_id_or_None)。
# kind ∈ {'read','ping','write','sync_write','reset'}。写帧按寄存器地址区分，
# 力矩写再按值（0=卸力 / 1=使能）区分。


def startup_tx(bus) -> list[tuple]:
    """把假总线的 tx_log 解析成结构化事务序列（独立于被测实现的字节真值）。"""
    out: list[tuple] = []
    for frame in bus.tx_log:
        inst = frame[4]
        if inst == VENDOR_INST_RESET:
            out.append(("reset", None, None, None))
            continue
        if inst == VENDOR_INST_PING:
            out.append(("ping", None, None, frame[2]))
            continue
        params = list(frame[5 : 5 + frame[3] - 2])
        if inst == VENDOR_INST_READ:
            out.append(("read", params[0], None, frame[2]))
        elif inst == VENDOR_INST_WRITE:
            addr = params[0]
            value = params[1] if len(params) > 1 else None
            out.append(("write", addr, value, frame[2]))
        elif inst == VENDOR_INST_SYNC_WRITE:
            out.append(("sync_write", params[0], None, None))
        else:
            raise AssertionError(f"未知指令 0x{inst:02X}")
    return out


def assert_no_torque_or_goal_write(tx) -> None:
    """零 Goal_Position 广播、零 Torque_Enable 单播写（含卸载/使能）。"""
    for kind, addr, value, sid in tx:
        assert kind not in ("reset",), "启动路径禁止 RESET"
        if kind == "sync_write":
            assert addr != STS_GOAL_POSITION, "握手准入阶段不得广播 Goal_Position"
        if kind == "write":
            assert addr != STS_TORQUE_ENABLE, "握手准入阶段不得写 Torque_Enable"
            assert addr not in STS_EEPROM_ADDRS, f"禁止写 EEPROM/配置寄存器 {addr}"


def assert_no_eeprom_write(tx) -> None:
    for kind, addr, value, sid in tx:
        assert kind != "reset", "启动路径禁止 RESET"
        assert not (kind == "write" and addr in STS_EEPROM_ADDRS), f"禁止写 EEPROM {addr}"


class SleepInterruptClock(Clock):
    """在第 nth 次 sleep 时抛 KeyboardInterrupt，注入"上力后 Ctrl-C"。"""

    def __init__(self, interrupt_on_sleep: int, **kw) -> None:
        super().__init__(**kw)
        self._interrupt_on = interrupt_on_sleep
        self._sleep_calls = 0

    def sleep(self, dt: float) -> None:
        self._sleep_calls += 1
        if self._sleep_calls == self._interrupt_on:
            raise KeyboardInterrupt("模拟：使能/预置窗口内按下 Ctrl-C")
        super().sleep(dt)


# ---------------------------------------------------------------------------
# 1. 共享换算函数（标定指南第 12 节）
# ---------------------------------------------------------------------------


def test_关节角正反正变换互逆():
    """4.2 的第二、三行公式必须互为反变换。"""
    sign = np.array([1, -1, 1, -1, 1])
    offset = np.array([0.0, 12.5, -7.25, 0.0, 3.0])
    q_urdf = np.linspace(-120.0, 120.0, 5)
    back = bus_deg_to_urdf_deg(urdf_deg_to_bus_deg(q_urdf, sign, offset), sign, offset)
    assert np.allclose(back, q_urdf, atol=1e-12)


def test_sign只能是pm1():
    with pytest.raises(MotorLimitError):
        bus_deg_to_urdf_deg(10.0, 0, 0.0)
    with pytest.raises(MotorLimitError):
        urdf_deg_to_bus_deg(10.0, 2, 0.0)


def test_拒绝非有限输入():
    with pytest.raises(MotorLimitError):
        bus_deg_to_urdf_deg(float("nan"), 1, 0.0)
    with pytest.raises(MotorLimitError):
        urdf_deg_to_bus_deg(np.array([1.0, np.inf, 0, 0, 0]), np.ones(5), np.zeros(5))


def test_夹爪百分比往返与方向():
    """4.4：0%=标定闭合端、100%=标定张开端，open_raw 允许小于 closed_raw。"""
    for closed, opened in ((500, 3600), (3600, 500)):
        assert raw_to_gripper_pct(closed, closed, opened) == pytest.approx(0.0)
        assert raw_to_gripper_pct(opened, closed, opened) == pytest.approx(100.0)
        mid = (closed + opened) / 2
        assert raw_to_gripper_pct(mid, closed, opened) == pytest.approx(50.0)
        # 取三个都落在两端之间的读数：方向反过来时 "closed+37" 会跑到区间外，
        # 而区间外本来就必须抛错，不能拿来做往返测试。
        span = opened - closed
        for frac in (0.1, 0.37, 0.5, 0.93):
            raw = int(round(closed + frac * span))
            pct = raw_to_gripper_pct(raw, closed, opened)
            assert gripper_pct_to_raw(pct, closed, opened) == raw


def test_夹爪分母为零必须失败():
    with pytest.raises(MotorLimitError):
        raw_to_gripper_pct(100, 700, 700)
    with pytest.raises(MotorLimitError):
        gripper_pct_to_raw(50.0, 700, 700)


@pytest.mark.parametrize("bad_raw", [499, 3601, -5, 5000])
def test_夹爪越界观测抛错而不是裁剪(bad_raw):
    """4.4：超出范围的观测属于异常，不是先裁剪成 0/100 掩盖故障。"""
    with pytest.raises(MotorLimitError):
        raw_to_gripper_pct(bad_raw, 500, 3600)


@pytest.mark.parametrize("bad_pct", [-0.5, 100.5])
def test_夹爪命令越界抛错而不是裁剪(bad_pct):
    """4.4：命令侧越界同样抛错（0~100% 之外属于异常）。"""
    with pytest.raises(MotorLimitError):
        gripper_pct_to_raw(bad_pct, 500, 3600)


# ---------------------------------------------------------------------------
# 2. MotorMapping：raw ↔ 角度，量化误差界限
# ---------------------------------------------------------------------------


@pytest.fixture()
def mapping(params):
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    return MC.MotorMapping.from_params(params, calib)


def test_校准文件必须逐项含五个姿态关节与夹爪(mapping):
    assert {a.name for a in mapping.axes} == set(MOTOR_NAMES)


def test_deg与raw往返误差不超过一个编码步长(mapping, params):
    """4.2：整数取整策略必须固定并用往返测试证明误差不超过实现界限。"""
    for name in MOTOR_NAMES[:5]:
        step = mapping.axis(name).count_to_deg     # 是属性，不是方法
        worst = 0.0
        for deg in np.linspace(-110.0, 110.0, 801):
            raw = mapping.degrees_to_raw(name, float(deg))
            back = mapping.raw_to_degrees(name, raw)
            worst = max(worst, abs(back - float(deg)))
        assert worst <= step + 1e-9, f"{name} 往返误差 {worst:.5f} > 一个步长 {step:.5f}"


def test_raw到deg到raw在整数上不漂移(mapping):
    for name in MOTOR_NAMES[:5]:
        calib = mapping.axis(name)
        for raw in (calib.range_min, calib.mid_raw - 33, int(calib.mid_raw), calib.range_max - 1):
            deg = mapping.raw_to_degrees(name, int(raw))
            back = mapping.degrees_to_raw(name, deg)
            assert abs(back - int(raw)) <= 1, (name, raw, deg, back)


def test_官方校准端点可直接派生总线与URDF行程():
    bus = MC.calibrated_range_to_bus_limits_deg(718, 3424, 4096)
    assert bus == pytest.approx((-118.94505494505495, 118.94505494505495))
    # 坐标反向后要重新排成 [lower, upper]，零位则只做平移。
    urdf = MC.calibrated_range_to_urdf_limits_deg(718, 3424, 4096, -1, 5.0)
    assert urdf == pytest.approx((-113.94505494505495, 123.94505494505495))


@pytest.mark.parametrize("lo,hi,res", [(1, 1, 4096), (2, 1, 4096), (0, 4095, 1)])
def test_官方校准行程派生拒绝退化输入(lo, hi, res):
    with pytest.raises(MotorLimitError):
        MC.calibrated_range_to_bus_limits_deg(lo, hi, res)


def test_homing_offset不参与换算(params):
    """4.1 第 3 条：不要在本项目里再加一次官方 homing_offset。

    Homing_Offset 是写进舵机 EEPROM 的，设备上报的 Present_Position 已经归零；
    驱动再减一次会把同一个物理姿态读成两个不同角度。
    """
    base = raw_profile()
    calib_path = SIM_PROFILE.parent / base["model"]["motor_calibration_path"]
    import json

    original = json.loads(calib_path.read_text(encoding="utf-8"))
    a = MC.MotorMapping.from_params(
        _params_with(params, base),
        {k: {**v, "homing_offset": 0} for k, v in original.items()},
    )
    b = MC.MotorMapping.from_params(
        _params_with(params, base),
        {k: {**v, "homing_offset": -2000} for k, v in original.items()},
    )
    for name in MOTOR_NAMES[:5]:
        assert a.degrees_to_raw(name, 33.3) == b.degrees_to_raw(name, 33.3)
        assert a.raw_to_degrees(name, 2048) == b.raw_to_degrees(name, 2048)


def _params_with(params, raw_json):
    """构造一份只改了校准内容的临时 MotionParams（其余字段完全相同）。"""
    import copy

    p = copy.deepcopy(params)
    return p


def test_电流换算带零偏与比例(mapping, params):
    for i, name in enumerate(MOTOR_NAMES):
        zero = params.motor.current_zero_raw[i]
        scale = params.motor.current_ma_per_raw[i]
        assert mapping.raw_current_to_ma(name, int(zero)) == pytest.approx(0.0)
        assert mapping.raw_current_to_ma(name, int(zero) + 10) == pytest.approx(10 * scale)
        assert mapping.sign_bit(name, "Present_Current") == 15
        # 飞特 SMS/STS 的电流为 bit15 符号-幅值，不是 16 位补码。
        assert mapping.raw_current_to_ma(name, 0x8000 | 10) == pytest.approx(
            (-10 - zero) * scale
        )


def test_关节速度乘sign夹爪速度用百分比(mapping, params):
    """4.3：速度先按厂商比例转换，关节再乘 joints.sign；夹爪走 4.4 的 %/s。"""
    i = 1                                   # shoulder_lift，标定里常见反向
    sign = int(params.joints.sign[i])
    name = MOTOR_NAMES[i]
    v = mapping.raw_velocity_to_deg_s(name, 100)
    assert v == pytest.approx(sign * 100 * params.motor.velocity_deg_s_per_raw[i])
    # 4.4 的 %/s 公式里 raw_count_speed 是"由已换算的轴角速度换回编码计数/秒"，
    # 不能直接把寄存器读数当计数：先乘厂商比例得到 deg/s，再除以每计数角度。
    g = mapping.raw_velocity_to_deg_s("gripper", 100)
    span = params.motor.gripper_open_raw - params.motor.gripper_closed_raw
    axis_deg_s = 100 * params.motor.velocity_deg_s_per_raw[5]
    counts_s = axis_deg_s / mapping.axis("gripper").count_to_deg
    assert g == pytest.approx(counts_s * 100.0 / span)
    # 符号：开度增大为正，所以分母符号决定方向
    assert (g > 0) == (span > 0)


def test_有效限位复用同一份交集(mapping, params):
    lo, up = mapping.clamp_free_limits_deg()
    e_lo, e_up = effective_joint_limits(params.urdf_limits_deg, params.joints)
    assert np.allclose(lo, e_lo) and np.allclose(up, e_up)


# ---------------------------------------------------------------------------
# 3. 协议层：独立标准报文（Prompt §7 第 1 条）
# ---------------------------------------------------------------------------


def test_厂商标准读报文逐字节相等():
    """读 ID 1、地址 56、长度 2 → FF FF 01 04 02 38 02 BE（手册示例）。

    这条就是旧版"37 passed 仍漏错指令码"要补的独立断言：期望字节直接来自
    手册，不含任何被测常量；假总线对未知指令码抛 AssertionError，指令码错
    一字节都过不了。
    """
    bus = FakeBus()
    bus.set_u16(1, 56, 0x1234)
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    got = proto.read_registers(1, 56, 2, deadline_s=99.0)
    assert bus.tx_log[-1] == bytes([0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE])
    assert StsProtocol.join_u16(got) == 0x1234


def test_厂商标准写报文逐字节相等():
    """单播 WRITE（0x03）：写 ID 1 的 42 号寄存器两字节 0x0800。"""
    bus = FakeBus()
    bus.set_write_level(1, 0)                # 该台全应答 → 默认策略下等状态帧
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    proto.write_registers(1, 42, [0x00, 0x08], deadline_s=99.0)
    expected = bytes([0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC4])
    assert bus.tx_log[-1] == expected
    # 设备侧寄存器真的落了值（等应答模式 → 收到的是干净的 0 错误状态帧）
    assert bus.registers[1][42] == 0x00 and bus.registers[1][43] == 0x08


def test_同步写报文符合独立预期():
    """SYNC_WRITE（0x83，广播 0xFE）整帧逐字节比对独立构造的期望帧。"""
    bus = FakeBus()
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    ids = [1, 2, 3, 4, 5, 6]
    values = [2048, 100, 4095, 0, 7, 3600]
    params = [42, 2]
    for sid, val in zip(ids, values):
        params.extend([sid, *vendor_split_u16(val)])
    proto.sync_write_targets(ids, 42, 2, [vendor_split_u16(v) for v in values],
                             deadline_s=99.0)
    frame = bus.tx_log[-1]
    assert frame == vendor_packet(VENDOR_BROADCAST_ID, VENDOR_INST_SYNC_WRITE, params)
    # 广播不落总线确认，但从机侧寄存器必须已经更新
    assert bus.registers[6][42] == (3600 & 0xFF) and bus.registers[6][43] == (3600 >> 8)


def test_REG_WRITE与ACTION是两个独立指令码():
    """旧版把 0x04/0x05 混成一条 0x06；常量表必须与手册逐项一致。"""
    assert StsProtocol.INST_READ_DATA == 0x02
    assert StsProtocol.INST_WRITE_DATA == 0x03
    assert StsProtocol.INST_REG_WRITE == 0x04
    assert StsProtocol.INST_ACTION == 0x05
    assert StsProtocol.INST_SYNC_WRITE == 0x83
    assert StsProtocol.INST_PING == 0x01
    # 且驱动不再提供被删掉的误导封装（复位与两段式提交不进入本任务）。
    assert not hasattr(StsProtocol, "reg_write_action")
    assert not hasattr(StsProtocol, "reset_servo")


def test_校验和定义():
    body = [1, 4, 2, 56, 2]     # 手册读示例的 ID..参数
    assert StsProtocol.checksum(body) == (~sum(body)) & 0xFF


# ---------------------------------------------------------------------------
# 3b. 协议层：应答配置、迟到应答与异常分类（Prompt §5.2/§5.4）
# ---------------------------------------------------------------------------


def test_写应答策略按设备实际配置生效():
    """等级 1 的设备 WRITE 不应答：驱动不得硬等（否则耗光预算）。"""
    bus = FakeBus()
    bus.set_write_level(1, 1)
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    proto.set_write_response(1, False)
    proto.write_registers(1, 40, [1], deadline_s=99.0)   # 不应超时、不应抛
    assert bus.registers[1][40] == 1
    assert proto.write_expects_ack(1) is False
    assert proto.write_expects_ack(2) is True             # 未配置的单元默认等待


def test_全应答设备的写状态帧校验错误位():
    bus = FakeBus()
    bus.set_write_level(2, 0)
    bus.fail_error_bit = 0x01
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    proto.set_write_response(2, True)
    with pytest.raises(MotorStateError):
        proto.write_registers(2, 40, [1], deadline_s=99.0)


def test_迟到应答在请求前被冲刷不污染当前读():
    """上一笔超时留下的完整旧应答躺在接收缓冲里，必须先冲掉再收新的。

    旧帧是"舵机 2 读到 0xFFFF"；新种子是 777。若驱动没有先清缓冲，会把
    迟到的 0xFFFF 当成本次读回 → 断言失败。
    """
    bus = FakeBus()
    bus.set_u16(2, 56, 777)
    bus.inject_stale(vendor_status(2, [0xFF, 0xFF]))
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    got = proto.read_registers(2, 56, 2, deadline_s=99.0)
    assert StsProtocol.join_u16(got) == 777


def test_底层写异常归类为通信错误并保留原因():
    """OSError 不得裸漏（审阅指引 3.3 复现项）。"""
    bus = FakeBus()

    def boom(data, timeout_s=None):
        raise OSError("模拟：串口写失败")

    bus.write = boom
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    with pytest.raises(MotorCommunicationError) as exc:
        proto.read_registers(1, 56, 2, deadline_s=99.0)
    assert isinstance(exc.value.__cause__, OSError)


def test_短写按通信错误处理():
    class ShortWriteBus(FakeBus):
        def write(self, data, timeout_s=None):
            super().write(data, timeout_s=timeout_s)
            return len(data) - 1          # 故意少提交一个字节

    bus = ShortWriteBus()
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    with pytest.raises(MotorCommunicationError):
        proto.sync_write_targets([1], 42, 2, [[0, 8]], deadline_s=99.0)


def test_阻塞读收到递减的剩余预算():
    """验证实际传给底层的读超时参数，而不是只看最终时钟检查（Prompt §7.5）。

    该 transport 的 read_available() 永远为空，逼协议层走阻塞 read() 路径；
    每次阻塞 read 必须拿到一个严格正的 timeout_s，且沿调用序列单调不增
    （预算只被消耗、绝不重置）。
    """
    class OnlyBlockingRead:
        def __init__(self, reply: bytes, clock: Clock) -> None:
            self._reply = bytearray(reply)
            self._clock = clock
            self.read_timeouts: list[float | None] = []

        def write(self, data, timeout_s=None):
            return len(data)

        def read(self, n, timeout_s=None):
            self.read_timeouts.append(timeout_s)
            out = bytes(self._reply[:n])
            del self._reply[:n]
            return out

        def read_available(self):
            self._clock.t += 1e-4        # 模拟一次非阻塞轮询的固定开销
            return b""

        def flush_input(self):
            pass

        def close(self):
            pass

    clock = Clock(step_s=0.0)
    bus = OnlyBlockingRead(vendor_status(1, vendor_split_u16(777)), clock)
    proto = StsProtocol(bus, clock=clock, sleep=clock.sleep)
    got = proto.read_registers(1, 56, 2, deadline_s=0.05)
    assert StsProtocol.join_u16(got) == 777
    ts = [t for t in bus.read_timeouts if t is not None]
    assert ts, "协议层从未把预算传给底层读"
    assert all(0.0 < t <= 0.05 for t in ts)
    assert all(b <= a + 1e-12 for a, b in zip(ts, ts[1:])), "每次读都重新拿到完整预算"


@pytest.mark.parametrize("flag", ["fail_bad_checksum", "fail_wrong_id", "fail_short_reply"])
def test_坏包一律按通信错误处理(flag):
    bus = FakeBus()
    setattr(bus, flag, True)
    bus.set_u16(3, 56, 2050)
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    with pytest.raises(MotorCommunicationError):
        proto.read_registers(3, 56, 2, deadline_s=99.0)


def test_ERROR位非零按设备状态错误处理():
    bus = FakeBus()
    bus.fail_error_bit = 0x02
    bus.set_u16(2, 56, 100)
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    with pytest.raises(MotorStateError) as exc:
        proto.read_registers(2, 56, 2, deadline_s=99.0)
    assert "0x02" in str(exc.value) or "ERROR" in str(exc.value).upper()


def test_总时限不因从机数量成倍放大():
    """技术文档 4.2：总时限覆盖方法内部全部寄存器访问，不随电机数量翻倍。"""
    bus = FakeBus()
    bus.fail_no_response = True
    # 假时钟每调用一次前进 1ms：6 台轮询必然在同一个绝对 deadline 处到期。
    clock, nap = Clock(step_s=0.001).pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    started = clock.t
    with pytest.raises(MotorCommunicationError):
        proto.bulk_read([1, 2, 3, 4, 5, 6], 56, 2, deadline_s=started + 0.020)
    # 到点就退出：允许一个很小的分片粒度余量，绝不允许 6×20ms。
    assert clock.t <= started + 0.020 + 0.005, f"实际耗时 {clock.t - started:.4f}s"


def test_运行期不重试():
    bus = FakeBus()
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    assert proto.num_retry == 0
    bus.set_u16(5, 58, 12)
    proto.read_registers(5, 58, 2, deadline_s=99.0)
    assert bus.read_request_count == 1, "同一次读出现了重试"


def test_应答延迟送达时阻塞路径正常工作():
    """应答不凑在 write() 同一瞬间出现时，分帧/pushback 逻辑仍须解对。"""
    bus = FakeBus()
    bus.deliver_after_polls = 6     # 轮询 4 次为空后协议转入阻塞 read()，第 7 次才送达
    bus.set_u16(3, 56, 0xBEEF)
    clock, nap = Clock(step_s=1e-4).pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    got = proto.read_registers(3, 56, 2, deadline_s=99.0)
    assert StsProtocol.join_u16(got) == 0xBEEF
    assert any(t is not None for t in bus.read_timeouts), "阻塞读没有带超时参数"


# ---------------------------------------------------------------------------
# 4. Sts3215MotorController：三个协议方法
# ---------------------------------------------------------------------------


def _controller(params, bus=None, **kwargs):
    bus = bus or FakeBus()
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    clock = kwargs.pop("clock", Clock())
    ctl = Sts3215MotorController(
        params, mapping, transport=bus, clock=clock,
        sleep=kwargs.pop("sleep", clock.sleep),
        monotonic_ns=lambda: int(clock.t * 1e9),
        initialize=False, **kwargs,
    )
    ctl._clock_obj = clock
    return ctl, bus


def _seed_feedback(bus, params, mapping):
    """给 6 个从机写入"当前姿态"的原始位置读数（连续块读回也覆盖这些地址）。"""
    q = np.array(params.workspace.home_joints_deg, float)
    for i, name in enumerate(MOTOR_NAMES[:5]):
        bus.set_u16(i + 1, 56, int(mapping.degrees_to_raw(name, float(q[i]))))
        bus.set_u16(i + 1, 58, 0)
        bus.set_u16(i + 1, 69, 0)
    bus.set_u16(6, 56, params.motor.gripper_closed_raw
                + (params.motor.gripper_open_raw - params.motor.gripper_closed_raw) // 2)
    bus.set_u16(6, 58, 0)
    bus.set_u16(6, 69, 0)


def _startup_controller(params, bus, *, clock=None, sleep=None, initialize=True):
    """构造一个走 §4.1 默认握手路径的驱动：seed 反馈（当前在 home）+ 注入时钟。

    返回 (ctl, bus, mapping, clock)。默认 ``initialize=True`` 让构造即握手；
    需要注入异常/在握手前改总线状态时传 ``initialize=False`` 再手动 initialize()。
    """
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    _seed_feedback(bus, params, mapping)
    clock = clock if clock is not None else Clock()
    ctl = Sts3215MotorController(
        params, mapping, transport=bus, clock=clock,
        sleep=sleep if sleep is not None else clock.sleep,
        monotonic_ns=lambda: int(clock.t * 1e9),
        initialize=initialize,
    )
    ctl._clock_obj = clock
    return ctl, bus, mapping, clock


def test_未初始化时三个方法都拒绝(params):
    """文档 4.2：设备未就绪或报告硬件故障抛 MotorStateError。

    先把从机寄存器灌成"停在待机位、夹爪居中"，否则读回来的 0 会先撞上夹爪端点
    越界（那是 MotorLimitError），测不到"未使能力矩"这条设备状态分支。
    """
    bus = FakeBus()
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    _seed_feedback(bus, params, mapping)
    clock, nap = Clock().pair()
    ctl = Sts3215MotorController(params, mapping, transport=bus, clock=clock,
                                 sleep=nap, initialize=False)
    # send_action 会真的驱动舵机，力矩没开时静默"成功"是最坏结果，必须拒绝。
    with pytest.raises(MotorStateError):
        ctl.send_action(np.array(params.workspace.home_joints_deg, float), 50.0)
    # 读反馈不需要力矩：故障诊断时恰恰要在力矩关闭的状态下读数。
    assert ctl.get_feedback().angles_deg.shape == (5,)
    # 关闭连接之后三个方法一律拒绝
    ctl.close()
    for call in (lambda: ctl.send_action(np.array(params.workspace.home_joints_deg, float), 50.0),
                 ctl.get_feedback, ctl.hold_current):
        with pytest.raises(MotorStateError):
            call()


def test_初始化链路_读应答配置核对身份能力矩_不碰EEPROM(params):
    """默认握手成功路径（§4.1）：准入卸力→预置→隐式核对→位移检查→READY。

    修订说明（P1-06）：旧断言"6+6+6+12+6=36 帧、写帧只有 Torque_Enable"是**旧
    initialize**（读应答配置→身份→直接 configure_torque(True)）的形态。§4.1 现在
    要求先核对六轴全 0、以当前反馈预置 Goal_Position、再按 CAL-052 处理隐式上力，
    因此本用例覆盖面**提升**：新增"预置广播 1 帧 SYNC_WRITE(Goal)"与"仅对力矩=0 的
    轴显式使能"，同时保留原有的两条底线——EEPROM/行程限位/Response_Status_Level 一次
    都不写、无 RESET、应答策略确实来自寄存器读回。
    """
    bus = StartupBus()                   # 默认全应答等级 0、无隐式上力：广播后仍需显式使能
    ctl, bus, mapping, clock = _startup_controller(params, bus)
    tx = startup_tx(bus)

    # 底线一：没有任何 EEPROM/行程限位/应答配置写，也没有 RESET。
    assert_no_eeprom_write(tx)
    # 底线二：Goal_Position 广播恰好 1 帧（一次 sync_write 六轴），力矩写只允许 =1。
    goal_syncs = [t for t in tx if t[0] == "sync_write" and t[1] == STS_GOAL_POSITION]
    torque_writes = [t for t in tx if t[0] == "write" and t[1] == STS_TORQUE_ENABLE]
    assert len(goal_syncs) == 1, "目标预置必须是恰好一次六轴广播"
    assert torque_writes and all(v == 1 for _, _, v, _ in torque_writes), "只允许使能写，不得写卸力"
    # 应答策略来自寄存器读回：默认等级 0 → 写要等应答。
    assert all(ctl._protocol.write_expects_ack(sid) for sid in range(1, 7))
    # 六轴最终都打开；内部状态兼容 _require_actuating_ready（READY）。
    addr, _ = mapping.common_register("Torque_Enable")
    assert all(bus.registers[sid][addr] == 1 for sid in range(1, 7))
    report = ctl.startup_report
    assert report["result"] == "ready"
    assert report["failed_phase"] is None
    assert report["torque_initial"] == [0] * 6
    assert ctl._torque_enabled is True and ctl._startup_ready is True


def test_初始化按等级1配置后力矩写不等应答(params):
    """等级 1（出厂常见"只回读"）下若硬等应答，握手会超时；必须按配置跳过。

    修订说明（P1-06）：默认路径现在在广播目标后需要对力矩=0 的轴**逐轴显式使能**，
    本用例即验证这些力矩写在等级 1 设备上不等待应答、从而整条握手按时完成并置 READY。
    """
    bus = StartupBus()
    for sid in range(1, 7):
        bus.set_write_level(sid, 1)
    ctl, bus, mapping, clock = _startup_controller(params, bus)
    assert all(not ctl._protocol.write_expects_ack(sid) for sid in range(1, 7))
    addr, _ = mapping.common_register("Torque_Enable")
    assert all(bus.registers[sid][addr] == 1 for sid in range(1, 7))
    assert ctl.startup_report["result"] == "ready"
    # 广播后读回全 0 → 显式使能全部六轴（无隐式上力时），逐轴写、每轴一帧。
    assert ctl.startup_report["torque_enable_source"] == "explicit_enable_zero_axes"
    assert len(ctl.startup_report["explicitly_enabled_axes"]) == 6


def test_只读路径_enable_torque_False_全程零WRITE(params):
    """enable_torque=False 是标定工具在用的只读路径：锁定"该路径零 WRITE"。

    既有脚本（validate_home_waypoint_motion / calibrate_joint_velocity /
    calibrate_joint_acceleration）都用 ``initialize(verify_identity=True,
    enable_torque=False)`` 只读握手，随后自行预置。这条断言保证新流程不会给
    它们追加任何 Goal_Position/Torque_Enable 写入。
    """
    bus = StartupBus()
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    ctl.initialize(verify_identity=True, enable_torque=False)
    tx = startup_tx(bus)
    assert_no_torque_or_goal_write(tx)
    # 只读路径不预置、不核对位移、不置 READY。
    report = ctl.startup_report
    assert report["result"] == "read_only"
    assert ctl._torque_enabled is False and ctl._startup_ready is False
    # 身份核对仍在只读路径里执行：应能读到型号/固件（有 PING）。
    assert any(t[0] == "ping" for t in tx)


# ---------------------------------------------------------------------------
# 4b. P1 §4.1 驱动启动握手（P1-T28 / T29 / T30）
# ---------------------------------------------------------------------------


def _startup_raises(params, bus, *, clock=None, sleep=None):
    """构造 initialize=False 的驱动并让握手跑起来，返回 (exc, ctl)。"""
    ctl, bus, mapping, clock = _startup_controller(
        params, bus, initialize=False, clock=clock, sleep=sleep
    )
    with pytest.raises(MC.MotorStartupError) as ei:
        ctl.initialize()
    return ei.value, ctl


# --- P1-T28：任一轴已上力 / (a) 阶段读失败 → MOTOR_INIT_FAILED，零写入 ---


def test_T28_任一轴已上力_拒绝启动且零目标力矩写入(params):
    # 轴 3（elbow_flex）启动前就已 Torque_Enable=1，D17 要求不得由启动自动卸力。
    bus = StartupBus()
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    bus.set_torque(3, 1)
    with pytest.raises(MC.MotorStartupError) as ei:
        ctl.initialize()
    assert_no_torque_or_goal_write(startup_tx(bus))     # 一条 Goal/Torque 写都没有
    report = ei.value.report
    assert report["result"] == "error"
    assert "卸力" in report["failed_phase"]
    assert report["torque_initial"][2] == 1


def test_T28_构造即握手失败也抛MotorStartupError供main捕获(params):
    """生产 main 走 ``Sts3215MotorController(params)``（构造即 initialize=True）；
    已上力时构造本身必须抛出可取的 MotorStartupError，且零目标/力矩写入。"""
    bus = StartupBus()
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    _seed_feedback(bus, params, mapping)
    bus.set_torque(6, 1)                                 # 夹爪轴启动前已上力
    clock = Clock()
    with pytest.raises(MC.MotorStartupError) as ei:
        Sts3215MotorController(
            params, mapping, transport=bus, clock=clock, sleep=clock.sleep,
            monotonic_ns=lambda: int(clock.t * 1e9), initialize=True,
        )
    assert_no_torque_or_goal_write(startup_tx(bus))
    assert ei.value.report["failed_phase"]
    assert bus.closed is True                            # 构造失败也不泄漏传输句柄


def test_T28_准入阶段力矩读失败_拒绝启动且零写入(params):
    # (a) 逐轴读 Torque_Enable 时设备报硬件故障（ERROR 位）→ 立即失败，零写入。
    bus = StartupBus()
    bus.read_error_by_addr[STS_TORQUE_ENABLE] = 0x02
    exc, ctl = _startup_raises(params, bus)
    assert_no_torque_or_goal_write(startup_tx(bus))
    assert exc.report["result"] == "error"
    assert "卸力" in exc.report["failed_phase"]
    assert ctl._connected is False                      # 失败关闭传输


# --- P1-T29：旧目标不一致 / 隐式全上力 / 部分上力 / 成功不额外写 / 全0+位移OK ---


def test_T29_旧目标不一致_只登记不先写力矩_随后以当前反馈预置(params):
    # 把轴 1 的旧 Goal_Position 灌成一个与当前反馈编码目标不同的值。
    bus = StartupBus()
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    home0_raw = int(mapping.degrees_to_raw("shoulder_pan", 0.0))
    bus.set_goal(1, (home0_raw + 700) & 0xFFFF)         # 制造旧目标≠当前反馈
    ctl.initialize()
    report = ctl.startup_report
    mismatch = {m["axis"] for m in report["old_target_mismatch"]}
    assert "shoulder_pan" in mismatch
    # 预置以"当前反馈"为准：广播后读回必须等于编码当前反馈的目标，而不是旧的 +700 偏移。
    assert report["goal_readback_raw"] == report["seeded_goal_positions_raw"]
    # 关键：绝不先写 Torque_Enable=1 —— 第一条写事务必须是 Goal 广播。
    tx = startup_tx(bus)
    first_write_kind = next(
        (t[0] for t in tx if t[0] in ("write", "sync_write")), None
    )
    assert first_write_kind == "sync_write"
    assert report["result"] == "ready"


def test_T29_广播后隐式全上力_读回全1不重复写力矩(params):
    # CAL-052：广播 Goal_Position 使六轴隐式上力 → 读回全 1 → 不得再逐轴写 Torque_Enable=1。
    bus = StartupBus(goal_write_torque_effect="all")
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    ctl.initialize()
    report = ctl.startup_report
    assert report["torque_enable_source"] == "goal_broadcast_implicit_all_verified"
    assert report["explicitly_enabled_axes"] == []
    assert report["torque_final_readback"] == [1] * 6
    assert report["result"] == "ready"
    tx = startup_tx(bus)
    # 只有一帧 Goal 广播；没有任何 Torque_Enable 单播写（既不使能也不卸力）。
    assert sum(1 for t in tx if t[0] == "sync_write" and t[1] == STS_GOAL_POSITION) == 1
    assert [t for t in tx if t[0] == "write" and t[1] == STS_TORQUE_ENABLE] == []


def test_T29_部分上力_只对为0轴显式使能再读回(params):
    # 广播后只有轴 1/3/5 隐式上力，轴 2/4/6 仍为 0 → 只对这 3 个轴显式使能。
    bus = StartupBus(goal_write_torque_effect={1, 3, 5})
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    ctl.initialize()
    report = ctl.startup_report
    assert report["torque_after_broadcast"] == [1, 0, 1, 0, 1, 0]
    assert report["explicitly_enabled_axes"] == ["shoulder_lift", "wrist_flex", "gripper"]
    assert report["torque_final_readback"] == [1] * 6
    assert report["result"] == "ready"
    tx = startup_tx(bus)
    enable_writes = [t for t in tx if t[0] == "write" and t[1] == STS_TORQUE_ENABLE]
    assert all(v == 1 for _, _, v, _ in enable_writes)
    assert {sid for _, _, _, sid in enable_writes} == {2, 4, 6}   # 只使能为 0 的轴


def test_T29_全0且位移OK成功路径不追加多余初始化写入(params):
    # 无隐式上力 → 广播后需显式使能全部六轴；成功路径写入集合必须精确，不夹带多余写。
    bus = StartupBus()
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    ctl.initialize()
    report = ctl.startup_report
    assert report["result"] == "ready"
    assert ctl._startup_ready is True and ctl._torque_enabled is True
    assert report["torque_initial"] == [0] * 6
    assert report["position_change_deg"] == pytest.approx([0.0] * 5)
    assert report["gripper_change_pct"] == pytest.approx(0.0)
    # 精确写事务：1 帧 Goal 广播 + 6 帧 Torque_Enable=1；除此之外零 WRITE、零卸力、无 EEPROM/RESET。
    tx = startup_tx(bus)
    from collections import Counter
    writes = Counter(
        (kind, addr, val) for kind, addr, val, _ in tx if kind in ("write", "sync_write")
    )
    assert writes == Counter({
        ("sync_write", STS_GOAL_POSITION, None): 1,
        ("write", STS_TORQUE_ENABLE, 1): 6,
    })
    assert_no_eeprom_write(tx)


def test_T29_预置前速度未静止_拒绝且零写入(params):
    """(b) 要求五关节速度逐轴 ≤ motion.settle_velocity_tol_deg_s；未静止时零写入退出。"""
    bus = StartupBus()
    ctl, bus, mapping, clock = _startup_controller(params, bus, initialize=False)
    tol = float(params.motion.settle_velocity_tol_deg_s[0])
    # 找一个换算后 deg/s 明确超过容差的原始速度值灌进轴 1 的 Present_Velocity。
    probe = 1
    while abs(mapping.raw_velocity_to_deg_s(0, probe)) <= tol and probe < 0x7FFF:
        probe = probe * 2 + 1
    assert abs(mapping.raw_velocity_to_deg_s(0, probe)) > tol, "假速度未越容差，测试构造失败"
    bus.set_u16(1, 58, probe)
    with pytest.raises(MC.MotorStartupError) as ei:
        ctl.initialize()
    assert_no_torque_or_goal_write(startup_tx(bus))       # 预置阶段就失败 → 零写入
    assert ei.value.report["failed_phase"] == "反馈预置"
    assert ei.value.report["result"] == "error"
    assert ei.value.report["cleanup"]["attempted"] is False




def _cleanup_states(report):
    return {ax["axis"]: ax["state"] for ax in report["cleanup"]["axes"]}


def test_T30_目标广播写失败_进入有界清理并卸力关闭(params):
    bus = StartupBus()
    bus.explode_sync_write = True                        # 广播帧 TX 失败
    exc, ctl = _startup_raises(params, bus)
    report = exc.report
    assert report["result"] == "error"
    assert report["cleanup"]["attempted"] is True        # 从首次可能上力的写入起
    states = _cleanup_states(report)
    assert set(states) == set(MOTOR_NAMES)
    # 广播从未落地、也没上力 → 清理逐轴失能写后读回 0 → 全 off。
    assert all(v == "off" for v in states.values())
    tx = startup_tx(bus)
    disable = [t for t in tx if t[0] == "write" and t[1] == STS_TORQUE_ENABLE and t[2] == 0]
    assert len(disable) == 6
    # 清理不得写 Goal_Position、不得 hold/reset/home。
    assert [t for t in tx if t[0] == "sync_write"] == []
    assert not any(t[0] == "reset" for t in tx)
    assert ctl._connected is False and bus.closed is True


def test_T30_目标读回不匹配_清理逐轴卸力(params):
    bus = StartupBus()
    bus.corrupt_goal_read_ids = {4}                      # 轴 4 Goal 读回被破坏
    exc, ctl = _startup_raises(params, bus)
    report = exc.report
    assert report["goal_readback_raw"] != report["seeded_goal_positions_raw"]
    assert report["cleanup"]["attempted"] is True
    assert all(v == "off" for v in _cleanup_states(report).values())
    assert ctl._connected is False and bus.closed is True


def test_T30_位移超限_上力后失败仍完整清理(params):
    # 广播时把轴 2 的 Present_Position 平移约 35°（> start_position_tol_deg=3°）。
    bus = StartupBus(goal_write_present_shift={2: 400})
    exc, ctl = _startup_raises(params, bus)
    report = exc.report
    assert report["failed_phase"] == "位移核对"
    assert report["position_change_deg"][1] > 3.0
    # 无隐式上力 → 曾显式使能六轴 → (d) 失败 → 清理必须把这六轴全部卸回。
    assert report["cleanup"]["attempted"] is True
    assert all(v == "off" for v in _cleanup_states(report).values())
    tx = startup_tx(bus)
    disable = [t for t in tx if t[0] == "write" and t[1] == STS_TORQUE_ENABLE and t[2] == 0]
    assert len(disable) == 6
    assert ctl._connected is False and bus.closed is True


def test_T30_CtrlC在上力窗口_有界清理后抛MotorStartupError(params):
    # 在 (d) 第一次等待（已广播+已上力）时注入 KeyboardInterrupt。
    clock = SleepInterruptClock(interrupt_on_sleep=1)
    bus = StartupBus(clock=clock)
    exc, ctl = _startup_raises(params, bus, clock=clock)
    report = exc.report
    assert report["result"] == "interrupted"
    assert report["cleanup"]["attempted"] is True
    assert all(v == "off" for v in _cleanup_states(report).values())
    assert ctl._connected is False and bus.closed is True
    # MotorStartupError 仍是 MotorStateError（既有 MOTOR_INIT_FAILED 归类接得住）。
    assert isinstance(exc, MC.MotorStateError)


def test_T30_清理某轴读失败登记unknown_不得当作off(params):
    # 位移失败触发清理；轴 5 的清理读回失败 → 只能记 unknown，其余 off。
    bus = StartupBus(goal_write_present_shift={2: 400})
    bus.cleanup_read_fail_ids = {5}
    exc, ctl = _startup_raises(params, bus)
    states = _cleanup_states(exc.report)
    assert states["wrist_roll"] == "unknown"
    # unknown ≠ off：只有轴 5 无法确认，其余成功读回 0。
    off_axes = {a for a, s in states.items() if s == "off"}
    assert off_axes == set(MOTOR_NAMES) - {"wrist_roll"}
    assert "unknown" in states.values()
    assert exc.report["cleanup"]["axes"][4]["servo_id"] == 5


def test_T30_清理某轴失能写失败_仍继续其余轴并如实记录(params):
    # 轴 2 失能写失败（寄存器仍停在 1）→ 读回 on；其余 off；循环不中断，六轴都有记录。
    bus = StartupBus(goal_write_present_shift={3: 400})
    bus.cleanup_write_fail_ids = {2}
    exc, ctl = _startup_raises(params, bus)
    report = exc.report
    states = _cleanup_states(report)
    assert len(states) == 6                              # 失败仍继续，逐轴都有条目
    assert states["shoulder_lift"] == "on"               # 失能写没落地 → 读回 1
    off_axes = {a for a, s in states.items() if s == "off"}
    assert off_axes == set(MOTOR_NAMES) - {"shoulder_lift"}
    axis2 = [ax for ax in report["cleanup"]["axes"] if ax["servo_id"] == 2][0]
    assert axis2["disable_error"]                        # 记录了该轴失能写失败


def test_T30_清理预算独立有界_慢总线尾部轴记unknown(params):
    # 清理阶段每事务推进假时钟 1s，超过 STARTUP_CLEANUP_TIMEOUT_S 后剩余轴不再尝试 → unknown。
    clock = Clock(step_s=0.0)
    bus = StartupBus(clock=clock, goal_write_present_shift={2: 400}, cleanup_write_delay_s=1.0)
    exc, ctl = _startup_raises(params, bus, clock=clock)
    report = exc.report
    assert report["cleanup"]["attempted"] is True
    assert report["cleanup"]["deadline_exceeded"] is True
    states = [ax["state"] for ax in report["cleanup"]["axes"]]
    assert len(states) == 6
    assert "off" in states and "unknown" in states       # 有界：早期卸成功、尾部超预算 unknown
    assert states[0] == "off"                            # 第一轴一定在预算内先卸力
    assert ctl._connected is False and bus.closed is True


@pytest.mark.parametrize("bad_index", [0, 4])
def test_任意关节NaN或Inf整条拒绝且零运动写入(params, bad_index):
    """Prompt §7.2：最后一个姿态关节出 NaN/Inf 时，运动写入次数必须为 0。"""
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    before_w, before_s = bus.write_count, bus.sync_write_count
    home = np.array(params.workspace.home_joints_deg, float)
    for bad in (np.nan, np.inf, -np.inf):
        q = home.copy()
        q[bad_index] = bad
        with pytest.raises(MotorLimitError):
            ctl.send_action(q, 50.0)
    assert bus.write_count == before_w
    assert bus.sync_write_count == before_s


def test_夹爪NaN_Inf_越界整条拒绝且零运动写入(params):
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    before_w, before_s = bus.write_count, bus.sync_write_count
    home = np.array(params.workspace.home_joints_deg, float)
    for bad in (float("nan"), float("inf"), -5.0, 140.0):
        with pytest.raises(MotorLimitError):
            ctl.send_action(home, bad)
    assert bus.write_count == before_w
    assert bus.sync_write_count == before_s


def test_越限指令在写出任何字节之前整条拒绝(params):
    """文档 4.2：整条参数/限位校验在写入前完成，不静默裁剪。"""
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    before = bus.write_count
    bad = np.array(params.workspace.home_joints_deg, float)
    bad[0] = float(ctl.mapping.clamp_free_limits_deg()[1][0] + 5.0)
    with pytest.raises(MotorLimitError):
        ctl.send_action(bad, 50.0)
    assert bus.write_count == before, "校验失败前已经向总线写过字节"
    # 夹爪开度越界同样整条拒绝
    with pytest.raises(MotorLimitError):
        ctl.send_action(np.array(params.workspace.home_joints_deg, float), 140.0)
    assert bus.write_count == before
    # shape 与 NaN 也拒
    with pytest.raises(MotorLimitError):
        ctl.send_action(np.zeros(4), 50.0)
    with pytest.raises(MotorLimitError):
        ctl.send_action(np.array([0, np.nan, 0, 0, 0], float), 50.0)


def test_send_action恰好一帧六轴同步写(params):
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    q = np.array(params.workspace.home_joints_deg, float)
    before = bus.write_count
    ctl.send_action(q, 62.0)
    assert bus.sync_write_count == 1, "六个目标必须装进恰好一条 SYNC_WRITE 帧"
    assert bus.write_count == before + 1
    assert bus.read_request_count == 0, "send_action 不得夹带读事务（不等到位）"
    written = StsProtocol.join_u16(bytes([bus.registers[1][42], bus.registers[1][43]]))
    assert written == ctl.mapping.degrees_to_raw("shoulder_pan", float(q[0]))


def test_get_feedback六台各一次块读并正确解码(params):
    """文档 4.2 + 审阅指引 3.6：每台一次 15 字节连续块，六台共 6 次事务。

    Present_Load(60..61) 被灌入脏数据：电流必须取自 Present_Current(69)，
    绝不能把 Load 当 Current（标定指南 4.3）。
    """
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    bus.set_u16(2, 58, 20)
    bus.set_u16(2, 69, 40)
    bus.set_u16(2, 60, 3000)         # Present_Load：读得到但绝不使用
    reads_before = bus.read_request_count
    fb = ctl.get_feedback()
    assert bus.read_request_count - reads_before == 6, "块读应恰好每台一次"
    assert fb.angles_deg.shape == (5,) and fb.angles_deg.dtype == np.float64
    assert fb.speeds_deg_s.shape == (5,) and fb.currents_ma.shape == (5,)
    assert fb.angles_deg[0] == pytest.approx(params.workspace.home_joints_deg[0], abs=0.1)
    assert fb.speeds_deg_s[1] == pytest.approx(
        mapping.raw_velocity_to_deg_s("shoulder_lift", 20))
    assert fb.currents_ma[1] == pytest.approx(mapping.raw_current_to_ma("shoulder_lift", 40))
    assert fb.currents_ma[1] != pytest.approx(
        (3000 - params.motor.current_zero_raw[1]) * params.motor.current_ma_per_raw[1]), \
        "电流读成了 Present_Load"
    assert 0.0 <= fb.gripper_pct <= 100.0
    assert fb.sample_start_ns <= fb.sample_end_ns
    seq = fb.sequence
    assert ctl.get_feedback().sequence > seq, "sequence 必须每次完整成功读回后递增"


def test_第六台失败整次失败不返回部分数据不递增(params):
    """Prompt §7.4：任一台失败 → 整次失败；无缓存、无部分数据、无重试。"""
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    _seed_feedback(bus, params, mapping)
    good = ctl.get_feedback()
    bus.fail_ids = {6}
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()
    assert ctl._sequence == good.sequence, "失败读回不得递增 sequence"
    # 每台一次、失败即止：6 次 READ 请求（第 6 台等预算到期），没有第二轮重试
    assert bus.read_request_count == good.sequence * 6 + 6


def test_读回跨度过大视为无效采样且不返回缓存(params):
    """文档 4.2：失败时不返回缓存数据。

    把读预算临时放大（内存副本，不改共享 fixture），让 6 次事务都能完成，
    但假时钟推进使**跨度**超过 max_feedback_span_s → 必须按无效采样抛错；
    再把时钟步进归零，确认成功读回能继续（没有卡死的脏缓存）。
    """
    from dataclasses import replace as dc_replace

    p2 = dc_replace(params, timing=dc_replace(
        params.timing, read_timeout_s=0.5, max_feedback_span_s=params.timing.max_feedback_span_s))
    ctl, bus = _controller(p2)
    calib = load_motor_calibration(SIM_PROFILE.parent / p2.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    good = ctl.get_feedback()
    # 每次 deadline 检查前进 ~2ms：一次完整块读回约 30+ 次检查 → 跨度 >30ms
    ctl._clock_obj.step_s = 0.002
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()
    ctl._clock_obj.step_s = 0.0
    assert ctl.get_feedback().sequence > good.sequence


def test_慢写不能骗过整次写预算(params):
    """Prompt §7.5：100ms 的写不能声称满足 15ms 预算；且底层真收到了剩余预算参数。"""
    clock = Clock()
    bus = SlowWriteBus(clock, delay_s=0.1)
    ctl, _ = _controller(params, bus=bus, clock=clock)
    ctl._torque_enabled = True
    assert params.timing.write_timeout_s <= 0.015 + 1e-12   # SIM 配置：15ms
    with pytest.raises(MotorCommunicationError):
        ctl.send_action(np.array(params.workspace.home_joints_deg, float), 50.0)
    # 阻塞调用返回后复查：帧虽写完，方法仍报超时，不返回"成功"
    assert bus.write_timeouts and 0.0 < bus.write_timeouts[-1] <= params.timing.write_timeout_s


def test_慢读跨六台共享同一deadline(params):
    """多设备共享绝对预算：每次应答 4ms × 6 台 = 24ms > 20ms 读预算 → 必失败。

    若实现按"每台一个超时"组织，6 台各自 20ms 会全部成功返回——那正是
    本条要拒绝的行为。
    """
    clock = Clock()
    bus = SlowReadBus(clock, delay_s=0.004)
    ctl, _ = _controller(params, bus=bus, clock=clock)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    _seed_feedback(bus, params, mapping)
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()
    assert bus.read_request_count <= 6, "不得在下一台上重新获得完整预算"


def test_通信失败向上抛出不吞(params):
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    bus.fail_no_response = True
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()


def test_hold_current恰好一读一写且目标等于反馈(params):
    """文档 4.2：有界地读一次并写一次相同五关节角与夹爪开度，返回所用反馈。"""
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    reads_before = bus.read_request_count
    threads_before = threading.active_count()
    fb = ctl.hold_current()
    assert bus.read_request_count - reads_before == 6, "保持前只允许一次完整块读回"
    assert bus.sync_write_count == 1, "保持目标必须是一次同步写，不循环、不续写"
    written = StsProtocol.join_u16(bytes([bus.registers[1][42], bus.registers[1][43]]))
    assert written == mapping.degrees_to_raw("shoulder_pan", float(fb.angles_deg[0]))
    assert threading.active_count() <= threads_before, "hold_current 不得留下后台活动"


def test_hold_current读失败绝不发送猜测的保持目标(params):
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    bus.fail_no_response = True
    with pytest.raises(MotorCommunicationError):
        ctl.hold_current()
    assert bus.sync_write_count == 0, "读失败后不得拿未知位置当保持目标"


def test_hold_current写失败不伪报保持成功(params):
    ctl, bus = _controller(params, bus=ExplodingSyncWriteBus())
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    with pytest.raises(MotorCommunicationError):
        ctl.hold_current()


def test_型号或协议不符拒绝构造(params):
    """文档 3.1/2.1：motor.ids 与型号要逐项核对。"""
    bus = FakeBus(model_number=999)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    clock, nap = Clock().pair()
    with pytest.raises(Exception):
        Sts3215MotorController(params, mapping, transport=bus, clock=clock,
                               sleep=nap, initialize=True)


def test_没有pyserial时仍可import并使用换算层():
    """pyserial 只在构造真机串口对象时才需要。

    这条保证在没有串口的开发机/CI 上，驱动模块本身与全部换算逻辑依然可用。
    """
    import builtins
    import sys

    saved = sys.modules.pop("serial", None)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("模拟：本机没有 pyserial")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        assert raw_to_gripper_pct(2050, 500, 3600) == pytest.approx(50.0)
        with pytest.raises(MotorStateError) as exc:
            MC.SerialTransport("/dev/nonexistent", 1000000)
        assert "pyserial" in str(exc.value) or "serial" in str(exc.value)
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["serial"] = saved


def test_校准文件缺关节时报错(params):
    import json

    path = SIM_PROFILE.parent / params.model.motor_calibration_path
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["wrist_roll"]
    with pytest.raises((MotorLimitError, ValueError, KeyError)) as exc:
        load_motor_calibration(_write_tmp(params.profile_dir, data))
    assert exc.type is not None


def _write_tmp(base_dir, payload):
    import json
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    p = d / "motor_calibration.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 5. CalibrationReader：只读采集契约（标定指南 3.3/12 节，Prompt §7.8）
# ---------------------------------------------------------------------------


def test_CalibrationReader快照键与只读行为(params):
    """原始快照键、时间戳、未完成标定时 gripper_pct=None 的只读采集行为。

    本条只验证驱动侧契约（键集合 + 全程零运动写）；draft 配置加载与
    capture CLI 的集成回归由 tests/test_calibration.py 覆盖。
    """
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    profile = SimpleNamespace(
        motor=params.motor,
        joints=None,                        # joints 未拟合 → q_urdf_deg 必须为 None
        motor_calibration=calib,
        urdf_limits_deg=params.urdf_limits_deg,
    )
    bus = FakeBus()
    bus.set_u16(6, 56, 2050)   # 夹爪落在标定端点 [500, 3600] 内；其余通道默认 0
    reader = MC.CalibrationReader(profile, transport=bus)
    snap = reader.read_snapshot()
    assert set(snap) == {
        "raw_position", "raw_velocity", "raw_current",
        "sample_start_ns", "sample_end_ns",
        "q_bus_deg", "q_urdf_deg", "gripper_pct",
    }
    assert len(snap["raw_position"]) == 6
    assert len(snap["q_bus_deg"]) == 5
    assert snap["gripper_pct"] == pytest.approx(50.0, abs=1.0)
    assert snap["q_urdf_deg"] is None, "joints 未拟合时不得输出 URDF 角"
    assert snap["sample_start_ns"] <= snap["sample_end_ns"]
    # 六台各一次块读，且全程只有 READ：没有任何运动/寄存器写帧
    assert bus.read_request_count == 6
    assert bus.sync_write_count == 0
    assert not any(f[4] in (VENDOR_INST_WRITE, VENDOR_INST_SYNC_WRITE) for f in bus.tx_log)
    bus.registers[3][40] = 1
    assert reader.read_torque_enabled() == [0, 0, 1, 0, 0, 0]
    assert bus.read_request_count == 12
    assert bus.sync_write_count == 0
    assert not any(f[4] in (VENDOR_INST_WRITE, VENDOR_INST_SYNC_WRITE) for f in bus.tx_log)
    for sid in range(1, 7):
        bus.registers[sid][5] = sid
        bus.set_u16(sid, 31, MC.encode_sign_magnitude(-10 * sid, 11))
    eeprom = reader.read_eeprom_snapshot()
    assert [row["registers"]["ID"]["raw"] for row in eeprom] == list(range(1, 7))
    assert [row["registers"]["Model_Number"]["raw"] for row in eeprom] == [777] * 6
    assert [row["registers"]["Homing_Offset"]["decoded"] for row in eeprom] == [
        -10, -20, -30, -40, -50, -60]
    assert all(len(bytes.fromhex(row["raw_hex"])) == 40 for row in eeprom)
    assert bus.read_request_count == 18
    assert bus.sync_write_count == 0
    assert not any(f[4] in (VENDOR_INST_WRITE, VENDOR_INST_SYNC_WRITE) for f in bus.tx_log)
    for sid in range(1, 7):
        bus.registers[sid][82] = 40 + sid
    assert reader.read_common_register("Velocity_Unit_factor") == [41, 42, 43, 44, 45, 46]
    assert bus.read_request_count == 24
    assert bus.sync_write_count == 0
    assert not any(f[4] in (VENDOR_INST_WRITE, VENDOR_INST_SYNC_WRITE) for f in bus.tx_log)
    reader.close()
    with pytest.raises(MotorStateError):
        reader.read_snapshot()
    with pytest.raises(MotorStateError):
        reader.read_torque_enabled()
    with pytest.raises(MotorStateError):
        reader.read_eeprom_snapshot()
    with pytest.raises(MotorStateError):
        reader.read_common_register("Velocity_Unit_factor")
