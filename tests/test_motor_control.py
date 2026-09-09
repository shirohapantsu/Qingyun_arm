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
    """initialize() 走完整链路：应答配置(READ)+身份(PING/READ)+力矩(WRITE)。

    除 Torque_Enable 外不得有任何 WRITE，尤其禁止 Homing_Offset(31)、
    Min/Max 行程限位(9/11) 或 Response_Status_Level(8) 被写（技术文档 4.2、
    标定指南 4.1：不隐藏自动回零 / EEPROM 写入）。
    """
    bus = FakeBus()                     # 默认全应答等级 0，固件 2.54，型号 777
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    clock, nap = Clock().pair()
    ctl = Sts3215MotorController(params, mapping, transport=bus, clock=clock,
                                 sleep=nap, initialize=True)
    # 应答策略真的来自寄存器读回：默认等级 0 → 写要等应答
    assert all(ctl._protocol.write_expects_ack(sid) for sid in range(1, 7))
    # 六台力矩都已打开
    addr, _ = mapping.common_register("Torque_Enable")
    assert all(bus.registers[sid][addr] == 1 for sid in range(1, 7))
    # 全部 WRITE 帧只允许写 Torque_Enable；也没有 RESET 指令帧
    for frame in bus.tx_log:
        inst = frame[4]
        assert inst != VENDOR_INST_RESET
        if inst == VENDOR_INST_WRITE:
            assert frame[5] == addr, f"初始化写出了预期外的寄存器 {frame[5]}"
    # 事务数固定：6 应答配置 + 6 PING + 6 型号 + 12 固件 + 6 力矩 = 36
    assert len(bus.tx_log) == 36


def test_初始化按等级1配置后力矩写不等应答(params):
    """等级 1（出厂常见"只回读"）下若硬等应答，初始化会超时；必须按配置跳过。"""
    bus = FakeBus()
    for sid in range(1, 7):
        bus.set_write_level(sid, 1)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    clock, nap = Clock().pair()
    ctl = Sts3215MotorController(params, mapping, transport=bus, clock=clock,
                                 sleep=nap, initialize=True)
    assert all(not ctl._protocol.write_expects_ack(sid) for sid in range(1, 7))
    addr, _ = mapping.common_register("Torque_Enable")
    assert all(bus.registers[sid][addr] == 1 for sid in range(1, 7))


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
    reader.close()
    with pytest.raises(MotorStateError):
        reader.read_snapshot()
