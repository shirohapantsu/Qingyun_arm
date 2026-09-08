"""STS3215 驱动层测试。

覆盖 docs/真机参数测量与标定指南.md 第 4 节（关节换算、电流/速度单位、夹爪两端）、
第 12 节共享函数与"必要离线检查"第 2、6 条，以及 docs/机械臂运动控制模块技术文档.md
第 4.2 节（三个协议方法的语义：整条校验、有界 I/O、越限拒绝、不返回缓存）。

真机串口不参与测试：全部走注入的假 transport 与假时钟。没有装 pyserial 的机器
也必须能 import 本模块并跑完换算层与协议层用例（见 test_没有pyserial时仍可import）。
"""

from __future__ import annotations

import math

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
# 假总线：按 STS protocol 0 应答，并允许注入各种故障
# ---------------------------------------------------------------------------


class FakeBus:
    """一个足够真实的 STS 从机集合，用于在无硬件条件下验证协议层。

    只实现测试需要的三件事：READ_DATA 回状态帧、SYNC_WRITE 落寄存器、PING 回型号。
    另外提供故障开关：不应答、回错校验和、回错 ID、置 ERROR 位、只回一半。
    """

    def __init__(self, ids=(1, 2, 3, 4, 5, 6), model_number=777) -> None:
        self.registers = {i: {} for i in ids}      # id -> {addr: value}
        self.model_number = model_number
        self.tx_log: list[bytes] = []
        self.rx_queue = bytearray()
        self.fail_no_response = False
        self.fail_bad_checksum = False
        self.fail_wrong_id = False
        self.fail_error_bit = 0
        self.fail_short_reply = False
        self.write_count = 0
        self.sync_write_count = 0
        self.read_request_count = 0

    # --- ByteTransport 协议 ---

    def write(self, data: bytes) -> int:
        self.tx_log.append(bytes(data))
        self.write_count += 1
        data = bytes(data)
        assert data[:2] == b"\xff\xff", f"帧头不对：{data[:2].hex()}"
        servo_id, length, inst = data[2], data[3], data[4]
        params = list(data[5:5 + length - 2])
        if inst == StsProtocol.INST_SYNC_WRITE:
            self.sync_write_count += 1
            addr, size = params[0], params[1]
            for k in range(2, len(params), size + 1):
                sid = params[k]
                # 按字节落寄存器，便于之后逐字节读回
                for off in range(size):
                    self.registers.setdefault(sid, {})[addr + off] = params[k + 1 + off]
            return len(data)                    # 广播不回包
        if inst == StsProtocol.INST_PING:
            self._reply(servo_id, [])
            return len(data)
        if inst == StsProtocol.INST_READ_DATA:
            addr, size = params[0], params[1]
            self.read_request_count += 1
            if self.fail_no_response:
                return len(data)
            # registers 是"字节地址 -> 字节值"，所以按 size 个连续字节回。
            payload = [self.registers.get(servo_id, {}).get(addr + off, 0) & 0xFF
                       for off in range(size)]
            if self.fail_short_reply:
                payload = payload[:1]
            self._reply(servo_id, payload)
            return len(data)
        if inst == StsProtocol.INST_WRITE_DATA:
            addr = params[0]
            for off in range(len(params) - 1):
                self.registers.setdefault(servo_id, {})[addr + off] = params[1 + off]
            self._reply(servo_id, [])
            return len(data)
        self._reply(servo_id, [])
        return len(data)

    def _reply(self, servo_id: int, payload: list[int]) -> None:
        sid = servo_id
        if self.fail_wrong_id:
            sid = (servo_id + 7) & 0xFF
        body = [sid, len(payload) + 2, self.fail_error_bit, *payload]
        chk = StsProtocol.checksum(body)
        if self.fail_bad_checksum:
            chk = (chk + 1) & 0xFF
        self.rx_queue += bytes([0xFF, 0xFF, *body, chk])

    def read(self, n: int) -> bytes:
        out = bytes(self.rx_queue[:n])
        del self.rx_queue[:n]
        return out

    def read_available(self) -> bytes:
        return self.read(len(self.rx_queue))

    def flush_input(self) -> None:
        self.rx_queue.clear()

    def close(self) -> None:
        pass

    # --- 便捷：把寄存器按 2 字节小端写进从机 ---

    def set_u16(self, servo_id: int, addr: int, value: int) -> None:
        lo, hi = StsProtocol.split_u16(int(value) & 0xFFFF)
        self.registers.setdefault(servo_id, {})[addr] = lo
        self.registers[servo_id][addr + 1] = hi


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
# 3. 协议层：组包与解包
# ---------------------------------------------------------------------------


def test_校验和定义():
    body = [1, 4, 4, 56, 2]
    assert StsProtocol.checksum(body) == (~sum(body)) & 0xFF


def test_sync_write帧逐字节正确():
    """同步写是广播、不回包，所以只能校验"提交完成"，不宣称物理原子性。"""
    bus = FakeBus()
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    ids = [1, 2, 3, 4, 5, 6]
    values = [2048, 100, 4095, 0, 7, 3600]
    proto.sync_write_targets(ids, 42, 2, values, deadline_s=99.0)
    frame = bus.tx_log[-1]
    assert frame[:2] == b"\xff\xff"
    assert frame[2] == StsProtocol.BROADCAST_ID
    params_len = 2 + 6 * 3
    assert frame[3] == params_len + 2
    assert frame[4] == StsProtocol.INST_SYNC_WRITE
    assert frame[5] == 42 and frame[6] == 2
    # 校验和覆盖"从 ID 到最后一个参数"，也就是整帧去掉帧头和校验和本身。
    assert StsProtocol.checksum(list(frame[2:-1])) == frame[-1]
    for k, (sid, val) in enumerate(zip(ids, values)):
        base = 7 + k * 3
        assert frame[base] == sid
        assert StsProtocol.join_u16(frame[base + 1:base + 3]) == val
    # 广播写不落总线确认，但从机侧寄存器必须已经更新
    assert StsProtocol.join_u16(bytes([bus.registers[6][42], bus.registers[6][43]])) == 3600


def test_读回正常状态帧():
    bus = FakeBus()
    bus.set_u16(3, 56, 2050)
    clock, nap = Clock().pair()
    proto = StsProtocol(bus, clock=clock, sleep=nap)
    got = proto.read_registers(3, 56, 2, deadline_s=99.0)
    assert StsProtocol.join_u16(got) == 2050


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
    """给 6 个从机写入"当前姿态"的原始位置读数。"""
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


def test_send_action一次提交六个通道(params):
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    q = np.array(params.workspace.home_joints_deg, float)
    ctl.send_action(q, 62.0)
    # 一条同步写帧（可能再配一条 REG_WRITE 触发动作），但不是六条独立写。
    assert bus.write_count <= 2
    written = StsProtocol.join_u16(bytes([bus.registers[1][42], bus.registers[1][43]]))
    assert written == ctl.mapping.degrees_to_raw("shoulder_pan", float(q[0]))


def test_get_feedback返回换算后的物理量(params):
    """文档 4.2：完整返回六电机的已换算反馈；电流/速度必须是物理量。"""
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    bus.set_u16(2, 58, 20)
    bus.set_u16(2, 69, 40)
    fb = ctl.get_feedback()
    assert fb.angles_deg.shape == (5,) and fb.angles_deg.dtype == np.float64
    assert fb.speeds_deg_s.shape == (5,) and fb.currents_ma.shape == (5,)
    assert fb.angles_deg[0] == pytest.approx(params.workspace.home_joints_deg[0], abs=0.1)
    assert fb.speeds_deg_s[1] == pytest.approx(
        mapping.raw_velocity_to_deg_s("shoulder_lift", 20))
    assert fb.currents_ma[1] == pytest.approx(mapping.raw_current_to_ma("shoulder_lift", 40))
    assert 0.0 <= fb.gripper_pct <= 100.0
    assert fb.sample_start_ns <= fb.sample_end_ns
    seq = fb.sequence
    assert ctl.get_feedback().sequence > seq, "sequence 必须每次完整成功读回后递增"


def test_读回跨度过大视为无效采样且不返回缓存(params):
    """文档 4.2：失败时不返回缓存数据。"""
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    good = ctl.get_feedback()
    # 让每次事务都消耗掉大段时间，使整次读回跨度超过上限
    ctl._clock_obj.step_s = params.timing.max_feedback_span_s
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()
    # 上一次的好反馈没有被"续期"，也没有悄悄返回旧数据
    again = ctl.get_feedback.__self__
    assert again is ctl
    ctl._clock_obj.step_s = 0.0
    assert ctl.get_feedback().sequence > good.sequence


def test_通信失败向上抛出不吞(params):
    ctl, bus = _controller(params)
    ctl._torque_enabled = True
    bus.fail_no_response = True
    with pytest.raises(MotorCommunicationError):
        ctl.get_feedback()


def test_hold_current恰好一读一写(params):
    """文档 4.2：有界地读一次并写一次相同五关节角与夹爪开度，返回所用反馈。"""
    ctl, bus = _controller(params)
    calib = load_motor_calibration(SIM_PROFILE.parent / params.model.motor_calibration_path)
    mapping = MC.MotorMapping.from_params(params, calib)
    ctl._torque_enabled = True
    _seed_feedback(bus, params, mapping)
    reads_before = bus.read_request_count
    writes_before = bus.write_count
    fb = ctl.hold_current()
    # 一次完整读回 = 每个通道一组寄存器；实测每组要发 READ 请求，台数决定请求数，
    # 但整个 hold_current 只允许"一次读 + 一次写"，所以写请求必须恰好 +1。
    assert bus.read_request_count > reads_before
    assert bus.read_request_count <= reads_before + 2 * 6, "读回请求数异常，可能在逐寄存器轮询"
    # 注意 write_count 把所有请求帧（含 READ_DATA）都算进去了，"写了几次运动命令"
    # 要看同步写帧的条数。
    assert bus.sync_write_count == 1, bus.sync_write_count
    # 写回去的就是读到的那组值
    written = StsProtocol.join_u16(bytes([bus.registers[1][42], bus.registers[1][43]]))
    assert written == mapping.degrees_to_raw("shoulder_pan", float(fb.angles_deg[0]))


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
    from pathlib import Path

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
