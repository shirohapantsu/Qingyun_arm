# -*- coding: utf-8 -*-
"""feetech_bus.py — Feetech STS 系列总线舵机协议层（自实现，仅依赖 pyserial）

帧格式（STS/Feetech 总线，与 Dynamixel v1 风格一致，官方手册 + lerobot v0.6.1
src/lerobot/motors/feetech/ 交叉核对；寄存器表来源：STS3215 数据手册、
lerobot motors/feetech/tables.py、commanderfun/STS3215 REGISTER_REFERENCE）：

    请求:  FF FF | ID | LEN | INS | 参数... | CHK
    应答:  FF FF | ID | LEN | ERR | 数据... | CHK

    LEN = 参数个数 + 2（即 INS/ERR..CHK 的字节数）
    CHK = ~( ID + LEN + INS/ERR + 参数... ) & 0xFF   （低字节取反）
    多字节参数小端序（低字节在前）。

指令：PING=0x01  READ=0x02  WRITE=0x03
应答等级出厂默认“只回读指令”（Response_Status_Level=1）：WRITE 无应答 → 写即返回
（满足 send_action 非阻塞约定）；若固件配成全应答(2)，未读应答会在下次总线
操作前的清缓冲中被丢弃，不串帧。需要校验写结果时可显式 expect_ack=True。

半双工方向切换由 Feetech USB2Serial 转换板硬件自动完成（/dev/ttyACM0），
本层只负责帧级收发、超时与重试。
"""

from __future__ import annotations

import threading
import time

# ------------------------------------------------------------------ 协议常量
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03

BROADCAST_ID = 0xFE   # 广播（本层暂不使用）

# STS3215 控制表关键地址（详见模块 docstring 数据来源）
REG_ID = 0x05               # 舵机 ID（EEPROM）
REG_TORQUE_ENABLE = 0x28    # 40：力矩开关
REG_GOAL_POSITION = 0x2A    # 42：目标位置（2B, 0..4095）
REG_GOAL_TIME = 0x2C        # 44：到达时间 ms
REG_GOAL_SPEED = 0x2E       # 46：速度上限（0=最大）
REG_LOCK = 0x37             # 55：EEPROM 写锁
REG_PRESENT_POSITION = 0x38 # 56：当前角度（2B, 0..4095）
REG_PRESENT_SPEED = 0x3A    # 58：当前速度（2B, 符号位 bit15，单位 步/s）
REG_PRESENT_LOAD = 0x3C     # 60：当前负载（2B, 符号位 bit10）
REG_PRESENT_VOLTAGE = 0x3E  # 62：电压（×0.1V）
REG_PRESENT_TEMPERATURE = 0x3F  # 63：温度 ℃
REG_MOVING = 0x42           # 66：运动中标志
REG_PRESENT_CURRENT = 0x45  # 69：电流（2B, ×6.5mA）

MAX_FRAME_LEN = 64          # 应答帧长度上限（防御性）

# ------------------------------------------------------------------ 异常
class FeetechError(Exception):
    """总线错误基类。"""

class FeetechTimeoutError(FeetechError):
    """应答超时（舵机不在线/线序错/波特率错时最常见）。"""

class FeetechProtocolError(FeetechError):
    """帧不合法（帧头/长度/ID 不符）。"""

class FeetechChecksumError(FeetechProtocolError):
    """校验和不符（总线噪声或坏包，重试通常可恢复）。"""

class FeetechServoError(FeetechError):
    """舵机应答 ERR 字节非 0（过压/过温/超行程/堵转等）。"""

# ------------------------------------------------------------------ 帧构造/校验
def compute_checksum(parts: list[int]) -> int:
    """校验和 = 对 ID..末参数 求和取反的低字节。"""
    total = 0
    for b in parts:
        total += b
    return (~total) & 0xFF

def build_packet(servo_id: int, instruction: int, params: list[int]) -> bytes:
    """构造请求帧（含 0xFF 0xFF 帧头）。"""
    if not 0 <= servo_id <= 0xFD:
        raise ValueError(f"servo_id 越界: {servo_id}")
    if not 0 <= instruction <= 0xFF:
        raise ValueError(f"instruction 越界: {instruction}")
    length = len(params) + 2          # INS + 参数 + CHK
    mid = [servo_id, length, instruction]
    chk = compute_checksum(mid + params)
    return bytes([0xFF, 0xFF, *mid, *params, chk])

def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    """Feetech 符号-幅值编码：最高位为方向位，低位为幅值。"""
    if raw & (1 << sign_bit):
        return -(raw & ((1 << sign_bit) - 1))
    return raw & ((1 << sign_bit) - 1)

def le_u16(b: bytes, offset: int) -> int:
    return b[offset] | (b[offset + 1] << 8)

# ------------------------------------------------------------------ 总线
class FeetechBus:
    """单条 STS 串口总线的帧级收发。线程安全（TX/RX 互斥），
    典型单舵机整块反馈读 ≈ 1 次往返 ~2.5ms。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        response_timeout_s: float = 0.05,
        read_retries: int = 2,
        ack_peek_s: float = 0.008,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = response_timeout_s
        self.retries = read_retries
        self.ack_peek_s = ack_peek_s
        self._ser = None            # pyserial Serial，connect() 打开
        self._io_lock = threading.RLock()

    # ------------------------------------------------------------ 生命周期
    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self) -> None:
        if self.is_open:
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - 依赖缺失提示
            raise FeetechError(
                "缺少 pyserial，请先安装: pip install pyserial"
            ) from exc
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01,          # read(1) 粒度的短超时，外层有 deadline
                write_timeout=0.1,
            )
        except Exception as exc:
            raise FeetechError(
                f"无法打开串口 {self.port!r}: {exc}\n"
                "  请检查：转换板是否插入；端口号是否正确 "
                "(Linux: /dev/ttyACM0，Windows: COMx，可用环境变量 "
                "QINGYUN_SERVO_PORT 指定)"
            ) from exc
        self._flush_input()

    def close(self) -> None:
        with self._io_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    def __enter__(self) -> "FeetechBus":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------ 内部收发
    def _require_open(self):
        if not self.is_open:
            raise FeetechError(f"总线未连接: {self.port!r}，请先调用 connect()")

    def _flush_input(self) -> None:
        """丢弃输入缓冲残留（噪声/过期应答），防串帧。"""
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def _send(self, packet: bytes) -> None:
        try:
            self._ser.write(packet)
            self._ser.flush()
        except Exception as exc:
            raise FeetechError(f"串口写失败: {exc}") from exc

    def _read_status_packet(self, expected_id: int | None,
                            timeout_s: float | None = None) -> tuple[int, bytes]:
        """读一帧完整状态包。

        返回 (err, payload)，payload 不含 ERR 与 CHK 字节。
        超时抛 FeetechTimeoutError；帧不合法/校验错抛 FeetechProtocolError。
        timeout_s 缺省用总线级 self.timeout_s（应答探测等短窗口可显式传小值）。
        """
        ser = self._ser
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else self.timeout_s)

        def read_byte() -> int:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FeetechTimeoutError(
                        f"等待应答超时(>{self.timeout_s*1000:.0f}ms)"
                    )
                b = ser.read(1)
                if b:
                    return b[0]

        def read_exact(n: int) -> bytearray:
            buf = bytearray()
            while len(buf) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FeetechTimeoutError(
                        f"应答不完整: 期望{n}字节, 实收{len(buf)}"
                    )
                chunk = ser.read(n - len(buf))
                if chunk:
                    buf.extend(chunk)
            return buf

        # 1) 扫描帧头 FF FF（容忍总线噪声）
        while True:
            b = read_byte()
            if b == 0xFF and read_byte() == 0xFF:
                break
        # 2) ID / LEN
        servo_id = read_byte()
        length = read_byte()
        if length < 2 or length > MAX_FRAME_LEN:
            raise FeetechProtocolError(f"非法帧长 LEN={length}")
        if expected_id is not None and servo_id != expected_id:
            raise FeetechProtocolError(
                f"应答 ID 不符: 期望{expected_id}, 实收{servo_id}"
            )
        # 3) ERR..CHK 共 LEN 字节
        rest = read_exact(length)
        err, payload, chk = rest[0], bytes(rest[1:-1]), rest[-1]
        expect_chk = compute_checksum([servo_id, length, err, *payload])
        if chk != expect_chk:
            raise FeetechChecksumError(
                f"校验和不符: 实收0x{chk:02X}, 期望0x{expect_chk:02X}"
            )
        return err, payload

    def _transact(self, servo_id: int, instruction: int, params: list[int],
                  reply_expected: bool) -> tuple[int, bytes]:
        """带锁收发一次，坏包按 self.retries 重试（重试前清缓冲）。"""
        last_exc: FeetechError | None = None
        for attempt in range(self.retries + 1):
            with self._io_lock:
                try:
                    self._require_open()
                    self._flush_input()
                    self._send(build_packet(servo_id, instruction, params))
                    if not reply_expected:
                        return 0, b""
                    err, payload = self._read_status_packet(servo_id)
                    return err, payload
                except FeetechError as exc:
                    last_exc = exc
                    try:
                        self._flush_input()
                    except Exception:
                        pass
        raise FeetechError(
            f"ID={servo_id} 指令0x{instruction:02X} 连续{self.retries+1}次失败: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------ 公共读写
    def ping(self, servo_id: int) -> int:
        """PING 单舵机，返回 ERR 字节（0=正常）；不在线抛超时。"""
        err, _ = self._transact(servo_id, INST_PING, [], reply_expected=True)
        return err

    def read_block(self, servo_id: int, addr: int, size: int) -> bytes:
        """连续读 size 字节（小端寄存器区）。"""
        if size < 1 or size > 30:
            raise ValueError(f"size 越界: {size}")
        err, payload = self._transact(
            servo_id, INST_READ, [addr & 0xFF, size & 0xFF], reply_expected=True
        )
        if err != 0:
            raise FeetechServoError(f"ID={servo_id} 读@0x{addr:02X} 返回错误码0x{err:02X}")
        if len(payload) != size:
            raise FeetechProtocolError(
                f"读@0x{addr:02X} 数据长度不符: 期望{size}, 实收{len(payload)}"
            )
        return payload

    def read_register(self, servo_id: int, addr: int, size: int) -> int:
        """读单个寄存器（多字节小端）。"""
        data = self.read_block(servo_id, addr, size)
        value = 0
        for i in range(size):
            value |= data[i] << (8 * i)
        return value

    def write(self, servo_id: int, addr: int, data: bytes | list[int],
              expect_ack: bool = False) -> None:
        """WRITE 指令。默认 expect_ack=False：不等待应答，写完立即返回（非阻塞语义）。

        STS 出厂“只回读指令”(Response_Status_Level=1) 时写本无应答；即便固件配为
        全应答，未读的应答也会在下次总线操作前的清缓冲中被丢弃，不会串帧，
        因此常规下发无需探测。需要校验写结果时传 expect_ack=True，将在
        ack_peek_s 短窗口内探测并校验应答。
        """
        if isinstance(data, bytes):
            data_list = list(data)
        else:
            data_list = list(data)
        if not data_list or len(data_list) > 30:
            raise ValueError(f"写入数据长度非法: {len(data_list)}")

        last_exc: FeetechError | None = None
        for _attempt in range(self.retries + 1):
            with self._io_lock:
                try:
                    self._require_open()
                    self._flush_input()
                    self._send(build_packet(servo_id, INST_WRITE, [addr & 0xFF, *data_list]))
                    if not expect_ack:
                        return              # 无应答模式：写入完成
                    err, _payload = self._read_status_packet(servo_id, timeout_s=self.ack_peek_s)
                    if err != 0:
                        raise FeetechServoError(
                            f"ID={servo_id} 写@0x{addr:02X} 返回错误码0x{err:02X}"
                        )
                    return
                except FeetechError as exc:
                    last_exc = exc
                    try:
                        self._flush_input()
                    except Exception:
                        pass
        raise FeetechError(
            f"ID={servo_id} 写@0x{addr:02X} 连续{self.retries+1}次失败: {last_exc}"
        ) from last_exc
