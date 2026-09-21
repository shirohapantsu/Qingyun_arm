from __future__ import annotations

from dataclasses import dataclass

import scripts.sync_motor_calibration_eeprom as sync


@dataclass(frozen=True)
class _Axis:
    name: str
    servo_id: int


class _LockMapping:
    axes = tuple(_Axis(f"motor_{index}", index) for index in range(1, 7))

    @staticmethod
    def common_register(name: str) -> tuple[int, int]:
        assert name == "Lock"
        return 55, 1


class _FlakyLockProtocol:
    """模拟等级1无写应答，2号舵机第一次 Lock 写入丢失。"""

    def __init__(self) -> None:
        self.values = {axis.servo_id: (0 if axis.servo_id == 2 else 1)
                       for axis in _LockMapping.axes}
        self.writes: list[tuple[int, int]] = []
        self._drop_id2_once = True

    def bulk_read(self, ids, address, length, deadline):
        assert address == 55 and length == 1
        return {servo_id: bytes([self.values[servo_id]]) for servo_id in ids}

    def write_registers(self, servo_id, address, data, deadline):
        assert address == 55 and len(data) == 1
        self.writes.append((servo_id, data[0]))
        if servo_id == 2 and self._drop_id2_once:
            self._drop_id2_once = False
            return
        self.values[servo_id] = data[0]


def test_lock_restore_retries_only_the_unconfirmed_idempotent_lock_write(monkeypatch):
    monkeypatch.setattr(sync, "EEPROM_SETTLE_S", 0.0)
    protocol = _FlakyLockProtocol()

    restored = sync._set_and_verify_locks(protocol, _LockMapping(), [1] * 6)

    assert restored == [1] * 6
    assert protocol.writes == [(2, 1), (2, 1)]


class _CalibrationMapping:
    axes = (_Axis("shoulder_pan", 1),)

    @staticmethod
    def register(axis, name: str) -> tuple[int, int]:
        assert axis.name == "shoulder_pan"
        assert name == "Homing_Offset"
        return 31, 2

    @staticmethod
    def encode_signed(axis, name: str, value: int) -> int:
        assert axis.name == "shoulder_pan" and name == "Homing_Offset"
        return abs(value) | 0x800 if value < 0 else value

    @staticmethod
    def decode_signed(axis, name: str, raw: int) -> int:
        assert axis.name == "shoulder_pan" and name == "Homing_Offset"
        return -(raw & 0x7FF) if raw & 0x800 else raw


class _CalibrationProtocol:
    def __init__(self) -> None:
        self.raw = 0
        self.writes: list[tuple[int, int, bytes]] = []

    def write_registers(self, servo_id, address, data, deadline):
        payload = bytes(data)
        self.writes.append((servo_id, address, payload))
        self.raw = int.from_bytes(payload, "little")

    def read_registers(self, servo_id, address, length, deadline):
        return self.raw.to_bytes(length, "little")


def test_calibration_field_is_encoded_then_immediately_read_back(monkeypatch):
    monkeypatch.setattr(sync, "EEPROM_SETTLE_S", 0.0)
    protocol = _CalibrationProtocol()
    axis = _CalibrationMapping.axes[0]

    result = sync._write_and_verify(
        protocol, _CalibrationMapping(), axis, "Homing_Offset", -438
    )

    assert protocol.writes == [(1, 31, bytes([0xB6, 0x09]))]
    assert result["readback_decoded"] == -438
    assert result["verified"] is True
