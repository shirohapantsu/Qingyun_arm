# -*- coding: utf-8 -*-
"""motor_control.py — 电机驱动实现（C 号真机驱动，满足 MotorController 冻结契约）

契约：configs/common_interface.py 的 MotorController Protocol / JointFeedback；
本类对应“3号负责”的电机驱动，B 侧（arm_control）只按 Protocol 编程，
真机/Mock 由 main.py 一个开关切换。

第一个里程碑：**舵机实时回传角度**。真机验收（在 Qingyun_arm 根目录）：

    # Linux(香橙派): 端口默认 /dev/ttyACM0
    python -m qingyun.grabbing.motor_control --hz 30
    # Windows 开发机: 先设端口
    set QINGYUN_SERVO_PORT=COM5 && python -m qingyun.grabbing.motor_control

    # 关闭力矩后手掰舵机，实时角度会跟着变（验证“实测回传”而非“目标回显”）
    python -m qingyun.grabbing.motor_control --no-torque

    # 真机自检“目标角→运动→实测回传”全链路（单关节慢速往返扫掠，不依赖 B 的解算；
    # 验证 send_action 能让舵机按目标角动起来、get_feedback 全程跟随）：
    python -m qingyun.grabbing.motor_control --sweep pan --amp 30 --rate 15
    #    --sweep 关节名(pan/lift/elbow/wrF/wrR/grip 或长名) --amp 单侧幅值(°) --rate 角速度(°/s)

依赖：pyserial（串口）、numpy（契约数组）。协议层见 feetech_bus.py（自实现，无官方 SDK）。

时序约定（技术文档 §5.2）：
- send_action 仅写不读（STS 默认写无应答），非阻塞返回；
- 时钟节拍（30Hz、绝对时间对齐）归 B；本文件无自持定时循环，
  仅 wait_until_settled / emergency_stop(freeze) 按自身阻塞语义短循环；
- 软限位最后一道钳位在此处（send_action 内），沿袭 lerobot motors_bus 标定范围钳位思路。

角度换算：原始磁编码 0..4095（0~360°）→ 关节 deg 的线性标定模型在
configs/servo_params.py 的 ServoJointConfig（零位/方向待 3号标定后回填，
未标定时按 zero_raw=2048、direction=+1 显示为 -180°..+180° 相对角）。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import numpy.typing as _npt  # noqa: F401  副作用: 先导入以绑定 np.typing 属性
# （numpy<2 未显式导入 numpy.typing 前 np.typing 不可用；冻结契约 common_interface
#   的类注解会即时求值 np.typing.NDArray，故必须在本文件 import 契约前完成绑定）

from configs.common_interface import JointFeedback
from configs.servo_params import (
    CURRENT_MA_PER_BIT,
    FEEDBACK_LEN,
    FEEDBACK_START_ADDR,
    GRIPPER_DEG_AT_0_PCT,
    GRIPPER_DEG_AT_100_PCT,
    MOTOR_CONFIGS,
    SERVO_ACK_PEEK_S,
    SERVO_BAUDRATE,
    SERVO_PORT,
    SERVO_READ_RETRIES,
    SERVO_RESPONSE_TIMEOUT_S,
    SPEED_DEG_S_PER_BIT,
    ServoJointConfig,
)
from qingyun.grabbing.feetech_bus import (
    REG_GOAL_POSITION,
    REG_TORQUE_ENABLE,
    FeetechBus,
    FeetechError,
    decode_sign_magnitude,
    le_u16,
)

# 显示用短名（与官方索引 0..5 同序）
_SHORT_NAMES = ("pan", "lift", "elbow", "wrF", "wrR", "grip")
_MODEL_NUMBER_ADDR = 0x03    # 2B 只读，STS3215 = 777 (0x0309)
_MODEL_NUMBER_EXPECTED = 0x0309

FREEZE_HZ = 30.0             # freeze 锁定频率
SETTLE_SAMPLE_S = 0.05       # wait_until_settled 采样间隔


class MotorError(Exception):
    """电机侧通用错误。"""

class ServoFeedbackError(MotorError):
    """单舵机反馈读取失败（携带关节名与舵机 ID）。"""

    def __init__(self, joint_name: str, servo_id: int, cause: Exception):
        super().__init__(f"关节 {joint_name}(舵机ID={servo_id}) 反馈读取失败: {cause}")
        self.joint_name = joint_name
        self.servo_id = servo_id


class STS3215MotorController:
    """STS3215 ×6 驱动：实现 MotorController 协议的 4 个方法 + 生命周期/诊断。

    用法：
        ctrl = STS3215MotorController()      # 端口取自 servo_params / 环境变量
        ctrl.connect()                       # 打开串口
        ctrl.enable_torque(True)             # 需要执行时使能力矩
        fb = ctrl.get_feedback()             # ← 实时回传角度/速度/电流
        ctrl.send_action(q5, gripper_pct=50)
        ctrl.emergency_stop(); ctrl.release_freeze()
        ctrl.close()
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int | None = None,
        motor_configs: tuple[ServoJointConfig, ...] | None = None,
        bus: FeetechBus | None = None,
    ) -> None:
        self.configs: tuple[ServoJointConfig, ...] = (
            tuple(motor_configs) if motor_configs is not None else MOTOR_CONFIGS
        )
        if len(self.configs) != 6:
            raise ValueError(f"motor_configs 必须为 6 个关节，实收 {len(self.configs)}")
        self._bus = bus or FeetechBus(
            port=port or SERVO_PORT,
            baudrate=baudrate or SERVO_BAUDRATE,
            response_timeout_s=SERVO_RESPONSE_TIMEOUT_S,
            read_retries=SERVO_READ_RETRIES,
            ack_peek_s=SERVO_ACK_PEEK_S,
        )
        # ---- freeze 状态（emergency_stop 后台线程） 急停冻结
        self._freeze_lock = threading.RLock()
        self._freeze_thread: threading.Thread | None = None
        self._freeze_stop = threading.Event()
        self._frozen = False

    # ================================================================ 生命周期
    def connect(self) -> None:
        """打开串口总线（不使能力矩，防止上电即误动）。"""
        self._bus.connect()

    def close(self) -> None:
        self.release_freeze()
        self._bus.close()

    def __enter__(self) -> "STS3215MotorController":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def enable_torque(self, on: bool) -> None:
        """6 舵机力矩开关。读角度/速度/电流不需要力矩；下发运动前必须先使能。"""
        for cfg in self.configs:
            self._bus.write(cfg.servo_id, REG_TORQUE_ENABLE, [1 if on else 0])

    # ================================================================ 诊断
    def diagnose(self) -> None:
        """上电自检：逐个 PING + 读型号/整块状态，打印在线表格。

        用于排查“舵机不上线”（供电/线序/ID 烧录/波特率）。
        """
        print(f"[连接] 串口 {self._bus.port} @ {self._bus.baudrate} bps")
        online = 0
        for cfg in self.configs:
            try:
                model = self._bus.read_register(cfg.servo_id, _MODEL_NUMBER_ADDR, 2)
                st = self._read_raw_state(cfg)
                online += 1
                mark = "OK " if model == _MODEL_NUMBER_EXPECTED else "??"
                print(
                    f"  ID={cfg.servo_id:>2} {cfg.name:<13} 在线[{mark}]  "
                    f"model=0x{model:04X}  raw={st['position']:4d}  "
                    f"{st['voltage'] / 10:.1f}V  {st['temperature']:>2}°C"
                )
            except FeetechError as exc:
                print(f"  ID={cfg.servo_id:>2} {cfg.name:<13} 离线   ({exc})")
        if online == 0:
            raise MotorError(
                "6 个舵机全部不在线。请依次检查：总线供电、USB2Serial 接线、"
                "端口号(SERVO_PORT)、波特率 1Mbps、舵机 ID 烧录。"
            )
        if online < len(self.configs):
            print(f"[警告] 仅 {online}/{len(self.configs)} 个舵机在线")
        else:
            print(f"[诊断] {online}/{len(self.configs)} 个舵机全部在线")

    # ================================================================ 反馈（本里程碑核心）
    def get_feedback(self) -> JointFeedback: #把 6 台舵机的原始量变成关节量
        """同步读 6 舵机角度/速度/电流（冻结契约）。

        每舵机 1 次整块寄存器读（Present_Position..Present_Current，15B），
        6 次往返 ≈ 15ms@1Mbps，满足 30Hz 节拍内富余（10ms 同步读为后续优化项）。
        """
        angles = np.empty(6, dtype=np.float64)
        speeds = np.empty(6, dtype=np.float64)
        currents = np.empty(6, dtype=np.float64)
        for i, cfg in enumerate(self.configs):
            try:
                raw = self._read_raw_state(cfg)
            except FeetechError as exc:
                raise ServoFeedbackError(cfg.name, cfg.servo_id, exc) from exc
            angles[i] = cfg.raw_to_deg(raw["position"])
            # Present_Speed: 符号-幅值编码(bit15)，单位 步/s → deg/s
            speeds[i] = raw["speed"] * SPEED_DEG_S_PER_BIT * cfg.direction
            # Present_Current: 1 bit = 6.5mA
            currents[i] = raw["current"] * CURRENT_MA_PER_BIT
        return JointFeedback(angles_deg=angles, speeds_deg_s=speeds, currents_ma=currents)

    def _read_raw_state(self, cfg: ServoJointConfig) -> dict: #底层拆包
        """单舵机整块读反馈寄存器，返回原始量 dict（含标定与调参用细节）。"""
        data = self._bus.read_block(cfg.servo_id, FEEDBACK_START_ADDR, FEEDBACK_LEN)
        return {
            "position": le_u16(data, 0),
            "speed": decode_sign_magnitude(le_u16(data, 2), 15),
            "load": decode_sign_magnitude(le_u16(data, 4), 10),
            "voltage": data[6],
            "temperature": data[7],
            "moving": data[10],
            "current": le_u16(data, 13),
        }

    # ================================================================ 动作下发
    def send_action(self, joints_deg: np.ndarray, gripper_pct: float) -> None:
        """非阻塞下发 5 姿态关节 + 夹爪开度（冻结契约）。

        内部先做行程软限位钳位（最后一道安全闸）再逐舵机写目标位置；
        STS 写指令无应答，写完立即返回，无内部 sleep。
        """
        q = np.asarray(joints_deg, dtype=np.float64).reshape(-1)
        if q.size != 5:
            raise ValueError(f"joints_deg 应为 5 个姿态关节角，实收 shape={np.shape(joints_deg)}")

        with self._freeze_lock:
            if self._frozen:
                raise MotorError(
                    "电机处于 emergency_stop(freeze) 锁定中，拒绝新指令；"
                    "确认安全后需先调用 release_freeze()"
                )

        # ---- 最后一道闸：行程软限位钳位（真值来源 servo_params MOTOR_CONFIGS）
        limits = [(cfg.min_deg, cfg.max_deg) for cfg in self.configs[:5]]
        clipped = np.clip(
            q, [lo for lo, _ in limits], [hi for _, hi in limits]
        ).astype(np.float64)

        # ---- 夹爪：百分比 → deg → 钳位
        pct = float(np.clip(gripper_pct, 0.0, 100.0))
        gripper_deg = GRIPPER_DEG_AT_0_PCT + pct / 100.0 * (
            GRIPPER_DEG_AT_100_PCT - GRIPPER_DEG_AT_0_PCT
        )
        g_cfg = self.configs[5]
        gripper_deg = float(
            np.clip(gripper_deg, g_cfg.min_deg, g_cfg.max_deg)
        )

        for cfg, deg in zip(self.configs[:5], clipped):
            self._write_goal_deg(cfg, float(deg))
        self._write_goal_deg(g_cfg, gripper_deg)

    def _write_goal_deg(self, cfg: ServoJointConfig, deg: float) -> None:
        """关节角 → 舵机原始目标位（0..4095）并写入。"""
        raw = int(round(cfg.deg_to_raw(deg)))
        raw = int(np.clip(raw, 0, 4095))
        self._bus.write(cfg.servo_id, REG_GOAL_POSITION, [raw & 0xFF, (raw >> 8) & 0xFF])

    # ================================================================ 阻塞/停止原语
    def wait_until_settled(self, timeout_s: float, tol_deg: float) -> bool:  
        """阻塞等待全部关节静止（角度变化率 < tol_deg，单位 deg/s，按冻结契约“变化率”解释），
        超时返回 False。实现：位置差分测速（不依赖速度寄存器换算，抗单位不确定）。"""
        deadline = time.monotonic() + float(timeout_s)
        prev: np.ndarray | None = None
        prev_t = 0.0
        stable_rounds = 0
        while True:
            fb = self.get_feedback()           # 读一次全臂状态
            now_t = time.monotonic()
            now = fb.angles_deg                # 当前 6 关节角度
            if prev is not None and now_t > prev_t:
                rates = np.abs(now - prev) / (now_t - prev_t)    # 逐关节速度(deg/s)
                if float(np.max(rates)) < float(tol_deg):        # 最慢的那个关节……
                    stable_rounds += 1
                    if stable_rounds >= 2:      # 连续两次静止判据，滤抖
                        return True
                else:
                    stable_rounds = 0
            prev, prev_t = now, now_t           # 存下本次读数，供下次差分
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(SETTLE_SAMPLE_S, remaining))

    def emergency_stop(self) -> None:
        """freeze（冻结契约）：以当前实测角度为锁定位，后台 30Hz 持续下发，
        臂僵在原地（不切力矩）。恢复需人工确认后 release_freeze()。"""
        with self._freeze_lock:
            if self._frozen:
                return
            fb = self.get_feedback()   # 先取实测角度；失败则无法冻结，向上抛
            self._frozen = True
            self._freeze_stop = threading.Event()
            self._freeze_thread = threading.Thread(
                target=self._freeze_loop,
                args=(fb.angles_deg.copy(),),
                name="servo-freeze",
                daemon=True,
            )
            self._freeze_thread.start()
            print("[急停] 已进入 freeze：按实测角度持续锁定（release_freeze() 解除）",
                  file=sys.stderr)

    def _freeze_loop(self, _initial_angles: np.ndarray) -> None:
        """冻结循环：每拍实测 → 回写（动态跟随，抗外力推动漂移）。"""
        while not self._freeze_stop.is_set():
            try:
                fb = self.get_feedback()
            except MotorError:
                print("[急停] freeze 期间反馈读取失败，停止续写（物理急停归硬件供电回路）",
                      file=sys.stderr)
                break
            for cfg, deg in zip(self.configs, fb.angles_deg):
                try:
                    self._write_goal_deg(cfg, float(deg))
                except FeetechError:
                    pass
            self._freeze_stop.wait(1.0 / FREEZE_HZ)
        with self._freeze_lock:
            self._frozen = False

    def release_freeze(self) -> None:
        """人工确认安全后解除 freeze 锁定。

        注意：只置位事件并等待冻结线程退出，不替换事件对象——
        冻结循环每拍读取实例上的 _freeze_stop，替换会导致旧线程永远停不下来。
        """
        with self._freeze_lock:
            stop = self._freeze_stop
            thread = self._freeze_thread
        stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._freeze_lock:
            if self._freeze_thread is thread:
                self._freeze_thread = None
            self._frozen = False


# ================================================================ 运动链路自检（不依赖 B）
def _resolve_joint(ctrl: "STS3215MotorController", joint: str) -> int:
    """关节名解析：支持短名(pan/...)与 servo_params 里的长名。返回索引 0..5。"""
    by_long = {cfg.name: i for i, cfg in enumerate(ctrl.configs)}
    by_short = {s: i for i, s in enumerate(_SHORT_NAMES)}
    if joint in by_long:
        return by_long[joint]
    if joint in by_short:
        return by_short[joint]
    raise ValueError(
        f"未知关节 {joint!r}，可选短名: {', '.join(_SHORT_NAMES)}"
        f"（或长名: {', '.join(by_long)}）"
    )


def sweep_joint(
    ctrl: "STS3215MotorController",
    joint: str,
    amp_deg: float = 30.0,
    rate_deg_s: float = 15.0,
    dev_tol_deg: float = 8.0,
) -> bool:
    """单关节慢速往返扫掠自检：验证 C 侧"目标角 → 实际运动 → 实测回传"全链路。

    流程：其余 5 关节锁定在当前实测位 → 该关节从当前位慢速推进到 +amp → -amp
    → 回到起点，每步 send_action 后立即 get_feedback 比对，|实测-指令| 超
    dev_tol_deg 立即中止返回 False（典型原因：卡阻 / 换算方向或零点错 / 未使能）。
    结束时回到起点且静置偏差在容差内才算通过。执行前需 connect() 且
    enable_torque(True)；全程人离开机械范围、手放物理急停上。

    参数 amp_deg 单侧幅值(°)；rate_deg_s 目标角速度(°/s)，默认 15°/s 慢而稳。
    """
    idx = _resolve_joint(ctrl, joint)
    cfg = ctrl.configs[idx]

    def deg_to_pct(deg: float) -> float:
        span = GRIPPER_DEG_AT_100_PCT - GRIPPER_DEG_AT_0_PCT
        pct = (float(deg) - GRIPPER_DEG_AT_0_PCT) / span * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    fb0 = ctrl.get_feedback()
    start = float(fb0.angles_deg[idx])
    lo = max(float(cfg.min_deg), start - amp_deg)
    hi = min(float(cfg.max_deg), start + amp_deg)
    if hi - lo < 1.0:
        raise MotorError(
            f"{cfg.name} 扫掠幅值被软限位吃光: 当前 {start:.1f}°，"
            f"限位 [{cfg.min_deg:+.1f}, {cfg.max_deg:+.1f}]°，请减小 --amp"
        )
    print(f"[扫掠] {cfg.name} (ID={cfg.servo_id}): {start:+.2f}° → [{lo:+.2f}, {hi:+.2f}]°"
          f" 往返，速率≈{rate_deg_s:.0f}°/s，偏差容差 ±{dev_tol_deg:.1f}°")

    base = fb0.angles_deg.copy()
    base_pct = deg_to_pct(float(base[5]))

    # 生成往返目标序列（100ms 一拍，tick 角度 = 速率×0.1s，下限 0.5° 防停走）
    tick_s = 0.1
    tick_deg = max(rate_deg_s * tick_s, 0.5)
    targets: list[float] = []
    p = start
    while p < hi - 1e-9:
        p = min(p + tick_deg, hi)
        targets.append(p)
    while p > lo + 1e-9:
        p = max(p - tick_deg, lo)
        targets.append(p)
    while p < start - 1e-9:
        p = min(p + tick_deg, start)
        targets.append(p)

    t0 = time.monotonic()
    for i, cmd in enumerate(targets):
        v = base.copy()
        v[idx] = cmd
        ctrl.send_action(v[:5], base_pct)
        if i < len(targets) - 1:
            time.sleep(tick_s)          # 慢速推进；最后一步不睡，直接静置判偏差
        fb = ctrl.get_feedback()
        actual = float(fb.angles_deg[idx])
        dev = abs(actual - cmd)
        print(f"\r   [{time.monotonic() - t0:5.1f}s] 目标 {cmd:+9.2f}°  实测 {actual:+9.2f}°"
              f"  Δ {dev:5.2f}°", end="")
        if dev > dev_tol_deg:
            print(f"\n[扫掠中止] {cfg.name} 实测偏离指令 {dev:.1f}° > {dev_tol_deg:.1f}°："
                  f"卡阻？换算方向/零点未标定？力矩未使能？")
            return False

    time.sleep(tick_s)                  # 静置一拍，确认停在起点
    fb_end = ctrl.get_feedback()
    dev_end = abs(float(fb_end.angles_deg[idx]) - start)
    if dev_end <= dev_tol_deg:
        print(f"\n[扫掠通过] {cfg.name} 目标角往返运动正常，实测全程跟随，回到起点偏差 {dev_end:.2f}°")
        return True
    print(f"\n[扫掠未过] 回到起点偏差 {dev_end:.2f}° 超容差")
    return False


# ================================================================ 验收入口
def _demo_main(argv: list[str] | None = None) -> None:
    """python -m qingyun.grabbing.motor_control —— 实时回传验收脚本。"""
    ap = argparse.ArgumentParser(
        description="青云智臂 · 第一步验收：6 路舵机实时角度回传"
    )
    ap.add_argument("--port", default=None, help="串口（默认取 QINGYUN_SERVO_PORT 或 /dev/ttyACM0）")
    ap.add_argument("--hz", type=float, default=30.0, help="打印频率，默认 30")
    ap.add_argument("--no-torque", action="store_true",
                    help="不使能力矩：可手掰舵机验证实测回传")
    ap.add_argument("--once", type=int, default=0, metavar="N",
                    help="只读 N 次后退出（0=持续，默认）")
    ap.add_argument("--sweep", default=None, metavar="JOINT",
                    help="单关节慢速往返扫掠自检（运动链路验收），关节可选 "
                         "pan/lift/elbow/wrF/wrR/grip 或长名")
    ap.add_argument("--amp", type=float, default=30.0, metavar="DEG",
                    help="扫掠单侧幅值°，默认 30（会被软限位截断）")
    ap.add_argument("--rate", type=float, default=15.0, metavar="DEG_S",
                    help="扫掠角速度 °/s，默认 15")
    ap.add_argument("--dev-tol", type=float, default=8.0, metavar="DEG",
                    help="扫掠中 |实测-指令| 中止容差°，默认 8")
    args = ap.parse_args(argv)

    ctrl = STS3215MotorController(port=args.port)
    failures = 0
    try:
        ctrl.connect()
        ctrl.diagnose()
        if not args.no_torque:
            ctrl.enable_torque(True)
            print("[力矩] 已使能 6 舵机（--no-torque 可保持松垮手掰验证）")
        else:
            print("[力矩] 未使能：手掰关节，角度应实时跟随")

        # ---- 运动链路自检：目标角→运动→实测回传（无需 B 的解算）----
        if args.sweep:
            if args.no_torque:
                raise MotorError("--sweep 需要力矩使能（会真实运动），请去掉 --no-torque")
            print("\n[注意] 关节将真实运动！请确保人/物在机械臂范围外，手放物理急停上。")
            time.sleep(1.0)
            ok = sweep_joint(ctrl, args.sweep, amp_deg=args.amp,
                             rate_deg_s=args.rate, dev_tol_deg=args.dev_tol)
            print(f"[扫掠结果] {'通过 ✔：目标角→运动→回传链路 OK' if ok else '未通过 ✘：见上方中止原因'}")
            return

        hdr = "      " + "".join(f"{n:>11}" for n in _SHORT_NAMES)
        print(hdr + "     | |v|max°/s  ΣI mA   (Ctrl+C 退出)")
        t0 = time.monotonic()
        i = 0
        while True:
            try:
                fb = ctrl.get_feedback()
            except MotorError as exc:
                failures += 1
                if failures >= 5:
                    print(f"\n[错误] 连续 {failures} 次读取失败，退出: {exc}")
                    break
                print(f"\r[读取失败 {failures}/5 次] {exc}", end="")
                time.sleep(0.2)
                continue
            failures = 0
            t = time.monotonic() - t0
            body = "".join(
                f"{fb.angles_deg[j]:>11.2f}" for j in range(6)
            )
            vmax = float(np.max(np.abs(fb.speeds_deg_s)))
            isum = float(np.sum(fb.currents_ma))
            print(f"\r[{t:7.2f}s]" + body + f"  |  {vmax:8.2f}  {isum:6.0f}", end="")
            i += 1
            if args.once and i >= args.once:
                print("\n[完成]")
                break
            nxt = t0 + i / args.hz
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[退出]")
    except (MotorError, FeetechError) as exc:
        print(f"\n[错误] {exc}")
    finally:
        ctrl.close()


if __name__ == "__main__":
    _demo_main()
