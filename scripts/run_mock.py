#!/usr/bin/env python
"""独立 mock 入口（D15）：固定读 configs/app.mock.json，离线组装全链并复用生产会话。

规范来源：
    docs/实施文档/P5_集成测试与部署验收技术文档.md §2（装配契约逐字：
        load_app_config(mock) → runlog.init → startup → load_motion_params(mock) →
        preflight(MOCK_REGISTRY) → SimTime/MockMotorController → **显式
        motor.initialize() 一次**（mock 构造不自动初始化）→ ArmController(
        clock/sleep/clock_ns 全接 SimTime，R04 统一模拟时钟) → install_stubs →
        cold_start_recover → vision.configure(thresholds_from_params(params),
        cfg.vision_config_path) → vision.init() → ready → run_session(cfg, arm,
        registry=MOCK_REGISTRY)）
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md §2 末段（本入口是"生产模块
        禁止导入 tests"的**唯一豁免**：scripts 非生产运行链，可导入
        tests/mock_motor.py 的 MockMotorController/SimTime）、§4 末段/§5（复用共享
        函数，不复制业务循环）、§3.2（与生产**同一** preflight 函数、同一
        MOCK_REGISTRY 对象贯穿实例化）
    docs/实施文档/README.md 决策 D15（无 CLI、独立 mock 入口、跳过真实 asr/
    cloud_model init）、D14；复审 R04（统一模拟时钟/仿真 K）

边界（DEC-002 已定，P1-05 落实）：
    * vision **不自动回退替身**：装配默认引用真实 ``qingyun.grabbing.vision`` 桩——
      直接 ``python scripts/run_mock.py`` 时 ``vision.configure`` 抛 VisionHardError
      → ``terminate("VISION_INIT_FAILED")``，这是契约正确行为（P3 交付前 mock 全链
      到视觉为止）。pytest（P5 E2E / 本阶段 T22）经 ``assemble(..., vision=FakeVision
      实例)`` 注入替身复用**同一**装配函数与生产 ``run_session``，不复制业务循环。
    * ``install_stubs`` 只替换 asr/cloud_model 模块的**业务函数**（模块属性赋值），
      **不调用真实 init**（D15：跳过 VAD/麦克风/密钥加载）；重复调用拒绝。
    * attempts 由入口层 ``ScenarioArmController`` 消费：每次真实 grasp_and_place 前
      只设置 MockMotor 故障开关（present→object_gap_m=float(mean(
      contact_gap_range_m))；absent→None），随后原样 super() 调用、不替换返回结果；
      耗尽时拒绝场景（terminate，不默认成功）。
    * 生产 main 永不 import 本模块与 tests；本文件被 import 时零装配副作用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

# scripts/ 不是项目根的子包目录形态：直接 `python scripts/run_mock.py` 时
# sys.path[0] 是 scripts/，必须先把项目根插进来才能 import configs/qingyun/main/tests。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.app_config import (  # noqa: E402 - sys.path 引导之后才能 import
    MOCK_APP_PATH,
    AppConfig,
    PreflightError,
    load_app_config,
    preflight,
)
from configs.motion_params import MotionParams, load_motion_params  # noqa: E402
from main import (  # noqa: E402 - 生产共享函数：复用会话与复位/阈值辅助，不复制循环
    cold_start_recover,
    log_startup,
    run_session,
    thresholds_from_params,
)
from qingyun import asr as asr_module  # noqa: E402 - 替身替换的目标模块
from qingyun import cloud_model as cloud_module  # noqa: E402
from qingyun import runlog, shutdown  # noqa: E402
from qingyun.grabbing import vision as vision_stub  # noqa: E402 - 默认真实桩（DEC-002）
from qingyun.grabbing.arm_control import ArmController  # noqa: E402
from qingyun.plans.strawberry import StrawberryPlan  # noqa: E402
from tests.mock_motor import MockMotorController, SimTime  # noqa: E402 - D15 豁免

__all__ = [
    "MockStrawberryPlan",
    "MOCK_REGISTRY",
    "ScenarioArmController",
    "load_scenario",
    "ScenarioStubs",
    "install_stubs",
    "reset_stubs_for_tests",
    "assemble",
    "main",
]

# ---------------------------------------------------------------------------
# 1. MOCK_REGISTRY：注入"非实测"名义 K 的 Plan 变体（P5 §2，不改生产类）
# ---------------------------------------------------------------------------

# 名义 K（**非实测**，P4 回填实测值后才谈生产可用）：与 tests/support.make_target
# 的名义尺寸同源（0.05×0.024 m²），让 mock 场景的 A/B 分级可判定。
MOCK_AREA_THRESHOLD_M2: float = 0.05 * 0.024


class MockStrawberryPlan(StrawberryPlan):
    """草莓 Plan 的 mock 变体：仅覆盖 ``AREA_THRESHOLD_M2`` 为**名义值（非实测）**。

    生产 ``StrawberryPlan.AREA_THRESHOLD_M2`` 保持 None（预检拒绝），P4 回填路径
    不受影响；预检（§3.2 第 3 条）在本类上通过，使 mock 全链不依赖 P4。
    除 K 外不覆盖任何行为——业务循环与生产 Plan 完全同一实现。
    """

    AREA_THRESHOLD_M2 = MOCK_AREA_THRESHOLD_M2  # 名义值（非实测，R04）


# 模块级常量：与生产 PLAN_REGISTRY 平起平坐但互不覆盖（§4 末段）。
# 预检与会话必须拿到**同一个对象**（assemble/main 全程只引用本名字）。
MOCK_REGISTRY: dict[str, type[MockStrawberryPlan]] = {"strawberry": MockStrawberryPlan}

# scenario 耗尽后替身的空转节拍：按**真实墙钟**小段 sleep 再返回空串（P5 §2——
# "阻塞等 Ctrl-C、可用小段 sleep，不持续空串忙循环"）。
IDLE_POLL_S = 0.2

# ---------------------------------------------------------------------------
# 2. 场景资产（P5 §2 形态：{"utterances": [...], "attempts": [{"contact": ...}]}）
# ---------------------------------------------------------------------------


class ScenarioError(ValueError):
    """scenario.json 形态不合法（入口层资产错误，映射 terminate）。"""


def load_scenario(cfg: AppConfig) -> dict[str, Any]:
    """读取 ``cfg.mock.asr_script_path`` 的场景文件并做形态校验。

    只接受顶层对象含 ``utterances``（字符串列表）与 ``attempts``（每项
    ``{"contact": "present"|"absent"}``）；``_comment`` 为资产自述，允许但不消费。
    utterances 耗尽后的行为与 attempts 耗尽后的拒绝分别由替身/包装层实现。
    """
    if cfg.mock is None:
        raise ScenarioError("cfg.mock 缺失：mock 入口必须使用 app.mock.json")
    path = cfg.mock.asr_script_path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ScenarioError(f"场景文件不可读/非法 JSON：{path}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"场景文件顶层应为对象：{path}")
    unknown = set(raw) - {"utterances", "attempts", "_comment"}
    if unknown:
        raise ScenarioError(f"场景文件出现未知键 {sorted(unknown)}：{path}")
    utterances = raw.get("utterances")
    if not isinstance(utterances, list) or any(
        not isinstance(u, str) for u in utterances
    ):
        raise ScenarioError(f"utterances 应为字符串列表：{path}")
    attempts = raw.get("attempts")
    if not isinstance(attempts, list):
        raise ScenarioError(f"attempts 应为列表：{path}")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) != {"contact"}:
            raise ScenarioError(
                f"attempts[{index}] 应为恰含 contact 的对象：{attempt!r}"
            )
        if attempt["contact"] not in ("present", "absent"):
            raise ScenarioError(
                f"attempts[{index}].contact 只能是 present/absent，实际 {attempt['contact']!r}"
            )
    return {"utterances": list(utterances), "attempts": list(attempts)}


# ---------------------------------------------------------------------------
# 3. install_stubs：替换业务函数（模块属性赋值），**不调真实 init**（D15）
# ---------------------------------------------------------------------------


class ScenarioStubs:
    """asr/cloud_model 业务函数的脚本化替身（P5 §2 语义，零网络零麦克风）。

    * ``listen_and_transcribe``：utterances 逐次返回；耗尽后按真实墙钟小段 sleep
      返回空串（模拟静音、等待 Ctrl-C；不持续空串忙循环）。
    * ``select_plan``：本地确定性映射——文本含"草莓"→ ``strawberry``，其余→
      ``invalid``；零网络。
    """

    def __init__(self, utterances: list[str]) -> None:
        self._utterances = list(utterances)
        self.listen_calls = 0

    def listen_and_transcribe(self) -> str:
        self.listen_calls += 1
        if self._utterances:
            return self._utterances.pop(0)
        time.sleep(IDLE_POLL_S)
        return ""

    @staticmethod
    def select_plan(text: str) -> str:
        # 与生产 P2 契约同值域：本期合法 task 仅 strawberry/invalid（D01/D07）。
        return "strawberry" if "草莓" in text else "invalid"


_STUBS_ORIGINALS: dict[str, Any] = {}
_stubs_installed = False


def install_stubs(cfg: AppConfig) -> ScenarioStubs:
    """把场景替身以**模块属性赋值**装到 qingyun.asr / qingyun.cloud_model 上。

    重复调用拒绝（P5 §2"install_stubs 重复调用拒绝"）；不调用任何真实 init——
    D15：mock 跳过 VAD/麦克风/密钥加载，业务函数由本替身接管。
    """
    global _stubs_installed
    if _stubs_installed:
        raise RuntimeError("install_stubs 重复调用被拒绝：替身只允许装配一次（P5 §2）")
    scenario = load_scenario(cfg)
    stubs = ScenarioStubs(scenario["utterances"])
    # 保存原函数以便 reset（生产桩语义"调用即抛"必须在测试后回到原状）。
    _STUBS_ORIGINALS.setdefault(
        "listen_and_transcribe", asr_module.listen_and_transcribe
    )
    _STUBS_ORIGINALS.setdefault("select_plan", cloud_module.select_plan)
    asr_module.listen_and_transcribe = stubs.listen_and_transcribe
    cloud_module.select_plan = stubs.select_plan
    _stubs_installed = True
    return stubs


def reset_stubs_for_tests() -> None:
    """仅供测试：恢复两个模块的业务函数并解禁重复调用守卫（生产不得调用）。"""
    global _stubs_installed
    if "listen_and_transcribe" in _STUBS_ORIGINALS:
        asr_module.listen_and_transcribe = _STUBS_ORIGINALS["listen_and_transcribe"]
    if "select_plan" in _STUBS_ORIGINALS:
        cloud_module.select_plan = _STUBS_ORIGINALS["select_plan"]
    _STUBS_ORIGINALS.clear()
    _stubs_installed = False


# ---------------------------------------------------------------------------
# 4. attempts 消费：入口层 ArmController 包装（只动故障开关，不碰返回结果）
# ---------------------------------------------------------------------------


class ScenarioArmController(ArmController):
    """在每次**真实** grasp_and_place 前设置 MockMotor 故障开关，然后原样 super()。

    P5 §2：present → ``motor.object_gap_m = float(mean(contact_gap_range_m))``
    （物体堵住标定接触开口）；absent → ``None``（空夹可一路闭合）。沿用 MockMotor
    的真实字段，不发明 object_present 开关；**不替换返回结果**、不改运动实现。
    attempts 耗尽时拒绝场景（terminate），绝不默认成功。
    """

    def __init__(self, motor, params, *, attempts: list[dict[str, str]], **kwargs) -> None:
        super().__init__(motor, params, **kwargs)
        self.scenario_attempts: list[dict[str, str]] = list(attempts)
        self.attempt_order: list[str] = []

    def grasp_and_place(self, target, place_id: str = "default"):
        if not self.scenario_attempts:
            shutdown.terminate(
                "SCENARIO_ATTEMPTS_EXHAUSTED",
                "场景 attempts 已耗尽，仍被要求新的真实抓取——拒绝场景，不默认成功"
                f"（place_id={place_id!r}）",
            )
        attempt = self.scenario_attempts.pop(0)
        contact = attempt["contact"]
        if contact == "present":
            self.motor.object_gap_m = float(
                np.mean(self.params.gripper.contact_gap_range_m)
            )
        else:  # absent
            self.motor.object_gap_m = None
        self.attempt_order.append(contact)
        return super().grasp_and_place(target, place_id)


# ---------------------------------------------------------------------------
# 5. 装配（P5 §2 顺序固定；返回四件套供 pytest 复用）
# ---------------------------------------------------------------------------


def _abort_before_log(status: str, detail: object) -> NoReturn:
    print(f"[run_mock] {status}: {detail}", file=sys.stderr, flush=True)
    shutdown.terminate(status, str(detail))


def _safe_failure_event(kind: str, **fields: object) -> None:
    try:
        runlog.event(kind, **fields)
    except Exception as exc:  # pragma: no cover - 日志坏在极端环境
        print(f"[run_mock][WARN] {kind} 事件写入失败（{exc}）", file=sys.stderr, flush=True)


def assemble(
    cfg: AppConfig | None = None,
    *,
    vision: Any = None,
) -> tuple[AppConfig, MotionParams, MockMotorController, ScenarioArmController]:
    """按 P5 §2 装配契约顺序组装 mock 栈并发送 ready。

    返回 ``(cfg, params, motor, arm)``；``vision`` 缺省为**真实桩模块**（DEC-002：
    直接跑 main() 时在 vision.configure 处按契约终止，不静默回退替身），pytest 传入
    ``tests.support_upper.FakeVision`` 实例复用同一装配。预检与会话使用**同一个**
    ``MOCK_REGISTRY`` 对象；本函数不调用任何真实 asr/cloud init（D15）。
    """
    # 1. 固定 mock 配置
    try:
        if cfg is None:
            cfg = load_app_config(MOCK_APP_PATH, entry="mock")
    except Exception as exc:
        _abort_before_log("APP_CONFIG_INVALID", exc)

    # 2. 正式日志 + 3. startup（入口=mock；资源各算各的哈希，共用 main 的辅助）
    try:
        runlog.init(cfg.log_dir)
    except Exception as exc:
        _abort_before_log("RUNLOG_INIT_FAILED", exc)
    log_startup("mock", cfg, MOCK_APP_PATH)

    # 4. simulation profile 副本，mode="mock"
    try:
        params = load_motion_params(cfg.profile_path, mode="mock")
    except Exception as exc:
        _safe_failure_event("preflight_failed", stage="load_motion_params",
                            reason=str(exc))
        shutdown.terminate("MOTION_PARAMS_INVALID", str(exc))

    # 5. 与生产同一 preflight 函数、同一 MOCK_REGISTRY 对象（贯穿到实例化）
    try:
        preflight(cfg, params, MOCK_REGISTRY)
    except PreflightError as exc:
        _safe_failure_event("preflight_failed", check=exc.check, reason=exc.message)
        shutdown.terminate("PREFLIGHT_FAILED", f"preflight.{exc.check}: {exc.message}")
    except Exception as exc:
        _safe_failure_event("preflight_failed", stage="preflight", reason=str(exc))
        shutdown.terminate("PREFLIGHT_FAILED", str(exc))

    # 6. SimTime + MockMotor；mock 构造不自动初始化 → **显式一次**（P5 §2）
    sim_time = SimTime()
    motor = MockMotorController(params, sim_time)
    motor.initialize()

    # 7. R04：clock/sleep/clock_ns 全接 SimTime——默认真实时钟会把仿真反馈判过期
    try:
        scenario = load_scenario(cfg)
    except ScenarioError as exc:
        shutdown.terminate("MOCK_SCENARIO_INVALID", str(exc))
    arm = ScenarioArmController(
        motor,
        params,
        attempts=scenario["attempts"],
        clock=sim_time.monotonic,
        sleep=sim_time.sleep,
        clock_ns=sim_time.monotonic_ns,
    )

    # 8. 替身接管业务函数（不调真实 init，D15）
    try:
        install_stubs(cfg)
    except ScenarioError as exc:
        shutdown.terminate("MOCK_SCENARIO_INVALID", str(exc))
    except RuntimeError as exc:  # 重复安装守卫
        shutdown.terminate("MOCK_STUBS_REINSTALL", str(exc))

    # 9. 冷启动复位（与生产共享实现；异常 → COLD_START_RECOVERY_FAILED）
    try:
        cold_start_recover(arm)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        shutdown.terminate("COLD_START_RECOVERY_FAILED", repr(exc))

    # 10. vision：默认真实桩（调用即 VisionHardError → VISION_INIT_FAILED，契约
    #     正确行为）；注入 FakeVision 时记录 configure/init 后继续（DEC-002）。
    vision_module = vision if vision is not None else vision_stub
    try:
        vision_module.configure(thresholds_from_params(params), cfg.vision_config_path)
        vision_module.init()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - VisionHardError/装配期兜底；故障不得伪装 NoTarget
        shutdown.terminate("VISION_INIT_FAILED", repr(exc))

    # 11. ready（mock 无三 init：asr/cloud 由替身接管，D15；只待 vision 成功）
    runlog.event("ready")
    runlog.console("系统就绪（mock），等待语音指令")
    return cfg, params, motor, arm


def main() -> NoReturn:
    """mock 冒烟入口：装配（P5 §2 契约）→ 生产同一 run_session(MOCK_REGISTRY)。"""
    cfg, _params, _motor, arm = assemble()
    run_session(cfg, arm, registry=MOCK_REGISTRY)


if __name__ == "__main__":
    main()
