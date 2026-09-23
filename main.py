"""生产入口：固定配置装配、冷启动、顶层任务循环与共享会话函数（P1-05）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2（文件责任表：本文件 = 固定生产配置装配、冷启动、顶层任务循环（无 CLI，
           D15）、可被 mock 入口复用的会话函数；末段：**生产模块禁止导入 tests**）
        §3.2/§4 步骤 1–11（冷启动顺序**固定**：load_app_config → runlog.init →
           startup → load_motion_params(real) → preflight（初审 L1：**全部检查在
           motor 构造之前**，失败时未连串口、零舵机写入）→ Sts3215MotorController
           （构造即握手，main 不二次 initialize）→ ArmController（should_stop 不接）
           → 冷启动复位 → asr.init → cloud_model.init（R03）→ vision.configure →
           vision.init → ready；三个 init 任一失败不得 ready/进入会话）
        §4 末段（暴露 ``cold_start_recover`` / ``run_session`` 两个共享函数，生产与
           mock 均调用、不复制业务循环；预检与会话接收**同一 registry 对象**；main 的
           import 不执行装配）
        §5（顶层主循环逐字 + 异常映射表：ASRHardError/LLMHardError → terminate；
           VisionHardError 含 plan 外保险 → terminate；SystemExit 直通不包装；
           其余未预期 → terminate("UNEXPECTED", repr)；任务完成不退出、回 WAITING，
           无"跑完即退"分支；不重复打印"任务完成"——统一由 plan._finish 输出，初审 L9）
        §7（startup 事件带 profile/vision_config/prompt 各自路径与哈希——"不得把配置
           文件哈希当作其引用资源的哈希"；preflight_failed/ready/asr_result/llm_result
           字段）
    docs/实施文档/P0_公共契约与迁移技术文档.md §3（VisionThresholds 六字段来源逐一
        从 MotionParams 复制，不做单位换算）
    docs/实施文档/README.md 决策 D13（预检/JSONL/init 入口）、D14（正常完成出口仅余
    NoTarget×3，terminate 语义）、D15（main 无 CLI、固定读 configs/app.local.json；
    mock 走独立入口 scripts/run_mock.py）

设计要点：
    * 装配函数 :func:`assemble_production` 严格按 §4 步骤 1–11 顺序调用**模块级名字**
      （``load_app_config`` / ``runlog`` / ``load_motion_params`` / ``preflight`` /
      ``Sts3215MotorController`` / ``ArmController`` / ``cold_start_recover`` /
      ``asr`` / ``cloud_model`` / ``vision`` / ``shutdown``），测试可对任一名字注入
      探针；步骤 1 之前与 ``runlog.init`` 本身失败时没有正式日志，只 stderr +
      terminate（§4 步骤 1）。
    * 冷启动任意异常（含 KeyboardInterrupt）与 WAITING 阶段 Ctrl-C 都映射统一退出；
      ``terminate`` 的日志/资源关闭不覆盖原失败原因（shutdown §3.3 幂等 + 降级）。
    * 模块 import 零装配副作用：只有 ``if __name__ == "__main__"`` 才启动。
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import numpy as np

from configs.app_config import (
    PRODUCTION_APP_PATH,
    AppConfig,
    PreflightError,
    load_app_config,
    preflight,
)
from configs.common_interface import VisionThresholds
from configs.motion_params import MotionParams, load_motion_params
from qingyun import asr, cloud_model, runlog, shutdown
from qingyun.asr import ASRHardError
from qingyun.cloud_model import LLMHardError
from qingyun.grabbing import vision
from qingyun.grabbing.arm_control import ArmController
from qingyun.grabbing.motor_control import Sts3215MotorController
from qingyun.grabbing.vision import VisionHardError
from qingyun.plans import PLAN_REGISTRY
from qingyun.plans.base import BasePlan

__all__ = [
    "thresholds_from_params",
    "log_startup",
    "cold_start_recover",
    "run_session",
    "assemble_production",
    "main",
]

# 顶层兜底捕获的异常族：SystemExit 必须直通（§5 表），其余 BaseException 一律
# 收敛到 terminate，绝不让栈回溯裸奔到终端。
_CAUGHT_FOR_UNEXPECTED = BaseException


# ---------------------------------------------------------------------------
# 1. P0 六字段阈值快照构造辅助（入口层纯函数；run_mock 复用同一份，P5 §2）
# ---------------------------------------------------------------------------


def thresholds_from_params(params: MotionParams) -> VisionThresholds:
    """按 P0 §3 登记的六字段来源**逐一复制**，不做单位换算。

    ``VisionThresholds.__post_init__`` 负责数组的 float64 归一与深拷贝，本函数
    不重复处理；来源字段与 ``configs/common_interface.VisionThresholds`` docstring
    的映射逐字一致（workspace.target_bounds_m / grasp.object_envelope_m /
    collision.clearance_m / grasp.approach_height_m / workspace.table_z_m /
    workspace.table_flatness_m）。
    """
    return VisionThresholds(
        target_bounds_m=params.workspace.target_bounds_m,
        object_envelope_m=params.grasp.object_envelope_m,
        clearance_m=params.collision.clearance_m,
        approach_height_m=params.grasp.approach_height_m,
        table_z_m=params.workspace.table_z_m,
        table_flatness_m=params.workspace.table_flatness_m,
    )


# ---------------------------------------------------------------------------
# 2. startup 事件（P1 §7：profile/vision_config/prompt 各算**各自文件**的 SHA256，
#    不得把配置文件哈希当作其引用资源的哈希）
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """流式计算单个文件的 SHA256；读不到就抛 OSError，由调用方就地降级。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_or_reason(path: Path) -> str:
    """startup 事件里的哈希字段：文件此刻可能尚不可读（存在性检查属后续步骤），
    读取失败时把原因如实写成字符串——startup 只负责登记，不替后续步骤判生死。
    """
    try:
        return file_sha256(path)
    except OSError as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"


def log_startup(entry: str, cfg: AppConfig, app_config_path: Path) -> None:
    """发一条 startup 事件（P1 §7）：入口、app 配置文件路径、三个被引用资源的
    路径与**各自**文件哈希。由生产装配与 mock 装配共用，避免两处各抄一份。
    """
    runlog.event(
        "startup",
        entry=entry,
        app_config_path=str(app_config_path),
        profile_path=str(cfg.profile_path),
        profile_sha256=_sha256_or_reason(cfg.profile_path),
        vision_config_path=str(cfg.vision_config_path),
        vision_config_sha256=_sha256_or_reason(cfg.vision_config_path),
        prompt_path=str(cfg.prompt_path),
        prompt_sha256=_sha256_or_reason(cfg.prompt_path),
    )


# ---------------------------------------------------------------------------
# 3. 冷启动复位（§4 步骤 7 / 上层文档 §8；仅步骤 7，读 arm.motor / arm.params）
# ---------------------------------------------------------------------------


def cold_start_recover(arm: ArmController) -> None:
    """读一次反馈，与 home 逐轴比较；容差内**零动作**，否则开爪→逐个安全中转位→home。

    判据（§4 步骤 7 伪代码逐字）：``|fb.angles_deg - home| <= settle_position_tol_deg``
    逐轴成立即视为已在家（开机已空爪、场地已清是操作前提，不自动判空、不开爪）。
    回程路线按 ``safe_waypoints_deg`` **列表顺序**（冻结），最后 ``move_joints(home)``。

    本函数不做任何 terminate 映射：任一并发异常（运动原语异常 / wait 失败 /
    KeyboardInterrupt）向上抛，由入口映射 ``terminate("COLD_START_RECOVERY_FAILED")``
    （§4 步骤 7 括注；生产装配与 mock 装配各自完成这层映射）。
    """
    fb = arm.motor.get_feedback()
    home = np.asarray(arm.params.workspace.home_joints_deg, dtype=np.float64)
    tol = np.asarray(arm.params.motion.settle_position_tol_deg, dtype=np.float64)
    within = np.all(np.abs(np.asarray(fb.angles_deg, dtype=np.float64) - home) <= tol)
    if within:
        return
    arm.open_gripper()
    for waypoint in arm.params.workspace.safe_waypoints_deg:
        arm.move_joints(np.asarray(waypoint, dtype=np.float64))
    arm.move_joints(home)


# ---------------------------------------------------------------------------
# 4. 顶层主循环（§5 逐字；生产与 mock 共享，不做装配/复位/任何 init）
# ---------------------------------------------------------------------------


def run_session(
    cfg: AppConfig,
    arm: ArmController,
    *,
    registry: Mapping[str, type[BasePlan]],
) -> NoReturn:
    """§5 主循环：ASR → LLM → 每任务新建 Plan → plan.run() 后回 WAITING。

    * ``cfg`` 按签名接收（会话本体不消费：一切装配属入口，§4 末段）；
    * ``registry`` 必须是与预检**同一个对象**（§4 末段）；生产传 ``PLAN_REGISTRY``，
      mock 入口传自己的注册表，不覆盖生产全局变量；
    * 空转写/``invalid`` 一律 continue（T01 不创建 Plan、不触视觉/抓取）；
    * 有效任务：``task_instance_id`` 在 session 内单调 +1（§3.4），用
      ``runlog.context(task_id=..., task_instance_id=...)`` 包住整个任务（§5）；
    * plan 正常返回 = NoTarget×3 唯一完成出口（D14）；**不在此重复打印"任务完成"**
      （初审 L9，统一由 ``_finish`` 输出）；
    * 异常映射（§5 表）：ASRHardError → ``ASR_HARD_ERROR``；LLMHardError →
      ``LLM_HARD_ERROR``；task ∉ registry → ``LLM_CONTRACT_VIOLATION``（detail 用
      repr）；VisionHardError（plan 侧已兜底，这里是 plan 外保险）→
      ``VISION_HARD_ERROR``；SystemExit 直通不包装；KeyboardInterrupt（WAITING 阶段
      路径，臂在 home）→ terminate；其余未预期 → ``UNEXPECTED`` + repr。
    """
    task_instance_id = 0
    while True:
        try:
            # 一次只处理一个任务：PLAN 运行期间不录音、不提交云请求、不缓存下一任务
            # （§5 末段——同步循环天然保证，无需额外机制）。
            try:
                t_asr0 = time.monotonic()
                text = asr.listen_and_transcribe()  # 桩：P2 替换；mock：脚本替身
            except ASRHardError as exc:
                shutdown.terminate("ASR_HARD_ERROR", f"ASR 硬故障：{exc}")
            duration_asr_s = time.monotonic() - t_asr0
            runlog.event("asr_result", text=text, duration_s=duration_asr_s)
            if not text.strip():
                continue

            try:
                t_llm0 = time.monotonic()
                task = cloud_model.select_plan(text)
            except LLMHardError as exc:
                shutdown.terminate("LLM_HARD_ERROR", f"云模型硬故障：{exc}")
            duration_llm_s = time.monotonic() - t_llm0
            runlog.event(
                "llm_result", text=text, task=task, duration_s=duration_llm_s
            )
            if task == "invalid":
                continue
            if task not in registry:
                shutdown.terminate("LLM_CONTRACT_VIOLATION", repr(task))

            task_instance_id += 1  # session 内单调编号（§3.4）
            with runlog.context(task_id=task, task_instance_id=task_instance_id):
                plan = registry[task](arm)  # 每任务新实例，计数器归零（§5/§6.1）
                plan.run()  # 正常返回 = NoTarget×3；完成日志由 plan._finish 输出
        except SystemExit:
            # §5：terminate 自己发的 exit(1) 直通，不包装、不双重打印。
            raise
        except KeyboardInterrupt:
            # WAITING/ASR 阶段的 Ctrl-C（臂在 home）：统一 terminate（§5 表）。
            # 抓取中的 Ctrl-C 由 arm_control 转 ABORTED 走分发表、恢复 input() 中的
            # Ctrl-C 由 plan 转 MANUAL_CONFIRM_ABORTED——都不到这里（P1-04 已测）。
            shutdown.terminate(
                "KEYBOARD_INTERRUPT", "会话 WAITING 阶段收到 Ctrl-C（臂在 home），统一退出"
            )
        except VisionHardError as exc:
            # plan 内 get_target 的 VisionHardError 由 plan 自己映射；能冒到这里的
            # 属 plan 外保险（如 terminate 前读取路径上的异常实现），同码退出。
            shutdown.terminate("VISION_HARD_ERROR", f"视觉管线故障（plan 外保险）：{exc}")
        except _CAUGHT_FOR_UNEXPECTED as exc:  # noqa: BLE001 - §5"其余未预期异常"兜底
            shutdown.terminate("UNEXPECTED", repr(exc))


# ---------------------------------------------------------------------------
# 5. 生产装配（§4 步骤 1–11，顺序固定；每个名字都是模块级引用，测试可注入探针）
# ---------------------------------------------------------------------------


def _abort_before_log(status: str, detail: object) -> NoReturn:
    """还没有正式日志时的退出（§4 步骤 1/2）：stderr 一行 + terminate。

    ``terminate`` 自己的事件日志失败会降级为 stderr（shutdown §3.4），不会递归。
    """
    print(f"[main] {status}: {detail}", file=sys.stderr, flush=True)
    shutdown.terminate(status, str(detail))


def _safe_failure_event(kind: str, **fields: object) -> None:
    """尽力发一条启动阶段事件；写失败只 stderr 告警，绝不掩盖随后的 terminate 原因。"""
    try:
        runlog.event(kind, **fields)
    except Exception as exc:  # pragma: no cover - 日志坏在极端环境
        print(f"[main][WARN] {kind} 事件写入失败（{exc}）", file=sys.stderr, flush=True)


def assemble_production() -> tuple[AppConfig, MotionParams, ArmController]:
    """按 §4 步骤 1–11 完成冷启动装配并发送 ready；返回 (cfg, params, arm)。

    任一步失败都在此处映射 terminate（唯一退出出口），绝不进入会话；
    三 init（asr/cloud/vision）全部成功后才 ready（R03）。
    """
    # 1. 固定生产配置（失败时尚无正式日志：仅 stderr + terminate）
    try:
        cfg = load_app_config(PRODUCTION_APP_PATH, entry="production")
    except Exception as exc:  # AppConfigError 及其 IO 起因
        _abort_before_log("APP_CONFIG_INVALID", exc)

    # 2. 正式日志
    try:
        runlog.init(cfg.log_dir)
    except Exception as exc:  # 目录不可建/不可写等
        _abort_before_log("RUNLOG_INIT_FAILED", exc)

    # 3. startup 事件（入口=production；引用资源各算各的文件哈希）
    log_startup("production", cfg, PRODUCTION_APP_PATH)

    # 4. 运动参数（real 模式要求 verified，加载器已保证——§3.2 第 7 条不重复）
    try:
        params = load_motion_params(cfg.profile_path, mode="real")
    except Exception as exc:
        _safe_failure_event("preflight_failed", stage="load_motion_params",
                            reason=str(exc))
        shutdown.terminate("MOTION_PARAMS_INVALID", str(exc))

    # 5. 启动预检（初审 L1：必须在步骤 6 之前——失败时未连串口、零舵机写入）
    try:
        preflight(cfg, params, PLAN_REGISTRY)
    except PreflightError as exc:
        _safe_failure_event("preflight_failed", check=exc.check, reason=exc.message)
        shutdown.terminate("PREFLIGHT_FAILED", f"preflight.{exc.check}: {exc.message}")
    except Exception as exc:
        _safe_failure_event("preflight_failed", stage="preflight", reason=str(exc))
        shutdown.terminate("PREFLIGHT_FAILED", str(exc))

    # 6. 驱动：构造即连接 + §4.1 握手（main 不二次 initialize）；异常/Ctrl-C 都属
    #    启动事务失败（驱动内部已完成有界清理），映射 MOTOR_INIT_FAILED。
    try:
        motor = Sts3215MotorController(params)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 含 KeyboardInterrupt（§4.1 第 5 步）
        shutdown.terminate("MOTOR_INIT_FAILED", repr(exc))

    # 7. 臂控制器：should_stop 不接（Ctrl-C 走 KeyboardInterrupt 现成路径，§1）
    arm = ArmController(motor, params)

    # 8. 冷启动复位（§4 步骤 7 本体；任一异常/Ctrl-C → COLD_START_RECOVERY_FAILED）
    try:
        cold_start_recover(arm)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        shutdown.terminate("COLD_START_RECOVERY_FAILED", repr(exc))

    # 9–10. 三 init：任一失败不得 ready、不得进入会话（R03，§4 末段）
    try:
        asr.init(cfg)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - ASRHardError 及装配期兜底
        shutdown.terminate("ASR_INIT_FAILED", repr(exc))
    try:
        cloud_model.init(cfg)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        shutdown.terminate("LLM_INIT_FAILED", repr(exc))
    try:
        vision.configure(thresholds_from_params(params), cfg.vision_config_path)
        vision.init()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - VisionHardError；init 故障不得伪装 NoTarget
        shutdown.terminate("VISION_INIT_FAILED", repr(exc))

    # 11. ready
    runlog.event("ready")
    runlog.console("系统就绪，等待语音指令")
    return cfg, params, arm


def main() -> NoReturn:
    """生产入口：装配（§4 步骤 1–11）→ 会话（§5）。无 CLI、固定配置（D15）。"""
    cfg, _params, arm = assemble_production()
    run_session(cfg, arm, registry=PLAN_REGISTRY)


if __name__ == "__main__":
    main()
