"""P3 视觉入口：已替换契约桩，文件名与公开签名不得改变。

configure → init → get_target；相机、RKNN 和 OpenCV 仅在 init 加载。
内部管线见 _vision_pipeline.py；公共接口以本地 common_interface 为准。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.common_interface import VisionInterface, VisionThresholds

__all__ = ["NoTarget", "VisionHardError", "configure", "init", "get_target"]


class NoTarget(Exception):
    """本次扫描无合法目标，异常消息为业务原因码。"""


class VisionHardError(Exception):
    """配置、设备、帧或推理故障，不能作为空场景处理。"""


_thresholds = None
_config = None
_config_path = None
_pipeline = None


def configure(thresholds: VisionThresholds, config_path: Path) -> None:
    """保存阈值快照，路径相对配置文件解析；只允许调用一次。"""
    global _thresholds, _config, _config_path
    if _config is not None:
        raise VisionHardError("configure: 重复调用或 init 后调用")
    try:
        from configs.common_interface import VisionThresholds
        if not isinstance(thresholds, VisionThresholds):
            raise ValueError("thresholds 必须是 VisionThresholds")
        path = Path(config_path).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("配置顶层必须是对象")
        snapshot = copy.deepcopy(thresholds)
    except Exception as exc:
        raise VisionHardError(f"configure: {exc}") from exc
    _thresholds, _config, _config_path = snapshot, config, path


def init() -> None:
    """加载并验证模型、标定和首帧；回放首帧只验证、不消费。"""
    global _pipeline
    if _config is None:
        raise VisionHardError("init: 尚未 configure")
    if _pipeline is not None:
        raise VisionHardError("init: 不允许重复初始化")
    try:
        from qingyun.grabbing._vision_pipeline import Pipeline
        pipeline = Pipeline(_thresholds, _config, _config_path)
        pipeline.initialize()
    except Exception as exc:
        if isinstance(exc, VisionHardError):
            raise
        raise VisionHardError(f"init: {exc}") from exc
    _pipeline = pipeline


def get_target(ignore: int = 0) -> VisionInterface:
    """重新扫描，返回确定性排序第 ignore+1 个合法候选。"""
    from qingyun import runlog
    audit = {"ignore": ignore if type(ignore) is int else repr(ignore),
             "valid_count": 0, "candidates": []}
    try:
        if _pipeline is None:
            raise VisionHardError("get_target: 尚未 init")
        if type(ignore) is not int or ignore < 0:
            raise VisionHardError("get_target: ignore 必须是非负 int")
        target = _pipeline.scan(ignore, audit)
        audit["outcome"] = "target"
        return target
    except NoTarget as exc:
        audit.update(outcome="no_target", reason=str(exc))
        raise
    except Exception as exc:
        audit.update(outcome="hard_error", reason=str(exc))
        if isinstance(exc, VisionHardError):
            raise
        raise VisionHardError(f"get_target: {exc}") from exc
    finally:
        runlog.event("vision_scan", **audit)


def _close() -> None:
    """幂等设备清理；用于进程退出和测试隔离。"""
    if _pipeline is not None:
        _pipeline.close()


def _reset_for_tests() -> None:
    global _thresholds, _config, _config_path, _pipeline
    _close()
    _thresholds = _config = _config_path = _pipeline = None
