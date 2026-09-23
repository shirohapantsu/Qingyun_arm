"""JSONL 详细日志 + 终端简洁输出 + 同步日志上下文（P1-02）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2（文件责任表：本模块 = JSONL 详细日志 + 终端简洁输出）
        §3.4（公开签名 init/event/console/context；单线程同步无锁；ndarray 先
           tolist()；JSON 禁止 NaN/Infinity；关联键
           (session_id, task_instance_id, scan_id, attempt_id)；
           "单测可注入内存日志接收器"）
        §3.4 末段（日志失败抛异常并交顶层 terminate；terminate 的日志再次失败
           时仅向 stderr 输出，仍退出，不能递归调用自己）
        §5（主循环用 ``with runlog.context(task_id=..., task_instance_id=...)``
           包住任务；scan_id/attempt_id 由 plan 以同样的上下文携带）
        §7（日志事件最低字段——本模块只提供机制，事件由各后续模块产生）
        §8 步骤 2
    docs/实施文档/README.md 决策 D13（JSONL 详细日志纳入）

设计要点：
    * **一次进程一个 runlog**：``init`` 只允许成功一次，重复 init 抛
      ``RuntimeError``。这与 P1 §4 冷启动"步骤 2 建日志、进程活到 terminate
      为止"的形态一致；测试需要重新初始化时显式调用 ``reset_for_tests()``。
    * 未 init（或已 close 且未注入 sink）时调用 ``event``/``console`` 抛
      ``RuntimeError``——terminate 侧捕获它并降级为 stderr 输出（P1 §3.4 末段）。
      ``context``/``snapshot_context`` 是纯内存关联状态，不要求 init。
    * ``event`` 先整体序列化成一行 JSON 字符串、**成功后才落盘**：非有限值
      （NaN/Infinity）经 ``json.dumps(allow_nan=False)`` 明确拒绝并抛
      ``ValueError``，由顶层映射 terminate；不会写出半行非法 JSON。
    * ``set_sink`` 是 §3.4"单测可注入内存日志接收器"的注入点：sink 收到的是
      与落盘完全相同的 JSON 行（不含换行符），设置后事件**只进 sink 不写文件**，
      传 ``None`` 恢复文件输出。
    * 单线程同步，无锁；每条事件写入后立即 ``flush()``，进程被 kill 前也已有行。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np

__all__ = [
    "init",
    "event",
    "console",
    "context",
    "snapshot_context",
    "current_session_id",
    "current_log_path",
    "set_sink",
    "close",
    "reset_for_tests",
]

# 事件记录里由 runlog 自动生成的三个保留键：调用方（fields 或 context）不得
# 占用，否则等于伪造会话/时间戳，一律 ValueError 拒绝。
RESERVED_KEYS = ("kind", "ts", "session_id")

# --- 模块级状态（单线程同步，无锁；P1 §3.4） ---

_session_id: Optional[str] = None
_log_path: Optional[Path] = None
_file: Optional[IO[str]] = None
_context: dict[str, Any] = {}
_sink: Optional[Callable[[str], None]] = None

_MISSING = object()


def _utc_now_iso() -> str:
    """事件时间戳：UTC、毫秒精度、Z 后缀（与文件名同一时间基准）。"""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _normalize(value: Any) -> Any:
    """把 ndarray 递归转成可 JSON 序列化的原生结构（P1 §3.4"ndarray 先 tolist()"）。

    只处理 ndarray / Mapping / list / tuple 三种容器；其余原样交给 json.dumps，
    非可序列化类型会得到 TypeError——同样在落盘前抛出，不产生半行 JSON。
    数值是否为有限值由 ``json.dumps(allow_nan=False)`` 把关，这里不做转换。
    """
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _validate_field_names(where: str, keys) -> None:
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValueError(f"runlog.{where}: 字段名必须是非空字符串，收到 {key!r}")
        if key in RESERVED_KEYS:
            raise ValueError(
                f"runlog.{where}: 字段名 {key!r} 是保留键（自动附加），不允许调用方提供"
            )


def init(log_dir: Path) -> str:
    """创建 ``<log_dir>/run-<UTC时间戳>-<pid>.jsonl`` 并返回 session_id。

    session_id 取文件主名（``run-YYYYMMDDTHHMMSSZ-<pid>``），文件名与会话一一对应。
    目录不存在则 parents 创建；文件以追加模式打开。

    **一次进程一个 runlog**：重复调用抛 ``RuntimeError``（冷启动 §4 步骤 2 只会
    调用一次，二次 init 属编程错误；测试请走 ``reset_for_tests()``）。
    """
    global _session_id, _log_path, _file
    if _session_id is not None:
        raise RuntimeError(
            f"runlog 已初始化（session_id={_session_id!r}）：一个进程只允许一次 init；"
            "测试复用请 runlog.reset_for_tests()"
        )
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"run-{stamp}-{os.getpid()}.jsonl"
    path = directory / name
    handle = path.open("a", encoding="utf-8")
    _session_id = Path(name).stem
    _log_path = path
    _file = handle
    return _session_id


def event(kind: str, **fields: Any) -> None:
    """追加一行 JSON 事件并立即 flush。

    自动附加 ``kind`` / ``ts``（UTC ISO 毫秒）/ ``session_id``，随后合入当前
    ``context`` 的关联字段（task_id/task_instance_id/scan_id/attempt_id 等），
    最后合入显式 fields（fields 覆盖同名 context 键）。

    失败语义（P1 §3.4 末段）：未 init / 已 close → ``RuntimeError``；
    保留键被占用、字段名非法 → ``ValueError``；含 NaN/Infinity 或非可序列化
    值 → ``ValueError`` / ``TypeError``。全部在写盘之前抛出，日志文件不会
    出现半行；调用方（顶层 main / terminate）负责映射处置。
    """
    if _session_id is None:
        raise RuntimeError("runlog 未初始化：先 init(log_dir) 再调用 event")
    if _sink is None and _file is None:
        raise RuntimeError(f"runlog 已关闭（session_id={_session_id!r}），无法写事件")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"runlog.event: kind 必须是非空字符串，收到 {kind!r}")
    _validate_field_names("event", fields)

    record: dict[str, Any] = {
        "kind": kind,
        "ts": _utc_now_iso(),
        "session_id": _session_id,
    }
    for key, value in _context.items():
        record[key] = _normalize(value)
    for key, value in fields.items():
        record[key] = _normalize(value)

    # allow_nan=False：NaN/Infinity 明确拒绝（抛 ValueError），不做安全转换；
    # 先序列化成功再落盘，保证文件里逐行都是合法 JSON。
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if _sink is not None:
        _sink(line)
    else:
        assert _file is not None  # 上方守卫已保证
        _file.write(line + "\n")
        _file.flush()


def console(msg: str) -> None:
    """终端简洁输出一行（不进 JSONL；需要留档由调用方另发 event）。

    与 event 同样要求已 init（未 init 抛 RuntimeError，terminate 侧捕获）；
    输出即时 flush，便于操作者顺序看到进度。
    """
    if _session_id is None:
        raise RuntimeError("runlog 未初始化：先 init(log_dir) 再调用 console")
    print(msg, flush=True)


@contextlib.contextmanager
def context(**fields: Any) -> Iterator[None]:
    """同步日志上下文：进入时合并字段，退出时恢复旧值（可嵌套）。

    同一键嵌套进入时覆盖外层值，退出后恢复外层值；外层没有该键则退出后删除。
    期间所有 ``event`` 自动携带这些关联字段（P1 §3.4：日志关联键为
    (session_id, task_instance_id, scan_id, attempt_id)）。
    纯内存状态，不要求 init；字段名不得占用保留键。
    """
    _validate_field_names("context", fields)
    saved = {key: _context.get(key, _MISSING) for key in fields}
    _context.update(fields)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is _MISSING:
                _context.pop(key, None)
            else:
                _context[key] = old


def snapshot_context() -> dict[str, Any]:
    """当前上下文的浅拷贝（terminate 用它打印/记录"所处阶段"；不要求 init）。"""
    return dict(_context)


def current_session_id() -> Optional[str]:
    """当前 session_id；未 init 时为 None。"""
    return _session_id


def current_log_path() -> Optional[Path]:
    """当前 JSONL 路径；未 init 时为 None。"""
    return _log_path


def set_sink(sink: Optional[Callable[[str], None]]) -> None:
    """注入/恢复内存日志接收器（P1 §3.4"单测可注入内存日志接收器"）。

    ``sink(line)`` 收到与落盘逐字节相同的 JSON 行（不含换行符）；设置后事件
    **只进 sink、不写文件**；``set_sink(None)`` 恢复默认的文件输出。
    """
    global _sink
    if sink is not None and not callable(sink):
        raise TypeError(f"runlog.set_sink: sink 必须可调用或 None，收到 {type(sink).__name__}")
    _sink = sink


def close() -> None:
    """flush 并关闭日志文件句柄（幂等；atexit 钩子调用，不发任何设备指令）。

    关闭后 session_id 保留（可审计），再写 event 抛 RuntimeError（除非注入了
    sink）。关闭错误只向 stderr 告警，不向外抛。
    """
    global _file
    handle, _file = _file, None
    if handle is None:
        return
    try:
        handle.close()
    except OSError as exc:
        print(f"[runlog][WARN] 关闭日志文件失败: {exc}", file=sys.stderr, flush=True)


def reset_for_tests() -> None:
    """仅供测试：关闭句柄并清空 session/上下文/sink，回到未初始化状态。

    生产代码不得调用；它存在的意义是让同一 pytest 进程里的每个用例都能
    拿到干净的 runlog 状态（一次进程一个 runlog 是生产语义，不是测试语义）。
    """
    global _session_id, _log_path, _file, _sink
    close()
    _session_id = None
    _log_path = None
    _sink = None
    _context.clear()
