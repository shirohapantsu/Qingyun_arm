"""qingyun/runlog.py 的离线单测（P1-02）。

覆盖：init 文件名（UTC+pid）与 session_id、重复 init 拒绝、未 init 的
event/console 行为、事件行逐行合法 JSON 与自动字段、立即 flush、ndarray
tolist、NaN/Infinity 明确拒绝且不落半行、context 嵌套覆盖与恢复、
fields 覆盖 context、保留键拒绝、set_sink 内存注入点、close/幂等、
reset_for_tests。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from qingyun import runlog

# run-<UTC YYYYMMDDTHHMMSSZ>-<pid>.jsonl
NAME_RE = re.compile(r"^run-(\d{8}T\d{6}Z)-(\d+)\.jsonl$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


@pytest.fixture(autouse=True)
def clean_state():
    """每个用例前后把模块级状态复位（一次进程一个 runlog 是生产语义）。"""
    runlog.reset_for_tests()
    yield
    runlog.reset_for_tests()


def raw_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return text.splitlines()


def records(path: Path) -> list[dict]:
    """逐行 json.loads——任何半行/非法 JSON 都会在这里炸。"""
    return [json.loads(line) for line in raw_lines(path)]


# --- init ------------------------------------------------------------------


def test_init创建UTC带pid命名的jsonl并返回session_id(tmp_path):
    log_dir = tmp_path / "logs" / "nested"
    session = runlog.init(log_dir)

    files = list(log_dir.iterdir())
    assert len(files) == 1
    name = files[0].name
    match = NAME_RE.match(name)
    assert match, f"文件名 {name!r} 不符合 run-<UTC时间戳>-<pid>.jsonl"
    assert int(match.group(2)) == os.getpid()
    assert session == name[: -len(".jsonl")]
    assert runlog.current_session_id() == session
    assert runlog.current_log_path() == log_dir / name
    assert files[0].is_file()


def test_init文件名时间戳为UTC(tmp_path, monkeypatch):
    # 用固定 UTC 时刻钉死"时间戳是 UTC 而非本地时间"。
    import datetime as dt

    class FrozenTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 4, 5, 6, 7, 891000, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(runlog, "datetime", FrozenTime)
    session = runlog.init(tmp_path)
    assert session == "run-20260304T050607Z-{}".format(os.getpid())


def test_重复init被拒绝且原日志继续可用(tmp_path):
    first_dir = tmp_path / "a"
    session = runlog.init(first_dir)
    with pytest.raises(RuntimeError) as exc:
        runlog.init(tmp_path / "b")
    assert "已初始化" in str(exc.value)
    assert runlog.current_session_id() == session
    assert not (tmp_path / "b").exists()  # 拒绝发生在建目录之前
    runlog.event("still_writable")
    assert records(runlog.current_log_path())[-1]["kind"] == "still_writable"


def test_reset_for_tests允许重新init(tmp_path):
    session1 = runlog.init(tmp_path / "a")
    runlog.reset_for_tests()
    assert runlog.current_session_id() is None
    assert runlog.current_log_path() is None
    runlog.init(tmp_path / "b")
    assert runlog.current_log_path().parent == tmp_path / "b"
    assert isinstance(runlog.current_session_id(), str)
    assert session1  # 前一会话的 id 仍归前一会话


# --- 未 init 行为（terminate 侧要捕获的 RuntimeError） ----------------------


def test_未init调用event抛RuntimeError():
    with pytest.raises(RuntimeError, match="未初始化"):
        runlog.event("ready")


def test_未init调用console抛RuntimeError(capsys):
    with pytest.raises(RuntimeError, match="未初始化"):
        runlog.console("不应输出")
    assert capsys.readouterr().out == ""


def test_context与snapshot不要求init():
    with runlog.context(scan_id=7):
        assert runlog.snapshot_context() == {"scan_id": 7}
    assert runlog.snapshot_context() == {}


# --- event：合法 JSON、自动字段、立即 flush --------------------------------


def test_事件行是合法JSON并自动附kind_ts_session(tmp_path):
    runlog.init(tmp_path)
    runlog.event("ready", note="ok", count=3, flag=True)

    path = runlog.current_log_path()
    recs = records(path)  # 未 close 即可逐行 loads：写入即 flush
    assert len(recs) == 1
    rec = recs[0]
    assert rec["kind"] == "ready"
    assert rec["session_id"] == runlog.current_session_id()
    assert TS_RE.match(rec["ts"]), rec["ts"]
    assert rec["note"] == "ok" and rec["count"] == 3 and rec["flag"] is True


def test_多条事件session_id一致(tmp_path):
    session = runlog.init(tmp_path)
    for i in range(5):
        runlog.event("tick", i=i)
    recs = records(runlog.current_log_path())
    assert [r["session_id"] for r in recs] == [session] * 5
    assert [r["i"] for r in recs] == list(range(5))


def test_中文与Unicode值原样写出(tmp_path):
    runlog.init(tmp_path)
    runlog.event("任务", 消息="就绪 ✓")
    raw = raw_lines(runlog.current_log_path())
    assert "就绪 ✓" in raw[0]  # ensure_ascii=False
    assert json.loads(raw[0])["消息"] == "就绪 ✓"


# --- ndarray / 非有限值 -----------------------------------------------------


def test_ndarray字段递归tolist(tmp_path):
    runlog.init(tmp_path)
    runlog.event(
        "frame",
        vec=np.array([1.0, 2.0, 3.0]),
        mat=np.arange(6).reshape(2, 3),
        nested={"items": [np.array([1, 2]), {"inner": np.array([[3.5]])}]},
    )
    rec = records(runlog.current_log_path())[0]
    assert rec["vec"] == [1.0, 2.0, 3.0]
    assert rec["mat"] == [[0, 1, 2], [3, 4, 5]]
    assert rec["nested"]["items"][0] == [1, 2]
    assert rec["nested"]["items"][1]["inner"] == [[3.5]]


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        np.float64("nan"),
        np.float64("inf"),
        np.array([1.0, float("nan")]),
        {"deep": [float("inf")]},
    ],
    ids=["nan", "inf", "neg-inf", "np-nan", "np-inf", "ndarray-nan", "nested-inf"],
)
def test_非有限值明确拒绝且不写出半行(tmp_path, value):
    runlog.init(tmp_path)
    with pytest.raises(ValueError):
        runlog.event("bad", payload=value)
    # 拒绝 = 抛异常；序列化在写盘之前，文件仍是空的（无半行非法 JSON）。
    assert raw_lines(runlog.current_log_path()) == []


def test_非可序列化对象抛TypeError且不留半行(tmp_path):
    runlog.init(tmp_path)
    with pytest.raises(TypeError):
        runlog.event("bad", payload=object())
    assert raw_lines(runlog.current_log_path()) == []


# --- 保留键与字段名校验 ------------------------------------------------------


def test_event拒绝保留键(tmp_path):
    runlog.init(tmp_path)
    # kind 是具名位置参：event("x", kind=...) 由 Python 直接判重复实参（TypeError）；
    # ts/session_id 走 runlog 的保留键守卫（ValueError）。两者都必须拒。
    with pytest.raises(TypeError):
        runlog.event("x", kind="spoofed")
    for key in ("ts", "session_id"):
        with pytest.raises(ValueError, match="保留键"):
            runlog.event("x", **{key: "spoofed"})
    assert raw_lines(runlog.current_log_path()) == []


@pytest.mark.parametrize("key", ["kind", "ts", "session_id"])
def test_context拒绝保留键(tmp_path, key):
    runlog.init(tmp_path)
    with pytest.raises(ValueError, match="保留键"):
        with runlog.context(**{key: 1}):
            pass


def test_event拒绝空kind(tmp_path):
    runlog.init(tmp_path)
    with pytest.raises(ValueError):
        runlog.event("")


# --- context：嵌套、覆盖、恢复、异常退出 -------------------------------------


def test_context嵌套进入退出逐层恢复(tmp_path):
    runlog.init(tmp_path)
    runlog.event("root")
    with runlog.context(task_id="strawberry", task_instance_id=1):
        runlog.event("task_enter")
        with runlog.context(scan_id=0):
            runlog.event("scan0")
            with runlog.context(scan_id=1, attempt_id=2):
                runlog.event("scan1_attempt2")
            runlog.event("back_to_scan0")
            assert runlog.snapshot_context() == {
                "task_id": "strawberry",
                "task_instance_id": 1,
                "scan_id": 0,
            }
        runlog.event("back_to_task")
        assert runlog.snapshot_context() == {
            "task_id": "strawberry",
            "task_instance_id": 1,
        }
    runlog.event("after_task")
    assert runlog.snapshot_context() == {}

    by_kind = {r["kind"]: r for r in records(runlog.current_log_path())}
    for kind, expect in [
        ("root", {}),
        ("task_enter", {"task_id": "strawberry", "task_instance_id": 1}),
        ("scan0", {"task_id": "strawberry", "task_instance_id": 1, "scan_id": 0}),
        (
            "scan1_attempt2",
            {
                "task_id": "strawberry",
                "task_instance_id": 1,
                "scan_id": 1,
                "attempt_id": 2,
            },
        ),
        ("back_to_scan0", {"task_id": "strawberry", "task_instance_id": 1, "scan_id": 0}),
        ("after_task", {}),
    ]:
        rec = by_kind[kind]
        for key, value in expect.items():
            assert rec[key] == value
        for key in ("scan_id", "attempt_id", "task_id", "task_instance_id"):
            if key not in expect:
                assert key not in rec, f"{kind} 行不该带 {key}"


def test_context同键深层覆盖退出恢复前值(tmp_path):
    runlog.init(tmp_path)
    with runlog.context(scan_id="outer"):
        with runlog.context(scan_id="middle"):
            with runlog.context(scan_id="inner"):
                runlog.event("inner")
            runlog.event("middle")
        runlog.event("outer")
    runlog.event("none")
    kinds = [r.get("scan_id", "∅") for r in records(runlog.current_log_path())]
    assert kinds == ["inner", "middle", "outer", "∅"]


def test_context体抛异常也恢复旧值(tmp_path):
    runlog.init(tmp_path)
    with runlog.context(task_instance_id=9):
        with pytest.raises(KeyError):
            with runlog.context(task_instance_id=10):
                raise KeyError("boom")
        assert runlog.snapshot_context() == {"task_instance_id": 9}
    assert runlog.snapshot_context() == {}


def test_event显式fields覆盖同名context键(tmp_path):
    runlog.init(tmp_path)
    with runlog.context(scan_id=1):
        runlog.event("assign", scan_id=42)
    rec = records(runlog.current_log_path())[0]
    assert rec["scan_id"] == 42


def test_snapshot_context是拷贝改动不回写(tmp_path):
    runlog.init(tmp_path)
    with runlog.context(a=1):
        snap = runlog.snapshot_context()
        snap["a"] = 999
        snap["b"] = 2
        assert runlog.snapshot_context() == {"a": 1}


# --- console ---------------------------------------------------------------


def test_console输出到终端且不进jsonl(tmp_path, capsys):
    runlog.init(tmp_path)
    runlog.console("系统就绪，等待语音指令")
    assert capsys.readouterr().out == "系统就绪，等待语音指令\n"
    assert raw_lines(runlog.current_log_path()) == []


# --- set_sink：P1 §3.4 的单测注入点 -----------------------------------------


def test_set_sink注入内存接收器并接管输出(tmp_path):
    runlog.init(tmp_path)
    captured: list[str] = []
    runlog.set_sink(captured.append)
    runlog.event("in_memory", n=1)
    # sink 收到与落盘逐字节同形的 JSON 行；文件不再增长。
    assert len(captured) == 1
    assert not captured[0].endswith("\n")
    rec = json.loads(captured[0])
    assert rec["kind"] == "in_memory" and rec["n"] == 1
    assert rec["session_id"] == runlog.current_session_id()
    assert raw_lines(runlog.current_log_path()) == []


def test_set_sink恢复None后继续写文件(tmp_path):
    runlog.init(tmp_path)
    captured: list[str] = []
    runlog.set_sink(captured.append)
    runlog.event("mem")
    runlog.set_sink(None)
    runlog.event("file")
    assert [json.loads(line)["kind"] for line in captured] == ["mem"]
    assert [r["kind"] for r in records(runlog.current_log_path())] == ["file"]


def test_set_sink拒绝非可调用对象(tmp_path):
    runlog.init(tmp_path)
    with pytest.raises(TypeError):
        runlog.set_sink("not-callable")


def test_注入sink后非有限值同样在进sink前被拒(tmp_path):
    runlog.init(tmp_path)
    captured: list[str] = []
    runlog.set_sink(captured.append)
    with pytest.raises(ValueError):
        runlog.event("bad", x=float("nan"))
    assert captured == []


# --- close ------------------------------------------------------------------


def test_close后event抛RuntimeError(tmp_path):
    runlog.init(tmp_path)
    runlog.event("before_close")
    runlog.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        runlog.event("after_close")
    # session_id 保留可审计，已写内容完整。
    assert runlog.current_session_id() is not None
    assert [r["kind"] for r in records(runlog.current_log_path())] == ["before_close"]


def test_close幂等且close后sink仍可写(tmp_path):
    runlog.init(tmp_path)
    captured: list[str] = []
    runlog.close()
    runlog.close()  # 幂等，不抛
    runlog.set_sink(captured.append)
    runlog.event("via_sink")
    assert len(captured) == 1
