"""qingyun/shutdown.py（terminate 唯一退出出口）的离线单测（P1-02）。

覆盖 P1 §3.3 全部契约点与 §8 步骤 2 的必测项：
    * 打印 status/detail/所处阶段/现场处置指引；
    * terminate 事件行（含 status、detail、stage 与 context 关联键）；
    * 幂等：第二次显式调用仍 SystemExit(1)，但不重复打印、不重复日志；
    * runlog 未 init / 写失败：stderr 降级、仍 exit(1)、不递归自身；
    * atexit 钩子：导入时注册一次、只 flush/关闭、不再调 terminate、不二次退出；
    * 零运动探针：sys.modules 装替身模块记录调用，terminate 全路径零调用；
      另以 AST 钉死 shutdown/runlog 的 import 白名单与运动原语零引用；
    * 子进程端到端：真实退出码 1、日志完整、重复 terminate 不重复记录。
"""

from __future__ import annotations

import ast
import atexit
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from qingyun import runlog, shutdown

# 项目根：本文件在 tests/ 下。
ROOT = Path(__file__).resolve().parents[1]
SHUTDOWN_SRC = ROOT / "qingyun" / "shutdown.py"
RUNLOG_SRC = ROOT / "qingyun" / "runlog.py"

# P1 §3.3 点名的运动/驱动原语 + 同族接口名：terminate 全路径不得触碰。
MOTION_PRIMITIVES = frozenset(
    {
        "hold_current",
        "open_gripper",
        "close_gripper",
        "move_joints",
        "move_linear",
        "configure_torque",
        "reset_fault",
        "grasp_and_place",
        "get_feedback",
        "set_servo_torque",
    }
)
# 允许出现在测试替身名里的"串口/驱动实现模块"根（import 层面全禁）。
DEVICE_MODULE_ROOTS = ("serial", "qingyun.grabbing", "placo", "configs.motion_params")


@pytest.fixture(autouse=True)
def clean_state():
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    yield
    runlog.reset_for_tests()
    shutdown.reset_for_tests()


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def log_lines_of_kind(path: Path, kind: str) -> list[dict]:
    return [rec for rec in read_records(path) if rec["kind"] == kind]


# --- 静态边界：AST 钉死"零运动 import / 零运动原语引用" ----------------------


def _import_roots(tree: ast.Module) -> list[str]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # 相对导入在本仓库不出现；出现即拒绝
                base = "." * node.level + base
            roots.append(base)
    return roots


@pytest.mark.parametrize(
    "src, allowed",
    [
        (
            SHUTDOWN_SRC,
            {"__future__", "atexit", "sys", "typing", "qingyun"},
        ),
        (
            RUNLOG_SRC,
            {
                "__future__",
                "contextlib",
                "json",
                "os",
                "sys",
                "collections.abc",
                "datetime",
                "pathlib",
                "typing",
                "numpy",
            },
        ),
    ],
    ids=["shutdown", "runlog"],
)
def test_import白名单_无任何通向串口或运动实现的路径(src, allowed):
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for imported in _import_roots(tree):
        top = imported.split(".")[0]
        assert top in {a.split(".")[0] for a in allowed}, f"{src.name} 导入了 {imported!r}"
        assert not any(
            imported == dev or imported.startswith(dev + ".") for dev in DEVICE_MODULE_ROOTS
        ), f"{src.name} 导入了设备实现模块 {imported!r}"
    # shutdown 对 qingyun 的唯一依赖必须是 qingyun.runlog。
    if src.name == "shutdown.py":
        qingyun_imports = [n for n in _import_roots(tree) if n.split(".")[0] == "qingyun"]
        assert qingyun_imports and all(n == "qingyun" or n.startswith("qingyun.runlog")
                                       for n in qingyun_imports)
        from_qingyun = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "") == "qingyun"
        ]
        for node in from_qingyun:
            assert {a.name for a in node.names} == {"runlog"}


def test_源码AST零运动原语引用():
    for src in (SHUTDOWN_SRC, RUNLOG_SRC):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in MOTION_PRIMITIVES, f"{src.name} 引用了 {node.attr}"
            if isinstance(node, ast.Name):
                assert node.id not in MOTION_PRIMITIVES, f"{src.name} 引用了 {node.id}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in MOTION_PRIMITIVES
        # import 名字里也不允许出现原语（from x import move_joints 之类）。
        for imported in _import_roots(tree):
            assert imported.split(".")[-1] not in MOTION_PRIMITIVES


# --- terminate：打印四要素 + 事件 + SystemExit(1) ----------------------------


def test_terminate打印status_detail_阶段与处置指引并抛SystemExit1(tmp_path, capsys):
    session = runlog.init(tmp_path)
    with runlog.context(task_id="strawberry", task_instance_id=3, scan_id=2):
        with pytest.raises(SystemExit) as exc:
            shutdown.terminate("NO_GRASPABLE_TARGET", "ignore=10 ≥ valid_count=10")
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "status=NO_GRASPABLE_TARGET" in out
    assert "detail=ignore=10 ≥ valid_count=10" in out
    assert f"session={session}" in out
    assert "task_instance_id=3" in out and "scan_id=2" in out  # 所处阶段=上下文快照
    # 现场处置指引语义：臂冻结、不回 home、不卸力、操作者负责。
    assert "冻结" in out and "不回 home" in out and "不卸力" in out and "操作者" in out
    # 指引不得反向承诺任何动作（只声明"不做什么 + 交操作者"）。
    assert "不发送任何舵机/串口指令" in out


def test_terminate记录terminate事件行含status_detail_阶段(tmp_path, capsys):
    runlog.init(tmp_path)
    with runlog.context(task_id="strawberry", task_instance_id=7, scan_id=1, attempt_id=2):
        with pytest.raises(SystemExit):
            shutdown.terminate("VISION_HARD_ERROR", "管线故障")

    events = log_lines_of_kind(runlog.current_log_path(), "terminate")
    assert len(events) == 1
    rec = events[0]
    assert rec["status"] == "VISION_HARD_ERROR"
    assert rec["detail"] == "管线故障"
    assert rec["session_id"] == runlog.current_session_id()
    assert rec["ts"]
    # §7 terminate 行"所处阶段"：显式 stage 字段 + context 关联键自动合并。
    assert rec["stage"] == {
        "task_id": "strawberry",
        "task_instance_id": 7,
        "scan_id": 1,
        "attempt_id": 2,
    }
    assert rec["task_instance_id"] == 7 and rec["scan_id"] == 1 and rec["attempt_id"] == 2


def test_未init时terminate打印阶段标注未初始化(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        shutdown.terminate("APP_CONFIG_INVALID", "configs/app.local.json 不存在")
    assert exc.value.code == 1
    captured = capsys.readouterr()  # 一次读取 out 与 err（readouterr 会清双缓冲）
    out, err = captured.out, captured.err
    assert "runlog 未初始化" in out
    assert "status=APP_CONFIG_INVALID" in out
    # §3.4 末段：terminate 的日志失败只 stderr 告警，仍退出，不递归。
    assert "日志写入失败" in err
    assert "RuntimeError" in err


# --- 幂等 -------------------------------------------------------------------


def test_幂等_第二次仍SystemExit1且无重复打印与日志(tmp_path, capsys):
    runlog.init(tmp_path)
    with pytest.raises(SystemExit):
        shutdown.terminate("BOOM", "第一次")
    after_first = capsys.readouterr()
    assert "status=BOOM" in after_first.out

    with pytest.raises(SystemExit) as second:
        shutdown.terminate("BOOM_AGAIN", "第二次")
    assert second.value.code == 1
    after_second = capsys.readouterr()
    assert after_second.out == "" and after_second.err == ""  # 不重复打印

    events = log_lines_of_kind(runlog.current_log_path(), "terminate")
    assert len(events) == 1 and events[0]["status"] == "BOOM"


def test_幂等在日志失败路径下同样生效(tmp_path, capsys):
    # 未 init 的首次调用（日志失败 + 退出）之后，第二次仍只抛 SystemExit(1)。
    with pytest.raises(SystemExit):
        shutdown.terminate("X", "first")
    capsys.readouterr()
    with pytest.raises(SystemExit) as second:
        shutdown.terminate("Y", "second")
    assert second.value.code == 1
    assert capsys.readouterr().out == ""


def test_幂等标志先于一切动作设置(tmp_path, monkeypatch, capsys):
    # 即使打印/记录中途出乱子（此处模拟 print 抛异常），标志也已置位，
    # 后续调用直接 SystemExit(1)，不会重走记录路径。
    runlog.init(tmp_path)

    real_print = print
    boom = {"n": 0}

    def flaky_print(*args, **kwargs):
        boom["n"] += 1
        if boom["n"] == 3:  # 第三行（阶段行）坏掉
            raise OSError("stdout 故障")
        real_print(*args, **kwargs)

    monkeypatch.setattr(shutdown, "print", flaky_print, raising=False)
    with pytest.raises(SystemExit):
        shutdown.terminate("PARTIAL", "部分打印失败")
    assert "日志写入失败" not in capsys.readouterr().err  # 事件本身仍成功
    with pytest.raises(SystemExit) as second:
        shutdown.terminate("PARTIAL", "第二次不应记录")
    assert second.value.code == 1
    assert len(log_lines_of_kind(runlog.current_log_path(), "terminate")) == 1


# --- 日志写失败降级 ----------------------------------------------------------


def test_事件写失败时terminate仍退出且stderr告警(tmp_path, capsys):
    runlog.init(tmp_path)

    def bad_sink(line: str) -> None:
        raise OSError("磁盘故障")

    runlog.set_sink(bad_sink)
    with pytest.raises(SystemExit) as exc:
        shutdown.terminate("DISK_BROKEN", "写不进去")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "terminate 事件日志写入失败" in err and "OSError" in err and "磁盘故障" in err
    # 不递归：terminate 只被显式调用了一次。
    assert err.count("日志写入失败") == 1


def test_非字符串入参被安全字符串化(tmp_path, capsys):
    class Toxic:
        def __str__(self):
            raise RuntimeError("str 坏")

        def __repr__(self):
            raise RuntimeError("repr 坏")

    runlog.init(tmp_path)
    with pytest.raises(SystemExit) as exc:
        shutdown.terminate(Toxic(), "ok")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "unprintable" in out  # 三级兜底：类型名占位，绝不抛非 SystemExit


# --- atexit 钩子 -------------------------------------------------------------


def test_atexit钩子在导入时注册一次且不在terminate内注册():
    tree = ast.parse(SHUTDOWN_SRC.read_text(encoding="utf-8"))
    module_level_registers = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "register"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "atexit"
    ]
    assert len(module_level_registers) == 1
    hook_arg = module_level_registers[0].value.func  # noqa: F841（结构自查）
    (arg,) = module_level_registers[0].value.args
    assert isinstance(arg, ast.Name) and arg.id == "_atexit_flush"

    terminate_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "terminate"
    )
    for node in ast.walk(terminate_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (node.func.attr == "register" and
                        isinstance(node.func.value, ast.Name) and node.func.value.id == "atexit")


def test_atexit钩子源码不调terminate不exit只关日志():
    tree = ast.parse(SHUTDOWN_SRC.read_text(encoding="utf-8"))
    hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_atexit_flush")
    called: list[str] = []
    for node in ast.walk(hook):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.append(f.id)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                called.append(f"{f.value.id}.{f.attr}")
    assert "terminate" not in called
    assert "sys.exit" not in called and "exit" not in called
    assert "runlog.close" in called
    # 除日志关闭与就地告警外没有其他副作用面（type/print 只服务异常摘要文本）。
    assert set(called) <= {"runlog.close", "_warn_stderr", "print", "type"}


def test_atexit钩子二次调用不二次退出不重复日志(tmp_path, capsys):
    runlog.init(tmp_path)
    runlog.event("work")
    with pytest.raises(SystemExit):
        shutdown.terminate("DONE_ONE_WAY", "首次退出")
    capsys.readouterr()

    # 解释器退出时 atexit 会调用钩子；测试里直接手动调用模拟。
    shutdown._atexit_flush()  # 不得抛 SystemExit / 任何异常
    shutdown._atexit_flush()  # 幂等
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""

    assert len(log_lines_of_kind(runlog.current_log_path(), "terminate")) == 1
    # 钩子确实把文件关了：再写 event 报"已关闭"。
    with pytest.raises(RuntimeError, match="已关闭"):
        runlog.event("too_late")
    # 内容完整（close 前已 flush）。
    kinds = [r["kind"] for r in read_records(runlog.current_log_path())]
    assert kinds == ["work", "terminate"]


def test_atexit钩子未terminate过时只flush关闭(tmp_path):
    runlog.init(tmp_path)
    runlog.event("just_work")
    shutdown._atexit_flush()
    assert [r["kind"] for r in read_records(runlog.current_log_path())] == ["just_work"]


def test_导入模块恰好注册一个atexit钩子():
    # atexit 没有公开的列举 API；用注册数增量证明"模块 import 注册恰好一个钩子"
    # （_ncallbacks 是 CPython 稳定私有 API，缺失时退回模块属性检查）。
    import importlib

    if hasattr(atexit, "_ncallbacks"):
        before = atexit._ncallbacks()
        importlib.reload(shutdown)
        assert atexit._ncallbacks() - before == 1
    assert shutdown._atexit_flush.__module__ == "qingyun.shutdown"
    assert callable(shutdown._atexit_flush)


# --- 零运动探针：替身模块记录一切属性访问/调用 --------------------------------


class _ProbeModule(types.ModuleType):
    """任意属性访问都返回记录调用的替身函数。"""

    def __init__(self, name: str, calls: list[str]) -> None:
        super().__init__(name)
        self._calls = calls

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def _recorded(*args, **kwargs):
            self._calls.append(f"{self.__name__}.{attr}")
            return None

        return _recorded


PROBE_MODULES = (
    "qingyun.grabbing",
    "qingyun.grabbing.motor_control",
    "qingyun.grabbing.arm_control",
    "qingyun.grabbing.executor",
    "qingyun.grabbing.safety",
    "qingyun.grabbing.vision",
    "serial",
)


@pytest.fixture()
def motion_probe(monkeypatch):
    calls: list[str] = []
    modules = {}
    for name in PROBE_MODULES:
        mod = _ProbeModule(name, calls)
        monkeypatch.setitem(sys.modules, name, mod)
        modules[name] = mod

    # 探针自检：机制本身能记录调用（防止"零调用"是假阳性）。
    modules["qingyun.grabbing.arm_control"].move_joints([0] * 6)
    modules["serial"].write(b"frame")
    assert calls == [
        "qingyun.grabbing.arm_control.move_joints",
        "serial.write",
    ]
    calls.clear()
    return calls


def test_探针_terminate全路径零运动调用(motion_probe, tmp_path):
    calls = motion_probe
    # 路径 1：正常 init + context + event + console，然后首次 terminate。
    runlog.init(tmp_path)
    first_path = runlog.current_log_path()
    with runlog.context(task_id="strawberry", task_instance_id=1, scan_id=0):
        runlog.event("assign", grade="A", place_id="strawberry_A_01")
        runlog.console("分配完成")
        with pytest.raises(SystemExit) as exc:
            shutdown.terminate("FAULT_UNCONTROLLED", "失控")
        assert exc.value.code == 1
        with runlog.context(scan_id=1):
            pass
    # 路径 2：幂等第二次调用。
    with pytest.raises(SystemExit) as exc2:
        shutdown.terminate("FAULT_UNCONTROLLED", "再来一次也不记录")
    assert exc2.value.code == 1
    # 路径 3：atexit 钩子（解释器退出时执行的动作）。
    shutdown._atexit_flush()
    shutdown._atexit_flush()
    # 路径 4：未 init 场景的 terminate（日志失败降级路径）。
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    with pytest.raises(SystemExit):
        shutdown.terminate("APP_CONFIG_INVALID", "无日志出口也要退")
    # 路径 5：sink 抛异常场景的 terminate（日志写失败降级路径）。
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    runlog.init(tmp_path / "b")
    runlog.set_sink(_explode)
    with pytest.raises(SystemExit):
        shutdown.terminate("DISK", "坏")
    runlog.set_sink(None)

    assert calls == []  # 全路径零运动/串口调用
    assert len(log_lines_of_kind(first_path, "terminate")) == 1  # 首次只记录一次（幂等）
    b_logs = list((tmp_path / "b").glob("run-*.jsonl"))
    assert len(b_logs) == 1
    assert log_lines_of_kind(b_logs[0], "terminate") == []  # sink 故障路径不落任何行


def _explode(line: str) -> None:
    raise RuntimeError("sink 爆炸")


def test_探针_模块重复导入不触发任何设备路径(motion_probe):
    import importlib

    importlib.reload(runlog)
    importlib.reload(shutdown)
    assert motion_probe == []


# --- 子进程端到端：真实退出码 + 日志完整 + 幂等 ------------------------------


def _run_snippet(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from qingyun import runlog, shutdown\n"
        + body
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )


def test_子进程terminate真实退出码1且日志带终止行(tmp_path):
    proc = _run_snippet(
        tmp_path,
        f"""
session = runlog.init({str(tmp_path)!r})
with runlog.context(task_id="strawberry", task_instance_id=1):
    shutdown.terminate("SUBPROCESS_DEMO", "端到端演示")
print("UNREACHABLE")
""",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "UNREACHABLE" not in proc.stdout
    assert "status=SUBPROCESS_DEMO" in proc.stdout
    assert "冻结" in proc.stdout and "操作者" in proc.stdout
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    events = [r for r in read_records(logs[0]) if r["kind"] == "terminate"]
    assert len(events) == 1
    assert events[0]["status"] == "SUBPROCESS_DEMO" and events[0]["task_instance_id"] == 1


def test_子进程重复terminate只记录一次且退出码1(tmp_path):
    proc = _run_snippet(
        tmp_path,
        f"""
runlog.init({str(tmp_path)!r})
try:
    shutdown.terminate("FIRST", "首次")
except SystemExit as e:
    first_code = e.code
print(f"CAUGHT={{first_code}}")
shutdown.terminate("SECOND", "不应被记录")
""",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "CAUGHT=1" in proc.stdout
    assert "status=FIRST" in proc.stdout
    assert "SECOND" not in proc.stdout
    logs = list(tmp_path.glob("run-*.jsonl"))
    assert len(logs) == 1
    events = [r for r in read_records(logs[0]) if r["kind"] == "terminate"]
    assert len(events) == 1 and events[0]["status"] == "FIRST"


# --- 与 atexit 生态共存的最后核对：atexit 自身不再新增退出码来源 --------------


def test_atexit钩子异常不外溢为terminate调用(monkeypatch, tmp_path, capsys):
    # 把 runlog.close 换坏：钩子必须就地降级（stderr），不调 terminate、不抛。
    runlog.init(tmp_path)

    def explode() -> None:
        raise RuntimeError("close 坏了")

    monkeypatch.setattr(runlog, "close", explode)
    shutdown._atexit_flush()
    err = capsys.readouterr().err
    assert "退出时日志关闭失败" in err and "RuntimeError" in err
    assert not shutdown._terminated  # 钩子没有置位/触碰幂等标志
