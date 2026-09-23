"""main / run_mock 装配层场景测试（P1-05，覆盖 P1 §9 的 T01/T02/T20/T21/T22/T25/
T27/T31 与预检前移（初审 L1）、SystemExit 直通、WAITING Ctrl-C、attempts 包装、
startup 哈希、mock 冒烟不静默回退替身（DEC-002））。

驱动方式（P1 §2 末段）：
    * 生产装配的全部依赖都以 ``main`` 模块级名字被引用（``load_app_config`` /
      ``runlog`` / ``load_motion_params`` / ``preflight`` / ``Sts3215MotorController`` /
      ``ArmController`` / ``asr`` / ``cloud_model`` / ``vision`` / ``run_session`` /
      ``PLAN_REGISTRY``），本文件用 monkeypatch 对任一名字注入探针/替身；mock 入口
      同理经 ``run_mock`` 命名空间。
    * 会话替身复用 ``tests/support_upper`` 的 FakeASR/FakeLLM/FakeVision/FakeArm
      （真实异常类），日志复用 ``runlog.set_sink`` 内存接收器（P1 §3.4）。
    * 真实驱动类 ``Sts3215MotorController.__init__`` 被换成失败探针，钉死
      "mock 无真实串口"（T22）；``install_stubs`` 的模块属性替换由 autouse fixture
      逐用例恢复（``run_mock.reset_stubs_for_tests``）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import configs.app_config as app_config_module
import main as production_main
import qingyun.asr as asr_module
import qingyun.cloud_model as cloud_module
from configs.app_config import (
    MOCK_APP_PATH,
    AppConfig,
    AsrSettings,
    LlmSettings,
    MockSettings,
    PreflightError,
    load_app_config,
)
from configs.common_interface import GraspStatus, JointFeedback, VisionThresholds
from configs.motion_params import load_motion_params
from main import cold_start_recover, log_startup, run_session, thresholds_from_params
from qingyun import runlog, shutdown
from qingyun import plans as plans_package
from qingyun.asr import ASRHardError
from qingyun.cloud_model import LLMHardError
from qingyun.grabbing.motor_control import Sts3215MotorController
from qingyun.grabbing.vision import NoTarget, VisionHardError
from qingyun.plans import PLAN_REGISTRY, StrawberryPlan
from qingyun.plans import base as base_module
from scripts import run_mock
from tests.mock_motor import MockMotorController, SimTime
from tests.support_upper import (
    FAKE_HOME_JOINTS_DEG,
    FAKE_SAFE_WAYPOINTS_DEG,
    FakeArm,
    FakeASR,
    FakeLLM,
    FakeVision,
    make_arm_params,
)

ROOT = MOCK_APP_PATH.parent.parent
MOCK_PROFILE_PATH = ROOT / "tests" / "fixtures" / "mock" / "profile.json"


# ---------------------------------------------------------------------------
# 公共设施
# ---------------------------------------------------------------------------


class Sink:
    """``runlog.set_sink`` 内存接收器（与 test_plan_flow 同一形态）。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(line)

    @property
    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.lines]

    @property
    def kinds(self) -> list[str]:
        return [record["kind"] for record in self.records]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["kind"] == kind]

    def sole(self, kind: str) -> dict[str, Any]:
        found = self.of(kind)
        assert len(found) == 1, f"{kind} 应恰好一条，实际 {len(found)}：{found}"
        return found[0]


@pytest.fixture(autouse=True)
def runtime():
    """复位单进程状态：runlog / terminate 幂等标志 / mock 替身安装守卫。"""
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    run_mock.reset_stubs_for_tests()
    yield
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    run_mock.reset_stubs_for_tests()


class RunlogRecorder:
    """入口模块 ``runlog`` 名字的替身：init 成功后立即改道 sink（步骤 2 的时机本身
    是被测语义，所以不能在测试前置 init）。事件/输出全部转发真实模块。"""

    def __init__(self, receiver: Sink, order: list[str] | None = None) -> None:
        self.receiver = receiver
        self.order = order
        self.init_count = 0

    def init(self, log_dir):
        if self.order is not None:
            self.order.append("runlog.init")
        session_id = runlog.init(log_dir)  # 真实 init（tmp 下的文件即 T31 断言对象）
        runlog.set_sink(self.receiver)
        self.init_count += 1
        return session_id

    def event(self, kind, **fields):
        if self.order is not None:
            self.order.append(f"event:{kind}")
        runlog.event(kind, **fields)

    def console(self, msg):
        runlog.console(msg)

    def context(self, **fields):
        return runlog.context(**fields)


def _terminate_record(sink: Sink) -> dict[str, Any]:
    events = sink.of("terminate")
    assert len(events) == 1, f"terminate 事件应恰好一条（幂等不重复记录）：{events}"
    return events[0]


def _joint_feedback(angles_deg) -> JointFeedback:
    a = np.asarray(angles_deg, dtype=np.float64)
    return JointFeedback(
        angles_deg=a,
        speeds_deg_s=np.zeros(5),
        currents_ma=np.zeros(5),
        gripper_pct=50.0,
        gripper_speed_pct_s=0.0,
        gripper_current_ma=0.0,
        sample_start_ns=0,
        sample_end_ns=1,
        sequence=0,
    )


class _FeedbackMotor:
    """冷启动读反馈用的假电机：只给 get_feedback；动作断言看 FakeArm.events。"""

    def __init__(self, angles_deg) -> None:
        self.feedback = _joint_feedback(angles_deg)
        self.reads = 0

    def get_feedback(self) -> JointFeedback:
        self.reads += 1
        return self.feedback


def _cold_params():
    """FakeArm.params 形状 + cold_start_recover 需要的 motion.settle_position_tol_deg。"""
    params = make_arm_params()
    params.motion = SimpleNamespace(
        settle_position_tol_deg=np.full(5, 2.5, dtype=np.float64)
    )
    return params


def _home_arm() -> FakeArm:
    """反馈恰在家（逐轴差 0）的 FakeArm：装配成功路径的默认臂。"""
    return FakeArm(
        [], params=_cold_params(), motor=_FeedbackMotor(np.asarray(FAKE_HOME_JOINTS_DEG))
    )


def _mock_cfg(tmp_path) -> AppConfig:
    """结构合法的生产形态 cfg（装配用例都 monkeypatch 掉真实加载，不触盘上资源）。"""
    return AppConfig(
        profile_path=tmp_path / "profile.json",
        vision_config_path=tmp_path / "vision.json",
        prompt_path=tmp_path / "prompt.md",
        log_dir=tmp_path / "custom_logs",
        asr=AsrSettings("m", "https://example.invalid/api", 30.0),
        llm=LlmSettings("m", "https://example.invalid", 30.0),
        mock=None,
    )


class _NeverBuiltPlan(StrawberryPlan):
    """T01：会话被要求实例化即为失败。"""

    def __init__(self, arm) -> None:  # noqa: ANN001
        raise AssertionError("invalid/空转写不得创建 Plan（P1-T01）")


class _SpyPlan(StrawberryPlan):
    """记录每个任务实例的 Plan 子类（T02/T25：实例身份与计数器独立性）。"""

    instances: list["_SpyPlan"] = []

    def __init__(self, arm) -> None:  # noqa: ANN001
        super().__init__(arm)
        _SpyPlan.instances.append(self)


_KI = KeyboardInterrupt("测试注入：WAITING 阶段 Ctrl-C")


# ---------------------------------------------------------------------------
# T01 / T02 / SystemExit 直通 / WAITING Ctrl-C / §5 映射（run_session）
# ---------------------------------------------------------------------------


@pytest.fixture
def session_logs(tmp_path) -> Sink:
    """会话用例：正式 init（tmp）+ 事件改道内存接收器。"""
    receiver = Sink()
    runlog.init(tmp_path / "session_logs")
    runlog.set_sink(receiver)
    return receiver


def test_T01_invalid与空转写不创建Plan不触视觉不触抓取(monkeypatch, session_logs: Sink,
                                                  tmp_path):
    asr = FakeASR(["", "   ", "帮我拿个杯子", _KI])
    llm = FakeLLM(["invalid"])
    fake_vision = FakeVision([])  # 空脚本：被调用即以 VisionHardError 失败
    arm = FakeArm([])  # 空脚本：被调用即以 AssertionError 失败
    monkeypatch.setattr(production_main, "asr", asr)
    monkeypatch.setattr(production_main, "cloud_model", llm)
    monkeypatch.setattr(base_module, "vision", fake_vision)
    with pytest.raises(SystemExit) as excinfo:
        run_session(_mock_cfg(tmp_path), arm, registry={"strawberry": _NeverBuiltPlan})
    assert excinfo.value.code == 1
    assert _terminate_record(session_logs)["status"] == "KEYBOARD_INTERRUPT"
    assert fake_vision.scan_count == 0 and fake_vision.configure_calls == []
    assert arm.events == []
    assert session_logs.of("task_done") == []
    # 三条非硬错误转写各一条 asr_result（含空串）；仅一条非空转写触发 llm_result。
    assert len(session_logs.of("asr_result")) == 3
    assert len(session_logs.of("llm_result")) == 1
    sole_llm = session_logs.sole("llm_result")
    assert sole_llm["text"] == "帮我拿个杯子" and sole_llm["task"] == "invalid"


def test_T02_连续两个有效任务各自独立Plan实例计数器从零(monkeypatch, session_logs: Sink):
    asr = FakeASR(["分拣草莓", "分拣草莓", _KI])
    llm = FakeLLM(["strawberry", "strawberry"])
    fake_vision = FakeVision([NoTarget("no_detection")] * 6)
    arm = FakeArm([])
    monkeypatch.setattr(production_main, "asr", asr)
    monkeypatch.setattr(production_main, "cloud_model", llm)
    monkeypatch.setattr(base_module, "vision", fake_vision)
    _SpyPlan.instances = []
    with pytest.raises(SystemExit) as excinfo:
        run_session(None, arm, registry={"strawberry": _SpyPlan})
    assert excinfo.value.code == 1

    assert len(_SpyPlan.instances) == 2, "两个任务 = 两个独立 Plan 实例（§5）"
    first, second = _SpyPlan.instances
    assert first is not second
    # 每任务计数器从 0 起、NoTarget 三连各自结束：互不沾染（T31 关联无串号）。
    assert first.scan_id == 3 and second.scan_id == 3
    assert first.no_target_streak == 3 and second.no_target_streak == 3
    assert first.a_count == 0 and second.a_count == 0
    assert fake_vision.scan_count == 6, "两任务各三连，不多不少"

    done = session_logs.of("task_done")
    assert len(done) == 2
    assert [d["task_instance_id"] for d in done] == [1, 2]
    assert [d["scan_id"] for d in done] == [3, 3]
    assert [d["task_id"] for d in done] == ["strawberry", "strawberry"]


def test_SystemExit直通terminate不被包装成其他status(monkeypatch, session_logs: Sink):
    """plan 内 terminate（D14 封顶同款出口）冒泡：退出码 1、terminate 恰一条。"""

    class _TerminatingPlan(StrawberryPlan):
        def run(self) -> None:
            shutdown.terminate("NO_GRASPABLE_TARGET", "测试注入：封顶直通")

    asr = FakeASR(["分拣草莓", _KI])
    monkeypatch.setattr(production_main, "asr", asr)
    monkeypatch.setattr(production_main, "cloud_model", FakeLLM(["strawberry"]))
    with pytest.raises(SystemExit) as excinfo:
        run_session(None, FakeArm([]), registry={"strawberry": _TerminatingPlan})
    assert excinfo.value.code == 1
    assert _terminate_record(session_logs)["status"] == "NO_GRASPABLE_TARGET"
    assert len(asr.listen_calls) == 1, "terminate 后循环不再推进、不二次包装"


def test_KeyboardInterrupt在WAITING阶段走terminate统一退出(monkeypatch, session_logs: Sink):
    monkeypatch.setattr(production_main, "asr", FakeASR([_KI]))
    with pytest.raises(SystemExit) as excinfo:
        run_session(None, FakeArm([]), registry={"strawberry": _NeverBuiltPlan})
    assert excinfo.value.code == 1
    assert _terminate_record(session_logs)["status"] == "KEYBOARD_INTERRUPT"


@pytest.mark.parametrize(
    "asr_script, llm_script, expected_status, expected_detail",
    [
        (["", ASRHardError("设备掉了")], [], "ASR_HARD_ERROR", None),
        (["分拣草莓"], [LLMHardError("云 502")], "LLM_HARD_ERROR", None),
        (["分拣草莓"], ["blueberry"], "LLM_CONTRACT_VIOLATION", repr("blueberry")),
    ],
)
def test_ASR硬错误_LLM硬错误_违约task的映射(monkeypatch, tmp_path, asr_script, llm_script,
                                       expected_status, expected_detail):
    """§5 表：三种映射各自唯一 status；terminate 幂等只一条。"""
    receiver = Sink()
    runlog.init(tmp_path / "logs")
    runlog.set_sink(receiver)
    monkeypatch.setattr(production_main, "asr", FakeASR(asr_script))
    monkeypatch.setattr(production_main, "cloud_model", FakeLLM(llm_script))
    with pytest.raises(SystemExit) as excinfo:
        run_session(None, FakeArm([]), registry={"strawberry": _NeverBuiltPlan})
    assert excinfo.value.code == 1
    report = _terminate_record(receiver)
    assert report["status"] == expected_status
    if expected_detail is not None:
        assert report["detail"] == expected_detail
    assert receiver.of("ready") == []


# ---------------------------------------------------------------------------
# T20 / T21：冷启动复位
# ---------------------------------------------------------------------------


def test_T20_home容差内零搬臂且只读一次反馈():
    arm = FakeArm(
        [],
        params=_cold_params(),
        # 逐轴 ≤ tol（含恰好等于 2.5 的边界轴）→ 容差内
        motor=_FeedbackMotor(np.asarray(FAKE_HOME_JOINTS_DEG) + [2.5, 0, 0, 0, 0]),
    )
    cold_start_recover(arm)
    assert arm.events == [], "容差内不开爪、不搬臂（§4 步骤 7）"
    assert arm.motor.reads == 1


def test_T20_容差外顺序为open_逐个waypoints_home():
    out_of_home = np.asarray(FAKE_HOME_JOINTS_DEG) + [0, 0, 6.0, 0, 0]
    arm = FakeArm([], params=_cold_params(), motor=_FeedbackMotor(out_of_home))
    cold_start_recover(arm)
    assert arm.command_sequence() == [
        "open_gripper", "move_joints", "move_joints", "move_joints",
    ]
    expected = [tuple(float(v) for v in wp) for wp in FAKE_SAFE_WAYPOINTS_DEG] + [
        tuple(float(v) for v in FAKE_HOME_JOINTS_DEG)
    ]
    assert arm.moved_positions() == expected, "中转位按列表顺序（冻结），最后 home"


def test_T21_复位原语异常就地向上抛由入口负责映射():
    arm = FakeArm(
        [],
        params=_cold_params(),
        motor=_FeedbackMotor(np.asarray(FAKE_HOME_JOINTS_DEG) + [9, 0, 0, 0, 0]),
    )
    arm.fail("move_joints", RuntimeError("总线读回中断"), at_call=2)
    with pytest.raises(RuntimeError, match="总线读回中断"):
        cold_start_recover(arm)  # 复位本体不 terminate：映射是入口职责（§4 末段）


def test_T21_装配把冷启动任意异常与Ctrl_C统一映射terminate(monkeypatch, tmp_path):
    for injected in (RuntimeError("复位原语炸了"), KeyboardInterrupt("复位中 Ctrl-C")):
        shutdown.reset_for_tests()
        runlog.reset_for_tests()
        sink = Sink()
        arm = FakeArm(
            [],
            params=_cold_params(),
            motor=_FeedbackMotor(np.asarray(FAKE_HOME_JOINTS_DEG) + [9, 0, 0, 0, 0]),
        )
        arm.fail("move_joints", injected)
        _patch_production_assembly_ok(monkeypatch, tmp_path, sink, arm=arm)
        with pytest.raises(SystemExit) as excinfo:
            production_main.assemble_production()
        assert excinfo.value.code == 1
        assert _terminate_record(sink)["status"] == "COLD_START_RECOVERY_FAILED"


# ---------------------------------------------------------------------------
# 生产装配（§4 步骤 1–11）：顺序、ready、T25、预检前移、早期失败
# ---------------------------------------------------------------------------


def _patch_production_assembly_ok(monkeypatch, tmp_path, receiver: Sink, *,
                                  arm: FakeArm | None = None,
                                  cfg: AppConfig | None = None,
                                  asr=None, cloud=None, vision=None) -> dict[str, Any]:
    """装配成功路径的默认探针套件：外部名字全换替身，返回关键对象供断言。"""
    cfg = cfg if cfg is not None else _mock_cfg(tmp_path)
    monkeypatch.setattr(production_main, "runlog", RunlogRecorder(receiver))
    monkeypatch.setattr(production_main, "load_app_config", lambda *a, **k: cfg)
    monkeypatch.setattr(
        production_main, "load_motion_params",
        lambda path, *, mode: load_motion_params(MOCK_PROFILE_PATH, mode="mock"),
    )
    monkeypatch.setattr(production_main, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(production_main, "Sts3215MotorController",
                        lambda params: SimpleNamespace(name="probe_motor"))
    fake_arm = arm if arm is not None else _home_arm()
    monkeypatch.setattr(production_main, "ArmController", lambda motor, params: fake_arm)
    # cold_start_recover **不**替换：真实共享函数在"反馈在家"的假臂上零动作。
    fake_asr = asr if asr is not None else FakeASR([])
    fake_cloud = cloud if cloud is not None else FakeLLM([])
    fake_vision = vision if vision is not None else FakeVision([])
    monkeypatch.setattr(production_main, "asr", fake_asr)
    monkeypatch.setattr(production_main, "cloud_model", fake_cloud)
    monkeypatch.setattr(production_main, "vision", fake_vision)
    return {"cfg": cfg, "arm": fake_arm, "asr": fake_asr,
            "cloud": fake_cloud, "vision": fake_vision}


def test_装配步骤顺序逐字等于P1_4的1到11(monkeypatch, tmp_path):
    order: list[str] = []
    sink = Sink()
    cfg = _mock_cfg(tmp_path)
    params = load_motion_params(MOCK_PROFILE_PATH, mode="mock")
    fake_arm = _home_arm()

    def _rec(name, value=None):
        order.append(name)
        return value

    monkeypatch.setattr(production_main, "runlog",
                        RunlogRecorder(sink, order=order))
    monkeypatch.setattr(production_main, "load_app_config",
                        lambda *a, **k: _rec("load_app_config", cfg))
    monkeypatch.setattr(production_main, "load_motion_params",
                        lambda p, *, mode: _rec("load_motion_params", params))
    monkeypatch.setattr(production_main, "preflight",
                        lambda *a, **k: _rec("preflight"))
    monkeypatch.setattr(production_main, "Sts3215MotorController",
                        lambda p: _rec("motor", SimpleNamespace()))
    monkeypatch.setattr(production_main, "ArmController",
                        lambda m, p: _rec("arm", fake_arm))
    real_cold = production_main.cold_start_recover
    monkeypatch.setattr(production_main, "cold_start_recover",
                        lambda a: _rec("cold_start", real_cold(a)))
    monkeypatch.setattr(production_main, "asr",
                        SimpleNamespace(init=lambda c: order.append("asr.init")))
    monkeypatch.setattr(production_main, "cloud_model",
                        SimpleNamespace(init=lambda c: order.append("cloud.init")))
    monkeypatch.setattr(
        production_main, "vision",
        SimpleNamespace(
            configure=lambda t, c: order.append("vision.configure"),
            init=lambda: order.append("vision.init"),
        ),
    )
    production_main.assemble_production()
    assert order == [
        "load_app_config", "runlog.init", "event:startup", "load_motion_params",
        "preflight", "motor", "arm", "cold_start", "asr.init", "cloud.init",
        "vision.configure", "vision.init", "event:ready",
    ], "冷启动顺序固定（P1 §4）；ready 在三 init 之后（R03）"


def test_T25_三init逐个失败对应terminate且ready不出现会话不进(monkeypatch, tmp_path):
    cases = ["asr", "cloud", "vision_configure", "vision_init"]
    for stage in cases:
        shutdown.reset_for_tests()
        runlog.reset_for_tests()
        sink = Sink()
        entered: list[str] = []
        monkeypatch.setattr(production_main, "run_session",
                            lambda *a, **k: entered.append("entered"))
        error = {
            "asr": ASRHardError("麦克风设备缺失"),
            "cloud": LLMHardError("prompt 读取失败"),
        }.get(stage, VisionHardError("相机离线"))
        asr = FakeASR([])
        cloud = FakeLLM([])
        vision = FakeVision([])
        if stage == "asr":
            asr.init = lambda c: (_ for _ in ()).throw(error)
        if stage == "cloud":
            cloud.init = lambda c: (_ for _ in ()).throw(error)
        if stage == "vision_configure":
            vision.configure = lambda t, c: (_ for _ in ()).throw(error)
        if stage == "vision_init":
            vision.init = lambda: (_ for _ in ()).throw(error)
        _patch_production_assembly_ok(monkeypatch, tmp_path, sink,
                                      asr=asr, cloud=cloud, vision=vision)
        with pytest.raises(SystemExit) as excinfo:
            production_main.assemble_production()
        assert excinfo.value.code == 1, stage
        assert _terminate_record(sink)["status"] == {
            "asr": "ASR_INIT_FAILED",
            "cloud": "LLM_INIT_FAILED",
            "vision_configure": "VISION_INIT_FAILED",
            "vision_init": "VISION_INIT_FAILED",
        }[stage]
        assert sink.of("ready") == [], f"{stage} 失败不得 ready（§4 末段）"
        assert entered == [], f"{stage} 失败不得进入会话"


def test_T25_全部init成功后ready恰一次且首个有效口令进入Plan(monkeypatch, tmp_path):
    """一个用例串起：三 init 各一次 → ready 恰一次 → 真实 run_session 进入首任务。"""
    _SpyPlan.instances = []
    sink = Sink()
    asr = FakeASR(["分拣草莓", _KI])
    cloud = FakeLLM(["strawberry"])
    vision = FakeVision([])
    fake_arm = _home_arm()
    monkeypatch.setattr(base_module, "vision", FakeVision([NoTarget("空桌")] * 3))
    monkeypatch.setattr(production_main, "PLAN_REGISTRY", {"strawberry": _SpyPlan})
    # 不替换 run_session：走真实会话实现（"复用生产会话"即被测语义）。
    _patch_production_assembly_ok(monkeypatch, tmp_path, sink, arm=fake_arm,
                                  asr=asr, cloud=cloud, vision=vision)
    with pytest.raises(SystemExit) as excinfo:
        production_main.main()
    assert excinfo.value.code == 1
    assert len(asr.init_calls) == 1 and len(cloud.init_calls) == 1
    assert len(vision.configure_calls) == 1 and len(vision.init_calls) == 1
    assert len(sink.of("ready")) == 1, "ready 恰一次"
    kinds = sink.kinds
    assert kinds.index("ready") < kinds.index("task_done"), "先 ready 后会话完成"
    assert len(_SpyPlan.instances) == 1
    assert _SpyPlan.instances[0].arm is fake_arm
    assert sink.sole("task_done")["task_instance_id"] == 1
    call = vision.configure_calls[0]
    assert isinstance(call["thresholds"], VisionThresholds)
    assert call["config_path"] == _mock_cfg(tmp_path).vision_config_path


def test_T31_自定义log_dir生效且任务日志关联无串号(monkeypatch, tmp_path):
    sink = Sink()
    _patch_production_assembly_ok(monkeypatch, tmp_path, sink)
    production_main.assemble_production()
    log_path = runlog.current_log_path()
    assert log_path is not None
    assert log_path.parent == tmp_path / "custom_logs", "cfg.log_dir 自定义生效（T31）"
    assert log_path.is_file() and log_path.name.startswith("run-")


def test_预检前移preflight失败时驱动零构造并记preflight_failed事件(monkeypatch, tmp_path):
    built: list[Any] = []

    class _MotorProbe:
        def __init__(self, params, *a, **k):
            built.append(self)
            raise AssertionError("预检失败后绝不允许实例化驱动（初审 L1）")

    def failing_preflight(cfg, params, registry):
        raise PreflightError("places", "strawberry_A_01 不在 places 中（注入）")

    sink = Sink()
    _patch_production_assembly_ok(monkeypatch, tmp_path, sink)
    monkeypatch.setattr(production_main, "Sts3215MotorController", _MotorProbe)
    monkeypatch.setattr(production_main, "preflight", failing_preflight)
    with pytest.raises(SystemExit) as excinfo:
        production_main.assemble_production()
    assert excinfo.value.code == 1
    assert built == [], "预检在 motor 构造之前（§3.2/§4 步骤 4→5）：零串口、零舵机写入"
    assert sink.sole("preflight_failed")["check"] == "places"
    report = _terminate_record(sink)
    assert report["status"] == "PREFLIGHT_FAILED"
    assert "places" in report["detail"]


def test_配置与日志初始化失败仅stderr加terminate无正式日志(monkeypatch, capsys, tmp_path):
    from configs.app_config import AppConfigError

    sink = Sink()
    monkeypatch.setattr(
        production_main, "load_app_config",
        lambda *a, **k: (_ for _ in ()).throw(
            AppConfigError("mock", "production 不得含 mock 段")
        ),
    )
    with pytest.raises(SystemExit) as excinfo:
        production_main.assemble_production()
    assert excinfo.value.code == 1
    assert "APP_CONFIG_INVALID" in capsys.readouterr().err, "步骤 1 失败：stderr 可见"
    assert runlog.current_session_id() is None, "此时还没有正式日志"

    # runlog.init 自身失败：同样 stderr + terminate（不递归、不吞退出码）。
    shutdown.reset_for_tests()
    monkeypatch.setattr(production_main, "load_app_config", lambda *a, **k: _mock_cfg(tmp_path))
    monkeypatch.setattr(production_main, "runlog", RunlogRecorder(sink))
    monkeypatch.setattr(runlog, "init",
                        lambda d: (_ for _ in ()).throw(PermissionError("只读盘")))
    with pytest.raises(SystemExit) as excinfo:
        production_main.assemble_production()
    assert excinfo.value.code == 1
    assert "RUNLOG_INIT_FAILED" in capsys.readouterr().err


def test_日志写失败不递归terminate仍退出码1(session_logs: Sink, capsys, monkeypatch):
    def boom(kind, **fields):
        raise OSError("磁盘掉了")

    monkeypatch.setattr(runlog, "event", boom)
    monkeypatch.setattr(production_main, "asr", FakeASR(["分拣草莓"]))
    with pytest.raises(SystemExit) as excinfo:
        run_session(None, FakeArm([]), registry={"strawberry": _NeverBuiltPlan})
    assert excinfo.value.code == 1, "event 抛错 → UNEXPECTED terminate，仍退出码 1"
    captured = capsys.readouterr()
    assert "[TERMINATE] status=UNEXPECTED" in captured.out
    assert "terminate 事件日志写入失败" in captured.err, (
        "terminate 自身日志失败降级 stderr，不递归（§3.4 末段）"
    )


# ---------------------------------------------------------------------------
# startup 哈希与阈值构造（§7 / P0）
# ---------------------------------------------------------------------------


def test_startup事件带三个资源各自的文件哈希(session_logs: Sink):
    cfg = load_app_config(MOCK_APP_PATH, entry="mock")
    log_startup("mock", cfg, MOCK_APP_PATH)
    record = session_logs.sole("startup")
    assert record["entry"] == "mock"
    assert record["app_config_path"] == str(MOCK_APP_PATH)
    for field, path in (
        ("profile", cfg.profile_path),
        ("vision_config", cfg.vision_config_path),
        ("prompt", cfg.prompt_path),
    ):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert record[f"{field}_path"] == str(path)
        assert record[f"{field}_sha256"] == actual, f"{field} 必须是**各自文件**的哈希"
    own = hashlib.sha256(MOCK_APP_PATH.read_bytes()).hexdigest()
    assert record["profile_sha256"] != own, "不得把配置文件哈希当作引用资源哈希（§7）"


def test_thresholds_from_params按P0六字段来源逐一复制():
    params = load_motion_params(MOCK_PROFILE_PATH, mode="mock")
    thresholds = thresholds_from_params(params)
    assert np.array_equal(thresholds.target_bounds_m, params.workspace.target_bounds_m)
    assert np.array_equal(thresholds.object_envelope_m, params.grasp.object_envelope_m)
    assert thresholds.clearance_m == params.collision.clearance_m
    assert thresholds.approach_height_m == params.grasp.approach_height_m
    assert thresholds.table_z_m == params.workspace.table_z_m
    assert thresholds.table_flatness_m == params.workspace.table_flatness_m
    # 数组是副本（VisionThresholds 契约：冻结 + 与来源解耦）
    assert thresholds.target_bounds_m is not params.workspace.target_bounds_m


# ---------------------------------------------------------------------------
# T22 / mock 冒烟 / attempts / 替身（run_mock）
# ---------------------------------------------------------------------------


def _mock_cfg_with_tmp_logs() -> AppConfig:
    """真实 app.mock.json，但 log_dir 改到测试专属目录（不污染仓库 logs/）。"""
    base = load_app_config(MOCK_APP_PATH, entry="mock")
    return replace(base, log_dir=ROOT / "logs" / "mock-pytest-tmp")


def test_T22_mock装配用仿真副本且无真实串口与init(monkeypatch, tmp_path):
    seen_registries: list[Any] = []
    real_preflight = run_mock.preflight
    assert real_preflight is app_config_module.preflight, "与生产同一预检函数"

    def spy_preflight(cfg, params, registry):
        seen_registries.append(registry)
        return real_preflight(cfg, params, registry)

    motor_booms: list[Any] = []

    def boom(self, *a, **k):
        motor_booms.append(self)
        raise AssertionError("mock 入口不得实例化真实驱动（T22）")

    init_probes: list[str] = []
    sink = Sink()
    cfg = replace(_mock_cfg_with_tmp_logs(), log_dir=tmp_path / "mock_logs")
    orig_listen = asr_module.listen_and_transcribe
    orig_select = cloud_module.select_plan
    monkeypatch.setattr(run_mock, "runlog", RunlogRecorder(sink))
    monkeypatch.setattr(run_mock, "preflight", spy_preflight)
    monkeypatch.setattr(Sts3215MotorController, "__init__", boom)
    monkeypatch.setattr(asr_module, "init", lambda c: init_probes.append("asr.init"))
    monkeypatch.setattr(cloud_module, "init", lambda c: init_probes.append("cloud.init"))
    fake_vision = FakeVision([])
    cfg, params, motor, arm = run_mock.assemble(cfg=cfg, vision=fake_vision)

    assert production_main.preflight is app_config_module.preflight, "与生产同一预检函数"
    assert params.status == "simulation" and params.profile_id == "sim_v0_mock_copy"
    assert cfg.profile_path == MOCK_PROFILE_PATH.resolve(), "用 simulation profile 副本"
    assert cfg.mock is not None
    assert seen_registries == [run_mock.MOCK_REGISTRY], "预检与实例化贯穿同一对象"
    assert isinstance(motor, MockMotorController)
    assert len(motor.initialize_calls) == 1, "显式 initialize **恰一次**（P5 §2）"
    assert not motor_booms, "真实串口驱动零构造（探针断言）"
    assert init_probes == [], "真实 asr/cloud init 零调用（D15）"
    # R04：ArmController 三时钟全部接到同一个 SimTime（motor.time 即装配内 SimTime）。
    assert isinstance(motor.time, SimTime)
    assert arm.clock == motor.time.monotonic
    assert arm.sleep == motor.time.sleep
    assert arm.clock_ns == motor.time.monotonic_ns
    # 冷启动在 home：反馈在家 → 恰一次反馈读、零运动命令。
    assert motor.read_count == 1 and motor.command_count == 0
    # 替身接管业务函数（模块属性赋值）；生产 main 与真实 init 未被触碰。
    assert asr_module.listen_and_transcribe is not orig_listen
    assert cloud_module.select_plan is not orig_select
    assert len(fake_vision.configure_calls) == 1 and len(fake_vision.init_calls) == 1
    assert len(sink.of("ready")) == 1
    assert sink.sole("startup")["entry"] == "mock"
    assert isinstance(fake_vision.configure_calls[0]["thresholds"], VisionThresholds)


def test_run_mock_main默认vision桩按契约terminate不回退替身(monkeypatch, tmp_path, capsys):
    """DEC-002：直接跑入口时真实桩抛 VisionHardError → VISION_INIT_FAILED。"""
    sink = Sink()
    entered: list[str] = []
    cfg = replace(_mock_cfg_with_tmp_logs(), log_dir=tmp_path / "mock_smoke_logs")
    monkeypatch.setattr(run_mock, "load_app_config", lambda *a, **k: cfg)
    monkeypatch.setattr(run_mock, "runlog", RunlogRecorder(sink))
    monkeypatch.setattr(run_mock, "run_session",
                        lambda *a, **k: entered.append("session"))
    with pytest.raises(SystemExit) as excinfo:
        run_mock.main()
    assert excinfo.value.code == 1
    assert entered == [], "视觉未就绪绝不进入会话"
    report = _terminate_record(sink)
    assert report["status"] == "VISION_INIT_FAILED"
    assert "视觉未实现" in report["detail"]
    assert sink.of("ready") == [], "桩故障不得洗白成 ready，也不得自动回退替身"


@pytest.fixture(scope="module")
def scenario_arm():
    """模块级一次性构建 MockMotor+ScenarioArmController（placo 建模有成本）。"""
    params = load_motion_params(MOCK_PROFILE_PATH, mode="mock")
    sim = SimTime()
    motor = MockMotorController(params, sim)
    motor.initialize()
    arm = run_mock.ScenarioArmController(
        motor,
        params,
        attempts=[{"contact": "present"}, {"contact": "absent"}],
        clock=sim.monotonic,
        sleep=sim.sleep,
        clock_ns=sim.monotonic_ns,
    )
    return motor, arm


def test_attempts按调用序消费并只设置object_gap_m不替换返回(scenario_arm):
    motor, arm = scenario_arm
    present_gap = float(np.mean(motor.params.gripper.contact_gap_range_m))

    assert arm.scenario_attempts == [{"contact": "present"}, {"contact": "absent"}]
    first = arm.grasp_and_place(None, "strawberry_B_bin")
    assert motor.object_gap_m == present_gap, "present → mean(contact_gap_range_m)"
    second = arm.grasp_and_place(None, "strawberry_B_bin")
    assert motor.object_gap_m is None, "absent → None（空夹可一路闭合）"
    assert arm.attempt_order == ["present", "absent"]
    # 原样 super()：返回是真实控制器对 target=None 的 INVALID_INPUT 拒绝，
    # 包装层没有伪造 SUCCESS。
    assert first.status is GraspStatus.INVALID_INPUT
    assert second.status is GraspStatus.INVALID_INPUT
    assert arm.scenario_attempts == []


def test_attempts耗尽时拒绝场景不默认成功(scenario_arm):
    motor, arm = scenario_arm
    arm.scenario_attempts = []
    motor.object_gap_m = 0.0123  # 哨兵：耗尽路径不得再动开关
    with pytest.raises(SystemExit) as excinfo:
        arm.grasp_and_place(None, "strawberry_B_bin")
    assert excinfo.value.code == 1
    assert motor.object_gap_m == 0.0123, "拒绝发生在设置开关之前"
    motor.object_gap_m = None


def test_install_stubs的替身语义与重复调用拒绝(tmp_path, monkeypatch):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps({"utterances": ["分拣草莓", "你好"], "attempts": []}),
        encoding="utf-8",
    )
    cfg = replace(_mock_cfg(tmp_path), mock=MockSettings(scenario, tmp_path))
    sleeps: list[float] = []
    monkeypatch.setattr(run_mock.time, "sleep", lambda s: sleeps.append(s))

    stubs = run_mock.install_stubs(cfg)
    assert asr_module.listen_and_transcribe() == "分拣草莓"
    assert cloud_module.select_plan("帮我分拣草莓") == "strawberry"
    assert cloud_module.select_plan("随便说点什么") == "invalid"
    assert asr_module.listen_and_transcribe() == "你好"
    # 耗尽：小段真实墙钟 sleep 后返回空串（静音语义，不持续空串忙循环）。
    assert asr_module.listen_and_transcribe() == ""
    assert sleeps and all(0 < s < 5.0 for s in sleeps)
    with pytest.raises(RuntimeError, match="重复调用"):
        run_mock.install_stubs(cfg)
    assert stubs.listen_calls == 3


def test_load_scenario形态校验(tmp_path):
    def scenario_cfg(payload: Any) -> AppConfig:
        path = tmp_path / "s.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return replace(_mock_cfg(tmp_path), mock=MockSettings(path, tmp_path))

    with pytest.raises(run_mock.ScenarioError, match="未知键"):
        run_mock.load_scenario(scenario_cfg(
            {"utterances": [], "attempts": [], "extra": 1}))
    with pytest.raises(run_mock.ScenarioError, match="contact"):
        run_mock.load_scenario(scenario_cfg(
            {"utterances": [], "attempts": [{"contact": "slip"}]}))
    with pytest.raises(run_mock.ScenarioError, match="字符串列表"):
        run_mock.load_scenario(scenario_cfg({"utterances": [1], "attempts": []}))
    ok = run_mock.load_scenario(scenario_cfg(
        {"utterances": ["a"], "attempts": [{"contact": "absent"}]}))
    assert ok == {"utterances": ["a"], "attempts": [{"contact": "absent"}]}


# ---------------------------------------------------------------------------
# T27：生产 K=None 与 MOCK_REGISTRY 贯穿
# ---------------------------------------------------------------------------


def test_T27_生产K为None被预检拒绝而MOCK_REGISTRY贯穿且PLAN_REGISTRY未被修改():
    params = load_motion_params(MOCK_PROFILE_PATH, mode="mock")
    cfg = load_app_config(MOCK_APP_PATH, entry="mock")

    with pytest.raises(PreflightError) as excinfo:
        app_config_module.preflight(cfg, params, PLAN_REGISTRY)
    assert excinfo.value.check == "area_threshold", "生产 K=None → 拒绝运行（§3.2 第 3 条）"

    # 同一 MOCK_REGISTRY 从预检贯穿实例化成功（T22 的 spy 证明贯穿；此处钉"同一
    # 对象 + 生产表未被改动 + 名义 K 只存在于变体类"）。
    app_config_module.preflight(cfg, params, run_mock.MOCK_REGISTRY)
    assert plans_package.PLAN_REGISTRY is production_main.PLAN_REGISTRY
    assert production_main.PLAN_REGISTRY == {"strawberry": StrawberryPlan}
    assert production_main.PLAN_REGISTRY["strawberry"] is StrawberryPlan
    assert StrawberryPlan.AREA_THRESHOLD_M2 is None, "不许把名义 K 写回生产类"
    assert run_mock.MOCK_REGISTRY["strawberry"] is run_mock.MockStrawberryPlan
    assert issubclass(run_mock.MockStrawberryPlan, StrawberryPlan)
    assert run_mock.MockStrawberryPlan.AREA_THRESHOLD_M2 == pytest.approx(0.05 * 0.024)
    plan = run_mock.MOCK_REGISTRY["strawberry"](_home_arm())
    assert isinstance(plan, StrawberryPlan) and plan.AREA_THRESHOLD_M2 > 0
