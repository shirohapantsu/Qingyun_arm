"""P5 §3 离线集成业务链测试（E2E-01–E2E-15，FakeVision 版 mock 全链）。

本文件干什么（P5 §3 表 + 任务.md「使用 FakeVision 复用生产业务逻辑的 mock 链路」）：
    在 **pytest 进程内**用 ``run_mock.assemble(cfg, vision=FakeVision(...))`` 做生产装配，
    再调用**生产** ``main.run_session`` 与**真实** ``plans`` 业务循环，跑通
    语音→方案→视觉→抓取→完成/失败→回 WAITING 的全链业务分支；关键场景
    （E2E-01 / E2E-02R / E2E-04 / E2E-05a / E2E-08b / E2E-10）用真实
    ``ArmController`` + ``tests/mock_motor.MockMotorController``（统一 SimTime 时钟，R04），
    其余场景把运动侧换成 ``FakeArm``（P1 §6.5 的典型返回组合）以聚焦业务分支，
    每条 docstring 标出该条覆盖的业务分支与运动侧形态。

本期 **不切换原生 replay**（任务.md §非视觉范围、视觉接口交接 §4/§8）：
    ``qingyun/grabbing/vision.py`` 仍是契约桩，原生 ``source.mode=replay`` 帧链路归队友。
    因此 E2E-01/09/14 的**视觉部分**标注「队友负责」，本文件只钉业务分支。

规范依据（逐条）：
    docs/实施文档/P5_集成测试与部署验收技术文档.md
        §2（装配契约；仿真 K 与 places「由现有 IK 和真实 ArmController 离线验证目标/
            落点组合可达，不随意填坐标后把 IK_FAILED 当 E2E 成功」；replay 一致性 R04
            两条；attempts 按真实 grasp_and_place 顺序消费、耗尽拒绝；故障开关登记）
        §3（E2E-01–15 场景矩阵与验收命令）
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §5（顶层循环与异常映射）/§6.1–§6.6（计数器、分发顺序、槽位、契约复核、恢复
        矩阵、完成出口）/§7（日志最低字段）/§9（业务语义基准 T01–T32）
    docs/实施文档/README.md 决策 D14（ignore 封顶→terminate）、D15（独立 mock 入口）、
        复审 R04（统一模拟时钟 / 仿真 K / replay 一致性）
    dev_logs/upper开发/视觉接口交接.md §6/§7（FakeVision 用法与注入边界）

替身边界（不冒充）：
    * FakeVision 只做"按脚本返回预设 ``VisionInterface`` / 抛**真实** ``NoTarget``
      ``VisionHardError``"，不做候选筛选、几何、replay（交接 §7）。逐扫描的
      ``vision_scan`` 审计因此在本文件缺失，其验收归队友（P3 §6 第 8 步）。
    * FakeArm 只按脚本返回 ``GraspResult`` 并记录事件顺序（P1 §9 末段），不假装运动
      仿真；凡用它承载的条目，docstring 明确"运动侧=替身"。
    * 真实运动条目一律断言 MockMotor 的物理量（``object_gap_m`` / ``command_count`` /
      关节角 / initialize 调用数），不接受"替身结果冒充原生视觉或运动条目通过"。

跨界改动（详见 dev_logs/upper开发/P5-01_开发日志.md）：本文件**不修改**
scripts/run_mock.py、main.py、tests/support_upper.py。多场景靠 ``assemble(cfg=…)``
注入的 cfg 指向不同 fixture 场景文件；ASR/LLM 用测试内
``monkeypatch.setattr(main, "asr"/"cloud_model", FakeASR/FakeLLM)`` 接管（P1 §2 末段的
注入方式）；``install_stubs`` 的模块属性替换由 ``run_mock.reset_stubs_for_tests()``
逐用例（以及 E2E-10 的同用例二次装配前）恢复。
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
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
    MockSettings,
    PreflightError,
    load_app_config,
)
from configs.common_interface import GraspStatus, HoldingState, PlacePose, VisionInterface
from configs.motion_params import MotionParams, load_motion_params
from main import run_session
from qingyun import runlog, shutdown
from qingyun.grabbing import kinematics_ext
from qingyun.grabbing.kinematics_ext import (
    ArmModel,
    IkNotConverged,
    IkSolver,
    gap_to_gripper_pct,
    grasp_target_pose,
    holding_transform,
    object_pose,
    place_tcp_pose,
)
from qingyun.grabbing.vision import NoTarget
from qingyun.plans import PLAN_REGISTRY, StrawberryPlan
from qingyun.plans import base as base_module
from qingyun.plans.base import MAX_CONSECUTIVE_FAILURES
from scripts import run_mock
from tests.mock_motor import MockMotorController
from tests.support import make_target
from tests.support_upper import FakeArm, FakeASR, FakeLLM, FakeVision, make_result

ROOT = MOCK_APP_PATH.parent.parent
MOCK_DIR = ROOT / "tests" / "fixtures" / "mock"
MOCK_PROFILE_PATH = MOCK_DIR / "profile.json"
MOCK_VISION_PATH = MOCK_DIR / "vision.json"

# Ctrl-C 哨兵：放在 ASR 脚本末尾让会话确定结束（P5 §2「单测注入可结束的输入源，
# 捕获预期 SystemExit；生产会话不增加"跑完即退"」）。
_KI = KeyboardInterrupt("测试注入：会话收到 Ctrl-C")

# 名义 K（与 run_mock.MOCK_AREA_THRESHOLD_M2 同源：0.05×0.024 m²，**非实测**）。
K_AREA_M2 = run_mock.MOCK_AREA_THRESHOLD_M2
_A_SIZE = (0.05, 0.03)     # 0.0015 ≥ K → A
_B_SIZE = (0.05, 0.02)     # 0.0010 < K → B
_C_SIZE = (0.05, 0.03)     # ripe=False → C（不成熟优先，与面积无关）

# ---------------------------------------------------------------------------
# 1. 名义坐标表与场景清单（全部 simulation / 非实测）
# ---------------------------------------------------------------------------
# 目标坐标：x/y/z 落在 mock profile ``workspace.target_bounds_m``
# （[0.325,-0.07,0.02] ~ [0.4,0.07,0.028]）内，z=0.022（≈ table_z + 半高），
# yaw 按 ``tests/support.graspable_target_at`` 的可达流形语义 = atan2(y, x)。
# 落点坐标见 tests/fixtures/mock/profile.json 的 strawberry_* places（同样
# yaw=atan2(y,x)，x/y 落在 workspace.tcp_bounds_m 内）。两者的可达组合由
# ``test_IK_可达性_名义places与目标网格与旧坐标不可达对照`` 钉住（P5 §2）。
TARGET_XY: dict[str, tuple[float, float]] = {
    "t1": (0.34, 0.02),
    "t2": (0.36, -0.04),
    "t3": (0.38, 0.05),
    "t4": (0.34, 0.0),
    "t5": (0.36, -0.02),
    "t6": (0.33, -0.06),
}
# E2E-01 的逐目标分级放置链（(目标键, 期望 grade, 期望 place_id)）。
E2E01_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("t1", "A", "strawberry_A_01"),
    ("t2", "A", "strawberry_A_02"),
    ("t3", "B", "strawberry_B_bin"),
    ("t4", "C", "strawberry_C_bin"),
)
# E2E 全部真实运动用例实际用到的 (目标, 落点) 组合——IK 网格验证的覆盖清单。
REAL_MOTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("t1", "strawberry_A_01"), ("t2", "strawberry_A_01"), ("t2", "strawberry_A_02"),
    ("t3", "strawberry_B_bin"), ("t4", "strawberry_C_bin"), ("t5", "strawberry_A_01"),
    ("t5", "strawberry_A_02"), ("t6", "strawberry_A_03"), ("t1", "strawberry_B_bin"),
    ("t1", "strawberry_C_bin"), ("t1", "strawberry_A_03"),
)

# P5 §2「场景清单登记开关值和触发时刻」：每条真实运动场景使用的 attempts 与
# MockMotor 故障开关（present/absent 由 scenario.json 经 ScenarioArmController 设置；
# 其余开关在测试内按登记的触发点注入）。
PRESENT_GAP_M = 0.026          # 与 profile 的 mean(contact_gap_range_m) 同源，IK 用例核验
_SCENE_MANIFEST: dict[str, dict[str, Any]] = {
    "E2E-01": {
        "scenario": "scenario_e2e01_full.json",
        "attempts": ["present"] * 4,
        "switches": {},
        "trigger": "无额外开关：present → object_gap_m=mean(contact_gap_range_m)=0.026",
    },
    "E2E-02R": {
        "scenario": "scenario_e2e02_miss_limit.json",
        "attempts": ["absent", "absent"],
        "switches": {},
        "trigger": "absent → object_gap_m=None（空夹可一路闭合到空闭合端 → 真实 GRASP_MISS）",
    },
    "E2E-04": {
        "scenario": "scenario_e2e04_post_motion_ik.json",
        "attempts": ["present"],
        "switches": {},
        "trigger": "无 MockMotor 开关：测试内 monkeypatch **该实例** 的 _replan_after_close "
                   "抛真实 IkNotConverged（闭爪之后、LIFT 之前）",
    },
    "E2E-05a": {
        "scenario": "scenario_e2e05_uncertain.json",
        "attempts": ["present"],
        "switches": {"object_gap_m": 0.011},
        "trigger": "闭爪阶段首条"
                   "（gripper 命令相对上一条减小）send_action 时把 object_gap_m 改为 0.011m："
                   "开口低于 contact_gap_range_m 下限、又高于 empty_closed_pct+empty_tol_pct，"
                   "既不判接触也不判空闭合 → gripper.close_timeout_s=6.0（仿真时钟）后 UNKNOWN",
    },
    "E2E-08b": {
        "scenario": "scenario_e2e08_abort.json",
        "attempts": ["present"],
        "switches": {"KeyboardInterrupt": "第 12 次 get_feedback 读"},
        "trigger": "运动已开始（OPEN 阶段小步张开途中）读反馈处抛 KeyboardInterrupt，"
                   "由运动层按既定契约转 ABORTED/FAULT_HOLD",
    },
}


# ---------------------------------------------------------------------------
# 2. 通用设施
# ---------------------------------------------------------------------------

_PARAMS_CACHE: list[MotionParams] = []


def load_mock_params() -> MotionParams:
    """mock 场景 profile（simulation 副本）加载一次、只读复用（placo 建模有成本）。"""
    if not _PARAMS_CACHE:
        _PARAMS_CACHE.append(load_motion_params(MOCK_PROFILE_PATH, mode="mock"))
    return _PARAMS_CACHE[0]


@pytest.fixture
def mock_params() -> MotionParams:
    return load_mock_params()


@pytest.fixture(autouse=True)
def runtime_state():
    """逐用例复位单进程状态：runlog（一进程一日志）、terminate 幂等标志、
    ``install_stubs`` 的模块属性替换与重复调用守卫。

    同进程多次装配彼此隔离的三要素都在这里：
      1. ``runlog.reset_for_tests()`` → 每个用例的 ``assemble`` 都能 init 自己的 tmp
         JSONL（不同 session_id，日志互不串号）；
      2. ``shutdown.reset_for_tests()`` → terminate 可在同一进程内被多次演练；
      3. ``run_mock.reset_stubs_for_tests()`` → 恢复 asr/cloud_model 原函数并解禁守卫，
         下一次装配的 ``install_stubs`` 才能装上新场景。
    ``qingyun.plans.base.vision`` 与 ``main.asr``/``main.cloud_model`` 的替换走
    monkeypatch（用例结束自动还原）。
    """
    _reset_process_state()
    yield
    _reset_process_state()


def _reset_process_state() -> None:
    runlog.reset_for_tests()
    shutdown.reset_for_tests()
    run_mock.reset_stubs_for_tests()


@dataclass
class Assembled:
    """一次进程内装配的产物 + 该用例专属 JSONL 的读取入口。"""

    cfg: AppConfig
    params: MotionParams
    motor: MockMotorController
    arm: run_mock.ScenarioArmController
    vision: FakeVision
    log_path: Path
    move_log: list[str] = field(default_factory=list)

    def events(self) -> list[dict[str, Any]]:
        """读**真实落盘**的 JSONL（不用 sink：断言对象就是生产日志文件本身）。"""
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [row for row in self.events() if row["kind"] == kind]

    def sole(self, kind: str) -> dict[str, Any]:
        found = self.of(kind)
        assert len(found) == 1, f"{kind} 应恰好一条，实际 {len(found)}：{found}"
        return found[0]

    @property
    def kinds(self) -> list[str]:
        return [row["kind"] for row in self.events()]


def scenario_cfg(tmp_path: Path, scenario_file: str, *, utterances: Sequence[str] | None = None,
                 attempts: Sequence[str] | None = None) -> AppConfig:
    """指向某个场景资产的 cfg；``log_dir`` 改到本用例 tmp（不污染仓库 logs/）。

    不传 utterances/attempts 时按文件名读 ``tests/fixtures/mock`` 下已提交的资产；
    传了则在 tmp 里生成同形态场景文件（只用于纯替身驱动、不需要 attempts 的业务用例）。
    两条都不改 ``configs/app.mock.json`` 的已提交内容（P5 §2）。
    """
    base = load_app_config(MOCK_APP_PATH, entry="mock")
    assert base.mock is not None
    if utterances is None and attempts is None:
        script = MOCK_DIR / scenario_file
    else:
        payload = {
            "utterances": list(utterances or []),
            "attempts": [{"contact": c} for c in (attempts or [])],
            "_comment": "测试内生成的同形态场景资产（P5 §2 形态，simulation，非实测）。",
        }
        script = tmp_path / scenario_file
        script.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert script.is_file(), f"场景资产缺失：{script}"
    cfg = replace(base, log_dir=tmp_path / "logs",
                  mock=MockSettings(script, base.mock.vision_replay_dir))
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _spy_primitive(arm: Any, names: Iterable[str], sink: list[str]) -> None:
    """在**实例上**包一层记录（原样转发，不改行为），用于运动命令顺序断言。"""
    for name in names:
        original = getattr(arm, name)

        def wrapper(*args: Any, _name: str = name, _orig: Any = original,
                    **kwargs: Any) -> Any:
            sink.append(_name)
            return _orig(*args, **kwargs)

        setattr(arm, name, wrapper)


def assemble_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, vision: FakeVision,
                  scenario_file: str = "scenario.json",
                  utterances: Sequence[str] | None = None,
                  attempts: Sequence[str] | None = None) -> Assembled:
    """按 P5 §2 装配契约在本进程内装配一套 mock 栈，并注入 FakeVision。

    FakeVision 同时承担两处契约面（交接 §6/§7）：
      * ``assemble(cfg, vision=fake)`` → 装配期的 configure/init（P1 §4 步骤 10）；
      * ``qingyun.plans.base.vision = fake`` → 业务期的 get_target（P1 §2 末段注入方式）。
    两处用**同一个实例**，因此"装配时 configure/init 的那个视觉就是会话在用的视觉"
    可被断言。日志目录固定在本用例 tmp 下。
    """
    cfg = scenario_cfg(tmp_path, scenario_file, utterances=utterances, attempts=attempts)
    loaded_cfg, params, motor, arm = run_mock.assemble(cfg=cfg, vision=vision)
    monkeypatch.setattr(base_module, "vision", vision)
    log_path = runlog.current_log_path()
    assert log_path is not None and log_path.is_file(), "装配必须留下正式 JSONL"
    case = Assembled(cfg=loaded_cfg, params=params, motor=motor, arm=arm, vision=vision,
                     log_path=log_path)
    _spy_primitive(arm, ("move_joints", "reset_fault", "open_gripper"), case.move_log)
    return case


def capturing_registry(instances: list[StrawberryPlan]) -> dict[str, type[StrawberryPlan]]:
    """返回"会记录实例"的注册表：类是 ``MockStrawberryPlan`` 的空子类（K 与全部行为
    照旧继承，生产类与 mock 变体都不改），仅用于取回每任务 Plan 的计数器做断言。"""

    class _SpyPlan(run_mock.MockStrawberryPlan):
        def __init__(self, arm: Any) -> None:  # noqa: ANN001
            super().__init__(arm)
            instances.append(self)

    _SpyPlan.__name__ = "MockStrawberryPlanSpy"
    return {"strawberry": _SpyPlan}


def exploding_registry() -> dict[str, type[StrawberryPlan]]:
    """被要求实例化即失败（E2E-07：invalid/空转写不得创建 Plan）。"""

    class _NeverPlan(run_mock.MockStrawberryPlan):
        def __init__(self, arm: Any) -> None:  # noqa: ANN001
            raise AssertionError("invalid/空转写不得创建 Plan（P1-T01/E2E-07）")

    _NeverPlan.__name__ = "NeverPlan"
    return {"strawberry": _NeverPlan}


def drive_session(monkeypatch: pytest.MonkeyPatch, case: Assembled, *, arm: Any,
                  asr_script: Sequence[Any], llm_script: Sequence[Any],
                  registry: Mapping | None = None) -> tuple[int, FakeASR, FakeLLM]:
    """用 FakeASR/FakeLLM 驱动**生产** ``run_session``，直到出现预期退出。

    返回 ``(退出码, asr, llm)``。FakeASR 脚本耗尽会抛真实 ASRHardError，所以脚本末尾
    一律 ``_KI``（会话级 Ctrl-C → terminate）；"循环比规格多跑一轮"因此必然显式失败。
    """
    asr = FakeASR(list(asr_script))
    llm = FakeLLM(list(llm_script))
    monkeypatch.setattr(production_main, "asr", asr)
    monkeypatch.setattr(production_main, "cloud_model", llm)
    with pytest.raises(SystemExit) as excinfo:
        run_session(case.cfg, arm, registry=registry or run_mock.MOCK_REGISTRY)
    assert excinfo.value.code == 1
    return excinfo.value.code, asr, llm


def forbid_manual_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 plans 的人工输入换成"被调用即失败"的哨兵（不该进人工确认的分支用）。"""
    def _boom(prompt: str) -> str:
        raise AssertionError("该业务分支不得进入人工确认（P1 §6.5）")

    monkeypatch.setattr(base_module, "_input", _boom)


def target_from(key: str, params: MotionParams, *, valid_count: int,
                ripe: bool = True, size: tuple[float, float] = _A_SIZE) -> VisionInterface:
    """按名义坐标表构造一个真实可达的六字段目标（yaw=atan2(y,x) 可达流形）。"""
    x, y = TARGET_XY[key]
    return make_target((x, y, 0.022), math.degrees(math.atan2(y, x)),
                       length_m=size[0], width_m=size[1], ripe=ripe,
                       valid_count=valid_count)


def size_for_grade(grade: str) -> tuple[tuple[float, float], bool]:
    """grade → (尺寸, ripe)：A/B 成熟按面积分界；C 由 ripe=False 优先判出。"""
    if grade == "A":
        return _A_SIZE, True
    if grade == "B":
        return _B_SIZE, True
    return _C_SIZE, False


def nontarget(reason: str = "no_detection") -> NoTarget:
    """真实 ``NoTarget`` 异常实例（P3 §3：ROI 无物体/全部被过滤/跳尽）。"""
    return NoTarget(reason)


def motor_pinch_object_during_close(motor: MockMotorController, gap_m: float) -> dict[str, Any]:
    """在**闭爪阶段首条命令**（gripper 命令第一次减小）把 ``object_gap_m`` 改为 gap_m。

    只包 MockMotor 这个测试驱动自身的 ``send_action``（记录/改自己的字段），不改运动
    判据、不碰控制器，因此返回状态是运动层按既定判据真实算出来的。
    """
    state: dict[str, Any] = {"fired": False, "cmd": None}
    original = motor.send_action
    holder = {"prev": None}

    def hooked(joints_deg: Any, gripper_pct: float) -> None:
        pct = float(gripper_pct)
        prev = holder["prev"]
        holder["prev"] = pct
        if not state["fired"] and prev is not None and pct < prev - 1e-9:
            motor.object_gap_m = float(gap_m)
            state["fired"] = True
            state["cmd"] = pct
        return original(joints_deg, pct)

    motor.send_action = hooked
    return state


def motor_interrupt_on_read(motor: MockMotorController, at_read: int) -> dict[str, int]:
    """让第 ``at_read`` 次反馈读取抛 ``KeyboardInterrupt``（Ctrl-C 抓取中分支）。"""
    original = motor.get_feedback
    counter = {"reads": 0}

    def hooked() -> Any:
        counter["reads"] += 1
        if counter["reads"] == at_read:
            raise KeyboardInterrupt("测试注入：抓取中 Ctrl-C")
        return original()

    motor.get_feedback = hooked  # type: ignore[assignment]
    return counter


# ---------------------------------------------------------------------------
# 3. IK 可达性验证（P5 §2「由现有 IK 和真实 ArmController 离线验证目标/落点组合可达」）
# ---------------------------------------------------------------------------


def _solve_chain(params: MotionParams, model: ArmModel, ik: IkSolver,
                 target: VisionInterface, place: PlacePose) -> dict[str, Any]:
    """复算一次抓放的**关键位姿 IK**（公式与 ``_preplan``/``_build_post_grasp_steps`` 同源）。

    抓取段：T_grasp（物体中心 + 固定偏移，方向 = 长轴 + yaw_offset）→ 上方接近位；
    闭爪开度 g_close 取 present 名义接触开口经 gap_table 反查；持物关系
    ``holding_transform`` 换算放置 TCP 与其上方接近位。每步都要求真实 ``IkSolver.solve``
    收敛（不可达即抛 ``IkNotConverged``），并检查 TCP 落在标定的
    ``workspace.tcp_bounds_m`` 内、接近轴竖直度不超过 ``ik.tilt_tol_deg``——与
    ``ArmController._check_grasp_pose`` 同判据。
    """
    g_open = params.gripper.preopen_pct
    g_close = gap_to_gripper_pct(PRESENT_GAP_M, params.gripper)
    home = np.asarray(params.workspace.home_joints_deg, dtype=np.float64)
    position = np.asarray(target.position, dtype=np.float64)
    p_grasp, _spin, T_grasp = grasp_target_pose(position, float(target.yaw_deg), params)
    bounds = np.asarray(params.workspace.tcp_bounds_m, dtype=np.float64)

    def _within_bounds(T: Any, label: str) -> None:
        p = np.asarray(T[:3, 3], dtype=np.float64)
        assert np.all(p >= bounds[0] - 1e-9) and np.all(p <= bounds[1] + 1e-9), (
            f"{label} TCP {np.round(p, 4).tolist()} 超出 tcp_bounds "
            f"{np.round(bounds, 4).tolist()}")
        tilt = kinematics_ext.approach_tilt_deg(T)
        assert tilt <= params.ik.tilt_tol_deg, f"{label} 接近轴偏离竖直 {tilt:.3f}deg"

    _within_bounds(T_grasp, "抓取")
    T_above = T_grasp.copy()
    T_above[2, 3] += params.grasp.approach_height_m
    q_above = ik.solve(T_above, g_open, home).joints_deg
    ik.solve(T_grasp, g_open, q_above)
    q_close = ik.solve(T_grasp, g_close, q_above).joints_deg
    T_TCP_O = holding_transform(model.fk_tcp(q_close, g_close),
                                object_pose(position, float(target.yaw_deg)))
    T_place = place_tcp_pose(object_pose(np.asarray(place.position, dtype=np.float64),
                                         float(place.yaw_deg)), T_TCP_O)
    _within_bounds(T_place, "放置")
    above_place = T_place.copy()
    above_place[2, 3] += params.grasp.approach_height_m
    ik.solve(above_place, g_close, q_close)
    return {"g_close_pct": g_close, "q_close": np.asarray(q_close, dtype=np.float64),
            "place_tcp": np.asarray(T_place[:3, 3], dtype=np.float64)}


def test_IK_可达性_名义places与目标网格与旧坐标不可达对照(mock_params):
    """P5 §2 反"随意填坐标后把 IK_FAILED 当 E2E 成功"的门禁（纯计算层，快）。

    两半：
      1. E2E 真实运动用到的每个 (目标, 落点) 组合，抓取与放置关键位姿 IK 都必须收敛、
         TCP 在标定域内、接近轴竖直；
      2. **对照**（证明断言有牙）：P5-01 之前那组名义落点（y=-0.15、yaw=0）必须被
         同一判据拒绝——它们越出 ``tcp_bounds_m`` 的 ±0.1 y 半宽；把落点 yaw 改成 0
         （离开 yaw≈atan2(y,x) 流形）同样必须不可达；域外目标同理。若日后有人把
         落点改回那种"看起来对称"的坐标，本用例即失败，而 E2E 会把它们演成 CHECK
         阶段的 PLAN_INVALID（原始坐标实测数据见 P5-01 开发日志 §4）。
    """
    model = ArmModel(mock_params)
    ik = IkSolver(model, mock_params, lambda: None)
    assert PRESENT_GAP_M == pytest.approx(
        float(np.mean(mock_params.gripper.contact_gap_range_m))), "present 名义开口与 profile 同源"

    checked = 0
    for target_key, place_id in REAL_MOTION_PAIRS:
        chain = _solve_chain(mock_params, model, ik,
                             target_from(target_key, mock_params, valid_count=1),
                             mock_params.place(place_id))
        assert np.all(np.isfinite(chain["q_close"]))
        assert 0.0 < chain["g_close_pct"] < 100.0
        checked += 1
    assert checked == len(REAL_MOTION_PAIRS)

    # 对照 1：越出 tcp_bounds y 半宽的旧名义落点（0.30, -0.15, yaw=0）。
    with pytest.raises(AssertionError, match="tcp_bounds"):
        _solve_chain(mock_params, model, ik,
                     target_from("t1", mock_params, valid_count=1),
                     PlacePose(position=np.array([0.3, -0.15, 0.03]), yaw_deg=0.0))
    # 对照 2：位置在域内但 yaw 不在可达流形上（yaw=0，而 atan2(-0.055,0.33)≈-9.46°）。
    with pytest.raises(IkNotConverged):
        _solve_chain(mock_params, model, ik,
                     target_from("t1", mock_params, valid_count=1),
                     PlacePose(position=np.array([0.33, -0.055, 0.03]), yaw_deg=0.0))
    # 对照 3：目标本身越出工作域（x=0.25 < target_bounds 下限，也出 tcp_bounds 边界内缘）。
    far = make_target(np.array([0.25, 0.02, 0.022]), math.degrees(math.atan2(0.02, 0.25)))
    with pytest.raises(AssertionError, match="tcp_bounds|接近轴"):
        _solve_chain(mock_params, model, ik, far, mock_params.place("strawberry_A_01"))

    # 名义落点仍是 simulation 身份、非实测；坐标与 K 都只存在于 mock 资产。
    assert mock_params.status == "simulation" and mock_params.verification is None
    for place_id in ("strawberry_A_01", "strawberry_A_02", "strawberry_A_03",
                     "strawberry_B_bin", "strawberry_C_bin"):
        pose = mock_params.place(place_id)
        x, y = float(pose.position[0]), float(pose.position[1])
        assert 0.30 <= x <= 0.42 and abs(y) <= 0.1, f"{place_id} 名义落点越出 tcp_bounds"
        assert pose.yaw_deg == pytest.approx(math.degrees(math.atan2(y, x)), abs=2e-4), (
            f"{place_id}.yaw_deg 必须等于 atan2(y,x)（可达流形），实际 {pose.yaw_deg}")


# ---------------------------------------------------------------------------
# E2E-01 草莓全链正常：A/B/C 混合多目标逐目标分级放置 + 三连 NoTarget → task_done
# ---------------------------------------------------------------------------


def test_E2E_01_真实臂多目标分级放置全成功_三连NoTarget完成任务完成(
        tmp_path, monkeypatch, capsys, mock_params):
    """E2E-01（FakeVision 版）。

    覆盖业务分支：语音→select_plan→Plan 实例化→分级（A/A/B/C 由预设 ripe+面积给出）→
    ``next_place_id`` 按 D04 顺序消耗 A 槽位→真实 ``grasp_and_place`` SUCCESS→
    NoTarget×3 唯一正常完成出口（D14）→回 WAITING；日志链 assign/grasp_result/
    task_done；attempts（present×4）由 ``ScenarioArmController`` 按真实调用顺序消费，
    MockMotor 的 ``object_gap_m`` 每轮被设为 mean(contact_gap_range_m)。
    运动侧=**真实** ArmController+MockMotor（SimTime 统一时钟，R04）。

    本条视觉部分=**队友负责**：原生 replay 帧编排、真实过滤/排名、逐扫描
    ``vision_scan`` 审计。本测试覆盖的业务分支为上列内容；"三张显式空桌帧"在替身侧
    由三个真实 ``NoTarget`` 事件承载。R04 一致性由脚本承担：每个目标只出现一次、
    ``valid_count`` 逐轮递减（抓走一个少一个），见下方断言。
    """
    manifest = _SCENE_MANIFEST["E2E-01"]
    total = len(E2E01_CHAIN)
    script: list[Any] = []
    for index, (target_key, grade, _place) in enumerate(E2E01_CHAIN):
        size, ripe = size_for_grade(grade)
        script.append(target_from(target_key, mock_params, valid_count=total - index,
                                  ripe=ripe, size=size))
    script += [nontarget("no_detection")] * 3
    fake = FakeVision(script)
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file=manifest["scenario"])
    assert case.arm.scenario_attempts == [{"contact": "present"}] * total
    code, asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                    asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1

    # --- 业务链：逐目标 place_id 与 grade 对应、计数与审计链 ---
    assigns = case.of("assign")
    assert [a["place_id"] for a in assigns] == [p for _, _, p in E2E01_CHAIN]
    assert [a["grade"] for a in assigns] == [g for _, g, _ in E2E01_CHAIN]
    assert [a["a_count"] for a in assigns] == [1, 2, 2, 2], "A 分配即消耗（D04），B/C 不消耗"
    assert [a["valid_count"] for a in assigns] == [4, 3, 2, 1], "R04：抓走一个少一个"
    assert [a["ignore_used"] for a in assigns] == [0, 0, 0, 0]
    assert [a["attempt_id"] for a in assigns] == [1, 2, 3, 4]
    assert [a["scan_id"] for a in assigns] == [1, 2, 3, 4]
    results = case.of("grasp_result")
    assert [r["status"] for r in results] == ["SUCCESS"] * 4
    assert [r["place_id"] for r in results] == [p for _, _, p in E2E01_CHAIN]
    assert [r["grade"] for r in results] == [g for _, g, _ in E2E01_CHAIN]
    assert all(r["recovery_required"] is False and r["stage"] == "DONE" for r in results)
    assert case.of("recovery") == [], "全成功路径不得出现恢复"
    done = case.sole("task_done")
    assert done["reason"] == "NO_TARGET_STREAK"
    assert done["success"] == 4 and done["failure_counts"] == {}
    assert done["a_slots_used"] == 2 and done["no_target_streak"] == 3
    assert done["attempts"] == 4 and done["ignore"] == 0 and done["consecutive_failures"] == 0
    # 唯一的 terminate 是会话末尾注入的 Ctrl-C；任务本身以 NoTarget×3 正常返回（D14）。
    terminate = case.sole("terminate")
    assert terminate["status"] == "KEYBOARD_INTERRUPT"
    assert "task_instance_id" not in terminate, "任务级 terminate 不得出现"

    # --- 视觉/会话侧 ---
    assert fake.ignore_values == [0] * 7, "4 次目标 + 3 次 NoTarget，各一次真实扫描"
    assert fake.remaining == 0, "脚本恰好用完（多跑一轮即以 VisionHardError 显式失败）"
    assert len(asr.listen_calls) == 2
    assert case.move_log == [], "任务链内不得出现回程/复位运动"

    # --- 真实运动侧物理量 ---
    assert case.arm.attempt_order == manifest["attempts"], "attempts 按真实调用顺序消费"
    assert case.motor.object_gap_m == pytest.approx(PRESENT_GAP_M)
    assert len(case.motor.initialize_calls) == 1, "mock 构造不自动初始化，装配显式调用一次"
    assert case.motor.command_count > 100, "真实回放产生了大量舵机指令（非替身短路）"
    home = np.asarray(mock_params.workspace.home_joints_deg, dtype=np.float64)
    tol = np.asarray(mock_params.motion.settle_position_tol_deg, dtype=np.float64)
    assert np.all(np.abs(np.asarray(case.motor.q, dtype=np.float64) - home) <= tol), (
        "每次成功由 RETURN 段回 home（任务结束时在家）")
    terminal = capsys.readouterr().out
    assert "任务完成：结束原因=NO_TARGET_STREAK，成功=4" in terminal


# ---------------------------------------------------------------------------
# E2E-02 GRASP_MISS 序列 + ignore 递增 + 逐次自动恢复 + 第 V 次 D14 封顶
# ---------------------------------------------------------------------------


def test_E2E_02_D14封顶_ignore递增逐次自动恢复_第V次终止后零恢复零运动(
        tmp_path, monkeypatch, mock_params):
    """E2E-02。

    覆盖业务分支：GRASP_MISS（重试类）→ ``ignore`` 与 ``consecutive_failures`` 逐次
    +1 → §6.5 自动恢复 ``reset_fault → safe_waypoints(列表顺序) → home`` → 下一轮视觉
    （D03 临时跳选）；第 V 次失败使 ``ignore ≥ valid_count`` → D14
    ``terminate("NO_GRASPABLE_TARGET")``：当次起不恢复、不人工确认、不回 home、
    零新运动命令（P1 §6.2 第 5 步、§9 T26 的全链版）。V=2 由 FakeVision 事件给出，
    R04：抓空后目标仍留在原帧里，故第 2 次扫描返回**同一目标**、valid_count 不变。
    运动侧=``FakeArm``（典型组合 GRASP_MISS/FAULT_HOLD/EMPTY/recovery_required=True）；
    真实运动侧的 GRASP_MISS 与恢复路线见 ``test_E2E_02R_...``。
    """
    same = target_from("t1", mock_params, valid_count=2, size=_A_SIZE)
    fake = FakeVision([same, same, nontarget()])
    miss = make_result(GraspStatus.GRASP_MISS)
    arm = FakeArm([miss, miss], params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    forbid_manual_confirm(monkeypatch)
    code, asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                     asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert len(asr.listen_calls) == 1, "封顶 terminate 后会话不再推进"

    assigns = case.of("assign")
    assert [a["place_id"] for a in assigns] == ["strawberry_A_01", "strawberry_A_02"]
    assert [a["ignore_used"] for a in assigns] == [0, 1], "D03 跳选游标逐次递增"
    results = case.of("grasp_result")
    assert [r["status"] for r in results] == ["GRASP_MISS", "GRASP_MISS"]
    assert [r["attempt_id"] for r in results] == [1, 2]
    assert [r["stage"] for r in results] == ["FAULT_HOLD", "FAULT_HOLD"]
    recoveries = case.of("recovery")
    assert len(recoveries) == 1, "第 V 次（第 2 次）封顶不恢复"
    assert recoveries[0]["mode"] == "auto" and recoveries[0]["reset"] == "ok"
    assert recoveries[0]["home"] == "ok"
    assert recoveries[0]["waypoint_moves"] == len(mock_params.workspace.safe_waypoints_deg)
    assert recoveries[0]["attempt_id"] == 1
    terminate = case.sole("terminate")
    assert terminate["status"] == "NO_GRASPABLE_TARGET"
    assert "ignore=2" in terminate["detail"] and "valid_count=2" in terminate["detail"]
    assert "末次 GRASP_MISS/FAULT_HOLD" in terminate["detail"]
    assert case.of("task_done") == [], "D14 封顶不调 _finish（P1 §6.6）"

    # 顺序与"当次起零运动"：reset→wp→wp→home 之后第二次失败不再有任何命令。
    wp = [tuple(float(v) for v in row) for row in mock_params.workspace.safe_waypoints_deg]
    home = tuple(float(v) for v in mock_params.workspace.home_joints_deg)
    assert arm.command_sequence() == ["grasp_and_place", "reset_fault", "move_joints",
                                      "move_joints", "move_joints", "grasp_and_place"]
    assert arm.moved_positions() == [wp[0], wp[1], home], "恢复路线按列表顺序 + home（§6.5 冻结）"
    assert arm.count_of("reset_fault") == 1
    assert arm.place_ids() == ["strawberry_A_01", "strawberry_A_02"]
    assert fake.ignore_values == [0, 1]


def test_E2E_02R_真实臂attempts消费链_真实GRASP_MISS与恢复路线的集成发现(
        tmp_path, monkeypatch, mock_params):
    """E2E-02 的真实运动版（attempts 消费链必须走 ScenarioArmController + scenario.json）。

    覆盖：``scenario_e2e02_miss_limit.json`` 的 absent 被按真实调用顺序消费
    （``object_gap_m=None``）；运动层**真实**返回 GRASP_MISS（不是伪造 SUCCESS、不是
    替身）；plan 审计后按 §6.5 进入自动恢复 ``reset_fault → safe_waypoints[0]``。

    与 E2E-02 的差异（P5-01 开发日志登记的集成发现）：在当前 simulation profile 下，
    空闭合位姿的 ``moving_pad`` 胶囊已吃到桌面判定阈值之下（穿透约 10.9mm，阈值约
    13.2mm 含误差膨胀/平度/净距余量），因此"从真实空抓位出发的一跳关节运动"必然
    CollisionViolation → 自动恢复走不完 → 第 V 次 D14 封顶在真实臂上不可达。逐次恢复
    的顺序与封顶后零运动由上一条 FakeArm 用例承载；本条只钉真实侧事实：attempts
    消费、真实返回、恢复入口与 §6.5 末行 ``RECOVERY_FAILED``（不追加 ignore 再试恢复），
    不削弱、不冒充。
    """
    manifest = _SCENE_MANIFEST["E2E-02R"]
    fake = FakeVision([target_from("t5", mock_params, valid_count=2)])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file=manifest["scenario"])
    assert case.arm.scenario_attempts == [{"contact": "absent"}] * 2
    code, asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                    asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert len(asr.listen_calls) == 1
    assert case.arm.attempt_order == ["absent"], "第 1 次 attempts 已按真实调用消费"
    assert case.arm.scenario_attempts == [{"contact": "absent"}], "剩余 1 条未被吞掉"
    assert case.motor.object_gap_m is None, "absent → object_gap_m=None（MockMotor 真实字段）"

    results = case.of("grasp_result")
    assert len(results) == 1
    row = results[0]
    assert row["status"] == "GRASP_MISS" and row["stage"] == "FAULT_HOLD"
    assert row["holding"] == HoldingState.EMPTY.value
    assert row["recovery_required"] is True
    assert row["place_id"] == "strawberry_A_01" and row["grade"] == "A"
    assert row["valid_count"] == 2 and row["ignore_used"] == 0
    recovery = case.sole("recovery")
    assert recovery["mode"] == "auto"
    assert recovery["failed_step"] == "move_joints(safe_waypoints_deg[0])"
    assert "碰桌" in recovery["error"]
    terminate = case.sole("terminate")
    assert terminate["status"] == "RECOVERY_FAILED"
    assert "不追加 ignore 再试恢复" in terminate["detail"]
    assert case.of("task_done") == []
    assert case.move_log == ["reset_fault", "move_joints"], "只做了 reset + 第一跳即退出"
    assert case.vision.scan_count == 1, "恢复失败后不再请求视觉"


# ---------------------------------------------------------------------------
# E2E-03 V>N 连续失败 → 失败预算耗尽（无第 10 次恢复）
# ---------------------------------------------------------------------------


def test_E2E_03_连续十次可恢复失败触发失败预算耗尽_无第十次恢复(tmp_path, monkeypatch, mock_params):
    """E2E-03（P1-T05 的全链版）。

    覆盖业务分支：V=12 > N=10，连续 10 次 GRASP_MISS 使 ``ignore`` 递增到 10 仍不封顶
    （10 < 12，第 5 步不触发）→ 第 6 步预算 ``consecutive_failures ≥ 10`` 触发
    ``terminate("FAILURE_BUDGET_EXHAUSTED")``，**无第 10 次恢复**、臂冻结。
    "封顶先于预算"的次序由 valid_count=12 与同一次失败共同咬合（P1 §6.2 第 5/6 步）。
    运动侧=``FakeArm``；目标全为 B 级（不消耗 A 槽位，避免与 E2E-11 混因）。
    """
    limit = MAX_CONSECUTIVE_FAILURES
    keys = ["t1", "t2", "t3", "t4", "t5", "t6", "t1", "t2", "t3", "t4"]
    assert len(keys) == limit
    fake = FakeVision([target_from(k, mock_params, valid_count=limit + 2, size=_B_SIZE)
                       for k in keys])
    arm = FakeArm([make_result(GraspStatus.GRASP_MISS)] * limit, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert len(case.of("grasp_result")) == limit
    assert len(case.of("recovery")) == limit - 1, "无第 10 次恢复"
    assert [a["ignore_used"] for a in case.of("assign")] == list(range(limit))
    assert [a["place_id"] for a in case.of("assign")] == ["strawberry_B_bin"] * limit
    terminate = case.sole("terminate")
    assert terminate["status"] == "FAILURE_BUDGET_EXHAUSTED"
    assert f"consecutive_failures={limit}" in terminate["detail"]
    assert "ignore=10/valid_count=12" in terminate["detail"], "封顶未触发才轮到预算"
    assert '失败码计数={"GRASP_MISS":10}' in terminate["detail"]
    assert case.of("task_done") == []
    assert arm.count_of("reset_fault") == limit - 1
    assert fake.scan_count == limit


# ---------------------------------------------------------------------------
# E2E-04 运动后 IK_FAILED（D08）→ 立即 terminate，JSONL 含冻结位记录
# ---------------------------------------------------------------------------


def test_E2E_04_运动后IK失败按D08终止_臂冻结于失败位且零恢复(tmp_path, monkeypatch, mock_params):
    """E2E-04。

    覆盖业务分支：抓取**运动已开始**之后的规划失败（IK_FAILED 且
    ``recovery_required=True``）→ §6.5 D08 行一律 ``terminate("IK_FAILED")``：不区分
    holding、不自动恢复、不人工确认、不回 home，臂冻结于失败位；审计先于终止（§6.2
    第 0 步），JSONL 的 grasp_result/terminate 两行都带失败现场。

    构造方式与差异（开发日志登记）：``_preplan`` 已按 ``contact_gap_range_m`` **两端**
    预规划并校验整条放置路线，而真实闭爪确认的开度必然落在该区间内（present 名义
    0.026→约 42%），所以"仅闭爪后才不可达"的落点在本仿真里无法有机构造（试过：改
    落点 → CHECK 阶段就 PLAN_INVALID/出域；持物关系由 FK 决定，无独立开关）。这里在
    **该控制器实例**上把闭爪后的 ``_replan_after_close`` 换成抛真实 ``IkNotConverged``
    ——测试内 monkeypatch，不改运动模块文件、不改任何判据；其余全走真实代码：真实
    夹爪接触确认、真实 ``_early_fail`` 分派、真实 ``_fault_result`` 保持提交
    （FAULT_HOLD）与真实实例锁存。
    """
    manifest = _SCENE_MANIFEST["E2E-04"]
    fake = FakeVision([target_from("t1", mock_params, valid_count=3, size=_A_SIZE)])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file=manifest["scenario"])
    forbid_manual_confirm(monkeypatch)
    arm = case.arm

    def failing(*args: Any, **kwargs: Any):
        raise IkNotConverged("测试注入：闭爪后重规划不收敛（E2E-04 构造）")

    monkeypatch.setattr(arm, "_replan_after_close", failing)
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert case.arm.attempt_order == ["present"] and case.arm.scenario_attempts == []

    row = case.sole("grasp_result")
    assert row["status"] == "IK_FAILED" and row["stage"] == "FAULT_HOLD"
    assert row["recovery_required"] is True
    assert row["holding"] == HoldingState.HOLDING.value, "真实闭爪已建立接触之后才失败"
    assert row["place_id"] == "strawberry_A_01" and row["grade"] == "A"
    assert row["attempt_id"] == 1 and row["valid_count"] == 3 and row["ignore_used"] == 0
    assert "E2E-04 构造" in row["reason"]
    terminate = case.sole("terminate")
    assert terminate["status"] == "IK_FAILED"
    assert "D08" in terminate["detail"] and "recovery_required=True" in terminate["detail"]
    assert "FAULT_HOLD" in terminate["detail"]
    assert case.of("recovery") == [], "D08 行不进恢复矩阵其余分支"
    assert case.of("task_done") == []
    # 冻结位记录：terminate 行的 stage 上下文 + 关节停在失败现场（未回零、无追加命令）。
    assert terminate["stage"]["task_instance_id"] == 1
    assert terminate["stage"]["scan_id"] == 1
    home = np.asarray(mock_params.workspace.home_joints_deg, dtype=np.float64)
    frozen = np.asarray(case.motor.q, dtype=np.float64)
    assert np.max(np.abs(frozen - home)) > 5.0, "臂停在失败位，不在 home"
    assert terminate["detail"].count("stage=FAULT_HOLD") == 1
    assert case.move_log == [], "无 reset_fault、无回程 move_joints、无 open_gripper"
    assert case.vision.scan_count == 1


# ---------------------------------------------------------------------------
# E2E-05 GRASP_UNCERTAIN（+ stdin 回车）→ 恢复继续
# ---------------------------------------------------------------------------


def test_E2E_05a_真实臂闭爪超时返回GRASP_UNCERTAIN并进入人工确认(
        tmp_path, monkeypatch, capsys, mock_params):
    """E2E-05 的真实运动版（构造登记见 _SCENE_MANIFEST["E2E-05a"]）。

    覆盖业务分支：真实运动层在闭爪阶段得到 ``holding=UNKNOWN`` →
    ``GRASP_UNCERTAIN``（重试类、recovery_required=True、stage=FAULT_HOLD）→ §6.5
    人工行：打印现场状态并 ``input()`` 阻塞等待确认（D13 回车即确认）。构造用
    MockMotor 真实故障开关 ``object_gap_m=0.011``（触发时刻=闭爪首条命令），既不判
    接触也不判空闭合，``gripper.close_timeout_s`` 到点后如实返回 UNKNOWN——本测试不
    伪造该状态。

    恢复路线走完（reset→中转→home→下一轮）由 ``test_E2E_05b_...`` 承载：本 profile
    下从这种低位闭爪姿态出发的一跳回程会被桌面碰撞校验拒绝（同 E2E-02R 的集成发现），
    所以本条把"人工确认被真实结果触发"这一段钉住，确认之后按 §6.5 末行
    ``RECOVERY_FAILED`` 终止，不追加恢复。
    """
    manifest = _SCENE_MANIFEST["E2E-05a"]
    fake = FakeVision([target_from("t1", mock_params, valid_count=3, size=_B_SIZE)])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file=manifest["scenario"])
    prompts: list[str] = []

    def confirm(prompt: str) -> str:
        prompts.append(prompt)
        return ""                      # D13：回车即确认（空回车同样算确认）

    monkeypatch.setattr(base_module, "_input", confirm)
    switch = motor_pinch_object_during_close(case.motor, 0.011)
    code, _asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert switch["fired"] is True, "故障开关按登记的触发点（闭爪首条命令）生效"
    row = case.sole("grasp_result")
    assert row["status"] == "GRASP_UNCERTAIN" and row["stage"] == "FAULT_HOLD"
    assert row["holding"] == HoldingState.UNKNOWN.value
    assert row["recovery_required"] is True
    assert row["grade"] == "B" and row["place_id"] == "strawberry_B_bin"
    assert len(prompts) == 1 and "回车" in prompts[0]
    out = capsys.readouterr().out
    assert "[需要人工确认] status=GRASP_UNCERTAIN" in out
    assert "holding=UNKNOWN" in out
    terminate = case.sole("terminate")
    assert terminate["status"] == "RECOVERY_FAILED"
    recovery = case.sole("recovery")
    assert recovery["mode"] == "manual", "人工确认后将走 reset→中转→home"
    assert recovery["status"] == "GRASP_UNCERTAIN" and recovery["holding"] == "UNKNOWN"
    assert recovery["failed_step"] == "move_joints(safe_waypoints_deg[0])"
    assert case.move_log == ["reset_fault", "move_joints"], "确认后才做 reset + 第一跳"
    assert case.of("task_done") == []
    assert case.vision.scan_count == 1


def test_E2E_05b_人工确认回车后恢复继续到任务完成(tmp_path, monkeypatch, mock_params):
    """E2E-05 的业务完整版（运动侧=``FakeArm``，返回与 E2E-05a 同形状的典型结果）。

    覆盖业务分支：GRASP_UNCERTAIN → 回车确认 → ``reset_fault → 逐个中转位 → home`` →
    下一轮视觉（ignore 已 +1、D03）→ NoTarget×3 正常完成；确认期间不请求视觉（同步
    阻塞天然保证）；恢复后 ``failure_counts`` 保留该次失败（只有 SUCCESS 清零，D02）。
    """
    fake = FakeVision([target_from("t2", mock_params, valid_count=2, size=_B_SIZE),
                       nontarget(), nontarget(), nontarget()])
    arm = FakeArm([make_result(GraspStatus.GRASP_UNCERTAIN)], params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    prompts: list[str] = []
    monkeypatch.setattr(base_module, "_input",
                        lambda prompt: (prompts.append(prompt), "")[1])
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert len(prompts) == 1
    # 恢复后的下一轮带 ignore=1 请求（D03 临时跳选），NoTarget 三连各自把 ignore 归零。
    assert fake.ignore_values == [0, 1, 0, 0]
    assert [arm.calls_of("grasp_and_place")[0].args[1]] == ["strawberry_B_bin"]
    recovery = case.sole("recovery")
    assert recovery["mode"] == "manual" and recovery["confirm"] == "manual_confirmed"
    assert recovery["reset"] == "ok" and recovery["home"] == "ok"
    wp = [tuple(float(v) for v in row) for row in mock_params.workspace.safe_waypoints_deg]
    home = tuple(float(v) for v in mock_params.workspace.home_joints_deg)
    assert arm.command_sequence() == ["grasp_and_place", "reset_fault", "move_joints",
                                      "move_joints", "move_joints"]
    assert arm.moved_positions() == [wp[0], wp[1], home]
    done = case.sole("task_done")
    assert done["reason"] == "NO_TARGET_STREAK" and done["success"] == 0
    assert done["failure_counts"] == {"GRASP_UNCERTAIN": 1}
    assert done["ignore"] == 0 and done["no_target_streak"] == 3
    assert done["consecutive_failures"] == 0, "NoTarget 不清零失败预算，但 ignore 归零"
    assert case.of("terminate") == []


# ---------------------------------------------------------------------------
# E2E-06 连续两个任务：两实例、计数归零、第二任务新目标序列
# ---------------------------------------------------------------------------


def test_E2E_06_连续两个任务两实例计数归零第二任务新目标序列(tmp_path, monkeypatch, mock_params):
    """E2E-06。

    覆盖业务分支：每任务新建 Plan 实例（§5）、四个计数器与成功/失败统计从零重记
    （§6.1 第一行）、``task_instance_id`` session 内单调 +1、第二任务获得**新的**目标
    序列（R04：两批独立事件，不靠脚本/帧耗尽冒充连续任务通过——脚本剩余为 0 且
    两次任务的坐标集合互不相交）。运动侧=``FakeArm``（全 SUCCESS，聚焦计数与序列）。
    """
    batch1 = [target_from("t1", mock_params, valid_count=2),
              target_from("t2", mock_params, valid_count=1),
              nontarget(), nontarget(), nontarget()]
    batch2 = [target_from("t3", mock_params, valid_count=2),
              target_from("t4", mock_params, valid_count=1),
              nontarget(), nontarget(), nontarget()]
    fake = FakeVision(batch1 + batch2)
    arm = FakeArm([make_result(GraspStatus.SUCCESS)] * 4, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓", "分拣草莓"], attempts=[])
    instances: list[StrawberryPlan] = []
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", "分拣草莓", _KI],
                                      llm_script=["strawberry", "strawberry"],
                                      registry=capturing_registry(instances))
    assert code == 1
    assert len(instances) == 2 and instances[0] is not instances[1]
    first, second = instances
    assert [p.scan_id for p in instances] == [5, 5], "scan_id 每任务从 0 起重记"
    assert [p.attempt_id for p in instances] == [2, 2]
    assert [p.a_count for p in instances] == [2, 2], "D04 槽位计数按任务独立"
    assert [p.no_target_streak for p in instances] == [3, 3]
    assert [p.ignore for p in instances] == [0, 0]
    assert [p.consecutive_failures for p in instances] == [0, 0]
    assert [p.success_count for p in instances] == [2, 2]
    assert first.failure_counts == {} and second.failure_counts == {}
    assert fake.remaining == 0 and fake.scan_count == 10, "两批事件各 5 条，全部真消费"
    grabbed = [tuple(np.asarray(c.args[0].position, float).tolist())
               for c in arm.calls_of("grasp_and_place")]
    assert len(set(grabbed)) == 4, "四个目标位置互不相同（第二任务不是复用第一任务）"
    assert arm.place_ids() == ["strawberry_A_01", "strawberry_A_02",
                               "strawberry_A_01", "strawberry_A_02"]
    done = case.of("task_done")
    assert [d["task_instance_id"] for d in done] == [1, 2]
    assert [d["scan_id"] for d in done] == [5, 5]
    assert [d["a_slots_used"] for d in done] == [2, 2]
    # 唯一的 terminate 是会话级 Ctrl-C（任务本身全部走 NoTarget×3 正常出口）。
    terminate = case.sole("terminate")
    assert terminate["status"] == "KEYBOARD_INTERRUPT"
    assert "task_instance_id" not in terminate, "会话级退出不在任务上下文里"


# ---------------------------------------------------------------------------
# E2E-07 invalid 语音 / 空转写：回 WAITING 不建 Plan
# ---------------------------------------------------------------------------


def test_E2E_07_invalid与空转写回WAITING不创建Plan(tmp_path, monkeypatch, mock_params):
    """E2E-07（P1-T01 的 mock 全链版）。

    覆盖业务分支：空转写 continue、``invalid`` continue、``task_instance_id`` 不被这些
    轮次消耗、**不创建 Plan**、不触视觉、不触抓取；会话继续回 WAITING 等下一条口令。
    注册表换成"实例化即失败"的探针，FakeVision/FakeArm 脚本为空——被调用即抛真实
    异常，等于"多调一次就失败"。
    """
    fake = FakeVision([])
    arm = FakeArm([], params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=[], attempts=[])
    code, asr, llm = drive_session(monkeypatch, case, arm=arm,
                                   asr_script=["", "   ", "今天天气不错", "帮我拿个杯子", _KI],
                                   llm_script=["invalid", "invalid"],
                                   registry=exploding_registry())
    assert code == 1
    assert len(asr.listen_calls) == 5, "4 条转写各一次 + 第 5 次 Ctrl-C"
    assert len(llm.select_calls) == 2, "空转写不提交云请求（P1 §5）"
    assert [c["text"] for c in llm.select_calls] == ["今天天气不错", "帮我拿个杯子"]
    assert fake.scan_count == 0 and arm.events == []
    assert case.of("assign") == [] and case.of("task_done") == []
    assert [r["text"] for r in case.of("asr_result")] == ["", "   ", "今天天气不错",
                                                           "帮我拿个杯子"]
    assert [r["task"] for r in case.of("llm_result")] == ["invalid", "invalid"]
    rows = case.events()
    assert not any("task_instance_id" in row for row in rows), "无效口令不得占用任务编号"
    terminate = case.sole("terminate")
    assert terminate["status"] == "KEYBOARD_INTERRUPT"
    assert fake.configure_calls and fake.init_calls, "装配期的 configure/init 仍照常发生"


# ---------------------------------------------------------------------------
# E2E-08 Ctrl-C 三分支
# ---------------------------------------------------------------------------


def test_E2E_08a_WAITING阶段CtrlC走terminate统一退出(tmp_path, monkeypatch):
    """E2E-08(a)：WAITING/ASR 阶段 Ctrl-C → ``terminate("KEYBOARD_INTERRUPT")``，
    零追加运动命令（臂在 home）。运动侧=真实 ArmController+MockMotor（装配后零动作）。
    """
    fake = FakeVision([])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file="scenario_e2e08_abort.json")
    commands = case.motor.command_count
    code, _asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                      asr_script=[_KI], llm_script=[])
    assert code == 1
    terminate = case.sole("terminate")
    assert terminate["status"] == "KEYBOARD_INTERRUPT"
    assert case.motor.command_count == commands == 0, "装配到 WAITING 之间零舵机指令"
    assert case.move_log == []
    assert fake.scan_count == 0 and case.of("task_done") == []


def test_E2E_08b_抓取中CtrlC经ABORTED分发终止(tmp_path, monkeypatch, mock_params):
    """E2E-08(b)：抓取**运动中** Ctrl-C。

    真实 ``ArmController`` 把 ``KeyboardInterrupt`` 按既定契约转成
    ``ABORTED``/``FAULT_HOLD``/``recovery_required=True``（运动文档 4.3），plan 按
    §6.2 第 3 步"非重试类以状态名直死" → ``terminate("ABORTED")``：不恢复、不回 home、
    当次起零追加运动命令。Ctrl-C 注入点=MockMotor 第 12 次反馈读（登记于
    _SCENE_MANIFEST），此时运动已开始。
    """
    manifest = _SCENE_MANIFEST["E2E-08b"]
    fake = FakeVision([target_from("t1", mock_params, valid_count=2)])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file=manifest["scenario"])
    counter = motor_interrupt_on_read(case.motor, 12)
    forbid_manual_confirm(monkeypatch)
    code, _asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert counter["reads"] >= 12, "注入点确实落在运动中"
    row = case.sole("grasp_result")
    assert row["status"] == "ABORTED" and row["stage"] == "FAULT_HOLD"
    assert row["recovery_required"] is True
    assert "KeyboardInterrupt" in row["reason"]
    terminate = case.sole("terminate")
    assert terminate["status"] == "ABORTED"
    assert case.of("recovery") == [] and case.of("task_done") == []
    assert case.move_log == [], "中止后不做任何恢复运动"
    assert fake.scan_count == 1


def test_E2E_08c_恢复确认中CtrlC走MANUAL_CONFIRM_ABORTED(tmp_path, monkeypatch, mock_params):
    """E2E-08(c)：``input()`` 阻塞中 Ctrl-C（P1 §5 表第三行）→
    ``terminate("MANUAL_CONFIRM_ABORTED")``，且不 reset、不搬臂（T16 的全链版）。
    运动侧=``FakeArm``（真实形状的 GRIP_SLIP 结果）。
    """
    fake = FakeVision([target_from("t6", mock_params, valid_count=2, size=_A_SIZE)])
    arm = FakeArm([make_result(GraspStatus.GRIP_SLIP)], params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    monkeypatch.setattr(base_module, "_input",
                        lambda prompt: (_ for _ in ()).throw(
                            KeyboardInterrupt("测试注入：确认中 Ctrl-C")))
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    terminate = case.sole("terminate")
    assert terminate["status"] == "MANUAL_CONFIRM_ABORTED"
    assert "未执行任何复位或回程" in terminate["detail"]
    assert arm.count_of("grasp_and_place") == 1
    assert arm.count_of("reset_fault") == 0 and arm.count_of("move_joints") == 0
    assert case.of("recovery") == [] and case.of("task_done") == []


# ---------------------------------------------------------------------------
# E2E-09 NoTarget 链（FakeVision 等价；原生 replay 耗尽归队友）
# ---------------------------------------------------------------------------


def test_E2E_09_NoTarget三连正常完成_真实扫描计数排除耗尽冒充(tmp_path, monkeypatch, mock_params):
    """E2E-09（FakeVision 等价版）。

    覆盖业务分支：``NoTarget`` → ignore 归零、``no_target_streak`` +1、三连为唯一正常
    完成出口（D14）；三连之间不 sleep、每次都是真实新扫描（P1 §6.2 末段、T03）；
    全程零抓取调用。

    本条视觉部分=**队友负责**：原生 ``replay_exhausted``（帧目录读尽后由真实管线抛
    ``NoTarget(skip_exhausted)``）依赖 P3 replay 资产。本测试覆盖的业务分支为 NoTarget
    计数、完成出口与零运动调用。
    """
    fake = FakeVision([nontarget("no_detection"), nontarget("no_detection"),
                       nontarget("skip_exhausted")])
    arm = FakeArm([], params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    wall0 = time.monotonic()
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI],
                                      llm_script=["strawberry"])
    assert code == 1
    assert time.monotonic() - wall0 < 10.0, "三连之间不引入 sleep（T03）"
    assert fake.ignore_values == [0, 0, 0] and fake.scan_count == 3 and fake.remaining == 0
    assert arm.events == [], "全程不触运动"
    assert case.of("assign") == [] and case.of("grasp_result") == []
    done = case.sole("task_done")
    assert done["reason"] == "NO_TARGET_STREAK" and done["success"] == 0
    assert done["no_target_streak"] == 3 and done["attempts"] == 0
    assert done["failure_counts"] == {} and done["a_slots_used"] == 0
    # 三连之后任务正常返回、会话继续回 WAITING，直到注入的 Ctrl-C 才 terminate。
    assert case.sole("terminate")["status"] == "KEYBOARD_INTERRUPT"


# ---------------------------------------------------------------------------
# E2E-10 真实 ArmController + MockMotor 的 SUCCESS / GRASP_MISS 双路径贯通 plans
# ---------------------------------------------------------------------------


def test_E2E_10_真实控制器SUCCESS与GRASP_MISS双路径贯通plans分发(tmp_path, monkeypatch,
                                                                mock_params):
    """E2E-10（P1 §9 末段"真实返回路径的集成由 P5 覆盖"）。

    覆盖业务分支：真实 ``grasp_and_place`` 的**两条返回路径**被 plans 分发消费——
      * SUCCESS（stage=DONE、recovery_required=False、holding=EMPTY）→ §6.2 第 2 步：
        ignore/连续失败清零、success_count+1、a_count 不回滚（D04）；
      * GRASP_MISS（stage=FAULT_HOLD、holding=EMPTY、recovery_required=True）→ 第 4 步
        重试类计数 + 第 7 步恢复入口。
    两条各用一次独立装配（避免恢复路线的集成发现遮蔽 SUCCESS 断言），运动侧全真实。
    """
    # --- SUCCESS 路径（真实） ---
    fake_ok = FakeVision([target_from("t2", mock_params, valid_count=1, size=_A_SIZE),
                          nontarget(), nontarget(), nontarget()])
    case_ok = assemble_case(tmp_path / "success", monkeypatch, vision=fake_ok,
                            scenario_file="scenario_e2e01_full.json")
    assert case_ok.cfg.log_dir == tmp_path / "success" / "logs"
    code, _asr, _llm = drive_session(monkeypatch, case_ok, arm=case_ok.arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert case_ok.arm.attempt_order == ["present"]
    row = case_ok.sole("grasp_result")
    assert row["status"] == "SUCCESS" and row["stage"] == "DONE"
    assert row["holding"] == HoldingState.EMPTY.value and row["recovery_required"] is False
    assert row["place_id"] == "strawberry_A_01"
    done = case_ok.sole("task_done")
    assert done["success"] == 1 and done["a_slots_used"] == 1
    assert done["failure_counts"] == {} and done["consecutive_failures"] == 0
    assert case_ok.motor.command_count > 100

    # 同一用例内二次装配：先复位单进程状态（runlog/terminate/stubs 守卫）。
    _reset_process_state()
    fake_miss = FakeVision([target_from("t5", mock_params, valid_count=3, size=_A_SIZE)])
    case_miss = assemble_case(tmp_path / "miss", monkeypatch, vision=fake_miss,
                              scenario_file="scenario_e2e02_miss_limit.json")
    code, _asr, _llm = drive_session(monkeypatch, case_miss, arm=case_miss.arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert case_miss.motor.object_gap_m is None
    assert case_miss.cfg.log_dir == tmp_path / "miss" / "logs", "二次装配用各自日志目录"
    miss = case_miss.sole("grasp_result")
    assert miss["status"] == "GRASP_MISS" and miss["stage"] == "FAULT_HOLD"
    assert miss["holding"] == HoldingState.EMPTY.value and miss["recovery_required"] is True
    assert miss["attempt_id"] == 1 and miss["ignore_used"] == 0
    terminate = case_miss.sole("terminate")
    assert terminate["status"] in {"RECOVERY_FAILED", "NO_GRASPABLE_TARGET"}, (
        f"真实空抓必须走重试类计数与恢复/封顶，实际 {terminate['status']}"
        f"：{terminate['detail']}")
    assert terminate["stage"]["scan_id"] == 1
    assert case_miss.of("task_done") == []


# ---------------------------------------------------------------------------
# E2E-11 A 槽位耗尽 → terminate，零越界抓取
# ---------------------------------------------------------------------------


def test_E2E_11_A槽位耗尽在第四次运动调用前终止零越界抓取(tmp_path, monkeypatch, mock_params):
    """E2E-11（D04 全链版，P1-T10）。

    覆盖业务分支：A 名额 3 个按顺序消耗，第 4 个 A 目标在 ``next_place_id`` 即
    ``terminate("A_SLOTS_EXHAUSTED")``——**运动调用之前**，故零越界抓取、无第 4 条
    assign/grasp_result 行、``a_count`` 保持 3（分配即消耗、不降级、不循环覆盖 A_01）。
    运动侧=``FakeArm``（三次真实形状的 SUCCESS）。
    """
    keys = ["t1", "t2", "t3", "t4"]
    fake = FakeVision([target_from(k, mock_params, valid_count=4 - i, size=_A_SIZE)
                       for i, k in enumerate(keys)])
    arm = FakeArm([make_result(GraspStatus.SUCCESS)] * 3, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    assert [a["place_id"] for a in case.of("assign")] == ["strawberry_A_01",
                                                           "strawberry_A_02",
                                                           "strawberry_A_03"]
    assert [a["a_count"] for a in case.of("assign")] == [1, 2, 3]
    assert len(case.of("grasp_result")) == 3
    assert arm.count_of("grasp_and_place") == 3, "第 4 个 A 目标不得触运动（零越界抓取）"
    assert fake.scan_count == 4, "第 4 次扫描已发生（目标已入选），终止发生在分配处"
    assert fake.ignore_values == [0, 0, 0, 0]
    terminate = case.sole("terminate")
    assert terminate["status"] == "A_SLOTS_EXHAUSTED"
    assert terminate["stage"]["scan_id"] == 4
    assert case.of("task_done") == []


# ---------------------------------------------------------------------------
# E2E-12 完成审计（NoTarget×3 与 D14 封顶两条出口的 JSONL 字段）
# ---------------------------------------------------------------------------


def test_E2E_12_完成审计_三连NoTarget终端行与JSONL字段齐全(tmp_path, monkeypatch, capsys,
                                                            mock_params):
    """E2E-12(a)（P1 §6.6/§7）。

    覆盖业务分支：D14 后唯一正常出口的**终端行**与 ``task_done`` 行的全部最低字段
    （结束原因/成功数/失败码计数/已分配槽位 + 会话关联键），并且"不掩盖未抓走目标"
    ——本任务 2 次成功、消耗 1 个 A 槽位、局部 NoTarget 打断三连各计数如实入行；
    grasp_result 的 §7 最低字段逐条齐全。运动侧=``FakeArm``。
    """
    fake = FakeVision([target_from("t1", mock_params, valid_count=2, size=_A_SIZE),
                       nontarget(), target_from("t2", mock_params, valid_count=1, size=_B_SIZE),
                       nontarget(), nontarget(), nontarget()])
    arm = FakeArm([make_result(GraspStatus.SUCCESS)] * 2, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", _KI],
                                      llm_script=["strawberry"])
    assert code == 1
    out = capsys.readouterr().out
    assert "任务完成：结束原因=NO_TARGET_STREAK，成功=2" in out
    assert "已分配优品槽位=1" in out and "失败码计数={}" in out
    assert out.count("任务完成") == 1, "每任务只一行任务完成（初审 L9）"
    done = case.sole("task_done")
    for key in ("reason", "success", "failure_counts", "a_slots_used", "no_target_streak",
                "consecutive_failures", "ignore", "attempts", "session_id", "ts", "kind",
                "task_id", "task_instance_id", "scan_id"):
        assert key in done, f"task_done 缺字段 {key}：{sorted(done)}"
    assert done["task_id"] == "strawberry" and done["task_instance_id"] == 1
    assert done["scan_id"] == 6 and done["attempts"] == 2
    assert done["no_target_streak"] == 3 and done["success"] == 2
    assert done["a_slots_used"] == 1 and done["ignore"] == 0
    kinds = case.kinds
    assert kinds.index("task_done") < kinds.index("terminate"), "完成先于会话级 Ctrl-C"
    assert [(a["grade"], a["place_id"], a["ignore_used"], a["scan_id"])
            for a in case.of("assign")] == [("A", "strawberry_A_01", 0, 1),
                                            ("B", "strawberry_B_bin", 0, 3)]
    for row in case.of("grasp_result"):
        for key in ("status", "stage", "reason", "holding", "recovery_required",
                    "attempt_id", "place_id", "grade", "ignore_used", "valid_count",
                    "scan_id", "task_instance_id"):
            assert key in row, f"grasp_result 缺字段 {key}：{sorted(row)}"


def test_E2E_12_完成审计_D14封顶terminate审计字段齐全(tmp_path, monkeypatch, mock_params):
    """E2E-12(b)（P1 §7 terminate 行 + D14 审计面；单目标 V=1 形态，同 T26）。

    覆盖业务分支：D14 封顶出口的完整性——``grasp_result``（含末次 status/stage）先于
    ``terminate`` 落盘；terminate 行带 status/detail/stage 上下文与全部会话关联键；
    失败码计数可由逐条 grasp_result 复核并与 ``plan.failure_counts`` 一致；封顶当次
    **不恢复**（无 recovery 行）、``task_done`` **不**出现（D14 不走完成出口）。
    运动侧=``FakeArm``。
    """
    target = target_from("t1", mock_params, valid_count=1, size=_B_SIZE)
    fake = FakeVision([target])
    arm = FakeArm([make_result(GraspStatus.GRASP_MISS)], params=mock_params)
    instances: list[StrawberryPlan] = []
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓"], attempts=[])
    code, asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                     asr_script=["分拣草莓", _KI], llm_script=["strawberry"],
                                     registry=capturing_registry(instances))
    assert code == 1
    assert len(asr.listen_calls) == 1, "封顶 terminate 后会话不再推进"
    kinds = case.kinds
    assert kinds.index("grasp_result") < kinds.index("terminate"), "审计先于终止（§6.2）"
    assert kinds.count("grasp_result") == 1 and kinds.count("terminate") == 1
    assert kinds.count("recovery") == 0, "封顶当次不恢复（P1 §6.2 第 5 步）"
    last = case.of("grasp_result")[-1]
    assert last["status"] == "GRASP_MISS" and last["stage"] == "FAULT_HOLD"
    terminate = case.sole("terminate")
    assert terminate["status"] == "NO_GRASPABLE_TARGET"
    for key in ("status", "detail", "stage", "session_id", "ts", "kind", "task_id",
                "task_instance_id", "scan_id"):
        assert key in terminate, f"terminate 缺字段 {key}：{sorted(terminate)}"
    assert terminate["stage"] == {"task_id": "strawberry", "task_instance_id": 1, "scan_id": 1}
    assert "末次 GRASP_MISS/FAULT_HOLD" in terminate["detail"]
    assert "ignore=1 ≥ valid_count=1" in terminate["detail"]
    counts: dict[str, int] = {}
    for row in case.of("grasp_result"):
        if row["status"] != "SUCCESS":
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    plan, = instances
    assert plan.failure_counts == counts == {"GRASP_MISS": 1}
    assert plan.ignore == 1 and plan.consecutive_failures == 1
    assert plan.a_count == 0, "B 级不消耗 A 槽位"
    assert arm.count_of("reset_fault") == 0 and arm.count_of("move_joints") == 0
    assert case.of("task_done") == []


# ---------------------------------------------------------------------------
# E2E-13 生产 K=None 的两任务全链（MOCK_REGISTRY 贯穿）
# ---------------------------------------------------------------------------


def test_E2E_13_生产K为None的两任务全链_mock注册表从预检贯穿实例化(tmp_path, monkeypatch,
                                                                   mock_params):
    """E2E-13（P1-T27 的全链版，R04 仿真 K）。

    覆盖业务分支：生产 ``StrawberryPlan.AREA_THRESHOLD_M2`` 仍为 None（K 未标定），
    预检对生产注册表拒绝、对 ``MOCK_REGISTRY``（注入名义 K 的变体）通过，且预检收到的
    注册表对象就是会话实例化用的**同一个对象**；两个任务的成功数都 > 0。
    运动侧=``FakeArm``（K 分级由 FakeVision 的 A 尺寸目标驱动）。
    """
    seen: list[Mapping] = []
    real_preflight = run_mock.preflight

    def spy_preflight(cfg, loaded_params, registry):
        seen.append(registry)
        return real_preflight(cfg, loaded_params, registry)

    monkeypatch.setattr(run_mock, "preflight", spy_preflight)
    fake = FakeVision([target_from("t1", mock_params, valid_count=1, size=_A_SIZE),
                       nontarget(), nontarget(), nontarget(),
                       target_from("t3", mock_params, valid_count=1, size=_A_SIZE),
                       nontarget(), nontarget(), nontarget()])
    arm = FakeArm([make_result(GraspStatus.SUCCESS)] * 2, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file="scenario_e2e01_full.json")
    assert seen == [run_mock.MOCK_REGISTRY], "预检收到的就是 MOCK_REGISTRY 本身"
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", "分拣草莓", _KI],
                                      llm_script=["strawberry", "strawberry"],
                                      registry=seen[0])
    assert code == 1
    done = case.of("task_done")
    assert [d["success"] for d in done] == [1, 1] and all(d["success"] > 0 for d in done)
    assert [d["a_slots_used"] for d in done] == [1, 1]
    assert [d["task_instance_id"] for d in done] == [1, 2]
    # 生产类与生产注册表未被污染；名义 K 只存在于 mock 变体（非实测，R04）。
    assert StrawberryPlan.AREA_THRESHOLD_M2 is None
    assert PLAN_REGISTRY == {"strawberry": StrawberryPlan}
    assert PLAN_REGISTRY["strawberry"] is StrawberryPlan
    assert run_mock.MOCK_REGISTRY["strawberry"] is run_mock.MockStrawberryPlan
    assert run_mock.MockStrawberryPlan.AREA_THRESHOLD_M2 == pytest.approx(K_AREA_M2)
    cfg = load_app_config(MOCK_APP_PATH, entry="mock")
    with pytest.raises(PreflightError) as excinfo:
        app_config_module.preflight(cfg, mock_params, PLAN_REGISTRY)
    assert excinfo.value.check == "area_threshold"
    app_config_module.preflight(cfg, mock_params, run_mock.MOCK_REGISTRY)


# ---------------------------------------------------------------------------
# E2E-14 无密钥 / 无音频权重环境下的全链装配（FakeVision 等价）
# ---------------------------------------------------------------------------


def test_E2E_14_无密钥无音频权重环境装配全链_零网络零密钥读取(tmp_path, monkeypatch, mock_params):
    """E2E-14（FakeVision 等价版）。

    覆盖业务分支：在没有密钥、没有音频/VAD/相机/YOLO 权重、清掉相关环境变量的进程里
    完成 P5 §2 的全链装配与会话（D15：mock 跳过真实 asr/cloud init，不加载 VAD/麦克风/
    密钥），并证明零出站网络连接、零密钥读取路径、真实运动栈（MockMotor+SimTime）与
    真实 plans 分发被消费、资产身份为 simulation/fixture。

    本条视觉部分=**队友负责**：``model.backend="fixture"`` 的真实几何与 replay 管线
    贯通依赖 P3 交付（本期 vision 仍是契约桩）。本测试覆盖的业务分支为"无外部凭据也
    能装配并跑通业务链"。运动侧=真实 ArmController+MockMotor。
    """
    for name in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def _no_connect(*args: Any, **kwargs: Any):
        raise AssertionError("mock 全链不得发起网络请求（D15 离线验收）")

    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    monkeypatch.setattr(socket, "create_connection", _no_connect)

    def _no_secrets(*args: Any, **kwargs: Any):
        raise AssertionError("mock 入口不得读取 secrets（P1 §3.2 第 7 条属生产附加）")

    monkeypatch.setattr(app_config_module, "load_secrets", _no_secrets)
    init_probes: list[str] = []
    monkeypatch.setattr(asr_module, "init", lambda cfg: init_probes.append("asr.init"))
    monkeypatch.setattr(cloud_module, "init", lambda cfg: init_probes.append("cloud.init"))

    fake = FakeVision([target_from("t6", mock_params, valid_count=1, size=_A_SIZE),
                       nontarget(), nontarget(), nontarget()])
    case = assemble_case(tmp_path, monkeypatch, vision=fake,
                         scenario_file="scenario_e2e01_full.json")
    assert init_probes == [], "真实 asr/cloud init 零调用（D15）"
    assert len(case.motor.initialize_calls) == 1
    assert case.params.status == "simulation" and case.params.verification is None
    vision_payload = json.loads(MOCK_VISION_PATH.read_text(encoding="utf-8"))
    assert isinstance(vision_payload, dict) and "P3" in vision_payload["_comment"]
    code, _asr, _llm = drive_session(monkeypatch, case, arm=case.arm,
                                      asr_script=["分拣草莓", _KI], llm_script=["strawberry"])
    assert code == 1
    startup = case.sole("startup")
    assert startup["entry"] == "mock"
    assert Path(startup["profile_path"]) == MOCK_PROFILE_PATH.resolve()
    assert Path(startup["vision_config_path"]) == MOCK_VISION_PATH.resolve()
    kinds = case.kinds
    assert "ready" in kinds and kinds.index("ready") < kinds.index("task_done")
    done = case.sole("task_done")
    assert done["success"] == 1, "无凭据环境仍跑通真实抓放业务链"
    assert case.sole("terminate")["status"] == "KEYBOARD_INTERRUPT", "只有会话级 Ctrl-C"
    assert case.motor.command_count > 100


# ---------------------------------------------------------------------------
# E2E-15 两任务 + 局部 NoTarget 的日志追踪
# ---------------------------------------------------------------------------


def test_E2E_15_两任务含局部NoTarget的日志关联唯一可审计(tmp_path, monkeypatch, mock_params):
    """E2E-15（P1 §3.4 关联键 / T31 的连续任务版）。

    覆盖业务分支：两个任务、任务内**局部** NoTarget（打断 streak 但不结束任务）时，
    ``task_instance_id`` 区分两个任务、``(session_id, task_instance_id, scan_id,
    attempt_id)`` 关联唯一不串号、每次产出目标的扫描都有 assign+grasp_result 可对账。
    NoTarget 扫描不产生业务行（其身份由 FakeVision 的 ignore 序列 + streak/scan 计数
    交叉核对）；逐扫描 ``vision_scan`` 审计（含候选与拒绝原因明细）属队友的 P3 侧，
    本条注明、不冒充。运动侧=``FakeArm``。
    """
    def t(key: str, vc: int) -> VisionInterface:
        return target_from(key, mock_params, valid_count=vc, size=_A_SIZE)

    fake = FakeVision([t("t1", 2), nontarget(), t("t2", 1),
                       nontarget(), nontarget(), nontarget(),
                       t("t3", 2), nontarget(), nontarget(), t("t4", 1),
                       nontarget(), nontarget(), nontarget()])
    arm = FakeArm([make_result(GraspStatus.SUCCESS)] * 4, params=mock_params)
    case = assemble_case(tmp_path, monkeypatch, vision=fake, scenario_file="scenario.json",
                         utterances=["分拣草莓", "分拣草莓"], attempts=[])
    code, _asr, _llm = drive_session(monkeypatch, case, arm=arm,
                                      asr_script=["分拣草莓", "分拣草莓", _KI],
                                      llm_script=["strawberry", "strawberry"])
    assert code == 1
    rows = case.events()
    assert {r["session_id"] for r in rows} == {rows[0]["session_id"]}, "一会话一个 session_id"
    task_rows = [r for r in rows if "task_instance_id" in r]
    assert {r["task_instance_id"] for r in task_rows} == {1, 2}
    assert all(r["task_id"] == "strawberry" for r in task_rows)
    assert all("scan_id" in r for r in task_rows), "任务内全部事件都带 scan_id 上下文"

    business = [r for r in rows if r["kind"] in {"assign", "grasp_result", "recovery"}]
    assert len(business) == 8, "4 次目标扫描 × (assign + grasp_result)"
    triples = {(r["session_id"], r["task_instance_id"], r["scan_id"], r["attempt_id"])
               for r in business}
    assert len(triples) == 4, "关联键唯一（同扫描的 assign 与 grasp_result 同 attempt）"
    per_task: dict[int, list[int]] = {}
    for r in case.of("grasp_result"):
        per_task.setdefault(r["task_instance_id"], []).append(r["attempt_id"])
    assert per_task == {1: [1, 2], 2: [1, 2]}, "attempt_id 每任务从 1 起、不跨任务累加"
    for task_id, last_scan in ((1, 6), (2, 7)):
        scans = {r["scan_id"] for r in rows if r.get("task_instance_id") == task_id}
        assert 1 in scans and last_scan in scans, f"任务{task_id} 扫描覆盖不全：{sorted(scans)}"
    done = case.of("task_done")
    assert [(d["task_instance_id"], d["success"], d["no_target_streak"], d["scan_id"],
             d["attempts"]) for d in done] == [(1, 2, 3, 6, 2), (2, 2, 3, 7, 2)]
    assert fake.ignore_values == [0] * 13 and fake.scan_count == 13
    assert [r["valid_count"] for r in case.of("assign")] == [2, 1, 2, 1]
    assert [r["scan_id"] for r in case.of("assign")] == [1, 3, 1, 4]
    assert case.of("terminate")[0]["status"] == "KEYBOARD_INTERRUPT"
