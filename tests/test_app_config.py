"""configs/app_config.py 的加载、校验与启动预检测试。

规格：docs/实施文档/P1_任务编排与运行生命周期技术文档.md §3.1（app.json 与公开签名）、
§3.2（启动预检 8 条）、§8 步骤 1；决策 D13/D15/D16/D20。mock 配置结构取自 P5 §2。

做法沿用 tests/support.py 的"临时目录改写"模式：所有负向用例都在内存里改 JSON、
落到 tmp_path 再加载，绝不改动仓库真配置；需要改 params 的用例用 SimpleNamespace
替身 params，不复制 profile 结构。P1-04 的 StrawberryPlan 尚不存在，注册表一律用
只提供契约类属性的替身类。
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from configs import app_config
from configs.app_config import (
    SECRET_KEYS,
    AppConfig,
    AppConfigError,
    MockSettings,
    PreflightError,
    load_app_config,
    load_secrets,
    preflight,
)
from configs.motion_params import load_motion_params
from tests.support import DELETE, _set_path, load_sim, raw_profile, write_profile

ROOT = Path(app_config.__file__).resolve().parents[1]
APP_TEMPLATE = ROOT / "configs" / "app.json"
SECRETS_TEMPLATE = ROOT / "configs" / "secrets.json"

# 替身 plan 的落点 ID 与 P1 §3.5 的 StrawberryPlan 同构（真实值由 P4 落点标定回填）。
A_IDS = ("strawberry_A_01", "strawberry_A_02")
B_ID = "strawberry_B_bin"
C_ID = "strawberry_C_bin"


class StubPlan:
    """只提供 preflight 鸭子类型检查会读到的五个类属性（无 __init__、不碰 arm）。"""

    TASK_ID = "strawberry"
    AREA_THRESHOLD_M2 = 0.0004  # 名义值，仅用于让"K 已设置"分支通过；非实测
    A_PLACE_IDS = A_IDS
    B_PLACE_ID = B_ID
    C_PLACE_ID = C_ID


def registry_with(**overrides: Any) -> dict[str, type]:
    """返回 {"strawberry": 覆盖过若干属性的替身类}，用于注册表负向用例。"""
    cls = type("StrawberryPlanStub", (StubPlan,), overrides)
    return {"strawberry": cls}


def stub_params(places=A_IDS + (B_ID, C_ID), yaw_offset_deg: float = 0.0):
    """替身 params：preflight 只读 places 与 grasp.yaw_offset_deg 两处。"""
    return SimpleNamespace(
        places={pid: SimpleNamespace(position=None, yaw_deg=0.0) for pid in places},
        grasp=SimpleNamespace(yaw_offset_deg=yaw_offset_deg),
    )


# ---------------------------------------------------------------------------
# 测试资产构造（全部落在 tmp_path，不触碰仓库真配置）
# ---------------------------------------------------------------------------


def app_data(tmp: Path) -> dict[str, Any]:
    """一份合法生产配置，路径全部按"相对 tmp/configs"书写。"""
    return {
        "profile_path": "../profile/profile.json",
        "vision_config_path": "../assets/vision.json",
        "prompt_path": "../assets/prompt.md",
        "log_dir": "../logs",
        "asr": {
            "model": "qwen-audio-3.1-asr-flash",
            "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
                        "multimodal-generation/generation",
            "cloud_total_timeout_s": 30.0,
        },
        "llm": {
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "total_timeout_s": 30.0,
        },
    }


def mock_section(tmp: Path) -> dict[str, str]:
    """P5 §2 的 mock 段：脚本化转写文件 + 视觉 replay 帧目录（真实建出资产）。"""
    base = tmp / "fixtures" / "mock"
    (base / "frames").mkdir(parents=True, exist_ok=True)
    (base / "scenario.json").write_text(
        json.dumps({"utterances": ["分拣草莓"], "attempts": [{"contact": "present"}]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "asr_script_path": "../fixtures/mock/scenario.json",
        "vision_replay_dir": "../fixtures/mock/frames",
    }


def make_assets(tmp: Path) -> None:
    """造出预检第 4/5 条要读的两个文件（vision 配置只要求是合法 JSON）。"""
    assets = tmp / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "vision.json").write_text(
        json.dumps({"note": "字段 schema 由 P3 定义；预检只看存在与合法 JSON（L7 收窄）"}),
        encoding="utf-8",
    )
    (assets / "prompt.md").write_text("任务：分拣草莓。只输出 JSON。\n", encoding="utf-8")


def write_app(tmp: Path, data: dict[str, Any], *, name: str = "app.json") -> Path:
    cfg_dir = tmp / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / name
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load(tmp: Path, data: dict[str, Any], *, entry: str = "production") -> AppConfig:
    return load_app_config(write_app(tmp, data), entry=entry)


def loaded_production_cfg(tmp: Path) -> AppConfig:
    make_assets(tmp)
    return load(tmp, app_data(tmp))


def loaded_mock_cfg(tmp: Path) -> AppConfig:
    make_assets(tmp)
    data = app_data(tmp)
    data["mock"] = mock_section(tmp)
    return load(tmp, data, entry="mock")


def sim_params_with_strawberry_places(tmp: Path):
    """仿真 profile 的**副本**追加 strawberry_* 落点（P1 §3.5：不改仓库真配置）。"""
    data = raw_profile()
    proto = data["places"]["bin"]
    for pid in (*A_IDS, B_ID, C_ID):
        data["places"][pid] = {"position": list(proto["position"]), "yaw_deg": proto["yaw_deg"]}
    prof_dir = tmp / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)
    return load_motion_params(write_profile(prof_dir, data), mode="mock")


@pytest.fixture
def real_secrets(tmp_path, monkeypatch):
    """把 secrets 定位改到临时目录并填入非空值，供"非生产附加检查之外"的用例复用。"""
    path = tmp_path / "secrets.local.json"
    path.write_text(
        json.dumps({key: f"sk-local-{key}" for key in SECRET_KEYS}), encoding="utf-8"
    )
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", path)
    return path


# ---------------------------------------------------------------------------
# 1. 正向：模板与目录相对解析
# ---------------------------------------------------------------------------


def test_生产模板可加载且字段与P1一致(tmp_path):
    cfg = load_app_config(APP_TEMPLATE, entry="production")
    assert cfg.mock is None
    assert cfg.asr.model == "qwen-audio-3.1-asr-flash"
    assert cfg.asr.endpoint == (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    assert cfg.asr.cloud_total_timeout_s == 30.0
    assert cfg.llm.model == "deepseek-flash"
    assert cfg.llm.base_url == "https://api.deepseek.com"
    assert cfg.llm.total_timeout_s == 30.0


def test_模板相对该配置文件目录解析为绝对路径():
    # 模板在 configs/ 下，所以 ../ 指向项目根（D16）。
    cfg = load_app_config(APP_TEMPLATE, entry="production")
    assert cfg.profile_path == ROOT / "calibration/REAL_ARM/PC/profile.json"
    assert cfg.vision_config_path == ROOT / "configs/vision.local.json"
    assert cfg.prompt_path == ROOT / "prompt.md"
    assert cfg.log_dir == ROOT / "logs"
    for p in (cfg.profile_path, cfg.vision_config_path, cfg.prompt_path, cfg.log_dir):
        assert p.is_absolute()


def test_模板引用的仓库资源确实存在():
    # 防 D16 类回归：模板照抄就必须能解析到真实文件（vision.local.json/logs 属本地
    # 文件与运行期目录，本条只查仓库内应有的两个）。
    cfg = load_app_config(APP_TEMPLATE, entry="production")
    assert cfg.profile_path.is_file()
    assert cfg.prompt_path.is_file()


def test_生产模板不含mock段():
    assert "mock" not in json.loads(APP_TEMPLATE.read_text(encoding="utf-8"))


def test_副本配置按配置文件所在目录解析(tmp_path):
    make_assets(tmp_path)
    cfg = load(tmp_path, app_data(tmp_path))
    assert cfg.profile_path == tmp_path / "profile/profile.json"
    assert cfg.vision_config_path == tmp_path / "assets/vision.json"
    assert cfg.prompt_path == tmp_path / "assets/prompt.md"
    assert cfg.log_dir == tmp_path / "logs"


def test_绝对路径原样保留(tmp_path):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    data["profile_path"] = str(tmp_path / "somewhere/else/profile.json")
    cfg = load(tmp_path, data)
    assert cfg.profile_path == tmp_path / "somewhere/else/profile.json"


def test_mock配置按P5结构可加载(tmp_path):
    # P5 §2 的完整 app.mock.json 示例（去掉不需要的 fixture 之外字段），逐路径断言。
    make_assets(tmp_path)
    data = {
        "profile_path": "../tests/fixtures/mock/profile.json",
        "vision_config_path": "../tests/fixtures/mock/vision.json",
        "prompt_path": "../prompt.md",
        "log_dir": "../logs/mock",
        "asr": {"model": "mock",
                "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
                            "multimodal-generation/generation",
                "cloud_total_timeout_s": 30.0},
        "llm": {"model": "mock", "base_url": "https://api.deepseek.com",
                "total_timeout_s": 30.0},
        "mock": {"asr_script_path": "../tests/fixtures/mock/scenario.json",
                 "vision_replay_dir": "../tests/fixtures/mock/frames"},
    }
    cfg = load(tmp_path, data, entry="mock")
    assert cfg.mock is not None
    assert cfg.profile_path == tmp_path / "tests/fixtures/mock/profile.json"
    assert cfg.vision_config_path == tmp_path / "tests/fixtures/mock/vision.json"
    assert cfg.prompt_path == tmp_path / "prompt.md"
    assert cfg.log_dir == tmp_path / "logs/mock"
    assert cfg.mock.asr_script_path == tmp_path / "tests/fixtures/mock/scenario.json"
    assert cfg.mock.vision_replay_dir == tmp_path / "tests/fixtures/mock/frames"


def test_数据类全部冻结():
    cfg = load_app_config(APP_TEMPLATE, entry="production")
    for obj, field, value in (
        (cfg, "log_dir", Path("/tmp")),
        (cfg.asr, "model", "x"),
        (cfg.llm, "total_timeout_s", 1.0),
        (MockSettings(Path("a"), Path("b")), "asr_script_path", Path("c")),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


def test_secrets模板可加载且允许空值():
    assert json.loads(SECRETS_TEMPLATE.read_text(encoding="utf-8")) == {
        "dashscope_api_key": "",
        "deepseek_api_key": "",
    }
    assert load_secrets(SECRETS_TEMPLATE) == {key: "" for key in SECRET_KEYS}


def test_secrets加载返回两键字符串(tmp_path):
    p = tmp_path / "secrets.local.json"
    p.write_text(json.dumps({"dashscope_api_key": "a", "deepseek_api_key": "b"}))
    assert load_secrets(p) == {"dashscope_api_key": "a", "deepseek_api_key": "b"}


# ---------------------------------------------------------------------------
# 2. 负向：load_app_config 的每一条校验规则
# ---------------------------------------------------------------------------


def test_拒绝未知顶层键(tmp_path):
    data = app_data(tmp_path)
    data["mode"] = "mock"
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "未知键" in str(exc.value)
    assert "mode" in str(exc.value)


@pytest.mark.parametrize("missing", [
    "profile_path", "vision_config_path", "prompt_path", "log_dir", "asr", "llm",
])
def test_拒绝缺失必填键(tmp_path, missing):
    data = app_data(tmp_path)
    del data[missing]
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "缺失键" in str(exc.value)
    assert missing in str(exc.value)


def test_拒绝非对象顶层(tmp_path):
    out = write_app(tmp_path, {})
    out.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AppConfigError) as exc:
        load_app_config(out, entry="production")
    assert "应为对象" in str(exc.value)


def test_拒绝非法JSON(tmp_path):
    out = write_app(tmp_path, {})
    out.write_text("{not json", encoding="utf-8")
    with pytest.raises(AppConfigError) as exc:
        load_app_config(out, entry="production")
    assert "不是合法 JSON" in str(exc.value)


def test_拒绝不存在的配置文件(tmp_path):
    with pytest.raises(AppConfigError) as exc:
        load_app_config(tmp_path / "configs" / "app.local.json", entry="production")
    assert "不存在" in str(exc.value)


def test_拒绝非法entry(tmp_path):
    make_assets(tmp_path)
    out = write_app(tmp_path, app_data(tmp_path))
    with pytest.raises(AppConfigError) as exc:
        load_app_config(out, entry="real")  # type: ignore[arg-type]
    assert "entry" in str(exc.value)


@pytest.mark.parametrize("field", ["asr.cloud_total_timeout_s", "llm.total_timeout_s"])
@pytest.mark.parametrize("bad", [0.0, -1.0, True, False, "30", None, []])
def test_超时字段只收正数且拒绝bool与非数值(tmp_path, field, bad):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    _set_path(data, field, bad)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert field in str(exc.value)


@pytest.mark.parametrize("field", ["asr.cloud_total_timeout_s", "llm.total_timeout_s"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_超时字段拒绝非有限值(tmp_path, field, bad):
    # JSON 里的 NaN/Infinity 扩展字面量在解析阶段就被拒；有限性检查再兜住非 JSON 来源。
    make_assets(tmp_path)
    data = app_data(tmp_path)
    _set_path(data, field, bad)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "非有限浮点字面量" in str(exc.value)
    with pytest.raises(AppConfigError) as exc2:
        app_config._require_positive_finite(field, bad)
    assert "有限" in str(exc2.value)


def test_超时字段接受整数与浮点正数(tmp_path):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    _set_path(data, "asr.cloud_total_timeout_s", 45)
    _set_path(data, "llm.total_timeout_s", 7.5)
    cfg = load(tmp_path, data)
    assert cfg.asr.cloud_total_timeout_s == 45.0
    assert cfg.llm.total_timeout_s == 7.5


def test_超时的JSON扩展字面量也被拒绝(tmp_path):
    make_assets(tmp_path)
    out = write_app(tmp_path, app_data(tmp_path))
    text = out.read_text(encoding="utf-8").replace('"cloud_total_timeout_s": 30.0',
                                                  '"cloud_total_timeout_s": NaN')
    assert "NaN" in text
    out.write_text(text, encoding="utf-8")
    with pytest.raises(AppConfigError) as exc:
        load_app_config(out, entry="production")
    assert "非有限浮点字面量" in str(exc.value)


@pytest.mark.parametrize("field", ["asr.model", "asr.endpoint", "llm.model",
                                   "llm.base_url", "profile_path", "prompt_path"])
def test_字符串字段拒绝空白与非字符串(tmp_path, field):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    _set_path(data, field, "   ")
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "不能为空字符串" in str(exc.value)
    _set_path(data, field, 5)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert field in str(exc.value)


@pytest.mark.parametrize("section,extra", [
    ("asr", {"asr.provider": "dashscope"}),
    ("llm", {"llm.api_key": "secret"}),
    ("mock", {"mock.frames_dir": "x"}),
])
def test_子段未知键同样拒绝(tmp_path, section, extra):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    data["mock"] = mock_section(tmp_path)
    for path, value in extra.items():
        _set_path(data, path, value)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data, entry="mock")
    assert "未知键" in str(exc.value)


@pytest.mark.parametrize("missing", ["asr.endpoint", "llm.base_url"])
def test_子段缺失键被拒绝(tmp_path, missing):
    data = app_data(tmp_path)
    _set_path(data, missing, DELETE)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "缺失键" in str(exc.value)


@pytest.mark.parametrize("section", ["asr", "llm"])
def test_子段非对象被拒绝(tmp_path, section):
    data = app_data(tmp_path)
    data[section] = "mock"
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data)
    assert "应为对象" in str(exc.value)


def test_production入口拒绝mock段(tmp_path):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    data["mock"] = mock_section(tmp_path)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data, entry="production")
    # 精确断言到"专用拒绝分支"，而不是被泛化的未知键错误顺带命中
    # （tmp_path 目录名含测试名，只有消息正文才能证明走了 entry 校验）。
    assert exc.value.field_path == "mock"
    assert "不得包含" in exc.value.message
    assert "app.mock.json" in exc.value.message


def test_mock入口必须有mock段(tmp_path):
    make_assets(tmp_path)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, app_data(tmp_path), entry="mock")
    assert "缺失键" in exc.value.message
    assert "mock" in exc.value.message


@pytest.mark.parametrize("missing", ["asr_script_path", "vision_replay_dir"])
def test_mock段两个路径都必填(tmp_path, missing):
    make_assets(tmp_path)
    data = app_data(tmp_path)
    data["mock"] = mock_section(tmp_path)
    _set_path(data, f"mock.{missing}", DELETE)
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, data, entry="mock")
    assert "缺失键" in str(exc.value)
    assert missing in str(exc.value)


def test_appconfig错误带字段路径(tmp_path):
    with pytest.raises(AppConfigError) as exc:
        load(tmp_path, {"profile_path": "p"})
    assert exc.value.field_path.endswith("app.json")
    assert isinstance(exc.value, ValueError)


# ---------------------------------------------------------------------------
# 3. 负向：load_secrets
# ---------------------------------------------------------------------------


def test_secrets拒绝缺键(tmp_path):
    p = tmp_path / "secrets.local.json"
    p.write_text(json.dumps({"dashscope_api_key": "a"}))
    with pytest.raises(AppConfigError) as exc:
        load_secrets(p)
    assert "缺失键" in str(exc.value)
    assert "deepseek_api_key" in str(exc.value)


def test_secrets拒绝未知键(tmp_path):
    p = tmp_path / "secrets.local.json"
    p.write_text(json.dumps({"dashscope_api_key": "a", "deepseek_api_key": "b",
                             "openai_api_key": "c"}))
    with pytest.raises(AppConfigError) as exc:
        load_secrets(p)
    assert "未知键" in str(exc.value)


@pytest.mark.parametrize("value", [None, 5, True, ["a"]])
def test_secrets值必须是字符串(tmp_path, value):
    p = tmp_path / "secrets.local.json"
    p.write_text(json.dumps({"dashscope_api_key": "a", "deepseek_api_key": value}))
    with pytest.raises(AppConfigError) as exc:
        load_secrets(p)
    assert "deepseek_api_key" in str(exc.value)


def test_secrets拒绝不存在与非法JSON(tmp_path):
    with pytest.raises(AppConfigError):
        load_secrets(tmp_path / "secrets.local.json")
    p = tmp_path / "broken.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(AppConfigError):
        load_secrets(p)


# ---------------------------------------------------------------------------
# 4. preflight 第 1 条：注册表
# ---------------------------------------------------------------------------


def _expect_check(check: str, cfg: AppConfig, params, registry) -> pytest.ExceptionInfo:
    """断言预检抛 PreflightError 且落在指定检查项上，返回 ExceptionInfo 供继续断言。"""
    with pytest.raises(PreflightError) as exc:
        preflight(cfg, params, registry)
    assert exc.value.check == check, f"期望检查项 {check}，实际 {exc.value.check}"
    return exc


def test_preflight_异常类型可区分且可被基类捕获():
    assert issubclass(PreflightError, AppConfigError)
    assert issubclass(AppConfigError, ValueError)


def test_注册表多余键被拒绝(tmp_path):
    cfg = loaded_production_cfg(tmp_path)
    reg = dict(registry_with())
    reg["blueberry"] = type("BlueberryPlan", (StubPlan,), {"TASK_ID": "blueberry"})
    err = _expect_check("registry", cfg, stub_params(), reg)
    assert "strawberry" in str(err.value)


def test_空注册表被拒绝(tmp_path):
    cfg = loaded_production_cfg(tmp_path)
    _expect_check("registry", cfg, stub_params(), {})


def test_TASK_ID与键不一致被拒绝(tmp_path):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("registry", cfg, stub_params(), registry_with(TASK_ID="other"))
    assert "TASK_ID" in str(err.value)


def test_缺少契约属性被拒绝(tmp_path):
    cfg = loaded_production_cfg(tmp_path)

    class NoThreshold:
        """不继承 StubPlan 的替身：刻意缺少 AREA_THRESHOLD_M2 属性。"""

        TASK_ID = "strawberry"
        A_PLACE_IDS = A_IDS
        B_PLACE_ID = B_ID
        C_PLACE_ID = C_ID

    err = _expect_check("registry", cfg, stub_params(), {"strawberry": NoThreshold})
    assert "AREA_THRESHOLD_M2" in str(err.value)


@pytest.mark.parametrize("a_ids,keyword", [
    ((), "非空"),
    (["strawberry_A_01"], "tuple"),
    (("strawberry_A_01", "strawberry_A_01"), "重复"),
    (("strawberry_A_01", ""), "非字符串或空白"),
    (("strawberry_A_01", 7), "非字符串或空白"),
])
def test_A_PLACE_IDS形状要求(tmp_path, a_ids, keyword):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("registry", cfg, stub_params(), registry_with(A_PLACE_IDS=a_ids))
    assert keyword in str(err.value)


@pytest.mark.parametrize("over", [
    {"B_PLACE_ID": ""},
    {"B_PLACE_ID": None},
    {"C_PLACE_ID": "  "},
])
def test_B与C必须非空字符串(tmp_path, over):
    cfg = loaded_production_cfg(tmp_path)
    _expect_check("registry", cfg, stub_params(), registry_with(**over))


@pytest.mark.parametrize("over,overlap", [
    ({"B_PLACE_ID": A_IDS[0]}, "A/B"),
    ({"C_PLACE_ID": A_IDS[1]}, "A/C"),
    ({"C_PLACE_ID": B_ID}, "B/C"),
])
def test_三组落点不得重叠(tmp_path, over, overlap):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("registry", cfg, stub_params(), registry_with(**over))
    assert "互不重叠" in str(err.value)


def test_注册表检查先于其他检查项(tmp_path):
    # 同时坏掉注册表与 vision 路径，必须报第 1 条（§3.2 顺序）。
    data = app_data(tmp_path)
    data["vision_config_path"] = "../assets/缺失.json"
    cfg = load(tmp_path, data)
    err = _expect_check("registry", cfg, stub_params(), registry_with(TASK_ID="bad"))
    assert "TASK_ID" in str(err.value)


# ---------------------------------------------------------------------------
# 5. preflight 第 2、3、6 条（依赖 params / plan 类属性）
# ---------------------------------------------------------------------------


def test_落点必须存在于params(tmp_path):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("places", cfg, stub_params(places=(*A_IDS, B_ID)),
                        registry_with())
    assert C_ID in str(err.value)


def test_仓库仿真profile的places只有default与bin(tmp_path):
    # 环境事实登记：P1 §3.5 要求在仿真 profile **副本**里追加 strawberry_* 落点，
    # 因此用 load_sim() 直接跑预检必然停在第 2 条，全通过用例改用 tmp 副本。
    cfg = loaded_production_cfg(tmp_path)
    sim = load_sim()
    assert set(sim.places) == {"default", "bin"}
    assert sim.grasp.yaw_offset_deg == 0.0  # 第 6 条对仿真 profile 天然通过
    err = _expect_check("places", cfg, sim, registry_with())
    assert A_IDS[0] in str(err.value)


@pytest.mark.parametrize("k", [None, 0.0, -1.0, float("nan"), float("inf"), True, "0.4"])
def test_面积阈值K的要求(tmp_path, k):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("area_threshold", cfg, stub_params(), registry_with(AREA_THRESHOLD_M2=k))
    assert "AREA_THRESHOLD_M2" in str(err.value)
    if k is None:
        assert "未标定" in str(err.value)


@pytest.mark.parametrize("yaw", [45.0, -0.001, 90.0001, 180.0, float("nan"), True, None])
def test_yaw_offset产品限制断言(tmp_path, real_secrets, yaw):
    cfg = loaded_production_cfg(tmp_path)
    err = _expect_check("yaw_offset", cfg, stub_params(yaw_offset_deg=yaw), registry_with())
    assert "yaw_offset_deg" in str(err.value)


@pytest.mark.parametrize("yaw", [0, 0.0, 90, 90.0])
def test_yaw_offset接受0与90(tmp_path, real_secrets, yaw):
    cfg = loaded_production_cfg(tmp_path)
    preflight(cfg, stub_params(yaw_offset_deg=yaw), registry_with())


def test_params缺少grasp时报错而非AttributeError(tmp_path, real_secrets):
    cfg = loaded_production_cfg(tmp_path)
    broken = SimpleNamespace(places={pid: None for pid in A_IDS + (B_ID, C_ID)})
    err = _expect_check("yaw_offset", cfg, broken, registry_with())
    assert "yaw_offset_deg" in str(err.value)


# ---------------------------------------------------------------------------
# 6. preflight 第 4、5 条（文件资产）
# ---------------------------------------------------------------------------


def test_vision配置缺失被拒绝(tmp_path, real_secrets):
    data = app_data(tmp_path)
    make_assets(tmp_path)
    data["vision_config_path"] = "../assets/没有这个文件.json"
    cfg = load(tmp_path, data)
    err = _expect_check("vision_config", cfg, stub_params(), registry_with())
    assert "不存在" in str(err.value)


def test_vision配置非法JSON被拒绝(tmp_path, real_secrets):
    make_assets(tmp_path)
    (tmp_path / "assets/vision.json").write_text("{坏的", encoding="utf-8")
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("vision_config", cfg, stub_params(), registry_with())
    assert "合法 JSON" in str(err.value)


def test_vision配置顶层非对象被拒绝(tmp_path, real_secrets):
    make_assets(tmp_path)
    (tmp_path / "assets/vision.json").write_text("[1,2]", encoding="utf-8")
    cfg = load(tmp_path, app_data(tmp_path))
    _expect_check("vision_config", cfg, stub_params(), registry_with())


def test_preflight不解析vision配置内部字段(tmp_path, real_secrets):
    # L7 收窄：字段语义属 P3 init，预检只看"是不是合法 JSON 对象"。
    make_assets(tmp_path)
    (tmp_path / "assets/vision.json").write_text(
        json.dumps({"whichever_p3_schema": {"model": {"backend": "yolo"}}}),
        encoding="utf-8",
    )
    cfg = load(tmp_path, app_data(tmp_path))
    preflight(cfg, stub_params(), registry_with())


def test_prompt缺失与空文件被拒绝(tmp_path, real_secrets):
    make_assets(tmp_path)
    (tmp_path / "assets/prompt.md").unlink()
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("prompt", cfg, stub_params(), registry_with())
    assert "不存在" in str(err.value)

    make_assets(tmp_path)
    (tmp_path / "assets/prompt.md").write_text("   \n\t\n", encoding="utf-8")
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("prompt", cfg, stub_params(), registry_with())
    assert "为空" in str(err.value)


# ---------------------------------------------------------------------------
# 7. preflight 第 7 条：生产附加 secrets 检查
# ---------------------------------------------------------------------------


def test_生产secrets文件缺失被拒绝(tmp_path, monkeypatch):
    make_assets(tmp_path)
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", tmp_path / "没有.json")
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("secrets", cfg, stub_params(), registry_with())
    assert "不存在" in str(err.value)


@pytest.mark.parametrize("empty", ["dashscope_api_key", "deepseek_api_key"])
def test_生产secrets空值被拒绝且不回显键值(tmp_path, monkeypatch, empty):
    make_assets(tmp_path)
    secret_path = tmp_path / "secrets.local.json"
    values = {k: "TOP-SECRET-VALUE-不入日志" for k in SECRET_KEYS}
    values[empty] = "   "
    secret_path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", secret_path)
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("secrets", cfg, stub_params(), registry_with())
    assert empty in str(err.value)
    assert "TOP-SECRET-VALUE" not in str(err.value)


def test_生产secrets结构错误映射为secrets检查项(tmp_path, monkeypatch):
    make_assets(tmp_path)
    secret_path = tmp_path / "secrets.local.json"
    secret_path.write_text(json.dumps({"dashscope_api_key": "a"}), encoding="utf-8")
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", secret_path)
    cfg = load(tmp_path, app_data(tmp_path))
    err = _expect_check("secrets", cfg, stub_params(), registry_with())
    assert "deepseek_api_key" in str(err.value)


def test_mock入口跳过secrets检查(tmp_path, monkeypatch):
    make_assets(tmp_path)
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", tmp_path / "不存在.json")
    cfg = loaded_mock_cfg(tmp_path)
    preflight(cfg, stub_params(), registry_with())  # 不抛即通过


# ---------------------------------------------------------------------------
# 8. preflight 第 8 条：mock 资产与 log_dir
# ---------------------------------------------------------------------------


def test_mock脚本缺失被拒绝(tmp_path):
    cfg = loaded_mock_cfg(tmp_path)
    cfg.mock.asr_script_path.unlink()
    err = _expect_check("mock_assets", cfg, stub_params(), registry_with())
    assert "scenario.json" in str(err.value)


def test_mock_replay目录缺失被拒绝(tmp_path):
    cfg = loaded_mock_cfg(tmp_path)
    (tmp_path / "fixtures/mock/frames/").rmdir()
    err = _expect_check("mock_assets", cfg, stub_params(), registry_with())
    assert "replay" in str(err.value)


def test_不存在的log_dir会被创建(tmp_path, real_secrets):
    cfg = loaded_production_cfg(tmp_path)
    assert not cfg.log_dir.exists()
    preflight(cfg, stub_params(), registry_with())
    assert cfg.log_dir.is_dir()


def test_嵌套log_dir支持parents创建(tmp_path):
    data = app_data(tmp_path)
    make_assets(tmp_path)
    data["log_dir"] = "../logs/mock"
    data["mock"] = mock_section(tmp_path)
    cfg = load(tmp_path, data, entry="mock")
    preflight(cfg, stub_params(), registry_with())
    assert cfg.log_dir.is_dir()


def test_log_dir是普通文件时被拒绝(tmp_path, real_secrets):
    data = app_data(tmp_path)
    make_assets(tmp_path)
    (tmp_path / "logs").write_text("我不是目录", encoding="utf-8")
    cfg = load(tmp_path, data)
    err = _expect_check("log_dir", cfg, stub_params(), registry_with())
    assert "不是目录" in str(err.value)


def test_log_dir无法创建时被拒绝(tmp_path, real_secrets):
    data = app_data(tmp_path)
    make_assets(tmp_path)
    (tmp_path / "logs").write_text("占位文件", encoding="utf-8")
    data["log_dir"] = "../logs/nested"
    cfg = load(tmp_path, data)
    err = _expect_check("log_dir", cfg, stub_params(), registry_with())
    assert "无法创建" in str(err.value)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root 可写只读目录，权限断言不成立")
def test_log_dir不可写时被拒绝(tmp_path, real_secrets):
    data = app_data(tmp_path)
    make_assets(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o500)  # 只读：存在但不可写
    try:
        cfg = load(tmp_path, data)
        err = _expect_check("log_dir", cfg, stub_params(), registry_with())
        assert "不可写" in str(err.value)
    finally:
        logs.chmod(0o700)


# ---------------------------------------------------------------------------
# 9. 全通过路径与边界
# ---------------------------------------------------------------------------


def test_preflight生产全通过(tmp_path, real_secrets):
    params = sim_params_with_strawberry_places(tmp_path)
    make_assets(tmp_path)
    cfg = load(tmp_path, app_data(tmp_path))
    preflight(cfg, params, registry_with())  # 不抛即通过
    assert cfg.log_dir.is_dir()
    assert cfg.mock is None


def test_preflight_mock全通过并写日志目录(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "SECRETS_LOCAL_PATH", tmp_path / "不参与.json")
    params = sim_params_with_strawberry_places(tmp_path)
    make_assets(tmp_path)
    data = app_data(tmp_path)
    data["log_dir"] = "../logs/mock"
    data["mock"] = mock_section(tmp_path)
    cfg = load(tmp_path, data, entry="mock")
    preflight(cfg, params, registry_with())
    assert cfg.mock is not None
    assert cfg.log_dir.is_dir()


def test_preflight只依赖入参不触碰设备实现():
    # §3.2 强调预检先于 motor 构造：本模块的 import 只允许标准库 + configs.motion_params。
    tree = ast.parse(Path(app_config.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "json", "math", "os", "collections", "dataclasses",
                        "pathlib", "typing", "configs"}
    forbidden = {"qingyun", "serial", "libs", "placo", "numpy", "httpx", "tests"}
    assert not (imported & forbidden)


def test_两个入口的固定配置路径常量():
    # D15：入口固定，无 CLI；main/run_mock（P1-05）按这两个常量装配。
    assert app_config.PRODUCTION_APP_PATH == ROOT / "configs/app.local.json"
    assert app_config.MOCK_APP_PATH == ROOT / "configs/app.mock.json"
    assert app_config.SECRETS_LOCAL_PATH.parent.name == "configs"


def test_gitignore屏蔽本地配置与密钥():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()]
    for line in ("configs/app.local.json", "configs/secrets.local.json",
                 "configs/vision.local.json"):
        assert line in lines, line
    # 文本之外再验一次语义：这三条规则确实让 git 忽略这些路径（D16 的实际效果）。
    git = shutil.which("git")
    if git is None:
        pytest.skip("环境无 git，只比对 .gitignore 文本")
    paths = ("configs/app.local.json", "configs/secrets.local.json",
             "configs/vision.local.json")
    r = subprocess.run(
        [git, "check-ignore", "--stdin"], cwd=str(ROOT),
        input="\n".join(paths) + "\n", capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert sorted(r.stdout.split()) == sorted(paths)
