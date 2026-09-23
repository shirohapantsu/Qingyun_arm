"""上层运行期配置 AppConfig 与启动预检。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §2（文件责任表：本模块=AppConfig 等纯数据类、app.json 加载校验、secrets 加载、
           启动预检）
        §3.1（app.json 模板与公开签名、相对路径解析规则、secrets.local.json 固定位置）
        §3.2（启动预检 8 条，**全部检查在 motor 构造之前完成**）
        §8 步骤 1
    docs/实施文档/README.md 决策 D13（启动预检纳入）、D15（配置入口固定、无 CLI）、
    D16（模板命名；相对路径相对该配置文件所在目录解析）、D20（ASR/LLM 模型与 endpoint）

设计要点：
    * 本模块只做配置层的活：纯数据类 + JSON 加载校验 + 启动预检。不导入电机、串口、
      视觉、云的任何实现，preflight 因此可以在驱动构造之前安全调用（初审 L1）。
    * ``preflight`` 的 registry 形参注解为 ``Mapping[str, type]``：``plans.BasePlan``
      是 P1-04 的产物，本阶段尚不存在，不做 import，只做鸭子类型属性检查
      （TASK_ID / AREA_THRESHOLD_M2 / A_PLACE_IDS / B_PLACE_ID / C_PLACE_ID）。
    * 异常分层：``AppConfigError(ValueError)`` 是配置层专用可区分异常；预检失败抛其
      子类 ``PreflightError``，额外携带 ``check`` 名（§3.2 八条检查项的稳定标识），
      便于入口映射 terminate 状态、也便于测试逐条断言。调用方（P1-05 的 main）
      可以只捕获 ``AppConfigError`` 一网打尽。
    * 所有相对路径（含 secrets 的固定位置）都不依赖启动目录：配置内路径相对该配置
      文件所在目录解析，secrets 路径由 ``__file__`` 定位项目根（P1 §3.1）。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# 只为类型注解与 place 查询复用 MotionParams；motion_params 本身不依赖电机/串口。
from configs.motion_params import MotionParams

# --- 固定路径（D15/D16：入口配置名与 secrets 位置都不由 CLI/环境变量决定） ---

# 项目根：本文件在 configs/ 下，父目录的父目录就是项目根。
ROOT = Path(__file__).resolve().parents[1]
# 生产入口固定读的 app 配置（P1-05 main 使用）。
PRODUCTION_APP_PATH = ROOT / "configs" / "app.local.json"
# 独立 mock 入口固定读的 app 配置（P1-05 scripts/run_mock.py 使用，D15）。
MOCK_APP_PATH = ROOT / "configs" / "app.mock.json"
# 密钥文件固定位置（P1 §3.1）。仓库只提交空值模板 configs/secrets.json，本文件进
# .gitignore。preflight 的第 7 条按本模块级变量定位，测试可 monkeypatch 本变量。
SECRETS_LOCAL_PATH = ROOT / "configs" / "secrets.local.json"

# secrets 的两个键（P2 §2/D12）；load_secrets 只保证"两键存在且为字符串"，
# 非空要求属 preflight 的生产附加检查（P1 §3.2 第 7 条）。
SECRET_KEYS = ("dashscope_api_key", "deepseek_api_key")

# 本期唯一注册键（D06/D07：只实例化草莓 Plan，蓝莓口令在 P2 按 invalid 处理）。
ALLOWED_PLAN_KEYS = frozenset({"strawberry"})

# yaw_offset 产品限制集合（P1 §3.2 第 6 条）：本期只接受已标定姿态范围。
ALLOWED_YAW_OFFSET_DEG = (0.0, 90.0)


class AppConfigError(ValueError):
    """app.json / secrets.json 不满足 P1 §3.1 约束时抛出。

    与 ``ParamsError`` 同样的"字段路径 + 具体消息"两参数形态：入口要把它翻译成
    ``terminate("APP_CONFIG_INVALID", detail)``，所以消息必须能直接定位到 JSON 的
    哪一个键。
    """

    def __init__(self, field_path: str, message: str) -> None:
        super().__init__(f"{field_path}: {message}")
        self.field_path = field_path
        self.message = message


class PreflightError(AppConfigError):
    """启动预检（P1 §3.2）某一条不通过。

    ``check`` 是检查项的稳定标识，取值：
    registry / places / area_threshold / vision_config / prompt / yaw_offset /
    secrets / mock_assets / log_dir。调用方只捕获 ``AppConfigError`` 即可同时接住
    加载错误与预检错误；需要区分时用 ``isinstance(exc, PreflightError)``。
    """

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"preflight.{check}", message)
        self.check = check


# ---------------------------------------------------------------------------
# 第 1 节：纯数据类（字段与 P1 §3.1 的签名逐字一致）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsrSettings:
    """asr 段：模型 ID、完整 HTTPS endpoint、单次云请求总时限（D20）。"""

    model: str
    endpoint: str
    cloud_total_timeout_s: float


@dataclass(frozen=True)
class LlmSettings:
    """llm 段：模型 ID、DeepSeek base_url、单次云请求总时限（D20）。"""

    model: str
    base_url: str
    total_timeout_s: float


@dataclass(frozen=True)
class MockSettings:
    """mock 段：脚本化转写序列与视觉 replay 帧目录（P5 §2）。"""

    asr_script_path: Path
    vision_replay_dir: Path


@dataclass(frozen=True)
class AppConfig:
    """app.json 的内存表示；四个路径字段均为已解析的绝对 Path（P1 §3.1）。

    ``mock`` 只在 mock 入口下非 None，其存在性由 ``load_app_config`` 按 entry 校验，
    ``preflight`` 也据此区分生产/替身附加检查（§3.2 第 7、8 条）。
    """

    profile_path: Path
    vision_config_path: Path
    prompt_path: Path
    log_dir: Path
    asr: AsrSettings
    llm: LlmSettings
    mock: MockSettings | None = None


# ---------------------------------------------------------------------------
# 第 2 节：校验原语（与 motion_params 同一套严格度：未知键拒绝、bool 不当数值）
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> float:
    """禁止 NaN/Infinity 进入配置：JSON 扩展字面量一律视为非法。"""
    raise AppConfigError("-", f"不允许出现非有限浮点字面量 {name}")


def _load_json(path: Path, field_path: str) -> Any:
    """读取并解析 JSON 文件，非有限字面量按 _reject_constant 拒绝。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AppConfigError(field_path, f"读取失败：{exc}") from exc
    except UnicodeDecodeError as exc:
        raise AppConfigError(field_path, f"不是 UTF-8 文本：{exc}") from exc
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except AppConfigError:
        raise
    except json.JSONDecodeError as exc:
        raise AppConfigError(field_path, f"不是合法 JSON：{exc}") from exc


def _require_object(payload: Any, expected: set[str] | frozenset[str], path: str) -> dict:
    """拒绝非对象、未知键与缺失键（P1 §3.1"未知键拒绝、必填键齐全"）。"""
    if not isinstance(payload, dict):
        raise AppConfigError(path, f"应为对象，实际是 {type(payload).__name__}")
    got = set(payload)
    unknown = sorted(got - set(expected))
    if unknown:
        raise AppConfigError(path, f"出现未知键 {unknown}")
    missing = sorted(set(expected) - got)
    if missing:
        raise AppConfigError(path, f"缺失键 {missing}")
    return dict(payload)


def _require_str(path: str, v: Any) -> str:
    """必填字符串，且不允许空白串（模型名/endpoint/URL 空值没有可用语义）。"""
    if v is None:
        raise AppConfigError(path, "缺失值（null）")
    if not isinstance(v, str):
        raise AppConfigError(path, f"应为字符串，实际是 {type(v).__name__}")
    if not v.strip():
        raise AppConfigError(path, "不能为空字符串")
    return v


def _require_positive_finite(path: str, v: Any) -> float:
    """超时数值：有限正数；bool 是 int 子类但不是合法数值参数，单独拒绝。"""
    if v is None:
        raise AppConfigError(path, "缺失值（null）")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AppConfigError(path, f"应为数值，实际是 {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise AppConfigError(path, f"必须是有限值，实际是 {f}")
    if f <= 0.0:
        raise AppConfigError(path, f"必须为正，实际是 {f}")
    return f


def _resolve_path(base: Path, path: str, v: Any) -> Path:
    """相对路径以该配置文件所在目录为基准（D16），返回绝对 Path。

    这里只解析、不检查存在性：存在性属 §3.2 预检（profile 的存在/哈希另属
    load_motion_params），加载器保持"结构校验"单一职责，也便于测试注入替身路径。
    """
    raw = _require_str(path, v)
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


# ---------------------------------------------------------------------------
# 第 3 节：app.json / secrets.json 加载
# ---------------------------------------------------------------------------


def load_app_config(path: Path, *, entry: Literal["production", "mock"]) -> AppConfig:
    """加载并校验非机密运行配置，返回路径已解析为绝对的 AppConfig。

    entry 由入口硬编码（D15：无 CLI，生产 main 传 "production"、mock 入口传 "mock"），
    不从 JSON 内容、文件名或环境变量猜测。校验规则（P1 §3.1）：

    * 顶层与 asr/llm/mock 子段的未知键一律拒绝，必填键必须齐全；
    * 四个路径字段与 mock 段两个路径字段为非空字符串，相对该配置文件所在目录解析；
    * 两个超时值为有限正数，bool 拒绝；字符串字段不允许空白；
    * production 出现 mock 段 → 拒绝；mock 缺 mock 段 → 拒绝。
    """
    if entry not in ("production", "mock"):
        raise AppConfigError("entry", f"entry 只能是 'production' 或 'mock'，实际 {entry!r}")

    path = Path(path)
    if not path.is_file():
        raise AppConfigError(str(path), "app 配置文件不存在")
    raw = _load_json(path, str(path))

    base = path.resolve().parent
    expected = {"profile_path", "vision_config_path", "prompt_path", "log_dir", "asr", "llm"}
    if entry == "production" and isinstance(raw, dict) and "mock" in raw:
        raise AppConfigError(
            "mock", "production 入口不得包含 mock 段（D15：mock 走独立入口与 app.mock.json）"
        )
    if entry == "mock":
        expected = expected | {"mock"}
    d = _require_object(raw, expected, str(path))

    asr_d = _require_object(d["asr"], {"model", "endpoint", "cloud_total_timeout_s"}, "asr")
    asr = AsrSettings(
        model=_require_str("asr.model", asr_d["model"]),
        endpoint=_require_str("asr.endpoint", asr_d["endpoint"]),
        cloud_total_timeout_s=_require_positive_finite(
            "asr.cloud_total_timeout_s", asr_d["cloud_total_timeout_s"]
        ),
    )

    llm_d = _require_object(d["llm"], {"model", "base_url", "total_timeout_s"}, "llm")
    llm = LlmSettings(
        model=_require_str("llm.model", llm_d["model"]),
        base_url=_require_str("llm.base_url", llm_d["base_url"]),
        total_timeout_s=_require_positive_finite("llm.total_timeout_s", llm_d["total_timeout_s"]),
    )

    mock: MockSettings | None = None
    if entry == "mock":
        mock_d = _require_object(
            d["mock"], {"asr_script_path", "vision_replay_dir"}, "mock"
        )
        mock = MockSettings(
            asr_script_path=_resolve_path(
                base, "mock.asr_script_path", mock_d["asr_script_path"]
            ),
            vision_replay_dir=_resolve_path(
                base, "mock.vision_replay_dir", mock_d["vision_replay_dir"]
            ),
        )

    return AppConfig(
        profile_path=_resolve_path(base, "profile_path", d["profile_path"]),
        vision_config_path=_resolve_path(base, "vision_config_path", d["vision_config_path"]),
        prompt_path=_resolve_path(base, "prompt_path", d["prompt_path"]),
        log_dir=_resolve_path(base, "log_dir", d["log_dir"]),
        asr=asr,
        llm=llm,
        mock=mock,
    )


def load_secrets(path: Path) -> dict[str, str]:
    """读取密钥文件，返回两个键的原始字符串值（P1 §3.1）。

    这里只保证"文件是合法 JSON、恰好两个已知键、值都是字符串"；**空字符串允许返回**，
    非空要求属生产入口的预检附加检查（§3.2 第 7 条）。返回值只在本进程内使用，
    任何日志/异常消息都不得带上键值（D12）。
    """
    path = Path(path)
    if not path.is_file():
        raise AppConfigError(str(path), "secrets 文件不存在")
    d = _require_object(_load_json(path, str(path)), set(SECRET_KEYS), str(path))
    out: dict[str, str] = {}
    for key in SECRET_KEYS:
        v = d[key]
        if v is None:
            raise AppConfigError(f"secrets.{key}", "缺失值（null）")
        if not isinstance(v, str):
            raise AppConfigError(f"secrets.{key}", f"应为字符串，实际是 {type(v).__name__}")
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# 第 4 节：启动预检（P1 §3.2 八条，顺序与编号一致；全部只依赖 cfg/params/registry）
# ---------------------------------------------------------------------------


def _plan_class_attrs(key: str, cls: Any) -> tuple[tuple[str, ...], str, str]:
    """第 1 条：鸭子类型核验单个 plan 类的契约形状，返回 (A, B, C)。"""
    for attr in ("TASK_ID", "AREA_THRESHOLD_M2", "A_PLACE_IDS", "B_PLACE_ID", "C_PLACE_ID"):
        if not hasattr(cls, attr):
            raise PreflightError("registry", f"注册项 {key!r} 的类缺少契约属性 {attr}")
    task_id = cls.TASK_ID
    if not isinstance(task_id, str) or task_id != key:
        raise PreflightError(
            "registry", f"注册键 {key!r} 与类属性 TASK_ID={task_id!r} 不一致"
        )
    a = cls.A_PLACE_IDS
    if not isinstance(a, tuple):
        raise PreflightError(
            "registry", f"{key}.A_PLACE_IDS 必须是 tuple，实际是 {type(a).__name__}"
        )
    if not a:
        raise PreflightError("registry", f"{key}.A_PLACE_IDS 为空：优品槽位必须非空（D04）")
    for pid in a:
        if not isinstance(pid, str) or not pid.strip():
            raise PreflightError("registry", f"{key}.A_PLACE_IDS 含非字符串或空白项：{pid!r}")
    if len(set(a)) != len(a):
        raise PreflightError("registry", f"{key}.A_PLACE_IDS 存在重复槽位：{a!r}")
    b = cls.B_PLACE_ID
    c = cls.C_PLACE_ID
    for name, value in (("B_PLACE_ID", b), ("C_PLACE_ID", c)):
        if not isinstance(value, str) or not value.strip():
            raise PreflightError("registry", f"{key}.{name} 必须是非空字符串，实际 {value!r}")
    if set(a) & {b} or set(a) & {c} or b == c:
        raise PreflightError(
            "registry", f"{key} 的 A/B/C 三组落点 ID 必须互不重叠：A={a!r} B={b!r} C={c!r}"
        )
    return a, b, c


def _check_registry(registry: Mapping[str, type]) -> list[tuple[str, Any, tuple[str, ...], str, str]]:
    """第 1 条：注册表仅 strawberry，且每个类的 ID 契约成立。"""
    if not isinstance(registry, Mapping):
        raise PreflightError("registry", f"registry 应为映射，实际是 {type(registry).__name__}")
    if set(registry) != set(ALLOWED_PLAN_KEYS):
        raise PreflightError(
            "registry",
            f"注册表键集合必须恰为 {sorted(ALLOWED_PLAN_KEYS)}（D06/D07），"
            f"实际 {sorted(registry)}",
        )
    out = []
    for key, cls in registry.items():
        a, b, c = _plan_class_attrs(key, cls)
        out.append((key, cls, a, b, c))
    return out


def _check_places(plan_rows, params: MotionParams) -> None:
    """第 2 条：每个 place_id 必须存在于 params.places（落点坐标归 P4 标定）。"""
    places = getattr(params, "places", None)
    if not isinstance(places, Mapping):
        raise PreflightError("places", "params.places 不是映射，无法核对落点 ID")
    for key, _cls, a, b, c in plan_rows:
        for pid in (*a, b, c):
            if pid not in places:
                raise PreflightError(
                    "places",
                    f"{key} 的落点 {pid!r} 不在 profile 的 places 中，"
                    f"已有 {sorted(places)}（落点标定归 P4）",
                )


def _check_area_threshold(plan_rows) -> None:
    """第 3 条：AREA_THRESHOLD_M2 已设置、>0、有限、非 bool（D05）。"""
    for key, cls, _a, _b, _c in plan_rows:
        k = cls.AREA_THRESHOLD_M2
        if k is None:
            raise PreflightError(
                "area_threshold",
                f"{key}.AREA_THRESHOLD_M2 为 None：K 未标定，P4 回填实测值前拒绝运行（D05）",
            )
        if isinstance(k, bool) or not isinstance(k, (int, float)):
            raise PreflightError(
                "area_threshold",
                f"{key}.AREA_THRESHOLD_M2 应为数值，实际是 {type(k).__name__}",
            )
        kf = float(k)
        if not math.isfinite(kf) or kf <= 0.0:
            raise PreflightError(
                "area_threshold",
                f"{key}.AREA_THRESHOLD_M2 必须是有限正数（单位 m²），实际是 {kf}",
            )


def _check_vision_config(cfg: AppConfig) -> None:
    """第 4 条（L7 收窄）：只要求文件存在且是合法 JSON，不解析内部字段。"""
    vp = cfg.vision_config_path
    if not vp.is_file():
        raise PreflightError("vision_config", f"vision 配置文件不存在：{vp}")
    try:
        parsed = _load_json(vp, str(vp))
    except AppConfigError as exc:
        raise PreflightError("vision_config", f"vision 配置文件不是合法 JSON：{exc.message}") from exc
    if not isinstance(parsed, dict):
        raise PreflightError(
            "vision_config",
            f"vision 配置文件顶层应为 JSON 对象，实际是 {type(parsed).__name__}"
            "（字段语义校验属 P3 init）",
        )


def _check_prompt(cfg: AppConfig) -> None:
    """第 5 条：prompt 文件存在且非空。"""
    pp = cfg.prompt_path
    if not pp.is_file():
        raise PreflightError("prompt", f"prompt 文件不存在：{pp}")
    try:
        text = pp.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PreflightError("prompt", f"prompt 文件读取失败：{exc}") from exc
    if not text.strip():
        raise PreflightError("prompt", f"prompt 文件为空：{pp}")


def _check_yaw_offset(params: MotionParams) -> None:
    """第 6 条：本期已标定姿态范围的产品限制，yaw_offset_deg ∈ {0, 90}。"""
    grasp = getattr(params, "grasp", None)
    if grasp is None or not hasattr(grasp, "yaw_offset_deg"):
        raise PreflightError("yaw_offset", "params.grasp.yaw_offset_deg 不存在，无法断言")
    v = grasp.yaw_offset_deg
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PreflightError(
            "yaw_offset", f"params.grasp.yaw_offset_deg 应为数值，实际是 {type(v).__name__}"
        )
    if float(v) not in ALLOWED_YAW_OFFSET_DEG:
        raise PreflightError(
            "yaw_offset",
            f"params.grasp.yaw_offset_deg 必须是 {list(ALLOWED_YAW_OFFSET_DEG)} 之一"
            f"（本期已标定姿态范围的产品限制，P1 §3.2 第 6 条），实际是 {float(v)}",
        )


def _check_secrets() -> None:
    """第 7 条（生产附加）：固定位置的 secrets.local.json 两个 API key 非空。"""
    try:
        secrets = load_secrets(SECRETS_LOCAL_PATH)
    except AppConfigError as exc:
        raise PreflightError(
            "secrets", f"读取 {SECRETS_LOCAL_PATH} 失败：{exc.message}"
        ) from exc
    for key in SECRET_KEYS:
        if not secrets[key].strip():
            raise PreflightError(
                "secrets",
                f"{SECRETS_LOCAL_PATH} 的 {key} 为空：生产入口需要两个非空 API key（D12）"
                "（只登记键名，不回显任何键值）",
            )


def _check_mock_assets(cfg: AppConfig) -> None:
    """第 8 条（mock 附加）：脚本化转写文件与 replay 帧目录都存在。"""
    assert cfg.mock is not None  # 调用方按 cfg.mock 判定，此处不重复分支
    if not cfg.mock.asr_script_path.is_file():
        raise PreflightError(
            "mock_assets", f"mock 转写脚本不存在：{cfg.mock.asr_script_path}"
        )
    if not cfg.mock.vision_replay_dir.is_dir():
        raise PreflightError(
            "mock_assets", f"mock 视觉 replay 帧目录不存在：{cfg.mock.vision_replay_dir}"
        )


def _check_log_dir(cfg: AppConfig) -> None:
    """第 8 条（两入口都查）：log_dir 已存在且可写，或可创建后再可写。

    只做目录创建与 os.access 判定，不写探针文件，避免在只读或生产根目录留下垃圾。
    """
    log_dir = cfg.log_dir
    if log_dir.exists() and not log_dir.is_dir():
        raise PreflightError("log_dir", f"log_dir 已存在但不是目录：{log_dir}")
    if not log_dir.is_dir():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PreflightError("log_dir", f"log_dir 无法创建：{log_dir}（{exc}）") from exc
    if not os.access(log_dir, os.W_OK | os.X_OK):
        raise PreflightError("log_dir", f"log_dir 不可写：{log_dir}")


def preflight(cfg: AppConfig, params: MotionParams, registry: Mapping[str, type]) -> None:
    """执行 P1 §3.2 的全部 8 条启动预检；任一条不通过抛 ``PreflightError``。

    只依赖传入的 ``cfg`` / 已加载的 ``params`` / ``registry``，不导入也不触碰电机、
    串口、视觉、云实现——§4 的顺序保证本函数在首次串口连接之前执行，失败时零舵机
    写入（初审 L1）。第 7/8 条按 ``cfg.mock`` 是否为 None 区分入口，入口合法性已由
    ``load_app_config`` 的 entry 校验保证。
    """
    # 1) 注册表：仅 strawberry，TASK_ID 与键一致，A/B/C 契约形状与非重叠
    plan_rows = _check_registry(registry)
    # 2) 每个 place_id 存在于 params.places
    _check_places(plan_rows, params)
    # 3) AREA_THRESHOLD_M2 已标定（K 单位 m²，>0 有限非 bool）
    _check_area_threshold(plan_rows)
    # 4) vision 配置文件存在且为合法 JSON（L7 收窄，不解析内部字段）
    _check_vision_config(cfg)
    # 5) prompt 文件存在且非空
    _check_prompt(cfg)
    # 6) yaw_offset 产品限制断言
    _check_yaw_offset(params)
    # 7) 生产附加：两个 API key 非空；mock 入口跳过（D15 不加载密钥）
    if cfg.mock is None:
        _check_secrets()
    else:
        # 8) mock 附加：替身资产存在
        _check_mock_assets(cfg)
    # 8) 两入口都要求 log_dir 可创建可写
    _check_log_dir(cfg)
