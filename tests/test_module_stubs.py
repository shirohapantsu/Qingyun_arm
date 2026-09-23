"""P1 §3.6 三个契约桩（asr / cloud_model / grabbing.vision）的契约测试 + MockMotor 的
``initialize()`` 记录桩测试（P1-03）。

覆盖 P1 §3.6/§8 步骤 3、P2 §2、P3 §3 与任务.md"视觉分工与接口边界"，分两组：

**A. 常驻契约用例**（桩被 P2/P3/队友的真实现替换后**仍然必须成立**，属验收基线）：
    * 三个桩文件存在于固定路径、固定文件名（P1 §1/§2："桩文件不得被重命名或移位"）；
    * 头部 docstring 明示"契约桩——在原文件替换桩体，文件名与公开签名不得改变"；
    * 异常类名、基类与同级关系（``ASRHardError``/``LLMHardError`` 是 ``RuntimeError``
      子类；``NoTarget``/``VisionHardError`` 是 ``Exception`` 子类且互不为父子）；
    * 公开函数的参数名 / 参数种类（keyword-only 形态）/ 默认值 / 类型注解名；
      其中 MockMotor 与真机 ``initialize`` 的**签名形态**必须逐位对等；
    * 干净子进程 import 桩模块成功、无额外输出，且不拉起任何音频/云/相机/检测运行时
      重依赖（P2-T11 / P3-T17 的常驻形态；AST import 白名单给环境无关的第二重保险）。

**B. 桩阶段专属用例**（真实现替换桩体时**必须随之替换**：那时"调用即抛未实现"
    就不再是行为了）：一切公开调用抛对应硬错误、消息含"未实现"与归属阶段
    （P2 / P3-队友）、``get_target`` 任何参数都不抛 ``NoTarget``、桩不返回目标、
    运行时不 import 配置/公共契约层（TYPE_CHECKING 延迟求值）。

MockMotorController 部分（记录桩常驻到 P1-06 真机握手交付之后）：
    显式调用记录参数、构造不自动调用、零总线动作、零时钟推进、与真机签名形态对等。
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from configs.common_interface import MotorController
from qingyun import asr, cloud_model
from qingyun.grabbing import motor_control as MC
from qingyun.grabbing import vision
from tests.mock_motor import MOCK_INITIALIZE_TIMEOUT_S, MockMotorController, SimTime

# 项目根：本文件在 tests/ 下（与 tests/support.py 同一做法）。
ROOT = Path(__file__).resolve().parents[1]

# --- 固定路径与固定文件名（P1 §2 文件责任表；§1"桩文件不得被重命名或移位"） ---
STUB_PATHS: dict[str, Path] = {
    "qingyun.asr": ROOT / "qingyun" / "asr.py",
    "qingyun.cloud_model": ROOT / "qingyun" / "cloud_model.py",
    "qingyun.grabbing.vision": ROOT / "qingyun" / "grabbing" / "vision.py",
}

MODULE_NAMES = tuple(STUB_PATHS)

# 桩消息前缀（三条硬错误消息的公共开头；测试按字面量比对，不引用桩的私有常量，
# 这样"消息文案"本身是被钉死的契约，而不是"与实现一致"的循环论证）。
# 现状：asr（P2-01）与 cloud_model（P2-03）已换成真实实现，各自的前缀只剩历史意义
# ——对应的 B 组用例已按 P2-01 同法改写为"未就绪仍精确抛 HardError + 归因文案"；
# 常量保留是为了让"哪些文案随桩退役"在同一个地方可见。
ASR_MESSAGE_PREFIX = "ASR 未实现（P2 替换本桩）"
LLM_MESSAGE_PREFIX = "LLM 未实现（P2 替换本桩）"
VISION_MESSAGE_PREFIX = "视觉未实现（P3/队友替换本桩）"

# 干净进程 import 桩模块时**绝不允许**出现在 sys.modules 里的重依赖（A 组）。
# 逐名列出，便于审查"检查了哪些"：串口 / 音频采集 / HTTP 客户端 / 检测运行时 /
# 相机 SDK / 运动学栈 / 其他常见重包。
HEAVY_IMPORTS: tuple[str, ...] = (
    "serial",            # pyserial：真机串口
    "sounddevice",       # P2 录音（PortAudio）
    "soundfile",
    "pyaudio",
    "httpx",             # P2 两家云 REST 适配
    "requests",
    "urllib3",
    "aiohttp",
    "ultralytics",       # P3 YOLO-OBB
    "torch",
    "torchvision",
    "onnxruntime",       # P2 §3 silero VAD 首选运行时
    "cv2",               # OpenCV
    "pyorbbecsdk",       # P3 §5 奥比中光 SDK
    "OpenNI2",
    "openni2",
    "placo",             # 运动学栈（语音/视觉桩都不该碰到）
    "pin",
    "hppfcl",
    "coal",
    "scipy",
    "numba",
    "PIL",
    "matplotlib",
)

# 桩阶段专属（B 组）：桩连配置层/公共契约层都不应在运行时拉起——它们只出现在
# TYPE_CHECKING 的注解里。真实现（P2 读 cfg 字段、P3 构造 VisionInterface）会合法地
# 导入这些模块甚至 numpy，所以本名单**只**对桩生效，与 HEAVY_IMPORTS 分开维护。
STUB_ONLY_ABSENT_IMPORTS: tuple[str, ...] = (
    "configs",
    "numpy",
    "qingyun.runlog",
    "qingyun.shutdown",
    "qingyun.grabbing.motor_control",
    "qingyun.grabbing.arm_control",
)

# 每个桩文件"允许出现在顶层（非 TYPE_CHECKING）import 里的根"（AST 静态白名单，
# 与环境装了哪些包无关，故黑名单不会因目标机缺包而空转）。
ALLOWED_IMPORT_ROOTS: dict[str, set[str]] = {
    "qingyun.asr": {"__future__", "typing"},
    "qingyun.cloud_model": {"__future__", "typing"},
    "qingyun.grabbing.vision": {"__future__", "typing", "pathlib"},
}

EMPTY = inspect.Parameter.empty


def _param(name: str, kind: str, default=EMPTY, annotation: str | None = None):
    """期望签名的一行：参数名 / 种类 / 默认值 / 注解名。"""
    return (name, kind, default, annotation)


# 公开签名期望（P1 §3.6 与 P3 §3 逐字）。注解按"名字字符串"比对：桩用
# `from __future__ import annotations` 存字符串，真实现若不加该 future import 就存
# 类对象——_annotation_text 把两者归一，使本组用例在桩被替换后仍可比对（A 组）。
EXPECTED_SIGNATURES: dict[tuple[str, str], tuple[list[tuple], str]] = {
    ("qingyun.asr", "init"): (
        [_param("cfg", "POSITIONAL_OR_KEYWORD", annotation="AppConfig")],
        "None",
    ),
    ("qingyun.asr", "listen_and_transcribe"): ([], "str"),
    ("qingyun.cloud_model", "init"): (
        [_param("cfg", "POSITIONAL_OR_KEYWORD", annotation="AppConfig")],
        "None",
    ),
    ("qingyun.cloud_model", "select_plan"): (
        [_param("text", "POSITIONAL_OR_KEYWORD", annotation="str")],
        "str",
    ),
    ("qingyun.grabbing.vision", "configure"): (
        [
            _param("thresholds", "POSITIONAL_OR_KEYWORD", annotation="VisionThresholds"),
            _param("config_path", "POSITIONAL_OR_KEYWORD", annotation="Path"),
        ],
        "None",
    ),
    ("qingyun.grabbing.vision", "init"): ([], "None"),
    ("qingyun.grabbing.vision", "get_target"): (
        [_param("ignore", "POSITIONAL_OR_KEYWORD", default=0, annotation="int")],
        "VisionInterface",
    ),
}

# 契约面公开名字（B 组：桩阶段的公开面精确等于规格点名的集合）。
# 三张表分别写死（不从签名表反推），再由 test_契约面三张表自洽 交叉核对，
# 这样"漏了一个函数"会在两处同时暴露，而不是被派生关系吞掉。
EXPECTED_EXCEPTIONS: dict[str, set[str]] = {
    "qingyun.asr": {"ASRHardError"},
    "qingyun.cloud_model": {"LLMHardError"},
    "qingyun.grabbing.vision": {"NoTarget", "VisionHardError"},
}
EXPECTED_FUNCTIONS: dict[str, set[str]] = {
    "qingyun.asr": {"init", "listen_and_transcribe"},
    "qingyun.cloud_model": {"init", "select_plan"},
    "qingyun.grabbing.vision": {"configure", "init", "get_target"},
}
EXPECTED_PUBLIC_NAMES: dict[str, set[str]] = {
    name: EXPECTED_FUNCTIONS[name] | EXPECTED_EXCEPTIONS[name] for name in MODULE_NAMES
}
# 桩的公开调用面总数（asr 2 + cloud 2 + vision 3 = 7）。
PUBLIC_CALL_COUNT = sum(len(v) for v in EXPECTED_FUNCTIONS.values())


def _module(name: str):
    return importlib.import_module(name)


def _annotation_text(ann: object) -> str:
    """把注解归一成"名字字符串"：字符串直接取，None/NoneType→'None'，类取 __name__。"""
    if ann is EMPTY:
        return "<empty>"
    if isinstance(ann, str):
        return ann.strip()
    if ann is None or ann is type(None):
        return "None"
    return getattr(ann, "__name__", str(ann))


def _signature_shape(func) -> tuple[list[tuple], str]:
    """(参数四元组列表, 返回注解名)——与 EXPECTED_SIGNATURES 同形，便于直接比对。"""
    sig = inspect.signature(func)
    rows = [
        (p.name, p.kind.name, p.default, _annotation_text(p.annotation))
        for p in sig.parameters.values()
    ]
    return rows, _annotation_text(sig.return_annotation)


def _parameter_rows(sig: inspect.Signature) -> list[tuple]:
    """签名的"形状"四元组：参数名 / 种类 / **是否有默认值**（不比数值）/ 注解名。

    用于 Mock 与真机 initialize 的协议对等比较——两边的 timeout_s 默认值允许取
    各自的名义值（P1 §4.1 末段：真机沿用驱动常量作待实测值），但形状必须同。
    """
    return [
        (p.name, p.kind.name, p.default is not EMPTY, _annotation_text(p.annotation))
        for p in sig.parameters.values()
    ]


def _guarded_node_ids(tree: ast.Module) -> set[int]:
    """所有位于 ``if TYPE_CHECKING:`` 块内部的节点 id（含块本身）。"""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            guarded.update(id(child) for child in ast.walk(node))
    return guarded


def _import_roots(tree: ast.Module, *, skip_guarded: bool) -> list[str]:
    """全部 import 根（``from x import y`` 取 x）；skip_guarded 时排除 TYPE_CHECKING 块。"""
    guarded = _guarded_node_ids(tree) if skip_guarded else set()
    roots: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            roots.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对导入在本仓库不出现；出现即按原样进黑名单核对
                roots.append("." * node.level + (node.module or ""))
            else:
                roots.append(node.module or "")
    return roots


def _clean_import_report(module_names: list[str]) -> subprocess.CompletedProcess:
    """在**全新子进程**里 import 指定桩模块，返回含 sys.modules 与输出的完成对象。

    用子进程而不是当前进程，是因为 pytest 进程早已 import 了 numpy/placo/serial 等，
    "import 无副作用"只有在干净解释器里才可证。子进程 cwd 取项目根（`-c` 会把 cwd
    放进 sys.path），并在代码里再显式 insert 一次项目根，与根 conftest 给 pytest
    注入根目录的做法等价且双保险。
    """
    code = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import importlib
mods = {{n: importlib.import_module(n) for n in {module_names!r}}}

def present(names):
    return [n for n in names if any(m == n or m.startswith(n + ".") for m in sys.modules)]

print("@" + json.dumps({{
    "files": {{n: getattr(m, "__file__", None) for n, m in mods.items()}},
    "heavy": present({list(HEAVY_IMPORTS)!r}),
    "stub_only": present({list(STUB_ONLY_ABSENT_IMPORTS)!r}),
    "qingyun_loaded": sorted(m for m in sys.modules if m.startswith("qingyun")),
}}))
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


def _parse_report(done: subprocess.CompletedProcess) -> dict:
    """取出子进程打印的那一行报告；顺带证明"除这一行外 import 没有输出任何东西"。"""
    lines = done.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("@"), (
        f"import 期间桩自行产生了输出：\n{done.stdout!r}\nstderr:\n{done.stderr!r}"
    )
    return json.loads(lines[0][1:])


# =========================================================================
# A 组：常驻契约（文件位置、头部声明、异常分层、签名形态、import 零副作用）
# =========================================================================


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_桩文件在固定路径且文件名固定(module_name):
    """P1 §1/§2：桩文件不得被重命名或移位——路径本身是契约的一部分。"""
    path = STUB_PATHS[module_name]
    assert path.is_file(), f"{module_name} 的桩文件不在 {path}"
    assert path.stat().st_size > 0, f"{path} 仍是 0 字节占位，P1-03 未落地"
    # 名字形态核对（父目录 + 文件名逐字来自模块名）。
    assert path.name == module_name.rsplit(".", 1)[-1] + ".py"
    assert path.parent == ROOT / Path(*module_name.split(".")[:-1])


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_桩文件头部声明契约桩与替换方式(module_name):
    """P1 §1 + 任务.md：头部必须写明"契约桩 / 在原文件替换桩体 / 签名不得改变"。"""
    doc = ast.get_docstring(ast.parse(STUB_PATHS[module_name].read_text(encoding="utf-8")))
    assert doc, f"{module_name} 缺模块 docstring"
    head = "\n".join(doc.splitlines()[:6])
    assert "契约桩" in head, head
    assert "替换桩体" in head, head
    assert "文件名与公开签名不得改变" in head, head
    # 归属阶段必须写在头部（P2 替换 / P3 替换 / P3-队友）。
    assert ("P2" in head) or ("P3" in head), head


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_干净进程import成功且无输出无重依赖(module_name):
    """P2 §2 要点/P2-T11、P3-T17：import 时不加载模型、不开设备、不请求网络。

    顺带证明"桩可被正常导入"（任务.md 视觉边界）与命名空间包导入形态可用
    （``qingyun``/``qingyun.grabbing`` 无 ``__init__.py``，与仓库既有风格一致）。
    """
    done = _clean_import_report([module_name])
    assert done.returncode == 0, f"import {module_name} 失败：\n{done.stderr}"
    report = _parse_report(done)
    assert report["files"][module_name], f"{module_name} 未成功 import"
    assert Path(report["files"][module_name]).resolve() == STUB_PATHS[module_name].resolve()
    assert report["heavy"] == [], f"{module_name} import 拉起了重依赖 {report['heavy']}"
    # qingyun 包里只应出现"包链 + 本桩"，不得连带拉起兄弟模块。
    parts = module_name.split(".")
    assert set(report["qingyun_loaded"]) == {
        ".".join(parts[:i]) for i in range(1, len(parts) + 1)
    }, report["qingyun_loaded"]


def test_三桩同进程共同import仍无重依赖():
    """P1 §4 步骤 8–10 会同时 import 三个模块；两两之间不得互相牵出设备/云依赖。"""
    done = _clean_import_report(list(MODULE_NAMES))
    assert done.returncode == 0, f"三桩同进程 import 失败：\n{done.stderr}"
    report = _parse_report(done)
    assert set(report["files"]) == set(MODULE_NAMES)
    assert all(report["files"].values()), report["files"]
    assert report["heavy"] == [], report["heavy"]
    assert set(report["qingyun_loaded"]) == set(MODULE_NAMES) | {"qingyun", "qingyun.grabbing"}


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_顶层import白名单_源码层面不出现设备与云依赖(module_name):
    """与子进程检查互补的静态保险：黑名单命中不依赖目标机装了哪些包。

    顶层（非 TYPE_CHECKING）只允许 ``__future__``/``typing``（vision 额外允许标准库
    ``pathlib``——它进的是注解位，实际写法放在 TYPE_CHECKING 块里，白名单取超集）。
    """
    tree = ast.parse(STUB_PATHS[module_name].read_text(encoding="utf-8"))
    allowed_roots = ALLOWED_IMPORT_ROOTS[module_name]
    for imported in _import_roots(tree, skip_guarded=True):
        assert imported.split(".")[0] in allowed_roots, f"{module_name} 顶层导入了 {imported!r}"
    # 全文件（含 TYPE_CHECKING 块）都不允许出现设备/云/检测运行时依赖。
    heavy_roots = {h.split(".")[0] for h in HEAVY_IMPORTS}
    for imported in _import_roots(tree, skip_guarded=False):
        assert imported.split(".")[0] not in heavy_roots, (
            f"{module_name} 即便在 TYPE_CHECKING 块里也不得引用 {imported!r}"
        )


def test_异常类名与基类符合P1_3_6与P3_3逐字():
    """P1 §3.6 / P2 §2 / P3 §3 的三行类声明是逐字契约，基类不许换成别的族。"""
    assert issubclass(asr.ASRHardError, RuntimeError)
    assert asr.ASRHardError.__bases__ == (RuntimeError,)
    assert issubclass(cloud_model.LLMHardError, RuntimeError)
    assert cloud_model.LLMHardError.__bases__ == (RuntimeError,)
    # P3 §3 逐字：两个视觉类都直接继承 Exception，**不是** RuntimeError 族。
    assert vision.NoTarget.__bases__ == (Exception,)
    assert vision.VisionHardError.__bases__ == (Exception,)
    assert not issubclass(vision.NoTarget, RuntimeError)
    assert not issubclass(vision.VisionHardError, RuntimeError)


def test_NoTarget与VisionHardError互不为父子():
    """P1 §6.2 的两条分支靠类型分派，父子类化会让"空"与"坏"重新混起来。"""
    assert not issubclass(vision.NoTarget, vision.VisionHardError)
    assert not issubclass(vision.VisionHardError, vision.NoTarget)
    assert vision.NoTarget is not vision.VisionHardError
    # ASR/LLM 两个 HardError 也各属自己的模块，不共享中间基类（顶层映射按类型分流）。
    assert not issubclass(asr.ASRHardError, cloud_model.LLMHardError)
    assert not issubclass(cloud_model.LLMHardError, asr.ASRHardError)


@pytest.mark.parametrize("module_name,func_name", sorted(EXPECTED_SIGNATURES))
def test_公开函数签名与规格逐项一致(module_name, func_name):
    """参数名 / 参数种类 / 默认值 / 类型注解名逐项比对 P1 §3.6 与 P3 §3。"""
    func = getattr(_module(module_name), func_name)
    expected_rows, expected_return = EXPECTED_SIGNATURES[(module_name, func_name)]
    rows, ret = _signature_shape(func)
    label = f"{module_name}.{func_name}"
    assert [r[0] for r in rows] == [e[0] for e in expected_rows], f"{label} 参数名 {rows}"
    assert [r[1] for r in rows] == [e[1] for e in expected_rows], f"{label} 参数种类 {rows}"
    assert [r[2] for r in rows] == [e[2] for e in expected_rows], f"{label} 默认值 {rows}"
    assert [r[3] for r in rows] == [e[3] for e in expected_rows], f"{label} 注解 {rows}"
    assert ret == expected_return, f"{label} 返回注解 {ret!r} ≠ {expected_return!r}"
    # 规格写"无参"的函数不允许偷带任何位置参数（signature.parameters 会含 self，
    # 而这三个都是模块级函数，故真·零参时参数表必须确实为空）。
    if not expected_rows:
        assert rows == [], f"{label} 应为无参函数，实际 {rows}"
    # 公开面全部是普通函数（不是描述符/类），可按 P1 §4/§5 的模块属性形式调用。
    assert inspect.isfunction(func), label


def test_契约面三张表自洽():
    """签名表 / 异常表 / 函数表必须覆盖同一份规格清单——防止漏测某个函数。"""
    assert set(EXPECTED_SIGNATURES) == {
        (module, func) for module, funcs in EXPECTED_FUNCTIONS.items() for func in funcs
    }
    assert set(EXPECTED_PUBLIC_NAMES) == set(MODULE_NAMES)
    assert PUBLIC_CALL_COUNT == 7 == len(EXPECTED_SIGNATURES)


def test_桩模块的公开名字面即规格所列():
    """P1 §3.6 的契约面是封闭集：桩阶段不多一个公开名（``__all__`` 精确等于规格）。"""
    for module_name, expected in EXPECTED_PUBLIC_NAMES.items():
        module = _module(module_name)
        assert set(module.__all__) == expected, module_name
        # __all__ 里每个名字都必须真的存在，且类/函数属性与表一致。
        for name in expected:
            assert hasattr(module, name), f"{module_name}.__all__ 含不存在的 {name}"
            obj = getattr(module, name)
            if name in EXPECTED_EXCEPTIONS[module_name]:
                assert isinstance(obj, type) and issubclass(obj, BaseException)
            else:
                assert inspect.isfunction(obj)


# =========================================================================
# B 组：桩阶段专属（真实现替换桩体时**必须同步替换本组**——见模块 docstring）
# =========================================================================


def test_asr桩一切公开调用抛ASRHardError(monkeypatch):
    """B 组·桩阶段专属——**P2-01 已把 asr.py 桩体替换为真实录音管线，本用例随之改写**。

    原断言钉的是"调用即抛、消息含'未实现（P2 替换本桩）'"。替换后 asr 已实现，
    "未实现"文案自然不再成立；但**契约不变量必须仍然成立且不弱化**：一切公开调用
    在**未就绪**时仍精确抛 ``ASRHardError``（不返回空串/成功值冒充"本轮无口令"）——

        * ``listen_and_transcribe()`` 在 init 前调用 → 未初始化硬错误（P2-T12）；
        * ``init(cfg)`` 收到畸形 cfg → 校验失败硬错误。

    为此先把运行态强制复位为"未初始化"（monkeypatch 私有 ``_state``），断言异常
    **精确类型**为 ``asr.ASRHardError``（不放宽为 Exception），并钉消息点明所属契约
    与"未就绪/初始化"语义。真实录音管线的正常路径由 ``tests/test_asr_flow.py`` 覆盖。
    """
    monkeypatch.setattr(asr, "_state", None)
    calls = [
        ("init(cfg)", lambda: asr.init(object()), "cfg.asr"),
        ("listen_and_transcribe()", asr.listen_and_transcribe, "未初始化"),
    ]
    assert len(calls) == len(EXPECTED_FUNCTIONS["qingyun.asr"])
    for label, call, note in calls:
        with pytest.raises(asr.ASRHardError) as info:
            call()
        assert type(info.value) is asr.ASRHardError, label
        message = str(info.value)
        # 契约标识（函数名）与"未就绪"归因都必须在消息里可见（否定式，不给成功值）。
        assert label.split("(")[0] in message, f"{label} 消息未点明是哪个契约：{message}"
        assert note in message, f"{label}: {message}"


def test_cloud_model桩一切公开调用抛LLMHardError(monkeypatch):
    """B 组·桩阶段专属——**P2-03 已把 cloud_model.py 桩体替换为 DeepSeek 真实适配，本用例随之改写**。

    原断言钉的是"调用即抛、消息含'未实现（P2 替换本桩）'"。替换后 LLM 已实现，
    "未实现"文案自然不再成立；但**契约不变量必须仍然成立且不弱化**：一切公开调用
    在**未就绪**时仍精确抛 ``LLMHardError``（不返回 "invalid" 冒充"模型判定与本期无关"，
    那会被 P1 §5 主循环读成合法跳过而洗白缺失的模块）——

        * ``select_plan(text)`` 在 init 前调用 → 未初始化硬错误（P2-T12）；
        * ``init(cfg)`` 收到畸形 cfg → ``cfg.llm`` 校验失败硬错误。

    为此先把运行态强制复位为"未初始化"（monkeypatch 私有 ``_state``），断言异常
    **精确类型**为 ``cloud_model.LLMHardError``（不放宽为 Exception），并钉消息点明所属
    契约与"未就绪/初始化"语义。真实请求组装/严格解析/一次重问由
    ``tests/test_cloud_model.py`` 覆盖（与 asr 侧 P2-01 的改写方式同形）。
    """
    monkeypatch.setattr(cloud_model, "_state", None)
    calls = [
        ("init(cfg)", lambda: cloud_model.init(object()), "cfg.llm"),
        ("select_plan(text)", lambda: cloud_model.select_plan("帮我分拣草莓"), "未初始化"),
    ]
    assert len(calls) == len(EXPECTED_FUNCTIONS["qingyun.cloud_model"])
    for label, call, note in calls:
        with pytest.raises(cloud_model.LLMHardError) as info:
            call()
        assert type(info.value) is cloud_model.LLMHardError, label
        message = str(info.value)
        # 契约标识（函数名）与"未就绪"归因都必须在消息里可见（否定式，不给成功值）。
        assert label.split("(")[0] in message, f"{label} 消息未点明是哪个契约：{message}"
        assert note in message, f"{label}: {message}"


def test_vision桩configure抛VisionHardError且消息含契约():
    """configure 的失败族：由 main 在 init 前调用恰好一次（P3 §3/T15）。"""
    with pytest.raises(vision.VisionHardError) as info:
        vision.configure(object(), Path("configs/vision.local.json"))
    exc = info.value
    assert type(exc) is vision.VisionHardError
    assert not isinstance(exc, vision.NoTarget), "configure 不得抛 NoTarget"
    message = str(exc)
    assert message.startswith(VISION_MESSAGE_PREFIX), message
    assert "未实现" in message and "P3" in message and "队友" in message, message
    assert "configure" in message, message


def test_vision桩init抛VisionHardError绝不伪装NoTarget():
    """P1 §4 步骤 10 括注原文：init 阶段故障不得伪装成 NoTarget。"""
    with pytest.raises(vision.VisionHardError) as info:
        vision.init()
    exc = info.value
    assert type(exc) is vision.VisionHardError
    assert not isinstance(exc, vision.NoTarget), "init 故障伪装成 NoTarget 会洗白缺失模块"
    message = str(exc)
    assert message.startswith(VISION_MESSAGE_PREFIX), message
    assert "不得伪装成 NoTarget" in message, message


@pytest.mark.parametrize(
    "ignore",
    [0, 1, 5, 99, -1, -100, 2.0, "0", None, [], object()],
    ids=["0", "1", "5", "99", "-1", "-100", "float", "str", "None", "list", "object"],
)
def test_get_target任何参数都抛VisionHardError绝不抛NoTarget(ignore):
    """任务.md：未实现调用抛 VisionHardError，不能返回固定目标/空成功/用 NoTarget 掩盖。

    含负值与非整数（P3 §3 末段：真实实现里它们同样是 VisionHardError，不静默取 0），
    所以"绝不抛 NoTarget"这半截在替换后仍成立；只有"消息含未实现"半截属桩阶段专属。
    """
    with pytest.raises(vision.VisionHardError) as info:
        vision.get_target(ignore)
    exc = info.value
    assert type(exc) is vision.VisionHardError
    assert not isinstance(exc, vision.NoTarget), "桩绝不允许用 NoTarget 掩盖模块缺失"
    message = str(exc)
    assert message.startswith(VISION_MESSAGE_PREFIX), message
    assert "未实现" in message and "P3" in message and "队友" in message


def test_get_target关键字与位置两种形态都抛错():
    """P3 §3 冻结的默认值 ignore=0 允许 get_target() / get_target(2) / get_target(ignore=2)。"""
    for label, call in (
        ("get_target()", vision.get_target),
        ("get_target(2)", lambda: vision.get_target(2)),
        ("get_target(ignore=2)", lambda: vision.get_target(ignore=2)),
    ):
        with pytest.raises(vision.VisionHardError) as info:
            call()
        assert "get_target" in str(info.value), label


def test_桩公开调用一次都不正常返回():
    """显式反向断言：七个公开函数面全部以异常结束，没有任何一次给出成功值。"""
    stub_calls = [
        ("asr.init", lambda: asr.init(object())),
        ("asr.listen_and_transcribe", asr.listen_and_transcribe),
        ("cloud_model.init", lambda: cloud_model.init(object())),
        ("cloud_model.select_plan", lambda: cloud_model.select_plan("strawberry")),
        ("vision.configure", lambda: vision.configure(object(), Path("x.json"))),
        ("vision.init", vision.init),
        ("vision.get_target", lambda: vision.get_target(0)),
    ]
    assert len(stub_calls) == PUBLIC_CALL_COUNT == 7
    for label, call in stub_calls:
        try:
            value = call()
        except (asr.ASRHardError, cloud_model.LLMHardError, vision.VisionHardError):
            continue
        pytest.fail(f"{label} 正常返回了 {value!r}——桩不得给出任何成功值")


def test_桩运行时不拉起配置层与公共契约层():
    """B 组：注解用 TYPE_CHECKING 延迟求值，运行时零配置依赖（与仓库风格一致）。

    当前进程早已 import 过 configs/numpy，所以这里反查桩模块自己的全局命名空间：
    被延迟导入的名字不得出现在模块 globals 里（出现即说明顶层真的 import 了它们）。
    """
    delayed_names = {
        "qingyun.asr": {"AppConfig", "VisionInterface", "VisionThresholds", "np", "Path"},
        "qingyun.cloud_model": {"AppConfig", "VisionInterface", "np", "Path"},
        "qingyun.grabbing.vision": {
            "AppConfig",
            "VisionInterface",
            "VisionThresholds",
            "np",
            "MotionParams",
        },
    }
    for module_name, forbidden in delayed_names.items():
        namespace = vars(_module(module_name))
        leaked = sorted(name for name in forbidden if name in namespace)
        assert leaked == [], f"{module_name} 运行时全局里出现了 {leaked}"
    # 反向确认：注解仍然是字符串（PEP 563 生效），故 A 组按名字比对成立。
    ann = inspect.signature(vision.configure).parameters["thresholds"].annotation
    assert isinstance(ann, str) and ann == "VisionThresholds", ann
    # 干净进程里的第二重证据：三桩一起 import 也不拉起 configs / numpy / 兄弟模块。
    done = _clean_import_report(list(MODULE_NAMES))
    assert done.returncode == 0, done.stderr
    report = _parse_report(done)
    assert report["stub_only"] == [], f"桩 import 拉起了 {report['stub_only']}"


# =========================================================================
# MockMotorController.initialize —— 协议对等记录桩（P1 §2 表 / §4.1 / §8 步骤 3）
# =========================================================================


@pytest.fixture
def sim_time() -> SimTime:
    return SimTime(start_s=1000.0)


@pytest.fixture
def mock_motor(params, sim_time) -> MockMotorController:
    return MockMotorController(params, sim_time)


def _observable_state(motor: MockMotorController) -> dict:
    """mock 的全部可观察物理量与计数器（构造后与 initialize 后逐项比对用）。"""
    return {
        "q": motor.q.tolist(),
        "q_cmd": motor.q_cmd.tolist(),
        "g": motor.g,
        "g_cmd": motor.g_cmd,
        "g_vel": motor._g_vel,
        "command_count": motor.command_count,
        "read_count": motor.read_count,
        "sequence": motor._sequence,
        "now_s": motor.time.now_s,
    }


def test_mock构造不自动调用initialize(mock_motor):
    """现状钉死：构造即得可用仿真体，且 initialize 台账为空（P1-03 明确保持的行为）。"""
    assert mock_motor.initialize_calls == []
    # 构造也不产生任何总线动作计数——与既有运动回归用例的起点一致。
    assert mock_motor.command_count == 0
    assert mock_motor.read_count == 0


def test_mock显式调用记录参数与次数(mock_motor):
    """默认参调用与两种关键字形态都被原样登记；调用次数 = 台账长度。"""
    mock_motor.initialize()
    mock_motor.initialize(verify_identity=True, enable_torque=False)
    mock_motor.initialize(verify_identity=False, enable_torque=False, timeout_s=0.25)
    assert len(mock_motor.initialize_calls) == 3
    assert mock_motor.initialize_calls == [
        {
            "verify_identity": True,
            "enable_torque": True,
            "timeout_s": MOCK_INITIALIZE_TIMEOUT_S,
        },
        {
            "verify_identity": True,
            "enable_torque": False,
            "timeout_s": MOCK_INITIALIZE_TIMEOUT_S,
        },
        {
            "verify_identity": False,
            "enable_torque": False,
            "timeout_s": 0.25,
        },
    ]
    # 默认值与真机同为 True/True（P1 §4.1：生产不用 verify_identity=False；
    # enable_torque=False 是标定脚本在用的只读路径）。
    defaults = inspect.signature(MockMotorController.initialize).parameters
    assert defaults["verify_identity"].default is True
    assert defaults["enable_torque"].default is True
    timeout_default = defaults["timeout_s"].default
    assert timeout_default == MOCK_INITIALIZE_TIMEOUT_S
    assert isinstance(timeout_default, float) and timeout_default > 0


def test_mock_initialize不做任何总线动作也不推进时钟(mock_motor, sim_time):
    """P1 §4.1 的"零舵机写入"在 mock 侧天然成立：initialize 只记不改。"""
    before = _observable_state(mock_motor)
    assert mock_motor.initialize() is None
    assert _observable_state(mock_motor) == before, "initialize 改变了仿真状态或计数器"
    assert sim_time.now_s == before["now_s"], "initialize 推进了虚拟时钟"
    assert mock_motor.command_count == 0 and mock_motor.read_count == 0
    assert len(mock_motor.initialize_calls) == 1


def test_mock_initialize参数是keyword_only不接受位置参数(mock_motor):
    """真机调用形态 ``initialize(verify_identity=..., enable_torque=...)`` 必须唯一可用。"""
    with pytest.raises(TypeError):
        mock_motor.initialize(True)  # type: ignore[misc]
    with pytest.raises(TypeError):
        mock_motor.initialize(False, True)  # type: ignore[misc]


def test_mock_initialize签名形态与真机逐位对等(mock_motor):
    """P1 §4.1 公开签名对等：参数名、种类、是否带默认值、类型注解全部一致。

    **默认值数值不比对**：真机 timeout_s 沿用驱动私有常量作为待实测预算
    （P1 §4.1 末段），mock 用一个明确的名义值（``MOCK_INITIALIZE_TIMEOUT_S``，
    见开发日志 §7 自主决策）；比对"是否有默认值"即可保证两边的调用方式可互换。
    """
    real_rows = _parameter_rows(inspect.signature(MC.Sts3215MotorController.initialize))
    fake_rows = _parameter_rows(inspect.signature(MockMotorController.initialize))
    # ① 与真机逐位对等（名字/种类/有无默认值/注解名四列全同）。
    assert fake_rows == real_rows, (real_rows, fake_rows)
    # ② 同时独立钉一遍 P1 §4.1 的形态：三个参数全部 keyword-only 且带默认值。
    assert [(name, kind, has_default) for name, kind, has_default, _ in fake_rows] == [
        ("self", "POSITIONAL_OR_KEYWORD", False),
        ("verify_identity", "KEYWORD_ONLY", True),
        ("enable_torque", "KEYWORD_ONLY", True),
        ("timeout_s", "KEYWORD_ONLY", True),
    ], fake_rows
    assert [ann for _, _, _, ann in fake_rows[1:]] == ["bool", "bool", "float"], fake_rows
    # ③ 返回 None；真机确实有公开 initialize（本用例前提），标定脚本的只读路径在用
    #    verify_identity=True/enable_torque=False 这一形态（P1 §4.1）。
    real_sig = inspect.signature(MC.Sts3215MotorController.initialize)
    assert _annotation_text(real_sig.return_annotation) == "None"
    assert _annotation_text(inspect.signature(MockMotorController.initialize).return_annotation) == "None"
    assert callable(MC.Sts3215MotorController.initialize)
    assert mock_motor.initialize.__kwdefaults__ is not None


def test_mock仍然满足MotorController协议且新增方法不破坏既有用法(mock_motor, params):
    """补 initialize 只是**增项**：协议三方法语义与故障注入开关一字不动。"""
    assert isinstance(mock_motor, MotorController)
    for name in ("send_action", "get_feedback", "hold_current", "initialize"):
        assert callable(getattr(mock_motor, name))
    # 既有运动用例依赖的公开面仍在。
    assert mock_motor.params is params
    assert mock_motor.object_gap_m is None
    assert mock_motor.command_count == 0
    mock_motor.all_faults_off()
    assert mock_motor.initialize_calls == [], "all_faults_off 不得清空生命周期台账"
