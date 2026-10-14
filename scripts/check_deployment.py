#!/usr/bin/env python3
"""部署发布验收工具：deployment_manifest.json 逐项强制检查（P5 §5/§6；任务 P5-02）。

规范来源（逐字引用，不发明判据）：
    docs/实施文档/P5_集成测试与部署验收技术文档.md
        §5：「生成 deployment_manifest.json，逐项记录资源相对路径、SHA256、对应
             报告；密钥不入哈希清单。发布验收工具检查文件存在、哈希匹配、报告通过、
             未测字段非null，并以real loader重新加载profile。首次生产main真机动作前
             缺任一强制报告/条件不得用"演示过一次"替代。」
        §6：「'演示成功一次'不豁免 verified profile 的发布要求（落地方案 §9.2）」
    docs/实施文档/P4_标定工具与参数发布技术文档.md
        §2.7（promote 产物：verified profile + 验收报告哈希绑定）、
        §3.1（demo_acceptance 严格 schema；「样本清单与报告、vision配置、权重、K、
             motion profile必须在P5部署清单中绑定哈希」；未测 null → 不可验收）
    configs/motion_params.py：load_motion_params(mode="real") 的生产判据
        （status=verified、全部运行期字段非 null、资源哈希、_check_report_binding
        读报告的 overall_pass / operator_reviewed / stages physical 证据 /
        parameter_sha256 绑定）；本工具**原样调用该 real loader**，不复制、不放宽。
    configs/app_config.py：SECRET_KEYS（密钥键名单一来源）；configs/demo_acceptance.json：
        demo_acceptance 四键模板。

本工具的强制检查面（任一不满足即非零退出；工具不实现任何"豁免"开关）：
    a. 每个资源条目/报告条目/声明的 motion profile：相对路径文件存在，且 SHA256 与
       清单一致（哈希用实际 64 位 hex，无占位值）；
    b. 清单引用的每个验收报告（kind=profile_validation，即 calibrate.py validate/
       promote 形态的报告）可解析为 JSON，且其真实 schema 的通过字段 overall_pass 与
       operator_reviewed 为真——判据与生产 real loader configs/motion_params.py
       _check_report_binding 消费的是同一对字段，不发明字段；
    c. demo_acceptance 类报告（kind=demo_acceptance）复用 scripts/fit_grade_threshold.py
       的共用 loader（validate_demo_acceptance，P4 §3.1）：严格 schema 校验 + 四个数值键
       （near_round_min_ratio / max_ripe_error_rate / reference_position_uncertainty_m /
       reference_yaw_uncertainty_deg）任一为 null 即"不完整、不可验收"，不得结论通过
       （P4-T13）；
    d. 对 manifest 声明的 motion profile 用生产 real loader 重新加载：
       load_motion_params(path, mode="real")，抛出任何异常即失败；
    e. 密钥不入哈希清单（P5 §5）：manifest 中任何路径的基名等于 secrets.json 或以
       "secrets." 开头、任何路径段等于 "secrets"、或任何字符串值出现 SECRET_KEYS 键名
       （dashscope_api_key / deepseek_api_key）即拒绝整个清单。本工具从不读取、
       哈希或校验 secrets 文件内容——密钥由操作者现场填写，不在验收面内。

manifest schema 由本工具定义并在此钉死（v1，严格：未知键拒绝、必填键齐全）；
文档模板见 docs/部署检查与验收清单.md。相对路径一律相对 manifest 所在目录解析，
必须是相对 POSIX 风格路径（拒绝绝对路径、反斜杠、空段、"."、".."，拒绝越出根）。

顶层键（全部必填，provenance 除外）：
    schema_version      整数 1（bool/字符串拒绝）
    git_commit          40 位小写十六进制完整 commit（P5 §5「代码版本（git commit）」；
                        工具只校验形态——目标机是否为该 commit 由现场记录与 §4 步骤 1 核对）
    provenance          可选，∈ {"measured","simulated"}，缺省 "measured"。**工具自述的
                        来源标记字段，仅如实回显进结论，绝不用于放宽或收紧任何判定**；
                        synthetic/演示用途的部署树必须显式标 "simulated"（对齐 P5 §2
                        「每个 fixture/场景应标明 simulation，禁止进入实测证据目录」）。
    motion_profile      {"relative_path", "sha256", "verification_report"}——promote
                        产物；verification_report 必须是 reports[] 中 kind=
                        profile_validation 的条目（即 promote 时绑定的那份验收报告）。
    resources           [{"relative_path", "sha256", "role", "reports"}]。
                        REQUIRED_RESOURCE_ROLES 中的每个角色须恰好声明一次，并按
                        REQUIRED_REPORT_LINKS 关联实测报告。profile 引用的模型资源
                        须与声明的 URDF/电机校准资源相同。
    reports             [{"relative_path", "sha256", "kind"}]。
                        必需类别：profile_validation/demo_acceptance/grade_threshold/
                        vision_accuracy/footprint_validation/ripe_confusion；另支持
                        可选 place_calibration。完整要求见部署检查与验收清单。

退出码：
    0  全部强制项通过；
    2  manifest 不可信/不可用（manifest 不存在、非 JSON、schema 违例、不安全路径、
       路径重复、**密钥类条目被拒**——拒绝即不再检查，等同"缺任一强制条件"）；
    3  manifest 合法，但存在未满足的强制项（缺文件/哈希不符/报告不可解析/报告通过
       字段为假/未测字段为 null/real loader 重载失败/引用悬空）。

机器可读结论：stdout 末段输出完整 JSON verdict（含逐项 check_id/ok/detail），可另用
--report PATH 落盘同一 JSON。本工具对部署树只读，绝不写回、绝不修改任何被检文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

# 从任意工作目录执行都要能 import 项目模块（与 scripts/calibrate.py 同一 bootstrap）。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.app_config import SECRET_KEYS  # noqa: E402
from configs.motion_params import load_motion_params, validate_calibration_report, ParamsError  # noqa: E402
from scripts.fit_grade_threshold import (  # noqa: E402
    AcceptanceConfigError,
    validate_demo_acceptance,
)

TOOL_NAME = "check_deployment"
SPEC_REF = "P5 §5/§6；P4 §2.7/§3.1（P4-T13）"

EXIT_OK = 0
EXIT_MANIFEST_REJECTED = 2
EXIT_CHECK_FAILED = 3

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

#: 不豁免条款的逐字提示（P5 §6 / 落地方案 §9.2，经 P5 §5 转引）。
NO_WAIVER_NOTICE = (
    "P5 §6：两种完成标准不互相替代；'演示成功一次'不豁免 verified profile 的发布要求"
    "（落地方案 §9.2）。本工具不提供任何豁免开关；provenance 仅是打包方自述的来源标记，"
    "不参与任何判定。"
)

#: 密钥不入哈希清单（P5 §5）判据的实现口径。
SECRET_STRICT_BASENAMES = {"secrets.json"}
SECRET_BASENAME_PREFIX = "secrets."
SECRET_SEGMENT = "secrets"

MANIFEST_TOP_KEYS = {"schema_version", "git_commit", "motion_profile", "resources", "reports"}
MANIFEST_TOP_OPTIONAL = {"provenance"}
PROVENANCE_VALUES = ("measured", "simulated")
MOTION_PROFILE_KEYS = {"relative_path", "sha256", "verification_report"}
RESOURCE_KEYS = {"relative_path", "sha256", "role", "reports"}
REPORT_KEYS = {"relative_path", "sha256", "kind"}
REPORT_KINDS = ("profile_validation", "demo_acceptance", "grade_threshold", "vision_accuracy",
                "footprint_validation", "ripe_confusion", "place_calibration")
REQUIRED_RESOURCE_ROLES = frozenset({
    "urdf", "motor_calibration", "requirements", "weights", "class_map",
    "camera_intrinsics", "camera_extrinsic", "app_config", "vision_config", "prompt",
    "mock_config", "sample_manifest", "grade_samples",
})
REQUIRED_REPORT_KINDS = frozenset(REPORT_KINDS) - {"place_calibration"}
REQUIRED_REPORT_LINKS = {
    "urdf": {"profile_validation"}, "motor_calibration": {"profile_validation"},
    "camera_intrinsics": {"vision_accuracy"}, "camera_extrinsic": {"vision_accuracy"},
    "weights": {"vision_accuracy", "footprint_validation", "ripe_confusion"},
    "class_map": {"vision_accuracy", "ripe_confusion"},
    "vision_config": {"vision_accuracy", "footprint_validation", "ripe_confusion"},
    "sample_manifest": {"vision_accuracy", "footprint_validation", "ripe_confusion"},
    "grade_samples": {"grade_threshold"},
}


class ManifestRejected(Exception):
    """manifest 不可信/不可用 → 退出码 2（拒绝整个清单，不再逐项检查）。"""


class _ReportFields:
    """profile_validation 报告的通过字段 = 生产 real loader 实际消费的那一对字段。

    configs/motion_params.py::_check_report_binding 用 ``report.get("overall_pass")``
    与 ``report.get("operator_reviewed")`` 判"报告通过"；本工具逐字对齐，不发明字段。
    """

    PASS_FIELDS = ("overall_pass", "operator_reviewed")


def sha256_file(path: Path) -> str:
    """流式实算文件 SHA256（与标定链其余工具同一口径）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. manifest 严格 schema 加载（含路径安全与密钥拒绝）—— 违例即 ManifestRejected
# ---------------------------------------------------------------------------


def _require_keys(payload: Any, expected: set[str], where: str, optional: set[str] = frozenset()) -> dict:
    if not isinstance(payload, dict):
        raise ManifestRejected(f"{where}: 应为 JSON 对象，实际 {type(payload).__name__}")
    unknown = sorted(set(payload) - expected - optional)
    if unknown:
        raise ManifestRejected(f"{where}: 出现未知键 {unknown}（严格 schema 拒绝）")
    missing = sorted(expected - set(payload))
    if missing:
        raise ManifestRejected(f"{where}: 缺失键 {missing}")
    return dict(payload)


def _validate_relpath(value: Any, where: str) -> str:
    """相对路径判据：非空字符串、非绝对、POSIX 风格、无 ./../空段、不越出根。"""
    if not isinstance(value, str) or not value.strip():
        raise ManifestRejected(f"{where}: relative_path 必须是非空字符串，实际 {value!r}")
    if "\\" in value:
        raise ManifestRejected(f"{where}: 不允许反斜杠路径（部署根为 POSIX 相对路径）：{value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or value.startswith("~"):
        raise ManifestRejected(f"{where}: 拒绝绝对路径 {value!r}（清单路径必须相对 manifest 所在目录）")
    for seg in value.split("/"):
        if seg in ("", ".", ".."):
            raise ManifestRejected(f"{where}: 路径含空段或 . / ..（拒绝越根/遍历）：{value!r}")
    if p.name in SECRET_STRICT_BASENAMES or p.name.startswith(SECRET_BASENAME_PREFIX) \
            or SECRET_SEGMENT in p.parts:
        raise ManifestRejected(_secret_rejection_text(value))
    return p.as_posix()


def _validate_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise ManifestRejected(
            f"{where}: sha256 必须是 64 位小写十六进制（实际哈希，不接受占位值），实际 {value!r}")
    return value


def _secret_rejection_text(value: str) -> str:
    return (
        f"密钥不入哈希清单（P5 §5）：清单出现凭据类路径 {value!r}，拒绝整个 manifest。"
        "secrets 由操作者现场填写（configs/secrets.local.json 固定位置，P1 §3.1），"
        "不属于可哈希分发的资源。"
    )


def _scan_secrets(node: Any, where: str = "manifest") -> None:
    """递归扫描 manifest 全部字符串值：SECRET_KEYS 键名一旦出现在任何字符串即拒绝。"""
    if isinstance(node, str):
        for key_name in SECRET_KEYS:
            if key_name in node:
                raise ManifestRejected(
                    f"{where}: 字符串值出现密钥键名 {key_name!r}（configs/app_config.SECRET_KEYS）。"
                    "密钥不入哈希清单（P5 §5），manifest 任何字段都不得携带密钥键名或值。")
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str):
                _scan_secrets(k, f"{where}.{k}")
            _scan_secrets(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_secrets(v, f"{where}[{i}]")


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    """读取并严格校验 deployment_manifest.json，返回 (归一化 manifest, 部署根)。

    部署根 = manifest 所在目录；相对路径全部相对它解析（与 profile.json 相对其所在
    目录解析的仓库惯例一致，P1 §3.1/D16）。任何违例抛 :class:`ManifestRejected`。
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestRejected(f"manifest 不存在：{manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestRejected(f"manifest 不是合法 JSON：{manifest_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ManifestRejected(f"manifest 不是 UTF-8 文本：{manifest_path}: {exc}") from exc

    d = _require_keys(raw, MANIFEST_TOP_KEYS, "manifest", MANIFEST_TOP_OPTIONAL)

    version = d["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ManifestRejected(
            f"schema_version 必须是整数 1，实际 {type(version).__name__}={version!r}")
    commit = d["git_commit"]
    if not isinstance(commit, str) or not GIT_COMMIT_RE.match(commit):
        raise ManifestRejected(
            f"git_commit 必须是 40 位小写十六进制完整 commit（P5 §5 代码版本），实际 {commit!r}")
    provenance = d.get("provenance", "measured")
    if provenance not in PROVENANCE_VALUES:
        raise ManifestRejected(
            f"provenance 只能是 {list(PROVENANCE_VALUES)} 之一（工具自述来源标记，不参与判定），"
            f"实际 {provenance!r}")

    mp = _require_keys(d["motion_profile"], MOTION_PROFILE_KEYS, "motion_profile")
    motion_profile = {
        "relative_path": _validate_relpath(mp["relative_path"], "motion_profile"),
        "sha256": _validate_sha(mp["sha256"], "motion_profile"),
        "verification_report": _validate_relpath(mp["verification_report"],
                                                  "motion_profile.verification_report"),
    }

    if not isinstance(d["resources"], list):
        raise ManifestRejected("resources 必须是数组")
    resources: list[dict[str, Any]] = []
    for i, item in enumerate(d["resources"]):
        rd = _require_keys(item, RESOURCE_KEYS, f"resources[{i}]")
        rel = _validate_relpath(rd["relative_path"], f"resources[{i}]")
        role = rd["role"]
        if not isinstance(role, str) or not role.strip():
            raise ManifestRejected(f"resources[{i}].role 必须是非空字符串，实际 {role!r}")
        if not isinstance(rd["reports"], list):
            raise ManifestRejected(f"resources[{i}].reports 必须是数组（关联报告 relative_path 列表）")
        reports = [_validate_relpath(r, f"resources[{i}].reports[{j}]")
                    for j, r in enumerate(rd["reports"])]
        resources.append({"relative_path": rel, "sha256": _validate_sha(rd["sha256"], f"resources[{i}]"),
                          "role": role, "reports": reports})

    if not isinstance(d["reports"], list):
        raise ManifestRejected("reports 必须是数组")
    reports_list: list[dict[str, Any]] = []
    for i, item in enumerate(d["reports"]):
        rd = _require_keys(item, REPORT_KEYS, f"reports[{i}]")
        rel = _validate_relpath(rd["relative_path"], f"reports[{i}]")
        kind = rd["kind"]
        if kind not in REPORT_KINDS:
            raise ManifestRejected(
                f"reports[{i}].kind 只能是 {list(REPORT_KINDS)}（本工具按报告实际 schema 选判据，"
                f"不发明字段），实际 {kind!r}")
        reports_list.append({"relative_path": rel,
                             "sha256": _validate_sha(rd["sha256"], f"reports[{i}]"),
                             "kind": kind})

    # 全清单路径唯一性：同一文件不得以多重身份出现两次（motion profile / resources /
    # reports 之间也不允许重复声明——重复条目会让"逐项"结论出现两套真相）。
    # verification_report 与 resources[].reports 是**引用**不是声明，不参与声明唯一性；
    # 其悬空/Kind 错配由 _check_references 逐项判定。
    seen: dict[str, str] = {}

    def _claim(path: str, who: str) -> None:
        if path in seen:
            raise ManifestRejected(f"{who}: relative_path {path!r} 与 {seen[path]} 重复声明")
        seen[path] = who

    _claim(motion_profile["relative_path"], "motion_profile")
    for r in resources:
        _claim(r["relative_path"], f"resource:{r['relative_path']}")
    for r in reports_list:
        _claim(r["relative_path"], f"report:{r['relative_path']}")

    # 密钥键名递归扫描（放在结构校验之后：报错信息能引用已解析的字段路径）。
    _scan_secrets(raw)

    return {
        "schema_version": 1,
        "git_commit": commit,
        "provenance": provenance,
        "motion_profile": motion_profile,
        "resources": resources,
        "reports": reports_list,
    }, manifest_path.resolve().parent


# ---------------------------------------------------------------------------
# 2. 逐项强制检查（a 存在+哈希 / b 报告通过 / c 未测 null / d real 重载 / e 已在上）
# ---------------------------------------------------------------------------


def _item(check_id: str, target: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check_id": check_id, "target": target, "ok": bool(ok), "detail": detail}


def _check_exists_and_hash(root: Path, rel: str, declared_sha: str, where: str) -> dict[str, Any]:
    """判据 a：相对路径文件存在 + SHA256 与清单一致（缺文件时不再求哈希，如实报因）。"""
    path = root / rel
    if not path.resolve().is_relative_to(root.resolve()):
        return _item("resource.exists", f"{where}:{rel}", False, "资源解析后越出部署根")
    if not path.is_file():
        return _item("resource.exists", f"{where}:{rel}", False, f"文件不存在：{path}")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        return _item("resource.exists", f"{where}:{rel}", False, f"无法读取以计算 SHA256：{exc}")
    if actual != declared_sha:
        return _item("resource.sha256", f"{where}:{rel}", False,
                     f"哈希与清单不一致：manifest={declared_sha}，实算={actual}")
    return _item("resource.sha256", f"{where}:{rel}", True, "存在且 SHA256 与清单一致")


def _check_profile_validation_report(root: Path, rel: str) -> dict[str, Any]:
    """判据 b：calibrate.py validate/promote 报告可解析且通过字段为真。

    通过字段 = configs/motion_params.py::_check_report_binding 实际消费的
    overall_pass 与 operator_reviewed（truthiness 口径逐字一致：``not report.get(k)``
    即失败），不发明字段。
    """
    path = root / rel
    target = f"report:{rel}"
    if not path.is_file():
        return _item("report.pass", target, False, "报告文件不存在，无法读取通过字段")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _item("report.pass", target, False, f"报告不是合法 JSON：{exc}")
    if not isinstance(report, dict):
        return _item("report.pass", target, False,
                     f"报告顶层应为对象，实际 {type(report).__name__}")
    bad: list[str] = []
    for field in _ReportFields.PASS_FIELDS:
        if field not in report:
            bad.append(f"{field} 缺失（calibrate.py 报告 schema 的通过字段，不允许缺）")
        elif report[field] is not True:
            bad.append(f"{field}={report[field]!r} 不为真")
    if bad:
        return _item("report.pass", target, False, "；".join(bad) +
                     "（P5 §5：报告通过；判据与生产 real loader _check_report_binding 同源）")
    try:
        validate_calibration_report(report)
    except ParamsError as exc:
        return _item("report.pass", target, False, str(exc))
    return _item("report.pass", target, True,
                 "overall_pass 与 operator_reviewed 均为真（字段判据同 configs/motion_params.py）")


def _check_demo_acceptance_report(root: Path, rel: str) -> dict[str, Any]:
    """判据 c：demo_acceptance 类报告的未测字段非 null（P4 §3.1 四键 + 共用 loader）。

    复用 scripts/fit_grade_threshold.validate_demo_acceptance：严格 schema 违例
    （未知键/坏版本/非法值）→ 拒绝；含 null → "不完整、不可验收"，不得结论通过
    （P4-T13）。source=simulated 的名义值如实标注，不冒充实测，但不豁免 null 检查。
    """
    path = root / rel
    target = f"report:{rel}"
    if not path.is_file():
        return _item("report.demo_acceptance", target, False, "报告文件不存在，无法校验四键")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _item("report.demo_acceptance", target, False, f"不是合法 JSON：{exc}")
    try:
        loaded = validate_demo_acceptance(data)
    except AcceptanceConfigError as exc:
        return _item("report.demo_acceptance", target, False, f"demo_acceptance schema 违例：{exc}")
    if not loaded["acceptance_allowed"]:
        return _item(
            "report.demo_acceptance", target, False,
            f"未测字段仍为 null：{loaded['missing_fields']}（P4 §3.1/P4-T13：null → 不完整、"
            "不可验收，工具不得结论通过；对齐 configs/demo_acceptance.json 四键判据）")
    return _item("report.demo_acceptance", target, True,
                 f"四键全部非 null 且合法（source={loaded['source']}；simulated 名义值"
                 "仅作工具链验证，不冒充实测验收）")


def _check_completeness(m: dict[str, Any]) -> list[dict[str, Any]]:
    """检查必需声明与关联；资源不存在和资源未声明都必须拒绝发布。"""
    out = []
    for role in sorted(REQUIRED_RESOURCE_ROLES):
        count = sum(r["role"] == role for r in m["resources"])
        out.append(_item("manifest.required_resource", role, count == 1,
                         f"必需资源角色必须恰好声明一次，实际 {count} 次"))
    for kind in sorted(REQUIRED_REPORT_KINDS):
        count = sum(r["kind"] == kind for r in m["reports"])
        out.append(_item("manifest.required_report", kind, count == 1,
                         f"必需报告类别必须恰好声明一次，实际 {count} 次"))
    reports = {r["relative_path"]: r["kind"] for r in m["reports"]}
    for resource in m["resources"]:
        required = REQUIRED_REPORT_LINKS.get(resource["role"], set())
        linked = {reports.get(path) for path in resource["reports"]}
        missing = sorted(required - linked)
        out.append(_item("manifest.required_binding", resource["relative_path"], not missing,
                         f"缺少关联报告类别：{missing}" if missing else "必需报告关联齐备"))
    return out


def _check_evidence_report(root: Path, rel: str, kind: str, m: dict[str, Any]) -> dict[str, Any]:
    """分级数值报告使用工具的真实 schema；视觉报告必须声明人工复核及通过。"""
    try:
        report = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        return _item("report.pass", rel, False, f"无法读取报告：{exc}")
    if not isinstance(report, dict) or report.get("overall_pass") is not True:
        return _item("report.pass", rel, False, "报告 overall_pass 必须严格为 true")
    if kind == "grade_threshold":
        fit = report.get("fit")
        value = fit.get("area_threshold_m2") if isinstance(fit, dict) else None
        samples = next((r for r in m["resources"] if r["role"] == "grade_samples"), None)
        inputs = report.get("inputs")
        ok = (isinstance(fit, dict) and fit.get("proposable") is True
              and isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(float(value)) and value > 0
              and samples is not None and isinstance(inputs, dict)
              and inputs.get("input_sha256") == samples["sha256"])
        return _item("report.pass", rel, ok, "K 建议与样本哈希绑定合法" if ok
                     else "K 建议缺失/非法或未绑定声明的 grade_samples")
    ok = report.get("operator_reviewed") is True
    return _item("report.pass", rel, ok, "通过且人工复核" if ok
                 else "报告 operator_reviewed 必须严格为 true")


def _check_model_resources(root: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    """profile 实际使用的两个资源必须正是清单中按角色声明的文件。"""
    out = []
    profile_path = root / m["motion_profile"]["relative_path"]
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
        for role, key in (("urdf", "urdf_path"), ("motor_calibration", "motor_calibration_path")):
            entry = next((r for r in m["resources"] if r["role"] == role), None)
            actual = (profile_path.parent / raw["model"][key]).resolve()
            ok = entry is not None and actual == (root / entry["relative_path"]).resolve()
            out.append(_item("resource.profile_binding", role, ok,
                             "与 profile 资源引用一致" if ok else "profile 引用未绑定到声明的资源"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.append(_item("resource.profile_binding", "model", False, str(exc)))
    return out


def _check_references(m: dict[str, Any]) -> list[dict[str, Any]]:
    """关联完整性：资源/ profile 引用的每个报告路径必须能在 reports[] 解析到。"""
    out: list[dict[str, Any]] = []
    declared = {r["relative_path"]: r for r in m["reports"]}
    vr = m["motion_profile"]["verification_report"]
    entry = declared.get(vr)
    if entry is None:
        out.append(_item("report.reference", f"motion_profile.verification_report:{vr}", False,
                         "引用的报告不在 reports[] 中（悬空引用）"))
    elif entry["kind"] != "profile_validation":
        out.append(_item("report.reference", f"motion_profile.verification_report:{vr}", False,
                         f"promote 绑定报告的 kind 必须是 profile_validation，实际 {entry['kind']!r}"))
    else:
        out.append(_item("report.reference", f"motion_profile.verification_report:{vr}", True,
                         "已声明且 kind 匹配"))
    for r in m["resources"]:
        for ref in r["reports"]:
            ok = ref in declared
            out.append(_item(
                "report.reference", f"resource:{r['relative_path']} -> {ref}", ok,
                "已声明" if ok else "关联的报告不在 reports[] 中（悬空引用）"))
    return out


def _check_real_reload(root: Path, m: dict[str, Any]) -> dict[str, Any]:
    """判据 d：对声明的 motion profile 用**生产 real loader**重新加载，异常即失败。

    load_motion_params(mode="real") 自身的判据（status=verified、全部运行期字段非
    null、URDF/电机校准哈希、verification 报告绑定与 parameter_sha256、各阶段
    physical 证据）原样生效；本工具不复制、不放宽任何一条。
    """
    rel = m["motion_profile"]["relative_path"]
    profile_path = root / rel
    try:
        params = load_motion_params(profile_path, mode="real")
    except Exception as exc:  # noqa: BLE001 —— 生产 loader 的任何异常都是强制项失败
        return _item(
            "motion_profile.real_reload", rel, False,
            f"real loader 拒绝加载：{type(exc).__name__}: {exc}。"
            f"注意：{NO_WAIVER_NOTICE}")
    return _item("motion_profile.real_reload", rel, True,
                 f"load_motion_params(mode='real') 通过：profile_id={params.profile_id!r}, "
                 f"status={params.status!r}")


def _check_binding_consistency(root: Path, m: dict[str, Any]) -> dict[str, Any]:
    """manifest 声明的 verification_report 必须与 profile 内部实际绑定的报告同一路径。

    P5 §5「逐项记录资源相对路径、SHA256、对应报告」要求清单与真相一致：若 manifest
    指 A、profile 内部 verification 绑 B，"对应报告"就成了两套真相。profile 不可解析、
    缺 verification 或字段形态不对时本项记"不适用"通过——那些缺陷由判据 a（文件/哈希）
    与判据 d（real loader 重载）各自把住强制闸，这里只补"声明对应性"。
    """
    rel = m["motion_profile"]["relative_path"]
    declared = (root / m["motion_profile"]["verification_report"]).resolve()
    target = f"motion_profile.verification_report:{m['motion_profile']['verification_report']}"
    try:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return _item("report.binding_consistency", target, True,
                     "不适用：profile 当前不可解析，对应性无从对照（存在性与加载由 a/d 强制把关）")
    ver = data.get("verification") if isinstance(data, dict) else None
    if not isinstance(ver, dict) or not isinstance(ver.get("report_path"), str):
        return _item("report.binding_consistency", target, True,
                     "不适用：profile 无 verification 绑定（real loader 将独立拒绝，见 d）")
    try:
        bound = (root / rel).parent / ver["report_path"]
        # 拒绝 profile 内部写越根路径；resolve 归一后仅做对应性比较，不做存在性判定。
        bound = bound.resolve()
    except OSError:
        return _item("report.binding_consistency", target, False, "profile 内部 report_path 无法归一")
    if bound != declared:
        return _item("report.binding_consistency", target, False,
                     f"manifest 声明的绑定报告与 profile 内部 verification.report_path 不一致："
                     f"manifest={declared}，profile={bound}")
    return _item("report.binding_consistency", target, True, "与 profile 内部绑定一致")


def run_deployment_check(manifest_path: Path) -> tuple[int, dict[str, Any]]:
    """执行全部强制检查，返回 (退出码, 机器可读 verdict dict)。"""
    manifest_path = Path(manifest_path)
    try:
        m, root = load_manifest(manifest_path)
    except ManifestRejected as exc:
        verdict = _base_verdict(manifest_path, git_commit=None, provenance=None,
                                 verdict="MANIFEST_REJECTED")
        verdict["checks"] = [_item("manifest.schema", str(manifest_path), False, str(exc))]
        verdict["failed"] = [c["check_id"] + ":" + c["target"] for c in verdict["checks"]]
        verdict["exit_code"] = EXIT_MANIFEST_REJECTED
        return EXIT_MANIFEST_REJECTED, verdict

    checks = _check_completeness(m)
    # a. 逐项：存在 + 哈希（motion profile / reports / resources）
    checks.append(_check_exists_and_hash(root, m["motion_profile"]["relative_path"],
                                         m["motion_profile"]["sha256"], "motion_profile"))
    for r in m["reports"]:
        checks.append(_check_exists_and_hash(root, r["relative_path"], r["sha256"], "report"))
    for r in m["resources"]:
        checks.append(_check_exists_and_hash(root, r["relative_path"], r["sha256"],
                                             f"resource:{r['role']}"))
    # 关联完整性（引用悬空属强制项失败）+ manifest 与 profile 内部绑定的对应性
    checks.extend(_check_references(m))
    checks.append(_check_binding_consistency(root, m))
    checks.extend(_check_model_resources(root, m))
    # b + c. 报告判据（仅对已声明、已解析的报告逐项执行；缺文件也如实产出失败项）
    for r in m["reports"]:
        if r["kind"] == "profile_validation":
            checks.append(_check_profile_validation_report(root, r["relative_path"]))
        elif r["kind"] == "demo_acceptance":
            checks.append(_check_demo_acceptance_report(root, r["relative_path"]))
        else:
            checks.append(_check_evidence_report(root, r["relative_path"], r["kind"], m))
    # d. real loader 重载（无条件执行——这正是"首发生产 main 真机动作前"的那道闸）
    checks.append(_check_real_reload(root, m))

    failed = [c for c in checks if not c["ok"]]
    exit_code = EXIT_OK if not failed else EXIT_CHECK_FAILED
    verdict = _base_verdict(manifest_path, git_commit=m["git_commit"],
                            provenance=m["provenance"],
                            verdict="PASS" if not failed else "FAIL")
    verdict["root"] = str(root)
    verdict["motion_profile"] = m["motion_profile"]["relative_path"]
    verdict["counts"] = {"resources": len(m["resources"]), "reports": len(m["reports"]),
                         "checks": len(checks), "failed": len(failed)}
    verdict["checks"] = checks
    verdict["failed"] = [f"{c['check_id']}:{c['target']}" for c in failed]
    verdict["exit_code"] = exit_code
    return exit_code, verdict


def _base_verdict(manifest_path: Path, *, git_commit: str | None, provenance: str | None,
                  verdict: str) -> dict[str, Any]:
    return {
        "tool": TOOL_NAME,
        "spec": SPEC_REF,
        "verdict": verdict,
        "manifest": str(manifest_path),
        "git_commit": git_commit,
        "provenance": provenance,
        "notice": NO_WAIVER_NOTICE,
    }


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_deployment.py",
        description=(__doc__.splitlines()[0] + "\n\n完整 schema、判据 a–e 与退出码见 "
                     "docs/部署检查与验收清单.md；不豁免条款见 P5 §6（本工具 help 逐字引用："
                     + NO_WAIVER_NOTICE + "）"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="deployment_manifest.json 路径（相对路径以该文件所在目录为部署根）")
    ap.add_argument("--report", type=Path, default=None,
                    help="把机器可读 JSON verdict 另存到该路径（stdout 始终输出同一 JSON）")
    args = ap.parse_args(argv)

    exit_code, verdict = run_deployment_check(args.manifest)
    for c in verdict["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        print(f"[{mark}] {c['check_id']}  {c['target']}  —— {c['detail']}")
    print(f"VERDICT={verdict['verdict']} (exit {exit_code})  "
          f"provenance={verdict['provenance']}  git_commit={verdict['git_commit']}")
    payload = json.dumps(verdict, ensure_ascii=False, indent=2)
    print(payload)
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(payload + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
