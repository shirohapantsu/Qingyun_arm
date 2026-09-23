#!/usr/bin/env python3
"""面积分级阈值 K 拟合工具（P4 §2.4，D05；A 类：智能体可编码、离线可验证）。

职责边界（§2.4 逐字）：本工具**消费已提供的**尺寸与人工定级标签 CSV，做纯数值拟合，
输出建议 K 与依据报告。**不做图像采集、不做标签识别**——尺寸 ``length_m×width_m``
来自 ``validate_vision_accuracy``（P4 §2.3）或专用采集模式（视觉队友交付）；人工定级
责任在用户/视觉负责人，工具不做自动标签。不成熟样本判 C 由 ``classify`` 的
``ripe is False`` 优先规则决定、与面积无关，因此 **C 样本不得进入本工具输入**（出现
即拒绝并注明）。

固定 CLI（P4 §3.1「最小 CLI 统一提供 --input --output --report」惯例）::

    fit_grade_threshold.py --input CSV --output 建议K文件 --report PATH [--policy midpoint|a_min]

输入 CSV 列（列名在本 --help 与报告双处固化，逐字）::

    sample_id,grade_label,length_m,width_m

* ``sample_id``：非空、全文唯一（重复拒绝）；
* ``grade_label``：严格 ``A`` 或 ``B``（其他标签含 ``C`` 一律拒绝——不成熟判 C 与面积
  无关，该类样本不属于 K 拟合的分母）；
* ``length_m`` / ``width_m``：有限数值，``length_m >= width_m > 0``（与 P1 §6.4 视觉
  契约同式）；面积 ``area = length_m * width_m``，单位 m²。

拟合与分界语义（必须与 P1 §3.5 ``BasePlan.classify`` 一致：``area >= K → A``，**等号
归 A**）：

* ``a_min = min(area | A 组)``，``b_max = max(area | B 组)``；
* **当且仅当 ``a_min > b_max``**（两组面积严格分离）才给出建议 K；``a_min <= b_max``
  时两组重叠，任何 K 都会把某个人工 A 判成 B 或把人工 B 判成 A——拒绝建议（非零退
  出），报告输出两组分位数/散布数据供人工复核；
* 策略 ``--policy``：
  - ``midpoint``（默认）：``K = (b_max + a_min) / 2``，两组都留正余量（浮点相邻导致
    中点并入 ``b_max`` 时如实上调，绝不允许 ``K <= b_max``——那会把 B 边界样本判成
    A）；
  - ``a_min``：``K = a_min``，人工 A 组最小样本恰好 ``area == K`` 判 A（等号归 A 合
    法，B 侧余量最大、A 侧余量为 0）。
  两种策略都满足「边界样本归 A 验证 ``area == K`` → A」；报告说明依据与两组余量。

输出（**工具绝不写入 plans 代码**）：

* ``--output``：建议 K 文件（JSON），供人工回填
  ``qingyun/plans/strawberry.py::AREA_THRESHOLD_M2``；含回填提示「回填后须以 classify
  单测回归」；
* ``--report``：JSON 报告（输入 SHA256、样本数与全部样本分母、两组面积分位数/散布、
  a_min/b_max、建议 K、策略、余量、职责与来源注记）。K 单位 m²，断言有限正数。

退出码：``0`` 给出建议并写出两份文件；``2`` 输入级错误（缺文件/坏 CSV/坏数值/重复
ID/C 或其他标签混入/空组/未知策略参数——argparse 缺参同样非零）；``3`` 两组面积重
叠，拒绝建议（报告已写、建议文件不写）。

同文件附带 **demo_acceptance 共用 loader**（P4 §3.1）：提交模板
``configs/demo_acceptance.json`` 逐字 schema
``{"schema_version":1,"near_round_min_ratio":null,"max_ripe_error_rate":null,
"reference_position_uncertainty_m":null,"reference_yaw_uncertainty_deg":null}``，
实测结果写 ``configs/demo_acceptance.local.json``（gitignore）。严格校验：未知键拒
绝、``schema_version`` 必须是整数 1、数值合法性 ``near_round_min_ratio > 1``、
``max_ripe_error_rate ∈ [0,1]``、两个不确定度有限非负、**bool 非法**。模板 null →
消费工具只能输出「不完整、不可验收」报告（``acceptance_allowed=False``），**不允许
结论通过**；独立模拟配置允许名义值但必须携带可选键 ``"source": "simulated"`` 如实标
注（缺省视为 ``measured``）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_NO_PROPOSAL = 3
#: 「不完整、不可验收」专用退出码（demo_acceptance 消费工具可直接透传；
#: 与本工具的 2/3 段错开，不与既有惯例冲突）。
EXIT_NOT_ACCEPTABLE = 4

TOOL_NAME = "fit_grade_threshold"
SPEC_REF = "P4 §2.4/§3.1（D05）"

#: 输入 CSV 列（列名在此与 --help、报告三处固化为同一常量）。
CSV_COLUMNS = ("sample_id", "grade_label", "length_m", "width_m")

POLICY_MIDPOINT = "midpoint"
POLICY_A_MIN = "a_min"
POLICIES = (POLICY_MIDPOINT, POLICY_A_MIN)

BACKFILL_TARGET = "qingyun/plans/strawberry.py::AREA_THRESHOLD_M2"
CLASSIFY_SEMANTICS_NOTE = (
    "分界语义与 P1 §3.5 BasePlan.classify 一致：area = length_m×width_m ≥ K → A（等号归 A）、"
    "area < K → B；不成熟（ripe is False）优先判 C、与面积无关。"
)
AREA_SOURCE_NOTE = (
    "面积输入来源 = scripts/validate_vision_accuracy.py（P4 §2.3）或专用采集模式（视觉队友交付）；"
    "本工具不做任何图像逻辑、不采集、不识别标签。"
)
LABEL_RESPONSIBILITY_NOTE = (
    "人工定级责任在用户/视觉负责人（§2.4）；本工具只消费提供的 A/B 标签做数值拟合，不做自动标签。"
)
C_REJECTION_NOTE = (
    "不成熟由 classify 判 C、与面积无关（P1 §3.5）；C/未成熟样本不得进入 K 拟合输入，"
    "出现 C 或其他标签一律拒绝。"
)
BACKFILL_NOTE = (
    f"建议 K 仅供人工复核回填 {BACKFILL_TARGET}；本工具绝不写入 plans 代码。"
    "回填后须以 classify 单测回归（tests/test_plan_flow.py P1-T32 等），"
    "并验证边界样本 area == K 判 A。"
)


class InputError(RuntimeError):
    """输入级错误 → 退出码 2（不写任何输出文件）。"""


# ---------------------------------------------------------------------------
# 0. 通用小工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """流式实算文件 SHA256（与标定链其余工具同一口径）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数（输入必须已升序非空）；q ∈ [0,100]，P05/P50/P95 等。"""
    vals = list(sorted_values)
    if not vals:
        raise ValueError("percentile: 空序列")
    if len(vals) == 1:
        return float(vals[0])
    pos = (q / 100.0) * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo]) * (1.0 - frac) + float(vals[hi]) * frac


def _plain_float(where: str, text: Any) -> float:
    """CSV 字段 → 有限 float；空/非数值/非有限拒绝。"""
    if text is None or not str(text).strip():
        raise ValueError(f"{where}: 缺值（空字段）")
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{where}: 无法解析为数值：{text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: 非有限数值 {value!r}")
    return value


# ---------------------------------------------------------------------------
# 1. 样本 CSV 加载与校验（§2.4 输入契约）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradeSample:
    line_no: int          # CSV 物理行号（含表头，从 2 起）——报告定位用
    sample_id: str
    grade: str            # 严格 "A" / "B"
    length_m: float
    width_m: float

    @property
    def area_m2(self) -> float:
        """面积与 classify 完全同一表达式：float(length) * float(width)。"""
        return self.length_m * self.width_m


def load_grade_samples(path: Path) -> list[GradeSample]:
    """读取并严格校验样本 CSV；任何一行不合法即 InputError（整体拒绝，不做静默剔除）。

    判据（逐条对应任务书/§2.4）：
    * 表头逐字等于 ``,``.join(CSV_COLUMNS)（缺列/多列/改名拒绝）；
    * ``sample_id`` 非空且全文唯一；
    * ``grade_label`` 严格 A/B；C 或其他 → 拒绝并附 :data:`C_REJECTION_NOTE`；
    * ``length_m``/``width_m`` 有限、``width_m > 0``、``length_m >= width_m``；
    * A、B 两组皆非空（空组拒绝）。
    """
    if not path.is_file():
        raise InputError(f"输入样本 CSV 不存在：{path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InputError(f"输入样本 CSV 不可读：{path}: {exc}") from exc
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration:
        raise InputError("输入 CSV 为空（至少需要表头行）") from None
    header_norm = [h.strip() for h in header]
    if header_norm != list(CSV_COLUMNS):
        raise InputError(
            f"CSV 表头应逐字为 {','.join(CSV_COLUMNS)}，实际 {header!r}"
            f"（缺列/多列/改名都会破坏输入 schema，拒绝）")

    samples: list[GradeSample] = []
    problems: list[str] = []
    seen_ids: dict[str, int] = {}
    for offset, row in enumerate(reader):
        line_no = offset + 2
        if not any((cell or "").strip() for cell in row):
            continue  # 纯空行（如文件尾换行）跳过，不计入分母
        if len(row) != len(CSV_COLUMNS):
            problems.append(
                f"第 {line_no} 行：应为 {len(CSV_COLUMNS)} 列，实际 {len(row)} 列：{row!r}")
            continue
        sid_raw, grade_raw, length_raw, width_raw = row
        sid = (sid_raw or "").strip()
        if not sid:
            problems.append(f"第 {line_no} 行：sample_id 为空")
            continue
        if sid in seen_ids:
            problems.append(
                f"第 {line_no} 行：sample_id {sid!r} 与第 {seen_ids[sid]} 行重复（拒绝）")
            continue
        seen_ids[sid] = line_no
        grade = (grade_raw or "").strip()
        if grade not in ("A", "B"):
            note = "（仅接受 A/B）"
            if grade.upper() == "C" or grade.lower() in ("c", "unripe", "unmature", "未成熟"):
                note = f"（{C_REJECTION_NOTE}）"
            problems.append(f"第 {line_no} 行：grade_label {grade!r} 非法 {note}")
            continue
        try:
            length_m = _plain_float(f"第 {line_no} 行 length_m", length_raw)
            width_m = _plain_float(f"第 {line_no} 行 width_m", width_raw)
        except ValueError as exc:
            problems.append(f"第 {line_no} 行：{exc}")
            continue
        if not width_m > 0.0:
            problems.append(f"第 {line_no} 行：width_m 必须 > 0，实际 {width_m}")
            continue
        if length_m < width_m:
            problems.append(
                f"第 {line_no} 行：length_m({length_m}) < width_m({width_m})，"
                "违反 P1 §6.4 视觉契约（OBB 长短轴）")
            continue
        samples.append(GradeSample(line_no, sid, grade, length_m, width_m))

    if problems:
        detail = "；".join(problems[:20])
        more = f"（另有 {len(problems) - 20} 条同类问题）" if len(problems) > 20 else ""
        raise InputError(f"样本 CSV 存在 {len(problems)} 处非法记录：{detail}{more}")
    if not samples:
        raise InputError("样本 CSV 无任何数据行（空组拒绝：K 拟合需要 A/B 两组已定级样本）")
    n_a = sum(1 for s in samples if s.grade == "A")
    n_b = sum(1 for s in samples if s.grade == "B")
    if n_a == 0:
        raise InputError("A 组为空（空组拒绝：无成熟优品样本则无法给出下界）")
    if n_b == 0:
        raise InputError("B 组为空（空组拒绝：无成熟次品样本则无法给出上界）")
    return samples


# ---------------------------------------------------------------------------
# 2. 拟合核心（纯函数：不写文件、不抛判据异常）
# ---------------------------------------------------------------------------


def group_area_stats(samples: Sequence[GradeSample], grade: str) -> dict[str, Any]:
    """某组面积散布画像：n/min/max/mean + 分位数（重叠拒绝时的复核材料）。"""
    areas = sorted(s.area_m2 for s in samples if s.grade == grade)
    return {
        "count": len(areas),
        "min_m2": areas[0] if areas else None,
        "max_m2": areas[-1] if areas else None,
        "mean_m2": (sum(areas) / len(areas)) if areas else None,
        "p05_m2": percentile(areas, 5.0) if areas else None,
        "p25_m2": percentile(areas, 25.0) if areas else None,
        "p50_m2": percentile(areas, 50.0) if areas else None,
        "p75_m2": percentile(areas, 75.0) if areas else None,
        "p95_m2": percentile(areas, 95.0) if areas else None,
    }


def fit_grade_threshold(samples: Sequence[GradeSample], *, policy: str = POLICY_MIDPOINT) -> dict[str, Any]:
    """按 §2.4 给出建议 K 或拒绝理由；返回结构化结果（含两组余量与逐样本回归标记）。

    不变量（成功时，全部由 classify 语义 ``area >= K → A`` 反推）：
    * ``b_max < K <= a_min``——B 组全部面积严格小于 K（否则 B 边界会被判 A），
      A 组全部面积不小于 K（等号归 A 合法）；
    * ``K`` 为有限正数（单位 m²）；
    * 逐样本回代：所有人工 A 判 A、所有人工 B 判 B（``verify_against_samples``）。
    """
    a_areas = [s.area_m2 for s in samples if s.grade == "A"]
    b_areas = [s.area_m2 for s in samples if s.grade == "B"]
    a_min = min(a_areas)
    b_max = max(b_areas)
    result: dict[str, Any] = {
        "policy": policy,
        "unit": "m^2",
        "a_min_m2": a_min,
        "b_max_m2": b_max,
        "separation_gap_m2": a_min - b_max,
        "proposable": False,
        "reason": None,
        "area_threshold_m2": None,
        "margin_a_m2": None,   # a_min - K：A 组最小样本到分界的余量（>=0，等号归 A）
        "margin_b_m2": None,   # K - b_max：B 组最大样本到分界的余量（必须 >0）
        "policy_adjustment": None,
    }
    if a_min <= b_max:
        overlap_lo = a_min
        overlap_hi = b_max
        a_in_overlap = sum(1 for v in a_areas if v <= overlap_hi)
        b_in_overlap = sum(1 for v in b_areas if v >= overlap_lo)
        result["reason"] = (
            f"两组面积重叠：a_min({a_min}) <= b_max({b_max})，重叠区间 "
            f"[{overlap_lo}, {overlap_hi}]（A 组落入上界的样本 {a_in_overlap} 个、"
            f"B 组达到下界的样本 {b_in_overlap} 个）。任何 K 都无法同时保住全部人工 A "
            "判 A 与全部人工 B 判 B——拒绝建议，请人工复核下方分位数/散布数据、"
            "定级口径与尺寸测量后重新提交。"
        )
        return result

    if policy == POLICY_A_MIN:
        k = a_min
        result["policy_basis"] = (
            "K = a_min：A 组最小样本恰好 area == K，依『等号归 A』判 A（边界样本归 A "
            "验证兼容）；B 侧余量 = a_min - b_max 最大，A 侧余量为 0。"
        )
    else:
        k = (b_max + a_min) / 2.0
        result["policy_basis"] = (
            "K = (b_max + a_min)/2：分界严格落在两组之间，两侧余量都为正；"
            "area == K 的边界样本按 classify 等号归 A 判 A，与『边界样本归 A 验证』兼容。"
        )
        if k <= b_max:
            # 仅在 b_max 与 a_min 浮点相邻时可能发生：中点向下舍入并入 b_max。
            # K == b_max 会把 B 边界样本判成 A（等号归 A），绝不允许——如实上调到 a_min。
            k = a_min
            result["policy_adjustment"] = (
                "两组边界浮点相邻，中点舍入并入 b_max 会把 B 边界样本判成 A；"
                "已如实上调为 K = a_min（仍满足 b_max < K <= a_min）。"
            )

    if not (math.isfinite(k) and k > 0.0):
        result["reason"] = f"内部一致性失败：建议 K 非有限正数（{k!r}，单位 m²），拒绝。"
        return result
    if not (b_max < k <= a_min):
        result["reason"] = f"内部一致性失败：K({k}) 不满足 b_max < K <= a_min，拒绝。"
        return result

    verify = verify_against_samples(samples, k)
    if not verify["all_consistent"]:
        result["reason"] = f"内部一致性失败：建议 K 与逐样本回代矛盾：{verify['inconsistent']}"
        return result

    result["proposable"] = True
    result["area_threshold_m2"] = k
    result["margin_a_m2"] = a_min - k
    result["margin_b_m2"] = k - b_max
    result["verify"] = verify
    return result


def classify_area(area_m2: float, k: float) -> str:
    """与 P1 §3.5 ``BasePlan.classify`` 逐字同式：``area >= k → A``（等号归 A），否则 B。"""
    return "A" if area_m2 >= k else "B"


def verify_against_samples(samples: Sequence[GradeSample], k: float) -> dict[str, Any]:
    """逐样本回代：每个人工 A 必须判 A、每个人工 B 必须判 B；返回不一致清单（分母=全部样本）。"""
    inconsistent = []
    for s in samples:
        got = classify_area(s.area_m2, k)
        if got != s.grade:
            inconsistent.append({
                "sample_id": s.sample_id, "line_no": s.line_no,
                "grade_label": s.grade, "area_m2": s.area_m2, "classify_with_k": got,
            })
    return {
        "total_samples": len(samples),
        "all_consistent": not inconsistent,
        "inconsistent": inconsistent,
    }


# ---------------------------------------------------------------------------
# 3. demo_acceptance 共用 loader（P4 §3.1；模板 null → 不可验收、不允许结论通过）
# ---------------------------------------------------------------------------

#: §3.1 逐字 schema 的五个必填键（顺序即模板顺序）。
DEMO_ACCEPTANCE_KEYS = (
    "schema_version",
    "near_round_min_ratio",
    "max_ripe_error_rate",
    "reference_position_uncertainty_m",
    "reference_yaw_uncertainty_deg",
)
#: 四个数值键（null = 未测；完成度按它们判定）。
DEMO_ACCEPTANCE_VALUE_KEYS = DEMO_ACCEPTANCE_KEYS[1:]
#: 可选标注键：独立模拟配置的名义值必须如实标 ``simulated``（缺省视为实测/本地文件）。
DEMO_ACCEPTANCE_SOURCE_KEY = "source"
DEMO_ACCEPTANCE_SOURCES = ("measured", "simulated")

DEMO_ACCEPTANCE_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo_acceptance.json"


class AcceptanceConfigError(ValueError):
    """demo_acceptance 严格 schema 违例（未知键/非法值/坏版本——一律拒绝加载）。"""


def _acceptance_number(where: str, value: Any, *, lo: float | None, hi: float | None,
                       lo_open: bool = False) -> float:
    """单个数值判据：非 bool 的 int/float、有限、落在给定区间（lo_open=True 时 lo 取开区间）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceConfigError(
            f"{where}: 应为数值或 null（bool 非法），实际 {type(value).__name__}={value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise AcceptanceConfigError(f"{where}: 非有限数值 {f}")
    if lo is not None and (f <= lo if lo_open else f < lo):
        bound = "> " if lo_open else ">= "
        raise AcceptanceConfigError(f"{where}: 必须 {bound}{lo}（真值语义），实际 {f}")
    if hi is not None and f > hi:
        raise AcceptanceConfigError(f"{where}: 必须 <= {hi}，实际 {f}")
    return f


def validate_demo_acceptance(data: Any) -> dict[str, Any]:
    """严格校验 demo_acceptance 配置（§3.1），返回归一化结果。

    返回 ``{"schema_version", "values": {四键: float|None}, "missing_fields",
    "complete", "acceptance_allowed", "source"}``：

    * 未知键拒绝（仅允许五键 + 可选 ``source``；``source`` ∈ {measured, simulated}）；
    * ``schema_version`` 必须是整数 1（bool/字符串/其他数值拒绝）；
    * 数值合法性：ratio > 1、rate ∈ [0,1]、两不确定度有限非负；bool 非法；
    * 含 null → ``complete=False`` → ``acceptance_allowed=False``（工具只能输出
      「不完整、不可验收」报告，**不得结论通过**）；
    * 全非 null 且合法 → ``complete=True``，允许验收结论（是否真实测量另看 ``source``）。
    """
    if not isinstance(data, dict):
        raise AcceptanceConfigError(f"demo_acceptance 顶层应为对象，实际 {type(data).__name__}")
    unknown = sorted(set(data) - set(DEMO_ACCEPTANCE_KEYS) - {DEMO_ACCEPTANCE_SOURCE_KEY})
    if unknown:
        raise AcceptanceConfigError(f"demo_acceptance 含未知键 {unknown}（严格 schema 拒绝）")
    missing_keys = sorted(set(DEMO_ACCEPTANCE_KEYS) - set(data))
    if missing_keys:
        raise AcceptanceConfigError(f"demo_acceptance 缺键 {missing_keys}（五键必须齐备）")

    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise AcceptanceConfigError(
            f"demo_acceptance.schema_version 必须是整数 1，实际 {type(version).__name__}={version!r}")

    source = data.get(DEMO_ACCEPTANCE_SOURCE_KEY, "measured")
    if source not in DEMO_ACCEPTANCE_SOURCES:
        raise AcceptanceConfigError(
            f"demo_acceptance.source 只能是 {list(DEMO_ACCEPTANCE_SOURCES)}，实际 {source!r}"
            "（独立模拟配置的名义值必须显式标 simulated，不得混充实测）")

    rules = {
        "near_round_min_ratio": dict(lo=1.0, hi=None, lo_open=True),   # ratio > 1
        "max_ripe_error_rate": dict(lo=0.0, hi=1.0, lo_open=False),    # rate ∈ [0,1]
        "reference_position_uncertainty_m": dict(lo=0.0, hi=None),     # 有限非负
        "reference_yaw_uncertainty_deg": dict(lo=0.0, hi=None),        # 有限非负
    }
    values: dict[str, float | None] = {}
    for key, kwargs in rules.items():
        raw = data[key]
        if raw is None:
            values[key] = None
        else:
            values[key] = _acceptance_number(f"demo_acceptance.{key}", raw, **kwargs)

    missing = [k for k in DEMO_ACCEPTANCE_VALUE_KEYS if values[k] is None]
    complete = not missing
    return {
        "schema_version": 1,
        "values": values,
        "missing_fields": missing,
        "complete": complete,
        # null → 不可验收（P4-T13）：这是消费工具输出验收结论前必须过的闸。
        "acceptance_allowed": complete,
        "source": source,
    }


def load_demo_acceptance(path: Path) -> dict[str, Any]:
    """从 JSON 文件加载并严格校验；文件级错误也走 :class:`AcceptanceConfigError`。"""
    path = Path(path)
    if not path.is_file():
        raise AcceptanceConfigError(f"demo_acceptance 文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcceptanceConfigError(f"demo_acceptance 不是合法 JSON：{path}: {exc}") from exc
    loaded = validate_demo_acceptance(data)
    loaded["path"] = str(path)
    return loaded


def evaluate_demo_acceptance(path_or_data: Path | Mapping[str, Any] | str) -> dict[str, Any]:
    """消费侧统一入口：加载/校验 + 给出「可验收 / 不完整、不可验收」结论块。

    ``verdict`` ∈ {"acceptable", "incomplete_not_acceptable"}；``exit_code`` 供只以
    本文件为准的工具直接透传（0 可验收 / 4 不完整、不可验收；1..3 段留给输入/判据类
    失败，与其他工具惯例不冲突）。不完整时**永远给不出 acceptable**。
    """
    if isinstance(path_or_data, Mapping):
        loaded = validate_demo_acceptance(dict(path_or_data))
    else:
        loaded = load_demo_acceptance(Path(path_or_data))
    complete = loaded["complete"]
    return {
        "valid": True,
        "source": loaded["source"],
        "values": loaded["values"],
        "missing_fields": loaded["missing_fields"],
        "complete": complete,
        "acceptance_allowed": complete,
        "verdict": "acceptable" if complete else "incomplete_not_acceptable",
        "verdict_text": (
            "完备：允许验收结论（source=simulated 时为名义模拟值，仅作工具/回放验证，"
            "不得冒充实测验收）" if complete and loaded["source"] == "simulated" else
            "完备：允许验收结论" if complete else
            "不完整、不可验收：存在未测 null 字段，工具不得结论通过（P4 §3.1/P4-T13）"
        ),
        "exit_code": EXIT_OK if complete else EXIT_NOT_ACCEPTABLE,
    }


# ---------------------------------------------------------------------------
# 4. 报告与建议值文件
# ---------------------------------------------------------------------------


def build_report(input_path: Path, input_sha: str, samples: Sequence[GradeSample],
                 fit: Mapping[str, Any], output_path: Path, *, output_written: bool) -> dict[str, Any]:
    per_sample = [{
        "sample_id": s.sample_id, "line_no": s.line_no, "grade_label": s.grade,
        "length_m": s.length_m, "width_m": s.width_m, "area_m2": s.area_m2,
    } for s in samples]
    report: dict[str, Any] = {
        "tool": TOOL_NAME,
        "spec": SPEC_REF,
        "inputs": {
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "input_schema": ",".join(CSV_COLUMNS),
            "policy": fit["policy"],
        },
        "samples": {
            "total_rows": len(samples),
            "a_count": sum(1 for s in samples if s.grade == "A"),
            "b_count": sum(1 for s in samples if s.grade == "B"),
            "denominator_note": "全部样本都计入分母与逐样本回代，不为拟合便利剔除任何样本。",
            "detail": per_sample,
        },
        "area_stats": {
            "unit": "m^2",
            "a_group": group_area_stats(samples, "A"),
            "b_group": group_area_stats(samples, "B"),
        },
        "fit": {k: fit[k] for k in (
            "policy", "unit", "a_min_m2", "b_max_m2", "separation_gap_m2", "proposable",
            "reason", "area_threshold_m2", "margin_a_m2", "margin_b_m2", "policy_basis",
            "policy_adjustment", "verify") if k in fit},
        "notes": {
            "classify_semantics": CLASSIFY_SEMANTICS_NOTE,
            "area_source": AREA_SOURCE_NOTE,
            "label_responsibility": LABEL_RESPONSIBILITY_NOTE,
            "c_samples": C_REJECTION_NOTE,
            "backfill": BACKFILL_NOTE,
        },
        "output": {
            "suggestion_path": str(output_path),
            "suggestion_written": output_written,
            "policy": "重叠/坏输入不写建议值文件；成功时报告与建议值文件同时写出。",
        },
        "overall_pass": bool(fit["proposable"]),
    }
    return report


def build_suggestion(input_sha: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    """建议 K 文件内容（供人工回填；本工具绝不写入 plans 代码）。"""
    return {
        "tool": TOOL_NAME,
        "spec": SPEC_REF,
        "unit": "m^2",
        "area_threshold_m2": fit["area_threshold_m2"],
        "policy": fit["policy"],
        "policy_basis": fit.get("policy_basis"),
        "a_min_m2": fit["a_min_m2"],
        "b_max_m2": fit["b_max_m2"],
        "margin_a_m2": fit["margin_a_m2"],
        "margin_b_m2": fit["margin_b_m2"],
        "input_sha256": input_sha,
        "boundary_semantics": "area == K → A（classify 等号归 A，P1 §3.5）",
        "backfill_target": BACKFILL_TARGET,
        "backfill_notice": BACKFILL_NOTE,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{TOOL_NAME}.py",
        description=(
            "K 面积分级阈值拟合（P4 §2.4/D05）。输入 CSV 列逐字固定为 "
            f"{','.join(CSV_COLUMNS)}；grade_label 仅 A/B（不成熟判 C 与面积无关，"
            "C 样本不得进入输入）；输出建议 K 文件 + JSON 报告，人工回填 "
            f"{BACKFILL_TARGET}，本工具不写 plans 代码。"
        ),
        epilog=(
            "退出码：0 给出建议；2 输入级错误；3 两组面积重叠（拒绝建议，报告含"
            "分位数/散布数据）。策略 --policy：midpoint（默认，两组之间取保守分界）"
            "/ a_min（K=A 组下界，边界样本恰好 area==K 判 A）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help=f"样本 CSV（列：{','.join(CSV_COLUMNS)}）")
    ap.add_argument("--output", required=True, type=Path,
                    help="建议 K 文件（JSON，供人工回填 AREA_THRESHOLD_M2）")
    ap.add_argument("--report", required=True, type=Path,
                    help="JSON 报告（输入 SHA256、样本数、a_min/b_max、建议 K、策略、余量）")
    ap.add_argument("--policy", choices=POLICIES, default=POLICY_MIDPOINT,
                    help="分界策略：midpoint=两组中点（默认，两侧余量均正）；a_min=取 A 组下界"
                         "（A 边界样本 area==K 判 A，等号归 A 合法）")
    return ap


def run(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output
    report_path: Path = args.report

    # --- 1. 输入级校验（任一失败 → 退出码 2，绝不写文件）---
    try:
        if output_path.resolve() == input_path.resolve():
            raise InputError("--output 不得等于 --input（禁止改写输入样本 CSV）")
        samples = load_grade_samples(input_path)
        input_sha = sha256_file(input_path)
    except InputError as exc:
        print(f"[{TOOL_NAME}][输入错误] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    # --- 2. 拟合（纯计算；重叠 → 报告 + 退出码 3，不写建议值文件）---
    fit = fit_grade_threshold(samples, policy=args.policy)
    report = build_report(input_path, input_sha, samples, fit, output_path,
                          output_written=fit["proposable"])
    try:
        if fit["proposable"]:
            _write_json(output_path, build_suggestion(input_sha, fit))
        _write_json(report_path, report)
    except OSError as exc:
        print(f"[{TOOL_NAME}][写盘失败] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    if fit["proposable"]:
        print(
            f"[{TOOL_NAME}] 建议 K = {fit['area_threshold_m2']} m²"
            f"（policy={args.policy}，margin_a={fit['margin_a_m2']}，margin_b={fit['margin_b_m2']}）；"
            f"请人工回填 {BACKFILL_TARGET} 并以 classify 单测回归。建议文件 {output_path}；报告 {report_path}",
            file=sys.stderr, flush=True)
        return EXIT_OK
    print(
        f"[{TOOL_NAME}][拒绝建议] {fit['reason']} 报告（含分位数/散布数据）见 {report_path}",
        file=sys.stderr, flush=True)
    return EXIT_NO_PROPOSAL


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
