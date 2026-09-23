"""P4 §2.4/§3.1 定向测试（P4-T02 / P4-T13）：`scripts/fit_grade_threshold.py` 与
`configs/demo_acceptance.json` 模板 + 共用 loader。

**测试范围（用户 2026-09-23 指示）：P4 阶段只做定向测试，不跑全量。**

覆盖（对齐 §4 表 P4-T02 / P4-T13 与任务书逐条）：

* P4-T02（K 拟合）：合成 A/B 分离样本 → 建议 K 落在 (b_max, a_min)（midpoint）或恰为
  a_min（a_min 策略）；`area == K` 用 **qingyun.plans.strawberry 的真实 classify** 实测
  判 "A"（等号归 A，P1 §3.5）；不成熟 ripe=False 大样本仍判 C（K 与 C 无关）；两组边界
  恰好重叠（a_min == b_max）拒绝建议（非零、不写建议文件、报告含分位数/散布）；C 标签
  混入拒绝；空组/坏数值/重复 sample_id/坏表头拒绝；输入 SHA256 进报告；**工具不写
  plans**（strawberry.py 哈希不变、类属性仍 None）；CLI 缺参/未知参非零。
* P4-T13（demo_acceptance 侧）：仓库模板可加载（全 null → incomplete、不可验收、不得
  结论通过）；完备合法值 → 可验收；ratio ≤ 1 / rate 越界 / 不确定度负 / bool / 字符串 /
  NaN / 未知键 / 缺键 / schema_version 非 1 全部拒绝；simulated 名义值样例可解析且如
  实标注 source=simulated；.gitignore 已含本地实例行。

全部离线：只在 tmp_path 写合成 CSV/JSON；classify 互证用 tests/support 的
``make_target``（其数值仅让测试对象字段完整，绝非真实标定）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import fit_grade_threshold as fgt  # noqa: E402
from qingyun.plans.strawberry import StrawberryPlan  # noqa: E402
from tests.support import make_target  # noqa: E402
from tests.support_upper import FakeArm  # noqa: E402

TEMPLATE_PATH = ROOT / "configs" / "demo_acceptance.json"

# 分离良好的合成样本（A 大 B 小）；面积为 float 乘积，与 classify 完全同式。
# A: 0.050*0.040=0.002（下界 a_min）、0.055*0.038=0.00209
# B: 0.045*0.028=0.00126（上界 b_max）、0.035*0.030=0.00105
GOOD_ROWS = [
    ["a1", "A", "0.050", "0.040"],
    ["a2", "A", "0.055", "0.038"],
    ["b1", "B", "0.045", "0.028"],
    ["b2", "B", "0.035", "0.030"],
]
A_MIN = 0.050 * 0.040
B_MAX = 0.045 * 0.028


def _write_csv(path: Path, rows: list[list[str]], header: list[str] | None = None) -> Path:
    head = header if header is not None else ["sample_id", "grade_label", "length_m", "width_m"]
    path.write_text("\n".join([",".join(head)] + [",".join(r) for r in rows]) + "\n",
                    encoding="utf-8")
    return path


def _run(tmp: Path, csv_path: Path, *, policy: str | None = None,
         out_name: str = "k.json", report_name: str = "report.json") -> tuple[int, Path, Path]:
    out = tmp / out_name
    rep = tmp / report_name
    argv = ["--input", str(csv_path), "--output", str(out), "--report", str(rep)]
    if policy is not None:
        argv += ["--policy", policy]
    return fgt.run(argv), out, rep


def _plan_with_k(k: float) -> StrawberryPlan:
    """以真实 StrawberryPlan 子类覆盖 K（等号归 A 的语义必须走 classify 本体）。"""
    plan_cls = type("PlanWithK", (StrawberryPlan,), {"AREA_THRESHOLD_M2": k})
    return plan_cls(FakeArm())


# ===========================================================================
# P4-T02：K 拟合主路径 + classify 语义互证
# ===========================================================================


def test_T02_分离样本_midpoint建议K严格落在两组之间(tmp_path):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path)
    assert code == 0
    assert out.is_file() and rep.is_file()
    sugg = json.loads(out.read_text(encoding="utf-8"))
    k = sugg["area_threshold_m2"]
    assert sugg["a_min_m2"] == A_MIN and sugg["b_max_m2"] == B_MAX
    assert B_MAX < k < A_MIN, "midpoint 必须严格在 (b_max, a_min) 内"
    assert k > 0.0 and k == k and k != float("inf"), "K 单位 m² 断言正有限"
    assert sugg["unit"] == "m^2"
    assert sugg["margin_a_m2"] > 0.0 and sugg["margin_b_m2"] > 0.0


def test_T02_area等于K用真实classify判A(tmp_path):
    """边界样本归 A 验证：area == K → "A"（P1 §3.5 等号归 A），走 StrawberryPlan.classify 本体。"""
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == 0
    k = json.loads(out.read_text(encoding="utf-8"))["area_threshold_m2"]
    plan = _plan_with_k(k)
    # length_m=K, width_m=1.0 → area 恰等于 float(K)*1.0 == K（同一乘式，无容差）。
    assert plan.classify(make_target(length_m=k, width_m=1.0, ripe=True, valid_count=1)) == "A"


def test_T02_a_min策略_真实A边界样本恰好area等K判A(tmp_path):
    """policy=a_min：K=A 组下界；CSV 里真实 A 边界样本 (0.050,0.040) 经 classify 判 A。"""
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path, policy="a_min")
    assert code == 0
    sugg = json.loads(out.read_text(encoding="utf-8"))
    k = sugg["area_threshold_m2"]
    assert k == A_MIN and k > B_MAX
    assert sugg["margin_a_m2"] == 0.0 and sugg["margin_b_m2"] == A_MIN - B_MAX
    plan = _plan_with_k(k)
    # 边界样本回归 classify 本体：面积乘式与工具完全一致 → area == K → "A"。
    assert plan.classify(make_target(length_m=0.050, width_m=0.040, ripe=True, valid_count=1)) == "A"
    assert plan.classify(make_target(length_m=0.045, width_m=0.028, ripe=True, valid_count=1)) == "B"


def test_T02_全部样本按建议K逐条回代与人工标签一致(tmp_path):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path)
    k = json.loads(out.read_text(encoding="utf-8"))["area_threshold_m2"]
    plan = _plan_with_k(k)
    for sid, grade, length, width in GOOD_ROWS:
        got = plan.classify(make_target(length_m=float(length), width_m=float(width),
                                        ripe=True, valid_count=1))
        assert got == grade, f"{sid} 人工判 {grade}、classify 判 {got}"
    assert json.loads(rep.read_text(encoding="utf-8"))["fit"]["verify"]["all_consistent"] is True


def test_T02_不成熟判C不受建议K影响(tmp_path):
    """ripe=False 优先 C（P1 §3.5）：大/小面积的未成熟样本在建议 K 下仍判 C。"""
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, _ = _run(tmp_path, csv_path)
    k = json.loads(out.read_text(encoding="utf-8"))["area_threshold_m2"]
    plan = _plan_with_k(k)
    assert plan.classify(make_target(length_m=0.9, width_m=0.8, ripe=False, valid_count=1)) == "C"
    assert plan.classify(make_target(length_m=0.001, width_m=0.001, ripe=False, valid_count=1)) == "C"


def test_T02_边界恰好重叠_a_min等于b_max_拒绝建议(tmp_path):
    """A 组下界与 B 组上界同为 0.05*0.04：a_min == b_max → 拒绝（非零、不写建议文件）。"""
    rows = [
        ["a1", "A", "0.050", "0.040"],
        ["b1", "B", "0.050", "0.040"],   # 与 a1 面积完全相同
        ["b2", "B", "0.035", "0.030"],
    ]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, rep = _run(tmp_path, csv_path)
    assert code != 0
    assert not out.exists(), "拒绝建议时不得产出建议值文件"
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["fit"]["proposable"] is False
    assert "重叠" in report["fit"]["reason"]


def test_T02_区间重叠拒绝时报告给分位数散布供人工复核(tmp_path):
    rows = [
        ["a1", "A", "0.050", "0.040"],
        ["a2", "A", "0.060", "0.050"],
        ["b1", "B", "0.060", "0.050"],   # B 顶进 A 组：b_max == 0.003 > a_min 0.002
        ["b2", "B", "0.035", "0.030"],
    ]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, rep = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_NO_PROPOSAL and not out.exists()
    report = json.loads(rep.read_text(encoding="utf-8"))
    for grp in ("a_group", "b_group"):
        stats = report["area_stats"][grp]
        for key in ("min_m2", "max_m2", "p05_m2", "p25_m2", "p50_m2", "p75_m2", "p95_m2"):
            assert key in stats and stats[key] is not None
    assert report["inputs"]["input_schema"] == "sample_id,grade_label,length_m,width_m"


def test_T02_C标签混入拒绝并注明不成熟与面积无关(tmp_path, capsys):
    rows = GOOD_ROWS + [["x1", "C", "0.020", "0.020"]]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, rep = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists() and not rep.exists()
    err = capsys.readouterr().err
    assert "grade_label" in err and "不成熟" in err and "classify" in err


def test_T02_其他非法标签拒绝(tmp_path):
    rows = GOOD_ROWS + [["x1", "ripe", "0.020", "0.020"]]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists()


@pytest.mark.parametrize("rows,expect_in_msg", [
    pytest.param([["a1", "A", "0.05", "0.04"]], "B 组为空", id="空B组"),
    pytest.param([["b1", "B", "0.05", "0.04"]], "A 组为空", id="空A组"),
    pytest.param([], "无任何数据行", id="仅表头"),
])
def test_T02_空组拒绝(tmp_path, rows, expect_in_msg, capsys):
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists()
    assert expect_in_msg in capsys.readouterr().err


@pytest.mark.parametrize("row", [
    pytest.param(["a1", "A", "nan", "0.04"], id="nan长轴"),
    pytest.param(["a1", "A", "inf", "0.04"], id="inf长轴"),
    pytest.param(["a1", "A", "0.05", "-0.04"], id="负短轴"),
    pytest.param(["a1", "A", "0.05", "0"], id="零短轴"),
    pytest.param(["a1", "A", "abc", "0.04"], id="非数值"),
    pytest.param(["a1", "A", "", "0.04"], id="缺值"),
    pytest.param(["a1", "A", "0.03", "0.04"], id="长小于宽违反P1契约"),
])
def test_T02_坏数值拒绝(tmp_path, row):
    rows = [r for r in GOOD_ROWS if r[0] != "a1"] + [row]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists()


def test_T02_重复sample_id拒绝(tmp_path, capsys):
    rows = GOOD_ROWS + [["a1", "B", "0.040", "0.030"]]
    csv_path = _write_csv(tmp_path / "s.csv", rows)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists()
    assert "重复" in capsys.readouterr().err


@pytest.mark.parametrize("header", [
    pytest.param(["sample_id", "grade", "length_m", "width_m"], id="列改名"),
    pytest.param(["sample_id", "grade_label", "length_m"], id="缺列"),
    pytest.param(["sample_id", "grade_label", "length_m", "width_m", "note"], id="多列"),
])
def test_T02_表头schema不符拒绝(tmp_path, header):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS, header=header)
    code, out, _ = _run(tmp_path, csv_path)
    assert code == fgt.EXIT_INPUT and not out.exists()


def test_T02_输入SHA256进报告且实算一致(tmp_path):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path)
    assert code == 0
    expected = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["inputs"]["input_sha256"] == expected
    assert json.loads(out.read_text(encoding="utf-8"))["input_sha256"] == expected


def test_T02_报告含职责来源与回归提示(tmp_path):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path)
    report = json.loads(rep.read_text(encoding="utf-8"))
    notes = report["notes"]
    assert "validate_vision_accuracy" in notes["area_source"], "面积来源=validate 工具或专用采集模式（队友）"
    assert "人工定级" in notes["label_responsibility"] and "不做自动标签" in notes["label_responsibility"]
    assert "不成熟" in notes["c_samples"]
    assert "classify 单测回归" in notes["backfill"] and "绝不写入 plans" in notes["backfill"]
    assert "等号归 A" in notes["classify_semantics"]
    assert report["samples"]["total_rows"] == 4 and report["samples"]["a_count"] == 2
    sugg = json.loads(out.read_text(encoding="utf-8"))
    assert sugg["backfill_target"] == "qingyun/plans/strawberry.py::AREA_THRESHOLD_M2"
    assert "area == K → A" in sugg["boundary_semantics"]


def test_T02_工具绝不写plans代码(tmp_path):
    """成功跑一遍：qingyun/plans/ 全部文件哈希不变，生产类属性仍 None。"""
    plans_dir = ROOT / "qingyun" / "plans"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(plans_dir.glob("*.py"))}
    assert StrawberryPlan.AREA_THRESHOLD_M2 is None, "P4-05 不得回填生产 K（D05：人工回填）"
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    code, out, rep = _run(tmp_path, csv_path)
    assert code == 0
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(plans_dir.glob("*.py"))}
    assert after == before, "fit_grade_threshold 绝不写入 plans 代码"
    assert StrawberryPlan.AREA_THRESHOLD_M2 is None
    assert {p.name for p in tmp_path.iterdir()} == {"s.csv", "k.json", "report.json"}, \
        "工具只写 --output 与 --report 两个文件"


def test_T02_CLI缺参或非零退出():
    with pytest.raises(SystemExit) as exc:
        fgt.run([])
    assert exc.value.code != 0
    with pytest.raises(SystemExit) as exc:
        fgt.run(["--input", "x.csv"])       # 缺 --output/--report
    assert exc.value.code != 0


def test_T02_CLI未知策略与未知参数非零(tmp_path):
    with pytest.raises(SystemExit) as exc:
        fgt.run(["--input", "x.csv", "--output", "o.json", "--report", "r.json",
                 "--policy", "aggressive"])
    assert exc.value.code != 0
    with pytest.raises(SystemExit) as exc:
        fgt.run(["--input", "x.csv", "--output", "o.json", "--report", "r.json",
                 "--auto-label", "yes"])
    assert exc.value.code != 0


def test_T02_缺输入文件退出码2且不写文件(tmp_path):
    code, out, rep = _run(tmp_path, tmp_path / "nope.csv")
    assert code == fgt.EXIT_INPUT and not out.exists() and not rep.exists()


def test_T02_output不得覆盖输入CSV(tmp_path):
    csv_path = _write_csv(tmp_path / "s.csv", GOOD_ROWS)
    before = csv_path.read_bytes()
    code = fgt.run(["--input", str(csv_path), "--output", str(csv_path),
                    "--report", str(tmp_path / "r.json")])
    assert code == fgt.EXIT_INPUT
    assert csv_path.read_bytes() == before


# ===========================================================================
# P4-T13（demo_acceptance 侧）：严格 schema + 缺值语义
# ===========================================================================

_COMPLETE = {
    "schema_version": 1,
    "near_round_min_ratio": 1.35,
    "max_ripe_error_rate": 0.05,
    "reference_position_uncertainty_m": 0.003,
    "reference_yaw_uncertainty_deg": 2.0,
}


def _complete(**overrides) -> dict:
    data = dict(_COMPLETE)
    data.update(overrides)
    return data


def test_T13_模板五键逐字且全null(tmp_path):
    data = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert list(data) == list(fgt.DEMO_ACCEPTANCE_KEYS), "模板键集合与顺序按 §3.1 逐字"
    assert data["schema_version"] == 1
    for key in fgt.DEMO_ACCEPTANCE_VALUE_KEYS:
        assert data[key] is None, "模板（提交版）一律 null，实测值走 gitignore 的 local 文件"


def test_T13_模板加载为不完整不可验收():
    result = fgt.evaluate_demo_acceptance(TEMPLATE_PATH)
    assert result["valid"] is True
    assert result["complete"] is False and result["acceptance_allowed"] is False
    assert result["verdict"] == "incomplete_not_acceptable"
    assert "不完整、不可验收" in result["verdict_text"]
    assert result["exit_code"] == fgt.EXIT_NOT_ACCEPTABLE != 0, "null 模板绝不能给出通过结论"
    assert sorted(result["missing_fields"]) == sorted(fgt.DEMO_ACCEPTANCE_VALUE_KEYS)


def test_T13_完备合法值可验收(tmp_path):
    path = tmp_path / "demo_acceptance.local.json"
    path.write_text(json.dumps(_COMPLETE), encoding="utf-8")
    result = fgt.evaluate_demo_acceptance(path)
    assert result["complete"] is True and result["acceptance_allowed"] is True
    assert result["verdict"] == "acceptable" and result["exit_code"] == 0
    assert result["source"] == "measured"


def test_T13_边界合法值_zero_rate与非零ratio大于1与零不确定度():
    data = _complete(near_round_min_ratio=1.0000001, max_ripe_error_rate=0.0,
                     reference_position_uncertainty_m=0.0, reference_yaw_uncertainty_deg=0.0)
    result = fgt.evaluate_demo_acceptance(data)
    assert result["acceptance_allowed"] is True


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda d: d.update(near_round_min_ratio=1.0), id="ratio等于1非法(须>1)"),
    pytest.param(lambda d: d.update(near_round_min_ratio=0.5), id="ratio小于1"),
    pytest.param(lambda d: d.update(near_round_min_ratio=True), id="ratio为bool"),
    pytest.param(lambda d: d.update(max_ripe_error_rate=-0.01), id="rate负"),
    pytest.param(lambda d: d.update(max_ripe_error_rate=1.01), id="rate大于1"),
    pytest.param(lambda d: d.update(max_ripe_error_rate=True), id="rate为bool"),
    pytest.param(lambda d: d.update(max_ripe_error_rate="0.05"), id="rate为字符串"),
    pytest.param(lambda d: d.update(reference_position_uncertainty_m=-0.001), id="位置不确定度负"),
    pytest.param(lambda d: d.update(reference_yaw_uncertainty_deg=-1e-9), id="yaw不确定度负"),
    pytest.param(lambda d: d.update(reference_yaw_uncertainty_deg=False), id="不确定度为bool"),
    pytest.param(lambda d: d.update(schema_version=2), id="schema_version为2"),
    pytest.param(lambda d: d.update(schema_version="1"), id="schema_version字符串"),
    pytest.param(lambda d: d.update(schema_version=True), id="schema_version为bool"),
    pytest.param(lambda d: d.pop("max_ripe_error_rate"), id="缺键"),
    pytest.param(lambda d: d.update(extra_field=1), id="未知键"),
    pytest.param(lambda d: d.update(soucrce="simulated"), id="source拼写错误按未知键"),
    pytest.param(lambda d: d.update(source="real"), id="source取值非法"),
    pytest.param(lambda d: d.update(near_round_min_ratio=float("inf")), id="ratio为inf"),
])
def test_T13_严格schema违例全部拒绝(mutate):
    data = _complete()
    mutate(data)
    with pytest.raises(fgt.AcceptanceConfigError):
        fgt.validate_demo_acceptance(data)


def test_T13_JSON的NaN字面量被拒绝(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_COMPLETE).replace("1.35", "NaN"), encoding="utf-8")
    with pytest.raises(fgt.AcceptanceConfigError):
        fgt.load_demo_acceptance(path)


def test_T13_simulated名义值可解析且如实标注(tmp_path):
    """§3.1：独立模拟配置可给名义值，但必须显式 source=simulated，不得混充实测。"""
    path = tmp_path / "sim.json"
    path.write_text(json.dumps(_complete(source="simulated")), encoding="utf-8")
    result = fgt.evaluate_demo_acceptance(path)
    assert result["valid"] is True and result["complete"] is True
    assert result["source"] == "simulated", "报告必须如实携带 simulated 标注"
    assert "simulated" in result["verdict_text"] and "不得冒充实测" in result["verdict_text"]


def test_T13_文件不存在与坏JSON拒绝(tmp_path):
    with pytest.raises(fgt.AcceptanceConfigError):
        fgt.load_demo_acceptance(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(fgt.AcceptanceConfigError):
        fgt.load_demo_acceptance(bad)


def test_T13_gitignore含本地实例行():
    lines = [ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "configs/demo_acceptance.local.json" in lines


def test_T13_报告输出闸_incomplete时永远给不出acceptable():
    """null → 工具可出「不完整、不可验收」报告，但不得结论通过（P4-T13）。"""
    data = _complete(max_ripe_error_rate=None, reference_yaw_uncertainty_deg=None)
    result = fgt.evaluate_demo_acceptance(data)
    assert result["complete"] is False and result["acceptance_allowed"] is False
    assert result["verdict"] != "acceptable"
    assert sorted(result["missing_fields"]) == ["max_ripe_error_rate",
                                                "reference_yaw_uncertainty_deg"]
