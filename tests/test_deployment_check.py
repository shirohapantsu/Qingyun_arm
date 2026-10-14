"""P5-02 离线测试：部署发布验收工具 `scripts/check_deployment.py`（P5 §5/§6）。

范围（任务书 P5-02 逐条）：

* 全通过：tmp_path 迷你部署树 + 合成 verified profile **真实通过生产 real loader**
  （``load_motion_params(mode="real")`` 的 verified+报告绑定判据可被合成满足——
  tests/test_motion_params.py 已有同构正向用例；本测试按其同一配方构造），证明
  判据 a/b/c/d 的正向分支被真实调用；manifest 以 ``provenance="simulated"`` 如实
  标注合成来源，工具只回显、不据此放宽或收紧任何判定。
* 负向：缺文件 / 哈希篡改 / 报告缺失 / 报告通过字段为假或缺失 / demo_acceptance
  未测 null（P4-T13 四键判据）/ real 重载失败（draft 状态、含 null 字段、指向
  mock 副本）/ 密钥路径与 SECRET_KEYS 键名被拒 / manifest 形态非法（参数化）。
* CLI：--help 退出 0；故意缺项树真实退出非零（3）；坏 manifest 退出 2；stdout 含
  机器可读 JSON verdict。
* 只读性：跑完检查后整棵树文件哈希不变（工具绝不写回被检部署树）。

禁止事项核对：本文件不向 tests/fixtures 落任何合成"部署证据"（全部 tmp_path）；
不放宽生产校验（负向用例断言的正是生产 loader/fit_grade_threshold 的原始拒绝）。
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.support_validation import complete_report_stages

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_deployment as cd  # noqa: E402
from configs.motion_params import parameter_sha256  # noqa: E402
from tests.support import SIM_PROFILE, raw_profile  # noqa: E402

# 报告字段缺失哨兵（区别于 None/False）。
OMIT = object()

GIT_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha(path: Path) -> str:
    return cd.sha256_file(path)


# ---------------------------------------------------------------------------
# 1. 迷你部署树构造（全部落在 tmp_path，合成、标注 simulated，绝不进 tests/fixtures）
# ---------------------------------------------------------------------------


def build_tree(tmp_path: Path, *,
               profile_status: str = "verified",
               null_field: str | None = None,
               report_overall_pass: Any = True,
               report_operator_reviewed: Any = True,
               acceptance: str = "complete") -> Path:
    """写一棵合成部署树并返回部署根 root。

    profile 采用 tests/test_motion_params.py::test_real模式能加载一份完整验收过的配置
    的同一配方（simulation profile 副本 → status/verification 两步写入 → 报告绑定
    parameter_sha256），因此正向树能被**生产 real loader 原样加载**，无需任何放宽。
    """
    root = tmp_path / "deploy"
    pdir = root / "profile"
    (pdir / "reports").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "calib").mkdir()
    (root / "configs").mkdir()
    (root / "reports").mkdir()
    (root / "samples").mkdir()

    src = SIM_PROFILE.parent
    data = raw_profile()
    urdf_src = (src / data["model"]["urdf_path"]).resolve()
    shutil.copy(urdf_src, pdir / "model.urdf")
    data["model"]["urdf_path"] = "model.urdf"
    data["model"]["urdf_sha256"] = _sha(pdir / "model.urdf")
    shutil.copy(src / data["model"]["motor_calibration_path"],
                pdir / "motor_calibration.json")
    data["model"]["motor_calibration_path"] = "motor_calibration.json"
    data["model"]["motor_calibration_sha256"] = _sha(pdir / "motor_calibration.json")

    # 第一步：verification 置 null，先定稿"被验收的参数"。
    data["status"] = profile_status
    data["verification"] = None
    if null_field is not None:
        grp, key = null_field.split(".")
        data[grp][key] = None
    # status 不进 parameter_sha256；null 字段进哈希（它本来就是要被 loader 拒绝的改动）。
    param_hash = parameter_sha256(data)

    report: dict[str, Any] = {
        "profile_id": data["profile_id"],
        "parameter_sha256": param_hash,
        "operator": "synthetic-tester",
        "stages": complete_report_stages(),
        "overall_pass": report_overall_pass if report_overall_pass is not OMIT else True,
        "operator_reviewed": report_operator_reviewed
        if report_operator_reviewed is not OMIT else True,
    }
    if report_overall_pass is OMIT:
        del report["overall_pass"]
    if report_operator_reviewed is OMIT:
        del report["operator_reviewed"]
    report_path = pdir / "reports" / "validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    data["verification"] = {
        "report_path": "reports/validation.json",
        "report_sha256": _sha(report_path),
    }
    (pdir / "profile.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # demo_acceptance 本地文件（P4 §3.1 四键；complete=合成名义值并如实标 simulated）
    acc = root / "configs" / "demo_acceptance.local.json"
    if acceptance == "complete":
        acc.write_text(json.dumps({
            "schema_version": 1,
            "near_round_min_ratio": 1.35,
            "max_ripe_error_rate": 0.05,
            "reference_position_uncertainty_m": 0.001,
            "reference_yaw_uncertainty_deg": 1.0,
            "source": "simulated",
        }), encoding="utf-8")
    elif acceptance == "nulls":
        # 逐字取仓库模板（全 null → 不完整、不可验收）。
        acc.write_text((ROOT / "configs" / "demo_acceptance.json").read_text(encoding="utf-8"),
                       encoding="utf-8")
    elif acceptance == "badschema":
        acc.write_text(json.dumps({
            "schema_version": 1, "near_round_min_ratio": 1.35, "max_ripe_error_rate": 0.05,
            "reference_position_uncertainty_m": 0.001, "reference_yaw_uncertainty_deg": 1.0,
            "unexpected_key": 1,
        }), encoding="utf-8")
    else:
        raise AssertionError(f"未知 acceptance 形态 {acceptance!r}")

    # 其余资源占位文件（内容仅为合成字节串；名称对应 P5 §5 固定清单角色）。
    (root / "requirements.txt").write_text("numpy==1.2.3  # synthetic\n", encoding="utf-8")
    (root / "prompt.md").write_text("# 合成 prompt（simulated）\n", encoding="utf-8")
    (root / "weights" / "strawberry.pt").write_bytes(b"synthetic-weights-v0\n")
    (root / "weights" / "classes.json").write_text('["strawberry"]\n', encoding="utf-8")
    (root / "calib" / "camera_intrinsics.json").write_text(
        json.dumps({"comment": "synthetic placeholder，真实标定包归 P4/队友", "measured": False}),
        encoding="utf-8")
    (root / "configs" / "app.local.json").write_text(
        json.dumps({"comment": "synthetic config template"}), encoding="utf-8")
    (root / "configs" / "vision.local.json").write_text(
        json.dumps({"comment": "synthetic vision config"}), encoding="utf-8")
    (root / "configs" / "app.mock.json").write_text(
        json.dumps({"comment": "synthetic mock config"}), encoding="utf-8")
    (root / "samples" / "manifest.csv").write_text("sample_id,grade_label\n", encoding="utf-8")
    for name in ("cal040_report.json", "confusion_report.json"):
        (root / "reports" / name).write_text(json.dumps({
            "profile_id": data["profile_id"], "overall_pass": True,
            "operator_reviewed": True, "stages": complete_report_stages(),
        }), encoding="utf-8")
    (root / "calib" / "T_base_camera.json").write_text('{"source":"synthetic"}')
    (root / "samples" / "grades.csv").write_text(
        "sample_id,grade_label,length_m,width_m\na,A,0.06,0.04\nb,B,0.04,0.02\n")
    (root / "reports" / "grade_threshold.json").write_text(json.dumps({
        "overall_pass": True, "fit": {"proposable": True, "area_threshold_m2": 0.0016},
        "inputs": {"input_sha256": _sha(root / "samples" / "grades.csv")}}))
    (root / "reports" / "footprint.json").write_text(json.dumps({
        "overall_pass": True, "operator_reviewed": True}))
    return root


def base_manifest(root: Path) -> dict[str, Any]:
    """按当前树内容计算全部 SHA256，生成合法 manifest dict（尚未落盘）。"""
    def res(rel: str, role: str, reports: list[str] | None = None) -> dict[str, Any]:
        return {"relative_path": rel, "sha256": _sha(root / rel),
                "role": role, "reports": reports or []}

    return {
        "schema_version": 1,
        "git_commit": GIT_COMMIT,
        "provenance": "simulated",
        "motion_profile": {
            "relative_path": "profile/profile.json",
            "sha256": _sha(root / "profile/profile.json"),
            "verification_report": "profile/reports/validation.json",
        },
        "resources": [
            res("profile/model.urdf", "urdf", ["profile/reports/validation.json"]),
            res("profile/motor_calibration.json", "motor_calibration", ["profile/reports/validation.json"]),
            res("requirements.txt", "requirements"),
            res("prompt.md", "prompt"),
            res("weights/strawberry.pt", "weights", ["reports/cal040_report.json", "reports/footprint.json", "reports/confusion_report.json"]),
            res("weights/classes.json", "class_map", ["reports/cal040_report.json", "reports/footprint.json", "reports/confusion_report.json"]),
            res("calib/camera_intrinsics.json", "camera_intrinsics", ["reports/cal040_report.json"]),
            res("calib/T_base_camera.json", "camera_extrinsic", ["reports/cal040_report.json"]),
            res("configs/app.local.json", "app_config"),
            res("configs/vision.local.json", "vision_config", ["reports/cal040_report.json", "reports/footprint.json", "reports/confusion_report.json"]),
            res("configs/app.mock.json", "mock_config"),
            res("samples/manifest.csv", "sample_manifest",
                ["reports/cal040_report.json", "reports/footprint.json", "reports/confusion_report.json"]),
            res("samples/grades.csv", "grade_samples", ["reports/grade_threshold.json"]),
        ],
        "reports": [
            {"relative_path": "profile/reports/validation.json",
             "sha256": _sha(root / "profile/reports/validation.json"),
             "kind": "profile_validation"},
            {"relative_path": "configs/demo_acceptance.local.json",
             "sha256": _sha(root / "configs/demo_acceptance.local.json"),
             "kind": "demo_acceptance"},
            {"relative_path": "reports/cal040_report.json",
             "sha256": _sha(root / "reports/cal040_report.json"),
             "kind": "vision_accuracy"},
            {"relative_path": "reports/confusion_report.json",
             "sha256": _sha(root / "reports/confusion_report.json"),
             "kind": "ripe_confusion"},
            {"relative_path": "reports/grade_threshold.json", "sha256": _sha(root / "reports/grade_threshold.json"), "kind": "grade_threshold"},
            {"relative_path": "reports/footprint.json", "sha256": _sha(root / "reports/footprint.json"), "kind": "footprint_validation"},
        ],
    }


def finalize(root: Path, manifest: dict[str, Any]) -> Path:
    mpath = root / "deployment_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return mpath


def run(root: Path) -> tuple[int, dict[str, Any]]:
    return cd.run_deployment_check(root / "deployment_manifest.json")


def failed_checks(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in verdict["checks"] if not c["ok"]]


def first_failed(verdict: dict[str, Any], check_id: str, needle: str = "") -> dict[str, Any]:
    for c in failed_checks(verdict):
        if c["check_id"] == check_id and needle in c["detail"]:
            return c
    raise AssertionError(
        f"未找到失败项 {check_id!r}（needle={needle!r}）；实际失败项：{failed_checks(verdict)}")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# 2. 正向：全通过（含判据 a/b/c/d 的正向分支被真实调用）
# ---------------------------------------------------------------------------


def test_全通过_合成verified树经生产real_loader真实加载(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == 0, verdict["checks"]
    assert verdict["verdict"] == "PASS"
    assert all(c["ok"] for c in verdict["checks"])
    # d 的正向分支确实调用了生产 real loader（返回体带 profile 身份）。
    reload_item = next(c for c in verdict["checks"]
                       if c["check_id"] == "motion_profile.real_reload")
    assert "status='verified'" in reload_item["detail"]
    # 工具只读：整棵树逐文件哈希不变。
    before = _tree_hashes(root)
    run(root)
    assert _tree_hashes(root) == before
    # provenance 只回显、不改变判定（verdict 仍 PASS 且携带标记）；不豁免条款在文案中。
    assert verdict["provenance"] == "simulated"
    assert verdict["git_commit"] == GIT_COMMIT
    assert "不豁免" in verdict["notice"]


# ---------------------------------------------------------------------------
# 3. 判据 a：缺文件 / 哈希篡改
# ---------------------------------------------------------------------------


def test_资源文件缺失_非零并逐项报因(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, base_manifest(root))
    (root / "weights" / "strawberry.pt").unlink()
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    assert verdict["verdict"] == "FAIL"
    item = first_failed(verdict, "resource.exists", "文件不存在")
    assert "weights/strawberry.pt" in item["target"]


def test_哈希篡改被发现(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, base_manifest(root))
    with (root / "weights" / "strawberry.pt").open("ab") as fh:
        fh.write(b"tampered\n")
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    item = first_failed(verdict, "resource.sha256", "实算=")
    assert "manifest=" in item["detail"] and item["target"].endswith("weights/strawberry.pt")
    # 其余资源不受影响（逐项独立）。
    assert not any(c["check_id"] == "resource.sha256" and not c["ok"]
                   and "prompt.md" in c["target"] for c in verdict["checks"])


# ---------------------------------------------------------------------------
# 4. 判据 b/c：报告缺失 / 未通过 / 字段缺失 / demo_acceptance 未测 null
# ---------------------------------------------------------------------------


def test_验收报告文件缺失_b与d同时报因(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, base_manifest(root))
    (root / "profile" / "reports" / "validation.json").unlink()
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "resource.exists", "validation.json")
    first_failed(verdict, "report.pass", "报告文件不存在")
    # real loader 因绑定报告缺失同样拒绝（两道闸独立生效）。
    first_failed(verdict, "motion_profile.real_reload", "validation.json")


def test_报告overall_pass为假_按实际schema字段判失败(tmp_path):
    root = build_tree(tmp_path, report_overall_pass=False)
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    item = first_failed(verdict, "report.pass", "overall_pass=False")
    assert item["target"].endswith("profile/reports/validation.json")
    # 生产 loader 的绑定检查也独立拒绝（工具不替代、只复核）。
    first_failed(verdict, "motion_profile.real_reload", "overall_pass")


def test_报告缺operator_reviewed字段不允许缺省通过(tmp_path):
    root = build_tree(tmp_path, report_operator_reviewed=OMIT)
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "report.pass", "operator_reviewed 缺失")


def test_demo_acceptance未测null_不可验收(tmp_path):
    root = build_tree(tmp_path, acceptance="nulls")
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    item = first_failed(verdict, "report.demo_acceptance", "未测字段仍为 null")
    for key in ("near_round_min_ratio", "max_ripe_error_rate",
                "reference_position_uncertainty_m", "reference_yaw_uncertainty_deg"):
        assert key in item["detail"]


def test_demo_acceptance坏schema被严格loader拒绝(tmp_path):
    root = build_tree(tmp_path, acceptance="badschema")
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "report.demo_acceptance", "schema 违例")


# ---------------------------------------------------------------------------
# 5. 判据 d：real 重载失败三形态（draft 状态 / 含 null 字段 / 指向 mock 副本）
# ---------------------------------------------------------------------------


def test_real重载失败_draft状态(tmp_path):
    root = build_tree(tmp_path, profile_status="draft")
    finalize(root, base_manifest(root))  # manifest 哈希与新内容一致 → 只有 d 失败
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    item = first_failed(verdict, "motion_profile.real_reload", "mode=real 要求 status=verified")
    assert "'draft'" in item["detail"]
    assert "不豁免" in item["detail"]  # 错误信息如实引用 P5 §6 不豁免条款
    # 其余强制项仍全绿 → 失败唯一归因于生产 loader。
    assert all(c["ok"] for c in verdict["checks"]
               if c["check_id"] != "motion_profile.real_reload")


def test_real重载失败_运行期字段含null(tmp_path):
    root = build_tree(tmp_path, null_field="grasp.approach_height_m")
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "motion_profile.real_reload", "仍为 null")


def test_real重载失败_指向mock副本(tmp_path):
    # 未改写的 simulation profile 副本（configs/app.mock.json 指向的同类形态）。
    root = build_tree(tmp_path, profile_status="simulation")
    finalize(root, base_manifest(root))
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "motion_profile.real_reload", "'simulation'")
    assert all(c["ok"] for c in verdict["checks"]
               if c["check_id"] != "motion_profile.real_reload")


# ---------------------------------------------------------------------------
# 6. 关联完整性：悬空引用 / verification_report 错配
# ---------------------------------------------------------------------------


def test_资源关联报告悬空引用失败(tmp_path):
    root = build_tree(tmp_path)
    m = base_manifest(root)
    m["resources"][-1]["reports"] = ["reports/ghost.json"]
    finalize(root, m)
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    item = first_failed(verdict, "report.reference")
    assert "ghost.json" in item["target"]


def test_verification_report未声明或kind错配失败(tmp_path):
    root = build_tree(tmp_path)
    m = base_manifest(root)
    m["motion_profile"]["verification_report"] = "reports/undeclared.json"
    finalize(root, m)
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_CHECK_FAILED
    first_failed(verdict, "report.reference", "悬空")

    root2 = build_tree(tmp_path / "second")
    m2 = base_manifest(root2)
    m2["motion_profile"]["verification_report"] = "reports/cal040_report.json"  # 存在但非 profile 实际绑定的报告
    # 该报告 kind 恰为 profile_validation，reference 判据本身无悬空；错配由
    # binding_consistency（声明对应性）与 d（loader 哈希/参数绑定）共同把关。
    finalize(root2, m2)
    exit_code2, verdict2 = run(root2)
    assert exit_code2 == cd.EXIT_CHECK_FAILED
    first_failed(verdict2, "report.binding_consistency", "不一致")

    root3 = build_tree(tmp_path / "third")
    m3 = base_manifest(root3)
    m3["motion_profile"]["verification_report"] = "configs/demo_acceptance.local.json"
    finalize(root3, m3)
    exit_code3, verdict3 = run(root3)
    assert exit_code3 == cd.EXIT_CHECK_FAILED
    first_failed(verdict3, "report.reference", "kind 必须是 profile_validation")


# ---------------------------------------------------------------------------
# 7. 判据 e：密钥不入哈希清单
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate, needle", [
    (lambda m: m["resources"].append({"relative_path": "configs/secrets.local.json",
                                      "sha256": "0" * 64, "role": "config", "reports": []}),
     "密钥不入哈希清单"),
    (lambda m: m["reports"].append({"relative_path": "configs/secrets.json",
                                    "sha256": "0" * 64, "kind": "profile_validation"}),
     "密钥不入哈希清单"),
    (lambda m: m["motion_profile"].update({"relative_path": "secrets.json"}),
     "密钥不入哈希清单"),
])
def test_密钥类路径被拒_退出2(tmp_path, mutate, needle):
    root = build_tree(tmp_path)
    m = base_manifest(root)
    mutate(m)
    finalize(root, m)
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_MANIFEST_REJECTED
    assert verdict["verdict"] == "MANIFEST_REJECTED"
    assert needle in verdict["checks"][0]["detail"]


def test_SECRET_KEYS键名出现在任意字符串即拒(tmp_path):
    root = build_tree(tmp_path)
    m = base_manifest(root)
    m["resources"][2]["role"] = "requirements_with_dashscope_api_key"
    finalize(root, m)
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_MANIFEST_REJECTED
    assert "dashscope_api_key" in verdict["checks"][0]["detail"]

    root2 = build_tree(tmp_path / "second")
    m2 = base_manifest(root2)
    m2["resources"][0]["relative_path"] = "configs/secrets/dump.json"  # secrets 目录段
    finalize(root2, m2)
    exit_code2, verdict2 = run(root2)
    assert exit_code2 == cd.EXIT_MANIFEST_REJECTED


def test_工具不读取secrets文件内容(tmp_path):
    """被拒发生在校验阶段：树里根本不需要存在任何 secrets 文件（拒绝先于接触内容）。"""
    root = build_tree(tmp_path)
    m = base_manifest(root)
    m["resources"].append({"relative_path": "configs/secrets.local.json",
                           "sha256": "0" * 64, "role": "config", "reports": []})
    finalize(root, m)
    assert not (root / "configs" / "secrets.local.json").exists()
    exit_code, _verdict = run(root)
    assert exit_code == cd.EXIT_MANIFEST_REJECTED


# ---------------------------------------------------------------------------
# 8. manifest 形态非法（严格 schema，逐项拒绝 → 退出 2）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate, needle", [
    (lambda m: m.update({"unknown_top": 1}), "未知键"),
    (lambda m: m.pop("git_commit"), "缺失键"),
    (lambda m: m.update({"schema_version": "1"}), "schema_version"),
    (lambda m: m.update({"schema_version": 2}), "schema_version"),
    (lambda m: m.update({"git_commit": GIT_COMMIT.upper()}), "git_commit"),
    (lambda m: m.update({"git_commit": "abc123"}), "git_commit"),
    (lambda m: m.update({"provenance": "magic"}), "provenance"),
    (lambda m: m["motion_profile"].update({"relative_path": "/etc/passwd"}), "绝对路径"),
    (lambda m: m["motion_profile"].update({"relative_path": "../../outside.json"}), "越根"),
    (lambda m: m["motion_profile"].update({"sha256": "0" * 63}), "64 位"),
    (lambda m: m["resources"][0].update({"relative_path": "a\\b.json"}), "反斜杠"),
    (lambda m: m["resources"][0].update({"role": "   "}), "非空字符串"),
    (lambda m: m["resources"].append(copy.deepcopy(m["resources"][1])), "重复声明"),
    (lambda m: m["reports"][0].update({"kind": "mystery"}), "kind 只能是"),
    (lambda m: m.update({"reports": "不是数组"}), "数组"),
    (lambda m: m.update({"resources": {"不是": "数组"}}), "数组"),
])
def test_manifest形态非法一律拒绝(tmp_path, mutate, needle):
    root = build_tree(tmp_path)
    m = base_manifest(root)
    try:
        mutate(m)
    except (KeyError, IndexError, TypeError):
        pytest.skip("该形态对本参数化树不适用")
    finalize(root, m)
    exit_code, verdict = run(root)
    assert exit_code == cd.EXIT_MANIFEST_REJECTED, verdict["checks"]
    assert needle in verdict["checks"][0]["detail"], verdict["checks"][0]["detail"]


def test_manifest文件缺失或坏JSON退出2(tmp_path):
    root = build_tree(tmp_path)
    exit_code, verdict = run(root)  # 还没有 deployment_manifest.json
    assert exit_code == cd.EXIT_MANIFEST_REJECTED
    assert "不存在" in verdict["checks"][0]["detail"]

    finalize(root, base_manifest(root))
    (root / "deployment_manifest.json").write_text("{broken", encoding="utf-8")
    exit_code2, verdict2 = run(root)
    assert exit_code2 == cd.EXIT_MANIFEST_REJECTED
    assert "不是合法 JSON" in verdict2["checks"][0]["detail"]


# ---------------------------------------------------------------------------
# 9. CLI 冒烟（子进程真实退出码 + stdout 机器可读 JSON）
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> "subprocess.CompletedProcess[str]":  # noqa: F821
    import subprocess
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_deployment.py"), *args],
        capture_output=True, text=True, cwd=ROOT, timeout=180)


def test_CLI_help退出0():
    proc = _run_cli("--help")
    assert proc.returncode == 0
    assert "--manifest" in proc.stdout


def test_CLI_故意缺项树真实退出非零(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, base_manifest(root))
    (root / "prompt.md").unlink()
    proc = _run_cli("--manifest", str(root / "deployment_manifest.json"),
                    "--report", str(tmp_path / "verdict.json"))
    assert proc.returncode == cd.EXIT_CHECK_FAILED
    assert "VERDICT=FAIL" in proc.stdout
    assert '"verdict": "FAIL"' in proc.stdout  # 机器可读 JSON 已在 stdout
    saved = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL" and saved["exit_code"] == cd.EXIT_CHECK_FAILED
    assert any(not c["ok"] for c in saved["checks"])


def test_CLI_坏manifest退出2与通过树退出0(tmp_path):
    root = build_tree(tmp_path)
    finalize(root, {"schema_version": 99})
    proc = _run_cli("--manifest", str(root / "deployment_manifest.json"))
    assert proc.returncode == cd.EXIT_MANIFEST_REJECTED
    assert "MANIFEST_REJECTED" in proc.stdout

    root2 = build_tree(tmp_path / "ok")
    finalize(root2, base_manifest(root2))
    proc2 = _run_cli("--manifest", str(root2 / "deployment_manifest.json"))
    assert proc2.returncode == 0, proc2.stdout[-2000:]
    assert "VERDICT=PASS" in proc2.stdout
