"""P4 §2.8/§2.8.3 抓放验收入口工具链的离线测试（P4-T08 + 补充）。

覆盖：
* 合成 hardware pick_place 样例 → validate 完整性关卡通过（P4-T08 正例）；
* 变异拒绝：缺观察字段 / 损伤时长不足 / 参数哈希不一致 / 重复 trial_id /
  began_descend=null（unknown 保留分母但不通过）/ source 冒充 hardware 失败 /
  simulated 不成为 physical 证据 / 实际成功缺位置或误差不可复算 / yaw P95 超限；
* 导出器结构性：metadata + sample 形状、trial_summary 不作分母；
* 工具 CLI：缺列/坏 CSV/未知 place 在连接驱动前失败（探针证明驱动与 vision 未被触碰）、
  FakeVision valid_count≠1 记失败、unknown 下降样本保留；
* 修订工作流：unknown 样本人工把 began_descend 由 null 定为 bool 后 validate 通过、
  仍 null 者保留分母但不通过。

全部离线：只在 tmp 目录写合成 JSONL/资源文件，绝不触碰硬件；工具流测试用模块级
工厂/属性注入替身（P4 §2.8：注入点仅测试用）。
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import calibrate  # noqa: E402
import record_pick_place_trials as rppt  # noqa: E402
from configs.common_interface import (  # noqa: E402
    GraspResult,
    GraspStatus,
    HoldingState,
    VisionInterface,
)
from configs.motion_params import parameter_sha256  # noqa: E402
from qingyun.grabbing.arm_control import MotionTraceEvent, TRACE_DESCENT_SUBMIT  # noqa: E402


# ---------------------------------------------------------------------------
# 合成样例基础设施
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_thresholds(monkeypatch):
    """工具装配会调用 thresholds_from_params(完整 MotionParams)；测试用 FakeParams 时
    只需装配流程可跑通，视觉阈值构造不是被测对象，故整体打桩为 None。"""
    monkeypatch.setattr(rppt, "thresholds_from_params", lambda params: None)

_ACCEPTANCE = {
    "min_grasp_trials": 1, "min_grasp_success_rate": 0.5, "max_drop_rate": 0.5,
    "max_damage_rate": 0.5, "max_position_error_m": 0.01, "max_yaw_error_deg": 4.0,
    "damage_observe_minutes": 30.0,
}


def _minimal_profile_dict() -> dict:
    return {
        "schema_version": 1, "profile_id": "T08", "robot_id": "TEST", "status": "draft",
        "model": {"urdf_path": "urdf.txt", "motor_calibration_path": "motor_calib.json"},
        "places": {"bin": {"position": [0.32, 0.05, 0.02], "yaw_deg": 0.0}},
        "acceptance": dict(_ACCEPTANCE),
    }


def _write_resources(tmp: Path) -> dict:
    (tmp / "urdf.txt").write_text("<dummy urdf for hashing>\n", encoding="utf-8")
    (tmp / "motor_calib.json").write_text('{"dummy": "calib"}\n', encoding="utf-8")
    return {
        "urdf": calibrate.sha256_file(tmp / "urdf.txt"),
        "motor_calibration": calibrate.sha256_file(tmp / "motor_calib.json"),
    }


def _success_sample(trial_id: str, *, param_sha: str, grade: str = "B",
                    place_id: str = "bin", observe: float = 30.0,
                    began_descend: bool | None = True,
                    evidence: str = "submitted",
                    yaw_err: float = 1.0) -> dict:
    tx, ty, tz = 0.35, 0.0, 0.02
    ax, ay, az = tx + 0.002, ty + 0.001, tz + 0.0005
    center = round(calibrate._place_center_error_m([tx, ty, tz], [ax, ay, az]), 9)
    yaw = round(calibrate._place_yaw_error_deg(0.0, yaw_err), 9)
    return {
        "record_type": "sample", "trial_id": trial_id,
        "profile_parameter_sha256": param_sha,
        "target_position_m": [tx, ty, tz], "target_yaw_deg": 0.0,
        "grade": grade, "place_id": place_id, "start_ns": 1, "end_ns": 2,
        "software_status": "SUCCESS", "software_stage": "DONE", "holding": "EMPTY",
        "recovery_required": False, "began_descend": began_descend,
        "descent_evidence": evidence, "actual_success": True, "actual_drop": False,
        "actual_place_position_m": [ax, ay, az], "actual_place_yaw_deg": yaw_err,
        "place_error_center_m": center, "place_error_yaw_deg": yaw,
        "feedback_span_s": None, "stop_latency_s": None, "damage_label": "none",
        "damage_observe_minutes": observe, "image_refs": [f"{trial_id}.jpg"],
        "trace_ref": f"{trial_id}.trace.json",
    }


def _meta(param_sha: str, resource: dict, *, operator: str = "op-1",
          source: str = "hardware") -> dict:
    return {
        "record_type": "metadata", "source": source, "operator": operator,
        "profile_path": "profile.json", "profile_parameter_sha256": param_sha,
        "resource_sha256": resource, "code_commit": "abc123",
    }


def _validate(tmp: Path, meta: dict, samples: list[dict]) -> list[dict]:
    """把 metadata + samples 组成 Record 序列，走真实 _stage_checks 的 pick_place 分支。"""
    raw = _minimal_profile_dict()
    records = [calibrate.Record("pick_place", "r1", "meta", 1, meta)]
    for i, p in enumerate(samples):
        records.append(calibrate.Record("pick_place", "r1", f"sid-{i}", 2 + i, p))
    calibrate.args_profile_dir["current"] = str(tmp)
    return calibrate._stage_checks("pick_place", raw, records, "insha",
                                   {"sid-%d" % i for i in range(len(samples))}, {})


def _failed(checks: list[dict]) -> list[str]:
    return [c["name"] for c in checks if not c["passed"]]


def _passed(checks: list[dict]) -> bool:
    return bool(checks) and all(c["passed"] for c in checks)


def _param_sha() -> str:
    return parameter_sha256(_minimal_profile_dict())


# ---------------------------------------------------------------------------
# P4-T08 正例：合成完整 hardware 样例 → validate 通过
# ---------------------------------------------------------------------------


def test_合成完整hardware样例通过(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    samples = [_success_sample(f"t{i}", param_sha=sha) for i in range(3)]
    checks = _validate(tmp_path, _meta(sha, resource), samples)
    assert _passed(checks), _failed(checks)
    # 硬件记录的关键检查以 physical 证据登记（含新增的 yaw P95）。
    assert any("yaw 误差" in c["name"] and c["evidence_type"] == "physical" for c in checks)
    assert any(c["name"].startswith("完整性：") and c["evidence_type"] == "physical"
               for c in checks)


# ---------------------------------------------------------------------------
# 变异拒绝
# ---------------------------------------------------------------------------


def test_缺观察字段被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    s = _success_sample("t0", param_sha=sha)
    del s["trace_ref"]
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("观察/证据字段齐备" in n for n in _failed(checks))


def test_损伤观察时长不足记pending不通过(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    s = _success_sample("t0", param_sha=sha, observe=5.0)   # < 30 门槛
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("损伤观察时长" in n for n in _failed(checks))


def test_损伤观察时长缺失记pending不通过(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    s = _success_sample("t0", param_sha=sha)
    s["damage_observe_minutes"] = None
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("损伤观察时长" in n for n in _failed(checks))


def test_参数哈希与候选不一致被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    bad = _success_sample("t0", param_sha="0" * 64)
    checks = _validate(tmp_path, _meta(sha, resource), [bad])
    assert not _passed(checks)
    assert any("参数哈希" in n for n in _failed(checks))


def test_资源哈希不匹配被拒(tmp_path):
    _write_resources(tmp_path)
    sha = _param_sha()
    fake_resource = {"urdf": "1" * 64, "motor_calibration": "2" * 64}
    checks = _validate(tmp_path, _meta(sha, fake_resource),
                       [_success_sample("t0", param_sha=sha)])
    assert not _passed(checks)
    assert any("resource_sha256" in n for n in _failed(checks))


def test_重复trial_id被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    checks = _validate(tmp_path, _meta(sha, resource),
                       [_success_sample("dup", param_sha=sha),
                        _success_sample("dup", param_sha=sha)])
    assert not _passed(checks)
    assert any("trial_id" in n for n in _failed(checks))


def test_空operator被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    checks = _validate(tmp_path, _meta(sha, resource, operator="  "),
                       [_success_sample("t0", param_sha=sha)])
    assert not _passed(checks)
    assert any("operator" in n for n in _failed(checks))


def test_began_descend_null保留分母但报告不通过(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    good = [_success_sample(f"t{i}", param_sha=sha) for i in range(2)]
    unknown = _success_sample("u0", param_sha=sha, began_descend=None,
                              evidence="unknown")
    checks = _validate(tmp_path, _meta(sha, resource), good + [unknown])
    assert not _passed(checks)
    assert any("began_descend" in n for n in _failed(checks))
    # 分母仍把 unknown 计入（不从分母删除）：独立实物试验数 measured == 3。
    trial_count = next(c for c in checks if c["name"] == "独立实物试验数")
    assert trial_count["measured"] == 3.0


def test_source非hardware时完整性关卡不施加但记mock(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    # 模拟源样本仍带 not-bound-here 参数哈希（模拟语义），只应走 mock 证据。
    s = _success_sample("t0", param_sha="not-bound-here")
    checks = _validate(tmp_path, _meta(sha, resource, source="simulated"), [s],
    )
    # simulated 不得成为 physical 证据。
    assert all(c["evidence_type"] != "physical" for c in checks), checks
    assert not any(c["name"].startswith("完整性：") for c in checks)


def test_实际成功缺实际位置被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    s = _success_sample("t0", param_sha=sha)
    s["actual_place_position_m"] = None
    s["place_error_center_m"] = None
    s["place_error_yaw_deg"] = None
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("可复算一致" in n for n in _failed(checks))


def test_误差不可复算被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    s = _success_sample("t0", param_sha=sha)
    s["place_error_center_m"] = 0.000001        # 与 target/actual 重算不符
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("可复算一致" in n for n in _failed(checks))


def test_yaw误差P95超限被拒(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    # yaw_err=6.0 落在 [-90,90)，重算 yaw 误差 6.0 > max_yaw_error_deg 4.0。
    s = _success_sample("t0", param_sha=sha, yaw_err=6.0)
    checks = _validate(tmp_path, _meta(sha, resource), [s])
    assert not _passed(checks)
    assert any("yaw 误差" in n for n in _failed(checks))


def test_trial_summary不作为分母(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    samples = [_success_sample(f"t{i}", param_sha=sha) for i in range(3)]
    summary = {"record_type": "trial_summary", "note": "手工汇总，不是样本"}
    raw = _minimal_profile_dict()
    records = [calibrate.Record("pick_place", "r1", "meta", 1, _meta(sha, resource))]
    for i, p in enumerate(samples):
        records.append(calibrate.Record("pick_place", "r1", f"sid-{i}", 2 + i, p))
    records.append(calibrate.Record("pick_place", "r1", "sum-1", 99, summary))
    calibrate.args_profile_dir["current"] = str(tmp_path)
    checks = calibrate._stage_checks("pick_place", raw, records, "x", set(), {})
    trial_count = next(c for c in checks if c["name"] == "独立实物试验数")
    assert trial_count["measured"] == 3.0        # summary 未计入分母
    assert _passed(checks), _failed(checks)


# ---------------------------------------------------------------------------
# 修订工作流（unknown → 人工定值 + validate 复算）
# ---------------------------------------------------------------------------


def test_修订工作流_unknown保留不通过定为bool后通过(tmp_path):
    resource = _write_resources(tmp_path)
    sha = _param_sha()
    good = [_success_sample(f"t{i}", param_sha=sha) for i in range(2)]
    unknown = _success_sample("u0", param_sha=sha, began_descend=None,
                              evidence="unknown")
    # 修订前：unknown 使报告不通过（且保留在分母）。
    assert not _passed(_validate(tmp_path, _meta(sha, resource), good + [unknown]))
    # 操作者结合现场/视频把该条 began_descend 定为 bool、descent_evidence 改为一致值。
    revised = copy.deepcopy(unknown)
    revised["began_descend"] = True
    revised["descent_evidence"] = "manual_observed"
    assert _passed(_validate(tmp_path, _meta(sha, resource), good + [revised]))


# ---------------------------------------------------------------------------
# 工具注入替身
# ---------------------------------------------------------------------------


class _Probe:
    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("连接驱动/视觉初始化被提前触发")


class FakeParams:
    def __init__(self, tmp: Path, places: dict):
        self._tmp = tmp
        self.places = places
        self.profile_dir = tmp

    def urdf_path_resolved(self) -> Path:
        return self._tmp / "urdf.txt"

    def motor_calibration_path_resolved(self) -> Path:
        return self._tmp / "motor_calib.json"


def _target(valid_count: int = 1, pos=(0.35, 0.0, 0.02), yaw=0.0) -> VisionInterface:
    return VisionInterface(position=[*pos], yaw_deg=yaw, length_m=0.05, width_m=0.04,
                           ripe=True, valid_count=valid_count)


class FakeVisionModule:
    def __init__(self, target=None, raise_init=False):
        self._target = target or _target()
        self.raise_init = raise_init
        self.init_called = False

    def configure(self, thresholds, config_path):
        return None

    def init(self):
        self.init_called = True
        if self.raise_init:
            from qingyun.grabbing.vision import VisionHardError
            raise VisionHardError("桩：视觉未实现（真实调用如实失败）")

    def get_target(self, ignore: int = 0):
        return self._target


class FakeArm:
    def __init__(self, trace_events, result):
        self._trace = list(trace_events)
        self._result = result
        self.grasp_calls: list[str] = []

    def grasp_and_place(self, target, place_id="default"):
        self.grasp_calls.append(place_id)
        return self._result

    def get_last_motion_trace(self):
        return tuple(self._trace)


def _result() -> GraspResult:
    return GraspResult(status=GraspStatus.SUCCESS, stage="DONE", reason="ok",
                       place_id="bin", holding=HoldingState.EMPTY,
                       recovery_required=False)


def _submitted_trace():
    return [MotionTraceEvent(0, 1, "DESCEND", TRACE_DESCENT_SUBMIT, "submitted")]


def _unknown_trace():
    return [MotionTraceEvent(0, 1, "DESCEND", TRACE_DESCENT_SUBMIT, "unknown")]


class PromptScript:
    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, text: str) -> str:
        self.prompts.append(text)
        if not self._answers:
            raise EOFError("脚本耗尽")
        return self._answers.pop(0)


def _write_profile(tmp: Path) -> Path:
    p = tmp / "profile.json"
    p.write_text(json.dumps(_minimal_profile_dict()), encoding="utf-8")
    return p


def _write_trials(tmp: Path, rows: list[dict]) -> Path:
    import csv
    cols = list(rppt.REQUIRED_COLUMNS)
    p = tmp / "trials.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def _base_row(trial_id: str, place_id: str = "bin") -> dict:
    return {"trial_id": trial_id, "sample_id": f"s-{trial_id}", "place_id": place_id,
            "grade": "B", "sample_truth": f"object-{trial_id}", "scene": "lab-table"}


# ---------------------------------------------------------------------------
# 工具 CLI：连接驱动前的失败（探针证明驱动/视觉未被触碰）
# ---------------------------------------------------------------------------


def _guard_against_connect(monkeypatch):
    motor_probe, arm_probe = _Probe(), _Probe()
    vision_probe = FakeVisionModule(raise_init=False)
    monkeypatch.setattr(rppt, "make_motor", motor_probe)
    monkeypatch.setattr(rppt, "make_arm", arm_probe)
    monkeypatch.setattr(rppt, "cold_start_recover", lambda arm: None)
    monkeypatch.setattr(rppt, "vision", vision_probe)
    return motor_probe, arm_probe, vision_probe


def test_缺CSV列在连接驱动前失败(tmp_path, monkeypatch):
    resource = _write_resources(tmp_path)  # noqa: F841 (让资源文件存在，供 sha256)
    profile = _write_profile(tmp_path)
    p = tmp_path / "trials.csv"
    p.write_text("trial_id,sample_id,place_id\nT1,S1,bin\n", encoding="utf-8")
    motor_probe, arm_probe, vision_probe = _guard_against_connect(monkeypatch)
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    code = rppt.run(["--profile", str(profile), "--trials", str(p),
                     "--output", str(tmp_path / "out.jsonl"), "--operator", "op",
                     "--vision-config", str(profile)])
    assert code == rppt.EXIT_INPUT
    assert not motor_probe.called and not arm_probe.called
    assert not vision_probe.init_called


def test_未知place在连接驱动前失败(tmp_path, monkeypatch):
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("T1", place_id="ghost_bin")])
    motor_probe, arm_probe, vision_probe = _guard_against_connect(monkeypatch)
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(tmp_path / "out.jsonl"), "--operator", "op",
                     "--vision-config", str(profile)])
    assert code == rppt.EXIT_INPUT
    assert not motor_probe.called and not vision_probe.init_called


def test_重复trial_id在连接驱动前失败(tmp_path, monkeypatch):
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("DUP"), _base_row("DUP")])
    # 复用 sample_id 也行，因为 trial_id 先判；这里保持 sample_id 唯一以隔离断言。
    trials.read_text()
    motor_probe, _arm, vision_probe = _guard_against_connect(monkeypatch)
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(tmp_path / "out.jsonl"), "--operator", "op",
                     "--vision-config", str(profile)])
    assert code == rppt.EXIT_INPUT
    assert not motor_probe.called and not vision_probe.init_called


def test_未测完整参数profile在连接驱动前失败(tmp_path, monkeypatch):
    # 不注入替身：用真实 load_calibration_motion_params 拒绝不完整 draft。
    _write_resources(tmp_path)
    incomplete = _minimal_profile_dict()
    incomplete["motor"] = None            # 缺运行期字段 → 真实 loader 拒绝
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(incomplete), encoding="utf-8")
    trials = _write_trials(tmp_path, [_base_row("T1")])
    motor_probe, _arm, vision_probe = _guard_against_connect(monkeypatch)
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(tmp_path / "out.jsonl"), "--operator", "op",
                     "--vision-config", str(profile)])
    assert code == rppt.EXIT_INPUT
    assert not motor_probe.called and not vision_probe.init_called


# ---------------------------------------------------------------------------
# 工具流：视觉桩真实失败 / 正常导出 / 多目标记失败 / unknown 保留
# ---------------------------------------------------------------------------


def test_vision桩真实调用如实失败退出码3(tmp_path, monkeypatch):
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("T1")])
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    monkeypatch.setattr(rppt, "make_motor", lambda params: object())
    monkeypatch.setattr(rppt, "make_arm", lambda motor, params: FakeArm([], _result()))
    monkeypatch.setattr(rppt, "cold_start_recover", lambda arm: None)
    monkeypatch.setattr(rppt, "vision", FakeVisionModule(raise_init=True))
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(tmp_path / "out.jsonl"), "--operator", "op",
                     "--vision-config", str(profile)])
    assert code == rppt.EXIT_RUNTIME


def _happy_run_one_trial(tmp_path, monkeypatch, *, trace=None, valid_count=1):
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("T1")])
    arm = FakeArm(trace or _submitted_trace(), _result())
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    monkeypatch.setattr(rppt, "make_motor", lambda params: object())
    monkeypatch.setattr(rppt, "make_arm", lambda motor, params: arm)
    monkeypatch.setattr(rppt, "cold_start_recover", lambda a: None)
    monkeypatch.setattr(rppt, "vision", FakeVisionModule(_target(valid_count=valid_count)))
    # 观察脚本：确认开始→成功→未掉落→x/y/z→yaw→损伤none→观察30→图像默认
    monkeypatch.setattr(rppt, "_prompt", PromptScript(
        ["yes", "y", "n", "0.352", "0.001", "0.0205", "1.0", "", "30", ""]))
    out = tmp_path / "out.jsonl"
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(out), "--operator", "op-1",
                     "--vision-config", str(profile)])
    return code, out, arm


def test_正常导出metadata与sample并走通validate(tmp_path, monkeypatch):
    code, out, arm = _happy_run_one_trial(tmp_path, monkeypatch)
    assert code == rppt.EXIT_OK
    assert arm.grasp_calls == ["bin"]
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    assert lines[0]["payload"]["record_type"] == "metadata"
    assert lines[0]["payload"]["source"] == "hardware"
    assert lines[0]["payload"]["operator"] == "op-1"
    assert lines[0]["payload"]["code_commit"]           # git 实际值（unknown 也算如实登记）
    assert isinstance(lines[0]["payload"]["resource_sha256"], dict)
    sample = lines[1]["payload"]
    assert sample["record_type"] == "sample"
    assert sample["began_descend"] is True and sample["descent_evidence"] == "submitted"
    assert sample["feedback_span_s"] is None and sample["stop_latency_s"] is None
    assert sample["profile_parameter_sha256"] == lines[0]["payload"]["profile_parameter_sha256"]
    # 导出的这一条本身能通过 validate 的 pick_place 完整性关卡。
    raw = _minimal_profile_dict()
    records = [calibrate.Record("pick_place", "r", l["sample_id"], 1, l["payload"])
               for l in lines]
    calibrate.args_profile_dir["current"] = str(tmp_path)
    checks = calibrate._stage_checks("pick_place", raw, records, "x", set(), {})
    assert _passed(checks), _failed(checks)


def test_多目标场景记失败且不执行抓取(tmp_path, monkeypatch):
    code, out, arm = _happy_run_one_trial(
        tmp_path, monkeypatch, valid_count=2)
    assert code == rppt.EXIT_OK
    assert arm.grasp_calls == []                     # valid_count!=1 → 不抓取
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    sample = lines[1]["payload"]
    assert sample["actual_success"] is False
    assert sample["software_status"] == "SKIPPED_MULTI_TARGET"
    assert sample["began_descend"] is False          # 未抓取→无下降事件


def test_unknown下降样本保留且began_descend为null(tmp_path, monkeypatch):
    # 观察脚本：unknown 时操作者如实记未成功（无落点），保留整条。
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("T1")])
    arm = FakeArm(_unknown_trace(), _result())
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    monkeypatch.setattr(rppt, "make_motor", lambda params: object())
    monkeypatch.setattr(rppt, "make_arm", lambda motor, params: arm)
    monkeypatch.setattr(rppt, "cold_start_recover", lambda a: None)
    monkeypatch.setattr(rppt, "vision", FakeVisionModule(_target(valid_count=1)))
    monkeypatch.setattr(rppt, "_prompt", PromptScript(
        ["yes", "n", "n", "", "30", ""]))
    out = tmp_path / "out.jsonl"
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(out), "--operator", "op", "--vision-config", str(profile)])
    assert code == rppt.EXIT_OK
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    sample = lines[1]["payload"]
    assert sample["began_descend"] is None
    assert sample["descent_evidence"] == "unknown"   # §2.8.2：unknown 保留待操作者补充


def test_操作者EOF干净退出且已记录样本保留(tmp_path, monkeypatch):
    _write_resources(tmp_path)
    profile = _write_profile(tmp_path)
    trials = _write_trials(tmp_path, [_base_row("T1"), _base_row("T2")])
    arm = FakeArm(_submitted_trace(), _result())
    monkeypatch.setattr(rppt, "load_calibration_motion_params",
                        lambda path: FakeParams(tmp_path, {"bin": None}))
    monkeypatch.setattr(rppt, "make_motor", lambda params: object())
    monkeypatch.setattr(rppt, "make_arm", lambda motor, params: arm)
    monkeypatch.setattr(rppt, "cold_start_recover", lambda a: None)
    monkeypatch.setattr(rppt, "vision", FakeVisionModule(_target()))
    # 第一次完整记录，第二次"开始确认"即 EOF → 干净退出。
    monkeypatch.setattr(rppt, "_prompt", PromptScript(
        ["yes", "y", "n", "0.352", "0.001", "0.0205", "1.0", "", "30", ""]))
    out = tmp_path / "out.jsonl"
    code = rppt.run(["--profile", str(profile), "--trials", str(trials),
                     "--output", str(out), "--operator", "op", "--vision-config", str(profile)])
    assert code == rppt.EXIT_OK
    lines = [json.loads(l) for l in out.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) == 2                            # metadata + 1 sample（第二次未记录）
