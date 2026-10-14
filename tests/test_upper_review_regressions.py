"""2026-10-14 upper 审查 R01–R11 回归：设备适配器和验收边界。"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from configs.common_interface import GraspResult, GraspStatus, HoldingState, PlacePose
from configs.motion_params import ParamsError, load_motion_params
from qingyun import asr, runlog
from tests import test_calibration_tools as ct
from tests import test_deployment_check as dt
from tests import test_motor_control as mt
from tests.support import load_sim


@pytest.fixture(autouse=True)
def log_lifecycle():
    runlog.reset_for_tests()
    yield
    runlog.reset_for_tests()


def test_R01_无积压时不阻塞读取():
    raw = SimpleNamespace(samplerate=16000, channels=1, read_available=0,
                          read=lambda n: pytest.fail("空缓冲不得阻塞读取"))
    asr._SoundDeviceInput(raw).drain()


def test_R01_只清空进入时的积压不追逐新数据():
    class Raw:
        samplerate, channels, read_available = 16000, 1, 600
        def __init__(self): self.reads = []
        def read(self, n):
            self.reads.append(n)
            self.read_available += 10000
            return b"\0" * (n * 2), False
    raw = Raw()
    asr._SoundDeviceInput(raw).drain()
    assert raw.reads == [512, 88]


@pytest.mark.parametrize("channels,samples", [(1, [0, 256, -1, -32768, 32767]),
                                            (2, [256, -256, 32767, -32768])])
def test_R02_按本机PCM16解码带符号样本及多声道(channels, samples):
    data = memoryview(struct.pack(f"={len(samples)}h", *samples))
    raw = SimpleNamespace(samplerate=16000, channels=channels, read=lambda n: (data, False))
    assert asr._SoundDeviceInput(raw).read() == (samples, channels)


def test_R02_缓冲溢出不能作为完整口令():
    raw = SimpleNamespace(samplerate=16000, channels=1, read=lambda n: (b"\0\0", True))
    with pytest.raises(asr.ASRHardError, match="溢出"):
        asr._SoundDeviceInput(raw).read()


def test_R03_VAD归一化且不改变WAV原始幅值(monkeypatch):
    seen = []
    class Probability:
        def view(self, *args): return self
        def __getitem__(self, index): return self
        def item(self): return 0.8
    def model(window, sr):
        assert sr == 16000
        seen.extend(window)
        return Probability()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        FloatTensor=lambda samples: samples, no_grad=contextlib.nullcontext))
    pcm = [-32768, 0, 32767]
    assert asr._SileroVad(model, None).probability(pcm) == 0.8
    assert seen == pytest.approx([-1.0, 0.0, 32767 / 32768])
    import io
    import wave
    with wave.open(io.BytesIO(asr._encode_wav_mono16(pcm))) as wav:
        assert struct.unpack("<3h", wav.readframes(3)) == tuple(pcm)


def test_R04_原生DashScope请求及多段消息响应():
    body = asr._build_request_payload("qwen3-asr-flash", "data:audio/wav;base64,AA==")
    assert body["input"]["messages"] == [{"role": "user", "content": [
        {"audio": "data:audio/wav;base64,AA=="}]}]
    response = {"output": {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": [{"text": "分拣"}, {"text": "草莓"}]}}]}}
    assert asr._extract_transcript_text(json.dumps(response).encode()) == "分拣草莓"
    response["output"]["choices"][0]["finish_reason"] = "length"
    with pytest.raises(asr._TranscribeFailure):
        asr._extract_transcript_text(json.dumps(response).encode())


def test_R05_按配置落点正确放置得到零误差():
    row = dict(trial_id="t", place_id="bin", grade="B", sample_truth="s", scene="one")
    place = PlacePose(np.array([0.32, 0.055, 0.024]), 30.0)
    result = GraspResult(GraspStatus.SUCCESS, "DONE", "ok", "bin", HoldingState.EMPTY, False)
    sample = ct.rppt._build_sample_payload(
        trial_row=row, target=SimpleNamespace(position=[0.35, 0.0, 0.024], yaw_deg=0.0),
        result=result, trace=(), params=SimpleNamespace(places={"bin": place}), param_sha="test",
        observations=dict(actual_success=True, actual_place_position_m=place.position.tolist(),
                          actual_place_yaw_deg=30.0))
    assert sample["target_position_m"] == [0.35, 0.0, 0.024]
    assert sample["planned_place_position_m"] == place.position.tolist()
    assert sample["place_error_center_m"] == 0.0
    assert sample["place_error_yaw_deg"] == 0.0


@pytest.mark.parametrize("role", ["weights", "requirements", "prompt", "camera_intrinsics",
                                  "camera_extrinsic", "app_config", "vision_config", "grade_samples"])
def test_R06_漏掉必需资源声明同样拒绝(tmp_path, role):
    root = dt.build_tree(tmp_path)
    manifest = dt.base_manifest(root)
    dt.finalize(root, manifest)
    assert dt.run(root)[0] == 0
    manifest["resources"] = [r for r in manifest["resources"] if r["role"] != role]
    dt.finalize(root, manifest)
    code, verdict = dt.run(root)
    assert code != 0
    assert any(c["check_id"] == "manifest.required_resource" and c["target"] == role
               and not c["ok"] for c in verdict["checks"])


@pytest.mark.parametrize("kind", ["demo_acceptance", "vision_accuracy", "footprint_validation",
                                  "ripe_confusion", "grade_threshold"])
def test_R06_漏掉必需报告拒绝(tmp_path, kind):
    root = dt.build_tree(tmp_path)
    manifest = dt.base_manifest(root)
    manifest["reports"] = [r for r in manifest["reports"] if r["kind"] != kind]
    dt.finalize(root, manifest)
    code, verdict = dt.run(root)
    assert code != 0
    assert any(c["check_id"] == "manifest.required_report" and c["target"] == kind
               and not c["ok"] for c in verdict["checks"])


def test_R07_采集入口在视觉初始化前建立真实日志会话(tmp_path, monkeypatch):
    class LoggingVision(ct.FakeVisionModule):
        def init(self):
            super().init()
            runlog.event("vision_init_probe")
        def get_target(self, ignore=0):
            runlog.event("vision_scan_probe")
            return super().get_target(ignore)
    monkeypatch.setattr(ct, "FakeVisionModule", LoggingVision)
    monkeypatch.setattr(ct.rppt, "thresholds_from_params", lambda params: None)
    code, _, _ = ct._happy_run_one_trial(tmp_path, monkeypatch)
    assert code == 0
    events = [json.loads(line) for line in runlog.current_log_path().read_text().splitlines()]
    assert events[0]["kind"] == "calibration_startup"
    scan = next(e for e in events if e["kind"] == "vision_scan_probe")
    assert scan["trial_id"] == "T1" and scan["scan_id"] == 1


def test_R06_旧通用角色不能代替必需声明_可选来源可省略(tmp_path):
    root = dt.build_tree(tmp_path)
    manifest = dt.base_manifest(root)
    del manifest["provenance"]
    dt.finalize(root, manifest)
    assert dt.run(root)[0] == 0
    next(r for r in manifest["resources"] if r["role"] == "urdf")["role"] = "model_resource"
    dt.finalize(root, manifest)
    assert dt.run(root)[0] != 0


def test_R06_K报告不得绑定另一批样本(tmp_path):
    root = dt.build_tree(tmp_path)
    manifest = dt.base_manifest(root)
    entry = next(r for r in manifest["reports"] if r["kind"] == "grade_threshold")
    path = root / entry["relative_path"]
    report = json.loads(path.read_text())
    report["inputs"]["input_sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    entry["sha256"] = dt._sha(path)
    dt.finalize(root, manifest)
    code, verdict = dt.run(root)
    assert code != 0
    assert any(c["check_id"] == "report.pass" and not c["ok"] and "K 建议" in c["detail"]
               for c in verdict["checks"])


@pytest.mark.parametrize("change", [
    {"actual_success": "false", "actual_place_position_m": None, "actual_place_yaw_deg": None},
    {"actual_drop": "false"}, {"recovery_required": "false"},
    {"actual_place_yaw_deg": float("nan"), "place_error_yaw_deg": 0.0},
    {"damage_observe_minutes": float("nan")}, {"damage_observe_minutes": float("inf")},
    {"place_error_center_m": float("nan")}, {"planned_place_position_m": [0.35, 0.0, 0.02]},
    {"damage_label": None},
])
def test_R08_坏观察数据不能通过hardware验收(tmp_path, change):
    resource = ct._write_resources(tmp_path)
    digest = ct._param_sha()
    sample = ct._success_sample("t", param_sha=digest)
    meta = ct._meta(digest, resource)
    assert all(c["passed"] for c in ct._validate(tmp_path, meta, [sample]))
    sample.update(change)
    assert not all(c["passed"] for c in ct._validate(tmp_path, meta, [sample]))


@pytest.mark.parametrize("defect", ["empty", "missing", "partial", "mock", "failed_stage",
                                    "failed_check", "string_flag", "empty_checks"])
def test_R09_real加载器独立拒绝不完整或未通过的报告(tmp_path, defect):
    root = dt.build_tree(tmp_path)
    profile_path = root / "profile/profile.json"
    assert load_motion_params(profile_path, mode="real").status == "verified"
    report_path = root / "profile/reports/validation.json"
    report = json.loads(report_path.read_text())
    if defect == "empty": report["stages"] = {}
    if defect == "missing": del report["stages"]
    if defect == "partial": del report["stages"]["pick_place"]
    if defect == "mock": report["stages"]["motor"]["checks"][0]["evidence_type"] = "mock"
    if defect == "failed_stage": report["stages"]["motor"]["passed"] = False
    if defect == "failed_check": report["stages"]["motor"]["checks"][0]["passed"] = False
    if defect == "string_flag": report["overall_pass"] = "false"
    if defect == "empty_checks": report["stages"]["motor"]["checks"] = []
    report_path.write_text(json.dumps(report))
    profile = json.loads(profile_path.read_text())
    profile["verification"]["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ParamsError):
        load_motion_params(profile_path, mode="real")


@pytest.mark.parametrize("alias", ["same_input", "same_output", "hardlink", "symlink"])
def test_R10_报告输出不可覆盖样本或建议文件(tmp_path, alias):
    from scripts import fit_grade_threshold as tool
    samples, output, report = tmp_path / "samples.csv", tmp_path / "k.json", tmp_path / "report.json"
    samples.write_text("sample_id,grade_label,length_m,width_m\na,A,0.06,0.04\nb,B,0.04,0.02\n")
    before = samples.read_bytes()
    if alias == "same_input": report = samples
    if alias == "same_output": report = output
    if alias == "hardlink": os.link(samples, report)
    if alias == "symlink": report.symlink_to(samples)
    assert tool.run(["--input", str(samples), "--output", str(output), "--report", str(report)]) != 0
    assert samples.read_bytes() == before and not output.exists()


def test_R10_抓放输出覆盖profile须在连接驱动前拒绝(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text("original input")
    monkeypatch.setattr(ct.rppt, "make_motor", lambda params: pytest.fail("不得连接驱动"))
    code = ct.rppt.run(["--profile", str(profile), "--trials", str(tmp_path / "trials.csv"),
                        "--output", str(profile), "--operator", "op", "--vision-config", "vision.json"])
    assert code == ct.rppt.EXIT_INPUT and profile.read_text() == "original input"


def test_R11_最终反馈跨总时限必须清理并拒绝READY(monkeypatch):
    params = load_sim()
    ctl, bus, _, clock = mt._startup_controller(params, mt.StartupBus(), initialize=False)
    original = ctl._protocol.bulk_read
    deadlines = []
    def delayed(ids, address, length, deadline):
        deadlines.append(deadline)
        if len(deadlines) == 2: clock.sleep(0.015)
        return original(ids, address, length, deadline)
    monkeypatch.setattr(ctl._protocol, "bulk_read", delayed)
    with pytest.raises(mt.MC.MotorStartupError):
        ctl.initialize(timeout_s=0.31)
    assert deadlines[-1] <= 0.31
    assert ctl.startup_report["result"] != "ready"
    assert ctl.startup_report["cleanup"]["attempted"] is True and bus.closed
