"""upper 独立代码审查：确认现有缺陷的离线复现，非回归通过证明。

所有写入只发生在 TemporaryDirectory；无真实串口、麦克风或云请求。
assert 验证的是审查时观察到的错误行为，修复后应改成正确行为的回归测试。
"""
from __future__ import annotations

import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from qingyun import asr, runlog
from configs.common_interface import GraspResult, GraspStatus, HoldingState
from configs.motion_params import load_motion_params, parameter_sha256
import record_pick_place_trials as recorder
import fit_grade_threshold as grade_tool
from tests import test_deployment_check as deployment_fixture
from tests import test_calibration_tools as calibration_fixture


def emit(finding, **observations):
    print(json.dumps({"finding": finding, **observations}, ensure_ascii=False))


def microphone_drain():
    source = '''
import time
from qingyun.asr import _SoundDeviceInput
class Raw:
    samplerate=16000
    channels=1
    read_available=0
    def read(self, n):
        time.sleep(0.01)
        return b"\\0"*(n*2), False
print("drain entered", flush=True)
_SoundDeviceInput(Raw()).drain()
print("drain returned", flush=True)
'''
    try:
        subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                       capture_output=True, timeout=0.6, check=True)
    except subprocess.TimeoutExpired as exc:
        assert b"drain entered" in exc.stdout
        assert b"drain returned" not in exc.stdout
        emit("R01", no_overflow=True, read_available=0, drain_returned=False)
    else:
        raise AssertionError("R01 no longer reproduced")


def pcm_decode():
    pcm = struct.pack("<3h", 0, 256, -1)
    raw = SimpleNamespace(samplerate=16000, channels=1,
                          read=lambda n: (pcm, False))
    samples, _ = asr._SoundDeviceInput(raw).read()
    assert samples != [0, 256, -1] and len(samples) == 6
    emit("R02", expected_pcm_samples=[0, 256, -1], actual=samples)


def vad_scale():
    seen = []
    class Probability:
        def view(self, *args): return self
        def __getitem__(self, index): return self
        def item(self): return 0.5
    fake_torch = SimpleNamespace(FloatTensor=lambda x: x,
                                 no_grad=contextlib.nullcontext)
    def model(window, sample_rate):
        seen.extend(window)
        return Probability()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        asr._SileroVad(model, None).probability([32767] * 512)
    assert max(seen) == 32767.0
    emit("R03", peak_received_by_model=max(seen), expected_peak=32767 / 32768)


def asr_protocol():
    payload = asr._build_request_payload("qwen3-asr-flash", "data:audio/wav;base64,AA==")
    entry = payload["input"]["messages"][0]["content"][0]
    assert "audio" not in entry and "input_audio" in entry
    body = json.dumps({"output": {"choices": [{"finish_reason": "stop",
                       "message": {"role": "assistant", "content": [
                           {"text": "分拣草莓"}]}}]}}).encode()
    try:
        asr._extract_transcript_text(body)
    except asr._TranscribeFailure as exc:
        emit("R04", request_entry_keys=list(entry), official_response_rejected=str(exc))
    else:
        raise AssertionError("R04 no longer reproduced")


def place_error_reference():
    pickup = [0.35, 0.0, 0.024]
    place = [0.32, 0.055, 0.024]
    target = SimpleNamespace(position=pickup, yaw_deg=0.0)
    params = SimpleNamespace(places={"bin": SimpleNamespace(position=place, yaw_deg=30.0)})
    row = dict(trial_id="t1", sample_id="s1", place_id="bin", grade="B",
               sample_truth="sample-1", scene="one target")
    result = GraspResult(GraspStatus.SUCCESS, "DONE", "ok", "bin", HoldingState.EMPTY, False)
    observations = dict(actual_success=True, actual_drop=False,
                        actual_place_position_m=place, actual_place_yaw_deg=30.0,
                        damage_label="none", damage_observe_minutes=30.0)
    sample = recorder._build_sample_payload(trial_row=row, target=target,
        result=result, trace=(), observations=observations, param_sha="demo")
    assert sample["place_error_center_m"] > 0.06 and sample["place_error_yaw_deg"] == 30.0
    emit("R05", actual_place_equals_configured_place=True,
         reported_center_error_m=sample["place_error_center_m"],
         reported_yaw_error_deg=sample["place_error_yaw_deg"], expected_errors=0)


def deployment_omissions(tmp):
    root = deployment_fixture.build_tree(tmp)
    params = load_motion_params(root / "profile/profile.json", mode="real")
    report_path = root / "profile/reports/validation.json"
    report = json.loads(report_path.read_text())
    assert report["stages"] == {} and params.status == "verified"
    emit("R09", inherited_loader_gap=True, stages=report["stages"], real_loader_accepted=True)
    # R06 单独使用声明了全部八阶段的合成报告，避免依赖 R09 的空 stages 缺口。
    # 这些 physical 字段仅是临时负向测试数据，不是实际物理证据。
    report["stages"] = {stage: {"passed": True, "checks": [
        {"passed": True, "evidence_type": "physical", "name": "synthetic probe"}]}
        for stage in calibration_fixture.calibrate.STAGES}
    report_path.write_text(json.dumps(report))
    profile_path = root / "profile/profile.json"
    profile = json.loads(profile_path.read_text())
    profile["verification"]["report_sha256"] = deployment_fixture._sha(report_path)
    profile_path.write_text(json.dumps(profile))
    manifest = deployment_fixture.base_manifest(root)
    manifest["resources"] = []
    manifest["reports"] = manifest["reports"][:1]
    for name in ["requirements.txt", "prompt.md", "weights/strawberry.pt",
                 "configs/demo_acceptance.local.json", "reports/cal040_report.json",
                 "reports/confusion_report.json"]:
        (root / name).unlink()
    manifest_path = deployment_fixture.finalize(root, manifest)
    code, verdict = deployment_fixture.cd.run_deployment_check(manifest_path)
    assert code == 0 and verdict["verdict"] == "PASS"
    emit("R06", verdict=verdict["verdict"], resources=0,
         demo_acceptance_and_weights_and_visual_reports_missing=True,
         complete_profile_stage_count=len(report["stages"]))


def recorder_logging():
    runlog.reset_for_tests()
    fake_vision = SimpleNamespace(configure=lambda *args: None,
                                init=lambda: runlog.event("vision_ready"))
    with patch.object(recorder, "make_motor", lambda params: object()), \
         patch.object(recorder, "make_arm", lambda motor, params: object()), \
         patch.object(recorder, "cold_start_recover", lambda arm: None), \
         patch.object(recorder, "thresholds_from_params", lambda params: None), \
         patch.object(recorder, "vision", fake_vision):
        try:
            recorder._assemble(None, Path("unused.json"))
        except RuntimeError as exc:
            assert "runlog 未初始化" in str(exc)
            emit("R07", conforming_vision_log_failed=str(exc), actual_motor_writes=0)
        else:
            raise AssertionError("R07 no longer reproduced")


def observation_validation(tmp):
    from tests.support import SIM_PROFILE, raw_profile
    raw = raw_profile()
    calibrate = calibration_fixture.calibrate
    resource = {"urdf": calibrate.sha256_file(SIM_PROFILE.parent / raw["model"]["urdf_path"]),
                "motor_calibration": calibrate.sha256_file(
                    SIM_PROFILE.parent / raw["model"]["motor_calibration_path"])}
    digest = parameter_sha256(raw)
    count = raw["acceptance"]["min_grasp_trials"]
    def validate_samples(samples):
        records = [calibrate.Record("pick_place", "r1", "meta", 1,
                   calibration_fixture._meta(digest, resource))]
        records += [calibrate.Record("pick_place", "r1", f"s{i}", i+2, p)
                    for i, p in enumerate(samples)]
        calibrate.args_profile_dir["current"] = str(SIM_PROFILE.parent)
        return calibrate._stage_checks("pick_place", raw, records, "insha", set(), {})
    samples = []
    for i in range(count):
        sample = calibration_fixture._success_sample(f"t{i}", param_sha=digest)
        sample.update(actual_success="false", actual_place_position_m=None,
                      actual_place_yaw_deg=None, place_error_center_m=None,
                      place_error_yaw_deg=None,
                      damage_observe_minutes=raw["acceptance"]["damage_observe_minutes"])
        samples.append(sample)
    checks = validate_samples(samples)
    assert all(c["passed"] for c in checks)
    emit("R08", actual_success="false", actual_pose_missing=True,
         all_pick_place_checks_passed=True, unchanged_acceptance_min_trials=count)
    samples = []
    for i in range(count):
        sample = calibration_fixture._success_sample(f"nan{i}", param_sha=digest)
        sample.update(actual_place_yaw_deg=float("nan"), place_error_yaw_deg=0.0,
                      damage_observe_minutes=float("nan"))
        samples.append(sample)
    checks = validate_samples(samples)
    assert all(c["passed"] for c in checks)
    emit("R08", actual_yaw="NaN", damage_observe_minutes="NaN",
         all_pick_place_checks_passed=True, unchanged_acceptance_min_trials=count)


def report_overwrites_input(tmp):
    sample_csv = tmp / "samples.csv"
    sample_csv.write_text("sample_id,grade_label,length_m,width_m\n"
                          "a,A,0.06,0.04\nb,B,0.04,0.02\n")
    with contextlib.redirect_stderr(io.StringIO()):
        code = grade_tool.run(["--input", str(sample_csv), "--output", str(tmp / "k.json"),
                               "--report", str(sample_csv)])
    assert code == 0 and json.loads(sample_csv.read_text())["tool"] == "fit_grade_threshold"
    emit("R10", exit_code=code, original_sample_csv_overwritten_by_report=True)


def startup_deadline():
    from tests import test_motor_control as motor_fixture
    from tests.support import load_sim
    params = load_sim()
    ctl, _, _, clock = motor_fixture._startup_controller(
        params, motor_fixture.StartupBus(), initialize=False)
    original = ctl._protocol.bulk_read
    reads = 0
    def late_feedback(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            clock.sleep(0.015)  # 小于 read_timeout=20 ms 和允许采样跨度=30 ms
        return original(*args, **kwargs)
    with patch.object(ctl._protocol, "bulk_read", late_feedback):
        ctl.initialize(timeout_s=0.31)
    assert ctl.startup_report["result"] == "ready" and clock.t > 0.31
    emit("R11", configured_startup_timeout_s=0.31, ready_at_s=round(clock.t, 4),
         result=ctl.startup_report["result"])


def main():
    microphone_drain()
    pcm_decode()
    vad_scale()
    asr_protocol()
    place_error_reference()
    with tempfile.TemporaryDirectory(prefix="upper-review-") as name:
        tmp = Path(name)
        deployment_omissions(tmp)
        calibration = tmp / "calibration"
        calibration.mkdir()
        observation_validation(calibration)
        report_overwrites_input(tmp)
    recorder_logging()
    startup_deadline()


if __name__ == "__main__":
    main()
