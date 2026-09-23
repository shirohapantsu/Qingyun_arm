"""P2 语音链路流程测试（覆盖 P2 §5 的 T01–T05/T04b、T11–T19 中 **ASR 侧**全部条目）。

    * T01–T05/T04b/T11–T15：录音状态机、采样链、WAV 编码、init 契约（P2-01 交付）。
    * T16–T19：DashScope 私有适配与**端到端 deadline**（P2-02 交付，见 §3.1/§3.2）。

驱动方式（P2 §5 末段"替身"）：
    * **设备**用假流注入（``_open_input_stream``）：按块返回 (frames, channels)，块数
      即"假时钟"——状态机的一切时长（1000ms/200ms/30s）都换算成 16k 音频样本计数推进，
      测试不真实等待、不触任何麦克风。
    * **VAD**用固定概率序列替身（脚本 VAD）或按内容判峰替身（content VAD），
      接口 16k/512 帧、输出语音概率。
    * **云转写**在录音管线用例里由假钩子（``_transcribe`` 模块属性）接管；适配层自身
      的用例（T16/T18/T19）把 **HTTP 层**换成注入的本地 stub transport
      （``httpx.MockTransport``，P2 §5"不 monkeypatch socket"），T17 则反过来——用
      **真实 transport + 127.0.0.1 本地 HTTP 服务子进程**做墙钟证明（既不求云也不消耗
      账号），并显式不使用任何 stub。
    * 16k/单声道走 _AudioConverter 的恒等快速路径（1 块 = 1 帧），故精确状态机用例
      可按帧锁定；44.1k/立体声走带状态线性重采样，另用重采样正确性用例证明与整段
      一次性参考实现逐样本相等、块边界无毛刺（T05）。

依赖现状：sounddevice/onnxruntime/torch/silero_vad **未装**（本文件全程注入替身，
绝不触碰真实音频/VAD 库）；httpx 0.28.1 **已装**（DEC-005），仅供 T16–T19 的本地
stub transport 与 127.0.0.1 服务用例使用——``qingyun.asr`` 自身对 httpx 是惰性导入，
T11 用干净子进程二次证明 import 零副作用。**本文件零外网请求**。
"""

from __future__ import annotations

import base64
import io
import json
import select
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import configs.app_config as app_config_module
import qingyun.asr as asr
from qingyun import runlog
from qingyun.asr import (
    AUDIO_DATA_URI_PREFIX,
    MAX_RECORD_SAMPLES,
    PRE_ROLL_SAMPLES,
    SILENCE_END_SAMPLES,
    TARGET_SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
)

ROOT = Path(__file__).resolve().parents[1]
_VALID_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 帧数换算（identity 路径：1 块 = 1 帧 = 512 样本）
SILENCE_END_FRAMES = SILENCE_END_SAMPLES // VAD_FRAME_SAMPLES          # 31 余 128
SILENCE_END_FRAMES_TRIGGER = -(-SILENCE_END_SAMPLES // VAD_FRAME_SAMPLES)  # 32
MAX_RECORD_FRAMES_TRIGGER = -(-MAX_RECORD_SAMPLES // VAD_FRAME_SAMPLES)    # 938


# ---------------------------------------------------------------------------
# 替身：假输入流 / 脚本 VAD / 内容 VAD / 假转写
# ---------------------------------------------------------------------------


class FakeStream:
    """假输入流：drain/read/close 全记录到共享 timeline；read 逐块交计划或返回 None。"""

    def __init__(self, blocks, *, samplerate=16000, channels=1, timeline=None):
        self._blocks = list(blocks)
        self.samplerate = float(samplerate)
        self.channels = int(channels)
        self.timeline = timeline if timeline is not None else []
        self.drain_count = 0
        self.read_count = 0
        self.close_count = 0
        self.stopped = False

    def drain(self) -> None:
        self.drain_count += 1
        self.timeline.append("drain")

    def read(self):
        if not self._blocks:
            self.timeline.append("read:exhausted")
            return None
        item = self._blocks.pop(0)
        if isinstance(item, BaseException):
            raise item
        self.read_count += 1
        self.timeline.append(f"read{self.read_count}")
        return (item, self.channels)

    def close(self) -> None:
        self.close_count += 1
        self.stopped = True
        self.timeline.append("close")


class ScriptVAD:
    """固定概率序列替身：每帧依次交出一个概率（与计划逐帧对齐）。"""

    frame_size = VAD_FRAME_SAMPLES

    def __init__(self, probs):
        self._probs = list(probs)
        self.calls = 0

    def probability(self, frame):
        self.calls += 1
        if self._probs:
            return self._probs.pop(0)
        return 0.0


class ContentVAD:
    """按帧内峰值判语音的替身（用于非恒等重采样：块≠帧，不靠脚本对齐）。"""

    frame_size = VAD_FRAME_SAMPLES

    def __init__(self, speech_min=1500):
        self.speech_min = speech_min

    def probability(self, frame):
        peak = max((abs(s) for s in frame), default=0)
        return 1.0 if peak >= self.speech_min else 0.0


class FakeTranscribe:
    """假转写钩子：记录收到的 WAV；把 'transcribe' 记进共享 timeline 以证明停止先于转写。"""

    def __init__(self, text="hello", *, timeline=None):
        self.text = text
        self.calls: list[bytes] = []
        self.timeline = timeline if timeline is not None else []

    def __call__(self, wav_bytes):
        self.timeline.append("transcribe")
        self.calls.append(wav_bytes)
        if isinstance(self.text, BaseException):
            raise self.text
        return self.text


# ---------------------------------------------------------------------------
# 计划构造（identity 16k 单声道：每个计划项 = 1 块 = 1 帧）
# ---------------------------------------------------------------------------

SILENCE = (0.0, 100)     # (概率, 样本值)
PRE_SILENCE = (0.0, 100)
SPEECH = (1.0, 5000)
END_SILENCE = (0.0, 0)


def make_identity_stream(plan, *, timeline=None):
    """(prob, value) 计划 → 单声道 16k 假流 + 脚本 VAD（块与帧一一对应）。"""
    blocks = [[value] * VAD_FRAME_SAMPLES for _prob, value in plan]
    probs = [prob for prob, _value in plan]
    return FakeStream(blocks, samplerate=16000, channels=1, timeline=timeline), ScriptVAD(probs)


def arm_pipeline(monkeypatch, stream_factory, vad, transcribe):
    """把注入边界装到模块上并把运行态置为"已初始化"（listen 只读 _state.vad）。"""
    monkeypatch.setattr(asr, "_open_input_stream", stream_factory)
    monkeypatch.setattr(asr, "_transcribe", transcribe)
    monkeypatch.setattr(asr, "_state", SimpleNamespace(vad=vad))


# ---------------------------------------------------------------------------
# WAV 解码
# ---------------------------------------------------------------------------


def decode_wav(raw: bytes) -> dict:
    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        meta = {
            "nframes": wav_file.getnframes(),
            "framerate": wav_file.getframerate(),
            "channels": wav_file.getnchannels(),
            "sampwidth": wav_file.getsampwidth(),
        }
        data = wav_file.readframes(meta["nframes"])
    import array as _array

    samples = _array.array("h")
    samples.frombytes(data[: meta["nframes"] * meta["sampwidth"]])
    meta["samples"] = list(samples)
    return meta


# ---------------------------------------------------------------------------
# 公共夹具：每例前后把 asr 运行态复位为"未初始化"（避免跨例污染）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_asr_state():
    asr._state = None
    yield
    asr._state = None


# ---------------------------------------------------------------------------
# P2-T01：始终静音 → 永远 WAITING、零转写
# ---------------------------------------------------------------------------


def test_T01_持续静音永远等待且零转写调用(monkeypatch):
    plan = [SILENCE] * 5000  # ~160s 音频的静音，远超 30s，但从未触发
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe(timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    assert result == ""
    assert transcribe.calls == []          # 零转写（§3 WAITING 无总超时）
    assert stream.read_count == 5000       # 长时间按块推进未提前退出
    assert vad.calls == 5000


# ---------------------------------------------------------------------------
# P2-T02：语音→999ms 静音→语音 → 不结束该段（静音计数被语音复位）
# ---------------------------------------------------------------------------


def test_T02_短于1000ms静音后再现语音不结束该段(monkeypatch):
    # 连续静音帧数始终 < 32（<1000ms）：段永不因静音结束。
    plan = (
        [SPEECH]
        + [END_SILENCE] * (SILENCE_END_FRAMES_TRIGGER - 1)  # 31 帧 ≈ 992ms
        + [SPEECH]
        + [END_SILENCE] * (SILENCE_END_FRAMES_TRIGGER - 1)  # 再来 31 帧
        + [SPEECH]
    )
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe(timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()  # 读完后 stream 返回 None（源结束）

    assert transcribe.calls == []   # 复位生效：两段 31 帧静音从未连续到 1000ms
    assert result == ""


def test_T02_随后补齐1000ms静音即正常结束一次(monkeypatch):
    # 对照：中间 31 帧静音未结束，最后连续 32 帧静音才结束——恰一次转写。
    plan = (
        [SPEECH]
        + [END_SILENCE] * (SILENCE_END_FRAMES_TRIGGER - 1)  # 31（不结束）
        + [SPEECH]                                            # 复位
        + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER          # 32（结束）
    )
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe("分拣草莓", timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    assert result == "分拣草莓"
    assert len(transcribe.calls) == 1


# ---------------------------------------------------------------------------
# P2-T03：语音后连续静音 ≥1000ms → 一次完整 WAV 交转写；含约 200ms 前置缓冲
# ---------------------------------------------------------------------------


def test_T03_静音满1000ms交付恰好一个完整WAV并含前置音频(monkeypatch):
    pre_frames = PRE_ROLL_SAMPLES // VAD_FRAME_SAMPLES + 2  # 8 帧足以填满 200ms 前置
    plan = (
        [PRE_SILENCE] * pre_frames
        + [SPEECH] * 2
        + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER
    )
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe("拿草莓", timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    assert result == "拿草莓"
    assert len(transcribe.calls) == 1, "恰好一次完整 WAV"
    decoded = decode_wav(transcribe.calls[0])
    assert decoded["channels"] == 1
    assert decoded["sampwidth"] == 2
    assert decoded["framerate"] == TARGET_SAMPLE_RATE
    # 头部 = 触发前的 200ms 前置音频（值 100），其后才是触发的语音帧（值 5000）。
    assert decoded["samples"][:PRE_ROLL_SAMPLES] == [100] * PRE_ROLL_SAMPLES
    trigger_slice = decoded["samples"][PRE_ROLL_SAMPLES:PRE_ROLL_SAMPLES + VAD_FRAME_SAMPLES]
    assert trigger_slice == [5000] * VAD_FRAME_SAMPLES
    # 关闭输入流发生在转写之前（§3 要点：进入云 ASR 前停止流）。
    assert timeline.index("close") < timeline.index("transcribe")


# ---------------------------------------------------------------------------
# P2-T04：触发后持续语音至 30s → 丢弃、返回空串、一行终端日志、零上传
# ---------------------------------------------------------------------------


def test_T04_录音满30s丢弃并返回空串零上传(monkeypatch):
    console_lines: list[str] = []
    monkeypatch.setattr(runlog, "console", lambda msg: console_lines.append(msg))
    plan = [SPEECH] * (MAX_RECORD_FRAMES_TRIGGER + 60)  # 远超 30s 的连续语音
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe(timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    assert result == ""
    assert transcribe.calls == []          # 零上传
    assert len(console_lines) == 1         # 一行终端日志
    assert "30s" in console_lines[0]
    assert stream.stopped is True          # 仍停止流
    assert stream.read_count == MAX_RECORD_FRAMES_TRIGGER


# ---------------------------------------------------------------------------
# P2-T04b：静音终点与 30s 同步到达 → 先判静音终点（上传，而非丢弃）
# ---------------------------------------------------------------------------


def test_T04b_静音终点与30s同步到达时先判静音终点(monkeypatch):
    console_lines: list[str] = []
    monkeypatch.setattr(runlog, "console", lambda msg: console_lines.append(msg))
    # 让第 938 帧同时满足"连续静音≥1000ms（第 32 个静音帧）"与"自触发≥30s"。
    speech_frames = MAX_RECORD_FRAMES_TRIGGER - SILENCE_END_FRAMES_TRIGGER  # 906
    plan = [SPEECH] * speech_frames + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe("草莓", timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    # 静音终点优先：上传一次、返回文本，而非按 30s 丢弃。
    assert len(transcribe.calls) == 1
    assert result == "草莓"
    assert console_lines == []             # 未走 30s 丢弃分支
    assert stream.read_count == MAX_RECORD_FRAMES_TRIGGER


# ---------------------------------------------------------------------------
# P2-T05：44.1k 立体声 → WAV 头/声道/采样率/位宽/时长正确
# ---------------------------------------------------------------------------


def _stereo_blocks(n_blocks, value, per_channel=441, channels=2):
    """n_blocks 个交织 int16 块（每块 per_channel×channels 样本），值恒为 value。"""
    frame = [value] * (per_channel * channels)
    return [list(frame) for _ in range(n_blocks)]


def test_T05_双声道44k输入得到合规16k单声道WAV(monkeypatch):
    # 触发在首个内容帧（无 WAITING 前置），段持续到静音终点。
    blocks = _stereo_blocks(20, 20000) + _stereo_blocks(120, 0)
    timeline: list[str] = []
    stream = FakeStream(blocks, samplerate=44100, channels=2, timeline=timeline)
    transcribe = FakeTranscribe("草莓", timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, ContentVAD(), transcribe)

    result = asr.listen_and_transcribe()

    assert result == "草莓"
    assert len(transcribe.calls) == 1
    decoded = decode_wav(transcribe.calls[0])
    assert decoded["channels"] == 1
    assert decoded["sampwidth"] == 2
    assert decoded["framerate"] == TARGET_SAMPLE_RATE
    assert decoded["nframes"] % VAD_FRAME_SAMPLES == 0
    # 时长（帧数/16000）与已消费输入音频时长一致，容差在约 1 块内。
    consumed_s = stream.read_count * 441 / 44100
    duration_s = decoded["nframes"] / TARGET_SAMPLE_RATE
    assert abs(duration_s - consumed_s) <= (441 / 44100) + (VAD_FRAME_SAMPLES / TARGET_SAMPLE_RATE)


def test_T05_连续降混与带状态重采样与整段一次性参考逐样本相等():
    # 44.1k 立体声正弦叠加：流式分块 vs 整段一次性参考，必须逐样本一致（无块边界毛刺）。
    per_channel = 337   # 故意不整除，考验跨块余量与相位残留
    channels = 2
    n_frames = per_channel * 200
    left = [int(9000 * ((i * 13) % 89 - 44) / 44) for i in range(n_frames)]
    right = [int(7000 * ((i * 7) % 97 - 48) / 48) for i in range(n_frames)]
    interleaved: list[int] = []
    for i in range(n_frames):
        interleaved.append(left[i])
        interleaved.append(right[i])

    converter = asr._AudioConverter(44100, TARGET_SAMPLE_RATE)
    streaming: list[float] = []
    # 故意用奇数块长（不整除声道数），迫使降混跨块余量在 _chan_buf 携带、跨块不跳变。
    chunk = per_channel * channels + 1
    for start in range(0, len(interleaved), chunk):
        block = interleaved[start:start + chunk]
        streaming.extend(converter.process(block, channels))

    reference = asr._reference_downmix_resample(interleaved, channels, 44100, TARGET_SAMPLE_RATE)

    assert len(streaming) == len(reference)
    tol = 1e-9
    max_err = max((abs(a - b) for a, b in zip(streaming, reference)), default=0.0)
    assert max_err <= tol, f"流式与参考最大误差 {max_err}"
    # 块边界无周期毛刺：以相邻样本一阶差衡量，边界处的跳变不高于内部。
    boundary = per_channel * TARGET_SAMPLE_RATE // 44100
    deltas = [abs(streaming[i] - streaming[i - 1]) for i in range(1, len(streaming))]
    interior_max = max(deltas)
    at_boundary = [deltas[i - 1] for i in range(boundary, len(deltas), boundary)] if boundary > 0 else []
    for jump in at_boundary:
        assert jump <= interior_max + tol, "块边界出现高于内部的跳变（周期毛刺）"


# ---------------------------------------------------------------------------
# P2-T11：干净子进程 import 不拉起音频/云/视觉/配置运行时
# ---------------------------------------------------------------------------


def test_T11_干净子进程import零副作用():
    code = f"""
import sys, json
sys.path.insert(0, {str(ROOT)!r})
import qingyun.asr
forbidden = ["sounddevice", "onnxruntime", "torch", "httpx", "requests",
             "silero_vad", "cv2", "ultralytics", "serial"]
hit = [m for m in forbidden if any(x == m or x.startswith(m + ".") for x in sys.modules)]
print("@@" + json.dumps({{"hit": hit, "asr": qingyun.asr.__file__}}))
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert len(lines) == 1, f"import 期间有额外输出：{done.stdout!r}"
    report = json.loads(lines[0][2:])
    assert report["hit"] == [], f"import 拉起了重依赖 {report['hit']}"
    assert report["asr"]


# ---------------------------------------------------------------------------
# P2-T12：init 前调业务函数 / init 重复调用
# ---------------------------------------------------------------------------


def test_T12_init前调用listen抛ASRHardError():
    assert asr._state is None
    with pytest.raises(asr.ASRHardError) as info:
        asr.listen_and_transcribe()
    assert type(info.value) is asr.ASRHardError
    assert "未初始化" in str(info.value)


def _valid_init_env(monkeypatch, tmp_path, *, key="sk-dashscope-real-key"):
    secrets = tmp_path / "secrets.local.json"
    secrets.write_text(json.dumps({"dashscope_api_key": key, "deepseek_api_key": "x"}),
                       encoding="utf-8")
    monkeypatch.setattr(app_config_module, "SECRETS_LOCAL_PATH", secrets)
    monkeypatch.setattr(asr, "_enumerate_input_devices", lambda: ["mic0"])
    monkeypatch.setattr(asr, "_load_vad", lambda: ScriptVAD([]))


def test_T12_init重复调用抛ASRHardError(monkeypatch, tmp_path):
    _valid_init_env(monkeypatch, tmp_path)
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=_VALID_ENDPOINT,
        cloud_total_timeout_s=30.0))
    asr.init(cfg)
    assert asr._state is not None
    with pytest.raises(asr.ASRHardError) as info:
        asr.init(cfg)
    assert "只允许调用一次" in str(info.value)


# ---------------------------------------------------------------------------
# init 校验：拒绝占位模型名 / 非 https endpoint / 非正超时
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["mock", "", "   ", "MOCK", "placeholder", "changeme"])
def test_init拒绝占位模型名(monkeypatch, tmp_path, model):
    _valid_init_env(monkeypatch, tmp_path)
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model=model, endpoint=_VALID_ENDPOINT, cloud_total_timeout_s=30.0))
    with pytest.raises(asr.ASRHardError):
        asr.init(cfg)
    assert asr._state is None


@pytest.mark.parametrize("endpoint", [
    "http://dashscope.aliyuncs.com/x", "dashscope.aliyuncs.com", "", "   ",
])
def test_init要求https完整endpoint(monkeypatch, tmp_path, endpoint):
    _valid_init_env(monkeypatch, tmp_path)
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=endpoint, cloud_total_timeout_s=30.0))
    with pytest.raises(asr.ASRHardError):
        asr.init(cfg)


@pytest.mark.parametrize("timeout", [0, -1, 0.0, True])
def test_init要求正有限cloud_total_timeout(monkeypatch, tmp_path, timeout):
    _valid_init_env(monkeypatch, tmp_path)
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=_VALID_ENDPOINT,
        cloud_total_timeout_s=timeout))
    with pytest.raises(asr.ASRHardError):
        asr.init(cfg)


def test_init成功设置运行态且日志不回显key(monkeypatch, tmp_path, capsys):
    _valid_init_env(monkeypatch, tmp_path, key="super-secret-dashscope")
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=_VALID_ENDPOINT,
        cloud_total_timeout_s=30.0))
    asr.init(cfg)
    assert asr._state is not None
    assert asr._state.api_key == "super-secret-dashscope"
    assert asr._state.model == "qwen-audio-3.1-asr-flash"
    # 任何输出都不得含 key 值（D12）。
    captured = capsys.readouterr()
    assert "super-secret-dashscope" not in captured.out + captured.err


# ---------------------------------------------------------------------------
# P2-T13（asr 侧）：secrets 缺失/全空白 → init 抛 ASRHardError，且失败态干净可重试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["", "     ", None])
def test_T13缺失或全空白密钥init拒绝(monkeypatch, tmp_path, key):
    secrets = tmp_path / "secrets.local.json"
    payload = {"dashscope_api_key": key, "deepseek_api_key": "x"}
    if key is None:
        payload = {"deepseek_api_key": "x"}  # 缺 dashscope 键
    secrets.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_config_module, "SECRETS_LOCAL_PATH", secrets)
    monkeypatch.setattr(asr, "_enumerate_input_devices", lambda: ["mic0"])
    monkeypatch.setattr(asr, "_load_vad", lambda: ScriptVAD([]))
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=_VALID_ENDPOINT,
        cloud_total_timeout_s=30.0))
    with pytest.raises(asr.ASRHardError):
        asr.init(cfg)
    assert asr._state is None, "失败路径状态保持未初始化（可修复后重试）"
    # 异常消息不得含任何 key 值。
    # （None 键场景 load_secrets 因缺键报错；空/空白键场景命中显式拒绝——都归 ASRHardError。）


def test_T13异常消息不回显key(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.local.json"
    secrets.write_text(json.dumps({"dashscope_api_key": "   ", "deepseek_api_key": "x"}),
                       encoding="utf-8")
    monkeypatch.setattr(app_config_module, "SECRETS_LOCAL_PATH", secrets)
    monkeypatch.setattr(asr, "_enumerate_input_devices", lambda: ["mic0"])
    monkeypatch.setattr(asr, "_load_vad", lambda: ScriptVAD([]))
    cfg = SimpleNamespace(asr=SimpleNamespace(
        model="qwen-audio-3.1-asr-flash", endpoint=_VALID_ENDPOINT,
        cloud_total_timeout_s=30.0))
    with pytest.raises(asr.ASRHardError) as info:
        asr.init(cfg)
    msg = str(info.value)
    assert "   " not in msg or "dashscope_api_key" in msg  # 只点名键，不泄漏值形态


# ---------------------------------------------------------------------------
# P2-T14：转写返回空/纯空白 → 清理后返回空串
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\t\n "])
def test_T14转写返回空白则listen返回空串(monkeypatch, text):
    plan = [SPEECH] + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    transcribe = FakeTranscribe(text, timeline=timeline)
    arm_pipeline(monkeypatch, lambda: stream, vad, transcribe)

    result = asr.listen_and_transcribe()

    assert result == ""
    assert len(transcribe.calls) == 1      # 确实上传了一次
    assert stream.stopped is True


# ---------------------------------------------------------------------------
# P2-T15：段结束后回到 WAITING、缓冲/流复位；每次 listen 重启流并先 drain 再读
# ---------------------------------------------------------------------------


def test_T15_段结束后新listen回到WAITING且先排空再读(monkeypatch):
    timeline: list[str] = []
    # 第一段：语音 + 连续 1000ms 静音 → 结束一次。
    first, vad1 = make_identity_stream([SPEECH] + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER,
                                       timeline=timeline)
    # 第二段：全新计划（全静音）→ 应保持 WAITING、不再转写。
    second, vad2 = make_identity_stream([SILENCE] * 40, timeline=timeline)
    streams = [first, second]
    factory_calls: list[int] = []

    def factory():
        factory_calls.append(1)
        return streams.pop(0)

    transcribe = FakeTranscribe("草莓", timeline=timeline)
    monkeypatch.setattr(asr, "_open_input_stream", factory)
    monkeypatch.setattr(asr, "_transcribe", transcribe)
    holder = SimpleNamespace(vad=None)
    monkeypatch.setattr(asr, "_state", holder)

    # 第一 listen：注入 vad1 段脚本 VAD
    holder.vad = vad1
    r1 = asr.listen_and_transcribe()
    assert r1 == "草莓"
    assert len(transcribe.calls) == 1
    assert first.drain_count == 1 and first.read_count > 0
    assert first.stopped is True
    # drain 必须先于任何读取（排空 PLAN 期间积压输入）
    assert timeline.index("drain") < timeline.index("read1")

    # 第二 listen：新流 + 复位后的状态机（历史/段缓冲都是 _capture 局部，天然复位）
    holder.vad = vad2
    r2 = asr.listen_and_transcribe()
    assert r2 == ""                       # 回到 WAITING，全静音不触发、零额外转写
    assert len(transcribe.calls) == 1     # 未新增转写
    assert factory_calls == [1, 1]        # 每次 listen 重启流
    assert second.drain_count == 1 and second.stopped is True


# ===========================================================================
# P2-02：DashScope 私有适配（§3.1）与端到端 deadline（§3.2）——T16/T17/T18/T19
#
# 替身纪律（P2 §5 末段）：HTTP 层一律用**注入的本地 stub transport**
# （httpx.MockTransport，绝不 monkeypatch socket）；只有 T17 反过来**刻意不用任何
# stub**——真实 transport + 127.0.0.1 本地服务子进程做墙钟证明（R05）。全部用例零外网、
# 零真实账号。
# ===========================================================================

DASHSCOPE_KEY_FOR_TESTS = "sk-unit-test-dashscope-0123456789-abcdef"
AUDIO_FIXTURE = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + bytes(48)
MODEL_ID = "qwen-audio-3.1-asr-flash"


def arm_cloud_state(monkeypatch, **overrides):
    """装上 init 成功后的运行态（云适配只读 model/endpoint/cloud_total_timeout_s/api_key）。"""
    fields = {
        "model": MODEL_ID,
        "endpoint": _VALID_ENDPOINT,
        "cloud_total_timeout_s": 30.0,
        "api_key": DASHSCOPE_KEY_FOR_TESTS,
        "vad": None,
    }
    fields.update(overrides)
    state = SimpleNamespace(**fields)
    monkeypatch.setattr(asr, "_state", state)
    return state


class StubClient(httpx.Client):
    """记录 close() 次数的本地 Client（外加测试侧观察到的分项超时值）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_count = 0
        self.observed_timeout_s = None

    def close(self) -> None:
        self.close_count += 1
        super().close()


class StubHttp:
    """HTTP 层替身的记录本：到达过的请求、建过的 client、每次的分项超时。"""

    def __init__(self, responder):
        self._responder = responder
        self.requests: list[httpx.Request] = []
        self.clients: list[StubClient] = []
        self.item_timeouts: list[float] = []

    def __call__(self, request):
        self.requests.append(request)
        return self._responder(request)

    @property
    def count(self) -> int:
        return len(self.requests)


def install_stub_http_layer(monkeypatch, responder) -> StubHttp:
    """把 ``asr._new_http_client`` 换成"本地 stub transport 的 Client"工厂（注入式）。"""
    stub = StubHttp(responder)

    def factory(*, timeout_s):
        client = StubClient(
            transport=httpx.MockTransport(stub),
            timeout=httpx.Timeout(connect=timeout_s, read=timeout_s, write=timeout_s,
                                  pool=timeout_s),
            follow_redirects=False,
        )
        client.observed_timeout_s = float(timeout_s)
        stub.clients.append(client)
        stub.item_timeouts.append(float(timeout_s))
        return client

    monkeypatch.setattr(asr, "_new_http_client", factory)
    return stub


def ok_response(text="分拣草莓"):
    """DashScope 成功响应的最小合法形态（§3.1：读 output.text）。"""
    return httpx.Response(200, json={"output": {"text": text}})


def response_body(payload):
    return httpx.Response(200, json=payload)


class ChunkStream(httpx.SyncByteStream):
    """按片交付的响应体。

    ``gap_s`` 让每片之间真实等待（制造"数据在流、整体很慢"的形态）；
    ``raise_after=(n, exc)`` 在交出第 n 片后抛该异常（用于注入 Ctrl-C）。
    ``closed`` 是"未完成响应是否被关闭"的直接证据（httpx 只关 SyncByteStream 实例）。
    """

    def __init__(self, chunks, *, gap_s=0.0, raise_after=None):
        self.chunks = list(chunks)
        self.gap_s = float(gap_s)
        self.raise_after = raise_after
        self.delivered = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.gap_s:
                time.sleep(self.gap_s)
            self.delivered += 1
            yield chunk
            if self.raise_after and self.delivered >= self.raise_after[0]:
                raise self.raise_after[1]

    def close(self) -> None:
        self.closed = True


def stream_response(status, stream, **headers):
    return httpx.Response(status, stream=stream,
                          headers={"Content-Type": "application/json", **headers})


def alarm_state():
    """(当前 SIGALRM 处理器, ITIMER_REAL 读数)——T18 的状态卫生断言用。"""
    return signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)


# ---------------------------------------------------------------------------
# P2-T16：假时钟推满 cloud_total_timeout_s → ASRHardError，且只有单次请求
# ---------------------------------------------------------------------------


def test_T16_假时钟推满总时限得到deadline硬错误且只发一次请求(monkeypatch):
    budget = 30.0
    clock = {"now": 1_000.0}
    monkeypatch.setattr(asr, "_monotonic", lambda: clock["now"])   # 不真实等待
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=budget)
    handler_before = signal.getsignal(signal.SIGALRM)
    armed: list[tuple[float, float]] = []

    def responder(request):
        # 请求进行中的时刻：端到端闹钟必须已经装填（真 signal 接口，只是永不到点）。
        armed.append(signal.getitimer(signal.ITIMER_REAL))
        clock["now"] += budget           # 这次请求"耗尽"了整段总时限
        return ok_response()

    stub = install_stub_http_layer(monkeypatch, responder)

    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)

    assert stub.count == 1, "单次请求：适配层不得自行重试"
    assert type(info.value) is asr.ASRHardError
    assert "deadline_timeout" in str(info.value)
    assert "cloud_total_timeout_s" in str(info.value)
    assert isinstance(info.value.__cause__, asr._AsrDeadlineExceeded), "保留来源"
    # 分项超时全部取剩余预算（≤ 总时限），且起算就是整段预算
    assert stub.item_timeouts == [pytest.approx(budget)]
    assert armed and 0.0 < armed[0][0] <= budget, "请求期间 ITIMER_REAL 处于装填状态"
    # 退出后：计时器取消、旧处理器恢复
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert stub.clients[0].close_count == 1, "客户端一定被关闭"


def test_T16_剩余预算非正时不再发起请求且退出即取消计时器(monkeypatch):
    """deadline 的"预算耗尽"闸门：分项超时**不超过**剩余预算，预算没了就不该再扔音频。

    直接对 deadline 上下文做单元测试（不经 HTTP），把"绝对截止点"这一实现细节钉死：
    剩余 ≤ 0 一律判到期，且到期不会留下任何计时器/处理器状态。
    """
    monkeypatch.setattr(asr, "_monotonic", lambda: 100.0)
    stale = asr._CloudDeadline(30.0, request_id="stale")
    stale.deadline_at = 90.0                       # 绝对截止点已在过去
    assert stale.remaining_s() == pytest.approx(-10.0)
    with pytest.raises(asr._AsrDeadlineExceeded):
        stale.item_timeout_s()

    clock = {"now": 0.0}
    monkeypatch.setattr(asr, "_monotonic", lambda: clock["now"])
    handler_before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(asr._AsrDeadlineExceeded):
        with asr._CloudDeadline(2.0, request_id="burn") as live:
            assert live.item_timeout_s() == pytest.approx(2.0)
            clock["now"] = 2.5                     # 请求途中预算被吃完
            live.item_timeout_s()                  # → 抛（并被 with 的退出校验兜住）
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before


# ---------------------------------------------------------------------------
# P2-T17（R05 核心证明）：127.0.0.1 本地服务**持续缓慢返回数据**，
# 端到端 deadline 必须在总时限处打断整次请求——不用"立即超时"的 stub。
# ---------------------------------------------------------------------------

# 服务替身的源码（内联，避免新增文件）：由子进程 `python -c` 执行。
_SLOW_SERVER_SOURCE = '''
"""P2-T17 本地慢响应 ASR 服务替身（只绑 127.0.0.1，绝不触云、不消耗账号）。

用法：python -c <本源码> <状态文件> <块间隔秒> <stall总秒> <慢响应文本>
路径分派：
  /fast_ok        立即返回完整合法 JSON，并把"收到的请求特征"写进状态文件
  /slow_stall     声明超大 Content-Length，每 gap 秒挤 8 字节，共 stall 秒 —— 永不读完
  /slow_complete  **完整合法**的 JSON 响应，每 gap 秒挤 12 字节慢慢送完（响应正常但慢）
  /redirect       302 + Location（验证不跟随 redirect）
每送一片追加一行 "<path> sent=<n> t=<相对秒>"，供测试证明"数据一直在流"。
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS_PATH = sys.argv[1]
GAP_S = float(sys.argv[2])
STALL_S = float(sys.argv[3])
SLOW_TEXT = sys.argv[4]
SLOW_BODY = json.dumps({"output": {"text": SLOW_TEXT}}).encode("utf-8")
OK_BODY = json.dumps({"output": {"text": "分拣草莓"}}).encode("utf-8")
STALL_DECLARE = 100000


def note(line):
    with open(STATUS_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\\n")


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 响应：答完即关连接，keep-alive 不会把两次请求混进同一条连接。
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        return

    def _read_request_body(self):
        declared = int(self.headers.get("Content-Length") or 0)
        left = declared
        while left > 0:
            chunk = self.rfile.read(min(left, 65536))
            if not chunk:
                break
            left -= len(chunk)
        return declared

    def do_POST(self):
        declared = self._read_request_body()
        path = self.path.strip("/")
        port = self.server.server_address[1]
        if path == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d/redirected" % port)
            self.end_headers()
            note("redirect sent")
            return
        if path == "fast_ok":
            note("fast_ok received clen=%d sse=%s ctype=%s bearer=%s" % (
                declared,
                self.headers.get("X-DashScope-SSE"),
                self.headers.get("Content-Type"),
                (self.headers.get("Authorization") or "").startswith("Bearer "),
            ))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(OK_BODY)))
            self.end_headers()
            self.wfile.write(OK_BODY)
            return
        stalled = path == "slow_stall"
        piece = 8 if stalled else 12
        body = (OK_BODY + b"p" * STALL_DECLARE) if stalled else SLOW_BODY
        total = STALL_DECLARE if stalled else len(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(total))
        self.end_headers()
        started = time.monotonic()
        index = 0
        sent = 0
        try:
            while index < total:
                chunk = body[index:index + piece]
                self.wfile.write(chunk)
                index += len(chunk)
                sent += 1
                note("%s sent=%d t=%.3f" % (path, sent, time.monotonic() - started))
                time.sleep(GAP_S)
                if stalled and (time.monotonic() - started) >= STALL_S:
                    break
        except OSError as exc:
            note("%s aborted sent=%d t=%.3f err=%s" % (
                path, sent, time.monotonic() - started, type(exc).__name__))


SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
sys.stdout.write("PORT=%d\\n" % SERVER.server_address[1])
sys.stdout.flush()
SERVER.serve_forever(poll_interval=0.05)
'''

# 慢响应文本：让"完整合法但要慢慢送完"的总时长落在 ~0.9s（远大于失败用例的预算）。
SLOW_TEXT = "分拣草莓" + "嗯" * 6 + "x" * 120
SERVER_GAP_S = 0.05
SERVER_STALL_S = 3.0


@pytest.fixture(scope="module")
def local_asr_server(tmp_path_factory):
    """启动/收尾 127.0.0.1 慢响应服务子进程（模块内复用，避免多次付启动成本）。"""
    directory = tmp_path_factory.mktemp("asrslowserver")
    status = directory / "sent.log"
    status.write_text("", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", _SLOW_SERVER_SOURCE, str(status),
         str(SERVER_GAP_S), str(SERVER_STALL_S), SLOW_TEXT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT),
    )
    try:
        readable, _, _ = select.select([process.stdout], [], [], 20.0)
        assert readable, f"本地服务未就绪：{process.stderr.read()[:2000]}"
        first = process.stdout.readline().strip()
        assert first.startswith("PORT="), first
        port = int(first.split("=", 1)[1])
        yield {"port": port, "base": f"http://127.0.0.1:{port}", "status": status}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:      # pragma: no cover - 兜底
            process.kill()


def sent_lines(status_path: Path, mode: str) -> list[str]:
    """状态文件里属于该模式且"确实送出一片"的记录行。"""
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.startswith(f"{mode} sent=")]


def test_T17_持续缓慢返回数据被端到端deadline在总时限处打断(local_asr_server, monkeypatch):
    """R05 证明：块间隔 50ms ≪ read 超时，**没有任何分项超时会触发**；
    只有 signal.setitimer(ITIMER_REAL) 能在总时限处打断这条阻塞中的读取。
    """
    server = local_asr_server
    budget = 0.6
    arm_cloud_state(monkeypatch, endpoint=f"{server['base']}/slow_stall",
                    cloud_total_timeout_s=budget)
    handler_before, _ = alarm_state()
    before = len(sent_lines(server["status"], "slow_stall"))

    started = time.monotonic()
    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)          # 真实 transport、真实墙钟
    elapsed = time.monotonic() - started
    delivered = len(sent_lines(server["status"], "slow_stall")) - before

    assert "deadline_timeout" in str(info.value)
    assert isinstance(info.value.__cause__, asr._AsrDeadlineExceeded)
    # ① 打断发生在总时限处：不早到"没等够"、不晚到"等服务自己结束"
    assert elapsed >= budget * 0.9, f"早于总时限退出，不像墙钟中断：{elapsed:.3f}s"
    assert elapsed <= budget + 0.5, f"未被及时中断：{elapsed:.3f}s"
    assert elapsed < SERVER_STALL_S - 0.5, "远早于服务自身结束（不是等它收尾才失败）"
    # ② 数据一直在流动：至少 3 片已送达，故"立即返回超时"的 stub 不能替代本证明
    assert delivered >= 3, f"服务端只送了 {delivered} 片，未构成持续慢响应"
    # ③ 分项 read 超时（= 剩余预算）在这条间隔下**永不**可能触发
    assert SERVER_GAP_S * 5 < budget
    # ④ 被打断的是真实 socket 读取：signal 状态与计时器仍被完整复原
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_T17_响应正常但整体超预算同样被硬中断(local_asr_server, monkeypatch):
    """变体：响应**最终是完整合法 JSON**，只是送得比总时限慢——同样必须失败。"""
    server = local_asr_server
    budget = 0.4
    arm_cloud_state(monkeypatch, endpoint=f"{server['base']}/slow_complete",
                    cloud_total_timeout_s=budget)
    before = len(sent_lines(server["status"], "slow_complete"))

    started = time.monotonic()
    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)
    elapsed = time.monotonic() - started
    delivered = len(sent_lines(server["status"], "slow_complete")) - before

    assert "deadline_timeout" in str(info.value)
    assert budget * 0.9 <= elapsed <= budget + 0.5, f"{elapsed:.3f}s"
    assert delivered >= 3
    # 服务侧总耗时（≈ len(body)/12 * gap）明显大于预算：预算内本来读不完
    assert budget < 1.5, "本例的预算必须小于送完全部数据所需时间"


def test_T17_同一慢服务在充足预算下正常成功_对照证明中断归因于总时限(
        local_asr_server, monkeypatch):
    """对照组：**同一个**慢响应、同样的代码路径，只要预算够就成功。

    这排除"连接坏了/服务立刻超时"之类的替代解释——失败的唯一原因是墙钟总时限。
    """
    server = local_asr_server
    arm_cloud_state(monkeypatch, endpoint=f"{server['base']}/slow_complete",
                    cloud_total_timeout_s=8.0)
    started = time.monotonic()
    text = asr._transcribe(AUDIO_FIXTURE)
    elapsed = time.monotonic() - started

    assert text == SLOW_TEXT, "分片送达的完整响应必须被原样解析"
    assert 0.2 <= elapsed < 5.0
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_T17_真实transport下的正常请求成功且signal状态干净(local_asr_server, monkeypatch):
    """真实 HTTP 往返（127.0.0.1）+ deadline：正常响应不受计时器机制影响。

    顺带在服务端留痕上核对 wire 级事实：认证头形态、Content-Type、SSE 头、
    请求体确实按声明的 Content-Length 完整上传。
    """
    server = local_asr_server
    arm_cloud_state(monkeypatch, endpoint=f"{server['base']}/fast_ok",
                    cloud_total_timeout_s=10.0)
    handler_before, _ = alarm_state()
    status_before = server["status"].read_text(encoding="utf-8").count("fast_ok received")

    text = asr._transcribe(AUDIO_FIXTURE)

    assert text == "分拣草莓"
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    lines = [line for line in server["status"].read_text(encoding="utf-8").splitlines()
             if line.startswith("fast_ok received")]
    assert len(lines) == status_before + 1
    received = lines[-1]
    assert "sse=disable" in received and "ctype=application/json" in received
    assert "bearer=True" in received
    assert "clen=" in received and not received.endswith("clen=0")
    # 脱敏：留痕里绝不允许出现密钥值
    assert DASHSCOPE_KEY_FOR_TESTS not in received


# ---------------------------------------------------------------------------
# P2-T18：deadline 成功/失败/Ctrl-C 之后的 signal 状态与资源清理
# ---------------------------------------------------------------------------


def test_T18_专用deadline异常不继承OSError也不继承Exception():
    """§3.2 的字面要求：处理器抛的专用异常**不继承 OSError**（防 EINTR 自动重试吞掉）。"""
    assert issubclass(asr._AsrDeadlineExceeded, BaseException)
    assert not issubclass(asr._AsrDeadlineExceeded, OSError)
    assert not issubclass(asr._AsrDeadlineExceeded, Exception)
    # 且它永不出现在公开面上（__all__ 只有契约三件套）
    assert "_AsrDeadlineExceeded" not in asr.__all__


def test_T18_成功请求后旧处理器恢复计时器取消且响应已关闭(monkeypatch):
    handler_before, _ = alarm_state()
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=10.0)
    body = json.dumps({"output": {"text": "拿草莓"}}).encode("utf-8")
    stream = ChunkStream([body[:10], body[10:]])
    stub = install_stub_http_layer(
        monkeypatch, lambda request: stream_response(200, stream))

    text = asr._transcribe(AUDIO_FIXTURE)

    assert text == "拿草莓"
    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before, "SIGALRM 处理器必须交还原主"
    assert timer_after == (0.0, 0.0), "ITIMER_REAL 必须已取消"
    assert stream.closed is True and stream.delivered == 2
    assert stub.clients and all(c.close_count == 1 for c in stub.clients)


def test_T18_到期失败后同样恢复signal并关闭未完成响应(monkeypatch):
    handler_before, _ = alarm_state()
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=0.3)
    body = json.dumps({"output": {"text": "拿草莓"}}).encode("utf-8")
    # 两片之间睡 0.2s：0.3s 的闹钟必然落在"响应还没读完"的时候
    stream = ChunkStream([body[:10], body[10:]], gap_s=0.2)
    stub = install_stub_http_layer(
        monkeypatch, lambda request: stream_response(200, stream))

    started = time.monotonic()
    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)
    elapsed = time.monotonic() - started

    assert "deadline_timeout" in str(info.value)
    assert elapsed <= 1.0, f"闹钟未及时中断：{elapsed:.3f}s"
    assert 0 < stream.delivered < 2, "响应应处于未完成状态（被打断在中途）"
    assert stream.closed is True, "未完成的响应必须被关闭"
    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before
    assert timer_after == (0.0, 0.0)
    assert stub.clients[0].close_count == 1


def test_T18_CtrlC中断后signal状态与资源清理同样成立(monkeypatch):
    """把 KeyboardInterrupt 塞进响应迭代：清理照做，异常**原样上抛**不洗白。"""
    handler_before, _ = alarm_state()
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=10.0)
    stream = ChunkStream([b'{"output": {"tex', b't": ""}}'],
                         raise_after=(1, KeyboardInterrupt()))
    stub = install_stub_http_layer(monkeypatch,
                                   lambda request: stream_response(200, stream))

    with pytest.raises(KeyboardInterrupt):
        asr._transcribe(AUDIO_FIXTURE)

    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before, "Ctrl-C 也必须走 finally 恢复处理器"
    assert timer_after == (0.0, 0.0)
    assert stream.closed is True
    assert stub.clients[0].close_count == 1


def test_T18_到期后的下一次请求不被旧闹钟打断(monkeypatch):
    handler_before, _ = alarm_state()
    # 第一次：0.3s 预算 + 慢响应 → 到期
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=0.3)
    slow = ChunkStream([b'{"output": {"te', b'xt": ""}}'], gap_s=0.25)
    first_stub = install_stub_http_layer(monkeypatch,
                                         lambda request: stream_response(200, slow))
    with pytest.raises(asr.ASRHardError):
        asr._transcribe(AUDIO_FIXTURE)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    # 等过"旧闹钟"的原始到点时刻：若计时器没被取消，这期间必然再触发一次 SIGALRM。
    time.sleep(0.6)

    # 第二次：同样的响应节奏、充足预算 → 必须成功（说明没有遗留闹钟在打断）
    body = json.dumps({"output": {"text": "第二条"}}).encode("utf-8")
    second = ChunkStream([body[:12], body[12:]], gap_s=0.1)
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=10.0)
    second_stub = install_stub_http_layer(monkeypatch,
                                          lambda request: stream_response(200, second))
    assert asr._transcribe(AUDIO_FIXTURE) == "第二条"

    assert first_stub.count == 1 and second_stub.count == 1
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before


def test_T18_已有活动ITIMER_REAL时拒绝执行且不动别人的定时器(monkeypatch):
    handler_before, _ = alarm_state()
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response())
    arm_cloud_state(monkeypatch)
    signal.setitimer(signal.ITIMER_REAL, 30.0)     # 别的组件的定时器
    try:
        with pytest.raises(asr.ASRHardError) as info:
            asr._transcribe(AUDIO_FIXTURE)
        still_running, interval = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    assert "不支持的调用环境" in str(info.value)
    assert "ITIMER_REAL" in str(info.value)
    assert stub.count == 0, "拒绝必须发生在发请求之前"
    assert still_running > 25.0, "别人的计时器不能被本模块取消或改写"
    assert interval == 0.0
    assert signal.getsignal(signal.SIGALRM) is handler_before


def test_T18_非主线程调用被明确拒绝(monkeypatch):
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response())
    arm_cloud_state(monkeypatch)
    outcomes: list[BaseException] = []

    def worker():
        try:
            asr._transcribe(AUDIO_FIXTURE)
        except BaseException as exc:      # noqa: BLE001 - 收集线程内结果交主线程断言
            outcomes.append(exc)

    thread = threading.Thread(target=worker, name="not-main")
    thread.start()
    thread.join(timeout=20.0)
    assert not thread.is_alive()

    assert len(outcomes) == 1
    assert type(outcomes[0]) is asr.ASRHardError
    assert "不支持的调用环境" in str(outcomes[0])
    assert "主线程" in str(outcomes[0])
    assert stub.count == 0


def test_T18_每次请求单独起表_第二次请求拿到完整预算(monkeypatch):
    """第一次把预算烧掉 90%（假时钟），第二次仍从**完整**总时限起算——两表互不相干。"""
    budget = 20.0
    clock = {"now": 500.0}
    monkeypatch.setattr(asr, "_monotonic", lambda: clock["now"])   # 按调用次数推进无意义：
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=budget)     # 只在 responder 里推进
    burns = iter([budget * 0.9, 0.0])     # 第一次烧掉 18s（仍在预算内），第二次不烧

    def responder(request):
        clock["now"] += next(burns)
        return ok_response()

    stub = install_stub_http_layer(monkeypatch, responder)

    assert asr._transcribe(AUDIO_FIXTURE) == "分拣草莓"
    assert asr._transcribe(AUDIO_FIXTURE) == "分拣草莓"

    assert stub.count == 2
    # 每次请求的分项超时都等于**当时**的剩余预算：两次都是完整 20s（单独起表）。
    assert len(stub.item_timeouts) % 2 == 0
    assert all(value == pytest.approx(budget) for value in stub.item_timeouts)
    assert clock["now"] == pytest.approx(518.0)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# P2-T19（ASR 侧）：固定请求/响应 fixture 逐字段核对（§3.1 协议 + 无重试 + 禁 redirect）
# ---------------------------------------------------------------------------


def test_T19_请求逐字段符合DashScope协议并正确解析output_text(monkeypatch):
    arm_cloud_state(monkeypatch)
    captured: list[httpx.Request] = []

    def responder(request):
        captured.append(request)
        return ok_response("请帮我分拣草莓")

    stub = install_stub_http_layer(monkeypatch, responder)
    wav = asr._encode_wav_mono16([1000] * 512)     # 真实 WAV 字节（16k/mono/PCM16）

    text = asr._transcribe(wav)

    assert text == "请帮我分拣草莓"
    assert stub.count == 1
    request = captured[0]
    # URL / 方法：完全取配置值，不改写、不换域名、不跨地域 fallback
    assert request.method == "POST"
    assert str(request.url) == _VALID_ENDPOINT
    # 请求头逐字（§3.1）
    assert request.headers["Authorization"] == f"Bearer {DASHSCOPE_KEY_FOR_TESTS}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["X-DashScope-SSE"] == "disable"
    # body 结构逐字
    body = json.loads(request.content.decode("utf-8"))
    assert set(body) == {"model", "input", "parameters"}
    assert body["model"] == MODEL_ID                      # 模型 ID 取配置（D12/D20）
    assert body["parameters"] == {"format": "wav", "sample_rate": "16000"}
    assert set(body["input"]) == {"messages"}
    messages = body["input"]["messages"]
    assert len(messages) == 1, "不携带历史音频/对话"
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list) and len(content) == 1
    assert set(content[0]) == {"type", "input_audio"}
    assert content[0]["type"] == "input_audio"
    assert set(content[0]["input_audio"]) == {"data"}
    data_uri = content[0]["input_audio"]["data"]
    assert data_uri.startswith(AUDIO_DATA_URI_PREFIX), "data:audio/wav;base64, 前缀"
    assert base64.b64decode(data_uri.split(",", 1)[1]) == wav, "上传的就是这段 WAV"
    # 分项超时 = 剩余预算（起算即整段总时限，且绝不超过它）
    assert stub.item_timeouts and all(
        29.0 < value <= 30.0 for value in stub.item_timeouts)


@pytest.mark.parametrize("payload,expected", [
    ({"output": {"text": "分拣草莓"}}, "分拣草莓"),
    ({"output": {"text": ""}}, ""),                     # 空串是合法结果（§3.1）
    ({"output": {"text": "   "}}, "   "),               # 空白由 listen 侧判为"无口令"
    ({"output": {"text": "草莓"}, "request_id": "r-1", "usage": {"characters": 8}}, "草莓"),
])
def test_T19_成功响应fixture解析(monkeypatch, payload, expected):
    arm_cloud_state(monkeypatch)
    stub = install_stub_http_layer(monkeypatch, lambda request: response_body(payload))
    assert asr._transcribe(AUDIO_FIXTURE) == expected
    assert stub.count == 1


@pytest.mark.parametrize("status,category", [
    (401, "auth"),
    (403, "auth"),
    (429, "rate_limited"),
    (400, "request_rejected"),
    (500, "service_error"),
    (503, "service_error"),
])
def test_T19_服务错误按类别硬失败且只请求一次(monkeypatch, status, category):
    """鉴权/限流/服务错误分类记录，统一 ASRHardError；**无自动重试**。"""
    arm_cloud_state(monkeypatch)
    error_body = {"code": "InvalidApiKey", "message": "绝不回显的内容 " + "z" * 400}
    stub = install_stub_http_layer(
        monkeypatch,
        lambda request: httpx.Response(status, json=error_body),
    )

    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)

    assert stub.count == 1, f"HTTP {status} 不得触发任何重试"
    assert category in str(info.value)
    assert f"HTTP {status}" in str(info.value)
    assert "InvalidApiKey" in str(info.value)          # 服务侧 code 可用于定位
    assert "绝不回显的内容" not in str(info.value)      # 响应正文不入异常/日志


def test_T19_3xx不跟随redirect直接判错且不再发第二次请求(monkeypatch):
    """禁自动 redirect（§3.2）：302 是**错误**，不是"再发一次到 Location"。"""
    arm_cloud_state(monkeypatch)
    stub = install_stub_http_layer(monkeypatch, lambda request: httpx.Response(
        302, headers={"Location": "https://other-region.invalid/generation"}))

    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)

    assert stub.count == 1
    assert "redirect_not_followed" in str(info.value)
    assert str(stub.requests[0].url) == _VALID_ENDPOINT   # 仍然只碰配置里的 endpoint


@pytest.mark.parametrize("body", [
    b"not json at all",                                   # 坏 JSON
    b"",                                                  # 空响应体
    b"[1,2,3]",                                           # 顶层非对象
    b'{"output": "text"}',                                # output 不是对象（多余结构）
    b'{"output": {"text": {"nested": 1}}}',               # text 是对象（多余结构）
    b'{"output": {"text": ["fen", "tiao"]}}',             # text 是数组
    b'{"output": {"text": 123}}',                          # text 是数字
    b'{"output": {"text": null}}',                         # text 是 null
    b'{"output": {}}',                                    # 缺 text
    b'{"output": {"sentence": "mei"}}',                   # 缺 text（别的字段）
    b'{"text": "mei"}',                                    # 缺 output
    b'{}',                                                # 只有空对象
    b'{"request_id": "r-1"}',                              # 缺 output
    b"\xff\xfe not utf-8",                                 # 非 UTF-8 字节
])
def test_T19_坏响应一律ASRHardError(monkeypatch, body):
    arm_cloud_state(monkeypatch)
    stream = ChunkStream([body])
    stub = install_stub_http_layer(
        monkeypatch,
        lambda request: stream_response(200, stream),
    )

    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)

    assert type(info.value) is asr.ASRHardError
    message = str(info.value)
    assert ("bad_json" in message) or ("bad_payload" in message)
    assert stub.count == 1, "协议违约也不得自动重试"


def test_T19_传输层异常保留来源且不重试(monkeypatch):
    arm_cloud_state(monkeypatch)
    attempts: list[int] = []

    def responder(request):
        attempts.append(1)
        raise httpx.ConnectError("connection refused")

    stub = install_stub_http_layer(monkeypatch, responder)

    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(AUDIO_FIXTURE)

    assert len(attempts) == 1
    assert "transport" in str(info.value)
    assert isinstance(info.value.__cause__, httpx.ConnectError), "保留来源异常"


def test_T19_默认客户端显式关闭重试与redirect并压平分项超时():
    """§3.2 的三条硬要求在默认工厂里是**显式字面量**，不依赖库默认值（白盒核对）。"""
    client = asr._new_http_client(timeout_s=1.75)
    try:
        assert client.follow_redirects is False
        assert client.timeout == httpx.Timeout(connect=1.75, read=1.75, write=1.75,
                                              pool=1.75)
        pool = client._transport._pool                  # noqa: SLF001 - 配置白盒核对
        assert pool._retries == 0                       # transport 层零重试
        assert pool._http2 is False                     # 不走 HTTP/2（无隐式多路复用面）
        assert pool._http1 is True
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 脱敏：日志事件与异常消息都不含密钥/Authorization/音频 base64
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_events(tmp_path):
    """把 runlog 接到内存 sink（P1 §3.4 的注入点），返回收到的 JSON 行列表。"""
    runlog.reset_for_tests()
    runlog.init(tmp_path / "logs")
    lines: list[str] = []
    runlog.set_sink(lines.append)
    yield lines
    runlog.reset_for_tests()


def test_T19_日志与异常均脱敏_不含密钥与音频base64(monkeypatch, captured_events):
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=15.0)
    wav = asr._encode_wav_mono16([123] * 2048)
    needle = base64.b64encode(wav).decode("ascii")

    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response("草莓"))
    assert asr._transcribe(wav) == "草莓"
    assert all(needle not in line for line in captured_events)
    assert all(DASHSCOPE_KEY_FOR_TESTS not in line for line in captured_events)
    kinds = [json.loads(line) for line in captured_events]
    assert [row["kind"] for row in kinds] == ["asr_transcribe"]
    assert kinds[0]["outcome"] == "ok"
    assert kinds[0]["model"] == MODEL_ID
    assert kinds[0]["audio_bytes"] == len(wav)
    assert kinds[0]["elapsed_ms"] >= 0.0

    # 失败路径同样脱敏，并带上错误类别；底层异常若回显密钥也要被抹掉
    leaked = httpx.ConnectError(f"tunneling Bearer {DASHSCOPE_KEY_FOR_TESTS} failed")

    def failing(request):
        raise leaked

    stub2 = install_stub_http_layer(monkeypatch, failing)
    with pytest.raises(asr.ASRHardError) as info:
        asr._transcribe(wav)
    assert stub2.count == 1, "传输层异常同样不得自动重试"
    assert info.value.__cause__ is leaked
    assert DASHSCOPE_KEY_FOR_TESTS not in str(info.value)
    assert "[已脱敏]" in str(info.value)
    assert stub.count == 1                       # 第一次成功的那一发没有被重放
    failures = [json.loads(line) for line in captured_events][-3:]
    text = json.dumps(failures, ensure_ascii=False)
    assert DASHSCOPE_KEY_FOR_TESTS not in text
    assert needle not in text
    assert "transport" in text


def test_T19_事件字段可定位_请求标识与耗时与类别(monkeypatch, captured_events):
    arm_cloud_state(monkeypatch)
    install_stub_http_layer(monkeypatch, lambda request: httpx.Response(500, json={
        "code": "Throttling.RateQuota"}))
    with pytest.raises(asr.ASRHardError):
        asr._transcribe(AUDIO_FIXTURE)
    row = json.loads(captured_events[-1])
    assert row["kind"] == "asr_transcribe"
    assert row["outcome"] == "error"
    assert row["category"] == "service_error"
    assert isinstance(row["request_id"], str) and len(row["request_id"]) == 16
    assert isinstance(row["elapsed_ms"], (int, float))
    assert row["endpoint"] == _VALID_ENDPOINT
    assert set(row) <= {"kind", "ts", "session_id", "request_id", "outcome",
                        "elapsed_ms", "category", "detail", "model", "endpoint",
                        "audio_bytes", "http_status", "error_class"}
    assert "api_key" not in json.dumps(row)


# ---------------------------------------------------------------------------
# 整链路（P2-01 录音管线 + P2-02 真实适配）：停止流 → 真实 _transcribe → 上传的就是
# 状态机产出的那段 16k/mono/PCM16 WAV。两侧不再靠假钩子缝合，而是同一条代码路径。
# ---------------------------------------------------------------------------


def test_整链路录音结束后由真实适配上传并返回文本(monkeypatch):
    plan = [SPEECH] + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    monkeypatch.setattr(asr, "_open_input_stream", lambda: stream)
    arm_cloud_state(monkeypatch, cloud_total_timeout_s=25.0, vad=vad)
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response("分拣草莓"))

    assert asr.listen_and_transcribe() == "分拣草莓"

    assert stub.count == 1 and stream.stopped is True
    body = json.loads(stub.requests[0].content.decode("utf-8"))
    data = body["input"]["messages"][0]["content"][0]["input_audio"]["data"]
    uploaded = decode_wav(base64.b64decode(data.split(",", 1)[1]))
    assert uploaded["channels"] == 1
    assert uploaded["sampwidth"] == 2
    assert uploaded["framerate"] == TARGET_SAMPLE_RATE
    assert uploaded["samples"][:VAD_FRAME_SAMPLES] == [5000] * VAD_FRAME_SAMPLES
    assert uploaded["samples"][VAD_FRAME_SAMPLES:] == [0] * (
        SILENCE_END_FRAMES_TRIGGER * VAD_FRAME_SAMPLES)
    assert uploaded["nframes"] == (1 + SILENCE_END_FRAMES_TRIGGER) * VAD_FRAME_SAMPLES
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 默认转写钩子的"未就绪"拒绝（P2-02 已接线 DashScope，本例随之改写，见日志 §7 DEC-P2-02-F）：
# 运行态缺云配置时必须明确拒绝，**一次 HTTP 都不发**、绝不伪造转写文本。
# ---------------------------------------------------------------------------


def test_默认转写钩子缺云配置即拒绝且零HTTP请求(monkeypatch):
    plan = [SPEECH] + [END_SILENCE] * SILENCE_END_FRAMES_TRIGGER
    timeline: list[str] = []
    stream, vad = make_identity_stream(plan, timeline=timeline)
    monkeypatch.setattr(asr, "_open_input_stream", lambda: stream)
    # P2-01 时代的运行态只有 vad（没有 model/endpoint/key）——正是"未就绪"形态。
    monkeypatch.setattr(asr, "_state", SimpleNamespace(vad=vad))
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response())

    # 不 monkeypatch _transcribe：使用模块默认体（P2-02 的 DashScope 适配）。
    with pytest.raises(asr.ASRHardError) as info:
        asr.listen_and_transcribe()

    message = str(info.value)
    assert type(info.value) is asr.ASRHardError
    assert "ASR 未初始化" in message or "配置不完整" in message
    assert stub.requests == []              # 拒绝发生在发起请求之前
    assert stream.stopped is True           # 进入云转写前已停止流（P2 §3 要点）
    assert "分拣草莓" not in message        # 不返回、也不回显任何"假成功"文本
