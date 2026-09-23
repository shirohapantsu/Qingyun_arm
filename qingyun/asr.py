"""ASR 录音管线——**P2-01 在原文件替换契约桩体，文件名与公开签名不得改变**（P1 §1/§3.6）。

契约声明（P1 §1、§2 文件责任表）：
    本文件曾是 P1 的**契约桩**；P2-01 在**同名同路径**上替换桩体为真实录音管线，
    始终保留 ``ASRHardError`` 类名与 ``init`` / ``listen_and_transcribe`` 的签名
    （异常类基类 ``RuntimeError`` 不动）。契约桩的阶段边界不因替换而失效：
    调用方（P1 ``main``/``run_session``）与 P2 实现共享同一公开面，不得重命名、
    移位或以别的名字重建。

本阶段（P2-01 录音管线 + P2-02 云适配）实现与未实现：
    * 已实现（P2-01）：设备抽象（注入式）→ 连续降混 + 带状态重采样到 16k mono → VAD
      起止状态机 → 30s 上限与静音终点 → WAV 编码 → 调用私有转写钩子；``init(cfg)`` 的
      真实语义（只允许一次、cfg 校验、拒绝占位模型名、读密钥、加载 VAD、枚举设备）。
    * 已实现（P2-02）：私有 ``_transcribe(wav_bytes) -> str`` 的 **DashScope 真实 REST
      适配**（httpx 同步 POST、§3.1 逐字请求/响应协议、transport 层 retries=0、禁自动
      redirect、分项超时 ≤ 剩余预算）与 §3.2 的**端到端 deadline**
      （``time.monotonic()`` 绝对截止 + ``signal.setitimer(ITIMER_REAL, remaining)`` 中断
      整次请求 + 不继承 OSError 的专用异常 + finally 取消计时器/恢复旧处理器/关闭未完成
      响应；非主线程或已有活动 ITIMER_REAL 一律拒绝）。故障分类记录**脱敏**日志后统一
      ``ASRHardError`` 并保留来源，不静默替换/降级模型（D12/D20）。
    * 仍未实现（他阶段）：LLM 侧适配属 P2-03；requirements 增补属 P2-04；真实云烟雾
      实测属 P5。

规范来源：
    docs/实施文档/P2_语音与云模型接入技术文档.md
        §1–§3（录音状态机表、边界优先级"先判静音终点"、采样链"连续降混 + 带状态
            重采样到 16k mono，禁止逐帧独立重采样"、设备生命周期"每次 listen 重启并
            排空旧数据、离开函数前停止流（含进入云 ASR 前）、PLAN 期间不采集"、
            30s 仅限录音）
        §2（import 时不加载模型/不开设备/不请求网络；init 只允许一次；init 前调业务
            函数或重复 init → HardError；日志/异常不含 key；D15 mock 不调本 init）
        §5 测试表（T01–T05/T04b/T11–T15 归本任务；T16+ 归 P2-02）
        §6 步骤 1（先录音状态机与 WAV，再 DashScope 适配）
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md §3.6（冻结的契约面逐字）、
        §3.1 末段（模板占位模型名不能用于真实请求，P2 init 拒绝占位值）
    docs/实施文档/README.md 决策 D12/D13/D15/D20

import 零副作用（P2-T11 与常驻契约测试 test_module_stubs A 组）：
    模块顶层**只** import ``__future__``/``typing``（``AppConfig`` 走 TYPE_CHECKING
    延迟求值）。所有重依赖——sounddevice（采集）、onnxruntime/torch/silero（VAD）、
    configs.app_config（load_secrets）、qingyun.runlog（终端日志）、以及标准库
    wave/io/array——一律延迟到 ``init`` 或私有工厂内部，且经 ``__import__`` 动态载入，
    因此 ``import qingyun.asr`` 不拉起音频/云/视觉/配置层，也不触发任何设备或网络。
    （这是为同时满足"顶层 import 白名单 = {__future__, typing}"这条常驻 AST 契约而做的
    刻意选择，见开发日志 §7 DEC-P2-01-A。）

可注入边界（私有、公开签名不变；形态与协议见开发日志 §2）：
    * 设备层 ``_open_input_stream()``：返回阻塞式按块读取的输入流对象
      （``.samplerate`` / ``.channels`` / ``.drain()`` / ``.read()->(frames,channels)
      | None`` / ``.close()``）。默认工厂 lazy import sounddevice。每次 listen 开始
      新建并 drain（重启 + 排空旧数据），离开函数前停止流。
    * VAD 层 ``_load_vad()``：返回 16k/``.frame_size``(=512) 帧接口对象
      （``.probability(frame)->float``，可选 ``.reset()``）。默认 loader 经官方
      silero 入口优先 onnxruntime、备选 torch，阈值按官方示例默认 0.5 不臆造；
      每段起点调用 ``.reset()`` 复位，段间不积压。
    * 设备枚举 ``_enumerate_input_devices()``：init 期能力检查（默认 lazy sounddevice）。
    * 转写层 ``_transcribe(wav_bytes)->str``：私有云适配（P2-02 已接线 DashScope）。
      其 HTTP 客户端由工厂 ``_new_http_client(timeout_s=…)`` 提供，测试以**本地 stub
      transport 的 Client** 接管该名字（不 monkeypatch socket，P2 §5 末段）；时钟与
      计时器原语同样是私有名字（``_monotonic`` / ``signal.setitimer``），分别支撑
      T16 假时钟与 T17 真实墙钟（详见 P2-02 开发日志 §2）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — 仅用于类型检查，运行时不导入配置层
    from configs.app_config import AppConfig

__all__ = ["ASRHardError", "init", "listen_and_transcribe"]

# ---------------------------------------------------------------------------
# 采样链 / 状态机常量（§3；1000ms / 200ms / 30s 均以 16k 音频样本时长语义换算）
# ---------------------------------------------------------------------------

# 上传与 VAD 工作采样率：16 kHz 单声道 PCM16（§3 末段）。
TARGET_SAMPLE_RATE = 16000
# silero 官方推荐帧长（16 kHz 下 512 采样）；阈值取官方示例默认 0.5，不臆调。
VAD_FRAME_SAMPLES = 512
VAD_SPEECH_THRESHOLD = 0.5
# 连续静音达到该样本数即结束本段（1000ms @ 16k = 16000 样本）。
SILENCE_END_SAMPLES = 16000
# 触发语音时纳入段缓冲的前置音频长度（约 200ms @ 16k = 3200 样本）。
PRE_ROLL_SAMPLES = 3200
# 单次录音上限（30s @ 16k = 480000 样本，从触发起表；仅限录音，不含云转写）。
MAX_RECORD_SAMPLES = 480000

# 模板占位模型名（P1 §3.1 末段：模板占位模型名不能用于真实请求，init 拒绝）。
# 以 app.mock.json 的字面占位 "mock" 为核心，附带常见模板占位写法（大小写不敏感、
# 去空白后比对）；真实模型 "qwen-audio-3.1-asr-flash" 不在其中。
_PLACEHOLDER_MODEL_NAMES = frozenset(
    {
        "mock",
        "mock-model",
        "mock_model",
        "placeholder",
        "changeme",
        "change_me",
        "replace_me",
        "replace-me",
        "your-model",
        "your_model",
        "todo",
        "tbd",
        "none",
        "null",
        "unknown",
    }
)

# secrets 键名与固定路径在运行期经 __import__ 从 configs.app_config 取，
# 模块级不缓存（避免把配置层名字泄进模块全局命名空间）。
_DASHSCOPE_KEY_NAME = "dashscope_api_key"

# ---------------------------------------------------------------------------
# DashScope 请求协议固定字面量（P2 §3.1 逐字；模型名与 endpoint **不**硬编码，
# 取 init 校验后存于 _state 的配置值——换地域/换域名只改 app.json，不改代码）
# ---------------------------------------------------------------------------

# 音频以 data URI 内联上传（§3.1：input_audio.data = "data:audio/wav;base64,<音频>"）。
AUDIO_DATA_URI_PREFIX = "data:audio/wav;base64,"
# parameters 逐字：{"format":"wav","sample_rate":"16000"}（sample_rate 是**字符串**）。
AUDIO_FORMAT = "wav"
AUDIO_SAMPLE_RATE = str(TARGET_SAMPLE_RATE)
# 关闭 DashScope 的服务端 SSE：本适配是同步一次性请求。
SSE_HEADER_NAME = "X-DashScope-SSE"
SSE_HEADER_VALUE = "disable"
CONTENT_TYPE_HEADER_VALUE = "application/json"

# deadline 取消计时器后，用来吸收"已送达但尚未派发"的在途 SIGALRM 的让出次数
# （见 §7 DEC-P2-02-C：恢复旧处理器前必须先把在途信号消费掉）。
_DISARM_GRACE_TICKS = 8
# 异常/日志里对"外部异常文本"的截断长度（配合脱敏，避免把响应原文刷进日志）。
_DETAIL_MAX_CHARS = 200


class ASRHardError(RuntimeError):
    """ASR 侧硬故障（P1 §3.6 冻结的契约面；P2 §3 表末行"设备/重采样/VAD/云请求硬故障"）。

    基类固定 ``RuntimeError``：顶层映射（P1 §5）按类型分派到
    ``terminate("ASR_HARD_ERROR")``。替换桩体后它承载真实硬故障：未初始化、重复
    init、cfg/密钥校验失败、设备/重采样/VAD 故障、云转写的网络/鉴权/限流/服务/
    协议故障与**端到端 deadline 到期**（P2 §3.1/§3.2）。
    """


# ---------------------------------------------------------------------------
# 模块级运行态（单线程同步；import 时只声明、不加载任何重资源）
# ---------------------------------------------------------------------------


class _AsrState:
    """init 成功后常驻的运行态：cfg 派生设置 + VAD 实例。

    设备句柄不常驻于此——每次 listen 重启输入流（§3 要点）。VAD 在 init 加载一次。
    密钥仅在内存保留（不写日志/异常）。
    """

    def __init__(self, *, model: str, endpoint: str, cloud_total_timeout_s: float,
                 api_key: str, vad) -> None:
        self.model = model
        self.endpoint = endpoint
        self.cloud_total_timeout_s = cloud_total_timeout_s
        self.api_key = api_key  # 内存保留；任何日志/异常都不得回显其值（D12）
        self.vad = vad


_state: "_AsrState | None" = None


# ---------------------------------------------------------------------------
# 采样链：连续降混 + 带状态线性重采样（禁止逐帧独立重采样）
# ---------------------------------------------------------------------------


class _AudioConverter:
    """跨块连续的降混 + 带状态线性插值重采样器。

    输入：交织 int16 帧（长度可为任意块长）+ 声道数；输出：目标采样率单声道 float
    样本（保留 int16 量级）。降混对每帧多声道求均值；重采样以绝对读位置 ``pos`` 递增
    ``in_rate/out_rate``，对 ``floor(pos)`` 与 ``floor(pos)+1`` 线性插值，并保留尚未
    取到的边界样本在内部缓冲——因此逐块喂入与整段一次性喂入产生**完全一致**的前缀
    （T05）。声道分组不满一组时把余量留到下一块，块间不跳变。
    """

    def __init__(self, in_rate: float, out_rate: float) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ASRHardError(
                f"重采样率非法：in_rate={in_rate} out_rate={out_rate}（应为正数）"
            )
        self.in_rate = float(in_rate)
        self.out_rate = float(out_rate)
        self.ratio = self.in_rate / self.out_rate
        self._chan_buf: list = []      # 交织余量（不足一帧的尾部样本）
        self._mono: list = []          # 已降混但尚未被插值消费的样本
        self._mono_start = 0           # _mono[0] 的绝对下标
        self._pos = 0.0                # 绝对读位置（mono 域）

    def _at(self, abs_idx: int):
        return self._mono[abs_idx - self._mono_start]

    def process(self, frames, channels: int) -> list:
        """喂入一块交织 int16 帧，返回该块内**可确定**的目标采样率单声道样本。"""
        if channels < 1:
            raise ASRHardError(f"设备声道数非法：channels={channels}")
        # 同率单声道：降混与重采样都是恒等映射——逐样本原样透传（合法的精确快速路径，
        # 无相位残留、无边界误差）。异率或多声道走下面的带状态线性插值。
        if channels == 1 and self.ratio == 1.0:
            return list(frames)
        data = self._chan_buf + list(frames)
        n_full = (len(data) // channels) * channels
        self._chan_buf = data[n_full:]
        if channels == 1:
            mono_new = data[:n_full]
        else:
            mono_new = [
                sum(data[i:i + channels]) / channels for i in range(0, n_full, channels)
            ]
        self._mono.extend(mono_new)

        out: list = []
        end_abs = self._mono_start + len(self._mono)
        while True:
            idx = int(self._pos)
            if idx + 1 >= end_abs:  # 需要 floor(pos)+1，尚未到达
                break
            frac = self._pos - idx
            a = self._at(idx)
            b = self._at(idx + 1)
            out.append((1.0 - frac) * a + frac * b)
            self._pos += self.ratio

        keep_from = int(self._pos) - self._mono_start
        if keep_from < 0:
            keep_from = 0
        # 至少保留末尾一样本，令 _mono[0] 的绝对下标始终 = _mono_start：
        # 否则当整块被丢弃时（读位置跨过了缓冲末尾）下一块的绝对对齐会错位一格。
        max_keep = len(self._mono) - 1
        if keep_from > max_keep:
            keep_from = max_keep
        if keep_from > 0:
            del self._mono[:keep_from]
            self._mono_start += keep_from
        return out

    def reset(self) -> None:
        """丢弃残留状态（丢弃段后由调用方新建实例；此处保留同形接口便于测试）。"""
        self._chan_buf = []
        self._mono = []
        self._mono_start = 0
        self._pos = 0.0


def _reference_downmix_resample(frames, channels: int, in_rate: float,
                                out_rate: float) -> list:
    """整段一次性降混 + 线性重采样的**参考实现**（T05 用它与流式实现比对）。

    与 :class:`_AudioConverter` 使用同一绝对位置序列与同一插值式，故二者逐样本相等
    （仅在末尾同样停止于 ``floor(pos)+1`` 越界处）。
    """
    if channels < 1:
        raise ASRHardError(f"参考实现声道数非法：channels={channels}")
    if channels == 1 and float(in_rate) == float(out_rate):
        # 同率单声道恒等映射：与 _AudioConverter 的精确快速路径一致。
        return list(frames)
    if channels == 1:
        mono = list(frames)
    else:
        n_full = (len(frames) // channels) * channels
        mono = [
            sum(frames[i:i + channels]) / channels for i in range(0, n_full, channels)
        ]
    ratio = float(in_rate) / float(out_rate)
    out: list = []
    pos = 0.0
    while True:
        idx = int(pos)
        if idx + 1 >= len(mono):
            break
        frac = pos - idx
        out.append((1.0 - frac) * mono[idx] + frac * mono[idx + 1])
        pos += ratio
    return out


def _encode_wav_mono16(samples) -> bytes:
    """把 16k 单声道样本编成 PCM16 WAV 字节（标准库 wave/BytesIO/array，§3）。

    采样按 int16 截断四舍五入；头字段固定 nchannels=1、sampwidth=2、framerate=16000。
    """
    wave = __import__("wave")
    io = __import__("io")
    array = __import__("array")
    pcm = array.array("h")
    for value in samples:
        v = int(value + (0.5 if value >= 0 else -0.5))  # 四舍五入到整数
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        pcm.append(v)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 可注入边界：默认实现（惰性 __import__；本机这些包未装，测试全程注入替身）
# ---------------------------------------------------------------------------


def _open_input_stream():
    """默认输入流工厂（lazy sounddevice）。

    返回对象满足设备层协议：``samplerate`` / ``channels`` / ``drain()`` /
    ``read()->(frames, channels)|None`` / ``close()``。每次 listen 调用本工厂即为
    "重启输入流"，随后由调用方 ``drain()`` 排空旧数据。
    """
    try:
        sounddevice = __import__("sounddevice")
    except Exception as exc:  # ImportError 或 PortAudio 缺失
        raise ASRHardError(f"打开输入设备失败（sounddevice 不可用）：{exc}") from exc
    try:
        raw = sounddevice.RawInputStream(
            samplerate=TARGET_SAMPLE_RATE, channels=None, dtype="int16",
            blocksize=VAD_FRAME_SAMPLES,
        )
        raw.start()
    except Exception as exc:
        raise ASRHardError(f"启动输入流失败：{exc}") from exc
    return _SoundDeviceInput(raw)


class _SoundDeviceInput:
    """sounddevice.RawInputStream 的设备层薄封装（不在测试路径执行）。"""

    def __init__(self, raw) -> None:
        self._raw = raw
        self.samplerate = float(raw.samplerate)
        self.channels = int(raw.channels) or 1

    def drain(self) -> None:
        # 排空旧数据：丢弃进入本函数前积压的缓冲（§3 要点）。
        try:
            while self._raw.read(VAD_FRAME_SAMPLES)[1] == False:  # noqa: E712
                pass
        except Exception:
            pass

    def read(self):
        try:
            data, overflowed = self._raw.read(VAD_FRAME_SAMPLES)
        except Exception as exc:
            raise ASRHardError(f"读取输入流失败：{exc}") from exc
        return (list(data), self.channels)

    def close(self) -> None:
        try:
            self._raw.stop()
            self._raw.close()
        except Exception:
            pass


def _enumerate_input_devices():
    """init 期输入设备能力检查（lazy sounddevice）。返回设备标识列表。"""
    try:
        sounddevice = __import__("sounddevice")
    except Exception as exc:
        raise ASRHardError(f"枚举输入设备失败（sounddevice 不可用）：{exc}") from exc
    try:
        devices = sounddevice.query_devices()
        inputs = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]
    except Exception as exc:
        raise ASRHardError(f"查询输入设备失败：{exc}") from exc
    if not inputs:
        raise ASRHardError("未发现任何可用输入设备（麦克风缺失或被占用）")
    return [d.get("name", "<unnamed>") for d in inputs]


def _load_vad():
    """默认 VAD 加载器：优先 onnxruntime、备选 torch，经官方 silero-vad 入口封装。

    帧长 512、阈值官方示例默认 0.5（不臆调）。运行时选择与张量管道交给官方
    ``silero_vad.load_silero_vad(onnx=...)``——本模块不自行臆造 ONNX 状态张量形状。
    本机依赖未装时导入失败 → 抛 ASRHardError（VAD 硬故障）；pytest 从不执行本默认体
    （注入替身）。权重/接线在目标机 P5 烟雾实测后锁定（§3 末段）。
    """
    try:
        silero_vad = __import__("silero_vad", fromlist=["load_silero_vad"])
    except Exception as exc:
        raise ASRHardError(
            f"VAD 运行时缺失（silero_vad/onnxruntime/torch 不可用）：{exc}"
        ) from exc
    try:
        model = silero_vad.load_silero_vad(onnx=True)   # onnxruntime 优先
    except Exception:
        model = silero_vad.load_silero_vad(onnx=False)  # torch 备选
    return _SileroVad(model, silero_vad)


class _SileroVad:
    """官方 silero VAD 的 16k/512 帧概率封装（目标机接线；pytest 用替身，不执行本类）。

    协议：``frame_size`` / ``probability(frame)->float`` / ``reset()``。逐帧概率与
    跨帧隐状态由官方 model 维护；每段结束/丢弃经 ``reset()`` 复位，段间不积压。
    """

    frame_size = VAD_FRAME_SAMPLES

    def __init__(self, model, silero_vad) -> None:
        self._model = model
        self._silero = silero_vad

    def reset(self) -> None:
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()

    def probability(self, frame):  # pragma: no cover - 目标机路径（本机无 torch）
        torch = __import__("torch")
        window = torch.FloatTensor([float(s) for s in frame])
        with torch.no_grad():
            prob = self._model(window, TARGET_SAMPLE_RATE)
        return float(prob.view(-1)[0].item())


# ---------------------------------------------------------------------------
# 端到端 deadline（P2 §3.2 固定实现边界）
#
# 为什么必须有它：HTTP 客户端的分项超时（connect/read/write/pool）里，read 只约束
# "等待下一块数据"——一个每 50ms 挤一点数据的慢服务可以**永远不触发**任何分项超时，
# 而整次调用早已超过 cloud_total_timeout_s（R05）。因此对"连接 + DNS + TLS + 上传 +
# 完整响应读取 + 解析"施加一个可中止的**墙钟**上限：time.monotonic() 记绝对截止点，
# signal.setitimer(ITIMER_REAL, remaining) 到点用 SIGALRM 打断阻塞中的系统调用。
# ---------------------------------------------------------------------------


class _AsrDeadlineExceeded(BaseException):
    """端到端 deadline 到期的专用异常（P2 §3.2"处理器抛专用异常"）。

    刻意**不继承 ``OSError``**，甚至不继承 ``Exception``，两个理由都是实现性的：
        1. PEP 475 之后 CPython 的系统调用在 EINTR 时会自动重试，只有信号处理器
           "设置了异常"才中断重试；若处理器抛的是 ``OSError(EINTR)`` 一类，语义上
           与"底层 socket 自己报的可重试错误"混在一起，中断有被吞掉的风险。
        2. httpx/httpcore 内部以 ``except Exception`` 把底层异常映射成自己的
           ``TransportError`` 族。继承 ``BaseException`` 保证"到期"原样穿过去，
           不会在最后一刻被改写成"传输错误"而丢掉类别。
    它只在 deadline 上下文与本模块内部产生，**绝不**外泄给调用方：``_transcribe``
    统一把它转成 ``ASRHardError``（公开异常面只有 P1 冻结的那一个）。
    """


class _TranscribeFailure(Exception):
    """内部分类载体：把"哪一类故障"从协议解析处带到 ``_transcribe`` 的统一出口。

    公开面仍然只有 ``ASRHardError``；本类只用于在私有函数之间传递**错误类别**
    （``auth`` / ``rate_limited`` / ``service_error`` / ``bad_json`` / …），
    使日志分类不被字符串匹配猜测。
    """

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def _monotonic() -> float:
    """单调时钟源（**注入点**：T16 用假时钟推满 cloud_total_timeout_s，不真实等待）。"""
    return __import__("time").monotonic()


def _yield_to_signals() -> None:
    """让出一次解释器/系统调用边界，使**已送达但尚未派发**的 SIGALRM 得到执行。"""
    __import__("time").sleep(0)


def _deadline_handler(signum, frame) -> None:
    """ITIMER_REAL 到点处理器：抛专用异常，把阻塞中的整次请求打断（§3.2）。"""
    raise _AsrDeadlineExceeded(
        f"ASR 云请求端到端 deadline 到点（SIGALRM signum={signum}）"
    )


def _drain_handler(signum, frame) -> None:
    """取消计时器后的"在途信号吸收"处理器：只消费迟到 SIGALRM，不做任何事。

    没有这一步会留下一个真实风险：``setitimer(0)`` 只能阻止**未来**的投递，已经
    进入 pending 的信号会在下一个字节码边界执行"当时的处理器"。若此时旧处理器已
    被恢复成 ``SIG_DFL``，一个迟到的 SIGALRM 会**终止进程**。
    """
    return None


def _reject_unsupported_call_environment() -> None:
    """进入 deadline 前的环境检查（§3.2 末段）：非主线程 / 已有活动 ITIMER_REAL → 拒绝。

    活动计时器的检测用 ``signal.getitimer(ITIMER_REAL)``（**只读**）而不是
    ``setitimer`` 试探——后者会顺手取消别人的定时器，正是要避免的事。任一字段 > 0
    即认为有活动计时器：本模块不覆盖其他组件的定时器，直接拒绝本次云请求。
    """
    threading = __import__("threading")
    if threading.current_thread() is not threading.main_thread():
        raise ASRHardError(
            "ASR 云转写拒绝执行：不支持的调用环境（端到端 deadline 依赖主线程信号派发，"
            "本期运行环境限定 Linux 主线程同步调用；P2 §3.2）"
        )
    signal = __import__("signal")
    try:
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    except Exception as exc:  # 平台不支持（无 setitimer 的环境）等
        raise ASRHardError(
            "ASR 云转写拒绝执行：不支持的调用环境（无法探测 ITIMER_REAL："
            f"{type(exc).__name__}）"
        ) from exc
    if remaining > 0.0 or interval > 0.0:
        raise ASRHardError(
            "ASR 云转写拒绝执行：不支持的调用环境（已有活动的 ITIMER_REAL"
            f"（remaining={remaining} interval={interval}）；本模块不覆盖其他组件的"
            "定时器，P2 §3.2）"
        )


class _CloudDeadline:
    """一次云请求的端到端 deadline 上下文（P2 §3.2；每次请求**单独起表**）。

    生命周期：
        * ``__enter__``：环境检查 → 以 ``_monotonic()`` 记**绝对**截止点 → 安装
          ``_deadline_handler`` → ``setitimer(ITIMER_REAL, remaining)``（一次性）。
          装表失败会立刻回滚处理器。
        * 期间：``item_timeout_s()`` 给出分项 HTTP 超时（= 剩余预算，**不超过**预算；
          预算已耗尽时直接判到期，绝不带着 0 预算再发请求）。
        * ``__exit__``：**总是**（含 Ctrl-C 路径）取消计时器 → 安装吸收处理器并让出
          若干次以消费在途 SIGALRM → 恢复旧处理器；随后若属正常返回而绝对截止点已到，
          仍判到期——这是预算的最后一道校验，**中断能力由 ITIMER 提供**，本模块的
          正确性不依赖该事后判断（T17 用真实慢响应服务证明前者有效）。
    """

    def __init__(self, total_s: float, *, request_id: str) -> None:
        self.total_s = float(total_s)
        self.request_id = request_id
        self.deadline_at = 0.0
        self._previous_handler = None
        self._armed = False

    def remaining_s(self) -> float:
        return self.deadline_at - _monotonic()

    def item_timeout_s(self) -> float:
        """分项 HTTP 超时上限：取剩余预算；剩余 ≤ 0 即判端到端到期。"""
        remaining = self.remaining_s()
        if remaining <= 0.0:
            raise _AsrDeadlineExceeded(
                f"ASR 云请求总时限 {self.total_s}s 的预算已耗尽，拒绝再发起请求"
            )
        return remaining

    def __enter__(self) -> "_CloudDeadline":
        _reject_unsupported_call_environment()
        signal = __import__("signal")
        self.deadline_at = _monotonic() + self.total_s
        budget = self.remaining_s()
        if budget <= 0.0:
            # 还没装表就没预算：直接判到期（也避免把负数喂给 setitimer）。
            raise _AsrDeadlineExceeded(
                f"ASR 云请求总时限 {self.total_s}s 无剩余预算，拒绝发起请求"
            )
        self._previous_handler = signal.signal(signal.SIGALRM, _deadline_handler)
        try:
            pending, interval = signal.setitimer(signal.ITIMER_REAL, budget)
        except BaseException:
            signal.signal(signal.SIGALRM, self._previous_handler)  # 装表失败 → 回滚处理器
            raise
        if pending > 0.0 or interval > 0.0:
            # 进入前已探测为零，这里非零说明检查被并发绕过：立刻撤销并拒绝。
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            raise ASRHardError(
                "ASR 云转写拒绝执行：不支持的调用环境（装表瞬间仍看到活动 ITIMER_REAL）"
            )
        self._armed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._armed:
            self._disarm()
        if exc_type is None and self.remaining_s() <= 0.0:
            raise _AsrDeadlineExceeded(
                f"ASR 云请求整次调用超出总时限 {self.total_s}s（绝对截止点已越过）"
            )
        return False

    def _disarm(self) -> None:
        """取消计时器、消费在途信号、恢复旧处理器（顺序即安全性，见 DEC-P2-02-C）。"""
        signal = __import__("signal")
        self._armed = False
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)          # 1) 停掉信号源
        finally:
            try:
                signal.signal(signal.SIGALRM, _drain_handler)  # 2) 换吸收处理器
                for _ in range(_DISARM_GRACE_TICKS):
                    _yield_to_signals()                      # 3) 让出，派发在途信号
            finally:
                signal.signal(signal.SIGALRM, self._previous_handler)  # 4) 交还原主


# ---------------------------------------------------------------------------
# DashScope 私有适配（P2 §3.1）
# ---------------------------------------------------------------------------


def _import_httpx():
    """惰性载入 httpx（import 零副作用；缺依赖时明确硬故障，不静默降级为别的库）。"""
    try:
        return __import__("httpx")
    except Exception as exc:
        raise ASRHardError(
            f"DashScope 适配需要 httpx（P2 §3.1）；当前不可用：{type(exc).__name__}"
        ) from exc


def _new_http_client(*, timeout_s: float):
    """默认 HTTP 客户端工厂（**注入点**：测试注入本地 stub transport 的同形 Client）。

    §3.2 的三条硬要求全部在这里显式写出，不依赖库默认值：
        * ``HTTPTransport(retries=0, http2=False)``——transport 层**零重试**、不走 HTTP/2；
        * ``follow_redirects=False``——禁自动 redirect（3xx 由本适配判为错误）；
        * ``connect/read/write/pool`` 四个分项超时**全部等于**调用方传入的剩余预算
          （分项超时只是"别单次阻塞太久"，整次墙钟上限由 :class:`_CloudDeadline` 保证）。
    """
    httpx = _import_httpx()
    transport = httpx.HTTPTransport(retries=0, http2=False)
    timeout = httpx.Timeout(connect=timeout_s, read=timeout_s, write=timeout_s,
                            pool=timeout_s)
    return httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)


def _new_request_id() -> str:
    """日志关联用的请求标识（随机、不含密钥/音频内容；每次云请求一个）。"""
    uuid = __import__("uuid")
    return uuid.uuid4().hex[:16]


def _redact(text, secret: str) -> str:
    """把外部异常/响应文本里的密钥形态一律替换为占位，再截断（§3.1 脱敏）。

    只针对**值**做替换：Authorization 头本身不进日志（我们从不记录 headers），
    但底层库的异常字符串理论上可能回显上下文，故对 key 与 ``Bearer <key>`` 两种
    形态都做兜底替换。
    """
    if not isinstance(text, str):
        text = str(text)
    for needle in (f"Bearer {secret}", secret):
        if needle:
            text = text.replace(needle, "[已脱敏]")
    if len(text) > _DETAIL_MAX_CHARS:
        text = text[:_DETAIL_MAX_CHARS] + "…(截断)"
    return text


def _audio_data_uri(wav_bytes: bytes) -> str:
    """16k mono PCM16 WAV → ``data:audio/wav;base64,<音频>``（§3.1 逐字前缀）。"""
    base64 = __import__("base64")
    return AUDIO_DATA_URI_PREFIX + base64.b64encode(bytes(wav_bytes)).decode("ascii")


def _build_request_payload(model: str, data_uri: str) -> dict:
    """按 §3.1 组装 body：**一条** user 消息、只含本次音频，不携带历史音频/对话。"""
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data_uri}}
                    ],
                }
            ]
        },
        "parameters": {"format": AUDIO_FORMAT, "sample_rate": AUDIO_SAMPLE_RATE},
    }


def _build_request_headers(api_key: str) -> dict:
    """Bearer 密钥 + application/json + ``X-DashScope-SSE: disable``（§3.1）。"""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": CONTENT_TYPE_HEADER_VALUE,
        SSE_HEADER_NAME: SSE_HEADER_VALUE,
    }


def _http_status_category(status: int) -> str:
    """HTTP 状态 → 错误类别（鉴权/限流/重定向/请求被拒/服务故障）。"""
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if 300 <= status < 400:
        return "redirect_not_followed"
    if 400 <= status < 500:
        return "request_rejected"
    if status >= 500:
        return "service_error"
    return "unexpected_status"


def _provider_error_code(response) -> "str | None":
    """非 2xx 时尽力取 DashScope 的 ``code`` 字段用于分类日志（取不到就 None）。

    只用 ``code``（形如 ``InvalidApiKey`` / ``Throttling.RateQuota``）；服务侧
    ``message`` 可能含请求上下文，不入日志。读取本身仍在 deadline 覆盖之内。
    """
    json_module = __import__("json")
    try:
        raw = response.read()
        document = json_module.loads(raw.decode("utf-8", "replace"))
    except _AsrDeadlineExceeded:
        raise
    except Exception:
        return None
    if not isinstance(document, dict):
        return None
    code = document.get("code")
    if not isinstance(code, str) or not code or len(code) > 64:
        return None
    return code


def _extract_transcript_text(body: bytes) -> str:
    """成功路径的唯一出口：读 ``output.text``（**允许空串**）。

    坏 JSON / 顶层非对象 / 缺 output / 缺 text / text 非字符串（含"多余嵌套结构"）
    一律 :class:`_TranscribeFailure`，由 ``_transcribe`` 归类后抛 ASRHardError（§3.1）。
    """
    json_module = __import__("json")
    try:
        document = json_module.loads(body.decode("utf-8"))
    except _AsrDeadlineExceeded:
        raise
    except UnicodeError as exc:
        raise _TranscribeFailure("bad_json", f"响应不是合法 UTF-8：{exc}") from exc
    except ValueError as exc:
        raise _TranscribeFailure(
            "bad_json", f"响应不是合法 JSON（{type(exc).__name__}）"
        ) from exc
    if not isinstance(document, dict):
        raise _TranscribeFailure(
            "bad_payload", f"响应顶层不是 JSON 对象（{type(document).__name__}）"
        )
    output = document.get("output")
    if output is None:
        raise _TranscribeFailure("bad_payload", "响应缺 output 字段")
    if not isinstance(output, dict):
        raise _TranscribeFailure(
            "bad_payload", f"output 不是 JSON 对象（{type(output).__name__}）"
        )
    if "text" not in output:
        raise _TranscribeFailure("bad_payload", "响应缺 output.text 字段")
    transcript = output["text"]
    if not isinstance(transcript, str):
        raise _TranscribeFailure(
            "bad_payload", f"output.text 非字符串（{type(transcript).__name__}）"
        )
    return transcript


def _close_quietly(closer) -> None:
    """关闭客户端/响应：清理失败不外抛（清理不得盖掉真实故障，也不得盖掉 Ctrl-C）。"""
    try:
        closer.close()
    except Exception:
        pass


def _log_transcribe_event(request_id: str, started_at: float, outcome: str, **fields) -> None:
    """写一条**脱敏** JSONL 事件（§3.1：请求标识、耗时、错误类别）。

    永不写入 api_key / Authorization / 音频 base64 / 响应原文。日志自身的失败
    （未 init、字段非法等）在这里吞掉——它不得掩盖转写故障。
    """
    try:
        runlog = __import__("qingyun.runlog", fromlist=["event"])
    except Exception:
        return
    record = {
        "request_id": request_id,
        "outcome": outcome,
        "elapsed_ms": round(max(0.0, (_monotonic() - started_at) * 1000.0), 3),
    }
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    try:
        runlog.event("asr_transcribe", **record)
    except Exception:
        pass


def _transcribe_failure(request_id: str, started_at: float, category: str, detail: str,
                        *, api_key: str = "", **extra) -> ASRHardError:
    """统一出口：记一条脱敏错误事件，并**返回**（不抛）待抛出的 ASRHardError。

    返回值形式让调用点写成 ``raise _transcribe_failure(...) from source``，从而
    ``__cause__`` 保留来源异常（§3"保留来源与原因"），消息里只出现类别与脱敏细节。
    """
    safe_detail = _redact(detail, api_key)
    _log_transcribe_event(request_id, started_at, "error", category=category,
                          detail=safe_detail, **extra)
    elapsed_s = max(0.0, _monotonic() - started_at)
    return ASRHardError(
        f"DashScope 语音转写失败（{category}）：{safe_detail}"
        f"（请求 {request_id}，已耗时 {elapsed_s:.3f}s；不重试、不降级、不跨地域 fallback）"
    )


def _perform_transcribe_request(deadline: "_CloudDeadline", state, payload: dict,
                                headers: dict) -> bytes:
    """发起**一次** POST 并返回 2xx 响应体字节（在 deadline 覆盖之内）。"""
    client = _new_http_client(timeout_s=deadline.item_timeout_s())
    try:
        with client.stream(
            "POST",
            state.endpoint,
            json=payload,
            headers=headers,
            timeout=deadline.item_timeout_s(),
            follow_redirects=False,
        ) as response:
            status = int(response.status_code)
            if not 200 <= status < 300:
                provider_code = _provider_error_code(response)
                raise _TranscribeFailure(
                    _http_status_category(status),
                    f"HTTP {status} 非 2xx（禁自动 redirect，服务侧 code={provider_code}）"
                    if provider_code else f"HTTP {status} 非 2xx（禁自动 redirect）",
                )
            return response.read()        # 完整响应体读取（慢响应的中断点）
    finally:
        _close_quietly(client)            # 客户端/连接一定关闭（含未完成响应）


def _transcribe(wav_bytes: bytes) -> str:
    """DashScope 非实时 ASR 私有适配（P2 §3.1）+ 端到端 deadline（§3.2）。

    步骤：取 ``_state`` 的 model/endpoint/cloud_total_timeout_s/api_key → 组装 §3.1
    逐字请求 → 在 :class:`_CloudDeadline` 下发起**一次** httpx 同步 POST → 读
    ``output.text``（允许空串）。任何网络/鉴权/限流/服务/坏 JSON/缺字段/超时都归类
    记脱敏日志后抛 :class:`ASRHardError` 并保留来源；不自动重试、不跨地域 fallback、
    不替换模型（D12/D20）。
    """
    state = _state
    if state is None:
        raise ASRHardError(
            "ASR 未初始化：云转写需要 init(cfg) 校验后的 model/endpoint/密钥（P2-T12）"
        )
    missing = [
        name for name in ("model", "endpoint", "cloud_total_timeout_s", "api_key")
        if getattr(state, name, None) in (None, "")
    ]
    if missing:
        raise ASRHardError(
            f"ASR 云转写配置不完整：运行态缺 {'、'.join(missing)}（不猜测默认值、不降级）"
        )
    if not isinstance(wav_bytes, (bytes, bytearray)):
        raise ASRHardError(f"待上传音频必须是 bytes，实际 {type(wav_bytes).__name__}")
    audio = bytes(wav_bytes)
    if not audio:
        raise ASRHardError("待上传音频为空字节：无音频可转写")

    httpx = _import_httpx()
    api_key = state.api_key
    model = state.model
    total_s = float(state.cloud_total_timeout_s)
    request_id = _new_request_id()
    started_at = _monotonic()
    payload = _build_request_payload(model, _audio_data_uri(audio))
    headers = _build_request_headers(api_key)
    common = {"model": model, "endpoint": state.endpoint, "audio_bytes": len(audio)}

    try:
        with _CloudDeadline(total_s, request_id=request_id) as deadline:
            body = _perform_transcribe_request(deadline, state, payload, headers)
            transcript = _extract_transcript_text(body)
    except _AsrDeadlineExceeded as exc:
        # 端到端 deadline 到期（含"分项超时都不触发、但总时长已超"的慢响应，R05）。
        raise _transcribe_failure(
            request_id, started_at, "deadline_timeout",
            f"整次云请求超过 cloud_total_timeout_s={total_s}s：{type(exc).__name__}",
            api_key=api_key, model=model, endpoint=state.endpoint,
            audio_bytes=len(audio),
        ) from exc
    except _TranscribeFailure as exc:
        raise _transcribe_failure(request_id, started_at, exc.category, exc.detail,
                                  api_key=api_key, **common) from exc
    except httpx.TimeoutException as exc:
        # 分项超时：只约束"某一次等待过久"，与端到端到期分列两类便于运维定位。
        raise _transcribe_failure(
            request_id, started_at, "item_timeout",
            f"HTTP 分项超时（{type(exc).__name__}），分项预算 = 剩余总时限"
            f"（起算 {total_s}s）：{exc}",
            api_key=api_key, **common,
        ) from exc
    except httpx.HTTPError as exc:
        raise _transcribe_failure(
            request_id, started_at, "transport",
            f"传输层故障（{type(exc).__name__}），transport retries=0 未重试：{exc}",
            api_key=api_key, **common,
        ) from exc
    except ASRHardError as exc:
        # 环境/依赖拒绝（不支持的调用环境、缺 httpx）：消息已定性，只补一条事件。
        _log_transcribe_event(request_id, started_at, "error", category="refused",
                              detail=_redact(exc, api_key), **common)
        raise
    except (ValueError, TypeError, UnicodeError) as exc:
        raise _transcribe_failure(
            request_id, started_at, "local_error",
            f"本地序列化/解析故障（{type(exc).__name__}）",
            api_key=api_key, **common,
        ) from exc
    except OSError as exc:
        raise _transcribe_failure(
            request_id, started_at, "os_error",
            f"系统调用故障（{type(exc).__name__}）",
            api_key=api_key, **common,
        ) from exc
    except BaseException as exc:            # KeyboardInterrupt / SystemExit 等
        # 用户中断不是"云故障"：记录后**原样上抛**，交 P1 顶层生命周期处置，
        # 绝不转成 ASRHardError（那会把 Ctrl-C 洗白成一次可归因于服务的失败）。
        _log_transcribe_event(request_id, started_at, "aborted",
                              error_class=type(exc).__name__, **common)
        raise
    _log_transcribe_event(request_id, started_at, "ok", http_status="2xx", **common)
    return transcript



# ---------------------------------------------------------------------------
# 录音状态机（§3 表逐行 + 边界优先级）
# ---------------------------------------------------------------------------


def _emit_console(message: str) -> None:
    """写一行终端日志（runlog.console）；未 init 时（纯单测环境）降级为不写。"""
    try:
        runlog = __import__("qingyun.runlog", fromlist=["console"])
    except Exception:
        return
    try:
        runlog.console(message)
    except Exception:
        # 运行期终端日志失败不得改变"丢弃/返回空串"的结果；生产入口此时必已 init runlog。
        pass


def _capture(vad, stream) -> "bytes | None":
    """驱动 WAITING/TRIGGERED 状态机，返回待上传 WAV 字节或 None（无需上传）。

    ``stream.read()`` 返回 ``(frames, channels)`` 或 ``None``（注入源结束）。返回：
        * 静音终点命中 → 编码后的段 WAV 字节（调用方停止流后转写）。
        * 30s 到点未成终点 → 打一行终端日志、返回 None（丢弃、零上传）。
        * 源在 WAITING 结束或段未达边界即结束 → 返回 None（本轮无口令）。
    """
    frame_size = int(vad.frame_size)
    reset_vad = getattr(vad, "reset", None)
    if callable(reset_vad):
        reset_vad()  # 每次进入 = 一段的起点：复位 VAD 内部状态，段间不积压（§3 要点）
    history: list = []                 # 触发前的前置缓冲（最近 PRE_ROLL_SAMPLES 样本）
    out_buf: list = []                 # 尚未切满一帧的 16k 样本
    seg = None                         # 触发后的段样本缓冲（含前置缓冲）
    seg_samples = 0                    # 自触发起累计样本（30s 计时口径）
    silence_run = 0                    # 连续静音样本数
    converter = _AudioConverter(stream.samplerate, TARGET_SAMPLE_RATE)
    triggered = False

    while True:
        block = stream.read()
        if block is None:
            # 注入源结束：无口令（WAITING），或未达边界的残段（TRIGGERED）都不上传。
            return None
        frames, channels = block
        mono16k = converter.process(frames, int(channels))
        out_buf.extend(mono16k)

        while len(out_buf) >= frame_size:
            chunk = out_buf[:frame_size]
            del out_buf[:frame_size]
            speech = vad.probability(chunk) >= VAD_SPEECH_THRESHOLD

            if not triggered:
                if speech:
                    triggered = True
                    seg = list(history) + list(chunk)  # 纳入约 200ms 前置缓冲
                    seg_samples = len(chunk)
                    silence_run = 0
                else:
                    history.extend(chunk)
                    if len(history) > PRE_ROLL_SAMPLES:
                        del history[: len(history) - PRE_ROLL_SAMPLES]
                continue

            # TRIGGERED：语音或静音 <1s 连续积累
            seg.extend(chunk)
            seg_samples += len(chunk)
            if speech:
                silence_run = 0
            else:
                silence_run += len(chunk)

            # 边界优先级：先判静音终点（完整语音优先），再判 30s 到点。
            if silence_run >= SILENCE_END_SAMPLES:
                return _encode_wav_mono16(seg)
            if seg_samples >= MAX_RECORD_SAMPLES:
                _emit_console("ASR 录音达到 30s 上限，丢弃本段（未上传）")
                return None


def listen_and_transcribe() -> str:
    """阻塞式完成"等待口令 → 录音 → 转写"，返回转写文本（允许空串）。

    前置条件：``init`` 已成功，否则抛 ``ASRHardError``（P2-T12）。每次进入本函数
    重启输入流并排空旧数据，离开函数前停止流（**包括进入云转写前**）；PLAN 运行期间
    不采集。语义见 §3 状态机与本模块头 docstring。
    """
    if _state is None:
        raise ASRHardError(
            "ASR 未初始化：listen_and_transcribe() 前必须先成功调用 init(cfg)"
        )
    vad = _state.vad
    stream = None
    try:
        stream = _open_input_stream()  # 重启输入流
    except ASRHardError:
        raise
    except Exception as exc:
        raise ASRHardError(f"重启输入流失败：{exc}") from exc

    wav_bytes: "bytes | None"
    try:
        stream.drain()                 # 排空旧数据（先于任何读取）
        wav_bytes = _capture(vad, stream)
    finally:
        _stop_stream(stream)           # 停止流（含进入云转写前）

    if wav_bytes is None:
        return ""                      # 无口令 / 30s 丢弃 / 残段：本轮不上传

    text = _transcribe(wav_bytes)
    if not isinstance(text, str):
        raise ASRHardError(f"转写钩子返回非字符串：{type(text).__name__}")
    if not text.strip():               # ASR 返回空/纯空白（P2-T14）
        return ""
    return text


def _stop_stream(stream) -> None:
    """停止并关闭输入流；关闭异常不外抛（离开函数前的清理）。"""
    try:
        stream.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# init（真实语义）
# ---------------------------------------------------------------------------


def init(cfg: AppConfig) -> None:
    """加载 ASR 运行期资源：校验 cfg.asr、读密钥、枚举设备、加载 VAD（各为注入点）。

    契约（P2 §2/§3.1）：
        * 由 ``main`` 冷启动段调用**恰好一次**；重复调用抛 ``ASRHardError``。
        * 校验 ``cfg.asr``：拒绝占位模型名（空串/"mock" 等模板占位）；endpoint 必须
          为 https 完整 URL；``cloud_total_timeout_s`` 必须为正有限数。
        * 经 ``configs.app_config.load_secrets`` 读 ``dashscope_api_key``，缺失或
          全空白抛 ``ASRHardError``（P2-T13 的 asr 侧兜底）；key 只在内存保留，日志
          与异常不回显其值。
        * 加载 VAD（注入点 ``_load_vad``）与设备枚举（注入点
          ``_enumerate_input_devices``）；失败即抛，状态保持未初始化可重试。
        * 独立 mock 入口**不调用**本函数（D15）；本 init 语义不为 mock 分支修改。
    """
    global _state
    if _state is not None:
        raise ASRHardError(
            "ASR 已初始化：init(cfg) 只允许调用一次（重复调用属内部不变量破坏）"
        )

    model, endpoint, cloud_total_timeout_s = _validate_asr_settings(cfg)
    api_key = _read_dashscope_key()

    try:
        _enumerate_input_devices()
        vad = _load_vad()
    except ASRHardError:
        raise
    except Exception as exc:
        # VAD/设备硬故障：状态保持未初始化，允许修复后重试（不留下半成品）。
        raise ASRHardError(f"加载 ASR 运行资源失败：{exc}") from exc

    _state = _AsrState(
        model=model,
        endpoint=endpoint,
        cloud_total_timeout_s=cloud_total_timeout_s,
        api_key=api_key,
        vad=vad,
    )


def _validate_asr_settings(cfg):
    """从 cfg.asr 取三字段并逐项校验；结构缺失/非法一律抛 ASRHardError。"""
    try:
        asr_settings = cfg.asr
        model = asr_settings.model
        endpoint = asr_settings.endpoint
        cloud_total_timeout_s = asr_settings.cloud_total_timeout_s
    except AttributeError as exc:
        raise ASRHardError(
            "init(cfg) 失败：cfg.asr 结构不合法"
            f"（缺 model/endpoint/cloud_total_timeout_s）：{exc}"
        ) from exc

    if not isinstance(model, str) or not model.strip():
        raise ASRHardError("asr.model 为空或非法：真实请求不允许占位/空模型名（P1 §3.1）")
    if model.strip().lower() in _PLACEHOLDER_MODEL_NAMES:
        raise ASRHardError(
            f"asr.model 是模板占位值 {model!r}，不能用于真实请求（P1 §3.1：P2 init 拒绝占位值）"
        )
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ASRHardError("asr.endpoint 为空或非字符串：需要完整 https URL")
    ep = endpoint.strip()
    if not ep.startswith("https://") or len(ep) <= len("https://"):
        raise ASRHardError(f"asr.endpoint 必须是完整 https URL，实际 {endpoint!r}")

    if isinstance(cloud_total_timeout_s, bool) or not isinstance(
        cloud_total_timeout_s, (int, float)
    ):
        raise ASRHardError(
            f"asr.cloud_total_timeout_s 应为数值，实际 {type(cloud_total_timeout_s).__name__}"
        )
    timeout = float(cloud_total_timeout_s)
    if timeout != timeout or timeout in (float("inf"), float("-inf")) or timeout <= 0.0:
        raise ASRHardError(f"asr.cloud_total_timeout_s 必须为有限正数，实际 {timeout}")

    return model, ep, timeout


def _read_dashscope_key() -> str:
    """读取 dashscope_api_key（内存保留）；缺失/全空白/读取失败抛 ASRHardError。

    异常与日志**不含**任何 key 值（D12）。经 __import__ 动态载入配置层，避免把
    configs 名字泄进模块全局。
    """
    try:
        app_config = __import__(
            "configs.app_config", fromlist=["load_secrets", "SECRETS_LOCAL_PATH"]
        )
    except Exception as exc:
        raise ASRHardError(f"加载配置层失败，无法读取密钥：{exc}") from exc
    try:
        secrets = app_config.load_secrets(app_config.SECRETS_LOCAL_PATH)
    except Exception as exc:  # AppConfigError / OSError 等：统一 asr 侧硬故障
        raise ASRHardError(
            f"读取 secrets 失败（不回显任何键值）：{type(exc).__name__}"
        ) from exc
    key = secrets.get(_DASHSCOPE_KEY_NAME)
    if not isinstance(key, str) or not key.strip():
        raise ASRHardError(
            "secrets 的 dashscope_api_key 缺失或全空白：生产入口需要非空密钥（P2-T13）"
        )
    return key


def _reset_state_for_tests() -> None:
    """仅供测试：清空运行态回到"未初始化"（与 runlog/shutdown 同族惯例；生产不调用）。"""
    global _state
    _state = None
