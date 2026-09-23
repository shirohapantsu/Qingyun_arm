"""P2-03 云模型（LLM）适配测试——覆盖 P2 §5 的 **T06–T10、T11、T12、T13（LLM 侧）、
T17/T18 的 LLM 变体与 T19 的 DeepSeek 侧**。

    * T06–T10：请求组装、严格 JSON 判定、一次原样重问、两次违约 → ``"invalid"``、
      硬故障立即抛（不消耗重问）（P2 §4）。
    * T11：干净子进程 import 零副作用（不拉起 httpx/configs/runlog/音频/视觉）。
    * T12/T13：init 前后契约（未 init、重复 init）与 cfg.llm / 密钥 / prompt 校验。
    * T17/T18（LLM 变体）：端到端 deadline 的**真实墙钟**证明与 signal 状态卫生
      （P2 §3.2——本模块**私有**一份 deadline 上下文，两次语义请求各自独立起表，R05）。
    * T19（DeepSeek 侧）：固定 POST 地址派生、逐字段请求体、choices 解析、无重试、
      禁 redirect、脱敏（P2 §4 末段）。

驱动方式（P2 §5 末段"替身"）：
    * **HTTP 层**一律用注入的本地 stub transport（``httpx.MockTransport``）——把
      ``cloud_model._new_http_client`` 换成"同形 Client 工厂"（**不** monkeypatch socket）。
      与 asr 侧的差别是：本模块的客户端由 **init 构造一次**并常驻 ``_state.client``，
      所以"工厂被调用几次"本身就是被测事实（T19 断言"恰好一次"、两次请求仍只一个客户端）。
      请求级分项超时通过 ``StubClient.stream`` 记录的 ``timeout=`` 观察。
    * **T17 刻意不用任何 stub**：真实 transport + 127.0.0.1 本地 HTTP 服务子进程做
      墙钟证明（既不求云也不消耗账号），并在服务端留痕核对 wire 级事实。
    * 时钟：R05 独立起表与预算耗尽用例注入假时钟 ``_monotonic``，不真实等待；T17 用真墙钟。
    * prompt/secrets：``tmp_path`` 里的临时文件 + monkeypatch ``SECRETS_LOCAL_PATH``，
      绝不读仓库真实 secrets。

**本文件零外网请求、零真实账号。**
"""

from __future__ import annotations

import json
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import configs.app_config as app_config_module
import qingyun.cloud_model as cloud_model
from qingyun import runlog
from qingyun.cloud_model import CHAT_COMPLETIONS_PATH, MAX_COMPLETION_TOKENS

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_KEY = "sk-unit-test-deepseek-0123456789-abcdef"

# 测试用 prompt（可辨识文本，便于断言"system 消息 = prompt 全文"）；仓库真实
# prompt.md 的形态由 test_T19_仓库真实prompt全文原样作为system消息 一例覆盖。
PROMPT_TEXT = "## 核心要求\n只返回一个 JSON 对象，键集合恰为 {\"task\"}\n示例输出\n" \
              "{\"task\": \"strawberry\"}\n"

TRANSCRIPT = "帮我分拣一些草莓"

_UNSET = object()


def task_content(task: str) -> str:
    """§4 唯一合法响应形态的 content 文本。"""
    return json.dumps({"task": task})


def deepseek_body(content, finish_reason="stop", **extra_message):
    """DeepSeek chat/completions 成功包体的最小合法形态（§4 末段）。"""
    message = {"role": "assistant", "content": content}
    message.update(extra_message)
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "object": "chat.completion",
        "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
    }


def ok_response(task="strawberry"):
    return httpx.Response(200, json=deepseek_body(task_content(task)))


def payload_response(payload):
    """任意 JSON 包体（用于"包结构损坏"与多选项用例）。"""
    return httpx.Response(200, json=payload)


def alarm_state():
    """(当前 SIGALRM 处理器, ITIMER_REAL 读数)——T18 的状态卫生断言用。"""
    return signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)


# ---------------------------------------------------------------------------
# 公共夹具：每例前后把 cloud_model 运行态复位为"未初始化"（避免跨例污染）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_llm_state():
    cloud_model._state = None
    yield
    cloud_model._state = None


# ---------------------------------------------------------------------------
# 替身：记录型 Client / HTTP 层工厂注入 / 运行态直装
# ---------------------------------------------------------------------------


class StubClient(httpx.Client):
    """记录 close() 次数与**请求级** timeout / URL 的本地 Client。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_count = 0
        self.request_timeouts: list = []
        self.request_urls: list = []
        self.stream_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()

    def stream(self, method, url, **kwargs):          # 只做记录后原样转发
        self.request_timeouts.append(kwargs.get("timeout"))
        self.request_urls.append(str(url))
        self.stream_count += 1
        return super().stream(method, url, **kwargs)


class StubHttp:
    """HTTP 层替身的记录本：到达过的请求、建过的 client、客户端级超时值。"""

    def __init__(self, responder):
        self._responder = responder
        self.requests: list[httpx.Request] = []
        self.clients: list[StubClient] = []
        self.client_timeouts: list[float] = []

    def __call__(self, request):
        self.requests.append(request)
        return self._responder(request)

    @property
    def count(self) -> int:
        return len(self.requests)

    @property
    def client(self) -> StubClient:
        assert self.clients, "尚未构造任何客户端"
        return self.clients[-1]

    @property
    def request_timeouts(self) -> list:
        return [timeout for client in self.clients for timeout in client.request_timeouts]


def install_stub_http_layer(monkeypatch, responder) -> StubHttp:
    """把 ``cloud_model._new_http_client`` 换成"本地 stub transport 的 Client"工厂。

    本模块的工厂**只应被 init 调用一次**（P2 §4），所以工厂入参（客户端级超时）与
    被建出的客户端个数都是可断言的事实；请求级超时与关闭次数由 ``StubClient`` 记录。
    """
    stub = StubHttp(responder)

    def factory(*, timeout_s):
        client = StubClient(
            transport=httpx.MockTransport(stub),
            timeout=httpx.Timeout(connect=timeout_s, read=timeout_s, write=timeout_s,
                                  pool=timeout_s),
            follow_redirects=False,
        )
        stub.clients.append(client)
        stub.client_timeouts.append(float(timeout_s))
        return client

    monkeypatch.setattr(cloud_model, "_new_http_client", factory)
    return stub


def arm_llm_state(monkeypatch, responder=None, **overrides):
    """装上"init 成功后"的运行态（用例可覆盖任意字段，含显式 ``client=``）。

    ``responder`` 为 None 且未显式给 ``client`` 时，客户端保持 None——用于"缺配置即
    拒绝、且一次请求都不发"的用例；否则用注入工厂造一个常驻客户端（与 init 的产物同形）。
    """
    explicit_client = "client" in overrides
    stub = install_stub_http_layer(monkeypatch, responder) if responder is not None else None
    fields = {
        "model": MODEL_ID,
        "endpoint": ENDPOINT,
        "total_timeout_s": 30.0,
        "api_key": DEEPSEEK_KEY,
        "prompt_text": PROMPT_TEXT,
        "prompt_sha256": cloud_model._sha256_of(PROMPT_TEXT),
        "prompt_chars": len(PROMPT_TEXT),
        "client": _UNSET,
    }
    fields.update(overrides)
    if fields["client"] is _UNSET:
        fields["client"] = None
    state = SimpleNamespace(**fields)
    if state.client is None and stub is not None and not explicit_client:
        # 客户端级超时用运行态的预算；被覆盖成 None 的字段（"缺配置即拒绝"用例）
        # 不该把 None 喂给工厂，因此回退到一个名义值——反正被测的是"拒绝在发请求之前"。
        nominal = state.total_timeout_s if isinstance(state.total_timeout_s,
                                                      (int, float)) else 30.0
        state.client = cloud_model._new_http_client(timeout_s=nominal)
    monkeypatch.setattr(cloud_model, "_state", state)
    return state, stub


# ---------------------------------------------------------------------------
# init 环境：临时 prompt 文件 + 临时 secrets + stub 客户端工厂
# ---------------------------------------------------------------------------


def make_cfg(tmp_path, *, prompt_text=PROMPT_TEXT, model=MODEL_ID,
             base_url=BASE_URL, total_timeout_s=30.0, prompt_name="prompt.md"):
    prompt = tmp_path / prompt_name
    prompt.write_text(prompt_text, encoding="utf-8")
    return SimpleNamespace(
        prompt_path=prompt,
        llm=SimpleNamespace(model=model, base_url=base_url,
                            total_timeout_s=total_timeout_s),
    )


def make_secrets(monkeypatch, tmp_path, *, deepseek=DEEPSEEK_KEY, dashscope="x"):
    """写临时 secrets.local.json 并把 ``SECRETS_LOCAL_PATH`` 指过去。

    ``deepseek=...``（Ellipsis）表示**整键缺失**；``None`` 表示键在但值为 null。
    """
    secrets = tmp_path / "secrets.local.json"
    payload = {"dashscope_api_key": dashscope}
    if deepseek is not ...:
        payload["deepseek_api_key"] = deepseek
    secrets.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_config_module, "SECRETS_LOCAL_PATH", secrets)
    return secrets


def init_ok(monkeypatch, tmp_path, responder=None, **cfg_kwargs):
    """走完整 init（stub 客户端工厂），返回 (cfg, stub)。"""
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, **cfg_kwargs)
    stub = install_stub_http_layer(
        monkeypatch, responder or (lambda request: ok_response()))
    cloud_model.init(cfg)
    return cfg, stub


# ===========================================================================
# 替身层：分片响应（T18 的"响应是否被关闭"直接证据）
# ===========================================================================


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


def deepseek_bytes(content, finish_reason="stop"):
    return json.dumps(deepseek_body(content, finish_reason)).encode("utf-8")


# ===========================================================================
# 日志夹具（P1 §3.4 的内存 sink 注入点）
# ===========================================================================


@pytest.fixture
def captured_events(tmp_path):
    """把 runlog 接到内存 sink，返回收到的 JSON 行列表。"""
    runlog.reset_for_tests()
    runlog.init(tmp_path / "logs")
    lines: list[str] = []
    runlog.set_sink(lines.append)
    yield lines
    runlog.reset_for_tests()


# ===========================================================================
# P2-T06：第一响应即合法英文 task → 仅一次请求
# ===========================================================================


def test_T06_首次响应合法只发一次请求并返回该task(monkeypatch):
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 1, "合法响应不得触发任何额外请求"
    assert stub.client.stream_count == 1


def test_T06_首次合法invalid同样只发一次请求(monkeypatch):
    """``{"task":"invalid"}`` 是**合法**响应（一次即返），与"两次违约后兜底 invalid"可分。"""
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("invalid"))
    assert cloud_model.select_plan("今天天气如何") == "invalid"
    assert stub.count == 1
    assert len(stub.client.request_timeouts) == 1


def test_T06_请求上下文恰为system加user两条消息(monkeypatch):
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"))
    cloud_model.select_plan(TRANSCRIPT)
    body = json.loads(stub.requests[0].content.decode("utf-8"))
    messages = body["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == PROMPT_TEXT, "system = prompt.md 全文"
    assert messages[1]["content"] == TRANSCRIPT, "user = 转写文本"
    assert all(set(m) == {"role", "content"} for m in messages)


def test_T06_每次调用都是新上下文_第二次调用不带第一次转写(monkeypatch):
    """同一常驻客户端上连续两次 select_plan：各自 messages 只含本次转写、共两条消息。"""
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"))
    assert cloud_model.select_plan("帮我分拣草莓") == "strawberry"
    assert cloud_model.select_plan("请把蓝莓挑出来") == "strawberry"
    assert stub.count == 2
    first = json.loads(stub.requests[0].content.decode("utf-8"))["messages"]
    second = json.loads(stub.requests[1].content.decode("utf-8"))["messages"]
    assert [m["role"] for m in second] == ["system", "user"], "不累积历史对话"
    assert second[1]["content"] == "请把蓝莓挑出来"
    assert first[1]["content"] not in second[1]["content"]
    assert second[0]["content"] == first[0]["content"] == PROMPT_TEXT
    assert stub.client.stream_count == 2, "客户端常驻复用（init 只构造一次）"


# ===========================================================================
# P2-T07：第一响应违约、第二合法 → 两次请求上下文一致；返回第二值
# ===========================================================================


VIOLATION_TEXT = "I cannot answer with JSON, sorry"


def test_T07_首次违约后原样重问第二次合法返回第二值(monkeypatch):
    iterator = iter([
        httpx.Response(200, json=deepseek_body(VIOLATION_TEXT)),
        ok_response("strawberry"),
    ])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))

    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 2, "第一次违约 → 恰好重问一次"


def test_T07_两次请求体逐字节一致_prompt与转写均原样(monkeypatch):
    iterator = iter([
        httpx.Response(200, json=deepseek_body('{"task": "blueberry"}')),
        ok_response("strawberry"),
    ])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert len(stub.requests) == 2
    assert stub.requests[0].content == stub.requests[1].content, "重问必须原样重发"
    body = json.loads(stub.requests[1].content.decode("utf-8"))
    assert body["messages"][0]["content"] == PROMPT_TEXT
    assert body["messages"][1]["content"] == TRANSCRIPT


def test_T07_重问不携带上次输出也不做纠错对话(monkeypatch):
    iterator = iter([
        httpx.Response(200, json=deepseek_body(VIOLATION_TEXT)),
        ok_response("invalid"),
    ])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    raw = stub.requests[1].content.decode("utf-8")
    assert VIOLATION_TEXT not in raw, "上次输出绝不进第二次上下文"
    assert "blueberry" not in raw
    assert "sorry" not in raw
    assert json.loads(raw)["messages"][0]["role"] == "system"
    assert [m["role"] for m in json.loads(raw)["messages"]] == ["system", "user"]


def test_T07_首次违约第二次合法strawberry_返回第二值不被首值顶替(monkeypatch):
    """顺序敏感性：违约后的第二值才是返回值。"""
    iterator = iter([
        httpx.Response(200, json=deepseek_body("[oops")),
        ok_response("strawberry"),
    ])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 2


# ===========================================================================
# P2-T08：两次均违约 → 返回 invalid，无第三次请求
# ===========================================================================


def test_T08_两次均违约返回invalid且无第三次请求(monkeypatch):
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(
            200, json=deepseek_body('{"task": "blueberry"}')))
    assert cloud_model.select_plan("帮我分拣蓝莓") == "invalid"
    assert stub.count == 2, "两次语义请求为上限"
    assert stub.client.stream_count == 2


def test_T08_两次违约各记一条attempt事件(monkeypatch, captured_events):
    _state, _stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(
            200, json=deepseek_body("not json at all")))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    rows = [json.loads(line) for line in captured_events]
    assert [row["kind"] for row in rows] == ["llm_request", "llm_request"]
    assert [row["attempt"] for row in rows] == [1, 2]
    assert [row["outcome"] for row in rows] == ["violation", "violation"]
    assert rows[0]["request_id"] != rows[1]["request_id"], "两次请求各有可定位标识"


def test_T08_兜底invalid不再额外发第三次请求(monkeypatch):
    """兜底值 ``"invalid"`` 是本模块常量，**不**经由模型再问一次确认。"""
    counter = {"n": 0}

    def responder(request):
        counter["n"] += 1
        return httpx.Response(200, json=deepseek_body('{"task": 42}'))

    _state, stub = arm_llm_state(monkeypatch, responder)
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    assert counter["n"] == 2 == stub.count


# ===========================================================================
# P2-T09：网络/鉴权/限流/服务异常与包结构损坏 → 立即 LLMHardError，不消耗重问
# ===========================================================================


@pytest.mark.parametrize("status,category", [
    (401, "auth"),
    (403, "auth"),
    (429, "rate_limited"),
    (400, "request_rejected"),
    (500, "service_error"),
    (503, "service_error"),
])
def test_T09_服务侧错误立即硬故障且不消耗违约重问(monkeypatch, status, category):
    """关键断言：即使**第二次**请求会给出合法响应，硬故障也绝不重发（无隐式重试）。"""
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(status, json={
                "error": {"code": "some_code", "message": "绝不回显 " + "z" * 300}})
        return ok_response("strawberry")

    _state, stub = arm_llm_state(monkeypatch, responder)
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert type(info.value) is cloud_model.LLMHardError
    assert stub.count == 1, f"HTTP {status} 不得触发任何重试或违约重问"
    assert category in str(info.value)
    assert f"HTTP {status}" in str(info.value)
    assert "绝不回显" not in str(info.value), "响应正文不入异常"


def test_T09_传输层异常立即硬故障保留来源且不重试(monkeypatch):
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    _state, stub = arm_llm_state(monkeypatch, responder)
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert calls["n"] == 1 == stub.count
    assert "transport" in str(info.value)
    assert isinstance(info.value.__cause__, httpx.ConnectError), "保留来源异常"


def test_T09_读超时按分项超时归类且不重问(monkeypatch):
    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out")

    _state, stub = arm_llm_state(monkeypatch, responder)
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert calls["n"] == 1
    assert "item_timeout" in str(info.value)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_T09_3xx不跟随redirect直接判错且不再发第二次请求(monkeypatch):
    _state, stub = arm_llm_state(monkeypatch, lambda request: httpx.Response(
        302, headers={"Location": "https://other-region.invalid/chat/completions"}))
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert stub.count == 1
    assert "redirect_not_followed" in str(info.value)
    assert str(stub.requests[0].url) == ENDPOINT, "仍然只碰派生出的配置地址"


@pytest.mark.parametrize("body,note", [
    (b"not json at all", "坏 JSON 包体"),
    (b"", "空包体"),
    (b'{"choices": [1,2]}', "choices[0] 非对象"),
    (b'{"choices": []}', "choices 为空数组"),
    (b'{"choices": "x"}', "choices 非数组"),
    (b'{"id": "1"}', "缺 choices"),
    (b'{"choices": [{"message": {}}]}', "缺 message.content"),
    (b'{"choices": [{"message": {"content": null}}]}', "content 为 null"),
    (b'{"choices": [{"message": {"content": {"task": "strawberry"}}}]}', "content 非字符串"),
    (b'{"choices": [{"message": "text"}]}', "message 非对象"),
    (b'[1,2,3]', "顶层非对象"),
])
def test_T09_包结构损坏属硬故障_只发一次请求(monkeypatch, body, note):
    """§4 末段："HTTP失败/返回包结构损坏为 HardError"——**不**消耗违约重问。"""
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: stream_response(200, ChunkStream([body])))
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert stub.count == 1, f"{note}：硬故障不得重问"
    message = str(info.value)
    assert ("bad_json" in message) or ("bad_payload" in message)


def test_T09_硬故障异常消息与事件均不含密钥(monkeypatch, captured_events):
    leaked = httpx.ConnectError(f"tunnel Bearer {DEEPSEEK_KEY} failed")

    def responder(request):
        raise leaked

    _state, _stub = arm_llm_state(monkeypatch, responder)
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert DEEPSEEK_KEY not in str(info.value)
    assert "[已脱敏]" in str(info.value)
    assert DEEPSEEK_KEY not in "\n".join(captured_events)


# ===========================================================================
# P2-T10：违约形态矩阵（blueberry/数字/布尔/重复键/额外键/代码块/截断/未知 ID…）
# ===========================================================================


VIOLATIONS = [
    ('{"task": "blueberry"}', "stop", "unknown_task"),                   # D07：blueberry 违约
    ('{"task": "blueberry", "x": 1}', "stop", "extra_keys"),
    ('{"task": "1"}', "stop", "unknown_task"),                           # 旧数字字符串
    ('{"task": "0"}', "stop", "unknown_task"),
    ('{"task": 1}', "stop", "task_not_string"),                          # 数字
    ('{"task": 1.5}', "stop", "task_not_string"),
    ('{"task": true}', "stop", "task_not_string"),                       # 布尔
    ('{"task": false}', "stop", "task_not_string"),
    ('{"task": null}', "stop", "task_not_string"),
    ('{"task": {"value": "strawberry"}}', "stop", "task_not_string"),    # 多余嵌套
    ('{"task": ["strawberry"]}', "stop", "task_not_string"),
    ('{"task": "strawberry", "task": "invalid"}', "stop", "duplicate_keys"),
    ('{"task": "strawberry", "extra": 1}', "stop", "extra_keys"),
    ('{"task": "strawberry"}\n{"task": "invalid"}', "stop", "not_json"),  # 两个对象
    ('{"other": "strawberry"}', "stop", "missing_task"),
    ('{}', "stop", "missing_task"),
    ('{"task ": "strawberry"}', "stop", "missing_task"),                  # 键名不精确
    ('{"task": "STRAWBERRY"}', "stop", "unknown_task"),                   # 大小写敏感
    ('{"task": " strawberry"}', "stop", "unknown_task"),                  # 值含空白
    ('{"task": "strawberry", }', "stop", "not_json"),                      # 尾逗号
    ('```json\n{"task": "strawberry"}\n```', "stop", "not_json"),          # 代码块包装
    ('前言 {"task": "strawberry"}', "stop", "not_json"),
    ('{"task": "strawberry"} 结尾说明', "stop", "not_json"),
    ('{"task": "straw', "stop", "not_json"),                              # 截断输出
    ('', "stop", "not_json"),
    ('   \n ', "stop", "not_json"),
    ('strawberry', "stop", "not_json"),
    ('123', "stop", "not_object"),
    ('true', "stop", "not_object"),
    ('null', "stop", "not_object"),
    ('[{"task": "strawberry"}]', "stop", "not_object"),
    ('{"task": "strawberry"}', "length", "truncated"),                    # finish_reason 截断
    ('{"task": "strawberry"}', "content_filter", "truncated"),
]


@pytest.mark.parametrize("content,finish_reason,reason", VIOLATIONS,
                         ids=[f"{index}-{v[2]}" for index, v in enumerate(VIOLATIONS)])
def test_T10_违约形态一律走一次重问两次即兜底invalid(monkeypatch, captured_events,
                                                     content, finish_reason, reason):
    _state, stub = arm_llm_state(
        monkeypatch,
        lambda request: httpx.Response(200, json=deepseek_body(content, finish_reason)))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    assert stub.count == 2, "违约上限两次，不多问"
    rows = [json.loads(line) for line in captured_events]
    assert [row["category"] for row in rows] == [reason, reason]
    assert [row["outcome"] for row in rows] == ["violation", "violation"]


@pytest.mark.parametrize("content", [
    '  {"task": "strawberry"}  \n',
    '{"task":"invalid"}',
    '{ "task" : "strawberry" }',
])
def test_T10_合法对象的空白与键间距不影响判定(monkeypatch, content):
    """JSON 语义允许的**空白**不是"代码块包装/前后缀文本"，一次即返。"""
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body(content)))
    assert cloud_model.select_plan(TRANSCRIPT) == json.loads(content)["task"]
    assert stub.count == 1


def test_T10_蓝莓口令的正确模型输出是合法invalid_一次请求即返回(monkeypatch):
    """D07 的 prompt 行为：说"分拣蓝莓"时合法输出是 ``{"task":"invalid"}``（替身固定）。"""
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("invalid"))
    assert cloud_model.select_plan("帮我分拣一些蓝莓") == "invalid"
    assert stub.count == 1


def test_T10_重复键不会被json后者覆盖洗白(monkeypatch):
    """标准 json.loads 对重复键取后者——本模块用 object_pairs_hook 显式判违约。"""
    duplicate = '{"task": "strawberry", "task": "invalid"}'
    assert json.loads(duplicate) == {"task": "invalid"}, "前提：库默认静默取后者"
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body(duplicate)))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    assert stub.count == 2, "重复键判违约，而不是按后者胜出当合法响应"


def test_T10_reasoning_content里的合法JSON不被采信(monkeypatch):
    """§4：只解析 choices[0].message.content，**不**把 reasoning_content 当 JSON。"""
    def responder(request):
        return httpx.Response(200, json=deepseek_body(
            "我不能按 JSON 回答", reasoning_content='{"task": "strawberry"}'))

    _state, stub = arm_llm_state(monkeypatch, responder)
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    assert stub.count == 2


def test_T10_违约与硬故障分界_违约第二次合法即返回(monkeypatch):
    """对照：违约消耗的是"重问"额度；硬故障走另一条出口（见 T09 各例）。"""
    iterator = iter([
        httpx.Response(200, json=deepseek_body('{"task": "blueberry"}')),
        ok_response("strawberry"),
    ])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 2


def test_T10_两次都违约后signal状态仍然干净(monkeypatch):
    handler_before, _ = alarm_state()
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body("nope")))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before
    assert timer_after == (0.0, 0.0)
    assert stub.count == 2


# ===========================================================================
# P2-T11：干净子进程 import 零副作用
# ===========================================================================


def test_T11_干净子进程import零副作用():
    code = f"""
import sys, json
sys.path.insert(0, {str(ROOT)!r})
import qingyun.cloud_model
forbidden = ["httpx", "requests", "urllib3", "sounddevice", "onnxruntime", "torch",
             "silero_vad", "cv2", "ultralytics", "serial", "configs", "numpy",
             "qingyun.runlog", "qingyun.asr", "qingyun.grabbing"]
hit = [m for m in forbidden if any(x == m or x.startswith(m + ".") for x in sys.modules)]
print("@@" + json.dumps({{"hit": hit, "path": qingyun.cloud_model.__file__}}))
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert len(lines) == 1, f"import 期间有额外输出：{done.stdout!r}"
    report = json.loads(lines[0][2:])
    assert report["hit"] == [], f"import 拉起了重依赖 {report['hit']}"
    assert report["path"]


def test_T11_import不读prompt也不建客户端():
    """import 后运行态必须是 None（不预取任何资源）。"""
    assert cloud_model._state is None
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.select_plan(TRANSCRIPT)


# ===========================================================================
# P2-T12：init 前调业务函数 / init 重复调用
# ===========================================================================


def test_T12_init前调select_plan抛LLMHardError():
    assert cloud_model._state is None
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert type(info.value) is cloud_model.LLMHardError
    assert "未初始化" in str(info.value)
    assert "select_plan" in str(info.value)


def test_T12_init重复调用抛LLMHardError(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    cloud_model.init(cfg)
    assert cloud_model._state is not None
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "只允许调用一次" in str(info.value)
    assert "init" in str(info.value)


def test_T12_init成功后select_plan可用而再init仍被拒(monkeypatch, tmp_path):
    cfg, stub = init_ok(monkeypatch, tmp_path)
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 1
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.init(cfg)
    assert stub.count == 1, "重复 init 不得顺手发起任何请求"


@pytest.mark.parametrize("missing", ["model", "endpoint", "total_timeout_s", "api_key",
                                     "prompt_text", "client"])
def test_T12_运行态缺任一必需字段即拒绝且零请求(monkeypatch, missing):
    """未就绪形态一律硬故障，**不发请求**、不猜默认值。"""
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response(),
                                 **{missing: None})
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert "配置不完整" in str(info.value)
    assert missing in str(info.value)
    assert stub.count == 0


def test_T12_select_plan拒绝非字符串转写(monkeypatch):
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(None)          # type: ignore[arg-type]
    assert "必须是 str" in str(info.value)
    assert stub.count == 0


# ===========================================================================
# P2-T13（LLM 侧）与 init 校验
# ===========================================================================


@pytest.mark.parametrize("key", ["", "     ", ..., None])
def test_T13密钥缺失或全空白时init拒绝(monkeypatch, tmp_path, key):
    """缺失键 / 空串 / 全空白 / null 值四种形态都要拒绝，且失败态干净可重试。"""
    make_secrets(monkeypatch, tmp_path, deepseek=key)
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.init(cfg)
    assert cloud_model._state is None, "失败态保持未初始化（可修复后重试）"


def test_T13空白密钥的异常消息不回显任何键值(monkeypatch, tmp_path, capsys):
    make_secrets(monkeypatch, tmp_path, deepseek="   ")
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "deepseek_api_key" in str(info.value)
    captured = capsys.readouterr()
    assert DEEPSEEK_KEY not in captured.out + captured.err


def test_T13密钥文件不存在时init拒绝(monkeypatch, tmp_path):
    monkeypatch.setattr(app_config_module, "SECRETS_LOCAL_PATH",
                        tmp_path / "nope" / "secrets.local.json")
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "secrets" in str(info.value)
    assert cloud_model._state is None


def test_T13失败修复后可以再次init成功(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path, deepseek="   ")
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.init(cfg)
    make_secrets(monkeypatch, tmp_path)         # 换上一份合法 secrets
    cloud_model.init(cfg)
    assert cloud_model._state is not None


@pytest.mark.parametrize("model", ["mock", "", "   ", "MOCK", "Placeholder", "changeme",
                                   "tbd", "none", "your-model", "replace-me"])
def test_init拒绝占位模型名(monkeypatch, tmp_path, model):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, model=model)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "模型名" in str(info.value) or "llm.model" in str(info.value)
    assert cloud_model._state is None


def test_init接受真实模型名deepseek_flash(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, model="deepseek-flash")
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    cloud_model.init(cfg)
    assert cloud_model._state.model == "deepseek-flash"


@pytest.mark.parametrize("base_url", [
    "http://api.deepseek.com", "api.deepseek.com", "", "   ", "https://", "ftp://x/y",
])
def test_init要求https_base_url(monkeypatch, tmp_path, base_url):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, base_url=base_url)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "base_url" in str(info.value)


@pytest.mark.parametrize("timeout", [0, -1, 0.0, True, float("nan"), float("inf"), "30", None])
def test_init要求正有限total_timeout_s(monkeypatch, tmp_path, timeout):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, total_timeout_s=timeout)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "total_timeout_s" in str(info.value)


def test_init拒绝缺llm结构的cfg并点名cfg_llm(monkeypatch):
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(object())             # type: ignore[arg-type]
    assert "cfg.llm" in str(info.value)
    assert "init" in str(info.value)


@pytest.mark.parametrize("prompt_text", ["", "   \n  ", "\t"])
def test_init拒绝空白prompt(monkeypatch, tmp_path, prompt_text):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, prompt_text=prompt_text)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "prompt" in str(info.value)
    assert cloud_model._state is None


def test_init拒绝不存在的prompt文件(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path)
    cfg.prompt_path.unlink()
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "prompt" in str(info.value)


def test_init拒绝缺prompt_path的cfg(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path)
    del cfg.prompt_path
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.init(cfg)
    assert "prompt_path" in str(info.value)


def test_init成功后运行态含prompt摘要与派生地址(monkeypatch, tmp_path, capsys):
    make_secrets(monkeypatch, tmp_path, deepseek="super-secret-deepseek-key")
    cfg = make_cfg(tmp_path)
    install_stub_http_layer(monkeypatch, lambda request: ok_response())
    cloud_model.init(cfg)
    state = cloud_model._state
    assert state.prompt_text == PROMPT_TEXT
    assert state.prompt_chars == len(PROMPT_TEXT)
    assert state.prompt_sha256 == cloud_model._sha256_of(PROMPT_TEXT)
    assert state.endpoint == ENDPOINT
    assert state.total_timeout_s == 30.0
    assert state.api_key == "super-secret-deepseek-key"
    captured = capsys.readouterr()
    assert "super-secret-deepseek-key" not in captured.out + captured.err


def test_prompt全文只在init读一次_运行期改写文件不影响请求(monkeypatch, tmp_path):
    """§4 逐字："读 prompt.md 一次（运行期不重读）"。"""
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path)
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response("strawberry"))
    cloud_model.init(cfg)
    cfg.prompt_path.write_text("篡改后的 prompt，绝不该出现在请求里", encoding="utf-8")
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    raw = stub.requests[0].content.decode("utf-8")
    assert json.loads(raw)["messages"][0]["content"] == PROMPT_TEXT
    assert "篡改" not in raw
    cfg.prompt_path.unlink()                    # 运行期连文件都可以不在
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 2


def test_init只构造一次HTTP客户端且客户端级超时等于total_timeout_s(monkeypatch, tmp_path):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, total_timeout_s=12.5)
    stub = install_stub_http_layer(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body("nope")))
    cloud_model.init(cfg)
    assert len(stub.clients) == 1
    assert stub.client_timeouts == [12.5]
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"   # 两次违约
    assert len(stub.clients) == 1, "请求期间不得再构造客户端"
    assert stub.count == 2
    assert stub.client.close_count == 0, "客户端常驻，不随单次请求关闭"


# ===========================================================================
# P2-T17（LLM 变体，R05 核心证明）：127.0.0.1 本地服务持续缓慢返回数据
# ===========================================================================

_SLOW_SERVER_SOURCE = '''
"""P2-T17 本地慢响应 DeepSeek 服务替身（只绑 127.0.0.1，绝不触云、不消耗账号）。

用法：python -c <本源码> <状态文件> <块间隔秒> <stall总秒> <慢响应文本>
路径分派：
  /fast_ok        立即返回完整合法 chat/completions 包体，并把**收到的请求特征**写进状态文件
  /slow_stall     声明超大 Content-Length，每 gap 秒挤 8 字节，共 stall 秒 —— 永不读完
  /slow_complete  **完整合法**的 task JSON 包体，每 gap 秒挤 12 字节慢慢送完（响应正常但慢）
  /redirect       302 + Location（验证不跟随 redirect）
每收到一次请求追加一行 "<path> received"，每送一片追加 "<path> sent=<n> t=<相对秒>"，
供测试证明"数据一直在流"与"没有第二次请求"。
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS_PATH = sys.argv[1]
GAP_S = float(sys.argv[2])
STALL_S = float(sys.argv[3])
SLOW_TEXT = sys.argv[4]
SLOW_BODY = json.dumps({
    "choices": [{"message": {"role": "assistant", "content": SLOW_TEXT},
                 "finish_reason": "stop"}],
}).encode("utf-8")
OK_BODY = json.dumps({
    "choices": [{"message": {"role": "assistant",
                             "content": "{\\"task\\": \\"strawberry\\"}"},
                 "finish_reason": "stop"}],
}).encode("utf-8")
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
        chunks = []
        while left > 0:
            chunk = self.rfile.read(min(left, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return declared, b"".join(chunks)

    def do_POST(self):
        declared, raw = self._read_request_body()
        path = self.path.strip("/")
        port = self.server.server_address[1]
        note("%s received" % path)
        if path == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d/redirected" % port)
            self.end_headers()
            return
        if path == "fast_ok":
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                messages = parsed.get("messages") or []
                roles = ",".join(str(m.get("role")) for m in messages
                                 if isinstance(m, dict))
                lengths = ",".join(str(len(str(m.get("content")))) for m in messages
                                   if isinstance(m, dict))
                note("fast_ok wire clen=%d ctype=%s bearer=%s keys=%s model=%s stream=%s "
                     "max_tokens=%s thinking=%s rfmt=%s roles=%s lengths=%s" % (
                     declared,
                     self.headers.get("Content-Type"),
                     (self.headers.get("Authorization") or "").startswith("Bearer "),
                     ",".join(sorted(parsed)),
                     parsed.get("model"),
                     json.dumps(parsed.get("stream")),
                     parsed.get("max_tokens"),
                     json.dumps(parsed.get("thinking"), sort_keys=True),
                     json.dumps(parsed.get("response_format"), sort_keys=True),
                     roles,
                     lengths,
                 ))
            else:
                note("fast_ok wire clen=%d unparsable" % declared)
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

# 慢响应文本：合法 task JSON + 允许的尾随空白（送完约 0.9s，远大于失败用例预算）。
SLOW_TEXT = '{"task": "strawberry"}' + " " * 160
SERVER_GAP_S = 0.05
SERVER_STALL_S = 3.0


@pytest.fixture(scope="module")
def local_llm_server(tmp_path_factory):
    """启动/收尾 127.0.0.1 慢响应服务子进程（模块内复用，避免多次付启动成本）。"""
    directory = tmp_path_factory.mktemp("llmslowserver")
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


def status_lines(status_path: Path, prefix: str) -> list[str]:
    """状态文件里以该前缀开头的记录行。"""
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.startswith(prefix)]


def sent_lines(status_path: Path, mode: str) -> list[str]:
    return status_lines(status_path, f"{mode} sent=")


def received_lines(status_path: Path, mode: str) -> list[str]:
    return status_lines(status_path, f"{mode} received")


def arm_real_transport_state(monkeypatch, server, path, budget):
    """直装运行态 + **真实** HTTPTransport 客户端（T17/T19 的 wire 级用例）。"""
    client = cloud_model._new_http_client(timeout_s=budget)
    arm_llm_state(monkeypatch, endpoint=f"{server['base']}/{path}",
                  total_timeout_s=budget, client=client)
    return client


def test_T17_持续缓慢返回数据被端到端deadline在总时限处打断(local_llm_server, monkeypatch):
    """R05 证明：块间隔 50ms ≪ read 超时，**没有任何分项超时会触发**；
    只有 signal.setitimer(ITIMER_REAL) 能在总时限处打断这条阻塞中的读取。
    """
    server = local_llm_server
    budget = 0.6
    arm_real_transport_state(monkeypatch, server, "slow_stall", budget)
    handler_before, _ = alarm_state()
    delivered_before = len(sent_lines(server["status"], "slow_stall"))
    received_before = len(received_lines(server["status"], "slow_stall"))

    started = time.monotonic()
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)          # 真实 transport、真实墙钟
    elapsed = time.monotonic() - started
    delivered = len(sent_lines(server["status"], "slow_stall")) - delivered_before

    assert "deadline_timeout" in str(info.value)
    assert isinstance(info.value.__cause__, cloud_model._LlmDeadlineExceeded)
    assert elapsed >= budget * 0.9, f"早于总时限退出，不像墙钟中断：{elapsed:.3f}s"
    assert elapsed <= budget + 0.5, f"未被及时中断：{elapsed:.3f}s"
    assert elapsed < SERVER_STALL_S - 0.5, "远早于服务自身结束（不是等它收尾才失败）"
    assert delivered >= 3, f"服务端只送了 {delivered} 片，未构成持续慢响应"
    assert SERVER_GAP_S * 5 < budget, "该间隔下分项 read 超时永不可能触发"
    # 端到端到期属硬故障：**不**消耗违约重问（服务端只收到一次请求）
    assert len(received_lines(server["status"], "slow_stall")) == received_before + 1
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_T17_响应正常但整体超预算同样被硬失败且不重问(local_llm_server, monkeypatch):
    """变体：响应**最终是完整合法 task JSON**，只是送得比总时限慢——同样必须失败。"""
    server = local_llm_server
    budget = 0.4
    arm_real_transport_state(monkeypatch, server, "slow_complete", budget)
    received_before = len(received_lines(server["status"], "slow_complete"))

    started = time.monotonic()
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    elapsed = time.monotonic() - started

    assert "deadline_timeout" in str(info.value)
    assert budget * 0.9 <= elapsed <= budget + 0.5, f"{elapsed:.3f}s"
    assert len(received_lines(server["status"], "slow_complete")) == received_before + 1
    assert budget < 1.5, "本例的预算必须小于送完全部数据所需时间"


def test_T17_同一慢服务在充足预算下正常成功_对照证明中断归因于总时限(
        local_llm_server, monkeypatch):
    """对照组：**同一个**慢响应、同样的代码路径，只要预算够就成功。

    这排除"连接坏了/服务立刻超时"之类的替代解释——失败的唯一原因是墙钟总时限。
    """
    server = local_llm_server
    arm_real_transport_state(monkeypatch, server, "slow_complete", 8.0)
    started = time.monotonic()
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < 5.0
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_T17_真实transport下的正常请求wire留痕逐字段符合协议(local_llm_server, monkeypatch):
    """真实 HTTP 往返（127.0.0.1）：服务端留痕核对 wire 级事实——认证头形态、
    Content-Type、Content-Length 完整上传、六个请求体字段与两条消息的长度。
    """
    server = local_llm_server
    arm_real_transport_state(monkeypatch, server, "fast_ok", 10.0)
    handler_before, _ = alarm_state()
    before = len(status_lines(server["status"], "fast_ok wire"))

    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"

    lines = status_lines(server["status"], "fast_ok wire")
    assert len(lines) == before + 1
    received = lines[-1]
    assert "ctype=application/json" in received and "bearer=True" in received
    assert "clen=" in received and not received.endswith("clen=0")
    assert "model=deepseek-flash" in received
    assert "stream=false" in received
    assert "max_tokens=128" in received
    assert 'thinking={"type": "disabled"}' in received
    assert 'rfmt={"type": "json_object"}' in received
    assert "roles=system,user" in received
    assert f"lengths={len(PROMPT_TEXT)},{len(TRANSCRIPT)}" in received
    assert ("keys=" + ",".join(sorted([
        "model", "messages", "stream", "response_format", "thinking",
        "max_tokens"]))) in received
    assert DEEPSEEK_KEY not in received, "留痕绝不含密钥值"
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


# ===========================================================================
# P2-T18（LLM 变体）：deadline 成功/违约/失败/Ctrl-C 后的 signal 状态与资源清理
# ===========================================================================


def test_T18_专用deadline异常不继承OSError也不继承Exception():
    """§3.2 的字面要求：处理器抛的专用异常**不继承 OSError**（防 EINTR 自动重试吞掉）。"""
    assert issubclass(cloud_model._LlmDeadlineExceeded, BaseException)
    assert not issubclass(cloud_model._LlmDeadlineExceeded, OSError)
    assert not issubclass(cloud_model._LlmDeadlineExceeded, Exception)
    assert "_LlmDeadlineExceeded" not in cloud_model.__all__


def test_T18_成功请求后旧处理器恢复计时器取消且响应已关闭(monkeypatch):
    handler_before, _ = alarm_state()
    body = deepseek_bytes('{"task": "strawberry"}')
    stream = ChunkStream([body[:12], body[12:]])
    _state, stub = arm_llm_state(monkeypatch, lambda request: stream_response(200, stream))

    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"

    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before, "SIGALRM 处理器必须交还原主"
    assert timer_after == (0.0, 0.0), "ITIMER_REAL 必须已取消"
    assert stream.closed is True and stream.delivered == 2
    assert stub.client.close_count == 0, "客户端常驻（init 构造一次），不随单次请求关闭"


def test_T18_违约路径两次响应都被关闭且signal干净(monkeypatch):
    handler_before, _ = alarm_state()
    body = deepseek_bytes("nope")
    streams = [ChunkStream([body[:6], body[6:]]), ChunkStream([body])]
    iterator = iter(streams)
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: stream_response(200, next(iterator)))

    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"

    assert all(stream.closed for stream in streams)
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert stub.client.stream_count == 2


def test_T18_到期失败后同样恢复signal并关闭未完成响应(monkeypatch):
    handler_before, _ = alarm_state()
    body = deepseek_bytes('{"task": "strawberry"}')
    # 两片之间睡 0.2s：0.3s 的闹钟必然落在"响应还没读完"的时候
    stream = ChunkStream([body[:10], body[10:]], gap_s=0.2)
    _state, stub = arm_llm_state(monkeypatch, lambda request: stream_response(200, stream),
                                 total_timeout_s=0.3)

    started = time.monotonic()
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    elapsed = time.monotonic() - started

    assert "deadline_timeout" in str(info.value)
    assert elapsed <= 1.0, f"闹钟未及时中断：{elapsed:.3f}s"
    assert 0 < stream.delivered < 2, "响应应处于未完成状态（被打断在中途）"
    assert stream.closed is True, "未完成的响应必须被关闭"
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert stub.count == 1, "端到端到期属硬故障：不消耗违约重问"


def test_T18_CtrlC中断后signal状态与资源清理同样成立(monkeypatch):
    """把 KeyboardInterrupt 塞进响应迭代：清理照做，异常**原样上抛**不洗白。"""
    handler_before, _ = alarm_state()
    stream = ChunkStream([deepseek_bytes('{"task": "strawberry"}')],
                         raise_after=(1, KeyboardInterrupt()))
    _state, stub = arm_llm_state(monkeypatch, lambda request: stream_response(200, stream))

    with pytest.raises(KeyboardInterrupt):
        cloud_model.select_plan(TRANSCRIPT)

    handler_after, timer_after = alarm_state()
    assert handler_after is handler_before, "Ctrl-C 也必须走 finally 恢复处理器"
    assert timer_after == (0.0, 0.0)
    assert stream.closed is True
    assert stub.count == 1


def test_T18_到期后的下一次调用不被旧闹钟打断(monkeypatch):
    handler_before, _ = alarm_state()
    body = deepseek_bytes('{"task": "strawberry"}')
    slow = ChunkStream([body[:12], body[12:]], gap_s=0.25)
    first_stub = arm_llm_state(
        monkeypatch, lambda request: stream_response(200, slow), total_timeout_s=0.3)[1]
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.select_plan(TRANSCRIPT)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)

    time.sleep(0.6)      # 等过"旧闹钟"的原始到点时刻：没取消就必然再触发一次 SIGALRM

    second = ChunkStream([body])
    second_stub = arm_llm_state(monkeypatch, lambda request: stream_response(200, second),
                                total_timeout_s=10.0)[1]
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"

    assert first_stub.count == 1 and second_stub.count == 1
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before


def test_T18_已有活动ITIMER_REAL时拒绝执行且不动别人的定时器(monkeypatch):
    handler_before, _ = alarm_state()
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response())
    signal.setitimer(signal.ITIMER_REAL, 30.0)     # 别的组件的定时器
    try:
        with pytest.raises(cloud_model.LLMHardError) as info:
            cloud_model.select_plan(TRANSCRIPT)
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
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response())
    outcomes: list[BaseException] = []

    def worker():
        try:
            cloud_model.select_plan(TRANSCRIPT)
        except BaseException as exc:      # noqa: BLE001 - 收集线程内结果交主线程断言
            outcomes.append(exc)

    thread = threading.Thread(target=worker, name="not-main")
    thread.start()
    thread.join(timeout=20.0)
    assert not thread.is_alive()

    assert len(outcomes) == 1
    assert type(outcomes[0]) is cloud_model.LLMHardError
    assert "不支持的调用环境" in str(outcomes[0])
    assert "主线程" in str(outcomes[0])
    assert stub.count == 0


def test_T18_两次语义请求各自独立起表_重问不共享剩余预算(monkeypatch):
    """R05 计时口径：第一次请求把预算烧掉 90%（假时钟），第二次仍从**完整**总时限起算。"""
    budget = 20.0
    clock = {"now": 500.0}
    monkeypatch.setattr(cloud_model, "_monotonic", lambda: clock["now"])
    burns = iter([budget * 0.9, 0.0])

    def responder(request):
        burn = next(burns)
        clock["now"] += burn
        content = '{"task": "blueberry"}' if burn > 0 else task_content("strawberry")
        return httpx.Response(200, json=deepseek_body(content))

    _state, stub = arm_llm_state(monkeypatch, responder, total_timeout_s=budget)
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"

    assert stub.count == 2
    timeouts = stub.request_timeouts
    assert len(timeouts) == 2
    # 每次请求的分项超时都等于**当时**的剩余预算：两次都是完整 20s（单独起表）
    assert all(value == pytest.approx(budget) for value in timeouts), timeouts
    assert clock["now"] == pytest.approx(518.0)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_T18_假时钟推满总时限得到deadline硬错误且只发一次请求(monkeypatch):
    """LLM 侧的预算耗尽证明（T16 同族）：硬故障、单次请求、**不**消耗违约重问。"""
    budget = 30.0
    clock = {"now": 1_000.0}
    monkeypatch.setattr(cloud_model, "_monotonic", lambda: clock["now"])
    handler_before = signal.getsignal(signal.SIGALRM)
    armed: list[tuple[float, float]] = []

    def responder(request):
        armed.append(signal.getitimer(signal.ITIMER_REAL))
        clock["now"] += budget            # 这次请求"耗尽"了整段总时限
        return ok_response("strawberry")

    _state, stub = arm_llm_state(monkeypatch, responder, total_timeout_s=budget)
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)

    assert stub.count == 1, "适配层不得自行重试，也不得把到期当违约再问一次"
    assert "deadline_timeout" in str(info.value)
    assert "total_timeout_s" in str(info.value)
    assert isinstance(info.value.__cause__, cloud_model._LlmDeadlineExceeded), "保留来源"
    assert all(value == pytest.approx(budget) for value in stub.request_timeouts)
    assert armed and 0.0 < armed[0][0] <= budget, "请求期间 ITIMER_REAL 处于装填状态"
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before


def test_T18_预算非正时不再发起请求_上下文级(monkeypatch):
    """deadline 的"预算耗尽"闸门：分项超时**不超过**剩余预算，预算没了就不该再发请求。"""
    monkeypatch.setattr(cloud_model, "_monotonic", lambda: 100.0)
    stale = cloud_model._CloudDeadline(30.0, request_id="stale")
    stale.deadline_at = 90.0
    assert stale.remaining_s() == pytest.approx(-10.0)
    with pytest.raises(cloud_model._LlmDeadlineExceeded):
        stale.item_timeout_s()

    clock = {"now": 0.0}
    monkeypatch.setattr(cloud_model, "_monotonic", lambda: clock["now"])
    handler_before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(cloud_model._LlmDeadlineExceeded):
        with cloud_model._CloudDeadline(2.0, request_id="burn") as live:
            assert live.item_timeout_s() == pytest.approx(2.0)
            clock["now"] = 2.5
            live.item_timeout_s()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is handler_before


# ===========================================================================
# P2-T19（DeepSeek 侧）：固定请求/响应 fixture 逐字段核对
# ===========================================================================


@pytest.mark.parametrize("base_url,expected", [
    ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com///", "https://api.deepseek.com/chat/completions"),
    ("https://proxy.example.invalid/v1", "https://proxy.example.invalid/v1/chat/completions"),
    ("https://api.deepseek.com/beta/", "https://api.deepseek.com/beta/chat/completions"),
])
def test_T19_地址由base_url逐字派生不改写不跨域(monkeypatch, tmp_path, base_url, expected):
    make_secrets(monkeypatch, tmp_path)
    cfg = make_cfg(tmp_path, base_url=base_url)
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response("strawberry"))
    cloud_model.init(cfg)
    assert cloud_model._state.endpoint == expected
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert str(stub.requests[0].url) == expected
    assert stub.requests[0].method == "POST"


def test_T19_请求体逐字段符合DeepSeek协议(monkeypatch):
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"))
    cloud_model.select_plan(TRANSCRIPT)
    request = stub.requests[0]
    assert request.method == "POST"
    assert str(request.url) == ENDPOINT
    assert request.headers["Authorization"] == f"Bearer {DEEPSEEK_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.content.decode("utf-8"))
    assert set(body) == {"model", "messages", "stream", "response_format", "thinking",
                         "max_tokens"}
    assert body["model"] == MODEL_ID                    # D20：精确模型 ID
    assert body["stream"] is False                      # 同步一次性请求
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == MAX_COMPLETION_TOKENS == 128
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_T19_协议常量与规格逐字一致():
    """字面量本身是契约（§4 末段），不靠"与实现一致"的循环论证。"""
    assert CHAT_COMPLETIONS_PATH == "/chat/completions"
    assert cloud_model.JSON_MODE_TYPE == "json_object"
    assert cloud_model.THINKING_DISABLED_TYPE == "disabled"
    assert cloud_model.STREAM_DISABLED is False
    assert cloud_model.VALID_TASKS == frozenset({"strawberry", "invalid"})
    assert cloud_model._DEEPSEEK_KEY_NAME == "deepseek_api_key"
    assert cloud_model._MAX_SEMANTIC_REQUESTS == 2


def test_T19_仓库真实prompt全文原样作为system消息(monkeypatch, tmp_path):
    """配置里的 prompt_path 指向仓库 prompt.md 时，system 消息就是它的全字节。"""
    real_prompt = (ROOT / "prompt.md").read_text(encoding="utf-8")
    make_secrets(monkeypatch, tmp_path)
    cfg = SimpleNamespace(prompt_path=ROOT / "prompt.md",
                          llm=SimpleNamespace(model=MODEL_ID, base_url=BASE_URL,
                                              total_timeout_s=30.0))
    stub = install_stub_http_layer(monkeypatch, lambda request: ok_response("strawberry"))
    cloud_model.init(cfg)
    assert cloud_model._state.prompt_text == real_prompt
    cloud_model.select_plan(TRANSCRIPT)
    body = json.loads(stub.requests[0].content.decode("utf-8"))
    assert body["messages"][0]["content"] == real_prompt
    assert '{"task": "strawberry"}' in real_prompt, "JSON mode 前提：prompt 含 JSON 示例"


@pytest.mark.parametrize("task", ["strawberry", "invalid"])
def test_T19_成功响应只解析choices首项content(monkeypatch, task):
    payload = {
        "id": "uuid-1", "object": "chat.completion", "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": task_content(task)},
             "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "ignored"},
             "finish_reason": "stop"},
        ],
        "usage": {"total_tokens": 128},
    }
    _state, stub = arm_llm_state(monkeypatch, lambda request: payload_response(payload))
    assert cloud_model.select_plan(TRANSCRIPT) == task
    assert stub.count == 1


@pytest.mark.parametrize("finish_reason", [None, "stop", "stop_sequence"])
def test_T19_finish_reason非截断值仍按content判定(monkeypatch, finish_reason):
    """协议演进：finish_reason 不是"length"/"content_filter"就不算截断（§4 只点名截断）。"""
    iterator = iter([httpx.Response(
        200, json=deepseek_body(task_content("strawberry"), finish_reason))])
    _state, stub = arm_llm_state(monkeypatch, lambda request: next(iterator))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 1


def test_T19_默认客户端显式关闭重试与redirect并压平分项超时():
    """§3.2/§4 的硬要求在默认工厂里是**显式字面量**，不依赖库默认值（白盒核对）。"""
    client = cloud_model._new_http_client(timeout_s=1.75)
    try:
        assert client.follow_redirects is False
        assert client.timeout == httpx.Timeout(connect=1.75, read=1.75, write=1.75,
                                              pool=1.75)
        pool = client._transport._pool                  # noqa: SLF001 - 配置白盒核对
        assert pool._retries == 0                       # transport 层零重试
        assert pool._http2 is False
        assert pool._http1 is True
    finally:
        client.close()


def test_T19_请求级分项超时不超过剩余预算并由deadline供给(monkeypatch):
    budget = 20.0
    clock = {"now": 0.0}
    monkeypatch.setattr(cloud_model, "_monotonic", lambda: clock["now"])

    def responder(request):
        clock["now"] += 5.0            # 本次请求耗时 5s（假时钟）
        return httpx.Response(200, json=deepseek_body("nope"))

    _state, stub = arm_llm_state(monkeypatch, responder, total_timeout_s=budget)
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    # 第一次起算整段预算；第二次**重新起表**（不是剩余 15s）——R05
    assert stub.request_timeouts[0] == pytest.approx(budget)
    assert stub.request_timeouts[1] == pytest.approx(budget)


def test_T19_服务错误码只取短码且不回显响应正文(monkeypatch):
    body = {"error": {"code": "rate_limit_exceeded",
                      "message": "内含上下文 " + "y" * 500}}
    _state, stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(429, json=body))
    with pytest.raises(cloud_model.LLMHardError) as info:
        cloud_model.select_plan(TRANSCRIPT)
    assert "rate_limit_exceeded" in str(info.value)
    assert "内含上下文" not in str(info.value)
    assert stub.count == 1


# ===========================================================================
# 日志（llm_request）：字段可定位、职责与 main 的 llm_result 不重叠、全链脱敏
# ===========================================================================


def test_日志每次合法请求一条ok事件含attempt与耗时(monkeypatch, captured_events):
    _state, _stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"),
                                  total_timeout_s=15.0)
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    rows = [json.loads(line) for line in captured_events]
    assert [row["kind"] for row in rows] == ["llm_request"]
    row = rows[0]
    assert row["outcome"] == "ok"
    assert row["attempt"] == 1
    assert row["task"] == "strawberry"
    assert row["model"] == MODEL_ID
    assert row["endpoint"] == ENDPOINT
    assert row["elapsed_ms"] >= 0.0
    assert isinstance(row["request_id"], str) and len(row["request_id"]) == 16
    assert set(row) <= {"kind", "ts", "session_id", "request_id", "attempt", "outcome",
                        "elapsed_ms", "category", "detail", "model", "endpoint",
                        "prompt_sha256", "prompt_chars", "text_chars", "task",
                        "finish_reason", "http_status", "error_class"}


def test_日志不含密钥prompt全文与转写文本(monkeypatch, captured_events):
    """脱敏三禁 + 职责边界：转写文本由 main 的 llm_result 承载，本模块只记长度。"""
    _state, _stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body("nope")))
    secret_transcript = "帮我分拣草莓 关键个人内容"
    assert cloud_model.select_plan(secret_transcript) == "invalid"
    blob = "\n".join(captured_events)
    assert DEEPSEEK_KEY not in blob
    assert "Authorization" not in blob
    assert "核心要求" not in blob            # prompt 全文不入日志
    assert "关键个人内容" not in blob        # 转写文本由 main 的 llm_result 承载
    rows = [json.loads(line) for line in captured_events]
    assert [row["prompt_chars"] for row in rows] == [len(PROMPT_TEXT)] * 2
    assert [row["text_chars"] for row in rows] == [len(secret_transcript)] * 2
    assert rows[0]["prompt_sha256"] == cloud_model._sha256_of(PROMPT_TEXT)
    assert "text" not in rows[0] and "prompt_text" not in rows[0]


def test_日志记录违约类别与finish_reason(monkeypatch, captured_events):
    _state, _stub = arm_llm_state(
        monkeypatch,
        lambda request: httpx.Response(200, json=deepseek_body(
            '{"task": "strawberry"}', "length")))
    assert cloud_model.select_plan(TRANSCRIPT) == "invalid"
    rows = [json.loads(line) for line in captured_events]
    assert [row["category"] for row in rows] == ["truncated", "truncated"]
    assert [row["finish_reason"] for row in rows] == ["length", "length"]
    assert all(row["outcome"] == "violation" for row in rows)


def test_日志在硬故障时带类别并截断外部文本(monkeypatch, captured_events):
    long_detail = "x" * 5_000
    _state, _stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(500, text=long_detail))
    with pytest.raises(cloud_model.LLMHardError):
        cloud_model.select_plan(TRANSCRIPT)
    row = json.loads(captured_events[-1])
    assert row["outcome"] == "error"
    assert row["category"] == "service_error"
    assert len(row["detail"]) <= cloud_model._DETAIL_MAX_CHARS + len("…(截断)")
    assert long_detail not in json.dumps(row)


def test_日志未init_runlog时不影响业务结果(monkeypatch):
    """runlog 自身的失败绝不掩盖云请求结果（与 P1 §3.4 末段一致）。"""
    runlog.reset_for_tests()
    _state, stub = arm_llm_state(monkeypatch, lambda request: ok_response("strawberry"))
    assert cloud_model.select_plan(TRANSCRIPT) == "strawberry"
    assert stub.count == 1


def test_CtrlC事件记为aborted且不改写异常类型(monkeypatch, captured_events):
    stream = ChunkStream([deepseek_bytes('{"task": "strawberry"}')],
                         raise_after=(1, KeyboardInterrupt()))
    _state, _stub = arm_llm_state(monkeypatch, lambda request: stream_response(200, stream))
    with pytest.raises(KeyboardInterrupt):
        cloud_model.select_plan(TRANSCRIPT)
    rows = [json.loads(line) for line in captured_events]
    assert rows[-1]["outcome"] == "aborted"
    assert rows[-1]["error_class"] == "KeyboardInterrupt"


# ===========================================================================
# 公开面收口：select_plan 的返回类型与合法值封闭
# ===========================================================================


@pytest.mark.parametrize("content,expected", [
    ('{"task": "strawberry"}', "strawberry"),
    ('{"task": "invalid"}', "invalid"),
    ('{"task": "blueberry"}', "invalid"),
    ("junk", "invalid"),
])
def test_select_plan始终返回str且取值属于合法集(monkeypatch, content, expected):
    _state, _stub = arm_llm_state(
        monkeypatch, lambda request: httpx.Response(200, json=deepseek_body(content)))
    value = cloud_model.select_plan(TRANSCRIPT)
    assert isinstance(value, str)
    assert value == expected
    assert value in cloud_model.VALID_TASKS


def test_公开面只有契约三件套():
    """A 组常驻契约的第二重表达：新增的全是私有名字。"""
    assert set(cloud_model.__all__) == {"LLMHardError", "init", "select_plan"}
    assert issubclass(cloud_model.LLMHardError, RuntimeError)
    assert cloud_model.LLMHardError.__bases__ == (RuntimeError,)
