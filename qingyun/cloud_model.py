"""云模型（LLM 方案识别）——**P2-03 在原文件替换契约桩体，文件名与公开签名不得改变**（P1 §1/§3.6）。

契约声明（P1 §1、§2 文件责任表）：
    本文件曾是 P1 的**契约桩**；P2-03 在**同名同路径**上替换桩体为 DeepSeek 真实
    REST 适配，替换桩体的同时**始终保留** ``LLMHardError`` 类名与 ``init`` /
    ``select_plan`` 的签名（异常类基类 ``RuntimeError`` 不动）。桩阶段确立的阶段边界
    不因替换而失效：调用方（P1 ``main``/``run_session``）与 P2 实现共享同一公开面，
    不得重命名、移位或以别的名字重建。

本阶段（P2-03）实现与未实现：
    * 已实现：``init(cfg)`` 的真实语义（只允许一次、``cfg.llm`` 校验、拒绝占位模型名、
      prompt.md 全文一次读入常驻、读 ``deepseek_api_key``、**一次**构造 HTTP 客户端）；
      ``select_plan(text)`` 的 DeepSeek 请求组装（每调用**新上下文**：
      system=prompt 全文 + user=转写文本；JSON mode）、严格 JSON 解析、第一次违约的
      **一次原样重问**、两次都违约返回 ``"invalid"``、硬故障立即抛不消耗重问，以及
      P2 §3.2 的**端到端 deadline**（本模块**私有**一份：两次语义请求各自独立起表）。
    * 仍未实现（他阶段）：requirements 增补属 P2-04；真实 DeepSeek 账号烟雾实测属 P5。

规范来源：
    docs/实施文档/P2_语音与云模型接入技术文档.md
        §4（init 读 prompt.md 一次、运行期不重读、key 与 cfg.llm、一次构造 HTTP 客户端；
            每调用新上下文 messages=[system: prompt 全文, user: 转写文本]；
            JSON mode ``response_format={"type":"json_object"}``；**合法当且仅当**单个
            JSON 对象、键集合恰 ``{"task"}``、task ∈ ``{"strawberry","invalid"}``；
            数字/旧数字字符串/布尔/blueberry/未知 ID/缺失/额外/重复键/代码块包装/截断
            输出均**违约**；第一次违约→新上下文同一 prompt 同一转写原样重发一次（不带
            上次输出）；第二次仍违约→返回 ``"invalid"``；网络/鉴权/限流/服务异常→立即
            LLMHardError 不消耗重问；两次语义请求为上限；每次耗时入日志；
            **计时口径 R05**：两次请求各自独立计时，重问不共享剩余预算；固定 POST
            ``base_url.rstrip('/')+'/chat/completions'``、Bearer、stream=false、
            model=deepseek-flash、thinking={"type":"disabled"}、max_tokens=128；只解析
            ``choices[0].message.content``，不把 reasoning_content 当 JSON；HTTP 失败/
            包结构损坏=HardError；有 content 但任务 JSON 不满足或 finish_reason 截断
            =语义违约）、
        §3.2（deadline 固定实现边界——本模块**私有**一份，规格明确"两适配器各有私有
            deadline 上下文"）、§5 表（T06–T10、T11–T13、T17、T18、T19 DeepSeek 侧）与
            末段替身纪律、§6 步骤 3
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md §3.6（冻结契约面逐字）、
        §4 步骤 9（``cloud_model.init(cfg)`` 异常 → ``terminate("LLM_INIT_FAILED")``）、
        §5（LLMHardError → ``terminate("LLM_HARD_ERROR")``；task ∉ registry →
        ``LLM_CONTRACT_VIOLATION``）、§3.1 末段（模板占位模型名不能用于真实请求）、
        §7（``llm_result`` 由 main 产生：转写文本/task/耗时——与本模块的 ``llm_request``
        事件职责不重叠，见开发日志 §6）
    docs/实施文档/P0_公共契约与迁移技术文档.md / prompt.md：task 契约只有
        ``strawberry`` 与 ``invalid`` 两个合法值（D01/D07；蓝莓属合法语音输入，
        模型应输出 invalid；模型输出 ``blueberry`` 属**违约**，不是第三种 task）
    docs/实施文档/README.md 决策 D01（英文 task）、D07（蓝莓→invalid）、
        D12（LLM=DeepSeek，JSON mode 与超时/关闭 SDK 自动重试在接入时实测核对，
        key 存 configs/secrets.local.json）、D15（mock 入口不调本 init）、
        D20（LLM 模型 ID 精确 ``deepseek-flash``、DeepSeek 官方 API）、复审 R05

可注入边界（私有、公开签名不变；形态与 P2-01/P2-02 交付的 asr 侧对齐）：
    * HTTP 客户端工厂 ``_new_http_client(*, timeout_s)``：由 **init 调用恰好一次**
      （P2 §4"构造一次 HTTP 客户端"），产物常驻 ``_state.client``；测试以本地 stub
      transport 的同形 ``httpx.Client`` 接管该名字（不 monkeypatch socket，P2 §5 末段）。
      每次请求再传**请求级** ``timeout=`` 与 ``follow_redirects=False``（分项超时不得
      超过剩余预算，§3.2）。
    * 时钟 ``_monotonic()``：假时钟用例（独立起表/R05）注入点；T17 用真实墙钟。
    * 信号原语走 ``__import__("signal")`` 的真实接口（与 asr 同形），支撑 T18 的
      处理器恢复/计时器取消/活动计时器与非主线程拒绝。
    * 密钥读取 ``_read_deepseek_key()`` 与 prompt 读取 ``_read_prompt()``：都在 init 内
      调用一次，测试通过 ``configs.app_config.SECRETS_LOCAL_PATH`` 与 ``cfg.prompt_path``
      注入临时文件，不读仓库真实 secrets。

与 qingyun/asr.py 的关系（刻意**不**共享代码）：
    ``_CloudDeadline`` / ``_reject_unsupported_call_environment`` / ``_new_http_client``
    / 脱敏辅助在两个模块里各有一份私有实现。理由：①P2 §3.2 的字面要求是"两适配器各
    有私有 deadline 上下文"，跨模块 import ``asr._CloudDeadline`` 会把两个供应商的超时
    与清理耦合在一起（一方改形状就悄悄破坏另一方），且**跨模块私有依赖**本身是坏形态；
    ②两侧的类别词表与出口异常不同（ASRHardError vs LLMHardError、deadline 到期在 LLM 侧
    不得消耗重问）；③复制的代价是两个模块各自可删可改。取舍记录见开发日志 §7 DEC-P2-03-A。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — 仅用于类型检查，运行时不导入配置层
    from configs.app_config import AppConfig

__all__ = ["LLMHardError", "init", "select_plan"]

# ---------------------------------------------------------------------------
# 配置校验常量
# ---------------------------------------------------------------------------

# 模板占位模型名（P1 §3.1 末段：占位模型名不能用于真实请求，init 拒绝）。
# **复制**自 asr 侧同一名单（跨模块私有依赖不可取，见头 docstring 与 DEC-P2-03-A）：
# 以 app.mock.json 的字面占位 "mock" 为核心，附常见模板占位写法（大小写不敏感、去空白
# 后比对）；真实模型 "deepseek-flash" 不在其中。
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

# secrets 键名（P2 §2/D12）；运行期经 __import__ 从 configs.app_config 取路径与 loader，
# 模块级不缓存配置层名字（避免泄进模块全局命名空间，常驻契约测试 A/B 组都查这一点）。
_DEEPSEEK_KEY_NAME = "deepseek_api_key"

# ---------------------------------------------------------------------------
# DeepSeek 请求协议固定字面量（P2 §4 逐字；model 与 base_url **不**硬编码，
# 取 init 校验后存于 _state 的配置值——换域名/换模型只改 app.json，不改代码）
# ---------------------------------------------------------------------------

# 固定 POST 路径：base_url.rstrip('/') + CHAT_COMPLETIONS_PATH。
CHAT_COMPLETIONS_PATH = "/chat/completions"
# JSON mode（§4：response_format={"type":"json_object"}，以 DeepSeek 实测为准）。
JSON_MODE_TYPE = "json_object"
# 关闭思考链：thinking={"type":"disabled"}——同时保证 content 就是任务 JSON，
# 不把 reasoning_content 当 JSON 解析（§4 末段）。
THINKING_DISABLED_TYPE = "disabled"
# 方案 JSON 极短，128 token 足够；max_tokens 与 finish_reason="length" 一起构成
# "截断输出=语义违约"的判据（§4）。
MAX_COMPLETION_TOKENS = 128
# 本适配是同步一次性请求。
STREAM_DISABLED = False
CONTENT_TYPE_HEADER_VALUE = "application/json"

# 合法 task 值的封闭集（P0/D01/D07）：**当且仅当**单个 JSON 对象、键集合恰为 {"task"}、
# 值 ∈ 本集合才算合法。blueberry 明确不在集合内——出现即违约（D07）。
VALID_TASKS = frozenset({"strawberry", "invalid"})

# finish_reason 里表示"输出被截断/未完整交付"的取值（§4：截断属语义违约，不是硬故障）。
_TRUNCATED_FINISH_REASONS = frozenset({"length", "content_filter"})

# deadline 取消计时器后，用来吸收"已送达但尚未派发"的在途 SIGALRM 的让出次数
# （与 asr 侧同一实现理由：恢复旧处理器前必须先把在途信号消费掉，否则迟到的 SIGALRM
# 会在 SIG_DFL 下终止进程）。
_DISARM_GRACE_TICKS = 8
# 异常/日志里对"外部异常文本"的截断长度（配合脱敏，避免把响应原文刷进日志）。
_DETAIL_MAX_CHARS = 200
# 语义请求次数上限（§4：两次为上限；第一次违约重问一次，第二次仍违约 → invalid）。
_MAX_SEMANTIC_REQUESTS = 2


class LLMHardError(RuntimeError):
    """云模型侧硬故障（P1 §3.6 冻结的契约面；P2 §4"网络/鉴权/限流/服务异常"）。

    与"语义违约"严格区分：响应内容不满足 task JSON 契约属**违约**（走一次原样重问，
    最终可能返回 ``"invalid"``），**不**抛本异常；只有传输/服务/协议级故障
    （HTTP 非 2xx、返回包结构损坏、端到端 deadline 到期、未 init、重复 init、
    cfg/密钥/prompt 校验失败、不支持的调用环境）才抛它。基类固定 ``RuntimeError``，
    顶层映射按类型分派 ``terminate("LLM_HARD_ERROR")``（P1 §5）。
    """


# ---------------------------------------------------------------------------
# 模块级运行态（单线程同步；import 时只声明、不加载任何重资源）
# ---------------------------------------------------------------------------


class _LlmState:
    """init 成功后常驻的运行态：prompt 全文 + cfg.llm 派生值 + 密钥 + HTTP 客户端。

    * ``prompt_text``：prompt.md **全文一次读入**，运行期不再读文件（§4）。
    * ``prompt_sha256`` / ``prompt_chars``：日志用的**摘要**（全文不入日志）。
    * ``client``：init 用 ``_new_http_client`` **构造一次**的 HTTP 客户端，进程内常驻
      （不因单次请求关闭；``P2 §4`` 的"构造一次"就是这个语义）。
    * ``api_key``：只在内存保留，任何日志/异常都不回显其值（D12）。
    """

    def __init__(self, *, model: str, endpoint: str, total_timeout_s: float,
                 api_key: str, prompt_text: str, client) -> None:
        self.model = model
        self.endpoint = endpoint
        self.total_timeout_s = total_timeout_s
        self.api_key = api_key  # 内存保留；日志与异常不得含认证头或完整 key
        self.prompt_text = prompt_text
        self.prompt_sha256 = _sha256_of(prompt_text)
        self.prompt_chars = len(prompt_text)
        self.client = client


_state: "_LlmState | None" = None


# ---------------------------------------------------------------------------
# 端到端 deadline（P2 §3.2 固定实现边界；本模块**私有**一份）
#
# 为什么必须有它：HTTP 客户端的分项超时（connect/read/write/pool）里，read 只约束
# "等待下一块数据"——一个每 50ms 挤一点数据的慢服务可以**永远不触发**任何分项超时，
# 而整次调用早已超过 total_timeout_s（R05）。因此对"连接 + DNS + TLS + 上传 + 完整
# 响应读取 + 解析"施加一个可中止的**墙钟**上限：time.monotonic() 记绝对截止点，
# signal.setitimer(ITIMER_REAL, remaining) 到点用 SIGALRM 打断阻塞中的系统调用。
# 计时口径（R05）：两次语义请求**各自独立起表**，违约重问**不**共享首次剩余预算。
# ---------------------------------------------------------------------------


class _LlmDeadlineExceeded(BaseException):
    """端到端 deadline 到期的专用异常（P2 §3.2"处理器抛专用异常"）。

    刻意**不继承 ``OSError``**，甚至不继承 ``Exception``，两个理由都是实现性的：
        1. PEP 475 之后 CPython 的系统调用在 EINTR 时会自动重试，只有信号处理器
           "设置了异常"才中断重试；若处理器抛的是 ``OSError(EINTR)`` 一类，语义上
           与"底层 socket 自己报的可重试错误"混在一起，中断有被吞掉的风险。
        2. httpx/httpcore 内部以 ``except Exception`` 把底层异常映射成自己的
           ``TransportError`` 族。继承 ``BaseException`` 保证"到期"原样穿过去，
           不会在最后一刻被改写成"传输错误"而丢掉 ``deadline_timeout`` 这一类别。
    它只在 deadline 上下文与本模块内部产生，**绝不**外泄给调用方；在 LLM 侧它属于
    "网络/服务异常"，一律立即转 ``LLMHardError``，**不**消耗"再问一次"（§4）。
    """


class _LlmHardFailure(Exception):
    """传输/协议级故障的内部载体：把"哪一类硬故障"从解析处带到统一出口。

    公开面仍然只有 ``LLMHardError``；本类只用于在私有函数之间传递**错误类别**
    （``auth`` / ``rate_limited`` / ``service_error`` / ``bad_json`` / ``bad_payload``
    / …），使日志分类不被字符串匹配猜测。语义是"立即失败、不重问、不重试"。
    """

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


class _LlmContractViolation(Exception):
    """**语义违约**（响应内容不满足 task JSON 契约）的内部载体（P2 §4）。

    与 :class:`_LlmHardFailure` 严格分立：违约消耗"再问一次"，第二次仍违约则返回
    ``"invalid"``；``reason`` 是给日志看的稳定标识（not_json / not_object /
    duplicate_keys / missing_task / extra_keys / task_not_string / unknown_task /
    truncated）。
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class _DuplicateJsonKey(ValueError):
    """重复键标记异常：由 ``object_pairs_hook`` 抛出（实现见 :func:`_parse_task`）。

    必须是 ``ValueError`` 子类——标准库 ``json`` 只对 ``ValueError`` 族一视同仁地
    原样向外传播（``JSONDecodeError`` 本身就是它的子类），既不被包装也不被吞掉。
    """


def _single_object_pairs(pairs):
    """``json.loads(object_pairs_hook=...)`：把**重复键**判为违约。

    标准 ``json.loads`` 对重复键静默取最后一个值（``{"task":"strawberry","task":
    "invalid"}`` 会得到 ``invalid``），而 §4 要求"重复键属违约"。用 hook 在构造 dict
    之前逐对检查键名，第一次出现重复即抛 :class:`_DuplicateJsonKey`（任何层级的重复键
    都判违约：合法的 task 响应只有 ``{"task": <字符串>}`` 一层，不存在合法嵌套重复键）。
    """
    seen: set = set()
    for key, _value in pairs:
        if key in seen:
            raise _DuplicateJsonKey(f"JSON 对象出现重复键 {key!r}")
        seen.add(key)
    return dict(pairs)


def _monotonic() -> float:
    """单调时钟源（**注入点**：R05 独立起表用例用假时钟推预算，不真实等待）。"""
    return __import__("time").monotonic()


def _yield_to_signals() -> None:
    """让出一次解释器/系统调用边界，使**已送达但尚未派发**的 SIGALRM 得到执行。"""
    __import__("time").sleep(0)


def _deadline_handler(signum, frame) -> None:
    """ITIMER_REAL 到点处理器：抛专用异常，把阻塞中的整次请求打断（§3.2）。"""
    raise _LlmDeadlineExceeded(
        f"DeepSeek 云请求端到端 deadline 到点（SIGALRM signum={signum}）"
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
        raise LLMHardError(
            "DeepSeek 方案识别拒绝执行：不支持的调用环境（端到端 deadline 依赖主线程"
            "信号派发，本期运行环境限定 Linux 主线程同步调用；P2 §3.2）"
        )
    signal = __import__("signal")
    try:
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    except Exception as exc:  # 平台不支持（无 setitimer 的环境）等
        raise LLMHardError(
            "DeepSeek 方案识别拒绝执行：不支持的调用环境（无法探测 ITIMER_REAL："
            f"{type(exc).__name__}）"
        ) from exc
    if remaining > 0.0 or interval > 0.0:
        raise LLMHardError(
            "DeepSeek 方案识别拒绝执行：不支持的调用环境（已有活动的 ITIMER_REAL"
            f"（remaining={remaining} interval={interval}）；本模块不覆盖其他组件的"
            "定时器，P2 §3.2）"
        )


class _CloudDeadline:
    """一次云请求的端到端 deadline 上下文（P2 §3.2；**每次语义请求单独起表**）。

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
            raise _LlmDeadlineExceeded(
                f"DeepSeek 云请求总时限 {self.total_s}s 的预算已耗尽，拒绝再发起请求"
            )
        return remaining

    def __enter__(self) -> "_CloudDeadline":
        _reject_unsupported_call_environment()
        signal = __import__("signal")
        self.deadline_at = _monotonic() + self.total_s
        budget = self.remaining_s()
        if budget <= 0.0:
            # 还没装表就没预算：直接判到期（也避免把负数喂给 setitimer）。
            raise _LlmDeadlineExceeded(
                f"DeepSeek 云请求总时限 {self.total_s}s 无剩余预算，拒绝发起请求"
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
            raise LLMHardError(
                "DeepSeek 方案识别拒绝执行：不支持的调用环境（装表瞬间仍看到活动 ITIMER_REAL）"
            )
        self._armed = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._armed:
            self._disarm()
        if exc_type is None and self.remaining_s() <= 0.0:
            raise _LlmDeadlineExceeded(
                f"DeepSeek 云请求整次调用超出总时限 {self.total_s}s（绝对截止点已越过）"
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
# 惰性依赖与脱敏辅助
# ---------------------------------------------------------------------------


def _import_httpx():
    """惰性载入 httpx（import 零副作用；缺依赖时明确硬故障，不静默降级为别的库）。"""
    try:
        return __import__("httpx")
    except Exception as exc:
        raise LLMHardError(
            f"DeepSeek 适配需要 httpx（P2 §4）；当前不可用：{type(exc).__name__}"
        ) from exc


def _new_http_client(*, timeout_s: float):
    """默认 HTTP 客户端工厂（**注入点**：测试注入本地 stub transport 的同形 Client）。

    **只由 init 调用一次**（P2 §4"构造一次 HTTP 客户端"），产物常驻 ``_state.client``。
    §3.2 的三条硬要求全部在这里显式写出，不依赖库默认值：
        * ``HTTPTransport(retries=0, http2=False)``——transport 层**零重试**、不走 HTTP/2；
        * ``follow_redirects=False``——禁自动 redirect（3xx 由本适配判为错误）；
        * ``connect/read/write/pool`` 四个分项超时**全部等于** ``total_timeout_s``
          （客户端级只是基线；每次请求再用请求级 ``timeout=`` 压到**当时的剩余预算**）。
    """
    httpx = _import_httpx()
    transport = httpx.HTTPTransport(retries=0, http2=False)
    timeout = httpx.Timeout(connect=timeout_s, read=timeout_s, write=timeout_s,
                            pool=timeout_s)
    return httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)


def _new_request_id() -> str:
    """日志关联用的请求标识（随机、不含密钥/正文；每次语义请求一个）。"""
    uuid = __import__("uuid")
    return uuid.uuid4().hex[:16]


def _sha256_of(text: str) -> str:
    """prompt 全文的 SHA256（日志只记摘要与长度，全文不入日志）。"""
    hashlib = __import__("hashlib")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _redact(text, secret: str) -> str:
    """把外部异常/响应文本里的密钥形态一律替换为占位，再截断（§4 脱敏）。

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


# ---------------------------------------------------------------------------
# 请求组装（P2 §4：每调用新上下文 + JSON mode + 固定参数）
# ---------------------------------------------------------------------------


def _build_request_payload(model: str, prompt_text: str, transcript: str) -> dict:
    """按 §4 逐字组装 body：**新上下文**——messages 恰好两条（system prompt 全文、
    user 本次转写文本），不携带历史对话、不携带上一次的模型输出。

    两次语义请求**共享同一个 payload 对象**（在 ``select_plan`` 里构造一次），这是
    "原样重发"的结构性保证：不存在"第二次悄悄多带一条消息"的代码路径。而每次
    ``select_plan`` 调用都重新构造列表与字典字面量（不跨调用复用可变对象），因此
    相邻两次会话之间也不会串上下文。
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": transcript},
        ],
        "stream": STREAM_DISABLED,
        "response_format": {"type": JSON_MODE_TYPE},
        "thinking": {"type": THINKING_DISABLED_TYPE},
        "max_tokens": MAX_COMPLETION_TOKENS,
    }


def _build_request_headers(api_key: str) -> dict:
    """Bearer 密钥 + application/json（§4；DeepSeek 无 SSE 开关头，SSE 由 stream=false 表达）。"""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": CONTENT_TYPE_HEADER_VALUE,
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
    """非 2xx 时尽力取 DeepSeek 的错误码用于分类日志（取不到就 None）。

    DeepSeek 的错误形态是 ``{"error": {"code": "...", "message": "..."}}``，也兼容
    顶层 ``code``/``type``。只用**短码**（形如 ``authentication_fails`` /
    ``rate_limit_reached``）；``message`` 可能含请求上下文，一律不入日志/异常。
    读取本身仍在 deadline 覆盖之内。
    """
    json_module = __import__("json")
    try:
        raw = response.read()
        document = json_module.loads(raw.decode("utf-8", "replace"))
    except _LlmDeadlineExceeded:
        raise
    except Exception:
        return None
    if not isinstance(document, dict):
        return None
    candidates = []
    error = document.get("error")
    if isinstance(error, dict):
        candidates.append(error.get("code"))
        candidates.append(error.get("type"))
    candidates.append(document.get("code"))
    candidates.append(document.get("type"))
    for code in candidates:
        if isinstance(code, str) and code and len(code) <= 64:
            return code
    return None


# ---------------------------------------------------------------------------
# 响应解析：包结构（硬故障）与 task JSON（语义违约）两层，严格分立
# ---------------------------------------------------------------------------


def _extract_content(body: bytes) -> "tuple[str, object]":
    """协议层：取 ``choices[0].message.content`` 与 ``finish_reason``。

    **只**读这一个字段——``reasoning_content``（思考链）永不参与 JSON 解析（§4）。
    坏 JSON / 顶层非对象 / 缺 choices / choices 空 / message 非对象 / 缺 content /
    content 非字符串 = "返回包结构损坏" → :class:`_LlmHardFailure`（硬故障，**不**消耗
    重问）。注意与 :func:`_parse_task` 的分界：content **存在且是字符串**之后，内容
    不合契约才是语义违约。
    """
    json_module = __import__("json")
    try:
        document = json_module.loads(body.decode("utf-8"))
    except _LlmDeadlineExceeded:
        raise
    except UnicodeError as exc:
        raise _LlmHardFailure("bad_json", f"响应不是合法 UTF-8：{exc}") from exc
    except ValueError as exc:
        raise _LlmHardFailure(
            "bad_json", f"响应不是合法 JSON（{type(exc).__name__}）"
        ) from exc
    if not isinstance(document, dict):
        raise _LlmHardFailure(
            "bad_payload", f"响应顶层不是 JSON 对象（{type(document).__name__}）"
        )
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _LlmHardFailure("bad_payload", "响应缺非空 choices 数组")
    first = choices[0]
    if not isinstance(first, dict):
        raise _LlmHardFailure(
            "bad_payload", f"choices[0] 不是 JSON 对象（{type(first).__name__}）"
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise _LlmHardFailure(
            "bad_payload", f"choices[0].message 不是 JSON 对象（{type(message).__name__}）"
        )
    if "content" not in message:
        raise _LlmHardFailure("bad_payload", "响应缺 choices[0].message.content")
    content = message["content"]
    if not isinstance(content, str):
        raise _LlmHardFailure(
            "bad_payload",
            f"choices[0].message.content 非字符串（{type(content).__name__}）",
        )
    return content, first.get("finish_reason")


def _parse_task(content: str, finish_reason) -> str:
    """契约层：把 content 判成 task，或判为**语义违约**。

    合法**当且仅当**（P2 §4 逐字）：单个 JSON 对象 + 键集合恰为 ``{"task"}`` +
    task ∈ ``{"strawberry","invalid"}``。逐项否决：
        * ``finish_reason`` 表示截断（``length``/``content_filter``）→ truncated；
        * 代码块包装、前后缀文本、语法不完整（含输出被截断的半截 JSON）→ not_json；
        * 数字 / 布尔 / null / 字符串 / 数组等顶层非对象 → not_object；
        * **重复键** → duplicate_keys（标准 json.loads 会静默取后者，故用
          ``object_pairs_hook`` 显式拦截，见 :func:`_single_object_pairs`）；
        * 缺 task / 有额外键 / task 非字符串（含旧数字字符串之外的所有非串）/
          未知 ID（含 ``blueberry``）→ 各自一类。
    前后**空白**允许（JSON 语义本身允许空白，``{"task": "invalid"}\\n`` 是单个对象）。
    """
    if isinstance(finish_reason, str) and finish_reason.lower() in _TRUNCATED_FINISH_REASONS:
        raise _LlmContractViolation(
            "truncated",
            f"finish_reason={finish_reason!r} 表示输出被截断（max_tokens="
            f"{MAX_COMPLETION_TOKENS}）",
        )
    json_module = __import__("json")
    try:
        document = json_module.loads(content, object_pairs_hook=_single_object_pairs)
    except _DuplicateJsonKey as exc:
        raise _LlmContractViolation("duplicate_keys", str(exc)) from exc
    except _LlmDeadlineExceeded:
        raise
    except ValueError as exc:
        # 覆盖：代码块 ```json …``` 包装、对象外多写文字、半截 JSON、空串。
        raise _LlmContractViolation(
            "not_json", f"content 不是可直接解析的单个 JSON 值（{type(exc).__name__}）"
        ) from exc
    if not isinstance(document, dict):
        raise _LlmContractViolation(
            "not_object",
            f"顶层不是 JSON 对象（{type(document).__name__}）：数字/布尔/null/数组一律违约",
        )
    keys = set(document)
    if keys != {"task"}:
        if "task" not in keys:
            raise _LlmContractViolation(
                "missing_task", f"缺 task 键（实际键集合 {sorted(keys)}）"
            )
        raise _LlmContractViolation(
            "extra_keys", f"键集合必须恰为 {{'task'}}，实际 {sorted(keys)}"
        )
    value = document["task"]
    if isinstance(value, bool) or not isinstance(value, str):
        raise _LlmContractViolation(
            "task_not_string",
            f"task 必须是字符串，实际 {type(value).__name__}（数字/布尔一律违约）",
        )
    if value not in VALID_TASKS:
        raise _LlmContractViolation(
            "unknown_task",
            f"task={value!r} 不在合法集合 {sorted(VALID_TASKS)}（blueberry 属违约，D07）",
        )
    return value


# ---------------------------------------------------------------------------
# 日志（脱敏）与统一出口
# ---------------------------------------------------------------------------


def _log_request_event(request_id: str, started_at: float, attempt: int, outcome: str,
                       *, state=None, **fields) -> None:
    """写一条**脱敏** JSONL 事件（类别 ``llm_request``：每次语义请求一条，上限两条）。

    永不写入 api_key / Authorization / prompt 全文 / 转写文本 / 响应原文：
        * prompt 以 ``prompt_sha256`` + ``prompt_chars`` 登记（全文不入日志）；
        * 转写文本与其终值由 main 的 ``llm_result`` 事件承载（P1 §7），本模块只记
          ``text_chars`` 长度，两侧职责不重叠（见开发日志 §6）；
        * 外部文本先过 :func:`_redact` 再入 ``detail``。
    日志自身的失败（未 init runlog、字段非法等）在这里吞掉——它不得掩盖云请求结果。
    """
    try:
        runlog = __import__("qingyun.runlog", fromlist=["event"])
    except Exception:
        return
    record = {
        "request_id": request_id,
        "attempt": attempt,
        "outcome": outcome,
        "elapsed_ms": round(max(0.0, (_monotonic() - started_at) * 1000.0), 3),
    }
    if state is not None:
        record["model"] = getattr(state, "model", None)
        record["endpoint"] = getattr(state, "endpoint", None)
        record["prompt_sha256"] = getattr(state, "prompt_sha256", None)
        record["prompt_chars"] = getattr(state, "prompt_chars", None)
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    try:
        runlog.event("llm_request", **record)
    except Exception:
        pass


def _hard_failure(request_id: str, started_at: float, attempt: int, category: str,
                  detail: str, *, state, text_chars: int) -> LLMHardError:
    """统一硬故障出口：记一条脱敏事件，并**返回**（不抛）待抛出的 LLMHardError。

    返回值形式让调用点写成 ``raise _hard_failure(...) from source``，从而
    ``__cause__`` 保留来源异常（"保留来源与原因"），消息里只出现类别与脱敏细节，
    并明确"不消耗违约重问、不重试、不降级模型"（D12/D20）。
    """
    safe_detail = _redact(detail, getattr(state, "api_key", "") or "")
    _log_request_event(request_id, started_at, attempt, "error", state=state,
                       category=category, detail=safe_detail, text_chars=text_chars)
    elapsed_s = max(0.0, _monotonic() - started_at)
    return LLMHardError(
        f"DeepSeek 方案识别失败（{category}）：{safe_detail}"
        f"（请求 {request_id}，第 {attempt} 次语义请求，已耗时 {elapsed_s:.3f}s；"
        "硬故障不消耗违约重问、不重试、不降级模型）"
    )


# ---------------------------------------------------------------------------
# 单次语义请求（组装 + deadline + HTTP + 两层解析）
# ---------------------------------------------------------------------------


def _perform_llm_request(deadline: "_CloudDeadline", state, payload: dict,
                         headers: dict) -> bytes:
    """发起**一次** POST 并返回 2xx 响应体字节（在 deadline 覆盖之内）。

    响应体在 ``with client.stream(...)`` 内完整读取：非 2xx 立即按状态归类为硬故障，
    2xx 才交出字节交给协议层。上下文管理器保证**未完成或被中断的响应一定关闭**
    （客户端本身常驻 ``_state.client``，不在此关闭——P2 §4 只要求"构造一次"）。
    """
    client = state.client
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
            raise _LlmHardFailure(
                _http_status_category(status),
                f"HTTP {status} 非 2xx（禁自动 redirect，服务侧 code={provider_code}）"
                if provider_code else f"HTTP {status} 非 2xx（禁自动 redirect）",
            )
        return response.read()        # 完整响应体读取（慢响应的中断点）


def _run_semantic_request(state, *, attempt: int, payload: dict, headers: dict,
                          text_chars: int) -> str:
    """一次语义请求的全部动作：新上下文 → deadline → POST → 协议 → 契约。

    返回合法 task 字符串；违约时抛 :class:`_LlmContractViolation`（由 ``select_plan``
    决定是否重问）；硬故障一律以 :class:`LLMHardError` 离开本函数（不重问）。
    """
    request_id = _new_request_id()
    started_at = _monotonic()
    common = {"state": state, "text_chars": text_chars}
    httpx = _import_httpx()
    finish_reason = None               # 违约日志用（协议层成功后才有值）
    try:
        with _CloudDeadline(float(state.total_timeout_s),
                            request_id=request_id) as deadline:
            body = _perform_llm_request(deadline, state, payload, headers)
            content, finish_reason = _extract_content(body)
            task = _parse_task(content, finish_reason)
    except _LlmContractViolation as exc:
        logged_reason = finish_reason if isinstance(finish_reason, str) else None
        _log_request_event(request_id, started_at, attempt, "violation",
                           category=exc.reason,
                           detail=_redact(exc.detail, state.api_key),
                           state=state, text_chars=text_chars,
                           finish_reason=logged_reason)
        raise
    except _LlmDeadlineExceeded as exc:
        # 端到端 deadline 到期（含"分项超时都不触发、但总时长已超"的慢响应，R05）。
        raise _hard_failure(
            request_id, started_at, attempt, "deadline_timeout",
            f"整次云请求超过 total_timeout_s={state.total_timeout_s}s："
            f"{type(exc).__name__}",
            **common,
        ) from exc
    except _LlmHardFailure as exc:
        raise _hard_failure(request_id, started_at, attempt, exc.category, exc.detail,
                            **common) from exc
    except httpx.TimeoutException as exc:
        # 分项超时：只约束"某一次等待过久"，与端到端到期分列两类便于运维定位。
        raise _hard_failure(
            request_id, started_at, attempt, "item_timeout",
            f"HTTP 分项超时（{type(exc).__name__}），分项预算 = 剩余总时限"
            f"（起算 {state.total_timeout_s}s）：{exc}",
            **common,
        ) from exc
    except httpx.HTTPError as exc:
        raise _hard_failure(
            request_id, started_at, attempt, "transport",
            f"传输层故障（{type(exc).__name__}），transport retries=0 未重试：{exc}",
            **common,
        ) from exc
    except LLMHardError as exc:
        # 环境/依赖拒绝（不支持的调用环境、缺 httpx）：消息已定性，只补一条事件。
        _log_request_event(request_id, started_at, attempt, "error",
                           category="refused",
                           detail=_redact(str(exc), state.api_key), **common)
        raise
    except (ValueError, TypeError, UnicodeError) as exc:
        raise _hard_failure(
            request_id, started_at, attempt, "local_error",
            f"本地序列化/解析故障（{type(exc).__name__}）",
            **common,
        ) from exc
    except OSError as exc:
        raise _hard_failure(
            request_id, started_at, attempt, "os_error",
            f"系统调用故障（{type(exc).__name__}）",
            **common,
        ) from exc
    except BaseException as exc:            # KeyboardInterrupt / SystemExit 等
        # 用户中断不是"云故障"：记录后**原样上抛**，交 P1 顶层生命周期处置，
        # 绝不转成 LLMHardError（那会把 Ctrl-C 洗白成一次可归因于服务的失败）。
        _log_request_event(request_id, started_at, attempt, "aborted",
                           error_class=type(exc).__name__, **common)
        raise
    _log_request_event(request_id, started_at, attempt, "ok", task=task,
                       http_status="2xx", **common)
    return task


# ---------------------------------------------------------------------------
# 公开契约面（P1 §3.6 冻结：类名与签名不得改变）
# ---------------------------------------------------------------------------


def init(cfg: AppConfig) -> None:
    """加载云模型侧运行期资源：prompt 全文、DeepSeek 密钥、cfg.llm 校验、HTTP 客户端。

    契约（P2 §2/§4，P2-03 实现）：
        * 由 ``main`` 冷启动段调用**恰好一次**（P1 §4 步骤 9，R03）；重复调用抛
          ``LLMHardError``，交 ``terminate("LLM_INIT_FAILED")``。
        * 校验 ``cfg.llm``：拒绝占位模型名（空串/"mock" 等模板占位，P1 §3.1）；
          ``base_url`` 必须是 https URL，请求地址固定为
          ``base_url.rstrip('/') + '/chat/completions'``；``total_timeout_s`` 必须是
          有限正数。
        * 读 ``cfg.prompt_path`` **一次**并常驻（运行期不重读该文件）。
        * 经 ``configs.app_config.load_secrets`` 读 ``deepseek_api_key``，缺失或全空白
          抛 ``LLMHardError``（P1 预检之外的兜底，P2-T13）；key 只在内存保留，日志与
          异常不回显其值（D12）。
        * **构造一次** HTTP 客户端（超时 = ``total_timeout_s``、transport 重试 0、
          禁自动 redirect）；失败即抛，状态保持未初始化可重试。
        * import 时不读文件、不请求网络、不建客户端（P2-T11）。
        * 独立 mock 入口**不调用**本函数（D15）；本 init 语义不为 mock 分支修改。
    """
    global _state
    if _state is not None:
        raise LLMHardError(
            "云模型已初始化：init(cfg) 只允许调用一次（重复调用属内部不变量破坏）"
        )

    model, endpoint, total_timeout_s = _validate_llm_settings(cfg)
    prompt_text = _read_prompt(_prompt_path_of(cfg))
    api_key = _read_deepseek_key()

    try:
        client = _new_http_client(timeout_s=total_timeout_s)
    except LLMHardError:
        raise
    except Exception as exc:
        raise LLMHardError(
            f"init(cfg) 失败：构造 DeepSeek HTTP 客户端失败（{type(exc).__name__}）；"
            "状态保持未初始化，可修复后重试"
        ) from exc

    _state = _LlmState(
        model=model,
        endpoint=endpoint,
        total_timeout_s=total_timeout_s,
        api_key=api_key,
        prompt_text=prompt_text,
        client=client,
    )


def select_plan(text: str) -> str:
    """把转写文本交给 DeepSeek，返回英文 task 标识（P2-03 真实实现）。

    契约（P2 §4）：
        * 前置条件：``init`` 已成功；否则抛 ``LLMHardError``（P2-T12）。
        * 每次请求**新上下文**：messages = [system: prompt.md 全文, user: text]，
          不携带历史对话；JSON mode（``response_format={"type":"json_object"}``）。
        * 严格解析：合法当且仅当响应为单个 JSON 对象、键集合恰为 ``{"task"}``、
          task ∈ ``{"strawberry","invalid"}``。数字/旧数字字符串/布尔/``blueberry``/
          未知 ID/缺失或额外或重复键/代码块包装/截断输出均属**违约**。
        * 第一次违约：新上下文、同一 prompt 与同一转写文本**原样重发一次**（不携带
          上次输出、不做纠错对话）；第二次仍违约 → 返回 ``"invalid"``。两次语义请求
          为上限，**各自独立端到端计时**（R05：重问不共享首次剩余预算）。
        * 网络/鉴权/限流/服务/协议异常与端到端到期 → **立即**抛 ``LLMHardError``，
          不消耗"再问一次"、不重试、不降级模型（D12/D20）。
        * 返回值必须是 ``str``；调用方（P1 §5）按 ``"invalid"`` 跳过、按 registry
          查表，未知值 ``terminate("LLM_CONTRACT_VIOLATION")``。
    """
    state = _require_state()
    if not isinstance(text, str):
        raise LLMHardError(
            f"select_plan(text) 的转写文本必须是 str，实际 {type(text).__name__}"
        )
    text_chars = len(text)
    # 两次语义请求共享的是**同一份**请求体（同一 prompt 全文、同一转写文本、同一固定
    # 参数），而每次请求重新起表；"重发"必须是原样重发，不带任何纠错对话。
    payload = _build_request_payload(state.model, state.prompt_text, text)
    headers = _build_request_headers(state.api_key)

    for attempt in range(1, _MAX_SEMANTIC_REQUESTS + 1):
        try:
            return _run_semantic_request(state, attempt=attempt, payload=payload,
                                         headers=headers, text_chars=text_chars)
        except _LlmContractViolation:
            continue  # 违约：还有额度就原样重问一次，用尽后落到下面的 invalid
    return "invalid"


# ---------------------------------------------------------------------------
# init 的校验与读取辅助
# ---------------------------------------------------------------------------


def _validate_llm_settings(cfg):
    """从 ``cfg.llm`` 取三字段并逐项校验；结构缺失/非法一律抛 LLMHardError。"""
    try:
        llm_settings = cfg.llm
        model = llm_settings.model
        base_url = llm_settings.base_url
        total_timeout_s = llm_settings.total_timeout_s
    except AttributeError as exc:
        raise LLMHardError(
            "init(cfg) 失败：cfg.llm 结构不合法"
            f"（缺 model/base_url/total_timeout_s）：{exc}"
        ) from exc

    if not isinstance(model, str) or not model.strip():
        raise LLMHardError(
            "llm.model 为空或非法：真实请求不允许占位/空模型名（P1 §3.1、D20）"
        )
    if model.strip().lower() in _PLACEHOLDER_MODEL_NAMES:
        raise LLMHardError(
            f"llm.model 是模板占位值 {model!r}，不能用于真实请求"
            "（P1 §3.1：P2 init 拒绝占位值；D20 要求精确 deepseek-flash）"
        )
    if not isinstance(base_url, str) or not base_url.strip():
        raise LLMHardError("llm.base_url 为空或非字符串：需要完整 https URL（D20）")
    raw = base_url.strip()
    if not raw.startswith("https://") or len(raw) <= len("https://"):
        raise LLMHardError(
            f"llm.base_url 必须是完整 https URL，实际 {base_url!r}"
            "（明文 http 不允许承载 Bearer 密钥）"
        )
    endpoint = raw.rstrip("/") + CHAT_COMPLETIONS_PATH

    if isinstance(total_timeout_s, bool) or not isinstance(
        total_timeout_s, (int, float)
    ):
        raise LLMHardError(
            f"llm.total_timeout_s 应为数值，实际 {type(total_timeout_s).__name__}"
        )
    timeout = float(total_timeout_s)
    if timeout != timeout or timeout in (float("inf"), float("-inf")) or timeout <= 0.0:
        raise LLMHardError(f"llm.total_timeout_s 必须为有限正数，实际 {timeout}")

    return model, endpoint, timeout


def _prompt_path_of(cfg):
    """取 ``cfg.prompt_path``（结构缺失即硬故障，不猜默认路径）。"""
    try:
        return cfg.prompt_path
    except AttributeError as exc:
        raise LLMHardError(
            f"init(cfg) 失败：cfg.prompt_path 不存在，无法加载 prompt：{exc}"
        ) from exc


def _read_prompt(prompt_path) -> str:
    """一次读入 prompt 全文（常驻内存；**运行期不再重读**，§4 逐字）。

    不存在 / 读不出 / 非 UTF-8 / 全空白都属初始化硬故障。异常消息只点名路径，
    不含任何密钥（本就没有）。
    """
    pathlib = __import__("pathlib")
    path = pathlib.Path(prompt_path)
    if not path.is_file():
        raise LLMHardError(
            f"init(cfg) 失败：prompt 文件不存在（prompt_path={path}）"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LLMHardError(
            f"init(cfg) 失败：prompt 文件读取失败（prompt_path={path}）："
            f"{type(exc).__name__}"
        ) from exc
    if not text.strip():
        raise LLMHardError(
            f"init(cfg) 失败：prompt 文件为空或全空白（prompt_path={path}）——"
            "JSON mode 需要 prompt 内含 JSON 输出要求，空 prompt 不予接受"
        )
    return text


def _read_deepseek_key() -> str:
    """读取 ``deepseek_api_key``（内存保留）；缺失/全空白/读取失败抛 LLMHardError。

    异常与日志**不含**任何 key 值（D12）。经 __import__ 动态载入配置层，避免把
    configs 名字泄进模块全局。
    """
    try:
        app_config = __import__(
            "configs.app_config", fromlist=["load_secrets", "SECRETS_LOCAL_PATH"]
        )
    except Exception as exc:
        raise LLMHardError(f"加载配置层失败，无法读取密钥：{exc}") from exc
    try:
        secrets = app_config.load_secrets(app_config.SECRETS_LOCAL_PATH)
    except Exception as exc:  # AppConfigError / OSError 等：统一 llm 侧硬故障
        raise LLMHardError(
            f"读取 secrets 失败（不回显任何键值）：{type(exc).__name__}"
        ) from exc
    key = secrets.get(_DEEPSEEK_KEY_NAME)
    if not isinstance(key, str) or not key.strip():
        raise LLMHardError(
            "secrets 的 deepseek_api_key 缺失或全空白：生产入口需要非空密钥（P2-T13）"
        )
    return key


def _require_state() -> "_LlmState":
    """select_plan 的前置检查：未 init → 硬故障；运行态缺字段 → 硬故障（不猜默认值）。"""
    state = _state
    if state is None:
        raise LLMHardError(
            "云模型未初始化：select_plan(text) 前必须先成功调用 init(cfg)（P2-T12）"
        )
    missing = [
        name for name in ("model", "endpoint", "total_timeout_s", "api_key",
                          "prompt_text", "client")
        if getattr(state, name, None) in (None, "")
    ]
    if missing:
        raise LLMHardError(
            f"云模型配置不完整：运行态缺 {'、'.join(missing)}（不猜测默认值、不降级）"
        )
    return state


def _reset_state_for_tests() -> None:
    """仅供测试：清空运行态回到"未初始化"（与 asr/runlog/shutdown 同族惯例；生产不调用）。"""
    global _state
    _state = None
