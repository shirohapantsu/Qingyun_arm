"""terminate——唯一退出出口（P1-02）。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §3.3（shutdown 契约：打印 status/detail/阶段/现场处置指引 + sys.exit(1)；
           幂等；不调用 hold_current/open_gripper/move_joints/configure_torque；
           atexit 钩子只做日志 flush 与无串口写入的资源关闭，不再调用 terminate）
        §3.4 末段（terminate 的日志再次失败时仅向 stderr 输出，仍退出，不能递归
           调用自己）
        §5（顶层异常映射：除 SystemExit 直通外，一切失败都收敛到 terminate）
        §6.6（_finish 是任务完成出口、**不**调 terminate——本模块无需感知，
           只需保证 terminate 自身语义成立）
        §8 步骤 2
    CAL-052 底线（P1 §3.3/§4.1）：任何退出不发新舵机指令、不回 home、
    不关扭矩；"臂冻结"是沿用既有保持状态的业务描述。

设计要点：
    * 本模块 **只** import 标准库（atexit/sys/typing）与 ``qingyun.runlog``：
      不存在任何通向串口/运动原语的 import 路径（tests 用 AST + 运行时替身
      探针双重钉死这一点）。
    * 终端报告用内置 print 直接输出，不经 ``runlog.console``——runlog 未 init
      时 console 会抛 RuntimeError，而 terminate 必须无条件把现场信息打出来。
    * 幂等靠模块级 ``_terminated`` 标志：首次调用完成"打印 + 记 terminate 事件
      + sys.exit(1)"；此后任何调用直接再抛 SystemExit(1)，不重复日志、不重复
      任何清理动作。
    * atexit 钩子在 **模块导入时注册一次**（不在 terminate 内注册，避免重入与
      重复注册）；钩子只调 ``runlog.close()``（flush + 关句柄），绝不调用
      terminate、绝不 sys.exit、绝不触碰串口。
"""

from __future__ import annotations

import atexit
import sys
from typing import NoReturn

from qingyun import runlog

__all__ = ["terminate", "reset_for_tests"]

# 幂等标志：首次 terminate 置位后不再记录/清理（P1 §3.3）。
_terminated = False

# 现场处置指引（CAL-052/D14 语义）：terminate 只退出，不接管物理现场。
_GUIDANCE_LINES = (
    "本进程直接退出（sys.exit(1)），不发送任何舵机/串口指令：",
    "不回 home、不卸力、不关扭矩、不松开夹爪；机械臂冻结在当前位姿"
    "（沿用既有保持状态，不代表软件能保证物理静止）。",
    "现场处置由操作者负责：请确认臂与末端物体状态后，人工决定卸力/断电与清理方式；",
    "排除故障后重新启动程序（冷启动预检先于任何串口连接执行）。",
)


def _format_stage() -> str:
    """所处阶段一行文本：优先 runlog 上下文快照，附带 session 信息。"""
    session = runlog.current_session_id()
    stage = runlog.snapshot_context()
    prefix = f"session={session}" if session else "runlog 未初始化"
    if not stage:
        return f"{prefix}；阶段上下文=（根上下文，无阶段字段）"
    pairs = "，".join(f"{key}={value!r}" for key, value in stage.items())
    return f"{prefix}；阶段上下文={pairs}"


def _warn_stderr(message: str) -> None:
    """降级输出：stderr 本身再坏也只能沉默——绝不回调 terminate（防递归）。"""
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover — stderr 不可用属极端环境
        pass


def _safe_str(value: object) -> str:
    """无条件产出字符串：str/repr 都可能坏，最后一级退化为类型名。

    terminate 是所有错误的收敛点（§5"其余未预期异常 terminate("UNEXPECTED",
    repr)"），它自己绝不允许再抛非 SystemExit。
    """
    try:
        return str(value)
    except Exception:
        pass
    try:
        return repr(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def terminate(status: str, detail: str) -> NoReturn:
    """唯一退出出口：打印 status/detail/所处阶段/现场处置指引 + 记 terminate 事件
    + ``sys.exit(1)``。

    - **幂等**：首次调用记录并退出；此后重复显式调用仍抛 ``SystemExit(1)``，
      但不重复日志、不重复任何清理。
    - **零运动**：不调用 hold_current/open_gripper/move_joints/configure_torque
      等任何运动/驱动原语，不发任何串口语义操作——任何出口都不发新舵机指令、
      不回 home、不关扭矩（CAL-052 底线）。
    - **日志失败降级**：runlog 未 init 或写失败时仅向 stderr 告警，仍 exit(1)，
      绝不递归调用自身。

    ``status``/``detail`` 防御性 ``str()``：terminate 是所有错误（含怪异
    ``__str__`` 的对象）的收敛点，自身绝不允许因格式化失败而抛非 SystemExit。
    """
    global _terminated
    if _terminated:
        # 幂等：不重复打印、不重复日志、不重复清理，只重申退出码。
        raise SystemExit(1)
    _terminated = True

    status = _safe_str(status)
    detail = _safe_str(detail)

    try:
        print(f"[TERMINATE] status={status}", flush=True)
        print(f"[TERMINATE] detail={detail}", flush=True)
        print(f"[TERMINATE] {_format_stage()}", flush=True)
        print("[TERMINATE] 现场处置指引：" + _GUIDANCE_LINES[0], flush=True)
        for line in _GUIDANCE_LINES[1:]:
            print(f"[TERMINATE] 　{line}", flush=True)
    except Exception as exc:  # 终端坏了也不能拦住退出（§3.4 末段降级原则）
        _warn_stderr(f"[SHUTDOWN][WARN] 终端报告输出失败（{type(exc).__name__}: {exc}），继续退出")

    try:
        # stage 作为显式字段冗余记录一份"所处阶段"（P1 §7 terminate 行最低字段）；
        # context 的关联键（task_id/task_instance_id/scan_id/attempt_id）由
        # runlog.event 自动合并，天然带在同一行里。
        runlog.event("terminate", status=status, detail=detail, stage=runlog.snapshot_context())
    except Exception as exc:
        _warn_stderr(
            f"[SHUTDOWN][WARN] terminate 事件日志写入失败"
            f"（{type(exc).__name__}: {exc}）；仅 stderr 输出，仍退出"
        )

    sys.exit(1)


def _atexit_flush() -> None:
    """atexit 钩子：只做日志 flush/关闭（无串口写入的资源关闭）。

    绝不再调用 terminate（防重入 SystemExit）；绝不 sys.exit；异常一律就地
    降级为 stderr 告警。runlog.close 本身幂等且不抛，双保险仍 try/except。
    """
    try:
        runlog.close()
    except Exception as exc:  # pragma: no cover — close 内部已自兜底
        _warn_stderr(f"[SHUTDOWN][WARN] 退出时日志关闭失败（{type(exc).__name__}: {exc}）")


# 导入时注册一次：terminate 内不再注册，幂等重复调用也就不产生重复钩子。
atexit.register(_atexit_flush)


def reset_for_tests() -> None:
    """仅供测试：清除幂等标志，让 terminate 可以在同一进程内被再次演练。

    atexit 钩子保持原样（模块 import 只发生一次，钩子也只注册一次；重复
    注册反而是要测的反模式）。生产代码不得调用。
    """
    global _terminated
    _terminated = False
