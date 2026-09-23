"""视觉契约桩——**P3/队友在原文件替换桩体，文件名与公开签名不得改变**（P1 §1/§3.6）。

契约桩声明（P1 §1、§2 文件责任表；任务.md"视觉分工与接口边界"）：
    本文件是**契约桩**，不是实现，而且**视觉实现归用户的队友**，不属本任务的开发范围。
    队友在**同名同路径**的 ``qingyun/grabbing/vision.py`` 上替换桩体（P3 §1："前置
    P0 的 VisionInterface/VisionThresholds 与 P1 冻结的模块桩（异常类名与签名不变，
    替换桩体）"）。桩文件不得被重命名或移位；调用方（P1 ``qingyun/plans/base.py``
    以 ``from qingyun.grabbing import vision`` 引用并可在测试中整体替换该名字）
    只按本文的契约面使用它。

桩阶段的行为：一切公开调用直接抛 ``VisionHardError``，消息含"未实现（P3/队友替换本桩）"。
    * **不**返回任何固定目标 / 伪造的 ``VisionInterface``；
    * **不**返回 ``None``、空成功或"视野干净"的假象；
    * **绝不**抛 ``NoTarget`` 掩盖模块缺失——``NoTarget`` 是"这次真实扫描里没有可抓
      目标"的业务结论，会让 plan 累加 ``no_target_streak`` 并在三连后**正常完成**任务
      （P1 §6.2/D14），用它冒充未实现等于把缺失模块洗白成"场地是空的"。
    * 抛 ``VisionHardError`` 则按 P1 §5/§6.2 映射 ``terminate("VISION_HARD_ERROR")``
      （不参与 NoTarget 计数，P1-T23），故障可见。

规范来源：
    docs/实施文档/P1_任务编排与运行生命周期技术文档.md
        §3.6（桩契约面逐字：``class NoTarget(Exception)`` /
           ``class VisionHardError(Exception)`` / ``configure(thresholds, config_path)`` /
           ``init()`` / ``get_target(ignore=0)``）
        §4 步骤 10（``vision.configure(VisionThresholds(…P0 契约), cfg.vision_config_path)``
           → ``vision.init()``；``VisionHardError`` → ``terminate("VISION_INIT_FAILED")``；
           **init 阶段故障不得伪装成 NoTarget**）
        §5/§6.2（get_target 的 NoTarget 与 VisionHardError 两条分支的计数语义）
        §2 表（本行=契约桩，P3 替换）、§8 步骤 3
    docs/实施文档/P3_视觉检测与目标选择技术文档.md
        §2（依赖方向：允许标准库、``configs.common_interface`` 类型、``qingyun.runlog``
           与自身模型栈；**不 import motion_params/plans/main/app_config**；
           vision.json 内相对路径相对该文件所在目录解析）
        §3（本文逐字重述的公开签名与语义）、§4（thresholds 之外的视觉参数归属：
           detection_floor/conf_threshold 等是视觉参数，**不进入公共目标类型**；
           阈值 ``VisionThresholds`` 由 P1 从已校验 profile 注入，P3 只消费）
        §6（get_target 管线七步）、§7（P3-T10/T11/T12/T15/T17/T18 场景）
    docs/实施文档/P0_公共契约与迁移技术文档.md：六字段 ``VisionInterface`` 与
        ``VisionThresholds``（本桩只在注解里引用它们，运行时不导入）。

P3/队友实现时必须保持的语义（本桩只登记，不实现任何视觉逻辑）：
    * 见下方各函数 docstring——逐条重述 P3 §3，签名与异常类名不得改变。
    * 桩阶段不做深度/几何/检测相关计算，也不为"跑通离线测试"在别处代写视觉逻辑；
      测试侧替身是 ``tests/support_upper.py`` 的 FakeVision（P1-04 交付），
      替身必须复用本文件的**真实** ``NoTarget`` / ``VisionHardError`` 类。

依赖边界（本桩阶段）：
    模块顶层**只** import 标准库 ``typing``（TYPE_CHECKING 守卫）：
    ``import qingyun.grabbing.vision`` 不加载相机 SDK（pyorbbecsdk/OpenNI2/UVC 绑定）、
    不加载检测运行时（ultralytics/torch/onnxruntime）、不加载 OpenCV、不加载权重、
    不打开设备（P3-T17），也不 import ``configs.common_interface``（连带 numpy）——
    ``VisionThresholds``/``VisionInterface``/``Path`` 只作类型注解，用
    ``from __future__ import annotations`` + ``TYPE_CHECKING`` 延迟求值（与仓库风格一致）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — 仅用于类型检查，运行时不导入公共契约层
    from pathlib import Path

    from configs.common_interface import VisionInterface, VisionThresholds

__all__ = ["NoTarget", "VisionHardError", "configure", "init", "get_target"]

# 统一前缀：三条硬错误消息都以它开头，测试据此钉死"未实现"与归属方（P3/队友）可见。
_NOT_IMPLEMENTED = "视觉未实现（P3/队友替换本桩）"


class NoTarget(Exception):
    """NoTarget——"这次扫描没有可抓目标"的业务结论（P3 §3 逐字：``class NoTarget(Exception)``）。

    触发情形（P3 §3、§6 第 7/8 步）：ROI 内无物体、全部候选被过滤、
    ``ignore ≥ valid_count``（跳尽，原因 ``skip_exhausted``）、
    replay 帧序列耗尽（``replay_exhausted``）。
    它是**正常业务信号**，不是故障：plan 捕获后 ``ignore=0``、``no_target_streak += 1``，
    三连即任务正常完成出口（P1 §6.1/§6.2/D14）。

    因此本桩**绝不**抛它：桩没有扫描过任何画面，无权断言场地为空。
    基类固定为 ``Exception``，与 ``VisionHardError`` 互不为父子（同级两类），
    顶层与 plan 靠类型分派严格区分"空"与"坏"。
    """


class VisionHardError(Exception):
    """视觉侧硬故障（P3 §3 逐字：``class VisionHardError(Exception)``）。

    触发情形（P3 §3/§6 第 1/2 步、§4 标定包校验、§7 P3-T12/T15/T22）：相机断开、
    帧读取失败或配对超时、整帧深度不可用、检测管线崩溃、init 阶段任何故障、
    configure/init 调用次序被破坏、``ignore`` 为负值或非整数（调用方程序错误）、
    标定包缺文件/缺键/非法值。一律交 ``terminate``（P1 §5：
    ``VISION_INIT_FAILED`` / ``VISION_HARD_ERROR``），**不**计入 NoTarget 连续计数。

    与 ``NoTarget`` 是同级兄弟类（都直接继承 ``Exception``），不可互相捕获。
    本桩的一切公开调用都抛它，消息明确"待队友实现"。
    """


def configure(thresholds: VisionThresholds, config_path: Path) -> None:
    """注入阈值快照与视觉配置文件路径（P3 §3 重述；桩阶段调用即抛 ``VisionHardError``）。

    契约（P3 §3/§4、P1 §4 步骤 10、D13）：
        * 由 ``main`` 冷启动段在 ``init()`` **之前**调用**恰好一次**；
          未 configure 即 init、重复 configure、init 之后再 configure 都是程序错误，
          抛 ``VisionHardError``（P3-T15）。
        * ``thresholds`` 是 P0 的 ``VisionThresholds``：P1 从**已加载并校验**的 motion
          profile 提取的六字段快照（target_bounds_m / object_envelope_m / clearance_m /
          approach_height_m / table_z_m / table_flatness_m），与运动同源同值；
          视觉不 import motion_params 自己去读 profile，也不复制第二份阈值来源。
        * ``config_path`` 是 ``cfg.vision_config_path`` 指向的视觉配置文件
          （``configs/vision.local.json``，schema 由 P3 §4 定义；P1 预检只做
          "文件存在且为合法 JSON"的 L7 收窄检查，其内部的模型/权重/标定包引用
          整体移交 P3 的 init 解析——P1 §3.2 第 4 条）。
          文件内相对路径相对**该文件所在目录**解析。
        * 检测阈值 ``detection_floor`` / ``conf_threshold`` 等属视觉参数，
          由配置文件承载，**不进入** ``VisionThresholds`` 公共目标类型（P3 §4）。
        * 本函数不做权重加载、不开相机（重活属 init）；P1 在调用前也不会因为
          本桩抛错而进入会话（三个 init 任一失败不得 ready）。

    桩阶段两个形参都不被读取：既不解析 ``config_path`` 指向的文件，也不校验
    ``thresholds`` 的字段，一律直接抛未实现错误。
    """
    raise VisionHardError(
        f"{_NOT_IMPLEMENTED}：契约 configure(thresholds: VisionThresholds, "
        "config_path: Path) -> None 待队友实现 —— 由 main 在 init 前调用恰好一次，"
        "重复调用或 init 后调用属程序错误，抛 VisionHardError（P3 §3/T15）"
    )


def init() -> None:
    """一次性加载模型与相机并验证可采集（P3 §3 重述；桩阶段调用即抛 ``VisionHardError``）。

    契约（P3 §3、P1 §4 步骤 10、§7 末段）：
        * 前置条件：``configure`` 已成功；未 configure 即 init → ``VisionHardError``
          （P3-T15）。
        * YOLO 权重**一次性**加载（不懒加载、不放在 import 时；P3-T17 要求 import
          阶段不开设备不加载权重，本桩同时满足两侧）。
        * 打开相机并验证可采集一帧；``source.mode = "replay"`` 时验证目录非空且
          首帧可读，**init 只验证不消费首帧**（P3 §4.1、P3-T23）。
        * 加载并严格校验标定包（内参/外参）；缺文件、缺键、非法值、模板 null
          用于 live 均属故障（P3 §4、P3-T22）。
        * init 阶段的任何故障抛 ``VisionHardError``，**不得伪装成 NoTarget**
          （P1 §4 步骤 10 括注原文）；由 main 映射 ``terminate("VISION_INIT_FAILED")``。
        * ready 之前完成权重、内外参哈希与相机/流身份的记录（P1 §7：另记日志）。
        * 只允许成功初始化一次；重复 init 属程序错误（与 ``configure`` 的
          "恰好一次"同一族不变量）。

    桩不加载权重、不打开相机、不校验标定包，直接抛未实现错误。
    """
    raise VisionHardError(
        f"{_NOT_IMPLEMENTED}：契约 init() -> None 待队友实现 —— 一次性加载权重、"
        "打开相机并验证可采集一帧（replay 只验证不消费）；init 阶段故障一律 "
        "VisionHardError，不得伪装成 NoTarget（P3 §3；P1 §4 步骤 10）"
    )


def get_target(ignore: int = 0) -> VisionInterface:
    """完整跑一次采集-检测-测量-过滤-排名，返回第 ``ignore+1`` 名候选（P3 §3 重述；桩即抛）。

    契约（P3 §3/§6，P1 §6.2/§6.4）：
        * **前置条件：机械臂在 home**（由 plan/运动 RETURN 维持）——相机视野固定，
          臂不在 home 时画面不可信。
        * **一次调用 = 一次完整采集-检测管线，无跨调用缓存**；本体内部多帧融合不受限
          （首版每调用一对帧，P3-T18）。每次 NoTarget 之间不加 sleep，三连调每次
          都是真实新扫描（P1 §6.2）。
        * 返回确定性排名第 ``ignore + 1`` 名的 ``VisionInterface``（六字段构造，
          position 为 float64 ``(3,)`` 副本）；排名键固定
          ``(-d_i, dist_to_roi_xy_center, x, y, z)``，禁随机数/时间戳/置信度 tie-break。
        * ``valid_count = len(ranked)``；**``ignore ≥ valid_count`` 必为 ``NoTarget``**
          （原因 ``skip_exhausted``）；无检出 → ``no_detection``；全被过滤 →
          ``all_filtered``。调用侧（P1 §6.4）还会复核 ``valid_count > ignore``。
        * ``ignore`` 必须为**非负 int**；负值或非整数是**调用方程序错误**，抛
          ``VisionHardError``（**不静默取 0**，P3 §3 末段）。
        * 相机离线、整帧深度不可用、推理崩溃、配对超时 → ``VisionHardError``；
          单个候选局部缺深度等"该候选被拒绝"属过滤，不是硬错误（P3 §6 末段、T04/T12）。
        * 每次调用（含 NoTarget 与 HardError）恰好一条 ``vision_scan`` 结论日志，
          经 ``qingyun.runlog`` 自动携带 task_instance_id/scan_id（P1 §3.4/§7、P3 §6 第 8 步）。

    桩**不**返回目标、**不**抛 ``NoTarget``：任何 ``ignore`` 取值（含 0、正数、负数、
    非整数）都只抛 ``VisionHardError``，并在消息里说明本函数的契约要点。
    """
    raise VisionHardError(
        f"{_NOT_IMPLEMENTED}：契约 get_target(ignore: int = 0) -> VisionInterface 待队友实现 "
        "—— 一次调用=一次完整管线、无跨调用缓存，返回确定性排名第 ignore+1 名；"
        "ignore 须为非负 int（负值/非整数属调用方程序错误，抛 VisionHardError 而非静默取 0）；"
        "ignore ≥ valid_count 才该抛 NoTarget，桩从未扫描，一律硬错误（P3 §3/§6）"
    )
