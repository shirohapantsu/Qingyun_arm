# upper 开发前方案初审

> 日期：2026-09-22。审查对象：`docs/` 下全部方案文档（上层契约、视觉契约、完整落地方案、P0–P5 实施文档、运动文档、姿态数学、标定指南）与现有源码（`configs/`、`qingyun/grabbing/`、`tests/`、`scripts/`、`prompt.md`、`calibration/`）。
> 裁定原则（用户 2026-09-22 明确）：**readme 与文档/实际代码冲突时，以文档和实际代码为准**。readme 的过时内容不构成对文档的否定，只列为 readme 待同步项。
> 结论先行：P0–P5 对源码的事实陈述经逐条复核**基本全部属实**；但有 **4 处阻断级逻辑问题**（集中在 P1 冷启动顺序、mock 装配、P1 配置示例）和若干文档间未同步项，建议在派发编码前修订 P1/P5 相关章节。

## 一、结论摘要

| 类别 | 数量 | 阻断级 |
|---|---:|---:|
| 文档 ↔ 源码冲突（P0 已登记，复核属实） | 3 | 0（P0 即为修复方案） |
| 文档 ↔ 源码冲突（本次新发现） | 2 | 0 |
| 文档间冲突 / 决策未同步回源文档 | 4 | 0 |
| 未实现功能的逻辑问题 | 9 | 4 |

## 二、文档 ↔ 源码冲突

### 2.1 P0 已登记、本次复核属实的冲突（不重复展开，仅确认证据）

| # | 冲突 | 代码证据 | 文档证据 |
|---|---|---|---|
| C1 | `VisionInterface` 已迁六字段，但运动入口仍强制 `grade` 存在且为 str，新对象必判 `INVALID_INPUT` | [arm_control.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/arm_control.py#L406-L441)：L412 字段循环含 `"grade"`，L438-440 类型检查 | 落地方案 §1.1、P0 §2 如实登记；**视觉契约 §一"运动侧运行代码零引用 grade"一句是错误的，P0 修订文档时应一并删除该句** |
| C2 | 测试仍按旧接口构造 `grade=...`，对冻结 dataclass 直接 TypeError；含"grade 非字符串拒绝"用例 | [support.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/tests/support.py#L75-L93) 5 行、[test_grasp_flow.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/tests/test_grasp_flow.py#L160-L164) 16 行（含 L160-164 拒绝用例） | P0 §2"约 18 处"与口径有关，实际含 grade 的行共 21 行，实施时以全仓搜索为准 |
| C3 | `prompt.md` 仍列 blueberry 方案，与 D07"本期仅 strawberry/invalid"冲突 | [prompt.md](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/prompt.md) L9-10 蓝莓方案段 | P0 §4 已含收窄任务，属待办非错误 |

### 2.2 本次新发现的文档 ↔ 源码冲突

| # | 冲突 | 证据与影响 |
|---|---|---|
| C4 | **标定指南附录 A.2 第 1 条自相矛盾且与现状不符**：称 profile"现为 `verified` 但 `verification: null`，两种模式都加载不了"；实际 [profile.json](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/calibration/REAL_ARM/PC/profile.json) `status="draft"`，且同文附录 A.4 第 1 条已写"`status` 已为 `draft`"。 | A.2 是撤回前快照未更新。后果：读者按 A.2 会误判"当前连采集链都不通"。建议修订 A.2 第 1 条为"已退回 draft"。 |
| C5 | **readme 多处过时**（按裁定以文档/代码为准，此处仅登记 readme 待同步项）：§3 接口表仍写 `VisionInterface(position, yaw_deg, grade)`、品级 `A+/A/B/C`；工程结构仍列 `watchdog.py`（P1 改为 `shutdown.py`）；行数统计（motor_control 1966 / arm_control 949）与实际（2104 / 1016）不符。 | [readme.md](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/readme.md) L111-126、L45-61、L13-26。P0 §4 已含"README 同步"，建议把这三处明确列入 P0  checklist，避免只同步接口一节。 |

## 三、文档间冲突 / 决策未同步

| # | 冲突 | 说明 |
|---|---|---|
| D-A | **D08 未同步回两份源契约**：[落地方案](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/docs/上层视觉运动完整功能落地方案.md) §7.4 恢复矩阵仍写"IK_FAILED/PLAN_INVALID 且 recovery_required=True、holding=EMPTY → 可自动 reset 回 home"；上层文档 §6 解锁流程同样未区分运动前后。D08 与 P1 §6.5 已定稿为**一律 terminate**。 | 落地方案头注有"以实施文档为最终规格"的总声明，但正文矩阵与之直接矛盾，编码智能体若误读源契约会实现错误分支。建议在落地方案 §7.4 与上层文档 §6 各加一行"D08 已废止本行"。 |
| D-B | **尺寸过滤表述不一致**：落地方案 §6.3 写"按固定 yaw_offset/R_TOP_DOWN 约定计算**对应投影尺寸**"；D10 与 P3 §6 冻结为**逐轴比较**（`length_m ≤ envelope[0]` 且 `width_m ≤ envelope[1]`）。两者只在 yaw_offset∈{0°,90°} 时等价（见 L1）。 | 以 D10/P3 为准，落地方案 §6.3 该行需修订，否则验收口径不清。 |
| D-C | 落地方案 §7.4"人工确认建议要求明确输入 `已空爪`"已被 D13 否决（回车即确认），正文未标注。 | 同 D-A，加废止标注。 |
| D-D | 上层文档 §11 待办清单中"`common_interface.py` VisionInterface 改造"实际已完成（数据类已迁移），未勾选；"places 标定"等状态与 P4 文档有重叠分工。 | P0 §4 负责同步原契约，建议把 §11 勾销纳入 P0。 |

## 四、未实现功能的逻辑问题

### 4.1 阻断级（建议冻结派发前修订）

**L1（P1 §4 vs §3.2 自相矛盾）：预检"零舵机写入"承诺与冷启动顺序不可兼得。**
P1 §3.2 声明预检失败 → terminate，"此时未连串口、零舵机写入"；但 §4 把 `preflight` 放在步骤 7，而步骤 5 `Sts3215MotorController(params)` **构造即连接并自动 `initialize(enable_torque=True)`**（[motor_control.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/motor_control.py#L1426-L1499) 默认 `initialize=True`），此时已发生 Torque_Enable 写入。§4 注解允许"依赖 params 的检查置于步骤 5 之后"同样破坏承诺。实际上 `load_motion_params` 在步骤 4 已完成，全部预检项（注册表、place_id、K、文件、key）都**不依赖 motor**，可以完全移到步骤 5 之前。**建议：P1 §4 顺序改为 preflight 紧随步骤 4；同时澄清 mock 侧 motor 是否也统一调 `initialize()`（real 侧构造已自动完成，main 再调一次会重复执行约 36 次事务，幂等性未验证）。**

**L2（P1+P5 组装矛盾）：mock 模式的 motor 替身无合法来源。**
P1 §2 规定"生产模块禁止导入 tests"，但 mock 分支需要的 `MockMotorController` 只存在于 [tests/mock_motor.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/tests/mock_motor.py)（且其构造签名是 `(params, time: SimTime, ...)`，比 real 侧多一个必需的 `SimTime`，P1 §4 步骤 5 的 `MockMotorController(...)` 未交代 SimTime 由谁创建）。P5 §2 的 `mock_runtime.install()` 只替换 asr/cloud_model 函数，未覆盖 motor 构造；P5 §3 验收命令又要求 `python3 main.py --mode mock` 可运行。**建议三选一并写入 P1/P5：①MockMotor 移入 `qingyun/`（如 `qingyun/mock_motor.py`）；②`mock_runtime.install()` 增加 motor 工厂替换；③main 的 mock 分支改为接受外部注入的 motor 实例。**

**L3（P5/P2 缺口）：mock 模式下 `asr.init()` / `cloud_model.init()` 行为未定义。**
P5 §2 `install()` 只替换 `listen_and_transcribe` / `select_plan`，main 随后仍会调用真实 `init(cfg)`：P2 的 init 要加载 VAD 权重、打开麦克风、读 secrets——与 mock "不联网、不用真麦克风、secrets 不需要"的目标直接冲突；P2-T13"key 缺失 → init 抛 HardError"也未区分模式，mock 下无 key 会让 init 必炸。**建议：在 P2 §2 明确"init 感知 `cfg.mock is not None` 时跳过设备/权重/密钥加载"，或在 P5 §2 明确 install 一并替换 init 为 no-op。**

**L4（P1 §3.1 文档内部 bug）：`app.example.json` 全部相对路径与自身解析规则矛盾。**
规则是"相对路径相对 app.json 所在目录解析"，而示例位于 `configs/`，却写 `"profile_path": "calibration/REAL_ARM/PC/profile.json"`、`"prompt_path": "prompt.md"`、`"vision_config_path": "configs/vision.local.json"`、`"log_dir": "logs"`——按规则分别解析为 `configs/calibration/...`、`configs/prompt.md`、`configs/configs/...`、`configs/logs`，**全部不存在**，照抄示例必然预检失败。应改为 `../calibration/...`、`../prompt.md`、`vision.local.json`（同目录）、`../logs` 等。P5 的 `app.mock.json` 同样需要注意。

### 4.2 需在编码前澄清的逻辑问题（非阻断，但影响验收）

**L5 尺寸过滤与运动包络的隐含前提：yaw_offset 必须是 0°/90°。**
运动侧按 `close_axis_width_at_yaw(L,W,yaw_offset)=|cosβ|·L+|sinβ|·W` 定预张与包络（[kinematics_ext.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/kinematics_ext.py#L118-L125)），且 P0 迁移后**运动完全不校验目标尺寸——视觉过滤是唯一防线**。P3 的逐轴检查只在 β∈{0°,90°} 时与投影公式等价。现状恰好满足（real profile `yaw_offset_deg=90.0`，sim profile `0.0`），但这是**无代码把守的巧合**：P4 若重新标定出中间角度，视觉会放进闭合轴投影超宽的目标，运动侧持物包络仍按 config envelope 建模，误差无任一层的拦截。另注意 sim/real 的 yaw_offset 差 90°，mock 全链跑通的闭合轴语义与真机不同。**建议：在 P0 VisionThresholds 组装或 P1 预检中断言 `yaw_offset_deg ∈ {0, 90}`（或显式换用投影公式），并在 P4 标定规程中加"修改 yaw_offset 必须回审 P3 尺寸过滤"。**

> **2026-09-22 复审更正（复审报告 §4）**：本条"仅在 0°/90° 等价"的论证不成立——若 `L ≤ E_L` 且 `W ≤ E_W`，则任意偏角 β 都有 `|cosβ|·L+|sinβ|·W ≤ |cosβ|·E_L+|sinβ|·E_W`，逐轴包含本来就是"投影不超包络投影"的**充分条件**（初审混淆了"等价"与"保守充分条件"）；且 preopen 由同一 envelope+yaw_offset+误差带派生（`calibrate.py::_derive_preopen_pct`）。`{0,90}` 断言经用户确认后**保留**，但定性改为"本期已标定姿态范围的产品限制"；真正需要把守的不变量是：视觉过滤与运动/preopen 使用同源同值的 envelope（VisionThresholds 注入已保证）+ 修改 yaw_offset 后重新派生 preopen 等下游参数。

**L6 近圆形草莓的 yaw 语义缺口。**
落地方案 §6.2 只对蓝莓提出长轴退化问题，但草莓同样近圆：OBB 的 w≈h 时长轴角噪声大、随帧抖动，而 P3 §8 仍按 CAL-041" yaw ≤5°"验收（只说"近圆形样本重点抽查"），未定义退化约定。后果：近圆样本系统性超差 → CAL-041 无法通过，或被迫放宽成走过场。且姿态文档 §2 明确 θ 与 θ+180° 对单动指夹爪接触几何不同，yaw 抖动有真实物理后果。**建议 P3/P4 补充：定义圆度阈值（如 L/W < 1.15 视为退化），退化样本 yaw 输出约定值并在 CAL-041 中单独口径验收。**

**L7 P1 预检第 4 项依赖 P3 尚未定义的 schema，阶段先后倒置。**
P1 §3.2 要求预检"vision_config 内部引用的模型/标定文件存在"，但 vision.json 的键名（`model.path`、`calibration.*`）归 P3 §4 定义，P1 先冻结。P3 §9 只说 schema 变更同步 `app.example.json` 注释，未提同步预检。**建议：P1 预检收窄为"vision_config_path 存在且为合法 JSON"，引用文件存在性检查整体移交 P3 init（P3 本来就有严格校验）；或在 P1 冻结一个最小键集合契约。**

**L8 ignore 封顶"正常出口"可能阻塞在人工确认上。**
P1 §6.2 exhausted 分支先 `_recover(result)` 再 `_finish`：若压垮封顶的那次失败是 GRASP_UNCERTAIN/GRIP_SLIP，恢复路径是阻塞 `input()` 人工确认。即"任务完成"这一正常出口可能挂起等人——与无人值守 demo 的预期有张力。逻辑本身自洽（不能带着未清现场回 WAITING），但 P5 现场操作规范应写明"任务完成前可能需要一次人工确认"，避免验收时把挂起误判为死机。

### 4.3 次要问题（记录备查，不阻塞）

- **L9 终端"任务完成"重复打印**：P1 §6.6 `_finish` 已输出"任务完成：原因=…"，§5 主循环在 `plan.run()` 返回后再 `runlog.console("任务完成")`，会出现两行。建议主循环不再打印，统一由 `_finish` 输出。
- **L10 K 拟合未定义分布重叠行为**：P4 §2.4"取人工 A 组下界与 B 组上界之间的保守分界"——两组重叠时无此区间，工具行为未定义，建议明确"重叠即拒绝并报告样本"。
- **L11 P3 `below_table` 判别复用 `table_flatness_m`**：该字段语义是桌面平整度（桌面采样点最大偏离），被借用作"物体 vs 桌面"判别阈值，语义混用；当前量级下无害，建议在 P3 注释中说明借用理由。
- **L12 LLM"重复 task 键"违约检测**：`json.loads` 默认后键覆盖前键，检测重复键需 `object_pairs_hook` 自定义解析，P2 §4 未提示该实现手段，执行智能体可能漏检。
- **L13 runlog 初始化前的失败无 JSONL**：`load_app_config` 失败时 runlog 尚未 init，terminate 只有终端输出。可接受，建议 P1 明确说明该窗口。

## 五、已核对属实的关键声明（防误报清单）

以下文档声明已逐条对照源码/数据验证**无误**，编码时可直接采信：

1. 五个 0 字节占位文件（`main.py`、`asr.py`、`cloud_model.py`、`watchdog.py`、`vision.py`）属实。
2. `load_motion_params(profile_path, *, mode)`：real 要求 `status=verified`，mock 要求 `simulation`，draft 两边都拒——属实。
3. P0 的 `VisionThresholds` 六个来源字段（`workspace.target_bounds_m/table_z_m/table_flatness_m`、`grasp.object_envelope_m/approach_height_m`、`collision.clearance_m`）在 [motion_params.py](file:///home/shiroha_pantsu/learn/labroratry/Qingyun_arm/configs/motion_params.py) 全部存在且同名。
4. 真机 profile：`status=draft`、`verification=null`、places 仅 `default/bin`、**bin 中心 [0.34,-0.06,0.024] 确在 target_bounds [[0.325,-0.07,0.02],[0.4,0.07,0.028]] 内**、`target_position_error_bound_m=0.004`（D09 按 4mm 统一正确）、`target_yaw_error_bound_deg=5`、`min_grasp_trials=30`——全部属实。
5. `Sts3215MotorController.initialize(*, verify_identity=True, enable_torque=True, timeout_s=...)` 签名与 P1/落地方案引用一致；构造默认自动连接+初始化。
6. `ArmController.__init__` 的 `clock/sleep/clock_ns` 均有默认值，main 可 `ArmController(motor, params)` 直接构造；`reset_fault` 确认不自动回零；`KeyboardInterrupt`→`ABORTED` 属实；`FAULT_UNCONTROLLED` 确为 stage 字符串；故障时 stage 被改写为 FAULT_HOLD/FAULT_UNCONTROLLED（原始阶段丢失，落地方案 §8 的日志条款已正确虑及）。
7. `open_gripper/move_joints/reset_fault` 失败抛异常（非返回码）；`wait_settled` 超时返回 `False`（执行器内部段末转 TIMEOUT）；未知 place_id → `CONFIG_INVALID` 且臂不动——与 P1 恢复矩阵/冷启动的异常处理设计匹配。
8. `scripts/calibrate.py` 含 `promote` 子命令、`scripts/read_workcell_waypoint.py` 存在，P4 的复用声明成立。
9. `prompt.md` 含"JSON"字样与 `{"task": "invalid"}` 示例，满足 DeepSeek JSON mode 前提（P2 §4 声明属实）。
10. `MockMotorController` 目前**无** `initialize()`，P1"补协议对等桩"的任务描述准确。

## 六、处置建议（按归口）

| 优先级 | 事项 | 归口 |
|---|---|---|
| 派发前必改 | L1 预检顺序前移、L2 mock motor 来源、L3 mock init 语义、L4 示例路径 | 修订 P1（L1/L4）、P1+P5（L2）、P2+P5（L3）后重新冻结 |
| 派发前宜改 | D-A/D-B/D-C 源契约废止标注；C4 标定指南 A.2 修订；L5 预检断言；L6 yaw 退化约定；L7 预检收窄 | P0 范围文档同步 + P1/P3/P4 小节修订 |
| 编码中注意 | L8 写入 P5 操作规范；L9–L13 列入对应阶段 checklist；C5 readme 三处同步列入 P0 | 各阶段文档 checklist |
| 已就绪 | 第五节全部声明 | 无需动作 |

## 七、处理决定（2026-09-22 用户逐条确认）

> 本节为用户对第三、四章全部问题的逐项裁定，优先级高于前文"处置建议"列。其中 L8 构成对既有契约的**新决策（记为 D14）**，推翻原 IGNORE_EXHAUSTED 正常完成出口。

### 7.1 阻断级问题

| 项 | 用户决定 | 修订范围 |
|---|---|---|
| L1 预检顺序 | **预检整体前移**：preflight 紧随 `load_motion_params`（步骤 4），全部检查在 motor 构造前完成，"零舵机写入"承诺按此成立；同时澄清 main 不再对已自动初始化的 motor 二次调用 `initialize()` | P1 §4 冷启动顺序、§3.2 注解、P1-T18 |
| L2 mock 替身 | **取消 main 的 CLI 参数功能，单独构建 mock 入口**：main 不再提供 `--mode/--config`；mock 由独立入口承担，该入口自由组装 MockMotor/SimTime/替身，不受"生产模块禁止导入 tests"约束 | 推翻 P1 §3.1"mode 唯一来源是 CLI"、§4 步骤 1；P5 §2 mock_runtime、§3 验收命令 `python3 main.py --mode mock` 全部改写为独立入口形态；D13 中"main CLI 纳入"的决定随之作废 |
| L3 mock init | **mock 跳过这些环节**：独立 mock 入口不执行真实 asr/cloud_model init（不加载 VAD、不开麦克风、不读密钥），由替身直接接管业务函数；P2 的 init 语义不必为 mock 修改 | P5 §2 install 语义；P2 仅需一句"mock 入口不调 init"的说明 |
| L4 示例路径 | **修正示例路径，并将模板文件名中的 example 去掉**：模板改为 `configs/app.json`、`configs/secrets.json`、`configs/vision.json`（提交空值/占位）；实际本地文件为 `app.local.json`、`secrets.local.json`（gitignore）、`vision.local.json`；示例内路径按"相对该文件所在目录"规则改写为可真实解析的形式 | P1 §3.1/§3.2、P3 §4（vision.example.json 引用同步改名）、P5 §2 |

### 7.2 编码前澄清项

| 项 | 用户决定 | 修订范围 |
|---|---|---|
| L5 尺寸过滤 | **预检断言**：P0 VisionThresholds 组装或 P1 预检中断言 `yaw_offset_deg ∈ {0, 90}`，违约拒绝运行；P4 标定规程加"修改 yaw_offset 必须回审 P3 尺寸过滤"。**依据经复审 §4 更正**：断言定性为"本期产品限制"，非几何学必然（逐轴包含对任意偏角已是充分条件） | P0/P1 预检条款、P4 §3 |
| L6 近圆 yaw | **验收剔除近圆**：不改视觉管线；P3 §8 与 P4 CAL-041 规程写明"近圆样本（需定义剔除口径，如 L/W 小于某阈值）不参与 5°  yaw 验收，仅验明显长形样本" | P3 §8、P4 §2.3/§3 |
| L7 预检倒置 | **收窄预检**：P1 预检仅查 vision_config 存在且为合法 JSON；模型/标定引用文件检查整体移交 P3 init | P1 §3.2 第 4 条 |
| L8 封顶出口 | **新决策 D14：ignore 封顶判定为无可抓取目标，走 terminate 任务失败流程（臂冻结），不再是恢复后正常完成**。"任务完成"正常出口只剩 NoTarget×3；封顶 terminate 不做恢复、不做人工确认、不回 home，现场处置由操作规范承担 | **推翻**：上层契约 §5 终止条款/§6 分发表、视觉契约 §五、落地方案 §7.3；**改写**：P1 §6.2 分发顺序（exhausted 分支）、§6.5 恢复矩阵、§9 P1-T06/T19、P5 E2E-02/E2E-12、实施文档 README 决策记录（追加 D14 并标注 D02 语境中"候选先跳尽仍按原封顶规则结束"失效） |

### 7.3 文档同步与次要项

| 项 | 用户决定 |
|---|---|
| D 类（决策未同步） | **全部修订**：落地方案 §7.4 加 D08/D13 废止标注、§6.3 尺寸表述按 D10 改为逐轴；上层文档 §6 恢复流程加 D08 废止标注、§11 已完成待办勾销；标定指南附录 A.2 第 1 条修订为"已退回 draft"的事实状态 |
| C5 readme | **列入 P0**：grade 接口表、watchdog、行数统计三处明确写入 P0 文档同步 checklist，随 P0 一并修订 |
| L9 | **采纳**：主循环不再重复打印"任务完成"，统一由 `_finish` 输出（修订 P1 §5） |
| L10 / L11 / L12 / L13 | **不采纳**：K 拟合重叠、below_table 借用 flatness、重复键解析提示、runlog 前窗口均不写入规格，维持现状 |

### 7.4 D14 决策登记（待同步至实施文档 README）

| 编号 | 事项 | 决定 | 影响阶段 |
|---|---|---|---|
| D14 | ignore 封顶（`ignore ≥ valid_count`）的出口语义 | 判定无可抓取目标 → terminate 任务失败流程，臂冻结于当次失败位；不恢复、不人工确认、不回 WAITING；正常完成出口仅余 NoTarget×3 | P0/P1/P5（源契约同步、run() 分发、测试场景） |
