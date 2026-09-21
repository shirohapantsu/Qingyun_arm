# 最终审阅报告

## 1. Executive Summary

**结论：FAIL_NEEDS_FIX。** 当前版本不是可交付的最终版本。

本轮以当前工作区代码为对象，而非采用历史报告结论。电机驱动的基础协议码、离线超时处理、应答策略和连续反馈块读取都有实质性改进，专项离线测试也通过；但仍发现一个确认的核心 P1：抓取后各段在 `CHECK` 中不是按照实际的段间终点链式构造和验证。因此“开始动作前预规划、校验整条名义路线”的要求没有实现；真实路径可能在已经抓取、提起物体之后才首次被验证并失败。

另发现本次电机修改引入了一个 P2 接口回归：先前只实现 `write(data)` / `read(n)` 的自定义 transport 不能再被注入，尽管修改说明称新增可选 `timeout_s` 向后兼容。历史 P1/P2 修复大多仍在，但 P1-4、P2-4、P2-6 均不能继续标为完全稳定修复。没有真机、串口或实物验收证据，真机形态同样不得视为通过。

## 2. Review Scope

- 需求：`docs/机械臂运动控制模块技术文档.md`、`docs/顶抓姿态构造与R_top_down标定.md`、`docs/真机参数测量与标定指南.md`。
- 当前代码：`configs/`、`qingyun/grabbing/`、`scripts/calibrate.py`、vendor 运动学/电机表、配置和全部测试。
- 历史材料：`dev_logs/代码审查报告.md`、`dev_logs/修复报告.md`、`dev_logs/复审报告.md`。
- 最近修改：`电机修改.md` 与当前 `motor_control.py`、`test_motor_control.py` 的 diff。
- 实际验证：`compileall`、`git diff --check`、pytest 专项/模块测试。未运行真机 I/O、力矩使能、运动或实物验收。

## 3. Historical Issue Verification

| Issue ID | 历史状态 | 当前状态 | 验证依据 |
|---|---|---|---|
| P1-1 | VERIFIED_FIXED | VERIFIED | `safety.py` 桌面/障碍判断扣除了净距和平度；`test_safety_geometry.py` 29 passed。 |
| P1-2 | VERIFIED_FIXED | VERIFIED | `_sweep_inflation` 以基座系端点和关节轴计算；几何测试通过。 |
| P1-3 | VERIFIED_FIXED | VERIFIED | 接触豁免限制为抓取组件，腕和上游连杆仍检查。 |
| P1-4 | VERIFIED_FIXED | **REGRESSED / 不完整** | `arm_control.py:609-612` 对每个抓取后步骤均以同一 `q_ref` 构造；见 FINAL-P1-001。 |
| P1-5 | VERIFIED_FIXED | VERIFIED（离线） | 标定采集入口和硬件采集 Mock 回归测试所在 `test_calibration.py` 当前 42 passed。真机仍未验证。 |
| P2-1 | VERIFIED_FIXED | VERIFIED | 几何测试使用独立距离断言，未与实现共享判定。 |
| P2-2 | VERIFIED_FIXED | VERIFIED（离线） | 标定工具/共享换算测试当前通过。 |
| P2-3 | PARTIALLY_FIXED | PARTIAL | 仍有手工/启发式输出值；真机标定和人工复核前不可作为实测参数。 |
| P2-4 | VERIFIED_FIXED | PARTIAL | 公开原语仍在真实提交前设置 `_motion_started`，保留历史 N2。 |
| P2-5 | VERIFIED_FIXED | VERIFIED | 标定工具调用生产 gap 换算并拒绝外插。 |
| P2-6 | VERIFIED_FIXED | PARTIAL | `report.get("stages") or {}` 仍允许空阶段集合绕过 real-load 的逐阶段物理证据检查。 |
| P2-7 | VERIFIED_FIXED | VERIFIED | 偏差时从实测起点重建或拒绝不可重建的竖直段；对应回归测试通过。 |

历史 P3-10（会话级共享 fixture 改写）和 P3-12（空 `main.py` 与 README 入口描述不一致）仍存在，均为非阻塞问题。

## 4. Motor Modification Review

| 修改项 | 实现位置 | 验证结果 | Regression 风险 | 状态 |
|---|---|---|---|---|
| READ/WRITE/REG_WRITE/ACTION 指令码修正 | `motor_control.py:833-840` | 独立字节级报文测试通过；57 项电机测试通过。 | 未见当前仓内回归。 | VERIFIED |
| 按 Response_Status_Level 决定 WRITE ACK | `:874-888`, `:1493-1512` | Mock 覆盖等级 0/1；实际型号、固件与等级行为未实测。 | 实机不确定。 | PARTIAL |
| deadline 传至 pyserial 读写 | `:984-1006`, `:1017-1059`, `:1310-1331` | 慢读/慢写和底层异常的离线测试通过。 | 旧式 transport 注入已破坏，见 FINAL-P2-001。 | PARTIAL |
| 15-byte 连续反馈块 | `:744-768`, `:1436-1443`, `:1605-1617` | 当前 vendor 表确为 56..70；测试确认不把 Load 当 Current。 | 电流符号位和 mA 比例尚无真机证据。 | PARTIAL |
| 全量预校验与单帧 SYNC_WRITE | `:1563-1580`, `:1687-1749` | NaN/Inf/限位、零写入和六轴单帧测试通过。 | 未见仓内回归。 | VERIFIED |
| 无后台线程、`hold_current` 单读单写 | `:1663-1683` | 专项测试通过。 | 真机保持效果未验证。 | PARTIAL |

## 5. New Findings

### FINAL-P1-001

- **Severity:** P1
- **Category:** Requirement / Logic / Safety
- **问题描述：** 抓取后的完整路线没有在 `CHECK` 中按实际执行顺序进行链式预规划和校验。
- **对应技术文档：** 运动控制文档 §5.3 要求开始动作前预规划完整名义路线；§6.3 的 CHECK 要求预规划、校验整条名义路线。
- **涉及代码：** `qingyun/grabbing/arm_control.py:576-612`；实际执行路径在 `:620-627`。
- **根因：** `_build_post_grasp_steps()` 对 `lift`、`transfer`、`lower`、`retreat`、`return_home` 的每一个 `step.build()` 都传入同一个 `q_ref`。实际执行时，`_execute_step()` 却以该段开始时的实测位置调用 `build()`：例如 `transfer` 实际从 lift 终点开始，但 CHECK 校验的是 `q_ref → q_above_place` 的另一条路线。
- **复现 / 验证方式：** 静态追踪上述两处即可确定起点不一致；现有 `test_放置位姿不可达时在CHECK就被拒绝` 只覆盖放置目标不可达，未让“从 lift 终点到 transfer 终点”的真实路线与 `q_ref` 路线产生不同碰撞/限位结果，因而没有检出该错误。
- **实际影响：** 可能先完成张爪、下降、闭爪、提起，再在真实 transfer/lower/retreat 路线首次校验时失败并故障锁存。虽然运行时再次校验降低了碰撞发生概率，但违反了动作前拒绝的安全与行为契约，也会让物体在非预期位置悬停。
- **建议修复方式：** 在构建阶段按真实序列传播每段的终点：构造并校验 lift 后，以其 `end_joints` 作为 transfer 起点，依次传给 lower、retreat、return。对两个接触开度端点分别完成这条链；增加“仅真实链式 transfer 碰撞/越限”的回归测试，并断言 CHECK 零写入。

### FINAL-P2-001

- **Severity:** P2
- **Category:** Motor Control / Regression / Interface Compatibility
- **问题描述：** `ByteTransport.write/read` 的“可选 timeout_s 向后兼容”声明不成立。
- **对应技术文档：** 电机修改说明 §3 声称不会影响调用方；技术文档 §4.2 要求通信错误明确可控。
- **涉及代码：** `qingyun/grabbing/motor_control.py:772-798`, `:993`, `:1048`；修改说明 `电机修改.md:121-134`。
- **根因：** 协议层总是用关键字参数调用 `transport.write(packet, timeout_s=...)` 和 `transport.read(..., timeout_s=...)`。旧实现仅有 `write(data)` / `read(n)` 的 adapter 会直接抛 `TypeError`，并被包装成通信错误。
- **复现 / 验证方式：** 注入仅实现旧签名的 transport，调用 `StsProtocol(..., clock=lambda: 0).sync_write_targets(...)`；实际得到 `MotorCommunicationError`，其原因是 `TypeError: Legacy.write() got an unexpected keyword argument 'timeout_s'`。
- **实际影响：** 所有既有自定义串口 adapter、板级桥接或第三方测试替身都可能在第一条命令失败。这是本次修改引入的兼容性回归；仓内测试仅使用已更新的 FakeBus，未覆盖旧契约。
- **建议修复方式：** 将 timeout 支持放在明确版本化的新 transport 协议中，并保留旧协议适配层；或用能力检测后回退到旧签名（回退路径必须有明确的有限阻塞保证，不能静默取消 deadline）。补充旧签名 adapter 的兼容性回归测试。

### FINAL-P3-001

- **Severity:** P3
- **Category:** Calibration / Error Handling
- **问题描述：** real 模式的验收报告可通过删除/置空 `stages` 绕过逐阶段 physical 证据检查。
- **涉及代码：** `configs/motion_params.py:1285-1293`。
- **根因与影响：** 空字典不会进入循环。正常 promote 会检查八阶段，但手工篡改且哈希链被同步更新时，加载侧没有独立要求阶段全集。
- **建议：** real 加载侧强制 `stages` 恰好覆盖 `STAGES`，每阶段含至少一项 physical check。

### FINAL-P3-002

- **Severity:** P3
- **Category:** State Management
- **问题描述：** `move_joints` / `move_cartesian_top_down` 在 `_execute_step()` 的提交前校验之前将 `_motion_started` 置为 True，预校验碰撞失败也会锁存。
- **涉及代码：** `arm_control.py:178-183`, `:213-217`, `:620-626`。
- **建议：** 以首次成功 `submit()` 作为运动开始边界，或单独记录是否已发送总线命令。

### FINAL-P3-003

- **Severity:** P3
- **Category:** Error Handling
- **问题描述：** `load_calibration_profile()` 对 `model`、`motor`、`joints` 使用直接下标访问，残缺 JSON 泄漏 `KeyError` 而非带路径的 `ParamsError`。
- **涉及代码：** `configs/motion_params.py:1415-1434`。
- **建议：** 复用 `_require_keys()` / `_str()` 的受控错误路径。

### FINAL-P3-004

- **Severity:** P3
- **Category:** Calibration / Maintainability
- **问题描述：** `_fit_motion()` 把 `+50.0` 加速度启发式及多项固定运行参数写入拟合结果，来源没有逐字段 provenance。
- **涉及代码：** `scripts/calibrate.py:1569-1601`。
- **建议：** 将实测、人工复核、默认值分别标记；真机 promote 前拒绝未复核的手工项。

## 6. Regression Findings

发现一项经验证的 Regression：**FINAL-P2-001**，本次 `ByteTransport` 超时参数修改破坏了旧签名 adapter 的运行时兼容性。

除该项和 FINAL-P1-001 所揭示的 P1-4 未稳定满足外，未发现已通过测试的协议编码、同步写、连续块读取、限位换算或固定几何检查发生新的经验证回归。

## 7. Requirement Traceability Matrix

| Requirement ID | 技术文档要求 | 实现位置 | 测试 / 证据 | 当前状态 |
|---|---|---|---|---|
| R-01 | 单线程、同步、固定目标的抓取—放置入口 | `arm_control.py` | 抓取流专项用例 | VERIFIED |
| R-02 | TCP/物体中心、顶抓姿态及闭爪后放置变换 | `kinematics_ext.py`, `arm_control.py` | `test_kinematics_math.py` 45 passed | VERIFIED |
| R-03 | CHECK 在动作前校验完整名义路线；闭爪后按实际 g 更新 | `arm_control.py` | 静态链路审阅；现有测试覆盖不足 | INCORRECT |
| R-04 | 关节/笛卡尔轨迹、限位、速度/加速度与 FK 残差 | `trajectory.py`, `safety.py` | 轨迹 20 passed、安全几何 29 passed | VERIFIED |
| R-05 | 反馈监控、停止、到位、故障锁存与一次保持 | `executor.py`, `arm_control.py` | 选定抓取回归用例通过；完整抓取流未重新跑完 | PARTIAL |
| R-06 | 驱动限位、错误分类、读写总时限、无缓存反馈 | `motor_control.py` | 电机专项 57 passed | PARTIAL |
| R-07 | 0/1 ACK 策略、连续块反馈和物理量换算 | `motor_control.py` | Mock 验证；真实设备/符号位未验证 | PARTIAL |
| R-08 | 标定状态、哈希链、实机证据关卡 | `motion_params.py`, `calibrate.py` | 标定 42 passed；空 stages 缺陷仍在 | PARTIAL |
| R-09 | 标定指南的实机精度、时限、抓取成功/掉落/损伤验收 | 标定流程和配置 | 无硬件及实物试验 | UNVERIFIED |
| R-10 | 视觉为外部职责，控制不重选目标 | 公共接口和 `arm_control.py` | 接口/流程审阅 | VERIFIED |

## 8. Build Results

- `conda run -n Qingyun python -m compileall -q configs qingyun scripts libs tests`：**PASS**。
- `git diff --check`：**PASS**，未发现补丁空白错误。
- 本项目为 Python；没有 C/C++ 编译、Sanitizer 或链接步骤。
- 观察到 `hppfcl` 的 `coal` 迁移弃用 warning，来自依赖链；本轮未发现项目代码编译 warning。

## 9. Test Results

本轮实际执行并通过：

| 范围 | 结果 |
|---|---|
| `tests/test_motor_control.py` | 57 passed |
| `tests/test_motion_params.py` | 52 passed |
| `tests/test_trajectory.py` | 20 passed，1 个依赖弃用 warning |
| `tests/test_safety_geometry.py` | 29 passed，1 个依赖弃用 warning |
| `tests/test_kinematics_math.py` | 45 passed，1 个依赖弃用 warning |
| `tests/test_calibration.py` | 42 passed |
| 3 个关键 `test_grasp_flow` 回归用例 | 3 passed，1 个依赖弃用 warning |

合计 **248 次通过的测试执行**（其中 3 项属于抓取流的局部选择）。全量收集数为 289。完整 `tests/` 与完整 `test_grasp_flow.py` 在本环境单命令 30 秒执行限制内未能完成，故剩余抓取流组合、全量集成结果标为 **UNVERIFIED**；不采用历史报告的“289 passed”替代本轮证据。

此外，现有测试未覆盖 FINAL-P1-001 的真实链式后抓取路径，也未覆盖旧 `ByteTransport` 签名兼容性。

## 10. Static / Dynamic Analysis

- 已执行 Python 字节码编译和 diff whitespace 检查，均通过。
- 未配置可运行的 ruff、flake8、mypy、clang-tidy、cppcheck 或 Sanitizer 流程；这些项为 UNVERIFIED。
- 未进行真机动态分析。Mock 不能证明串口时延、固件 ACK 行为、电流符号、物理停止效果或实物抓放指标。

## 11. Remaining Risks

1. FINAL-P1-001 修复前，动作前全路线安全判定不成立。
2. 新驱动的协议码在 FakeBus 上通过，但实机 Response_Status_Level、串口时限、块读跨度、断线和电流符号位均未验证。
3. 当前配置是 simulation；八阶段实机标定、哈希绑定后的物理验收及成功率/损伤率试验均未完成。
4. 全量抓取流在本轮未执行完，不能以局部成功代替端到端覆盖。
5. P2-3 / FINAL-P3-004 中的人工/启发式参数未带足够 provenance，必须在实机使用前测量或人工确认。

## 12. Issue Summary

| Severity | 数量 |
|---|---:|
| P0 | 0 |
| P1 | 1 |
| P2 | 1 |
| P3 | 4 |

**Blocking Issues**

- FINAL-P1-001：完整后抓取路线未在 CHECK 中按真实链式路径预校验。
- 真机关键功能与验收指标均为 UNVERIFIED；不得用于真机交付。

**Non-Blocking Issues**

- FINAL-P2-001、FINAL-P3-001 至 FINAL-P3-004，以及历史 P3-10/P3-12。

## 13. Final Delivery Decision

### FAIL_NEEDS_FIX

必须先修复 FINAL-P1-001，并增加能区分“错误的 `q_ref` 路径”和“真实链式路径”的 CHECK 零写入回归测试。随后修复或明确版本化 FINAL-P2-001 的 transport 契约，完整运行 289 项离线测试。

真机交付还必须完成标定指南要求的物理标定、串口/电流/ACK/停止验证和实物抓取验收；在这些证据产生前，真机形态不得标为 PASS。
