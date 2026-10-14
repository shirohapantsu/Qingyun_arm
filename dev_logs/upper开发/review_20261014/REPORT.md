# upper 独立代码审查（2026-10-14）

> 修复状态：R01–R11 已完成代码修复，最终全量测试 1264 passed。修复后定位和验证见[开发日志](../upper审查修复_20261014_开发日志.md)。以下正文及 reproduce.py/results.jsonl 保留修复前审查记录。

结论：发现 11 项可复现的问题，其中 R09 是开发前已有、被 upper 发布工具继续依赖的加载器缺口。现有测试通过不能支持“真实语音与标定发布链路已经可用”的结论。优先修复真实音频适配器、抓放误差基准和验收漏检。

## 审查范围与方法

- 基线为 `0a050b9`，审查 `feature-upper` 当前工作区，包括 `d1b5195` 后尚未提交的 P5 工具和测试。
- 实际阅读入口/配置、plans 状态机、shutdown/runlog、ASR/LLM、驱动启动握手、运动 trace、标定导出/拟合/落点工具、部署检查及相关测试。日志与文档用于确认约定，不作为实现正确性的证据。
- 以下发现均附实际函数调用复现。第三方协议和音频格式另外核对官方源码/文档；未向云服务发请求，未连接机械臂、相机或麦克风。
- 只新增本目录的报告与复现材料，未修复生产代码，未改动用户已有工作区修改。

## 验证结果

| 验证 | 结果 |
|---|---|
| app_config / plan_flow / shutdown / runlog / asr_flow / cloud_model / deployment_check / grade_threshold_fit / place_calibration | 首轮 648 passed、8 setup errors；8 项本地服务测试受沙箱端口绑定限制 |
| ASR/LLM 的 `-k T17` 定向复测（允许本机临时端口） | 8 passed、240 deselected；消除上述环境限制，未调用真实云服务 |
| motor_control / motion_trace / calibration_loader / calibration_tools / main_flow | 209 passed，245.44 秒 |
| 本次新增离线缺陷复现 | R01–R11 全部复现；R08 包含两个不同坏输入分支 |

合计覆盖 **865 个不同的现有测试，最终均通过**，并有既有第三方 hppfcl→coal 弃用警告。本次未跑完整 1217 项回归。复现脚本的断言确认的是“当前错误行为确实存在”，其退出 0 不代表功能通过验收。

复测命令（在项目根目录执行）：

```bash
/home/shiroha_pantsu/miniconda3/envs/Qingyun/bin/python dev_logs/upper开发/review_20261014/reproduce.py
```

逐项结果见 [results.jsonl](results.jsonl)。脚本使用真实生产函数、临时文件及小型设备/模型替身；部署树与样本构造复用了已有测试夹具，不修改被审查的校验函数。

## P1：应在真实功能验收/发布前修复

### R01：麦克风 drain 把“没有溢出”当作“继续清空”的条件，正常输入会一直阻塞

位置：[asr.py:347](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:347)，调用方 [asr.py:1047](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:1047)。

`while self._raw.read(512)[1] == False` 的第二个返回值是 overflow 标记。正常及时读取通常一直返回 False；缓冲区读空以后，阻塞式 read 等待新的音频，再继续读。于是程序不会进入 `_capture`，且耗时没有上限。只有出现溢出或异常才可能退出，异常还被吞掉。

复现：输入源 `read_available=0`，每次阻塞读提供新数据且 `overflowed=False`；子进程进入 drain 后在 0.6 秒限时内始终不返回。持续正常源会继续同样循环。sounddevice 官方源码确认 read 为阻塞读取，overflowed 不代表“仍有旧数据”。[官方源码](https://python-sounddevice.readthedocs.io/en/0.5.3/_modules/sounddevice.html)

建议：按进入 drain 时的可读帧数做有界清空，或使用可明确丢弃缓冲的流生命周期操作；不要用 overflow 标志作为结束条件。补“没有积压、没有溢出”的适配器测试。

### R02：RawInputStream 的字节缓冲没有按 int16 解码

位置：[asr.py:355](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:355)。

流配置为 `dtype="int16"`，但 `read()` 返回 `list(data)`。这不是 PCM 样本解码。官方实现返回 `_ffi.buffer(data)`；直接迭代缓冲仍是按字节处理，不会恢复带符号的 16 位样本。字节对象分支还可能在后续 `float(s)`/降混时直接抛异常。

复现：以同样的 Python buffer 接口传入三个 PCM 样本 `[0,256,-1]`，适配器输出六个字节值 `[0,0,0,1,255,255]`。样本数、幅值及符号全部错误。[官方 RawStream 源码](https://python-sounddevice.readthedocs.io/en/0.5.3/_modules/sounddevice.html)

建议：按实际 PCM 格式用 `np.frombuffer(..., dtype=np.int16)` 或等价方式解码，再按声道拆分；使用与真实 raw buffer 一致的测试输入，而非提前解码好的整数列表。

### R03：Silero VAD 收到的是 int16 原始幅值，缺少归一化

位置：[asr.py:425](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:425)。

`torch.FloatTensor([float(s) for s in frame])` 只转换类型，没有把 PCM 幅值转换到模型所需的归一化浮点音频尺度。当前录音与 WAV 编码链按 int16 幅值工作，因此仅修复 R02 后，这个错误仍会存在。

复现：模型探针实际收到的峰值是 `32767.0`；按 int16 满量程换算应约为 `0.9999695`。未声称执行了真实模型推理；模型输入尺度已由探针确定，官方麦克风示例也在送入 VAD 前先做 Int2Float 归一化。[Silero 官方麦克风示例](https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py)

建议：VAD 输入单独归一化为 float32 音频，WAV 输出保留正确的 PCM 尺度；用模型输入探针以及真实静音/口令音频验证。

### R04：DashScope 请求和响应结构与所调用线路不一致

位置：[asr.py:673](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:673)、[asr.py:758](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/asr.py:758)，配置 [app.json:6](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/configs/app.json:6)。

当前配置选择 DashScope 的 `/api/v1/services/aigc/multimodal-generation/generation`，请求却构造 `type/input_audio/data` 内容对象，这属于另一种兼容接口的结构。官方 DashScope 同步示例使用内容项的 `audio` 字段；官方消息响应的文本位于 `output.choices[0].message.content[*].text`。当前解析器只接受 `output.text`，会拒绝官方正常消息响应。[阿里云官方 API](https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference)

复现：生产请求构造器产物没有 `audio` 键；把官方消息响应形态交给生产解析器，会抛 `bad_payload: 响应缺 output.text 字段`。

建议：明确选择 DashScope 原生接口或兼容接口，统一 URL、请求体、参数与响应解析，并固定验证过的模型版本。官方核对样例为 qwen3-asr-flash；仓库模板中的 qwen-audio-3.1-asr-flash 尚没有真实调用证据，不能把任一模型名替换当作协议已验证。补官方形态的 HTTP fixture；已有本地 wire 测试只证明自定义协议的发送一致性。

### R05：放置误差计算用抓取位置/方向作为参考，正确放置也会验收失败

位置：[record_pick_place_trials.py:314](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/record_pick_place_trials.py:314)、[record_pick_place_trials.py:342](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/record_pick_place_trials.py:342)；同样错误基准被 [calibrate.py:2637](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/calibrate.py:2637) 的复算接受。

样本 `target_position_m/target_yaw_deg` 取自 vision，即抓取前目标位姿。误差随后用它与 `actual_place_position_m/actual_place_yaw_deg` 比较，完全未读 `params.places[place_id]`。这个函数甚至没有接收 params。运动执行实际使用的是配置落点，故验收参考与控制目标不同。

复现：抓取位置 `[0.35,0,0.024]`；配置落点与实际落点均为 `[0.32,0.055,0.024]`，配置与实际放置 yaw 均为 30°、抓取 yaw 为 0°。正确误差均为 0，但导出为 **0.06264982 m / 30°**。

影响：正常抓取搬运会被标成超差；物体留在原抓取位置反而可能得到更小误差，验收方向错误。

建议：分别保存视觉抓取位姿与计划放置位姿；从已加载 profile 的 `place_id` 取得验收基准。导出器及 validate 都按配置落点复算，不能仅验证两个错误字段互相一致。

### R06：部署门禁可以靠不声明必需项目而绕过

位置：[check_deployment.py:261](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/check_deployment.py:261)、[check_deployment.py:499](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/check_deployment.py:499)。

`resources=[]` 合法，reports 只须包含 motion_profile 引用的报告。后续只遍历声明的资源/报告，没有必需资源角色、必需报告类别或验收配置的完整性清单。所以“删文件但保留声明”会失败，“同时删声明”却没有检查项。

复现：保留合成 verified profile 及其 URDF/电机资源/绑定报告，清空 resources，reports 仅留 profile 报告；删除 requirements、prompt、权重、demo_acceptance、CAL-040 和混淆报告，真实 `run_deployment_check` 仍 **exit 0 / PASS**。没有 mock real loader。此项复现单独构造声明全部八阶段通过及 physical 检查项的临时报告，避免依赖 R09 的空 stages 缺口；这些字段仅用于离线检查逻辑复现，不是实际物理证据。

建议：先核对规定的资源角色和报告集合全部齐备且唯一，再验证每项内容/哈希/引用。未交付视觉资产时，发布结论必须仍是不完整；不能因资产未列入而省略强制检查。

### R07：真实抓放采集入口没有初始化 runlog，与视觉日志契约不兼容

位置：[record_pick_place_trials.py:394](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/record_pick_place_trials.py:394)、[record_pick_place_trials.py:293](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/record_pick_place_trials.py:293)，[runlog.py:149](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/runlog.py:149)。

该 CLI 全程没有 `runlog.init`，`runlog_event = _noop_event` 只替代工具自己的回调，不会初始化 `qingyun.runlog`。真实视觉 init/扫描按契约需要输出资源与 vision_scan 日志，一旦调用 runlog.event 就抛 RuntimeError。已有 FakeVision 不发日志，因此工具测试绕过了这条集成路径。

复现：使用会在 init 发一次真实 `runlog.event` 的契约一致视觉适配器，保留工具真实 `_assemble`，会失败于 `runlog 未初始化`。电机与复位仅在复现中替换为无动作对象；未把尚未实现的视觉算法列为本次代码缺陷。

建议：在视觉 configure/init 之前建立独立采集日志会话，并保证 task/scan/trial 的关联；测试应注入会调用真实 runlog 的视觉适配器。

### R08：hardware 验收缺少观察值的严格类型/有限性检查，可把无观测样本判通过

位置：[calibrate.py:2635](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/calibrate.py:2635)、[calibrate.py:2643](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/calibrate.py:2643)、[calibrate.py:2926](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/calibrate.py:2926)、[calibrate.py:2974](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/calibrate.py:2974)。

统计使用 Python truthiness，但完整性复核遇到非 bool 的 actual_success 直接 continue。例如字符串 `"false"` 会被计为成功，同时跳过成功样本实际位姿验证。所有误差为 null 时也根本不生成误差检查。另一分支仅校验 yaw/观察时长是数值，未检查有限性；`NaN` 比较结果为 False，能绕过误差复算和时长下限。

复现使用仓库完整 simulation profile 的原有验收阈值，保持 `min_grasp_trials=30`：

- 30 条 `actual_success="false"`、实际位置/yaw 和误差全部 null 的声称 hardware 样本，pick_place 所有检查仍 passed。
- 30 条实际 yaw=NaN、存储 yaw 误差=0、损伤观察时长=NaN 的样本，同样全部 passed。

建议：先做统一严格 schema 校验，actual_success/actual_drop/recovery_required 必须是 bool，必需观察字段不得缺失，所有数值必须有限；成功样本必须生成中心/yaw 两项验收。缺失/未知观测只能进入 pending/失败结论，不能按真值或比较漏检放行。

### R09：real loader 接受空 stages 的报告，upper 正向发布测试继续利用该缺口

位置：[motion_params.py:1391](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/configs/motion_params.py:1391)、[test_deployment_check.py:102](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/tests/test_deployment_check.py:102)。

`stages = report.get("stages") or {}` 后只遍历已有项。报告的 stages 为空/缺失时，physical 证据检查循环完全不执行；只要顶层标志与哈希绑定齐备，real loader 就通过。它还没有核对八阶段集合齐备或逐阶段 passed。

复现：用已有部署夹具产生 `stages={}` 的合成报告，直接调用真实 `load_motion_params(...,mode="real")` 成功；R06 中的部署检查也真实调用该 loader。

来源区分：已逐字比较 `0a050b9` 与当前版本的 `_check_report_binding`，函数未变。这是**既有问题**，不是 upper 新引入；但当前 upper 的发布工具与正向验收把该行为作为可通过路径，因此不能据这些测试证明真实物理证据门禁成立。正常 `calibrate promote` 会检查八阶段，问题在于 real loader 自己不能拒绝不完整报告。

建议：loader 独立检查完整阶段集合、严格 bool passed/overall_pass/operator_reviewed、每阶段有效检查与 physical 证据。离线正向夹具应完整声明模拟证据，生产加载边界应拒绝用它驱动真机，而不能依赖空循环获得通过。

## P2：常规逻辑与数据保护问题

### R10：报告输出路径可以覆盖原始输入样本

位置：[fit_grade_threshold.py:658](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/fit_grade_threshold.py:658)、[fit_grade_threshold.py:673](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/scripts/fit_grade_threshold.py:673)。

仅拒绝 output==input，没有拒绝 report==input 或 report==output。合法 CSV 读取完后，报告写入会覆盖该 CSV，或覆盖刚生成的 K 建议文件；命令依然返回成功。

复现：在临时目录执行 `--input samples.csv --output k.json --report samples.csv`，exit 0，原始 CSV 被 JSON 报告取代。

建议：读写前对所有输入/输出路径做归一化碰撞检查，已有文件还需考虑 samefile 别名；输入文件不可成为任何输出目标。类似路径保护应一起检查其余标定 CLI。

### R11：启动最终反馈重新起读超时，能在握手总时限之后标 READY

位置：[motor_control.py:1709](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/motor_control.py:1709)、[motor_control.py:1735](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/motor_control.py:1735)、[motor_control.py:2058](/home/shiroha_pantsu/learn/labroratry/Qingyun_arm/qingyun/grabbing/motor_control.py:2058)。

启动大部分事务使用握手的绝对 deadline，但两次 `self.get_feedback()` 各自用当前时间重新建立 read_timeout_s。最后一次读完成后没有再次检查握手 deadline，直接设置 READY。于是整体上限并不覆盖最后一次反馈。

复现：沿用真实启动逻辑与已有假总线，timeout=0.31 秒；前面等待占 0.30 秒，最后反馈用 0.015 秒（小于 read_timeout=0.02 秒、允许采样跨度=0.03 秒）。启动在 **0.315 秒**仍成功并报告 ready。

建议：让初始化反馈读取接收上层截止点，以 `min(握手截止点, 当前时间+读超时)` 为限，READY 前再核对总时限；补最后一笔读跨总时限的用例。

## 修复与复验顺序

1. 修复 R01–R04，加入真实 raw buffer/VAD 输入和官方云响应 fixture；随后再做麦克风与账号烟雾验证。
2. 修复 R05/R08，重新生成抓放验收数据；旧误差字段不可直接沿用。
3. 修复 R06/R09 的两层完整性检查，补“删掉必需声明”“空/缺失阶段”“模拟证据”拒绝用例。
4. 修复 R07，贯通独立标定入口与视觉日志生命周期。
5. 补 R10/R11 的路径冲突与启动时限回归，然后运行完整测试。

已明确属于待交付范围的视觉算法、未回填 K 和真机 draft 状态，不计为以上 11 项逻辑缺陷。本报告没有证明真实云协议、硬件启动或视觉物理精度已经验收。
