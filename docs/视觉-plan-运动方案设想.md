# 视觉-plan-运动方案（接口评审落定版）

> 2026-09-21 视觉接口 grill 评审结论。与 [上层任务编排模块技术文档](上层任务编排模块技术文档.md) §2/§5/§6/§11 同步；运动侧行为以 [机械臂运动控制模块技术文档](机械臂运动控制模块技术文档.md) 为准。
> 相机：RGB-D，眼在手外固定安装，home 位姿不遮挡抓取 ROI；相机→基座变换在视觉内部完成（CAL-040/041 归视觉验收）。

## 一、公共接口改造（common_interface.py，动工前先行 commit）

```py
@dataclass(frozen=True)
class VisionInterface:
    position: FloatArray  # (3,) 目标几何中心，base_link，m
    yaw_deg: float        # 目标长轴相对基座 +X，绕 +Z 为正，[-90, 90)，模 180°
    length_m: float       # 长轴长度
    width_m: float        # 短轴宽度
    ripe: bool            # YOLO 成熟度判定，C 级唯一来源
    valid_count: int      # 当次视野合法候选总数（ignore 封顶判据，§五）
```

- grade 字段删除：运动侧运行代码零引用（仅 tests 构造点与文档锁场），grade 完全 plans 内部化，place_id 是唯一放置可观测。
- 连带机械修订：motion 文档 §1.1 字段表；tests/test_grasp_flow.py、tests/support.py 构造点。
- grasp_and_place 对 length/width/ripe/valid_count 零消费，只验 position/yaw；五项可抓性保证语义挂在 get_target 返回值上（运动文档 §1.1 不变）。

## 二、vision.py 单体（import 同进程）

- 依赖方向：只准 import common_interface（类型）与自身模型栈；不 import 上层/运动任何东西。
- 子进程仅作依赖冲突逃生门（见下节）。

### 子进程逃生门（依赖冲突备用接入，demo代码暂时不用实现）

- 触发条件：视觉模型栈（torch/onnxruntime、YOLO、深度图处理库）与上层环境发生依赖冲突——版本互斥、目标机装不上——才启用；默认不启用。
- 形态：**多进程，非多线程**。vision.py 移入独立操作系统进程，主进程经 stdin/stdout 管道与其通信，一问一答、以行为单位：
  - 主进程写一行 JSON 请求：`{"ignore": 1}`
  - 子进程跑完整采集-检测管线，回一行 JSON：`{"position": [...], "yaw_deg": ..., "length_m": ..., "width_m": ..., "ripe": ..., "valid_count": ...}`
  - 异常序列化：`{"exc": "NoTarget"}` / `{"exc": "VisionHardError"}`，主进程侧代理还原为异常抛出
- 公共面不变：`get_target(ignore) -> VisionInterface` 与两个异常类的签名、语义、映射原样保留，plans/main 代码零改动——只是底层从进程内函数调用换成管道请求/应答。
- 并发语义：上层视角两种形态均为同步阻塞（发一行、等一行，等待期间无并发动作）；子进程内部开几个线程做帧融合是视觉本体的内部自由（"本体内部多帧融合不受限"），不泄漏到接口。

### 入口签名

```py
class NoTarget(Exception): ...        # ROI 无物体 / 全部候选被过滤 / ignore 跳尽
class VisionHardError(Exception): ... # 相机断开 / 帧读取失败 / 检测管线崩溃 → main terminate

def init() -> None
# main 冷启动段调用；YOLO 模型一次性加载（不懒加载、不放 import 时）

def get_target(ignore: int = 0) -> VisionInterface
```

### 调用语义

- 前置条件：机械臂在 home（运动 RETURN 阶段天然维持，plans 不用额外搬臂）。
- 一次调用 = 一次完整采集-检测管线，无跨调用缓存；本体内部多帧融合不受限。
- 流程：全量检出 → 五项可抓性过滤（§三）→ 确定性排名（§四）→ 返回第 ignore+1 名。
- valid_count = 当次通过过滤的候选总数（排名表长度）；ignore ≥ valid_count 的调用必跳尽（plan 以此封顶，§五）。
- 无目标一律抛 NoTarget（不返 None）；管线故障抛 VisionHardError。
- NoTarget×3 之间无间隔；三连调防检测抖动，不防场景变化（目标静止假设承担）。
- 每次调用日志 = 候选数 + 逐项拒绝原因——伪"任务完成"（标定漂移）的唯一审计手段。
- 低置信度候选直接过滤，阈值归视觉本体内部。

## 三、可抓性五项（过滤阈值来源）

| 项 | 规则 | 阈值来源 |
|---|---|---|
| 位置 | 目标中心在已标定目标域内 | `workspace.target_bounds_m` |
| 尺寸 | 闭合轴向尺寸 ≤ object_envelope_m 对应维（**不设下限**） | `grasp.object_envelope_m`；闭合轴向按 R_top_down 夹持约定（length/width） |
| 方向 | 非过滤项；yaw 测量误差验收 ≤5°（实测） | 误差预算：(L/2)·sinδ 须被接触带余量吸收 |
| 接近空间 | 头顶净空防御性检查（深度图）；单层场景近乎恒真 | `approach_height_m` + envelope 推导，不新增配置 |
| 邻物间距 | 边缘距 = 中心距 −（半长+半宽，矩形足迹近似）≥ 最小净距 | 复用 `collision.clearance_m` |

不设下限的依据：夹爪可完全闭合——过小目标首次接触开度低于 `contact_gap_range_m`，被判空闭合（空爪确定）→ GRASP_MISS → skip-ahead，由 ignore 封顶穷尽兜底（§五）。后果：过小目标逐个 MISS → 封顶 → 任务完成（失败码日志可审计，优于伪"任务完成"；摆桌不混入过小果）。

## 四、确定性排名（可复现：同场景两次调用同一结果）

1. 邻距最大者优先——邻距对**全体检出物体**量（含被过滤的），XY 平面距离，与过滤共用一份计算
2. 平局 → ROI 几何中心（target_bounds 中心）最近
3. 仍平局 → position 字典序

禁随机数、时间戳、置信度 tie-break。

## 五、ignore 封顶与失败兜底（重试 skip-ahead）

- 触发：重试类失败（上层文档 §6 分发表）→ ignore+1，下次 `get_target(ignore)`。
- 封顶（任务完成判据，主出口）：失败递增后 `ignore ≥ 当次 target.valid_count` → 视野中所有合法目标已各失败一次、无处可跳 → 解锁复位后正常返回（任务完成，回 WAITING）。判定先于失败预算——穷尽是正常出口。
- 失败预算（兜底）：连续 N 次重试类失败 → `terminate`（不再解锁复位）；计数**仅成功抓取清零**（NoTarget 不清零）；正常路径均被封顶先拦截，预算只兜 NoTarget 归零绕路（场景变化致 valid_count 波动的循环）；N 为 plan 常量（默认 3），实例状态、新任务归零。
- ignore 归零：任一次成功抓取；任一次 NoTarget。
- 语义：episode 级弃权非拉黑——被跳过物体若仍在 ROI，归零后按当次排名重新入选；ignore 跳的是当次排名（每次调用重新扫描，失败可能已碰歪物体）。

## 六、plans（以草莓方案为例）

- grade 判定（阈值常量 K 在 plan 文件内）：

```py
grade = "C" if not target.ripe else ("A" if target.length_m * target.width_m >= K else "B")
```

- 落点：A=优品多落点顺序分配；B/C=各自一个固定落点，落点标定为**箱口上方释放式堆放点**（贴桌面同点重复放置会压到前次放置物，§10.6）。
- 每轮打一行 grade→place_id 日志（grade 不出 plans）。
- 伪代码（修正版）：

```py
def next_place_id(self, grade: str) -> str:
    if grade == "A":
        place = self.a_places[self.a_count]   # 越界 → terminate（摆桌违约）
        self.a_count += 1
        return place
    elif grade == "B":
        return self.b_place
    elif grade == "C":
        return self.c_place
```

- `a_count` 实例状态，构造归零；新任务重新实例化 Plan（上层文档 §1）。连续失败计数同 `a_count` 为实例状态，新任务归零。
- 越界防护：GRASP_MISS 空耗槽位可捅穿"优品数≤落点数"（上层文档 §9），查界后走 `terminate`。
- a_places/b_place/c_place 在构造时从 profile.json `places` 建立 place_id 映射。

## 七、验收项（demo 前）

- CAL-040/041：base 系位置误差 ≤5mm（实测）。
- yaw 测量误差 ≤5°（实测）。
- YOLO unripe 误判率：混绿果场景人工验收。
- 审计日志抽查：候选数 + 逐项拒绝原因可回溯每一次"任务完成"。

## 八、已知风险（显式接受）

- 过小目标不设下限 → 逐个 GRASP_MISS → ignore 封顶穷尽（§五）→ 任务完成（失败码日志可审计；摆桌不混入过小果）。
- YOLO ripe/unripe 误判 → 放错堆，无任何一层拦截。
- 矩形足迹近似 vs 真实轮廓的邻距误差，由 clearance 余量吸收。
- GRIP_SLIP 物体去向不明（上层文档 §10 既有）。
