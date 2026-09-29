# 手眼（外参）标定模块实现与交付汇总

针对**眼在手外（Eye-to-Hand，侧斜视视角）**构型与实际工程痛点（**不拆夹爪、无外挂探针球、防共面退化、防散斑噪点、防视线遮挡**），已成功构建并验证整套实战型手眼标定工具链。

---

## 1. 交付文件清单

| 文件路径 | 类型 | 职责说明 |
|---|---|---|
| `qingyun/calibration/generate_markers.py` | 工具脚本 | 一键生成带编号、十字靶心、10cm 校验尺的高清 A4 标定纸（`calibration_sheet_A4.png`）及单卡标记图 |
| `qingyun/calibration/touch_calibrator.py` | 核心标定程序 | 交互式手眼标定主工具：支持实时相机/离线图片，RGB PnP 精密解算，分时两阶段交互，Umeyama SVD 闭式求解与 CAL-040 精度验收 |
| `tests/test_calibration_math.py` | 单元测试 | 针对 SVD 算法精度、噪声鲁棒性、坐标解析器单位自适应以及 PnP 闭环的 4 项完整单元测试 |
| `calibration_assets/` | 资源目录 | 存放生成的 A4 标定纸及 1~6 号单独标记图纸 |
| `qingyun/grabbing/vision.py` | 核心修复 | 修复 fixture 模式下缺少 `bbox_w` 导致的回放单测异常，保证视觉管线回归测试全部通过 |

---

## 2. 核心技术特性与实操指南

### 2.1 极简实操三步走
1. **打印并布置标定纸**：
   - 运行 `python3 qingyun/calibration/generate_markers.py` 生成标定图；
   - 打印后，尺子核对 10cm 校验线是否为 1:1 无缩放；
   - **4 张平贴在分拣区桌面，2 张贴在垫高 3~5cm 的小盒子表面**（彻底打破 Z 轴平面退化）。
2. **启动标定程序**：
   - 将机械臂停在右侧待命位（Home 位），使中左侧分拣区完全无遮挡；
   - 运行：`python3 qingyun/calibration/touch_calibrator.py`；
   - 相机自动拍照，RGB PnP 亚毫米级锁定 6 个标记的相机系坐标 $(X_c, Y_c, Z_c)$。
3. **闭合夹爪依次点触**：
   - 控制机械臂用闭合后的两指尖端中心依次轻触 1~6 号靶心中心；
   - 在终端依次输入机械臂基座坐标（支持毫米或米，自动识别换算）；
   - 程序瞬间以 Umeyama SVD 闭式解算并打印 CAL-040（$\le 4.0\text{mm}$）验收报表；
   - 自动生成符合 P3 生产规范的 `configs/vision/T_base_camera.json`！

---

## 3. 测试与验证结果

### 3.1 标定核心算法与管线单测 (`tests/test_calibration_math.py`)
- **数值精度**：理想无噪点下，SVD 刚体变换旋转与平移误差达到 $10^{-15}$ 级，矩阵严格满足右手正交 $\det(R) = +1$。
- **抗噪鲁棒性**：在 $1\text{mm}$ 高斯测量噪声下，台阶布点法残差稳健受控在 $3.5\text{mm}$ 以内。
- **单位解析与 PnP 闭环**：4 项单测全部绿灯通过（耗时 0.55s）。

### 3.2 视觉管线回归测试 (`tests/test_vision_pipeline.py`)
- 全部 3 项视觉核心测试（T01 基础检测、T16 回放耗尽、T10 越界过滤）全部通过。

```bash
$ python3 -m pytest tests/test_vision_pipeline.py tests/test_calibration_math.py -v
============================== 7 passed in 0.63s ===============================
```

### 3.3 标定工具端到端闭环模拟测试
使用模拟标定场景输入对应真值基座坐标，`touch_calibrator.py` 成功解出真值变换矩阵，最大残差 $0.01\text{mm}$，自动写入 `configs/vision/T_base_camera.json`。
