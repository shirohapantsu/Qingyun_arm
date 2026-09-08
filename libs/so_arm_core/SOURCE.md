# vendor 来源与验证记录

本目录（`libs/so_arm_core/`）内的文件是从外部项目原样复制进来的第三方源文件，
在本项目内锁定，不做本地改动。复制时保留了原始版权声明与许可证头。

## 1. 上游来源

| 项目 | 值 |
|---|---|
| 仓库 | Hugging Face `huggingface/lerobot` |
| 上游版本 | 0.5.1 |
| 许可证 | Apache License 2.0 |
| 本机取源目录 | `../lerobot/lerobot/src/lerobot/model/kinematics.py`<br>`../lerobot/lerobot/src/lerobot/motors/`<br>`../lerobot/SO-ARM100/Simulation/SO101/so101_new_calib.urdf` |
| 复制日期 | 2026-09-07 |

复制范围与上游原文件的对应关系：

| 本目录文件 | 上游路径 | 是否逐字节一致 |
|---|---|---|
| `kinematics.py` | `src/lerobot/model/kinematics.py` | 是 |
| `so101_new_calib.urdf` | `SO-ARM100/Simulation/SO101/so101_new_calib.urdf` | 是 |
| `assets/*.stl` | `SO-ARM100/Simulation/SO101/assets/*.stl` | 是 |
| `motors/encoding_utils.py` | `src/lerobot/motors/encoding_utils.py` | 是 |
| `motors/feetech/tables.py` | `src/lerobot/motors/feetech/tables.py` | 是 |
| `__init__.py`、`motors/__init__.py`、`motors/feetech/__init__.py` | 本项目新增的包声明文件 | 不适用 |

`assets/` 是 URDF 中 `<mesh filename="assets/..."/>` 相对引用的目标。placo 在解析
URDF 时对网格文件缺失会直接抛错（`Mesh assets/... could not be found.`），因此网格
必须与 URDF 一起 vendor，并保持在同一相对位置。上游 `assets/` 中的 `.part`（Onshape
CAD 源文件）不被 URDF 引用，未复制。

按技术文档要求，**未**复制 LeRobot 的 processor 注册与数据流水线代码。
`src/lerobot/motors/motors_bus.py` 与 `feetech/feetech.py` 也**未**复制：它们依赖
`deepdiff`、`tqdm`、`scservo_sdk` 与 `lerobot.utils.*` 整条传递链，vendor 成本远高于
收益。本项目用 `pyserial` 直接实现 STS 协议组包（见
`qingyun/grabbing/motor_control.py`），但寄存器地址、编码分辨率与符号位定义一律
以本目录 vendor 的 `motors/feetech/tables.py` 为准，避免第二份真值表。

## 2. 验证过的依赖版本

下列版本是在开发机上实际跑通 FK/IK 往返与 URDF 加载的组合，香橙派部署时按此锁定。

| 依赖 | 约束 | 开发机验证版本 | 说明 |
|---|---|---|---|
| Python | 3.12 | 3.12.x | conda 环境 `Qingyun` |
| numpy | `>=1.26` | 2.3.5 | |
| placo | `>=0.9.6,<0.9.17`（LeRobot 0.5.1 `placo-dep` 声明） | 0.9.16 | PyPI 提供 cp312 的 manylinux x86_64 与 aarch64 轮子 |
| pin (`pinocchio`) | placo 的传递依赖 | 3.4.0 | |
| cmeel-urdfdom | 必须 4.0.x | 4.0.1 | pip 默认会拉 6.0.0（soname `.so.6`），而 placo 0.9.16 / pin 3.4.0 链接的是 `liburdfdom_sensor.so.4.0`，装成 6.0.0 会 `ImportError` |
| cmeel-tinyxml2 | 必须 10.0.x | 10.0.0 | 同上，11.0.0 提供 `libtinyxml2.so.11` 会导致 `ImportError: libtinyxml2.so.10` |
| pyserial | — | 3.5 | 仅真机 `MotorController` 需要；未安装时本模块仍可 import |
| pytest | 开发期 | 9.1.1 | |

`pip install -r requirements.txt` 已把上述两个 cmeel 传递依赖钉住。若自行升级
placo 或 pin，需重新确认 `import placo` 通过。

## 3. 文件哈希（SHA256）

```text
356645b7f6c68172055c13642791d5eddcb5df49e4fdea12780e4b1bc01371fe  kinematics.py
3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c  so101_new_calib.urdf
5285eb2b3f62399e95a2fa4b2fcab631bcad547124e8cc9b6a61525148bc924a  motors/encoding_utils.py
71f7f7beb17169781bd26a33b165ed4c5c3df7b9c477d36860f9d6011eb00428  motors/feetech/tables.py
8cd2f241037ea377af1191fffe0dd9d9006beea6dcc48543660ed41647072424  assets/base_motor_holder_so101_v1.stl
bb12b7026575e1f70ccc7240051f9d943553bf34e5128537de6cd86fae33924d  assets/base_so101_v2.stl
31242ae6fb59d8b15c66617b88ad8e9bded62d57c35d11c0c43a70d2f4caa95b  assets/motor_holder_so101_base_v1.stl
887f92e6013cb64ea3a1ab8675e92da1e0beacfd5e001f972523540545e08011  assets/motor_holder_so101_wrist_v1.stl
785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d  assets/moving_jaw_so101_v1.stl
9be900cc2a2bf718102841ef82ef8d2873842427648092c8ed2ca1e2ef4ffa34  assets/rotation_pitch_so101_v1.stl
75ef3781b752e4065891aea855e34dc161a38a549549cd0970cedd07eae6f887  assets/sts3215_03a_no_horn_v1.stl
a37c871fb502483ab96c256baf457d36f2e97afc9205313d9c5ab275ef941cd0  assets/sts3215_03a_v1.stl
d01d1f2de365651dcad9d6669e94ff87ff7652b5bb2d10752a66a456a86dbc71  assets/under_arm_so101_v1.stl
475056e03a17e71919b82fd88ab9a0b898ab50164f2a7943652a6b2941bb2d4f  assets/upper_arm_so101_v1.stl
e197e24005a07d01bbc06a8c42311664eaeda415bf859f68fa247884d0f1a6e9  assets/waveshare_mounting_plate_so101_v2.stl
4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4  assets/wrist_roll_follower_so101_v1.stl
6c7ec5525b4d8b9e397a30ab4bb0037156a5d5f38a4adf2c7d943d6c56eda5ae  assets/wrist_roll_pitch_so101_v2.stl
```

配置里的 `model.urdf_sha256` 使用 `so101_new_calib.urdf` 的哈希
`3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`，由
`load_motion_params(..., mode="real")` 在加载时复核。

## 4. 许可证全文

Apache License 2.0 全文见 <http://www.apache.org/licenses/LICENSE-2.0>；
`kinematics.py`、`encoding_utils.py`、`tables.py` 文件头保留了上游版权声明。
