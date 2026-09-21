# 顶抓姿态构造与 TCP 标定约定

> 配套：[运动控制](机械臂运动控制模块技术文档.md)、[标定指南](真机参数测量与标定指南.md)、[公共接口](../configs/common_interface.py)。本文定义目标中心、工具姿态和配置放置位姿之间的数学关系。

## 1. 坐标系

- B：`base_link`，右手系，+Z 竖直向上。
- E：URDF `gripper_frame_link`。
- O：物体中心系，+X 沿水平长轴代表方向，+Z 与基座 +Z 平行。
- TCP：指垫接触区域中心，+X 为夹爪闭合方向的固定代表方向，+Z 为接近方向，+Y=`+Z × +X`。

`T_A_B` 将 B 系中的点变换到 A 系；矩阵中的平移单位 m。VisionInterface.position 和 PlacePose.position 都是物体中心，top_down_pose 的位置参数则是 TCP。

物体 yaw 是其水平长轴相对基座 +X 的角度，绕 +Z 右手为正，模 180°，范围 `[-90,90)`。TCP yaw 是工具闭合轴相对基座 +X 的有向角。实际工具方向由物体 yaw 与固定配置 `grasp.yaw_offset_deg` 相加确定。

视觉按该固定夹持方向及已标定工作区确定可抓取目标。控制保持下发目标与夹持方向不变，求解并校验对应关节轨迹。

## 2. R_TOP_DOWN 和目标位姿

零偏航顶抓时，TCP +X 沿 B +X，TCP +Y 沿 B -Y，TCP +Z 沿 B -Z。

```python
import numpy as np

R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])

def rz_deg(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def top_down_pose(tcp_xyz: np.ndarray, tcp_yaw_deg: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rz_deg(tcp_yaw_deg) @ R_TOP_DOWN
    T[:3, 3] = tcp_xyz
    return T
```

R_TOP_DOWN 是工具坐标约定产生的常数，满足正交性、det=+1。任意 yaw 下，旋转矩阵第三列均为 `[0,0,-1]`。实机工具安装旋转通过 tool.rotation_e_tcp 表达。

从目标中心构造抓取 TCP：

```python
p_grasp = target.position + rz_deg(target.yaw_deg) @ params.grasp.center_offset_object_m
theta_grasp = target.yaw_deg + params.grasp.yaw_offset_deg
T_B_TCP_grasp = top_down_pose(p_grasp, theta_grasp)
```

center_offset_object_m 在 O 系中表达；yaw_offset_deg 是固定夹持角度偏移，不用于补偿关节零位或相机外参。采用单动指夹爪时，theta 和 theta+180° 的实际接触几何可能不同；实现使用配置定义的方向。

## 3. 开度相关的工具变换

固定旋转 tool.rotation_e_tcp 定义工具轴在 E 系中的方向，平移由 tool.translation_samples 按开度分段线性插值：

```text
translation_samples = [
  {gripper_pct: g0, xyz_m: [x0,y0,z0]},
  {gripper_pct: g1, xyz_m: [x1,y1,z1]}, ...
]
T_E_TCP(g) = [rotation_e_tcp, interpolate_translation(g); 0,0,0,1]
```

样本开度严格递增，禁止外插；表覆盖预张开、接触和释放开度。活动指的关节角由 gripper.angle_table 供碰撞模型使用，TCP 轴固定在工具主体上。

```python
T_B_TCP = FK_E(q_urdf_deg) @ tool_transform(g, params.tool)
T_B_E_target = T_B_TCP_target @ inverse(tool_transform(g, params.tool))
```

FK_E 使用五个姿态关节角。工具中心测量和插值验证方法见标定指南；工具几何包络同时覆盖实际活动指、固定指与指垫。

## 4. 夹持关系与配置放置位姿

闭爪时五关节保持，夹爪开度变化会改变 TCP 位置。接触稳定后读取 q_close、g_close，建立名义物体相对工具变换：

```python
T_B_O_grasp = np.eye(4)
T_B_O_grasp[:3, :3] = rz_deg(target.yaw_deg)
T_B_O_grasp[:3, 3] = target.position
T_TCP_O = inverse(FK_TCP(q_close, g_close)) @ T_B_O_grasp

place = params.places[place_id]
T_B_O_place = np.eye(4)
T_B_O_place[:3, :3] = rz_deg(place.yaw_deg)
T_B_O_place[:3, 3] = place.position
T_B_TCP_place = T_B_O_place @ inverse(T_TCP_O)
```

该关系在一次持物过程中固定。闭合推动与持物滑移的测量上界纳入 grasp.object_shift_bound_m 和碰撞/定位误差预算。放置 TCP 以此完整变换为准，其位置一般不等于配置中的物体中心。

下降和抬升沿基座 Z 轴，段内固定完整 TCP 姿态；横向搬运与姿态调整在配置安全高度及中转路径上完成。

## 5. IK 验收

SO-ARM101 有五个姿态关节。指定位置、接近方向和 yaw 分别形成 3、2、1 个标量条件；实际可达性取决于结构、关节范围及这些条件的几何关系。标定覆盖视觉允许下发的位置与角度域，运行时检验实际 IK 结果。

令 T_act 为实际关节解的完整 TCP FK，T_des 为目标：

- `e_pos = norm(p_act-p_des)`，m。
- `e_tilt = degrees(acos(clip(dot(R_act[:,2],R_des[:,2]),-1,1)))`，deg。
- 工具 +X 投影到基座 XY 后用 atan2 计算 yaw，`e_yaw=abs((yaw_act-yaw_des+180)%360-180)`，deg；投影退化时解无效。

同时满足 ik.position_tol_m、ik.tilt_tol_deg 和 ik.yaw_tol_deg 才接受。夹持后更新的目标姿态可能含允许范围内的工具倾角，还需检查其接近轴与 `[0,0,-1]` 的偏差不超过 ik.tilt_tol_deg。位置或姿态未达标返回 IK_FAILED，不修改视觉目标。

关节解需要满足有效限位和路径连续性；备用初值用于求解同一目标。物体长轴的离线标定误差按模 180° 计算，工具实际方向误差按模 360° 计算。

## 6. 标定分工与数学检查

电机原始读数经 joints.sign、joints.zero_offset_deg 转成 q_urdf；FK_E 与 T_E_TCP 组成工具位姿。相机到基座的转换由视觉完成，控制直接使用下发中心。

关节零位用多个已知构型测量；工具旋转/中心用刚性夹具、多姿态和不同开度测量；独立测量验证点用于检查实际工具精度与抓取域覆盖。

必要数学检查：R_TOP_DOWN 与 rz_deg 的正交性和方向；工具变换与逆变换往返；TCP 插值端点与中点；目标中心到配置放置中心的刚体关系往返；合法关节 FK/IK 残差。具体输入输出与函数名称和标定指南保持一致。
