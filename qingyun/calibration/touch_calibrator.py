#!/usr/bin/env python3
"""手眼标定主程序：侧斜视与分拣区专属——桌面标定纸 + 夹爪尖端点触法 (Eye-to-Hand)。

特点：
  1. 夹爪零拆卸，零加装探针/标定球；
  2. 高分辨率 RGB PnP 精密解算靶心 3D 坐标，规避深度散斑噪点与空洞；
  3. 分时两阶段流程：机械臂在 Home 位先锁定视觉坐标，再分别示教触碰，彻底规避视线遮挡；
  4. Umeyama SVD 闭式解算，绝对收敛无 Bug；
  5. 实时输出逐点残差检验报表（CAL-040 <= 4.0mm 验收标准）；
  6. 自动保存为标准 configs/vision/T_base_camera.json。
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# 将项目根目录加入模块路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_camera_intrinsics(intrin_path: Path):
    """加载相机内参矩阵与畸变参数。"""
    with open(intrin_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rgb = data["rgb"]
    K = np.array([
        [rgb["fx"], 0.0, rgb["cx"]],
        [0.0, rgb["fy"], rgb["cy"]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist = np.array(rgb["dist"], dtype=np.float64)
    return K, dist, data


def solve_rigid_transform_3d(pts_cam: np.ndarray, pts_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """使用 Umeyama / Kabsch SVD 算法求解刚体变换: P_base = R * P_cam + t.

    参数:
        pts_cam: (N, 3) 相机系坐标，单位 m
        pts_base: (N, 3) 机械臂基座系坐标，单位 m
    返回:
        R: (3, 3) 正交旋转矩阵 (det(R) == +1)
        t: (3, 1) 平移向量，单位 m
    """
    assert pts_cam.shape == pts_base.shape, "点集维度必须相同"
    assert pts_cam.shape[0] >= 3, "至少需要 3 个非共线三维点"
    
    # 1. 计算两组点云中心
    centroid_cam = np.mean(pts_cam, axis=0)
    centroid_base = np.mean(pts_base, axis=0)
    
    # 2. 去中心化
    X = pts_cam - centroid_cam
    Y = pts_base - centroid_base
    
    # 3. 计算协方差矩阵 H
    H = X.T @ Y
    
    # 4. 奇异值分解 SVD
    U, S, Vt = np.linalg.svd(H)
    
    # 5. 求解旋转矩阵 R，确保右手坐标系 (det=1，防止镜像反射)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # 6. 求解平移向量 t
    t = centroid_base.reshape(3, 1) - R @ centroid_cam.reshape(3, 1)
    return R, t


def detect_markers_pnp(image_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, marker_length_m: float = 0.040):
    """从彩色图中检测 ArUco 标记并通过 PnP 求解其相机系 3D 中心坐标 (X, Y, Z)。

    参数:
        image_bgr: RGB 图像 (BGR 格式)
        K: 相机内参矩阵 (3, 3)
        dist: 畸变系数 (5,)
        marker_length_m: 标记物理边长 (默认 40mm = 0.040m)
    返回:
        detected_dict: {marker_id: {"P_cam": (3,) ndarray, "corners": (4,2) ndarray, "rvec": ndarray, "tvec": ndarray}}
        annotated_img: 绘制了标记边界和坐标轴的图像
    """
    # 兼容 OpenCV 4.x 与 OpenCV 4.7+/5.x
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

    if hasattr(cv2.aruco, 'DetectorParameters'):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    # 提高倾斜视角的角点检测精度
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image_bgr)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image_bgr, aruco_dict, parameters=parameters)
    
    annotated = image_bgr.copy()
    results = {}
    
    if ids is None or len(ids) == 0:
        return results, annotated
        
    ids = ids.flatten()
    
    # 单个 marker 的 3D 物体空间角点定义 (以标记中心为原点)
    half = marker_length_m / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float64)
    
    for i, mid in enumerate(ids):
        marker_corners = corners[i][0]  # (4, 2)
        
        # PnP 求解精确位姿
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, marker_corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            # 备用普通 PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, marker_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
        if success:
            P_cam = tvec.flatten()  # 标记中心在相机系坐标 (X, Y, Z) 单位 m
            results[int(mid)] = {
                "P_cam": P_cam,
                "corners": marker_corners,
                "rvec": rvec,
                "tvec": tvec
            }
            # 画标记边界
            cv2.polylines(annotated, [np.int32(marker_corners)], True, (0, 255, 0), 2)
            # 画中心十字红点
            cx = int(np.mean(marker_corners[:, 0]))
            cy = int(np.mean(marker_corners[:, 1]))
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            # 标注文字
            txt = f"ID:{mid} Z:{P_cam[2]:.3f}m"
            cv2.putText(annotated, txt, (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # 画 3D 坐标轴 (红X, 绿Y, 蓝Z)
            try:
                cv2.drawFrameAxes(annotated, K, dist, rvec, tvec, 0.03, 2)
            except AttributeError:
                cv2.aruco.drawAxis(annotated, K, dist, rvec, tvec, 0.03)

    return results, annotated


def parse_user_coord_input(input_str: str) -> np.ndarray:
    """解析用户输入的 3D 坐标，支持空格/逗号分隔，并智能识别 mm 或 m。"""
    raw = input_str.strip().replace(",", " ").split()
    if len(raw) != 3:
        raise ValueError(f"必须输入 3 个数值 (X Y Z)，当前输入: '{input_str}'")
    
    vals = [float(x) for x in raw]
    
    # 智能单位判断：如果绝对值 > 5.0，说明用户输入的是毫米 (mm)，自动转换为米 (m)
    if any(abs(v) > 5.0 for v in vals):
        vals_m = [v / 1000.0 for v in vals]
    else:
        vals_m = vals
        
    return np.array(vals_m, dtype=np.float64)


def run_touch_calibration(image_source: str = "live", intrin_file: str = "configs/vision/camera_intrinsics.json"):
    """执行交互式手眼标定全流程。"""
    print("\n=======================================================")
    print("   🍓 青云机械臂手眼标定工具 (眼在手外 Eye-to-Hand)")
    print("   标定方法: 桌面标定纸 + 夹爪闭合尖端点触法")
    print("=======================================================\n")
    
    intrin_path = PROJECT_ROOT / intrin_file
    if not intrin_path.exists():
        print(f"❌ 错误: 找不到相机内参配置文件: {intrin_path}")
        return False
        
    K, dist, intrin_data = load_camera_intrinsics(intrin_path)
    print(f"✅ 成功载入相机内参: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # 第一阶段：获取拍摄帧
    color_img = None
    if image_source == "live":
        print("\n[阶段 1/3] 正在启动相机拍摄分拣区画面...")
        print("💡 提示: 请确保机械臂停在右侧 Home 待命位，使中左侧分拣区完全无遮挡！")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            print("❌ 无法打开相机设备 (/dev/video0)，请检查 USB 连接或使用 --image 参数。")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 预读 8 帧让传感器自动曝光与白平衡稳定
        for _ in range(8):
            ret, color_img = cap.read()
        cap.release()
        
        if not ret or color_img is None:
            print("❌ 从相机获取画面失败！")
            return False
        print("✅ 成功获取相机实时高清画面 (1280x720)！")
    else:
        img_p = Path(image_source)
        if not img_p.exists():
            print(f"❌ 找不到指定的离线图片: {img_p}")
            return False
        color_img = cv2.imread(str(img_p))
        print(f"✅ 成功载入图片: {img_p}")

    # 视觉识别 6 个标记
    marker_results, annotated_img = detect_markers_pnp(color_img, K, dist, marker_length_m=0.040)
    
    # 保存调试快照
    snapshot_path = PROJECT_ROOT / "calibration_assets" / "calibration_snapshot.jpg"
    snapshot_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(snapshot_path), annotated_img)
    print(f"📸 视觉识别调试画面已保存至: {snapshot_path}")

    detected_ids = sorted(list(marker_results.keys()))
    print(f"\n[阶段 2/3] 视觉锁定结果: 共检测到 {len(detected_ids)} 个标记点: {detected_ids}")
    
    if len(detected_ids) < 4:
        print(f"❌ 检测到的标记数量不足 ({len(detected_ids)} < 4)。至少需要 4 个点（推荐 6 个）。")
        print("💡 建议检查:")
        print("  1. 标记是否放置在相机视野内的中左侧分拣区；")
        print("  2. 环境光线是否过暗或标定纸严重反光；")
        print(f"  3. 请在编辑器中打开 {snapshot_path} 查看相机实际拍摄画面。")
        return False

    # 打印检测到的相机系坐标
    print("\n-------------------------------------------------------")
    print("标定点相机系坐标 (rgb_optical，单位米):")
    for mid in detected_ids:
        pos = marker_results[mid]["P_cam"]
        print(f"  标记 #{mid}: X={pos[0]:+.4f}m, Y={pos[1]:+.4f}m, Z={pos[2]:+.4f}m")
    print("-------------------------------------------------------")

    # 第二阶段：交互录入机械臂基座坐标
    print("\n开始依次录入机械臂基座坐标 (base_link):")
    print("💡 提示:")
    print("  1. 请控制机械臂闭合夹爪尖端中心，依次轻触各标记正中央的十字靶心；")
    print("  2. 在示教器或上位机上读取当前坐标，支持毫米(mm)或米(m)，空格分隔直接输入；")
    print("  3. 示例输入: '185.3 -50.2 32.1' 或 '0.1853 -0.0502 0.0321'\n")

    pts_cam_list = []
    pts_base_list = []
    valid_ids = []

    for mid in detected_ids:
        while True:
            pos_c = marker_results[mid]["P_cam"]
            prompt = f"👉 请将闭合夹爪尖端对准【标记 #{mid}】中心，输入机械臂坐标 X Y Z (回车确认，跳过输 s): "
            user_input = input(prompt).strip()
            
            if user_input.lower() == 's':
                print(f"⏩ 已跳过标记 #{mid}")
                break
                
            try:
                pos_b = parse_user_coord_input(user_input)
                pts_cam_list.append(pos_c)
                pts_base_list.append(pos_b)
                valid_ids.append(mid)
                print(f"  已记录 #{mid}: Cam=[{pos_c[0]:.3f}, {pos_c[1]:.3f}, {pos_c[2]:.3f}]m <---> Base=[{pos_b[0]:.3f}, {pos_b[1]:.3f}, {pos_b[2]:.3f}]m")
                break
            except Exception as e:
                print(f"  ⚠️ 输入解析错误: {e}，请重新输入！")

    if len(valid_ids) < 4:
        print(f"❌ 有效匹配点数不足 ({len(valid_ids)} < 4)，无法解算三维外参！")
        return False

    # 第三阶段：Umeyama SVD 闭式解算
    print("\n[阶段 3/3] 正在使用 Umeyama SVD 算法进行高精度刚体变换解算...")
    pts_cam = np.array(pts_cam_list, dtype=np.float64)
    pts_base = np.array(pts_base_list, dtype=np.float64)
    
    R_base_cam, t_base_cam = solve_rigid_transform_3d(pts_cam, pts_base)
    
    # 组装 4x4 齐次矩阵
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3:4] = t_base_cam
    
    # 残差检验 (CAL-040 逐样本验证)
    residuals_mm = []
    print("\n=======================================================")
    print("          🏆 手眼标定精度验收报告 (P3 CAL-040)         ")
    print("=======================================================")
    for i, mid in enumerate(valid_ids):
        P_c = pts_cam[i]
        P_b_real = pts_base[i]
        P_b_pred = (R_base_cam @ P_c.reshape(3, 1) + t_base_cam).flatten()
        res_mm = np.linalg.norm(P_b_real - P_b_pred) * 1000.0
        residuals_mm.append(res_mm)
        status = "✅ PASS" if res_mm <= 4.0 else "⚠️ FAIL"
        print(f"  标记 #{mid:2d}: 实测=[{P_b_real[0]:.3f}, {P_b_real[1]:.3f}, {P_b_real[2]:.3f}] | 投影=[{P_b_pred[0]:.3f}, {P_b_pred[1]:.3f}, {P_b_pred[2]:.3f}] | 误差={res_mm:5.2f} mm  {status}")

    max_err = np.max(residuals_mm)
    rmse = np.sqrt(np.mean(np.array(residuals_mm)**2))
    print("-------------------------------------------------------")
    print(f"  最大三维位置误差 (Max Error): {max_err:.2f} mm (CAL-040 要求: <= 4.0 mm)")
    print(f"  均方根误差 (RMSE)          : {rmse:.2f} mm")
    print("=======================================================")

    if max_err <= 4.0:
        print("🎉 恭喜！本次标定完全满足 P3 §8 (CAL-040) 精度指标！")
    else:
        print("⚠️ 提示: 最大误差略高于 4mm。建议检查是否有某个点手抖碰偏，重新标定。")

    # 保存配置到 configs/vision/T_base_camera.json
    output_cfg_path = PROJECT_ROOT / "configs/vision/T_base_camera.json"
    result_data = {
        "schema_version": 1,
        "from_frame": "rgb_optical",
        "to_frame": "base_link",
        "matrix": T_base_cam.tolist()
    }
    
    with open(output_cfg_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
        
    print(f"\n💾 外参标定矩阵已成功写入生产配置文件: {output_cfg_path}")
    print("🚀 现在运行 demo_vision_ui.py，视觉输出的草莓坐标已是绝对真实的机械臂坐标！")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="青云机械臂手眼标定工具 (Eye-to-Hand)")
    parser.add_argument("--image", type=str, default="live", help="输入图像路径，默认 'live' 使用实时相机拍摄")
    parser.add_argument("--intrinsics", type=str, default="configs/vision/camera_intrinsics.json", help="相机内参配置文件")
    args = parser.parse_args()
    
    run_touch_calibration(image_source=args.image, intrin_file=args.intrinsics)
