"""手眼标定数学与核心算法单元测试。

验证内容:
  1. Umeyama / Kabsch SVD 刚体变换算法的数值精度（零噪声下达到 10^-15 级）；
  2. 侧斜视空间点集（含高低台阶）在噪声干扰下的鲁棒性；
  3. 用户坐标输入解析器的健壮性与智能单位识别；
  4. 标定板图像生成的几何正确性与 PnP 求解闭环。
"""

import math
import numpy as np
import pytest
import cv2

from qingyun.calibration.touch_calibrator import (
    solve_rigid_transform_3d,
    parse_user_coord_input,
    detect_markers_pnp
)
from qingyun.calibration.generate_markers import generate_printable_marker_card


def test_svd_exact_numerical_precision():
    """测试 Umeyama SVD 在无噪声理想点云上的数值极限精度。"""
    np.random.seed(42)
    # 构造真值旋转（绕各轴旋转任意角度）
    rvec_gt = np.array([0.3, -0.4, 0.5])
    R_gt, _ = cv2.Rodrigues(rvec_gt)
    t_gt = np.array([[-0.15], [0.35], [0.82]])  # 空间平移 82cm

    # 模拟 6 个非共面测试点（4个在基准面，2个垫高 4cm）
    pts_cam = np.array([
        [-0.10, -0.05, 0.60],
        [ 0.10, -0.05, 0.65],
        [-0.08,  0.15, 0.70],
        [ 0.12,  0.15, 0.75],
        [-0.02,  0.02, 0.56],  # 垫高 4cm (离相机更近，Z更小)
        [ 0.05,  0.08, 0.58],  # 垫高 4cm
    ], dtype=np.float64)

    # 生成真值基座坐标: P_base = R * P_cam + t
    pts_base = (R_gt @ pts_cam.T + t_gt).T

    # 算法求解
    R_est, t_est = solve_rigid_transform_3d(pts_cam, pts_base)

    # 验证旋转与平移误差（应接近浮点极限 10^-15）
    rot_err = np.linalg.norm(R_est - R_gt)
    trans_err = np.linalg.norm(t_est - t_gt)
    
    assert rot_err < 1e-12, f"旋转误差过大: {rot_err}"
    assert trans_err < 1e-12, f"平移误差过大: {trans_err}"

    # 验证行列式必须严格为 +1 (右手正交矩阵，绝不能产生镜像反射)
    det = np.linalg.det(R_est)
    assert abs(det - 1.0) < 1e-12, f"旋转矩阵非正交右手系: det={det}"


def test_svd_robustness_with_measurement_noise():
    """测试在毫米级真实测量噪声下的解算稳定性（验证台阶法抗噪性能）。"""
    np.random.seed(123)
    rvec_gt = np.array([0.2, 0.5, -0.1])
    R_gt, _ = cv2.Rodrigues(rvec_gt)
    t_gt = np.array([[0.20], [-0.10], [0.50]])

    # 6 个测试点（含高低落差）
    pts_cam = np.array([
        [-0.15, 0.10, 0.55],
        [ 0.05, 0.12, 0.60],
        [-0.12, 0.25, 0.68],
        [ 0.10, 0.22, 0.72],
        [-0.05, 0.18, 0.51],  # 垫高点
        [ 0.02, 0.15, 0.52],  # 垫高点
    ], dtype=np.float64)

    pts_base_clean = (R_gt @ pts_cam.T + t_gt).T

    # 加入模拟的真实测量噪声: 高斯分布，标准差 1.0 mm (0.001 m)
    noise_cam = np.random.normal(0, 0.001, pts_cam.shape)
    noise_base = np.random.normal(0, 0.001, pts_base_clean.shape)

    pts_cam_noisy = pts_cam + noise_cam
    pts_base_noisy = pts_base_clean + noise_base

    # 求解
    R_est, t_est = solve_rigid_transform_3d(pts_cam_noisy, pts_base_noisy)

    # 检验所有点的反投影残差
    pts_pred = (R_est @ pts_cam.T + t_est).T
    residuals_mm = np.linalg.norm(pts_base_clean - pts_pred, axis=1) * 1000.0

    max_err = np.max(residuals_mm)
    assert max_err < 3.5, f"在 1mm 测量噪声下残差超标 ({max_err}mm > 3.5mm)"


def test_coord_parser_units():
    """测试坐标解析器对空格、逗号及 mm/m 智能单位转换的正确性。"""
    # 毫米输入 -> 自动转为米
    p1 = parse_user_coord_input("150.0  -200.5   35.2")
    assert np.allclose(p1, [0.150, -0.2005, 0.0352])

    # 逗号分隔的毫米输入
    p2 = parse_user_coord_input("200, -50, 80")
    assert np.allclose(p2, [0.200, -0.050, 0.080])

    # 纯米制输入
    p3 = parse_user_coord_input("0.250 -0.120 0.045")
    assert np.allclose(p3, [0.250, -0.120, 0.045])

    # 异常输入报错
    with pytest.raises(ValueError):
        parse_user_coord_input("100 200")  # 只有两个数


def test_aruco_pnp_detection_pipeline():
    """测试生成的 ArUco 标记卡片并运行完整 PnP 识别闭环。"""
    # 生成 1 号标记卡片
    marker_size_px = 400
    card = generate_printable_marker_card(marker_id=1, marker_size_px=marker_size_px, card_w_px=600, card_h_px=600)

    # 虚拟一个标准相机内参
    K = np.array([
        [800.0, 0.0, 300.0],
        [0.0, 800.0, 300.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    # 执行检测与 PnP 解算 (物理边长 0.040m)
    results, annotated = detect_markers_pnp(card, K, dist, marker_length_m=0.040)

    assert 1 in results, "未能正确识别标记 ID 1"
    P_cam = results[1]["P_cam"]
    assert P_cam[2] > 0.0, "Z 轴深度必须为正"
    assert abs(P_cam[0]) < 0.1, "正对着的卡片中心 X 应接近 0"
