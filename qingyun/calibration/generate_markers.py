#!/usr/bin/env python3
"""生成 6 个带编号与十字靶心的 40mm ArUco 标定标记图纸（A4 打印排版）。

使用标准 ArUco 字典 DICT_5X5_100，生成尺寸严格定为 40mm x 40mm 的标记。
支持输出：
  1. calibration_sheet_A4.png：整张 A4 排版（300 DPI），包含 6 个标记、裁剪虚线、10cm 校验尺。
  2. 单独标记卡片（markers/marker_1.png ~ marker_6.png）。
"""

import os
from pathlib import Path
import cv2
import numpy as np


def generate_aruco_marker(marker_id: int, marker_size_px: int, dict_type=cv2.aruco.DICT_5X5_100) -> np.ndarray:
    """生成单个 ArUco 标记图像（含白边，兼容 OpenCV 各版本）。"""
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dict_type)

    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px, borderBits=1)
    else:
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size_px, borderBits=1)

    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    return marker_bgr


def generate_printable_marker_card(marker_id: int, marker_size_px: int, card_w_px: int, card_h_px: int) -> np.ndarray:
    """生成带编号大字、裁剪虚线和中心十字靶心的标记卡片。"""
    card = np.ones((card_h_px, card_w_px, 3), dtype=np.uint8) * 255
    
    # 放置 ArUco 标记在中间偏上
    marker_bgr = generate_aruco_marker(marker_id, marker_size_px)
    offset_x = (card_w_px - marker_size_px) // 2
    offset_y = 60
    
    card[offset_y:offset_y + marker_size_px, offset_x:offset_x + marker_size_px] = marker_bgr
    
    # 在 ArUco 标记正中心绘制精密十字靶心（绿色圆圈 + 红色细十字）
    center_x = offset_x + marker_size_px // 2
    center_y = offset_y + marker_size_px // 2
    
    # 中心微型红十字（便于机械臂尖端精准对准）
    cross_len = 16
    cv2.line(card, (center_x - cross_len, center_y), (center_x + cross_len, center_y), (0, 0, 255), 2)
    cv2.line(card, (center_x, center_y - cross_len), (center_x, center_y + cross_len), (0, 0, 255), 2)
    cv2.circle(card, (center_x, center_y), 4, (0, 255, 0), 1)

    # 标记下方绘制大号编号文字
    text = f"Marker #{marker_id}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.3
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (card_w_px - tw) // 2
    text_y = offset_y + marker_size_px + 55
    cv2.putText(card, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # 提示文字
    sub_text = "Size: 40mm x 40mm"
    (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(card, sub_text, ((card_w_px - stw) // 2, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

    # 绘制外边框（虚线效果或灰色细线）
    cv2.rectangle(card, (5, 5), (card_w_px - 5, card_h_px - 5), (180, 180, 180), 2)
    return card


def create_a4_calibration_sheet(output_path: str = "calibration_sheet_A4.png"):
    """创建整张 300 DPI A4 标定图纸（210mm x 297mm）。"""
    dpi = 300
    # A4 尺寸像素数 (210mm x 297mm)
    a4_w = int(210 / 25.4 * dpi)  # 2480 px
    a4_h = int(297 / 25.4 * dpi)  # 3508 px
    
    sheet = np.ones((a4_h, a4_w, 3), dtype=np.uint8) * 255
    
    # 40mm 转换为像素
    marker_size_px = int(40.0 / 25.4 * dpi)  # 约 472 px
    
    # 单卡片尺寸
    card_w = 950
    card_h = 700
    
    # 页眉标题
    title = "Qingyun Eye-to-Hand Calibration Sheet"
    cv2.putText(sheet, title, (200, 180), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 3)
    desc = "Print at 100% scale (Do not fit/shrink). Markers 1-4 on table, 5-6 elevated 3-5cm."
    cv2.putText(sheet, desc, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    
    # 10cm 比例尺（用于打印后尺子实测验证打印是否为 1:1 无缩放）
    ruler_x0, ruler_y = 200, 310
    ruler_len_px = int(100.0 / 25.4 * dpi)  # 100mm 对应像素
    cv2.line(sheet, (ruler_x0, ruler_y), (ruler_x0 + ruler_len_px, ruler_y), (0, 0, 0), 4)
    cv2.line(sheet, (ruler_x0, ruler_y - 15), (ruler_x0, ruler_y + 15), (0, 0, 0), 3)
    cv2.line(sheet, (ruler_x0 + ruler_len_px, ruler_y - 15), (ruler_x0 + ruler_len_px, ruler_y + 15), (0, 0, 0), 3)
    cv2.putText(sheet, "<--- 100 mm (10.0 cm) Verification Scale --->", (ruler_x0 + 120, ruler_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # 布局 2 列 x 3 行 = 6 个标记
    start_y = 420
    start_x = (a4_w - card_w * 2 - 100) // 2
    
    marker_id = 1
    for r in range(3):
        for c in range(2):
            card = generate_printable_marker_card(marker_id, marker_size_px, card_w, card_h)
            pos_x = start_x + c * (card_w + 100)
            pos_y = start_y + r * (card_h + 80)
            sheet[pos_y:pos_y + card_h, pos_x:pos_x + card_w] = card
            marker_id += 1
            
    cv2.imwrite(output_path, sheet)
    print(f"✅ 已成功生成 A4 标定纸图片: {output_path} (分辨率: {a4_w}x{a4_h} @ 300DPI)")
    return output_path


def main():
    out_dir = Path("calibration_assets")
    out_dir.mkdir(exist_ok=True)
    a4_file = str(out_dir / "calibration_sheet_A4.png")
    create_a4_calibration_sheet(a4_file)
    
    # 额外生成单独卡片方便直接查看
    dpi = 300
    marker_size_px = int(40.0 / 25.4 * dpi)
    for mid in range(1, 7):
        card = generate_printable_marker_card(mid, marker_size_px, 950, 700)
        single_path = out_dir / f"marker_{mid}.png"
        cv2.imwrite(str(single_path), card)
    print(f"✅ 6 张单卡标定块已同步保存至: {out_dir}/")


if __name__ == "__main__":
    main()
