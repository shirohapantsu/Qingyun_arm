#!/usr/bin/env python3
"""生成 80 mm 黑色方形 ArUco 标记：A4 纵向三页，每页两张。

PNG 含 300 DPI 元数据；默认输出到 calibration_assets/generated 的新目录。
PDF 为精确毫米尺寸的矢量图；每张保留约 12 mm 白边，卡片 104×120 mm。
触碰点为黑色正方形的几何中心，不在码内添加十字线。
"""

import argparse
from datetime import datetime, timezone
import os
import struct
import tempfile
import zlib
from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKER_MM = 80.
QUIET_MM = 12.
CARD_W_MM, CARD_H_MM = 104., 120.
DPI = 300
A4_W_MM, A4_H_MM = 210., 297.
CARD_X_MM, CARD_Y_MM = 53., (35., 165.)


def _px(mm):
    return round(mm / 25.4 * DPI)



def _output_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT) and not path.is_relative_to(PROJECT_ROOT / "calibration_assets" / "generated"):
        raise ValueError("仓库内新图只允许写 calibration_assets/generated，保留现有图纸")
    return path


def _new_output_dir(root=None):
    root = _output_path(root or PROJECT_ROOT / "calibration_assets" / "generated")
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def save_png_with_dpi(path, image, dpi=300):
    """用 PNG 标准 pHYs 块记录像素密度，无需新增图像库依赖。"""
    path = _output_path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有图纸：{path}")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("DPI 必须为有限正数")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG 编码失败")
    pixels_per_meter = round(dpi / 0.0254)
    payload = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    chunk_type = b"pHYs"
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload
    chunk += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff)
    png = encoded.tobytes()
    # PNG 签名 8 字节 + IHDR 块 25 字节；在 IHDR 后插入 pHYs。
    png = png[:33] + chunk + png[33:]
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(png)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def generate_aruco_marker(marker_id: int, marker_size_px: int, dict_type=cv2.aruco.DICT_5X5_100) -> np.ndarray:
    """生成含黑色边框的编码图，白色静区由卡片提供。"""
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
    """生成卡片；像素尺寸按 80 mm 黑框比例换算白边，不在编码区作任何标记。"""
    quiet = round(marker_size_px * QUIET_MM / MARKER_MM)
    label_height = round(marker_size_px * 16 / MARKER_MM)
    if marker_size_px < 20 or card_w_px < marker_size_px + 2 * quiet - 1 or card_h_px < marker_size_px + 2 * quiet + label_height - 1:
        raise ValueError("卡片尺寸不足以容纳 80 mm 标记、12 mm 白边和编号")
    card = np.full((card_h_px, card_w_px, 3), 255, dtype=np.uint8)
    x, y = (card_w_px - marker_size_px) // 2, quiet
    card[y:y + marker_size_px, x:x + marker_size_px] = generate_aruco_marker(marker_id, marker_size_px)
    # 白色静区内不加中心线；编号放在黑框下方至少一个静区之外。
    scale = marker_size_px / 945
    for text, baseline, font_scale in ((f"ArUco ID {marker_id} | black square 80 mm", y + marker_size_px + quiet + round(50 * scale), .85),
                                       ("Keep white border | Cut on dashed outline", card_h_px - round(30 * scale), .70)):
        font_scale *= scale
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(card, text, ((card_w_px - width) // 2, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_8)
    # 虚线贴近裁剪边缘，远离黑框。
    dash = max(3, round(24 * scale))
    for x0 in range(0, card_w_px, dash * 2):
        for y0 in (0, card_h_px - 1):
            cv2.line(card, (x0, y0), (min(x0 + dash, card_w_px - 1), y0), (0, 0, 0), 1)
    for y0 in range(0, card_h_px, dash * 2):
        for x0 in (0, card_w_px - 1):
            cv2.line(card, (x0, y0), (x0, min(y0 + dash, card_h_px - 1)), (0, 0, 0), 1)
    return card


def create_a4_calibration_sheet(output_path=None, marker_ids=(1, 2)):
    """生成一页 A4 PNG；80 mm 标记加 12 mm 白边后，每页纵向排两张。"""
    if len(marker_ids) != 2 or len(set(marker_ids)) != 2:
        raise ValueError("每页需要两个不同标记 ID")
    output_path = _output_path(output_path or _new_output_dir() / "A4_80mm_page_1.png")
    sheet = np.full((_px(A4_H_MM), _px(A4_W_MM), 3), 255, dtype=np.uint8)
    cv2.putText(sheet, "Qingyun ArUco | 80 mm | DICT_5X5_100", (_px(17), _px(12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 2)
    cv2.putText(sheet, "A4 portrait | Actual size / 100% | Black and white | One-sided", (_px(17), _px(18)),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 1)
    x0, x1, y = _px(55), _px(155), _px(27)
    cv2.line(sheet, (x0, y), (x1, y), (0, 0, 0), 2)
    for x in (x0, x1):
        cv2.line(sheet, (x, y - _px(1)), (x, y + _px(1)), (0, 0, 0), 2)
    cv2.putText(sheet, "100 mm - measure before cutting", (_px(57), _px(24)),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1)
    for mid, top in zip(marker_ids, CARD_Y_MM):
        card = generate_printable_marker_card(mid, _px(MARKER_MM), _px(CARD_W_MM), _px(CARD_H_MM))
        x, y = _px(CARD_X_MM), _px(top)
        sheet[y:y + card.shape[0], x:x + card.shape[1]] = card
    save_png_with_dpi(output_path, sheet, DPI)
    return output_path


def create_a4_print_pdf(output_path):
    """写精确 A4 / 80 mm 的三页矢量 PDF；只用标准库，不依赖图像打印缩放。"""
    output_path = _output_path(output_path)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有图纸：{output_path}")
    unit = 72 / 25.4  # PDF point / mm
    objects = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pages = []
    for page in range(3):
        commands = ["0 g 0 G"]
        def text(x, y, value, size=10):
            value = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F1 {size} Tf {x * unit:.6f} {(A4_H_MM - y) * unit:.6f} Td ({value}) Tj ET")
        def line(x0, y0, x1, y1):
            commands.append(f"{x0 * unit:.6f} {(A4_H_MM - y0) * unit:.6f} m {x1 * unit:.6f} {(A4_H_MM - y1) * unit:.6f} l S")
        text(17, 12, "Qingyun ArUco | Black square 80 mm | DICT_5X5_100", 13)
        text(17, 18, f"A4 portrait | Actual size / 100% | Black and white | One-sided | Page {page + 1}/3", 9)
        text(57, 24, "100 mm - measure before cutting", 9)
        commands.append("0.6 w")
        line(55, 27, 155, 27)
        for x in (55, 155):
            line(x, 26, x, 28)
        for mid, top in zip((page * 2 + 1, page * 2 + 2), CARD_Y_MM):
            commands.append("0.3 w [4 4] 0 d")
            commands.append(f"{CARD_X_MM * unit:.6f} {(A4_H_MM - top - CARD_H_MM) * unit:.6f} {CARD_W_MM * unit:.6f} {CARD_H_MM * unit:.6f} re S [] 0 d")
            # 5×5 内容加单格黑框，共 7×7；每格使用矢量矩形，物理外框恰为 80 mm。
            cells = generate_aruco_marker(mid, 7)[:, :, 0]
            cell = MARKER_MM / 7
            for row in range(7):
                for col in range(7):
                    if cells[row, col] == 0:
                        x = CARD_X_MM + QUIET_MM + col * cell
                        y = A4_H_MM - (top + QUIET_MM + (row + 1) * cell)
                        commands.append(f"{x * unit:.6f} {y * unit:.6f} {cell * unit:.6f} {cell * unit:.6f} re f")
            text(CARD_X_MM + 8, top + 111, f"ArUco ID {mid} | black square 80 mm | card 104 x 120 mm", 9)
            text(CARD_X_MM + 8, top + 116, "Keep white border. Cut on dashed outline.", 9)
        stream = ("\n".join(commands) + "\n").encode("ascii")
        stream_id = len(objects) + 1
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream")
        page_id = len(objects) + 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {A4_W_MM * unit:.6f} {A4_H_MM * unit:.6f}] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>".encode())
        pages.append(page_id)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>"
    objects[1] = f"<< /Type /Pages /Count 3 /Kids [{' '.join(f'{page} 0 R' for page in pages)}] >>".encode()
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(payload)
    payload.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    with output_path.open("xb") as f:
        f.write(payload)
    return output_path


PRINT_INSTRUCTIONS = """# 80 mm ArUco 图纸：打印与裁剪

这些是当前点触标定程序使用的独立 ArUco 码（DICT_5X5_100，ID 1～6），不是 ChArUco 棋盘。

1. 优先打开 `ArUco_80mm_A4.pdf`：A4 纵向，共 3 页，每页 2 个码。
2. 使用普通白色 A4 纸、黑白、单面、每张纸 1 页，选择“实际大小 / 100%”。关闭“适合页面 / 缩小到可打印区域 / 多页合一”；无需无边距打印。
3. 建议先打印第 1 页，用直尺量 100 mm 校验线（两端竖线中心之间），再量黑色正方形外框：水平和竖直边都应为 80 mm。尺寸正确后再打印第 2、3 页并复核。80 mm 指含黑边的整个编码方形，不包括白边，也不是单个内部格子。
4. 若偏差明显，先检查缩放设置后重打。标定界面填写黑框实测毫米边长；各张应一致，长宽明显不同或纸张变形应重打，不靠修改一个数值补偿。
5. 沿每张卡片外侧虚线裁剪，保留整张 104×120 mm 卡片与编号。黑框上、左、右各约 12 mm 白边，底部另留编号区；不要沿 80 mm 黑边裁掉白边。
6. 卡片贴在平整硬底板上，保持四角平整，避开反光胶带覆盖黑框；白边干净、黑色实心。标定接触点是黑框中心，距四条黑边各 40 mm，不在码上画线、打孔或添加靶心。
7. 默认标定和识别预览尺寸均为 80 mm。保持相机与卡片固定；正式页面填写打印后的实测边长（mm）。

`A4_80mm_page_1.png`～`page_3.png` 是 300 DPI 页图，`marker_1.png`～`marker_6.png` 是独立卡片备用图。优先打印 PDF，避免图片查看器自动缩放。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, help="新图目录的父目录；默认 calibration_assets/generated")
    args = parser.parse_args(argv)
    try:
        out_dir = _new_output_dir(args.output_root)
        create_a4_print_pdf(out_dir / "ArUco_80mm_A4.pdf")
        for page in range(3):
            create_a4_calibration_sheet(out_dir / f"A4_80mm_page_{page + 1}.png", (page * 2 + 1, page * 2 + 2))
        for mid in range(1, 7):
            card = generate_printable_marker_card(mid, _px(MARKER_MM), _px(CARD_W_MM), _px(CARD_H_MM))
            save_png_with_dpi(out_dir / f"marker_{mid}.png", card, DPI)
        (out_dir / "打印与裁剪说明.md").write_text(PRINT_INSTRUCTIONS, encoding="utf-8")
        print(f"80 mm 图纸已生成：{out_dir}")
        print("优先打印 ArUco_80mm_A4.pdf：A4 纵向、3页、黑白单面、实际大小100%；沿104×120 mm虚线裁剪，保留12 mm白边。")
        return 0
    except (OSError, ValueError) as exc:
        print(f"图纸生成失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
