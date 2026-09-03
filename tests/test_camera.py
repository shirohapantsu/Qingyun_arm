import cv2
import numpy as np
from openni import openni2

# 全局变量保存当前状态
current_depth = None
last_clicked_pos = None
last_clicked_dist = None

def mouse_callback(event, x, y, flags, param):
    """鼠标左键双击：记录并打印选中点的像素坐标与真实深度距离（毫米）"""
    global current_depth, last_clicked_pos, last_clicked_dist
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if current_depth is not None:
            dist_mm = int(current_depth[y, x])
            last_clicked_pos = (x, y)
            last_clicked_dist = dist_mm
            print(f"-> 选中点: (x={x}, y={y}) | 距离: {dist_mm} mm ({dist_mm / 1000.0:.3f} 米)")

def init_rgb_camera(preferred_idx=1):
    """
    初始化奥比中光 RGB 彩色摄像头（锁定为索引 1，失败则回退）
    """
    for cam_idx in [preferred_idx, 0]:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # 设置分辨率为 640x480 以与深度流对齐
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f" -> 成功连接 RGB 摄像头 (设备索引: {cam_idx})")
                return cap, cam_idx
            cap.release()
    return None, -1

def main():
    global current_depth, last_clicked_pos, last_clicked_dist

    # 1. 初始化 OpenNI
    print("[1/4] 正在初始化 OpenNI2...")
    openni2.initialize()

    # 2. 打开奥比中光深度设备
    print("[2/4] 正在连接奥比中光深度设备...")
    dev = openni2.Device.open_any()
    dev_info = dev.get_device_info()
    name = dev_info.name.decode('utf-8') if isinstance(dev_info.name, bytes) else str(dev_info.name)
    print(f" -> 深度设备连接成功: {name}")

    # 尝试开启硬件级深度与彩色图视场对齐（Image Registration）
    try:
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print(" -> 已开启深度图与彩色图视场对齐 (Image Registration ON)")
    except Exception as e:
        print(f" -> 对齐模式设置提示: {e}")

    # 3. 启动深度流并关闭镜像模式（保证与真实物理空间和 RGB 视角左右一致）
    print("[3/4] 正在启动深度流并关闭镜像...")
    depth_stream = dev.create_depth_stream()
    
    # 关键设置：关闭深度流镜像，确保机械臂抓取的左右物理坐标正确！
    depth_stream.set_mirroring_enabled(False)
    
    depth_stream.start()

    # 4. 连接奥比中光 RGB 彩色摄像头（锁定索引 1）
    print("[4/4] 正在连接奥比中光 RGB 彩色摄像头...")
    cap, cam_idx = init_rgb_camera(preferred_idx=1)
    if cap is None:
        print(" [提示] 未能检测到可用彩色摄像头，仅显示深度流。")

    print("\n" + "=" * 60)
    print(">> 视觉采集系统已就绪：")
    print(f"   - 彩色摄像头: 奥比中光镜头 (索引 ID: {cam_idx})")
    print("   - 深度镜像: 已关闭 (与 RGB 同向)")
    print("   - 鼠标双击: 在【Depth Heatmap】或【RGB Color】窗口双击可查看实际距离")
    print("   - 退出方式: 按键盘 'q' 或 'Esc' 键安全退出")
    print("=" * 60 + "\n")

    # 双窗口均绑定鼠标双击事件
    cv2.namedWindow('Depth Heatmap')
    cv2.namedWindow('RGB Color')
    cv2.setMouseCallback('Depth Heatmap', mouse_callback)
    cv2.setMouseCallback('RGB Color', mouse_callback)

    try:
        while True:
            # --- A. 读取深度图 ---
            frame = depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            current_depth = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))

            # 生成伪彩色深度热力图
            depth_visual = cv2.convertScaleAbs(current_depth, alpha=0.05)
            depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            # --- B. 读取 RGB 彩色图 ---
            color_frame = None
            if cap is not None:
                ret, frame_bgr = cap.read()
                if ret:
                    color_frame = frame_bgr

            # 如果用户双击了某个点，在两个窗口上同步标记十字准星与距离数值
            if last_clicked_pos is not None:
                cx, cy = last_clicked_pos
                dist_str = f"{last_clicked_dist} mm ({last_clicked_dist / 1000.0:.3f}m)"

                # 在深度图上画白准星
                cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(depth_colormap, dist_str, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 在 RGB 图上画红准星
                if color_frame is not None:
                    cv2.drawMarker(color_frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(color_frame, dist_str, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 显示画面
            cv2.imshow('Depth Heatmap', depth_colormap)
            if color_frame is not None:
                cv2.imshow('RGB Color', color_frame)

            # 按 'q' 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        print("\n正在释放相机资源...")
        depth_stream.stop()
        dev.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        openni2.unload()
        print("相机已安全释放。")

if __name__ == "__main__":
    main()
