import os
import sys
import time
import cv2
import numpy as np

# 尝试导入 openni 库
try:
    from openni import openni2
except ImportError:
    print("❌ 未检测到 openni 库，请先在当前环境运行: pip install openni")
    sys.exit(1)

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
            print(f"🎯 选中点: (x={x}, y={y}) | 距离: {dist_mm} mm ({dist_mm / 1000.0:.3f} 米)")

def init_rgb_camera(preferred_indices=None):
    """
    初始化奥比中光 RGB 彩色摄像头：
    自动根据操作系统适配底层驱动（Linux: CAP_V4L2, Windows: CAP_DSHOW）
    """
    if preferred_indices is None:
        preferred_indices = (0, 1, 2, 3, 4, 5)

    is_win = sys.platform.startswith('win')
    backend = cv2.CAP_DSHOW if is_win else cv2.CAP_V4L2

    for cam_idx in preferred_indices:
        cap = cv2.VideoCapture(cam_idx, backend)
        if cap.isOpened():
            # 【关键修改】：强制使用 MJPG 压缩格式，极大降低 USB 带宽消耗，避免死锁
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, _ = cap.read()
            if ret:
                print(f" -> 成功连接 RGB 彩色摄像头 (设备索引: {cam_idx}, 后端: {'DSHOW' if is_win else 'V4L2'})")
                return cap, cam_idx
            cap.release()
    return None, -1

def locate_openni_redist():
    """
    自动定位 OpenNI2 动态链接库所在的目录：
    - Windows: 寻找根目录下的 OpenNI2.dll
    - Linux (ARM64 香橙派): 寻找 libs/OpenNI2_arm64 或 sdk/libs
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if sys.platform.startswith('win'):
        return project_root
    else:
        import platform
        arch = platform.machine().lower()
        if 'aarch64' in arch or 'arm' in arch:
            candidate_paths = [
                os.path.join(project_root, "OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d", "samples", "samples", "ThirdParty", "OpenNI2", "arm", "Arm64"),
                os.path.join(project_root, "libs", "OpenNI2_arm64"),
                os.path.join(project_root, "OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d", "sdk", "libs"),
                os.path.join(project_root, "OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d", "samples", "bin"),
                "/usr/lib",
                "/usr/local/lib"
            ]
        else:
            candidate_paths = [
                os.path.join(project_root, "OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d", "samples", "samples", "ThirdParty", "OpenNI2", "linux", "x64"),
                "/usr/lib",
                "/usr/local/lib"
            ]
        for path in candidate_paths:
            if os.path.exists(os.path.join(path, "libOpenNI2.so")):
                return path
        return None

def main():
    global current_depth, last_clicked_pos, last_clicked_dist

    # 1. 初始化 OpenNI
    redist_path = locate_openni_redist()
    print(f"[1/4] 正在初始化 OpenNI2 (库路径: {redist_path})...")
    if redist_path:
        openni2.initialize(redist_path)
    else:
        openni2.initialize()

    # 2. 打开奥比中光深度设备
    print("[2/4] 正在连接奥比中光深度设备...")
    try:
        dev = openni2.Device.open_any()
    except Exception:
        print("❌ 无法打开奥比中光深度设备 (具体原因请看上方 OpenNI 底层输出)")
        print("💡 请排查：1. 相机 USB 是否插紧？ 2. Linux 是否已安装 udev 权限规则？")
        openni2.unload()
        return

    dev_info = dev.get_device_info()
    name = dev_info.name.decode('utf-8') if isinstance(dev_info.name, bytes) else str(dev_info.name)
    uri = dev_info.uri.decode('utf-8') if isinstance(dev_info.uri, bytes) else str(dev_info.uri)
    print(f" -> 深度设备连接成功: {name} ({uri})")

    # 开启硬件级深度与彩色图对齐（Image Registration）
    try:
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print(" -> 已开启深度图与彩色图对齐 (Image Registration: ON)")
    except Exception as e:
        print(f" -> 提示: 对齐模式未能开启或已默认对齐: {e}")

    # 3. 启动深度流
    print("[3/4] 正在启动深度流...")
    depth_stream = dev.create_depth_stream()
    
    # 【关键修改】：显式设置深度流格式，防止参数飘移
    try:
        video_mode = openni2.VideoMode()
        video_mode.resolutionX = 640
        video_mode.resolutionY = 480
        video_mode.fps = 30
        video_mode.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
        depth_stream.set_video_mode(video_mode)
    except Exception as e:
        pass

    depth_stream.set_mirroring_enabled(False)  # 关闭镜像，确保真实物理坐标
    depth_stream.start()

    # 4. 连接 RGB 彩色摄像头
    print("[4/4] 正在连接 RGB 彩色摄像头...")
    # 恢复 RGB 相机调用
    cap, cam_idx = init_rgb_camera()
    if cap is None:
        print(" [提示] 未能检测到彩色视频流，仅显示深度流预览。")

    has_display = bool(os.environ.get('DISPLAY')) and ('--headless' not in sys.argv)

    print("\n" + "=" * 60)
    print(">> 视觉采集系统已就绪：")
    print(f"   - 运行平台: {sys.platform} ({'Linux ARM64' if sys.platform.startswith('linux') else 'Windows'})")
    print(f"   - 彩色相机索引: {cam_idx}")
    if has_display:
        print("   - 运行模式: 图形交互模式 (GUI)")
        print("   - 鼠标双击: 在任意预览窗口双击可查看目标点的毫米距离 (mm)")
        print("   - 退出方式: 按键盘 'q' 或 'Esc' 键退出")
    else:
        print("   - 运行模式: 纯终端命令行模式 (SSH Headless，无需图形界面)")
        print("   - 实时反馈: 终端将实时刷新打印 FPS、中心点毫米测距和状态")
        print("   - 退出方式: 按终端键盘 Ctrl + C 退出")
    print("=" * 60 + "\n")

    if has_display:
        cv2.namedWindow('Depth Heatmap')
        cv2.setMouseCallback('Depth Heatmap', mouse_callback)
        if cap is not None:
            cv2.namedWindow('RGB Color')
            cv2.setMouseCallback('RGB Color', mouse_callback)

    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    try:
        while True:
            # 加入超时保护：如果有任一流有数据，再读取，防止彻底死锁
            try:
                openni2.wait_for_any_stream([depth_stream], timeout=1.0)
            except Exception:
                continue

            # --- A. 读取深度流 ---
            frame = depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            current_depth = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))

            # 生成伪彩色热力图
            depth_visual = cv2.convertScaleAbs(current_depth, alpha=0.05)
            depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            # --- B. 读取 RGB 彩色流 ---
            color_frame = None
            if cap is not None:
                ret, frame_bgr = cap.read()
                if ret:
                    color_frame = frame_bgr

            # 帧率与中心点统计
            frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                fps = frame_count / (now - fps_start_time)
                frame_count = 0
                fps_start_time = now

            center_depth_mm = int(current_depth[240, 320])

            if has_display:
                # 如果点击了某个点，绘制十字准星与距离文字
                if last_clicked_pos is not None:
                    cx, cy = last_clicked_pos
                    dist_str = f"{last_clicked_dist} mm ({last_clicked_dist / 1000.0:.3f}m)"

                    cv2.drawMarker(depth_colormap, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(depth_colormap, dist_str, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if color_frame is not None:
                        cv2.drawMarker(color_frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                        cv2.putText(color_frame, dist_str, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 刷新显示
                cv2.imshow('Depth Heatmap', depth_colormap)
                if color_frame is not None:
                    cv2.imshow('RGB Color', color_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            else:
                # 纯命令行模式，实时单行刷新输出
                status_text = f"\r📡 [实时采集] 帧率: {fps:4.1f} FPS | 画面中心距离: {center_depth_mm:4d} mm ({center_depth_mm / 1000.0:.3f}m) | RGB彩色: {'正常' if color_frame is not None else '未连接'}"
                sys.stdout.write(status_text)
                sys.stdout.flush()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n已捕获 Ctrl+C，正在退出...")
    finally:
        print("正在释放相机资源...")
        depth_stream.stop()
        dev.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("相机资源已释放完毕。")

if __name__ == "__main__":
    main()
