import os
import sys
import time
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------- 1. 加载所有内参外参 -----------------
def load_calibration_data():
    json_path = os.path.join(PROJECT_ROOT, "data", "camera_params", "astra_pro_calib.json")
    if not os.path.exists(json_path):
        print(f"❌ 找不到标定参数文件: {json_path}")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        calib = json.load(f)
        
    params = {}
    params['R'] = np.array(calib['stereo_calibration']['rotation_matrix'])
    params['T'] = np.array(calib['stereo_calibration']['translation_vector'])
    
    params['mtx_ir'] = np.array(calib['stereo_calibration']['ir_depth_camera']['camera_matrix'])
    params['dist_ir'] = np.array(calib['stereo_calibration']['ir_depth_camera']['distortion_coeffs'])
    
    params['mtx_rgb'] = np.array(calib['stereo_calibration']['rgb_camera']['camera_matrix'])
    params['dist_rgb'] = np.array(calib['stereo_calibration']['rgb_camera']['distortion_coeffs'])
    return params

# ----------------- 2. 核心算法：向量化深度图对齐 (支持双摄不同分辨率) -----------------
class DepthAligner:
    def __init__(self, params, ir_w=1280, ir_h=1024, rgb_w=1280, rgb_h=720):
        self.ir_w, self.ir_h = ir_w, ir_h
        self.rgb_w, self.rgb_h = rgb_w, rgb_h
        
        self.R_d2c = params['R'].T
        self.T_d2c = -self.R_d2c.dot(params['T'])
        
        self.fx_ir, self.fy_ir = params['mtx_ir'][0, 0], params['mtx_ir'][1, 1]
        self.cx_ir, self.cy_ir = params['mtx_ir'][0, 2], params['mtx_ir'][1, 2]
        self.fx_rgb, self.fy_rgb = params['mtx_rgb'][0, 0], params['mtx_rgb'][1, 1]
        self.cx_rgb, self.cy_rgb = params['mtx_rgb'][0, 2], params['mtx_rgb'][1, 2]

        self.mapx_ir, self.mapy_ir = cv2.initUndistortRectifyMap(
            params['mtx_ir'], params['dist_ir'], None, params['mtx_ir'], (ir_w, ir_h), cv2.CV_32FC1
        )

        v, u = np.indices((ir_h, ir_w))
        X_ray = (u - self.cx_ir) / self.fx_ir
        Y_ray = (v - self.cy_ir) / self.fy_ir
        
        V_flat = np.stack((X_ray, Y_ray, np.ones_like(X_ray)), axis=0).reshape(3, -1)
        V_rot_flat = self.R_d2c.dot(V_flat)
        self.V_rot_x = V_rot_flat[0].reshape(ir_h, ir_w)
        self.V_rot_y = V_rot_flat[1].reshape(ir_h, ir_w)
        self.V_rot_z = V_rot_flat[2].reshape(ir_h, ir_w)
        
        self.T_x, self.T_y, self.T_z = self.T_d2c[0, 0], self.T_d2c[1, 0], self.T_d2c[2, 0]

    def align(self, frame_rgb, frame_depth):
        if frame_depth.shape == (480, 640) and self.ir_h == 1024:
            depth_1280 = cv2.resize(frame_depth, (1280, 960), interpolation=cv2.INTER_NEAREST)
            # 【微调旋钮】：已知 pad 32 会导致深度图偏下十几个像素
            # 减小 top_pad 可以把深度图往上提。64 - top_pad 是底部的填充。
            top_pad = 16  # 如果还需要往上提，改为 0；如果提过头了，改为 24
            bottom_pad = 64 - top_pad
            frame_depth = np.pad(depth_1280, ((top_pad, bottom_pad), (0, 0)), 'constant', constant_values=0)

        depth_undist = cv2.remap(frame_depth, self.mapx_ir, self.mapy_ir, cv2.INTER_NEAREST)
        
        # 2. 提取有效深度点
        valid = (depth_undist > 0) & (depth_undist < 10000)
        z = depth_undist[valid]
        if len(z) == 0: return frame_rgb, np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint16)
            
        # 3. 极速重投影
        X_rgb = z * self.V_rot_x[valid] + self.T_x
        Y_rgb = z * self.V_rot_y[valid] + self.T_y
        Z_rgb = z * self.V_rot_z[valid] + self.T_z
        
        u_rgb = np.round((X_rgb * self.fx_rgb / Z_rgb) + self.cx_rgb).astype(int)
        v_rgb = np.round((Y_rgb * self.fy_rgb / Z_rgb) + self.cy_rgb).astype(int)
        
        # 4. 过滤越界与遮挡遮盖 (按照 RGB 分辨率边界过滤)
        valid_proj = (u_rgb >= 0) & (u_rgb < self.rgb_w) & (v_rgb >= 0) & (v_rgb < self.rgb_h) & (Z_rgb > 0)
        u_rgb, v_rgb, z_rgb = u_rgb[valid_proj], v_rgb[valid_proj], Z_rgb[valid_proj]
        
        sort_idx = np.argsort(z_rgb)[::-1]
        aligned_depth = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint16)
        aligned_depth[v_rgb[sort_idx], u_rgb[sort_idx]] = z_rgb[sort_idx]
        
        # 【工业级除噪】：OpenCV 的 filterSpeckles 只认 int16，所以需要先转换一下类型
        aligned_depth_s16 = aligned_depth.astype(np.int16)
        cv2.filterSpeckles(aligned_depth_s16, 0, 200, 150)
        aligned_depth = aligned_depth_s16.astype(np.uint16)
        
        aligned_depth = cv2.medianBlur(aligned_depth, 5)
        
        return frame_rgb, aligned_depth

# ----------------- 3. 主函数：启动相机进行验证 -----------------
def get_openni_path():
    paths = [os.path.join(PROJECT_ROOT, "libs", "OpenNI2_arm64"), "/usr/lib", "/usr/local/lib"]
    for p in paths:
        if os.path.exists(os.path.join(p, "libOpenNI2.so")): return p
    return None

aligned_depth_global = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and aligned_depth_global is not None:
        d = aligned_depth_global[y, x]
        print(f"🎯 点击位置 (x={x}, y={y}) -> 对齐深度: {d} mm ({d/1000.0:.3f} 米)")

def main():
    global aligned_depth_global
    print("⏳ 正在加载 1280 高清标定矩阵并初始化对齐器...")
    params = load_calibration_data()
    aligner = DepthAligner(params, ir_w=1280, ir_h=1024, rgb_w=1280, rgb_h=720)
    
    print("⏳ 正在启动硬件视频流...")
    try:
        from openni import openni2
        path = get_openni_path()
        print(" -> [Debug] 正在初始化 OpenNI2...")
        openni2.initialize(path) if path else openni2.initialize()
        dev = openni2.Device.open_any()
        try: dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)
        except: pass
        
        print(" -> [Debug] 正在创建深度流...")
        depth_stream = dev.create_depth_stream()
        
        print(" -> [Debug] 正在配置深度流 640x480 分辨率...")
        vm = openni2.VideoMode()
        vm.resolutionX, vm.resolutionY, vm.fps, vm.pixelFormat = 640, 480, 30, openni2.PIXEL_FORMAT_DEPTH_1_MM
        depth_stream.set_video_mode(vm)
        
        try: depth_stream.set_mirroring_enabled(False)
        except: pass
        
        print(" -> [Debug] 正在启动深度流...")
        depth_stream.start()
        print(" -> [Debug] 深度流启动完毕！")
    except Exception as e:
        print(f"❌ 深度摄像头启动失败: {e}")
        return

    # 启动 OpenCV RGB
    cap = None
    sys.path.append(os.path.join(PROJECT_ROOT, "calibration"))
    from capture import CAM_RGB_INDEX
    try_indices = [CAM_RGB_INDEX] if CAM_RGB_INDEX >= 0 else [0, 1, 2, 4]
    for idx in try_indices:
        print(f" -> [Debug] 准备探测 RGB 摄像头 idx={idx}...")
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            print(f" -> [Debug] 成功 open，准备设 FOURCC...")
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print(f" -> [Debug] 设宽 1280...")
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            print(f" -> [Debug] 设高 720...")
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f" -> [Debug] 设帧率 30...")
            c.set(cv2.CAP_PROP_FPS, 30)
            print(f" -> [Debug] 准备读取第一帧 (此处极易崩溃)...")
            ret, _ = c.read()
            if ret: 
                print(f" -> [Debug] 第一帧读取成功！")
                cap = c; break
            c.release()

    if cap is None:
        print("❌ RGB 摄像头启动失败")
        return

    print("\n✅ 启动成功！")
    print("💡 操作指南：")
    print("  - 画面是【畸变纠正 + 软对齐】后的融合效果（彩色图上覆了深度热力图）")
    print("  - 用鼠标左键点击画面，终端会输出该点精准的真实 3D 距离")
    print("  - 按 'Q' 键退出\n")

    cv2.namedWindow('Depth-To-Color Alignment', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Depth-To-Color Alignment', mouse_callback)

    while True:
        try:
            openni2.wait_for_any_stream([depth_stream], timeout=500)
            frame_data = depth_stream.read_frame().get_buffer_as_uint16()
            depth_raw = np.frombuffer(frame_data, dtype=np.uint16).reshape((480, 640))
        except Exception:
            continue
            
        ret, frame_rgb = cap.read()
        if not ret: continue
        
        rgb_undist, aligned_depth_global = aligner.align(frame_rgb, depth_raw)
        
        depth_visual = cv2.convertScaleAbs(aligned_depth_global, alpha=0.05)
        depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        depth_colormap[aligned_depth_global == 0] = 0
        
        combined = cv2.addWeighted(rgb_undist, 0.7, depth_colormap, 0.4, 0)
        
        # 缩放至方便查看的大小 (比如 1280x720 太大，我们用 WINDOW_NORMAL 让用户自适应拉伸，这里直接传原图即可)
        cv2.imshow('Depth-To-Color Alignment', combined)
        
        if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
            break

    depth_stream.stop()
    dev.close()
    openni2.unload()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

