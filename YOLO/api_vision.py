import os
import sys
import json
import time
import cv2
import numpy as np
from openni import openni2
from rknnlite.api import RKNNLite

# 引入队友定义的接口
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from configs.common_interface import VisionInterface

# ==========================================
# 用户配置区
# ==========================================
TARGET_MODEL = "11l_berry_best-rk3588.rknn"  # 当前使用的模型文件名
IMGSZ = 736                                  # 模型输入分辨率
CONF_THRES = 0.25                            # 置信度阈值
IOU_THRES = 0.45                             # 重叠框过滤阈值
# --- 高清双摄分辨率配置 (物理端) ---
CAM_RGB_WIDTH, CAM_RGB_HEIGHT = 1280, 720
CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT = 640, 480 # ⚠️ 硬件限制：Astra Pro深度芯片最高仅支持640x480，强开1280会导致底层驱动 Segfault
# --- 推理裁剪画幅 ---
ROI_SIZE = 640                               # 按照队友要求，在1280x720中裁切中心 640x640
# ==========================================

# ================= 深度对齐引擎 (双摄分离版 + 内置除噪) =================
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
            params['mtx_ir'], params['dist_ir'], None, params['mtx_ir'], (ir_w, ir_h), cv2.CV_32FC1)
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
            # 【微调旋钮】：减小 top_pad 把深度图往上提。
            top_pad = 16  # 如果还需要往上提，改为 0；如果提过头了，改为 24
            bottom_pad = 64 - top_pad
            frame_depth = np.pad(depth_1280, ((top_pad, bottom_pad), (0, 0)), 'constant', constant_values=0)

        depth_undist = cv2.remap(frame_depth, self.mapx_ir, self.mapy_ir, cv2.INTER_NEAREST)
        valid = (depth_undist > 0) & (depth_undist < 10000)
        z = depth_undist[valid]
        if len(z) == 0: return frame_rgb, np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint16)
        
        X_rgb = z * self.V_rot_x[valid] + self.T_x
        Y_rgb = z * self.V_rot_y[valid] + self.T_y
        Z_rgb = z * self.V_rot_z[valid] + self.T_z
        
        u_rgb = np.round((X_rgb * self.fx_rgb / Z_rgb) + self.cx_rgb).astype(int)
        v_rgb = np.round((Y_rgb * self.fy_rgb / Z_rgb) + self.cy_rgb).astype(int)
        valid_proj = (u_rgb >= 0) & (u_rgb < self.rgb_w) & (v_rgb >= 0) & (v_rgb < self.rgb_h) & (Z_rgb > 0)
        u_rgb, v_rgb, z_rgb = u_rgb[valid_proj], v_rgb[valid_proj], Z_rgb[valid_proj]
        
        sort_idx = np.argsort(z_rgb)[::-1]
        aligned_depth = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint16)
        aligned_depth[v_rgb[sort_idx], u_rgb[sort_idx]] = z_rgb[sort_idx]
        
        aligned_depth_s16 = aligned_depth.astype(np.int16)
        cv2.filterSpeckles(aligned_depth_s16, 0, 200, 150)
        aligned_depth = aligned_depth_s16.astype(np.uint16)
        aligned_depth = cv2.medianBlur(aligned_depth, 5)
        
        return frame_rgb, aligned_depth


# ================= 核心视觉 API 类 =================
class StrawberryVisionAPI:
    def __init__(self, model_name=TARGET_MODEL, imgsz=IMGSZ):
        print(f"[VisionAPI] 正在初始化视觉接口...")
        self.imgsz = imgsz
        self.conf_thres = CONF_THRES
        self.iou_thres = IOU_THRES
        
        script_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(script_dir, model_name)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型: {self.model_path}")
        
        self.rknn = RKNNLite()
        self.rknn.load_rknn(self.model_path)
        self.rknn.init_runtime()
        
        calib_file = os.path.join(PROJECT_ROOT, "data/camera_params/astra_pro_calib.json")
        with open(calib_file) as f:
            calib = json.load(f)
            
        params = {k: np.array(v) for section in ['ir_depth_camera', 'rgb_camera']
                  for k, v in [
                      (f'mtx_{"ir" if "ir" in section else "rgb"}',
                       calib['stereo_calibration'][section]['camera_matrix']),
                      (f'dist_{"ir" if "ir" in section else "rgb"}',
                       calib['stereo_calibration'][section]['distortion_coeffs'])]}
        params['R'] = np.array(calib['stereo_calibration']['rotation_matrix'])
        params['T'] = np.array(calib['stereo_calibration']['translation_vector'])
        
        # ⚠️ 直接传入 640 宽高，类内部会自动除以 2 缩放标定矩阵
        self.aligner = DepthAligner(params, ir_w=640, ir_h=480, rgb_w=1280, rgb_h=720)
        self.fx, self.fy = params['mtx_rgb'][0, 0], params['mtx_rgb'][1, 1]
        self.cx, self.cy = params['mtx_rgb'][0, 2], params['mtx_rgb'][1, 2]

        openni_paths = [os.path.join(PROJECT_ROOT, "libs", "OpenNI2_arm64"), "/usr/lib", "/usr/local/lib"]
        path = next((p for p in openni_paths if os.path.exists(os.path.join(p, "libOpenNI2.so"))), None)
        openni2.initialize(path) if path else openni2.initialize()
        
        self.dev = openni2.Device.open_any()
        try: self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)
        except: pass
        
        self.ds = self.dev.create_depth_stream()
        vm = openni2.VideoMode()
        # ✨ 重大修复：强制物理层索要 640x480 的标准深度流，完美避开 Segfault！
        vm.resolutionX, vm.resolutionY, vm.fps = CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT, 30
        vm.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
        self.ds.set_video_mode(vm)
        try: self.ds.set_mirroring_enabled(False)
        except: pass
        self.ds.start()

        self.cap = None
        sys.path.append(os.path.join(PROJECT_ROOT, "calibration"))
        try:
            from capture import CAM_RGB_INDEX
            try_indices = [CAM_RGB_INDEX] if CAM_RGB_INDEX >= 0 else [0, 1, 2, 4]
        except:
            try_indices = [2, 0, 1, 4]
            
        for idx in try_indices:
            c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RGB_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RGB_HEIGHT)
                c.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = c.read()
                if ret: 
                    self.cap = c
                    break
                c.release()
        if self.cap is None:
            raise RuntimeError("无法打开 RGB 摄像头")
            
        print(f"[VisionAPI] 初始化完成，相机与模型已就绪。")

    def _infer_npu(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(cv2.resize(img_rgb, (self.imgsz, self.imgsz)), axis=0)

        outputs = self.rknn.inference(inputs=[img_in])
        preds = outputs[0][0].T 

        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        if scores.max() > 1.0 or scores.min() < -0.01:
            scores = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))

        mask = scores > self.conf_thres
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if len(boxes) == 0:
            return []

        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        w  = boxes[:, 2] * sx
        h  = boxes[:, 3] * sy

        indices = cv2.dnn.NMSBoxes(np.stack([x1, y1, w, h], 1).tolist(), scores.tolist(), self.conf_thres, self.iou_thres)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx1, by1 = max(0, int(x1[i])), max(0, int(y1[i]))
                bx2, by2 = min(w0, int(x1[i]+w[i])), min(h0, int(y1[i]+h[i]))
                results.append((bx1, by1, bx2, by2, float(scores[i]), int(class_ids[i])))
        return results

    def get_best_strawberry(self, save_debug_img=False, debug_img_name="debug_shot.jpg") -> VisionInterface:
        for _ in range(5):
            self.cap.grab()
            
        ret, frame = self.cap.read()
        if not ret:
            print("[VisionAPI] 获取 RGB 图像失败")
            return None

        try:
            # timeout=500，15帧大约66ms出一帧，绝对不会超时黑屏
            openni2.wait_for_any_stream([self.ds], timeout=500)
            depth_raw = np.frombuffer(self.ds.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(CAM_DEPTH_HEIGHT, CAM_DEPTH_WIDTH)
        except Exception as e:
            print(f"[VisionAPI] 获取深度数据失败: {e}")
            return None
            
        _, aligned = self.aligner.align(frame, depth_raw)

        # 裁剪 640x640 中心画幅
        h, w = frame.shape[:2]
        start_y = (h - ROI_SIZE) // 2
        start_x = (w - ROI_SIZE) // 2
        frame_roi = frame[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE].copy()
        aligned_roi = aligned[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE].copy()

        dets = self._infer_npu(frame_roi)
        if len(dets) == 0:
            print("[VisionAPI] 视野中未检测到目标")
            if save_debug_img:
                cv2.imwrite(debug_img_name, frame_roi)
            return None

        dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
        best_target = dets_sorted[0]
        
        x1, y1, x2, y2, conf, cls_id = best_target
        u_roi, v_roi = (x1+x2)//2, (y1+y2)//2
        
        hw = 6
        depth_patch = aligned_roi[max(0,v_roi-hw):min(ROI_SIZE,v_roi+hw+1), max(0,u_roi-hw):min(ROI_SIZE,u_roi+hw+1)]
        vd = depth_patch[depth_patch > 0]
        Z = float(np.median(vd)) if len(vd) > 0 else 0

        grade_str = "RIPE" if cls_id == 0 else "UNRIPE"

        if Z == 0:
            print("[VisionAPI] 目标区域无有效深度信息")
            if save_debug_img:
                color = (0, 0, 255) # 红色框代表深度失效
                cv2.rectangle(frame_roi, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame_roi, (u_roi, v_roi), 4, color, -1)
                cv2.putText(frame_roi, "NO DEPTH", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imwrite(debug_img_name, frame_roi)
                print(f"[VisionAPI] 失败状态图像已保存至: {debug_img_name}")
            return None
            
        # 还原坐标映射
        u_global = u_roi + start_x
        v_global = v_roi + start_y
        X = (u_global - self.cx) * Z / self.fx
        Y = (v_global - self.cy) * Z / self.fy

        if save_debug_img:
            color = (0, 255, 255)
            cv2.rectangle(frame_roi, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame_roi, (u_roi, v_roi), 4, color, -1)
            txt = f"TARGET: {grade_str} Z:{Z:.0f}mm"
            cv2.putText(frame_roi, txt, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(debug_img_name, frame_roi)
            print(f"[VisionAPI] 测试图像已保存至: {debug_img_name}")

        # 6. 坐标变换 (待标定)，目前返回相机坐标系数据
        robot_position = np.array([X, Y, Z], dtype=np.float64)

        # 7. 组装接口数据 (yaw_deg 暂无法计算，留空置为 0.0)
        result = VisionInterface(
            position=robot_position,
            yaw_deg=float(0.0),
            grade=str(grade_str)
        )
        return result

    def close(self):
        try:
            self.ds.stop()
            self.dev.close()
        except:
            pass
        if self.cap:
            self.cap.release()

# ================= 静态拍照测试 =================
if __name__ == "__main__":
    # 模型名字由顶部的 TARGET_MODEL 控制
    api = StrawberryVisionAPI()
    
    print("\n[测试] 准备进行静态抓拍测试...")
    time.sleep(3)
    
    print("[测试] 正在进行模型推理计算...")
    result = api.get_best_strawberry(save_debug_img=True, debug_img_name="test_l_model.jpg")
    
    print("\n" + "="*40)
    if result:
        print("[测试] 成功提取最优目标，接口返回数据结构如下:")
        print(f"VisionInterface:")
        print(f"  ├─ position (X,Y,Z) : {result.position} 毫米")
        print(f"  ├─ yaw_deg  (偏航角) : {result.yaw_deg} 度")
        print(f"  └─ grade    (品级)   : {result.grade}")
        
        print("\n[测试] --- 严格数据类型校验 ---")
        print(f"  ├─ 对象本体 : {type(result).__name__}")
        print(f"  ├─ position : {type(result.position)} (内部浮点类型: {result.position.dtype})")
        print(f"  ├─ yaw_deg  : {type(result.yaw_deg).__name__}")
        print(f"  └─ grade    : {type(result.grade).__name__}")
        
        print("\n[测试] 调试图像已保存为 test_l_model.jpg")
    else:
        print("[测试] 警告：未识别到目标或深度数据无效，请检查测试图像。")
    print("="*40)
    
    api.close()

