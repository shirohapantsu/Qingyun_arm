import os
import sys
import math
import json
import numpy as np
import cv2
from openni import openni2
from rknnlite.api import RKNNLite

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.common_interface import VisionInterface

class NoTarget(Exception):
    pass

class VisionHardError(Exception):
    pass

# ================= 视觉物理与标定参数 =================
WORKSPACE_Z_MIN = 0.1   # m (相机坐标系)
WORKSPACE_Z_MAX = 0.8   # m
MAX_ENVELOPE_LENGTH = 0.10  # 10cm
MAX_ENVELOPE_WIDTH = 0.08   # 8cm
MIN_CLEARANCE_M = 0.03      # 最小净距 3cm

TARGET_MODEL = "YOLO/11l_berry_best-rk3588.rknn"
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

CAM_RGB_WIDTH, CAM_RGB_HEIGHT = 1280, 720
CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT = 640, 480
ROI_SIZE = 640

# --- 深度对齐引擎 ---
class DepthAligner:
    def __init__(self, params, ir_w=1280, ir_h=1024, rgb_w=1280, rgb_h=720):
        self.ir_w, self.ir_h = ir_w, ir_h
        self.rgb_w, self.rgb_h = rgb_w, rgb_h
        self.R_d2c = np.array(params['R']).T
        self.T_d2c = -self.R_d2c.dot(np.array(params['T']))
        mtx_ir = np.array(params['mtx_ir'])
        mtx_rgb = np.array(params['mtx_rgb'])
        dist_ir = np.array(params['dist_ir'])
        self.fx_ir, self.fy_ir = mtx_ir[0, 0], mtx_ir[1, 1]
        self.cx_ir, self.cy_ir = mtx_ir[0, 2], mtx_ir[1, 2]
        self.fx_rgb, self.fy_rgb = mtx_rgb[0, 0], mtx_rgb[1, 1]
        self.cx_rgb, self.cy_rgb = mtx_rgb[0, 2], mtx_rgb[1, 2]
        self.mapx_ir, self.mapy_ir = cv2.initUndistortRectifyMap(
            mtx_ir, dist_ir, None, mtx_ir, (ir_w, ir_h), cv2.CV_32FC1)
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
            top_pad = 16
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
        
        kernel = np.ones((3, 3), np.uint8)
        aligned_depth = cv2.dilate(aligned_depth, kernel, iterations=1)
        aligned_depth = cv2.medianBlur(aligned_depth, 5)
        
        return frame_rgb, aligned_depth


# ================= 核心视觉 API 类 =================
class StrawberryVisionAPI:
    def __init__(self):
        try:
            self._init_rknn()
            self._init_camera()
        except Exception as e:
            raise VisionHardError(f"初始化失败: {e}")

    def _init_rknn(self):
        model_path = os.path.join(PROJECT_ROOT, TARGET_MODEL)
        if not os.path.exists(model_path):
            raise VisionHardError(f"模型未找到: {model_path}")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise VisionHardError("模型加载失败")
        if self.rknn.init_runtime() != 0:
            raise VisionHardError("模型运行时初始化失败")

    def _init_camera(self):
        # 标定文件
        calib_file = os.path.join(PROJECT_ROOT, "data/camera_params/astra_pro_calib.json")
        if not os.path.exists(calib_file):
            raise VisionHardError(f"标定文件未找到: {calib_file}")
        with open(calib_file, 'r') as f:
            calib = json.load(f)
            
        self.aligner = DepthAligner(calib, ir_w=1280, ir_h=1024, rgb_w=1280, rgb_h=720)
        self.fx, self.fy = self.aligner.fx_rgb, self.aligner.fy_rgb
        self.cx, self.cy = self.aligner.cx_rgb, self.aligner.cy_rgb
        
        # 寻找 OpenNI 路径
        openni_path = None
        for p in [os.path.join(PROJECT_ROOT, "libs", "OpenNI2_arm64"), "/usr/lib", "/usr/local/lib"]:
            if os.path.exists(os.path.join(p, "libOpenNI2.so")):
                openni_path = p
                break
        if openni_path is None:
            raise VisionHardError("未找到 libOpenNI2.so")
            
        openni2.initialize(openni_path)
        self.dev = openni2.Device.open_any()
        self.ds = self.dev.create_depth_stream()
        vm = openni2.VideoMode()
        vm.resolutionX, vm.resolutionY = CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT
        vm.fps = 30
        vm.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
        self.ds.set_video_mode(vm)
        try: self.ds.set_mirroring_enabled(False)
        except: pass
        self.ds.start()

        self.cap = None
        for i in range(6):
            c = cv2.VideoCapture(i, cv2.CAP_V4L2)
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
            raise VisionHardError("无法打开 RGB 摄像头")

    def _infer_npu_obb(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(cv2.resize(img_rgb, (IMGSZ, IMGSZ)), axis=0)

        outputs = self.rknn.inference(inputs=[img_in])
        preds = outputs[0][0].T  # 通常 shape 为 (8400, 4 + C + 1)
        
        boxes = preds[:, :4]     # cx, cy, w, h
        class_scores = preds[:, 4:-1]
        angles_rad = preds[:, -1] # Ultralytics OBB 角度 (弧度)
        
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        if scores.max() > 1.0 or scores.min() < -0.01:
            scores = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))
            
        mask = scores > CONF_THRES
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        angles_rad = angles_rad[mask]
        
        if len(boxes) == 0:
            return []
            
        sx, sy = w0 / IMGSZ, h0 / IMGSZ
        
        bboxes_for_nms = []
        for i in range(len(boxes)):
            cx = boxes[i, 0] * sx
            cy = boxes[i, 1] * sy
            w  = boxes[i, 2] * sx
            h  = boxes[i, 3] * sy
            angle_deg = angles_rad[i] * 180.0 / math.pi
            bboxes_for_nms.append(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
            
        # OpenCV 的 OBB NMS
        indices = cv2.dnn.NMSBoxesRotated(bboxes_for_nms, scores.tolist(), CONF_THRES, IOU_THRES)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                cx, cy = bboxes_for_nms[i][0]
                w, h = bboxes_for_nms[i][1]
                angle_rad = angles_rad[i]
                results.append((cx, cy, w, h, angle_rad, float(scores[i]), int(class_ids[i])))
                
        return results

    def fetch_targets(self):
        # 冲刷旧帧
        for _ in range(5):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret:
            raise VisionHardError("RGB 抽帧失败")
            
        try:
            openni2.wait_for_any_stream([self.ds], timeout=500)
            depth_raw = np.frombuffer(self.ds.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        except Exception as e:
            raise VisionHardError(f"深度抽帧失败: {e}")

        _, aligned = self.aligner.align(frame, depth_raw)

        # 裁剪 640x640 中心画幅
        h_full, w_full = frame.shape[:2]
        start_y = (h_full - ROI_SIZE) // 2
        start_x = (w_full - ROI_SIZE) // 2
        frame_roi = frame[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE]
        aligned_roi = aligned[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE]

        dets = self._infer_npu_obb(frame_roi)
        
        all_candidates = []
        for det in dets:
            cx_roi, cy_roi, w_roi, h_roi, angle_rad, conf, cls_id = det
            u_roi, v_roi = int(cx_roi), int(cy_roi)
            
            Z = 0
            for hw in [8, 15, 25, 40]:  # 水波纹式扩大搜索区域
                patch = aligned_roi[max(0,v_roi-hw):min(ROI_SIZE,v_roi+hw+1), max(0,u_roi-hw):min(ROI_SIZE,u_roi+hw+1)]
                vd = patch[patch > 0]
                if len(vd) > (hw * hw * 0.1):
                    Z = float(np.median(vd)) / 1000.0  # 转为米
                    break
                    
            if Z == 0:
                continue
                
            u_global = cx_roi + start_x
            v_global = cy_roi + start_y
            
            X_m = (u_global - self.cx) * Z / self.fx
            Y_m = (v_global - self.cy) * Z / self.fy
            Z_m = Z
            
            # 计算物理长宽
            physical_w = w_roi * Z / self.fx
            physical_h = h_roi * Z / self.fy
            
            # 长短轴与偏航角对齐 (约定 length_m 为较长边)
            if physical_h > physical_w:
                length_m = physical_h
                width_m = physical_w
                yaw_deg = (angle_rad * 180.0 / math.pi) + 90
            else:
                length_m = physical_w
                width_m = physical_h
                yaw_deg = angle_rad * 180.0 / math.pi
                
            # 钳位至 [-90, 90)
            yaw_deg = (yaw_deg + 90) % 180 - 90
            
            # 类别 (0 为 ripe)
            ripe = (cls_id == 0)
            
            all_candidates.append({
                'X': X_m, 'Y': Y_m, 'Z': Z_m,
                'length_m': length_m, 'width_m': width_m,
                'yaw_deg': yaw_deg,
                'ripe': ripe,
                'u': u_global, 'v': v_global
            })
            
        # 计算 邻物间距 (clearance)
        for i, c in enumerate(all_candidates):
            min_clearance = float('inf')
            for j, oc in enumerate(all_candidates):
                if i == j: continue
                dist = math.hypot(c['X']-oc['X'], c['Y']-oc['Y'])
                edge_dist = dist - (max(c['length_m'], c['width_m'])/2 + max(oc['length_m'], oc['width_m'])/2)
                if edge_dist < min_clearance:
                    min_clearance = edge_dist
            c['clearance'] = min_clearance
            
        # 五项可抓性过滤 (无视下限)
        valid_cands = []
        for c in all_candidates:
            if not (WORKSPACE_Z_MIN <= c['Z'] <= WORKSPACE_Z_MAX):
                continue
            if c['length_m'] > MAX_ENVELOPE_LENGTH or c['width_m'] > MAX_ENVELOPE_WIDTH:
                continue
            if c['clearance'] < MIN_CLEARANCE_M:
                continue
            valid_cands.append(c)
            
        if not valid_cands:
            raise NoTarget("过滤后无合法候选目标")
            
        # 确定性排名: 1. clearance 降序; 2. 几何中心距离 升序; 3. position (X) 升序
        center_x, center_y = CAM_RGB_WIDTH / 2, CAM_RGB_HEIGHT / 2
        for c in valid_cands:
            c['roi_dist'] = math.hypot(c['u'] - center_x, c['v'] - center_y)
            
        valid_cands.sort(key=lambda c: (-c['clearance'], c['roi_dist'], c['X'], c['Y'], c['Z']))
        
        return valid_cands

# ================= 全局单例接口 =================
_api_instance = None

def init() -> None:
    global _api_instance
    if _api_instance is None:
        _api_instance = StrawberryVisionAPI()

def get_target(ignore: int = 0) -> VisionInterface:
    if _api_instance is None:
        raise VisionHardError("尚未调用 init() 初始化视觉模块")
        
    candidates = _api_instance.fetch_targets()
    valid_count = len(candidates)
    
    if ignore >= valid_count:
        raise NoTarget(f"穷尽兜底: 忽略次数({ignore})已覆盖当前全部合法目标数({valid_count})")
        
    c = candidates[ignore]
    
    # 目前先将相机坐标系当作 base_link 占位，待手眼标定后此处加入 T_cam2base 变换
    # 或者直接把 T_cam2base 的运算放在运动模块处理，这里遵守接口文档输出相机三维空间坐标
    pos = np.array([c['X'], c['Y'], c['Z']], dtype=np.float64)
    
    return VisionInterface(
        position=pos,
        yaw_deg=float(c['yaw_deg']),
        length_m=float(c['length_m']),
        width_m=float(c['width_m']),
        ripe=bool(c['ripe']),
        valid_count=int(valid_count)
    )
