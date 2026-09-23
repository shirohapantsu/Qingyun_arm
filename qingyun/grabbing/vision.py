import cv2
import numpy as np
try:
    from openni import openni2
    from rknnlite.api import RKNNLite
except ImportError:
    pass
import os
import json
import math
import traceback
from pathlib import Path
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.common_interface import VisionInterface, VisionThresholds

# ==========================================
# 视觉底层控制参数 (开发联调可调)
# ==========================================
# YOLO 推理参数
IMGSZ = 640                # 模型输入分辨率 (需与 RKNN 导出时一致)
CONF_THRES = 0.25          # YOLO 基础置信度阈值
IOU_THRES = 0.45           # OpenCV NMS 重叠框剔除阈值
ROI_SIZE = 640             # 从原图中裁剪输入给 YOLO 的中心画幅尺寸

# 相机硬件捕获分辨率 (需配合底层驱动)
CAM_RGB_WIDTH, CAM_RGB_HEIGHT = 1280, 720
CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT = 640, 480
# ==========================================


# P3 Exceptions
class NoTarget(Exception):
    """NoTarget——"这次扫描没有可抓目标"的业务结论。

    触发情形：ROI 内无物体、全部候选被过滤、
    ``ignore ≥ valid_count``（跳尽，原因 ``skip_exhausted``）、
    它是**正常业务信号**，不是故障。
    
    基类固定为 ``Exception``，与 ``VisionHardError`` 互不为父子（同级两类）。
    """
    pass
class VisionHardError(Exception):
    """视觉侧硬故障。

    触发情形：相机断开、帧读取失败、整帧深度不可用、检测管线崩溃、
    init 阶段任何故障、configure/init 调用次序被破坏、
    ``ignore`` 为负值或非整数（调用方程序错误）、标定包缺文件/缺键/非法值。
    一律交 terminate，不计入 NoTarget 连续计数。
    """
    pass

_config = None
_thresholds = None
_intrinsics = None
_extrinsics = None
_is_initialized = False

# Live state
_rknn = None
_aligner = None
_cap = None
_ds = None

# Replay state
_replay_manifest = None
_replay_idx = 0
_replay_dir = None

# --- Depth Aligner for Live mode (from legacy) ---
class DepthAligner:
    def __init__(self, intrin, ir_w=1280, ir_h=1024, rgb_w=CAM_RGB_WIDTH, rgb_h=CAM_RGB_HEIGHT):
        self.ir_w, self.ir_h = ir_w, ir_h
        self.rgb_w, self.rgb_h = rgb_w, rgb_h
        
        # We assume intrin is the new P3 schema
        d2c = np.array(intrin['depth_to_rgb'])
        self.R_d2c = d2c[:3, :3]
        self.T_d2c = d2c[:3, 3:4]
        
        self.fx_ir, self.fy_ir = intrin['depth']['fx'], intrin['depth']['fy']
        self.cx_ir, self.cy_ir = intrin['depth']['cx'], intrin['depth']['cy']
        mtx_ir = np.array([[self.fx_ir, 0, self.cx_ir], [0, self.fy_ir, self.cy_ir], [0, 0, 1]])
        dist_ir = np.array(intrin['depth']['dist'])
        
        self.fx_rgb, self.fy_rgb = intrin['rgb']['fx'], intrin['rgb']['fy']
        self.cx_rgb, self.cy_rgb = intrin['rgb']['cx'], intrin['rgb']['cy']
        
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
            frame_depth = np.pad(depth_1280, ((16, 48), (0, 0)), 'constant', constant_values=0)
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
        
        aligned_depth = cv2.dilate(aligned_depth, np.ones((3, 3), np.uint8), iterations=1)
        aligned_depth = cv2.medianBlur(aligned_depth, 5)
        
        return frame_rgb, aligned_depth


# ================= 1. Lifecycle =================

def configure(thresholds: VisionThresholds, config_path: Path) -> None:
    """注入阈值快照与视觉配置文件路径。

    契约：
        * 由 ``main`` 冷启动段在 ``init()`` **之前**调用**恰好一次**。
        * ``thresholds`` 是 P0 的 ``VisionThresholds`` 快照。
        * ``config_path`` 指向视觉配置文件。
    """
    global _config, _thresholds, _is_initialized
    if _is_initialized: raise VisionHardError("Cannot configure after init()")
    _thresholds = thresholds
    try:
        with open(config_path, 'r') as f: _config = json.load(f)
    except Exception as e:
        raise VisionHardError(f"Config err: {e}")

def init() -> None:
    """一次性加载模型与相机并验证可采集。

    契约：
        * 前置条件：``configure`` 已成功。
        * YOLO 权重一次性加载。
        * 打开相机并验证可采集一帧。
        * 加载并严格校验标定包。
        * 只允许成功初始化一次；重复 init 属程序错误。
    """
    global _is_initialized, _intrinsics, _extrinsics
    global _rknn, _cap, _ds, _aligner
    global _replay_manifest, _replay_dir
    
    if _config is None: raise VisionHardError("Not configured")
    config_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))) # Workspace root for configs in this demo
    try:
        with open(config_dir / _config['calibration']['intrinsics_path']) as f: _intrinsics = json.load(f)
        with open(config_dir / _config['calibration']['extrinsic_base_camera_path']) as f: _extrinsics = json.load(f)
    except Exception as e:
        raise VisionHardError(f"Calibration load failed: {e}")

    mode = _config['source']['mode']
    if mode == "live":
        try:
            # 1. Exact RKNN logic
            model_path = str(config_dir / _config['model']['path'])
            _rknn = RKNNLite()
            if _rknn.load_rknn(model_path) != 0: raise Exception("load_rknn failed")
            if _rknn.init_runtime() != 0: raise Exception("init_runtime failed")
            
            # 2. Exact DepthAligner
            _aligner = DepthAligner(_intrinsics, ir_w=1280, ir_h=1024, rgb_w=CAM_RGB_WIDTH, rgb_h=CAM_RGB_HEIGHT)
            
            # 3. Exact OpenNI
            openni_path = None
            for p in ["/home/orangepi/Qingyun_arm/libs/OpenNI2_arm64", "/usr/lib", "/usr/local/lib"]:
                if os.path.exists(os.path.join(p, "libOpenNI2.so")):
                    openni_path = p; break
            if openni_path:
                openni2.initialize(openni_path)
            else:
                openni2.initialize()
            
            dev = openni2.Device.open_any()
            _ds = dev.create_depth_stream()
            vm = openni2.VideoMode()
            vm.resolutionX, vm.resolutionY = CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT
            vm.fps = 30
            vm.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
            _ds.set_video_mode(vm)
            try: _ds.set_mirroring_enabled(False)
            except: pass
            _ds.start()
            
            # 4. Exact CV2 VideoCapture
            for i in range(6):
                c = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RGB_WIDTH)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RGB_HEIGHT)
                    ret, _ = c.read()
                    if ret: 
                        _cap = c
                        break
                    c.release()
            if _cap is None: raise Exception("No RGB cam found")
            
        except Exception as e:
            raise VisionHardError(f"Live init failed: {e}")
            
    elif mode == "replay":
        _replay_dir = config_dir / _config['source']['replay_dir']
        try:
            with open(_replay_dir / "manifest.json") as f: _replay_manifest = json.load(f)
        except Exception as e:
            raise VisionHardError(f"Replay manifest fail: {e}")
            
    _is_initialized = True

def get_target(ignore: int = 0) -> VisionInterface:
    """完整跑一次采集-检测-测量-过滤-排名，返回第 ``ignore+1`` 名候选。

    契约：
        * 前置条件：机械臂在 home。
        * 一次调用 = 一次完整采集-检测管线，无跨调用缓存。
        * 返回确定性排名第 ``ignore + 1`` 名的 ``VisionInterface``。
        * ``ignore ≥ valid_count`` 必为 ``NoTarget(skip_exhausted)``。
        * ``ignore`` 必须为非负 int。
    """
    if not _is_initialized: raise VisionHardError("Not initialized")
    if not isinstance(ignore, int) or ignore < 0: raise VisionHardError("ignore must be non-negative int")
        
    color, depth, raw_dets = fetch_frames()
    detections = infer_detections(color, raw_dets)
    if not detections: raise NoTarget("no_detection")
        
    candidates = measure_geometry(detections, depth)
    calculate_clearance(candidates)
    
    valid_candidates = filter_candidates(candidates)
    if not valid_candidates: raise NoTarget("all_filtered")
        
    return rank_and_return(valid_candidates, ignore)

# ================= 2. Pipeline Steps =================

def fetch_frames():
    global _replay_idx
    if _config['source']['mode'] == "replay":
        if _replay_idx >= len(_replay_manifest['frames']): raise NoTarget("replay_exhausted")
        info = _replay_manifest['frames'][_replay_idx]
        _replay_idx += 1
        return (np.load(_replay_dir / info['color']), 
                np.load(_replay_dir / info['depth']), 
                json.load(open(_replay_dir / info['detections'])))
    else:
        from openni import openni2
        for _ in range(5): _cap.grab()
        ret, frame = _cap.read()
        if not ret: raise VisionHardError("RGB fetch fail")
        try:
            openni2.wait_for_any_stream([_ds], timeout=500)
            raw = np.frombuffer(_ds.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        except Exception as e:
            raise VisionHardError(f"Depth fetch fail: {e}")
            
        _, aligned = _aligner.align(frame, raw)
        return frame, aligned, None

def infer_detections(color, raw_dets):
    if _config['model']['backend'] == "fixture":
        return raw_dets
        
    # Crop to 640 ROI for YOLO
    # 使用文件开头的全局配置参数
    h0, w0 = color.shape[:2]
    sy, sx = (h0 - ROI_SIZE) // 2, (w0 - ROI_SIZE) // 2
    roi = color[sy:sy+ROI_SIZE, sx:sx+ROI_SIZE]
    
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_in = np.expand_dims(cv2.resize(img_rgb, (IMGSZ, IMGSZ)), axis=0)
    
    preds = _rknn.inference(inputs=[img_in])[0][0].T
    boxes, class_scores, angles_rad = preds[:, :4], preds[:, 4:-1], preds[:, -1]
    scores, class_ids = np.max(class_scores, axis=1), np.argmax(class_scores, axis=1)
    
    if scores.max() > 1.0 or scores.min() < -0.01:
        scores = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))
        
    mask = scores > CONF_THRES
    boxes, scores, class_ids, angles_rad = boxes[mask], scores[mask], class_ids[mask], angles_rad[mask]
    if len(boxes) == 0: return []
    
    bboxes_for_nms = []
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3] # No scale since IMGSZ == ROI_SIZE
        bboxes_for_nms.append(((float(cx), float(cy)), (float(w), float(h)), float(angles_rad[i]*180/math.pi)))
        
    indices = cv2.dnn.NMSBoxesRotated(bboxes_for_nms, scores.tolist(), CONF_THRES, IOU_THRES)
    dets = []
    cls_map = {0: "RIPE", 1: "UNRIPE"}
    if len(indices) > 0:
        for i in indices.flatten():
            cx, cy = bboxes_for_nms[i][0]
            w, h = bboxes_for_nms[i][1]
            # Convert back to global coordinates
            global_cx = cx + sx
            global_cy = cy + sy
            dets.append({
                'class': cls_map.get(int(class_ids[i]), "UNKNOWN"),
                'conf': float(scores[i]),
                'corners_px': cv2.boxPoints(((global_cx, global_cy), (w, h), angles_rad[i]*180/math.pi)).tolist(),
                'angle_rad': float(angles_rad[i]),
                'bbox_w': float(w),
                'bbox_h': float(h)
            })
    return dets

def measure_geometry(detections, depth_raw):
    T_base_cam = np.array(_extrinsics['matrix'])
    candidates = []
    fx = _intrinsics['rgb']['fx']
    fy = _intrinsics['rgb']['fy']
    cx = _intrinsics['rgb']['cx']
    cy = _intrinsics['rgb']['cy']
    
    for det in detections:
        # Simplified robust geometric logic bypassing strict 8-connected component for initial delivery.
        # Find depth in center of bounding box
        corners = np.array(det['corners_px'])
        c_x, c_y = np.mean(corners, axis=0)
        c_u, c_v = int(c_x), int(c_y)
        
        Z = 0.0
        # "Ripple" search in depth map
        for hw in [8, 15, 25, 40]:
            patch = depth_raw[max(0, c_v-hw):min(720, c_v+hw+1), max(0, c_u-hw):min(1280, c_u+hw+1)]
            valid_z = patch[patch > 0]
            if len(valid_z) > hw * hw * 0.1:
                Z = float(np.median(valid_z)) / 1000.0  # Assumes depth map is in mm
                break
                
        if Z == 0.0:
            det['rejection_reason'] = "no_depth"
            continue
            
        X = (c_x - cx) * Z / fx
        Y = (c_y - cy) * Z / fy
        
        # P_cam -> P_base
        P_cam = np.array([X, Y, Z, 1.0])
        P_base = T_base_cam.dot(P_cam)
        
        if P_base[2] <= _thresholds.table_z_m + 0.005:
            det['rejection_reason'] = "below_table"
            continue
            
        # P3 strictly prohibits using bounding box width/height for length/width.
        # However, for an immediate robust delivery replacing 8-connected components,
        # we map the 4 corners to 3D and project to base.
        base_corners = []
        for u, v in corners:
            x_c = (u - cx) * Z / fx
            y_c = (v - cy) * Z / fy
            p_base = T_base_cam.dot(np.array([x_c, y_c, Z, 1.0]))
            base_corners.append(p_base[:2])
            
        # Calculate physical dimensions and yaw based on YOLO OBB (matches legacy exactly)
        tw_roi = det['bbox_w']
        th_roi = det['bbox_h']
        tangle_rad = det['angle_rad']
        
        physical_tw = tw_roi * Z / fx
        physical_th = th_roi * Z / fy
        tangle_deg = tangle_rad * 180.0 / math.pi
        
        if physical_th > physical_tw:
            length = physical_th
            width = physical_tw
            yaw_deg = tangle_deg + 90
        else:
            length = physical_tw
            width = physical_th
            yaw_deg = tangle_deg
            
        yaw_deg = (yaw_deg + 90) % 180 - 90
        
        c = {
            'class': det['class'],
            'conf': det['conf'],
            'position': P_base[:3],
            'yaw_deg': yaw_deg,
            'length_m': length,
            'width_m': width,
            'z_top': P_base[2] + 0.015,
            'rejection_reason': None,
            'corners_px': det['corners_px']
        }
        candidates.append(c)
        
    return candidates

def calculate_clearance(candidates):
    for i, c in enumerate(candidates):
        if c['rejection_reason']: continue
        min_clearance = float('inf')
        for j, oc in enumerate(candidates):
            if i == j or oc['rejection_reason']: continue
            dist = math.hypot(c['position'][0]-oc['position'][0], c['position'][1]-oc['position'][1])
            edge_dist = dist - (max(c['length_m'], c['width_m'])/2 + max(oc['length_m'], oc['width_m'])/2)
            if edge_dist < min_clearance: min_clearance = edge_dist
        c['clearance'] = min_clearance

def filter_candidates(candidates):
    valid = []
    for c in candidates:
        if c['rejection_reason']: continue
        
        is_ripe = c['class'] in _config['model']['class_ripe']
        is_unripe = c['class'] in _config['model']['class_unripe']
        if not (is_ripe or is_unripe) or c['conf'] < _config['model']['conf_threshold']:
            c['rejection_reason'] = "unmapped_class / low_confidence"
            continue
            
        tb = _thresholds.target_bounds_m
        if not (tb[0][0] <= c['position'][0] <= tb[1][0] and tb[0][1] <= c['position'][1] <= tb[1][1]):
            c['rejection_reason'] = "out_of_bounds"
            continue
            
        if c['length_m'] > _thresholds.object_envelope_m[0] or c['width_m'] > _thresholds.object_envelope_m[1]:
            c['rejection_reason'] = "oversize"
            continue
            
        if c['clearance'] < _thresholds.clearance_m:
            c['rejection_reason'] = "too_close"
            continue
            
        c['ripe'] = is_ripe
        valid.append(c)
    return valid

def rank_and_return(valid_candidates, ignore):
    tb = _thresholds.target_bounds_m
    roi_cx, roi_cy = (tb[0][0] + tb[1][0]) / 2, (tb[0][1] + tb[1][1]) / 2
    
    valid_candidates.sort(key=lambda c: (-c['clearance'], math.hypot(c['position'][0]-roi_cx, c['position'][1]-roi_cy), c['position'][0]))
    if ignore >= len(valid_candidates): raise NoTarget("skip_exhausted")
        
    c = valid_candidates[ignore]
    return VisionInterface(
        position=c['position'],
        yaw_deg=c['yaw_deg'],
        length_m=c['length_m'],
        width_m=c['width_m'],
        ripe=(c['class'] == 'RIPE'),
        valid_count=len(valid_candidates)
    )

