"""
纯 RKNN(NPU) 版 YOLO 3D 雷达
把 best.rknn.bak 改回 best.rknn 后直接运行
"""
import cv2
import numpy as np
from openni import openni2
from rknnlite.api import RKNNLite
import json
import os

# ==========================================
# 🍓 用户常用配置区 (方便手动修改)
# ==========================================
TARGET_MODEL = "11l_berry_best-rk3588.rknn"  # 当前使用的模型文件名
IMGSZ = 736                                  # 模型输入分辨率 (如 736, 640, 416)
CONF_THRES = 0.25                            # 置信度阈值 (越小越容易识别，越大越严格)
IOU_THRES = 0.45                             # 重叠框过滤阈值
# --- 相机物理分辨率配置 (重标定后若改动请修改此处) ---
CAM_RGB_WIDTH, CAM_RGB_HEIGHT = 1280, 720
CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT = 640, 480 # ⚠️ 硬件限制
# --- 推理裁剪画幅 ---
ROI_SIZE = 640
# ==========================================

# ---------- 深度对齐引擎 (与主版本完全一致) ----------
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
        
        # 补全点云投影到高清大图产生的稀疏黑洞 (形态学膨胀弥合，取局部非零深度)
        kernel = np.ones((3, 3), np.uint8)
        aligned_depth = cv2.dilate(aligned_depth, kernel, iterations=1)
        aligned_depth = cv2.medianBlur(aligned_depth, 5)
        
        return frame_rgb, aligned_depth
        return frame_rgb, aligned_depth

def get_openni_path():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for p in [os.path.join(PROJECT_ROOT, "libs", "OpenNI2_arm64"), "/usr/lib", "/usr/local/lib"]:
        if os.path.exists(os.path.join(p, "libOpenNI2.so")): return p
    return None

# ---------- RKNN NPU 推理 ----------
class YOLO_NPU:
    def __init__(self, model_path, conf=CONF_THRES, iou=IOU_THRES):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()
        self.conf = conf
        self.iou = iou
        print(f"  ✅ NPU 加载成功，输入尺寸: {IMGSZ}x{IMGSZ}")

    def detect(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        # 诊断确认的正确格式: NHWC, uint8, 不归一化
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in = np.expand_dims(cv2.resize(img_rgb, (IMGSZ, IMGSZ)), axis=0)

        outputs = self.rknn.inference(inputs=[img_in])
        preds = outputs[0][0].T  # (2100, 4 + classes)

        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # 自动补 sigmoid（如果模型输出的是 logits）
        if scores.max() > 1.0 or scores.min() < -0.01:
            scores = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))

        mask = scores > self.conf
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if len(boxes) == 0:
            return []

        # cx,cy,w,h → x1,y1,w,h 并还原到原始画面尺寸
        sx, sy = w0 / IMGSZ, h0 / IMGSZ
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        w  = boxes[:, 2] * sx
        h  = boxes[:, 3] * sy

        indices = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, w, h], 1).tolist(),
            scores.tolist(), self.conf, self.iou)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx1, by1 = max(0, int(x1[i])), max(0, int(y1[i]))
                bx2, by2 = min(w0, int(x1[i]+w[i])), min(h0, int(y1[i]+h[i]))
                results.append((bx1, by1, bx2, by2, float(scores[i]), int(class_ids[i])))
        return results

# ---------- 主程序 ----------
def main():
    print("🚀 [纯NPU版] 正在启动...")
    script_dir = os.path.dirname(__file__)

    model_path = os.path.join(script_dir, TARGET_MODEL)
    if not os.path.exists(model_path):
        print(f"❌ 找不到 {model_path}，请确保模型文件在 {script_dir} 目录下！")
        return
    detector = YOLO_NPU(model_path)

    # 标定参数
    with open(os.path.join(script_dir, "../data/camera_params/astra_pro_calib.json")) as f:
        calib = json.load(f)
    params = {k: np.array(v) for section in ['ir_depth_camera', 'rgb_camera']
              for k, v in [
                  (f'mtx_{"ir" if "ir" in section else "rgb"}',
                   calib['stereo_calibration'][section]['camera_matrix']),
                  (f'dist_{"ir" if "ir" in section else "rgb"}',
                   calib['stereo_calibration'][section]['distortion_coeffs'])]}
    params['R'] = np.array(calib['stereo_calibration']['rotation_matrix'])
    params['T'] = np.array(calib['stereo_calibration']['translation_vector'])

    aligner = DepthAligner(params)
    fx, fy = params['mtx_rgb'][0, 0], params['mtx_rgb'][1, 1]
    cx, cy = params['mtx_rgb'][0, 2], params['mtx_rgb'][1, 2]

    # 摄像头
    path = get_openni_path()
    openni2.initialize(path) if path else openni2.initialize()
    dev = openni2.Device.open_any()
    ds = dev.create_depth_stream()
    vm = openni2.VideoMode()
    vm.resolutionX, vm.resolutionY = CAM_DEPTH_WIDTH, CAM_DEPTH_HEIGHT
    vm.fps = 30
    vm.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
    ds.set_video_mode(vm)
    try: ds.set_mirroring_enabled(False)
    except: pass
    ds.start()

    cap = None
    for i in range(6):
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RGB_WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RGB_HEIGHT)
            c.set(cv2.CAP_PROP_FPS, 30)
            ret, _ = c.read()
            if ret: cap = c; break
            c.release()

    cv2.namedWindow('NPU 3D Tracking', cv2.WINDOW_AUTOSIZE)
    print("\n✅ [纯NPU] 系统就绪！按 Q 退出。")

    first = True
    import time
    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0
    while True:
        try:
            openni2.wait_for_any_stream([ds], timeout=500)
            depth_raw = np.frombuffer(
                ds.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        except: continue
        ret, frame = cap.read()
        if not ret: continue

        _, aligned = aligner.align(frame, depth_raw)
        start_x = (CAM_RGB_WIDTH - ROI_SIZE) // 2
        start_y = (CAM_RGB_HEIGHT - ROI_SIZE) // 2
        frame_roi = frame[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE]
        aligned_roi = aligned[start_y:start_y+ROI_SIZE, start_x:start_x+ROI_SIZE]
        dets = detector.detect(frame_roi)

        if first and len(dets) > 0:
            print(f"  📊 首次检测到 {len(dets)} 个目标")
            first = False

        for (x1, y1, x2, y2, conf, cls_id) in dets:
            u_roi, v_roi = (x1+x2)//2, (y1+y2)//2
            Z = 0
            for hw in [8, 15, 25, 40]:  # 水波纹式扩大搜索区域
                roi = aligned_roi[max(0,v_roi-hw):min(ROI_SIZE,v_roi+hw+1), max(0,u_roi-hw):min(ROI_SIZE,u_roi+hw+1)]
                vd = roi[roi > 0]
                if len(vd) > (hw * hw * 0.1):
                    Z = float(np.median(vd))
                    break
            
            g_x1, g_y1 = x1 + start_x, y1 + start_y
            g_x2, g_y2 = x2 + start_x, y2 + start_y
            g_u, g_v = u_roi + start_x, v_roi + start_y

            # --- 类别到颜色和名称的映射 ---
            CLASS_NAMES = {0: "RIPE", 1: "UNRIPE"}
            COLORS = {0: (0, 0, 255), 1: (0, 255, 0)} # 0:红, 1:绿
            
            cls_name = CLASS_NAMES.get(cls_id, f"C{cls_id}")
            box_color = COLORS.get(cls_id, (0, 255, 255))

            # 画极细的框和中心点
            cv2.rectangle(frame, (g_x1, g_y1), (g_x2, g_y2), box_color, 2)
            cv2.circle(frame, (g_u, g_v), 4, (0,0,255), -1)
            
            # 极简文本展示
            if Z > 0:
                X = (g_u - cx) * Z / fx
                Y = (g_v - cy) * Z / fy
                txt = f"{cls_name} Z:{Z:.0f} ({conf:.0%})"
            else:
                txt = f"{cls_name} NO-Z ({conf:.0%})"
                
            # 直接把字写在框上
            cv2.putText(frame, txt, (g_x1, max(15, g_y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # FPS 计算与显示
        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_t0 = time.time()
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('NPU 3D Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q'), ord('Q')]: 
            break
        elif key in [ord('p'), ord('P'), ord(' ')]: 
            print("\n" + "="*40)
            print("[系统] 触发手动抓拍，正在提取最优目标...")
            if len(dets) == 0:
                print("[系统] 警告：当前视野中未识别到任何目标")
            else:
                best_target = sorted(dets, key=lambda x: x[4], reverse=True)[0]
                tx1, ty1, tx2, ty2, tconf, tcls_id = best_target
                tu_roi, tv_roi = (tx1+tx2)//2, (ty1+ty2)//2
                
                tZ = 0
                for thw in [8, 15, 25, 40]:
                    troi = aligned_roi[max(0,tv_roi-thw):min(ROI_SIZE,tv_roi+thw+1), max(0,tu_roi-thw):min(ROI_SIZE,tu_roi+thw+1)]
                    tvd = troi[troi > 0]
                    if len(tvd) > (thw * thw * 0.1):
                        tZ = float(np.median(tvd))
                        break
                
                tgrade = "RIPE" if tcls_id == 0 else "UNRIPE"
                
                tg_x1, tg_y1 = tx1 + start_x, ty1 + start_y
                tg_x2, tg_y2 = tx2 + start_x, ty2 + start_y
                tg_u, tg_v = tu_roi + start_x, tv_roi + start_y
                
                if tZ == 0:
                    print(f"[系统] 警告：最优目标 [{tgrade}] 无有效深度信息")
                else:
                    tX = (tg_u - cx) * tZ / fx
                    tY = (tg_v - cy) * tZ / fy
                    print("[系统] 成功生成 VisionInterface 模拟数据:")
                    print(f"  ├─ position : [{tX:.1f}, {tY:.1f}, {tZ:.1f}] mm")
                    print(f"  ├─ yaw_deg  : 0.0")
                    print(f"  └─ grade    : {tgrade}")
                    
                    snap = frame.copy()
                    cv2.rectangle(snap, (tg_x1, tg_y1), (tg_x2, tg_y2), (0, 0, 255), 2)
                    cv2.circle(snap, (tg_u, tg_v), 4, (0, 0, 255), -1)
                    cv2.putText(snap, "TARGET", (tg_x1, max(15, tg_y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imwrite("manual_snapshot.jpg", snap)
                    print("[系统] 快照已保存为: manual_snapshot.jpg")
            print("="*40 + "\n")

    try: ds.stop(); dev.close();
    except: pass
    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

