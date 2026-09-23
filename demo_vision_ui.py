import cv2
import sys
import time
import math
import numpy as np
from pathlib import Path

from qingyun.grabbing.vision import (
    configure, init, fetch_frames, infer_detections, 
    measure_geometry, calculate_clearance, filter_candidates, rank_and_return,
    VisionThresholds, NoTarget
)

def main():
    print("====== 🍓 YOLO 3D 视觉管线实时可视化 ======")
    print("按 'q' 或 'ESC' 退出")
    
    thresholds = VisionThresholds(
        target_bounds_m=[[-2.0, -2.0, -0.5], [2.0, 2.0, 1.0]],
        object_envelope_m=[0.2, 0.2, 0.2],
        clearance_m=0.001,
        approach_height_m=0.1,
        table_z_m=0.0,
        table_flatness_m=0.02
    )
    configure(thresholds, Path("configs/vision.json"))
    init()
    
    fps_count = 0
    fps_t0 = time.time()
    fps_display = 0.0

    # Retrieve internal parameters from vision module
    import qingyun.grabbing.vision as vis
    intrin = vis._intrinsics
    fx, fy = intrin['rgb']['fx'], intrin['rgb']['fy']
    cx, cy = intrin['rgb']['cx'], intrin['rgb']['cy']
    
    while True:
        try:
            c, d, r = fetch_frames()
        except Exception as e:
            print("相机抓取失败:", e)
            break
            
        display_img = c.copy()
        
        # 截取中心画幅供显示用 (与网络输入相同视野)
        ROI_SIZE = 640
        h0, w0 = display_img.shape[:2]
        sy, sx = (h0 - ROI_SIZE) // 2, (w0 - ROI_SIZE) // 2
        roi_img = display_img[sy:sy+ROI_SIZE, sx:sx+ROI_SIZE].copy()
        
        dets = infer_detections(c, r)
        cands = measure_geometry(dets, d)
        
        # 计算净距和过滤
        calculate_clearance(cands)
        valid_candidates = filter_candidates(cands)
        
        # 找出最优的那个
        best_target = None
        try:
            best_target = rank_and_return(valid_candidates, ignore=0)
        except NoTarget:
            pass

        # 在图像上绘制所有的候选目标
        for cand in cands:
            # 还原到 roi 坐标系 (方便画图)
            corners_img = cand.get('corners_px', [])
            if not corners_img: continue
            
            # 画框
            pts = np.array(corners_img, np.int32)
            pts[:, 0] -= sx
            pts[:, 1] -= sy
            pts = pts.reshape((-1, 1, 2))
            
            # 如果是被过滤掉的，画灰色；如果是合法备选画黄色；如果被选中画绿色
            # Determine maturity color matching legacy code
            cls_name = cand['class']
            is_best = best_target and np.allclose(cand['position'], best_target.position)
            
            if cand['rejection_reason']:
                color = (128, 128, 128)
                txt = f"Rej: {cand['rejection_reason']}"
                thick = 1
            else:
                color = (0, 0, 255) if cls_name == "RIPE" else (0, 255, 0) # Red for RIPE, Green for UNRIPE
                txt = f"{cls_name} Z:{cand['position'][2]:.2f} Y:{cand['yaw_deg']:.0f}"
                if is_best: txt = "[BEST] " + txt
                thick = 3 if is_best else 2
                
            # If it's best, draw a bright white highlight box underneath
            if is_best:
                cv2.polylines(roi_img, [pts], True, (255, 255, 255), 5)
                
            cv2.polylines(roi_img, [pts], True, color, thick)
            
            # 写字
            top_pt = pts[pts[:, 0, 1].argmin()][0]
            cv2.putText(roi_img, txt, (top_pt[0], max(15, top_pt[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # 画一个指示偏航角的轴
            center_x = int(np.mean(pts[:, 0, 0]))
            center_y = int(np.mean(pts[:, 0, 1]))
            yaw_rad = math.radians(cand['yaw_deg'])
            line_len = 30
            end_x = int(center_x + line_len * math.cos(yaw_rad))
            end_y = int(center_y + line_len * math.sin(yaw_rad)) # Image Y is down
            cv2.line(roi_img, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)
            cv2.circle(roi_img, (center_x, center_y), 4, (0, 0, 255), -1)

        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_t0 = time.time()
        cv2.putText(roi_img, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite('demo_output.jpg', roi_img)
        print(f'\r[UI] {fps_count} 画面已保存 (检测到 {len(cands)} 个目标)    ', end='', flush=True)
        #项目根目录的 demo_output.jpg，请在左侧目录点击查看！(按 Ctrl+C 退出)', end='', flush=True)
        time.sleep(0.1)
        key = 255
        if key == ord('q') or key == 27:
            break
            
    print('\n退出可视化')
    import qingyun.grabbing.vision as vis
    if hasattr(vis, '_cap') and vis._cap: vis._cap.release()
    try: vis.openni2.unload()
    except: pass

if __name__ == "__main__":
    main()
