#!/usr/bin/env python3
"""绘制原生视觉扫描结果；默认 mock，不发送任何运动指令。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['mock','real'],default='mock')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--frames',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'logs/vision_preview.jpg')
    parser.add_argument('--display',action='store_true')
    args=parser.parse_args()
    if args.frames<=0: parser.error('--frames 必须大于零')
    import cv2
    import numpy as np
    from configs.app_config import load_app_config, MOCK_APP_PATH, PRODUCTION_APP_PATH
    from configs.motion_params import load_motion_params
    from main import thresholds_from_params
    from qingyun import runlog
    from qingyun.grabbing import vision
    cfg=load_app_config(args.config or (MOCK_APP_PATH if args.mode=='mock' else PRODUCTION_APP_PATH),entry='mock' if args.mode=='mock' else 'production')
    params=load_motion_params(cfg.profile_path,mode=args.mode)
    runlog.init(cfg.log_dir)
    scans=[]
    # 文件日志照常保留，绘图监听只捕获 vision_scan。
    original=runlog.event
    def event(kind,**fields):
        original(kind,**fields)
        if kind=='vision_scan': scans.append(fields)
    runlog.event=event
    try:
        vision.configure(thresholds_from_params(params),cfg.vision_config_path); vision.init()
        for _ in range(args.frames):
            try: vision.get_target()
            except vision.NoTarget as exc:
                if str(exc)=='replay_exhausted': break
            image=vision._pipeline.last_color.copy()
            scan=scans[-1]
            for candidate in scan['candidates']:
                if 'corners_px' not in candidate: continue
                pts=np.rint(candidate['corners_px']).astype(np.int32)
                reason=candidate['rejection_reason']
                selected=candidate['source_detection_id']==scan.get('selected_detection_id')
                color=(128,128,128) if reason else (0,255,0) if selected else (0,255,255)
                cv2.polylines(image,[pts],True,color,1)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(args.output),image): raise RuntimeError('图像保存失败')
            print(f"frame={scan['frame_id']}, outcome={scan['outcome']}, valid={scan['valid_count']}, output={args.output}")
            if args.display:
                cv2.imshow('Vision pipeline',image)
                if cv2.waitKey(1)&0xff in (27,ord('q')): break
    finally:
        runlog.event=original
        vision._close(); runlog.close()
        if args.display: cv2.destroyAllWindows()


if __name__=='__main__': main()
