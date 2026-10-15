#!/usr/bin/env python3
"""视觉单次诊断入口：复用生产契约，阈值从本地 profile 注入。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from configs.app_config import load_app_config, MOCK_APP_PATH, PRODUCTION_APP_PATH
    from configs.motion_params import load_motion_params
    from main import thresholds_from_params
    from qingyun import runlog
    from qingyun.grabbing import vision
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['mock', 'real'], default='mock')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    cfg = load_app_config(args.config or (MOCK_APP_PATH if args.mode == 'mock' else PRODUCTION_APP_PATH),
                          entry='mock' if args.mode == 'mock' else 'production')
    params = load_motion_params(cfg.profile_path, mode=args.mode)
    runlog.init(cfg.log_dir)
    try:
        vision.configure(thresholds_from_params(params), cfg.vision_config_path)
        vision.init()
        try:
            target = vision.get_target()
        except vision.NoTarget as exc:
            print(f'没有可抓目标: {exc}')
            return 0
        print(f'position={target.position.tolist()} m, yaw={target.yaw_deg:.3f} deg, '
              f'length={target.length_m:.6f} m, width={target.width_m:.6f} m, '
              f'ripe={target.ripe}, valid_count={target.valid_count}')
        return 0
    except vision.VisionHardError as exc:
        print(f'视觉硬故障: {exc}', file=sys.stderr)
        return 1
    finally:
        vision._close()
        runlog.close()


if __name__ == '__main__':
    raise SystemExit(main())
