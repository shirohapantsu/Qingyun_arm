import sys
from pathlib import Path
from qingyun.grabbing import vision
from qingyun.grabbing.vision import VisionThresholds

def main():
    print("====== 视觉模块真机集成联调 ======")
    
    # 模拟运动学组下发的阈值配置
    thresholds = VisionThresholds(
        target_bounds_m=[[-2.0, -2.0, -0.5], [2.0, 2.0, 1.0]], # (2,3)
        object_envelope_m=[0.2, 0.2, 0.2],
        clearance_m=0.001,
        approach_height_m=0.1,
        table_z_m=0.0,
        table_flatness_m=0.02
    )
    
    print("[1] 正在加载配置文件 configs/vision.json ...")
    config_path = Path("configs/vision.json")
    vision.configure(thresholds, config_path)
    
    print("[2] 正在初始化相机、载入真实的 YOLO 模型和相机标定参数...")
    vision.init()
    
    print("[3] 正在请求抓取一帧真实画面 (get_target)...")
    try:
        target = vision.get_target(ignore=0)
        print("\n✅ 成功获取目标！")
        print("-" * 40)
        print(f"坐标 (Position) : X={target.position[0]:.3f}m, Y={target.position[1]:.3f}m, Z={target.position[2]:.3f}m")
        print(f"偏航角 (Yaw)    : {target.yaw_deg:.1f}度")
        print(f"品级 (Grade)    : {target.ripe}")
        print("-" * 40)
        print("真机测试完美通过！")
    except vision.NoTarget as e:
        print(f"\n⚠️ 视野内无合适目标 (原因: {e})")
    except Exception as e:
        print(f"\n❌ 发生严重错误: {e}")

if __name__ == "__main__":
    main()
