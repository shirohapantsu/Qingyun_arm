"""青云智臂——第三方机械臂核心库的 vendor 目录。

本包只存放从外部项目原样复制进来、并在本项目内锁定的源文件，
不放本项目的业务逻辑。所有文件的来源、SHA256 与验证过的依赖版本
记录在同目录 SOURCE.md 中。

模块说明：
    kinematics        LeRobot 0.5.1 的 RobotKinematics（placo FK/IK 封装）
    motors            STS/SCS 系列舵机的寄存器表与原始量编码工具

目录内还包含 so101_new_calib.urdf 与其相对引用的 assets/ 网格，
placo 在解析 URDF 时会强制要求这些网格存在，因此必须一并 vendor。
"""
