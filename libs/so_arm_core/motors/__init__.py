"""舵机总线支持包的 vendor 入口。

只收录本项目 Python 侧需要的、无第三方依赖的两块内容：
    encoding_utils  符号量/补码等原始寄存器编码换算
    feetech.tables  STS/SCS 系列控制表（寄存器地址、分辨率、型号号）

C++ 的 scservo_sdk 端口层不在这里 vendor；本项目用 pyserial 直接
按 STS/SCS 协议组包，见 qingyun/grabbing/motor_control.py。
"""
