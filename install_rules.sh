#!/bin/bash
set -e
echo "[1/2] 正在安装奥比中光 USB 权限规则到 /etc/udev/rules.d/..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo cp "${SCRIPT_DIR}/orbbec-usb.rules" /etc/udev/rules.d/558-orbbec-usb.rules
echo "[2/2] 正在重新加载 udev 规则..."
sudo udevadm control --reload-rules && sudo udevadm trigger
echo "✅ 奥比中光相机 USB 权限配置成功！非 root 用户现已可直接读取深度相机。"
