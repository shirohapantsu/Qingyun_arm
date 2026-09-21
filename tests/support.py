"""测试公共夹具：加载仿真配置、按字段改写配置副本。

单独放一个模块，是为了让"改一个字段看加载器是否拒绝"这类负向测试在各测试文件里
复用同一份做法：先在内存里改 JSON，落到临时目录再加载，绝不改动仓库里的真配置。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from configs.common_interface import VisionInterface
from configs.motion_params import MotionParams, load_motion_params

# 项目根：本文件在 tests/ 下，父目录的父目录就是项目根。
ROOT = Path(__file__).resolve().parents[1]
SIM_PROFILE = ROOT / "calibration/SIM_ROBOT/simulation/profile.json"


def raw_profile() -> dict[str, Any]:
    """读出仿真配置的原始 dict（未校验），供负向测试改字段。"""
    return json.loads(SIM_PROFILE.read_text(encoding="utf-8"))


def _set_path(obj: Any, path: str, value: Any) -> None:
    """按 "a.b.0.c" 这样的路径写入/删除（value=DELETE 时删除该键）。"""
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else obj[p]
    last = parts[-1]
    if value is DELETE:
        del obj[last]
    else:
        obj[int(last)] if last.isdigit() else obj.__setitem__(last, value)


DELETE = object()


def write_profile(tmp_path: Path, data: dict[str, Any]) -> Path:
    """把改过的配置写进临时目录，并让它引用的两个资源文件也能解析到。

    profile.json 里的相对路径以它自己所在目录为基准，所以写到临时目录后 URDF 与
    电机校准文件都会找不到。URDF 改成绝对路径即可；电机校准文件必须留在配置同目录，
    因为 real 模式的测试要验证"配置 + 校准 + 报告"三者的哈希绑定链路。
    """
    data = copy.deepcopy(data)
    src = SIM_PROFILE.parent
    data["model"]["urdf_path"] = str(src / data["model"]["urdf_path"])
    calib_rel = data["model"]["motor_calibration_path"]
    (tmp_path / Path(calib_rel).name).write_bytes((src / calib_rel).read_bytes())
    data["model"]["motor_calibration_path"] = Path(calib_rel).name
    out = tmp_path / "profile.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def motor_calibration_sha256() -> str:
    """仿真包里电机校准文件的真实哈希，供 real 模式测试填字段。"""
    import hashlib

    path = SIM_PROFILE.parent / raw_profile()["model"]["motor_calibration_path"]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sim() -> MotionParams:
    """加载仓库里的仿真配置（mock 模式）。会话级缓存，测试不许改它的字段。"""
    return load_motion_params(SIM_PROFILE, mode="mock")


def make_target(position=(0.34, 0.04, 0.022), yaw_deg=0.0, grade="A") -> VisionInterface:
    """构造一个合法的视觉目标。position 必须是 np.float64 数组（文档 3.2）。"""
    return VisionInterface(
        position=np.asarray(position, dtype=np.float64), yaw_deg=float(yaw_deg), grade=grade
    )


def graspable_target_at(x: float, y: float, z: float = 0.022,
                        grade: str = "A") -> VisionInterface:
    """在 (x,y,z) 放一个"这台 5 自由度臂真的能顶抓"的目标。

    SO-ARM101 只有 5 个姿态关节，而顶抓位姿给出 3+2+1=6 个标量条件（姿态文档第 5
    节），因此抓取方向与目标方位角之间存在结构耦合。仿真实测表明只有
    yaw ≈ atan2(y,x) 的一族目标落在可达流形上；标定指南 7.4 也正是要求用扫描结果
    来确定 target_bounds_m、并由视觉只下发满足该条件的目标。
    """
    import math

    return make_target((x, y, z), math.degrees(math.atan2(y, x)), grade)
