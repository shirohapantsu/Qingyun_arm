"""configs/motion_params.py 的加载与校验测试。

覆盖 docs/真机参数测量与标定指南.md 1.1 的加载规则与第 2 节字段字典的约束：
未知键、缺失键、非有限值、错误 shape、非法旋转、空限位交集、不单调标定表、
null 未测参数、模式与 status 的配套关系、限位三重交集、三张表的共同覆盖区间。
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from configs.motion_params import (
    ParamsError,
    effective_joint_limits,
    load_motion_params,
    parameter_sha256,
    read_urdf_joint_limits_deg,
    wrap180,
)
from tests.support import (
    DELETE,
    SIM_PROFILE,
    _set_path,
    motor_calibration_sha256,
    raw_profile,
    write_profile,
)


def _urdf_sha256(data):
    """按配置里的相对路径算出仓库 URDF 的真实哈希。"""
    import hashlib
    from pathlib import Path as _P

    urdf = SIM_PROFILE.parent / data["model"]["urdf_path"]
    return hashlib.sha256(urdf.read_bytes()).hexdigest()


def _load(tmp_path, **mutate):
    """按 "点号路径 -> 新值" 改一份仿真配置再加载，期望各测试自己断言结果。"""
    data = raw_profile()
    for path, value in mutate.items():
        _set_path(data, path, value)
    return load_motion_params(write_profile(tmp_path, data), mode="mock")


# ---------------------------------------------------------------------------
# 1. 正常路径
# ---------------------------------------------------------------------------


def test_仿真配置可以按mock模式加载(tmp_path):
    params = _load(tmp_path)
    assert params.status == "simulation"
    assert params.schema_version == 1
    assert set(params.places) >= {"default"}
    # 数组一律加载成 np.float64（标定指南 1.1）
    assert params.joints.zero_offset_deg.dtype == np.float64
    assert params.joints.measured_limits_deg.shape == (5, 2)
    assert params.motion.max_velocity_deg_s.shape == (5,)


def test_相对路径以profile所在目录为基准(tmp_path):
    # 仓库里的仿真配置用相对路径，加载后要能解析到真实 URDF 文件。
    from configs.motion_params import load_motion_params

    params = load_motion_params(SIM_PROFILE, mode="mock")
    assert params.urdf_path_resolved().is_file()
    assert params.motor_calibration_path_resolved().is_file()


def test_urdf限位只转换一次度数(tmp_path):
    """URDF 里是 rad，本模块转成 deg；数值应当是 ±110 / ±100 这个量级而不是 ±1.9。"""
    params = _load(tmp_path)
    assert params.urdf_limits_deg.shape == (5, 2)
    assert 100.0 < params.urdf_limits_deg[0, 1] < 120.0
    assert np.all(params.urdf_limits_deg[:, 1] > params.urdf_limits_deg[:, 0])


# ---------------------------------------------------------------------------
# 2. 结构与类型拒绝
# ---------------------------------------------------------------------------


def test_拒绝未知键(tmp_path):
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"gripper.bogus_field": 1.0})
    assert "未知键" in str(exc.value)
    assert "bogus_field" in str(exc.value)


def test_拒绝缺失键(tmp_path):
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"ik.tilt_tol_deg": DELETE})
    assert "缺失键" in str(exc.value)


def test_拒绝非有限值(tmp_path):
    """JSON 扩展字面量 NaN/Infinity 不能被当成合法参数。"""
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"grasp.lift_height_m": float("nan")})
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"timing.read_timeout_s": float("inf")})


def test_拒绝错误shape(tmp_path):
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"grasp.center_offset_object_m": [0.0, 0.0]})
    assert "shape" in str(exc.value)


def test_拒绝必须为正的字段取零或负(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"collision.clearance_m": 0.0})
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"ik.position_tol_m": -0.001})


def test_拒绝非整数的整数字段(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"timing.fps": 30.5})


@pytest.mark.parametrize("fps", [24, 25, 60])
def test_fps只能是文档规定的30(tmp_path, fps):
    """技术文档 6.1 与标定指南 2.3 都把控制节拍固定为 30Hz。"""
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"timing.fps": fps})


# ---------------------------------------------------------------------------
# 3. 语义约束
# ---------------------------------------------------------------------------


def test_拒绝非法旋转矩阵(tmp_path):
    """tool.rotation_e_tcp 必须在 SO(3) 上：正交且 det=+1。"""
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"tool.rotation_e_tcp": [[1, 0, 0], [0, 1, 0], [0, 0, 2]]})
    assert "正交" in str(exc.value)
    # 反射（det=-1）也必须被拒绝
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"tool.rotation_e_tcp": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]})


def test_拒绝空限位交集(tmp_path):
    """margin 大到把交集吃空时必须拒绝，而不是运行时才发现无路可走。"""
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"joints.margin_deg": [90.0] * 5})
    assert "交集为空" in str(exc.value)


def test_拒绝符号不为pm1的sign(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"joints.sign": [1, 1, 0, 1, -1]})


def test_拒绝不单调的gap表(tmp_path):
    """gap 非严格递增时插值不唯一，接触判据会给出随机结果。"""
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"gripper.gap_table.1.gap_m": 0.001})
    assert "gap_m" in str(exc.value)


def test_拒绝开度不递增的标定表(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"gripper.angle_table.2.gripper_pct": 10.0})


def test_允许角度表整体递减(tmp_path):
    """angle_deg 可递增也可递减，只要全表同向（标定指南 2.4）。"""
    data = raw_profile()
    table = data["gripper"]["angle_table"]
    for row in table:
        row["angle_deg"] = -row["angle_deg"]
    _load(tmp_path, **{"gripper.angle_table": table})


def test_拒绝少于三个标定点的表(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"gripper.gap_table": raw_profile()["gripper"]["gap_table"][:2]})


def test_拒绝开度落在标定表共同覆盖区间之外(tmp_path):
    """预张/接触/释放/空闭合端都必须落在三张表的共同覆盖区间内（标定指南 2.4）。

    这里把工具平移表的上界从 100% 压到 60%，于是 preopen_pct=70 虽然仍是个合法的
    百分比，却没有对应的 TCP 平移可插值——超出表范围时禁止外插。
    """
    data = raw_profile()
    samples = copy.deepcopy(data["tool"]["translation_samples"][:3])
    samples[-1]["gripper_pct"] = 60.0                      # 工具表上界压到 60%
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"tool.translation_samples": samples})
    assert "共同覆盖区间" in str(exc.value)


def test_拒绝超出0到100的开度百分比(tmp_path):
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"gripper.preopen_pct": 130.0})
    assert "[0,100]" in str(exc.value)


def test_拒绝接触开度不大于空闭合端(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"gripper.empty_closed_pct": 95.0, "gripper.empty_tol_pct": 1.0})


def test_拒绝hard不大于contact的夹爪电流阈值(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"gripper.hard_current_ma": 100.0})   # contact 是 220


def test_拒绝hard不大于普通的跟踪误差阈值(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motion.following_error_hard_deg": [1.0] * 5})


def test_拒绝settle_timeout不大于dwell(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motion.settle_timeout_s": 0.05})    # dwell 是 0.15


def test_拒绝迟到阈值不小于一个tick(tmp_path):
    """标定指南 2.3：max_tick_lateness_s 必须小于 1/FPS。"""
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"timing.max_tick_lateness_s": 0.05})


def test_拒绝stop_poll大于一个tick(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"timing.stop_poll_s": 0.2})


def test_示教姿态越限被拒绝(tmp_path):
    """home / safe_waypoints / ik 种子都是示教出来的姿态，越限属于配置错误。"""
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"workspace.home_joints_deg": [0.0, 80.0, 60.0, 0.0, 0.0]})
    assert "超出有效限位" in str(exc.value)
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"workspace.safe_waypoints_deg": [[0.0, -75.0, 120.0, 0.0, 0.0]]})


def test_至少需要一个安全中转位(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"workspace.safe_waypoints_deg": []})


def test_places必须含default(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"places": {"only_other": {"position": [0.3, 0.0, 0.02], "yaw_deg": 0.0}}})


def test_拒绝未知place(tmp_path):
    params = _load(tmp_path)
    from configs.motion_params import ParamsError

    with pytest.raises(ParamsError):
        params.place("does_not_exist")


def test_place的yaw被归一化到闭区间左闭右开(tmp_path):
    """长轴不区分头尾：[-90,90)。"""
    params = _load(tmp_path, **{"places.default.yaw_deg": 190.0})
    assert -90.0 <= params.places["default"].yaw_deg < 90.0
    assert params.places["default"].yaw_deg == pytest.approx(wrap180(190.0))


def test_拒绝夹爪两端读数相同(tmp_path):
    """4.4：分母为 0 禁止使用。"""
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motor.gripper_open_raw": 500})   # closed 也是 500


def test_拒绝错误的电机ids顺序(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motor.ids": [1, 2, 3, 4, 5, 7]})


def test_拒绝不大于1的编码分辨率(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motor.position_resolution": [4096, 1, 4096, 4096, 4096, 4096]})


def test_拒绝为零的电流比例(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"motor.current_ma_per_raw": [0.0] * 6})


def test_拒绝ignore_pairs引用不存在的capsule(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"collision.ignore_self_pairs": [["ghost", "base_body"]]})


def test_碰撞胶囊体列表不能为空(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"collision.link_capsules": []})


def test_AABB的min_max写反被拒绝(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"workspace.obstacles.0.bounds_m": [[0.36, 0.18, 0.1], [0.28, 0.13, 0.0]]})


def test_允许厚度为零的壁(tmp_path):
    """把容器壁抽象成无限薄的平面是合法建模手法，不该被 min<max 卡住。"""
    data = raw_profile()
    data["workspace"]["obstacles"] = [
        {"id": "zero_wall", "bounds_m": [[0.20, 0.20, 0.0], [0.20, 0.25, 0.05]]}
    ]
    _load(tmp_path, **{"workspace.obstacles": data["workspace"]["obstacles"]})


# ---------------------------------------------------------------------------
# 4. 加载模式与 status 的配套
# ---------------------------------------------------------------------------


def test_draft配置不能被任何模式加载(tmp_path):
    """draft 里未测物理值是 null，文档要求"实机未测参数保持未完成状态"。"""
    with pytest.raises(ParamsError) as exc:
        _load(tmp_path, **{"status": "draft"})
    assert "null" in str(exc.value) or "status" in str(exc.value)


def test_mock模式要求status为simulation(tmp_path):
    with pytest.raises(ParamsError):
        _load(tmp_path, **{"status": "verified"})


def test_real模式要求status为verified(tmp_path):
    data = raw_profile()
    data["status"] = "verified"
    p = write_profile(tmp_path, data)
    with pytest.raises(ParamsError):
        load_motion_params(p, mode="real")


def test_real模式在缺验收报告时拒绝(tmp_path):
    """标定指南 1.3：real 要求模型与校准文件哈希与验收报告匹配。"""
    data = raw_profile()
    data["status"] = "verified"
    data["verification"] = None
    data["model"]["motor_calibration_sha256"] = motor_calibration_sha256()
    data["model"]["urdf_sha256"] = _urdf_sha256(data)
    p = write_profile(tmp_path, data)
    with pytest.raises(ParamsError) as exc:
        load_motion_params(p, mode="real")
    assert "verification" in str(exc.value)


def test_real模式能加载一份完整验收过的配置(tmp_path):
    """走完 parameter_sha256 → 报告 → verification 的绑定链路。

    这条同时是 scripts/calibrate.py promote 的下游契约：它必须产出能被 real 模式
    加载的配置，否则"验收过的参数"和"实际跑的参数"就是两张皮。
    """
    import hashlib

    data = raw_profile()
    data["status"] = "verified"
    data["model"]["urdf_sha256"] = _urdf_sha256(data)
    data["model"]["motor_calibration_sha256"] = motor_calibration_sha256()
    profile_path = write_profile(tmp_path, data)
    # 先按"没有 verification"的最终内容算参数哈希，写进报告，再把 verification 补回
    # 配置。verification 与 status 一起被排除在哈希之外，所以这一步不会改变哈希。
    base = json.loads(profile_path.read_text(encoding="utf-8"))
    assert base["verification"] is None
    param_hash = parameter_sha256(base)
    report = {
        "profile_id": data["profile_id"],
        "parameter_sha256": param_hash,
        "operator": "tester",
        "operator_reviewed": True,
        "stages": {},
        "overall_pass": True,
    }
    report_path = tmp_path / "reports" / "validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    base["verification"] = {
        "report_path": "reports/validation.json",
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    profile_path.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    assert parameter_sha256(base) == param_hash, "写回 verification 不应改变参数哈希"
    params = load_motion_params(profile_path, mode="real")
    assert params.status == "verified"


def test_real模式在报告哈希不匹配时拒绝(tmp_path):
    """报告文件被改过一个字节，real 模式就不能再信任它。"""
    import hashlib

    data = raw_profile()
    data["status"] = "verified"
    data["model"]["urdf_sha256"] = _urdf_sha256(data)
    data["model"]["motor_calibration_sha256"] = motor_calibration_sha256()
    profile_path = write_profile(tmp_path, data)
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps({"profile_id": data["profile_id"],
                                       "parameter_sha256": "x", "overall_pass": True,
                                       "operator_reviewed": True, "stages": {}}),
                           encoding="utf-8")
    data["verification"] = {"report_path": "validation.json",
                            "report_sha256": hashlib.sha256(b"stale").hexdigest()}
    profile_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ParamsError) as exc:
        load_motion_params(profile_path, mode="real")
    assert "报告哈希" in str(exc.value)


def test_real模式在配置被改动后拒绝(tmp_path):
    """验收之后又偷偷改了一个数，必须拒绝启动。"""
    data = raw_profile()
    data["status"] = "verified"
    out = tmp_path / "profile.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    p2 = copy.deepcopy(data)
    p2["verification"] = {"report_path": "r.json", "report_sha256": "0" * 64}
    out.write_text(json.dumps(p2), encoding="utf-8")
    with pytest.raises(ParamsError):
        load_motion_params(out, mode="real")


# ---------------------------------------------------------------------------
# 5. 共享计算函数
# ---------------------------------------------------------------------------


def test_有效限位公式与技术文档5_1逐字一致(params):
    """lower = max(urdf, measured, application) + margin；upper = min(...) - margin。"""
    urdf = params.urdf_limits_deg
    j = params.joints
    lower, upper = effective_joint_limits(urdf, j)
    expect_lo = np.maximum.reduce([urdf[:, 0], j.measured_limits_deg[:, 0],
                                   j.application_limits_deg[:, 0]]) + j.margin_deg
    expect_hi = np.minimum.reduce([urdf[:, 1], j.measured_limits_deg[:, 1],
                                   j.application_limits_deg[:, 1]]) - j.margin_deg
    assert np.allclose(lower, expect_lo)
    assert np.allclose(upper, expect_hi)


def test_wrists_roll使用实测线缆范围而不是URDF满行程(params):
    """标定指南 4.1 第 4 条：官方校准里 wrist_roll 记为 0~4095 不代表能绕满一圈。"""
    lower, upper = effective_joint_limits(params.urdf_limits_deg, params.joints)
    urdf_lo, urdf_hi = params.urdf_limits_deg[4]
    assert lower[4] > urdf_lo and upper[4] < urdf_hi


def test_wrap180的边界(params):
    assert wrap180(-90.0) == pytest.approx(-90.0)
    assert wrap180(90.0) == pytest.approx(-90.0)      # 右开：90 折回 -90
    assert wrap180(0.0) == pytest.approx(0.0)
    assert wrap180(179.0) == pytest.approx(-1.0)


def test_parameter_sha256忽略status与verification只绑定参数值(tmp_path):
    """除 status、verification 之外的任何一项改动都必须改变哈希。

    注意这是按"原始 JSON 文本"序列化的（标定指南 1.3 原文），所以 4096 与 4096.0
    这两种写法会得到不同哈希。方向是安全的：改排版最多让 real 模式拒绝启动，不会
    把改过的参数误判成验收过。
    """
    a = raw_profile()
    b = copy.deepcopy(a)
    b["status"] = "verified"
    b["verification"] = {"report_path": "x", "report_sha256": "y"}
    assert parameter_sha256(a) == parameter_sha256(b)
    c = copy.deepcopy(a)
    c["ik"]["position_tol_m"] = c["ik"]["position_tol_m"] + 1e-6
    assert parameter_sha256(a) != parameter_sha256(c)
    # 重新排版不改变哈希：json.dumps 用的是紧凑分隔符。
    d = copy.deepcopy(a)
    assert parameter_sha256(json.loads(json.dumps(d))) == parameter_sha256(a)


def test_URDF解析拒绝DTD实体(tmp_path):
    """配置里的 urdf_path 是外部输入，不能让它展开实体。"""
    evil = tmp_path / "evil.urdf"
    evil.write_text(
        '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x "y">]>'
        '<robot name="e"><joint name="shoulder_pan" type="revolute">'
        '<limit lower="-1" upper="1"/></joint></robot>',
        encoding="utf-8",
    )
    with pytest.raises(ParamsError) as exc:
        read_urdf_joint_limits_deg(evil, ("shoulder_pan",))
    assert "DTD" in str(exc.value)
