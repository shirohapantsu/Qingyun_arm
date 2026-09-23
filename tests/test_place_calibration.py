"""P4 §2.5 落点标定工具 `scripts/calibrate_place_poses.py` 的离线合成测试（P4-T03）。

定向测试（用户 2026-09-23 指示：P4 阶段只做定向测试，不跑全量）。覆盖 P4-T03「TCP→物体
中心换算与运动文档 §3.4 逆变换一致」及 §2.5 判据：

* 换算双向闭合：合成顶抓 TCP 释放位姿 → 换算物体中心 → 用正向 ``place_tcp_pose`` 复算回
  TCP 一致（数值容差）；``object_pose(position,yaw)`` 反算与 §3.4 正向互为逆（内部 yaw）；
  full-matrix 逆恒等任意持物关系下精确成立；joints/FK 模式与运动链同一语义。
* 判据（全部进 --report）：ROI 内落点拒绝（含现 bin 中心在 ROI 内）；A 槽位二维逐对净距
  不足拒绝；净距恰 = clearance 通过（闭区间语义）；yaw 归一化与 wrap180 边界；中心/姿态
  有限；default/bin 旧条目保留不被改名挪用。
* 交付规则：输入 profile 文件 SHA256 前后不变；候选在副本上产出、结构合法可被
  ``load_calibration_motion_params``（直接 / 注回副本后）加载、恒为 draft 且不覆盖 verified；
  任一判据不过 → 非零退出且不产出候选（只留报告）；CLI 缺参/未知参/缺输入非零。

与 P4-03 已交付用例分文件存放（`tests/test_calibration_tools.py` 专测 record_pick_place
工具链，本文件专测落点标定工具），避免改动既有用例——见开发日志说明。

全部离线：只在 tmp 目录写合成 profile/记录；除 joints 模式一例外不构造 placo 模型，
判据/换算用例走 TCP 自报模式；绝不触碰硬件或 vision。
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_place_poses as cpp  # noqa: E402
from configs.motion_params import (  # noqa: E402
    load_calibration_motion_params,
    wrap180,
)
from tests.support import (  # noqa: E402
    SIM_PROFILE,
    motor_calibration_sha256,
    raw_profile,
    write_profile,
)

# 由 sim profile 冻结的判据同源参数（测试直接引用，不硬编码第二套真值）。
_TB = np.array(raw_profile()["workspace"]["target_bounds_m"], dtype=float)   # (2,3)
_ENV = np.array(raw_profile()["grasp"]["object_envelope_m"], dtype=float)     # (L,W,H)
_CLR = float(raw_profile()["collision"]["clearance_m"])
_YAW_OFFSET = float(raw_profile()["grasp"]["yaw_offset_deg"])                 # 0.0
# 持物参考：抓闭 TCP z=0.05、抓物中心 z=0.022 → 顶抓下物体中心 = 释放 TCP 下移 0.028。
_DZ = 0.05 - 0.022


def _urdf_sha() -> str:
    return hashlib.sha256(
        (SIM_PROFILE.parent / raw_profile()["model"]["urdf_path"]).read_bytes()).hexdigest()


def _write_draft(tmp: Path) -> tuple[Path, dict]:
    """写出一份「完整 draft」profile 副本（全字段非 null、资源哈希回填），供工具加载。

    返回 (路径, 落盘后的原始 dict)——落盘后的 dict 已把 urdf_path 解析为绝对路径、
    与工具读到的 profile 内容一致，供"候选是否只改了 places/status/verification"逐字段比对。
    """
    data = raw_profile()
    data["status"] = "draft"
    data["verification"] = None
    data["model"]["urdf_sha256"] = _urdf_sha()
    data["model"]["motor_calibration_sha256"] = motor_calibration_sha256()
    path = write_profile(tmp, data)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _hold() -> dict:
    return {
        "method": "tcp_teaching_vision_center",
        "grasp_object_center_m": [0.35, 0.0, 0.022],
        "grasp_object_yaw_deg": 0.0,
        "grasp_close_tcp": {"mode": "tcp", "xyz_m": [0.35, 0.0, 0.05], "yaw_deg": 0.0},
    }


def _release_for(obj_center, obj_yaw: float = 0.0) -> dict:
    x, y, z = obj_center
    # 顶抓（yaw_offset=0）下物体 xy 与释放 TCP xy 相同、物体 z = 释放 z - _DZ。
    return {"mode": "tcp", "xyz_m": [x, y, z + _DZ], "yaw_deg": obj_yaw}


def _trial(pid: str, obj_center, obj_yaw: float = 0.0) -> dict:
    return {"place_id": pid, "release_tcp": _release_for(obj_center, obj_yaw)}


def _records(trials: list[dict]) -> dict:
    return {"schema_version": 1, "kind": "place_pose_teaching",
            "holding_reference": _hold(), "trials": trials}


def _outside_center(dx: float = 0.0, dy: float = 0.09) -> list[float]:
    """构造一个稳定落在 target_bounds 之外的物体中心（y 越过 ROI 上界 + dy）。"""
    return [0.46 + dx, _TB[1, 1] + dy, 0.022]


# ---------------------------------------------------------------------------
# P4-T03 换算：双向闭合 / 逆恒等 / 与 kinematics_ext 同一语义
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("obj_yaw", [0.0, 20.0, -37.5, 45.0, -80.0])
def test_T03_换算双向闭合顶抓内部yaw(obj_yaw):
    """合成释放 TCP → 物体中心 → 正向 place_tcp_pose 复算回 TCP（数值容差）。"""
    import qingyun.grabbing.kinematics_ext as K

    center = _outside_center()
    T_O, _meta = cpp.resolve_holding(_hold(), _load_params_safe(), cpp._ModelBox(None), "h")
    T_release = cpp.resolve_tcp_pose(_release_for(center, obj_yaw), None, cpp._ModelBox(None), "r")
    pos, yaw, T_B_O_full = cpp.tcp_to_object_place(T_release, T_O)

    # 物体中心 xy 与释放 TCP 相同、z 下移 _DZ；yaw 归一化。
    assert np.allclose(pos[:2], np.array(center[:2]), atol=1e-12)
    assert pos[2] == pytest.approx(center[2], abs=1e-12)
    assert yaw == pytest.approx(wrap180(obj_yaw), abs=1e-9)
    # 物体姿态是纯 Z 旋转（内部 yaw，未被 wrap180 翻折）⇒ object_pose(pos,yaw) 复原全矩阵。
    assert np.allclose(K.object_pose(pos, yaw), T_B_O_full, atol=1e-9)
    # 正向复算回释放 TCP：place_tcp_pose(object_pose(pos,yaw), T_TCP_O) == 释放 TCP。
    T_back = K.place_tcp_pose(K.object_pose(pos, yaw), T_O)
    assert np.max(np.abs(T_back - T_release)) < 1e-9


def test_T03_full矩阵逆恒等对任意持物关系精确():
    """§3.4 正向 place_tcp_pose 与换算 T_B_O = T_release @ T_TCP_O 互为逆（full matrix 精确）。"""
    import qingyun.grabbing.kinematics_ext as K

    # 任取一个非退化的持物关系（此处带一个非零 pitch 倾角，检验恒等不依赖顶抓假设）。
    R = K.rz_deg(25.0) @ K.R_TOP_DOWN
    T_O = np.eye(4)
    T_O[:3, :3] = R @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 纯旋转部分
    T_O[:3, 3] = [0.01, -0.02, 0.03]
    T_release = K.top_down_pose([0.44, 0.05, 0.06], 12.0)
    pos, yaw, T_B_O_full = cpp.tcp_to_object_place(T_release, T_O)
    # full-matrix 逆恒等（不经过 (pos,yaw) 投影）：place_tcp_pose(T_B_O_full, T_O) == T_release。
    assert np.max(np.abs(K.place_tcp_pose(T_B_O_full, T_O) - T_release)) < 1e-9
    assert np.all(np.isfinite(pos)) and -90.0 <= yaw < 90.0


def test_T03_joints模式FK与运动链同一语义():
    """关节角读回走 FK（ArmModel.fk_tcp）；full-matrix 逆恒等仍精确，证明复用同一 TCP 语义。"""
    import qingyun.grabbing.kinematics_ext as K
    from tests.support import load_sim

    params = load_sim()
    box = cpp._ModelBox(params)
    q_close = np.array([0.0, -60.0, 60.0, 0.0, 0.0])
    hold = {"method": "joints", "grasp_object_center_m": [0.35, 0.0, 0.022],
            "grasp_object_yaw_deg": 0.0,
            "grasp_close_tcp": {"mode": "joints", "q_urdf_deg": q_close.tolist(), "gripper_pct": 30.0}}
    T_O, _ = cpp.resolve_holding(hold, params, box, "h")
    rel = {"mode": "joints", "q_urdf_deg": [10.0, -50.0, 55.0, 5.0, -20.0], "gripper_pct": 30.0}
    T_release = cpp.resolve_tcp_pose(rel, params, box, "r")
    # FK 与运动链一致：resolve_tcp_pose(joints) 应等于 ArmModel.fk_tcp。
    assert np.allclose(T_release, box.get().fk_tcp(np.array([10.0, -50.0, 55.0, 5.0, -20.0]), 30.0),
                       atol=1e-12)
    _pos, _yaw, T_B_O_full = cpp.tcp_to_object_place(T_release, T_O)
    assert np.max(np.abs(K.place_tcp_pose(T_B_O_full, T_O) - T_release)) < 1e-6


def _load_params_safe():
    """换算用例只需 target_bounds/clearance/envelope/yaw_offset——借用已加载 sim params（内存）。"""
    from tests.support import load_sim

    return load_sim()


# ---------------------------------------------------------------------------
# §2.5 判据
# ---------------------------------------------------------------------------


def test_ROI内落点拒绝_含现bin中心():
    params = _load_params_safe()
    trials = [
        _trial("strawberry_B_bin", [0.46, 0.30, 0.022]),          # 外部
        _trial("strawberry_C_bin", list(_TB.mean(axis=0))),       # ROI 正中 → 内部（模拟现 bin 中心在 ROI 内）
    ]
    result = cpp.evaluate(_records(trials), params)
    a = next(c for c in result["checks"] if c["id"] == "a_outside_target_bounds")
    assert not a["passed"]
    assert "strawberry_C_bin" in a["failed_places"]


def test_ROI闭区间语义():
    """§2.5a 与视觉过滤同一闭区间语义：逐轴 lo<=v<=hi 命中即"in"，边界上算内部。"""
    lo, hi = _TB
    on_hi = np.array([lo[0] + 0.01, hi[1], hi[2]])           # y、z 恰好压上界 → 内部
    on_lo = np.array([lo[0], lo[1], lo[2]])                  # 全轴压下界 → 内部
    just_out = np.array([hi[0] + 1e-9, hi[1], hi[2]])        # x 越界 → 外部
    assert cpp.inside_closed_bounds(on_hi, _TB) is True
    assert cpp.inside_closed_bounds(on_lo, _TB) is True
    assert cpp.inside_closed_bounds(just_out, _TB) is False
    # 中心在 ROI 内 → 判据 a 拒绝该 place。
    params = _load_params_safe()
    result = cpp.evaluate(_records([_trial("strawberry_B_bin", list(_TB.mean(axis=0)))]), params)
    a = next(c for c in result["checks"] if c["id"] == "a_outside_target_bounds")
    assert not a["passed"] and "strawberry_B_bin" in a["failed_places"]


def test_A槽净距不足拒绝():
    params = _load_params_safe()
    L = _ENV[0]
    base = np.array(_outside_center())
    c1 = base.copy()
    c2 = base + np.array([L + _CLR - 0.002, 0.0, 0.0])   # 沿长轴，净距 = (L+clr-0.002)-L = clr-0.002 不足
    trials = [_trial("strawberry_A_01", c1.tolist(), 0.0),
              _trial("strawberry_A_02", c2.tolist(), 0.0)]
    result = cpp.evaluate(_records(trials), params)
    b = next(c for c in result["checks"] if c["id"] == "b_A_pairwise_clearance")
    assert not b["passed"]
    d = cpp.net_distance_2d(c1, 0.0, c2, 0.0, _ENV[0], _ENV[1])
    assert d < _CLR


def test_A槽净距恰等于clearance通过_闭区间():
    params = _load_params_safe()
    L = _ENV[0]
    base = np.array(_outside_center())
    c1 = base.copy()
    c2 = base + np.array([L + _CLR, 0.0, 0.0])   # 净距恰好 = clearance
    d = cpp.net_distance_2d(c1, 0.0, c2, 0.0, _ENV[0], _ENV[1])
    assert d == pytest.approx(_CLR, abs=1e-12)
    trials = [_trial("strawberry_A_01", c1.tolist(), 0.0),
              _trial("strawberry_A_02", c2.tolist(), 0.0)]
    result = cpp.evaluate(_records(trials), params)
    b = next(c for c in result["checks"] if c["id"] == "b_A_pairwise_clearance")
    assert b["passed"] and b["pair_count"] == 1


def test_A槽净距按各自yaw构造矩形_二维逐对():
    params = _load_params_safe()
    base = np.array(_outside_center())
    # 一个 yaw=0（长轴沿 x）、一个 yaw=90（长轴沿 y）；沿连线 x 方向的投影半径各异。
    c1 = base.copy()
    c2 = base + np.array([_ENV[1] + _CLR + 0.02, 0.0, 0.0])
    r1 = (_ENV[0] * 1.0 + _ENV[1] * 0.0) / 2.0     # yaw0 沿 x：长轴贡献
    r2 = (_ENV[0] * 0.0 + _ENV[1] * 1.0) / 2.0     # yaw90 沿 x：短轴贡献
    expect_net = (c2[0] - c1[0]) - r1 - r2
    got = cpp.net_distance_2d(c1, 0.0, c2, 90.0, _ENV[0], _ENV[1])
    assert got == pytest.approx(expect_net, abs=1e-12)
    assert got >= _CLR   # 用足二维投影后判定通过


def test_中心重合按净距不足处理():
    params = _load_params_safe()
    c = _outside_center()
    trials = [_trial("strawberry_A_01", list(c), 0.0),
              _trial("strawberry_A_02", list(c), 0.0)]   # 完全重合
    result = cpp.evaluate(_records(trials), params)
    b = next(c for c in result["checks"] if c["id"] == "b_A_pairwise_clearance")
    assert not b["passed"]
    assert cpp.net_distance_2d(np.array(c), 0.0, np.array(c), 0.0, _ENV[0], _ENV[1]) == float("-inf")


def test_yaw归一化与wrap180边界():
    params = _load_params_safe()
    trials = [
        _trial("strawberry_A_01", _outside_center(dy=0.10), 100.0),    # → wrap180(100) = -80
        _trial("strawberry_A_02", _outside_center(dy=0.20), 90.0),     # → wrap180(90) = -90
        _trial("strawberry_A_03", _outside_center(dy=0.30), -95.0),    # → wrap180(-95) = 85
    ]
    result = cpp.evaluate(_records(trials), params)
    yaws = {e["place_id"]: e["converted_yaw_deg"] for e in result["entries"]}
    assert yaws["strawberry_A_01"] == pytest.approx(wrap180(100.0), abs=1e-9)
    assert yaws["strawberry_A_02"] == pytest.approx(-90.0, abs=1e-9)
    assert yaws["strawberry_A_03"] == pytest.approx(85.0, abs=1e-9)
    c = next(x for x in result["checks"] if x["id"] == "c_finite_yaw_range")
    assert c["passed"] and all(-90.0 <= v < 90.0 for v in yaws.values())


def test_非有限中心拒绝():
    params = _load_params_safe()
    trials = [{"place_id": "strawberry_B_bin",
               "release_tcp": {"mode": "tcp", "xyz_m": [0.46, 0.30, float("nan")], "yaw_deg": 0.0}}]
    with pytest.raises(cpp.InputError):
        cpp.evaluate(_records(trials), params)


def test_default_bin保留不被改名挪用(tmp_path):
    params = _load_params_safe()
    assert "default" in params.places and "bin" in params.places
    # 旧条目 key 不会被本工具的候选 ID 命名空间命中。
    for legacy in ("default", "bin"):
        with pytest.raises(cpp.InputError):
            cpp.classify_place(legacy)
    old_default = params.places["default"].position.tolist()
    old_bin = params.places["bin"].position.tolist()

    tmp = tmp_path
    profile, raw = _write_draft(tmp)
    trials = [_trial("strawberry_A_01", _outside_center(), 0.0),
              _trial("strawberry_B_bin", _outside_center(dy=0.30), 10.0)]
    inp = tmp / "in.json"
    inp.write_text(json.dumps(_records(trials)), encoding="utf-8")
    out = tmp / "cand.json"
    code = cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(out), "--report", str(tmp / "rep.json")])
    assert code == cpp.EXIT_OK
    cand = json.loads(out.read_text(encoding="utf-8"))
    # 旧条目原样保留（逐字节等价），且未被删改挪用。
    assert cand["places"]["default"] == raw["places"]["default"]
    assert cand["places"]["bin"] == raw["places"]["bin"]
    assert cand["places"]["default"]["position"] == old_default
    assert cand["places"]["bin"]["position"] == old_bin
    # 新候选都是 strawberry_*，不与旧 key 冲突。
    new_ids = [k for k in cand["places"] if k.startswith("strawberry")]
    assert set(new_ids) == {"strawberry_A_01", "strawberry_B_bin"}
    # 报告记录旧条目处置留给实测决定（判据 d）。
    rep = json.loads((tmp / "rep.json").read_text(encoding="utf-8"))
    d = next(c for c in rep["criteria"] if c["id"] == "d_legacy_entries_preserved")
    assert d["passed"] and set(["default", "bin"]).issubset(set(d["preserved_legacy_ids"]))


# ---------------------------------------------------------------------------
# 交付规则：输入不变 / 候选合法 draft / 判据不过不产出
# ---------------------------------------------------------------------------


def test_输入profile文件SHA256前后不变(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    before_profile = cpp.sha256_file(profile)
    inp = tmp_path / "in.json"
    rec = _records([_trial("strawberry_A_01", _outside_center(), 0.0),
                    _trial("strawberry_B_bin", _outside_center(dy=0.30), 5.0)])
    inp.write_text(json.dumps(rec), encoding="utf-8")
    before_input = cpp.sha256_file(inp)
    code = cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(tmp_path / "cand.json"), "--report", str(tmp_path / "rep.json")])
    assert code == cpp.EXIT_OK
    assert cpp.sha256_file(profile) == before_profile          # 输入 profile 绝不被改写
    assert cpp.sha256_file(inp) == before_input                # 输入记录亦不被改写


def test_候选输出合法可加载且为draft不覆盖verified(tmp_path):
    profile, raw = _write_draft(tmp_path)
    inp = tmp_path / "in.json"
    trials = [_trial("strawberry_A_01", _outside_center(), 0.0),
              _trial("strawberry_A_02", _outside_center(dx=0.06), 0.0),
              _trial("strawberry_B_bin", _outside_center(dy=0.30), 12.0),
              _trial("strawberry_C_bin", _outside_center(dy=0.40), -20.0)]
    inp.write_text(json.dumps(_records(trials)), encoding="utf-8")
    out = tmp_path / "cand.json"
    code = cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(out), "--report", str(tmp_path / "rep.json")])
    assert code == cpp.EXIT_OK and out.exists()
    # 候选直接可被完整加载；status=draft、verification=null，不改写运行字段。
    params = load_calibration_motion_params(out)
    assert params.status == "draft"
    cand = json.loads(out.read_text(encoding="utf-8"))
    assert cand["verification"] is None
    assert cand["grasp"] == raw["grasp"] and cand["workspace"] == raw["workspace"]
    assert cand["model"] == raw["model"]
    for pid in ("strawberry_A_01", "strawberry_A_02", "strawberry_B_bin", "strawberry_C_bin"):
        assert pid in params.places
        assert -90.0 <= params.places[pid].yaw_deg < 90.0
    # A 槽位有序：先 01 后 02。
    a_keys = [k for k in cand["places"] if k.startswith("strawberry_A_")]
    assert a_keys == sorted(a_keys)


def test_候选place注回原副本后仍可加载(tmp_path):
    profile, raw = _write_draft(tmp_path)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(_records([
        _trial("strawberry_A_01", _outside_center(), 0.0),
        _trial("strawberry_B_bin", _outside_center(dy=0.30), 10.0)])), encoding="utf-8")
    out = tmp_path / "cand.json"
    assert cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(out), "--report", str(tmp_path / "rep.json")]) == cpp.EXIT_OK
    cand = json.loads(out.read_text(encoding="utf-8"))
    injected = copy.deepcopy(raw)
    injected["places"] = cand["places"]
    inj_path = tmp_path / "injected.json"
    inj_path.write_text(json.dumps(injected), encoding="utf-8")
    params = load_calibration_motion_params(inj_path)     # 注回副本后可加载（结构合法）
    assert {"strawberry_A_01", "strawberry_B_bin", "default", "bin"} <= set(params.places)


def test_判据不过不产出候选只留报告(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    inp = tmp_path / "in.json"
    # C_bin 落在 ROI 正中 → 判据 a 失败。
    inp.write_text(json.dumps(_records([
        _trial("strawberry_A_01", _outside_center(), 0.0),
        _trial("strawberry_C_bin", list(_TB.mean(axis=0)), 0.0)])), encoding="utf-8")
    out = tmp_path / "cand.json"
    rep = tmp_path / "rep.json"
    code = cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(out), "--report", str(rep)])
    assert code == cpp.EXIT_CRITERIA
    assert not out.exists()                        # 候选不产出（不覆盖已发布 profile）
    assert rep.exists()                            # 报告始终产出
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["overall_pass"] is False
    assert report["output"]["candidate_written"] is False
    assert "strawberry_C_bin" in next(
        c for c in report["criteria"] if c["id"] == "a_outside_target_bounds")["failed_places"]


# ---------------------------------------------------------------------------
# 报告结构
# ---------------------------------------------------------------------------


def test_报告含哈希换算前后与待实测清单(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(_records([
        _trial("strawberry_A_01", _outside_center(), 0.0),
        _trial("strawberry_B_bin", _outside_center(dy=0.30), 10.0)])), encoding="utf-8")
    out = tmp_path / "cand.json"
    rep = tmp_path / "rep.json"
    assert cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(out), "--report", str(rep)]) == cpp.EXIT_OK
    report = json.loads(rep.read_text(encoding="utf-8"))
    # 资源哈希：输入 profile / 记录 / URDF / 电机校准 / 候选参数哈希。
    assert len(report["inputs"]["profile_sha256"]) == 64
    assert len(report["inputs"]["records_sha256"]) == 64
    assert len(report["resource_sha256"]["urdf"]) == 64
    assert len(report["resource_sha256"]["motor_calibration"]) == 64
    assert report["output"]["candidate_sha256"] == cpp.sha256_file(out)
    # 每条 place 含换算前后与闭合残差。
    b = next(p for p in report["places"] if p["place_id"] == "strawberry_B_bin")
    assert b["before_release_tcp"]["mode"] == "tcp"
    assert len(b["after_position_m"]) == 3 and "after_yaw_deg" in b
    assert b["closure_residual_m"] < 1e-7
    # B/C 待实测清单存在，且不谎称通过（判据 e passed 但不设伪造门槛）。
    assert report["pending_field_verification"]
    e = next(c for c in report["criteria"] if c["id"] == "e_B_C_shape_and_roi_only")
    assert e["b_c_present"] and e["pending_field_items"]


# ---------------------------------------------------------------------------
# CLI 参数校验（P4-T06 惯例）
# ---------------------------------------------------------------------------


def test_CLI缺参非零(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cpp.run(["--profile", str(profile)])            # 缺 --input/--output/--report
    assert exc.value.code != 0


def test_CLI未知参非零(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cpp.run(["--profile", str(profile), "--input", "x", "--output", "y",
                 "--report", "z", "--bogus", "1"])
    assert exc.value.code != 0


def test_CLI缺输入文件退出码2(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    code = cpp.run(["--profile", str(profile), "--input", str(tmp_path / "nope.json"),
                    "--output", str(tmp_path / "c.json"), "--report", str(tmp_path / "r.json")])
    assert code == cpp.EXIT_INPUT


def test_CLI输出等于profile被拒(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    code = cpp.run(["--profile", str(profile), "--input", str(tmp_path / "nope.json"),
                    "--output", str(profile), "--report", str(tmp_path / "r.json")])
    assert code == cpp.EXIT_INPUT


def test_未知place在换算前失败退出码2(tmp_path):
    profile, _raw = _write_draft(tmp_path)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(_records([{"place_id": "garbage",
                                          "release_tcp": _release_for(_outside_center())}])),
                   encoding="utf-8")
    code = cpp.run(["--profile", str(profile), "--input", str(inp),
                    "--output", str(tmp_path / "c.json"), "--report", str(tmp_path / "r.json")])
    assert code == cpp.EXIT_INPUT


def test_simulation_profile被拒退出码2(tmp_path):
    # 生产 sim 配置不属于标定链：load_calibration_motion_params 拒绝 → 工具退出码 2。
    sim = tmp_path / "sim.json"
    sim.write_text(json.dumps(raw_profile()), encoding="utf-8")
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(_records([_trial("strawberry_A_01", _outside_center(), 0.0)])),
                   encoding="utf-8")
    code = cpp.run(["--profile", str(sim), "--input", str(inp),
                    "--output", str(tmp_path / "c.json"), "--report", str(tmp_path / "r.json")])
    assert code == cpp.EXIT_INPUT
