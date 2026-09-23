"""P4-T09：完整 draft 参数加载器 load_calibration_motion_params 与其生产隔离。

覆盖 P4 §2.8.1「完整 draft 参数加载」：标定工具专用入口接受完整 draft/verified、
拒绝 null/未测物理字段、拒绝 simulation/坏 status、拒绝资源缺失/哈希不匹配；
唯一豁免「必须 verified 及已完成验收报告绑定」；既有 real/mock 两模式对外行为
逐字不变（draft 仍被 real 拒绝，完整 draft 也仍被 mock 拒绝）；原文件不被修改。

draft 样例的构造方式：以仓库仿真 profile（全字段非 null）的**副本**为底，把
status 改成 draft、verification 置 None，并把 model 内两项资源哈希回填成两份
真实文件的实算值（标定加载与 real 同款要核对哈希）。真机 draft profile 只读
加载做集成用例，绝不改写。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from configs.motion_params import (
    ParamsError,
    load_calibration_motion_params,
    load_motion_params,
)
from tests.support import (
    ROOT,
    SIM_PROFILE,
    _set_path,
    motor_calibration_sha256,
    raw_profile,
    write_profile,
)

REAL_DRAFT_PROFILE = ROOT / "calibration/REAL_ARM/PC/profile.json"


def _actual_urdf_sha256() -> str:
    """仓库 URDF 的实算哈希（标定/real 加载要核对它）。"""
    urdf = SIM_PROFILE.parent / raw_profile()["model"]["urdf_path"]
    return hashlib.sha256(urdf.read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draft_profile_path(
    tmp_path: Path,
    mutate: dict | None = None,
    *,
    status: str = "draft",
    correct_hashes: bool = True,
) -> Path:
    """写出一份「完整 draft」profile 副本并返回其路径。

    底稿用仿真配置（全字段非 null），仅把 status 换成 draft、verification 置 None；
    再回填两份真实资源哈希，使其通过 real 同款的哈希核对。mutate 在回填之后应用，
    便于负向用例覆盖单个字段（注入 null / 写坏哈希 / 指向缺失文件等）。
    """
    data = raw_profile()
    data["status"] = status
    data["verification"] = None
    if correct_hashes:
        data["model"]["urdf_sha256"] = _actual_urdf_sha256()
        data["model"]["motor_calibration_sha256"] = motor_calibration_sha256()
    for path, value in (mutate or {}).items():
        _set_path(data, path, value)
    return write_profile(tmp_path, data)


# ---------------------------------------------------------------------------
# 1. 完整 draft 被接受（且保留原始 status / 路径语义）
# ---------------------------------------------------------------------------


def test_完整draft可被标定加载器接受(tmp_path):
    p = _draft_profile_path(tmp_path)
    params = load_calibration_motion_params(p)
    assert params.status == "draft"          # 保留原始状态，不改写、不伪装 simulation
    assert params.schema_version == 1
    # places / 运行阈值可用（完整解析后各字段就位）
    assert "default" in params.places
    place = params.place("default")
    assert place.position.shape == (3,)
    assert np.isfinite(place.position).all()
    assert params.grasp.lift_height_m > 0.0
    assert params.collision.clearance_m > 0.0


def test_标定加载器profile_dir指向原始目录(tmp_path):
    p = _draft_profile_path(tmp_path)
    params = load_calibration_motion_params(p)
    assert params.profile_dir == p.parent
    assert params.urdf_path_resolved().is_file()
    assert params.motor_calibration_path_resolved().is_file()


def test_verified配置标定加载器也接受(tmp_path):
    p = _draft_profile_path(tmp_path, status="verified")
    params = load_calibration_motion_params(p)
    assert params.status == "verified"


def test_真机draft_profile只读加载不被修改():
    """集成用例：仓库里的真机 draft profile 能被标定加载器读成完整 MotionParams，
    且加载前后文件字节完全一致（不写文件、不改 status）。"""
    if not REAL_DRAFT_PROFILE.is_file():
        pytest.skip("仓库真机 profile 不存在")
    before = _file_sha256(REAL_DRAFT_PROFILE)
    params = load_calibration_motion_params(REAL_DRAFT_PROFILE)
    after = _file_sha256(REAL_DRAFT_PROFILE)
    assert before == after
    assert params.status == "draft"


# ---------------------------------------------------------------------------
# 2. 不豁免 null / 未测物理字段
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "grasp.lift_height_m",
        "collision.clearance_m",
        "ik.position_tol_m",
        "workspace.table_z_m",
        "gripper.preopen_pct",
    ],
)
def test_拒绝运行期null字段(tmp_path, field):
    p = _draft_profile_path(tmp_path, mutate={field: None})
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert field in str(exc.value)
    assert "null" in str(exc.value) or "未测" in str(exc.value)


def test_拒绝数组内的null未测元素(tmp_path):
    # 把一处示教姿态向量里的某个分量改成 null：整体结构仍在，但该物理值未测。
    p = _draft_profile_path(tmp_path, mutate={"motion.max_velocity_deg_s": [90.0, None, 90.0, 90.0, 90.0]})
    with pytest.raises(ParamsError):
        load_calibration_motion_params(p)


# ---------------------------------------------------------------------------
# 3. 拒绝 simulation / 坏 status（标定链语义）
# ---------------------------------------------------------------------------


def test_拒绝simulation配置(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"status": "simulation"})
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert "simulation" in str(exc.value)
    # 错误信息须点明：生产加载走 mode、仿真配置不属于标定链
    msg = str(exc.value)
    assert "标定" in msg and "load_motion_params" in msg


def test_拒绝未知status(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"status": "bogus-state"})
    with pytest.raises(ParamsError):
        load_calibration_motion_params(p)


def test_拒绝非字符串status(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"status": 3})
    with pytest.raises(ParamsError):
        load_calibration_motion_params(p)


# ---------------------------------------------------------------------------
# 4. 资源存在性与哈希（与 real 同款）
# ---------------------------------------------------------------------------


def test_拒绝urdf哈希不匹配(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"model.urdf_sha256": "0" * 64})
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert "model.urdf_sha256" in str(exc.value)


def test_拒绝电机校准哈希不匹配(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"model.motor_calibration_sha256": "f" * 64})
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert "motor_calibration_sha256" in str(exc.value)


def test_拒绝缺失的电机校准文件(tmp_path):
    # correct_hashes=False 保留仿真包里写死的电机校准文件不存在的相对名，
    # 直接把校准文件路径改到一个不存在的位置，存在性检查应先于哈希失败。
    p = _draft_profile_path(tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["model"]["motor_calibration_path"] = "no_such_calibration.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert "文件不存在" in str(exc.value)


def test_拒绝缺失的urdf文件(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"model.urdf_path": "missing.urdf"})
    with pytest.raises(ParamsError) as exc:
        load_calibration_motion_params(p)
    assert "model.urdf_path" in str(exc.value) and "文件不存在" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. 唯一豁免边界：不校验验收报告绑定
# ---------------------------------------------------------------------------


def test_豁免报告绑定_verified无报告仍被标定加载接受(tmp_path):
    """verified 但没有已完成验收报告绑定：real 必须拒绝，标定加载仍接受。

    这正是 §2.8.1 规定的唯一豁免——不要求 verified、不做 _check_report_binding。
    """
    p = _draft_profile_path(tmp_path, status="verified")
    # 标定加载接受（不碰报告）
    params = load_calibration_motion_params(p)
    assert params.status == "verified"
    assert params.verification is None
    # real 因缺少验收报告绑定而拒绝
    with pytest.raises(ParamsError) as exc:
        load_motion_params(p, mode="real")
    assert "verification" in str(exc.value)


def test_draft没有reports也可通过标定加载(tmp_path):
    """draft 常态即无 reports：目录里没有 reports/、verification=None，仍应接受。"""
    p = _draft_profile_path(tmp_path)
    assert not (p.parent / "reports").exists()
    params = load_calibration_motion_params(p)
    assert params.status == "draft"


# ---------------------------------------------------------------------------
# 6. 生产隔离：real/mock 行为逐字不变（draft 仍被拒）
# ---------------------------------------------------------------------------


def test_real仍拒绝完整draft(tmp_path):
    p = _draft_profile_path(tmp_path)
    with pytest.raises(ParamsError) as exc:
        load_motion_params(p, mode="real")
    assert "verified" in str(exc.value)


def test_mock仍拒绝完整draft(tmp_path):
    p = _draft_profile_path(tmp_path)
    with pytest.raises(ParamsError) as exc:
        load_motion_params(p, mode="mock")
    assert "simulation" in str(exc.value)


def test_标定能力不可经mode字符串获得(tmp_path):
    """P4 §2.8.1 要求不新增 mode="calibration"。既有签名与两模式行为逐字不变，
    因此生产入口没有任何 mode 取值能同时满足「接受 draft」且「核对资源哈希但不绑报告」
    —— 用 mock 加载 draft 会因 status 门槛被拒，即无 mode 可替代标定入口。"""
    p = _draft_profile_path(tmp_path)
    with pytest.raises(ParamsError):        # mock 拒 draft
        load_motion_params(p, mode="mock")
    with pytest.raises(ParamsError):        # real 拒 draft
        load_motion_params(p, mode="real")


# ---------------------------------------------------------------------------
# 7. 原文件不被修改
# ---------------------------------------------------------------------------


def test_标定加载不修改profile文件(tmp_path):
    p = _draft_profile_path(tmp_path, mutate={"status": "draft"})
    before = _file_sha256(p)
    snapshot = json.loads(p.read_text(encoding="utf-8"))
    load_calibration_motion_params(p)
    assert _file_sha256(p) == before
    assert json.loads(p.read_text(encoding="utf-8")) == snapshot
    assert snapshot["status"] == "draft"
    assert snapshot["verification"] is None
