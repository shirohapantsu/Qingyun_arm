"""运行期运动参数 MotionParams 与 profile.json 加载/校验。

规范来源：
    docs/真机参数测量与标定指南.md 第 1、2 节（字段字典与加载规则）
    docs/机械臂运动控制模块技术文档.md 第二节（文件结构）

设计要点：
    * MotionParams 的嵌套属性与 profile.json 字段逐项同名，数组一律加载为
      np.float64；places 映射的每个值加载为公共接口里的 PlacePose。
    * 相对路径以 profile.json 所在目录为基准，因此加载结果里额外携带
      profile_dir 与从 URDF 解析出的 urdf_limits_deg 两个派生字段。
      它们不出现在 JSON 中，键校验时被显式标记为“派生”。
    * real 模式要求 status=verified、全部运行期字段非 null、模型与电机校准
      文件哈希与验收报告匹配；mock 模式使用独立 status=simulation 配置。
    * draft（未测物理值为 null 的骨架配置）只能由 scripts/calibrate.py init
      生成，不能用本模块加载：两种模式都要求运行期字段完整。
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# 公共接口里已冻结的类型直接复用，不在本模块重复定义。
from configs.common_interface import (
    FloatArray,
    JOINT_NAMES,
    MOTOR_NAMES,
    PlacePose,
)

SCHEMA_VERSION = 1
# 控制节拍固定 30Hz（技术文档 6.1 与标定指南 2.3 一致）。
TIMING_FPS = 30
# 单位向量正交性、SO(3) 投影残差的允许公差。
ROTATION_ORTHO_TOL = 1e-6
# 检查示教姿态是否落在有效限位内时额外放给的数值容差（deg）。margin_deg 已经
# 把运行余量扣进限位里，这里只吸收 JSON 写法带来的浮点误差。
LOAD_LIMIT_TOL_DEG = 1e-6
# AABB 每轴必须保留的最小跨度（m）。配置里允许"厚度为 0 的壁"，所以只要求
# min 严格小于 max，用这个量挡住 min/max 写反和恰好相等的退化输入。
LOAD_AXIS_SPAN_M = 1e-9

STATUSES = ("draft", "verified", "simulation")


class ParamsError(ValueError):
    """profile.json 不满足技术文档约束时抛出。

    入口把它统一翻译成 GraspStatus.CONFIG_INVALID，因此消息里必须带上具体的
    字段路径，方便标定人员直接定位到 JSON 的哪一个键。
    """

    def __init__(self, field_path: str, message: str) -> None:
        super().__init__(f"{field_path}: {message}")
        self.field_path = field_path
        self.message = message


# ---------------------------------------------------------------------------
# 第 1 节：字段容器
#
# 每个 dataclass 的字段名与 profile.json 的键一一对应；字段顺序与标定指南
# 第 2 节的表格顺序保持一致，便于逐行核对。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelParams:
    """model 组：模型资源路径与哈希。"""

    urdf_path: str
    urdf_sha256: str
    motor_calibration_path: str
    motor_calibration_sha256: str


@dataclass(frozen=True)
class MotorParams:
    """motor 组：串口身份与 C 侧单位换算比例，数组按 MOTOR_NAMES 长度 6。"""

    port: str
    baudrate: int
    ids: NDArray[np.int64]                      # (6,) 固定 [1..6]
    models: tuple[str, ...]                     # (6,)
    firmware: tuple[str, ...]                   # (6,)
    position_resolution: NDArray[np.int64]      # (6,) 编码分辨率，每项 >1
    current_ma_per_raw: FloatArray              # (6,) mA/raw，每项非零
    current_zero_raw: FloatArray                # (6,) 零偏 raw
    velocity_deg_s_per_raw: FloatArray          # (6,) deg/s per raw count
    gripper_closed_raw: int
    gripper_open_raw: int


@dataclass(frozen=True)
class JointsParams:
    """joints 组：q_bus→q_urdf 映射、行程与运行上限，数组按 JOINT_NAMES 长度 5。"""

    sign: NDArray[np.int64]        # (5,) ±1
    zero_offset_deg: FloatArray    # (5,) deg
    measured_limits_deg: FloatArray        # (5,2) [lower,upper]，URDF 坐标
    application_limits_deg: FloatArray     # (5,2) 线缆/支架/桌面限制
    margin_deg: FloatArray         # (5,) 端点重复误差与运行余量
    hard_current_ma: FloatArray    # (5,) mA


@dataclass(frozen=True)
class ToolTranslationSample:
    """tool.translation_samples 的一项。"""

    gripper_pct: float
    xyz_m: FloatArray  # (3,) 在 E 系中的 TCP 平移


@dataclass(frozen=True)
class GapSample:
    """gripper.gap_table 的一项：开度 -> 指垫内表面间距。"""

    gripper_pct: float
    gap_m: float


@dataclass(frozen=True)
class AngleSample:
    """gripper.angle_table 的一项：开度 -> 活动指 URDF 角。"""

    gripper_pct: float
    angle_deg: float


@dataclass(frozen=True)
class ToolParams:
    """tool 组：固定工具旋转与随开度变化的 TCP 平移。"""

    rotation_e_tcp: FloatArray            # (3,3) SO(3)
    translation_samples: tuple[ToolTranslationSample, ...]
    position_error_bound_m: float
    orientation_error_bound_deg: float


@dataclass(frozen=True)
class Obstacle:
    """workspace.obstacles 的一项：固定障碍 AABB。"""

    id: str
    bounds_m: FloatArray  # (2,3) 第 0 行 XYZ 最小值，第 1 行最大值


@dataclass(frozen=True)
class WorkspaceParams:
    """workspace 组：工作域、桌面、待机位、中转位与固定障碍。"""

    target_bounds_m: FloatArray      # (2,3) m
    tcp_bounds_m: FloatArray         # (2,3) m
    table_z_m: float
    table_flatness_m: float
    home_joints_deg: FloatArray      # (5,) deg
    safe_waypoints_deg: FloatArray   # (K,5) deg，K>=1
    obstacles: tuple[Obstacle, ...]


@dataclass(frozen=True)
class LinkCapsule:
    """collision.link_capsules 的一项：端点定义在对应 link 系内的胶囊体。"""

    id: str
    link: str
    p0_m: FloatArray  # (3,)
    p1_m: FloatArray  # (3,)
    radius_m: float


@dataclass(frozen=True)
class CollisionParams:
    """collision 组。"""

    link_capsules: tuple[LinkCapsule, ...]
    ignore_self_pairs: tuple[tuple[str, str], ...]
    clearance_m: float
    max_joint_substep_deg: float


@dataclass(frozen=True)
class TimingParams:
    """timing 组：节拍与有界 I/O 时限。"""

    fps: int
    read_timeout_s: float
    write_timeout_s: float
    max_feedback_age_s: float
    max_feedback_span_s: float
    max_tick_lateness_s: float
    stop_poll_s: float
    planning_poll_s: float


@dataclass(frozen=True)
class MotionConstraints:
    """motion 组：关节/TCP 运动学上限与到位、跟踪判据。"""

    max_velocity_deg_s: FloatArray          # (5,)
    max_acceleration_deg_s2: FloatArray     # (5,)
    max_command_step_deg: FloatArray        # (5,)
    tcp_max_velocity_m_s: float
    tcp_max_acceleration_m_s2: float
    start_position_tol_deg: FloatArray      # (5,)
    following_error_deg: FloatArray         # (5,)
    following_error_hard_deg: FloatArray    # (5,)
    following_error_dwell_s: float
    settle_position_tol_deg: FloatArray     # (5,)
    settle_velocity_tol_deg_s: FloatArray   # (5,)
    settle_dwell_s: float
    settle_timeout_s: float


@dataclass(frozen=True)
class IkParams:
    """ik 组：残差容限、权重、迭代与停滞判据、备用初值。"""

    position_tol_m: float
    tilt_tol_deg: float
    yaw_tol_deg: float
    position_weight: float
    orientation_weight: float
    max_iterations: int
    stagnation_iterations: int
    max_solve_s: float
    min_progress: float
    seed_joints_deg: FloatArray  # (K,5)，允许 K=0


@dataclass(frozen=True)
class GraspParams:
    """grasp 组：中心到抓取 TCP 的固定偏移、包络与竖直高度。"""

    center_offset_object_m: FloatArray  # (3,) 在物体 O 系中
    yaw_offset_deg: float
    object_envelope_m: FloatArray       # (3,) 长、宽、高
    target_position_error_bound_m: float
    target_yaw_error_bound_deg: float
    object_shift_bound_m: float
    approach_height_m: float
    lift_height_m: float
    retreat_height_m: float


@dataclass(frozen=True)
class GripperParams:
    """gripper 组：三张标定表与开合、接触、保持参数。"""

    gap_table: tuple[GapSample, ...]
    angle_table: tuple[AngleSample, ...]
    preopen_pct: float
    release_pct: float
    open_speed_pct_s: float
    close_speed_pct_s: float
    contact_gap_range_m: FloatArray  # (2,) 0<min<max
    contact_current_ma: float
    hard_current_ma: float
    window_samples: int
    contact_dwell_s: float
    empty_closed_pct: float
    empty_tol_pct: float
    close_timeout_s: float
    open_timeout_s: float
    hold_dwell_s: float
    release_dwell_s: float
    slip_opening_change_pct: float
    settle_tol_pct: float
    settle_speed_pct_s: float


@dataclass(frozen=True)
class AcceptanceParams:
    """acceptance 组：验收要求（调参前确定）。"""

    max_position_error_m: float
    max_tilt_error_deg: float
    max_yaw_error_deg: float
    max_stop_latency_s: float
    min_grasp_success_rate: float
    max_drop_rate: float
    max_damage_rate: float
    damage_observe_minutes: float
    min_grasp_trials: int


@dataclass(frozen=True)
class VerificationParams:
    """verification 组：promote 写入的验收报告路径与哈希。"""

    report_path: str
    report_sha256: str


@dataclass(frozen=True)
class MotionParams:
    """profile.json 的完整内存表示。

    JSON 中出现的字段：schema_version/profile_id/robot_id/status/model/motor/
    joints/tool/workspace/collision/places/timing/motion/ik/grasp/gripper/
    acceptance/verification。

    派生字段（不出现在 JSON 中，由加载器填写）：
        profile_dir      profile.json 所在目录，用于解析相对路径
        urdf_limits_deg  从 URDF 读出的 5 个姿态关节限位（已转 deg）
    """

    schema_version: int
    profile_id: str
    robot_id: str
    status: str
    model: ModelParams
    motor: MotorParams
    joints: JointsParams
    tool: ToolParams
    workspace: WorkspaceParams
    collision: CollisionParams
    places: Mapping[str, PlacePose]
    timing: TimingParams
    motion: MotionConstraints
    ik: IkParams
    grasp: GraspParams
    gripper: GripperParams
    acceptance: AcceptanceParams
    verification: VerificationParams | None
    # --- 派生字段 ---
    profile_dir: Path = field(default=Path("."))
    urdf_limits_deg: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))

    # --- 常用派生查询 ---

    def urdf_path_resolved(self) -> Path:
        """把 model.urdf_path 相对 profile.json 目录解析成绝对路径。"""
        return _resolve(self.profile_dir, self.model.urdf_path)

    def motor_calibration_path_resolved(self) -> Path:
        """同上，解析电机校准文件路径。"""
        return _resolve(self.profile_dir, self.model.motor_calibration_path)

    def place(self, place_id: str) -> PlacePose:
        """按 place_id 取放置位姿；未知 id 抛 ParamsError（入口转 CONFIG_INVALID）。"""
        if place_id not in self.places:
            raise ParamsError(
                f"places.{place_id}",
                f"配置里没有该 place_id，已有 {sorted(self.places)}",
            )
        return self.places[place_id]


# ---------------------------------------------------------------------------
# 第 2 节：标量/向量/矩阵校验原语
# ---------------------------------------------------------------------------


def _resolve(base: Path, value: str) -> Path:
    """相对路径以 profile.json 所在目录为基准（标定指南 1.1）。"""
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _reject_constant(name: str) -> float:
    """禁止 NaN/Infinity 进入配置：JSON 扩展字面量一律视为非法。"""
    raise ParamsError("-", f"配置不允许出现非有限浮点字面量 {name}")


def _require_keys(payload: Any, expected: set[str], path: str, derived: set[str] = frozenset()) -> dict:
    """拒绝未知键与缺失键（标定指南 1.1）。"""
    if not isinstance(payload, dict):
        raise ParamsError(path, f"应为对象，实际是 {type(payload).__name__}")
    got = set(payload)
    unknown = sorted(got - expected - derived)
    if unknown:
        raise ParamsError(path, f"出现未知键 {unknown}")
    missing = sorted(expected - got)
    if missing:
        raise ParamsError(path, f"缺失键 {missing}")
    return dict(payload)


def _null_check(path: str, v: Any) -> None:
    """null 表示"该物理值尚未测量"，在 real/mock 加载时一律拒绝。"""
    if v is None:
        raise ParamsError(path, "仍为 null（未测量），不能用于运行")


def _str(path: str, v: Any, *, allow_empty: bool = False) -> str:
    _null_check(path, v)
    if not isinstance(v, str):
        raise ParamsError(path, f"应为字符串，实际是 {type(v).__name__}")
    if not allow_empty and not v.strip():
        raise ParamsError(path, "不能为空字符串")
    return v


def _int(path: str, v: Any, *, minimum: int | None = None, equals: int | None = None) -> int:
    _null_check(path, v)
    # bool 是 int 的子类，但它不是合法的数值参数，单独拒绝。
    if isinstance(v, bool) or not isinstance(v, int):
        raise ParamsError(path, f"应为整数，实际是 {type(v).__name__}")
    if equals is not None and v != equals:
        raise ParamsError(path, f"必须等于 {equals}，实际是 {v}")
    if minimum is not None and v < minimum:
        raise ParamsError(path, f"必须 >= {minimum}，实际是 {v}")
    return v


def _float(
    path: str,
    v: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    nonzero: bool = False,
) -> float:
    _null_check(path, v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ParamsError(path, f"应为数值，实际是 {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise ParamsError(path, f"必须是有限值，实际是 {f}")
    if positive and not f > 0.0:
        raise ParamsError(path, f"必须为正，实际是 {f}")
    if nonnegative and f < 0.0:
        raise ParamsError(path, f"必须非负，实际是 {f}")
    if nonzero and f == 0.0:
        raise ParamsError(path, "不能为 0")
    return f


def _fvec(path: str, v: Any, shape: tuple[int, ...]) -> FloatArray:
    """把 JSON 数组转换成指定形状的 np.float64 数组并检查有限性。

    shape 里的 None 表示该维任意（但至少为 0）。
    """
    _null_check(path, v)
    arr = np.asarray(v, dtype=np.float64)
    want = tuple(s if s is not None else arr.shape[i] for i, s in enumerate(shape))
    if arr.ndim != len(shape) or arr.shape != want:
        raise ParamsError(path, f"shape 应为 {shape}，实际是 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ParamsError(path, "含有非有限值")
    return arr


def _imat(path: str, v: Any, n: int | None) -> NDArray[np.int64]:
    """整数向量：允许 JSON 写 2048 或 2048.0，但值必须是整数。"""
    _null_check(path, v)
    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ParamsError(path, f"应为一维数组，实际 ndim={arr.ndim}")
    if n is not None and arr.shape[0] != n:
        raise ParamsError(path, f"长度应为 {n}，实际是 {arr.shape[0]}")
    out = np.empty(arr.shape[0], dtype=np.int64)
    for i, item in enumerate(arr.tolist()):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ParamsError(f"{path}[{i}]", f"应为整数，实际是 {item!r}")
        if float(item) != int(item):
            raise ParamsError(f"{path}[{i}]", f"应为整数，实际是 {item!r}")
        out[i] = int(item)
    return out


def _strvec(path: str, v: Any, n: int) -> tuple[str, ...]:
    _null_check(path, v)
    if not isinstance(v, list) or len(v) != n:
        raise ParamsError(path, f"应为长度 {n} 的字符串数组")
    return tuple(_str(f"{path}[{i}]", item) for i, item in enumerate(v))


def _strictly_increasing(path: str, xs: NDArray, *, name: str) -> None:
    """标定表公共要求：节点严格递增，否则插值不唯一。"""
    if xs.size < 3:
        raise ParamsError(path, f"{name}至少需要 3 个标定点，实际 {xs.size}")
    if np.any(np.diff(xs) <= 0.0):
        raise ParamsError(path, f"{name}必须严格递增，实际为 {xs.tolist()}")


def _assert_rotation(path: str, r: FloatArray) -> None:
    """检查 3x3 矩阵确实是 SO(3)：正交且行列式为 +1。"""
    err = float(np.max(np.abs(r.T @ r - np.eye(3))))
    if err > ROTATION_ORTHO_TOL:
        raise ParamsError(path, f"不是正交矩阵，最大正交残差 {err:.3e} > {ROTATION_ORTHO_TOL:.1e}")
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > 1e-4:
        raise ParamsError(path, f"行列式应为 +1，实际 {det:.6f}")


# ---------------------------------------------------------------------------
# 第 3 节：分组解析
# ---------------------------------------------------------------------------


def _parse_model(path: str, payload: Any) -> ModelParams:
    d = _require_keys(payload, set(ModelParams.__dataclass_fields__), path)
    return ModelParams(
        urdf_path=_str(f"{path}.urdf_path", d["urdf_path"]),
        urdf_sha256=_str(f"{path}.urdf_sha256", d["urdf_sha256"]),
        motor_calibration_path=_str(f"{path}.motor_calibration_path", d["motor_calibration_path"]),
        motor_calibration_sha256=_str(f"{path}.motor_calibration_sha256", d["motor_calibration_sha256"]),
    )


def _parse_motor(path: str, payload: Any) -> MotorParams:
    d = _require_keys(payload, set(MotorParams.__dataclass_fields__), path)
    ids = _imat(f"{path}.ids", d["ids"], 6)
    # 电机身份与顺序核对（标定指南 2.1）：固定 [1..6]。
    if ids.tolist() != [1, 2, 3, 4, 5, 6]:
        raise ParamsError(f"{path}.ids", f"必须为 [1,2,3,4,5,6]，实际 {ids.tolist()}")
    res = _imat(f"{path}.position_resolution", d["position_resolution"], 6)
    if np.any(res <= 1):
        raise ParamsError(f"{path}.position_resolution", f"每项必须大于 1，实际 {res.tolist()}")
    cpm = _fvec(f"{path}.current_ma_per_raw", d["current_ma_per_raw"], (6,))
    if np.any(cpm == 0.0):
        # 比例为 0 会让所有电流读数都算成 0mA，接触判据与过流保护同时失效。
        raise ParamsError(f"{path}.current_ma_per_raw", f"每项必须非零，实际 {cpm.tolist()}")
    vpr = _fvec(f"{path}.velocity_deg_s_per_raw", d["velocity_deg_s_per_raw"], (6,))
    if np.any(vpr <= 0.0):
        raise ParamsError(f"{path}.velocity_deg_s_per_raw", f"每项必须为正，实际 {vpr.tolist()}")
    closed = _int(f"{path}.gripper_closed_raw", d["gripper_closed_raw"])
    open_ = _int(f"{path}.gripper_open_raw", d["gripper_open_raw"])
    # 0%=标定闭合端、100%=标定张开端，两端读数相同会让百分比换算除零。
    if closed == open_:
        raise ParamsError(
            f"{path}", f"gripper_open_raw({open_}) 与 gripper_closed_raw({closed}) 不能相同"
        )
    return MotorParams(
        port=_str(f"{path}.port", d["port"]),
        baudrate=_int(f"{path}.baudrate", d["baudrate"], minimum=1),
        ids=ids,
        models=_strvec(f"{path}.models", d["models"], 6),
        firmware=_strvec(f"{path}.firmware", d["firmware"], 6),
        position_resolution=res,
        current_ma_per_raw=cpm,
        current_zero_raw=_fvec(f"{path}.current_zero_raw", d["current_zero_raw"], (6,)),
        velocity_deg_s_per_raw=vpr,
        gripper_closed_raw=closed,
        gripper_open_raw=open_,
    )


def _parse_joints(path: str, payload: Any) -> JointsParams:
    d = _require_keys(payload, set(JointsParams.__dataclass_fields__), path)
    sign = _imat(f"{path}.sign", d["sign"], 5)
    if not np.all((sign == 1) | (sign == -1)):
        raise ParamsError(f"{path}.sign", f"每项只能是 ±1，实际 {sign.tolist()}")
    measured = _fvec(f"{path}.measured_limits_deg", d["measured_limits_deg"], (5, 2))
    application = _fvec(f"{path}.application_limits_deg", d["application_limits_deg"], (5, 2))
    margin = _fvec(f"{path}.margin_deg", d["margin_deg"], (5,))
    if np.any(margin <= 0.0):
        raise ParamsError(f"{path}.margin_deg", f"必须每项为正，实际 {margin.tolist()}")
    for name, lim in (("measured_limits_deg", measured), ("application_limits_deg", application)):
        if np.any(lim[:, 1] <= lim[:, 0]):
            raise ParamsError(f"{path}.{name}", "每行必须满足 lower < upper")
    return JointsParams(
        sign=sign,
        zero_offset_deg=_fvec(f"{path}.zero_offset_deg", d["zero_offset_deg"], (5,)),
        measured_limits_deg=measured,
        application_limits_deg=application,
        margin_deg=margin,
        hard_current_ma=_fvec(f"{path}.hard_current_ma", d["hard_current_ma"], (5,)),
    )


def _parse_tool(path: str, payload: Any) -> ToolParams:
    d = _require_keys(payload, set(ToolParams.__dataclass_fields__), path)
    rot = _fvec(f"{path}.rotation_e_tcp", d["rotation_e_tcp"], (3, 3))
    _assert_rotation(f"{path}.rotation_e_tcp", rot)

    samples_raw = d["translation_samples"]
    if not isinstance(samples_raw, list):
        raise ParamsError(f"{path}.translation_samples", "应为数组")
    samples: list[ToolTranslationSample] = []
    for i, item in enumerate(samples_raw):
        sp = f"{path}.translation_samples[{i}]"
        dd = _require_keys(item, {"gripper_pct", "xyz_m"}, sp)
        samples.append(
            ToolTranslationSample(
                gripper_pct=_float(f"{sp}.gripper_pct", dd["gripper_pct"]),
                xyz_m=_fvec(f"{sp}.xyz_m", dd["xyz_m"], (3,)),
            )
        )
    pcts = np.array([s.gripper_pct for s in samples], dtype=np.float64)
    _check_pct_domain(f"{path}.translation_samples", pcts)
    return ToolParams(
        rotation_e_tcp=rot,
        translation_samples=tuple(samples),
        position_error_bound_m=_float(
            f"{path}.position_error_bound_m", d["position_error_bound_m"], nonnegative=True
        ),
        orientation_error_bound_deg=_float(
            f"{path}.orientation_error_bound_deg", d["orientation_error_bound_deg"],
            nonnegative=True,
        ),
    )


def _check_pct_domain(path: str, pcts: NDArray) -> None:
    """三张表的 gripper_pct 都必须在 [0,100] 且严格递增（标定指南 2.4）。"""
    if np.any((pcts < 0.0) | (pcts > 100.0)):
        raise ParamsError(path, f"gripper_pct 必须落在 [0,100]，实际 {pcts.tolist()}")
    _strictly_increasing(path, pcts, name="gripper_pct ")


def _pct(path: str, v: Any) -> float:
    """夹爪开度百分比字段：类型列写的是 float，[0,100]（标定指南 2.4）。

    单独的 0~100 范围检查不能省：0%=标定闭合端、100%=标定张开端，越出这个区间
    的"开度"在总线上没有对应读数，只能靠裁剪掩盖，而 4.4 明确禁止裁剪。
    """
    f = _float(path, v)
    if not 0.0 <= f <= 100.0:
        raise ParamsError(path, f"开度百分比应落在 [0,100]，实际 {f}")
    return f


def _bounds(path: str, v: Any) -> FloatArray:
    """解析 [2,3] 的 AABB：逐轴必须 min <= max（允许厚度为 0 的壁）。

    用 LOAD_AXIS_SPAN_M 做严格包含判断，能挡住把 min/max 写反的输入，同时不
    禁止一维或二维退化的盒（例如把容器壁抽象成一个无限薄的平面）。
    """
    arr = _fvec(path, v, (2, 3))
    if np.any(arr[0] - arr[1] > LOAD_AXIS_SPAN_M):
        raise ParamsError(path, f"每轴需满足 min <= max，实际 {arr.tolist()}")
    return arr


def _check_within_limits(path: str, poses_deg: NDArray, lower: NDArray, upper: NDArray) -> None:
    """检查一批 (M,5) 关节姿态是否逐项落在有效限位内，越限直接报字段路径。"""
    for row, pose in enumerate(poses_deg):
        bad = np.where((pose < lower - LOAD_LIMIT_TOL_DEG) | (pose > upper + LOAD_LIMIT_TOL_DEG))[0]
        for idx in bad:
            raise ParamsError(
                f"{path}[{row}]" if poses_deg.shape[0] > 1 else path,
                f"{JOINT_NAMES[idx]} = {pose[idx]:.3f}deg 超出有效限位"
                f" [{lower[idx]:.3f}, {upper[idx]:.3f}]",
            )


def _parse_workspace(path: str, payload: Any, limit_lower: NDArray, limit_upper: NDArray) -> WorkspaceParams:
    """解析 workspace 组。

    limit_lower/limit_upper 是已经算好的有效关节限位：home_joints_deg 与
    safe_waypoints_deg 是示教出来的姿态，越限属于配置错误，必须在加载期就报
    CONFIG_INVALID，而不是等到规划阶段才发现走不到。
    """
    d = _require_keys(payload, set(WorkspaceParams.__dataclass_fields__), path)
    obs_raw = d["obstacles"]
    if not isinstance(obs_raw, list):
        raise ParamsError(f"{path}.obstacles", "应为数组（可以为空数组）")
    obstacles: list[Obstacle] = []
    seen: set[str] = set()
    for i, item in enumerate(obs_raw):
        sp = f"{path}.obstacles[{i}]"
        dd = _require_keys(item, {"id", "bounds_m"}, sp)
        oid = _str(f"{sp}.id", dd["id"])
        # 障碍 id 重复会让碰撞报告无法定位。
        if oid in seen:
            raise ParamsError(f"{sp}.id", f"障碍 id {oid!r} 重复")
        seen.add(oid)
        obstacles.append(Obstacle(id=oid, bounds_m=_bounds(f"{sp}.bounds_m", dd["bounds_m"])))

    home = _fvec(f"{path}.home_joints_deg", d["home_joints_deg"], (5,))
    waypoints = _fvec(f"{path}.safe_waypoints_deg", d["safe_waypoints_deg"], (None, 5))
    if waypoints.shape[0] < 1:
        raise ParamsError(f"{path}.safe_waypoints_deg", "至少需要 1 个中转位（K>=1）")
    _check_within_limits(f"{path}.home_joints_deg", home[None, :], limit_lower, limit_upper)
    _check_within_limits(f"{path}.safe_waypoints_deg", waypoints, limit_lower, limit_upper)
    return WorkspaceParams(
        target_bounds_m=_bounds(f"{path}.target_bounds_m", d["target_bounds_m"]),
        tcp_bounds_m=_bounds(f"{path}.tcp_bounds_m", d["tcp_bounds_m"]),
        table_z_m=_float(f"{path}.table_z_m", d["table_z_m"]),
        table_flatness_m=_float(f"{path}.table_flatness_m", d["table_flatness_m"], nonnegative=True),
        home_joints_deg=home,
        safe_waypoints_deg=waypoints,
        obstacles=tuple(obstacles),
    )


def _parse_collision(path: str, payload: Any) -> CollisionParams:
    d = _require_keys(payload, set(CollisionParams.__dataclass_fields__), path)
    caps_raw = d["link_capsules"]
    if not isinstance(caps_raw, list) or not caps_raw:
        raise ParamsError(f"{path}.link_capsules", "不能为空，必须至少一个胶囊体")
    capsules: list[LinkCapsule] = []
    cap_ids: set[str] = set()
    for i, item in enumerate(caps_raw):
        sp = f"{path}.link_capsules[{i}]"
        dd = _require_keys(item, {"id", "link", "p0_m", "p1_m", "radius_m"}, sp)
        cid = _str(f"{sp}.id", dd["id"])
        if cid in cap_ids:
            raise ParamsError(f"{sp}.id", f"capsule id {cid!r} 重复")
        cap_ids.add(cid)
        capsules.append(
            LinkCapsule(
                id=cid,
                link=_str(f"{sp}.link", dd["link"]),
                p0_m=_fvec(f"{sp}.p0_m", dd["p0_m"], (3,)),
                p1_m=_fvec(f"{sp}.p1_m", dd["p1_m"], (3,)),
                radius_m=_float(f"{sp}.radius_m", dd["radius_m"], positive=True),
            )
        )

    pairs_raw = d["ignore_self_pairs"]
    if not isinstance(pairs_raw, list):
        raise ParamsError(f"{path}.ignore_self_pairs", "应为数组（可以为空数组）")
    pairs: list[tuple[str, str]] = []
    for i, item in enumerate(pairs_raw):
        sp = f"{path}.ignore_self_pairs[{i}]"
        if not isinstance(item, list) or len(item) != 2:
            raise ParamsError(sp, "每项应为 [capsule_id_a, capsule_id_b]")
        a = _str(f"{sp}[0]", item[0])
        b = _str(f"{sp}[1]", item[1])
        # 豁免表引用不存在的 id 会让"人工核对每个接触豁免"（标定指南 7.3）失效。
        for which, val in (("a", a), ("b", b)):
            if val not in cap_ids:
                raise ParamsError(sp, f"{which} 端 {val!r} 不在 link_capsules 的 id 中")
        pairs.append((a, b))

    return CollisionParams(
        link_capsules=tuple(capsules),
        ignore_self_pairs=tuple(pairs),
        clearance_m=_float(f"{path}.clearance_m", d["clearance_m"], positive=True),
        max_joint_substep_deg=_float(
            f"{path}.max_joint_substep_deg", d["max_joint_substep_deg"], positive=True
        ),
    )


def _parse_places(path: str, payload: Any) -> dict[str, PlacePose]:
    """places：键为 place_id，值严格为 {position:[3], yaw_deg:float}。"""
    if not isinstance(payload, dict) or not payload:
        raise ParamsError(path, "必须是非空映射")
    if "default" not in payload:
        raise ParamsError(path, "必须包含 default")
    places: dict[str, PlacePose] = {}
    for pid, item in payload.items():
        sp = f"{path}.{pid}"
        dd = _require_keys(item, {"position", "yaw_deg"}, sp)
        pos = _fvec(f"{sp}.position", dd["position"], (3,))
        yaw = _float(f"{sp}.yaw_deg", dd["yaw_deg"])
        # 长轴不区分头尾，统一归一化到 [-90,90)（技术文档 3.2）。
        yaw = wrap180(yaw)
        places[str(pid)] = PlacePose(position=pos, yaw_deg=yaw)
    return places


def _parse_timing(path: str, payload: Any) -> TimingParams:
    d = _require_keys(payload, set(TimingParams.__dataclass_fields__), path)
    fps = _int(f"{path}.fps", d["fps"], equals=TIMING_FPS)
    tick = 1.0 / fps
    timing = TimingParams(
        fps=fps,
        read_timeout_s=_float(f"{path}.read_timeout_s", d["read_timeout_s"], positive=True),
        write_timeout_s=_float(f"{path}.write_timeout_s", d["write_timeout_s"], positive=True),
        max_feedback_age_s=_float(f"{path}.max_feedback_age_s", d["max_feedback_age_s"], positive=True),
        max_feedback_span_s=_float(f"{path}.max_feedback_span_s", d["max_feedback_span_s"], positive=True),
        max_tick_lateness_s=_float(f"{path}.max_tick_lateness_s", d["max_tick_lateness_s"], positive=True),
        stop_poll_s=_float(f"{path}.stop_poll_s", d["stop_poll_s"], positive=True),
        planning_poll_s=_float(f"{path}.planning_poll_s", d["planning_poll_s"], positive=True),
    )
    # 标定指南 2.3：max_tick_lateness_s < 1/FPS，stop_poll_s 不大于 1/FPS。
    if not timing.max_tick_lateness_s < tick:
        raise ParamsError(f"{path}.max_tick_lateness_s", f"必须小于一个 tick（{tick:.6f}s）")
    if timing.stop_poll_s > tick:
        raise ParamsError(f"{path}.stop_poll_s", f"不得大于一个 tick（{tick:.6f}s）")
    return timing


def _parse_motion(path: str, payload: Any) -> MotionConstraints:
    d = _require_keys(payload, set(MotionConstraints.__dataclass_fields__), path)
    following = _fvec(f"{path}.following_error_deg", d["following_error_deg"], (5,))
    following_hard = _fvec(f"{path}.following_error_hard_deg", d["following_error_hard_deg"], (5,))
    # 立即停止阈值必须严于持续阈值，否则 hard 永远不会先触发（标定指南 2.3）。
    if np.any(following_hard <= following):
        raise ParamsError(
            f"{path}.following_error_hard_deg", "逐项必须大于 following_error_deg"
        )
    dwell = _float(f"{path}.settle_dwell_s", d["settle_dwell_s"], positive=True)
    timeout = _float(f"{path}.settle_timeout_s", d["settle_timeout_s"], positive=True)
    if not timeout > dwell:
        raise ParamsError(f"{path}.settle_timeout_s", f"必须大于 settle_dwell_s（{dwell}）")
    return MotionConstraints(
        max_velocity_deg_s=_fvec(f"{path}.max_velocity_deg_s", d["max_velocity_deg_s"], (5,)),
        max_acceleration_deg_s2=_fvec(
            f"{path}.max_acceleration_deg_s2", d["max_acceleration_deg_s2"], (5,)
        ),
        max_command_step_deg=_fvec(f"{path}.max_command_step_deg", d["max_command_step_deg"], (5,)),
        tcp_max_velocity_m_s=_float(f"{path}.tcp_max_velocity_m_s", d["tcp_max_velocity_m_s"], positive=True),
        tcp_max_acceleration_m_s2=_float(
            f"{path}.tcp_max_acceleration_m_s2", d["tcp_max_acceleration_m_s2"], positive=True
        ),
        start_position_tol_deg=_fvec(
            f"{path}.start_position_tol_deg", d["start_position_tol_deg"], (5,)
        ),
        following_error_deg=following,
        following_error_hard_deg=following_hard,
        following_error_dwell_s=_float(
            f"{path}.following_error_dwell_s", d["following_error_dwell_s"], positive=True
        ),
        settle_position_tol_deg=_fvec(
            f"{path}.settle_position_tol_deg", d["settle_position_tol_deg"], (5,)
        ),
        settle_velocity_tol_deg_s=_fvec(
            f"{path}.settle_velocity_tol_deg_s", d["settle_velocity_tol_deg_s"], (5,)
        ),
        settle_dwell_s=dwell,
        settle_timeout_s=timeout,
    )


def _parse_ik(path: str, payload: Any) -> IkParams:
    d = _require_keys(payload, set(IkParams.__dataclass_fields__), path)
    # 文档 2.3 允许 K=0（还没有经过验证的备用初值）。空列表的 np.asarray 形状是
    # (0,)，要按最后一维的宽度补成 (0,5)，否则会被 shape 检查误判成格式错误。
    seeds_raw = d["seed_joints_deg"]
    if isinstance(seeds_raw, list) and len(seeds_raw) == 0:
        seeds_raw = np.zeros((0, 5))
    seeds = _fvec(f"{path}.seed_joints_deg", seeds_raw, (None, 5))
    return IkParams(
        position_tol_m=_float(f"{path}.position_tol_m", d["position_tol_m"], positive=True),
        tilt_tol_deg=_float(f"{path}.tilt_tol_deg", d["tilt_tol_deg"], positive=True),
        yaw_tol_deg=_float(f"{path}.yaw_tol_deg", d["yaw_tol_deg"], positive=True),
        position_weight=_float(f"{path}.position_weight", d["position_weight"], positive=True),
        orientation_weight=_float(f"{path}.orientation_weight", d["orientation_weight"], positive=True),
        max_iterations=_int(f"{path}.max_iterations", d["max_iterations"], minimum=1),
        stagnation_iterations=_int(f"{path}.stagnation_iterations", d["stagnation_iterations"], minimum=1),
        max_solve_s=_float(f"{path}.max_solve_s", d["max_solve_s"], positive=True),
        min_progress=_float(f"{path}.min_progress", d["min_progress"], positive=True),
        seed_joints_deg=seeds,
    )


def _parse_grasp(path: str, payload: Any) -> GraspParams:
    d = _require_keys(payload, set(GraspParams.__dataclass_fields__), path)
    return GraspParams(
        center_offset_object_m=_fvec(
            f"{path}.center_offset_object_m", d["center_offset_object_m"], (3,)
        ),
        yaw_offset_deg=_float(f"{path}.yaw_offset_deg", d["yaw_offset_deg"]),
        object_envelope_m=_fvec(f"{path}.object_envelope_m", d["object_envelope_m"], (3,)),
        target_position_error_bound_m=_float(
            f"{path}.target_position_error_bound_m", d["target_position_error_bound_m"],
            nonnegative=True,
        ),
        target_yaw_error_bound_deg=_float(
            f"{path}.target_yaw_error_bound_deg", d["target_yaw_error_bound_deg"],
            nonnegative=True,
        ),
        object_shift_bound_m=_float(f"{path}.object_shift_bound_m", d["object_shift_bound_m"],
                                    nonnegative=True),
        approach_height_m=_float(f"{path}.approach_height_m", d["approach_height_m"], positive=True),
        lift_height_m=_float(f"{path}.lift_height_m", d["lift_height_m"], positive=True),
        retreat_height_m=_float(f"{path}.retreat_height_m", d["retreat_height_m"], positive=True),
    )


def _parse_gripper(path: str, payload: Any) -> GripperParams:
    d = _require_keys(payload, set(GripperParams.__dataclass_fields__), path)

    gap_samples = _parse_pair_table(
        f"{path}.gap_table", d["gap_table"], ("gripper_pct", "gap_m"), GapSample
    )
    angle_samples = _parse_pair_table(
        f"{path}.angle_table", d["angle_table"], ("gripper_pct", "angle_deg"), AngleSample
    )

    gap_pcts = np.array([s.gripper_pct for s in gap_samples], dtype=np.float64)
    gaps = np.array([s.gap_m for s in gap_samples], dtype=np.float64)
    ang_pcts = np.array([s.gripper_pct for s in angle_samples], dtype=np.float64)
    angles = np.array([s.angle_deg for s in angle_samples], dtype=np.float64)
    _check_pct_domain(f"{path}.gap_table", gap_pcts)
    _check_pct_domain(f"{path}.angle_table", ang_pcts)
    # gap 非负且严格递增；angle 可增可减但全表方向一致（标定指南 2.4）。
    if np.any(gaps < 0.0):
        raise ParamsError(f"{path}.gap_table", "gap_m 必须非负")
    if np.any(np.diff(gaps) <= 0.0):
        raise ParamsError(f"{path}.gap_table", "gap_m 必须严格递增")
    _check_consistent_direction(f"{path}.angle_table", angles)

    contact_gap = _fvec(f"{path}.contact_gap_range_m", d["contact_gap_range_m"], (2,))
    if not (0.0 < contact_gap[0] < contact_gap[1]):
        raise ParamsError(f"{path}.contact_gap_range_m", f"需满足 0 < min < max，实际 {contact_gap.tolist()}")
    contact_i = _float(f"{path}.contact_current_ma", d["contact_current_ma"], positive=True)
    hard_i = _float(f"{path}.hard_current_ma", d["hard_current_ma"], positive=True)
    if not hard_i > contact_i:
        raise ParamsError(f"{path}.hard_current_ma", f"必须大于 contact_current_ma（{contact_i}）")

    return GripperParams(
        gap_table=tuple(gap_samples),
        angle_table=tuple(angle_samples),
        preopen_pct=_pct(f"{path}.preopen_pct", d["preopen_pct"]),
        release_pct=_pct(f"{path}.release_pct", d["release_pct"]),
        open_speed_pct_s=_float(f"{path}.open_speed_pct_s", d["open_speed_pct_s"], positive=True),
        close_speed_pct_s=_float(f"{path}.close_speed_pct_s", d["close_speed_pct_s"], positive=True),
        contact_gap_range_m=contact_gap,
        contact_current_ma=contact_i,
        hard_current_ma=hard_i,
        window_samples=_int(f"{path}.window_samples", d["window_samples"], minimum=1),
        contact_dwell_s=_float(f"{path}.contact_dwell_s", d["contact_dwell_s"], positive=True),
        empty_closed_pct=_pct(f"{path}.empty_closed_pct", d["empty_closed_pct"]),
        empty_tol_pct=_float(f"{path}.empty_tol_pct", d["empty_tol_pct"], positive=True),
        close_timeout_s=_float(f"{path}.close_timeout_s", d["close_timeout_s"], positive=True),
        open_timeout_s=_float(f"{path}.open_timeout_s", d["open_timeout_s"], positive=True),
        hold_dwell_s=_float(f"{path}.hold_dwell_s", d["hold_dwell_s"], positive=True),
        release_dwell_s=_float(f"{path}.release_dwell_s", d["release_dwell_s"], positive=True),
        slip_opening_change_pct=_float(
            f"{path}.slip_opening_change_pct", d["slip_opening_change_pct"], positive=True
        ),
        settle_tol_pct=_float(f"{path}.settle_tol_pct", d["settle_tol_pct"], positive=True),
        settle_speed_pct_s=_float(f"{path}.settle_speed_pct_s", d["settle_speed_pct_s"], positive=True),
    )


def _parse_pair_table(path: str, payload: Any, keys: tuple[str, str], cls: type) -> list:
    """把 [{k0:.., k1:..}, ...] 形式的标定表逐项转成 dataclass。"""
    if not isinstance(payload, list):
        raise ParamsError(path, "应为数组")
    out = []
    for i, item in enumerate(payload):
        sp = f"{path}[{i}]"
        dd = _require_keys(item, set(keys), sp)
        out.append(cls(**{keys[0]: _float(f"{sp}.{keys[0]}", dd[keys[0]]),
                          keys[1]: _float(f"{sp}.{keys[1]}", dd[keys[1]])}))
    return out


def _check_consistent_direction(path: str, ys: NDArray) -> None:
    """angle_table 允许整体递增或整体递减，但不允许中途变号。"""
    diffs = np.diff(ys)
    if np.any(diffs == 0.0):
        raise ParamsError(path, "angle_deg 相邻节点不能相等（表退化）")
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ParamsError(path, "angle_deg 必须全表同向单调")


def _parse_acceptance(path: str, payload: Any) -> AcceptanceParams:
    d = _require_keys(payload, set(AcceptanceParams.__dataclass_fields__), path)

    def rate(key: str) -> float:
        v = _float(f"{path}.{key}", d[key])
        if not 0.0 <= v <= 1.0:
            raise ParamsError(f"{path}.{key}", f"应落在 [0,1]，实际 {v}")
        return v

    return AcceptanceParams(
        max_position_error_m=_float(f"{path}.max_position_error_m", d["max_position_error_m"], positive=True),
        max_tilt_error_deg=_float(f"{path}.max_tilt_error_deg", d["max_tilt_error_deg"], positive=True),
        max_yaw_error_deg=_float(f"{path}.max_yaw_error_deg", d["max_yaw_error_deg"], positive=True),
        max_stop_latency_s=_float(f"{path}.max_stop_latency_s", d["max_stop_latency_s"], positive=True),
        min_grasp_success_rate=rate("min_grasp_success_rate"),
        max_drop_rate=rate("max_drop_rate"),
        max_damage_rate=rate("max_damage_rate"),
        damage_observe_minutes=_float(
            f"{path}.damage_observe_minutes", d["damage_observe_minutes"], positive=True
        ),
        min_grasp_trials=_int(f"{path}.min_grasp_trials", d["min_grasp_trials"], minimum=1),
    )


def _parse_verification(path: str, payload: Any) -> VerificationParams | None:
    """verification 在 draft/simulation 里允许整体缺省为 null。"""
    if payload is None:
        return None
    d = _require_keys(payload, set(VerificationParams.__dataclass_fields__), path)
    return VerificationParams(
        report_path=_str(f"{path}.report_path", d["report_path"]),
        report_sha256=_str(f"{path}.report_sha256", d["report_sha256"]),
    )


# ---------------------------------------------------------------------------
# 第 4 节：URDF 限位读取与跨字段一致性检查
# ---------------------------------------------------------------------------


def read_urdf_joint_limits_deg(urdf_path: Path, joint_names: tuple[str, ...]) -> FloatArray:
    """从 URDF 读出各关节 limit 并把 rad 转成 deg —— 只在这里转换一次。

    技术文档 5.1 要求 "URDF rad 转 deg 一次"；B/C 共用同一份 deg 结果，
    不允许在别处再乘一次 180/pi。
    """
    try:
        text = urdf_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParamsError(str(urdf_path), f"无法读取：{exc}") from exc
    # 标准库 ET 的默认解析器会展开 DTD 实体。URDF 本该是纯数据模型，
    # 这里直接拒绝任何 DTD，既挡住 billion laughs 类攻击，也挡住外部实体。
    head = text[:4096]
    if "<!DOCTYPE" in head or "<!ENTITY" in head:
        raise ParamsError("model.urdf_path", "URDF 不允许包含 DTD/ENTITY 声明")
    parser = ET.XMLParser(target=ET.TreeBuilder())
    try:
        root = ET.fromstring(text, parser=parser)
    except ET.ParseError as exc:
        raise ParamsError("model.urdf_path", f"无法解析 URDF：{exc}") from exc
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        lim = joint.find("limit")
        if name is None or lim is None:
            continue
        lower = lim.get("lower")
        upper = lim.get("upper")
        if lower is None or upper is None:
            continue
        # URDF 的 limit 本来就是 rad，这里原样存下来，只在返回前转换一次。
        limits[name] = (float(lower), float(upper))
    out = np.empty((len(joint_names), 2), dtype=np.float64)
    for i, name in enumerate(joint_names):
        if name not in limits:
            raise ParamsError("model.urdf_path", f"URDF 里找不到关节 {name!r} 的 limit")
        lo, up = limits[name]
        out[i] = (math.degrees(lo), math.degrees(up))
    return out


def _check_gripper_table_coverage(path_prefix: str, params: "MotionParams") -> None:
    """三张表共同覆盖区间必须包含所有会实际使用的开度（标定指南 2.4 末段）。

    这里不用 kinematics_ext 的运行时插值函数，避免 configs 反向依赖
    qingyun；数学是同一件事：先求接触 gap 区间对应的开度区间，再做
    区间包含判断。
    """
    tool_pcts = np.array([s.gripper_pct for s in params.tool.translation_samples])
    gap_pcts = np.array([s.gripper_pct for s in params.gripper.gap_table])
    ang_pcts = np.array([s.gripper_pct for s in params.gripper.angle_table])
    lo = max(tool_pcts[0], gap_pcts[0], ang_pcts[0])
    hi = min(tool_pcts[-1], gap_pcts[-1], ang_pcts[-1])
    if lo >= hi:
        raise ParamsError(
            "gripper", f"TCP/gap/angle 三张表没有共同覆盖区间（交集 [{lo}, {hi}]）"
        )

    g = params.gripper
    # 接触 gap 区间换算成开度区间：gap_table 的 gap 严格递增，所以反查唯一。
    # 越出表的定义域必须当场拒绝——np.interp 会把越界值静默钳位成端点，
    # 让"接触区间超出标定覆盖"这种配置错误伪装成合法开度（P3-8）。
    gaps = np.array([s.gap_m for s in g.gap_table])
    lo_g, hi_g = (float(v) for v in g.contact_gap_range_m)
    if not (float(gaps[0]) - 1e-12 <= lo_g and hi_g <= float(gaps[-1]) + 1e-12):
        raise ParamsError(
            "gripper.contact_gap_range_m",
            f"接触区间 [{lo_g:.6f}, {hi_g:.6f}]m 越出 gap_table 定义域 "
            f"[{float(gaps[0]):.6f}, {float(gaps[-1]):.6f}]m",
        )
    contact_lo = float(np.interp(lo_g, gaps, gap_pcts))
    contact_hi = float(np.interp(hi_g, gaps, gap_pcts))
    required = [
        (f"{path_prefix}.gripper.preopen_pct", g.preopen_pct),
        (f"{path_prefix}.gripper.release_pct", g.release_pct),
        (f"{path_prefix}.gripper.empty_closed_pct", g.empty_closed_pct),
        ("gripper.contact_gap_range_m(下界对应开度)", contact_lo),
        ("gripper.contact_gap_range_m(上界对应开度)", contact_hi),
    ]
    for name, pct in required:
        if not (lo - 1e-9 <= pct <= hi + 1e-9):
            raise ParamsError(
                name, f"开度 {pct:.3f} 超出三张表共同覆盖区间 [{lo:.3f}, {hi:.3f}]"
            )
    # 接触开度必须明显大于空闭合端，否则无法区分"抓空"和"抓到物体"。
    if not contact_lo > g.empty_closed_pct + g.empty_tol_pct:
        raise ParamsError(
            "gripper.contact_gap_range_m",
            f"接触开度下限 {contact_lo:.3f} 必须大于 empty_closed_pct+empty_tol_pct"
            f" = {g.empty_closed_pct + g.empty_tol_pct:.3f}",
        )


def _check_limit_intersection(urdf_limits_deg: FloatArray, joints: JointsParams) -> tuple[FloatArray, FloatArray]:
    """URDF ∩ 实测 ∩ 应用（各减 margin）不得为空，返回算出的有效限位。

    与 safety 使用同一个 effective_joint_limits 实现，这里只做加载期预检，
    让"空交集"在拿不到 MotionSegment 之前就报 CONFIG_INVALID。返回结果还给
    workspace 组复用，避免同一件事在配置文件里出现第二份真值。
    """
    lower, upper = effective_joint_limits(urdf_limits_deg, joints)
    empty = np.where(lower > upper)[0]
    if empty.size:
        names = ", ".join(JOINT_NAMES[i] for i in empty)
        raise ParamsError("joints", f"有效限位交集为空（{names}）；检查 margin_deg 是否过大")
    return lower, upper


# ---------------------------------------------------------------------------
# 第 5 节：标定指南第 12 节要求的共享换算函数
#
# 有效限位与关节/夹爪单位换算被标定工具和运动控制共用。单位换算放在
# motor_control.py（C 的职责），有效限位是 B/C 共用的几何约束，因此和
# URDF 限位一样放在这里，保证只有一份实现。
# ---------------------------------------------------------------------------


def effective_joint_limits(urdf_limits_deg: FloatArray, joints_params: JointsParams) -> tuple[FloatArray, FloatArray]:
    """逐关节求 URDF/实测/应用三重交集并扣掉 margin。

    技术文档 5.1 给出的公式，原样实现：

        lower = max(urdf_lower, measured_lower, application_lower) + margin
        upper = min(urdf_upper, measured_upper, application_upper) - margin

    返回 (lower, upper)，单位 deg，形状均为 (5,)。交集可能为空（lower>upper），
    调用方必须检查结果；本模块在加载时已预检一次。
    """
    urdf = np.asarray(urdf_limits_deg, dtype=np.float64)
    measured = np.asarray(joints_params.measured_limits_deg, dtype=np.float64)
    application = np.asarray(joints_params.application_limits_deg, dtype=np.float64)
    margin = np.asarray(joints_params.margin_deg, dtype=np.float64)
    lower = np.maximum.reduce([urdf[:, 0], measured[:, 0], application[:, 0]]) + margin
    upper = np.minimum.reduce([urdf[:, 1], measured[:, 1], application[:, 1]]) - margin
    return lower, upper


# ---------------------------------------------------------------------------
# 第 6 节：角度归一化与参数哈希
# ---------------------------------------------------------------------------


def wrap180(angle_deg: float) -> float:
    """把长轴角归一化到 [-90,90)：长轴不区分头尾，所以周期是 180°。"""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def parameter_sha256(profile: Mapping[str, Any]) -> str:
    """对除 status、verification 之外的全部字段做排序键、紧凑 JSON 后求 SHA256。

    定义来自标定指南 1.3：验收报告用这个哈希把"被验收的参数"和"实际加载的
    参数"绑定起来，任何一项改动都会让 real 模式加载失败。
    """
    subset = {k: v for k, v in profile.items() if k not in ("status", "verification")}
    blob = json.dumps(subset, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 第 7 节：加载入口
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """流式计算文件哈希，避免把整份 mesh 读进内存。"""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
    except OSError as exc:
        raise ParamsError(str(path), f"无法读取：{exc}") from exc
    return h.hexdigest()


def load_motion_params(profile_path: Path, *, mode: Literal["real", "mock"]) -> MotionParams:
    """加载并校验 profile.json。

    mode="real"  ：要求 status=verified、运行期字段全部测完、资源文件哈希与
                   验收报告匹配，任何一项不满足都拒绝启动（标定指南 1.1）。
    mode="mock"  ：用于仿真与单元测试，要求 status=simulation，跳过哈希与
                   报告匹配，但结构、shape、单调性、限位交集等全部照查。
    """
    profile_path = Path(profile_path)
    if not profile_path.is_file():
        raise ParamsError(str(profile_path), "profile.json 不存在")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ParamsError:
        raise
    except json.JSONDecodeError as exc:
        raise ParamsError(str(profile_path), f"不是合法 JSON：{exc}") from exc

    expected_top = {
        "schema_version", "profile_id", "robot_id", "status", "model", "motor", "joints",
        "tool", "workspace", "collision", "places", "timing", "motion", "ik", "grasp",
        "gripper", "acceptance", "verification",
    }
    d = _require_keys(raw, expected_top, str(profile_path))

    status = _str("status", d["status"])
    if status not in STATUSES:
        raise ParamsError("status", f"只能是 {STATUSES}，实际 {status!r}")

    # 加载模式与配置状态必须配套：draft 还没测完，real/mock 都不接受。
    if mode == "real" and status != "verified":
        raise ParamsError("status", f"mode=real 要求 status=verified，实际 {status!r}")
    if mode == "mock" and status != "simulation":
        raise ParamsError("status", f"mode=mock 要求独立配置 status=simulation，实际 {status!r}")

    model = _parse_model("model", d["model"])
    collision = _parse_collision("collision", d["collision"])
    base_dir = profile_path.parent

    # --- 资源文件存在性与哈希 ---
    urdf_file = _resolve(base_dir, model.urdf_path)
    if not urdf_file.is_file():
        raise ParamsError("model.urdf_path", f"文件不存在：{urdf_file}")
    urdf_limits = read_urdf_joint_limits_deg(urdf_file, JOINT_NAMES)
    if mode == "real":
        actual = _sha256_file(urdf_file)
        if actual != model.urdf_sha256:
            raise ParamsError("model.urdf_sha256", f"与实算哈希不一致：{actual}")
        calib_file = _resolve(base_dir, model.motor_calibration_path)
        if not calib_file.is_file():
            raise ParamsError("model.motor_calibration_path", f"文件不存在：{calib_file}")
        actual = _sha256_file(calib_file)
        if actual != model.motor_calibration_sha256:
            raise ParamsError("model.motor_calibration_sha256", f"与实算哈希不一致：{actual}")

    # 关节组必须先解出来：它是 workspace/places 越限检查的基准。
    joints = _parse_joints("joints", d["joints"])
    limit_lower, limit_upper = _check_limit_intersection(urdf_limits, joints)
    workspace = _parse_workspace("workspace", d["workspace"], limit_lower, limit_upper)

    params = MotionParams(
        schema_version=_int("schema_version", d["schema_version"], equals=SCHEMA_VERSION),
        profile_id=_str("profile_id", d["profile_id"]),
        robot_id=_str("robot_id", d["robot_id"]),
        status=status,
        model=model,
        motor=_parse_motor("motor", d["motor"]),
        joints=joints,
        tool=_parse_tool("tool", d["tool"]),
        workspace=workspace,
        collision=collision,
        places=_parse_places("places", d["places"]),
        timing=_parse_timing("timing", d["timing"]),
        motion=_parse_motion("motion", d["motion"]),
        ik=_parse_ik("ik", d["ik"]),
        grasp=_parse_grasp("grasp", d["grasp"]),
        gripper=_parse_gripper("gripper", d["gripper"]),
        acceptance=_parse_acceptance("acceptance", d["acceptance"]),
        verification=_parse_verification("verification", d["verification"]),
        profile_dir=base_dir,
        urdf_limits_deg=urdf_limits,
    )

    # --- 跨字段一致性 ---
    _check_gripper_table_coverage("", params)

    if mode == "real":
        _check_report_binding(profile_path, params, raw)
    return params


def _check_report_binding(profile_path: Path, params: MotionParams, raw: Mapping[str, Any]) -> None:
    """real 模式：确认加载的参数确实就是被验收过的那一份。

    链路是 profile.verification -> 报告文件 -> 报告里的 parameter_sha256 ->
    当前 profile 除 status/verification 外的全部字段。任何一环断裂都不允许
    驱动真机。
    """
    if params.verification is None:
        raise ParamsError("verification", "mode=real 必须提供 promote 写入的验收报告信息")
    report_path = _resolve(params.profile_dir, params.verification.report_path)
    if not report_path.is_file():
        raise ParamsError("verification.report_path", f"报告不存在：{report_path}")
    actual_report_hash = _sha256_file(report_path)
    if actual_report_hash != params.verification.report_sha256:
        raise ParamsError("verification.report_sha256", f"报告哈希不一致：{actual_report_hash}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParamsError("verification.report_path", f"报告不是合法 JSON：{exc}") from exc
    if report.get("profile_id") != params.profile_id:
        raise ParamsError(
            "verification.report_path",
            f"报告 profile_id={report.get('profile_id')!r} 与配置 {params.profile_id!r} 不符",
        )
    if not report.get("overall_pass"):
        raise ParamsError("verification.report_path", "报告 overall_pass 不为 true，禁止实机运行")
    if not report.get("operator_reviewed"):
        raise ParamsError("verification.report_path", "报告缺少人工复核标记 operator_reviewed")
    # P2-6：证据等级关卡。报告里的每项 check 都带 evidence_type；某个阶段的
    # 全部检查都只有 mock/offline 证据时，这份"verified"从未被真机数据支撑，
    # 绝不允许驱动实机（指南 1.3 的 physical/mock 区分就是为了这一刻，不能只
    # 靠 promote 时的一次终端提示）。哈希链路保证报告与参数不能单方面篡改。
    stages = report.get("stages") or {}
    for stage, info in sorted(stages.items()):
        checks = (info or {}).get("checks") or []
        if not any(c.get("evidence_type") == "physical" for c in checks):
            raise ParamsError(
                "verification.report_path",
                f"阶段 {stage} 的验收报告没有任何 physical 证据（全是 mock/offline），"
                "纯仿真证据的配置禁止用于实机运行",
            )
    expect = parameter_sha256(raw)
    if report.get("parameter_sha256") != expect:
        raise ParamsError(
            "verification.report_sha256",
            f"报告针对的参数哈希 {report.get('parameter_sha256')!r} 与当前配置 {expect!r} 不同，"
            "说明配置在验收后被改动过",
        )


# ---------------------------------------------------------------------------
# 第 7.1 节：标定采集入口（独立于运行期 status 关卡）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationProfile:
    """标定采集阶段驱动总线所需的最小参数集（标定指南 3.3/4.1）。

    与 load_motion_params 的区别：采集发生在 motor→…→pick_place 逐阶段推进的
    过程中，profile 恒为 draft、大量运行期字段还是 null，real/mock 链路一律
    拒绝 null，所以那条链不能用于采集（P1-5 的结构性死锁）。本入口只要求
    "读总线"所需的组确实已填：

      * motor 组：端口、波特率、身份/分辨率、官方校准文件（换算层的行程中点）；
        电流零点与 mA/deg_s 比例尚未测完时以 NaN 登记——换算函数对非有限输入
        会立即抛错，绝不用猜测值参与采集数据。
      * joints 组：允许尚未拟合（None）。tool/workcell 之后的 q_urdf 反算才
        需要它，此时 joints 阶段的报告已经保证它被填完。
      * 不核对 urdf/校准文件哈希：标定包里这两份文件本来就在被逐阶段更新。
    """

    motor: MotorParams
    joints: JointsParams | None
    urdf_limits_deg: FloatArray
    motor_calibration: Mapping[str, Mapping[str, int]]
    profile_dir: Path


def _has_null(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, Mapping):
        return any(_has_null(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_null(v) for v in obj)
    return False


def _parse_motor_for_calibration(path: str, payload: Any) -> MotorParams:
    """motor 组里"采集必需"的字段严格解析；电流/速度换算三件套允许缺测。"""
    d = _require_keys(payload, set(MotorParams.__dataclass_fields__), path)
    ids = _imat(f"{path}.ids", d["ids"], 6)
    if ids.tolist() != [1, 2, 3, 4, 5, 6]:
        raise ParamsError(f"{path}.ids", f"必须为 [1,2,3,4,5,6]，实际 {ids.tolist()}")
    res = _imat(f"{path}.position_resolution", d["position_resolution"], 6)
    if np.any(res <= 1):
        raise ParamsError(f"{path}.position_resolution", f"每项必须大于 1，实际 {res.tolist()}")
    port = _str(f"{path}.port", d["port"]) if d["port"] is not None else ""
    if not port.strip():
        raise ParamsError(
            f"{path}.port",
            "实机采集需要真实串口；请先由操作者把官方工具确认过的端口写入配置"
            "（不用猜测值自动填补）",
        )
    # 夹爪两端读数在 draft 期同样是 null（它们正是 motor 阶段要测的东西），
    # 未测时保持 None，让采集层跳过开度换算；已填时照常核对两端不重合。
    closed = d["gripper_closed_raw"]
    opened = d["gripper_open_raw"]
    if closed is not None or opened is not None:
        closed = _int(f"{path}.gripper_closed_raw", closed)
        opened = _int(f"{path}.gripper_open_raw", opened)
        if closed == opened:
            raise ParamsError(f"{path}", f"夹爪两端读数相同（{closed}），无法换算开度")

    def _scale_vec(key: str, shape: tuple[int, ...]) -> FloatArray:
        if d[key] is None or _has_null(d[key]):
            return np.full(shape, np.nan)
        return _fvec(f"{path}.{key}", d[key], shape)

    return MotorParams(
        port=port,
        baudrate=_int(f"{path}.baudrate", d["baudrate"], minimum=1),
        ids=ids,
        models=_strvec(f"{path}.models", d["models"], 6),
        firmware=_strvec(f"{path}.firmware", d["firmware"], 6),
        position_resolution=res,
        current_ma_per_raw=_scale_vec("current_ma_per_raw", (6,)),
        current_zero_raw=_scale_vec("current_zero_raw", (6,)),
        velocity_deg_s_per_raw=_scale_vec("velocity_deg_s_per_raw", (6,)),
        gripper_closed_raw=closed,
        gripper_open_raw=opened,
    )


def load_calibration_profile(profile_path: Path) -> CalibrationProfile:
    """标定采集入口：读 draft/verified 的真机配置里采集所需的最小字段集。

    只接受 status ∈ {draft, verified}：simulation 配置没有真实串口，指南 3.3
    的采集对象就是实机，拿模拟配置走 --hardware 路径一定是操作者搞错了包。
    """
    profile_path = Path(profile_path)
    if not profile_path.is_file():
        raise ParamsError(str(profile_path), "profile.json 不存在")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ParamsError:
        raise
    except json.JSONDecodeError as exc:
        raise ParamsError(str(profile_path), f"不是合法 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ParamsError(str(profile_path), "profile.json 应为对象")

    status = raw.get("status")
    if status not in ("draft", "verified"):
        raise ParamsError(
            "status",
            f"实机采集要求 draft 或 verified 的真机配置，实际 {status!r}"
            "（simulation 配置不能用于 --hardware 链路）",
        )
    # 只取两个资源路径：哈希字段在 draft 期还是 null（它们由 motor 阶段拟合
    # 写入），而采集入口本来就不核对哈希（见 CalibrationProfile docstring）。
    model = raw["model"]
    urdf_rel = _str("model.urdf_path", model["urdf_path"])
    calib_rel = _str("model.motor_calibration_path", model["motor_calibration_path"])
    base_dir = profile_path.parent
    urdf_file = _resolve(base_dir, urdf_rel)
    if not urdf_file.is_file():
        raise ParamsError("model.urdf_path", f"文件不存在：{urdf_file}")
    urdf_limits = read_urdf_joint_limits_deg(urdf_file, JOINT_NAMES)
    calib_file = _resolve(base_dir, calib_rel)
    # 延迟导入：load_motor_calibration 住在 motor_control，而 motor_control
    # 导入本模块，顶层引用会形成循环。
    from qingyun.grabbing.motor_control import load_motor_calibration

    try:
        calibration = load_motor_calibration(calib_file)
    except ValueError as exc:
        raise ParamsError("model.motor_calibration_path", str(exc)) from exc
    motor = _parse_motor_for_calibration("motor", raw["motor"])
    joints_raw = raw["joints"]
    joints = None if _has_null(joints_raw) else _parse_joints("joints", joints_raw)
    return CalibrationProfile(
        motor=motor,
        joints=joints,
        urdf_limits_deg=urdf_limits,
        motor_calibration=calibration,
        profile_dir=base_dir,
    )
