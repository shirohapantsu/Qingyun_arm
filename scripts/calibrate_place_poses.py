#!/usr/bin/env python3
"""落点标定工具（P4 §2.5）：把示教的 TCP 释放位姿换算为物体中心 ``PlacePose``、
按 §2.5 判据校验，并在**原 profile 的副本**上产出 **draft 候选 places**——
通过前绝不覆盖已发布 profile，输入文件从头到尾不被改写。

固定 CLI（P4 §3.1「最小 CLI 统一提供 --input --output --report」惯例；本工具是纯输入
输出运动几何工具，不连接驱动，因此不涉及运动脚本"连接前确认"那一支）：

    calibrate_place_poses.py --profile PATH --input PATH --output PATH --report PATH

* ``--profile``：draft/verified 完整加载，走 ``load_calibration_motion_params``
  （拒绝 simulation、拒绝任何 null/未测物理字段），失败即退出、绝不触碰硬件。
* ``--input``：示教记录 JSON（复用 ``read_workcell_waypoint.py`` 读反馈的做法；每条
  trial 含 ``place_id`` 与释放时 TCP 位姿）。
* ``--output``：候选 profile——原 profile 的**副本**上把 ``strawberry_A_01..N``（有序）、
  ``strawberry_B_bin``、``strawberry_C_bin`` 替换/新增；``default``/``bin`` 等旧条目原样
  保留、绝不改名挪用。输出候选恒为 ``status=draft``、``verification=null``（未复核的
  place 候选按定义不是已发布参数）。**输入 profile 文件永不被写**。
* ``--report``：结构化 JSON 报告（资源哈希、每条换算前后与闭合残差、判据逐项结果、
  待实测清单）。判据全部通过才写 ``--output``；任一不过 → 非零退出且**不产出候选**
  （只留报告），对齐 §2.5「通过前不覆盖已发布 profile」。

输入 JSON 结构（明确格式）::

    {
      "schema_version": 1,
      "kind": "place_pose_teaching",
      "holding_reference": {                       # 建立名义持物变换 T_TCP_O（§3.4）
         "method": "vision_center_at_grasp | independent_gauge | ...",   # B 类证据说明
         "grasp_object_center_m": [x, y, z],        # 抓取时物体中心 T_B_O_grasp 位置
         "grasp_object_yaw_deg":  a,                # 抓取时物体长轴 T_B_O_grasp yaw
         "grasp_close_tcp":       <pose>            # 闭爪 TCP  T_B_TCP_close
      },
      "trials": [
         {
            "place_id": "strawberry_A_01",
            "release_tcp": <pose>,                  # 释放时 TCP 位姿（物体仍在爪内、未张开）
            "holding": <holding_reference>,          # 可选：本条自带持物关系，否则继承顶层
            "object_observation": "可选：物体中心观测方式说明"
         }, ...
      ]
    }

其中 ``<pose>`` 为释放/闭爪 TCP 的两种来源之一（§2.5「关节角读回或 TCP xyz+yaw 自报」）::

    {"mode": "tcp",    "xyz_m": [x, y, z], "yaw_deg": a}                 # 顶抓 TCP 自报
    {"mode": "joints", "q_urdf_deg": [5], "gripper_pct": g}              # 关节角读回 -> FK

换算（§2.5 核心，方向与运动文档 §3.4 一致，复用同一语义而非另推公式）：
``T_TCP_O = holding_transform(T_B_TCP_close, object_pose(grasp_center, grasp_yaw))``；
放置物体中心是 §3.4 正向关系 ``place_tcp_pose(T_B_O_place, T_TCP_O)`` 的**逆**：

    T_B_O_place = T_B_TCP_release @ T_TCP_O
    position    = T_B_O_place[:3, 3]
    yaw         = wrap180(atan2(R[1,0], R[0,0]))        # 长轴角，归一化 [-90,90)

即 ``place_tcp_pose(object_pose(position, yaw), T_TCP_O)`` 复算回释放 TCP——测试双向钉住。

夹爪闭合方向与长轴关系：抓取时 ``spin = 物体 yaw + grasp.yaw_offset_deg``（§3.3），
故 ``T_TCP_O`` 的旋转 = ``R_TOP_DOWN @ rz(-yaw_offset)`` 已把该固定偏角吃进刚体关系里；
反算物体 yaw 时对释放 TCP 顶抓方位角做同样还原（物体 yaw = 释放 spin − yaw_offset 的
模 180 结果），因此换算与 ``yaw_offset ∈ {0, 90}`` 的既有语义自洽（P4-T03 双向覆盖）。

退出码：``0`` 全部判据通过并产出候选；``2`` 连接/输入级错误（坏 profile、缺/坏输入、
未知 place、记录字段非法、输出即输入 profile）；``3`` §2.5 判据不通过（报告已写、
候选未产出）。缺 CLI 参数或未知参数由 argparse 以非零码（2）+ 明确用法信息退出。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.motion_params import (  # noqa: E402
    ParamsError,
    load_calibration_motion_params,
    parameter_sha256,
    wrap180,
)
from qingyun.grabbing import kinematics_ext as K  # noqa: E402

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_CRITERIA = 3

TOOL_NAME = "calibrate_place_poses"
INPUT_SCHEMA_VERSION = 1
_A_RE = re.compile(r"strawberry_A_(\d+)$")

# §2.5 B 类现场判据：工具不伪造，只在报告里列为"待实测"。
PENDING_B_C = (
    "箱口开度足以容纳当前物体（B 类·现场）",
    "释放高度：口上方到已堆物的净空（B 类·现场）",
    "撤回竖直路线不碰箱壁与已堆物（B 类·损伤控制）",
    "逐点实测一次释放验证（B 类·现场）",
)
PENDING_A = (
    "A 槽位独立逐点释放验证一次（B 类·现场）",
    "A 槽位清空为落点标定操作前提（P4 §3.4 前提）",
)


class InputError(RuntimeError):
    """连接/输入级错误 → 退出码 2（早于任何换算，绝不触碰硬件、绝不写文件）。"""


class CriteriaFailure(RuntimeError):
    """保留给未来：判据本身不以异常传播，只在 run() 汇总为非零退出。"""


# ---------------------------------------------------------------------------
# 0. 通用小工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """流式实算文件 SHA256（与标定链其余工具同一口径）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class _ModelBox:
    """延迟构造 ``ArmModel``：只有 joints 模式（关节角读回 → FK）才付 placo/URDF 成本。"""

    def __init__(self, params) -> None:
        self._params = params
        self._model = None

    def get(self):
        if self._model is None:
            self._model = K.ArmModel(self._params)
        return self._model


def _vec3(where: str, value: Any) -> np.ndarray:
    if value is None or _has_null(value):
        raise InputError(f"{where}: 缺值（null）")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise InputError(f"{where}: 应为长度 3 的数组，实际 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{where}: 含非有限值 {arr.tolist()}")
    return arr


def _scalar(where: str, value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise InputError(f"{where}: 应为有限数值，实际 {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{where}: 无法解析为数值：{exc}") from exc
    if not math.isfinite(f):
        raise InputError(f"{where}: 非有限数值 {f}")
    return f


def _has_null(obj: Any) -> bool:
    if obj is None:
        return True
    if isinstance(obj, dict):
        return any(_has_null(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_null(v) for v in obj)
    return False


# ---------------------------------------------------------------------------
# 1. TCP 位姿解析（§2.5：关节角读回 或 TCP xyz+yaw 自报）
# ---------------------------------------------------------------------------


def resolve_tcp_pose(spec: Any, params, model_box: _ModelBox, where: str) -> np.ndarray:
    """把 <pose> 规格解成 4x4 的 T_B_TCP，与运动链同一语义。

    * ``mode=tcp``：位置即 TCP、姿态为顶抓 → ``K.top_down_pose(xyz, yaw)``（姿态文档 §2）。
    * ``mode=joints``：位置由 ``FK_E(q) @ T_E_TCP(g)`` 得到 → ``ArmModel.fk_tcp``，
      开度 g 用闭爪保持开度（技术文档 3.3/5.3：闭爪后按实际开度算 TCP）。
    """
    if not isinstance(spec, dict):
        raise InputError(f"{where}: <pose> 应为对象，实际 {type(spec).__name__}")
    mode = spec.get("mode")
    if mode == "tcp":
        xyz = _vec3(f"{where}.xyz_m", spec.get("xyz_m"))
        yaw = _scalar(f"{where}.yaw_deg", spec.get("yaw_deg"))
        return K.top_down_pose(xyz, yaw)
    if mode == "joints":
        q = spec.get("q_urdf_deg")
        if q is None or _has_null(q):
            raise InputError(f"{where}.q_urdf_deg: 缺值（null）")
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (5,):
            raise InputError(f"{where}.q_urdf_deg: 应为长度 5，实际 {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise InputError(f"{where}.q_urdf_deg: 含非有限值")
        g = _scalar(f"{where}.gripper_pct", spec.get("gripper_pct"))
        try:
            return model_box.get().fk_tcp(arr, g)
        except K.MotionError as exc:   # 开度越出工具表覆盖区间等——如实记为输入错误
            raise InputError(f"{where}: FK 失败：{exc}") from exc
    raise InputError(f"{where}.mode: 只能是 'tcp' 或 'joints'，实际 {mode!r}")


# ---------------------------------------------------------------------------
# 2. 持物关系 T_TCP_O（技术文档 §3.4；复用 holding_transform/object_pose 同一语义）
# ---------------------------------------------------------------------------


def resolve_holding(hold: Any, params, model_box: _ModelBox, where: str) -> tuple[np.ndarray, dict]:
    """由抓取的"物体中心 + 闭爪 TCP"建立名义持物变换 T_TCP_O（§3.4）。

    返回 (T_TCP_O, 记录用 meta)。meta 只含换算输入的可读快照，不含任何伪造测量结论。
    """
    if not isinstance(hold, dict):
        raise InputError(f"{where}: holding_reference 应为对象")
    center = _vec3(f"{where}.grasp_object_center_m", hold.get("grasp_object_center_m"))
    yaw = _scalar(f"{where}.grasp_object_yaw_deg", hold.get("grasp_object_yaw_deg"))
    T_B_O_grasp = K.object_pose(center, yaw)
    T_B_TCP_close = resolve_tcp_pose(hold.get("grasp_close_tcp"), params, model_box,
                                     f"{where}.grasp_close_tcp")
    T_TCP_O = K.holding_transform(T_B_TCP_close, T_B_O_grasp)
    meta = {
        "method": hold.get("method"),
        "grasp_object_center_m": center.tolist(),
        "grasp_object_yaw_deg": yaw,
        "yaw_offset_deg_applied": float(params.grasp.yaw_offset_deg),
    }
    return T_TCP_O, meta


# ---------------------------------------------------------------------------
# 3. TCP → 物体中心换算（§3.4 正向关系的逆；复用同一语义，不另推公式）
# ---------------------------------------------------------------------------


def tcp_to_object_place(T_B_TCP_release: np.ndarray,
                        T_TCP_O: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """返回 (position, yaw, T_B_O_release)。

    ``T_B_O_release = T_B_TCP_release @ T_TCP_O`` 是 ``place_tcp_pose``（=
    ``T_B_O_place @ inv(T_TCP_O)``）的逆。yaw 取物体长轴角并 wrap180 归一化。
    """
    T_B_O = np.asarray(T_B_TCP_release, dtype=np.float64) @ np.asarray(T_TCP_O, dtype=np.float64)
    position = T_B_O[:3, 3].copy()
    yaw = wrap180(math.degrees(math.atan2(float(T_B_O[1, 0]), float(T_B_O[0, 0]))))
    return position, yaw, T_B_O


def closure_residual_m(T_B_TCP_release: np.ndarray, T_TCP_O: np.ndarray,
                       position: np.ndarray, yaw: float) -> float:
    """用运动模块同款正向 ``place_tcp_pose(object_pose(pos,yaw), T_TCP_O)`` 复算回 TCP，
    与示教释放 TCP 的最大逐元素差（米）。顶抓一致时≈0；残差来自示教姿态的额外倾角。"""
    T_back = K.place_tcp_pose(K.object_pose(position, yaw), T_TCP_O)
    return float(np.max(np.abs(T_back - np.asarray(T_B_TCP_release, dtype=np.float64))))


# ---------------------------------------------------------------------------
# 4. §2.5 判据几何
# ---------------------------------------------------------------------------


def inside_closed_bounds(point: np.ndarray, bounds: np.ndarray) -> bool:
    """逐轴闭区间 [lo, hi] 判定：与 P3 视觉过滤同一区间语义（in-bounds=保留）。"""
    lo = np.asarray(bounds, dtype=np.float64)[0]
    hi = np.asarray(bounds, dtype=np.float64)[1]
    p = np.asarray(point, dtype=np.float64)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _footprint_axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """O 系 +X=长轴代表方向、+Y=短轴（与 safety.object_box 的 rz_deg(yaw) 定义一致）。"""
    a = math.radians(float(yaw_deg))
    long_axis = np.array([math.cos(a), math.sin(a)])
    short_axis = np.array([-math.sin(a), math.cos(a)])
    return long_axis, short_axis


def net_distance_2d(ci: np.ndarray, yi: float, cj: np.ndarray, yj: float,
                    length_m: float, width_m: float) -> float:
    """P3 §6.4 同款二维净距：r=(L|dot(long,u)|+W|dot(short,u)|)/2，d=dist-ri-rj。

    中心重合/重叠（距离≈0，u 无定义）按净距不足处理（返回 -inf，令其必判不通过）。
    """
    ci2 = np.asarray(ci[:2], dtype=np.float64)
    cj2 = np.asarray(cj[:2], dtype=np.float64)
    delta = cj2 - ci2
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-12:
        return float("-inf")
    u = delta / dist
    li, si = _footprint_axes(yi)
    lj, sj = _footprint_axes(yj)
    ri = (length_m * abs(float(np.dot(li, u))) + width_m * abs(float(np.dot(si, u)))) / 2.0
    rj = (length_m * abs(float(np.dot(lj, u))) + width_m * abs(float(np.dot(sj, u)))) / 2.0
    return dist - ri - rj


def classify_place(pid: str) -> tuple[str, int | None]:
    """按 place_id 归类别：返回 ('A', 序号) / ('B', None) / ('C', None)；未知则 InputError。"""
    m = _A_RE.match(pid)
    if m:
        return "A", int(m.group(1))
    if pid == "strawberry_B_bin":
        return "B", None
    if pid == "strawberry_C_bin":
        return "C", None
    raise InputError(
        f"未知 place_id {pid!r}：仅接受 strawberry_A_<NN>、strawberry_B_bin、strawberry_C_bin")


# ---------------------------------------------------------------------------
# 5. 输入记录加载
# ---------------------------------------------------------------------------


def load_input_records(input_path: Path) -> dict:
    if not input_path.is_file():
        raise InputError(f"输入示教记录不存在：{input_path}")
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"输入示教记录不是合法 JSON：{input_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("输入示教记录顶层应为对象")
    if data.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise InputError(
            f"输入 schema_version 应为 {INPUT_SCHEMA_VERSION}，实际 {data.get('schema_version')!r}")
    if data.get("kind") != "place_pose_teaching":
        raise InputError(f"输入 kind 应为 'place_pose_teaching'，实际 {data.get('kind')!r}")
    trials = data.get("trials")
    if not isinstance(trials, list) or not trials:
        raise InputError("输入 trials 必须是非空数组")
    if not isinstance(data.get("holding_reference"), dict) and not any(
            isinstance(t, dict) and isinstance(t.get("holding"), dict) for t in trials):
        raise InputError("缺少 holding_reference，且没有任何自带 holding 的 trial —— 无法建立持物关系")
    return data


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


def evaluate(records: dict, params, *, verbose: bool = True) -> dict:
    """把示教记录换算为候选 places、逐项校验 §2.5 判据，返回结构化结果（含 pass 标志）。

    本函数不写文件、不抛判据异常：所有失败都体现在返回结果的 checks 与 overall_pass 上，
    交由 run() 决定退出码与是否产出候选。输入级错误（结构非法）在到达这里前已抛 InputError。
    """
    model_box = _ModelBox(params)
    bounds = np.asarray(params.workspace.target_bounds_m, dtype=np.float64)
    clearance_m = float(params.collision.clearance_m)
    envelope = np.asarray(params.grasp.object_envelope_m, dtype=np.float64)
    length_m, width_m = float(envelope[0]), float(envelope[1])
    global_holding = records.get("holding_reference")

    trials = records["trials"]
    seen: dict[str, int] = {}
    entries: list[dict] = []           # 每条候选（含换算结果与逐条判据）
    pair_results: list[dict] = []      # A 槽位二维逐对净距

    # --- 6.1 逐条换算 + 记录结构校验 ---
    for idx, trial in enumerate(trials):
        where = f"trials[{idx}]"
        if not isinstance(trial, dict):
            raise InputError(f"{where}: 应为对象")
        pid = trial.get("place_id")
        if not isinstance(pid, str) or not pid.strip():
            raise InputError(f"{where}.place_id: 应为非空字符串")
        pid = pid.strip()
        category, order = classify_place(pid)         # 未知 ID → InputError（早于换算）
        if pid in seen:
            raise InputError(f"{where}.place_id {pid!r} 与 trials[{seen[pid]}] 重复")
        seen[pid] = idx

        hold = trial.get("holding", global_holding)
        T_TCP_O, hold_meta = resolve_holding(hold, params, model_box, f"{where}.holding")
        T_release = resolve_tcp_pose(trial.get("release_tcp"), params, model_box,
                                     f"{where}.release_tcp")
        position, yaw, T_B_O = tcp_to_object_place(T_release, T_TCP_O)
        residual = closure_residual_m(T_release, T_TCP_O, position, yaw)

        entries.append({
            "place_id": pid,
            "category": category,
            "a_order": order,
            "object_observation": trial.get("object_observation"),
            "holding": hold_meta,
            "release_tcp": _pose_snapshot(trial.get("release_tcp")),
            "converted_position_m": [float(v) for v in position],
            "converted_yaw_deg": float(yaw),
            "closure_residual_m": residual,
        })

    # --- 6.2 判据 c：中心/姿态有限、yaw ∈ [-90,90) ---
    finite_fail = []
    for e in entries:
        pos = np.asarray(e["converted_position_m"], dtype=np.float64)
        yv = e["converted_yaw_deg"]
        if not (np.all(np.isfinite(pos)) and math.isfinite(yv)):
            finite_fail.append(e["place_id"])
        elif not (-90.0 <= yv < 90.0):
            finite_fail.append(e["place_id"])
    checks_c = {
        "id": "c_finite_yaw_range",
        "desc": "中心/姿态有限、yaw 归一化并落在 [-90,90)（wrap180）",
        "passed": not finite_fail,
        "failed_places": finite_fail,
    }

    # --- 6.3 判据 a：所有新 places 中心在 target_bounds 之外（逐轴闭区间）---
    roi_fail = []
    for e in entries:
        pos = np.asarray(e["converted_position_m"], dtype=np.float64)
        e["inside_target_bounds"] = inside_closed_bounds(pos, bounds)
        if e["inside_target_bounds"]:
            roi_fail.append(e["place_id"])
    checks_a = {
        "id": "a_outside_target_bounds",
        "desc": "所有新落点中心在 workspace.target_bounds_m 之外（§2.5：现 bin 中心在 ROI 内须重标）",
        "target_bounds_m": bounds.tolist(),
        "passed": not roi_fail,
        "failed_places": roi_fail,
    }

    # --- 6.4 判据 b：A 槽位二维逐对净距 >= collision.clearance_m ---
    a_entries = sorted([e for e in entries if e["category"] == "A"], key=lambda x: x["a_order"])
    clearance_fail_pairs = []
    for i in range(len(a_entries)):
        for j in range(i + 1, len(a_entries)):
            ei, ej = a_entries[i], a_entries[j]
            d = net_distance_2d(np.asarray(ei["converted_position_m"]), ei["converted_yaw_deg"],
                                np.asarray(ej["converted_position_m"]), ej["converted_yaw_deg"],
                                length_m, width_m)
            ok = d >= clearance_m - 1e-12
            pr = {
                "pair": [ei["place_id"], ej["place_id"]],
                "net_distance_m": None if math.isinf(d) else round(d, 9),
                "clearance_m": clearance_m,
                "envelope_L_m": length_m,
                "envelope_W_m": width_m,
                "passed": ok,
            }
            pair_results.append(pr)
            if not ok:
                clearance_fail_pairs.append((ei["place_id"], ej["place_id"]))
    checks_b = {
        "id": "b_A_pairwise_clearance",
        "desc": "A 槽位按各自 yaw 构造 object_envelope 水平矩形，二维逐对净距 >= "
                "collision.clearance_m（闭区间：d==clearance 判通过；不用标量代替）",
        "a_slot_count": len(a_entries),
        "pair_count": len(pair_results),
        "passed": not clearance_fail_pairs,
        "failed_pairs": [list(p) for p in clearance_fail_pairs],
    }

    # --- 6.5 判据 d：default/bin 旧条目保留、不被改名挪用 ---
    preserved = sorted(k for k in params.places if k not in seen)
    reused = [k for k in params.places if k in seen]   # 新 ID 恰与旧 key 同名（不应发生）
    checks_d = {
        "id": "d_legacy_entries_preserved",
        "desc": "default/bin 旧条目原样保留、绝不改名/挪用为新 ID；其处置留给实测决定",
        "preserved_legacy_ids": preserved,
        "legacy_reused_as_new_id": reused,
        "passed": not reused,
        "note": "旧条目是否删除/重标由实测（B 类）决定，本工具不自动充当新 ID。",
    }

    # --- 6.6 判据 e：B/C 只做形状与 ROI 外检查（ROI/有限已并入 a/c），其余列待实测 ---
    b_c_ids = [e["place_id"] for e in entries if e["category"] in ("B", "C")]
    checks_e = {
        "id": "e_B_C_shape_and_roi_only",
        "desc": "B/C 箱口释放式堆放点仅做形状/有限与 ROI 外检查；"
                "箱口开度/释放高度/撤回路线=现场 B 类，不伪造通过",
        "b_c_present": b_c_ids,
        "passed": True,   # 形状/ROI/有限性若失败已在 a/c 记为不过；此处不另设伪造门槛
        "pending_field_items": PENDING_B_C if b_c_ids else [],
    }

    checks = [checks_c, checks_a, checks_b, checks_d, checks_e]
    overall_pass = all(c["passed"] for c in checks)

    pending = list(PENDING_A) if a_entries else []
    pending += PENDING_B_C if b_c_ids else []

    return {
        "overall_pass": overall_pass,
        "entries": entries,
        "pairwise": pair_results,
        "checks": checks,
        "pending_field_verification": pending,
        "params_ref": {
            "target_bounds_m": bounds.tolist(),
            "object_envelope_m": envelope.tolist(),
            "clearance_m": clearance_m,
            "yaw_offset_deg": float(params.grasp.yaw_offset_deg),
        },
    }


def _pose_snapshot(spec: Any) -> dict:
    """把 <pose> 规格原样收进报告（换算前输入），非 dict 时记原值。"""
    if isinstance(spec, dict):
        out = {}
        for k, v in spec.items():
            out[k] = np.asarray(v, dtype=float).tolist() if isinstance(v, (list, tuple)) else v
        return out
    return {"raw": spec}


def build_candidate_profile(raw_profile: dict, entries: Sequence[dict]) -> dict:
    """在原 profile 的**深拷贝**上替换/新增 strawberry_* places；旧条目原样保留。

    输出候选恒为 draft、verification=null：未复核的 place 候选按定义不是已发布参数，
    这样它也绝不可能冒充已发布 verified（§2.5 通过前不覆盖已发布 profile）。
    """
    candidate = copy.deepcopy(raw_profile)
    places = candidate.get("places", {}) if isinstance(candidate.get("places"), dict) else {}
    # 先移除本工具将重写的 strawberry_* 旧值，保留其余（default/bin/...）原顺序。
    preserved = {k: v for k, v in places.items()
                 if not (k.startswith("strawberry_A_") or k in ("strawberry_B_bin", "strawberry_C_bin"))}
    a_entries = sorted([e for e in entries if e["category"] == "A"], key=lambda x: x["a_order"])
    other_entries = [e for e in entries if e["category"] != "A"]
    new_places: dict[str, dict] = dict(preserved)
    for e in list(a_entries) + other_entries:
        new_places[e["place_id"]] = {
            "position": [float(v) for v in e["converted_position_m"]],
            "yaw_deg": float(e["converted_yaw_deg"]),
        }
    candidate["places"] = new_places
    candidate["status"] = "draft"
    candidate["verification"] = None
    return candidate


def build_report(records_path: Path, profile_path: Path, output_path: Path,
                 raw_profile: dict, result: dict, *,
                 candidate_written: bool, candidate: dict | None) -> dict:
    per_place = []
    for e in result["entries"]:
        per_place.append({
            "place_id": e["place_id"],
            "category": e["category"],
            "holding": e["holding"],
            "before_release_tcp": e["release_tcp"],
            "object_observation": e["object_observation"],
            "after_position_m": e["converted_position_m"],
            "after_yaw_deg": e["converted_yaw_deg"],
            "closure_residual_m": round(e["closure_residual_m"], 9),
            "inside_target_bounds": e.get("inside_target_bounds"),
        })
    report = {
        "tool": TOOL_NAME,
        "spec": "P4 §2.5",
        "inputs": {
            "profile_path": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "records_path": str(records_path),
            "records_sha256": sha256_file(records_path),
        },
        "output": {
            "candidate_path": str(output_path),
            "candidate_written": candidate_written,
            "candidate_sha256": (sha256_file(output_path) if candidate_written
                                 else "not_written_criteria_failed"),
            "candidate_parameter_sha256": (parameter_sha256(candidate) if candidate else None),
            "candidate_status": "draft",
            "policy": "全部判据通过才产出候选；否则仅出报告、不写候选、非零退出（不覆盖已发布 profile）。",
        },
        "resource_sha256": _resource_hashes(raw_profile),
        "reference_params": result["params_ref"],
        "places": per_place,
        "pairwise_A": result["pairwise"],
        "criteria": result["checks"],
        "pending_field_verification": result["pending_field_verification"],
        "overall_pass": result["overall_pass"],
    }
    return report


def _resource_hashes(raw_profile: dict) -> dict:
    """从 profile 引用的 URDF/电机校准路径尽力实算哈希（相对 profile 目录解析）。"""
    model = raw_profile.get("model", {}) if isinstance(raw_profile.get("model"), dict) else {}
    base = Path(_profile_dir_hint or ".")
    out: dict[str, Any] = {}
    for key, field in (("urdf", "urdf_path"), ("motor_calibration", "motor_calibration_path")):
        rel = model.get(field)
        if not isinstance(rel, str):
            out[key] = "unavailable"
            continue
        p = Path(rel)
        p = p if p.is_absolute() else (base / p)
        try:
            out[key] = sha256_file(p) if p.is_file() else "missing"
        except OSError:
            out[key] = "unreadable"
    return out


# argparse 无法把 profile 目录透传给 _resource_hashes，用模块级桥接（单线程脚本）。
_profile_dir_hint: str | None = None


def run(argv: Sequence[str] | None = None) -> int:
    global _profile_dir_hint
    parser = _build_parser()
    args = parser.parse_args(argv)

    profile_path = Path(args.profile)
    records_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    # --- 1. 连接/输入级校验（任一失败 → 退出码 2，绝不写文件）---
    try:
        if not profile_path.is_file():
            raise InputError(f"profile 不存在：{profile_path}")
        if output_path.resolve() == profile_path.resolve():
            raise InputError("--output 不得等于 --profile（禁止改写输入 profile 源）")
        try:
            raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"profile 不是合法 JSON：{exc}") from exc
        _profile_dir_hint = str(profile_path.parent)
        try:
            # 完整 draft/verified 加载：拒绝 simulation、拒绝 null/未测物理字段（§2.8.1）。
            params = load_calibration_motion_params(profile_path)
        except ParamsError as exc:
            raise InputError(f"profile 未测完整/不可加载（load_calibration_motion_params 拒绝）：{exc}") from exc
        records = load_input_records(records_path)
    except InputError as exc:
        print(f"[{TOOL_NAME}][输入错误] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    # --- 2. 换算 + 判据 ---
    try:
        result = evaluate(records, params)
    except InputError as exc:   # 换算中暴露的记录结构问题（如未知 place/FK 越界）
        print(f"[{TOOL_NAME}][输入错误] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    candidate = build_candidate_profile(raw_profile, result["entries"]) if result["overall_pass"] else None

    # --- 3. 写候选（仅当全过）+ 写报告（总是）---
    try:
        if candidate is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        report = build_report(records_path, profile_path, output_path, raw_profile, result,
                              candidate_written=candidate is not None, candidate=candidate)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    except OSError as exc:
        print(f"[{TOOL_NAME}][写盘失败] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    failed = [c["id"] for c in result["checks"] if not c["passed"]]
    if result["overall_pass"]:
        print(f"[{TOOL_NAME}] §2.5 全部判据通过；候选 places 已写入 {output_path}；报告 {report_path}",
              file=sys.stderr, flush=True)
        return EXIT_OK
    print(f"[{TOOL_NAME}][判据不通过] 失败项 {failed}；未产出候选（见 {report_path}）",
          file=sys.stderr, flush=True)
    return EXIT_CRITERIA


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{TOOL_NAME}.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--profile", required=True, type=Path,
                    help="draft/verified profile.json（完整加载经 load_calibration_motion_params）")
    ap.add_argument("--input", required=True, type=Path,
                    help="示教记录 JSON（TCP 释放位姿 + 持物参考）")
    ap.add_argument("--output", required=True, type=Path,
                    help="候选 profile 输出路径（原 profile 的副本，绝不写回输入）")
    ap.add_argument("--report", required=True, type=Path,
                    help="结构化 JSON 报告输出路径")
    return ap


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
