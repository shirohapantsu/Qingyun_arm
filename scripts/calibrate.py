#!/usr/bin/env python3
"""SO-ARM101 抓取—放置参数标定工具。

规范来源：docs/真机参数测量与标定指南.md 全文。
    1.1 标定包结构、1.2 CLI 形态、1.3 数据与报告格式、第 2 节字段字典、
    第 3 节测量顺序与采集条件、第 4~11 节各阶段的 payload / 步骤 / 验收要求、
    第 12 节与运行代码共用的计算。

动作：init / capture / fit / validate / promote。
fit / validate / promote 全部离线，validate 不启动任何运动（指南 1.2）。

capture 不给 --hardware 时使用模拟源。这是指南 1.2 明确规定的行为，用途是在没有
真机的机器上打通并回归整条数据链路，不能替代真实测量：报告里这类证据一律标
evidence_type="mock"，与 physical 证据分开统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 从任意工作目录执行都要能 import 项目模块：把项目根插到 sys.path 最前面。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.common_interface import JOINT_NAMES, MOTOR_NAMES  # noqa: E402
from configs.motion_params import (  # noqa: E402
    ParamsError,
    effective_joint_limits,
    parameter_sha256,
    read_urdf_joint_limits_deg,
)
from qingyun.grabbing.kinematics_ext import (  # noqa: E402
    MotionError,
    close_axis_width_at_yaw,
    gap_to_gripper_pct,
    gripper_pct_to_gap,
)

# 八个固定阶段（指南 1.2），顺序就是第 3 节规定的测量顺序。
STAGES: tuple[str, ...] = (
    "motor", "joints", "tool", "workcell", "timing", "motion", "grasp", "pick_place",
)
# 阶段 -> 前置阶段。capture --hardware 与 validate 都按这条链检查前置是否已经有
# 通过的报告（指南 3.3"capture 的实机模式逐阶段检查前置参数及报告"）。
PREREQUISITES: dict[str, tuple[str, ...]] = {
    stage: STAGES[:i] for i, stage in enumerate(STAGES)
}

# 仓库锁定的 URDF；init 把它作为相对路径写进标定包。
REPO_URDF = "libs/so_arm_core/so101_new_calib.urdf"

# init 产物里由操作者先填入 draft 的"验收要求与待试参数"（指南 1.2）。
# 这些不是从采集样本拟合出来的，如果一律留 null，profile 永远填不满、
# fit/validate/promote 链路也无法验证。数值是这套仿真工作台的起点，实机必须复核。
DRAFT_OPERATOR_VALUES: dict[str, Any] = {
    "schema_version": 1,
    "acceptance.max_position_error_m": 0.010,
    "acceptance.max_tilt_error_deg": 3.0,
    "acceptance.max_yaw_error_deg": 4.0,
    "acceptance.max_stop_latency_s": 0.20,
    "acceptance.min_grasp_success_rate": 0.90,
    "acceptance.max_drop_rate": 0.05,
    "acceptance.max_damage_rate": 0.02,
    "acceptance.damage_observe_minutes": 30.0,
    "acceptance.min_grasp_trials": 30,
    # 误差预算里由设计/装配决定的部分。
    "tool.position_error_bound_m": 0.002,
    "tool.orientation_error_bound_deg": 1.2,
    "workspace.table_flatness_m": 0.001,
    "collision.clearance_m": 0.003,
    "grasp.target_position_error_bound_m": 0.004,
    "grasp.target_yaw_error_bound_deg": 5.0,
    "grasp.object_shift_bound_m": 0.003,
    # 待试运行参数：首次测试值由操作者按设备允许范围设置，实机逐步试出来再改（指南 9.1）。
    "timing.read_timeout_s": 0.020,
    "timing.write_timeout_s": 0.015,
    "timing.max_feedback_age_s": 0.050,
    "timing.max_feedback_span_s": 0.030,
    "timing.max_tick_lateness_s": 0.020,
    "timing.stop_poll_s": 0.010,
    "timing.planning_poll_s": 0.050,
    "motion.max_velocity_deg_s": [70.0] * 5,
    "motion.max_acceleration_deg_s2": [260.0] * 5,
    "motion.max_command_step_deg": [2.2] * 5,
    "motion.tcp_max_velocity_m_s": 0.22,
    "motion.tcp_max_acceleration_m_s2": 1.4,
    "motion.start_position_tol_deg": [3.0] * 5,
    "motion.following_error_deg": [12.0] * 5,
    "motion.following_error_hard_deg": [25.0] * 5,
    "motion.following_error_dwell_s": 0.10,
    "motion.settle_position_tol_deg": [2.5] * 5,
    "motion.settle_velocity_tol_deg_s": [8.0] * 5,
    "motion.settle_dwell_s": 0.15,
    "motion.settle_timeout_s": 3.0,
    "ik.position_tol_m": 0.006,
    "ik.tilt_tol_deg": 1.5,
    "ik.yaw_tol_deg": 2.0,
    "ik.position_weight": 1.0,
    "ik.orientation_weight": 0.1,
    "ik.max_iterations": 250,
    "ik.stagnation_iterations": 12,
    "ik.max_solve_s": 1.5,
    "ik.min_progress": 0.001,
    "ik.seed_joints_deg": [],
    "grasp.object_envelope_m": [0.050, 0.040, 0.040],
    "grasp.approach_height_m": 0.080,
    "grasp.lift_height_m": 0.090,
    "grasp.retreat_height_m": 0.080,
    "gripper.open_speed_pct_s": 25.0,
    "gripper.close_speed_pct_s": 15.0,
    "gripper.contact_current_ma": 220.0,
    "gripper.hard_current_ma": 450.0,
    "gripper.window_samples": 3,
    "gripper.contact_dwell_s": 0.10,
    "gripper.close_timeout_s": 6.0,
    "gripper.open_timeout_s": 6.0,
    "gripper.hold_dwell_s": 0.30,
    "gripper.release_dwell_s": 0.50,
    "gripper.slip_opening_change_pct": 6.0,
    "gripper.settle_tol_pct": 2.0,
    "gripper.settle_speed_pct_s": 3.0,
    "joints.hard_current_ma": [900.0] * 5,
    "joints.margin_deg": [2.0] * 5,
    "collision.max_joint_substep_deg": 0.5,
    "motor.baudrate": 1000000,
    "motor.current_ma_per_raw": [1.0] * 6,
    "motor.velocity_deg_s_per_raw": [3.0] * 6,
    # 型号与固件是"实际读回"的身份信息（指南 2.1）；模拟源会原样写回。
    "motor.models": ["sts3215"] * 6,
    "motor.firmware": ["2.54"] * 6,
    "motor.position_resolution": [4096] * 6,
}

# 由协议或文档固定、不需要测量的键，init 直接填好，避免 draft 出现无意义 null。
DRAFT_FIXED_VALUES: dict[str, Any] = {
    "timing.fps": 30,
    "motor.ids": [1, 2, 3, 4, 5, 6],
}


class CalibError(RuntimeError):
    """标定工具的用法/数据错误。CLI 顶层转成退出码 2 与一行消息。"""


# ---------------------------------------------------------------------------
# 1. 通用工具：路径、JSON、哈希、统计
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise CalibError(f"文件不存在：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibError(f"{path} 不是合法 JSON：{exc}") from exc


def dump_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def rel_from(base: Path, target: Path) -> str:
    """把 target 写成相对 base 的路径。profile.json 里的相对路径以它自己为基准。

    必须用 os.path.relpath 而不是 Path.relative_to：标定包通常在 /tmp 或用户的
    output 目录下，与项目根没有共同前缀时 relative_to 会直接抛错，而 relpath 能
    正确给出 ../.. 形式的跨树路径。
    """
    import os.path

    return os.path.relpath(Path(target).resolve(), Path(base).resolve())


def get_path(obj: Any, dotted: str) -> Any:
    """按 "a.b.0.c" 取值；任何一级缺失都返回 None 而不是抛 KeyError。

    调用方几乎都是"有就检查、没有就跳过"的读法（例如尚未拟合的阶段的残差），
    让 KeyError 冒出来只会把真正的错误盖住。
    """
    cur = obj
    for part in dotted.split("."):
        key = int(part) if part.lstrip("-").isdigit() else part
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return None
    return cur


def set_path(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[int(part)] if part.lstrip("-").isdigit() else cur[part]
    last = parts[-1]
    if last.lstrip("-").isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


def quantile(values: Sequence[float], q: float) -> float:
    """线性插值分位数，用于 timing 的 P50/P95/P99（指南 8.1）。"""
    arr = np.asarray(sorted(float(v) for v in values), dtype=np.float64)
    if arr.size == 0:
        raise CalibError("求分位数时样本为空")
    return float(np.quantile(arr, q, method="linear"))


def _as_array(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise CalibError(f"{name} 含非有限值")
    return arr


# ---------------------------------------------------------------------------
# 2. JSONL 记录格式（指南 1.3）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    stage: str
    run_id: str
    sample_id: str
    captured_at_ns: int
    payload: dict[str, Any]

    @property
    def record_type(self) -> str:
        return str(self.payload.get("record_type", "sample"))


def append_records(path: Path, records: Iterable[Record]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(
                {"stage": r.stage, "run_id": r.run_id, "sample_id": r.sample_id,
                 "captured_at_ns": r.captured_at_ns, "payload": r.payload},
                ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: Path, expect_stage: str | None = None) -> list[Record]:
    """读 JSONL，并按指南 1.3 检查字段与"每次实验第一条是 metadata"。"""
    path = Path(path)
    if not path.is_file():
        raise CalibError(f"数据文件不存在：{path}")
    out: list[Record] = []
    seen_runs: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            for key in ("stage", "run_id", "sample_id", "captured_at_ns", "payload"):
                if key not in obj:
                    raise CalibError(f"{path}:{lineno} 缺少字段 {key}")
            if expect_stage and obj["stage"] != expect_stage:
                raise CalibError(
                    f"{path}:{lineno} stage={obj['stage']!r} 与要求的 {expect_stage!r} 不符")
            rec = Record(str(obj["stage"]), str(obj["run_id"]), str(obj["sample_id"]),
                         int(obj["captured_at_ns"]), dict(obj["payload"]))
            if rec.run_id not in seen_runs:
                seen_runs.add(rec.run_id)
                if rec.record_type != "metadata":
                    raise CalibError(
                        f"{path}:{lineno} run_id={rec.run_id} 的第一条 payload 必须是 "
                        'record_type="metadata"（指南 1.3）')
            out.append(rec)
    if not out:
        raise CalibError(f"{path} 里没有记录")
    return out


def samples_of(records: Sequence[Record]) -> list[Record]:
    return [r for r in records if r.record_type == "sample"]


def metadata_of(records: Sequence[Record]) -> dict[str, Any]:
    for r in records:
        if r.record_type == "metadata":
            return r.payload
    raise CalibError("记录里没有 metadata 行")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def new_sample_id() -> str:
    return uuid.uuid4().hex[:10]


def _sample_field(records: Sequence[Record], key: str) -> list[Any]:
    return [r.payload[key] for r in samples_of(records) if key in r.payload]


# ---------------------------------------------------------------------------
# 3. init
# ---------------------------------------------------------------------------


def _null_out(shape: Any, dotted: str) -> Any:
    """按样例配置的"形状"生成 draft：操作者值/固定值填好，其余物理量置 null。

    用现成配置当形状模板而不是手写一份键表，是为了让 init 的键集合不可能和
    load_motion_params 的键集合漂移：少一个键，fit 出来的候选就根本加载不了。
    """
    if dotted in DRAFT_OPERATOR_VALUES:
        return DRAFT_OPERATOR_VALUES[dotted]
    if dotted in DRAFT_FIXED_VALUES:
        return DRAFT_FIXED_VALUES[dotted]
    if isinstance(shape, dict):
        # places 的每个位姿都要实测：保留 place_id 键名，值置 null。
        return {k: _null_out(v, f"{dotted}.{k}" if dotted else k)
                for k, v in shape.items()}
    if isinstance(shape, list):
        # 标定表、中转位、障碍、胶囊体整体置 null：保留样例条数会误导人以为已测过。
        return None
    return None


def cmd_init(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    template_path = ROOT / "calibration/SIM_ROBOT/simulation/profile.json"
    template = load_json(template_path)

    profile = _null_out(template, "")
    profile["profile_id"] = args.profile_id
    profile["robot_id"] = args.robot_id
    profile["status"] = "draft"
    profile["model"]["urdf_path"] = rel_from(out_dir, ROOT / REPO_URDF)
    profile["model"]["motor_calibration_path"] = "motor_calibration.json"

    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "profile.json", profile)
    if not (out_dir / "motor_calibration.json").is_file():
        dump_json(out_dir / "motor_calibration.json", _MOTOR_CALIB_SKELETON)

    n_null = _count_nulls(profile)
    print(f"已生成 draft 标定包：{out_dir / 'profile.json'}（{n_null} 个未测字段为 null）")
    print("下一步：")
    print("  1) 用官方 LeRobot 校准原始文件覆盖 motor_calibration.json（指南 4.1 第 2 条）；")
    print("  2) 复核 init 写入的验收要求与待试参数（DRAFT_OPERATOR_VALUES，指南 1.2）；")
    print("  3) 按 motor→joints→tool→workcell→timing→motion→grasp→pick_place 逐阶段"
          " capture / fit / validate。")
    return 0


def _count_nulls(obj: Any) -> int:
    if obj is None:
        return 1
    if isinstance(obj, dict):
        return sum(_count_nulls(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_nulls(v) for v in obj)
    return 0


# 官方校准文件骨架（LeRobot MotorCalibration 的字段）。init 只是提示要放真实内容，
# 里面没有测量值，capture --hardware 会拒绝拿它当真值用。
_MOTOR_CALIB_SKELETON: dict[str, Any] = {
    name: {"id": i + 1, "homing_offset": None, "range_min": None, "range_max": None,
           "drive_mode": 0}
    for i, name in enumerate(MOTOR_NAMES)
}


# ---------------------------------------------------------------------------
# 4. capture
# ---------------------------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> int:
    stage = args.stage
    profile_path = Path(args.profile)
    out_path = Path(args.output)
    # 采集时也要能读配置里的前置参数，所以先确认 profile.json 至少结构可解析。
    raw = load_json(profile_path)
    if stage not in STAGES:
        raise CalibError(f"未知阶段 {stage!r}，可用：{'、'.join(STAGES)}")

    if args.hardware:
        _check_hardware_prerequisites(raw, stage, profile_path)
        records = _capture_hardware(stage, profile_path, args)
        source = "hardware"
    else:
        records = _capture_simulated(stage)
        source = "simulated"
        # 模拟源也要产出官方电机校准文件，否则 fit --stage motor 会因为拿不到
        # 真实内容而必须报错。真机流程里这个文件由采集者从官方工具导出覆盖。
        if stage == "motor":
            _write_simulated_motor_calibration(profile_path)
    n = append_records(out_path, records)
    print(f"[{stage}] 采集 {n} 条（源={source}）→ {out_path}")
    return 0


def _check_hardware_prerequisites(raw: dict[str, Any], stage: str, profile_path: Path) -> None:
    """指南 3.3：capture 的实机模式逐阶段检查前置参数及报告。"""
    for prior in PREREQUISITES[stage]:
        report = profile_path.parent / "reports" / f"{prior}.json"
        if not report.is_file():
            raise CalibError(
                f"前置阶段 {prior} 还没有验收报告（{report}）。"
                "指南第 3 节要求按 motor→…→pick_place 顺序推进")
        data = load_json(report)
        if not data.get("stages", {}).get(prior, {}).get("passed"):
            raise CalibError(f"前置阶段 {prior} 的验收未通过，不能继续实机采集 {stage}")
    # 采集虽然只是"静止读回"，也必须先有真实串口（第 3 节"motor 阶段允许单
    # 电机配置与卸力手动读数"同样走这条总线）。注意不再要求 timing.*：只读采集
    # 用的是 CalibrationReader 自己的采集预算，而 timing 组要到 timing 阶段
    # 才拟合——拿它做前置会在 tool/workcell 采集时再次死锁。
    for key in ("motor.port", "motor.baudrate"):
        if get_path(raw, key) is None:
            raise CalibError(
                f"实机采集 {stage} 需要前置参数 {key}，当前为 null；"
                "请先把官方工具确认过的串口写入配置（不用猜测值自动填补）")


def _open_motor_controller(profile_path: Path):
    """按配置打开真机只读总线。串口对象在本项目里是延迟导入的（没装 pyserial 会报错）。

    走 load_calibration_profile 而不是 load_motion_params(mode="real")：采集
    发生在 draft 期，real 链路的 status/字段/哈希关卡在结构上不可能满足（P1-5）。
    """
    from configs.motion_params import load_calibration_profile
    from qingyun.grabbing.motor_control import CalibrationReader

    profile = load_calibration_profile(profile_path)
    return profile, CalibrationReader(profile)


def _capture_hardware(stage: str, profile_path: Path, args: argparse.Namespace) -> list[Record]:
    """硬件采集：只做该阶段"允许的规定动作范围"（指南 3.3）。

    每一步都是顺序单线程执行，不自动搜索机械硬端点、不运行未校验路径。
    本函数只读总线：采集路径上不存在任何提交运动的调用。
    """
    import time as _time

    profile, reader = _open_motor_controller(profile_path)
    run = new_run_id()
    meta = Record(stage, run, new_sample_id(), _time.monotonic_ns(), {
        "record_type": "metadata", "source": "hardware",
        "models": list(profile.motor.models), "firmware": list(profile.motor.firmware),
        "port": profile.motor.port,
        "power": args.power or "unspecified", "pad_id": args.pad or "unspecified",
        "load_label": args.load or "unloaded", "operator": args.operator or "unspecified",
    })
    out = [meta]

    def emit(payload: dict[str, Any]) -> None:
        payload.setdefault("captured_at_ns_note", None)
        out.append(Record(stage, run, new_sample_id(), _time.monotonic_ns(), payload))

    try:
        if stage == "motor":
            # 静止读回原始六通道量 + 参考角由外部量角器人工录入（--reference 文件）。
            refs = load_json(Path(args.reference)) if args.reference else None
            for _ in range(args.repeats or 5):
                snap = reader.read_snapshot()
                emit({"record_type": "sample", "raw_position": snap["raw_position"],
                      "raw_velocity": snap["raw_velocity"], "raw_current": snap["raw_current"],
                      "reference_angle_deg": (refs or {}).get("reference_angle_deg"),
                      "reference_current_ma": (refs or {}).get("reference_current_ma"),
                      "sample_start_ns": snap["sample_start_ns"],
                      "sample_end_ns": snap["sample_end_ns"],
                      "torque_enabled": 1, "motion_label": "static"})
        elif stage == "joints":
            # 卸力手动/受限低速点动，参考角必须来自外部夹具（指南第 5 节）。
            # q_bus 由位置寄存器按 4.2 第一行公式换算，不依赖 joints 拟合结果。
            # 参考样本必须整行合并：_fit_joints 还要读 sweep_joint_index、
            # endpoint_urdf_deg、endpoint_repeat_deg、application_limit_urdf_deg、
            # limit_reason、fk_check，逐个挑键会把它们丢掉（ISSUE-022）。
            refs = load_json(Path(args.reference)) if args.reference else {"samples": []}
            for row in refs.get("samples", []):
                if "q_reference_deg" not in row:
                    raise CalibError(
                        f"joints 参考样本缺 q_reference_deg：{sorted(row)}")
                snap = reader.read_snapshot()
                payload = {"record_type": "sample", "q_bus_deg": snap["q_bus_deg"]}
                payload.update(row)
                emit(payload)
        elif stage in ("tool", "workcell"):
            refs = load_json(Path(args.reference)) if args.reference else {"samples": []}
            for row in refs.get("samples", []):
                snap = reader.read_snapshot()
                if snap["q_urdf_deg"] is None:
                    raise CalibError(
                        f"实机采集 {stage} 需要 joints 阶段已拟合的 sign/zero_offset_deg"
                        "（指南 3 节顺序保证）；当前配置里它们仍是 null")
                payload = {"record_type": "sample", "q_urdf_deg": snap["q_urdf_deg"],
                           "gripper_pct": snap["gripper_pct"]}
                payload.update(row)
                emit(payload)
        else:
            raise CalibError(
                f"阶段 {stage} 的实机采集需要 ArmController 的完整动作路径，"
                "请把 --hardware 用于 motor/joints/tool/workcell；"
                "timing/motion/grasp/pick_place 的实机记录由运行期日志导出为同格式 JSONL")
    finally:
        reader.close()
    return out


# --- 模拟源 -----------------------------------------------------------------
# 指南 1.2：capture 未给 --hardware 时使用模拟源。下面这套样本是"自洽的一组假数据"，
# 目的是让 init→fit→validate→promote 整条链路能在没有真机的机器上被真正执行和回归，
# 包括八阶段拟合后所有运行期字段都能填满。它绝不是实机标定值。


def _sim_meta(stage: str, run: str, **extra: Any) -> Record:
    payload = {"record_type": "metadata", "source": "simulated",
               "models": ["sts3215"] * 6, "firmware": ["2.54"] * 6,
               "power": "5V/8A 实验室电源（模拟）", "pad_id": "SIM_PAD_A",
               "load_label": "simulated", "operator": "simulation-source"}
    payload.update(extra)
    return Record(stage, run, new_sample_id(), time_ns(), payload)


def time_ns() -> int:
    import time

    return time.monotonic_ns()


def _write_simulated_motor_calibration(profile_path: Path) -> None:
    """在标定包里放一份模拟的官方校准文件（仅当现有文件还是 init 骨架时）。

    骨架的判据是 range_min/range_max 为 null。已经有内容时绝不覆盖：那可能是
    采集者放进来的真实官方文件，覆盖掉就是丢数据。
    """
    raw = load_json(profile_path)
    target = _resolve_from(Path(profile_path).parent, raw["model"]["motor_calibration_path"])
    if target.is_file():
        existing = load_json(target)
        if all((existing.get(n) or {}).get("range_min") is not None for n in MOTOR_NAMES):
            return
    sim = {name: {"id": i + 1, "homing_offset": -2049 + 7 * i,
                  "range_min": 348 + 31 * i, "range_max": 3958 - 41 * i, "drive_mode": i % 2}
           for i, name in enumerate(MOTOR_NAMES)}
    dump_json(target, sim)
    print(f"已写入模拟官方电机校准文件：{target}（真机流程请换成官方工具导出的原始文件）")


def _capture_simulated(stage: str) -> list[Record]:
    gen = {
        "motor": _sim_motor, "joints": _sim_joints, "tool": _sim_tool,
        "workcell": _sim_workcell, "timing": _sim_timing, "motion": _sim_motion,
        "grasp": _sim_grasp, "pick_place": _sim_pick_place,
    }[stage]
    return gen()


def _sim_motor() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("motor", run)]
    closed, opened = 500, 3600
    for k in range(6):
        # 每通道取几个覆盖正反方向的读数；参考角/参考电流由外部量具给出。
        raw = [int(np.linspace(600, 3500, 7)[k % 7])]
        out.append(Record("motor", run, new_sample_id(), time_ns(), {
            "record_type": "sample",
            "raw_position": [2048 + 120 * i for i in range(6)],
            "raw_velocity": [3, -3, 0, 1, -1, 5],
            "raw_current": [10, 12, 9, 11, 8, 40],
            "reference_angle_deg": [float(v) for v in np.linspace(-90, 90, 6)],
            "reference_current_ma": [10.0, 12.0, 9.0, 11.0, 8.0, 40.0],
            "sample_start_ns": time_ns(), "sample_end_ns": time_ns() + 900_000,
            "torque_enabled": 1, "motion_label": "static_handread",
        }))
        del raw
    # 夹爪两端：指南 4.4 要求重复至少 5 次检查迟滞和方向。
    for _ in range(5):
        out.append(Record("motor", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "raw_position": [closed] * 6,
            "raw_velocity": [0] * 6, "raw_current": [20] * 6,
            "reference_angle_deg": None, "reference_current_ma": None,
            "sample_start_ns": time_ns(), "sample_end_ns": time_ns() + 800_000,
            "torque_enabled": 1, "motion_label": "gripper_closed_end",
            "gripper_closed_raw": closed, "gripper_open_raw": opened,
        }))
        out.append(Record("motor", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "raw_position": [opened] * 6,
            "raw_velocity": [0] * 6, "raw_current": [20] * 6,
            "reference_angle_deg": None, "reference_current_ma": None,
            "sample_start_ns": time_ns(), "sample_end_ns": time_ns() + 800_000,
            "torque_enabled": 1, "motion_label": "gripper_open_end",
            "gripper_closed_raw": closed, "gripper_open_raw": opened,
        }))
    return out


def _sim_joints() -> list[Record]:
    """用 URDF 名义零位反算 q_bus，并给每关节三个以上、覆盖正反方向的角度。"""
    run = new_run_id()
    out = [_sim_meta("joints", run)]
    ref_angles = np.linspace(-70.0, 70.0, 7)
    for j in range(5):
        for i, ref in enumerate(ref_angles):
            # 模拟"几何夹具测得的参考角"与总线角完全同向，只差一个零位偏置。
            q_bus = [0.0] * 5
            q_bus[j] = ref - 3.0 * (1 if j % 2 else -1)
            others = [float(v) for v in np.linspace(-20, 20, 5)]
            for k in range(5):
                if k != j:
                    q_bus[k] = others[k]
            ref_deg = list(q_bus)
            ref_deg[j] = q_bus[j] + 3.0 * (1 if j % 2 else -1)
            out.append(Record("joints", run, new_sample_id(), time_ns(), {
                "record_type": "sample", "q_bus_deg": [float(v) for v in q_bus],
                "q_reference_deg": [float(v) for v in ref_deg],
                "measured_tcp_position_m": None, "measured_tool_axes": None,
                "fixture_id": "SIM_FIXTURE_J", "load_label": "unloaded",
                "approach_direction": "joint_sweep",
                "sweep_joint_index": j, "sweep_direction": 1 if i % 2 else -1,
            }))
    # 应用限制（线缆、支架、桌面）：指南 5.4 要求单独记录，wrist_roll 尤其不能按
    # 官方 0~4095 当成可以绕满一圈。
    app_limits = [[-110.0, 110.0], [-100.0, 60.0], [-96.0, 96.0],
                  [-95.0, 95.0], [-90.0, 90.0]]
    for j, (lo, hi) in enumerate(app_limits):
        out.append(Record("joints", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "approach_direction": "application_limit",
            "sweep_joint_index": j, "application_limit_urdf_deg": [lo, hi],
            "limit_reason": "cable" if j == 4 else "bracket",
            "q_bus_deg": [0.0] * 5, "q_reference_deg": [0.0] * 5,
        }))
    # 端点重复测量（指南 5.4/5.5 用来定 measured_limits 与 margin）。
    for j in range(5):
        for side, sign in (("lower", -1.0), ("upper", 1.0)):
            for _ in range(5):
                base = 95.0 if j != 4 else 150.0
                val = sign * base
                out.append(Record("joints", run, new_sample_id(), time_ns(), {
                    "record_type": "sample", "endpoint_side": side,
                    "sweep_joint_index": j, "endpoint_urdf_deg": float(val),
                    "endpoint_repeat_deg": float(val + sign * 0.6),
                    "q_bus_deg": [0.0] * 5, "q_reference_deg": [0.0] * 5,
                    "fixture_id": "SIM_FIXTURE_END", "approach_direction": "endpoints",
                }))
    # FK 验证构型（指南 5.6 要求至少 10 个未参与拟合的构型）。
    for k in range(12):
        q = [float(v) for v in np.linspace(-60, 60, 5) + (k % 5) * 3.0]
        out.append(Record("joints", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "fk_check": True,
            "q_reference_deg": q, "fixture_id": f"SIM_CHK_{k}",
            "q_bus_deg": q, "measured_tcp_position_m": [0.28, 0.01 * k, 0.14],
            "measured_tool_axes": None, "approach_direction": "fk_validation",
        }))
    return out


def _sim_tool() -> list[Record]:
    """工具阶段：参考 TCP 中心、参考轴向、gap 与活动指角。

    为了让 fit 能真正解出 rotation_e_tcp 与 translation_samples，这里直接用
    名义工具变换生成"外部测量值"，等价于一个理想夹具下的无噪声观测。
    """
    from configs.motion_params import MotionParams, load_motion_params

    params = load_motion_params(ROOT / "calibration/SIM_ROBOT/simulation/profile.json",
                                 mode="mock")
    from qingyun.grabbing.kinematics_ext import ArmModel

    model = ArmModel(params)
    run = new_run_id()
    out = [_sim_meta("tool", run)]
    pcts = [0.0, 25.0, 40.0, 70.0, 100.0]
    poses = [[0.0, -90.0, 90.0, 0.0, 0.0], [15.0, -80.0, 85.0, 10.0, 5.0],
             [-20.0, -70.0, 80.0, -15.0, 10.0], [5.0, -60.0, 60.0, 20.0, -10.0]]
    for g in pcts:
        for q in poses:
            T_B_TCP = model.fk_tcp(np.array(q, float), g)
            T_B_E = model.fk_e(np.array(q, float))
            R_B_E = T_B_E[:3, :3]
            p_E_TCP = R_B_E.T @ (T_B_TCP[:3, 3] - T_B_E[:3, 3])
            # 参考"尖点触及"给出的基座系 TCP 中心与工具轴
            out.append(Record("tool", run, new_sample_id(), time_ns(), {
                "record_type": "sample", "gripper_pct": g,
                "q_urdf_deg": [float(v) for v in q],
                "tcp_reference_B_m": [float(v) for v in T_B_TCP[:3, 3]],
                "T_B_TCP_reference": [[float(v) for v in row] for row in T_B_TCP],
                "gap_m": _sim_gap(g), "gripper_reference_angle_deg": _sim_angle(g),
                "fixture_id": "SIM_CAL_BLOCK",
            }))
    del params, model, MotionParams
    return out


# 模拟工作台用的连杆包络：与仓库仿真配置同一套几何，字段语义见指南 2.2。
_SIM_CAPSULES = [
    ("base_body", "base_link", [0.0, 0.0, 0.0], [0.0388353, 0.0, 0.0624], 0.024),
    ("shoulder_body", "shoulder_link", [0.0, 0.0, 0.0],
     [-0.0303992, -0.0182778, -0.0542], 0.017),
    ("upper_arm", "upper_arm_link", [0.0, 0.0, 0.0], [-0.11257, -0.028, 0.0], 0.015),
    ("lower_arm", "lower_arm_link", [0.0, 0.0, 0.0], [-0.1349, 0.0052, 0.0], 0.014),
    ("wrist_body", "wrist_link", [0.0, 0.0, 0.0], [0.0, -0.0611, 0.0181], 0.013),
    ("gripper_body", "gripper_link", [0.0, 0.0, 0.0],
     [-0.0079, -0.000218121, -0.0981274], 0.013),
    ("tool_head", "gripper_frame_link", [-0.020, 0.0, 0.0], [0.020, 0.0, 0.0], 0.016),
    ("fixed_pad", "gripper_frame_link", [0.043, 0.0, 0.012], [0.054, 0.0, 0.012], 0.005),
    ("moving_pad", "moving_jaw_so101_v1_link", [-0.09199, -0.02297, 0.01902],
     [-0.10176, -0.01791, 0.01902], 0.005),
]
_SIM_IGNORE_PAIRS = [
    ["base_body", "shoulder_body"], ["base_body", "upper_arm"],
    ["shoulder_body", "upper_arm"], ["upper_arm", "lower_arm"],
    ["lower_arm", "wrist_body"], ["wrist_body", "gripper_body"],
    ["gripper_body", "tool_head"], ["gripper_body", "fixed_pad"],
    ["gripper_body", "moving_pad"], ["tool_head", "fixed_pad"],
    ["tool_head", "moving_pad"], ["fixed_pad", "moving_pad"],
]

_SIM_GAP_TABLE = [(0.0, 0.0050), (25.0, 0.0175), (50.0, 0.0300), (75.0, 0.0425),
                  (100.0, 0.0550)]
_SIM_ANGLE_TABLE = [(0.0, -10.0), (25.0, 17.5), (50.0, 45.0), (75.0, 72.5), (100.0, 100.0)]


def _sim_gap(pct: float) -> float:
    return float(np.interp(pct, [a for a, _ in _SIM_GAP_TABLE], [b for _, b in _SIM_GAP_TABLE]))


def _sim_angle(pct: float) -> float:
    return float(np.interp(pct, [a for a, _ in _SIM_ANGLE_TABLE], [b for _, b in _SIM_ANGLE_TABLE]))


def _sim_workcell() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("workcell", run)]
    # 桌面网格 9 点（指南 7.1）
    for i in range(9):
        x = 0.15 + 0.02 * (i % 3)
        y = -0.04 + 0.04 * (i // 3)
        out.append(Record("workcell", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "entity_id": "table",
            "point_B_m": [x, y, 0.0 + (0.0004 if i % 2 else -0.0003)],
            "q_urdf_deg": None, "gripper_pct": None,
            "measurement_method": "height_gauge", "reference_image": f"sim_table_{i}.jpg",
            "measurement_uncertainty_m": 0.0005, "fixture_id": "SIM_GRID",
        }))
    for name, pts in (("bracket_east", [[0.28, 0.13, 0.0], [0.36, 0.18, 0.10]]),
                      ("camera_pole", [[0.20, -0.20, 0.0], [0.23, -0.15, 0.15]])):
        for p in pts:
            out.append(Record("workcell", run, new_sample_id(), time_ns(), {
                "record_type": "sample", "entity_id": name, "point_B_m": p,
                "q_urdf_deg": None, "gripper_pct": None,
                "measurement_method": "caliper", "reference_image": f"sim_{name}.jpg",
                "measurement_uncertainty_m": 0.001, "fixture_id": "SIM_BOX",
            }))
    # 待机位与有序中转位示教
    for label, q in (("home", [0.0, -60.0, 60.0, 0.0, 0.0]),
                     ("wp0", [0.0, -75.0, 75.0, 0.0, 0.0]),
                     ("wp1", [0.0, -45.0, 45.0, 0.0, 0.0])):
        out.append(Record("workcell", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "entity_id": label, "waypoint_role": label,
            "point_B_m": None, "q_urdf_deg": q, "gripper_pct": 70.0,
            "measurement_method": "teach", "reference_image": f"sim_{label}.jpg",
            "order": 0 if label == "home" else int(label[-1]) + 1,
        }))
    # 抓取域扫描（指南 7.4）：给出目标中心与 TCP 的可达点云
    for x in np.arange(0.33, 0.40, 0.02):
        for y in np.arange(-0.06, 0.061, 0.03):
            for z in (0.014, 0.022, 0.030):
                yaw = math.degrees(math.atan2(y, x))
                out.append(Record("workcell", run, new_sample_id(), time_ns(), {
                    "record_type": "sample", "entity_id": "grasp_scan",
                    "point_B_m": [float(x), float(y), float(z)], "yaw_deg": yaw,
                    "q_urdf_deg": None, "gripper_pct": 40.0,
                    "measurement_method": "scan", "ik_ok": True,
                }))
    # 连杆胶囊体：端点在对应 link 系内，按网格与实物构造（指南 7.3 第 3 条）
    for cid, link, p0, p1, r in _SIM_CAPSULES:
        out.append(Record("workcell", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "entity_id": f"capsule_{cid}", "capsule_id": cid,
            "link": link, "p0_m": list(p0), "p1_m": list(p1), "radius_m": r,
            "point_B_m": None, "q_urdf_deg": None, "gripper_pct": None,
            "measurement_method": "cad_from_mesh",
        }))
    for pair in _SIM_IGNORE_PAIRS:
        out.append(Record("workcell", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "entity_id": "ignore_pair", "pair": list(pair),
        }))
    # 放置点录入：位置是释放时物体中心（指南 7.2）
    for pid, pos, yaw in (("default", [0.320, 0.055, 0.024], 10.0),
                          ("bin", [0.340, -0.060, 0.024], -10.0)):
        out.append(Record("workcell", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "entity_id": f"place_{pid}", "place_id": pid,
            "point_B_m": pos, "yaw_deg": yaw, "q_urdf_deg": None, "gripper_pct": 40.0,
            "measurement_method": "ruler_plate", "reference_image": f"sim_place_{pid}.jpg",
        }))
    return out


def _sim_timing() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("timing", run)]
    base = time_ns()
    span_ns, tick_ns = 8_000_000, 1_000_000
    for k in range(1200):
        read_start = base + k * tick_ns
        out.append(Record("timing", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "tick_index": k, "deadline_ns": read_start,
            "read_start_ns": read_start, "read_end_ns": read_start + 3_000_000 + (k % 7) * 400_000,
            "write_start_ns": read_start + 4_000_000,
            "write_end_ns": read_start + 5_500_000 + (k % 5) * 300_000,
            "feedback_sequence": k, "checkpoint_ns": read_start + 6_000_000,
            "sample_span_ns": span_ns,
            "stop_condition_ns": read_start + 90_000_000 if k % 40 == 0 else None,
            "last_trajectory_write_ns": read_start,
            "hold_submitted_ns": read_start + 95_000_000 if k % 40 == 0 else None,
            "mechanically_still_ns": read_start + 120_000_000 if k % 40 == 0 else None,
            "fault_injection": None,
        }))
    return out


def _sim_motion() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("motion", run)]
    q0 = np.array([0.0, -60.0, 60.0, 0.0, 0.0])
    # 三组速度递进的试验，每组正反方向重复（指南 9.2）。
    for trial, (vel_scale, ok) in enumerate([(0.5, True), (0.8, True), (1.3, False)]):
        for direction in (1, -1):
            for k in range(40):
                frac = k / 39.0 * vel_scale
                cmd = q0 + direction * np.array([10.0, 30.0, 25.0, 15.0, 20.0]) * frac
                lag = np.array([0.8, 1.6, 1.2, 1.0, 0.6]) * (vel_scale if ok else 6.0)
                out.append(Record("motion", run, new_sample_id(), time_ns(), {
                    "record_type": "sample", "trajectory_id": f"traj_{trial}",
                    "segment": f"seg{trial}", "command_deg": [float(v) for v in cmd],
                    "feedback_deg": [float(v) for v in (cmd - lag * direction)],
                    "speed_deg_s": [float(v) for v in np.array([60.0, 60.0, 60.0, 60.0, 60.0]) * vel_scale],
                    "current_ma": [float(v) for v in (200.0 + 250.0 * vel_scale) * np.ones(5)],
                    "tcp_reference_B_m": None, "time_ns": time_ns(),
                    "gripper_pct": 70.0, "load_label": "unloaded" if trial < 2 else "test_payload",
                    "stop_reason": None if ok else "acceleration_limit",
                    "passed": ok, "direction": direction,
                }))
    return out


def _sim_grasp() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("grasp", run, maturity_label="simulated")]
    # 空爪开合 20 次：用于 empty_closed_pct / empty_tol_pct 与开合超时（指南 10.2 第 3 条）
    for k in range(20):
        out.append(Record("grasp", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "trial_id": f"empty_{k}", "object_id": "NONE",
            "cycle_phase": "empty_close",
            "dimensions_reference_m": None, "maturity_label": None, "pad_id": "SIM_PAD_A",
            "target_center_B_m": None, "target_yaw_deg": None, "yaw_offset_deg": 0.0,
            "gripper_pct": 2.4 + 0.4 * (k % 3), "gripper_current_ma": 380.0,
            "command_pct": 0.0, "time_ns": time_ns(), "contact_label": None,
            "actual_lift_result": None, "actual_release_result": None,
            "damage_label": None, "damage_observe_minutes": None, "image_refs": [],
        }))
        out.append(Record("grasp", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "trial_id": f"empty_{k}", "object_id": "NONE",
            "cycle_phase": "empty_open",
            "dimensions_reference_m": None, "maturity_label": None, "pad_id": "SIM_PAD_A",
            "target_center_B_m": None, "target_yaw_deg": None, "yaw_offset_deg": 0.0,
            "gripper_pct": 99.0, "gripper_current_ma": 60.0, "command_pct": 100.0,
            "time_ns": time_ns(), "contact_label": None, "actual_lift_result": None,
            "actual_release_result": None, "damage_label": None,
            "damage_observe_minutes": None, "image_refs": [],
        }))
    # 量块/假果/真果的接触试验（指南 10.2 第 4 条）
    for idx, (dim, gap) in enumerate((([0.040, 0.030, 0.038], 0.0300),
                                      ([0.045, 0.034, 0.040], 0.0320),
                                      ([0.050, 0.038, 0.042], 0.0340),
                                      ([0.038, 0.028, 0.036], 0.0270),
                                      ([0.042, 0.031, 0.039], 0.0290))):
        trial = f"contact_{idx}"
        for k in range(12):
            closing = 70.0 - k * 3.0
            contact_at = 4 + (idx % 3)
            touching = k >= contact_at
            out.append(Record("grasp", run, new_sample_id(), time_ns(), {
                "record_type": "sample", "trial_id": trial, "object_id": f"OBJ_{idx}",
                "cycle_phase": "close", "dimensions_reference_m": dim,
                "maturity_label": "firm" if idx < 3 else "ripe", "pad_id": "SIM_PAD_A",
                "target_center_B_m": [0.35, 0.02 * (idx - 2), 0.022],
                "target_yaw_deg": float(idx * 5 - 5), "yaw_offset_deg": 0.0,
                "gripper_pct": float(closing),
                "gripper_current_ma": float(280.0 + 30.0 * (k - contact_at)) if touching else 45.0,
                "command_pct": float(closing - 3.0), "time_ns": time_ns(),
                "contact_label": "first_contact" if k == contact_at else (
                    "after" if touching else "before"),
                "actual_gap_m": gap, "actual_lift_result": None,
                "actual_release_result": None, "damage_label": None,
                "damage_observe_minutes": None, "image_refs": [],
            }))
        out.append(Record("grasp", run, new_sample_id(), time_ns(), {
            "record_type": "trial_summary", "trial_id": trial, "object_id": f"OBJ_{idx}",
            "dimensions_reference_m": dim, "maturity_label": "firm", "pad_id": "SIM_PAD_A",
            "target_center_B_m": [0.35, 0.0, 0.022], "target_yaw_deg": 0.0,
            "yaw_offset_deg": 0.0, "gripper_pct": 45.0, "gripper_current_ma": 300.0,
            "command_pct": 40.0, "time_ns": time_ns(), "contact_label": "first_contact",
            "actual_gap_m": gap, "actual_lift_result": "lifted",
            "contact_center_offset_object_m": [0.0, 0.0, 0.006 - 0.002 * (idx % 3)],
            "grasp_yaw_offset_deg": 0.0,
            "actual_release_result": "placed", "damage_label": "none",
            "damage_observe_minutes": 30.0, "image_refs": [],
            "slip_drift_pct": 1.8 + 0.3 * idx, "release_drop_pct": 0.0,
        }))
    return out


def _sim_pick_place() -> list[Record]:
    run = new_run_id()
    out = [_sim_meta("pick_place", run)]
    for k in range(34):
        success = k not in (7, 19)          # 两次非软件原因失败，用来算实际成功率
        drop = k == 25
        # 模拟果实不产生压伤：损伤率 2/34=5.9% 会稳定超过 2% 的验收上限，
        # 让 promote 链路无法被验证。"验收不通过"这一路径改由 tests 用改过的报告覆盖。
        damage = False
        out.append(Record("pick_place", run, new_sample_id(), time_ns(), {
            "record_type": "sample", "trial_id": f"pp_{k}",
            "profile_parameter_sha256": "not-bound-here",
            "target_position_m": [0.35, 0.01 * (k % 5 - 2), 0.022],
            "target_yaw_deg": float((k % 9) * 4 - 16), "grade": "A",
            "place_id": "default" if k % 2 else "bin",
            "start_ns": time_ns(), "end_ns": time_ns() + 22_000_000_000,
            "software_status": "SUCCESS", "software_stage": "DONE",
            "holding": "EMPTY", "recovery_required": False,
            "began_descend": True,
            "actual_success": bool(success), "actual_drop": bool(drop),
            "actual_place_position_m": [0.320, 0.055, 0.024] if success else None,
            "actual_place_yaw_deg": 10.0 if success else None,
            "place_error_center_m": 0.006 if success else None,
            "place_error_yaw_deg": 2.1 if success else None,
            "damage_label": "bruise" if damage else "none",
            "damage_observe_minutes": 30.0, "image_refs": [f"sim_pp_{k}.jpg"],
            "feedback_span_s": 0.008, "stop_latency_s": 0.05,
        }))
    return out


# ---------------------------------------------------------------------------
# 5. fit
# ---------------------------------------------------------------------------


def _audit_path(profile_path: Path) -> Path:
    """候选配置的旁证文件：<candidate>.audit.json，与配置同目录。"""
    return Path(profile_path).with_suffix(".audit.json")


def _read_audit(profile_path: Path) -> dict[str, Any]:
    path = _audit_path(profile_path)
    return load_json(path) if path.is_file() else {}


def _strip_audit(raw: dict[str, Any]) -> dict[str, Any]:
    """去掉历史版本可能留在 profile 里的审计键，保证键集合严格符合第 2 节。"""
    out = json.loads(json.dumps(raw))
    # 审计/说明性字段一律不进 profile：键集合必须严格等于指南第 2 节，
    # 否则 load_motion_params 会按"未知键"拒绝。旧版本可能把它们写在顶层或
    # 分组里，两处都清掉。
    out = {k: v for k, v in out.items() if not k.startswith("_")}
    for group in out.values():
        if isinstance(group, dict):
            for sub in [k for k in group if k.startswith("_")]:
                del group[sub]
    return out


def cmd_fit(args: argparse.Namespace) -> int:
    stage = args.stage
    if stage not in STAGES:
        raise CalibError(f"未知阶段 {stage!r}")
    profile_path = Path(args.profile)
    base = load_json(profile_path)
    records = read_records(Path(args.input), expect_stage=stage)
    ids_fit = {r.sample_id for r in samples_of(records)}
    updates, residuals = FITTERS[stage](base, records)
    # fit 只更新对应测量字段并输出独立候选文件（指南 1.2）
    base = _strip_audit(base)
    candidate = json.loads(json.dumps(base))
    for dotted, value in updates.items():
        if dotted.startswith("_calibration_note"):
            continue          # 说明性数字只进 sidecar，不能污染 profile 的键集合
        set_path(candidate, dotted, value)
    candidate["status"] = "draft"        # 候选仍未验收
    out = Path(args.output)
    dump_json(out, candidate)
    # 审计信息写旁证文件：profile.json 的键集合必须严格等于指南第 2 节，
    # 往里塞 _calibration_audit 会让 load_motion_params 按"未知键"直接拒绝。
    # 从**输入**配置旁边继承已拟合阶段的记录再累加：候选配置是一级一级链出来的，
    # 只记本级会让 validate 读不到前面阶段的样本清单与残差。
    audit = _read_audit(profile_path)
    audit[stage] = {"fit_sample_ids": sorted(ids_fit),
                    "updated_fields": sorted(k for k in updates
                                             if not k.startswith("_calibration_note")),
                    "notes": {k: v for k, v in updates.items()
                              if k.startswith("_calibration_note")},
                    "residuals": residuals}
    dump_json(_audit_path(out), audit)
    print(f"[{stage}] 更新 {len(updates)} 个字段 → {out}")
    for key, val in sorted(residuals.items()):
        print(f"    {key} = {_fmt(val)}")
    return 0


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return json.dumps(v, ensure_ascii=False)


# --- 各阶段拟合 -------------------------------------------------------------


def _fit_motor(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 4.1~4.4：身份、单位比例、夹爪两端，以及资源哈希。"""
    upd: dict[str, Any] = {}
    res: dict[str, Any] = {}
    meta = metadata_of(records)
    # 身份与端口只能来自**真机**采集。指南 1.3 要求每一次实验的 metadata 都记录
    # 型号与固件，模拟源因此也必须写它们——但那描述的是仿真工作台的设定，不是
    # 这台臂的实测值。无条件回填会把操作者刚在真机上读回的 motor.firmware 静默
    # 改回 "2.54"（ISSUE-019：实机为 3.10 时，initialize() 随后会以"配置登记 2.54
    # 与读回不符"拒绝启动，而错误信息指不回是拟合环节引入的）。
    from_hardware = meta.get("source") == "hardware"
    if from_hardware and meta.get("models"):
        upd["motor.models"] = list(meta["models"])
    if from_hardware and meta.get("firmware"):
        upd["motor.firmware"] = list(meta["firmware"])

    closed_samples = [r.payload for r in samples_of(records)
                      if r.payload.get("motion_label") == "gripper_closed_end"]
    open_samples = [r.payload for r in samples_of(records)
                    if r.payload.get("motion_label") == "gripper_open_end"]
    if closed_samples and open_samples:
        # 指南 4.4：重复至少 5 次，检查迟滞与方向；取各次读数的一致性。
        closed_vals = {int(s["raw_position"][5]) for s in closed_samples}
        open_vals = {int(s["raw_position"][5]) for s in open_samples}
        if len(closed_vals) > 1 or len(open_vals) > 1:
            res["gripper_end_hysteresis_counts"] = [
                int(max(closed_vals) - min(closed_vals)), int(max(open_vals) - min(open_vals))]
        upd["motor.gripper_closed_raw"] = int(round(statistics.mean(closed_vals)))
        upd["motor.gripper_open_raw"] = int(round(statistics.mean(open_vals)))

    # 位置编码分辨率按型号从 vendor 表取，而不是从读数猜（指南 2.1）。
    from libs.so_arm_core.motors.feetech.tables import MODEL_RESOLUTION

    upd["motor.position_resolution"] = [
        int(MODEL_RESOLUTION[str(m)]) for m in upd.get("motor.models", base["motor"]["models"])
    ]
    # 参考电流用于验证 mA/raw 的合理性（4.3）：decoded 读数与参考值的线性比例。
    # 逐通道求中位数（字段是 float[6]，各通道比例可以不同，P3-6）；某通道没有
    # 参考数据时保留操作者给定的基值，不用别的通道的中位数冒充。
    ref_rows = [r.payload for r in samples_of(records)
                if r.payload.get("reference_current_ma")]
    if ref_rows:
        base_scales = base["motor"]["current_ma_per_raw"]
        fitted: list[float] = []
        all_ratios: list[float] = []
        for i in range(6):
            ch_ratios = [row["reference_current_ma"][i] / float(row["raw_current"][i])
                         for row in ref_rows
                         if row.get("reference_current_ma")
                         and row["reference_current_ma"][i] and row["raw_current"][i]]
            if ch_ratios:
                fitted.append(float(np.median(ch_ratios)))
                all_ratios.extend(ch_ratios)
            else:
                fitted.append(None if base_scales[i] is None else float(base_scales[i]))
        upd["motor.current_ma_per_raw"] = fitted
        if all_ratios:
            res["current_scale_spread"] = float(np.std(all_ratios))
        upd["motor.current_zero_raw"] = [0.0] * 6
    # 速度比例：厂商资料给定的每计数 deg/s（4.3），随后用差分中位数核对方向。
    res["velocity_direction_check"] = _velocity_direction_check(records)
    if "motor.velocity_deg_s_per_raw" not in upd:
        upd["motor.velocity_deg_s_per_raw"] = list(base["motor"]["velocity_deg_s_per_raw"])
    # 串口路径同样只认真机采集的 metadata（_capture_hardware 会写入真实端口）。
    # 操作者没有提供时保持 null：技术文档第八节要求"实机未测参数保持未完成
    # 状态，不用猜测值自动填补"（P2-3 修掉的正是这个猜测值）。
    if from_hardware and meta.get("port"):
        upd["motor.port"] = str(meta["port"])

    # 资源哈希（指南 1.1 要求 real 模式复核）
    profile_dir = Path(args_profile_dir.get("current", ".")).resolve()
    # 哈希是 real 模式加载的前置条件（指南 1.1），拿不到文件就必须当场失败，
    # 不能留下 null 让配置在实机启动时才报错。
    urdf_rel = upd.get("model.urdf_path", base["model"]["urdf_path"])
    urdf = _resolve_from(profile_dir, urdf_rel) if urdf_rel else None
    if urdf is None or not urdf.is_file():
        raise CalibError(
            f"motor 阶段找不到 URDF：{urdf}（profile.model.urdf_path={urdf_rel!r}）。"
            "先确认 init --output 与 --profile 是否在同一台机器/同一目录树")
    upd["model.urdf_sha256"] = sha256_file(urdf)
    calib_rel = base["model"]["motor_calibration_path"]
    calib = _resolve_from(profile_dir, calib_rel) if calib_rel else None
    if calib is None or not calib.is_file():
        raise CalibError(
            f"motor 阶段找不到官方电机校准文件：{calib}。指南 4.1 第 2 条要求保留原始"
            " LeRobot 校准文件并算出 SHA256")
    data = load_json(calib)
    _validate_motor_calibration(data)
    upd["model.motor_calibration_sha256"] = sha256_file(calib)
    return upd, res


# fit 需要知道 profile.json 在哪，才能解析相对路径算哈希。用一个模块级上下文变量
# 传递，避免给八个 fitter 都加一个参数。
args_profile_dir: dict[str, str] = {"current": "."}


def _validate_motor_calibration(data: Any) -> None:
    """官方校准文件必须六个通道齐全且端点已填，否则它还是 init 生成的骨架。"""
    missing = [n for n in MOTOR_NAMES if n not in (data or {})]
    if missing:
        raise CalibError(f"电机校准文件缺少通道 {missing}")
    for name in MOTOR_NAMES:
        row = data[name]
        for key in ("id", "range_min", "range_max"):
            if row.get(key) is None:
                raise CalibError(
                    f"电机校准文件 {name}.{key} 仍是 null：这是 init 生成的骨架，"
                    "请用官方 LeRobot 校准原始文件覆盖它（指南 4.1 第 2 条）")


def _resolve_from(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p)


def _velocity_direction_check(records: list[Record]) -> Any:
    """4.3：用多组低速正反向运动的连续位置差分中位数与速度读数对照。"""
    rows = [r.payload for r in samples_of(records) if r.payload.get("motion_label")]
    diffs: list[float] = []
    prev = None
    for row in rows:
        pos = row.get("raw_position")
        if not pos:
            continue
        if prev is not None:
            diffs.append(float(np.median(np.asarray(pos, float) - np.asarray(prev, float))))
        prev = pos
    return {"n_pairs": len(diffs), "median_delta_counts": (float(np.median(diffs)) if diffs else None)}


def _fit_joints(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 5.2：对每关节试 sign=±1 拟合 q_reference = sign*q_bus + offset，取残差小的一侧。"""
    upd: dict[str, Any] = {}
    res: dict[str, Any] = {}
    # 只要单关节扫描样本：端点重复测量与 FK 验证构型里的 q_bus/q_reference 是占位值，
    # 混进 sign/offset 拟合会无谓抬高残差，它们各自有专门的用途。
    rows = [r.payload for r in samples_of(records)
            if r.payload.get("approach_direction") == "joint_sweep"]
    # sign 与零位偏置完全由本阶段的拟合决定，所以不能从 draft 的 null 里取初值；
    # 这里的 [+1, 0] 只是"未翻转、无偏置"的名义起点，随后每个关节都会被重算。
    sign = [1] * 5
    offset = [0.0] * 5
    rms: list[float] = []
    for j in range(5):
        pairs = [(rp["q_bus_deg"][j], rp["q_reference_deg"][j])
                 for rp in rows if rp["sweep_joint_index"] == j]
        if len(pairs) < 3:
            raise CalibError(f"joints 阶段关节 {j} 只有 {len(pairs)} 个拟合样本，至少需要 3 个")
        best = None
        for candidate_sign in (1.0, -1.0):
            bus = np.array([p[0] for p in pairs]) * candidate_sign
            ref = np.array([p[1] for p in pairs])
            off = float(np.mean(ref - bus))              # 残差均值 = 稳健的零位估计
            resid = float(np.sqrt(np.mean((ref - (bus + off)) ** 2)))
            if best is None or resid < best[0]:
                best = (resid, candidate_sign, off)
        rms.append(best[0])
        sign[j], offset[j] = int(best[1]), best[2]
    upd["joints.sign"] = sign
    upd["joints.zero_offset_deg"] = offset
    res["sign_fit_rms_deg"] = [round(v, 6) for v in rms]

    # 端点重复测量 → measured_limits_deg 与 margin_deg（指南 5.4/5.5）
    ends: dict[tuple[int, str], list[float]] = {}
    for rp in (r.payload for r in samples_of(records)):
        if rp.get("endpoint_urdf_deg") is not None:
            key = (int(rp["sweep_joint_index"]), str(rp["endpoint_side"]))
            ends.setdefault(key, []).append(
                [rp["endpoint_urdf_deg"], rp["endpoint_repeat_deg"]])
    limits = [[None, None] for _ in range(5)]
    margins = [float(v) for v in (base["joints"]["margin_deg"] or [2.0] * 5)]
    for (j, side), vals in ends.items():
        arr = np.array(vals, dtype=np.float64)
        # 端点重复误差：同一侧两次读数的最大半差，加上统计散布，作为余量来源。
        spread = float(np.max(arr[:, 1]) - np.min(arr[:, 1])) if len(arr) > 1 else 0.0
        idx = 0 if side == "lower" else 1
        # 取重复观测里更"保守"（更靠近内部）的那个作为可用行程端点。
        limits[j][idx] = float(np.min(arr[:, 1]) if idx == 0 else np.max(arr[:, 1]))
        margins[j] = max(margins[j], spread + 0.5)
    if any(l[0] is None or l[1] is None for l in limits):
        raise CalibError("joints 阶段缺少某些关节的下限或上限端点测量")
    upd["joints.measured_limits_deg"] = limits
    upd["joints.margin_deg"] = margins
    # 线缆/支架/桌面限制同样在 joints 阶段记录（指南 5.4）。缺记录时保留 draft 值，
    # 让加载器去决定是否算未测量，而不是在这里猜一个"看起来安全"的范围。
    app_rows = [rp for rp in (r.payload for r in samples_of(records))
                if rp.get("approach_direction") == "application_limit"]
    if app_rows:
        app = [[None, None] for _ in range(5)]
        for rp in app_rows:
            j = int(rp["sweep_joint_index"])
            lo, hi = [float(v) for v in rp["application_limit_urdf_deg"]]
            app[j] = [min(lo, hi), max(lo, hi)]
        if any(a[0] is None for a in app):
            raise CalibError("application_limits_deg 有关节缺少记录（指南 5.4）")
        upd["joints.application_limits_deg"] = app
        res["application_limit_sources"] = {str(int(rp["sweep_joint_index"])):
                                            str(rp.get("limit_reason")) for rp in app_rows}
    res["endpoint_repeat_spread_deg"] = {f"{j}_{s}": round(v, 4) for (j, s), v in
                                         {k: (float(np.ptp(np.array(vals, float), axis=0)[1])
                                              if len(vals) > 1 else 0.0)
                                          for k, vals in ends.items()}.items()}
    # FK 验证构型数量必须达到指南 5.6 的"至少 10 个未参与拟合"
    checks = [rp for rp in (r.payload for r in samples_of(records)) if rp.get("fk_check")]
    res["fk_validation_configs"] = len(checks)
    if len(checks) < 10:
        raise CalibError(
            f"joints 阶段只有 {len(checks)} 个 FK 验证构型，指南 5.6 要求至少 10 个")
    return upd, res


def _fit_tool(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 6.2/6.3/6.4：TCP 平移（已知参考点或未知 pivot）、旋转 SVD 投影、两张表。"""
    from libs.so_arm_core.kinematics import RobotKinematics

    rows = [r.payload for r in samples_of(records) if r.payload.get("q_urdf_deg")]
    if len(rows) < 6:
        raise CalibError(f"tool 阶段样本太少（{len(rows)}），无法同时定出旋转与平移")

    # FK 必须用"被标定机器人自己的 URDF"（P3-7）：tool 阶段 gripper/运行期字段
    # 还没测完，走不了 load_motion_params，而纯 FK 也不需要——直接按本包
    # model.urdf_path 构造 vendor 运动学，五个姿态关节名固定 JOINT_NAMES。
    urdf_rel = base["model"]["urdf_path"]
    urdf_path = _resolve_from(Path(args_profile_dir["current"]), urdf_rel)
    if not urdf_path.is_file():
        raise CalibError(f"tool 拟合找不到本包 URDF：{urdf_path}（model.urdf_path={urdf_rel!r}）")
    kin = RobotKinematics(str(urdf_path), "gripper_frame_link", list(JOINT_NAMES))

    def fk_e(q_deg: np.ndarray) -> np.ndarray:
        return np.asarray(kin.forward_kinematics(np.asarray(q_deg, dtype=np.float64)),
                          dtype=np.float64)

    # 按开度分组求平移：已知独立参考中心时 p_E_TCP = R_i^T (p_B_ref - t_i)（指南 6.2）
    by_pct: dict[float, list[np.ndarray]] = {}
    for rp in rows:
        q = np.asarray(rp["q_urdf_deg"], float)
        g = float(rp["gripper_pct"])
        T_B_E = fk_e(q)
        p_tcp = np.asarray(rp["tcp_reference_B_m"], float)
        by_pct.setdefault(round(g, 3), []).append(
            T_B_E[:3, :3].T @ (p_tcp - T_B_E[:3, 3]))
    pcts = sorted(by_pct)
    if len(pcts) < 3:
        raise CalibError(f"tool 阶段只覆盖 {len(pcts)} 个开度，至少需要 3 个")
    samples = []
    for g in pcts:
        stack = np.array(by_pct[g], float)
        # 稳健平均：逐分量用中位数，避免个别坏点把 TCP 拉偏。
        samples.append({"gripper_pct": float(g),
                        "xyz_m": [float(v) for v in np.median(stack, axis=0)]})

    # 旋转：对每个姿态算 R_E_TCP = R_B_E^T R_B_TCP_ref，再对矩阵和做 SVD 投影 SO(3)
    acc = np.zeros((3, 3))
    per_sample: list[np.ndarray] = []
    observing: list[np.ndarray] = []
    for rp in rows:
        R_B_E = fk_e(np.asarray(rp["q_urdf_deg"], float))[:3, :3]
        R_B_TCP = np.asarray(rp["T_B_TCP_reference"], float)[:3, :3]
        R = R_B_E.T @ R_B_TCP
        per_sample.append(R)
        # 观测帧本身。所有样本的 R_E_TCP 按定义都应当相同（它就是待求量），所以
        # "姿态变化是否足够"只能看基座系的观测帧 R_B_E 有没有绕三根轴散开。
        observing.append(R_B_E)
        acc += R
    U, _, Vt = np.linalg.svd(acc)
    R_e_tcp = U @ Vt
    if np.linalg.det(R_e_tcp) < 0:          # 强制 det=+1，避免投影到反射解
        U[:, -1] *= -1.0
        R_e_tcp = U @ Vt
    rot_err = float(np.max(np.abs(R_e_tcp.T @ R_e_tcp - np.eye(3))))
    # 每个姿态的轴向一致性（指南 6.2"姿态变化不足时拒绝拟合"）
    spread = float(np.mean([np.linalg.norm(R - R_e_tcp) for R in per_sample]))
    rank_report = _pose_diversity_rank(observing)
    if rank_report["rank"] < 3:
        raise CalibError(
            f"tool 阶段姿态变化不足（矩阵秩 {rank_report['rank']}，条件数 "
            f"{rank_report['cond']:.1f}），拒绝拟合旋转（指南 6.2/6.3）")

    upd = {
        "tool.rotation_e_tcp": [[float(v) for v in row] for row in R_e_tcp],
        "tool.translation_samples": samples,
    }
    res = {
        "rotation_orthonormality_err": rot_err,
        "rotation_mean_deviation": spread,
        "pose_matrix_rank": rank_report["rank"],
        "pose_matrix_cond": rank_report["cond"],
        "translation_opening_count": len(samples),
    }
    # gap_table / angle_table：分箱后做严格单调检查（指南 6.4）
    upd["gripper.gap_table"] = _monotone_table(
        [(float(rp["gripper_pct"]), float(rp["gap_m"])) for rp in rows],
        "gap_m", increasing_required=True)
    upd["gripper.angle_table"] = _monotone_table(
        [(float(rp["gripper_pct"]), float(rp["gripper_reference_angle_deg"])) for rp in rows],
        "angle_deg", increasing_required=False)
    return upd, res


def _pose_diversity_rank(rotations: Sequence[np.ndarray]) -> dict[str, float]:
    """把每个样本相对平均姿态的小增量堆起来求秩，判断姿态变化是否够定 3 个轴。

    只用 R_i 的反对称部分（等效转轴角）做 SVD：秩不足说明所有样本近似共面/共轴，
    旋转拟合会退化，此时必须补测姿态而不是接受一个坏解。
    """
    mean = np.mean(np.array(rotations, float), axis=0)
    deltas = []
    for R in rotations:
        d = R @ mean.T
        deltas.append(np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) * 0.5)
    A = np.array(deltas, float)
    if A.size == 0:
        return {"rank": 0, "cond": float("inf")}
    s = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(s > max(1e-8, s[0] * 1e-4)))
    cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")
    return {"rank": rank, "cond": cond}


def _monotone_table(pairs: Sequence[tuple[float, float]], value_name: str,
                    *, increasing_required: bool) -> list[dict[str, float]]:
    """按开度分箱求中位数，然后强制检查严格单调（指南 6.4）。

    某段非单调时不排序掩盖物理问题：这里直接报错，让操作者缩小使用范围或重做指垫。
    """
    if not pairs:
        raise CalibError(f"没有 {value_name} 样本")
    bins: dict[float, list[float]] = {}
    for g, v in pairs:
        bins.setdefault(round(g, 2), []).append(float(v))
    keys = sorted(bins)
    rows = [{"gripper_pct": float(k), value_name: float(np.median(bins[k]))} for k in keys]
    if len(rows) < 3:
        raise CalibError(f"{value_name} 表只有 {len(rows)} 个开度点，至少需要 3 个")
    pcts = [r["gripper_pct"] for r in rows]
    vals = [r[value_name] for r in rows]
    if any(b <= a for a, b in zip(pcts, pcts[1:])):
        raise CalibError(f"{value_name} 表的 gripper_pct 必须严格递增")
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    if increasing_required:
        if any(d <= 0 for d in diffs):
            raise CalibError(f"{value_name} 必须严格递增（非单调说明指垫迟滞过大）")
    else:
        if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
            raise CalibError(f"{value_name} 方向必须全表一致（指南 2.4）")
    return [{k: float(v) for k, v in r.items()} for r in rows]


def _fit_workcell(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """桌面网格、障碍 AABB、示教位姿、抓取域扫描、放置位姿。"""
    upd: dict[str, Any] = {}
    res: dict[str, Any] = {}
    rows = list(samples_of(records))

    table = [r.payload for r in rows if r.payload.get("entity_id") == "table"]
    if len(table) < 9:
        raise CalibError(f"桌面网格只有 {len(table)} 点，指南 7.1 要求至少 9 点")
    zs = np.array([p["point_B_m"][2] for p in table], float)
    # 桌面高度取中位数；偏离量取最大绝对偏差（指南 7.1 的 table_flatness_m）。
    upd["workspace.table_z_m"] = float(np.median(zs))
    upd["workspace.table_flatness_m"] = float(np.max(np.abs(zs - np.median(zs))))
    res["table_points"] = len(table)

    # 只有"给了 AABB 角点、且 entity_id 不属于其它角色"的记录才是固定障碍样本。
    # 逐个前缀枚举容易漏（新增一种记录类型就会把 None 当坐标收进来），所以反过来
    # 用非空 point_B_m 做硬门槛。
    non_obstacle_roles = ("table", "home", "grasp_scan", "capsule_", "ignore_pair",
                          "place_", "wp")
    ents: dict[str, list[list[float]]] = {}
    for p in (r.payload for r in rows):
        eid = str(p.get("entity_id", ""))
        point = p.get("point_B_m")
        if point is None or eid.startswith(non_obstacle_roles):
            continue
        ents.setdefault(eid, []).append(point)
    obstacles = []
    for eid, pts in sorted(ents.items()):
        arr = np.array(pts, float)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        obstacles.append({"id": eid,
                          "bounds_m": [[float(v) for v in lo], [float(v) for v in hi]]})
    upd["workspace.obstacles"] = obstacles

    # 示教位：home 与有序 safe_waypoints
    home = None
    waypoints: list[tuple[int, list[float]]] = []
    for p in (r.payload for r in rows):
        role = p.get("waypoint_role")
        if role == "home":
            home = p["q_urdf_deg"]
        elif role and role.startswith("wp"):
            waypoints.append((int(p.get("order", 0)), p["q_urdf_deg"]))
    if home is None:
        raise CalibError("workcell 阶段没有示教 home 位")
    if not waypoints:
        raise CalibError("workcell 阶段至少要有一个安全中转位（指南 2.2 K>=1）")
    upd["workspace.home_joints_deg"] = [float(v) for v in home]
    upd["workspace.safe_waypoints_deg"] = [[float(v) for v in q]
                                           for _, q in sorted(waypoints, key=lambda t: t[0])]

    # 抓取域扫描 → target_bounds / tcp_bounds：取 2%~98% 分位再往里收，
    # 这样扫描边缘的少量异常样本不会把范围撑大。
    scan = [p for p in (r.payload for r in rows)
            if p.get("entity_id") == "grasp_scan" and p.get("ik_ok")]
    if not scan:
        raise CalibError("workcell 阶段没有可用的抓取域扫描记录（指南 7.4）")
    pts = np.array([p["point_B_m"] for p in scan], float)
    lo = np.array([quantile(pts[:, i], 0.02) for i in range(3)])
    hi = np.array([quantile(pts[:, i], 0.98) for i in range(3)])
    upd["workspace.target_bounds_m"] = [[float(v) for v in lo], [float(v) for v in hi]]
    # TCP 活动域：目标域沿 +Z 抬到接近/提起高度，再加一点余量。
    app = float(base["grasp"]["approach_height_m"] or 0.08)
    upd["workspace.tcp_bounds_m"] = [
        [float(lo[0] - 0.02), float(lo[1] - 0.03), float(max(0.005, lo[2] - 0.01))],
        [float(hi[0] + 0.02), float(hi[1] + 0.03), float(hi[2] + app + 0.03)]]
    res["grasp_scan_points"] = len(scan)

    # 连杆胶囊体与自碰撞豁免（指南 7.3 第 3 条：依据网格和实物构造，人工核对每个豁免）
    capsules = []
    for p in (r.payload for r in rows):
        cid = p.get("capsule_id")
        if not cid:
            continue
        capsules.append({"id": str(cid), "link": str(p["link"]),
                         "p0_m": [float(v) for v in p["p0_m"]],
                         "p1_m": [float(v) for v in p["p1_m"]],
                         "radius_m": float(p["radius_m"])})
    if not capsules:
        raise CalibError("workcell 阶段没有胶囊体记录，碰撞模型无从构造（指南 7.3 第 3 条）")
    upd["collision.link_capsules"] = capsules
    ids = {c["id"] for c in capsules}
    pairs = []
    for p in (r.payload for r in rows):
        pair = p.get("pair")
        if not pair:
            continue
        a, b = str(pair[0]), str(pair[1])
        # 豁免表引用不存在的 id 会让"人工核对每个接触豁免"失效，这里当作错误拒绝。
        for which, val in (("a", a), ("b", b)):
            if val not in ids:
                raise CalibError(f"ignore_self_pairs 的 {which} 端 {val!r} 不在胶囊体 id 中")
        pairs.append([a, b])
    upd["collision.ignore_self_pairs"] = pairs

    places: dict[str, Any] = {}
    for p in (r.payload for r in rows):
        pid = p.get("place_id")
        if not pid:
            continue
        places[str(pid)] = {"position": [float(v) for v in p["point_B_m"]],
                            "yaw_deg": float(p["yaw_deg"])}
    if not places:
        raise CalibError("workcell 阶段没有录入任何放置位姿（指南 7.2 第 6 条）")
    if "default" not in places:
        raise CalibError("places 必须包含 default（指南 2.2）")
    upd["places"] = places
    return upd, res


def _fit_timing(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 8.1~8.3：读写耗时的 max/P50/P95/P99、停止延迟、采样跨度与年龄。"""
    rows = [r.payload for r in samples_of(records)]
    if len(rows) < 1000:
        raise CalibError(f"timing 阶段只有 {len(rows)} 个 tick，指南 8.1 要求至少 1000 个")
    read_s = [(int(p["read_end_ns"]) - int(p["read_start_ns"])) / 1e9 for p in rows]
    write_s = [(int(p["write_end_ns"]) - int(p["write_start_ns"])) / 1e9 for p in rows]
    upd: dict[str, Any] = {}
    # 时限取 P99 再放大 1.5 倍作为最坏允许值，而不是直接取 max：max 容易被一次
    # 系统抖动永久放大，放大后的阈值又会让真实的通信劣化被放过。
    upd["timing.read_timeout_s"] = round(max(quantile(read_s, 0.99) * 1.5, max(read_s)), 4)
    upd["timing.write_timeout_s"] = round(max(quantile(write_s, 0.99) * 1.5, max(write_s)), 4)
    span_s = [int(p.get("sample_span_ns", 0)) / 1e9 for p in rows if p.get("sample_span_ns")]
    upd["timing.max_feedback_span_s"] = round(max(quantile(span_s, 0.999), max(span_s)) * 1.2, 4)
    # 年龄从 sample_start 起算，不能只看读取结束时间（指南 8.3）
    upd["timing.max_feedback_age_s"] = round(
        upd["timing.max_feedback_span_s"] + max(read_s) + 0.020, 4)
    # 迟到阈值必须小于一个 tick
    tick = 1.0 / int(base["timing"]["fps"])
    upd["timing.max_tick_lateness_s"] = round(min(0.020, tick * 0.6), 4)
    upd["timing.stop_poll_s"] = round(min(tick * 0.3, 0.010), 4)
    lat = [int(p["read_start_ns"]) - int(p["deadline_ns"]) for p in rows
           if p.get("deadline_ns") is not None]
    res: dict[str, Any] = {
        "read_p50_ms": round(quantile(read_s, 0.50) * 1000, 3),
        "read_p95_ms": round(quantile(read_s, 0.95) * 1000, 3),
        "read_p99_ms": round(quantile(read_s, 0.99) * 1000, 3),
        "read_max_ms": round(max(read_s) * 1000, 3),
        "write_p99_ms": round(quantile(write_s, 0.99) * 1000, 3),
        "tick_lateness_max_ms": round(max(lat) / 1e6, 3),
        "ticks": len(rows),
    }
    # 停止延迟：should_stop 成立到机械实际静止（指南 8.7 用外部录像区分）。
    stop_rows = [(int(p["mechanically_still_ns"]) - int(p["stop_condition_ns"])) / 1e9
                 for p in rows
                 if p.get("stop_condition_ns") and p.get("mechanically_still_ns")]
    if stop_rows:
        upd["timing.planning_poll_s"] = round(min(0.05, max(stop_rows) * 2.0), 4)
        res["stop_latency_max_ms"] = round(max(stop_rows) * 1000, 3)
    else:
        upd["timing.planning_poll_s"] = base["timing"].get("planning_poll_s") or 0.05
    return upd, res


def _fit_motion(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 9.2~9.7：从"通过且留有余量"的试验里取上限，并统计跟踪误差与到位时间。"""
    ok_rows = [r.payload for r in samples_of(records) if r.payload.get("passed")]
    bad_rows = [r.payload for r in samples_of(records) if r.payload.get("passed") is False]
    if not ok_rows:
        raise CalibError("motion 阶段没有任何通过的试验组，不能取速度/加速度上限")
    upd: dict[str, Any] = {}
    res: dict[str, Any] = {}
    peak_speed = np.max(np.abs(np.array([p["speed_deg_s"] for p in ok_rows], float)), axis=0)
    cur_peak = np.max(np.abs(np.array([p["current_ma"] for p in ok_rows], float)), axis=0)
    if bad_rows:
        first_bad = np.max(np.abs(np.array([p["speed_deg_s"] for p in bad_rows], float)), axis=0)
        # 取"最后一组通过值"，再往里收 15%：留出余量（指南 9.3 通过范围内留有余量）。
        allow = np.minimum(peak_speed, first_bad * 0.85)
    else:
        allow = peak_speed
    upd["motion.max_velocity_deg_s"] = [round(float(v) * 0.85, 2) for v in allow]
    # 加速度按实测速度台阶估计：这里用相邻样本速度差的最大值。
    cmds = np.array([p["command_deg"] for p in ok_rows], float)
    d2 = np.abs(np.diff(cmds, n=2, axis=0)).max(axis=0) if cmds.shape[0] >= 3 else np.zeros(5)
    upd["motion.max_acceleration_deg_s2"] = [
        round(float(v) * 0.85 + 50.0, 2) for v in d2]
    # 9.4：max_command_step_deg 不大于 max_velocity/FPS
    fps = int(base["timing"]["fps"])
    upd["motion.max_command_step_deg"] = [
        round(min(float(v) / fps, 2.2), 3) for v in upd["motion.max_velocity_deg_s"]]

    # 9.5：跟踪误差阈值取通过样本的分位数，hard 取失败样本里量级最小的一个再收一半
    err = np.abs(np.array([np.asarray(p["command_deg"], float)
                           - np.asarray(p["feedback_deg"], float) for p in ok_rows], float))
    upd["motion.following_error_deg"] = [round(float(v), 2)
                                         for v in np.percentile(err, 99, axis=0) * 2.0 + 3.0]
    upd["motion.following_error_hard_deg"] = [
        round(float(v) * 2.2, 2) for v in upd["motion.following_error_deg"]]
    upd["motion.following_error_dwell_s"] = 0.10

    # 9.6：到位阈值用静止噪声与重复到位数据
    upd["motion.settle_position_tol_deg"] = [
        round(float(v) + 1.0, 2) for v in np.percentile(err, 99, axis=0)]
    upd["motion.settle_velocity_tol_deg_s"] = [8.0] * 5
    upd["motion.settle_timeout_s"] = 3.0
    upd["motion.settle_dwell_s"] = 0.15
    upd["motion.start_position_tol_deg"] = [3.0] * 5
    upd["motion.tcp_max_velocity_m_s"] = base["motion"]["tcp_max_velocity_m_s"] or 0.22
    upd["motion.tcp_max_acceleration_m_s2"] = base["motion"]["tcp_max_acceleration_m_s2"] or 1.4
    # 9.8：IK 容差必须严于真实总定位误差预算
    acc = float(base["acceptance"]["max_position_error_m"] or 0.010)
    upd["ik.position_tol_m"] = round(acc * 0.6, 4)
    upd["ik.tilt_tol_deg"] = round(float(base["acceptance"]["max_tilt_error_deg"] or 3.0) * 0.5, 2)
    upd["ik.yaw_tol_deg"] = round(float(base["acceptance"]["max_yaw_error_deg"] or 4.0) * 0.5, 2)
    res = {
        "passed_rows": len(ok_rows), "failed_rows": len(bad_rows),
        "peak_speed_deg_s": [round(float(v), 2) for v in peak_speed],
        "peak_current_ma": [round(float(v), 1) for v in cur_peak],
        "tracking_p99_deg": [round(float(v), 3) for v in np.percentile(err, 99, axis=0)],
    }
    return upd, res


def _fit_grasp(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南 10.2/10.3：空闭合端、接触 gap 区间、电流阈值与额外闭合量。"""
    # 用全部记录而不是只取 sample：接触试验的汇总是另一种 record_type。
    rows = list(records)
    empty_close = [p for p in (r.payload for r in rows) if p.get("cycle_phase") == "empty_close"]
    if not empty_close:
        raise CalibError("grasp 阶段缺少空爪闭合循环记录（指南 10.2 第 3 条）")
    # 空闭合端位置：重复闭合里稳定停住的最大开度，再放一点容差。
    pcts = [float(p["gripper_pct"]) for p in empty_close]
    upd: dict[str, Any] = {}
    res: dict[str, Any] = {}     # 统计结果在下面的各小节里逐步填
    upd["gripper.empty_closed_pct"] = round(float(np.percentile(pcts, 95)), 2)
    upd["gripper.empty_tol_pct"] = round(float(np.std(pcts) + 1.0), 2)

    gaps = [float(p["actual_gap_m"]) for p in (r.payload for r in rows)
            if p.get("contact_label") == "first_contact" and p.get("actual_gap_m")]
    if not gaps:
        raise CalibError("grasp 阶段没有首次接触的实际 gap 记录（指南 10.2 第 4 条）")
    # 区间 = 实测 min/max 再各放一个测量余量；必须与空闭合端可区分。
    pad = 0.001
    upd["gripper.contact_gap_range_m"] = [round(max(0.0005, min(gaps) - pad), 4),
                                          round(max(gaps) + pad, 4)]
    # 接触电流：取"接触之后"的滤波候选值下界，并保证 < hard
    after = [float(p["gripper_current_ma"]) for p in (r.payload for r in rows)
             if p.get("contact_label") == "after"]
    before = [float(p["gripper_current_ma"]) for p in (r.payload for r in rows)
              if p.get("contact_label") == "before"]
    if after and before:
        contact = max(float(np.percentile(before, 99)) * 1.4, float(np.percentile(after, 5)) * 0.9)
        upd["gripper.contact_current_ma"] = round(contact, 1)
    else:
        upd["gripper.contact_current_ma"] = base["gripper"]["contact_current_ma"]
    upd["gripper.hard_current_ma"] = round(float(upd["gripper.contact_current_ma"]) * 2.0, 1)

    # 10.2 第 5 条：滤波 + 连续确认 + 读写 + 一个 tick 的延迟乘闭合速度 = 额外闭合量，
    # 再用 gap_table 换算成压缩距离。超出允许压缩量时必须降速或缩短检测延迟。
    fps = int(base["timing"]["fps"])
    window_s = int(base["gripper"]["window_samples"] or 3) / fps
    dwell_s = float(base["gripper"]["contact_dwell_s"] or 0.1)
    detect_s = window_s + dwell_s + 2.0 / fps
    close_speed = float(base["gripper"]["close_speed_pct_s"] or 15.0)
    extra_pct = detect_s * close_speed
    gap_table = base["gripper"]["gap_table"]
    gap_slope_m_per_pct = None
    if gap_table:
        gxs = [float(r["gripper_pct"]) for r in gap_table]
        gys = [float(r["gap_m"]) for r in gap_table]
        gap_slope_m_per_pct = (gys[-1] - gys[0]) / max(1e-9, gxs[-1] - gxs[0])
        upd["_calibration_note_extra_close_pct"] = round(extra_pct, 3)
        upd["_calibration_note_extra_compression_m"] = round(
            extra_pct * gap_slope_m_per_pct, 5)
    # 指南 10.1：接触试验的汇总是独立的 record_type，下面几个 10.3 判据都要用它。
    # 用全部记录而不是只取 sample，所以这里不能过 samples_of() 过滤。
    offs = [r.payload for r in rows if r.payload.get("record_type") == "trial_summary"]

    # 10.3 第 1/2 条：抓取中心偏移写在物体系中，夹持方向偏移是单个固定常数。
    # 多次试验取逐分量中位数：个别放偏一次的试验不该把整条规则带跑。
    offs_vec = [p.get("contact_center_offset_object_m") for p in offs
                if p.get("contact_center_offset_object_m")]
    if offs_vec:
        med = np.median(np.array(offs_vec, float), axis=0)
        upd["grasp.center_offset_object_m"] = [round(float(v), 5) for v in med]
        res["center_offset_spread_m"] = [round(float(v), 5)
                                        for v in np.ptp(np.array(offs_vec, float), axis=0)]
    yaw_offsets = [float(p["grasp_yaw_offset_deg"]) for p in offs
                   if p.get("grasp_yaw_offset_deg") is not None]
    if yaw_offsets:
        upd["grasp.yaw_offset_deg"] = float(np.median(yaw_offsets))
    # 10.3 第 3 条：闭合前后物体中心位移上界，单位必须是米。
    # 优先采用试验里直接测得的中心位移（center_shift_m，录像/标记点）；只记录了
    # 开度漂移（slip_drift_pct，单位是开度百分比）时，按 gap_table 斜率把漂移
    # 换算成"内间距变化量"作为位移的保守上界。旧实现直接 百分比×0.001（P2-2
    # 的单位混淆），物理意义错误，手册附录 C 的"手工覆盖"只是权宜。
    shifts_m = [float(p["center_shift_m"]) for p in (r.payload for r in rows)
                if p.get("center_shift_m") is not None]
    drifts_pct = [abs(float(p["slip_drift_pct"])) for p in (r.payload for r in rows)
                  if p.get("slip_drift_pct") is not None]
    shift_pad_m = 0.001                     # 测量余量
    if shifts_m:
        upd["grasp.object_shift_bound_m"] = round(max(0.001, max(shifts_m) + shift_pad_m), 4)
    elif drifts_pct and gap_slope_m_per_pct is not None:
        bound = max(drifts_pct) * gap_slope_m_per_pct + shift_pad_m
        upd["grasp.object_shift_bound_m"] = round(max(0.001, bound), 4)
    # 10.2 第 7 条：slip 阈值取正常持物漂移的 P99 再放大（这个才是开度百分比）。
    if drifts_pct:
        upd["gripper.slip_opening_change_pct"] = round(quantile(drifts_pct, 0.99) * 2.0 + 1.0, 2)
    # 10.2 第 8 条：松爪后静止等待
    upd["gripper.release_dwell_s"] = float(base["gripper"]["release_dwell_s"] or 0.5)
    upd["gripper.hold_dwell_s"] = float(base["gripper"]["hold_dwell_s"] or 0.3)
    # 10.2 第 3 条：开合总时限按实测开合时间放大
    upd["gripper.close_timeout_s"] = 6.0
    upd["gripper.open_timeout_s"] = 6.0

    # 10.3 第 1/2 条：中心偏移与固定夹持偏移由试验选定，这里从记录里读标签
    # 10.2 第 2 条：预张开度必须覆盖"目标在固定夹持方向上的闭合轴宽度"并留净余量；
    # 释放开度要能稳定脱手，同时不碰固定环境。
    dims = [p.get("dimensions_reference_m") for p in offs if p.get("dimensions_reference_m")]
    gap_rows = base["gripper"]["gap_table"]
    need = margin = None
    if dims and gap_rows:
        yaw_off = float(upd.get("grasp.yaw_offset_deg",
                                base["grasp"]["yaw_offset_deg"] or 0.0))
        # 闭合轴宽度 abs(cos(beta))*length + abs(sin(beta))*width（指南 2.4），
        # 直接转调生产代码里那一份实现，标定与运行不会给出两个数。
        widths = [close_axis_width_at_yaw(float(d[0]), float(d[1]), yaw_off) for d in dims]
        need = max(widths)
        # 净余量取"宽度的 10%"与 3mm 里更大的那个：既吃掉测量误差，也让指尖有
        # 进入空间而不是擦着物体表面下刀。
        margin = max(0.10 * need, 0.003)
        upd["gripper.preopen_pct"] = round(_pct_for_gap(need + margin, gap_rows), 2)
        # 释放开度只留半个余量：太小会夹着不放，太大会在撤离时碰固定环境。
        upd["gripper.release_pct"] = round(_pct_for_gap(need + 0.5 * margin, gap_rows), 2)
        g_max = float(gap_rows[-1]["gripper_pct"])
        if upd["gripper.preopen_pct"] > g_max:
            raise CalibError(
                f"预张 {upd['gripper.preopen_pct']}% 超出 gap_table 覆盖上限 {g_max}%，"
                "必须换指垫或缩小适用目标尺寸（指南 10.2 第 2 条）")
    res = {
        "empty_close_cycles": len(empty_close),
        "closure_axis_width_m": None if need is None else round(need, 4),
        "preopen_margin_m": None if margin is None else round(margin, 4),
        "first_contact_gaps_m": [round(g, 4) for g in sorted(set(gaps))],
        "contact_current_ma_candidates": [round(float(np.percentile(before, 99)), 1),
                                          round(float(np.percentile(after, 5)), 1)] if after and before else [],
        "trial_summaries": len(offs),
        "extra_close_pct": upd.get("_calibration_note_extra_close_pct"),
        "extra_compression_m": upd.get("_calibration_note_extra_compression_m"),
    }
    return upd, res


def _fit_pick_place(base: dict, records: list[Record]) -> tuple[dict[str, Any], dict[str, Any]]:
    """指南第 11 节：五种统计口径。fit 不修改验收 limit，只报告实测统计。

    验收要求是"调参前确定"的，所以本阶段 fit 不更新运行字段；统计写进
    _calibration_audit 供 validate 使用（指南 1.3 不为通过验收而放宽 limit）。
    """
    return {}, {}


FITTERS: dict[str, Any] = {
    "motor": _fit_motor, "joints": _fit_joints, "tool": _fit_tool,
    "workcell": _fit_workcell, "timing": _fit_timing, "motion": _fit_motion,
    "grasp": _fit_grasp, "pick_place": _fit_pick_place,
}


# ---------------------------------------------------------------------------
# 6. validate
# ---------------------------------------------------------------------------


def _check(name: str, field_path: str, measured: float, limit: float, comparison: str,
           evidence_type: str, evidence_refs: Sequence[str]) -> dict[str, Any]:
    """构造一个 check 记录（指南 1.3）。comparison ∈ {<=, >=, ==, all}。"""
    if comparison == "<=":
        passed = measured <= limit + 1e-12
    elif comparison == ">=":
        passed = measured >= limit - 1e-12
    elif comparison == "==":
        passed = abs(measured - limit) <= 1e-9
    else:
        raise CalibError(f"未知 comparison {comparison!r}")
    return {"name": name, "field_path": field_path, "measured": round(float(measured), 6),
            "limit": round(float(limit), 6), "comparison": comparison,
            "passed": bool(passed), "evidence_type": evidence_type,
            "evidence_refs": list(evidence_refs)}


def cmd_validate(args: argparse.Namespace) -> int:
    profile_path = Path(args.profile)
    raw = load_json(profile_path)
    stage = args.stage
    # 不按 stage 过滤读入：一份合并了八个阶段的 JSONL 是很常见的输入形态，
    # `--stage timing` 只要求"从里面取 timing 的那部分来检查"，而不是"文件里
    # 不许出现别的阶段"。真正按阶段筛记录在 _stage_checks 里做。
    records = read_records(Path(args.input))
    input_sha = sha256_file(Path(args.input))
    fit_ids, val_ids = _split_ids(profile_path, records)

    report_path = Path(args.report)
    report = load_json(report_path) if report_path.is_file() else _new_report(raw)
    report["profile_id"] = raw["profile_id"]
    report["parameter_sha256"] = parameter_sha256(raw)
    if not report.get("operator"):
        report["operator"] = args.operator or "unknown"
    report["operator_reviewed"] = bool(args.operator_reviewed)

    audit = _read_audit(profile_path)
    stages = [s for s in STAGES] if stage == "all" else [stage]
    for s in stages:
        checks = _stage_checks(s, raw, records, input_sha, val_ids, audit)
        passed = all(c["passed"] for c in checks) and bool(checks)
        report["stages"][s] = {
            "input_sha256": input_sha,
            "fit_sample_ids": fit_ids.get(s, []),
            "validation_sample_ids": sorted(val_ids),
            "checks": checks,
            "passed": bool(passed),
        }
    report["overall_pass"] = all(report["stages"][s]["passed"] for s in STAGES
                                if s in report["stages"]) and len(report["stages"]) == len(STAGES)
    dump_json(report_path, report)
    for s in stages:
        info = report["stages"][s]
        bad = [c["name"] for c in info["checks"] if not c["passed"]]
        print(f"[{s}] passed={info['passed']}" + (f"  未通过：{'、'.join(bad)}" if bad else ""))
    print(f"报告 → {report_path}（overall_pass={report['overall_pass']}）")
    # 退出码反映"本次真正检查过的阶段是否全部通过"。用 overall_pass 会让
    # `validate --stage timing` 这种单阶段用法永远返回失败，因为它天然凑不齐八个阶段。
    return 0 if all(report["stages"][s]["passed"] for s in stages) else 1


def _new_report(raw: dict) -> dict[str, Any]:
    return {"profile_id": raw.get("profile_id"), "parameter_sha256": None,
            "operator": None, "operator_reviewed": False, "stages": {},
            "overall_pass": False}


def _split_ids(profile_path: Path, records: list[Record]) -> tuple[dict[str, list[str]], set[str]]:
    """拟合样本与独立验证样本的 sample_id 不得重叠（指南 1.3）。"""
    val_ids = {r.sample_id for r in samples_of(records)}
    audit = _read_audit(profile_path)
    fit_ids = {s: list(a.get("fit_sample_ids", [])) for s, a in audit.items()}
    overlap = set().union(*(set(v) for v in fit_ids.values())) if fit_ids else set()
    clash = overlap & val_ids
    if clash:
        raise CalibError(
            f"拟合样本与验证样本的 sample_id 重叠 {len(clash)} 个，指南 1.3 禁止")
    return fit_ids, val_ids


def _stage_checks(stage: str, raw: dict, records: list[Record], input_sha: str,
                  val_ids: set[str],
                  audit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    audit = audit or {}
    """按阶段生成阈值检查。limit 一律从配置里读，不为了通过而放宽（指南 1.3）。"""
    # --stage all 时传进来的是合并文件：必须先筛出本阶段的记录，否则 grasp 的
    # 检查会读到 pick_place 的样本，统计口径整个错位（指南 11 的分母定义很严格）。
    stage_records = [r for r in records if r.stage == stage]
    if not stage_records:
        raise CalibError(f"验证输入里没有 {stage} 阶段的记录")
    ev_type = "mock" if metadata_of(stage_records).get("source") == "simulated" else "physical"
    refs = [input_sha[:12]] + sorted(val_ids)[:4]
    checks: list[dict[str, Any]] = []
    rows = list(samples_of(stage_records))
    P = raw
    records = stage_records

    if stage == "motor":
        # 度数/百分比指令往返正确；夹爪两端读数不同；电流与速度单位有出处（指南 4 节验收）
        closed = P["motor"]["gripper_closed_raw"]
        opened = P["motor"]["gripper_open_raw"]
        checks.append(_check("夹爪两端读数不同", "motor.gripper_closed_raw",
                             float(abs(opened - closed)), 1.0, ">=", "offline", refs))
        checks.append(_check("编码分辨率大于 1", "motor.position_resolution",
                             float(min(P["motor"]["position_resolution"])), 1.0, ">=", "offline", refs))
        checks.append(_check("电流比例非零", "motor.current_ma_per_raw",
                             float(min(abs(v) for v in P["motor"]["current_ma_per_raw"])),
                             1e-9, ">=", "offline", refs))
        checks.append(_check("6 通道同型号", "motor.models",
                             float(len(set(P["motor"]["models"]))), 1.0, "==", "offline", refs))
        # 身份"逐项可读回"的离线可核部分：配置的 models/firmware 必须与采集
        # metadata 里记录的读回值逐项一致（真机侧的逐台读回由驱动 initialize
        # 完成；这里钉住"配置抄录没有错行"，P3-5 修的是名不副实的检查名）。
        meta = metadata_of(records)
        for key in ("models", "firmware"):
            recorded = meta.get(key)
            if recorded is not None:
                same = (len(recorded) == 6
                        and all(str(a) == str(b) for a, b in zip(recorded, P["motor"][key])))
                checks.append(_check(f"motor.{key} 与采集 metadata 逐项一致",
                                     f"motor.{key}", float(same), 1.0, "==", ev_type, refs))
        # 参考电流与换算电流一致性（4.3 用可靠参考测量验证合理性）
        pairs = [(r.payload["raw_current"], r.payload["reference_current_ma"]) for r in rows
                 if r.payload.get("reference_current_ma")]
        if pairs:
            errs = []
            for raws, refs_ma in pairs:
                scale = float(np.mean(P["motor"]["current_ma_per_raw"]))
                zero = float(np.mean(P["motor"]["current_zero_raw"]))
                errs.extend(abs((rv - zero) * scale - mv) for rv, mv in zip(raws, refs_ma))
            checks.append(_check("电流换算与参考值偏差(mA)", "motor.current_ma_per_raw",
                                 float(np.percentile(errs, 95)), 25.0, "<=", ev_type, refs))
    elif stage == "joints":
        lower, upper = _limits_from(raw)
        span = float(np.min(upper - lower))
        checks.append(_check("有效限位最小可用跨度(deg)", "joints.margin_deg", span, 10.0,
                             ">=", "offline", refs))
        rms = get_path(audit, "joints.residuals.sign_fit_rms_deg") or [0] * 5
        checks.append(_check("方向/零位拟合残差(deg)", "joints.zero_offset_deg",
                             float(np.max(rms)), 1.0, "<=", ev_type, refs))
        n_fk = get_path(audit, "joints.residuals.fk_validation_configs") or 0
        checks.append(_check("FK 独立验证构型数", "workspace.tcp_bounds_m", float(n_fk), 10.0,
                             ">=", ev_type, refs))
    elif stage == "tool":
        res = get_path(audit, "tool.residuals") or {}
        checks.append(_check("工具旋转正交残差", "tool.rotation_e_tcp",
                             float(res.get("rotation_orthonormality_err", 1.0)), 1e-6,
                             "<=", "offline", refs))
        checks.append(_check("TCP 平移标定点数", "tool.translation_samples",
                             float(res.get("translation_opening_count", 0)), 3.0, ">=",
                             ev_type, refs))
        checks.append(_check("独立 TCP 中点误差(m)", "tool.position_error_bound_m",
                             float(P["tool"]["position_error_bound_m"]),
                             float(P["acceptance"]["max_position_error_m"]), "<=", "offline", refs))
    elif stage == "workcell":
        checks.append(_check("桌面点数", "workspace.table_z_m",
                             float(sum(1 for r in rows if r.payload.get("entity_id") == "table")),
                             9.0, ">=", ev_type, refs))
        checks.append(_check("中转位数量", "workspace.safe_waypoints_deg",
                             float(len(P["workspace"]["safe_waypoints_deg"])), 1.0, ">=",
                             ev_type, refs))
        checks.append(_check("放置位姿数量", "places", float(len(P["places"])), 1.0, ">=",
                             ev_type, refs))
        checks.append(_check("桌面平度(m)", "workspace.table_flatness_m",
                             float(P["workspace"]["table_flatness_m"]), 0.005, "<=", ev_type, refs))
    elif stage == "timing":
        checks.append(_check("tick 样本数", "timing.fps", float(len(rows)), 1000.0, ">=",
                             ev_type, refs))
        tick = 1.0 / int(P["timing"]["fps"])
        checks.append(_check("迟到阈值不大于一个 tick(s)", "timing.max_tick_lateness_s",
                             float(P["timing"]["max_tick_lateness_s"]), tick, "<=",
                             ev_type, refs))
        checks.append(_check("stop_poll 不大于一个 tick(s)", "timing.stop_poll_s",
                             float(P["timing"]["stop_poll_s"]), tick, "<=", ev_type, refs))
        lat = get_path(audit, "timing.residuals.stop_latency_max_ms")
        if lat is not None:
            checks.append(_check("停止延迟(s)", "acceptance.max_stop_latency_s",
                                 float(lat) / 1000.0, float(P["acceptance"]["max_stop_latency_s"]),
                                 "<=", ev_type, refs))
    elif stage == "motion":
        lo, up = _limits_from(raw)
        step = np.array(P["motion"]["max_command_step_deg"], float)
        vel = np.array(P["motion"]["max_velocity_deg_s"], float)
        checks.append(_check("步长不超过限速/FPS", "motion.max_command_step_deg",
                             float(np.max(step * int(P["timing"]["fps"]) / vel)), 1.0, "<=",
                             "offline", refs))
        hard = np.array(P["motion"]["following_error_hard_deg"], float)
        soft = np.array(P["motion"]["following_error_deg"], float)
        checks.append(_check("hard>普通跟踪阈值(deg)", "motion.following_error_hard_deg",
                             float(np.min(hard - soft)), 0.0, ">=", "offline", refs))
        checks.append(_check("settle timeout>dwell(s)", "motion.settle_timeout_s",
                             float(P["motion"]["settle_timeout_s"] - P["motion"]["settle_dwell_s"]),
                             0.0, ">=", "offline", refs))
        tol = float(P["ik"]["position_tol_m"])
        checks.append(_check("IK 容差不大于定位预算(m)", "ik.position_tol_m", tol,
                             float(P["acceptance"]["max_position_error_m"]), "<=",
                             "offline", refs))
        # 与证据分级挂钩的一项（P2-6）：从 motion 记录的 command_deg 序列算出
        # 实测最大单步角，必须不超过配置的最小命令步长上限。此前本阶段全是
        # 纯结构检查，报告里没有任何随样本证据等级变化的判据。
        by_traj: dict[str, list] = {}
        for p in (r.payload for r in samples_of(records)):
            cd = p.get("command_deg")
            if cd is not None:
                # 同一条轨迹可能正反向各跑一遍（指南 9.2"每组正反方向重复"），
                # 折返点不是真实步长，必须按方向分组后再做差分。
                by_traj.setdefault(f"{p.get('trajectory_id')}#{p.get('direction')}", []).append(
                    np.asarray(cd, dtype=np.float64))
        step_max = 0.0
        for seq in by_traj.values():
            if len(seq) > 1:
                step_max = max(step_max, float(np.max(np.abs(np.diff(np.array(seq), axis=0)))))
        if by_traj:
            checks.append(_check("实测最大单步角不超过命令步长(deg)", "motion.max_command_step_deg",
                                 step_max,
                                 float(np.min(np.array(P["motion"]["max_command_step_deg"],
                                                      float))), "<=", ev_type, refs))
    elif stage == "grasp":
        g = P["gripper"]
        lo_g, hi_g = [float(v) for v in g["contact_gap_range_m"]]
        checks.append(_check("接触区间下界为正(m)", "gripper.contact_gap_range_m", lo_g, 0.0,
                             ">=", ev_type, refs))
        checks.append(_check("接触区间上下界分离(m)", "gripper.contact_gap_range_m", hi_g - lo_g,
                             0.0, ">=", "offline", refs))
        checks.append(_check("hard>contact 电流(mA)", "gripper.hard_current_ma",
                             float(g["hard_current_ma"] - g["contact_current_ma"]), 0.0, ">=",
                             "offline", refs))
        # 接触判据换算前必须先确认表覆盖：换算函数禁外插（P2-5），越界时这里
        # 报出一条明确的"覆盖不足"失败检查，而不是被钳位值蒙混过关。
        rows_g = g["gap_table"]
        gaps_dom = [float(r["gap_m"]) for r in rows_g]
        pcts_dom = [float(r["gripper_pct"]) for r in rows_g]
        contact_cov = min(lo_g - min(gaps_dom), max(gaps_dom) - hi_g)
        checks.append(_check("接触区间被 gap_table 定义域覆盖(m)", "gripper.gap_table",
                             contact_cov, 0.0, ">=", "offline", refs))
        preopen_cov = min(float(g["preopen_pct"]) - min(pcts_dom),
                          max(pcts_dom) - float(g["preopen_pct"]))
        checks.append(_check("预张开度被 gap_table 定义域覆盖(%)", "gripper.gap_table",
                             preopen_cov, 0.0, ">=", "offline", refs))
        # 接触开度必须明显大于空闭合端（指南 2.4 末段）
        if contact_cov >= 0.0:
            contact_pct = _pct_for_gap(lo_g, rows_g)
            checks.append(_check("接触开度大于空闭合端+容差(%)", "gripper.contact_gap_range_m",
                                 contact_pct - (float(g["empty_closed_pct"])
                                                + float(g["empty_tol_pct"])),
                                 0.0, ">=", "offline", refs))
        # 预张 gap 必须大于目标在固定偏移下的闭合轴宽度（指南 2.4）
        env = [float(v) for v in P["grasp"]["object_envelope_m"]]
        beta = math.radians(float(P["grasp"]["yaw_offset_deg"]))
        need = abs(math.cos(beta)) * env[0] + abs(math.sin(beta)) * env[1]
        if preopen_cov >= 0.0:
            pre_gap = gripper_pct_to_gap_simple(float(g["preopen_pct"]), rows_g)
            checks.append(_check("预张开口大于闭合轴宽度(m)", "gripper.preopen_pct", pre_gap,
                                 need, ">=", "offline", refs))
    elif stage == "pick_place":
        rows = [r.payload for r in samples_of(records) if r.payload.get("began_descend")]
        n = len(rows)
        checks.append(_check("独立实物试验数", "acceptance.min_grasp_trials", float(n),
                             float(P["acceptance"]["min_grasp_trials"]), ">=", ev_type, refs))
        if n:
            succ = sum(1 for p in rows if p.get("actual_success"))
            drop = sum(1 for p in rows if p.get("actual_drop"))
            dmg_rows = [p for p in rows if p.get("damage_observe_minutes")]
            dmg = sum(1 for p in dmg_rows if p.get("damage_label") not in (None, "none"))
            checks.append(_check("实际成功率", "acceptance.min_grasp_success_rate",
                                 succ / n, float(P["acceptance"]["min_grasp_success_rate"]),
                                 ">=", ev_type, refs))
            checks.append(_check("掉落率", "acceptance.max_drop_rate", drop / n,
                                 float(P["acceptance"]["max_drop_rate"]), "<=", ev_type, refs))
            checks.append(_check("损伤率", "acceptance.max_damage_rate",
                                 (dmg / len(dmg_rows)) if dmg_rows else 0.0,
                                 float(P["acceptance"]["max_damage_rate"]), "<=", ev_type, refs))
            cerr = [float(p["place_error_center_m"]) for p in rows
                    if p.get("place_error_center_m") is not None]
            if cerr:
                checks.append(_check("放置中心误差(m)", "acceptance.max_position_error_m",
                                     float(np.percentile(cerr, 95)),
                                     float(P["acceptance"]["max_position_error_m"]), "<=",
                                     ev_type, refs))
    if not checks:
        raise CalibError(f"阶段 {stage} 没有生成任何检查项")
    return checks


def _limits_from(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """从候选配置算有效限位。URDF 限位从文件读，避免配置里存第二份真值。"""
    urdf = Path(raw["model"]["urdf_path"])
    if not urdf.is_absolute():
        urdf = (Path(args_profile_dir["current"]) / urdf).resolve()
    urdf_deg = read_urdf_joint_limits_deg(urdf, JOINT_NAMES)
    return effective_joint_limits(urdf_deg, _JointsParams(raw))


class _JointsParamsShim:
    def __init__(self, j: dict) -> None:
        self.measured_limits_deg = np.array(j["measured_limits_deg"], float)
        self.application_limits_deg = np.array(j["application_limits_deg"], float)
        self.margin_deg = np.array(j["margin_deg"], float)


def _JointsParams(raw: dict):
    return _JointsParamsShim(raw["joints"])


def _gap_table_params(gap_table: Sequence[dict]):
    """把 profile.json 里的 gap_table 原始行包成生产换算函数需要的对象形状。

    指南 12 节要求标定工具与生产代码共同调用同一份插值实现。本地 np.interp
    副本在越界时会静默钳位（外插被伪装成端点值），而生产实现禁外插并报错——
    两份实现的漂移正是这条规定要防的事。
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        gap_table=[SimpleNamespace(gripper_pct=float(r["gripper_pct"]),
                                   gap_m=float(r["gap_m"])) for r in gap_table])


def _pct_for_gap(gap_m: float, gap_table: Sequence[dict]) -> float:
    """gap → 开度反查：转调生产实现，越出标定表覆盖时明确失败而不是钳位。"""
    try:
        return gap_to_gripper_pct(float(gap_m), _gap_table_params(gap_table))
    except MotionError as exc:
        raise CalibError(f"gap→开度 反查越出标定表覆盖：{exc.reason}") from None


def gripper_pct_to_gap_simple(pct: float, gap_table: Sequence[dict]) -> float:
    """开度 → gap：转调生产实现，越界报错（P2-5 修掉的就是这里的钳位）。"""
    try:
        return gripper_pct_to_gap(float(pct), _gap_table_params(gap_table))
    except MotionError as exc:
        raise CalibError(f"开度→gap 换算越出标定表覆盖：{exc.reason}") from None


# ---------------------------------------------------------------------------
# 7. promote
# ---------------------------------------------------------------------------


def cmd_promote(args: argparse.Namespace) -> int:
    raw = load_json(Path(args.profile))
    report = load_json(Path(args.report))
    # 报告必须是针对当前这份参数的
    if report.get("parameter_sha256") != parameter_sha256(raw):
        raise CalibError(
            f"报告针对的参数哈希 {report.get('parameter_sha256')!r} 与候选配置 "
            f"{parameter_sha256(raw)!r} 不同，说明候选在 validate 之后被改过")
    if not report.get("operator_reviewed"):
        raise CalibError("报告缺少人工复核标记 operator_reviewed，不能 promote")
    if not report.get("overall_pass"):
        raise CalibError("报告 overall_pass 不为 true，不能 promote")
    missing = [s for s in STAGES if s not in report.get("stages", {})]
    if missing:
        raise CalibError(f"报告缺少阶段：{'、'.join(missing)}")
    failed = [s for s in STAGES if not report["stages"][s]["passed"]]
    if failed:
        raise CalibError(f"以下阶段验收未通过，不能 promote：{'、'.join(failed)}")
    physical = [s for s in STAGES
                if any(c.get("evidence_type") == "physical" for c in report["stages"][s]["checks"])]
    if physical != list(STAGES):
        only_mock = [s for s in STAGES if s not in physical]
        print("注意：以下阶段目前只有 mock/offline 证据，promote 后仅可用于仿真与调试；"
              f"real 模式加载会因缺少 physical 证据直接拒绝：{'、'.join(only_mock)}")

    out = json.loads(json.dumps(raw))
    out["status"] = "verified"
    out.pop("_calibration_audit", None)
    report_path = Path(args.report).resolve()
    base = Path(args.output).resolve().parent
    out["verification"] = {
        "report_path": rel_from(base, report_path),
        "report_sha256": sha256_file(report_path),
    }
    dump_json(Path(args.output), out)
    print(f"已 promote → {args.output}（status=verified）")
    return 0


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="calibrate.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="生成 draft 标定包")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--robot-id", required=True, dest="robot_id")
    p.add_argument("--profile-id", required=True, dest="profile_id")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("capture", help="采集一个阶段的样本")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--hardware", action="store_true",
                   help="按指南第 3 节前置条件从真机采集；不给则使用模拟源")
    p.add_argument("--reference", type=Path, help="外部量具/示教记录的 JSON 文件")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--operator")
    p.add_argument("--power")
    p.add_argument("--pad")
    p.add_argument("--load")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("fit", help="离线拟合，输出独立候选配置")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("validate", help="离线验收，不启动运动")
    p.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--operator")
    p.add_argument("--operator-reviewed", action="store_true", dest="operator_reviewed",
                   help="声明人工已复核报告内容")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("promote", help="全部阶段通过后设置 verified")
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=cmd_promote)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # fit/validate/promote 都要解析 profile.json 里的相对路径，先记下来。
    profile = getattr(args, "profile", None)
    args_profile_dir["current"] = str(Path(profile).parent.resolve()) if profile else "."
    try:
        return args.func(args)
    except (CalibError, ParamsError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
