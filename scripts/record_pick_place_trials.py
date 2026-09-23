#!/usr/bin/env python3
"""抓放验收入口工具（P4 §2.8 / R08）：在 **draft 候选 profile** 上驱动 ArmController
执行经审查的抓取—放置试验，逐次导出 `scripts/calibrate.py validate --stage pick_place`
可接受的 hardware JSONL，补齐"首次 verified 所需 pick_place 真实证据此前没有产生路径"的缺口。

固定 CLI（P4 §2.8.3 逐字）：

    record_pick_place_trials.py --profile PATH --trials CSV \\
        --output JSONL --operator NAME --vision-config PATH

设计边界（贯穿 P4 §3）：

* 生产 `main` 永远要求 verified，本工具是标定链的一部分，**不绕过生产门槛**：
  profile 走 ``load_calibration_motion_params``（接受 draft/verified、拒绝 simulation、
  任何 null/未测物理字段一律拒绝），失败即退出。
* 装配复用 P1 默认安全启动：``Sts3215MotorController(params)``（构造即握手、D17 六轴
  卸力准入）→ ``ArmController`` → ``cold_start_recover`` → ``vision.configure`` →
  ``vision.init``。**不运行生产 Plan 的无限任务循环**。
* 对 vision 桩的依赖 = 真实调用：当前视觉尚未实现，``vision.init`` 如实抛
  ``VisionHardError`` 而退出——这是正确行为，工具**不内置任何回退替身**。测试通过替换
  模块级工厂 / 属性注入替身，生产路径不触碰。
* 一次只执行清单中一个 target/place_id；**每一次都要操作者明确开始**（stdin 确认），
  EOF / Ctrl-C 干净退出，不自动接下一次。工具失败（运动异常 / 现场终止）不自动恢复、
  终止不自动卸力（沿用运动既有保持 + terminate 语义）。

CSV 列口径（列名固定，前六个为必需；多余列忽略）：

    trial_id            本次试验唯一标识（重复即拒绝，且早于连接驱动）
    sample_id           本条样本唯一标识（写入 Record.sample_id；重复即拒绝）
    place_id            放置位姿 id，必须 ∈ profile.places（未知即拒绝，早于连接驱动）
    grade               本次样本分级（A/B/C，人工真值）
    sample_truth        样本真值 / 身份说明（如目标物体标识、量具编号）
    scene               场景说明（背景 / 光照 / 摆位等，供事后复核）

JSONL 与逐字段（P4 §2.8.3）：一个 run 首行为 metadata（record_type=metadata、
source=hardware、operator、profile 路径与 parameter_sha256、URDF/电机校准实算资源哈希、
code_commit），其后每试验一条 sample（record_type=sample，字段口径见标定指南 §11）。
``trial_summary`` 不作为 pick_place 统计分母。began_descend 依 §2.8.2 由运动 trace 的
descent_submit 事件导出（submitted→true、rejected/无事件→false、unknown→null +
descent_evidence="unknown" 保留待操作者补充）。无实测的 feedback_span/stop_latency 记 null，
不伪填 0。

**began_descend=unknown 的修订工作流（工具不做在线编辑）**：
工具只如实导出 ``began_descend=null, descent_evidence="unknown"`` 并保留整条样本；
操作者结合现场/视频确认后，**人工按 JSONL 修改**该行——把 ``began_descend`` 定值
（true/false）并把 ``descent_evidence`` 改为与之一致的 ``manual_observed``（或据 trace
改回 submitted/rejected）——随后重跑 ``calibrate.py validate --stage pick_place``：
仍为 null 或与其不一致者，validate 的完整性关卡会判报告不通过（不从分母删除，令报告失败）。
这条"人工改值 + validate 复算拒绝不一致/仍 unknown"即为修订闭环。

退出码：``0`` 正常结束（含操作者主动停止采集）；``2`` 连接驱动**之前**的输入/配置错误
（缺列 / 坏 CSV / 重复 trial_id / 未知 place / profile 未测完整参数无法加载 / 导出结构不完整）；
``3`` 连接驱动之后的运行期硬故障（vision/motion 现场终止）。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from configs.motion_params import (  # noqa: E402
    ParamsError,
    load_calibration_motion_params,
    parameter_sha256,
)
from main import cold_start_recover, thresholds_from_params  # noqa: E402
from qingyun.grabbing import vision  # noqa: E402
from qingyun.grabbing.arm_control import (  # noqa: E402
    ArmController,
    TRACE_DESCENT_SUBMIT,
)
from qingyun.grabbing.executor import (  # noqa: E402
    DESCENT_REJECTED,
    DESCENT_SUBMITTED,
    DESCENT_UNKNOWN,
)
from qingyun.grabbing.motor_control import Sts3215MotorController  # noqa: E402
from qingyun.grabbing.vision import VisionHardError  # noqa: E402

# 与 validate 双侧共用的导出/校验器：同一份格式与完整性清单，避免手工 JSONL 绕过。
from calibrate import (  # noqa: E402
    Record,
    append_records,
    pick_place_integrity_checks,
    sha256_file,
    time_ns,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "trial_id", "sample_id", "place_id", "grade", "sample_truth", "scene",
)

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUNTIME = 3

# 运行状态：工具进程内单调递增的试验序号，写入 metadata 供报告引用。
_TOOL_NAME = "record_pick_place_trials"


class InputError(RuntimeError):
    """连接驱动之前的输入/配置错误 → 退出码 2。"""


# ---------------------------------------------------------------------------
# 注入点：默认绑定真实实现；测试替换这些模块级名字即可脱离硬件/视觉。
# ---------------------------------------------------------------------------


def make_motor(params):  # pragma: no cover - 真实硬件路径，测试注入替身
    """P1 默认安全启动：构造即握手，D17 六轴卸力准入由驱动内部保证。"""
    return Sts3215MotorController(params)


def make_arm(motor, params):  # pragma: no cover - 真实硬件路径
    """抓放 trace 默认开启（enable_motion_trace=True），供 get_last_motion_trace 导出。"""
    return ArmController(motor, params)


def _prompt(text: str) -> str:  # pragma: no cover - 交互入口，测试注入脚本化替身
    return input(text)


def _confirm_start(trial_row: dict) -> bool:
    """操作者明确开始下一次：仅 y/yes 返回 True；其他答复返回 False（结束采集）。

    EOF / Ctrl-C 不在此捕获，由调用方映射为干净退出——工具不自动接下一次。
    """
    ans = _prompt(
        f"\n[{_TOOL_NAME}] trial_id={trial_row['trial_id']!r} "
        f"place_id={trial_row['place_id']!r}（grade={trial_row['grade']!r}，"
        f"truth={trial_row['sample_truth']!r}）。确认开始这一次？输入 y/yes 继续，"
        f"其他结束：").strip().lower()
    return ans in ("y", "yes")


def git_commit() -> str:
    """metadata.code_commit：git rev-parse HEAD 实际值；取不到如实写 unknown，不伪造。"""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and commit else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------------------
# 1. 输入加载与"连接驱动前"的严格校验
# ---------------------------------------------------------------------------


def _load_profile(profile_path: Path) -> tuple[dict, object]:
    """返回 (原始 profile dict, 完整 MotionParams)。未测完整参数在此失败（退出码 2）。"""
    if not profile_path.is_file():
        raise InputError(f"profile 不存在：{profile_path}")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"profile 不是合法 JSON：{profile_path}: {exc}") from exc
    try:
        # draft 候选完整加载：拒绝 simulation、拒绝 null/未测物理字段（P4 §2.8.1）。
        params = load_calibration_motion_params(profile_path)
    except ParamsError as exc:
        raise InputError(f"draft 参数未测完整，无法用于抓放验收：{exc}") from exc
    return raw, params


def load_trials(trials_path: Path, known_places: Sequence[str]) -> list[dict[str, str]]:
    """读 trials CSV，做"连接驱动前"的结构校验：必需列齐、trial_id/sample_id 唯一、
    place_id ∈ profile.places。任一不满足抛 InputError（退出码 2）。"""
    if not trials_path.is_file():
        raise InputError(f"trials CSV 不存在：{trials_path}")
    with trials_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise InputError(
                f"trials CSV 缺必需列 {missing}；必需列为 {list(REQUIRED_COLUMNS)}")
        rows: list[dict[str, str]] = []
        for lineno, row in enumerate(reader, start=2):
            for col in REQUIRED_COLUMNS:
                val = (row.get(col) or "").strip()
                if not val:
                    raise InputError(f"trials CSV 第 {lineno} 行列 {col!r} 为空")
            rows.append({k: (v if v is not None else "") for k, v in row.items()
                         if k is not None})
    if not rows:
        raise InputError("trials CSV 没有任何数据行")
    seen_trial: set[str] = set()
    for row in rows:
        tid = row["trial_id"].strip()
        if tid in seen_trial:
            raise InputError(f"trials CSV 出现重复 trial_id：{tid!r}")
        seen_trial.add(tid)
    seen_sample: set[str] = set()
    for row in rows:
        sid = row["sample_id"].strip()
        if sid in seen_sample:
            raise InputError(f"trials CSV 出现重复 sample_id：{sid!r}")
        seen_sample.add(sid)
    known = set(known_places)
    for row in rows:
        pid = row["place_id"].strip()
        if pid not in known:
            raise InputError(
                f"未知 place_id {pid!r}（trial_id={row['trial_id']!r}）；"
                f"profile.places 可用：{sorted(known)}")
    return rows


# ---------------------------------------------------------------------------
# 2. trace → began_descend / descent_evidence（P4 §2.8.2）
# ---------------------------------------------------------------------------


def _descent_from_trace(trace: Sequence) -> tuple[bool | None, str]:
    """从本次调用的运动 trace 导出下降证据。unknown 保留 null，绝不折算成 False 或丢弃。"""
    descent = next((e for e in trace if e.kind == TRACE_DESCENT_SUBMIT), None)
    if descent is None:
        return False, "none"                 # 无下降提交事件：明确未开始下发
    if descent.outcome == DESCENT_SUBMITTED:
        return True, "submitted"             # 驱动调用成功返回
    if descent.outcome == DESCENT_REJECTED:
        return False, "rejected"             # 驱动写前明确拒绝
    if descent.outcome == DESCENT_UNKNOWN:
        return None, "unknown"               # 通信异常/中断：不能证明是否写入
    return False, f"unrecognized:{descent.outcome!r}"


# ---------------------------------------------------------------------------
# 3. 观察录入（人工填写：位姿/损伤/掉落不能从软件 SUCCESS 推定）
# ---------------------------------------------------------------------------


def _ask_yes_no(label: str) -> bool:
    return _prompt(f"[观察] {label}（y/n，回车=n）：").strip().lower() in ("y", "yes")


def _ask_float(label: str) -> float | None:
    raw = _prompt(f"[观察] {label}（留空=null）：").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise InputError(f"{label} 不是合法数值：{raw!r}") from exc


def _read_observations(*, placed: bool) -> dict:
    """逐条读取人工观察。placed=False（未执行抓取，如多目标违约）时不追问落点。"""
    obs: dict = {}
    obs["actual_success"] = _ask_yes_no("实物是否成功放到配置位置")
    obs["actual_drop"] = _ask_yes_no("是否发生非预期掉落")
    success = obs["actual_success"]
    if success and placed:
        x = _ask_float("实际放置位置 X(m)")
        y = _ask_float("实际放置位置 Y(m)")
        z = _ask_float("实际放置位置 Z(m)")
        obs["actual_place_position_m"] = (
            None if x is None or y is None or z is None else [x, y, z])
        obs["actual_place_yaw_deg"] = _ask_float("实际放置长轴 yaw(deg)")
    else:
        obs["actual_place_position_m"] = None
        obs["actual_place_yaw_deg"] = None
    damage = _prompt("[观察] 损伤标签（none/bruise/…，回车=none）：").strip()
    obs["damage_label"] = damage or "none"
    obs["damage_observe_minutes"] = _ask_float("损伤观察时长(min)")
    images = _prompt("[观察] 图像引用（逗号分隔；回车=默认 trial 图名）：").strip()
    obs["image_refs"] = [s.strip() for s in images.split(",") if s.strip()]
    return obs


# ---------------------------------------------------------------------------
# 4. 装配 → 逐试验执行 → 导出
# ---------------------------------------------------------------------------


def _assemble(params, vision_config_path: Path):
    """按 §2.8.1 装配。任一步失败前置于本函数：视觉/运动硬故障如实向上抛，
    由调用方映射退出码——不内置回退替身、不自动恢复、不自动卸力。"""
    motor = make_motor(params)
    arm = make_arm(motor, params)
    cold_start_recover(arm)
    vision.configure(thresholds_from_params(params), vision_config_path)
    vision.init()
    return arm


def _metadata_payload(*, operator: str, profile_path: Path, param_sha: str,
                      urdf_sha: str, motor_sha: str, commit: str) -> dict:
    return {
        "record_type": "metadata", "source": "hardware", "operator": operator,
        "profile_path": str(profile_path), "profile_parameter_sha256": param_sha,
        "resource_sha256": {"urdf": urdf_sha, "motor_calibration": motor_sha},
        "code_commit": commit,
    }


def _build_sample_payload(*, trial_row: dict, target, result, trace,
                          observations: dict, param_sha: str) -> dict:
    payload = {
        "record_type": "sample",
        "trial_id": trial_row["trial_id"].strip(),
        "profile_parameter_sha256": param_sha,
        "target_position_m": (None if target is None
                              else [float(c) for c in target.position]),
        "target_yaw_deg": (None if target is None else float(target.yaw_deg)),
        "grade": trial_row["grade"].strip(),
        "place_id": trial_row["place_id"].strip(),
        "sample_truth": trial_row["sample_truth"].strip(),
        "scene": trial_row["scene"].strip(),
        "start_ns": None, "end_ns": None,
        "software_status": None if result is None else str(result.status.value),
        "software_stage": None if result is None else str(result.stage),
        "holding": None if result is None else str(result.holding.value),
        "recovery_required": None if result is None else bool(result.recovery_required),
        "feedback_span_s": None,     # 无实测记 null，引用相应 timing 证据，不伪填 0
        "stop_latency_s": None,
    }
    began, evidence = _descent_from_trace(trace)
    payload["began_descend"] = began
    payload["descent_evidence"] = evidence
    payload.update(observations)
    # 落点误差：仅实际成功且位置/yaw 齐全时可复算，与 validate 同一函数、口径一致。
    payload["place_error_center_m"] = None
    payload["place_error_yaw_deg"] = None
    if (payload["actual_success"] and payload["actual_place_position_m"] is not None
            and payload["actual_place_yaw_deg"] is not None
            and payload["target_position_m"] is not None):
        payload["place_error_center_m"] = round(
            _place_center_error(payload["target_position_m"],
                                payload["actual_place_position_m"]), 9)
        payload["place_error_yaw_deg"] = round(
            _place_yaw_error(payload["target_yaw_deg"],
                             payload["actual_place_yaw_deg"]), 9)
    return payload


# 误差重算：与 calibrate.py 同一实现，导出器写盘前与 validate 复算逐字对齐。
def _place_center_error(target, actual) -> float:
    import math
    return math.sqrt(sum((float(a) - float(t)) ** 2 for a, t in zip(actual, target)))


def _place_yaw_error(target_yaw: float, actual_yaw: float) -> float:
    from configs.motion_params import wrap180
    d = abs(wrap180(float(target_yaw) - float(actual_yaw)))
    return min(d, 180.0 - d)


def _write_trace_sidecar(output_path: Path, *, trial_id: str, target, result,
                         trace, began: bool | None, evidence: str,
                         image_refs: list[str]) -> str:
    """把本次视觉输出 + 运动 trace + 软件返回落到旁路文件；返回相对 trace_ref。"""
    sidecar_dir = output_path.parent / f"{output_path.stem}.traces"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    ref = f"{sidecar_dir.name}/{trial_id}.trace.json"
    doc = {
        "trial_id": trial_id,
        "began_descend": began, "descent_evidence": evidence,
        "vision": None if target is None else {
            "position": [float(c) for c in target.position],
            "yaw_deg": float(target.yaw_deg), "length_m": float(target.length_m),
            "width_m": float(target.width_m), "ripe": bool(target.ripe),
            "valid_count": int(target.valid_count)},
        "software": None if result is None else {
            "status": str(result.status.value), "stage": str(result.stage),
            "holding": str(result.holding.value),
            "recovery_required": bool(result.recovery_required),
            "reason": str(result.reason), "place_id": str(result.place_id)},
        "trace_events": [asdict(e) for e in trace],
        "image_refs": image_refs,
    }
    (output_path.parent / ref).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ref


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    runlog_event = _noop_event

    # --- 1. 连接驱动之前的输入/配置校验（任一失败 → 退出码 2，绝不触碰硬件）---
    try:
        raw, params = _load_profile(Path(args.profile))
        known_places = list(params.places)
        trial_rows = load_trials(Path(args.trials), known_places)
        if not args.operator or not args.operator.strip():
            raise InputError("--operator 不能为空（报告须绑定操作者标识）")
        param_sha = parameter_sha256(raw)
        urdf_sha = sha256_file(params.urdf_path_resolved())
        motor_sha = sha256_file(params.motor_calibration_path_resolved())
    except InputError as exc:
        print(f"[{_TOOL_NAME}][输入错误] {exc}", file=sys.stderr, flush=True)
        return EXIT_INPUT

    candidate = {
        "raw": raw, "params": params, "trial_rows": trial_rows, "param_sha": param_sha,
        "urdf_sha": urdf_sha, "motor_sha": motor_sha,
        "operator": args.operator.strip(), "output": output_path,
        "profile_path_str": str(args.profile),
        "vision_config": Path(args.vision_config), "commit": git_commit(),
    }

    # --- 2. 装配（硬件 / 视觉真实调用；桩环境如实抛 VisionHardError → 退出码 3）---
    try:
        arm = _assemble(params, candidate["vision_config"])
    except VisionHardError as exc:
        print(f"[{_TOOL_NAME}][视觉硬故障] 装配期 vision 真实失败（预期，无回退替身）：{exc}",
              file=sys.stderr, flush=True)
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print(f"[{_TOOL_NAME}][终止] 装配期收到 Ctrl-C，不自动卸力，退出", file=sys.stderr,
              flush=True)
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001 - 驱动/冷启动现场故障：不自动恢复
        print(f"[{_TOOL_NAME}][运行故障] 装配失败（不自动恢复/不自动卸力）：{exc!r}",
              file=sys.stderr, flush=True)
        return EXIT_RUNTIME

    try:
        return _collect_trials(arm, candidate, runlog_event)
    except KeyboardInterrupt:
        print(f"[{_TOOL_NAME}][终止] 采集期收到 Ctrl-C，停止且不自动卸力", file=sys.stderr,
              flush=True)
        return EXIT_RUNTIME


def _noop_event(*_a, **_k) -> None:
    return None


def _collect_trials(arm, candidate: dict, runlog_event) -> int:
    output_path: Path = candidate["output"]
    run_id = uuid.uuid4().hex[:12]
    meta_payload = _metadata_payload(
        operator=candidate["operator"], profile_path=candidate["profile_path_str"],
        param_sha=candidate["param_sha"], urdf_sha=candidate["urdf_sha"],
        motor_sha=candidate["motor_sha"], commit=candidate["commit"])

    # 首行 metadata：先清空 output（一个新 run 一份文件），再逐试验追加。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    append_records(output_path, [Record("pick_place", run_id, f"meta-{run_id}",
                                        time_ns(), meta_payload)])
    samples: list[tuple[str, dict]] = [(f"meta-{run_id}", meta_payload)]
    recorded = 0

    for trial_row in candidate["trial_rows"]:
        try:
            proceed = _confirm_start(trial_row)
        except (EOFError, KeyboardInterrupt):
            print(f"\n[{_TOOL_NAME}] 操作者结束采集（EOF/Ctrl-C）：已记录 {recorded} 次，"
                  f"不自动接下一次、不自动卸力", flush=True)
            break
        if not proceed:
            print(f"[{_TOOL_NAME}] 操作者未确认开始 trial_id="
                  f"{trial_row['trial_id']!r}，结束采集", flush=True)
            break

        target = None
        result = None
        trace: Sequence = ()
        multi_target = False
        try:
            target = vision.get_target(0)
            # 单目标场景验收：本次视野合法候选必须恰为 1；违反记失败、不执行抓取。
            if int(target.valid_count) != 1:
                multi_target = True
        except VisionHardError as exc:
            print(f"[{_TOOL_NAME}][视觉硬故障] get_target 失败（不自动恢复）：{exc}",
                  file=sys.stderr, flush=True)
            return EXIT_RUNTIME
        except (KeyboardInterrupt,):
            print(f"[{_TOOL_NAME}][终止] get_target 期间 Ctrl-C，停止且不自动卸力",
                  file=sys.stderr, flush=True)
            return EXIT_RUNTIME

        if not multi_target:
            try:
                result = arm.grasp_and_place(target, trial_row["place_id"].strip())
                trace = arm.get_last_motion_trace()
            except KeyboardInterrupt:
                print(f"[{_TOOL_NAME}][终止] grasp_and_place 期间 Ctrl-C：保留现场、"
                      f"不自动恢复/不自动卸力", file=sys.stderr, flush=True)
                return EXIT_RUNTIME
            except Exception as exc:  # noqa: BLE001 - 运动原语异常：不自动恢复
                print(f"[{_TOOL_NAME}][运行故障] grasp_and_place 抛出（不自动恢复）：{exc!r}",
                      file=sys.stderr, flush=True)
                return EXIT_RUNTIME

        try:
            observations = _read_observations(placed=not multi_target)
        except (EOFError, KeyboardInterrupt):
            print(f"\n[{_TOOL_NAME}] 观察录入被中断：本次 trial 不写入、结束采集",
                  flush=True)
            break

        if multi_target:
            observations["actual_success"] = False
            observations["actual_place_position_m"] = None
            observations["actual_place_yaw_deg"] = None

        began, evidence = _descent_from_trace(trace)
        image_refs = observations.pop("image_refs") or [f"{trial_row['trial_id']}.jpg"]
        trace_ref = _write_trace_sidecar(
            output_path, trial_id=trial_row["trial_id"].strip(), target=target,
            result=result, trace=trace, began=began, evidence=evidence,
            image_refs=image_refs)
        observations["image_refs"] = image_refs

        payload = _build_sample_payload(
            trial_row=trial_row, target=target, result=result, trace=trace,
            observations=observations, param_sha=candidate["param_sha"])
        payload["trace_ref"] = trace_ref
        if multi_target:
            payload["software_status"] = "SKIPPED_MULTI_TARGET"
            payload["descent_evidence"] = evidence

        samples.append((trial_row["sample_id"].strip(), payload))
        # 写盘前的结构性完整性检查：与 validate 同一份清单，拒绝写出结构非法记录。
        failures = _integrity_failures(meta_payload, samples[1:], candidate)
        if failures:
            print(f"[{_TOOL_NAME}][导出拒绝] 写盘前完整性不通过："
                  + "；".join(failures), file=sys.stderr, flush=True)
            return EXIT_INPUT

        append_records(output_path, [Record("pick_place", run_id,
                                             trial_row["sample_id"].strip(),
                                             time_ns(), payload)])
        recorded += 1
        runlog_event("pick_place_trial_recorded", trial_id=payload["trial_id"],
                     began_descend=payload["began_descend"],
                     descent_evidence=payload["descent_evidence"])

    print(f"[{_TOOL_NAME}] 完成：{recorded} 条试验写入 {output_path}"
          f"（source=hardware、operator={candidate['operator']}）", flush=True)
    return EXIT_OK


def _integrity_failures(meta: dict, sample_pairs: list[tuple[str, dict]],
                        candidate: dict) -> list[str]:
    results = pick_place_integrity_checks(
        meta, sample_pairs, candidate_param_sha=candidate["param_sha"],
        urdf_sha=candidate["urdf_sha"], motor_sha=candidate["motor_sha"])
    return [f"{r['name']}：{r['detail']}" for r in results if not r["ok"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="record_pick_place_trials.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, type=Path,
                    help="draft/verified 候选 profile.json（走 load_calibration_motion_params，"
                         "拒绝 simulation 与 null 未测字段）")
    ap.add_argument("--trials", required=True, type=Path,
                    help="试验清单 CSV，必需列：%s" % "、".join(REQUIRED_COLUMNS))
    ap.add_argument("--output", required=True, type=Path,
                    help="导出的 pick_place JSONL（首行 metadata，其后每试验一条 sample）")
    ap.add_argument("--operator", required=True,
                    help="操作者标识（写入 metadata.operator，报告须绑定，非空）")
    ap.add_argument("--vision-config", required=True, type=Path,
                    help="vision.configure 使用的视觉配置路径（真实调用，桩会如实失败）")
    return ap


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
