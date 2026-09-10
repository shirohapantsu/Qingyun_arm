"""scripts/calibrate.py 的端到端与负向测试。

覆盖 docs/真机参数测量与标定指南.md 1.2 的 CLI 形态、1.3 的数据与报告格式、
第 12 节"必要离线检查"，以及 promote 与 real 模式加载之间的哈希绑定。

全部用模拟采集源（指南 1.2 规定的行为），只在临时目录里写文件。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from configs.motion_params import ParamsError, load_motion_params, parameter_sha256
from tests.support import ROOT, SIM_PROFILE

PY = Path(sys.executable)
SCRIPT = ROOT / "scripts" / "calibrate.py"
STAGES = ("motor", "joints", "tool", "workcell", "timing", "motion", "grasp", "pick_place")


def run(*args: str, expect_ok: bool = True) -> str:
    proc = subprocess.run(
        [str(PY), str(SCRIPT), *args], cwd=str(ROOT), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if expect_ok and proc.returncode != 0:
        raise AssertionError(f"命令失败（{proc.returncode}）：calibrate {' '.join(args)}\n{out}")
    if not expect_ok and proc.returncode == 0:
        raise AssertionError(f"命令本应失败却成功了：calibrate {' '.join(args)}\n{out}")
    return out


def _mark_records_physical(jsonl_path: Path) -> None:
    """把验证样本的 metadata.source 改成 hardware：模拟"这组验收数据来自真机"。

    纯模拟链路的证据全是 mock/offline，P2-6 之后 real 加载会直接拒绝它——那条
    防线由 test_pure_mock证据的配置禁止real加载 单独钉住；本 fixture 需要走完
    哈希绑定链路，所以必须带上 physical 证据。
    """
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r.get("payload", {}).get("record_type") == "metadata":
            r["payload"]["source"] = "hardware"
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict[str, Path]:
    """跑一遍 init → 八阶段 capture+fit → validate → promote，返回关键产物路径。

    module 级共享：整条链路要跑十几秒，每个用例单独重跑会把测试时间拉长十几倍。
    用例只读产物，不改写。
    """
    d = tmp_path_factory.mktemp("calib")
    run("init", "--output", str(d), "--robot-id", "TEST", "--profile-id", "v0")
    profile = d / "profile.json"
    assert profile.is_file()
    # 实机串口属于"操作者提供的已测参数"（指南 3.3）：fit 不再用猜测值填补
    # （P2-3），所以真实流程里由操作者在这一步写入官方工具确认过的端口。
    data = json.loads(profile.read_text(encoding="utf-8"))
    data["motor"]["port"] = "/dev/ttyTEST-OPERATOR-CONFIRMED"
    profile.write_text(json.dumps(data), encoding="utf-8")
    prev = profile
    for stage in STAGES:
        fit = d / f"fit_{stage}.jsonl"
        run("capture", "--stage", stage, "--profile", str(prev), "--output", str(fit))
        out = d / f"c_{stage}.json"
        run("fit", "--stage", stage, "--profile", str(prev), "--input", str(fit),
            "--output", str(out))
        prev = out
    val = d / "val_all.jsonl"
    with val.open("w", encoding="utf-8") as fh:
        for stage in STAGES:
            # 每个阶段单独 capture 到一个文件再合并：既满足"每次实验首条是 metadata"，
            # 也让 --stage all 能拿到全部八个阶段的记录。
            v = d / f"val_{stage}.jsonl"
            run("capture", "--stage", stage, "--profile", str(prev), "--output", str(v))
            fh.writelines(v.read_text(encoding="utf-8"))
    _mark_records_physical(val)
    report = d / "report.json"
    run("validate", "--stage", "all", "--profile", str(prev), "--input", str(val),
        "--report", str(report), "--operator", "tester", "--operator-reviewed")
    verified = d / "verified.json"
    run("promote", "--profile", str(prev), "--report", str(report), "--output", str(verified))
    return {"dir": d, "profile": profile, "candidate": prev, "report": report,
            "verified": verified, "val": val}


# ---------------------------------------------------------------------------
# 1. init
# ---------------------------------------------------------------------------


def test_init生成完整键集合且未测值为null(tmp_path):
    out = run("init", "--output", str(tmp_path), "--robot-id", "R1", "--profile-id", "d0")
    assert "draft 标定包" in out
    data = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    template = json.loads(SIM_PROFILE.read_text(encoding="utf-8"))
    # 键集合必须与仿真配置一致：少一个键，fit 出的候选就加载不了
    assert set(data) == set(template)
    for group in ("model", "motor", "joints", "tool", "workspace", "collision",
                  "timing", "motion", "ik", "grasp", "gripper", "acceptance"):
        assert set(data[group]) == set(template[group]), group
    assert data["status"] == "draft"
    # 未测物理值保持 null，不用猜测值自动填补（指南 2 节、第八节）
    assert data["joints"]["measured_limits_deg"] is None
    assert data["workspace"]["home_joints_deg"] is None
    assert data["grasp"]["center_offset_object_m"] is None
    # 固定值与操作者预先填入的验收要求不是 null
    assert data["schema_version"] == 1
    assert data["timing"]["fps"] == 30
    assert data["motor"]["ids"] == [1, 2, 3, 4, 5, 6]
    assert data["acceptance"]["min_grasp_trials"] == 30
    assert (tmp_path / "raw").is_dir() and (tmp_path / "reports").is_dir()
    assert (tmp_path / "motor_calibration.json").is_file()


def test_init的相对路径以标定包目录为基准(tmp_path):
    run("init", "--output", str(tmp_path), "--robot-id", "R1", "--profile-id", "d0")
    data = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    urdf = (tmp_path / data["model"]["urdf_path"]).resolve()
    assert urdf.is_file(), f"相对路径解析失败：{data['model']['urdf_path']} -> {urdf}"
    assert urdf.name == "so101_new_calib.urdf"


def test_draft配置不能被加载(tmp_path):
    """未测参数为 null，两种模式都必须拒绝（指南 1.1）。"""
    run("init", "--output", str(tmp_path), "--robot-id", "R1", "--profile-id", "d0")
    with pytest.raises(ParamsError) as exc:
        load_motion_params(tmp_path / "profile.json", mode="mock")
    # 加载器先查"模式与 status 是否配套"，再逐字段查 null；两种报错都算正确拒绝，
    # 但都必须说明"这份配置还不能用来运行"。
    assert "null" in str(exc.value) or "status" in str(exc.value), str(exc.value)
    # 把 status 改成可加载的值之后，仍然要因为未测参数为 null 被拒
    data = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    data["status"] = "simulation"
    sim = tmp_path / "as_sim.json"
    sim.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ParamsError) as exc2:
        load_motion_params(sim, mode="mock")
    assert "null" in str(exc2.value), str(exc2.value)


def test_capture会补齐模拟官方校准文件(tmp_path):
    """fit motor 要算校准文件哈希，模拟源必须给出内容而不是留 init 骨架。"""
    run("init", "--output", str(tmp_path), "--robot-id", "R1", "--profile-id", "d0")
    calib_path = tmp_path / "motor_calibration.json"
    before = json.loads(calib_path.read_text(encoding="utf-8"))
    assert before["shoulder_pan"]["range_min"] is None
    run("capture", "--stage", "motor", "--profile", str(tmp_path / "profile.json"),
        "--output", str(tmp_path / "m.jsonl"))
    after = json.loads(calib_path.read_text(encoding="utf-8"))
    assert after["shoulder_pan"]["range_min"] is not None


# ---------------------------------------------------------------------------
# 2. capture 的数据格式（指南 1.3）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_每条记录含必需字段且首条是metadata(pipeline, stage):
    path = pipeline["dir"] / f"fit_{stage}.jsonl"
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines, stage
    for row in lines:
        assert set(row) >= {"stage", "run_id", "sample_id", "captured_at_ns", "payload"}
        assert row["stage"] == stage
        assert isinstance(row["captured_at_ns"], int) and row["captured_at_ns"] > 0
    assert lines[0]["payload"]["record_type"] == "metadata"
    meta = lines[0]["payload"]
    # metadata 必须记录设备型号、固件、供电、指垫、负载、操作者（指南 1.3）
    for key in ("models", "firmware", "power", "pad_id", "load_label", "operator"):
        assert key in meta, key


def test_拟合样本与验证样本不重叠(pipeline):
    fit_ids = set()
    for stage in STAGES:
        for line in (pipeline["dir"] / f"fit_{stage}.jsonl").read_text().splitlines():
            if line.strip():
                fit_ids.add(json.loads(line)["sample_id"])
    val_ids = {json.loads(l)["sample_id"] for l in
               (pipeline["val"]).read_text().splitlines() if l.strip()}
    assert not (fit_ids & val_ids), "指南 1.3 要求两者 sample_id 不重叠"


def test_拟合与验证是离线动作(pipeline):
    """fit/validate 不启动运动：整个过程里没有任何串口/驱动对象被创建。"""
    out = run("fit", "--stage", "motion", "--profile", str(pipeline["candidate"]),
              "--input", str(pipeline["dir"] / "fit_motion.jsonl"),
              "--output", str(pipeline["dir"] / "redo_motion.json"))
    assert "更新" in out


def test_未知阶段被拒绝(pipeline):
    run("capture", "--stage", "bogus", "--profile", str(pipeline["profile"]),
        "--output", "/tmp/never.jsonl", expect_ok=False)


def test_重复capture追加到同一文件形成多次实验(pipeline):
    src = pipeline["dir"] / "fit_motor.jsonl"
    first = src.read_text().count("\n")
    run("capture", "--stage", "motor", "--profile", str(pipeline["profile"]),
        "--output", str(src))
    after = src.read_text().count("\n")
    assert after > first
    # 每一次实验都要有自己的 metadata 首行
    lines = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    runs = {}
    for i, row in enumerate(lines):
        runs.setdefault(row["run_id"], i)
    firsts = set(runs.values())
    assert lines[min(firsts)]["payload"]["record_type"] == "metadata"


# ---------------------------------------------------------------------------
# 3. fit 的算法与约束
# ---------------------------------------------------------------------------


def test_fit产出可加载的候选配置(pipeline):
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    # profile.json 的键集合必须严格等于第 2 节，审计信息走旁证文件
    assert not any(k.startswith("_") for k in data), [k for k in data if k.startswith("_")]
    assert data["status"] == "draft"
    sidecar = pipeline["dir"] / "c_pick_place.audit.json"
    assert sidecar.is_file()
    audit = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(audit) >= set(STAGES)


def test_fit不修改输入配置(pipeline):
    """fit 只输出独立候选文件（指南 1.2），不能原地改写 --profile。"""
    c0 = pipeline["profile"]
    before = c0.read_text(encoding="utf-8")
    run("fit", "--stage", "joints", "--profile", str(c0),
        "--input", str(pipeline["dir"] / "fit_joints.jsonl"),
        "--output", str(pipeline["dir"] / "side.json"))
    assert c0.read_text(encoding="utf-8") == before, "fit 必须输出独立候选文件"


def test_模拟源拟合不得覆盖实机身份字段(tmp_path):
    """ISSUE-019：模拟源 metadata 里的型号/固件是仿真工作台的设定。

    指南 1.3 要求 metadata 必须记录它们，所以不能靠"模拟源不写"来修；真正的
    规则是**只有真机采集才允许回填身份字段**。这里两个方向都要钉住：模拟源不能
    覆盖实测值，真机采集仍然要把读回值取进来——只测前一个方向的话，"永远不回
    填"这种更错的实现也能让测试通过。
    """
    prof = tmp_path / "profile.json"
    run("init", "--output", str(tmp_path), "--robot-id", "HW", "--profile-id", "v0")
    data = json.loads(prof.read_text(encoding="utf-8"))
    # 操作者在真机上逐台读回的固件（实机案例：STS3215 3.10）
    data["motor"]["firmware"] = ["3.10"] * 6
    data["motor"]["port"] = "/dev/serial/by-id/usb-OPERATOR-CONFIRMED"
    prof.write_text(json.dumps(data), encoding="utf-8")

    sim = tmp_path / "fit_sim.jsonl"
    run("capture", "--stage", "motor", "--profile", str(prof), "--output", str(sim))
    # 前提：模拟源确实自带一个不同的固件号，否则这个测试没有任何判别力
    sim_meta = json.loads(sim.read_text(encoding="utf-8").splitlines()[0])
    assert sim_meta["payload"]["source"] == "simulated"
    assert sim_meta["payload"]["firmware"] != ["3.10"] * 6

    cand = tmp_path / "cand_sim.json"
    run("fit", "--stage", "motor", "--profile", str(prof), "--input", str(sim),
        "--output", str(cand))
    after = json.loads(cand.read_text(encoding="utf-8"))["motor"]
    assert after["firmware"] == ["3.10"] * 6, "模拟源把实测固件改回去了（ISSUE-019）"
    assert after["port"] == "/dev/serial/by-id/usb-OPERATOR-CONFIRMED"

    # 反方向：同一份样本改成真机来源，并带上读回值，fit 必须把它取进来
    rows = [json.loads(l) for l in sim.read_text(encoding="utf-8").splitlines()]
    rows[0]["payload"]["source"] = "hardware"
    rows[0]["payload"]["firmware"] = ["3.77"] * 6
    rows[0]["payload"]["port"] = "/dev/serial/by-id/usb-REAL-ARM"
    hw = tmp_path / "fit_hw.jsonl"
    hw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                  encoding="utf-8")
    cand_hw = tmp_path / "cand_hw.json"
    run("fit", "--stage", "motor", "--profile", str(prof), "--input", str(hw),
        "--output", str(cand_hw))
    got = json.loads(cand_hw.read_text(encoding="utf-8"))["motor"]
    assert got["firmware"] == ["3.77"] * 6, "真机采集的读回值没有被采纳"
    assert got["port"] == "/dev/serial/by-id/usb-REAL-ARM"


def test_joints拟合能定出方向与零位(pipeline):
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    assert data["joints"]["sign"] in ([1, 1, 1, 1, 1], [-1] * 5) or \
        all(s in (1, -1) for s in data["joints"]["sign"])
    assert len(data["joints"]["measured_limits_deg"]) == 5
    for lo, hi in data["joints"]["measured_limits_deg"]:
        assert lo < hi
    audit = json.loads((pipeline["dir"] / "c_pick_place.audit.json").read_text("utf-8"))
    # sign/offset 的最小二乘残差要足够小，否则方向判定不可信
    assert max(audit["joints"]["residuals"]["sign_fit_rms_deg"]) < 1.0


def test_工具旋转拟合结果落在SO3(pipeline):
    import numpy as np

    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    R = np.array(data["tool"]["rotation_e_tcp"], float)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    audit = json.loads((pipeline["dir"] / "c_pick_place.audit.json").read_text("utf-8"))
    # 观测位姿必须绕三根轴散开，否则 rotation_e_tcp 有一个方向没被约束
    assert audit["tool"]["residuals"]["pose_matrix_rank"] >= 3


def test_gap表拟合保持严格单调(pipeline):
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    gaps = [r["gap_m"] for r in data["gripper"]["gap_table"]]
    pcts = [r["gripper_pct"] for r in data["gripper"]["gap_table"]]
    assert all(b > a for a, b in zip(pcts, pcts[1:]))
    assert all(b > a for a, b in zip(gaps, gaps[1:]))


def test_workcell拟合出放置位姿与胶囊体(pipeline):
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    assert "default" in data["places"]
    for pid, place in data["places"].items():
        assert len(place["position"]) == 3
        assert -90.0 <= place["yaw_deg"] < 90.0, f"{pid} 的 yaw 必须已被归一化"
    assert len(data["collision"]["link_capsules"]) >= 5
    ids = {c["id"] for c in data["collision"]["link_capsules"]}
    for pair in data["collision"]["ignore_self_pairs"]:
        assert set(pair) <= ids, pair


def test_fit在样本不足时拒绝而不是猜值(pipeline, tmp_path):
    """指南 5.6 要求至少 10 个未参与拟合的 FK 验证构型；不足时必须失败。"""
    src = pipeline["dir"] / "fit_joints.jsonl"
    lines = src.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    fk = [l for l in lines[1:] if json.loads(l)["payload"].get("fk_check")]
    sweep = [l for l in lines[1:]
             if json.loads(l)["payload"].get("approach_direction") == "joint_sweep"]
    app = [l for l in lines[1:]
           if json.loads(l)["payload"].get("approach_direction") == "application_limit"]
    ends = [l for l in lines[1:] if json.loads(l)["payload"].get("endpoint_urdf_deg") is not None]
    # 只留 3 个 FK 构型，其余保持完整：失败原因必须落在 FK 构型数量上。
    thin = tmp_path / "thin.jsonl"
    thin.write_text("\n".join([header] + sweep + app + ends + fk[:3]) + "\n",
                    encoding="utf-8")
    out = run("fit", "--stage", "joints", "--profile", str(pipeline["profile"]),
              "--input", str(thin), "--output", str(tmp_path / "o.json"), expect_ok=False)
    assert "FK 验证构型" in out, out


# ---------------------------------------------------------------------------
# 4. validate 与报告格式
# ---------------------------------------------------------------------------


def test_报告结构逐条符合指南1_3(pipeline):
    report = json.loads(pipeline["report"].read_text(encoding="utf-8"))
    assert set(report) >= {"profile_id", "parameter_sha256", "operator",
                           "operator_reviewed", "stages", "overall_pass"}
    cand = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    assert report["parameter_sha256"] == parameter_sha256(cand)
    assert set(report["stages"]) == set(STAGES)
    for stage, info in report["stages"].items():
        assert set(info) >= {"input_sha256", "fit_sample_ids", "validation_sample_ids",
                             "checks", "passed"}
        assert info["checks"], stage
        for c in info["checks"]:
            assert set(c) >= {"name", "field_path", "measured", "limit", "comparison",
                              "passed", "evidence_type", "evidence_refs"}
            assert c["comparison"] in ("<=", ">=", "=="), c
            assert c["evidence_type"] in ("physical", "mock", "offline"), c
    assert report["overall_pass"] is True


def test_validate是离线动作(pipeline, tmp_path):
    """validate 不启动运动（指南 1.2）：单阶段离线跑通即可，不构造任何驱动对象。"""
    out = run("validate", "--stage", "timing", "--profile", str(pipeline["candidate"]),
              "--input", str(pipeline["val"]), "--report", str(tmp_path / "r.json"))
    assert "[timing] passed=" in out


def test_限制不为了通过验收而被放宽(pipeline, tmp_path):
    """limit 一律从配置里读：把 hard_current_ma 调到 contact 之下必须判不过。

    指南 1.3 明确"记录实际测量值，不为通过验收而自动放宽 limit"。这条用反向做法
    验证同一件事：改严 limit 之后结论必须跟着变，说明阈值不是写死在检查代码里的。
    """
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    data["gripper"]["hard_current_ma"] = data["gripper"]["contact_current_ma"] - 1.0
    bad = tmp_path / "bad_candidate.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    report_path = tmp_path / "r.json"
    # 检查未通过时返回码是 1，这是预期行为
    run("validate", "--stage", "grasp", "--profile", str(bad),
        "--input", str(pipeline["val"]), "--report", str(report_path), expect_ok=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stages"]["grasp"]["passed"] is False
    failed = [c["name"] for c in report["stages"]["grasp"]["checks"] if not c["passed"]]
    assert any("电流" in n for n in failed), failed


def test_sample_id重叠被拒绝(pipeline, tmp_path):
    """拟合样本被当成验证样本来用，等于没有独立验证（指南 1.3）。"""
    overlap = tmp_path / "overlap.jsonl"
    overlap.write_text((pipeline["dir"] / "fit_motor.jsonl").read_text(encoding="utf-8"),
                       encoding="utf-8")
    out = run("validate", "--stage", "motor", "--profile", str(pipeline["candidate"]),
              "--input", str(overlap), "--report", str(tmp_path / "r.json"),
              expect_ok=False)
    assert "重叠" in out


def test_stage_all需要每个阶段都有记录(pipeline, tmp_path):
    """--stage all 时缺任何一个阶段的记录都必须报错，不能默认给缺失阶段放行。

    用验证样本而不是拟合样本做输入：拟合样本会先撞上"拟合/验证 sample_id 不得
    重叠"那条检查，测不到"缺少阶段记录"这个分支。
    """
    only = tmp_path / "only_motor.jsonl"
    src = pipeline["dir"] / "val_motor.jsonl"
    only.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    out = run("validate", "--stage", "all", "--profile", str(pipeline["candidate"]),
              "--input", str(only), "--report", str(tmp_path / "r.json"), expect_ok=False)
    assert "没有" in out and "阶段" in out, out


# ---------------------------------------------------------------------------
# 5. promote 与 real 模式绑定
# ---------------------------------------------------------------------------


def test_promote产物可被real模式加载(pipeline):
    params = load_motion_params(pipeline["verified"], mode="real")
    assert params.status == "verified"
    assert params.verification is not None
    assert params.verification.report_path.endswith("report.json")


def test_pure_mock证据的配置禁止real加载(tmp_path):
    """P2-6：promote 不再只是打印提示；real 加载校验报告里有 physical 证据。

    与 test_promote产物可被real模式加载 的唯一差别：验证样本保持 simulated
    来源（证据全是 mock/offline）。这样的"verified"允许存在于仿真链路，但
    real 模式必须拒绝用它驱动真机。
    """
    run("init", "--output", str(tmp_path), "--robot-id", "MOCKEV", "--profile-id", "m0")
    profile = tmp_path / "profile.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    data["motor"]["port"] = "/dev/ttyMOCK-TEST"
    profile.write_text(json.dumps(data), encoding="utf-8")
    prev = profile
    for stage in STAGES:
        fit = tmp_path / f"fit_{stage}.jsonl"
        run("capture", "--stage", stage, "--profile", str(prev), "--output", str(fit))
        out = tmp_path / f"c_{stage}.json"
        run("fit", "--stage", stage, "--profile", str(prev), "--input", str(fit),
            "--output", str(out))
        prev = out
    val = tmp_path / "val_all.jsonl"
    with val.open("w", encoding="utf-8") as fh:
        for stage in STAGES:
            v = tmp_path / f"val_{stage}.jsonl"
            run("capture", "--stage", stage, "--profile", str(prev), "--output", str(v))
            fh.writelines(v.read_text(encoding="utf-8"))
    # 这里故意不调用 _mark_records_physical：证据保持 mock。
    report = tmp_path / "report.json"
    run("validate", "--stage", "all", "--profile", str(prev), "--input", str(val),
        "--report", str(report), "--operator", "tester", "--operator-reviewed")
    verified = tmp_path / "verified.json"
    run("promote", "--profile", str(prev), "--report", str(report), "--output", str(verified))
    with pytest.raises(ParamsError) as exc:
        load_motion_params(verified, mode="real")
    assert "physical" in str(exc.value), str(exc.value)


def test_promote拒绝未验收通过的报告(pipeline, tmp_path):
    report = json.loads(pipeline["report"].read_text(encoding="utf-8"))
    report["overall_pass"] = False
    p = tmp_path / "not_pass.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    out = run("promote", "--profile", str(pipeline["candidate"]), "--report", str(p),
              "--output", str(tmp_path / "v.json"), expect_ok=False)
    assert "overall_pass" in out


def test_promote拒绝缺人工复核的报告(pipeline, tmp_path):
    report = json.loads(pipeline["report"].read_text(encoding="utf-8"))
    report["operator_reviewed"] = False
    p = tmp_path / "no_review.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    out = run("promote", "--profile", str(pipeline["candidate"]), "--report", str(p),
              "--output", str(tmp_path / "v.json"), expect_ok=False)
    assert "operator_reviewed" in out


def test_promote拒绝某阶段未通过的报告(pipeline, tmp_path):
    report = json.loads(pipeline["report"].read_text(encoding="utf-8"))
    report["stages"]["tool"]["passed"] = False
    p = tmp_path / "one_fail.json"
    write = json.dumps(report)
    p.write_text(write, encoding="utf-8")
    out = run("promote", "--profile", str(pipeline["candidate"]), "--report", str(p),
              "--output", str(tmp_path / "v.json"), expect_ok=False)
    assert "未通过" in out and "tool" in out


def test_promote拒绝验收后被改动的配置(pipeline, tmp_path):
    """报告里的 parameter_sha256 必须绑定到当前这份参数（指南 1.3）。"""
    data = json.loads(pipeline["candidate"].read_text(encoding="utf-8"))
    data["ik"]["position_tol_m"] = data["ik"]["position_tol_m"] + 0.0005
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data), encoding="utf-8")
    out = run("promote", "--profile", str(tampered), "--report", str(pipeline["report"]),
              "--output", str(tmp_path / "v.json"), expect_ok=False)
    assert "不同" in out


def test_验收后改一个字也让real模式拒绝加载(pipeline, tmp_path):
    """promote 之后再改参数，real 加载必须失败——这条是整个绑定链路的意义。"""
    verified = json.loads(pipeline["verified"].read_text(encoding="utf-8"))
    verified["grasp"]["lift_height_m"] = verified["grasp"]["lift_height_m"] + 0.001
    # 整包复制到临时目录：verification 与 model 都用相对路径，只搬 profile 会让
    # 加载器先因为找不到校准文件而报"资源缺失"，测不到参数哈希那一步。
    for name in ("motor_calibration.json",):
        src = pipeline["verified"].parent / name
        if src.is_file():
            (tmp_path / name).write_bytes(src.read_bytes())
    report_src = pipeline["report"]
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "report.json").write_bytes(report_src.read_bytes())
    verified["verification"]["report_path"] = "reports/report.json"
    p = tmp_path / "after_edit.json"
    p.write_text(json.dumps(verified), encoding="utf-8")
    with pytest.raises(ParamsError) as exc:
        load_motion_params(p, mode="real")
    assert "参数哈希" in str(exc.value) or "不同" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. P1-5 回归：实机采集链路（假总线，无需硬件）
# ---------------------------------------------------------------------------

import importlib.util

import numpy as np

import qingyun.grabbing.motor_control as MC
from configs.motion_params import load_calibration_profile
from tests.test_motor_control import VENDOR_INST_READ, FakeBus


def _load_mod():
    spec = importlib.util.spec_from_file_location("calibrate_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_mod"] = mod        # @dataclass 需要模块在 sys.modules 里
    spec.loader.exec_module(mod)
    return mod


def _draft_package(tmp_path) -> tuple[object, Path]:
    """init 出的 draft 包 + 操作者填串口 + 官方校准文件已有内容。"""
    mod = _load_mod()
    assert mod.main(["init", "--output", str(tmp_path), "--robot-id", "HW",
                     "--profile-id", "h0"]) == 0
    prof = tmp_path / "profile.json"
    data = json.loads(prof.read_text(encoding="utf-8"))
    data["motor"]["port"] = "/dev/ttyHW-TEST"
    prof.write_text(json.dumps(data), encoding="utf-8")
    # 模拟源会把 init 的官方校准骨架补成有内容的文件（真机流程由官方工具导出）。
    assert mod.main(["capture", "--stage", "motor", "--profile", str(prof),
                     "--output", str(tmp_path / "seed.jsonl")]) == 0
    return mod, prof


def _seed_bus(pos_base: int = 2000, pos_step: int = 5) -> FakeBus:
    bus = FakeBus()
    for i in range(6):
        bus.set_u16(i + 1, 56, pos_base + pos_step * i)   # Present_Position
        bus.set_u16(i + 1, 58, 0)                          # Present_Velocity
        bus.set_u16(i + 1, 69, 10)                         # Present_Current
    return bus


def test_draft配置可被采集入口加载(tmp_path):
    """P1-5 缺陷 1 复现：real 链路要求 verified，draft 期 --hardware 必然抛错。"""
    _, prof = _draft_package(tmp_path)
    pfile = load_calibration_profile(prof)
    assert pfile.joints is None                  # joints 未拟合 → 只读换算可用
    assert pfile.motor.port == "/dev/ttyHW-TEST"
    assert np.isfinite(pfile.urdf_limits_deg).all()


def test_采集入口拒绝模拟包与缺失串口(tmp_path):
    with pytest.raises(ParamsError) as exc:
        load_calibration_profile(SIM_PROFILE)     # simulation 没有真机串口
    assert "simulation" in str(exc.value)
    # 官方校准骨架（range 为 null）在 port 之前就会被拒——这本身是正确顺序；
    # 要单独测 port 缺失，用补全过的采集包再把串口改回 null。
    _, prof = _draft_package(tmp_path)
    data = json.loads(prof.read_text(encoding="utf-8"))
    data["motor"]["port"] = None
    prof.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ParamsError) as exc2:     # port 仍是 null → 明确指向串口
        load_calibration_profile(prof)
    assert "port" in str(exc2.value)


def test_capture_hardware_motor_stage(tmp_path, monkeypatch):
    """P1-5 缺陷 2 复现：motor 采集走真实存在的只读快照，且全程只读。"""
    mod, prof = _draft_package(tmp_path)
    bus = _seed_bus()
    monkeypatch.setattr(MC, "SerialTransport", lambda port, baud: bus)
    out = tmp_path / "hw_motor.jsonl"
    rc = mod.main(["capture", "--stage", "motor", "--profile", str(prof),
                   "--output", str(out), "--hardware", "--repeats", "2"])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert records[0]["payload"]["source"] == "hardware"
    assert records[0]["payload"]["port"] == "/dev/ttyHW-TEST"
    assert len(records) == 1 + 2
    assert records[1]["payload"]["raw_position"] == [2000 + 5 * i for i in range(6)]
    # 采集路径绝不提交运动：没有任何 SYNC_WRITE / WRITE_DATA 帧。
    assert bus.sync_write_count == 0
    assert not any(f[4] in (MC.StsProtocol.INST_SYNC_WRITE,
                            MC.StsProtocol.INST_WRITE_DATA) for f in bus.tx_log)


def test_capture_hardware_joints_stage给出q_bus(tmp_path, monkeypatch):
    """P1-5 缺陷 3 复现：JointFeedback 没有 bus_degrees；q_bus 必须由换算层给出。"""
    mod, prof = _draft_package(tmp_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "motor.json").write_text(
        json.dumps({"stages": {"motor": {"passed": True}}}), encoding="utf-8")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"samples": [{
        "q_reference_deg": [0.0, 10.0, -20.0, 5.0, 5.0], "fixture_id": "F1",
        "load_label": "unloaded", "approach_direction": "sweep"}]}), encoding="utf-8")
    bus = _seed_bus(pos_base=2100)
    monkeypatch.setattr(MC, "SerialTransport", lambda port, baud: bus)
    out = tmp_path / "hw_joints.jsonl"
    rc = mod.main(["capture", "--stage", "joints", "--profile", str(prof),
                   "--output", str(out), "--hardware", "--reference", str(ref)])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    payload = records[1]["payload"]
    assert payload["q_reference_deg"] == [0.0, 10.0, -20.0, 5.0, 5.0]
    # 与生产换算层逐值一致（指南 12 节"共同调用同一份函数"的又一处体现）。
    rd = MC.CalibrationReader(load_calibration_profile(prof), transport=FakeBus())
    expect = [rd.mapping.raw_to_bus_deg(ax, 2100 + 5 * i)
              for i, ax in enumerate(rd.mapping.axes[:5])]
    assert payload["q_bus_deg"] == pytest.approx(expect, abs=1e-12)


def test_capture_hardware_joints的外部测量键能进记录并被fit消费(tmp_path, monkeypatch):
    """ISSUE-022：joints 采集分支曾逐个挑键，把 _fit_joints 必需的键全丢掉。

    端到端跑 capture --hardware --stage joints → fit --stage joints。旧写法在 fit
    里抛 KeyError: 'sweep_joint_index'，joints 整阶段无法拟合。
    """
    mod, prof = _draft_package(tmp_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "motor.json").write_text(
        json.dumps({"stages": {"motor": {"passed": True}}}), encoding="utf-8")

    # 真值定义在被标定模型自己的坐标里：用生产换算层把期望总线角折回寄存器计数。
    probe = MC.CalibrationReader(load_calibration_profile(prof), transport=FakeBus())
    mapping = probe.mapping
    probe.close()
    sign_true = [1, -1, 1, -1, 1]
    offset_true = [0.0, 5.0, -3.0, 12.0, 0.5]
    neutral = [int(mapping.bus_deg_to_raw(j, 0.0)) for j in range(5)]

    def q_ref(j: int, deg: float) -> list[float]:
        return [sign_true[k] * (deg if k == j else 0.0) + offset_true[k]
                for k in range(5)]

    samples: list[dict] = []
    positions: list[list[int]] = []

    def add(row: dict, pos: list[int]) -> None:
        samples.append(row)
        positions.append(pos)

    # 1) 每关节 3 个角度、覆盖正反方向 → sign / zero_offset_deg
    for j in range(5):
        for deg in (-20.0, 15.0, 40.0):
            pos = list(neutral)
            pos[j] = int(mapping.bus_deg_to_raw(j, deg))
            add({"approach_direction": "joint_sweep", "sweep_joint_index": j,
                 "sweep_direction": "forward" if deg >= 0 else "backward",
                 "q_reference_deg": q_ref(j, deg),
                 "fixture_id": "G1", "load_label": "unloaded"}, pos)
    # 2) 端点重复测量 → measured_limits_deg / margin_deg。关节 0 故意给 1.8° 的
    #    重复散布，用来验证指南 5.5 的 margin = max(draft, 散布 + 0.5) 真会变大。
    for j in range(5):
        spread = 1.8 if j == 0 else 0.4
        for side, base in (("lower", -95.0), ("upper", 88.0)):
            dirn = -1.0 if side == "lower" else 1.0
            for rep in (0.0, spread):
                v = base + dirn * rep
                add({"approach_direction": "endpoints", "sweep_joint_index": j,
                     "endpoint_side": side, "endpoint_urdf_deg": v,
                     "endpoint_repeat_deg": v, "q_reference_deg": q_ref(j, 0.0)},
                    list(neutral))
    # 3) 线缆/支架限制，每关节一条 → application_limits_deg
    for j in range(5):
        add({"approach_direction": "application_limit", "sweep_joint_index": j,
             "application_limit_urdf_deg": [-90.0 - j, 90.0 + j],
             "limit_reason": "harness", "q_reference_deg": q_ref(j, 0.0)},
            list(neutral))
    # 4) 至少 10 个未参与拟合的 FK 验证构型（指南 5.6）
    for k in range(10):
        add({"approach_direction": "fk_validation", "fk_check": True,
             "q_reference_deg": q_ref(k % 5, float(k)),
             "measured_tcp_position_m": [0.3, 0.01 * k, 0.05]},
            list(neutral))

    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    bus = _RowSteppingBus(positions)
    monkeypatch.setattr(MC, "SerialTransport", lambda port, baud: bus)
    out = tmp_path / "hw_joints.jsonl"
    assert mod.main(["capture", "--stage", "joints", "--profile", str(prof),
                     "--output", str(out), "--hardware", "--reference", str(ref)]) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    sweep = records[1]["payload"]
    for key in ("q_bus_deg", "sweep_joint_index", "sweep_direction", "fixture_id"):
        assert key in sweep, f"采集记录丢了 fit 必需的键 {key}（ISSUE-022）"
    assert not any(f[4] in (MC.StsProtocol.INST_SYNC_WRITE,
                            MC.StsProtocol.INST_WRITE_DATA) for f in bus.tx_log), \
        "joints 采集只允许读总线，不许提交运动"

    cand = tmp_path / "cand.json"
    assert mod.main(["fit", "--stage", "joints", "--profile", str(prof),
                     "--input", str(out), "--output", str(cand)]) == 0
    fitted = json.loads(cand.read_text(encoding="utf-8"))
    assert fitted["status"] == "draft"
    assert fitted["joints"]["sign"] == sign_true, "方向真值未被恢复"
    assert fitted["joints"]["zero_offset_deg"] == pytest.approx(
        offset_true, abs=0.06), "零位偏置未被恢复（编码步长 0.0879°/计数）"
    for j in range(5):
        lo, hi = fitted["joints"]["measured_limits_deg"][j]
        assert (lo, hi) == (-95.0 - (1.8 if j == 0 else 0.4), 88.0 + (1.8 if j == 0 else 0.4))
    assert fitted["joints"]["margin_deg"] == pytest.approx([2.3, 2.0, 2.0, 2.0, 2.0])
    assert fitted["joints"]["application_limits_deg"][2] == [-92.0, 92.0]
    audit = json.loads((tmp_path / "cand.audit.json").read_text(encoding="utf-8"))
    assert audit["joints"]["residuals"]["fk_validation_configs"] == 10
    assert audit["joints"]["updated_fields"] == sorted([
        "joints.sign", "joints.zero_offset_deg", "joints.measured_limits_deg",
        "joints.margin_deg", "joints.application_limits_deg"])


class _RowSteppingBus(FakeBus):
    """每完成一次 read_snapshot（6 台各一次块读）换一组 Present_Position。

    模拟操作者按 --reference 逐行把臂重新摆好——没有它就只能测"键是否进记录"，
    测不到"fit 能否从这些记录解出真值"。

    挂钩必须放在 write()：FakeBus 在 write() 里处理 READ 请求并自增
    read_request_count，而 read() 只是从应答缓冲取字节、一次应答会被分多次调用。
    挂在 read() 上会让行号在同一个 snapshot 内部错位。
    """

    def __init__(self, position_rows: list[list[int]]) -> None:
        super().__init__()
        self._rows = position_rows

    def write(self, data: bytes, timeout_s: float | None = None) -> int:
        frame = bytes(data)
        if len(frame) > 4 and frame[4] == VENDOR_INST_READ:
            row = self._rows[(self.read_request_count // 6) % len(self._rows)]
            for i, pos in enumerate(row):
                self.set_u16(i + 1, 56, pos)       # Present_Position
                self.set_u16(i + 1, 58, 0)         # Present_Velocity
                self.set_u16(i + 1, 69, 10)        # Present_Current
        return super().write(data, timeout_s)


def test_capture_hardware_tool阶段在joints未拟合时明确拒绝(tmp_path, monkeypatch, capsys):
    mod, prof = _draft_package(tmp_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    for prior in ("motor", "joints"):
        (tmp_path / "reports" / f"{prior}.json").write_text(
            json.dumps({"stages": {prior: {"passed": True}}}), encoding="utf-8")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"samples": [{
        "gripper_pct": 50.0, "q_urdf_deg_note": "external"}]}), encoding="utf-8")
    monkeypatch.setattr(MC, "SerialTransport", lambda port, baud: _seed_bus())
    rc = mod.main(["capture", "--stage", "tool", "--profile", str(prof),
                   "--output", str(tmp_path / "hw_tool.jsonl"), "--hardware",
                   "--reference", str(ref)])
    err = capsys.readouterr().err
    assert rc == 2 and "joints" in err, err


# ---------------------------------------------------------------------------
# 7. P2-2 / P2-5：fit 单位与共享换算函数回归
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calib_mod():
    return _load_mod()


def _grasp_records(mod, extra_summary: dict) -> list:
    recs = []

    def rec(payload):
        recs.append(mod.Record("grasp", "r1", f"s{len(recs)}", 0, payload))

    rec({"record_type": "empty_placeholder"})   # 占位行不参与 sample 过滤
    rec({"record_type": "sample", "cycle_phase": "empty_close", "gripper_pct": 2.5})
    rec({"record_type": "sample", "contact_label": "first_contact", "actual_gap_m": 0.026})
    summary = {"record_type": "trial_summary", "slip_drift_pct": 3.0}
    summary.update(extra_summary)
    rec(summary)
    return recs


def test_object_shift_bound必须是米量级的实测位移(calib_mod):
    """P2-2 复现：slip_drift_pct(%)×0.001 是单位混淆。

    有直接测得的中心位移 center_shift_m 时按"最大值+测量余量"；只有开度漂移
    时按 gap_table 斜率换算成内间距变化量的保守上界。两种路径的单位都必须是米。
    """
    from tests.support import raw_profile
    base = raw_profile()
    # 路径 1：实测中心位移 4mm + 1mm 余量 = 5mm
    upd, _ = calib_mod._fit_grasp(base, _grasp_records(calib_mod, {"center_shift_m": 0.004}))
    assert upd["grasp.object_shift_bound_m"] == pytest.approx(0.005)
    # 路径 2：漂移 3% × 斜率 (0.05m/100%) = 1.5mm + 1mm = 2.5mm。
    # 旧实现会给出 3×0.001=3.0mm——数值与含义都不对，这里用精确期望钉住。
    upd2, _ = calib_mod._fit_grasp(base, _grasp_records(calib_mod, {}))
    assert upd2["grasp.object_shift_bound_m"] == pytest.approx(0.0025)
    # slip 阈值仍然按开度百分比计算，不能被位移换算污染。
    assert upd2["gripper.slip_opening_change_pct"] == pytest.approx(3.0 * 2.0 + 1.0)


def test_gap换算共享实现禁止外插(calib_mod):
    """P2-5 复现：本地 np.interp 副本越界静默钳位，伪装成合法端点值。"""
    from tests.support import raw_profile
    rows = raw_profile()["gripper"]["gap_table"]
    # 表内：与生产实现同值。
    assert calib_mod.gripper_pct_to_gap_simple(50.0, rows) == pytest.approx(0.030)
    assert calib_mod._pct_for_gap(0.030, rows) == pytest.approx(50.0)
    # 表外：必须报错，不能返回端点值。
    with pytest.raises(calib_mod.CalibError):
        calib_mod.gripper_pct_to_gap_simple(120.0, rows)
    with pytest.raises(calib_mod.CalibError):
        calib_mod._pct_for_gap(0.20, rows)
