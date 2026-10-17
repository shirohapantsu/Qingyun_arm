#!/usr/bin/env python3
"""RGB-only 点触外参标定：PnP + Kabsch，结果仅保存为独立候选。

拟合残差不是 CAL-040 验收；深度配准、机器人运动与正式配置安装不在本工具内。
运行 --help 查看显式单位、TCP、质量阈值及独立验证点文件参数。
默认网页预览，点击开始后采满 10 次合格窗口，保存并停止检测，再到终端录入 mm 坐标。
纯识别用 --detect-only --browser，终端兼容入口用 --terminal。
"""

import argparse
import base64
from collections import deque
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TCP_DESCRIPTION = "闭合夹爪两指尖中心"


@dataclass(frozen=True)
class PreviewCriteria:
    """静止场景的建议采集门槛，可调整；不等同于真机精度验收。"""

    window_s: float = 3.0
    min_frames: int = 30
    min_rate: float = 0.95
    min_side_px: float = 60.0
    max_rms_px: float = 1.0
    max_corner_jitter_px: float = 0.5
    max_position_rms_mm: float = 2.0
    max_position_peak_mm: float = 4.0
    max_frame_age_s: float = 1.0

    def validate(self):
        if type(self.min_frames) is not int or self.min_frames < 2:
            raise ValueError("预览稳定性至少需要两帧")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"预览参数 {name} 必须为有限正数")
        if self.min_rate > 1:
            raise ValueError("预览检出率必须在 (0,1] 内")


class MarkerStability:
    """按最近时间窗口统计；未检出/坏姿态也计入分母，不使用累计次数代替稳定性。"""

    def __init__(self, expected_ids, criteria=None):
        self.criteria = criteria or PreviewCriteria()
        self.criteria.validate()
        self.expected_ids = tuple(expected_ids)
        self.history = deque()
        self.started_at = None

    def update(self, report, timestamp):
        if not np.isfinite(timestamp) or (self.history and timestamp < self.history[-1][0]):
            raise ValueError("预览时间戳必须有限且非递减")
        if self.started_at is None:
            self.started_at = timestamp
        self.history.append((timestamp, report["markers"]))
        # 保留窗口左边界的一帧，使完整窗口覆盖时间不会总是略小于 window_s。
        while len(self.history) > 1 and self.history[1][0] <= timestamp - self.criteria.window_s:
            self.history.popleft()
        elapsed = timestamp - self.started_at
        warmed = elapsed >= self.criteria.window_s and len(self.history) >= self.criteria.min_frames
        rows = []
        for mid in self.expected_ids:
            observed, valid = [], []
            current_valid = False
            for index, (_, items) in enumerate(self.history):
                matching = [item for item in items if item["id"] == mid]
                if len(matching) != 1:
                    continue
                item = matching[0]
                observed.append(item)
                if item["quality_pass"] and item.get("camera_xyz_m") is not None:
                    valid.append(item)
                    if index == len(self.history) - 1:
                        current_valid = True
            row = {"id": mid, "detection_rate": len(observed) / len(self.history),
                   "quality_rate": len(valid) / len(self.history), "min_side_px": None,
                   "max_rms_px": None, "corner_jitter_px": None,
                   "position_rms_mm": None, "position_peak_mm": None,
                   "mean_camera_xyz_m": None, "current_camera_xyz_m": None}
            failures = []
            if not warmed:
                failures.append("warming_up")
            if not current_valid:
                failures.append("missing_or_invalid_now")
            if row["detection_rate"] < self.criteria.min_rate:
                failures.append("low_detection_rate")
            if row["quality_rate"] < self.criteria.min_rate:
                failures.append("low_quality_rate")
            if observed:
                row["min_side_px"] = min(item["min_side_px"] for item in observed)
                if row["min_side_px"] < self.criteria.min_side_px:
                    failures.append("too_small")
            if valid:
                row["mean_camera_xyz_m"] = np.mean([item["camera_xyz_m"] for item in valid], axis=0).tolist()
                if current_valid:
                    row["current_camera_xyz_m"] = valid[-1]["camera_xyz_m"]
                row["max_rms_px"] = max(item["reprojection_error_px"] for item in valid)
                if row["max_rms_px"] > self.criteria.max_rms_px:
                    failures.append("reprojection")
            if len(valid) >= 2:
                corners = np.array([item["corners_px"] for item in valid])
                position = np.array([item["camera_xyz_m"] for item in valid])
                row["corner_jitter_px"] = float(np.sqrt(np.mean(np.sum((corners - corners.mean(axis=0)) ** 2, axis=2), axis=0)).max())
                distances_mm = np.linalg.norm(position - position.mean(axis=0), axis=1) * 1000
                row["position_rms_mm"] = float(np.sqrt(np.mean(distances_mm ** 2)))
                row["position_peak_mm"] = float(distances_mm.max())
                if row["corner_jitter_px"] > self.criteria.max_corner_jitter_px:
                    failures.append("corner_motion")
                if (row["position_rms_mm"] > self.criteria.max_position_rms_mm
                        or row["position_peak_mm"] > self.criteria.max_position_peak_mm):
                    failures.append("position_motion")
            else:
                failures.append("insufficient_geometry")
            row["reasons"] = failures
            row["status"] = "READY" if not failures else ("WARMUP" if not warmed else "CHECK")
            rows.append(row)
        ready = bool(rows) and all(row["status"] == "READY" for row in rows)
        return {"status": "READY" if ready else ("WARMUP" if not warmed else "CHECK"),
                "ready_for_sampling": ready, "elapsed_s": elapsed,
                "window_frames": len(self.history), "window_span_s": timestamp - self.history[0][0],
                "criteria": asdict(self.criteria), "markers": rows,
                "meaning": "READY 仅为静止场景采集质量筛选通过，不是坐标真值或 CAL-040 验收"}


class VisualCalibrationSession:
    """预览→点击开始→采满指定数量的合格窗口，自动冻结最佳批次。"""

    def __init__(self, expected_ids, criteria=None, marker_length_m=.080, target_windows=10):
        self.expected_ids = tuple(sorted(expected_ids))
        if (len(self.expected_ids) < 4 or len(set(self.expected_ids)) != len(self.expected_ids)
                or any(type(mid) is not int or not 0 <= mid < 100 for mid in self.expected_ids)):
            raise ValueError("正式标定至少需要四个不同的 0～99 标记 ID")
        if type(target_windows) is not int or target_windows < 1:
            raise ValueError("正式标定采集次数须为正整数")
        self.criteria = criteria or PreviewCriteria()
        self.criteria.validate()
        _positive_number(marker_length_m, "marker_length_m")
        self.marker_length_m = marker_length_m
        self.target_windows = target_windows
        self.lock = threading.RLock()
        self.generation = 0
        self.revision = 0
        self.phase = "preview"
        self.tracker = MarkerStability(self.expected_ids, self.criteria)
        self.comparisons = []
        self.best = None
        self.anchor = None
        self.last_window_at = None
        self.last_frame_at = None
        self.last_assessment = None
        self.coordinates_mm = {}
        self.attempt = 0
        self.error = ""
        self.result = None

    def frame_settings(self):
        with self.lock:
            return self.marker_length_m, self.generation

    def _window_points(self):
        return {mid: np.median([item["camera_xyz_m"] for _, items in self.tracker.history
                               for item in items if item["id"] == mid and item["quality_pass"]
                               and item.get("camera_xyz_m") is not None], axis=0)
                for mid in self.expected_ids}

    def _same_layout(self, first, second):
        return all(np.linalg.norm(np.asarray(first[mid]) - np.asarray(second[mid])) * 1000
                   <= self.criteria.max_position_peak_mm for mid in self.expected_ids)

    def observe(self, report, timestamp, raw, annotated, generation):
        with self.lock:
            if generation != self.generation or self.phase not in ("preview", "collecting"):
                return None  # 尺寸重置的在途帧与采满后的帧均不得进入记录。
            assessment = self.tracker.update(report, timestamp)
            self.revision += 1
            self.last_frame_at, self.last_assessment = timestamp, assessment
            if not assessment["ready_for_sampling"]:
                return assessment
            if self.phase != "collecting" or (self.last_window_at is not None
                    and timestamp - self.last_window_at < self.criteria.window_s):
                return assessment
            points = self._window_points()
            if self.anchor is not None and not self._same_layout(points, self.anchor):
                self.comparisons.clear()
                self.best = None
                self.error = "检测到布局变化，已清除旧批次并重新比较。"
            if not self.comparisons:
                self.anchor = points
            ratios = []
            for row in assessment["markers"]:
                ratios.extend([row["max_rms_px"] / self.criteria.max_rms_px,
                               row["corner_jitter_px"] / self.criteria.max_corner_jitter_px,
                               row["position_rms_mm"] / self.criteria.max_position_rms_mm,
                               row["position_peak_mm"] / self.criteria.max_position_peak_mm])
            rate = min(min(row["detection_rate"], row["quality_rate"]) for row in assessment["markers"])
            score = (max(ratios), -rate, float(np.mean(ratios)))
            summary = {"version": len(self.comparisons) + 1, "score": list(score),
                       "window_end_s": timestamp, "assessment": copy.deepcopy(assessment)}
            self.comparisons.append(summary)
            self.last_window_at = timestamp
            if self.best is None or score < tuple(self.best["summary"]["score"]):
                self.best = {"summary": copy.deepcopy(summary), "points": points,
                             "observations": [{"timestamp_s": t, "markers": copy.deepcopy(items)}
                                              for t, items in self.tracker.history],
                             "raw": raw.copy(), "annotated": annotated.copy()}
            if len(self.comparisons) == self.target_windows:
                self.phase = "captured"
            return assessment

    def snapshot(self, now=None):
        with self.lock:
            return {"phase": self.phase, "revision": self.revision, "unit": "mm",
                    "marker_length_mm": self.marker_length_m * 1000,
                    "compared_windows": len(self.comparisons), "target_windows": self.target_windows,
                    "best": copy.deepcopy(self.best["summary"]) if self.best else None,
                    "error": self.error, "result": copy.deepcopy(self.result)}

    def action(self, payload, now=None):
        if not isinstance(payload, dict):
            raise ValueError("操作必须为 JSON 对象")
        with self.lock:
            action = payload.get("action")
            if action == "start":
                if self.phase != "preview":
                    raise ValueError("采集已开始或已完成；请勿重复开始")
                value = _positive_number(payload.get("marker_length_mm"), "黑色正方形边长 mm")
                self.marker_length_m = value / 1000
                self._reset("collecting")
            elif action == "reset":
                if self.phase not in ("preview", "collecting"):
                    raise ValueError("采集已完成，请启动新会话")
                self._reset("preview")
            else:
                raise ValueError("未知标定操作")
            self.revision += 1
            return self.snapshot(now)

    def _reset(self, phase):
        self.generation += 1
        self.phase = phase
        self.tracker = MarkerStability(self.expected_ids, self.criteria)
        self.comparisons.clear()
        self.best = self.anchor = self.last_window_at = self.last_frame_at = self.last_assessment = None
        self.coordinates_mm.clear()
        self.error = ""

    def export(self):
        with self.lock:
            data = {"unit": "mm", "marker_length_m": self.marker_length_m,
                    "expected_ids": list(self.expected_ids), "target_windows": self.target_windows,
                    "comparisons": copy.deepcopy(self.comparisons),
                    "criteria": asdict(self.criteria), "coordinates_mm": copy.deepcopy(self.coordinates_mm),
                    "selection_method": "最低最差归一化误差优先，再按最低检出/质量率、平均归一化误差比较；取选定窗口中各点 PnP 中心逐轴中位数"}
            if self.best:
                data["selected_window"] = copy.deepcopy(self.best["summary"])
                data["selected_observations"] = copy.deepcopy(self.best["observations"])
                data["camera_points_m"] = {str(mid): point.tolist() for mid, point in self.best["points"].items()}
            return data


_PREVIEW_HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>标记实时预览</title><style>
body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}
img{max-width:100%;background:#000}table{border-collapse:collapse;width:100%;margin:18px 0}
td,th{padding:8px;border:1px solid #475569;text-align:left}.READY{color:#4ade80}
.CHECK{color:#fb923c}.WARMUP{color:#facc15}.STALE{color:#f87171}button{padding:10px;font-size:16px}
input,select{font-size:16px;padding:8px;margin:6px}input{width:130px}section{border:1px solid #64748b;padding:16px;margin:20px 0}pre{white-space:pre-wrap}button:disabled{opacity:.45}
</style><h2>RGB 标记实时预览</h2><p id="status">等待第一帧……</p>
<p>固定相机和标记，等待采样窗口。READY 只表示采集质量通过，不能证明毫米精度。XYZ 为 RGB 相机光学系，X向右、Y向下、Z向前。</p><p id="size"></p>
<img id="frame" alt="等待相机画面"><p id="limits"></p>
<table><thead><tr><th>ID</th><th>检出率</th><th>PnP通过率</th><th>最短边 px</th>
<th>最大重投影 px</th><th>角点抖动 px</th><th>位置RMS/峰值 mm</th><th>当前相机 XYZ mm</th><th>状态/原因</th></tr></thead><tbody id="rows"></tbody></table>
<section id="calibration" hidden><h2>正式标定 · 坐标默认 mm</h2>
<p id="cal-stage"></p><p id="cal-best"></p>
<div id="cal-size">黑色正方形边长多少 mm？<input id="marker-mm" type="number" min="0.001" step="any" value="80">
<button id="start">开始采集</button></div>
<p>开始前持续 Preview，不计入正式批次。机械臂先停在 Home，相机与标记保持固定；点击开始后，采满指定次数的合格窗口即自动停止，并保存最佳批次。每次为一个稳定时间窗口，未达标不计数。</p>
<button id="reset">清空本轮采集，返回 Preview</button>
<p id="cal-error" class="CHECK"></p><pre id="cal-result"></pre>
<p>评分比较的是观测质量，不是实测准确率。拟合通过仍需独立验证；正式配置不会自动覆盖。</p></section>
<button id="stop">保存当前记录并退出</button><p id="error"></p><script>
const reasons={warming_up:'等待足够帧数和时长',missing_or_invalid_now:'当前缺失或质量不合格',
low_detection_rate:'检出率不足',low_quality_rate:'PnP通过率不足',too_small:'像素边长不足',
reprojection:'重投影超限',corner_motion:'角点抖动过大',position_motion:'三维位置波动过大',insufficient_geometry:'有效几何不足',stale_frame:'画面已过期'};
const f=x=>x===null||x===undefined?'—':Number(x).toFixed(2);
let calibrationState=null,calibrationPhase=null,actionError='',calibrationRevision=-1;
function renderCalibration(c){document.getElementById('calibration').hidden=!c;if(!c||c.revision<calibrationRevision)return;calibrationRevision=c.revision;calibrationState=c;
const phases={preview:'1. 实时预览：填写边长后点击开始',collecting:'2. 正在采集合格数据',captured:'3. 已采满，正在停止相机并保存最佳数据',saved:'4. 最佳数据已保存，可以关闭网页，请转到终端输入机械臂坐标'};
document.getElementById('cal-stage').textContent=phases[c.phase]||c.phase;
document.getElementById('cal-size').hidden=c.phase!=='preview';
if(c.phase==='preview'&&calibrationPhase!=='preview')document.getElementById('marker-mm').value=c.marker_length_mm;
document.getElementById('start').textContent='开始采集 '+c.target_windows+' 次';
document.getElementById('cal-best').textContent='已采集 '+c.compared_windows+'/'+c.target_windows+' 个合格窗口；'+(c.best?'最佳第 '+c.best.version+' 批，质量评分 '+f(c.best.score[0])+'（越低越好）':'尚无合格窗口');
document.getElementById('reset').disabled=!['preview','collecting'].includes(c.phase);
document.getElementById('cal-error').textContent=actionError||c.error||'';
document.getElementById('cal-result').textContent=c.result?JSON.stringify(c.result,null,2):'';
if(c.phase==='saved'){stopping=true;document.getElementById('stop').disabled=true;}
calibrationPhase=c.phase;}
async function act(payload){actionError='';try{const response=await fetch('/calibration',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
const data=await response.json();if(!response.ok)throw Error(data.error||'操作失败');renderCalibration(data);return data;
}catch(e){actionError=e.message;document.getElementById('cal-error').textContent=actionError;return null;}}
document.getElementById('start').onclick=async()=>{const button=document.getElementById('start');button.disabled=true;
await act({action:'start',marker_length_mm:Number(document.getElementById('marker-mm').value)});button.disabled=false;};
document.getElementById('reset').onclick=()=>act({action:'reset'});
let stopping=false;async function refresh(){try{
const response=await fetch('/snapshot',{cache:'no-store'});const d=await response.json();
renderCalibration(d.calibration);
document.getElementById('status').textContent='已处理 '+d.frames_processed+' 帧；最新帧距今 '+f(d.frame_age_s)+' 秒；'+(d.stability?.status||'等待相机');
document.getElementById('size').textContent=d.assumed_marker_length_m?'当前物理边长设置：'+f(1000*d.assumed_marker_length_m)+' mm。须与黑色正方形实测边长一致，否则三维数值尺度错误。':'';
if(d.image)document.getElementById('frame').src='data:image/jpeg;base64,'+d.image;
const body=document.getElementById('rows');body.replaceChildren();
if(d.stability){let c=d.stability.criteria;
document.getElementById('limits').textContent='建议门槛：'+c.window_s+' 秒 / 至少 '+c.min_frames+' 帧；检出与PnP通过率 ≥ '+(100*c.min_rate)+'%；短边 ≥ '+c.min_side_px+' px；重投影 ≤ '+c.max_rms_px+' px；角点RMS ≤ '+c.max_corner_jitter_px+' px；位置RMS/峰值 ≤ '+c.max_position_rms_mm+'/'+c.max_position_peak_mm+' mm；画面更新时间 ≤ '+c.max_frame_age_s+' 秒。';
for(const m of d.stability.markers){const tr=document.createElement('tr');tr.className=m.status;
for(const text of [m.id,f(100*m.detection_rate)+'%',f(100*m.quality_rate)+'%',f(m.min_side_px),f(m.max_rms_px),f(m.corner_jitter_px),f(m.position_rms_mm)+' / '+f(m.position_peak_mm),m.current_camera_xyz_m?m.current_camera_xyz_m.map(x=>f(1000*x)).join(', '):'—',m.status+' '+m.reasons.map(x=>reasons[x]||x).join('，')]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}body.appendChild(tr);}}
document.getElementById('error').textContent=d.error||'';
}catch(e){document.getElementById('error').textContent=stopping?'预览结束，请查看终端保存路径。':calibrationState?.phase==='captured'?'采集已结束，请查看终端保存结果并输入坐标。':'连接中断，请查看终端。';}
if(!stopping)setTimeout(refresh,250);}
document.getElementById('stop').onclick=async()=>{stopping=true;await fetch('/stop',{method:'POST'});document.getElementById('status').textContent='正在保存并退出……';};refresh();
</script></html>"""


class BrowserPreview:
    """仅绑定本机回环地址，通过 SSH 端口转发给远程浏览器查看。"""

    def __init__(self, port, *, calibration=None):
        self.stopped = threading.Event()
        self.final_delivered = threading.Event()
        self.lock = threading.Lock()
        self.snapshot = {"frames_processed": 0, "image": None, "stability": None}
        self.updated_at = None
        self.calibration = calibration
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    payload, content_type = _PREVIEW_HTML.encode(), "text/html; charset=utf-8"
                elif self.path == "/snapshot":
                    with owner.lock:
                        state = dict(owner.snapshot)
                        state["frame_age_s"] = None if owner.updated_at is None else time.monotonic() - owner.updated_at
                    assessment = state.get("stability")
                    if assessment and state["frame_age_s"] > assessment.get("criteria", {}).get("max_frame_age_s", 1.0):
                        # 返回副本，避免把过期状态写回已保存的那一帧指标。
                        state["stability"] = dict(assessment, status="STALE", ready_for_sampling=False,
                            markers=[dict(row, status="STALE", reasons=row["reasons"] + ["stale_frame"])
                                     for row in assessment.get("markers", [])])
                    if owner.calibration is not None:
                        state["calibration"] = owner.calibration.snapshot()
                    payload = json.dumps(state, ensure_ascii=False, allow_nan=False).encode()
                    content_type = "application/json; charset=utf-8"
                elif self.path == "/candidate" and owner.calibration is not None:
                    result = owner.calibration.snapshot().get("result")
                    if not result or "candidate" not in result:
                        self.send_error(409, "Candidate not ready")
                        return
                    payload = json.dumps(result["candidate"], ensure_ascii=False, indent=2, allow_nan=False).encode()
                    content_type = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                    if self.path == "/snapshot" and state.get("calibration", {}).get("phase") == "saved":
                        owner.final_delivered.set()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if self.path == "/calibration" and owner.calibration is not None:
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 16384:
                            raise ValueError("请求长度不合法")
                        origin = self.headers.get("Origin")
                        host = self.headers.get("Host")
                        if origin and origin not in (f"http://{host}", f"https://{host}"):
                            raise ValueError("请从当前预览页面提交")
                        response = owner.calibration.action(json.loads(self.rfile.read(length)))
                        status = 200
                    except (ValueError, TypeError, KeyError) as exc:
                        response, status = {"error": str(exc)}, 400
                    payload = json.dumps(response, ensure_ascii=False, allow_nan=False).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path != "/stop":
                    self.send_error(404)
                    return
                owner.stopped.set()
                self.send_response(204)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def publish(self, image, metrics):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise OSError("浏览器预览 JPEG 编码失败")
        with self.lock:
            self.snapshot = dict(metrics, image=base64.b64encode(encoded).decode("ascii"))
            self.updated_at = time.monotonic()

    def close(self):
        self.stopped.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


@dataclass(frozen=True)
class CalibrationQuality:
    """本工具的质量筛选参数，不代表硬件精度保证。长度单位见字段名。"""

    min_span_m: float = 0.01
    min_second_ratio: float = 0.01
    duplicate_tolerance_m: float = 1e-6
    min_marker_side_px: float = 20.0
    max_reprojection_px: float = 1.5
    ambiguity_error_gap_px: float = 0.1
    max_ambiguity_translation_m: float = 0.004
    max_repeatability_m: float = 0.004
    max_fit_error_mm: float = 4.0
    max_validation_error_mm: float = 4.0

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"质量参数 {name} 必须为有限正数")
        if self.min_second_ratio > 1:
            raise ValueError("min_second_ratio 不能大于 1")
        if max(self.max_fit_error_mm, self.max_validation_error_mm) > 4:
            raise ValueError("拟合/外参验证门槛不能放宽至 4 mm 以上")


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为数值")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须为有限正数")
    return float(value)


def load_camera_intrinsics(intrin_path: Path):
    """只读校验 RGB 参数，不改写文件或处理其他模块的深度参数。"""
    with Path(intrin_path).open(encoding="utf-8") as f:
        data = json.load(f)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("不支持的内参 schema_version")
    rgb = data["rgb"]
    for name in ("width", "height"):
        if type(rgb[name]) is not int or rgb[name] <= 0:
            raise ValueError(f"RGB {name} 必须为正整数")
    fx = _positive_number(rgb["fx"], "fx")
    fy = _positive_number(rgb["fy"], "fy")
    K = np.array([[fx, 0, rgb["cx"]], [0, fy, rgb["cy"]], [0, 0, 1]], dtype=np.float64)
    dist = np.asarray(rgb["dist"], dtype=np.float64)
    if not np.isfinite(K).all() or dist.shape != (5,) or not np.isfinite(dist).all():
        raise ValueError("RGB 内参必须有限，畸变须为五项 Brown-Conrady 系数")
    return K, dist, data


def point_set_diagnostics(points, quality=None):
    quality = quality or CalibrationQuality()
    quality.validate()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("点集必须为 (N,3)，至少三个非共线点")
    if not np.isfinite(points).all():
        raise ValueError("点集含 NaN/Inf")
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    pairs = distances[np.triu_indices(len(points), 1)]
    if pairs.min() <= quality.duplicate_tolerance_m:
        raise ValueError("点集含重复或过近的点")
    span = float(pairs.max())
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    ratio = float(singular[1] / singular[0])
    if span < quality.min_span_m or ratio < quality.min_second_ratio:
        raise ValueError("点集跨度不足或共线/近共线")
    return {"span_m": span, "singular_values_m": singular.tolist(), "second_ratio": ratio}


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("外参必须为有限 4×4 矩阵")
    R = matrix[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("旋转矩阵不正交")
    if not np.isclose(np.linalg.det(R), 1, atol=1e-8, rtol=0):
        raise ValueError("旋转矩阵不是右手系")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("齐次矩阵末行错误")
    return matrix


def solve_rigid_transform_3d(pts_cam, pts_base, *, quality=None):
    """求 P_base = R @ P_cam + t，输入/平移为 m；共面非共线点合法。"""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    pts_base = np.asarray(pts_base, dtype=np.float64)
    if pts_cam.shape != pts_base.shape:
        raise ValueError("两组点形状必须相同")
    point_set_diagnostics(pts_cam, quality)
    point_set_diagnostics(pts_base, quality)
    centroid_cam, centroid_base = pts_cam.mean(axis=0), pts_base.mean(axis=0)
    H = (pts_cam - centroid_cam).T @ (pts_base - centroid_base)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = (centroid_base - R @ centroid_cam).reshape(3, 1)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.ravel()
    validate_transform(T)
    return R, t


def _marker_pose(corners, K, dist, marker_length_m, quality):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("标记角点必须为有限 (4,2)")
    if np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).min() < quality.min_marker_side_px:
        raise ValueError("标记可见边长过小")
    half = marker_length_m / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    poses = []
    # IPPE 提供平面候选；ITERATIVE 补充正视退化数值情形，同样接受质量检查。
    try:
        result = cv2.solvePnPGeneric(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if result[0]:
            poses.extend(zip(result[1], result[2]))
    except cv2.error:
        pass
    try:
        ok, rv, tv = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            poses.append((rv, tv))
    except cv2.error:
        pass
    scored = []
    for rv, tv in poses:
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            continue
        rotation, _ = cv2.Rodrigues(rv)
        if np.any((rotation @ obj.T + tv.reshape(3, 1))[2] <= 0):
            continue
        projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
        if np.isfinite(error):
            scored.append((error, rv, tv))
    scored.sort(key=lambda item: item[0])
    if not scored or scored[0][0] > quality.max_reprojection_px:
        raise ValueError("PnP 无合格正深度解或重投影误差超限")
    error, rv, tv = scored[0]
    for other_error, _, other_tv in scored[1:]:
        if (other_error - error <= quality.ambiguity_error_gap_px
                and np.linalg.norm(other_tv.ravel() - tv.ravel()) > quality.max_ambiguity_translation_m):
            raise ValueError("平面 PnP 候选中心存在歧义，请重新采集")
    return {"P_cam": tv.ravel(), "corners": corners, "rvec": rv, "tvec": tv,
            "reprojection_error_px": error, "candidate_errors_px": [item[0] for item in scored]}


def _aruco_observations(image_bgr):
    """保留解码出的全部 ID/角点及未解码四边形，供标定和预览共用。"""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ValueError("标定图像必须为非空 uint8 BGR")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    else:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                  else cv2.aruco.DetectorParameters_create())
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(image_bgr)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image_bgr, dictionary, parameters=parameters)
    return corners, ids, rejected


def detect_markers_pnp(image_bgr, K, dist, marker_length_m=0.080, *, expected_ids=None, quality=None):
    """检测标记中心，遇到重复 ID 或不合格姿态明确拒绝本帧。"""
    quality = quality or CalibrationQuality()
    quality.validate()
    _positive_number(marker_length_m, "marker_length_m")
    corners, ids, _ = _aruco_observations(image_bgr)
    annotated = image_bgr.copy()
    if ids is None:
        return {}, annotated
    ids = ids.ravel().tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("同一画面存在重复标记 ID")
    if expected_ids is not None and not set(ids).issubset(expected_ids):
        raise ValueError(f"发现非期望 ID：{sorted(set(ids) - set(expected_ids))}")
    results = {}
    for mid, points in zip(ids, corners):
        try:
            pose = _marker_pose(points[0], K, dist, marker_length_m, quality)
        except ValueError as exc:
            raise ValueError(f"标记 #{mid}: {exc}") from exc
        results[int(mid)] = pose
        cv2.polylines(annotated, [np.int32(pose["corners"])], True, (0, 255, 0), 2)
        center = tuple(np.int32(pose["corners"].mean(axis=0)))
        cv2.putText(annotated, f"ID:{mid} Z:{pose['P_cam'][2]:.3f}m", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(annotated, K, dist, pose["rvec"], pose["tvec"], marker_length_m / 2)
        else:
            cv2.aruco.drawAxis(annotated, K, dist, pose["rvec"], pose["tvec"], marker_length_m / 2)
    return results, annotated


def inspect_markers(image_bgr, K, dist, marker_length_m=0.080, *,
                    expected_ids=(1, 2, 3, 4, 5, 6), quality=None):
    """仅检查识别与单帧 PnP；保留已解码但质量不合格的标记以便诊断。"""
    quality = quality or CalibrationQuality()
    quality.validate()
    _positive_number(marker_length_m, "marker_length_m")
    corners, ids, rejected = _aruco_observations(image_bgr)
    ids = [] if ids is None else [int(mid) for mid in ids.ravel()]
    annotated = image_bgr.copy()
    items = []
    for mid, points in zip(ids, corners):
        points = np.asarray(points, dtype=np.float64).reshape(4, 2)
        edges = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        item = {"id": mid, "corners_px": points.tolist(), "side_lengths_px": edges.tolist(),
                "min_side_px": float(edges.min()), "quality_pass": False,
                "reprojection_error_px": None, "reason": None}
        if ids.count(mid) > 1:
            item["reason"] = "重复 ID"
        elif expected_ids is not None and mid not in expected_ids:
            item["reason"] = "非期望 ID"
        else:
            try:
                pose = _marker_pose(points, K, dist, marker_length_m, quality)
                item.update({"quality_pass": True,
                             "reprojection_error_px": pose["reprojection_error_px"],
                             "camera_xyz_m": pose["P_cam"].tolist()})
            except ValueError as exc:
                item["reason"] = str(exc)
        items.append(item)
        color = (0, 200, 0) if item["quality_pass"] else (0, 140, 255)
        cv2.polylines(annotated, [np.int32(points)], True, color, 2)
        for point in np.int32(points):
            cv2.circle(annotated, tuple(point), 3, color, -1)
        status = f"RMS={item['reprojection_error_px']:.2f}px" if item["quality_pass"] else "QUALITY FAIL"
        label = f"ID:{mid} min:{item['min_side_px']:.1f}px {status}"
        anchor = (max(0, int(points[:, 0].min())), max(18, int(points[:, 1].min()) - 7))
        cv2.putText(annotated, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    report = {"decoded_count": len(items), "quality_pass_count": sum(item["quality_pass"] for item in items),
              "undecoded_quad_count": len(rejected), "markers": items,
              "missing_ids": sorted(set(expected_ids or ()) - set(ids))}
    cv2.putText(annotated, f"Decoded:{len(items)} Quality:{report['quality_pass_count']} | q/ESC exit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    return report, annotated


def _preview_live_frames(device, intrinsics, frame_limit):
    """预览独立持有 RGB 资源；迭代器关闭时释放，不接入深度或机械臂。"""
    device = int(device) if str(device).isdigit() else str(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开指定 RGB 设备：{device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, intrinsics["rgb"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, intrinsics["rgb"]["height"])
        for _ in range(8):
            ok, _ = cap.read()
            if not ok:
                raise ValueError("RGB 预读失败")
        index = 0
        while frame_limit is None or index < frame_limit:
            ok, image = cap.read()
            if not ok:
                raise ValueError("RGB 采集失败")
            _check_image(image, intrinsics)
            yield image
            index += 1
    finally:
        cap.release()


def _stability_panel(image, assessment):
    panel = np.zeros((max(image.shape[0], 330), 530, 3), dtype=np.uint8)
    lines = [f"Sampling status: {assessment['status']}",
             f"Window: {assessment['window_frames']} frames, {assessment['window_span_s']:.1f}s",
             "READY = sampling check, NOT accuracy acceptance"]
    for row in assessment["markers"]:
        def fmt(value):
            return "--" if value is None else f"{value:.2f}"
        lines.extend([f"ID {row['id']} {row['status']} detect={row['detection_rate']:.0%} pose={row['quality_rate']:.0%}",
                      f" side={fmt(row['min_side_px'])}px rms={fmt(row['max_rms_px'])}px corner={fmt(row['corner_jitter_px'])}px",
                      f" XYZ RMS/peak={fmt(row['position_rms_mm'])}/{fmt(row['position_peak_mm'])}mm"])
    height = max(image.shape[0], len(lines) * 22 + 20)
    canvas = np.zeros((height, image.shape[1] + panel.shape[1], 3), dtype=np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (image.shape[1] + 10, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (200, 230, 230), 1)
    return canvas


def run_marker_preview(image_source="live", intrin_file="configs/vision/camera_intrinsics.json", *,
                       rgb_device="0", output_root=None, marker_length_m=0.080,
                       expected_ids=(1, 2, 3, 4, 5, 6), sample_count=5, no_display=False,
                       browser=False, preview_port=8765, criteria=None, quality=None):
    """独立实时识别与稳定性预览，无单位/TCP/坐标输入，不求机器人外参。"""
    frames = None
    window_created = False
    session_dir = None
    browser_view = None
    record = {"schema_version": 1, "mode": "detection_preview", "status": "failed",
              "source": str(image_source), "frames_processed": 0,
              "note": "READY 仅为采集质量筛选，不是物理尺寸、坐标真值或 CAL-040 验收；时间为主机处理时刻，非曝光时间"}
    try:
        quality = quality or CalibrationQuality()
        quality.validate()
        criteria = criteria or PreviewCriteria()
        criteria.validate()
        _positive_number(marker_length_m, "marker_length_m")
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("sample_count 必须为正整数")
        if type(preview_port) is not int or not 0 <= preview_port <= 65535:
            raise ValueError("预览端口须为 0～65535 的整数")
        if not expected_ids or any(type(mid) is not int or not 0 <= mid < 100 for mid in expected_ids) or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("期望 ID 须为不重复的 0～99 整数")
        intrin_path = Path(intrin_file)
        if not intrin_path.is_absolute():
            intrin_path = PROJECT_ROOT / intrin_path
        K, dist, intrinsics = load_camera_intrinsics(intrin_path)
        if browser:
            browser_view = BrowserPreview(preview_port)
            no_display = True
            print(f"浏览器预览：http://127.0.0.1:{browser_view.port}/（仅香橙派本机回环地址）")
            print(f"远程 VS Code 请转发端口 {browser_view.port}，再在电脑浏览器打开转发地址。")
        elif not no_display and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("未发现图形会话，使用无窗口识别；远程实时画面请用 --browser。")
            no_display = True
        elif not no_display:
            print(f"OpenCV 窗口显示在 DISPLAY={os.environ.get('DISPLAY', '本地桌面')}；SSH/远程 VS Code 请用 --browser。")
        session_dir = _new_session(output_root)
        record.update({"intrinsics_sha256": _sha256(intrin_path), "assumed_marker_length_m": marker_length_m,
                       "quality": asdict(quality), "criteria": asdict(criteria), "per_id": {},
                       "display_mode": "browser" if browser else ("none" if no_display else "desktop")})
        tracker = MarkerStability(expected_ids, criteria)
        if image_source == "live":
            frames = _preview_live_frames(rgb_device, intrinsics, sample_count if no_display and not browser else None)
        else:
            frames = iter(_acquire_images(image_source, rgb_device, intrinsics, 1))
        if not no_display:
            cv2.namedWindow("ArUco detection preview", cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow("ArUco detection preview", 1500, 720)
            waiting = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(waiting, "Opening RGB camera (MJPG)...", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("ArUco detection preview", waiting)
            cv2.waitKey(1)
        print("仅识别预览：固定相机和标记，观察最近时间窗口；READY 不是标定精度验收。")
        raw = annotated = None
        last_complete_frame = None
        last_print = -float("inf")
        try:
            for raw in frames:
                report, annotated = inspect_markers(raw, K, dist, marker_length_m,
                                                    expected_ids=expected_ids, quality=quality)
                now = time.monotonic()
                assessment = tracker.update(report, now)
                record["frames_processed"] += 1
                record["last_frame"], record["stability"] = report, assessment
                for mid in set(item["id"] for item in report["markers"]):
                    matched = [item for item in report["markers"] if item["id"] == mid]
                    stats = record["per_id"].setdefault(str(mid), {"decoded_frames": 0, "quality_pass_frames": 0,
                                                                "min_side_px": float("inf"), "max_min_side_px": 0.0})
                    stats["decoded_frames"] += 1
                    stats["quality_pass_frames"] += int(len(matched) == 1 and matched[0]["quality_pass"])
                    stats["min_side_px"] = min(stats["min_side_px"], *(item["min_side_px"] for item in matched))
                    stats["max_min_side_px"] = max(stats["max_min_side_px"], *(item["min_side_px"] for item in matched))
                annotated = _stability_panel(annotated, assessment)
                last_complete_frame = (raw, annotated)
                if browser_view is not None:
                    browser_view.publish(annotated, {"frames_processed": record["frames_processed"],
                                                   "assumed_marker_length_m": marker_length_m,
                                                   "stability": assessment})
                if now - last_print >= .5 or image_source != "live":
                    details = "; ".join(f"ID:{row['id']} {row['status']} 检出:{row['detection_rate']:.0%} PnP:{row['quality_rate']:.0%}"
                                        for row in assessment["markers"])
                    print(f"帧 {record['frames_processed']}：{assessment['status']}；{details}")
                    last_print = now
                if browser_view is not None and browser_view.stopped.is_set():
                    break
                if not no_display:
                    cv2.imshow("ArUco detection preview", annotated)
                    key = cv2.waitKey(1 if image_source == "live" else 0) & 0xff
                    if key in (ord("q"), 27) or image_source != "live":
                        break
                    if cv2.getWindowProperty("ArUco detection preview", cv2.WND_PROP_VISIBLE) < 1:
                        break
            if browser_view is not None and image_source != "live":
                while not browser_view.stopped.wait(.1):
                    pass
        except KeyboardInterrupt:
            if last_complete_frame is None:
                raise
            record["exit_reason"] = "keyboard_interrupt_after_frames"
        if last_complete_frame is None:
            raise ValueError("没有采集到预览帧")
        raw, annotated = last_complete_frame
        _write_image(session_dir / "preview_raw.png", raw)
        _write_image(session_dir / "preview_annotated.png", annotated)
        record["status"] = "completed"
        atomic_write_json(session_dir / "preview.json", record)
        print(f"识别结果与最后一帧已保存：{session_dir}")
        return True
    except (Exception, KeyboardInterrupt) as exc:
        record.update({"status": "failed", "error": str(exc) or type(exc).__name__})
        if session_dir is not None:
            try:
                atomic_write_json(session_dir / "preview_failure.json", record)
            except (OSError, ValueError) as save_exc:
                print(f"预览失败记录未保存：{save_exc}", file=sys.stderr)
        print(f"识别预览失败：{record['error']}；远程实时画面请用 --browser。", file=sys.stderr)
        return False
    finally:
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        if browser_view is not None:
            browser_view.close()
        if window_created:
            try:
                cv2.destroyWindow("ArUco detection preview")
            except cv2.error:
                pass


def parse_user_coord_input(input_str, unit):
    """显式单位解析，支持空格/逗号；不按坐标大小猜单位。"""
    if unit not in ("mm", "m"):
        raise ValueError("坐标单位必须显式选择 mm 或 m")
    raw = input_str.strip().replace(",", " ").split()
    if len(raw) != 3:
        raise ValueError("必须输入三个坐标 X Y Z")
    values = np.array([float(x) for x in raw], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("坐标含 NaN/Inf")
    return values / 1000 if unit == "mm" else values


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_output_path(path):
    """仓库内只允许新会话目录写入；不改写现有配置或标定资产。"""
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT) and not path.is_relative_to(PROJECT_ROOT / "calibration_assets" / "sessions"):
        raise ValueError("仓库内输出只允许 calibration_assets/sessions，不能写受保护路径")
    return path


def _new_session(output_root):
    root = _check_output_path(output_root or PROJECT_ROOT / "calibration_assets" / "sessions")
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def atomic_write_json(path, data):
    """原子创建 JSON，不覆盖已有目标；临时文件与目标在同一文件系统。"""
    path = _check_output_path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    json.loads(payload)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # hard link 原子发布且目标已存在时失败，避免 replace 的覆盖竞态。
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_image(path, image):
    if path.exists() or not cv2.imwrite(str(path), image):
        raise OSError(f"图像保存失败或目标已存在：{path}")


def _check_image(image, intrinsics):
    rgb = intrinsics["rgb"]
    if image is None or image.dtype != np.uint8 or image.shape != (rgb["height"], rgb["width"], 3):
        raise ValueError("图像解码失败，或实际 RGB 帧尺寸/类型与内参不匹配")


def _acquire_images(source, device, intrinsics, sample_count):
    if source != "live":
        image = cv2.imread(str(source))
        _check_image(image, intrinsics)
        return [image]
    device = int(device) if str(device).isdigit() else str(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开指定 RGB 设备：{device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, intrinsics["rgb"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, intrinsics["rgb"]["height"])
        for _ in range(8):
            ok, image = cap.read()
            if not ok:
                raise ValueError("RGB 预读失败")
        images = []
        for _ in range(sample_count):
            ok, image = cap.read()
            if not ok:
                raise ValueError("RGB 采集失败")
            _check_image(image, intrinsics)
            images.append(image)
        return images
    finally:
        cap.release()


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _residual_report(matrix, camera, base):
    predicted = np.asarray(camera) @ matrix[:3, :3].T + matrix[:3, 3]
    errors = np.linalg.norm(predicted - base, axis=1) * 1000
    if not np.isfinite(errors).all():
        raise ValueError("残差计算产生非有限结果")
    return {"errors_mm": errors.tolist(), "max_mm": float(errors.max()),
            "rmse_mm": float(np.sqrt(np.mean(errors ** 2)))}


def validate_independent_points(matrix, validation, fit_camera, fit_base, quality):
    """独立点文件不参与拟合，拒绝复用拟合位置；不等同于整机 CAL-040。"""
    quality.validate()
    if type(validation.get("schema_version")) is not int or validation["schema_version"] != 1 or validation.get("unit") not in ("mm", "m"):
        raise ValueError("独立验证文件必须指定 schema_version=1 与 unit=mm|m")
    if not isinstance(validation.get("reference"), str) or not validation["reference"].strip():
        raise ValueError("独立验证必须记录参考测量方法 reference")
    uncertainty = validation.get("reference_uncertainty_mm")
    if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) or not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("须记录有限非负的 reference_uncertainty_mm")
    points = validation["points"]
    if not isinstance(points, list) or len(points) < 4 or len({p["id"] for p in points}) != len(points):
        raise ValueError("至少四个独立验证点，ID 不得重复")
    camera = np.asarray([p["camera_xyz_m"] for p in points], dtype=np.float64)
    base = np.asarray([p["base_xyz"] for p in points], dtype=np.float64)
    if validation["unit"] == "mm":
        base = base / 1000
    point_set_diagnostics(camera, quality)
    point_set_diagnostics(base, quality)
    for current, fitted in ((camera, fit_camera), (base, fit_base)):
        if np.any(np.linalg.norm(current[:, None] - np.asarray(fitted)[None, :], axis=2) <= quality.duplicate_tolerance_m):
            raise ValueError("独立验证复用了拟合位置")
    report = _residual_report(validate_transform(matrix), camera, base)
    report.update({"status": "passed" if report["max_mm"] <= quality.max_validation_error_mm else "failed",
                   "threshold_mm": quality.max_validation_error_mm, "input": validation})
    return report


def _save_visual_candidate(session, directory, metadata, quality, validation_file=None):
    """使用锁定批次的多帧中位数求解；不以机器人拟合残差挑选观测批次。"""
    with session.lock:
        if session.phase != "solving" or len(session.comparisons) != session.target_windows:
            raise ValueError("正式标定状态不允许求解")
        data = session.export()
        raw = session.best["raw"].copy()
        annotated = session.best["annotated"].copy()
    pairs = [{"id": mid, "camera_xyz_m": data["camera_points_m"][str(mid)],
              "base_xyz_m": (np.asarray(data["coordinates_mm"][mid]) / 1000).tolist(),
              "raw_input_mm": data["coordinates_mm"][mid]} for mid in data["expected_ids"]]
    camera = np.asarray([pair["camera_xyz_m"] for pair in pairs])
    base = np.asarray([pair["base_xyz_m"] for pair in pairs])
    distribution = {"camera": point_set_diagnostics(camera, quality), "base": point_set_diagnostics(base, quality)}
    R, t = solve_rigid_transform_3d(camera, base, quality=quality)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = R, t.ravel()
    fit = _residual_report(matrix, camera, base)
    fit["threshold_mm"] = quality.max_fit_error_mm
    if fit["max_mm"] > quality.max_fit_error_mm:
        raise ValueError(f"拟合残差超限：{fit['max_mm']:.3f} mm，不生成合格候选")
    validation = {"status": "not_performed"}
    if validation_file is not None:
        with Path(validation_file).open(encoding="utf-8") as f:
            validation_input = json.load(f)
        validation = validate_independent_points(matrix, validation_input, camera, base, quality)
        validation["source_sha256"] = _sha256(validation_file)
        if validation["status"] != "passed":
            raise ValueError("独立外参验证超限，不生成合格候选")
    candidate = {"schema_version": 1, "from_frame": "rgb_optical", "to_frame": "base_link",
                 "matrix": validate_transform(matrix).tolist()}
    record = dict(metadata, schema_version=1, mode="visual_calibration", status="candidate", unit="mm",
                  marker_length_m=data["marker_length_m"], selection=data, correspondences=pairs,
                  point_distribution=distribution, fit=fit, independent_validation=validation,
                  depth_validation="not_performed_rgb_only", quality=asdict(quality))
    # 先保留数据和图像，再原子创建候选；失败时移除本次刚创建的候选。
    prefix = f"attempt_{session.attempt:03d}"
    _write_image(directory / f"{prefix}_selected_raw.png", raw)
    _write_image(directory / f"{prefix}_selected_annotated.png", annotated)
    record["selected_image"] = f"{prefix}_selected_raw.png"
    record["selected_image_role"] = "选定窗口末帧；相机点来自窗口内多帧中位数"
    path = directory / "T_base_camera.candidate.json"
    created = False
    try:
        atomic_write_json(path, candidate)
        created = True
        record["candidate_file"], record["candidate_sha256"] = path.name, _sha256(path)
        atomic_write_json(directory / "session.json", record)
    except (Exception, KeyboardInterrupt):
        if created:
            path.unlink(missing_ok=True)
        raise
    return {"candidate": candidate, "fit": fit, "independent_validation": validation["status"],
            "candidate_path": str(path), "session_path": str(directory / "session.json"),
            "selected_version": data["selected_window"]["version"],
            "note": "拟合残差不是 CAL-040；正式配置未替换"}


def _save_visual_selection(session, directory, metadata):
    """坐标录入前落盘：先存图像，最后创建完整选择记录作为保存成功标志。"""
    with session.lock:
        if session.phase != "captured" or len(session.comparisons) != session.target_windows:
            raise ValueError("尚未采满合格窗口，不能保存最佳批次")
        data = session.export()
        raw, annotated = session.best["raw"].copy(), session.best["annotated"].copy()
    _write_image(directory / "selected_raw.png", raw)
    _write_image(directory / "selected_annotated.png", annotated)
    record = dict(metadata, schema_version=1, status="observations_saved", selection=data,
                  selected_image="selected_raw.png", annotated_image="selected_annotated.png",
                  selected_image_role="选定窗口末帧；相机点来自窗口内多帧中位数")
    path = directory / "selection.json"
    atomic_write_json(path, record)
    return path


def _read_visual_coordinates(session):
    """只接收终端 mm 坐标，不再依赖相机或网页；q 保留已采集数据并取消求解。"""
    for mid in session.expected_ids:
        while True:
            text = input(f"#{mid} 接触点 X Y Z（mm；q 取消）：").strip()
            if text.lower() == "q":
                return False
            try:
                point = parse_user_coord_input(text, "mm")
            except ValueError as exc:
                print(f"输入错误：{exc}；请重新输入该点。")
                continue
            session.coordinates_mm[mid] = (point * 1000).tolist()
            break
    return True


def run_visual_calibration(intrin_file="configs/vision/camera_intrinsics.json", *, rgb_device="0",
                           output_root=None, marker_length_m=.080, expected_ids=(1, 2, 3, 4, 5, 6),
                           preview_port=8765, criteria=None, quality=None, target_windows=10,
                           tcp_description=DEFAULT_TCP_DESCRIPTION, tool_version="unspecified",
                           validation_file=None):
    """网页开始后采满合格窗口，保存最佳批次并释放相机，再到终端输入 mm 坐标。"""
    frames = view = session = directory = None
    try:
        quality = quality or CalibrationQuality()
        quality.validate()
        if type(preview_port) is not int or not 0 <= preview_port <= 65535:
            raise ValueError("预览端口须为 0～65535 的整数")
        intrin_path = Path(intrin_file)
        if not intrin_path.is_absolute():
            intrin_path = PROJECT_ROOT / intrin_path
        K, dist, intrinsics = load_camera_intrinsics(intrin_path)
        session = VisualCalibrationSession(expected_ids, criteria, marker_length_m, target_windows)
        view = BrowserPreview(preview_port, calibration=session)
        directory = _new_session(output_root)
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "intrinsics_path": str(intrin_path.resolve()), "intrinsics_sha256": _sha256(intrin_path),
                    "rgb_intrinsics": intrinsics["rgb"], "rgb_device": str(rgb_device),
                    "tcp_description": tcp_description or DEFAULT_TCP_DESCRIPTION, "tool_version": tool_version,
                    "coordinate_source": "contact", "robot_pose": "not_provided"}
        print(f"正式可视化标定：http://127.0.0.1:{view.port}/；远程电脑请转发端口 {view.port}。")
        print(f"默认 mm，接触点为闭合夹爪两指尖中心；页面点击开始后采集 {target_windows} 次合格窗口并自动停止。")
        frames = _preview_live_frames(rgb_device, intrinsics, None)
        frame_count = 0
        try:
            for raw in frames:
                length, generation = session.frame_settings()
                report, annotated = inspect_markers(raw, K, dist, length,
                                                    expected_ids=expected_ids, quality=quality)
                assessment = session.observe(report, time.monotonic(), raw, annotated, generation)
                if assessment is None:
                    continue
                frame_count += 1
                view.publish(_stability_panel(annotated, assessment),
                             {"frames_processed": frame_count, "assumed_marker_length_m": length,
                              "stability": assessment})
                if session.phase == "captured" or view.stopped.is_set():
                    break
        except KeyboardInterrupt:
            pass
        # 先关闭生成器以释放相机，后续保存与坐标录入绝不继续取帧。
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        frames = None
        if session.phase == "captured":
            selection_path = _save_visual_selection(session, directory, metadata)
            with session.lock:
                session.phase = "saved"
                session.result = {"selection_path": str(selection_path),
                                  "selected_version": session.best["summary"]["version"],
                                  "next": "可以关闭网页；请在终端输入机械臂坐标（mm）"}
                session.revision += 1
            # 给页面一次显示保存结果的机会；无人打开页面也会自动转入终端。
            view.final_delivered.wait(timeout=.75)
            view.close()
            view = None
            print(f"已采满 {target_windows} 次合格数据，相机检测已停止。最佳第 {session.best['summary']['version']} 批已保存：{selection_path}")
            print("现在可以关闭网页。保持相机和标记固定；用闭合夹爪两指尖中心触碰各标记中心，输入基座系接触点坐标。")
            session.phase = "coordinates"
            if _read_visual_coordinates(session):
                session.phase = "solving"
                session.attempt += 1
                result = _save_visual_candidate(session, directory, metadata, quality, validation_file)
                session.phase, session.result = "complete", result
                print(f"候选已保存：{result['candidate_path']}；拟合最大残差 {result['fit']['max_mm']:.3f} mm。")
                return True
        atomic_write_json(directory / "cancelled.json", dict(metadata, status="cancelled", selection=session.export()))
        print(f"未完成标定，当前记录已保存：{directory}；未生成候选。")
        return False
    except (KeyboardInterrupt, EOFError) as exc:
        if directory is not None:
            try:
                atomic_write_json(directory / "cancelled.json", {"status": "cancelled",
                    "reason": type(exc).__name__, "selection": session.export() if session else None})
            except (OSError, ValueError) as save_exc:
                print(f"取消记录未保存：{save_exc}", file=sys.stderr)
        print(f"已取消标定，已保存的数据保留：{directory}；未生成候选。")
        return False
    except Exception as exc:
        if directory is not None:
            try:
                atomic_write_json(directory / "failure.json", {"status": "failed", "error": str(exc),
                    "selection": session.export() if session else None})
            except (OSError, ValueError) as save_exc:
                print(f"失败记录未保存：{save_exc}", file=sys.stderr)
        print(f"可视化标定失败：{str(exc) or type(exc).__name__}", file=sys.stderr)
        return False
    finally:
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        if view is not None:
            view.close()


def run_touch_calibration(image_source="live", intrin_file="configs/vision/camera_intrinsics.json", *,
                          unit="mm", rgb_device="0", output_root=None, tcp_description=None,
                          tool_version="unspecified", coordinate_source="contact", marker_length_m=0.080,
                          expected_ids=(1, 2, 3, 4, 5, 6), sample_count=5,
                          validation_file=None, quality=None):
    """交互式标定，True 仅表示合格候选已保存；绝不安装到正式配置。"""
    session_dir = None
    record = {"schema_version": 1, "status": "failed", "depth_validation": "not_performed_rgb_only",
              "independent_validation": {"status": "not_performed"}, "source": str(image_source),
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    candidate_path = None
    candidate_created = False
    try:
        quality = quality or CalibrationQuality()
        quality.validate()
        if coordinate_source != "contact":
            raise ValueError("只接受基座系接触点坐标；法兰 XYZ 缺少姿态/工具转换，不能直接使用")
        if type(sample_count) is not int or sample_count < 1:
            raise ValueError("sample_count 必须为正整数")
        if not expected_ids or any(type(mid) is not int or not 0 <= mid < 100 for mid in expected_ids) or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("期望 ID 须为不重复的 0～99 整数")
        _positive_number(marker_length_m, "marker_length_m")
        if unit is None:
            unit = "mm"
        if unit not in ("mm", "m"):
            raise ValueError("单位必须为 mm 或 m")
        if tcp_description is None:
            tcp_description = DEFAULT_TCP_DESCRIPTION
        if not isinstance(tcp_description, str) or not tcp_description.strip():
            raise ValueError("必须记录接触点/TCP 定义")
        intrin_path = Path(intrin_file)
        if not intrin_path.is_absolute():
            intrin_path = PROJECT_ROOT / intrin_path
        K, dist, intrinsics = load_camera_intrinsics(intrin_path)
        session_dir = _new_session(output_root)
        record.update({"unit": unit, "coordinate_source": coordinate_source, "tcp_description": tcp_description,
                       "tool_version": tool_version, "robot_pose": "not_provided",
                       "intrinsics_path": str(intrin_path.resolve()), "intrinsics_sha256": _sha256(intrin_path),
                       "rgb_intrinsics": intrinsics["rgb"], "rgb_device": str(rgb_device) if image_source == "live" else None,
                       "marker_length_m": marker_length_m, "expected_ids": list(expected_ids),
                       "quality": asdict(quality), "frames": [], "correspondences": []})
        print("RGB-only 点触标定；请保持标记固定，机械臂先停在 Home。")
        images = _acquire_images(image_source, rgb_device, intrinsics, sample_count)
        results = []
        for index, image in enumerate(images):
            _write_image(session_dir / f"{index:03d}_raw.png", image)
            frame_record = {"index": index, "size": [image.shape[1], image.shape[0]],
                            "raw": f"{index:03d}_raw.png"}
            record["frames"].append(frame_record)
            markers, annotated = detect_markers_pnp(image, K, dist, marker_length_m,
                                                    expected_ids=expected_ids, quality=quality)
            frame_record["markers"] = _serializable(markers)
            _write_image(session_dir / f"{index:03d}_annotated.png", annotated)
            results.append(markers)
        common_ids = sorted(set.intersection(*(set(result) for result in results)))
        if len(common_ids) < 4:
            raise ValueError("所有采样帧共有的合格标记不足四个")
        camera_points, repeatability = {}, {}
        record["repeatability"] = repeatability
        for mid in common_ids:
            samples = np.array([result[mid]["P_cam"] for result in results])
            center = samples.mean(axis=0)
            deviation = float(np.linalg.norm(samples - center, axis=1).max())
            repeatability[str(mid)] = {"sample_count": len(samples), "max_deviation_m": deviation,
                                      "status": "measured" if len(samples) > 1 else "not_evaluated_single_frame"}
            if deviation > quality.max_repeatability_m:
                raise ValueError(f"标记 #{mid} 多帧中心波动超限")
            camera_points[mid] = center
        for mid in common_ids:
            while True:
                raw_input = input(f"#{mid} 接触点 X Y Z（{unit}；s 跳过，q 取消）：").strip()
                if raw_input.lower() == "q":
                    raise ValueError("用户取消")
                if raw_input.lower() == "s":
                    break
                try:
                    base = parse_user_coord_input(raw_input, unit)
                except ValueError as exc:
                    print(f"输入错误：{exc}")
                    continue
                record["correspondences"].append({"id": mid, "camera_xyz_m": camera_points[mid].tolist(),
                                                   "base_xyz_m": base.tolist(), "raw_input": raw_input})
                print(f"已记录 #{mid}：base={base.tolist()} m")
                break
        pairs = record["correspondences"]
        if len(pairs) < 4:
            raise ValueError("有效对应点不足四个")
        camera = np.array([p["camera_xyz_m"] for p in pairs])
        base = np.array([p["base_xyz_m"] for p in pairs])
        record["point_distribution"] = {"camera": point_set_diagnostics(camera, quality),
                                        "base": point_set_diagnostics(base, quality)}
        R, t = solve_rigid_transform_3d(camera, base, quality=quality)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = R, t.ravel()
        record["fit"] = _residual_report(matrix, camera, base)
        record["fit"]["threshold_mm"] = quality.max_fit_error_mm
        print(f"拟合残差：max={record['fit']['max_mm']:.3f} mm，RMSE={record['fit']['rmse_mm']:.3f} mm（不等于 CAL-040）")
        if record["fit"]["max_mm"] > quality.max_fit_error_mm:
            raise ValueError("拟合残差超限，不生成合格候选")
        if validation_file is not None:
            validation_path = Path(validation_file)
            record["independent_validation"] = {"status": "invalid", "source": str(validation_path)}
            with validation_path.open(encoding="utf-8") as f:
                validation = json.load(f)
            record["independent_validation"] = validate_independent_points(matrix, validation, camera, base, quality)
            record["independent_validation"]["source_sha256"] = _sha256(validation_path)
            if record["independent_validation"]["status"] != "passed":
                raise ValueError("独立外参验证超限")
        candidate = {"schema_version": 1, "from_frame": "rgb_optical", "to_frame": "base_link",
                     "matrix": validate_transform(matrix).tolist()}
        candidate_path = session_dir / "T_base_camera.candidate.json"
        atomic_write_json(candidate_path, candidate)
        candidate_created = True
        record.update({"status": "candidate", "candidate_file": candidate_path.name,
                       "candidate_sha256": _sha256(candidate_path)})
        atomic_write_json(session_dir / "session.json", record)
        print(f"候选已保存：{candidate_path}")
        print(f"独立外参验证：{record['independent_validation']['status']}；深度未验证（RGB-only）。正式配置未替换。")
        return True
    except (Exception, KeyboardInterrupt) as exc:
        if candidate_created:
            try:
                candidate_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                print(f"候选清理失败（不得使用无完整会话记录的候选）：{cleanup_exc}", file=sys.stderr)
        record.update({"status": "failed", "error": str(exc) or type(exc).__name__})
        record.pop("candidate_file", None)
        record.pop("candidate_sha256", None)
        if session_dir is not None:
            try:
                atomic_write_json(session_dir / "failure.json", record)
            except (OSError, ValueError) as save_exc:
                print(f"失败记录未保存：{save_exc}", file=sys.stderr)
        print(f"标定失败：{record['error']}", file=sys.stderr)
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="live", help="离线图路径，默认 live")
    parser.add_argument("--detect-only", action="store_true", help="只预览标记识别，不录入坐标或求外参")
    parser.add_argument("--browser", action="store_true", help="浏览器界面；live 正式标定默认启用，纯预览需显式指定")
    parser.add_argument("--terminal", action="store_true", help="兼容原终端坐标录入流程；默认单位 mm")
    parser.add_argument("--preview-port", type=int, default=8765, help="浏览器预览本机端口，默认 8765")
    parser.add_argument("--no-display", action="store_true", help="识别预览不打开窗口，保存 PNG/JSON；live 按 sample-count 采样")
    parser.add_argument("--intrinsics", default="configs/vision/camera_intrinsics.json")
    parser.add_argument("--unit", choices=("mm", "m"), default="mm", help="默认 mm，不询问；m 仅供终端兼容模式")
    parser.add_argument("--rgb-device", default="0", help="明确的设备路径或索引；推荐 /dev/v4l/by-id/...")
    parser.add_argument("--output-root", type=Path, help="新会话父目录；默认 calibration_assets/sessions")
    parser.add_argument("--tcp-description", default=DEFAULT_TCP_DESCRIPTION,
                        help=f"接触点/TCP 定义，默认：{DEFAULT_TCP_DESCRIPTION}；不交互询问")
    parser.add_argument("--tool-version", default="unspecified", help="工具/TCP 版本，缺省明确标为未提供")
    parser.add_argument("--coordinate-source", choices=("contact", "flange"), default="contact",
                        help="仅支持 contact；flange 会拒绝，不能直接用法兰 XYZ")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--marker-length-m", type=float, help="兼容旧参数：黑框实测边长，单位 m")
    size.add_argument("--marker-length-mm", type=float, help="黑框实测边长，单位 mm；正式界面中也可填写")
    parser.add_argument("--comparison-windows", type=int, default=10, help="点击开始后采满即停止的合格窗口数，默认 10")
    parser.add_argument("--expected-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--sample-count", type=int, default=5, help="标定或无窗口预览的 live 采样帧数；离线图为单帧")
    parser.add_argument("--validation-file", type=Path, help="独立点 JSON：schema_version=1, unit, reference, reference_uncertainty_mm, points（id/camera_xyz_m/base_xyz）")
    for name, value in asdict(CalibrationQuality()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value,
                            help=f"质量筛选参数，默认 {value}（非硬件精度保证）")
    for name, value in asdict(PreviewCriteria()).items():
        parser.add_argument("--preview-" + name.replace("_", "-"), dest="preview_" + name,
                            type=int if name == "min_frames" else float, default=value,
                            help=f"静止场景建议采集门槛，默认 {value}；不是精度验收")
    args = vars(parser.parse_args(argv))
    quality = CalibrationQuality(**{name: args.pop(name) for name in asdict(CalibrationQuality())})
    criteria = PreviewCriteria(**{name: args.pop("preview_" + name) for name in asdict(PreviewCriteria())})
    detect_only = args.pop("detect_only")
    no_display = args.pop("no_display")
    browser = args.pop("browser")
    terminal = args.pop("terminal")
    preview_port = args.pop("preview_port")
    comparison_windows = args.pop("comparison_windows")
    length_mm = args.pop("marker_length_mm")
    args["image_source"] = args.pop("image")
    args["intrin_file"] = args.pop("intrinsics")
    if args["marker_length_m"] is None:
        args["marker_length_m"] = length_mm / 1000 if length_mm is not None else .080
    if terminal and (browser or detect_only):
        parser.error("--terminal 与 --browser/--detect-only 不可同时使用")
    if detect_only:
        preview_args = {name: args[name] for name in (
            "image_source", "intrin_file", "rgb_device", "output_root", "marker_length_m", "expected_ids", "sample_count")}
        return 0 if run_marker_preview(**preview_args, no_display=no_display, browser=browser,
                                       preview_port=preview_port, criteria=criteria, quality=quality) else 1
    if no_display:
        parser.error("--no-display 仅用于 --detect-only；正式标定默认使用浏览器")
    if not terminal and args["image_source"] == "live":
        if args["unit"] != "mm" or args["coordinate_source"] != "contact":
            parser.error("正式可视化界面仅接受接触点坐标，单位固定 mm")
        visual_args = {name: args[name] for name in ("intrin_file", "rgb_device", "output_root",
            "marker_length_m", "expected_ids", "tcp_description", "tool_version", "validation_file")}
        return 0 if run_visual_calibration(**visual_args, preview_port=preview_port,
            target_windows=comparison_windows, criteria=criteria, quality=quality) else 1
    if browser:
        parser.error("正式浏览器标定使用 live；离线图可用 --detect-only --browser 预览或 --terminal 标定")
    return 0 if run_touch_calibration(**args, quality=quality) else 1


if __name__ == "__main__":
    sys.exit(main())
