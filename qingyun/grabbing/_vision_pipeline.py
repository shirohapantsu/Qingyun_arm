"""视觉私有管线：RGB-D 配准、OBB 深度测量、可抓性过滤及 RK3588 接入。

从 feature/vision 的 RKNN/OpenNI 接入迁移；几何和配置按本地 P3 契约修正。
本模块仅由 vision.init 延迟导入，不由主流程或 plans 直接依赖。
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from configs.common_interface import VisionInterface
from qingyun import runlog
from qingyun.grabbing.vision import NoTarget, VisionHardError


def _object(value, keys, where, optional=()):
    if not isinstance(value, dict) or set(value) - set(keys) - set(optional) or set(keys) - set(value):
        raise ValueError(f"{where}: 缺少字段或含未知字段")
    return value


def _number(value, where, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: 必须是有限数值")
    if positive and value <= 0 or minimum is not None and value < minimum:
        raise ValueError(f"{where}: 数值越界")
    return float(value)


def _integer(value, where, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: 必须是 >= {minimum} 的整数")
    return value


def _matrix(value, where):
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{where}: 必须是有限 4x4 矩阵")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError(f"{where}: 必须是右手刚体变换（平移单位 m）")
    return T


def _json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera(value, where):
    _object(value, ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'dist'), where)
    for key in ('width', 'height'):
        _integer(value[key], where + '.' + key)
    for key in ('fx', 'fy', 'cx', 'cy'):
        _number(value[key], where + '.' + key, positive=key in ('fx', 'fy'))
    dist = np.asarray(value['dist'], dtype=np.float64)
    if dist.shape != (5,) or not np.isfinite(dist).all():
        raise ValueError(where + ': dist 必须是五个有限系数')
    return np.array([[value['fx'], 0, value['cx']], [0, value['fy'], value['cy']], [0, 0, 1]], dtype=np.float64), dist


def align_depth(depth_raw, intrinsics):
    """原始轴向 Z → RGB 光学系 XYZ 网格；最近 Z 胜出，空洞保留 NaN。

    原始 RGB 不重映射；每个原始深度像素只去畸变一次，D2C 平移用米。
    """
    d, r = intrinsics['depth'], intrinsics['rgb']
    Kd, dd = _camera(d, 'depth')
    Kr, dr = _camera(r, 'rgb')
    T = _matrix(intrinsics['depth_to_rgb'], 'depth_to_rgb')
    v, u = np.indices(depth_raw.shape)
    z = depth_raw.astype(np.float64) * intrinsics['depth_unit_m']
    valid = np.isfinite(z) & (z > 0)
    pixels = np.column_stack((u[valid], v[valid])).astype(np.float64)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), Kd, dd).reshape(-1, 2)
    points = np.column_stack((rays * z[valid, None], z[valid]))
    rgb = points @ T[:3, :3].T + T[:3, 3]
    rgb = rgb[np.isfinite(rgb).all(axis=1) & (rgb[:, 2] > 0)]
    grid = np.full((r['height'], r['width'], 3), np.nan, dtype=np.float64)
    if not len(rgb):
        return grid
    projected, _ = cv2.projectPoints(rgb, np.zeros(3), np.zeros(3), Kr, dr)
    pixels = np.rint(projected.reshape(-1, 2)).astype(np.int64)
    good = (pixels[:, 0] >= 0) & (pixels[:, 0] < r['width']) & (pixels[:, 1] >= 0) & (pixels[:, 1] < r['height'])
    pixels, rgb = pixels[good], rgb[good]
    ids = pixels[:, 1] * r['width'] + pixels[:, 0]
    order = np.lexsort((rgb[:, 2], ids))
    ids, rgb = ids[order], rgb[order]
    _, first = np.unique(ids, return_index=True)
    grid.reshape(-1, 3)[ids[first]] = rgb[first]
    return grid


def _components(mask, points, jump):
    """8 邻接，三维相邻点距离受限；不跨空洞或深度跳变填补物体。"""
    remaining = mask.copy()
    height, width = mask.shape
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        queue = deque([(y, x)])
        component = []
        while queue:
            cy, cx = queue.popleft()
            component.append(cy * width + cx)
            for ny in range(max(0, cy-1), min(height, cy+2)):
                for nx in range(max(0, cx-1), min(width, cx+2)):
                    if remaining[ny, nx] and np.linalg.norm(points[ny, nx] - points[cy, cx]) <= jump:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
        components.append(np.array(component, dtype=np.int64))
    return components


def _rectangle(center, length, width, yaw):
    return cv2.boxPoints((tuple(map(float, center)), (float(length), float(width)), float(yaw))).astype(np.float64)


def _polygon_distance(a, b):
    if cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))[0] > 0:
        return 0.0
    distances = []
    for p, q in ((a, b), (b, a)):
        for start, end in zip(q, np.roll(q, -1, axis=0)):
            edge = end - start
            ratios = np.clip((p - start) @ edge / (edge @ edge), 0, 1)
            distances.extend(np.linalg.norm(p - (start + ratios[:, None] * edge), axis=1))
    return float(min(distances))


def measure_geometry(detections, xyz_rgb, thresholds, extrinsic, geometry):
    base = xyz_rgb @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    usable = np.isfinite(base).all(axis=2) & (base[:, :, 2] > thresholds.table_z_m + thresholds.table_flatness_m)
    height, width = usable.shape
    candidates = []
    for index, det in enumerate(detections):
        c = dict(source_detection_id=index, rejection_reason=None)
        candidates.append(c)
        try:
            _object(det, ('class', 'conf', 'corners_px'), 'detection')
            if not isinstance(det['class'], str) or not det['class']:
                raise ValueError('class 必须为字符串')
            conf = _number(det['conf'], 'conf', minimum=0)
            if conf > 1:
                raise ValueError('conf > 1')
            corners = np.asarray(det['corners_px'], dtype=np.float64)
            if corners.shape != (4, 2) or not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)) or abs(cv2.contourArea(corners.astype(np.float32))) < 1:
                raise ValueError('非法 OBB 四角')
            c.update({'class': det['class'], 'conf': conf, 'corners_px': corners.tolist()})
        except (ValueError, TypeError):
            c['rejection_reason'] = 'invalid_geometry'
            continue
        full = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(full, np.rint(corners).astype(np.int32), 1)
        inner = np.zeros_like(full)
        inner_corners = corners.mean(axis=0) + (corners - corners.mean(axis=0)) * geometry['inner_scale']
        cv2.fillConvexPoly(inner, np.rint(inner_corners).astype(np.int32), 1)
        components = _components(full.astype(bool) & usable, base, geometry['depth_neighbor_jump_m'])
        supported = [ids for ids in components if len(ids) >= geometry['min_points'] and np.count_nonzero(inner.ravel()[ids]) >= geometry['min_points']]
        if not supported:
            c['rejection_reason'] = 'no_depth'
            continue
        if len(supported) != 1:
            c['rejection_reason'] = 'ambiguous_geometry'
            continue
        ids = supported[0]
        pts = base.reshape(-1, 3)[ids]
        (cx, cy), (length, short), angle = cv2.minAreaRect(pts[:, :2].astype(np.float32))
        if short > length:
            length, short, angle = short, length, angle + 90
        if short <= 0 or not np.isfinite([cx, cy, length, short, angle]).all():
            c['rejection_reason'] = 'invalid_geometry'
            continue
        z_top = float(np.quantile(pts[:, 2], 0.95))
        center = np.array([cx, cy, (thresholds.table_z_m + z_top)/2], dtype=np.float64)
        yaw = (float(angle) + 90) % 180 - 90
        footprint = _rectangle(center[:2], length + 2*geometry['footprint_padding_m'], short + 2*geometry['footprint_padding_m'], yaw)
        c.update(position=center, yaw_deg=yaw, length_m=float(length), width_m=float(short), z_top=z_top,
                 footprint=footprint, component_ids=ids, point_count=len(ids))
    return candidates, base


def filter_and_rank(candidates, base, thresholds, model):
    measured = [c for c in candidates if c['rejection_reason'] is None]
    unverifiable = len(measured) != len(candidates)
    for c in measured:
        distances = [_polygon_distance(c['footprint'], other['footprint']) for other in measured if other is not c]
        c['clearance'] = min(distances, default=float('inf'))
        c['ripe'] = c['class'] in model['class_ripe']
        reason = None
        if c['class'] not in model['class_ripe'] + model['class_unripe']:
            reason = 'unmapped_class'
        elif c['conf'] < model['conf_threshold']:
            reason = 'low_confidence'
        elif not np.all((c['position'] >= thresholds.target_bounds_m[0]) & (c['position'] <= thresholds.target_bounds_m[1])):
            reason = 'out_of_bounds'
        elif c['length_m'] > thresholds.object_envelope_m[0] or c['width_m'] > thresholds.object_envelope_m[1]:
            reason = 'oversize'
        elif unverifiable:
            reason = 'neighbor_unverifiable'
        else:
            points = base.reshape(-1, 3)
            external = np.ones(len(points), dtype=bool)
            external[c['component_ids']] = False
            angle = math.radians(c['yaw_deg'])
            axis = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
            local = (points[:, :2] - c['position'][:2]) @ axis.T
            overhead = (np.abs(local[:, 0]) <= thresholds.object_envelope_m[0]/2) & (np.abs(local[:, 1]) <= thresholds.object_envelope_m[1]/2)
            overhead &= (points[:, 2] > c['z_top']) & (points[:, 2] <= c['z_top'] + thresholds.approach_height_m + thresholds.object_envelope_m[2])
            if np.any(overhead & external):
                reason = 'overhead_blocked'
            elif c['clearance'] < thresholds.clearance_m:
                reason = 'too_close'
        c['rejection_reason'] = reason
    valid = [c for c in measured if c['rejection_reason'] is None]
    center = thresholds.target_bounds_m.mean(axis=0)[:2]
    return sorted(valid, key=lambda c: (-c['clearance'], float(np.linalg.norm(c['position'][:2]-center)), *c['position']))


class Pipeline:
    def __init__(self, thresholds, config, config_path):
        self.thresholds, self.config = thresholds, config
        self.directory = config_path.parent
        self.index = 0
        self.cap = self.ds = self.device = self.rknn = self.openni = None
        self.frames = []

    def path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('资源路径必须为非空字符串')
        return (self.directory / value).resolve()

    def validate(self):
        cfg = _object(self.config, ('camera', 'model', 'calibration', 'geometry', 'source'), 'vision', ('_comment',))
        camera = _object(cfg['camera'], ('type', 'depth_mode', 'rgb_device', 'rgb_size', 'max_pair_delta_s', 'acquire_timeout_s'), 'camera', ('openni_path',))
        model = _object(cfg['model'], ('backend', 'path', 'detection_floor', 'conf_threshold', 'class_ripe', 'class_unripe'), 'model', ('input_size', 'crop_size', 'iou_threshold', 'class_names'))
        source = _object(cfg['source'], ('mode', 'replay_dir'), 'source')
        calibration = _object(cfg['calibration'], ('intrinsics_path', 'extrinsic_base_camera_path'), 'calibration')
        geom = _object(cfg['geometry'], ('min_points', 'inner_scale', 'depth_neighbor_jump_m', 'footprint_padding_m'), 'geometry')
        if source['mode'] not in ('live', 'replay') or model['backend'] not in ('fixture', 'yolo', 'rknn'):
            raise ValueError('未知 source.mode/model.backend')
        if source['mode'] == 'live' and model['backend'] == 'fixture':
            raise ValueError('live 禁止 fixture')
        floor = _number(model['detection_floor'], 'detection_floor', minimum=0)
        conf = _number(model['conf_threshold'], 'conf_threshold', minimum=0)
        if not floor <= conf <= 1:
            raise ValueError('要求 0 <= detection_floor <= conf_threshold <= 1')
        for key in ('class_ripe', 'class_unripe'):
            names = model[key]
            if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
                raise ValueError('类别映射必须非空且不重复')
        if set(model['class_ripe']) & set(model['class_unripe']):
            raise ValueError('成熟/未成熟类别必须互斥')
        _integer(geom['min_points'], 'min_points', 3)
        if _number(geom['inner_scale'], 'inner_scale', positive=True) > 1:
            raise ValueError('inner_scale > 1')
        _number(geom['depth_neighbor_jump_m'], 'depth_neighbor_jump_m', positive=True)
        _number(geom['footprint_padding_m'], 'footprint_padding_m', minimum=0)
        for key in ('max_pair_delta_s', 'acquire_timeout_s'):
            _number(camera[key], key, positive=True)
        for key in ('input_size', 'crop_size'):
            if key in model:
                _integer(model[key], key)
        if 'iou_threshold' in model and _number(model['iou_threshold'], 'iou_threshold', minimum=0) > 1:
            raise ValueError('iou_threshold > 1')
        t = self.thresholds
        bounds = np.asarray(t.target_bounds_m)
        envelope = np.asarray(t.object_envelope_m)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]) or envelope.shape != (3,) or not np.isfinite(envelope).all() or np.any(envelope <= 0):
            raise ValueError('非法 VisionThresholds 数组')
        for key in ('clearance_m', 'approach_height_m', 'table_flatness_m'):
            _number(getattr(t, key), key, minimum=0)
        _number(t.table_z_m, 'table_z_m')
        self.intrin_path = self.path(calibration['intrinsics_path'])
        self.extrin_path = self.path(calibration['extrinsic_base_camera_path'])
        intrin = _object(_json(self.intrin_path), ('schema_version', 'rgb', 'depth', 'depth_to_rgb', 'depth_unit_m'), 'intrinsics')
        if type(intrin['schema_version']) is not int or intrin['schema_version'] != 1:
            raise ValueError('intrinsics schema_version != 1')
        _camera(intrin['rgb'], 'rgb'); _camera(intrin['depth'], 'depth')
        _matrix(intrin['depth_to_rgb'], 'depth_to_rgb')
        _number(intrin['depth_unit_m'], 'depth_unit_m', positive=True)
        extrin = _object(_json(self.extrin_path), ('schema_version', 'from_frame', 'to_frame', 'matrix'), 'extrinsic')
        if type(extrin['schema_version']) is not int or extrin['schema_version'] != 1 or extrin['from_frame'] != 'rgb_optical' or extrin['to_frame'] != 'base_link':
            raise ValueError('外参坐标系/schema_version 不匹配')
        self.intrinsics, self.extrinsic = intrin, _matrix(extrin['matrix'], 'T_base_camera')
        if camera['type'] != 'orbbec_astra_pro':
            raise ValueError('camera.type 必须为 orbbec_astra_pro')
        if source['mode'] == 'live':
            if camera['depth_mode'] != '640x480@30' or (intrin['depth']['width'], intrin['depth']['height']) != (640, 480):
                raise ValueError('live 深度流与内参必须为 640x480，禁止自动缩放内参')
            if not isinstance(camera['rgb_size'], list) or len(camera['rgb_size']) != 2 or any(type(n) is not int or n <= 0 for n in camera['rgb_size']) or camera['rgb_size'] != [intrin['rgb']['width'], intrin['rgb']['height']]:
                raise ValueError('rgb_size 与内参不匹配')
            if not (type(camera['rgb_device']) is int and camera['rgb_device'] >= 0 or isinstance(camera['rgb_device'], str) and camera['rgb_device']):
                raise ValueError('必须明确指定 rgb_device')
        if model['backend'] == 'fixture' and model['path'] is not None:
            raise ValueError('fixture model.path 必须为 null')

    def initialize(self):
        try:
            self.validate()
            self.load_model()
            if self.config['source']['mode'] == 'replay':
                self.replay_dir = self.path(self.config['source']['replay_dir'])
                manifest = _object(_json(self.replay_dir/'manifest.json'), ('schema_version', 'frames'), 'manifest', ('_comment',))
                if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or not isinstance(manifest['frames'], list) or not manifest['frames']:
                    raise ValueError('replay manifest 必须含至少一帧')
                ids, previous_rgb, previous_depth = set(), -math.inf, -math.inf
                for info in manifest['frames']:
                    required = ('id', 'color', 'depth', 'rgb_time_s', 'depth_time_s')
                    _object(info, required, 'frame', ('detections',))
                    if not isinstance(info['id'], str) or not info['id'] or info['id'] in ids:
                        raise ValueError('frame id 必须非空且唯一')
                    rgb = _number(info['rgb_time_s'], 'rgb_time_s', minimum=0)
                    dep = _number(info['depth_time_s'], 'depth_time_s', minimum=0)
                    if rgb < previous_rgb or dep < previous_depth:
                        raise ValueError('帧时间必须非递减')
                    ids.add(info['id']); previous_rgb, previous_depth = rgb, dep
                    for key in ('color', 'depth') + (('detections',) if self.config['model']['backend'] == 'fixture' else ()):
                        self.replay_path(info[key])
                self.frames = manifest['frames']
                self.read_replay(self.frames[0])  # 验证，不推进 index
            else:
                self.open_camera()
                self.read_live()  # 验证，不复用此帧作为扫描结果
            runlog.event('vision_init', backend=self.config['model']['backend'], source=self.config['source']['mode'],
                         camera=self.config['camera'], weights_sha256=self.weights_hash,
                         intrinsics_sha256=_hash(self.intrin_path), extrinsic_sha256=_hash(self.extrin_path))
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def load_model(self):
        model = self.config['model']
        self.weights_hash = None
        if model['backend'] == 'fixture':
            return
        path = self.path(model['path'])
        if not path.is_file():
            raise ValueError(f'模型文件不存在（需自行部署）: {path}')
        self.weights_hash = _hash(path)
        # 原分支把 RKNN 标为 yolo；保留该配置兼容，同时接受显式 rknn。
        if model['backend'] == 'rknn' or path.suffix.lower() == '.rknn':
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            if self.rknn.load_rknn(str(path)) != 0 or self.rknn.init_runtime() != 0:
                raise ValueError('RKNN 初始化失败')
            self.class_names = model.get('class_names', ['RIPE', 'UNRIPE'])
            if not isinstance(self.class_names, list) or not self.class_names or any(not isinstance(n, str) or not n for n in self.class_names):
                raise ValueError('class_names 必须为非空字符串列表')
            names = self.class_names
        else:
            from ultralytics import YOLO
            self.yolo = YOLO(str(path))
            if self.yolo.task != 'obb':
                raise ValueError('模型必须为 OBB')
            names = list(self.yolo.names.values())
        if not set(model['class_ripe'] + model['class_unripe']) <= set(names):
            raise ValueError('模型类别与成熟度映射不匹配')

    def replay_path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('非法回放资源路径')
        path = (self.replay_dir/value).resolve()
        if not path.is_relative_to(self.replay_dir) or not path.is_file():
            raise ValueError(f'回放资源缺失或越出目录: {value}')
        return path

    def validate_frame(self, color, depth, rgb_time, depth_time):
        r, d = self.intrinsics['rgb'], self.intrinsics['depth']
        if color.shape != (r['height'], r['width'], 3) or color.dtype != np.uint8:
            raise ValueError('RGB 形状/类型与内参不匹配')
        if depth.shape != (d['height'], d['width']) or depth.dtype.kind not in 'uif' or not np.isfinite(depth).all() or np.any(depth < 0) or not np.any(depth > 0):
            raise ValueError('整帧深度不可用或与内参不匹配')
        if abs(rgb_time-depth_time) > self.config['camera']['max_pair_delta_s']:
            raise ValueError('RGB-D 配对超时')

    def read_replay(self, info):
        color = np.load(self.replay_path(info['color']), allow_pickle=False)
        depth = np.load(self.replay_path(info['depth']), allow_pickle=False)
        self.validate_frame(color, depth, info['rgb_time_s'], info['depth_time_s'])
        dets = _json(self.replay_path(info['detections'])) if self.config['model']['backend'] == 'fixture' else None
        if dets is not None and not isinstance(dets, list):
            raise ValueError('detections 必须为列表')
        return color, depth, dets, info

    def open_camera(self):
        from openni import openni2
        self.openni = openni2
        camera = self.config['camera']
        sdk = camera.get('openni_path')
        sdk_path = self.path(sdk) if sdk else Path(__file__).resolve().parents[2]/'libs/OpenNI2_arm64'
        openni2.initialize(str(sdk_path))
        self.device = openni2.Device.open_any()
        self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_OFF)
        self.ds = self.device.create_depth_stream()
        mode = openni2.VideoMode()
        mode.resolutionX, mode.resolutionY, mode.fps = 640, 480, 30
        mode.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM
        if self.intrinsics['depth_unit_m'] != 0.001:
            raise ValueError('OpenNI DEPTH_1_MM 要求 depth_unit_m=0.001')
        self.ds.set_video_mode(mode)
        self.ds.set_mirroring_enabled(False)
        self.ds.start()
        self.cap = cv2.VideoCapture(camera['rgb_device'], cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise ValueError('RGB 相机打开失败')
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera['rgb_size'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera['rgb_size'][1])

    def read_live(self):
        start = time.monotonic()
        ret, color = self.cap.read()
        rgb_time = time.monotonic()
        if not ret:
            raise ValueError('RGB 帧读取失败')
        timeout = self.config['camera']['acquire_timeout_s'] - (rgb_time-start)
        if timeout <= 0:
            raise ValueError('RGB 采集超时')
        # openni==2.3.0 的 Python timeout 单位是秒，内部再转换成毫秒。
        # 返回 None 表示超时；不能接着 read_frame，否则可能无界阻塞。
        if self.openni.wait_for_any_stream([self.ds], timeout=timeout) is None:
            raise ValueError('深度帧等待超时')
        frame = self.ds.read_frame()
        depth_time = time.monotonic()
        depth = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640).copy()
        if depth_time-start > self.config['camera']['acquire_timeout_s']:
            raise ValueError('RGB-D 采集超时')
        self.validate_frame(color, depth, rgb_time, depth_time)
        info = dict(id=f'live-{self.index}', rgb_time_s=rgb_time, depth_time_s=depth_time)
        self.index += 1
        return color, depth, None, info

    def infer(self, color, raw_dets):
        model = self.config['model']
        if model['backend'] == 'fixture':
            return raw_dets
        if self.rknn is None:
            results = self.yolo.predict(color, conf=model['detection_floor'], verbose=False)
            if len(results) != 1 or results[0].obb is None:
                raise ValueError('YOLO 未返回 OBB 输出')
            obb = results[0].obb
            return [{'class': self.yolo.names[int(cls)], 'conf': float(conf), 'corners_px': corners.tolist()}
                    for cls, conf, corners in zip(obb.cls.cpu().numpy(), obb.conf.cpu().numpy(), obb.xyxyxyxy.cpu().numpy())]
        size = model.get('input_size', 640)
        crop = model.get('crop_size', 640)
        height, width = color.shape[:2]
        if crop > min(height, width):
            raise ValueError('crop_size 大于 RGB 尺寸')
        sx, sy = (width-crop)//2, (height-crop)//2
        image = cv2.cvtColor(color[sy:sy+crop, sx:sx+crop], cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size))[None]
        outputs = self.rknn.inference(inputs=[image])
        if not outputs or np.asarray(outputs[0]).ndim != 3:
            raise ValueError('非法 RKNN 输出')
        predictions = np.asarray(outputs[0])[0].T
        if predictions.shape[1] != 5+len(self.class_names) or not np.isfinite(predictions).all():
            raise ValueError('RKNN 输出须为 [1, 4+classes+angle, N] OBB')
        if not len(predictions):
            return []
        scores_by_class = predictions[:, 4:-1]
        if np.any((scores_by_class < 0) | (scores_by_class > 1)):
            scores_by_class = 1/(1+np.exp(-np.clip(scores_by_class, -50, 50)))
        classes = scores_by_class.argmax(axis=1)
        scores = scores_by_class.max(axis=1)
        # 在低 detection_floor 上保留邻物，资格置信度由几何管线后置判断。
        eligible = np.flatnonzero(scores >= model['detection_floor'])
        rectangles, kept_scores, kept_classes = [], [], []
        for i in eligible:
            cx, cy, w, h = predictions[i, :4]*crop/size
            if w <= 0 or h <= 0:
                raise ValueError('RKNN 输出非正框尺寸')
            rectangles.append(((float(cx+sx), float(cy+sy)), (float(w), float(h)), float(math.degrees(predictions[i, -1]))))
            kept_scores.append(float(scores[i])); kept_classes.append(int(classes[i]))
        if not rectangles:
            return []
        indices = cv2.dnn.NMSBoxesRotated(rectangles, kept_scores, model['detection_floor'], model.get('iou_threshold', 0.45))
        return [{'class': self.class_names[kept_classes[i]], 'conf': kept_scores[i], 'corners_px': cv2.boxPoints(rectangles[i]).tolist()}
                for i in np.asarray(indices).reshape(-1)]

    def scan(self, ignore, audit):
        audit['backend'] = self.config['model']['backend']
        if self.config['source']['mode'] == 'replay':
            if self.index >= len(self.frames):
                raise NoTarget('replay_exhausted')
            result = self.read_replay(self.frames[self.index])
            self.index += 1
        else:
            result = self.read_live()
        color, depth, raw_dets, info = result
        self.last_color = color  # 仅供诊断绘图；下一次扫描仍重新采帧，不复用目标。
        audit.update(frame_id=info['id'], rgb_time_s=info['rgb_time_s'], depth_time_s=info['depth_time_s'])
        detections = self.infer(color, raw_dets)
        audit['detection_count'] = len(detections)
        if not detections:
            raise NoTarget('no_detection')
        xyz = align_depth(depth, self.intrinsics)
        if not np.isfinite(xyz).all(axis=2).any():
            raise VisionHardError('D2C 无有效配准点')
        candidates, base = measure_geometry(detections, xyz, self.thresholds, self.extrinsic, self.config['geometry'])
        ranked = filter_and_rank(candidates, base, self.thresholds, self.config['model'])
        for c in candidates:
            entry = {key: value for key, value in c.items() if key not in ('component_ids', 'clearance')}
            if 'clearance' in c:
                entry['isolated'] = math.isinf(c['clearance'])
                entry['clearance_m'] = None if entry['isolated'] else c['clearance']
            audit['candidates'].append(entry)
        audit['valid_count'] = len(ranked)
        if not ranked:
            raise NoTarget('all_filtered')
        if ignore >= len(ranked):
            raise NoTarget('skip_exhausted')
        c = ranked[ignore]
        audit['selected_index'] = ignore
        audit['selected_detection_id'] = c['source_detection_id']
        return VisionInterface(c['position'].astype(np.float64).copy(), c['yaw_deg'], c['length_m'], c['width_m'], bool(c['ripe']), len(ranked))

    def close(self):
        for name, method in (('cap', 'release'), ('ds', 'stop'), ('device', 'close'), ('rknn', 'release'), ('openni', 'unload')):
            resource = getattr(self, name)
            setattr(self, name, None)
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception:
                    pass
