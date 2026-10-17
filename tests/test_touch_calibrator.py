"""标定专项回归；只写 tmp_path，不连接设备或改写正式配置。"""

from dataclasses import replace
import json
import shutil
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import zlib

import cv2
import numpy as np
import pytest

from qingyun.calibration import touch_calibrator as cal
from qingyun.calibration import generate_markers as markers


CAMERA = np.array([[-.15, -.1, .6], [.15, -.1, .62], [-.15, .1, .65],
                   [.15, .1, .68], [-.04, .03, .55], [.06, .04, .58]])
ROTATION = cv2.Rodrigues(np.array([.2, -.3, .4]))[0]
TRANSLATION = np.array([.1, -.2, .3])
BASE = CAMERA @ ROTATION.T + TRANSLATION


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    intrinsics = {"schema_version": 1, "rgb": {"width": 640, "height": 480,
                  "fx": 800., "fy": 790., "cx": 320., "cy": 240., "dist": [0] * 5}}
    intrin_path = tmp_path / "intrinsics.json"
    intrin_path.write_text(json.dumps(intrinsics))
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.zeros((480, 640, 3), dtype=np.uint8))
    # 流程测试隔离检测，使用已知相机点；PnP 真值在下方单独验证。
    detections = {i + 1: {"P_cam": p, "corners": np.array([[1, 1], [2, 1], [2, 2], [1, 2]]),
                          "reprojection_error_px": 0.1} for i, p in enumerate(CAMERA)}
    monkeypatch.setattr(cal, "detect_markers_pnp", lambda image, *a, **kw: (detections, image.copy()))

    def run(base=BASE, inputs=None, **kwargs):
        responses = iter(inputs if inputs is not None else [" ".join(format(v * 1000, '.17g') for v in p) for p in base])
        monkeypatch.setattr("builtins.input", lambda prompt: next(responses))
        options = dict(image_source=str(image_path), intrin_file=str(intrin_path), unit="mm",
                       tool_version="tool-test-v1", output_root=tmp_path / "sessions")
        options.update(kwargs)
        return cal.run_touch_calibration(**options)

    return SimpleNamespace(run=run, root=tmp_path / "sessions", intrinsics=intrin_path,
                           image=image_path, detections=detections)


def read_record(workflow, name="session.json"):
    paths = list(workflow.root.glob(f"*/{name}"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text()), paths[0].parent


def independent_data(offset_mm=0):
    camera = CAMERA[:4] + [.031, .027, .043]
    base = camera @ ROTATION.T + TRANSLATION + [offset_mm / 1000, 0, 0]
    return {"schema_version": 1, "unit": "mm", "reference": "独立量具合成真值",
            "reference_uncertainty_mm": .2,
            "points": [{"id": i, "camera_xyz_m": c.tolist(), "base_xyz": (b * 1000).tolist()}
                       for i, (c, b) in enumerate(zip(camera, base))]}


@pytest.mark.parametrize("unit, expected", [("mm", [.001, .002, .003]), ("m", [1, 2, 3])])
def test_small_coordinates_have_explicit_unit(unit, expected):
    np.testing.assert_allclose(cal.parse_user_coord_input("1, 2, 3", unit), expected)


@pytest.mark.parametrize("text, unit", [("nan 1 2", "mm"), ("inf 1 2", "m"),
                                      ("1 2 -inf", "mm"), ("1 2", "m"),
                                      ("1 two 3", "mm"), ("1 2 3", "auto")])
def test_coordinate_input_rejects_invalid_values(text, unit):
    with pytest.raises(ValueError):
        cal.parse_user_coord_input(text, unit)


def test_coplanar_noncollinear_points_recover_transform():
    camera = np.array([[0, 0, .6], [.2, 0, .6], [0, .1, .6], [.2, .1, .6]])
    R, t = cal.solve_rigid_transform_3d(camera, camera @ ROTATION.T + TRANSLATION)
    np.testing.assert_allclose(R, ROTATION, atol=1e-12)
    np.testing.assert_allclose(t.ravel(), TRANSLATION, atol=1e-12)


@pytest.mark.parametrize("points", [np.zeros((4, 2)), np.ones((2, 3)),
    np.array([[0, 0, .6], [.1, 0, .6], [.2, 0, .6], [.3, 0, .6]]),
    np.array([[0, 0, .6], [.1, 1e-6, .6], [.2, 0, .6], [.3, 1e-6, .6]]),
    np.array([[0, 0, .6], [.1, 0, .6], [0, .1, .6], [0, .1, .6]]),
    np.array([[0, 0, .6], [.001, 0, .6], [0, .001, .6], [.001, .001, .6]]),
    np.array([[0, 0, .6], [.1, 0, .6], [0, .1, .6], [np.nan, 0, .6]])])
def test_invalid_point_sets_rejected(points):
    with pytest.raises(ValueError):
        cal.solve_rigid_transform_3d(points, points)


def test_point_shapes_must_match():
    with pytest.raises(ValueError):
        cal.solve_rigid_transform_3d(CAMERA, BASE[:4])


@pytest.mark.parametrize("name, value", [("max_fit_error_mm", 5), ("max_reprojection_px", float('nan')),
                                       ("min_second_ratio", 2), ("min_span_m", 0), ("min_span_m", True)])
def test_bad_quality_settings_rejected(name, value):
    with pytest.raises(ValueError):
        replace(cal.CalibrationQuality(), **{name: value}).validate()


def test_pnp_known_tilted_pose_with_distortion():
    K = np.array([[900., 0, 320], [0, 890., 240], [0, 0, 1]])
    dist = np.array([.1, -.03, .01, -.005, 0])
    obj = np.array([[-.02, .02, 0], [.02, .02, 0], [.02, -.02, 0], [-.02, -.02, 0]])
    rv, tv = np.array([2.8, .1, -.15]), np.array([.03, -.02, .7])
    corners = cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(4, 2)
    pose = cal._marker_pose(corners, K, dist, .04, cal.CalibrationQuality())
    np.testing.assert_allclose(pose["P_cam"], tv, atol=1e-8)
    assert pose["reprojection_error_px"] < 1e-6


@pytest.mark.parametrize("corners", [np.zeros((4, 2)), np.full((4, 2), np.nan),
    np.array([[280., 200], [360, 180], [400, 300], [275, 240]])])
def test_pnp_rejects_small_nonfinite_or_bad_reprojection(corners):
    K = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1]])
    with pytest.raises(ValueError):
        cal._marker_pose(corners, K, np.zeros(5), .04, cal.CalibrationQuality())


def test_pnp_rejects_negative_depth(monkeypatch):
    rv, tv = np.zeros((3, 1)), np.array([[0.], [0.], [-1.]])
    monkeypatch.setattr(cv2, "solvePnPGeneric", lambda *a, **kw: (1, [rv], [tv]))
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **kw: (True, rv, tv))
    with pytest.raises(ValueError, match="正深度"):
        cal._marker_pose(np.array([[100., 100], [200, 100], [200, 200], [100, 200]]),
                         np.eye(3), np.zeros(5), .04, cal.CalibrationQuality())


@pytest.mark.parametrize("ids, expected, error", [([1, 1], [1, 2], "重复"), ([9], [1, 2], "非期望")])
def test_detector_rejects_duplicate_or_unexpected_ids(monkeypatch, ids, expected, error):
    detector = SimpleNamespace(detectMarkers=lambda image: ([np.zeros((1, 4, 2))] * len(ids), np.array(ids).reshape(-1, 1), []))
    monkeypatch.setattr(cv2.aruco, "ArucoDetector", lambda *args: detector)
    with pytest.raises(ValueError, match=error):
        cal.detect_markers_pnp(np.zeros((480, 640, 3), np.uint8), np.eye(3), np.zeros(5), expected_ids=expected)


def test_successful_workflow_schema_and_reproducible_record(workflow):
    assert workflow.run()
    record, directory = read_record(workflow)
    candidate = json.loads((directory / "T_base_camera.candidate.json").read_text())
    assert set(candidate) == {"schema_version", "from_frame", "to_frame", "matrix"}
    assert candidate["schema_version"] == 1
    assert candidate["from_frame"] == "rgb_optical" and candidate["to_frame"] == "base_link"
    np.testing.assert_allclose(np.array(candidate["matrix"])[:3, :3], ROTATION, atol=1e-12)
    np.testing.assert_allclose(np.array(candidate["matrix"])[:3, 3], TRANSLATION, atol=1e-12)
    assert record["independent_validation"]["status"] == "not_performed"
    assert record["tcp_description"] == "闭合夹爪两指尖中心"
    assert record["depth_validation"] == "not_performed_rgb_only"
    assert record["intrinsics_sha256"] == cal._sha256(workflow.intrinsics)
    assert record["candidate_sha256"] == cal._sha256(directory / "T_base_camera.candidate.json")
    assert (directory / "000_raw.png").exists() and (directory / "000_annotated.png").exists()
    pairs = record["correspondences"]
    R, t = cal.solve_rigid_transform_3d([p["camera_xyz_m"] for p in pairs], [p["base_xyz_m"] for p in pairs])
    np.testing.assert_allclose(R, ROTATION, atol=1e-12)
    np.testing.assert_allclose(t.ravel(), TRANSLATION, atol=1e-12)


def test_bad_fit_does_not_produce_candidate(workflow):
    base = BASE.copy()
    base[0] += [.05, 0, 0]
    assert not workflow.run(base=base)
    record, directory = read_record(workflow, "failure.json")
    assert record["fit"]["max_mm"] > 4
    assert not list(directory.glob("*candidate*"))
    assert not (directory / "session.json").exists()


def test_shared_tcp_bias_is_not_discovered_by_fit(workflow):
    assert workflow.run(base=BASE + [.03, 0, 0])
    record, directory = read_record(workflow)
    assert record["fit"]["max_mm"] < 1e-8
    assert record["independent_validation"]["status"] == "not_performed"
    T = np.array(json.loads((directory / "T_base_camera.candidate.json").read_text())["matrix"])
    assert np.linalg.norm(T[:3, 3] - TRANSLATION) == pytest.approx(.03)


@pytest.mark.parametrize("inputs", [["q"], ["s"] * 6, []])
def test_cancel_skip_or_input_eof_is_failure(workflow, inputs):
    assert not workflow.run(inputs=inputs)
    record, directory = read_record(workflow, "failure.json")
    assert record["status"] == "failed"
    assert not list(directory.glob("*candidate*"))


@pytest.mark.parametrize("shape", [(240, 320, 3), None])
def test_mismatched_or_undecodable_image_rejected(workflow, shape):
    if shape is None:
        workflow.image.write_bytes(b"not an image")
    else:
        assert cv2.imwrite(str(workflow.image), np.zeros(shape, np.uint8))
    assert not workflow.run()
    record, directory = read_record(workflow, "failure.json")
    assert "解码失败" in record["error"]
    assert not list(directory.glob("*candidate*"))


@pytest.mark.parametrize("change", [{"fx": 0}, {"height": True}, {"dist": [0, 0]}, {"cx": float('inf')}])
def test_invalid_rgb_intrinsics_rejected(workflow, change):
    data = json.loads(workflow.intrinsics.read_text())
    data["rgb"].update(change)
    workflow.intrinsics.write_text(json.dumps(data))
    assert not workflow.run()
    assert not workflow.root.exists()


@pytest.mark.parametrize("offset, status", [(0, "passed"), (20, "failed")])
def test_independent_validation_pass_and_failure(workflow, offset, status):
    path = workflow.root.parent / "validation.json"
    path.write_text(json.dumps(independent_data(offset)))
    assert workflow.run(validation_file=path) is (status == "passed")
    record, directory = read_record(workflow, "session.json" if status == "passed" else "failure.json")
    assert record["independent_validation"]["status"] == status
    assert bool(list(directory.glob("*candidate*"))) is (status == "passed")


def test_validation_rejects_reused_fit_locations():
    data = independent_data()
    data["points"][0]["camera_xyz_m"] = CAMERA[0].tolist()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = ROTATION, TRANSLATION
    with pytest.raises(ValueError, match="复用了拟合位置"):
        cal.validate_independent_points(T, data, CAMERA, BASE, cal.CalibrationQuality())


def test_flange_input_and_protected_output_refused(workflow):
    assert not workflow.run(coordinate_source="flange")
    assert not workflow.root.exists()
    protected = cal.PROJECT_ROOT / "configs" / "vision" / "T_base_camera.json"
    before = protected.read_bytes()
    assert not workflow.run(output_root=protected)
    assert protected.read_bytes() == before


def test_json_atomic_creation_no_overwrite_or_nan(tmp_path):
    path = tmp_path / "result.json"
    cal.atomic_write_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        cal.atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(ValueError):
        cal.atomic_write_json(tmp_path / "nan.json", {"value": float('nan')})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


def test_json_publish_failure_cleans_temporary_file(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("模拟写入故障")
    monkeypatch.setattr(cal.os, "link", fail)
    with pytest.raises(OSError):
        cal.atomic_write_json(tmp_path / "candidate.json", {"value": 1})
    assert list(tmp_path.iterdir()) == []


def test_session_record_failure_removes_candidate(workflow, monkeypatch):
    original = cal.atomic_write_json
    def write(path, data):
        if Path(path).name == "session.json":
            raise OSError("模拟会话记录写入失败")
        original(path, data)
    monkeypatch.setattr(cal, "atomic_write_json", write)
    assert not workflow.run()
    record, directory = read_record(workflow, "failure.json")
    assert "记录写入失败" in record["error"]
    assert not list(directory.glob("*candidate*"))


def test_live_capture_failure_always_releases_camera(monkeypatch):
    released = []
    cap = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True,
                          read=lambda: (False, None), release=lambda: released.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
    with pytest.raises(ValueError, match="预读失败"):
        cal._acquire_images("live", "0", {"rgb": {"width": 640, "height": 480}}, 5)
    assert released == [True]


def test_multiframe_repeatability_rejects_unstable_marker(workflow, monkeypatch):
    image = np.zeros((480, 640, 3), np.uint8)
    monkeypatch.setattr(cal, "_acquire_images", lambda *a: [image] * 2)
    calls = []
    def detect(image, *a, **kw):
        shift = .02 if calls else 0
        calls.append(True)
        return {mid: {"P_cam": item["P_cam"] + [shift, 0, 0]} for mid, item in workflow.detections.items()}, image
    monkeypatch.setattr(cal, "detect_markers_pnp", detect)
    assert not workflow.run(image_source="live")
    record, directory = read_record(workflow, "failure.json")
    assert "波动超限" in record["error"]
    assert record["repeatability"]["1"]["sample_count"] == 2
    assert not list(directory.glob("*candidate*"))


def test_cli_failure_has_nonzero_exit_and_never_opens_camera(workflow):
    result = subprocess.run([sys.executable, str(Path(cal.__file__)), "--image", str(workflow.image.parent / 'missing.png'),
                             "--unit", "mm", "--tcp-description", "test TCP", "--intrinsics", str(workflow.intrinsics),
                             "--output-root", str(workflow.root)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "标定失败" in result.stderr


def test_marker_pixels_unchanged_and_png_dpi(tmp_path):
    card = markers.generate_printable_marker_card(1, 400, 600, 600)
    np.testing.assert_array_equal(card[60:460, 100:500], markers.generate_aruco_marker(1, 400))
    path = tmp_path / "marker.png"
    markers.save_png_with_dpi(path, card)
    data = path.read_bytes()
    offset = 8
    density = None
    while offset < len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        tag = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", data[offset + 8 + size:offset + 12 + size])[0]
        assert crc == zlib.crc32(tag + payload) & 0xffffffff
        if tag == b"pHYs":
            density = struct.unpack(">IIB", payload)
        offset += size + 12
    assert density[2] == 1
    assert density[0] * .0254 == pytest.approx(300, abs=.02)
    assert density[0] == density[1]
    np.testing.assert_array_equal(cv2.imread(str(path)), card)
    with pytest.raises(FileExistsError):
        markers.save_png_with_dpi(path, card)


def test_generator_protects_existing_assets_and_outputs_six_detectable_markers(tmp_path):
    protected = markers.PROJECT_ROOT / "calibration_assets" / "live_test.jpg"
    before = protected.read_bytes()
    with pytest.raises(ValueError):
        markers.create_a4_calibration_sheet(protected)
    assert protected.read_bytes() == before
    assert markers.main(["--output-root", str(tmp_path)]) == 0
    directories = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(directories) == 1
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    for page in range(1, 4):
        sheet = cv2.imread(str(directories[0] / f"A4_80mm_page_{page}.png"))
        assert sheet.shape[:2] == (round(297 / 25.4 * 300), round(210 / 25.4 * 300))
        _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(sheet)
        assert sorted(ids.ravel().tolist()) == [page * 2 - 1, page * 2]
    assert len(list(directories[0].glob("*.png"))) == 9
    assert (directories[0] / 'ArUco_80mm_A4.pdf').is_file()
    assert (directories[0] / '打印与裁剪说明.md').is_file()


def test_80mm_card_pixels_quiet_zone_and_real_detection():
    # 独立按 300 DPI 换算物理尺寸，不以生成器常量作为期望值。
    side, width, height = round(80 / 25.4 * 300), round(104 / 25.4 * 300), round(120 / 25.4 * 300)
    card = markers.generate_printable_marker_card(3, side, width, height)
    x, y = (width - side) // 2, round(12 / 25.4 * 300)
    np.testing.assert_array_equal(card[y:y + side, x:x + side], markers.generate_aruco_marker(3, side))
    for strip in (card[1:y, x:x + side], card[y:y + side, 1:x],
                  card[y:y + side, x + side:-1], card[y + side:y + side + y, x:x + side]):
        assert np.all(strip == 255)
    assert side / 300 * 25.4 == pytest.approx(80, abs=.02)
    assert x / 300 * 25.4 == pytest.approx(12, abs=.1)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    _, ids, _ = detector.detectMarkers(card)
    assert ids.ravel().tolist() == [3]


@pytest.mark.skipif(not shutil.which('pdfinfo') or not shutil.which('pdftoppm'), reason='PDF 渲染器不可用')
def test_a4_pdf_real_render_detects_six_80mm_markers(tmp_path):
    path = tmp_path / 'print.pdf'
    markers.create_a4_print_pdf(path)
    info = subprocess.run(['pdfinfo', str(path)], check=True, capture_output=True, text=True).stdout
    assert 'Pages:           3' in info and '(A4)' in info
    assert b'/PrintScaling /None' in path.read_bytes()
    subprocess.run(['pdftoppm', '-r', '150', '-gray', '-png', str(path), str(tmp_path / 'page')],
                   check=True, capture_output=True)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    for page in range(1, 4):
        image = cv2.imread(str(tmp_path / f'page-{page}.png'), cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = detector.detectMarkers(image)
        assert sorted(ids.ravel().tolist()) == [page * 2 - 1, page * 2]
        for quad in corners:
            points = quad.reshape(4, 2)
            sides_mm = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) * 25.4 / 150
            np.testing.assert_allclose(sides_mm, 80, atol=.35)
    with pytest.raises(FileExistsError):
        markers.create_a4_print_pdf(path)


def test_real_raster_detection_pnp_and_calibration_end_to_end(tmp_path, monkeypatch):
    """真实 80 mm 生成码的合成相机画面→ArUco→PnP→录入→SVD，不替换检测或求解。"""
    # 这里是模拟相机画面而非打印 A4，六张码排列为 2×3。
    side, card_w, card_h, gap = 420, 546, 630, 40
    w, h = card_w * 2 + gap * 3, card_h * 3 + gap * 4
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    camera = []
    fx = 2500.
    z = .08 * fx / side
    for row in range(3):
        for col in range(2):
            x, y = gap + col * (card_w + gap), gap + row * (card_h + gap)
            card = markers.generate_printable_marker_card(row * 2 + col + 1, side, card_w, card_h)
            image[y:y + card_h, x:x + card_w] = card
            u, v = x + (card_w - side) // 2 + (side - 1) / 2, y + 63 + (side - 1) / 2
            camera.append([(u - w / 2) * z / fx, (v - h / 2) * z / fx, z])
    image_path = tmp_path / 'camera.png'
    assert cv2.imwrite(str(image_path), image)
    intrinsics = {"schema_version": 1, "rgb": {"width": w, "height": h, "fx": fx, "fy": fx,
                  "cx": w / 2, "cy": h / 2, "dist": [0] * 5}}
    intrin_path = tmp_path / "camera.json"
    intrin_path.write_text(json.dumps(intrinsics))
    base = np.array(camera) @ ROTATION.T + TRANSLATION
    responses = iter([" ".join(format(v, '.17g') for v in point) for point in base])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))
    assert cal.run_touch_calibration(str(image_path), str(intrin_path), unit="m", tcp_description="synthetic contact",
                                     output_root=tmp_path / "results")
    candidates = list((tmp_path / "results").glob("*/T_base_camera.candidate.json"))
    T = np.array(json.loads(candidates[0].read_text())["matrix"])
    np.testing.assert_allclose(T[:3, :3], ROTATION, atol=1e-3)
    np.testing.assert_allclose(T[:3, 3], TRANSLATION, atol=1e-3)


@pytest.mark.parametrize('argv,entry', [([], 'run_visual_calibration'),
    (['--detect-only', '--browser'], 'run_marker_preview'), (['--terminal'], 'run_touch_calibration'),
    (['--image', 'input.png'], 'run_touch_calibration')])
def test_all_calibration_entry_points_default_to_80mm(monkeypatch, argv, entry):
    calls = []
    monkeypatch.setattr(cal, entry, lambda **kwargs: calls.append(kwargs) or True)
    assert cal.main(argv) == 0
    assert calls[0]['marker_length_m'] == .08


def test_pnp_ambiguous_candidate_centers_are_rejected():
    K = np.array([[800., 0, 320], [0, 790., 240], [0, 0, 1]])
    obj = np.array([[-.02, .02, 0], [.02, .02, 0], [.02, -.02, 0], [-.02, -.02, 0]])
    corners = cv2.projectPoints(obj, np.array([2.6, .1, -.15]), np.array([.03, -.02, .7]), K, np.zeros(5))[0].reshape(4, 2)
    # 将待验证的歧义门槛收紧，两个真实 IPPE 候选因此不能被当成唯一中心。
    quality = replace(cal.CalibrationQuality(), ambiguity_error_gap_px=1.5, max_ambiguity_translation_m=.0001)
    with pytest.raises(ValueError, match="歧义"):
        cal._marker_pose(corners, K, np.zeros(5), .04, quality)


def test_invalid_validation_file_is_not_marked_unperformed(workflow):
    path = workflow.root.parent / "bad_validation.json"
    path.write_text('{"schema_version":true}')
    assert not workflow.run(validation_file=path)
    record, directory = read_record(workflow, "failure.json")
    assert record["independent_validation"]["status"] == "invalid"
    assert not list(directory.glob("*candidate*"))


@pytest.fixture
def preview_scene(tmp_path):
    image = markers.generate_printable_marker_card(1, 200, 600, 600)
    image_path = tmp_path / "preview_input.png"
    assert cv2.imwrite(str(image_path), image)
    intrinsics = {"schema_version": 1, "rgb": {"width": 600, "height": 600,
                  "fx": 800., "fy": 800., "cx": 300., "cy": 300., "dist": [0] * 5}}
    intrin_path = tmp_path / "preview_intrinsics.json"
    intrin_path.write_text(json.dumps(intrinsics))
    return SimpleNamespace(image=image, image_path=image_path, intrinsics=intrin_path,
                           root=tmp_path / "preview_results")


def test_preview_reports_actual_pixel_size_and_missing_ids(preview_scene):
    K, dist, _ = cal.load_camera_intrinsics(preview_scene.intrinsics)
    report, annotated = cal.inspect_markers(preview_scene.image, K, dist)
    assert report["decoded_count"] == report["quality_pass_count"] == 1
    assert report["markers"][0]["id"] == 1
    assert report["markers"][0]["min_side_px"] == pytest.approx(200, abs=1)
    assert report["missing_ids"] == [2, 3, 4, 5, 6]
    assert annotated.shape == preview_scene.image.shape
    assert not np.array_equal(annotated, preview_scene.image)


@pytest.mark.parametrize("ids, side, reason", [([1], 10, "过小"), ([1, 1], 100, "重复 ID"), ([9], 100, "非期望 ID")])
def test_preview_keeps_decoded_but_rejected_markers(preview_scene, monkeypatch, ids, side, reason):
    corners = np.array([[[250., 250], [250 + side, 250], [250 + side, 250 + side], [250, 250 + side]]])
    monkeypatch.setattr(cal, "_aruco_observations", lambda image: ([corners] * len(ids), np.array(ids).reshape(-1, 1), []))
    K, dist, _ = cal.load_camera_intrinsics(preview_scene.intrinsics)
    report, _ = cal.inspect_markers(preview_scene.image, K, dist)
    assert report["decoded_count"] == len(ids)
    assert report["quality_pass_count"] == 0
    assert all(reason in item["reason"] for item in report["markers"])


def test_detection_only_cli_never_prompts_or_calibrates(preview_scene, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("仅识别模式不得询问单位/TCP/坐标或求解外参")
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(cal, "run_touch_calibration", forbidden)
    monkeypatch.setattr(cal, "solve_rigid_transform_3d", forbidden)
    assert cal.main(["--detect-only", "--no-display", "--image", str(preview_scene.image_path),
                     "--intrinsics", str(preview_scene.intrinsics), "--output-root", str(preview_scene.root)]) == 0
    paths = list(preview_scene.root.glob("*/preview.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["mode"] == "detection_preview" and record["status"] == "completed"
    assert record["frames_processed"] == 1
    assert record["per_id"]["1"]["decoded_frames"] == 1
    assert record["last_frame"]["decoded_count"] == 1
    assert {p.name for p in paths[0].parent.iterdir()} == {"preview.json", "preview_raw.png", "preview_annotated.png"}


def test_preview_no_marker_is_valid_observation(preview_scene):
    assert cv2.imwrite(str(preview_scene.image_path), np.full_like(preview_scene.image, 255))
    assert cal.run_marker_preview(str(preview_scene.image_path), str(preview_scene.intrinsics),
                                  no_display=True, output_root=preview_scene.root)
    record = json.loads(next(preview_scene.root.glob("*/preview.json")).read_text())
    assert record["last_frame"]["decoded_count"] == 0
    assert record["last_frame"]["missing_ids"] == [1, 2, 3, 4, 5, 6]
    assert record["per_id"] == {}


def test_preview_bad_image_is_failure(preview_scene):
    preview_scene.image_path.write_bytes(b"bad PNG")
    assert not cal.run_marker_preview(str(preview_scene.image_path), str(preview_scene.intrinsics),
                                      no_display=True, output_root=preview_scene.root)
    assert len(list(preview_scene.root.glob("*/preview_failure.json"))) == 1
    assert not list(preview_scene.root.glob("*/preview.json"))


def test_headless_live_preview_limits_frames_and_releases_camera(preview_scene, monkeypatch):
    reads, released = [], []
    def read():
        reads.append(True)
        return True, preview_scene.image.copy()
    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True, read=read,
                             release=lambda: released.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: camera)
    assert cal.run_marker_preview(intrin_file=str(preview_scene.intrinsics), no_display=True,
                                  sample_count=3, output_root=preview_scene.root)
    record = json.loads(next(preview_scene.root.glob("*/preview.json")).read_text())
    assert len(reads) == 8 + 3
    assert released == [True]
    assert record["frames_processed"] == 3
    assert record["per_id"]["1"]["decoded_frames"] == record["per_id"]["1"]["quality_pass_frames"] == 3


def test_live_preview_quit_closes_camera_and_window(preview_scene, monkeypatch):
    released, closed = [], []
    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True,
                             read=lambda: (True, preview_scene.image.copy()), release=lambda: released.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: camera)
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: ord('q'))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: closed.append(True))
    assert cal.run_marker_preview(intrin_file=str(preview_scene.intrinsics), output_root=preview_scene.root)
    assert released == closed == [True]
    record = json.loads(next(preview_scene.root.glob("*/preview.json")).read_text())
    assert record["frames_processed"] == 1


def _stability_report(side=80., rms=.2, corner_shift=0., position_shift=0.):
    corners = np.array([[100., 100.], [180, 100], [180, 180], [100, 180]]) + corner_shift
    return {"markers": [{"id": 1, "quality_pass": True, "corners_px": corners.tolist(),
                         "min_side_px": side, "reprojection_error_px": rms,
                         "camera_xyz_m": [.01 + position_shift, .02, .6]}]}


def test_stability_requires_time_and_frames_before_ready():
    tracker = cal.MarkerStability([1])
    initial = tracker.update(_stability_report(), 0.)
    assert initial["status"] == "WARMUP"
    for timestamp in np.linspace(.1, 3., 30):
        result = tracker.update(_stability_report(), timestamp)
    assert result["status"] == "READY"
    assert result["ready_for_sampling"]
    row = result["markers"][0]
    assert row["detection_rate"] == row["quality_rate"] == 1
    assert row["position_rms_mm"] < 1e-9
    assert row["corner_jitter_px"] == 0


@pytest.mark.parametrize("problem, reason", [("side", "too_small"), ("rms", "reprojection"),
    ("corner", "corner_motion"), ("position", "position_motion"), ("dropout", "low_detection_rate")])
def test_stability_rejects_unreliable_measurements(problem, reason):
    tracker = cal.MarkerStability([1])
    for i in range(32):
        sign = 1 if i % 2 else -1
        report = _stability_report(side=59 if problem == 'side' else 80,
                                   rms=1.1 if problem == 'rms' else .2,
                                   corner_shift=sign if problem == 'corner' else 0,
                                   position_shift=sign * .005 if problem == 'position' else 0)
        if problem == 'dropout' and i % 5 == 0:
            report = {"markers": []}
        result = tracker.update(report, i / 10)
    assert result["status"] == "CHECK"
    assert reason in result["markers"][0]["reasons"]


def test_stability_missing_current_frame_cannot_pass_on_historical_success():
    tracker = cal.MarkerStability([1])
    for i in range(32):
        tracker.update(_stability_report(), i / 10)
    result = tracker.update({"markers": []}, 3.2)
    assert not result["ready_for_sampling"]
    assert "missing_or_invalid_now" in result["markers"][0]["reasons"]


def test_stability_window_recovers_after_motion_leaves_window():
    tracker = cal.MarkerStability([1])
    for i in range(80):
        result = tracker.update(_stability_report(position_shift=.04 if i < 20 else 0), i / 10)
    assert result["ready_for_sampling"]
    assert result["window_frames"] < 40
    assert result["markers"][0]["position_peak_mm"] < 1e-9


def test_stability_requires_all_expected_ids():
    tracker = cal.MarkerStability([1, 2])
    for i in range(32):
        result = tracker.update(_stability_report(), i / 10)
    assert result["markers"][0]["status"] == "READY"
    assert result["markers"][1]["detection_rate"] == 0
    assert not result["ready_for_sampling"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field, value", [("min_frames", 1), ("min_frames", True),
    ("window_s", float('nan')), ("min_rate", 1.1), ("max_rms_px", 0)])
def test_preview_criteria_validation(field, value):
    with pytest.raises(ValueError):
        replace(cal.PreviewCriteria(), **{field: value}).validate()


def test_browser_preview_serves_synchronized_image_and_metrics(preview_scene):
    import urllib.request
    server = cal.BrowserPreview(0)
    address = f'http://127.0.0.1:{server.port}'
    try:
        with urllib.request.urlopen(address + '/', timeout=2) as response:
            assert '标记实时预览' in response.read().decode()
        server.publish(preview_scene.image, {"frames_processed": 7, "stability": {"status": "WARMUP"}})
        with urllib.request.urlopen(address + '/snapshot', timeout=2) as response:
            state = json.loads(response.read())
            assert response.headers['Cache-Control'] == 'no-store'
        import base64
        image = cv2.imdecode(np.frombuffer(base64.b64decode(state['image']), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == preview_scene.image.shape
        assert state['frames_processed'] == 7
        assert state['stability']['status'] == 'WARMUP'
        assert state['frame_age_s'] >= 0
        request = urllib.request.Request(address + '/stop', method='POST')
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 204
        assert server.stopped.is_set()
        assert server.server.server_address[0] == '127.0.0.1'
    finally:
        server.close()
    assert not server.thread.is_alive()


def test_browser_mode_skips_desktop_and_saves_on_browser_stop(preview_scene, monkeypatch):
    import urllib.request
    base = cal.BrowserPreview
    class AutoStop(base):
        def publish(self, image, metrics):
            super().publish(image, metrics)
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/snapshot', timeout=2) as response:
                state = json.loads(response.read())
            assert state['frames_processed'] == 1
            assert state['assumed_marker_length_m'] == .08
            self.stopped.set()
    def forbidden(*args, **kwargs):
        pytest.fail('远程浏览器模式不得打开桌面窗口或询问坐标')
    monkeypatch.setattr(cal, 'BrowserPreview', AutoStop)
    monkeypatch.setattr(cal, '_preview_live_frames', lambda *a: iter([preview_scene.image] * 2))
    monkeypatch.setattr(cv2, 'namedWindow', forbidden)
    monkeypatch.setattr('builtins.input', forbidden)
    assert cal.main(['--detect-only', '--browser', '--preview-port', '0', '--intrinsics', str(preview_scene.intrinsics),
                     '--output-root', str(preview_scene.root)]) == 0
    record = json.loads(next(preview_scene.root.glob('*/preview.json')).read_text())
    assert record['display_mode'] == 'browser'
    assert record['frames_processed'] == 1
    assert record['stability']['status'] == 'WARMUP'
    assert not list(preview_scene.root.glob('*/*candidate*'))


def test_preview_ctrl_c_preserves_last_complete_frame(preview_scene, monkeypatch):
    def images(*args):
        yield preview_scene.image
        raise KeyboardInterrupt
    monkeypatch.setattr(cal, '_preview_live_frames', images)
    assert cal.run_marker_preview(intrin_file=str(preview_scene.intrinsics), no_display=True, output_root=preview_scene.root)
    record = json.loads(next(preview_scene.root.glob('*/preview.json')).read_text())
    assert record['frames_processed'] == 1
    assert record['exit_reason'] == 'keyboard_interrupt_after_frames'
    assert not list(preview_scene.root.glob('*/preview_failure.json'))
    assert len(list(preview_scene.root.glob('*/preview_raw.png'))) == 1


def test_calibration_capture_sets_existing_project_mjpg_mode(preview_scene, monkeypatch):
    settings, released = [], []
    camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: settings.append(a),
                             read=lambda: (True, preview_scene.image.copy()), release=lambda: released.append(True))
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a: camera)
    intrinsics = json.loads(preview_scene.intrinsics.read_text())
    stream = cal._preview_live_frames('0', intrinsics, 1)
    assert next(stream).shape == preview_scene.image.shape
    stream.close()
    cal._acquire_images('live', '0', intrinsics, 1)
    assert settings[0] == (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    assert settings[3] == settings[0]
    assert released == [True, True]


def test_browser_marks_old_ready_snapshot_as_stale_without_mutating_metrics(preview_scene):
    import urllib.request
    tracker = cal.MarkerStability([1])
    for i in range(32):
        assessment = tracker.update(_stability_report(), i / 10)
    assert assessment['ready_for_sampling']
    server = cal.BrowserPreview(0)
    try:
        server.publish(preview_scene.image, {'frames_processed': 32, 'stability': assessment})
        with server.lock:
            server.updated_at -= 2
        with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/snapshot', timeout=2) as response:
            state = json.loads(response.read())
        assert state['stability']['status'] == 'STALE'
        assert not state['stability']['ready_for_sampling']
        assert 'stale_frame' in state['stability']['markers'][0]['reasons']
        assert assessment['status'] == 'READY'
        assert assessment['ready_for_sampling']
    finally:
        server.close()


def test_decoded_id_with_bad_pose_is_not_ready():
    tracker = cal.MarkerStability([1])
    report = _stability_report()
    report['markers'][0]['quality_pass'] = False
    for i in range(32):
        result = tracker.update(report, i / 10)
    assert result['markers'][0]['detection_rate'] == 1
    assert result['markers'][0]['quality_rate'] == 0
    assert not result['ready_for_sampling']


def test_wrong_marker_size_has_small_reprojection_but_wrong_position():
    K = np.array([[900., 0, 320], [0, 890., 240], [0, 0, 1]])
    obj = np.array([[-.04, .04, 0], [.04, .04, 0], [.04, -.04, 0], [-.04, -.04, 0]])
    true_position = np.array([.03, -.02, .7])
    corners = cv2.projectPoints(obj, np.array([2.8, .1, -.15]), true_position, K, np.zeros(5))[0].reshape(4, 2)
    wrong = cal._marker_pose(corners, K, np.zeros(5), .04, cal.CalibrationQuality())
    correct = cal._marker_pose(corners, K, np.zeros(5), .08, cal.CalibrationQuality())
    assert wrong['reprojection_error_px'] < 1e-6
    np.testing.assert_allclose(wrong['P_cam'], true_position / 2, atol=1e-8)
    np.testing.assert_allclose(correct['P_cam'], true_position, atol=1e-8)


def _visual_report(rms=.2, shift=0.):
    items = []
    for mid, point in enumerate(CAMERA, 1):
        item = _stability_report(rms=rms)['markers'][0]
        item.update(id=mid, camera_xyz_m=(point + [shift, 0, 0]).tolist())
        items.append(item)
    return {'markers': items}


def _feed_visual(session, start=0., count=24, rms=.2, shift=0.):
    raw = np.zeros((48, 64, 3), np.uint8)
    for i in range(count):
        timestamp = start + i * .05
        session.observe(_visual_report(rms, shift), timestamp, raw, raw, session.generation)
    return start + count * .05


def _started_visual():
    session = cal.VisualCalibrationSession(range(1, 7), replace(cal.PreviewCriteria(), window_s=.2, min_frames=2))
    session.action({'action': 'start', 'marker_length_mm': 80}, now=0.)
    return session


def _ready_visual():
    session = _started_visual()
    end = _feed_visual(session, count=100)
    assert session.phase == 'captured'
    assert session.snapshot()['compared_windows'] == 10
    return session, end


def _coordinates_visual(session, now, base=BASE):
    session.coordinates_mm = {mid: (point * 1000).tolist() for mid, point in enumerate(base, 1)}
    session.phase = 'solving'
    session.attempt = 1


def test_terminal_default_mm_never_asks_unit(workflow):
    assert workflow.run(unit=None)
    record, _ = read_record(workflow)
    assert record['unit'] == 'mm'


def test_default_cli_enters_visual_calibration_without_terminal_prompt(monkeypatch):
    calls = []
    monkeypatch.setattr(cal, 'run_visual_calibration', lambda **kwargs: calls.append(kwargs) or True)
    monkeypatch.setattr('builtins.input', lambda *args: pytest.fail('默认 mm 不应询问单位或 TCP'))
    assert cal.main([]) == 0
    assert calls[0]['marker_length_m'] == .08
    assert calls[0]['target_windows'] == 10
    assert calls[0]['tcp_description'] == '闭合夹爪两指尖中心'


def test_visual_mm_marker_size_cli_parameter(monkeypatch):
    calls = []
    monkeypatch.setattr(cal, 'run_visual_calibration', lambda **kwargs: calls.append(kwargs) or True)
    assert cal.main(['--marker-length-mm', '79.8']) == 0
    assert calls[0]['marker_length_m'] == pytest.approx(.0798)


def test_visual_preview_keeps_detecting_without_collecting_before_start():
    session = cal.VisualCalibrationSession(range(1, 7), replace(cal.PreviewCriteria(), window_s=.2, min_frames=2))
    _feed_visual(session, count=120)
    assert session.last_assessment['ready_for_sampling']
    assert session.phase == 'preview'
    assert session.snapshot()['compared_windows'] == 0
    assert session.best is None
    session.action({'action': 'start', 'marker_length_mm': 80}, now=6.)
    _feed_visual(session, start=6., count=4)
    assert session.snapshot()['compared_windows'] == 0  # 开始前的稳定历史不得计数
    _feed_visual(session, start=6.2, count=1)
    assert session.snapshot()['compared_windows'] == 1


def test_visual_stops_at_exactly_ten_and_rejects_further_frames():
    session = _started_visual()
    raw = np.zeros((48, 64, 3), np.uint8)
    for i in range(100):
        session.observe(_visual_report(), i * .05, raw, raw, session.generation)
        if session.phase == 'captured':
            break
    else:
        pytest.fail('十次后未自动结束')
    before = session.export()
    revision = session.revision
    _feed_visual(session, start=100., count=100, rms=.01, shift=.03)
    assert session.export() == before
    assert session.revision == revision
    assert len(session.comparisons) == 10
    with pytest.raises(ValueError):
        session.action({'action': 'start', 'marker_length_mm': 40})
    with pytest.raises(ValueError):
        session.action({'action': 'reset'})


def test_visual_unqualified_data_does_not_count():
    session = _started_visual()
    end = _feed_visual(session, count=100, rms=1.2)
    assert session.phase == 'collecting' and len(session.comparisons) == 0
    _feed_visual(session, start=end, count=100)
    assert session.phase == 'captured' and len(session.comparisons) == 10


def test_visual_keeps_lowest_quality_error_window_not_latest():
    session = _started_visual()
    end = _feed_visual(session, count=16, rms=.4)
    end = _feed_visual(session, start=end, count=20, rms=.1)
    best = session.snapshot()['best']
    assert best['score'][0] == pytest.approx(.1)
    _feed_visual(session, start=end, count=100, rms=.8)
    state = session.snapshot()
    assert state['compared_windows'] == 10
    assert state['best'] == best
    assert best['version'] < 10


def test_visual_layout_change_discards_old_comparisons():
    session = _started_visual()
    end = _feed_visual(session, count=16)
    raw = np.zeros((48, 64, 3), np.uint8)
    for i in range(20):
        now = end + i * .05
        session.observe(_visual_report(shift=.03), now, raw, raw, session.generation)
        state = session.snapshot(now)
        if '布局变化' in state['error']:
            assert state['compared_windows'] == 1
            for mid in session.expected_ids:
                np.testing.assert_allclose(session.best['points'][mid], CAMERA[mid - 1] + [.03, 0, 0])
            break
    else:
        pytest.fail('布局改变后没有重新比较')


def test_visual_reset_clears_old_size_frames_and_requires_start_again():
    session = _started_visual()
    end = _feed_visual(session, count=16)
    generation = session.generation
    session.action({'action': 'reset'})
    assert session.phase == 'preview'
    assert session.best is None and not session.comparisons
    session.action({'action': 'start', 'marker_length_mm': 40})
    raw = np.zeros((48, 64, 3), np.uint8)
    assert session.observe(_visual_report(), end, raw, raw, generation) is None
    assert session.marker_length_m == .04
    assert session.snapshot()['compared_windows'] == 0


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_visual_rejects_invalid_collection_limit(count):
    with pytest.raises(ValueError):
        cal.VisualCalibrationSession(range(1, 7), target_windows=count)


def test_visual_double_start_does_not_restart_collection():
    session = _started_visual()
    _feed_visual(session, count=16)
    before = session.export()
    with pytest.raises(ValueError):
        session.action({'action': 'start', 'marker_length_mm': 80})
    assert session.export() == before


@pytest.mark.parametrize('bad', ['1 2', '1 nan 3', 'inf 2 3', 'a b c'])
def test_visual_terminal_coordinates_retry_invalid_mm_input(monkeypatch, bad):
    session, _ = _ready_visual()
    responses = iter([bad] + [' '.join(format(v * 1000, '.17g') for v in p) for p in BASE])
    prompts = []
    def enter(prompt):
        prompts.append(prompt)
        return next(responses)
    monkeypatch.setattr('builtins.input', enter)
    assert cal._read_visual_coordinates(session)
    assert len(prompts) == 7 and prompts[0] == prompts[1]
    assert all('mm' in prompt for prompt in prompts)
    np.testing.assert_allclose([session.coordinates_mm[i] for i in range(1, 7)], BASE * 1000)


def test_visual_selection_saved_before_robot_coordinates(tmp_path):
    session, _ = _ready_visual()
    path = cal._save_visual_selection(session, tmp_path, {})
    data = json.loads(path.read_text())
    assert data['status'] == 'observations_saved'
    assert len(data['selection']['comparisons']) == 10
    assert data['selection']['coordinates_mm'] == {}
    assert (tmp_path / data['selected_image']).is_file()
    assert (tmp_path / data['annotated_image']).is_file()
    assert not list(tmp_path.glob('*candidate*'))


def test_visual_candidate_uses_selected_multi_frame_points_and_existing_schema(tmp_path):
    session, end = _ready_visual()
    _coordinates_visual(session, end - .05)
    result = cal._save_visual_candidate(session, tmp_path, {'tcp_description': cal.DEFAULT_TCP_DESCRIPTION}, cal.CalibrationQuality())
    matrix = np.array(result['candidate']['matrix'])
    np.testing.assert_allclose(matrix[:3, :3], ROTATION, atol=1e-12)
    np.testing.assert_allclose(matrix[:3, 3], TRANSLATION, atol=1e-12)
    assert set(result['candidate']) == {'schema_version', 'from_frame', 'to_frame', 'matrix'}
    record = json.loads((tmp_path / 'session.json').read_text())
    assert record['unit'] == 'mm'
    assert len(record['selection']['comparisons']) == 10
    assert len(record['selection']['selected_observations']) >= 2
    assert record['marker_length_m'] == .08
    assert record['independent_validation']['status'] == 'not_performed'
    assert record['fit']['max_mm'] < 1e-8


def test_visual_bad_fit_produces_no_candidate(tmp_path):
    session, end = _ready_visual()
    base = BASE.copy()
    base[0] += [.05, 0, 0]
    _coordinates_visual(session, end - .05, base)
    with pytest.raises(ValueError, match='超限'):
        cal._save_visual_candidate(session, tmp_path, {}, cal.CalibrationQuality())
    assert not (tmp_path / 'T_base_camera.candidate.json').exists()


def test_visual_session_record_failure_removes_candidate(tmp_path, monkeypatch):
    session, end = _ready_visual()
    _coordinates_visual(session, end - .05)
    write = cal.atomic_write_json
    def failing_write(path, data):
        if Path(path).name == 'session.json':
            raise OSError('simulated record failure')
        write(path, data)
    monkeypatch.setattr(cal, 'atomic_write_json', failing_write)
    with pytest.raises(OSError):
        cal._save_visual_candidate(session, tmp_path, {}, cal.CalibrationQuality())
    assert not (tmp_path / 'T_base_camera.candidate.json').exists()


@pytest.mark.parametrize('outcome', ['success', 'closed_page', 'bad_fit', 'cancel', 'interrupt', 'save_error'])
def test_visual_ten_batches_http_then_terminal_after_camera_and_server_close(tmp_path, preview_scene, monkeypatch, outcome):
    import urllib.request
    base_server = cal.BrowserPreview
    clock = [time_value := cal.time.monotonic()]
    resources = {'frames_closed': False, 'server_closed': False, 'reads': 0, 'terminal_reads': 0}
    root = tmp_path / 'formal'
    servers = []
    def frames(*args):
        try:
            for i in range(100):
                resources['reads'] += 1
                clock[0] = time_value + i * .11
                yield preview_scene.image.copy()
            pytest.fail('十次合格窗口后仍继续取帧')
        finally:
            resources['frames_closed'] = True
    class Operator(base_server):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)
            event = self.final_delivered
            owner = self
            class Delivery:
                def set(self):
                    event.set()
                def wait(self, timeout):
                    assert resources['frames_closed']
                    if outcome == 'closed_page':
                        return event.wait(timeout)  # 关闭网页也不阻塞转入终端
                    state = owner.get_state()
                    assert state['phase'] == 'saved'
                    assert len(list(root.glob('*/selection.json'))) == 1
                    assert state['compared_windows'] == 10
                    assert '终端' in state['result']['next']
                    return event.wait(timeout)
            self.final_delivered = Delivery()
        def get_state(self):
            with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/snapshot', timeout=2) as response:
                return json.loads(response.read())['calibration']
        def publish(self, image, metrics):
            super().publish(image, metrics)
            state = self.get_state()
            if state['phase'] == 'preview':
                assert state['compared_windows'] == 0
                request = urllib.request.Request(f'http://127.0.0.1:{self.port}/calibration',
                    data=json.dumps({'action': 'start', 'marker_length_mm': 80}).encode(),
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=2):
                    pass
        def close(self):
            super().close()
            resources['server_closed'] = True
    points = BASE.copy()
    if outcome == 'bad_fit':
        points[0] += [.05, 0, 0]
    responses = iter(['q'] if outcome == 'cancel' else [' '.join(format(v * 1000, '.17g') for v in p) for p in points])
    reads_at_input = []
    def enter(prompt):
        resources['terminal_reads'] += 1
        assert resources['frames_closed'] and resources['server_closed']
        assert not servers[0].thread.is_alive()
        assert servers[0].server.socket.fileno() == -1
        assert len(list(root.glob('*/selection.json'))) == 1
        assert len(list(root.glob('*/selected_raw.png'))) == 1
        reads_at_input.append(resources['reads'])
        assert len(set(reads_at_input)) == 1
        assert 'mm' in prompt
        if outcome == 'interrupt':
            raise KeyboardInterrupt
        return next(responses)
    if outcome == 'save_error':
        write = cal.atomic_write_json
        def fail_selection(path, data):
            if Path(path).name == 'selection.json':
                raise OSError('simulated selection save error')
            write(path, data)
        monkeypatch.setattr(cal, 'atomic_write_json', fail_selection)
    monkeypatch.setattr(cal.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(cal, 'BrowserPreview', Operator)
    monkeypatch.setattr(cal, '_preview_live_frames', frames)
    monkeypatch.setattr(cal, 'inspect_markers', lambda image, *a, **kw: (_visual_report(), image.copy()))
    monkeypatch.setattr('builtins.input', enter)
    criteria = replace(cal.PreviewCriteria(), window_s=.1, min_frames=2)
    success = outcome in ('success', 'closed_page')
    assert cal.run_visual_calibration(str(preview_scene.intrinsics), output_root=root,
                                      preview_port=0, criteria=criteria) is success
    assert resources['frames_closed'] and resources['server_closed']
    assert resources['reads'] == 12  # 1 帧预览、1 帧预热、10 个合格窗口，不读取第 13 帧
    if outcome == 'save_error':
        assert resources['terminal_reads'] == 0
    if success:
        data = json.loads(next(root.glob('*/session.json')).read_text())
        assert data['mode'] == 'visual_calibration' and data['unit'] == 'mm'
        assert len(data['selection']['comparisons']) == 10
        assert data['fit']['max_mm'] < 1e-8
    elif outcome in ('cancel', 'interrupt'):
        assert len(list(root.glob('*/cancelled.json'))) == 1
    else:
        assert len(list(root.glob('*/failure.json'))) == 1
    assert bool(list(root.glob('*/*candidate*'))) is success


def test_visual_cancel_saves_draft_and_never_candidate(preview_scene, monkeypatch):
    base_server = cal.BrowserPreview
    class Stop(base_server):
        def publish(self, image, metrics):
            super().publish(image, metrics)
            self.stopped.set()
    monkeypatch.setattr(cal, 'BrowserPreview', Stop)
    monkeypatch.setattr(cal, '_preview_live_frames', lambda *a: iter([preview_scene.image]))
    assert not cal.run_visual_calibration(str(preview_scene.intrinsics), output_root=preview_scene.root, preview_port=0)
    drafts = list(preview_scene.root.glob('*/cancelled.json'))
    assert len(drafts) == 1
    assert json.loads(drafts[0].read_text())['status'] == 'cancelled'
    assert not list(preview_scene.root.glob('*/*candidate*'))
