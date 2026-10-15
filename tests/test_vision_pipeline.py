"""P3 原生回放、几何与迁移接口回归，无权重、无设备。

合成下视 RGB-D，相机距桌面 0.5m；物体顶面 0.48m，中心高 0.01m。
所有内参/外参/深度均为明确名义值，不作为实测证据。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from configs.common_interface import VisionThresholds
from qingyun import runlog
from qingyun.grabbing import vision
from qingyun.grabbing._vision_pipeline import align_depth, filter_and_rank, measure_geometry, Pipeline


@pytest.fixture(autouse=True)
def isolated(tmp_path):
    vision._reset_for_tests()
    runlog.reset_for_tests()
    runlog.init(tmp_path/'logs')
    rows = []
    runlog.set_sink(lambda line: rows.append(json.loads(line)))
    yield rows
    vision._reset_for_tests()
    runlog.reset_for_tests()


def bundle(root, *, detections=None, depth=None, frame_count=1):
    root.mkdir(parents=True, exist_ok=True)
    intrinsics = dict(schema_version=1,
        rgb=dict(width=80, height=60, fx=100., fy=120., cx=40., cy=30., dist=[0.]*5),
        depth=dict(width=80, height=60, fx=100., fy=120., cx=40., cy=30., dist=[0.]*5),
        depth_to_rgb=np.eye(4).tolist(), depth_unit_m=.001)
    T = np.diag([1., -1., -1., 1.]); T[2, 3] = .5
    extrinsic = dict(schema_version=1, from_frame='rgb_optical', to_frame='base_link', matrix=T.tolist())
    if depth is None:
        depth = np.full((60, 80), 500, dtype=np.uint16)
        depth[28:32, 37:44] = 480
    if detections is None:
        detections = [dict(class_='RIPE', conf=.9, corners_px=[[36,27],[44,27],[44,32],[36,32]])]
        detections[0]['class'] = detections[0].pop('class_')
    np.save(root/'color.npy', np.zeros((60,80,3), dtype=np.uint8))
    np.save(root/'depth.npy', depth)
    (root/'detections.json').write_text(json.dumps(detections))
    (root/'intrinsics.json').write_text(json.dumps(intrinsics))
    (root/'extrinsic.json').write_text(json.dumps(extrinsic))
    (root/'manifest.json').write_text(json.dumps(dict(schema_version=1, frames=[dict(id=str(i),color='color.npy',depth='depth.npy',detections='detections.json',rgb_time_s=float(i),depth_time_s=float(i)) for i in range(frame_count)])))
    config = dict(camera=dict(type='orbbec_astra_pro',depth_mode='640x480@30',rgb_device=None,rgb_size=None,max_pair_delta_s=.03,acquire_timeout_s=1.),
        model=dict(backend='fixture',path=None,detection_floor=.05,conf_threshold=.5,class_ripe=['RIPE'],class_unripe=['UNRIPE']),
        calibration=dict(intrinsics_path='intrinsics.json',extrinsic_base_camera_path='extrinsic.json'),
        geometry=dict(min_points=3,inner_scale=.8,depth_neighbor_jump_m=.01,footprint_padding_m=0.),
        source=dict(mode='replay',replay_dir='.'))
    path=root/'vision.json'; path.write_text(json.dumps(config))
    thresholds=VisionThresholds([[-.2,-.2,0],[.2,.2,.1]], [.1,.1,.1], .001,.05,0.,.001)
    return path, thresholds, config, intrinsics, T


def start(path, thresholds):
    vision.configure(thresholds,path); vision.init()


def test_basic_geometry_and_init_does_not_consume(tmp_path, isolated):
    path,t,_,_,_=bundle(tmp_path/'case')
    start(path,t)
    with runlog.context(task_instance_id='task-1',scan_id=7):
        target=vision.get_target()
    np.testing.assert_allclose(target.position,[0,.002,.01],atol=1e-7)
    assert target.position.dtype == np.float64
    assert target.length_m == pytest.approx(.0288,abs=1e-7)
    assert target.width_m == pytest.approx(.012,abs=1e-7)
    assert target.yaw_deg == pytest.approx(0)
    assert target.ripe is True and target.valid_count == 1
    scans=[r for r in isolated if r['kind']=='vision_scan']
    assert len(scans)==1 and scans[0]['scan_id']==7 and scans[0]['task_instance_id']=='task-1'
    assert scans[0]['candidates'][0]['clearance_m'] is None
    with pytest.raises(vision.NoTarget,match='replay_exhausted'): vision.get_target()


def test_original_branch_three_smoke_cases(tmp_path):
    # 覆盖原分支 T01/T10/T16，但使用符合本地单位、深度配准规范的夹具。
    path,t,_,_,_=bundle(tmp_path/'case',frame_count=3)
    start(path,t)
    assert vision.get_target().valid_count==1
    with pytest.raises(vision.NoTarget,match='skip_exhausted'): vision.get_target(1)
    assert vision.get_target().ripe
    with pytest.raises(vision.NoTarget,match='replay_exhausted'): vision.get_target()


@pytest.mark.parametrize('ignore',[-1,True,False,1.,'1',None])
def test_ignore_program_errors(tmp_path,ignore,isolated):
    path,t,*_=bundle(tmp_path/'case'); start(path,t)
    with pytest.raises(vision.VisionHardError,match='非负 int'): vision.get_target(ignore)
    assert isolated[-1]['outcome']=='hard_error'
    assert vision.get_target().valid_count==1  # 非法调用未消费帧


def test_lifecycle_duplicate_and_snapshot(tmp_path):
    with pytest.raises(vision.VisionHardError): vision.init()
    path,t,*_=bundle(tmp_path/'case'); vision.configure(t,path)
    t.target_bounds_m[:]=99
    with pytest.raises(vision.VisionHardError): vision.configure(t,path)
    vision.init()
    with pytest.raises(vision.VisionHardError): vision.init()
    assert vision.get_target().valid_count==1


@pytest.mark.parametrize('change',[
    lambda c:c.update(unknown=True),
    lambda c:c['source'].update(mode='unknown'),
    lambda c:c['source'].update(mode='live'),
    lambda c:c['model'].update(detection_floor=.8),
    lambda c:c['model'].update(class_unripe=['RIPE']),
    lambda c:c['geometry'].update(min_points=True),
    lambda c:c['geometry'].update(inner_scale=0),
    lambda c:c['geometry'].update(footprint_padding_m=-1),
    lambda c:c['camera'].update(max_pair_delta_s=None),
    lambda c:c['calibration'].update(intrinsics_path='missing.json'),
])
def test_bad_configs_are_hard_errors(tmp_path,change):
    path,t,c,*_=bundle(tmp_path/'case'); change(c); path.write_text(json.dumps(c))
    vision.configure(t,path)
    with pytest.raises(vision.VisionHardError): vision.init()


@pytest.mark.parametrize('kind',['empty','missing','shape','zero_depth','time','pickle','extrinsic','intrinsic','duplicate_id'])
def test_bad_resources_never_no_target(tmp_path,kind):
    path,t,c,i,T=bundle(tmp_path/'case'); root=path.parent
    if kind=='empty': (root/'manifest.json').write_text('{"schema_version":1,"frames":[]}')
    if kind=='missing': (root/'depth.npy').unlink()
    if kind=='shape': np.save(root/'depth.npy',np.ones((10,10),dtype=np.uint16))
    if kind=='zero_depth': np.save(root/'depth.npy',np.zeros((60,80),dtype=np.uint16))
    if kind=='pickle': np.save(root/'depth.npy',np.array([object()]))
    if kind=='time':
        m=json.loads((root/'manifest.json').read_text()); m['frames'][0]['depth_time_s']=1.; (root/'manifest.json').write_text(json.dumps(m))
    if kind=='duplicate_id':
        m=json.loads((root/'manifest.json').read_text()); m['frames']*=2; (root/'manifest.json').write_text(json.dumps(m))
    if kind=='extrinsic':
        e=json.loads((root/'extrinsic.json').read_text()); e['matrix'][0][0]=2.; (root/'extrinsic.json').write_text(json.dumps(e))
    if kind=='intrinsic':
        i['rgb']['fx']=None; (root/'intrinsics.json').write_text(json.dumps(i))
    vision.configure(t,path)
    with pytest.raises(vision.VisionHardError): vision.init()


def test_no_detection_and_whole_frame_failure(tmp_path,isolated):
    path,t,*_=bundle(tmp_path/'case',detections=[]); start(path,t)
    with pytest.raises(vision.NoTarget,match='no_detection'): vision.get_target()
    assert isolated[-1]['outcome']=='no_target'
    vision._reset_for_tests()
    path,t,*_=bundle(tmp_path/'bad',frame_count=2); start(path,t)
    np.save(path.parent/'depth.npy',np.zeros((60,80),dtype=np.uint16))
    with pytest.raises(vision.VisionHardError): vision.get_target()
    assert isolated[-1]['outcome']=='hard_error'


def test_unmeasurable_neighbor_is_preserved(tmp_path,isolated):
    detections=[{'class':'RIPE','conf':.9,'corners_px':[[36,27],[44,27],[44,32],[36,32]]},
                {'class':'UNRIPE','conf':.1,'corners_px':[[1,1],[10,1],[10,10],[1,10]]}]
    path,t,*_=bundle(tmp_path/'case',detections=detections); start(path,t)
    with pytest.raises(vision.NoTarget,match='all_filtered'): vision.get_target()
    reasons=[c['rejection_reason'] for c in isolated[-1]['candidates']]
    assert reasons==['neighbor_unverifiable','no_depth']


def test_base_geometry_rotation_and_half_height(tmp_path):
    path,t,c,i,T=bundle(tmp_path/'case')
    depth=np.load(path.parent/'depth.npy'); xyz=align_depth(depth,i)
    a=.4; R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    T[:3,:3]=R@T[:3,:3]; T[:2,3]=[.03,.04]
    dets=json.loads((path.parent/'detections.json').read_text())
    candidates,_=measure_geometry(dets,xyz,t,T,c['geometry']); target=candidates[0]
    assert target['yaw_deg']==pytest.approx(np.degrees(a),abs=1e-4)
    assert target['position'][2]==pytest.approx(.01)
    assert target['z_top']==pytest.approx(.02)
    np.testing.assert_allclose(target['position'][:2], R[:2,:2]@[0,.002]+[.03,.04],atol=1e-6)


def test_d2c_distortion_translation_and_nearest_z(tmp_path):
    _,_,_,i,_=bundle(tmp_path/'case')
    i['rgb']['dist']=[.1,-.02,.003,-.002,.005]; i['depth']['dist']=[-.1,.02,-.001,.003,0]
    i['depth_to_rgb'][0][3]=.012
    depth=np.zeros((60,80),dtype=np.uint16); depth[20,20]=400
    xyz=align_depth(depth,i)
    K=np.array([[100,0,40],[0,120,30],[0,0,1]],dtype=float)
    ray=cv2.undistortPoints(np.array([[[20.,20.]]]),K,np.array(i['depth']['dist'])).reshape(2)
    expected=np.r_[ray*.4,.4]+[.012,0,0]
    uv,_=cv2.projectPoints(expected[None],np.zeros(3),np.zeros(3),K,np.array(i['rgb']['dist']))
    u,v=np.rint(uv).astype(int).reshape(2)
    np.testing.assert_allclose(xyz[v,u],expected,atol=1e-10)
    assert np.isfinite(xyz).all(axis=2).sum()==1
    i['rgb']['fx']=i['rgb']['fy']=1.; i['rgb']['dist']=[0]*5
    depth[20,21]=600
    xyz=align_depth(depth,i)
    finite=xyz[np.isfinite(xyz).all(axis=2)]
    assert len(finite)==1 and finite[0,2]==pytest.approx(.4)


def test_ambiguous_components_and_overhead(tmp_path,isolated):
    depth=np.full((60,80),500,dtype=np.uint16)
    depth[28:32,37:40]=480; depth[28:32,42:44]=480
    path,t,*_=bundle(tmp_path/'case',depth=depth); start(path,t)
    with pytest.raises(vision.NoTarget,match='all_filtered'): vision.get_target()
    assert isolated[-1]['candidates'][0]['rejection_reason']=='ambiguous_geometry'
    vision._reset_for_tests()
    depth[28:32,37:44]=480; depth[34:36,39:42]=450
    path,t,*_=bundle(tmp_path/'overhead',depth=depth); start(path,t)
    with pytest.raises(vision.NoTarget): vision.get_target()
    assert isolated[-1]['candidates'][0]['rejection_reason']=='overhead_blocked'


def test_z_bounds_maturity_and_padding(tmp_path,isolated):
    path,t,c,*_=bundle(tmp_path/'case'); c['model']['class_ripe']=['red']; path.write_text(json.dumps(c))
    d=json.loads((path.parent/'detections.json').read_text()); d[0]['class']='red'; (path.parent/'detections.json').write_text(json.dumps(d))
    start(path,t); assert vision.get_target().ripe is True
    vision._reset_for_tests()
    t.target_bounds_m[1,2]=.005; start(path,t)
    with pytest.raises(vision.NoTarget): vision.get_target()
    assert isolated[-1]['candidates'][0]['rejection_reason']=='out_of_bounds'


def test_deterministic_ranking_and_filtered_neighbor(tmp_path):
    _,t,c,_,_=bundle(tmp_path/'case')
    from qingyun.grabbing._vision_pipeline import _rectangle
    def candidate(x, cls='RIPE',conf=.9):
        return dict(source_detection_id=x,rejection_reason=None,position=np.array([x,0,.01]),yaw_deg=0.,length_m=.02,width_m=.01,z_top=.02,
                    footprint=_rectangle([x,0],.02,.01,0),component_ids=np.array([],dtype=int),**{'class':cls,'conf':conf})
    base=np.empty((0,0,3))
    a,b=candidate(-.05),candidate(.05)
    ranks=filter_and_rank([b,a],base,t,c['model'])
    assert [v['position'][0] for v in ranks]==[-.05,.05]
    bad=candidate(-.065,conf=.1)
    ranks=filter_and_rank([a,b,bad],base,t,c['model'])
    assert bad['rejection_reason']=='low_confidence' and a['rejection_reason']=='too_close'


def test_rknn_adapter_undoes_resize_and_keeps_low_confidence_neighbors(tmp_path):
    path,t,c,*_=bundle(tmp_path/'case')
    c['model'].update(backend='rknn',input_size=40,crop_size=40,class_names=['RIPE','UNRIPE'])
    p=Pipeline(t,c,path)
    class FakeRKNN:
        def inference(self,inputs):
            assert inputs[0].shape==(1,40,40,3)
            return [np.array([[[20],[20],[10],[6],[.2],[.1],[0.]]])]
    p.rknn=FakeRKNN(); p.class_names=c['model']['class_names']
    dets=p.infer(np.zeros((60,80,3),dtype=np.uint8),None)
    assert len(dets)==1 and dets[0]['conf']==pytest.approx(.2)
    np.testing.assert_allclose(np.mean(dets[0]['corners_px'],axis=0),[40,30])
    assert np.ptp(np.array(dets[0]['corners_px'])[:,0])==pytest.approx(10)


def test_two_tasks_have_new_frames_and_empty_scans(tmp_path,isolated):
    path,t,*_=bundle(tmp_path/'case',frame_count=2); start(path,t)
    with runlog.context(task_instance_id='one',scan_id=1): vision.get_target()
    with runlog.context(task_instance_id='two',scan_id=1): vision.get_target()
    scans=[r for r in isolated if r['kind']=='vision_scan']
    assert [s['frame_id'] for s in scans]==['0','1']
    assert [s['task_instance_id'] for s in scans]==['one','two']


def test_live_openni_timeout_is_seconds_and_never_reads_after_timeout(tmp_path,monkeypatch):
    path,t,c,i,T=bundle(tmp_path/'case')
    p=Pipeline(t,c,path); p.intrinsics=i
    from qingyun.grabbing import _vision_pipeline as module
    ticks=iter([10.,10.2])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(ticks))
    class Cap:
        def read(self): return True,np.zeros((60,80,3),dtype=np.uint8)
    class Stream:
        def read_frame(self): raise AssertionError('等待超时后禁止读取')
    class OpenNI:
        def wait_for_any_stream(self,streams,timeout):
            assert streams==[p.ds]
            assert timeout==pytest.approx(.8)  # Python 绑定转换秒至 ms；不可先乘 1000
            return None
    p.cap=Cap();p.ds=Stream();p.openni=OpenNI()
    with pytest.raises(ValueError,match='等待超时'): p.read_live()
