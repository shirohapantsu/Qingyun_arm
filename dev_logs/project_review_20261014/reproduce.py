"""本次项目审查的离线复现；仅使用替身电机、名义 RGB-D 和临时标定包。

用已安装 requirements.txt 的 Python 执行：
    python dev_logs/project_review_20261014/reproduce.py results.json
输出记录当前缺陷；不连接设备、不发送云请求、不修改运行代码或真实标定。
"""
import sys, pathlib, json, copy, tempfile, argparse, contextlib, io
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import replace
from configs.common_interface import VisionThresholds, JointFeedback
from configs.motion_params import load_motion_params, parameter_sha256
from qingyun.grabbing.trajectory import make_joint_segment
from qingyun.grabbing.safety import check_step_velocity_acceleration
from qingyun.grabbing.arm_control import ArmController, PlanStep
from qingyun.grabbing.executor import SyncExecutor
from qingyun.grabbing.kinematics_ext import StoppedByRequest, IkSolver, PoseError, IkNotConverged
from qingyun.grabbing._vision_pipeline import align_depth, measure_geometry, filter_and_rank
from tests.mock_motor import MockMotorController, SimTime
from scripts import calibrate, tool_tcp_ref
p=load_motion_params(ROOT/'calibration/SIM_ROBOT/simulation/profile.json',mode='mock')
results=[]
def record(name, fn):
 try:
  value=fn(); results.append({'check':name,'result':value})
 except Exception as e: results.append({'check':name,'unexpected':repr(e)})
def tiny():
 q=p.workspace.home_joints_deg.copy(); q[0]+=0.0001
 s=make_joint_segment('tiny',p.workspace.home_joints_deg,q,50.,p)
 try: check_step_velocity_acceleration(s.joints_deg,1/30,p,'tiny')
 except ValueError as e:return dict(nodes=s.nodes,error=str(e))
 raise AssertionError('did not reproduce')
record('two_node_acceleration',tiny)
def prim():
 st=SimTime();m=MockMotorController(p,st,gripper_pct=10.); calls=[]
 orig=m.hold_current
 def hold():calls.append('hold');return orig()
 m.hold_current=hold
 a=ArmController(m,p,lambda:m.command_count>=2,clock=st.monotonic,sleep=st.sleep,clock_ns=st.monotonic_ns)
 try:a.open_gripper()
 except StoppedByRequest:pass
 else:raise AssertionError('did not stop')
 return dict(latched=a._fault_latched,holds=len(calls),sent=m.command_count,goal=m.g_cmd,actual=m.g)
record('primitive_stops_without_hold',prim)
def timing():
 st=SimTime(); times=[]
 class Motor:
  def __init__(self):self.n=0;self.q=p.workspace.home_joints_deg.copy()
  def get_feedback(self):
   self.n+=1; start=st.monotonic_ns()
   if self.n==1:st.sleep(.02)
   return JointFeedback(self.q.copy(),np.zeros(5),np.zeros(5),50.,0.,0.,start,st.monotonic_ns(),self.n)
  def send_action(self,q,g):self.q=q.copy();times.append(st.monotonic())
 m=Motor();ex=SyncExecutor(m,p,clock=st.monotonic,sleep=st.sleep,clock_ns=st.monotonic_ns)
 q=p.workspace.home_joints_deg.copy();q[0]+=.1
 s=make_joint_segment('timing',p.workspace.home_joints_deg,q,50,p,min_duration_s=.1)
 ex.play_segment(s)
 return dict(tick_s=1/30,first_intervals_s=np.diff(times).tolist(),min_interval_s=min(np.diff(times)))
record('command_intervals_shorter_than_tick',timing)
def rebound():
 st=SimTime();m=MockMotorController(p,st);a=ArmController(m,p,clock=st.monotonic,sleep=st.sleep,clock_ns=st.monotonic_ns)
 q0=p.workspace.home_joints_deg.copy();w1=q0.copy();w1[0]+=1;w2=q0.copy();w2[0]+=2;goal=q0.copy();goal[0]+=3
 current=q0.copy();current[0]+=10;played=[]
 def build(q):return [make_joint_segment('route'+str(i),x,y,50,p) for i,(x,y) in enumerate(zip([q,w1,w2],[w1,w2,goal]))]
 def play(s):
  nonlocal current
  played.append({'name':s.name,'from':float(s.start_joints[0]),'to':float(s.end_joints[0])});current=s.end_joints.copy()
 a._measured_joints=lambda:current.copy();a._validate=lambda *args,**kwargs:None;a.executor.play_segment=play
 step=PlanStep('route','APPROACH',build=build)
 old=build(q0)
 for s in old:a._play_with_offset_guard(s,step)
 return dict(played=played,expected_final=goal[0])
record('whole_route_replayed_after_offset',rebound)
def height():
 t=VisionThresholds(p.workspace.target_bounds_m,p.grasp.object_envelope_m,.003,.08,0.,.001)
 xyz=np.full((20,20,3),np.nan)
 yy,xx=np.indices((10,10));xyz[5:15,5:15]=np.stack((.325+xx*.003,-.0135+yy*.003,np.full((10,10),.055)),axis=-1)
 det=[{'class':'RIPE','conf':.9,'corners_px':[[4,4],[16,4],[16,16],[4,16]]}]
 geom={'min_points':3,'inner_scale':.8,'depth_neighbor_jump_m':.02,'footprint_padding_m':0.}
 cs,base=measure_geometry(det,xyz,t,np.eye(4),geom)
 valid=filter_and_rank(cs,base,t,{'class_ripe':['RIPE'],'class_unripe':['UNRIPE'],'conf_threshold':.5})
 return dict(valid_count=len(valid),height_m=cs[0]['z_top'],envelope_height_m=t.object_envelope_m[2],position=cs[0]['position'].tolist())
record('oversize_height_accepted',height)
def sparse():
 intr={'rgb':{'width':1280,'height':720,'fx':975.,'fy':975.,'cx':640.,'cy':360.,'dist':[0.]*5},'depth':{'width':640,'height':480,'fx':570.,'fy':570.,'cx':320.,'cy':240.,'dist':[0.]*5},'depth_to_rgb':np.eye(4).tolist(),'depth_unit_m':.001}
 depth=np.zeros((480,640),np.uint16);depth[230:250,310:330]=500
 xyz=align_depth(depth,intr)
 t=VisionThresholds([[-1,-1,0],[1,1,1]],[1,1,1],.003,.08,0.,.001)
 geom={'min_points':32,'inner_scale':.8,'depth_neighbor_jump_m':.02,'footprint_padding_m':.01}
 det=[{'class':'RIPE','conf':.9,'corners_px':[[620,340],[660,340],[660,380],[620,380]]}]
 cs,_=measure_geometry(det,xyz,t,np.eye(4),geom)
 from qingyun.grabbing._vision_pipeline import _components
 mask=np.isfinite(xyz).all(axis=2); cc=_components(mask,xyz,.02)
 return dict(valid_depth_points=int(mask.sum()),max_component_points=max(map(len,cc)),required_points=32,rejection=cs[0]['rejection_reason'])
record('live_resolution_sparse_components',sparse)
def degrees():
 q=np.array([30.,-60.,60.,0.,0.]); kin=tool_tcp_ref._kin();correct=np.asarray(kin.forward_kinematics(q))[:3,:3]@tool_tcp_ref.R_E_TCP[:,0];actual=tool_tcp_ref.closing_dir_B(q)
 angle=float(np.degrees(np.arccos(np.clip(actual@correct/(np.linalg.norm(actual)*np.linalg.norm(correct)),-1,1))))
 return dict(correct=correct.tolist(),actual=actual.tolist(),axis_error_deg=angle)
record('tool_reference_double_radians',degrees)
def stale_report():
 with tempfile.TemporaryDirectory(prefix='qingyun-review-') as directory:
  d=pathlib.Path(directory);raw=json.loads((ROOT/'calibration/SIM_ROBOT/simulation/profile.json').read_text());raw['status']='draft';raw['model']['urdf_path']=str(p.urdf_path_resolved());raw['model']['urdf_sha256']=calibrate.sha256_file(p.urdf_path_resolved());raw['model']['motor_calibration_path']='motor_calibration.json';(d/'motor_calibration.json').write_bytes(p.motor_calibration_path_resolved().read_bytes());raw['model']['motor_calibration_sha256']=calibrate.sha256_file(d/'motor_calibration.json');old_hash=parameter_sha256(raw)
  report={'profile_id':raw['profile_id'],'parameter_sha256':old_hash,'operator':'review','operator_reviewed':True,'overall_pass':True,'stages':{s:{'passed':True,'checks':[{'passed':True,'evidence_type':'physical','name':'previous_measurement'}]} for s in calibrate.STAGES}}
  raw['motor']['current_ma_per_raw']=[x*10 for x in raw['motor']['current_ma_per_raw']]
  path=d/'profile.json';path.write_text(json.dumps(raw));rp=d/'report.json';rp.write_text(json.dumps(report))
  rec=[calibrate.Record('workcell','run','meta',0,{'record_type':'metadata','source':'hardware'})]+[calibrate.Record('workcell','run',f's{i}',i+1,{'record_type':'sample','entity_id':'table','point_B_m':[0.,0.,0.]}) for i in range(9)]
  ip=d/'input.jsonl';calibrate.append_records(ip,rec)
  with contextlib.redirect_stdout(io.StringIO()):
   calibrate.cmd_validate(argparse.Namespace(profile=path,input=ip,stage='workcell',report=rp,operator='review',operator_reviewed=True))
   calibrate.cmd_promote(argparse.Namespace(profile=path,report=rp,output=d/'verified.json'))
  new=json.loads(rp.read_text());out=json.loads((d/'verified.json').read_text())
  return dict(parameter_changed=old_hash!=new['parameter_sha256'],all_pass=new['overall_pass'],old_motor_checks_retained=new['stages']['motor']==report['stages']['motor'],promoted_status=out['status'],real_reload=load_motion_params(d/'verified.json',mode='real').status)
record('old_stage_report_rebound_and_promoted',stale_report)
def independent():
 raw=json.loads((ROOT/'calibration/SIM_ROBOT/simulation/profile.json').read_text())
 audit={'tool':{'residuals':{'rotation_orthonormality_err':0.,'translation_opening_count':3}}}
 rec=[calibrate.Record('tool','run','meta',0,{'record_type':'metadata','source':'hardware'}),calibrate.Record('tool','run','s',1,{'record_type':'sample','q_urdf_deg':[0]*5,'tcp_reference_B_m':[100.,100.,100.]})]
 checks=calibrate._stage_checks('tool',raw,rec,'0'*64,{'s'},audit)
 return dict(observed_tcp=[100]*3,checks=[{'name':c['name'],'passed':c['passed'],'measured':c['measured']} for c in checks],all_pass=all(c['passed'] for c in checks))
record('independent_tool_measurement_ignored',independent)
def config_invalid():
 raw=json.loads((ROOT/'calibration/SIM_ROBOT/simulation/profile.json').read_text());raw['model']['urdf_path']=str(p.urdf_path_resolved());raw['motion']['max_velocity_deg_s']=[0.]*5;raw['grasp']['object_envelope_m']=[-.1,.04,.04]
 with tempfile.TemporaryDirectory() as directory:
  path=pathlib.Path(directory)/'profile.json';path.write_text(json.dumps(raw));loaded=load_motion_params(path,mode='mock')
  return dict(velocity=loaded.motion.max_velocity_deg_s.tolist(),envelope=loaded.grasp.object_envelope_m.tolist())
record('nonpositive_physical_parameters_accepted',config_invalid)
def reset_moving():
 st=SimTime();m=MockMotorController(p,st);a=ArmController(m,p,clock=st.monotonic,sleep=st.sleep,clock_ns=st.monotonic_ns)
 q=p.workspace.home_joints_deg.copy();fb=JointFeedback(q,np.full(5,60.),np.zeros(5),50.,50.,0.,st.monotonic_ns(),st.monotonic_ns(),1)
 a._fault_latched=True;a.executor.state.last_submitted_deg=q;a.executor.read_feedback=lambda:fb
 a.reset_fault()
 return dict(joint_speeds_deg_s=fb.speeds_deg_s.tolist(),gripper_speed_pct_s=fb.gripper_speed_pct_s,latch_cleared=not a._fault_latched)
record('reset_accepts_moving_robot',reset_moving)
def ik_best():
 pp=replace(p,ik=replace(p.ik,max_iterations=1,seed_joints_deg=np.ones((1,5))))
 class Model:
  limit_lower=np.full(5,-180.);limit_upper=np.full(5,180.)
  def e_from_tcp(self,t,g):return t
  def single_solver_call(self,q,*args):return q
 solver=IkSolver(Model(),pp)
 def evaluate(q,g,t):
  return (PoseError(.9*pp.ik.position_tol_m,.9*pp.ik.tilt_tol_deg,.9*pp.ik.yaw_tol_deg,False) if q[0]==0 else PoseError(0.,0.,1.1*pp.ik.yaw_tol_deg,False))
 solver._evaluate=evaluate
 try:solver.solve(np.eye(4),50.,np.zeros(5))
 except IkNotConverged as e:return dict(valid_candidate_acceptable=evaluate(np.zeros(5),50,None).acceptable(pp),outcome='IK_FAILED',reason=str(e))
 raise AssertionError('did not reproduce')
record('acceptable_ik_candidate_discarded',ik_best)
def stale_live():
 from qingyun.grabbing._vision_pipeline import Pipeline
 from unittest.mock import patch
 config={'camera':{'acquire_timeout_s':1.,'max_pair_delta_s':.03}}
 pipeline=Pipeline(None,config,ROOT/'configs/vision.json')
 camera={'width':640,'height':480,'fx':570.,'fy':570.,'cx':320.,'cy':240.,'dist':[0.]*5}
 pipeline.intrinsics={'rgb':camera,'depth':camera}
 class Cap:
  def read(self):return True,np.zeros((480,640,3),np.uint8)
 class Frame:
  timestamp=1_000_000
  def get_buffer_as_uint16(self):return np.full((480,640),500,np.uint16).tobytes()
 class Stream:
  def read_frame(self):return Frame()
 class OpenNI:
  def wait_for_any_stream(self,*args,**kwargs):return 0
 pipeline.cap=Cap();pipeline.ds=Stream();pipeline.openni=OpenNI()
 with patch('qingyun.grabbing._vision_pipeline.time.monotonic',return_value=100.):
  _,_,_,info=pipeline.read_live()
 return dict(sensor_timestamp_s=1.,recorded_rgb_time_s=info['rgb_time_s'],recorded_depth_time_s=info['depth_time_s'],accepted=True)
record('old_live_frame_recorded_as_current',stale_live)
def open_jump():
 pp=replace(p,gripper=replace(p.gripper,preopen_pct=37.65,open_speed_pct_s=20.))
 st=SimTime();m=MockMotorController(pp,st,gripper_pct=55.)
 a=ArmController(m,pp,lambda:m.command_count>=1,clock=st.monotonic,sleep=st.sleep,clock_ns=st.monotonic_ns)
 try:a.open_gripper()
 except StoppedByRequest:pass
 return dict(initial_pct=55.,first_goal_pct=m.g_cmd,command_jump_pct=55.-m.g_cmd,allowed_step_pct=pp.gripper.open_speed_pct_s/pp.timing.fps)
record('open_gripper_closes_in_one_large_step',open_jump)
predicates = {
 'two_node_acceleration': lambda r:r['nodes']==2 and 'zero-size' in r['error'],
 'primitive_stops_without_hold': lambda r:r['latched'] and r['holds']==0 and r['goal']!=r['actual'],
 'command_intervals_shorter_than_tick': lambda r:r['min_interval_s']<r['tick_s']*.9,
 'whole_route_replayed_after_offset': lambda r:len(r['played'])>3,
 'oversize_height_accepted': lambda r:r['valid_count']==1 and r['height_m']>r['envelope_height_m'],
 'live_resolution_sparse_components': lambda r:r['valid_depth_points']>=r['required_points'] and r['max_component_points']<r['required_points'] and r['rejection']=='no_depth',
 'tool_reference_double_radians': lambda r:r['axis_error_deg']>1.,
 'old_stage_report_rebound_and_promoted': lambda r:r['parameter_changed'] and r['all_pass'] and r['old_motor_checks_retained'] and r['real_reload']=='verified',
 'independent_tool_measurement_ignored': lambda r:r['all_pass'] and r['observed_tcp']==[100]*3,
 'nonpositive_physical_parameters_accepted': lambda r:0 in r['velocity'] and min(r['envelope'])<0,
 'reset_accepts_moving_robot': lambda r:r['latch_cleared'] and max(r['joint_speeds_deg_s'])>0,
 'acceptable_ik_candidate_discarded': lambda r:r['valid_candidate_acceptable'] and r['outcome']=='IK_FAILED',
 'old_live_frame_recorded_as_current': lambda r:r['accepted'] and r['recorded_depth_time_s']!=r['sensor_timestamp_s'],
 'open_gripper_closes_in_one_large_step': lambda r:r['command_jump_pct']>r['allowed_step_pct'],
}
for entry in results:
 entry['bug_reproduced'] = 'unexpected' not in entry and bool(predicates[entry['check']](entry['result']))
output=json.dumps(results,ensure_ascii=False,indent=2)
if len(sys.argv)>1:pathlib.Path(sys.argv[1]).write_text(output+'\n',encoding='utf-8')
print(output)

raise SystemExit(0 if all(entry['bug_reproduced'] for entry in results) else 1)
