"""原生 fixture/replay → 生产 Plan → 真实运动栈 + MockMotor 集成。"""
from dataclasses import replace
import json

import pytest

import main as production_main
from configs.app_config import load_app_config, MOCK_APP_PATH
from qingyun import runlog, shutdown
from qingyun.grabbing import vision
from scripts import run_mock


def test_native_replay_drives_real_mock_grasp_and_task_completion(tmp_path, monkeypatch):
    vision._reset_for_tests(); runlog.reset_for_tests(); shutdown.reset_for_tests(); run_mock.reset_stubs_for_tests()
    rows=[]
    cfg=replace(load_app_config(MOCK_APP_PATH,entry='mock'),log_dir=tmp_path/'logs')
    try:
        cfg,params,motor,arm=run_mock.assemble(cfg=cfg)
        runlog.set_sink(lambda line:rows.append(json.loads(line)))
        utterances=iter(['分拣草莓'])
        def listen():
            try: return next(utterances)
            except StopIteration: raise KeyboardInterrupt
        monkeypatch.setattr(production_main.asr,'listen_and_transcribe',listen)
        with pytest.raises(SystemExit) as exc:
            production_main.run_session(cfg,arm,registry=run_mock.MOCK_REGISTRY)
        assert exc.value.code==1  # 第二轮等语音时 Ctrl-C 正常退出契约
        done=next(row for row in rows if row['kind']=='task_done')
        assert done['reason']=='NO_TARGET_STREAK'
        scans=[row for row in rows if row['kind']=='vision_scan']
        assert len(scans)==4
        assert scans[0]['outcome']=='target'
        assert [row['reason'] for row in scans[1:]]==['no_detection']*3
        assert len({row['frame_id'] for row in scans})==4
        assert len({row['task_instance_id'] for row in scans})==1
        assert motor.command_count>0
        # 生产运动已经完成抓取、放置、返回，不能拿 FakeArm 结果冒充。
        attempts=[row for row in rows if row['kind']=='grasp_result']
        assert len(attempts)==1 and attempts[0]['status']=='SUCCESS'
    finally:
        vision._reset_for_tests(); runlog.reset_for_tests(); shutdown.reset_for_tests(); run_mock.reset_stubs_for_tests()
