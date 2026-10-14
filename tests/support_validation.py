"""仅供离线测试构造完整报告结构，不能作为真实标定证据。"""
from configs.motion_params import CALIBRATION_STAGES


def complete_report_stages(evidence_type="physical"):
    return {stage: {"passed": True, "checks": [
        {"name": "synthetic test fixture", "passed": True,
         "evidence_type": evidence_type, "evidence_refs": ["synthetic-only"]}
    ]} for stage in CALIBRATION_STAGES}
