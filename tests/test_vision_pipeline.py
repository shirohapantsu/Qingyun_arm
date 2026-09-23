import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from qingyun.grabbing import vision
from qingyun.grabbing.vision import VisionThresholds, NoTarget, VisionHardError

@pytest.fixture
def setup_vision():
    # Provide dummy thresholds
    thresholds = VisionThresholds(
        target_bounds_m=[[-2.0, -2.0, -0.5], [2.0, 2.0, 1.0]],
        object_envelope_m=[0.2, 0.2, 0.2],
        clearance_m=0.001,
        approach_height_m=0.1,
        table_z_m=0.0,
        table_flatness_m=0.02
    )
    
    config_path = Path(PROJECT_ROOT) / "configs" / "vision.json"
    
    # Reset singleton states for testing
    vision._is_initialized = False
    vision._replay_idx = 0
    
    vision.configure(thresholds, config_path)
    
    # Force mock mode for testing
    vision._config['source']['mode'] = 'replay'
    vision._config['source']['replay_dir'] = '../tests/fixtures/vision'
    vision._config['model']['backend'] = 'fixture'
    
    vision.init()
    return vision

def test_p3_t01_basic_detection(setup_vision):
    vis = setup_vision
    target = vis.get_target(ignore=0)
    
    assert target.ripe is True
    assert target.position.shape == (3,)

def test_p3_t16_replay_exhausted(setup_vision):
    vis = setup_vision
    # Frame 0 consumed
    vis.get_target(0)
    
    # Frame 1 should exhaust the replay manifest
    with pytest.raises(NoTarget) as exc:
        vis.get_target(0)
    assert "replay_exhausted" in str(exc.value)

def test_p3_t10_ignore_bounds(setup_vision):
    vis = setup_vision
    with pytest.raises(NoTarget) as exc:
        # We only have 1 target in the fixture, ignore=1 should throw skip_exhausted
        vis.get_target(ignore=1)
    assert "skip_exhausted" in str(exc.value)
