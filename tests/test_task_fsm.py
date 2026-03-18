"""
tests/test_task_fsm.py — Unit tests for task_fsm.py FSM state transitions.

All external I/O (camera, arm, Gemini) is mocked so no hardware is needed.
"""

import pytest
from unittest.mock import MagicMock, patch, call

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_fsm import TaskFSM, State, TaskFailed, TaskResult


# ---------------------------------------------------------------------------
# Minimal config for tests
# ---------------------------------------------------------------------------
MINIMAL_CFG = {
    "arm":          {"ip": "192.168.4.1", "port": 80, "timeout_s": 1},
    "camera":       {"device_index": 0, "width": 640, "height": 480, "fps": 5, "jpeg_quality": 80},
    "gemini":       {"model": "gemini-robotics-er-1.5-preview",
                     "thinking_budget_detect": 0, "thinking_budget_plan": 0,
                     "max_retries": 0},
    "anti_gravity": {"rubber_bands_fitted": True, "shoulder_base_speed": 100,
                     "shoulder_gravity_k": 0.4, "elbow_base_speed": 80,
                     "elbow_gravity_k": 0.3, "base_speed": 120, "gripper_speed": 120},
    "task":         {"pick_height_mm": 80, "carry_height_mm": 150,
                     "verify_on_complete": False,  # most tests disable auto-verify
                     "max_retries_on_verify": 0},
    "calibration":  {"origin_y": 500, "origin_x": 500, "scale_mm_per_unit": 0.5,
                     "camera_height_cm": 60, "camera_angle_deg": 45},
    "gripper":      {"default_force_ma": 300},
    "logging":      {"level": "WARNING"},
}


# ---------------------------------------------------------------------------
# Happy-path FSM fixture
# ---------------------------------------------------------------------------
SIMPLE_PLAN = {
    "task_summary":     "Pick up red cube",
    "objects_detected": [{"label": "red cube", "point": [500, 400], "confidence": 0.95}],
    "trajectory":       [],
    "action_sequence":  [
        {"fn": "home",       "args": {}},
        {"fn": "move_joint", "args": {"joint": "T", "angle_deg": 45, "speed": 100}},
        {"fn": "wait",       "args": {"ms": 100}},
    ],
    "anti_gravity_notes": "none needed",
}


def make_fsm_with_mocks(cfg=None, plan=None, verify_success=True):
    """
    Create a TaskFSM with all external components mocked.
    Returns (fsm, mock_camera, mock_driver, mock_gemini).
    """
    cfg  = cfg  or MINIMAL_CFG
    plan = plan or SIMPLE_PLAN

    # Patch constructors
    with (
        patch("task_fsm.CameraCapture") as MockCamera,
        patch("task_fsm.ArmDriver")     as MockDriver,
        patch("task_fsm.GeminiClient")  as MockGemini,
    ):
        # Camera mock
        cam_inst = MagicMock()
        cam_inst.grab_frame_b64.return_value = ("base64string", MagicMock())
        MockCamera.return_value = cam_inst

        # Driver mock
        drv_inst = MagicMock()
        drv_inst.get_joint_states.return_value = {"T": 0.0, "B": 0.0, "A": 0.0, "G": 100.0}
        drv_inst.execute_action.return_value = None
        drv_inst.home.return_value = None
        MockDriver.return_value = drv_inst

        # Gemini mock
        gem_inst = MagicMock()
        gem_inst.plan_task.return_value = plan
        gem_inst.thinking_budget_detect = 0
        gem_inst.thinking_budget_plan   = 0
        gem_inst._call_gemini.return_value = (
            '{"success": true, "note": "ok"}' if verify_success
            else '{"success": false, "note": "mismatch"}'
        )
        gem_inst._parse_json.return_value = (
            {"success": True, "note": "ok"} if verify_success
            else {"success": False, "note": "mismatch"}
        )
        MockGemini.return_value = gem_inst

        fsm = TaskFSM(config=cfg, dry_run=True)
        # Return the mocks for assertions
        return fsm, cam_inst, drv_inst, gem_inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFSMStateTransitions:
    def test_initial_state_is_idle(self):
        fsm, _, _, _ = make_fsm_with_mocks()
        assert fsm.state == State.IDLE

    def test_happy_path_returns_idle(self):
        fsm, _, _, _ = make_fsm_with_mocks()
        result = fsm.run("test task")
        assert fsm.state == State.IDLE

    def test_returns_task_result(self):
        fsm, _, _, _ = make_fsm_with_mocks()
        result = fsm.run("test task")
        assert isinstance(result, TaskResult)

    def test_task_summary_populated(self):
        fsm, _, _, _ = make_fsm_with_mocks()
        result = fsm.run("test task")
        assert result.task_summary == "Pick up red cube"

    def test_objects_detected_propagated(self):
        fsm, _, _, _ = make_fsm_with_mocks()
        result = fsm.run("test task")
        assert len(result.objects) == 1
        assert result.objects[0]["label"] == "red cube"


class TestFSMExecution:
    def test_camera_called_once_without_verify(self):
        fsm, cam, _, _ = make_fsm_with_mocks()
        fsm.run("test task")
        assert cam.grab_frame_b64.call_count == 1

    def test_gemini_plan_called_once(self):
        fsm, _, _, gem = make_fsm_with_mocks()
        fsm.run("test task")
        gem.plan_task.assert_called_once()

    def test_execute_action_called_for_each_step(self):
        fsm, _, drv, _ = make_fsm_with_mocks()
        fsm.run("test task")
        # home + move_joint + wait = 3 actions
        assert drv.execute_action.call_count == 3

    def test_home_called_after_execution(self):
        fsm, _, drv, _ = make_fsm_with_mocks()
        fsm.run("test task")
        drv.home.assert_called()

    def test_verify_action_in_sequence_triggers_verify_state(self):
        """A 'verify' fn in action_sequence should trigger re-capture."""
        plan_with_verify = {
            **SIMPLE_PLAN,
            "action_sequence": [
                {"fn": "home",   "args": {}},
                {"fn": "verify", "args": {}},
            ],
        }
        cfg_verify = {
            **MINIMAL_CFG,
            "task": {**MINIMAL_CFG["task"], "verify_on_complete": False, "max_retries_on_verify": 0},
        }

        fsm, cam, drv, gem = make_fsm_with_mocks(cfg=cfg_verify, plan=plan_with_verify)
        # execute_action returns "VERIFY" for verify fn
        drv.execute_action.side_effect = lambda a: "VERIFY" if a["fn"] == "verify" else None

        fsm.run("test task")
        # Camera should be grabbed at least twice (PERCEIVE + VERIFY)
        assert cam.grab_frame_b64.call_count >= 2


class TestFSMErrorHandling:
    def test_camera_error_raises_task_failed(self):
        from camera import CameraError
        fsm, cam, _, _ = make_fsm_with_mocks()
        cam.grab_frame_b64.side_effect = CameraError("no device")
        with pytest.raises(TaskFailed, match="Camera failure"):
            fsm.run("test task")

    def test_gemini_error_raises_task_failed(self):
        fsm, _, _, gem = make_fsm_with_mocks()
        gem.plan_task.side_effect = RuntimeError("API down")
        with pytest.raises(TaskFailed, match="planning failed"):
            fsm.run("test task")

    def test_arm_connection_error_raises_task_failed(self):
        from arm_driver import ArmConnectionError
        fsm, _, drv, _ = make_fsm_with_mocks()
        drv.execute_action.side_effect = ArmConnectionError("unreachable")
        with pytest.raises(TaskFailed, match="communication error"):
            fsm.run("test task")


class TestFSMVerify:
    def test_verify_on_complete_calls_camera_twice(self):
        cfg_v = {
            **MINIMAL_CFG,
            "task": {**MINIMAL_CFG["task"], "verify_on_complete": True},
        }
        fsm, cam, _, gem = make_fsm_with_mocks(cfg=cfg_v, verify_success=True)
        result = fsm.run("test task")
        assert cam.grab_frame_b64.call_count == 2

    def test_task_result_success_true_on_good_verify(self):
        cfg_v = {
            **MINIMAL_CFG,
            "task": {**MINIMAL_CFG["task"], "verify_on_complete": True},
        }
        fsm, _, _, _ = make_fsm_with_mocks(cfg=cfg_v, verify_success=True)
        result = fsm.run("test task")
        assert result.success is True
