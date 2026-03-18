"""
tests/test_arm_driver.py — Unit tests for arm_driver.py (mock HTTP)
"""

import math
import time
import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arm_driver import ArmDriver, ArmConnectionError, ArmCommandError
import requests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def driver():
    """Live driver in dry-run mode — no actual HTTP calls."""
    return ArmDriver(ip="192.168.4.1", dry_run=True)


@pytest.fixture
def live_driver_mock():
    """Driver with requests.post patched."""
    with patch("arm_driver.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        drv = ArmDriver(ip="192.168.4.1", dry_run=False, timeout=1)
        yield drv, mock_post


# ---------------------------------------------------------------------------
# send_json
# ---------------------------------------------------------------------------
class TestSendJson:
    def test_dry_run_does_not_call_requests(self):
        with patch("arm_driver.requests.post") as mock_post:
            drv = ArmDriver(dry_run=True)
            drv.send_json({"T": 100})
            mock_post.assert_not_called()

    def test_dry_run_returns_empty_dict(self):
        drv = ArmDriver(dry_run=True)
        result = drv.send_json({"T": 100})
        assert result == {}

    def test_live_posts_correct_url(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.send_json({"T": 100})
        called_url = mock_post.call_args[0][0]
        assert called_url == "http://192.168.4.1:80/js"

    def test_connection_error_raises_arm_error(self):
        with patch("arm_driver.requests.post", side_effect=requests.ConnectionError()):
            drv = ArmDriver(dry_run=False)
            with pytest.raises(ArmConnectionError):
                drv.send_json({"T": 100})

    def test_timeout_raises_arm_connection_error(self):
        with patch("arm_driver.requests.post", side_effect=requests.Timeout()):
            drv = ArmDriver(dry_run=False)
            with pytest.raises(ArmConnectionError):
                drv.send_json({"T": 100})

    def test_http_error_raises_arm_command_error(self):
        with patch("arm_driver.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.HTTPError(
                response=MagicMock(status_code=500)
            )
            mock_post.return_value = mock_resp
            drv = ArmDriver(dry_run=False)
            with pytest.raises(ArmCommandError):
                drv.send_json({"T": 100})


# ---------------------------------------------------------------------------
# Command helpers → correct ESP32 payloads
# ---------------------------------------------------------------------------
class TestCommandPayloads:
    def test_home_sends_T100(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.home()
        sent = mock_post.call_args[1]["json"]
        assert sent == {"T": 100}

    def test_move_joint_T_base(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.move_joint("T", 90.0, speed=120)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"]     == 1
        assert sent["speed"] == 120
        assert abs(sent["base"] - math.radians(90.0)) < 1e-4

    def test_move_joint_B_shoulder(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.move_joint("B", 45.0, speed=80)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"] == 2
        assert abs(sent["shoulder"] - math.radians(45.0)) < 1e-4

    def test_move_joint_A_elbow(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.move_joint("A", -30.0, speed=70)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"] == 3
        assert abs(sent["elbow"] - math.radians(-30.0)) < 1e-4

    def test_move_joint_G_gripper(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.move_joint("G", 45.0)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"] == 4

    def test_move_xyz_payload(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.move_xyz(200, 0, 150, speed=80)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"]     == 104
        assert sent["x"]     == 200.0
        assert sent["y"]     == 0.0
        assert sent["z"]     == 150.0
        assert sent["speed"] == 80

    def test_set_gripper_100_percent_open(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.set_gripper(open_percent=100, force_limit_ma=300)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"]      == 121
        assert sent["cmd"]    == 0     # 100% open → angle 0
        assert sent["torque"] == 300

    def test_set_gripper_0_percent_closed(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.set_gripper(open_percent=0, force_limit_ma=150)
        sent = mock_post.call_args[1]["json"]
        assert sent["cmd"]    == 90    # 0% open → angle 90 (fully closed)
        assert sent["torque"] == 150

    def test_set_torque_threshold(self, live_driver_mock):
        drv, mock_post = live_driver_mock
        drv.set_torque_threshold("B", 80)
        sent = mock_post.call_args[1]["json"]
        assert sent["T"]         == 114
        assert sent["joint"]     == "B"
        assert sent["threshold"] == 80


# ---------------------------------------------------------------------------
# execute_action dispatcher
# ---------------------------------------------------------------------------
class TestExecuteAction:
    def test_home_action(self, driver):
        result = driver.execute_action({"fn": "home", "args": {}})
        assert result is None

    def test_verify_action_returns_VERIFY(self, driver):
        result = driver.execute_action({"fn": "verify", "args": {}})
        assert result == "VERIFY"

    def test_wait_action_dry_run_skips_sleep(self, driver):
        start = time.monotonic()
        driver.execute_action({"fn": "wait", "args": {"ms": 2000}})
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # dry_run should not sleep

    def test_unknown_fn_returns_none(self, driver):
        result = driver.execute_action({"fn": "explode", "args": {}})
        assert result is None

    def test_move_joint_action(self, driver):
        result = driver.execute_action({
            "fn": "move_joint",
            "args": {"joint": "T", "angle_deg": 45, "speed": 100}
        })
        assert result is None

    def test_set_gripper_action(self, driver):
        result = driver.execute_action({
            "fn": "set_gripper",
            "args": {"open_percent": 50, "force_limit_ma": 300}
        })
        assert result is None
