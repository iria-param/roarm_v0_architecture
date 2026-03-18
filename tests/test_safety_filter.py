"""
tests/test_safety_filter.py — Unit tests for safety_filter.py
"""

import math
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety_filter import (
    clip_joint,
    validate_xyz,
    validate_speed,
    validate_gripper_force,
    validate_action,
    validate_sequence,
    SafetyViolation,
    DOF_LIMITS,
    SOFT_MARGIN,
    WORKSPACE_MM,
)


# ---------------------------------------------------------------------------
# clip_joint
# ---------------------------------------------------------------------------
class TestClipJoint:
    def test_within_range_unchanged(self):
        assert clip_joint("T", 180.0) == 180.0
        assert clip_joint("B", 45.0)  == 45.0
        assert clip_joint("A", -30.0) == -30.0
        assert clip_joint("G", 50.0)  == 50.0

    def test_hard_limit_lower(self):
        # Use hard limits (use_soft_margin=False)
        lo, _ = DOF_LIMITS["B"]
        result = clip_joint("B", lo - 10, use_soft_margin=False)
        assert result == lo

    def test_hard_limit_upper(self):
        _, hi = DOF_LIMITS["B"]
        result = clip_joint("B", hi + 10, use_soft_margin=False)
        assert result == hi

    def test_soft_margin_lower_shoulder(self):
        # Soft margin for B is [-30, 120] per SOFT_MARGIN
        result = clip_joint("B", -45.0, use_soft_margin=True)
        assert result == SOFT_MARGIN["B"][0]  # -30

    def test_soft_margin_upper_shoulder(self):
        result = clip_joint("B", 135.0, use_soft_margin=True)
        assert result == SOFT_MARGIN["B"][1]  # 120

    def test_all_joints_accept_midpoint(self):
        for joint, (lo, hi) in DOF_LIMITS.items():
            mid = (lo + hi) / 2
            assert clip_joint(joint, mid, use_soft_margin=False) == mid

    def test_base_full_rotation_both_ends(self):
        assert clip_joint("T", 0.0,   use_soft_margin=False) == 0.0
        assert clip_joint("T", 360.0, use_soft_margin=False) == 360.0

    def test_gripper_clamped_to_100(self):
        assert clip_joint("G", 150.0, use_soft_margin=False) == 100.0

    def test_gripper_clamped_to_0(self):
        assert clip_joint("G", -10.0, use_soft_margin=False) == 0.0


# ---------------------------------------------------------------------------
# validate_xyz
# ---------------------------------------------------------------------------
class TestValidateXYZ:
    def test_valid_coords_pass_through(self):
        cx, cy, cz = validate_xyz(100, 50, 200)
        assert (cx, cy, cz) == (100.0, 50.0, 200.0)

    def test_x_clipped_positive(self):
        cx, _, _ = validate_xyz(600, 0, 100)
        assert cx == WORKSPACE_MM["x"][1]  # 450

    def test_x_clipped_negative(self):
        cx, _, _ = validate_xyz(-600, 0, 100)
        assert cx == WORKSPACE_MM["x"][0]  # -450

    def test_z_floor(self):
        _, _, cz = validate_xyz(0, 0, -50)
        assert cz == WORKSPACE_MM["z"][0]  # 0

    def test_z_ceiling(self):
        _, _, cz = validate_xyz(0, 0, 700)
        assert cz == WORKSPACE_MM["z"][1]  # 600

    def test_origin_valid(self):
        cx, cy, cz = validate_xyz(0, 0, 0)
        assert (cx, cy, cz) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# validate_speed
# ---------------------------------------------------------------------------
class TestValidateSpeed:
    def test_normal_speed(self):
        assert validate_speed(80) == 80

    def test_too_low_clamped(self):
        assert validate_speed(5) == 20

    def test_too_high_clamped(self):
        assert validate_speed(200) == 150

    def test_boundary_values(self):
        assert validate_speed(20)  == 20
        assert validate_speed(150) == 150


# ---------------------------------------------------------------------------
# validate_gripper_force
# ---------------------------------------------------------------------------
class TestValidateGripperForce:
    def test_valid_force(self):
        assert validate_gripper_force(300) == 300

    def test_below_minimum(self):
        assert validate_gripper_force(50) == 100

    def test_above_maximum(self):
        assert validate_gripper_force(600) == 500


# ---------------------------------------------------------------------------
# validate_action
# ---------------------------------------------------------------------------
class TestValidateAction:
    def test_home_passthrough(self):
        action = {"fn": "home", "args": {}}
        result = validate_action(action)
        assert result["fn"] == "home"

    def test_verify_passthrough(self):
        action = {"fn": "verify", "args": {}}
        result = validate_action(action)
        assert result["fn"] == "verify"

    def test_move_joint_clips_angle(self):
        action = {"fn": "move_joint", "args": {"joint": "B", "angle_deg": 150.0, "speed": 80}}
        result = validate_action(action)
        assert result["args"]["angle_deg"] <= SOFT_MARGIN["B"][1]

    def test_move_joint_clips_speed(self):
        action = {"fn": "move_joint", "args": {"joint": "T", "angle_deg": 90, "speed": 250}}
        result = validate_action(action)
        assert result["args"]["speed"] == 150

    def test_move_joint_unknown_joint_raises(self):
        action = {"fn": "move_joint", "args": {"joint": "X", "angle_deg": 45}}
        with pytest.raises(SafetyViolation):
            validate_action(action)

    def test_move_xyz_clips_all_axes(self):
        action = {"fn": "move_xyz", "args": {"x_mm": 999, "y_mm": -999, "z_mm": 999}}
        result = validate_action(action)
        assert result["args"]["x_mm"] == 450.0
        assert result["args"]["y_mm"] == -450.0
        assert result["args"]["z_mm"] == 600.0

    def test_set_gripper_clips_force(self):
        action = {"fn": "set_gripper", "args": {"open_percent": 50, "force_limit_ma": 999}}
        result = validate_action(action)
        assert result["args"]["force_limit_ma"] == 500

    def test_wait_clamps_ms(self):
        action = {"fn": "wait", "args": {"ms": 99999}}
        result = validate_action(action)
        assert result["args"]["ms"] == 10000

    def test_wait_zero_ms(self):
        action = {"fn": "wait", "args": {"ms": 0}}
        result = validate_action(action)
        assert result["args"]["ms"] == 0


# ---------------------------------------------------------------------------
# validate_sequence
# ---------------------------------------------------------------------------
class TestValidateSequence:
    def test_full_valid_sequence(self):
        actions = [
            {"fn": "home",       "args": {}},
            {"fn": "move_joint", "args": {"joint": "T", "angle_deg": 90, "speed": 100}},
            {"fn": "move_xyz",   "args": {"x_mm": 200, "y_mm": 0, "z_mm": 150}},
            {"fn": "set_gripper","args": {"open_percent": 100, "force_limit_ma": 300}},
            {"fn": "wait",       "args": {"ms": 500}},
            {"fn": "verify",     "args": {}},
        ]
        result = validate_sequence(actions)
        assert len(result) == 6

    def test_invalid_joint_propagates(self):
        actions = [{"fn": "move_joint", "args": {"joint": "Z", "angle_deg": 45}}]
        with pytest.raises(SafetyViolation):
            validate_sequence(actions)

    def test_empty_sequence(self):
        assert validate_sequence([]) == []
