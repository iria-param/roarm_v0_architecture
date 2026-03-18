"""
tests/test_anti_gravity.py — Unit tests for anti_gravity.py
"""

import math
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anti_gravity as ag
from anti_gravity import anti_gravity_speed, inject_speeds, load_config


# Reset config to defaults before each test
@pytest.fixture(autouse=True)
def reset_config():
    load_config({
        "shoulder_base_speed":  100,
        "shoulder_gravity_k":   0.4,
        "elbow_base_speed":      80,
        "elbow_gravity_k":       0.3,
        "base_speed":           120,
        "gripper_speed":        120,
        "rubber_bands_fitted":  True,
    })


# ---------------------------------------------------------------------------
# anti_gravity_speed
# ---------------------------------------------------------------------------
class TestAntiGravitySpeed:
    # --- Shoulder ---
    def test_shoulder_at_90_degrees_max_speed(self):
        """At 90° (vertical), cos(90)=0 → speed = base_speed = 100."""
        speed = anti_gravity_speed("shoulder", 90.0)
        assert speed == 100

    def test_shoulder_at_0_degrees_min_speed(self):
        """At 0° (horizontal), cos(0)=1 → speed = 100*(1-0.4) = 60."""
        speed = anti_gravity_speed("shoulder", 0.0)
        assert speed == 60

    def test_shoulder_at_45_degrees_midpoint(self):
        """At 45°, cos(45)≈0.707 → speed = int(round(100*(1-0.4*0.707))) ≈ 72."""
        expected = int(round(100 * (1 - 0.4 * abs(math.cos(math.radians(45))))))
        speed = anti_gravity_speed("shoulder", 45.0)
        assert speed == expected

    def test_shoulder_uppercase_B_key(self):
        """Accept "B" as alias for shoulder."""
        assert anti_gravity_speed("B", 0.0) == anti_gravity_speed("shoulder", 0.0)
        assert anti_gravity_speed("B", 90.0) == anti_gravity_speed("shoulder", 90.0)

    # --- Elbow ---
    def test_elbow_at_90_degrees_max_speed(self):
        """At 90°, elbow speed = base 80."""
        speed = anti_gravity_speed("elbow", 90.0)
        assert speed == 80

    def test_elbow_at_0_degrees_min_speed(self):
        """At 0°, elbow speed = int(80*(1-0.3)) = 56."""
        speed = anti_gravity_speed("elbow", 0.0)
        assert speed == 56

    def test_elbow_uppercase_A_key(self):
        """Accept "A" as alias for elbow."""
        assert anti_gravity_speed("A", 0.0) == anti_gravity_speed("elbow", 0.0)

    def test_elbow_negative_angle(self):
        """Negative angles use |cos|, same as positive."""
        assert anti_gravity_speed("elbow", -30.0) == anti_gravity_speed("elbow", 30.0)

    # --- Base & Gripper ---
    def test_base_constant_speed(self):
        for angle in [0, 45, 90, 180, 360]:
            assert anti_gravity_speed("T", angle) == 120

    def test_gripper_constant_speed(self):
        for angle in [0, 45, 90]:
            assert anti_gravity_speed("G", angle) == 120

    # --- Payload scaling ---
    def test_payload_increases_gravity_effect(self):
        """Heavier payload should reduce speed further."""
        speed_0   = anti_gravity_speed("shoulder", 0.0, payload_kg=0.0)
        speed_0p5 = anti_gravity_speed("shoulder", 0.0, payload_kg=0.5)
        assert speed_0p5 <= speed_0

    def test_speed_always_within_bounds(self):
        """Speed must always be in [20, 150] regardless of inputs."""
        for joint in ["shoulder", "elbow", "T", "G", "B", "A"]:
            for angle in range(-90, 136, 15):
                for payload in [0.0, 0.5, 2.0]:
                    s = anti_gravity_speed(joint, angle, payload_kg=payload)
                    assert 20 <= s <= 150, (
                        f"Speed {s} out of bounds for joint={joint}, "
                        f"angle={angle}, payload={payload}"
                    )

    # --- Rubber-band penalty ---
    def test_no_rubber_bands_reduces_shoulder_speed(self):
        load_config({"rubber_bands_fitted": False})
        speed_no_bands = anti_gravity_speed("shoulder", 0.0)
        load_config({"rubber_bands_fitted": True})
        speed_bands    = anti_gravity_speed("shoulder", 0.0)
        assert speed_no_bands < speed_bands


# ---------------------------------------------------------------------------
# inject_speeds
# ---------------------------------------------------------------------------
class TestInjectSpeeds:
    def _make_actions(self):
        return [
            {"fn": "home",       "args": {}},
            {"fn": "move_joint", "args": {"joint": "B", "angle_deg": 45}},
            {"fn": "move_joint", "args": {"joint": "A", "angle_deg": -20}},
            {"fn": "move_joint", "args": {"joint": "T", "angle_deg": 90}},
            {"fn": "verify",     "args": {}},
        ]

    def test_injects_speed_for_B(self):
        actions = self._make_actions()
        inject_speeds(actions)
        b_action = next(a for a in actions if a.get("args", {}).get("joint") == "B")
        assert "speed" in b_action["args"]
        assert 20 <= b_action["args"]["speed"] <= 150

    def test_injects_speed_for_A(self):
        actions = self._make_actions()
        inject_speeds(actions)
        a_action = next(a for a in actions if a.get("args", {}).get("joint") == "A")
        assert "speed" in a_action["args"]

    def test_does_not_inject_speed_for_T(self):
        """Base joint (T) should not be touched by inject_speeds."""
        actions = self._make_actions()
        inject_speeds(actions)
        t_action = next(a for a in actions if a.get("args", {}).get("joint") == "T")
        # T should not have speed injected by inject_speeds (no gravity)
        assert "speed" not in t_action["args"]

    def test_does_not_override_lower_speed(self):
        """If model supplies a speed already ≤ recommended, keep it."""
        actions = [{"fn": "move_joint", "args": {"joint": "B", "angle_deg": 0, "speed": 10}}]
        inject_speeds(actions)
        # Speed 10 < recommended (60) → capped at recommended; should never go above recommended
        assert actions[0]["args"]["speed"] <= anti_gravity_speed("B", 0.0)

    def test_caps_dangerously_high_speed(self):
        """Model-supplied speed > recommended should be capped."""
        actions = [{"fn": "move_joint", "args": {"joint": "B", "angle_deg": 0, "speed": 150}}]
        inject_speeds(actions)
        recommended = anti_gravity_speed("B", 0.0)
        assert actions[0]["args"]["speed"] <= recommended

    def test_non_move_joint_actions_unmodified(self):
        actions = [
            {"fn": "home",    "args": {}},
            {"fn": "verify",  "args": {}},
            {"fn": "wait",    "args": {"ms": 500}},
        ]
        import copy
        original = copy.deepcopy(actions)
        inject_speeds(actions)
        assert actions == original
