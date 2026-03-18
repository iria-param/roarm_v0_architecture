"""
anti_gravity.py — Gravity-aware velocity profiling for the RoArm-M2S.

The shoulder and elbow joints experience the highest gravitational load when
near horizontal (angle ≈ 0°). This module scales servo speed inversely with
the cosine of the joint angle so gravity is partially compensated in software.

Usage:
    from anti_gravity import anti_gravity_speed, inject_speeds

    speed = anti_gravity_speed("shoulder", angle_deg=30, payload_kg=0.1)
    actions = inject_speeds(action_sequence, joint_states={"T":0,"B":30,"A":-10,"G":50})
"""

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants (mirrors config.yaml — overridden by load_config())
# ---------------------------------------------------------------------------
_CFG = {
    "shoulder_base_speed":   100,
    "shoulder_gravity_k":    0.4,
    "elbow_base_speed":       80,
    "elbow_gravity_k":        0.3,
    "base_speed":            120,
    "gripper_speed":         120,
    "rubber_bands_fitted":  True,
}

# When rubber bands are NOT fitted, apply an additional speed reduction penalty
_NO_BANDS_PENALTY = 0.7   # 30% slower without mechanical assist


def load_config(cfg: dict) -> None:
    """
    Update tuning constants from a loaded config dict (call once at startup).

    Args:
        cfg: The "anti_gravity" sub-dict from config.yaml, e.g.:
             {"shoulder_base_speed": 100, "rubber_bands_fitted": True, ...}
    """
    for key in _CFG:
        if key in cfg:
            _CFG[key] = cfg[key]
    logger.info("Anti-gravity config loaded: %s", _CFG)


# ---------------------------------------------------------------------------
# Core speed formula
# ---------------------------------------------------------------------------

def anti_gravity_speed(
    joint: str,
    angle_deg: float,
    payload_kg: float = 0.0,
) -> int:
    """
    Return the recommended servo speed (units/s) for *joint* at *angle_deg*,
    accounting for gravitational load and optional payload mass.

    Args:
        joint:       "shoulder" | "B"  for shoulder
                     "elbow"   | "A"  for elbow
                     anything else     → constant base/gripper speed
        angle_deg:   Current (or target) joint angle in degrees.
        payload_kg:  Mass of carried object (kg); increases gravity effect.

    Returns:
        Integer speed in [20, 150] range.

    Speed formula (per spec §4.1 Level 3):
        speed_shoulder = int(base * (1 - k * |cos(θ)|))
        speed_elbow    = int(base * (1 - k * |cos(θ)|))

    Payload scaling: gravity_factor *= (1 + payload_kg * 0.5)
    so a 200 g payload increases the gravity penalty by 10%.
    """
    angle_rad    = math.radians(angle_deg)
    gravity_raw  = abs(math.cos(angle_rad))
    payload_mult = 1.0 + payload_kg * 0.5
    gravity_factor = min(1.0, gravity_raw * payload_mult)

    joint_key = joint.upper()

    if joint_key in ("SHOULDER", "B"):
        base  = _CFG["shoulder_base_speed"]
        k     = _CFG["shoulder_gravity_k"]
        speed = base * (1.0 - k * gravity_factor)
    elif joint_key in ("ELBOW", "A"):
        base  = _CFG["elbow_base_speed"]
        k     = _CFG["elbow_gravity_k"]
        speed = base * (1.0 - k * gravity_factor)
    else:
        # Base (T) and Gripper (G) — no gravity concern
        return max(20, min(150, _CFG.get("base_speed", 120)))

    # Apply rubber-band penalty if bands are not fitted
    if not _CFG["rubber_bands_fitted"]:
        speed *= _NO_BANDS_PENALTY
        if joint_key in ("SHOULDER", "B"):
            logger.warning(
                "Rubber bands NOT fitted — shoulder speed reduced by %.0f%%",
                (1 - _NO_BANDS_PENALTY) * 100,
            )

    speed = int(round(speed))
    speed = max(20, min(150, speed))
    logger.debug(
        "anti_gravity_speed(%s, %.1f°, %.2fkg) → %d units/s",
        joint, angle_deg, payload_kg, speed,
    )
    return speed


# ---------------------------------------------------------------------------
# Sequence-level helper
# ---------------------------------------------------------------------------

def inject_speeds(
    action_sequence: list[dict[str, Any]],
    joint_states: dict[str, float] | None = None,
    payload_kg: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Iterate over *action_sequence* and fill in gravity-corrected speed values
    for any `move_joint` actions targeting shoulder (B) or elbow (A) joints
    that do NOT already have an explicit speed set.

    Also fast-corrects pre-set speeds that are dangerously high for gravity
    joints when near horizontal.

    Args:
        action_sequence: List of VLaM action dicts (modified in-place).
        joint_states:    Current joint angles ({"T":…, "B":…, "A":…, "G":…}).
                         Used to estimate intermediate gravity load when the
                         target angle is not yet reached.
        payload_kg:      Current payload mass.

    Returns:
        The same list (modified in-place) for convenience.
    """
    joint_states = joint_states or {"T": 0.0, "B": 0.0, "A": 0.0, "G": 0.0}

    for action in action_sequence:
        if action.get("fn") != "move_joint":
            continue

        args  = action.get("args", {})
        joint = args.get("joint", "")

        if joint not in ("B", "A"):
            continue   # only gravity joints need correction

        target_angle = args.get("angle_deg", joint_states.get(joint, 0.0))
        # Use the midpoint between current and target for speed estimation
        current_angle = joint_states.get(joint, target_angle)
        mid_angle = (current_angle + target_angle) / 2.0

        recommended = anti_gravity_speed(joint, mid_angle, payload_kg)

        if "speed" not in args:
            args["speed"] = recommended
            logger.debug(
                "Injected speed %d for joint %s → %.1f°",
                recommended, joint, target_angle,
            )
        else:
            # Clamp any model-supplied speed to be ≤ recommended (safety)
            original = args["speed"]
            args["speed"] = min(original, recommended)
            if args["speed"] != original:
                logger.info(
                    "Speed for joint %s capped %d → %d (gravity safety)",
                    joint, original, args["speed"],
                )

        # Update virtual joint state for subsequent steps
        joint_states[joint] = target_angle

    return action_sequence


# ---------------------------------------------------------------------------
# Speed table printer (run directly: python anti_gravity.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"{'Angle':>6}  {'Shoulder':>10}  {'Elbow':>7}")
    print("-" * 28)
    for deg in range(-90, 136, 15):
        sh = anti_gravity_speed("shoulder", deg)
        el = anti_gravity_speed("elbow",    deg)
        bar_sh = "█" * (sh // 10)
        print(f"{deg:>+6}°  {sh:>4} {bar_sh:<8}  {el:>4}")
    print()
    print("Payload 0.3 kg penalty (shoulder at 0°):",
          anti_gravity_speed("shoulder", 0, payload_kg=0.3))
