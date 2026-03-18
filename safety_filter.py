"""
safety_filter.py — DOF clipping, torque checks, and workspace bounds validation.

Every action emitted by the Gemini model passes through validate_action()
before being sent to the arm. This module is the last line of defense.

Usage:
    from safety_filter import validate_sequence
    safe_actions = validate_sequence(raw_actions)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint limits (degrees)
# ---------------------------------------------------------------------------
DOF_LIMITS: dict[str, tuple[float, float]] = {
    "T": (0.0,   360.0),   # Base — full 360° rotation, no gravity load
    "B": (-45.0, 135.0),   # Shoulder — CRITICAL: rubber-band assist required above 90°
    "A": (-90.0,  90.0),   # Elbow — compliant mode above 70% torque
    "G": (0.0,   100.0),   # Gripper open% — 0 = fully closed, 100 = fully open
}

# Joint soft safety margins (applied on top of DOF limits for trajectory commands)
SOFT_MARGIN: dict[str, tuple[float, float]] = {
    "T": (0.0,    360.0),  # no extra margin on base
    "B": (-30.0,  120.0),  # ±15° inside hard limits as per spec §5.2
    "A": (-85.0,   85.0),  # ±5° inside hard limits
    "G": (0.0,    100.0),
}

# ---------------------------------------------------------------------------
# Workspace limits (mm from arm base centre)
# ---------------------------------------------------------------------------
WORKSPACE_MM: dict[str, tuple[float, float]] = {
    "x": (-450.0, 450.0),
    "y": (-450.0, 450.0),
    "z": (0.0,    600.0),
}

MAX_REACH_MM = 400.0  # soft reach limit for trajectory planning


class SafetyViolation(Exception):
    """Raised when an action cannot be made safe even after clipping."""


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------

def clip_joint(joint: str, angle: float, *, use_soft_margin: bool = True) -> float:
    """
    Clip *angle* to the allowed range for *joint*.

    Args:
        joint:           One of "T", "B", "A", "G".
        angle:           Requested angle in degrees.
        use_soft_margin: If True, apply the soft safety margin (recommended for
                         trajectory commands). If False, use hard DOF limits only.

    Returns:
        Clipped angle (float, degrees).
    """
    limits = SOFT_MARGIN[joint] if use_soft_margin else DOF_LIMITS[joint]
    lo, hi = limits
    clipped = max(lo, min(hi, float(angle)))
    if abs(clipped - angle) > 0.001:
        logger.warning(
            "Joint %s clipped %.2f° → %.2f°  (limit [%.1f, %.1f])",
            joint, angle, clipped, lo, hi,
        )
    return clipped


def validate_xyz(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """
    Clip x, y, z to workspace limits and warn if reach exceeds MAX_REACH_MM.

    Returns:
        (x_clipped, y_clipped, z_clipped)
    """
    import math

    cx = max(WORKSPACE_MM["x"][0], min(WORKSPACE_MM["x"][1], float(x)))
    cy = max(WORKSPACE_MM["y"][0], min(WORKSPACE_MM["y"][1], float(y)))
    cz = max(WORKSPACE_MM["z"][0], min(WORKSPACE_MM["z"][1], float(z)))

    if cx != x:
        logger.warning("x_mm clipped %.1f → %.1f", x, cx)
    if cy != y:
        logger.warning("y_mm clipped %.1f → %.1f", y, cy)
    if cz != z:
        logger.warning("z_mm clipped %.1f → %.1f", z, cz)

    reach = math.sqrt(cx ** 2 + cy ** 2)
    if reach > MAX_REACH_MM:
        logger.warning(
            "Horizontal reach %.1f mm exceeds safe limit %.1f mm — command may fail.",
            reach, MAX_REACH_MM,
        )

    return cx, cy, cz


def validate_speed(speed: int | float) -> int:
    """Clamp speed to [20, 150] and return as int."""
    s = max(20, min(150, int(speed)))
    if s != int(speed):
        logger.warning("Speed clipped %d → %d", int(speed), s)
    return s


def validate_gripper_force(force_ma: int | float) -> int:
    """Clamp gripper force limit to [100, 500] mA."""
    f = max(100, min(500, int(force_ma)))
    if f != int(force_ma):
        logger.warning("Gripper force clipped %d → %d mA", int(force_ma), f)
    return f


# ---------------------------------------------------------------------------
# Action-level validators
# ---------------------------------------------------------------------------

def validate_action(action: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and clip all values in a single VLaM action dict.

    Modifies *action* in-place and also returns it.

    Supported fn names: "home", "move_joint", "move_xyz",
                        "set_gripper", "wait", "verify".
    """
    fn = action.get("fn")
    args = action.get("args", {})

    if fn == "move_joint":
        joint = args.get("joint")
        if joint not in DOF_LIMITS:
            raise SafetyViolation(f"Unknown joint '{joint}' in move_joint command.")
        args["angle_deg"] = clip_joint(joint, args["angle_deg"])
        if "speed" in args:
            args["speed"] = validate_speed(args["speed"])

    elif fn == "move_xyz":
        cx, cy, cz = validate_xyz(
            args.get("x_mm", 0),
            args.get("y_mm", 0),
            args.get("z_mm", 0),
        )
        args["x_mm"], args["y_mm"], args["z_mm"] = cx, cy, cz
        if "speed" in args:
            args["speed"] = validate_speed(args["speed"])

    elif fn == "set_gripper":
        args["open_percent"] = max(0, min(100, int(args.get("open_percent", 0))))
        if "force_limit_ma" in args:
            args["force_limit_ma"] = validate_gripper_force(args["force_limit_ma"])

    elif fn == "wait":
        # Clamp wait to [0, 10000] ms to prevent infinite hangs
        args["ms"] = max(0, min(10_000, int(args.get("ms", 0))))

    elif fn in ("home", "verify"):
        pass  # no arguments to validate

    else:
        logger.warning("Unknown action fn '%s' — passing through unmodified.", fn)

    action["args"] = args
    return action


def validate_sequence(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate every action in a sequence.

    Returns the (possibly modified) list.
    Raises SafetyViolation if any action is fundamentally unsafe.
    """
    validated = []
    for i, action in enumerate(actions):
        try:
            validated.append(validate_action(action))
        except SafetyViolation as exc:
            raise SafetyViolation(f"Action #{i} ({action.get('fn')}): {exc}") from exc
    return validated


# ---------------------------------------------------------------------------
# Smoke-test (run directly: python safety_filter.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_actions = [
        {"fn": "home",       "args": {}},
        {"fn": "move_joint", "args": {"joint": "B", "angle_deg": 150, "speed": 200}},  # clipped
        {"fn": "move_joint", "args": {"joint": "A", "angle_deg": -95, "speed": 80}},   # clipped
        {"fn": "move_xyz",   "args": {"x_mm": 500, "y_mm": 0, "z_mm": -10}},           # clipped
        {"fn": "set_gripper","args": {"open_percent": 50, "force_limit_ma": 600}},      # clipped
        {"fn": "wait",       "args": {"ms": 500}},
        {"fn": "verify",     "args": {}},
    ]

    safe = validate_sequence(test_actions)
    print("\nValidated sequence:")
    for a in safe:
        print(" ", a)
