"""
arm_driver.py — HTTP JSON command dispatch to the RoArm-M2S ESP32.

Maps the VLaM function vocabulary (home, move_joint, move_xyz, set_gripper,
wait, verify) to the ESP32 native JSON protocol and sends them via HTTP POST
to http://{ip}/js.

Usage:
    driver = ArmDriver(ip="192.168.4.1", timeout=3)
    driver.home()
    driver.execute_action({"fn": "move_joint",
                           "args": {"joint": "B", "angle_deg": 45, "speed": 80}})
    states = driver.get_joint_states()
"""

import logging
import math
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ArmConnectionError(Exception):
    """Raised when the ESP32 cannot be reached."""


class ArmCommandError(Exception):
    """Raised when the ESP32 returns an error response."""


# ---------------------------------------------------------------------------
# Servo ID mapping (joint letter → ESP32 servo T-code)
# ---------------------------------------------------------------------------
_JOINT_T: dict[str, int] = {
    "T": 1,   # Base rotation
    "B": 2,   # Shoulder
    "A": 3,   # Elbow
    "G": 4,   # Gripper
}

# Default command parameters
_DEFAULT_ACC     = 10    # acceleration (units)
_DEFAULT_SPEED   = 100   # units/s
_GRIPPER_MAX_DEG = 90    # degrees for fully closed gripper


class ArmDriver:
    """
    HTTP driver for the RoArm-M2S ESP32 JSON API.

    Args:
        ip:        ESP32 IP address (AP mode default: "192.168.4.1").
        port:      HTTP port (default 80).
        timeout:   Request timeout in seconds.
        dry_run:   If True, log commands but do not send HTTP requests.
    """

    def __init__(
        self,
        ip: str = "192.168.4.1",
        port: int = 80,
        timeout: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.base_url = f"http://{ip}:{port}"
        self.endpoint = f"{self.base_url}/js"
        self.timeout  = timeout
        self.dry_run  = dry_run
        logger.info(
            "ArmDriver init: endpoint=%s, dry_run=%s", self.endpoint, dry_run
        )

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def send_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST *payload* as JSON to the ESP32 /js endpoint.

        Returns:
            Parsed JSON response dict (empty dict on dry run).

        Raises:
            ArmConnectionError: On network failures.
            ArmCommandError:    On non-2xx HTTP responses.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] Would send: %s", payload)
            return {}

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                data = {}
            logger.debug("ESP32 response: %s", data)
            return data

        except requests.ConnectionError as exc:
            raise ArmConnectionError(
                f"Cannot connect to arm at {self.endpoint}. "
                "Check Wi-Fi connection and ARM IP in config.yaml."
            ) from exc
        except requests.Timeout as exc:
            raise ArmConnectionError(
                f"Request to {self.endpoint} timed out after {self.timeout}s."
            ) from exc
        except requests.HTTPError as exc:
            raise ArmCommandError(
                f"ESP32 returned HTTP error: {exc.response.status_code}"
            ) from exc

    # ------------------------------------------------------------------
    # High-level command helpers
    # ------------------------------------------------------------------

    def home(self) -> None:
        """Send the home command — arm returns to safe default pose."""
        logger.info("HOME command sent.")
        self.send_json({"T": 100})

    def move_joint(
        self,
        joint: str,
        angle_deg: float,
        speed: int = _DEFAULT_SPEED,
        acc: int = _DEFAULT_ACC,
    ) -> None:
        """
        Move a single joint to *angle_deg*.

        The ESP32 accepts radians, so conversion is done here.
        """
        if joint not in _JOINT_T:
            raise ArmCommandError(f"Unknown joint '{joint}'.")

        rad = math.radians(angle_deg)
        t_code = _JOINT_T[joint]

        # The ESP32 protocol uses different field names per joint
        field_map = {"T": "base", "B": "shoulder", "A": "elbow", "G": "gripper"}
        payload = {
            "T":     t_code,
            field_map[joint]: round(rad, 6),
            "speed": int(speed),
            "acc":   int(acc),
        }
        logger.info(
            "MOVE_JOINT joint=%s angle=%.2f° (%.4f rad) speed=%d",
            joint, angle_deg, rad, speed,
        )
        self.send_json(payload)

    def move_xyz(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        speed: int = _DEFAULT_SPEED,
        acc: int = _DEFAULT_ACC,
    ) -> None:
        """Move end-effector to Cartesian position using ESP32 IK (T=104)."""
        logger.info(
            "MOVE_XYZ x=%.1f y=%.1f z=%.1f mm speed=%d", x_mm, y_mm, z_mm, speed
        )
        self.send_json({
            "T":     104,
            "x":     round(float(x_mm), 2),
            "y":     round(float(y_mm), 2),
            "z":     round(float(z_mm), 2),
            "speed": int(speed),
            "acc":   int(acc),
        })

    def set_gripper(
        self,
        open_percent: float,
        force_limit_ma: int = 300,
    ) -> None:
        """
        Set gripper opening.

        Args:
            open_percent:    0 = fully closed, 100 = fully open.
            force_limit_ma:  Current limit in mA (100–500).
        """
        # Convert open_percent → servo angle (0%=90°closed, 100%=0°open)
        angle = int((1.0 - open_percent / 100.0) * _GRIPPER_MAX_DEG)
        angle = max(0, min(_GRIPPER_MAX_DEG, angle))
        logger.info(
            "SET_GRIPPER open=%.0f%% → angle=%d° torque=%d mA",
            open_percent, angle, force_limit_ma,
        )
        self.send_json({
            "T":      121,
            "cmd":    angle,
            "torque": int(force_limit_ma),
        })

    def set_torque_threshold(self, joint: str, threshold_pct: int) -> None:
        """
        Enable compliance / torque threshold mode on a joint.

        Args:
            joint:          "B" (shoulder), "A" (elbow), or "G" (gripper).
            threshold_pct:  Percentage of max torque (0–100).
        """
        threshold_pct = max(0, min(100, int(threshold_pct)))
        logger.info(
            "SET_TORQUE_THRESHOLD joint=%s threshold=%d%%", joint, threshold_pct
        )
        self.send_json({
            "T":         114,
            "joint":     joint,
            "threshold": threshold_pct,
        })

    def get_joint_states(self) -> dict[str, float]:
        """
        Query all joint angles from the ESP32 (T=105).

        Returns:
            Dict {"T": deg, "B": deg, "A": deg, "G": pct}
            On failure or dry-run, returns zeros.
        """
        try:
            resp = self.send_json({"T": 105})
        except (ArmConnectionError, ArmCommandError) as exc:
            logger.warning("get_joint_states failed: %s — returning zeros.", exc)
            return {"T": 0.0, "B": 0.0, "A": 0.0, "G": 0.0}

        # Parse ESP32 feedback — field names may vary by firmware
        states: dict[str, float] = {}
        field_map = {
            "base":     "T",
            "shoulder": "B",
            "elbow":    "A",
            "gripper":  "G",
        }
        for esp_field, joint in field_map.items():
            if esp_field in resp:
                # ESP32 returns radians; convert to degrees
                states[joint] = math.degrees(float(resp[esp_field]))
        return states

    # ------------------------------------------------------------------
    # Action dispatcher (VLaM fn → arm command)
    # ------------------------------------------------------------------

    def execute_action(self, action: dict[str, Any]) -> str | None:
        """
        Execute a single validated VLaM action dict.

        Returns:
            "VERIFY" if the action is a verify step (signals FSM to re-capture),
            None otherwise.
        """
        fn   = action.get("fn")
        args = action.get("args", {})

        if fn == "home":
            self.home()

        elif fn == "move_joint":
            self.move_joint(
                joint     = args["joint"],
                angle_deg = args["angle_deg"],
                speed     = args.get("speed", _DEFAULT_SPEED),
                acc       = args.get("acc",   _DEFAULT_ACC),
            )

        elif fn == "move_xyz":
            self.move_xyz(
                x_mm  = args["x_mm"],
                y_mm  = args["y_mm"],
                z_mm  = args["z_mm"],
                speed = args.get("speed", _DEFAULT_SPEED),
                acc   = args.get("acc",   _DEFAULT_ACC),
            )

        elif fn == "set_gripper":
            self.set_gripper(
                open_percent   = args.get("open_percent",   100),
                force_limit_ma = args.get("force_limit_ma", 300),
            )

        elif fn == "wait":
            ms = args.get("ms", 0)
            logger.info("WAIT %.3f s", ms / 1000.0)
            if not self.dry_run:
                time.sleep(ms / 1000.0)

        elif fn == "verify":
            logger.info("VERIFY signal — FSM will re-capture and assess.")
            return "VERIFY"

        else:
            logger.warning("Unknown action fn '%s' — skipped.", fn)

        return None


# ---------------------------------------------------------------------------
# Quick connectivity check (run directly: python arm_driver.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
    print(f"Connecting to RoArm-M2S at {ip} …")
    driver = ArmDriver(ip=ip, dry_run=False)
    try:
        states = driver.get_joint_states()
        print(f"Joint states: {states}")
        print("Sending HOME …")
        driver.home()
        print("Done.")
    except ArmConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
