"""
task_fsm.py — Finite State Machine orchestrating the full VLaM task loop.

States:
    IDLE     → wait for a task string
    PERCEIVE → capture current webcam frame
    PLAN     → call Gemini; safety-filter; inject anti-gravity speeds
    EXECUTE  → dispatch action sequence to arm; on VERIFY signal → VERIFY state
    VERIFY   → re-capture frame, re-query model for success / failure
    (error)  → raise TaskFailed, return to IDLE

Usage:
    fsm = TaskFSM(config)
    result = fsm.run("Pick up the red cube and place it in the blue bowl")
"""

import logging
import time
from enum import Enum, auto
from typing import Any

from anti_gravity import inject_speeds, load_config as ag_load_config
from arm_driver    import ArmDriver, ArmConnectionError, ArmCommandError
from camera        import CameraCapture, CameraError
from gemini_client import GeminiClient
from safety_filter import validate_sequence, SafetyViolation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE    = auto()
    PERCEIVE = auto()
    PLAN    = auto()
    EXECUTE = auto()
    VERIFY  = auto()


class TaskFailed(Exception):
    """Raised when a task cannot be completed successfully."""


class TaskResult:
    """Value object returned by TaskFSM.run()."""

    def __init__(
        self,
        success:      bool,
        task_summary: str,
        objects:      list,
        retries:      int,
        elapsed_s:    float,
        verify_note:  str = "",
    ) -> None:
        self.success      = success
        self.task_summary = task_summary
        self.objects      = objects
        self.retries      = retries
        self.elapsed_s    = elapsed_s
        self.verify_note  = verify_note

    def __repr__(self) -> str:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return (
            f"TaskResult({status} | {self.task_summary!r} | "
            f"{len(self.objects)} objects | "
            f"{self.retries} retries | {self.elapsed_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------
class TaskFSM:
    """
    Orchestrate the IDLE → PERCEIVE → PLAN → EXECUTE → VERIFY → IDLE loop.

    Args:
        config: Loaded config.yaml dict (top-level).
        dry_run: If True, arm HTTP calls are skipped (logged only).
    """

    def __init__(self, config: dict[str, Any], dry_run: bool = False) -> None:
        self.cfg     = config
        self.dry_run = dry_run

        # --- Initialise components ---
        cam_cfg = config.get("camera", {})
        self.camera = CameraCapture(
            device_index  = cam_cfg.get("device_index", 0),
            width         = cam_cfg.get("width",  640),
            height        = cam_cfg.get("height", 480),
            fps           = cam_cfg.get("fps",    5),
            jpeg_quality  = cam_cfg.get("jpeg_quality", 85),
        )

        arm_cfg = config.get("arm", {})
        self.driver = ArmDriver(
            ip       = arm_cfg.get("ip",       "192.168.4.1"),
            port     = arm_cfg.get("port",     80),
            timeout  = arm_cfg.get("timeout_s", 3),
            dry_run  = dry_run,
        )

        gem_cfg = config.get("gemini", {})
        self.gemini = GeminiClient(
            model                  = gem_cfg.get("model", "gemini-robotics-er-1.5-preview"),
            thinking_budget_detect = gem_cfg.get("thinking_budget_detect", 0),
            thinking_budget_plan   = gem_cfg.get("thinking_budget_plan",   1024),
            max_retries            = gem_cfg.get("max_retries", 2),
        )

        ag_load_config(config.get("anti_gravity", {}))

        task_cfg = config.get("task", {})
        self.verify_on_complete = task_cfg.get("verify_on_complete", True)
        self.max_verify_retries = task_cfg.get("max_retries_on_verify", 1)

        cal = config.get("calibration", {})
        self._scene_base: dict[str, Any] = {
            "camera_height_cm":  cal.get("camera_height_cm", 60),
            "camera_angle_deg":  cal.get("camera_angle_deg", 45),
            "origin_y":          cal.get("origin_y",          500),
            "origin_x":          cal.get("origin_x",          500),
            "scale_mm_per_unit": cal.get("scale_mm_per_unit", 0.5),
            "pick_height_mm":    task_cfg.get("pick_height_mm",  80),
            "carry_height_mm":   task_cfg.get("carry_height_mm", 150),
            "rubber_band_status": (
                "enabled"
                if config.get("anti_gravity", {}).get("rubber_bands_fitted", True)
                else "rubber bands NOT fitted — use reduced speed"
            ),
            "gripper_force_ma": config.get("gripper", {}).get("default_force_ma", 300),
        }

        self.state = State.IDLE
        logger.info("TaskFSM initialised (dry_run=%s).", dry_run)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, task_description: str, payload_kg: float = 0.0) -> TaskResult:
        """
        Execute a complete task cycle from IDLE to IDLE.

        Args:
            task_description: Natural-language task prompt.
            payload_kg:       Estimated mass of object being manipulated.

        Returns:
            TaskResult with success flag and details.

        Raises:
            TaskFailed: If the task cannot be completed after retries.
        """
        start_time = time.monotonic()
        retries    = 0
        verify_note = ""

        logger.info("=== TASK START: %s ===", task_description)
        self._transition(State.IDLE, State.PERCEIVE)

        # ── PERCEIVE ──────────────────────────────────────────────────
        frame_b64, _frame_np = self._perceive()

        # ── PLAN ──────────────────────────────────────────────────────
        self._transition(State.PERCEIVE, State.PLAN)
        plan = self._plan(frame_b64, task_description, payload_kg)

        action_sequence = plan.get("action_sequence", [])
        task_summary    = plan.get("task_summary", "")
        objects         = plan.get("objects_detected", [])
        ag_note         = plan.get("anti_gravity_notes", "")

        logger.info("Plan summary  : %s", task_summary)
        logger.info("Objects found : %d — %s", len(objects), objects)
        logger.info("AG notes      : %s", ag_note)
        logger.info("Actions       : %d steps", len(action_sequence))

        # ── EXECUTE ───────────────────────────────────────────────────
        self._transition(State.PLAN, State.EXECUTE)
        triggered_verify = self._execute(action_sequence, payload_kg)

        # ── VERIFY ────────────────────────────────────────────────────
        if triggered_verify or self.verify_on_complete:
            for attempt in range(self.max_verify_retries + 1):
                self._transition(State.EXECUTE, State.VERIFY)
                success, verify_note = self._verify(task_description, payload_kg)
                if success:
                    break
                retries += 1
                logger.warning(
                    "Verify attempt %d/%d failed: %s",
                    attempt + 1, self.max_verify_retries + 1, verify_note,
                )
                if attempt < self.max_verify_retries:
                    # Re-plan and re-execute once
                    self._transition(State.VERIFY, State.PERCEIVE)
                    frame_b64, _ = self._perceive()
                    self._transition(State.PERCEIVE, State.PLAN)
                    plan = self._plan(frame_b64, task_description, payload_kg)
                    action_sequence = plan.get("action_sequence", [])
                    self._transition(State.PLAN, State.EXECUTE)
                    self._execute(action_sequence, payload_kg)
            else:
                success = False
        else:
            success = True

        # ── HOME & IDLE ────────────────────────────────────────────────
        self._home_safe()
        self._transition(self.state, State.IDLE)

        elapsed = time.monotonic() - start_time
        result  = TaskResult(
            success      = success,
            task_summary = task_summary,
            objects      = objects,
            retries      = retries,
            elapsed_s    = elapsed,
            verify_note  = verify_note,
        )
        logger.info("=== TASK END: %s (%.1fs) ===", result, elapsed)
        return result

    # ------------------------------------------------------------------
    # State handler methods
    # ------------------------------------------------------------------

    def _perceive(self) -> tuple[str, Any]:
        """Capture a frame from the webcam."""
        logger.info("[PERCEIVE] Capturing frame …")
        try:
            return self.camera.grab_frame_b64()
        except CameraError as exc:
            raise TaskFailed(f"Camera failure during PERCEIVE: {exc}") from exc

    def _plan(
        self,
        frame_b64: str,
        task_description: str,
        payload_kg: float,
    ) -> dict[str, Any]:
        """Call Gemini, safety-filter, inject anti-gravity speeds."""
        logger.info("[PLAN] Calling Gemini …")
        joint_states = {}
        try:
            joint_states = self.driver.get_joint_states()
        except Exception:
            logger.warning("Could not read joint states — using defaults.")

        scene_ctx = {
            **self._scene_base,
            "T_deg": joint_states.get("T", 0),
            "B_deg": joint_states.get("B", 0),
            "A_deg": joint_states.get("A", 0),
            "G_deg": joint_states.get("G", 100),
            "payload_kg": payload_kg,
        }

        try:
            plan = self.gemini.plan_task(
                frame_b64, task_description, scene_ctx, payload_kg
            )
        except Exception as exc:
            raise TaskFailed(f"Gemini planning failed: {exc}") from exc

        # Safety filter
        try:
            plan["action_sequence"] = validate_sequence(plan.get("action_sequence", []))
        except SafetyViolation as exc:
            raise TaskFailed(f"Safety violation in plan: {exc}") from exc

        # Anti-gravity speed injection
        plan["action_sequence"] = inject_speeds(
            plan["action_sequence"],
            joint_states=joint_states,
            payload_kg=payload_kg,
        )

        return plan

    def _execute(
        self,
        action_sequence: list[dict[str, Any]],
        payload_kg: float,
    ) -> bool:
        """
        Dispatch actions to the arm.

        Returns:
            True if a `verify` action was encountered mid-sequence.
        """
        logger.info("[EXECUTE] Dispatching %d actions …", len(action_sequence))
        triggered_verify = False
        try:
            for i, action in enumerate(action_sequence):
                logger.info("  Step %d/%d: fn=%s args=%s",
                            i + 1, len(action_sequence),
                            action.get("fn"), action.get("args"))
                result = self.driver.execute_action(action)
                if result == "VERIFY":
                    triggered_verify = True
        except (ArmConnectionError, ArmCommandError) as exc:
            raise TaskFailed(f"Arm communication error during EXECUTE: {exc}") from exc
        return triggered_verify

    def _verify(
        self,
        task_description: str,
        payload_kg: float,
    ) -> tuple[bool, str]:
        """
        Re-capture frame and ask Gemini if the task succeeded.

        Returns:
            (success: bool, note: str)
        """
        logger.info("[VERIFY] Re-capturing frame for verification …")
        frame_b64, _ = self._perceive()

        verify_prompt = (
            f"VERIFY: Was this task successfully completed? "
            f"Task: '{task_description}'. "
            "Return JSON: {\"success\": true/false, \"note\": \"reason\"}"
        )
        try:
            scene_ctx = {**self._scene_base, "payload_kg": payload_kg}
            response  = self.gemini._call_gemini(
                frame_b64, verify_prompt,
                thinking_budget=self.gemini.thinking_budget_detect,
            )
            data    = self.gemini._parse_json(response, context="verify")
            success = bool(data.get("success", False))
            note    = str(data.get("note", ""))
            logger.info("[VERIFY] success=%s note=%s", success, note)
            return success, note
        except Exception as exc:
            logger.error("[VERIFY] Error: %s — assuming failure.", exc)
            return False, str(exc)

    def _home_safe(self) -> None:
        """Send home command, suppressing errors so FSM always returns cleanly."""
        try:
            logger.info("[HOME] Returning arm to home position.")
            self.driver.home()
        except Exception as exc:
            logger.warning("Home command failed: %s", exc)

    def _transition(self, from_state: State, to_state: State) -> None:
        """Log and update FSM state."""
        logger.debug("FSM %s → %s", from_state.name, to_state.name)
        self.state = to_state

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release camera resource on exit."""
        self.camera.release()
        logger.info("TaskFSM shut down.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
