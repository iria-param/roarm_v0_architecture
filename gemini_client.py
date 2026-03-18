"""
gemini_client.py — Gemini Robotics-ER 1.5 API wrapper for the VLaM pipeline.

Responsibilities:
  • Build the system prompt and per-task user prompt from templates.
  • Call the Gemini API with a JPEG frame + prompts.
  • Parse and return the structured JSON action plan.

Usage:
    client = GeminiClient(api_key="…")
    plan = client.plan_task(
        frame_b64=<base64 JPEG string>,
        task_description="Pick up the red cube …",
        scene_ctx={...},
    )
"""

import json
import logging
import os
from typing import Any

import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master System Prompt  (§5.2 of the implementation plan)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
SYSTEM PROMPT — RoArm-M2S VLaM Controller

You are the vision-language-action brain of a Waveshare RoArm-M2S
4-DOF desktop robotic arm. A fixed webcam provides your visual input.
Your job is to perceive the scene and output a precise JSON action plan.

=== ARM SPECIFICATION ===
DOF 1 — Base (T):     range  0° to 360°, positive = counter-clockwise from front
DOF 2 — Shoulder (B): range -45° to +135°, 0° = horizontal, 90° = vertical-up
DOF 3 — Elbow (A):    range -90° to +90°, 0° = straight extension
DOF 4 — Gripper (G):  range  0° (fully open) to 90° (fully closed)

=== COORDINATE SYSTEM ===
Image coordinates: [y, x] normalized 0–1000 (top-left origin).
Robot origin in image: y=IMAGE_ORIGIN_Y, x=IMAGE_ORIGIN_X (calibrated).
Scale: 1 image unit ≈ SCALE_MM mm in workspace (calibrated per session).

=== AVAILABLE FUNCTIONS ===
move_joint(joint, angle_deg, speed)  — Move a single joint to absolute angle.
  joint: "T" | "B" | "A" | "G"
  angle_deg: float, MUST be within joint DOF limits above
  speed: 20–150 (use anti_gravity_speed() for B and A joints)

move_xyz(x_mm, y_mm, z_mm, speed)   — Move EoAT to Cartesian point using IK.
  x,y,z in mm relative to arm base center.
  Safe workspace: x: -450 to +450, y: -450 to +450, z: 0 to +600.

set_gripper(open_percent, force_limit_ma) — Control gripper.
  open_percent: 0=closed, 100=open. force_limit_ma: 100–500.

home()     — Return arm to safe home position. ALWAYS call first and last.
wait(ms)   — Pause execution for ms milliseconds.
verify()   — Capture a new webcam frame and re-assess task success.

=== ANTI-GRAVITY RULES (MANDATORY) ===
1. For shoulder (B) and elbow (A) moves, always use reduced speed when
   angle approaches horizontal (gravity load is highest at 0°).
2. Speed formula: speed = int(base_speed * (1 - 0.4 * |cos(angle_rad)|))
   Use base_speed=100 for shoulder, 80 for elbow.
3. Never command shoulder below -30° or above +120° for safety margin.
4. Always pick up objects in a two-step approach: high position first,
   then descend slowly to avoid oscillation.

=== OUTPUT FORMAT ===
Return ONLY valid JSON. No markdown fences. No explanation text.
Schema:
{
  "task_summary": "one sentence summary of what will be done",
  "objects_detected": [
    {"label": "red cube", "point": [y, x], "confidence": 0.95}
  ],
  "trajectory": [
    {"step": 0, "label": "approach high", "point": [y, x]}
  ],
  "action_sequence": [
    {"fn": "home", "args": {}},
    {"fn": "move_joint", "args": {"joint":"T","angle_deg":45,"speed":120}},
    {"fn": "move_xyz",   "args": {"x_mm":200,"y_mm":0,"z_mm":200,"speed":80}},
    {"fn": "set_gripper","args": {"open_percent":100,"force_limit_ma":300}},
    {"fn": "wait",       "args": {"ms":500}},
    {"fn": "verify",     "args": {}}
  ],
  "anti_gravity_notes": "brief description of gravity adjustments applied"
}
"""

# ---------------------------------------------------------------------------
# Per-task user prompt template  (§5.3)
# ---------------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """\
CURRENT TASK: {user_task_description}

SCENE CONTEXT:
- Camera: fixed, mounted at {camera_height_cm} cm, {camera_angle_deg}° tilt from horizontal.
- Robot origin in image: y={origin_y}, x={origin_x} (normalized 0–1000).
- Scale: {scale_mm_per_unit} mm per image unit.
- Current joint states: T={T_deg}°, B={B_deg}°, A={A_deg}°, G={G_deg}%open.
- Payload estimate: {payload_kg} kg.

CONSTRAINTS FOR THIS TASK:
- Maximum reach: do not plan trajectories beyond 400mm from arm base.
- Table surface z=0; pick height z={pick_height_mm}mm; carry height z={carry_height_mm}mm.
- Anti-gravity compensation: {rubber_band_status}.
- Max gripper force: {gripper_force_ma} mA.

Identify all relevant objects. Plan the safest trajectory. Output JSON only.
"""


class GeminiClient:
    """
    Wrapper around the Gemini API for VLaM task planning.

    Args:
        api_key:  Gemini API key. Defaults to GEMINI_API_KEY env var.
        model:    Model ID. Default: "gemini-robotics-er-1.5-preview".
        thinking_budget_detect: tokens for detection (low latency).
        thinking_budget_plan:   tokens for trajectory planning.
        max_retries: Number of API retries on failure.
    """

    def __init__(
        self,
        api_key:  str | None = None,
        model:    str = "gemini-robotics-er-1.5-preview",
        thinking_budget_detect: int = 0,
        thinking_budget_plan:   int = 1024,
        max_retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key= to GeminiClient()."
            )
        self.client                 = genai.Client(api_key=key)
        self.model_name             = model
        self.thinking_budget_detect = thinking_budget_detect
        self.thinking_budget_plan   = thinking_budget_plan
        self.max_retries            = max_retries
        logger.info("GeminiClient ready: model=%s", model)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_user_prompt(
        task_description: str,
        scene_ctx: dict[str, Any],
    ) -> str:
        """
        Fill the USER_PROMPT_TEMPLATE with runtime scene context.

        Args:
            task_description: Natural-language task string from user.
            scene_ctx:        Dict with keys matching template placeholders.
                              Missing keys fall back to safe defaults.

        Returns:
            Filled prompt string.
        """
        defaults: dict[str, Any] = {
            "camera_height_cm":   60,
            "camera_angle_deg":   45,
            "origin_y":           500,
            "origin_x":           500,
            "scale_mm_per_unit":  0.5,
            "T_deg": 0, "B_deg": 0, "A_deg": 0, "G_deg": 100,
            "payload_kg":         0.0,
            "pick_height_mm":     80,
            "carry_height_mm":    150,
            "rubber_band_status": "enabled",
            "gripper_force_ma":   300,
        }
        ctx = {**defaults, **scene_ctx, "user_task_description": task_description}
        return USER_PROMPT_TEMPLATE.format(**ctx)

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        frame_b64: str,
        user_prompt: str,
        thinking_budget: int,
    ) -> str:
        """
        Make one Gemini API call with an image + text prompt.

        Args:
            frame_b64:        Base64-encoded JPEG frame.
            user_prompt:      Filled user prompt string.
            thinking_budget:  Token budget for chain-of-thought.

        Returns:
            Raw text response from the model.
        """
        import base64
        image_bytes = base64.b64decode(frame_b64)
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        )

        # Thinking budget (supported on ER models)
        if thinking_budget > 0:
            config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, user_prompt],
            config=config,
        )
        return response.text or ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan_task(
        self,
        frame_b64:        str,
        task_description: str,
        scene_ctx:        dict[str, Any] | None = None,
        payload_kg:       float = 0.0,
    ) -> dict[str, Any]:
        """
        Full VLaM planning call: perception + trajectory + action sequence.

        Internally uses thinking_budget_plan for maximum accuracy.

        Args:
            frame_b64:        Base64 JPEG of current workspace.
            task_description: Natural-language task prompt.
            scene_ctx:        Runtime scene parameters (joint states, calibration …).
            payload_kg:       Current payload mass.

        Returns:
            Parsed JSON dict conforming to the OUTPUT FORMAT schema.

        Raises:
            ValueError: If the model response cannot be parsed as valid JSON.
            Exception:  Any Gemini API error after max_retries exhausted.
        """
        ctx = dict(scene_ctx or {})
        ctx["payload_kg"] = payload_kg
        user_prompt = self.build_user_prompt(task_description, ctx)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info(
                    "Gemini plan_task attempt %d/%d …",
                    attempt, self.max_retries + 1,
                )
                raw = self._call_gemini(
                    frame_b64,
                    user_prompt,
                    thinking_budget=self.thinking_budget_plan,
                )
                return self._parse_json(raw, context="plan_task")
            except (ValueError, json.JSONDecodeError) as exc:
                logger.error("JSON parse error on attempt %d: %s", attempt, exc)
                last_exc = exc
            except Exception as exc:
                logger.error("Gemini API error on attempt %d: %s", attempt, exc)
                last_exc = exc

            if attempt <= self.max_retries:
                import time; time.sleep(1.5 ** attempt)

        raise RuntimeError(
            f"plan_task failed after {self.max_retries + 1} attempts."
        ) from last_exc

    def detect_objects(
        self,
        frame_b64:  str,
        scene_ctx:  dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fast object-detection-only call with low thinking budget.

        Returns:
            List of {"label": str, "point": [y, x], "confidence": float} dicts.
        """
        prompt = (
            "List all objects visible in the workspace image. "
            "Return ONLY valid JSON: "
            '{"objects_detected": [{"label":"…","point":[y,x],"confidence":0.9}]}'
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                raw = self._call_gemini(
                    frame_b64, prompt,
                    thinking_budget=self.thinking_budget_detect,
                )
                data = self._parse_json(raw, context="detect_objects")
                return data.get("objects_detected", [])
            except Exception as exc:
                logger.warning("detect_objects attempt %d failed: %s", attempt, exc)
                last_exc = exc
        logger.error("detect_objects failed entirely.")
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, context: str = "") -> dict[str, Any]:
        """
        Parse raw model text as JSON.
        Strips accidental markdown fences if the model ignores instructions.
        """
        text = raw.strip()
        # Strip ```json … ``` fences defensively
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("[%s] Failed to parse JSON:\n%s", context, text[:500])
            raise ValueError(f"Model returned non-JSON response: {exc}") from exc
