"""
main.py — RoArm-M2S VLaM entry point (CLI mode).

Usage:
    # Run a task (live arm):
    python main.py --task "Pick up the red cube and put it in the blue bowl"

    # Dry-run (no HTTP calls; logs planned JSON to stdout):
    python main.py --task "Pick up the red cube" --dry-run

    # Override config values at runtime:
    python main.py --task "…" --arm-ip 192.168.1.50 --payload 0.1

    # Run calibration:
    python calibration.py --detect
    python calibration.py --verify
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR       = Path(__file__).parent
CONFIG_FILE      = SCRIPT_DIR / "config.yaml"
CALIBRATION_FILE = SCRIPT_DIR / "calibration.json"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(level_str: str, log_to_file: bool, log_file: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
        handlers = handlers,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] config.yaml not found at {path}.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def apply_calibration(cfg: dict, cal_path: Path) -> dict:
    """Merge calibration.json values into the config calibration section."""
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        cfg.setdefault("calibration", {}).update(cal)
        logging.getLogger(__name__).info(
            "Calibration loaded: origin=(%s,%s), scale=%.4f mm/unit",
            cal.get("origin_y"), cal.get("origin_x"), cal.get("scale_mm_per_unit", 0)
        )
    else:
        logging.getLogger(__name__).warning(
            "calibration.json not found — using config.yaml defaults. "
            "Run: python calibration.py --detect"
        )
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config keys with CLI arguments."""
    if args.arm_ip:
        cfg.setdefault("arm", {})["ip"] = args.arm_ip
    return cfg


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "roarm_vlam",
        description = "RoArm-M2S Vision-Language-Action Model Controller",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = """
Examples:
  python main.py --task "Pick up the red cube"
  python main.py --task "Sort all objects by shape" --dry-run
  python main.py --task "Hold bottle horizontally" --payload 0.15 --arm-ip 192.168.1.50

For calibration:
  python calibration.py --detect
  python calibration.py --verify
""",
    )

    parser.add_argument(
        "--task", "-t",
        required = True,
        metavar  = "TASK",
        help     = 'Natural-language task description. Example: "Pick up the red cube".',
    )
    parser.add_argument(
        "--dry-run", "-n",
        action   = "store_true",
        dest     = "dry_run",
        help     = "Simulate execution — log planned commands without sending to arm.",
    )
    parser.add_argument(
        "--payload",
        type     = float,
        default  = 0.0,
        metavar  = "KG",
        help     = "Estimated payload mass in kg (default: 0.0).",
    )
    parser.add_argument(
        "--arm-ip",
        metavar  = "IP",
        default  = None,
        help     = "Override arm IP from config.yaml (e.g. 192.168.1.50).",
    )
    parser.add_argument(
        "--config",
        metavar  = "PATH",
        default  = str(CONFIG_FILE),
        help     = f"Path to config.yaml (default: {CONFIG_FILE}).",
    )
    parser.add_argument(
        "--no-verify",
        action   = "store_true",
        dest     = "no_verify",
        help     = "Skip the verification frame at the end of task execution.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # --- Config ---
    cfg = load_config(Path(args.config))
    cfg = apply_calibration(cfg, CALIBRATION_FILE)
    cfg = apply_cli_overrides(cfg, args)

    # Respect --no-verify flag
    if args.no_verify:
        cfg.setdefault("task", {})["verify_on_complete"] = False

    # --- Logging ---
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level_str  = log_cfg.get("level",      "INFO"),
        log_to_file = log_cfg.get("log_to_file", False),
        log_file   = log_cfg.get("log_file",    "roarm_vlam.log"),
    )
    logger = logging.getLogger("main")

    # --- Print banner ---
    arm_ip = cfg.get("arm", {}).get("ip", "192.168.4.1")
    model  = cfg.get("gemini", {}).get("model", "gemini-robotics-er-1.5-preview")
    mode   = "[DRY-RUN]" if args.dry_run else "[LIVE]"
    bands  = "fitted ✓" if cfg.get("anti_gravity", {}).get("rubber_bands_fitted", True) else "NOT fitted ✗"
    print(
        f"\n{'─'*60}\n"
        f"  RoArm-M2S VLaM Controller\n"
        f"  Mode     : {mode}\n"
        f"  Model    : {model}\n"
        f"  Arm IP   : {arm_ip}\n"
        f"  Payload  : {args.payload} kg\n"
        f"  Rubber bands: {bands}\n"
        f"  Task     : {args.task}\n"
        f"{'─'*60}\n"
    )

    # --- API key check ---
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error(
            "GEMINI_API_KEY environment variable is not set.\n"
            "  Windows: $env:GEMINI_API_KEY = 'your_key_here'\n"
            "  Linux/Mac: export GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    # --- Lazy import FSM (so --help works without dependencies) ---
    from task_fsm import TaskFSM, TaskFailed

    # --- Run ---
    try:
        with TaskFSM(config=cfg, dry_run=args.dry_run) as fsm:
            result = fsm.run(
                task_description = args.task,
                payload_kg       = args.payload,
            )

        # --- Print result ---
        status_icon = "✓" if result.success else "✗"
        print(f"\n{'─'*60}")
        print(f"  {status_icon} Task {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Summary  : {result.task_summary}")
        print(f"  Objects  : {len(result.objects)} detected")
        print(f"  Retries  : {result.retries}")
        print(f"  Duration : {result.elapsed_s:.1f} s")
        if result.verify_note:
            print(f"  Verify   : {result.verify_note}")
        print(f"{'─'*60}\n")

        sys.exit(0 if result.success else 2)

    except TaskFailed as exc:
        logger.error("Task failed: %s", exc)
        print(f"\n[TASK FAILED] {exc}\n")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Sending home command …")
        # Best-effort home on Ctrl-C
        try:
            from arm_driver import ArmDriver
            ArmDriver(ip=arm_ip, dry_run=args.dry_run).home()
        except Exception:
            pass
        sys.exit(130)


if __name__ == "__main__":
    main()
