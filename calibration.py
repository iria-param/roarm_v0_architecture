"""
calibration.py — Camera-to-arm spatial calibration tool.

Modes:
  --detect   Auto-detect checkerboard corners + arm base in image;
             prompt user to confirm 4 reference arm positions; write calibration.json.
  --verify   Command arm to 4 known positions, measure pixel error vs. calibration.

Run:
    python calibration.py --detect
    python calibration.py --verify
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
CONFIG_FILE      = Path(__file__).parent / "config.yaml"

# Checkerboard inner corners (cols-1, rows-1)
BOARD_COLS = 6   # 7-column board → 6 inner corners
BOARD_ROWS = 4   # 5-row board    → 4 inner corners
SQUARE_MM  = 30  # mm per square


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    return {}


def open_camera(cfg: dict) -> cv2.VideoCapture:
    cam_cfg = cfg.get("camera", {})
    idx = cam_cfg.get("device_index", 0)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {idx}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    return cap


def send_arm(ip: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"http://{ip}/js", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[WARN] Arm command failed: {exc}")
        return {}


def grab_frame(cap: cv2.VideoCapture) -> np.ndarray:
    for _ in range(3):
        ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read camera frame.")
    return frame


def pixel_to_norm(px: int, py: int, width: int, height: int) -> tuple[int, int]:
    """Convert pixel coords → normalized 0–1000."""
    return int(py * 1000 / height), int(px * 1000 / width)


# ---------------------------------------------------------------------------
# detect mode
# ---------------------------------------------------------------------------

def run_detect(cfg: dict) -> None:
    """
    Interactive calibration:
    1. Find checkerboard corners in image.
    2. Identify arm base pixel position (user click).
    3. Ask arm to visit 4 reference points.
    4. User enters mm for each; compute scale & perspective transform.
    5. Write calibration.json.
    """
    print("\n=== CALIBRATION — DETECT MODE ===")
    print("Place the 5x7 checkerboard (30mm squares) flat on the work surface.")
    print("Press ENTER to capture …")
    input()

    cap   = open_camera(cfg)
    frame = grab_frame(cap)
    h, w  = frame.shape[:2]

    # --- Checkerboard detection ---
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (BOARD_COLS, BOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        cv2.drawChessboardCorners(frame, (BOARD_COLS, BOARD_ROWS), corners, found)
        # Use first corner as workspace reference
        ref_px = corners[0][0]
        print(f"Checkerboard detected! Reference corner: px=({ref_px[0]:.0f}, {ref_px[1]:.0f})")
    else:
        print("[WARN] Checkerboard not detected — proceeding with manual origin click.")
        ref_px = np.array([w / 2, h / 2])

    # --- User clicks arm base position ---
    origin_px = [None, None]

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            origin_px[0], origin_px[1] = x, y
            print(f"  Origin set at pixel ({x}, {y})")

    vis = frame.copy()
    cv2.putText(vis, "Click the arm BASE on the image, then press ENTER",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Calibration — click arm base", vis)
    cv2.setMouseCallback("Calibration — click arm base", on_click)
    while origin_px[0] is None:
        cv2.waitKey(50)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    ox, oy = origin_px
    norm_y, norm_x = pixel_to_norm(ox, oy, w, h)
    print(f"Arm base in image: pixel=({ox},{oy})  normalized=({norm_y},{norm_x})")

    # --- 4 reference points ---
    arm_ip = cfg.get("arm", {}).get("ip", "192.168.4.1")
    ref_points_mm   = []
    ref_points_norm = []

    REFERENCE_POSITIONS_MM = [
        (150, 0, 0),
        (0, 150, 0),
        (-150, 0, 0),
        (100, 100, 0),
    ]

    print("\nNow the arm will move to 4 reference positions.")
    print("After each move, click the end-effector tip in the image window.\n")

    for i, (rx, ry, rz) in enumerate(REFERENCE_POSITIONS_MM):
        print(f"  [{i+1}/4] Moving arm to ({rx}, {ry}, {rz}) mm …")
        send_arm(arm_ip, {"T": 104, "x": rx, "y": ry, "z": rz, "speed": 60, "acc": 10})

        import time; time.sleep(2.0)  # let arm settle

        frame = grab_frame(cap)
        click_pt = [None, None]

        def on_ref_click(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_pt[0], click_pt[1] = x, y

        vis2 = frame.copy()
        cv2.putText(vis2, f"Click EoAT tip for ref {i+1}/4, then ENTER",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.imshow("Calibration — reference", vis2)
        cv2.setMouseCallback("Calibration — reference", on_ref_click)
        while click_pt[0] is None:
            cv2.waitKey(50)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        ny, nx = pixel_to_norm(click_pt[0], click_pt[1], w, h)
        ref_points_mm.append((rx, ry))
        ref_points_norm.append((nx, ny))
        print(f"     Ref {i+1}: norm=({nx},{ny})")

    cap.release()

    # --- Compute scale (average of 4 reference distances) ---
    scales = []
    for (rx, ry), (nx, ny) in zip(ref_points_mm, ref_points_norm):
        dist_mm   = math.sqrt(rx**2 + ry**2) + 1e-9
        dist_norm = math.sqrt((nx - norm_x)**2 + (ny - norm_y)**2) + 1e-9
        scales.append(dist_mm / dist_norm)

    scale = float(np.median(scales))
    print(f"\nComputed scale: {scale:.4f} mm/unit  (median of {len(scales)} samples)")

    # --- Perspective transform (3x3 homography) ---
    src = np.float32([[nx, ny] for nx, ny in ref_points_norm])
    dst = np.float32([[rx, ry] for rx, ry in ref_points_mm])
    H, _mask = cv2.findHomography(src, dst)

    cal: dict[str, Any] = {
        "origin_y":                   norm_y,
        "origin_x":                   norm_x,
        "scale_mm_per_unit":          round(scale, 4),
        "camera_height_cm":           cfg.get("calibration", {}).get("camera_height_cm", 60),
        "camera_angle_deg":           cfg.get("calibration", {}).get("camera_angle_deg", 45),
        "perspective_transform_matrix": H.tolist(),
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"\n[OK] calibration.json written to: {CALIBRATION_FILE}")
    print(f"  origin: y={norm_y}, x={norm_x}")
    print(f"  scale:  {scale:.4f} mm/unit")


# ---------------------------------------------------------------------------
# verify mode
# ---------------------------------------------------------------------------

def run_verify(cfg: dict) -> None:
    """
    Verify calibration by commanding the arm to 4 checkerboard corners
    and measuring pixel error against calibration.json predictions.
    """
    print("\n=== CALIBRATION — VERIFY MODE ===")

    if not CALIBRATION_FILE.exists():
        print(f"[ERROR] calibration.json not found at {CALIBRATION_FILE}.")
        print("Run --detect first.")
        sys.exit(1)

    with open(CALIBRATION_FILE) as f:
        cal = json.load(f)

    arm_ip = cfg.get("arm", {}).get("ip", "192.168.4.1")
    cap    = open_camera(cfg)
    h, w   = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    TEST_POSITIONS_MM = [(150, 0), (0, 150), (-100, 100), (100, -100)]
    errors_mm = []

    H_inv = np.linalg.inv(np.array(cal["perspective_transform_matrix"]))

    for i, (tx, ty) in enumerate(TEST_POSITIONS_MM):
        print(f"  [{i+1}/4] Moving to ({tx}, {ty}) mm …")
        send_arm(arm_ip, {"T": 104, "x": tx, "y": ty, "z": 0, "speed": 60, "acc": 10})

        import time; time.sleep(2.0)
        frame = grab_frame(cap)

        click_pt = [None, None]

        def on_verify_click(event, x, y, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_pt[0], click_pt[1] = x, y

        vis = frame.copy()
        cv2.putText(vis, f"Click EoAT tip {i+1}/4 then ENTER",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Verify", vis)
        cv2.setMouseCallback("Verify", on_verify_click)
        while click_pt[0] is None:
            cv2.waitKey(50)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        ny, nx = pixel_to_norm(click_pt[0], click_pt[1], w, h)
        # Project clicked norm coords through inverse homography → mm estimate
        pt_h = H_inv @ np.array([nx, ny, 1.0])
        ex, ey = pt_h[:2] / pt_h[2]
        err = math.sqrt((ex - tx)**2 + (ey - ty)**2)
        errors_mm.append(err)
        print(f"     Expected ({tx:.0f},{ty:.0f}) mm  |  Estimated ({ex:.1f},{ey:.1f}) mm  |  Error {err:.1f} mm")

    cap.release()
    avg_err = float(np.mean(errors_mm))
    status  = "PASS ✓" if avg_err <= 5.0 else "FAIL ✗"
    print(f"\nAverage error: {avg_err:.2f} mm  — {status}  (threshold ±5 mm)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="RoArm-M2S camera-to-arm spatial calibration tool."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--detect", action="store_true",
                       help="Run detection + calibration wizard.")
    group.add_argument("--verify", action="store_true",
                       help="Verify an existing calibration.json.")
    args = parser.parse_args()
    cfg  = load_config()

    if args.detect:
        run_detect(cfg)
    elif args.verify:
        run_verify(cfg)


if __name__ == "__main__":
    main()
