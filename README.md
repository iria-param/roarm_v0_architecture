# RoArm-M2S — Vision-Language-Action Model (VLaM) Controller

**Powered by Gemini Robotics-ER 1.5 · 4-DOF Anti-Gravity Task Control · Fixed Webcam Vision**

---

## Overview

This project adds a complete Vision-Language-Action Model (VLaM) pipeline to the **Waveshare RoArm-M2S** 4-DOF desktop robotic arm. Type a natural-language task, the system perceives the workspace via a fixed webcam, reasons using Gemini Robotics-ER 1.5, plans a safe trajectory, and drives the arm — all with real-time anti-gravity compensation and strict joint-limit enforcement.

```
User prompt → Webcam frame → Gemini API → Safety filter → Anti-gravity → ESP32 HTTP → Arm
```

---

## Project Structure

```
roarm_vlam/
├── main.py            # CLI entry point
├── config.yaml        # All runtime parameters (edit this)
├── camera.py          # OpenCV webcam capture + base64 encoding
├── gemini_client.py   # Gemini API wrapper + prompt templates
├── safety_filter.py   # DOF clipping, workspace bounds, torque checks
├── anti_gravity.py    # Gravity-aware speed profiling
├── arm_driver.py      # HTTP JSON commands to ESP32
├── task_fsm.py        # Finite state machine orchestrator
├── calibration.py     # Camera-to-arm coordinate calibration tool
├── requirements.txt
└── tests/
    ├── test_safety_filter.py
    ├── test_anti_gravity.py
    ├── test_arm_driver.py
    └── test_task_fsm.py
```

---

## Hardware Requirements

| Item | Spec |
|---|---|
| Arm | Waveshare RoArm-M2S (4-DOF, ESP32, ST3215 servos) |
| Power | 12 V / 5 A DC supply or 3S LiPo |
| Camera | USB webcam ≥ 720p — fixed mount, clear top/side view |
| Host PC | Wi-Fi + Python 3.10+, Windows / Linux / macOS |
| Rubber bands | Install on **Shoulder** and **Elbow** joints (included in kit) |

---

## Installation

### 1. Clone / copy project

```powershell
cd "C:\Users\Hari S\Desktop\ROARM-AG\roarm_vlam"
```

### 2. Create virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set your Gemini API key

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your_key_here"

# To persist across sessions, add to your profile:
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY","your_key_here","User")
```

### 5. Verify arm connection

```powershell
# Connect to arm Wi-Fi (SSID: RoArm-M2S or similar), then:
curl http://192.168.4.1/js -d '{"T":100}'
# Expected response: {"T":100} or {"ret":0}
```

---

## Configuration

Edit **`config.yaml`** before first use:

| Key | Default | Description |
|---|---|---|
| `arm.ip` | `192.168.4.1` | ESP32 IP — change for STA (router) mode |
| `anti_gravity.rubber_bands_fitted` | `true` | **Set `false` if rubber bands not installed** |
| `camera.device_index` | `0` | USB camera index (try `1` if wrong camera) |
| `gripper.default_force_ma` | `300` | Lower for fragile objects (min 100) |

---

## Camera Calibration (one-time setup)

Run calibration any time the camera is moved:

```powershell
# Step 1 — Place 5×7 checkerboard on work surface, then:
python calibration.py --detect

# Follow on-screen instructions:
#   1. Click arm base position in the image window
#   2. Arm visits 4 reference points — click EoAT tip each time
#   -> writes calibration.json automatically

# Step 2 — Verify accuracy (should be ≤ 5 mm error):
python calibration.py --verify
```

---

## Running a Task

### Basic task (live arm):
```powershell
python main.py --task "Pick up the red cube and place it in the blue bowl"
```

### Dry-run (no arm movement — logs planned commands):
```powershell
python main.py --task "Pick up the red cube" --dry-run
```

### With payload and custom IP:
```powershell
python main.py --task "Hold the bottle for 5 seconds" --payload 0.15 --arm-ip 192.168.1.50
```

### Skip end-of-task verification:
```powershell
python main.py --task "Move arm to home" --no-verify
```

---

## Sample Task Prompts

| Task | Prompt |
|---|---|
| Basic pick & place | `"Pick up the red cube on the left and place it in the blue bowl on the right"` |
| Anti-gravity hold | `"Reach out horizontally and hold the yellow bottle for 5 seconds, then lower it"` |
| Multi-object sort | `"Sort all objects: round items left, square items right, others center"` |
| Force-sensitive grasp | `"Gently pick up the paper cup — use minimum gripper force, max 150 mA"` |
| Boundary awareness | `"Pick up whatever is closest to the far edge — stop if it's beyond 400 mm"` |

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

Expected output (no hardware required):
```
tests/test_safety_filter.py   PASSED  (20 tests)
tests/test_anti_gravity.py    PASSED  (15 tests)
tests/test_arm_driver.py      PASSED  (18 tests)
tests/test_task_fsm.py        PASSED  (12 tests)
```

---

## 4-DOF Joint Limits (Quick Reference)

| Joint | Key | Range | Anti-Gravity |
|---|---|---|---|
| Base | `T` | 0° – 360° | None needed |
| Shoulder | `B` | −45° – +135° | Rubber-band assist + speed scaling |
| Elbow | `A` | −90° – +90° | Rubber-band assist + compliance mode |
| Gripper | `G` | 0% – 100% open | Current-limit controlled |

> **Critical:** Never command Shoulder below −30° or above +120° (software soft limits enforced automatically).

---

## Anti-Gravity Speed Formula

Gravity load is highest when joints are horizontal (0°). Speed is automatically scaled:

```python
speed_shoulder = int(100 * (1 - 0.4 * |cos(θ)|))   # 60–100 units/s
speed_elbow    = int( 80 * (1 - 0.3 * |cos(θ)|))   # 56–80  units/s
```

Run `python anti_gravity.py` to print the full speed table.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Task completed successfully |
| `1` | Fatal error (camera, API, network) |
| `2` | Task executed but verification failed |
| `130` | Interrupted (Ctrl-C) — arm homed safely |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Try `camera.device_index: 1` in config.yaml |
| `Cannot connect to arm` | Check Wi-Fi SSID, try `python main.py --arm-ip <IP>` |
| `GEMINI_API_KEY not set` | Set environment variable (see Installation step 4) |
| Arm drifts under load | Ensure rubber bands are fitted; set `rubber_bands_fitted: true` |
| Calibration error > 5 mm | Re-run `python calibration.py --detect` with camera stationary |
| Gemini returns non-JSON | Retry — `max_retries: 2` in config.yaml handles this automatically |

---

## License

This project is provided as-is for educational and research purposes.
