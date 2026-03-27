# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Reachy Mini?

An open-source expressive robot SDK by Pollen Robotics. The robot has a 6-DOF head (Stewart platform), body rotation, and two antenna motors (also usable as buttons). Hardware variants: **Lite** (USB to laptop) and **Wireless** (onboard CM4, WiFi).

## Development Commands

```bash
# Install (use uv, not pip)
uv sync --frozen --all-extras --group dev

# Run tests (CI-safe, excludes hardware-dependent tests)
uv run pytest -vv -m 'not audio and not video and not audio_sounddevice and not video_opencv and not wireless' --tb=short

# Run a single test file
uv run pytest tests/unit_tests/test_daemon.py -vv

# Lint (auto-fix with --fix)
uv run ruff check src/ tests/

# Format (check only; remove --check to auto-format)
uv run ruff format --check src/

# Type check (strict mode, targets src/ and examples/)
uv run mypy

# Start daemon (simulation, headless)
reachy-mini-daemon --sim --headless --localhost-only

# Start daemon (real hardware, Lite)
reachy-mini-daemon --serialport auto --localhost-only

# Create a new app
reachy-mini-app-assistant create <app_name> <path> --publish
```

## Architecture

**Communication:** SDK uses **Eclipse Zenoh** pub/sub middleware for real-time client-daemon communication. The daemon publishes state (joint positions, head pose, IMU) at **50 Hz**; clients send commands (set_target, goto_target, wake_up, etc.) as Zenoh messages.

```
User App → ReachyMini (SDK client)
             → ZenohClient (io/zenoh_client.py)
               → [Zenoh network]
                 → ZenohServer (io/zenoh_server.py) [inside daemon]
                   → Backend (abstract.py)
                     ├── RobotBackend (real hardware via serial/USB)
                     ├── MujocoBackend (simulation)
                     └── MockupSimBackend (lightweight mockup)
```

**REST API:** The daemon also runs a **FastAPI** server on port 8000 (`/api`) with routers for apps, motors, moves, state, kinematics, volume, wifi, etc. Interactive docs at `http://{host}:8000/docs`.

**Motion control modes:**
- `goto_target()` — smooth interpolated motion (default, for gestures ≥0.5s). Interpolation methods: `linear`, `minjerk` (default), `ease`, `cartoon`. Concurrent goto/play_move calls are guarded by `_play_move_lock`.
- `set_target()` — immediate real-time control at 10Hz+ (tracking, games).
- `goto_joint_positions()` — joint-space interpolation (backend only, async).
- `create_head_pose(x, y, z, roll, pitch, yaw, mm=False, degrees=True)` from `reachy_mini.utils` — primary helper for building 4x4 head pose matrices.

**Kinematics engines (pluggable):**
- `AnalyticalKinematics` — default, Rust-backed (`reachy-mini-rust-kinematics`)
- `NNKinematics` — ONNX neural network (optional extra)
- `PlacoKinematics` — collision-aware solver (optional extra)

## Key Source Layout

- `src/reachy_mini/reachy_mini.py` — Main SDK class (`ReachyMini`), method signatures and docstrings
- `src/reachy_mini/daemon/` — FastAPI daemon: `app/main.py` (entrypoint), `app/routers/` (REST endpoints), `backend/` (hardware/sim backends)
- `src/reachy_mini/daemon/backend/abstract.py` — Base Backend with all motion/kinematics logic
- `src/reachy_mini/io/` — Zenoh communication layer (client, server, protocol types)
- `src/reachy_mini/kinematics/` — IK/FK engines
- `src/reachy_mini/motion/` — Move primitives (goto, recorded moves from HuggingFace datasets)
- `src/reachy_mini/media/` — Audio/video with pluggable backends (OpenCV, GStreamer, WebRTC)
- `src/reachy_mini/apps/` — App scaffolding, lifecycle management, CLI assistant
- `tests/unit_tests/` — pytest tests; some require hardware markers (`audio`, `video`, `wireless`)

## Linting & Style

- **Pre-commit** hooks are configured (`.pre-commit-config.yaml`) — runs Ruff checks and formatting automatically.
- **Ruff** (`ruff==0.12.0`): extends with `I` (isort) and `D` (pydocstyle). Excludes `__init__.py`, `build/`, `conftest.py`, `tests/`.
- **Mypy** (`mypy==1.18.2`): strict mode, Python 3.12, targets `src/` and `examples/`.
- Python ≥3.10 required.

## Testing Notes

- CI runs on both **Ubuntu and macOS**. Tests require `MUJOCO_GL=disable` for headless environments.
- Tests in `tests/unit_tests/`. Hardware-gated markers: `audio`, `audio_sounddevice`, `video`, `video_opencv`, `wireless`.
- Daemon tests are async (`@pytest.mark.asyncio`).
- Test fixture apps: `tests/unit_tests/ok_app/` (valid) and `faulty_app/` (intentionally broken).

## Building Apps (for AI agents)

Read `AGENTS.md` for the full AI agent development guide, including SDK patterns, safety limits, skills reference, and app creation workflow. Always create apps via `reachy-mini-app-assistant` (never manually).
