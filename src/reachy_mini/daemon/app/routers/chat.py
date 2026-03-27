"""Chat router for OpenRouter API with continuous voice conversation and robot control.

STT: faster-whisper (local)
TTS: ElevenLabs (cloud API)
LLM: OpenRouter (cloud API) with tool calling for robot movements
Voice activity detection: energy-based
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from reachy_mini.daemon.backend.abstract import Backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parents[5] / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "google/gemini-3.1-flash-lite-preview")
ELEVENLABS_API_KEY: str = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

MAX_HISTORY = 20
SAMPLE_RATE = 16000

# VAD thresholds
SPEECH_ENERGY_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MIN_SPEECH_DURATION = 0.5

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive robot made by Pollen Robotics. "
    "You can see through your camera when the user shares an image. "
    "Keep responses concise and friendly (1-2 sentences). "
    "Use the available tools to move your head, express emotions, and dance "
    "whenever it feels natural in the conversation. For example, nod when agreeing, "
    "shake your head when disagreeing, show emotions that match the conversation tone, "
    "or dance when the user asks you to."
)

# ---------------------------------------------------------------------------
# LLM Tools definition (OpenAI-compatible function calling)
# ---------------------------------------------------------------------------
LLM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_head",
            "description": "Move the robot head to a position. Use for nodding, looking around, tilting, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll": {
                        "type": "number",
                        "description": "Side tilt in degrees. Positive = tilt left, negative = tilt right. Range: -30 to 30.",
                    },
                    "pitch": {
                        "type": "number",
                        "description": "Up/down in degrees. Positive = look up, negative = look down. Range: -30 to 30.",
                    },
                    "yaw": {
                        "type": "number",
                        "description": "Left/right rotation in degrees. Positive = turn left, negative = turn right. Range: -45 to 45.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration of the movement in seconds. Default 0.8.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_emotion",
            "description": "Play an expressive emotion animation. Use to react to conversation naturally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "Emotion name. Examples: cheerful1, sad1, surprised1, laughing1, "
                        "curious1, amazed1, confused1, shy1, proud1, grateful1, yes1, no1, "
                        "thoughtful1, welcoming1, oops1, scared1, enthusiastic1, tired1, "
                        "loving1, frustrated1, relieved1, impatient1, attentive1.",
                    },
                },
                "required": ["emotion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_dance",
            "description": "Play a dance animation. Use when the user asks you to dance or when the mood is fun.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dance": {
                        "type": "string",
                        "description": "Dance name. Options: groovy_sway_and_roll, side_to_side_sway, "
                        "yeah_nod, simple_nod, dizzy_spin, chicken_peck, head_tilt_roll, "
                        "pendulum_swing, jackson_square, grid_snap, polyrhythm_combo.",
                    },
                },
                "required": ["dance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nod_yes",
            "description": "Nod the head up and down to indicate agreement or 'yes'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shake_no",
            "description": "Shake the head left and right to indicate disagreement or 'no'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiggle_antennas",
            "description": "Wiggle the antennas playfully. Use for excitement or greeting.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_body",
            "description": "Rotate the robot body left or right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees. Positive = turn left, negative = turn right. Range: -45 to 45.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds. Default 0.8.",
                    },
                },
                "required": ["angle"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the event loop, raising if not set."""
    assert _loop is not None, "Event loop not initialized"
    return _loop


def _execute_tool(name: str, args: dict[str, Any], backend: Backend) -> str:
    """Execute a robot control tool. Returns a result string for the LLM."""
    try:
        if name == "move_head":
            _exec_move_head(backend, args)
            return "Head moved."
        elif name == "play_emotion":
            _exec_play_recorded(backend, "pollen-robotics/reachy-mini-emotions-library", args.get("emotion", ""))
            return f"Played emotion: {args.get('emotion')}"
        elif name == "play_dance":
            _exec_play_recorded(backend, "pollen-robotics/reachy-mini-dances-library", args.get("dance", ""))
            return f"Played dance: {args.get('dance')}"
        elif name == "nod_yes":
            _exec_nod(backend)
            return "Nodded yes."
        elif name == "shake_no":
            _exec_shake(backend)
            return "Shook head no."
        elif name == "wiggle_antennas":
            _exec_wiggle_antennas(backend)
            return "Wiggled antennas."
        elif name == "turn_body":
            _exec_turn_body(backend, args)
            return f"Body turned {args.get('angle', 0)} degrees."
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return f"Tool error: {e}"


def _exec_move_head(backend: Backend, args: dict[str, Any]) -> None:
    """Move head to a roll/pitch/yaw position."""
    from scipy.spatial.transform import Rotation as R

    roll = float(args.get("roll", 0))
    pitch = float(args.get("pitch", 0))
    yaw = float(args.get("yaw", 0))
    duration = float(args.get("duration", 0.8))

    # Clamp values
    roll = max(-30, min(30, roll))
    pitch = max(-30, min(30, pitch))
    yaw = max(-45, min(45, yaw))
    duration = max(0.3, min(3.0, duration))

    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()

    asyncio.run_coroutine_threadsafe(
        backend.goto_target(head=pose, duration=duration), _get_loop()
    ).result(timeout=duration + 2)


def _exec_play_recorded(backend: Backend, dataset: str, move_name: str) -> None:
    """Play a recorded move from a HuggingFace dataset."""
    from reachy_mini.motion.recorded_move import RecordedMoves

    moves = RecordedMoves(dataset)
    available = moves.list_moves()
    if move_name not in available:
        logger.warning("Move %r not found in %s. Available: %s", move_name, dataset, available[:5])
        return

    move = moves.get(move_name)
    asyncio.run_coroutine_threadsafe(
        backend.play_move(move), _get_loop()
    ).result(timeout=move.duration + 2)


def _exec_nod(backend: Backend) -> None:
    """Nod head up and down twice."""
    from scipy.spatial.transform import Rotation as R

    for _ in range(2):
        pose_down = np.eye(4)
        pose_down[:3, :3] = R.from_euler("xyz", [0, -12, 0], degrees=True).as_matrix()
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(head=pose_down, duration=0.2), _get_loop()
        ).result(timeout=2)
        time.sleep(0.25)

        pose_up = np.eye(4)
        pose_up[:3, :3] = R.from_euler("xyz", [0, 5, 0], degrees=True).as_matrix()
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(head=pose_up, duration=0.2), _get_loop()
        ).result(timeout=2)
        time.sleep(0.25)

    # Return to neutral
    asyncio.run_coroutine_threadsafe(
        backend.goto_target(head=np.eye(4), duration=0.3), _get_loop()
    ).result(timeout=2)


def _exec_shake(backend: Backend) -> None:
    """Shake head left and right twice."""
    from scipy.spatial.transform import Rotation as R

    for _ in range(2):
        pose_left = np.eye(4)
        pose_left[:3, :3] = R.from_euler("xyz", [0, 0, 15], degrees=True).as_matrix()
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(head=pose_left, duration=0.2), _get_loop()
        ).result(timeout=2)
        time.sleep(0.25)

        pose_right = np.eye(4)
        pose_right[:3, :3] = R.from_euler("xyz", [0, 0, -15], degrees=True).as_matrix()
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(head=pose_right, duration=0.2), _get_loop()
        ).result(timeout=2)
        time.sleep(0.25)

    asyncio.run_coroutine_threadsafe(
        backend.goto_target(head=np.eye(4), duration=0.3), _get_loop()
    ).result(timeout=2)


def _exec_wiggle_antennas(backend: Backend) -> None:
    """Wiggle antennas back and forth."""
    for _ in range(3):
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(antennas=np.array([-0.5, 0.5]), duration=0.15), _get_loop()
        ).result(timeout=2)
        time.sleep(0.2)
        asyncio.run_coroutine_threadsafe(
            backend.goto_target(antennas=np.array([0.3, -0.3]), duration=0.15), _get_loop()
        ).result(timeout=2)
        time.sleep(0.2)

    # Return to default
    asyncio.run_coroutine_threadsafe(
        backend.goto_target(antennas=np.array([-0.1745, 0.1745]), duration=0.3), _get_loop()
    ).result(timeout=2)


def _exec_turn_body(backend: Backend, args: dict[str, Any]) -> None:
    """Turn robot body left or right."""
    import math

    angle_deg = float(args.get("angle", 0))
    duration = float(args.get("duration", 0.8))

    angle_deg = max(-45, min(45, angle_deg))
    duration = max(0.3, min(3.0, duration))
    angle_rad = math.radians(angle_deg)

    asyncio.run_coroutine_threadsafe(
        backend.goto_target(body_yaw=angle_rad, duration=duration), _get_loop()
    ).result(timeout=duration + 2)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_history: list[dict[str, Any]] = []
_session: aiohttp.ClientSession | None = None
_whisper_model: Any = None
_whisper_loading: bool = False
_conv_running: bool = False
_conv_thread: threading.Thread | None = None
_conv_backend: Backend | None = None
_ws_clients: list[WebSocket] = []
_loop: asyncio.AbstractEventLoop | None = None


def _trim_history() -> None:
    """Keep only the most recent MAX_HISTORY messages."""
    global _history
    if len(_history) > MAX_HISTORY:
        _history = _history[-MAX_HISTORY:]


async def _get_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp session."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


def _get_whisper() -> Any:
    """Lazy-load the faster-whisper model."""
    global _whisper_model, _whisper_loading
    if _whisper_model is not None:
        return _whisper_model
    if _whisper_loading:
        return None
    _whisper_loading = True
    try:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model (base)...")
        _whisper_model = WhisperModel("base", compute_type="int8")
        logger.info("Whisper model loaded successfully.")
        return _whisper_model
    except Exception:
        logger.exception("Failed to load Whisper model")
        _whisper_loading = False
        return None


def _broadcast(event: str, data: str = "") -> None:
    """Send a status event to all connected WebSocket clients."""
    if _loop is None:
        return
    msg = json.dumps({"event": event, "data": data})
    for ws in list(_ws_clients):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), _loop)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP session with retry + SSL tolerance
# ---------------------------------------------------------------------------
_http_session: Any = None


def _get_http_session() -> Any:
    """Get or create the shared requests session."""
    global _http_session
    if _http_session is None:
        import requests
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _http_session = session
    return _http_session


# ---------------------------------------------------------------------------
# LLM call with tool calling
# ---------------------------------------------------------------------------
def _call_llm_sync(text: str, image: str | None = None, backend: Backend | None = None) -> str:
    """Call OpenRouter with tool support. Execute tools and return final reply."""
    if image:
        content: list[dict[str, Any]] | str = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
        ]
    else:
        content = text
    _history.append({"role": "user", "content": content})
    _trim_history()

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *_history]
    session = _get_http_session()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/pollen-robotics/reachy_mini",
        "X-Title": "Reachy Mini Chat",
    }

    # Allow up to 3 rounds of tool calling
    for _round in range(3):
        payload: dict[str, Any] = {"model": MODEL_NAME, "messages": messages}
        if backend is not None:
            payload["tools"] = LLM_TOOLS
            payload["tool_choice"] = "auto"

        try:
            resp = session.post(
                f"{API_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
                verify=False,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("LLM call failed")
            return "Sorry, I had trouble thinking. Could you try again?"

        choice = data["choices"][0]
        message = choice["message"]

        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls and backend is not None:
            # Add assistant message with tool calls to history
            messages.append(message)

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    fn_args = {}

                logger.info("Executing tool: %s(%s)", fn_name, fn_args)
                _broadcast("status", f"Moving: {fn_name}...")
                result = _execute_tool(fn_name, fn_args, backend)
                logger.info("Tool result: %s", result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            # Continue loop to get final text reply
            continue

        # No tool calls — this is the final text reply
        reply = message.get("content", "")
        _history.append({"role": "assistant", "content": reply})
        _trim_history()
        return str(reply)

    return "I got a bit confused. Let me try again."


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _tts_and_play_sync(text: str, backend: Backend) -> None:
    """Call ElevenLabs TTS and play through robot speaker (blocking)."""
    import soundfile as sf

    if not ELEVENLABS_API_KEY or not backend.audio:
        return

    session = _get_http_session()

    try:
        resp = session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=pcm_16000",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=30,
            verify=False,
        )
        resp.raise_for_status()

        pcm = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, pcm, SAMPLE_RATE)
            backend.audio.play_sound(tmp.name)
            time.sleep(len(pcm) / SAMPLE_RATE + 0.5)
        finally:
            os.unlink(tmp.name)
    except Exception:
        logger.exception("TTS failed")


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------
def _conversation_loop() -> None:
    """Continuously listen, detect speech, transcribe, respond with movement, repeat."""
    global _conv_running

    backend = _conv_backend
    if backend is None or backend.audio is None:
        _broadcast("error", "No audio backend available")
        _conv_running = False
        return

    audio = backend.audio

    _broadcast("status", "Loading speech model...")
    model = _get_whisper()
    if model is None:
        _broadcast("error", "Failed to load Whisper model")
        _conv_running = False
        return

    _broadcast("status", "Ready! Start talking...")
    logger.info("Conversation loop started")

    while _conv_running:
        try:
            speech_chunks: list[np.ndarray[Any, np.dtype[np.floating[Any]]]] = []
            speech_active = False
            silence_start = 0.0

            audio.start_recording()
            _broadcast("status", "Listening...")

            while _conv_running:
                sample = audio.get_audio_sample()
                if sample is None:
                    time.sleep(0.02)
                    continue

                rms = float(np.sqrt(np.mean(sample ** 2)))

                if rms > SPEECH_ENERGY_THRESHOLD:
                    if not speech_active:
                        speech_active = True
                        _broadcast("status", "Hearing you...")
                        logger.info("Speech detected (rms=%.4f)", rms)
                    speech_chunks.append(sample)
                    silence_start = 0.0
                elif speech_active:
                    speech_chunks.append(sample)
                    if silence_start == 0.0:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start > SILENCE_DURATION:
                        break

            audio.stop_recording()

            if not _conv_running:
                break
            if not speech_chunks:
                continue

            total_samples = sum(c.shape[0] for c in speech_chunks)
            duration = total_samples / SAMPLE_RATE
            if duration < MIN_SPEECH_DURATION:
                continue

            _broadcast("status", "Transcribing...")
            audio_data = np.concatenate(speech_chunks, axis=0)
            if audio_data.ndim == 2 and audio_data.shape[1] > 1:
                audio_mono = audio_data.mean(axis=1).astype(np.float32)
            else:
                audio_mono = audio_data.flatten().astype(np.float32)

            segments, _info = model.transcribe(audio_mono, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("Transcription: %r", text)

            if not text or len(text) < 2:
                continue

            _broadcast("user", text)

            # LLM with tool calling (passes backend for movement execution)
            _broadcast("status", "Thinking...")
            reply = _call_llm_sync(text, backend=backend)
            logger.info("LLM reply: %r", reply)
            _broadcast("assistant", reply)

            # TTS
            _broadcast("status", "Speaking...")
            _tts_and_play_sync(reply, backend)

        except Exception:
            logger.exception("Error in conversation loop")
            _broadcast("error", "Something went wrong. Resuming...")
            time.sleep(1)

    _broadcast("status", "Conversation stopped.")
    logger.info("Conversation loop ended")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatConfig(BaseModel):
    """Configuration status."""

    model: str
    openrouter_configured: bool
    elevenlabs_configured: bool
    whisper_loaded: bool
    conversation_active: bool


class ChatRequest(BaseModel):
    """Request body for text chat."""

    message: str
    image: str | None = None


class ChatResponse(BaseModel):
    """Response body for chat."""

    response: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/config")
async def chat_config_route() -> ChatConfig:
    """Return current configuration status."""
    return ChatConfig(
        model=MODEL_NAME,
        openrouter_configured=bool(API_KEY),
        elevenlabs_configured=bool(ELEVENLABS_API_KEY),
        whisper_loaded=_whisper_model is not None,
        conversation_active=_conv_running,
    )


@router.post("/conversation/start")
async def conversation_start(request: Request) -> dict[str, str]:
    """Start the continuous conversation loop."""
    global _conv_running, _conv_thread, _conv_backend, _loop
    if _conv_running:
        return {"status": "already_running"}

    backend = request.app.state.daemon.backend
    if backend is None or not backend.ready.is_set():
        return {"status": "error", "message": "Backend not ready"}

    _conv_backend = backend
    _conv_running = True
    _loop = asyncio.get_event_loop()
    _conv_thread = threading.Thread(target=_conversation_loop, daemon=True)
    _conv_thread.start()
    return {"status": "started"}


@router.post("/conversation/stop")
async def conversation_stop() -> dict[str, str]:
    """Stop the continuous conversation loop."""
    global _conv_running, _conv_thread
    if not _conv_running:
        return {"status": "not_running"}

    _conv_running = False
    if _conv_thread is not None:
        _conv_thread.join(timeout=5.0)
        _conv_thread = None
    return {"status": "stopped"}


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket for real-time conversation status and messages."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "text":
                text = msg.get("text", "").strip()
                image = msg.get("image")
                if text:
                    _broadcast("user", text)
                    backend = _conv_backend or websocket.app.state.daemon.backend
                    reply = await asyncio.get_event_loop().run_in_executor(
                        None, _call_llm_sync, text, image, backend
                    )
                    _broadcast("assistant", reply)
                    if backend and backend.audio:
                        asyncio.get_event_loop().run_in_executor(None, _tts_and_play_sync, reply, backend)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@router.post("/completions")
async def chat_completions(req: ChatRequest) -> ChatResponse:
    """Send a text message to the LLM."""
    if not API_KEY:
        return ChatResponse(response="API key not configured.")

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, _call_llm_sync, req.message, req.image, _conv_backend)
    return ChatResponse(response=reply)


@router.post("/reset")
async def chat_reset() -> dict[str, str]:
    """Clear conversation history."""
    _history.clear()
    return {"status": "ok"}
