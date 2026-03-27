/* Voice & Vision Chat for Reachy Mini
 *
 * STT: SenseVoice via robot mic (server-side)
 * TTS: ElevenLabs via robot speaker (server-side)
 * LLM: OpenRouter (server-side)
 */

const State = { IDLE: "idle", RECORDING: "recording", PROCESSING: "processing", SPEAKING: "speaking" };
let state = State.IDLE;

// DOM refs
let messagesEl, inputEl, sendBtn, micBtn, photoToggle, statusEl, videoEl, canvasEl;

// Camera stream
let cameraStream = null;

// ── Helpers ──────────────────────────────────────────────────────────────

function setState(s) {
  state = s;
  micBtn.classList.toggle("active", s === State.RECORDING);
  statusEl.textContent =
    s === State.RECORDING  ? "Recording from robot mic..." :
    s === State.PROCESSING ? "Thinking..." :
    s === State.SPEAKING   ? "Speaking through robot..." : "";
  inputEl.disabled = s !== State.IDLE;
  sendBtn.disabled = s !== State.IDLE;
  micBtn.disabled  = s === State.PROCESSING || s === State.SPEAKING;
}

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = "chat-msg " + role;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Camera ───────────────────────────────────────────────────────────────

async function initCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    videoEl.srcObject = cameraStream;
    videoEl.play();
  } catch (e) {
    console.warn("Camera not available:", e);
    photoToggle.disabled = true;
    photoToggle.checked = false;
  }
}

function captureFrame() {
  if (!cameraStream) return null;
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;
  canvasEl.getContext("2d").drawImage(videoEl, 0, 0);
  return canvasEl.toDataURL("image/jpeg", 0.7).split(",")[1];
}

// ── STT (server-side via robot mic + SenseVoice) ─────────────────────────

async function toggleMic() {
  if (state === State.RECORDING) {
    // Stop recording and transcribe
    setState(State.PROCESSING);
    try {
      const resp = await fetch("/api/chat/stt/stop", { method: "POST" });
      const data = await resp.json();
      if (data.text && data.text.trim()) {
        inputEl.value = data.text;
        await sendMessage(data.text);
      } else {
        appendMessage("assistant", "Could not hear anything. Try again.");
        setState(State.IDLE);
      }
    } catch (e) {
      appendMessage("assistant", "STT error. Check server logs.");
      setState(State.IDLE);
    }
  } else if (state === State.IDLE) {
    // Start recording
    try {
      const resp = await fetch("/api/chat/stt/start", { method: "POST" });
      const data = await resp.json();
      if (data.status === "recording" || data.status === "already_recording") {
        setState(State.RECORDING);
      }
    } catch (e) {
      appendMessage("assistant", "Could not start recording.");
    }
  }
}

// ── TTS (server-side via ElevenLabs + robot speaker) ─────────────────────

async function speakOnRobot(text) {
  setState(State.SPEAKING);
  try {
    await fetch("/api/chat/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (e) {
    console.warn("TTS playback error:", e);
  }
  setState(State.IDLE);
}

// ── Send message to LLM ─────────────────────────────────────────────────

async function sendMessage(text) {
  if (!text.trim()) return;
  appendMessage("user", text);
  inputEl.value = "";
  setState(State.PROCESSING);

  const body = { message: text };
  if (photoToggle.checked) {
    const img = captureFrame();
    if (img) body.image = img;
  }

  try {
    const resp = await fetch("/api/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    appendMessage("assistant", data.response);
    // Fire TTS in background (don't block UI)
    speakOnRobot(data.response);
  } catch (e) {
    appendMessage("assistant", "Error: could not reach the server.");
    setState(State.IDLE);
  }
}

async function resetChat() {
  await fetch("/api/chat/reset", { method: "POST" });
  messagesEl.innerHTML = "";
  appendMessage("assistant", "Conversation reset. How can I help you?");
}

// ── Init ─────────────────────────────────────────────────────────────────

async function init() {
  messagesEl  = document.getElementById("messages");
  inputEl     = document.getElementById("chat-input");
  sendBtn     = document.getElementById("send-btn");
  micBtn      = document.getElementById("mic-btn");
  photoToggle = document.getElementById("photo-toggle");
  statusEl    = document.getElementById("chat-status");
  videoEl     = document.getElementById("camera-preview");
  canvasEl    = document.getElementById("capture-canvas");

  // Check config
  try {
    const resp = await fetch("/api/chat/config");
    const cfg = await resp.json();
    const issues = [];
    if (!cfg.openrouter_configured) issues.push("OPENAI_API_KEY");
    if (!cfg.elevenlabs_configured) issues.push("ELEVENLABS_API_KEY");
    if (issues.length > 0) {
      appendMessage("assistant", "Missing in .env: " + issues.join(", ") + ". Set them and restart the daemon.");
    } else {
      appendMessage("assistant",
        "Hi! I'm Reachy Mini (" + cfg.model + "). " +
        "Type or hold the mic button to talk. Toggle camera to share what I see."
      );
    }
    if (!cfg.whisper_loaded) {
      appendMessage("assistant", "Whisper model will load on first mic use (may take a moment).");
    }
  } catch {
    appendMessage("assistant", "Could not connect to the chat API.");
  }

  await initCamera();

  sendBtn.addEventListener("click", () => sendMessage(inputEl.value));
  inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(inputEl.value); });
  micBtn.addEventListener("click", toggleMic);
  document.getElementById("reset-btn").addEventListener("click", resetChat);

  photoToggle.addEventListener("change", () => {
    videoEl.style.display = photoToggle.checked ? "block" : "none";
  });
}

window.addEventListener("DOMContentLoaded", init);
