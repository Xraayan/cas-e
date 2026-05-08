import asyncio
import contextlib
import json
import os
import re
import logging
import time
import platform
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION
from tool_compat import RunContext, ToolRegistry
from tools import query_college_info, query_ec_faq, search_web, show_timetable
from website_context import get_context, init_context

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("casie")

MODEL = os.getenv("CASIE_MODEL", "gemini-3.1-flash-live-preview")
SAMPLE_RATE = 16000
OUTPUT_RATE = 24000
INPUT_BLOCK_MS = int(os.getenv("CASIE_INPUT_BLOCK_MS", "10"))
SILENCE_DURATION_MS = int(os.getenv("CASIE_VAD_SILENCE_MS", "420"))
PREFIX_PADDING_MS = int(os.getenv("CASIE_VAD_PREFIX_MS", "180"))
STARTUP_GREETING_ENABLED = os.getenv("CASIE_STARTUP_GREETING_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GREETING_TEXT = os.getenv(
    "CASIE_GREETING_TEXT",
    "Welcome to the Electronics and Communication Department. I am CASie. How can I help you today?",
)
ENGLISH_ONLY_MODE = os.getenv("CASIE_ENGLISH_ONLY", "true").strip().lower() in {"1", "true", "yes", "on"}
INPUT_DEVICE_HINT = os.getenv("CASIE_INPUT_DEVICE", "").strip()
OUTPUT_DEVICE_HINT = os.getenv("CASIE_OUTPUT_DEVICE", "").strip()
DEBUG_AUDIO = os.getenv("CASIE_DEBUG_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}
TURN_HINT_ENABLED = os.getenv("CASIE_TURN_HINT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
TURN_HINT_RMS = float(os.getenv("CASIE_TURN_HINT_RMS", "800"))
TURN_HINT_IDLE_MS = int(os.getenv("CASIE_TURN_HINT_IDLE_MS", "350"))
TURN_HINT_COOLDOWN_MS = int(os.getenv("CASIE_TURN_HINT_COOLDOWN_MS", "350"))
CLIENT_AUDIO_STREAM_END_ENABLED = os.getenv("CASIE_CLIENT_AUDIO_STREAM_END_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DROP_INPUT_WHILE_ASSISTANT = os.getenv("CASIE_DROP_INPUT_WHILE_ASSISTANT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ASSISTANT_AUDIO_HOLD_MS = int(os.getenv("CASIE_ASSISTANT_AUDIO_HOLD_MS", "650"))
FORCE_ECHO_GUARD = os.getenv("CASIE_FORCE_ECHO_GUARD", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIC_BLOCK_WHILE_ASSISTANT = FORCE_ECHO_GUARD or DROP_INPUT_WHILE_ASSISTANT
MIC_BLOCK_HOLD_MS = max(650, ASSISTANT_AUDIO_HOLD_MS) if FORCE_ECHO_GUARD else ASSISTANT_AUDIO_HOLD_MS
MIC_NOISE_GATE_ENABLED = os.getenv("CASIE_MIC_NOISE_GATE", "false").strip().lower() in {"1", "true", "yes", "on"}
MIC_NOISE_GATE_RMS = float(os.getenv("CASIE_MIC_NOISE_GATE_RMS", "180"))
MIC_NOISE_GATE_HANGOVER_MS = int(os.getenv("CASIE_MIC_NOISE_GATE_HANGOVER_MS", "260"))
MIC_NOISE_GATE_LEARN_MS = int(os.getenv("CASIE_MIC_NOISE_GATE_LEARN_MS", "220"))
MIC_NOISE_GATE_MULTIPLIER = float(os.getenv("CASIE_MIC_NOISE_GATE_MULTIPLIER", "2.2"))
ALLOW_ALPHA_API = os.getenv("CASIE_ALLOW_ALPHA_API", "false").strip().lower() in {"1", "true", "yes", "on"}


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    if machine not in {"armv7l", "aarch64", "arm64"}:
        return False

    model_paths = (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model"))
    for path in model_paths:
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore").strip("\x00").lower()
                if "raspberry pi" in text:
                    return True
        except Exception:
            continue

    return "raspberry" in platform.platform().lower()


IS_RASPBERRY_PI = _is_raspberry_pi()
PI_COMPAT_ENABLED = os.getenv("CASIE_PI_COMPAT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
INPUT_LATENCY = os.getenv("CASIE_INPUT_LATENCY", "low")
OUTPUT_LATENCY = os.getenv("CASIE_OUTPUT_LATENCY", "low")
PI_STABILIZE_MS = int(os.getenv("CASIE_PI_STABILIZE_MS", "12"))
COMMON_AUDIO_RATES = (48000, 44100, 32000, 24000, 22050, 16000)


def _start_sensitivity(value: str) -> types.StartSensitivity:
    return (
        types.StartSensitivity.START_SENSITIVITY_LOW
        if value.strip().lower() == "low"
        else types.StartSensitivity.START_SENSITIVITY_HIGH
    )


def _end_sensitivity(value: str) -> types.EndSensitivity:
    return (
        types.EndSensitivity.END_SENSITIVITY_LOW
        if value.strip().lower() == "low"
        else types.EndSensitivity.END_SENSITIVITY_HIGH
    )


def _build_quick_context_summary() -> str:
    try:
        init_context()
        data = get_context()
    except Exception as exc:
        logger.warning("Could not load website context snapshot: %s", exc)
        return ""

    college = data.get("college", {})
    principal = data.get("principal", {})
    contact = data.get("contact", {})
    ece = data.get("ece_department", {})
    ece_hod = ece.get("hod", {}) if isinstance(ece.get("hod"), dict) else {}

    lines: list[str] = []
    if college.get("name"):
        lines.append(f"College: {college.get('name')}")
    if college.get("short_name"):
        lines.append(f"Short name: {college.get('short_name')}")
    if college.get("location"):
        lines.append(f"Location: {college.get('location')}")
    if college.get("website"):
        lines.append(f"Website: {college.get('website')}")

    if principal.get("name"):
        lines.append(f"Principal: {principal.get('name')}")
    if principal.get("email"):
        lines.append(f"Principal email: {principal.get('email')}")
    if principal.get("phone"):
        lines.append(f"Principal phone: {principal.get('phone')}")

    if contact.get("office_phone"):
        lines.append(f"Office phone: {contact.get('office_phone')}")
    if contact.get("college_email"):
        lines.append(f"College email: {contact.get('college_email')}")
    if contact.get("ece_hod_email"):
        lines.append(f"ECE HOD email: {contact.get('ece_hod_email')}")

    if ece.get("name"):
        lines.append(f"ECE department: {ece.get('name')}")
    if ece_hod.get("name"):
        lines.append(f"ECE HOD: {ece_hod.get('name')}")
    if ece_hod.get("email"):
        lines.append(f"ECE HOD contact email: {ece_hod.get('email')}")

    return "\n".join(lines)


def _build_system_instruction() -> str:
    parts = [
        AGENT_INSTRUCTION.strip(),
        SESSION_INSTRUCTION.strip(),
    ]

    quick_context = _build_quick_context_summary()
    if quick_context:
        parts.append("Verified RIT context:\n" + quick_context)

    parts.append(
        "Use available tools for college questions instead of guessing. "
        "Prefer precise, short answers that are accurate for RIT Kottayam."
    )
    parts.append(
        "For EC department room/lab/building/floor/project-coordinator/student-count questions, "
        "prefer query_ec_faq before query_college_info."
    )

    if ENGLISH_ONLY_MODE:
        parts.append(
            "Always communicate in English only. If speech is unclear or sounds like "
            "background noise, ask the user to repeat it in English instead of guessing."
        )
    else:
        parts.append(
            "Do not force English. Detect the user's spoken language from the audio "
            "and reply in the same language unless the user asks otherwise."
        )

    return "\n\n".join(part for part in parts if part)


def _create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many([search_web, query_college_info, show_timetable, query_ec_faq])
    logger.info("Registered tools: %s", ", ".join(registry.names) if registry.names else "none")
    return registry


def _safe_tool_args(raw_args: object) -> dict[str, object]:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        with contextlib.suppress(Exception):
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
    return {}


_EC_FAQ_ROUTING_HINTS = (
    "where",
    "lab",
    "room",
    "floor",
    "building",
    "seminar",
    "workshop",
    "project coordinator",
    "student count",
    "how many students",
)

_EC_FAQ_TOPIC_HINTS = (
    "digital lab",
    "circuits lab",
    "communication lab",
    "electronics workshop",
    "lecture hall 1",
    "professors room",
    "systems lab",
    "hod cabin",
    "staff room",
    "project lab",
    "robotics lab",
    "pg lab",
    "casp lab",
)

_EC_SCOPE_HINTS = (
    " ec ",
    " ece ",
    "electronics",
    "communication",
    "rai",
)


def _should_route_to_ec_faq(tool_name: str, args: dict[str, object]) -> bool:
    if tool_name != "query_college_info":
        return False

    query = str(args.get("query", "") or "").strip().lower()
    if not query:
        return False

    padded_query = f" {query} "
    has_faq_shape = any(hint in query for hint in _EC_FAQ_ROUTING_HINTS)
    has_ec_scope = any(hint in padded_query for hint in _EC_SCOPE_HINTS)
    has_ec_topic = any(hint in query for hint in _EC_FAQ_TOPIC_HINTS)
    return (has_faq_shape and has_ec_scope) or has_ec_topic


async def _respond_to_tool_calls(
    session,
    tool_registry: ToolRegistry,
    run_context: RunContext,
    function_calls: list[types.FunctionCall],
) -> None:
    if not function_calls:
        return

    deduped: list[types.FunctionCall] = []
    seen_ids: set[str] = set()
    for call in function_calls:
        call_id = str(getattr(call, "id", "") or "")
        if call_id and call_id in seen_ids:
            continue
        if call_id:
            seen_ids.add(call_id)
        deduped.append(call)

    responses: list[types.FunctionResponse] = []
    for call in deduped:
        if bool(getattr(call, "will_continue", False)):
            continue

        name = str(getattr(call, "name", "") or "").strip() or "unknown_tool"
        call_id = str(getattr(call, "id", "") or "")
        args = _safe_tool_args(getattr(call, "args", {}))

        resolved_name = "query_ec_faq" if _should_route_to_ec_faq(name, args) else name
        if resolved_name != name:
            logger.info("Tool call rerouted: %s -> %s for query=%r", name, resolved_name, args.get("query", ""))

        logger.info("Tool call requested: %s(%s)", resolved_name, args)
        result = await tool_registry.execute(resolved_name, args, run_context)

        payload: dict[str, object] = {"name": name, "response": {"result": result}}
        if call_id:
            payload["id"] = call_id
        responses.append(types.FunctionResponse(**payload))

    if not responses:
        return

    await session.send_tool_response(function_responses=responses)
    logger.info("Sent %d tool response(s)", len(responses))


async def _send_text_turn(session, text: str) -> None:
    clean = text.strip()
    if not clean:
        return

    await session.send_realtime_input(text=clean)


async def _send_startup_greeting(session) -> None:
    clean = GREETING_TEXT.strip()
    if not clean:
        return

    await _send_text_turn(
        session,
        (
            "Say only this opening welcome line now, then wait for the visitor's "
            f'first question: "{clean}"'
        ),
    )


def _build_config(tool_registry: ToolRegistry) -> types.LiveConnectConfig:
    start_mode = os.getenv("CASIE_SERVER_START_SENSITIVITY", "high")
    end_mode = os.getenv("CASIE_SERVER_END_SENSITIVITY", "high")
    system_instruction = _build_system_instruction()
    gemini_tools = tool_registry.to_gemini_tools()

    logger.info(
        "Live config: model=%s tools=%d start_sensitivity=%s end_sensitivity=%s silence_ms=%d",
        MODEL,
        len(tool_registry.names),
        start_mode,
        end_mode,
        SILENCE_DURATION_MS,
    )

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=gemini_tools,
        realtime_input_config=types.RealtimeInputConfig(
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=_start_sensitivity(start_mode),
                end_of_speech_sensitivity=_end_sensitivity(end_mode),
                prefix_padding_ms=PREFIX_PADDING_MS,
                silence_duration_ms=SILENCE_DURATION_MS,
            ),
        ),
    )


def _stream_kwargs(io_kind: str) -> dict[str, object]:
    if not PI_COMPAT_ENABLED:
        return {}
    # Optional Pi mode: callback stream tweaks for lower-power boards.
    kwargs: dict[str, object] = {
        "latency": INPUT_LATENCY if io_kind == "input" else OUTPUT_LATENCY,
    }
    if IS_RASPBERRY_PI:
        kwargs["clip_off"] = True
        kwargs["dither_off"] = True
    return kwargs


def _safe_query_device(device_idx: int | None, io_kind: str) -> dict[str, object]:
    kind = "output" if io_kind == "output" else "input"
    try:
        if device_idx is not None:
            dev = sd.query_devices(device_idx, kind)
            if isinstance(dev, dict):
                return dev
    except Exception:
        pass

    try:
        defaults = sd.default.device
        if isinstance(defaults, (list, tuple)) and len(defaults) >= 2:
            fallback_idx = int(defaults[1] if kind == "output" else defaults[0])
            if fallback_idx >= 0:
                dev = sd.query_devices(fallback_idx, kind)
                if isinstance(dev, dict):
                    return dev
    except Exception:
        pass

    return {}


def _device_supports_rate(device_idx: int, io_kind: str, rate: int) -> bool:
    checker = sd.check_output_settings if io_kind == "output" else sd.check_input_settings
    try:
        checker(device=device_idx, channels=1, dtype="float32", samplerate=rate)
        return True
    except Exception:
        return False


def _hostapi_name(device: dict[str, object], hostapis: object) -> str:
    if not isinstance(hostapis, (list, tuple)):
        return ""
    try:
        hostapi_idx = int(device.get("hostapi", -1))
        if 0 <= hostapi_idx < len(hostapis):
            return str(hostapis[hostapi_idx].get("name", "")).lower()
    except Exception:
        pass
    return ""


def _is_mapper_device(device: dict[str, object]) -> bool:
    name = str(device.get("name", "")).lower()
    return "sound mapper" in name or name.startswith("primary ")


def _first_device_with_channels(io_kind: str, preferred_rate: int | None = None) -> int | None:
    want_output = io_kind == "output"
    key = "max_output_channels" if want_output else "max_input_channels"
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not query %s devices: %s", io_kind, exc)
        return None

    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []

    candidates = [
        (idx, dev)
        for idx, dev in enumerate(devices)
        if int(dev.get(key, 0)) > 0
    ]

    def supports(idx: int) -> bool:
        return preferred_rate is None or _device_supports_rate(idx, io_kind, preferred_rate)

    try:
        defaults = sd.default.device
        default_idx = int(defaults[1] if want_output else defaults[0])
        if default_idx >= 0:
            default_dev = devices[default_idx]
            if int(default_dev.get(key, 0)) > 0 and supports(default_idx):
                return default_idx
    except Exception:
        pass

    if platform.system().lower() == "windows":
        preferred_hosts = ("directsound", "wasapi", "wdm-ks", "mme")
        for host_name in preferred_hosts:
            for idx, dev in candidates:
                if _is_mapper_device(dev) or not supports(idx):
                    continue
                if host_name in _hostapi_name(dev, hostapis):
                    return idx

    for idx, _dev in candidates:
        if supports(idx):
            return idx

    for idx, dev in candidates:
        if not _is_mapper_device(dev):
            return idx
    if candidates:
        return candidates[0][0]
    return None


def _ensure_audio_runtime_ready() -> None:
    try:
        sd.query_devices()
    except Exception as exc:
        hint = (
            "On Raspberry Pi, install audio libs: "
            "`sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev`"
            if IS_RASPBERRY_PI
            else "Ensure PortAudio is installed and audio devices are available."
        )
        raise RuntimeError(f"Audio backend is unavailable: {exc}. {hint}") from exc

    in_dev = _first_device_with_channels("input")
    out_dev = _first_device_with_channels("output")
    if in_dev is None or out_dev is None:
        raise RuntimeError(
            "No usable audio devices detected. "
            "Please connect/configure both microphone and speaker output."
        )


def _restart_output_stream(
    stream: sd.OutputStream,
    rate: int,
    device_idx: int | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> sd.OutputStream:
    try:
        stream.stop()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass
    kwargs = dict(extra_kwargs or {})
    if device_idx is not None:
        kwargs["device"] = device_idx
    new_stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", **kwargs)
    new_stream.start()
    return new_stream


def _device_default_rate(device_idx: int | None, io_kind: str) -> int | None:
    dev = _safe_query_device(device_idx, io_kind)
    if not dev:
        return None
    try:
        rate = int(round(float(dev.get("default_samplerate", 0.0))))
    except Exception:
        return None
    return rate if rate > 0 else None


def _choose_stream_rate(device_idx: int | None, io_kind: str, preferred_rate: int) -> int:
    if device_idx is not None and _device_supports_rate(device_idx, io_kind, preferred_rate):
        return preferred_rate

    default_rate = _device_default_rate(device_idx, io_kind)
    if default_rate and (device_idx is None or _device_supports_rate(device_idx, io_kind, default_rate)):
        return default_rate

    for rate in COMMON_AUDIO_RATES:
        if device_idx is None or _device_supports_rate(device_idx, io_kind, rate):
            return rate

    return preferred_rate


def _resample_float32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    flat = samples.astype(np.float32, copy=False).reshape(-1)
    src_len = flat.shape[0]
    dst_len = max(1, int(round(src_len * (float(dst_rate) / float(src_rate)))))
    src_pos = np.linspace(0.0, src_len - 1, num=src_len, dtype=np.float32)
    dst_pos = np.linspace(0.0, src_len - 1, num=dst_len, dtype=np.float32)
    return np.interp(dst_pos, src_pos, flat).astype(np.float32)


def _pcm16_bytes_from_float32(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples.reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _resolve_device(device_hint: str, io_kind: str, preferred_rate: int | None = None) -> int | None:
    if not device_hint:
        return _first_device_with_channels(io_kind, preferred_rate)
    if device_hint.isdigit():
        idx = int(device_hint)
        if preferred_rate is None or _device_supports_rate(idx, io_kind, preferred_rate):
            return idx
        logger.warning(
            "%s device %s does not support %d Hz. Choosing a compatible %s device.",
            io_kind,
            idx,
            preferred_rate,
            io_kind,
        )
        return _first_device_with_channels(io_kind, preferred_rate)

    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not query %s devices: %s", io_kind, exc)
        return None

    want_output = io_kind == "output"
    needle = device_hint.lower()
    for idx, dev in enumerate(devices):
        channels = int(dev.get("max_output_channels" if want_output else "max_input_channels", 0))
        if channels <= 0:
            continue
        if needle in str(dev.get("name", "")).lower():
            if preferred_rate is None or _device_supports_rate(idx, io_kind, preferred_rate):
                return idx
            logger.warning(
                "%s device hint %r matched %s but not %d Hz. Choosing a compatible %s device.",
                io_kind,
                device_hint,
                dev.get("name", idx),
                preferred_rate,
                io_kind,
            )
            return _first_device_with_channels(io_kind, preferred_rate)

    logger.warning("%s device hint %r not found. Using default %s device.", io_kind, device_hint, io_kind)
    return _first_device_with_channels(io_kind, preferred_rate)


def _is_expected_disconnect(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc)
    text_lower = text.lower()
    return (
        "ConnectionClosed" in name
        or "WebSocket" in name
        or "keepalive ping timeout" in text_lower
        or "timed out while closing connection" in text_lower
        or "abnormal closure" in text_lower
        or "close code 1006" in text_lower
        or "code = 1006" in text_lower
        or "1006 (abnormal closure)" in text_lower
    )


async def _run_session(client: genai.Client) -> None:
    tool_registry = _create_tool_registry()
    config = _build_config(tool_registry)
    stop_event = asyncio.Event()
    assistant_block_until_ts = 0.0
    input_device = _resolve_device(INPUT_DEVICE_HINT, "input", SAMPLE_RATE)
    output_device = _resolve_device(OUTPUT_DEVICE_HINT, "output", OUTPUT_RATE)
    input_stream_rate = _choose_stream_rate(input_device, "input", SAMPLE_RATE)
    output_stream_rate = _choose_stream_rate(output_device, "output", OUTPUT_RATE)
    input_stream_kwargs = _stream_kwargs("input")
    output_stream_kwargs = _stream_kwargs("output")
    run_context = RunContext(session_id="live-session")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("CASie is live! Speak now. Ctrl+C to exit.\n")
        logger.info(
            "Echo guard: active=%s hold_ms=%d (force=%s)",
            MIC_BLOCK_WHILE_ASSISTANT,
            MIC_BLOCK_HOLD_MS,
            FORCE_ECHO_GUARD,
        )

        if STARTUP_GREETING_ENABLED:
            await _send_startup_greeting(session)

        loop = asyncio.get_running_loop()
        mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        speaker_queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=256)

        def mic_callback(indata, frames, time_info, status):
            del frames, time_info
            if status:
                logger.debug("Mic status: %s", status)
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            if input_stream_rate != SAMPLE_RATE:
                mono = _resample_float32(mono, input_stream_rate, SAMPLE_RATE)
            pcm = _pcm16_bytes_from_float32(mono)

            def _push() -> None:
                if mic_queue.full():
                    try:
                        mic_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                mic_queue.put_nowait(pcm)

            loop.call_soon_threadsafe(_push)

        async def listen_audio() -> None:
            blocksize = int(input_stream_rate * (INPUT_BLOCK_MS / 1000.0))
            with sd.InputStream(
                samplerate=input_stream_rate,
                channels=1,
                dtype="float32",
                blocksize=max(1, blocksize),
                device=input_device,
                callback=mic_callback,
                **input_stream_kwargs,
            ):
                try:
                    dev = _safe_query_device(input_device, "input")
                    logger.info(
                        "Microphone open (device_rate=%d, api_rate=%d, device=%s)",
                        input_stream_rate,
                        SAMPLE_RATE,
                        dev.get("name", "default") if dev else "default",
                    )
                except Exception:
                    logger.info("Microphone open (device_rate=%d, api_rate=%d, device=default)", input_stream_rate, SAMPLE_RATE)
                if PI_COMPAT_ENABLED and IS_RASPBERRY_PI and PI_STABILIZE_MS > 0:
                    await asyncio.sleep(PI_STABILIZE_MS / 1000.0)
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)

        async def send_audio() -> None:
            nonlocal assistant_block_until_ts
            last_sent_ts = time.monotonic()
            stream_active = False
            last_rms_log_ts = time.monotonic()
            peak_rms = 0.0
            last_voice_ts = time.monotonic()
            last_voice_gate_ts = 0.0
            last_stream_end_ts = 0.0
            had_voice_since_last_end = False
            gate_started_ts = time.monotonic()
            noise_floor_samples: list[float] = []
            adaptive_gate_rms = MIC_NOISE_GATE_RMS
            gate_ready = not MIC_NOISE_GATE_ENABLED
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(mic_queue.get(), timeout=0.15)
                except asyncio.TimeoutError:
                    if (
                        CLIENT_AUDIO_STREAM_END_ENABLED
                        and stream_active
                        and (time.monotonic() - last_sent_ts) > 0.45
                    ):
                        try:
                            await session.send_realtime_input(audio_stream_end=True)
                        except Exception as exc:
                            if _is_expected_disconnect(exc) or stop_event.is_set():
                                logger.info("Input stream closed while sending stream_end.")
                                break
                            raise
                        stream_active = False
                        logger.debug("Sent audio_stream_end after mic idle pause.")
                    continue

                if MIC_BLOCK_WHILE_ASSISTANT and time.monotonic() < assistant_block_until_ts:
                    continue

                pcm = np.frombuffer(chunk, dtype=np.int16)
                rms = 0.0
                if pcm.size:
                    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))

                if MIC_NOISE_GATE_ENABLED and not gate_ready:
                    elapsed_ms = (time.monotonic() - gate_started_ts) * 1000.0
                    noise_floor_samples.append(rms)
                    if elapsed_ms < MIC_NOISE_GATE_LEARN_MS:
                        continue
                    if noise_floor_samples:
                        floor = float(np.percentile(np.array(noise_floor_samples, dtype=np.float32), 80))
                        adaptive_gate_rms = max(MIC_NOISE_GATE_RMS, floor * MIC_NOISE_GATE_MULTIPLIER)
                        logger.info(
                            "Mic noise gate calibrated: floor=%.1f threshold=%.1f",
                            floor,
                            adaptive_gate_rms,
                        )
                    gate_ready = True

                if rms >= max(TURN_HINT_RMS, adaptive_gate_rms):
                    last_voice_ts = time.monotonic()
                    had_voice_since_last_end = True

                if MIC_NOISE_GATE_ENABLED:
                    now = time.monotonic()
                    if rms >= adaptive_gate_rms:
                        last_voice_gate_ts = now
                    else:
                        hangover_ms = (now - last_voice_gate_ts) * 1000.0
                        if hangover_ms > MIC_NOISE_GATE_HANGOVER_MS:
                            continue

                if DEBUG_AUDIO:
                    peak_rms = max(peak_rms, rms)
                    now = time.monotonic()
                    if now - last_rms_log_ts >= 1.0:
                        logger.info("Mic debug: peak_rms=%.1f stream_active=%s", peak_rms, stream_active)
                        peak_rms = 0.0
                        last_rms_log_ts = now

                try:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
                    )
                except Exception as exc:
                    if _is_expected_disconnect(exc) or stop_event.is_set():
                        logger.info("Input stream closed while sending audio.")
                        break
                    raise
                last_sent_ts = time.monotonic()
                stream_active = True

                if TURN_HINT_ENABLED:
                    now = time.monotonic()
                    idle_ms = (now - last_voice_ts) * 1000.0
                    cooldown_ms = (now - last_stream_end_ts) * 1000.0
                    if (
                        had_voice_since_last_end
                        and idle_ms >= TURN_HINT_IDLE_MS
                        and cooldown_ms >= TURN_HINT_COOLDOWN_MS
                    ):
                        try:
                            await session.send_realtime_input(audio_stream_end=True)
                            last_stream_end_ts = now
                            stream_active = False
                            had_voice_since_last_end = False
                            logger.info(
                                "Turn hint: sent audio_stream_end after silence (idle_ms=%d, rms=%.1f)",
                                int(idle_ms),
                                rms,
                            )
                        except Exception as exc:
                            if _is_expected_disconnect(exc) or stop_event.is_set():
                                logger.info("Input stream closed while sending turn hint.")
                                break
                            raise

        async def receive_audio() -> None:
            nonlocal assistant_block_until_ts
            current_rate = OUTPUT_RATE
            last_printed_user_text = ""
            last_printed_assistant_text = ""

            def print_user_once(text: str) -> None:
                nonlocal last_printed_user_text
                cleaned = text.strip()
                if cleaned and cleaned != last_printed_user_text:
                    print(f"You: {cleaned}")
                    last_printed_user_text = cleaned

            def print_assistant_once(text: str) -> None:
                nonlocal last_printed_assistant_text
                cleaned = text.strip()
                if cleaned and cleaned != last_printed_assistant_text:
                    print(f"CASie: {cleaned}")
                    last_printed_assistant_text = cleaned

            def queue_speaker_audio(raw: bytes, rate: int) -> None:
                if speaker_queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        speaker_queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    speaker_queue.put_nowait((raw, rate))

            while not stop_event.is_set():
                try:
                    turn = session.receive()
                    async for msg in turn:
                        if stop_event.is_set():
                            break

                        setup_complete = getattr(msg, "setup_complete", None)
                        if setup_complete and getattr(setup_complete, "session_id", None):
                            run_context.session_id = str(setup_complete.session_id)
                            logger.info("Live session established: %s", run_context.session_id)

                        if getattr(msg, "tool_call_cancellation", None):
                            logger.info("Received tool call cancellation from server.")

                        raw_audio = getattr(msg, "data", None)
                        used_direct_audio = False
                        if raw_audio:
                            queue_speaker_audio(raw_audio, current_rate)
                            used_direct_audio = True

                        pending_tool_calls = list(
                            getattr(getattr(msg, "tool_call", None), "function_calls", None) or []
                        )

                        # Some SDK/model variants emit transcription events at top level.
                        top_it = getattr(msg, "input_transcription", None)
                        if top_it and getattr(top_it, "text", None):
                            print_user_once(top_it.text)

                        top_ot = getattr(msg, "output_transcription", None)
                        if top_ot and getattr(top_ot, "text", None):
                            print_assistant_once(top_ot.text)

                        sc = getattr(msg, "server_content", None)
                        if not sc:
                            if pending_tool_calls:
                                await _respond_to_tool_calls(
                                    session=session,
                                    tool_registry=tool_registry,
                                    run_context=run_context,
                                    function_calls=pending_tool_calls,
                                )
                            continue

                        if getattr(sc, "interrupted", False) is True:
                            # Flush queued playback immediately on server interruption event.
                            while not speaker_queue.empty():
                                with contextlib.suppress(asyncio.QueueEmpty):
                                    speaker_queue.get_nowait()
                            # Keep a tiny guard window to avoid immediate speaker bleed-through.
                            assistant_block_until_ts = time.monotonic() + max(0.08, MIC_BLOCK_HOLD_MS / 1000.0 / 4.0)
                            logger.info("Server interruption detected; playback buffer cleared.")

                        it = getattr(sc, "input_transcription", None) or getattr(msg, "input_transcription", None)
                        if it and getattr(it, "text", None):
                            print_user_once(it.text)

                        ot = getattr(sc, "output_transcription", None) or getattr(msg, "output_transcription", None)
                        if ot and getattr(ot, "text", None):
                            print_assistant_once(ot.text)

                        mt = getattr(sc, "model_turn", None)
                        if mt:
                            for part in (getattr(mt, "parts", None) or []):
                                function_call = getattr(part, "function_call", None)
                                if function_call:
                                    pending_tool_calls.append(function_call)

                                if used_direct_audio:
                                    continue

                                idata = getattr(part, "inline_data", None)
                                if not idata:
                                    continue
                                raw = getattr(idata, "data", None)
                                if not raw:
                                    continue

                                mime = (getattr(idata, "mime_type", None) or "").lower()
                                m = re.search(r"rate=(\d+)", mime)
                                rate = int(m.group(1)) if m else current_rate
                                if rate != current_rate:
                                    logger.info("Output sample rate changed: %d -> %d", current_rate, rate)
                                    current_rate = rate

                                queue_speaker_audio(raw, rate)

                        if pending_tool_calls:
                            await _respond_to_tool_calls(
                                session=session,
                                tool_registry=tool_registry,
                                run_context=run_context,
                                function_calls=pending_tool_calls,
                            )

                    # Keep listening across turns: receive() iterator may end at turn boundaries.
                    await asyncio.sleep(0)
                except Exception as exc:
                    if _is_expected_disconnect(exc) or stop_event.is_set():
                        logger.info("Receive loop closed.")
                        return
                    raise

        async def play_audio() -> None:
            nonlocal assistant_block_until_ts
            output_stream = sd.OutputStream(
                samplerate=output_stream_rate,
                channels=1,
                dtype="float32",
                device=output_device,
                **output_stream_kwargs,
            )
            output_stream.start()
            try:
                dev = _safe_query_device(output_device, "output")
                logger.info(
                    "Speaker open (device_rate=%d, api_rate=%d, device=%s)",
                    output_stream_rate,
                    OUTPUT_RATE,
                    dev.get("name", "default") if dev else "default",
                )
            except Exception:
                logger.info("Speaker open (device_rate=%d, api_rate=%d, device=default)", output_stream_rate, OUTPUT_RATE)
            try:
                while not stop_event.is_set():
                    raw, desired_rate = await speaker_queue.get()
                    if not raw:
                        continue

                    if MIC_BLOCK_WHILE_ASSISTANT:
                        # Block mic while assistant audio is playing, plus a short cooldown.
                        samples = max(1, len(raw) // 2)
                        playback_secs = samples / float(max(1, desired_rate))
                        hold_secs = MIC_BLOCK_HOLD_MS / 1000.0
                        assistant_block_until_ts = max(
                            assistant_block_until_ts,
                            time.monotonic() + playback_secs + hold_secs,
                        )
                        while not mic_queue.empty():
                            with contextlib.suppress(asyncio.QueueEmpty):
                                mic_queue.get_nowait()

                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if desired_rate != output_stream_rate:
                        pcm = _resample_float32(pcm, desired_rate, output_stream_rate)
                    await asyncio.to_thread(output_stream.write, pcm.reshape(-1, 1))
            finally:
                try:
                    output_stream.stop()
                finally:
                    output_stream.close()

        listener = asyncio.create_task(listen_audio())
        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(receive_audio())
        player = asyncio.create_task(play_audio())
        tasks = {listener, sender, receiver, player}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is None:
                    logger.info("Live stream task ended; restarting session.")
                    return
                if _is_expected_disconnect(exc):
                    logger.info("Live socket closed; restarting session: %s", exc)
                    continue
                raise exc
        finally:
            stop_event.set()
            for task in tasks:
                task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)


async def run():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required.")
    _ensure_audio_runtime_ready()

    requested_api_version = os.getenv("CASIE_API_VERSION", "v1beta").strip() or "v1beta"
    api_version = requested_api_version
    if requested_api_version.lower() == "v1alpha" and not ALLOW_ALPHA_API:
        api_version = "v1beta"
        logger.warning(
            "CASIE_API_VERSION=%s can be unstable for realtime use. Using %s. "
            "Set CASIE_ALLOW_ALPHA_API=true to force alpha.",
            requested_api_version,
            api_version,
        )
    client = genai.Client(api_key=api_key, http_options={"api_version": api_version})

    reconnect_attempt = 0
    while True:
        try:
            await _run_session(client)
            reconnect_attempt = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reconnect_attempt += 1
            delay = min(8.0, 1.5 * reconnect_attempt)
            logger.warning(
                "Live session ended (%s). Reconnecting in %.1fs (attempt %d)...",
                exc,
                delay,
                reconnect_attempt,
            )
            await asyncio.sleep(delay)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBye!")
