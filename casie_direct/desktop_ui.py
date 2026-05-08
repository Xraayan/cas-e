import asyncio
import contextlib
import ctypes
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types
try:
    from PIL import Image, ImageOps, ImageTk
except ImportError:
    Image = None
    ImageOps = None
    ImageTk = None

import agent as core
from tools import TIMETABLE_DISPLAY_MS, TIMETABLE_IMAGE_PATH
from tool_compat import RunContext


load_dotenv()

UI_INPUT_BLOCK_MS = int(os.getenv("CASIE_UI_INPUT_BLOCK_MS", "6"))
UI_STREAM_IDLE_END_MS = int(os.getenv("CASIE_UI_STREAM_IDLE_END_MS", "360"))
UI_TURN_HINT_IDLE_MS = int(os.getenv("CASIE_UI_TURN_HINT_IDLE_MS", "350"))
UI_TURN_HINT_COOLDOWN_MS = int(os.getenv("CASIE_UI_TURN_HINT_COOLDOWN_MS", "350"))
UI_ASSISTANT_HOLD_MS = int(os.getenv("CASIE_UI_ASSISTANT_HOLD_MS", str(core.MIC_BLOCK_HOLD_MS)))
UI_TRANSCRIPT_UPDATE_INTERVAL_MS = int(os.getenv("CASIE_UI_TRANSCRIPT_UPDATE_INTERVAL_MS", "0"))
UI_TRANSCRIPT_JOIN_GAP_MS = int(os.getenv("CASIE_UI_TRANSCRIPT_JOIN_GAP_MS", "3800"))
UI_TRANSCRIPT_DUPLICATE_GAP_MS = int(os.getenv("CASIE_UI_TRANSCRIPT_DUPLICATE_GAP_MS", "120"))
UI_SERVER_SILENCE_MS = int(os.getenv("CASIE_UI_SERVER_SILENCE_MS", "360"))
UI_SERVER_PREFIX_MS = int(os.getenv("CASIE_UI_SERVER_PREFIX_MS", "160"))
UI_AUTO_SLEEP_IDLE_MS = int(os.getenv("CASIE_UI_AUTO_SLEEP_IDLE_MS", "60000"))
UI_DEBUG_AUDIO = os.getenv("CASIE_UI_DEBUG_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}
UI_REDRAW_INTERVAL_MS = int(os.getenv("CASIE_UI_REDRAW_INTERVAL_MS", "70"))
UI_MAX_MESSAGES = int(os.getenv("CASIE_UI_MAX_MESSAGES", "120"))
UI_LEVEL_EMIT_MS = int(os.getenv("CASIE_UI_LEVEL_EMIT_MS", "70"))
UI_VIZ_BARS = int(os.getenv("CASIE_UI_VIZ_BARS", "34"))
UI_VIZ_FPS_MS = int(os.getenv("CASIE_UI_VIZ_FPS_MS", "50"))
UI_EVENT_POLL_MS = int(os.getenv("CASIE_UI_EVENT_POLL_MS", "16"))
UI_TIMETABLE_SELF_TEST = os.getenv("CASIE_TIMETABLE_SELF_TEST", "false").strip().lower() in {"1", "true", "yes", "on"}
UI_TIMETABLE_MARGIN_PX = int(os.getenv("CASIE_TIMETABLE_MARGIN_PX", "24"))


class AutoSleepRequested(Exception):
    """Raised inside the worker to close the live session after idle timeout."""


class LiveUiWorker:
    def __init__(self, event_sink, worker_id: int):
        self._event_sink = event_sink
        self.worker_id = worker_id
        self._thread = None
        self._loop = None
        self._root_task = None
        self._stop_requested = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._thread_main, name="casie-ui-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_root_task)

    def _cancel_root_task(self) -> None:
        if self._root_task and not self._root_task.done():
            self._root_task.cancel()

    def _emit(self, event_type: str, **payload) -> None:
        self._event_sink({"type": event_type, "worker_id": self.worker_id, **payload})

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._root_task = loop.create_task(self._run())
        try:
            loop.run_until_complete(self._root_task)
        except asyncio.CancelledError:
            pass
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None
            self._root_task = None
            self._thread = None
            self._emit("stopped")

    async def _run(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self._emit("error", text="GOOGLE_API_KEY is missing. Add it to .env.")
            return
        try:
            core._ensure_audio_runtime_ready()
        except Exception as exc:
            self._emit("error", text=str(exc))
            return

        requested_api_version = os.getenv("CASIE_API_VERSION", "v1beta").strip() or "v1beta"
        api_version = requested_api_version
        if requested_api_version.lower() == "v1alpha" and not core.ALLOW_ALPHA_API:
            api_version = "v1beta"
            self._emit(
                "status",
                text=f"CASIE_API_VERSION={requested_api_version} is unstable; using {api_version}.",
                level="warn",
            )
        client = genai.Client(api_key=api_key, http_options={"api_version": api_version})
        reconnect_attempt = 0

        while not self._stop_requested.is_set():
            try:
                await self._run_single_session(client)
                reconnect_attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reconnect_attempt += 1
                delay = min(8.0, 1.5 * reconnect_attempt)
                self._emit("status", text=f"Reconnecting in {delay:.1f}s...", level="warn")
                self._emit("error", text=f"Session ended: {exc}")
                await asyncio.sleep(delay)

    async def _run_single_session(self, client: genai.Client) -> None:
        tool_registry = core._create_tool_registry()
        config = core._build_config(tool_registry)
        with contextlib.suppress(Exception):
            aad = config.realtime_input_config.automatic_activity_detection
            aad.silence_duration_ms = UI_SERVER_SILENCE_MS
            aad.prefix_padding_ms = UI_SERVER_PREFIX_MS
        stop_event = asyncio.Event()
        assistant_block_until_ts = 0.0
        input_device = core._resolve_device(core.INPUT_DEVICE_HINT, "input", core.SAMPLE_RATE)
        output_device = core._resolve_device(core.OUTPUT_DEVICE_HINT, "output", core.OUTPUT_RATE)
        input_stream_rate = core._choose_stream_rate(input_device, "input", core.SAMPLE_RATE)
        output_stream_rate = core._choose_stream_rate(output_device, "output", core.OUTPUT_RATE)
        input_stream_kwargs = core._stream_kwargs("input")
        output_stream_kwargs = core._stream_kwargs("output")
        run_context = RunContext(
            session_id="desktop-session",
            metadata={"event_sink": self._event_sink, "worker_id": self.worker_id},
        )
        transcript_cache: dict[str, str] = {"you": "", "casie": ""}
        transcript_last_emit_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}
        transcript_last_chunk_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}
        transcript_last_raw_chunk: dict[str, str] = {"you": "", "casie": ""}
        transcript_last_raw_chunk_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}

        self._emit("status", text="Connecting to CASie...", level="info")

        async with client.aio.live.connect(model=core.MODEL, config=config) as session:
            self._emit("status", text="Connected. Microphone is live.", level="live")
            if core.STARTUP_GREETING_ENABLED:
                await core._send_startup_greeting(session)

            loop = asyncio.get_running_loop()
            mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
            speaker_queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=256)
            level_state = {"mic": 0.0, "out": 0.0}

            def _norm_level(rms: float, ref: float) -> float:
                if ref <= 0:
                    return 0.0
                return max(0.0, min(1.0, rms / ref))

            def merge_with_overlap(previous: str, chunk: str) -> str:
                prev = " ".join((previous or "").split())
                cur = " ".join((chunk or "").split())
                if not prev:
                    return cur
                if not cur:
                    return prev
                if cur == prev:
                    return prev
                if cur.startswith(prev):
                    return cur
                if prev.startswith(cur):
                    return prev
                if prev.endswith(cur):
                    return prev
                if cur.lower() in prev.lower():
                    return prev

                max_overlap = min(len(prev), len(cur))
                overlap = 0
                for size in range(max_overlap, 0, -1):
                    if prev[-size:].lower() == cur[:size].lower():
                        overlap = size
                        break

                if overlap > 0:
                    merged = f"{prev}{cur[overlap:]}"
                else:
                    sep = "" if cur[:1] in {".", ",", "!", "?", ":", ";"} else " "
                    merged = f"{prev}{sep}{cur}"
                return " ".join(merged.split())

            def emit_transcript(role: str, text: str) -> None:
                clean = " ".join((text or "").split())
                if not clean:
                    return

                now = time.monotonic()
                last_raw = transcript_last_raw_chunk.get(role, "")
                last_raw_ts = transcript_last_raw_chunk_ts.get(role, 0.0)
                if (
                    clean == last_raw
                    and (now - last_raw_ts) * 1000.0 < UI_TRANSCRIPT_DUPLICATE_GAP_MS
                ):
                    return

                transcript_last_raw_chunk[role] = clean
                transcript_last_raw_chunk_ts[role] = now

                previous = transcript_cache.get(role, "")
                if previous == clean:
                    return

                gap_ms = (now - transcript_last_chunk_ts.get(role, 0.0)) * 1000.0
                is_cumulative = bool(previous) and clean.startswith(previous)
                can_join_chunks = (
                    bool(previous)
                    and gap_ms <= UI_TRANSCRIPT_JOIN_GAP_MS
                    and not re.search(r"[.!?]\s*$", previous)
                )

                if is_cumulative:
                    merged = clean
                    event_type = "transcript_update"
                elif can_join_chunks:
                    merged = merge_with_overlap(previous, clean)
                    event_type = "transcript_update"
                else:
                    merged = clean
                    event_type = "transcript"

                if merged == previous:
                    return

                transcript_cache[role] = merged
                transcript_last_chunk_ts[role] = now

                if (
                    UI_TRANSCRIPT_UPDATE_INTERVAL_MS > 0
                    and
                    event_type == "transcript_update"
                    and (now - transcript_last_emit_ts.get(role, 0.0)) * 1000.0
                    < UI_TRANSCRIPT_UPDATE_INTERVAL_MS
                ):
                    return

                transcript_last_emit_ts[role] = now
                self._emit(event_type, role=role, text=merged)
                if event_type == "transcript" and role == "you":
                    self._emit("status", text="Listening...", level="live")

            def mic_callback(indata, frames, time_info, status) -> None:
                del frames, time_info
                if status:
                    core.logger.debug("Mic status: %s", status)
                mono = np.asarray(indata[:, 0], dtype=np.float32)
                if input_stream_rate != core.SAMPLE_RATE:
                    mono = core._resample_float32(mono, input_stream_rate, core.SAMPLE_RATE)
                pcm = core._pcm16_bytes_from_float32(mono)

                def push() -> None:
                    if mic_queue.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            mic_queue.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        mic_queue.put_nowait(pcm)

                try:
                    loop.call_soon_threadsafe(push)
                except RuntimeError:
                    pass

            async def listen_audio() -> None:
                blocksize = int(input_stream_rate * (UI_INPUT_BLOCK_MS / 1000.0))
                with sd.InputStream(
                    samplerate=input_stream_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=max(1, blocksize),
                    device=input_device,
                    callback=mic_callback,
                    **input_stream_kwargs,
                ):
                    if core.PI_COMPAT_ENABLED and core.IS_RASPBERRY_PI and core.PI_STABILIZE_MS > 0:
                        await asyncio.sleep(core.PI_STABILIZE_MS / 1000.0)
                    self._emit("status", text="Listening...", level="live")
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
                adaptive_gate_rms = core.MIC_NOISE_GATE_RMS
                gate_ready = not core.MIC_NOISE_GATE_ENABLED
                auto_sleep_secs = UI_AUTO_SLEEP_IDLE_MS / 1000.0 if UI_AUTO_SLEEP_IDLE_MS > 0 else 0.0

                def request_auto_sleep(now: float) -> None:
                    if not auto_sleep_secs:
                        return
                    if now < assistant_block_until_ts:
                        return
                    if (now - last_voice_ts) < auto_sleep_secs:
                        return
                    self._stop_requested.set()
                    stop_event.set()
                    self._emit("auto_sleep", idle_seconds=int(auto_sleep_secs))
                    raise AutoSleepRequested()

                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic_queue.get(), timeout=0.08)
                    except asyncio.TimeoutError:
                        request_auto_sleep(time.monotonic())
                        level_state["mic"] *= 0.9
                        if (
                            core.CLIENT_AUDIO_STREAM_END_ENABLED
                            and stream_active
                            and (time.monotonic() - last_sent_ts) > (UI_STREAM_IDLE_END_MS / 1000.0)
                        ):
                            try:
                                await session.send_realtime_input(audio_stream_end=True)
                            except Exception as exc:
                                if core._is_expected_disconnect(exc) or stop_event.is_set():
                                    return
                                raise
                            stream_active = False
                        continue

                    if (
                        core.DROP_INPUT_WHILE_ASSISTANT
                        and time.monotonic() < assistant_block_until_ts
                    ):
                        last_voice_ts = time.monotonic()
                        continue

                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    rms = 0.0
                    if pcm.size:
                        rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
                    level_state["mic"] = (level_state["mic"] * 0.72) + (_norm_level(rms, 3200.0) * 0.28)

                    if core.MIC_NOISE_GATE_ENABLED and not gate_ready:
                        elapsed_ms = (time.monotonic() - gate_started_ts) * 1000.0
                        noise_floor_samples.append(rms)
                        if elapsed_ms < core.MIC_NOISE_GATE_LEARN_MS:
                            continue
                        if noise_floor_samples:
                            floor = float(np.percentile(np.array(noise_floor_samples, dtype=np.float32), 80))
                            adaptive_gate_rms = max(
                                core.MIC_NOISE_GATE_RMS,
                                floor * core.MIC_NOISE_GATE_MULTIPLIER,
                            )
                            core.logger.info(
                                "Mic noise gate calibrated(UI): floor=%.1f threshold=%.1f",
                                floor,
                                adaptive_gate_rms,
                            )
                        gate_ready = True

                    if rms >= max(core.TURN_HINT_RMS, adaptive_gate_rms):
                        last_voice_ts = time.monotonic()
                        had_voice_since_last_end = True
                    else:
                        request_auto_sleep(time.monotonic())

                    if core.MIC_NOISE_GATE_ENABLED:
                        now = time.monotonic()
                        if rms >= adaptive_gate_rms:
                            last_voice_gate_ts = now
                        else:
                            hangover_ms = (now - last_voice_gate_ts) * 1000.0
                            if hangover_ms > core.MIC_NOISE_GATE_HANGOVER_MS:
                                continue

                    if UI_DEBUG_AUDIO:
                        peak_rms = max(peak_rms, rms)
                        now = time.monotonic()
                        if now - last_rms_log_ts >= 1.0:
                            core.logger.info("Mic debug(UI): peak_rms=%.1f stream_active=%s", peak_rms, stream_active)
                            peak_rms = 0.0
                            last_rms_log_ts = now

                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={core.SAMPLE_RATE}")
                        )
                    except Exception as exc:
                        if core._is_expected_disconnect(exc) or stop_event.is_set():
                            return
                        raise
                    last_sent_ts = time.monotonic()
                    stream_active = True

                    if core.TURN_HINT_ENABLED:
                        now = time.monotonic()
                        idle_ms = (now - last_voice_ts) * 1000.0
                        cooldown_ms = (now - last_stream_end_ts) * 1000.0
                        if (
                            had_voice_since_last_end
                            and idle_ms >= UI_TURN_HINT_IDLE_MS
                            and cooldown_ms >= UI_TURN_HINT_COOLDOWN_MS
                        ):
                            try:
                                await session.send_realtime_input(audio_stream_end=True)
                            except Exception as exc:
                                if core._is_expected_disconnect(exc) or stop_event.is_set():
                                    return
                                raise
                            last_stream_end_ts = now
                            stream_active = False
                            had_voice_since_last_end = False

            async def receive_audio() -> None:
                nonlocal assistant_block_until_ts
                current_rate = core.OUTPUT_RATE

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

                            raw_audio = getattr(msg, "data", None)
                            used_direct_audio = False
                            if raw_audio:
                                queue_speaker_audio(raw_audio, current_rate)
                                used_direct_audio = True

                            pending_tool_calls = list(
                                getattr(getattr(msg, "tool_call", None), "function_calls", None) or []
                            )

                            sc = getattr(msg, "server_content", None)
                            if not sc:
                                top_it = getattr(msg, "input_transcription", None)
                                if top_it and getattr(top_it, "text", None):
                                    emit_transcript("you", top_it.text)

                                top_ot = getattr(msg, "output_transcription", None)
                                if top_ot and getattr(top_ot, "text", None):
                                    emit_transcript("casie", top_ot.text)

                                if pending_tool_calls:
                                    await core._respond_to_tool_calls(
                                        session=session,
                                        tool_registry=tool_registry,
                                        run_context=run_context,
                                        function_calls=pending_tool_calls,
                                    )
                                continue

                            if getattr(sc, "interrupted", False) is True:
                                while not speaker_queue.empty():
                                    with contextlib.suppress(asyncio.QueueEmpty):
                                        speaker_queue.get_nowait()
                                assistant_block_until_ts = 0.0
                                self._emit("status", text="Interrupted. Listening...", level="live")

                            it = getattr(sc, "input_transcription", None) or getattr(msg, "input_transcription", None)
                            if it and getattr(it, "text", None):
                                emit_transcript("you", it.text)

                            ot = getattr(sc, "output_transcription", None) or getattr(msg, "output_transcription", None)
                            if ot and getattr(ot, "text", None):
                                emit_transcript("casie", ot.text)

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
                                    match = re.search(r"rate=(\d+)", mime)
                                    rate = int(match.group(1)) if match else current_rate
                                    current_rate = rate
                                    queue_speaker_audio(raw, rate)

                            if pending_tool_calls:
                                await core._respond_to_tool_calls(
                                    session=session,
                                    tool_registry=tool_registry,
                                    run_context=run_context,
                                    function_calls=pending_tool_calls,
                                )

                        await asyncio.sleep(0)
                    except Exception as exc:
                        if core._is_expected_disconnect(exc) or stop_event.is_set():
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
                current_rate = output_stream_rate

                try:
                    while not stop_event.is_set():
                        raw, desired_rate = await speaker_queue.get()
                        if not raw:
                            level_state["out"] *= 0.9
                            continue

                        if core.MIC_BLOCK_WHILE_ASSISTANT:
                            samples = max(1, len(raw) // 2)
                            playback_secs = samples / float(max(1, desired_rate))
                            hold_secs = UI_ASSISTANT_HOLD_MS / 1000.0
                            assistant_block_until_ts = max(
                                assistant_block_until_ts,
                                time.monotonic() + playback_secs + hold_secs,
                            )
                            while not mic_queue.empty():
                                with contextlib.suppress(asyncio.QueueEmpty):
                                    mic_queue.get_nowait()

                        pcm_i16 = np.frombuffer(raw, dtype=np.int16)
                        out_rms = 0.0
                        if pcm_i16.size:
                            out_rms = float(np.sqrt(np.mean(np.square(pcm_i16.astype(np.float32)))))
                        level_state["out"] = (level_state["out"] * 0.72) + (_norm_level(out_rms, 4200.0) * 0.28)

                        pcm = pcm_i16.astype(np.float32) / 32768.0
                        if desired_rate != output_stream_rate:
                            pcm = core._resample_float32(pcm, desired_rate, output_stream_rate)
                        await asyncio.to_thread(output_stream.write, pcm.reshape(-1, 1))
                finally:
                    with contextlib.suppress(Exception):
                        output_stream.stop()
                    with contextlib.suppress(Exception):
                        output_stream.close()

            async def emit_levels() -> None:
                interval = max(0.03, UI_LEVEL_EMIT_MS / 1000.0)
                while not stop_event.is_set():
                    level_state["mic"] *= 0.95
                    level_state["out"] *= 0.95
                    self._emit("levels", mic=level_state["mic"], out=level_state["out"])
                    await asyncio.sleep(interval)

            listener = asyncio.create_task(listen_audio())
            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_audio())
            player = asyncio.create_task(play_audio())
            meter = asyncio.create_task(emit_levels())
            stream_tasks = {listener, sender, receiver, player}
            tasks = set(stream_tasks)
            tasks.add(meter)
            try:
                done, _ = await asyncio.wait(stream_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if isinstance(exc, AutoSleepRequested):
                        return
                    if exc is None:
                        if self._stop_requested.is_set() or stop_event.is_set():
                            return
                        self._emit("status", text="Session dropped. Reconnecting...", level="warn")
                        return
                    if core._is_expected_disconnect(exc):
                        if not self._stop_requested.is_set():
                            self._emit("status", text="Connection dropped. Reconnecting...", level="warn")
                        continue
                    if self._stop_requested.is_set():
                        return
                    raise exc
            finally:
                stop_event.set()
                for task in tasks:
                    task.cancel()
                with contextlib.suppress(Exception):
                    await asyncio.gather(*tasks, return_exceptions=True)


class CasieDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CASie Desktop")
        self.geometry("880x610")
        self.minsize(680, 460)
        self.configure(bg="#030303")
        self._is_linux = sys.platform.startswith("linux")

        self._events = queue.Queue()
        self._worker = None
        self._next_worker_id = 1
        self._active_worker_id = 0
        self._running = False
        self._close_after_stop = False
        self._destroy_after_stop = False
        self._last_transcript_role = None
        self._last_transcript_index = None
        self._last_transcript_text = ""
        self._messages: list[tuple[str, str]] = []
        self._chat_font = tkfont.Font(family="Consolas", size=17, weight="bold")
        self._line_height = max(14, int(self._chat_font.metrics("linespace")))
        self._avg_char_width = max(6, int(self._chat_font.measure("n")))
        self._redraw_job = None
        self._viz_job = None
        self._viz_phase = 0.0
        self._viz_level_mic = 0.0
        self._viz_level_out = 0.0
        self._viz_target_mic = 0.0
        self._viz_target_out = 0.0
        self._status_text = "Ready"
        self._status_level = "idle"
        self._title_font = tkfont.Font(family="Consolas", size=24, weight="bold")
        self._subtitle_font = tkfont.Font(family="Consolas", size=11, weight="bold")
        self._status_font = tkfont.Font(family="Consolas", size=13, weight="bold")
        self._footer_font = tkfont.Font(family="Consolas", size=9, weight="bold")
        self._timetable_window = None
        self._timetable_photo = None
        self._timetable_close_job = None
        self._timetable_canvas_active = False
        self._timetable_error_text = ""

        self._build_layout()
        self._set_status("Ready", "idle")
        self._apply_fullscreen()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _event: self._on_close())
        self.bind("<F8>", lambda _event: self._show_timetable(str(TIMETABLE_IMAGE_PATH), TIMETABLE_DISPLAY_MS))
        self.after_idle(self._finish_fullscreen_layout)
        self.after(250, self._finish_fullscreen_layout)
        self.after(1000, self._finish_fullscreen_layout)
        if UI_TIMETABLE_SELF_TEST:
            self.after(2500, lambda: self._show_timetable(str(TIMETABLE_IMAGE_PATH), TIMETABLE_DISPLAY_MS))
        self.after(40, self._drain_events)

    def _new_worker(self) -> LiveUiWorker:
        worker = LiveUiWorker(self._events.put, self._next_worker_id)
        self._next_worker_id += 1
        return worker

    def _screen_size(self) -> tuple[int, int]:
        return max(1, self.winfo_screenwidth()), max(1, self.winfo_screenheight())

    def _apply_fullscreen(self) -> None:
        """Enable fullscreen with platform-specific window management."""
        screen_w, screen_h = self._screen_size()

        with contextlib.suppress(tk.TclError):
            self.overrideredirect(self._is_linux)
        with contextlib.suppress(tk.TclError):
            self.attributes("-fullscreen", True)
        if self._is_linux:
            with contextlib.suppress(tk.TclError):
                self.geometry(f"{screen_w}x{screen_h}+0+0")
        else:
            with contextlib.suppress(tk.TclError):
                self.attributes("-topmost", True)
                self.state("zoomed")

        with contextlib.suppress(tk.TclError):
            self.lift()
            self.focus_force()

        self._canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _finish_fullscreen_layout(self) -> None:
        """Retry fullscreen after the window manager settles and redraw to the live size."""
        self._apply_fullscreen()
        self.update_idletasks()
        self._canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._draw_visualizer()

    def _build_layout(self) -> None:
        self._canvas = tk.Canvas(
            self,
            bg="#030303",
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self._canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._canvas.bind("<Configure>", lambda _e: self._request_redraw())
        self._start_visualizer()

        self._left_text = tk.Text(
            self,
            bg="#030303",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            wrap="word",
            font=self._chat_font,
            padx=0,
            pady=0,
            spacing1=2,
            spacing2=2,
            spacing3=8,
            state="disabled",
            cursor="arrow",
        )
        self._right_text = tk.Text(
            self,
            bg="#030303",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            wrap="word",
            font=self._chat_font,
            padx=0,
            pady=0,
            spacing1=2,
            spacing2=2,
            spacing3=8,
            state="disabled",
            cursor="arrow",
        )
        self._left_text.place_forget()
        self._right_text.place_forget()

        self.end_button = tk.Button(
            self,
            text="Wake",
            command=self._toggle_session,
            bg="#050505",
            fg="#d7c4aa",
            activebackground="#13100d",
            activeforeground="#fff0d8",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=5,
            font=("Consolas", 12, "bold"),
        )
        self.end_button.place_forget()

        self.close_button = tk.Button(
            self,
            text="X",
            command=self._on_close,
            bg="#160707",
            fg="#fff2e8",
            activebackground="#351010",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=4,
            font=("Consolas", 13, "bold"),
            cursor="hand2",
        )
        self.close_button.place_forget()

    def _start_visualizer(self) -> None:
        if self._viz_job is not None:
            return
        self._viz_job = self.after(max(24, UI_VIZ_FPS_MS), self._tick_visualizer)

    def _tick_visualizer(self) -> None:
        self._viz_job = None
        self._viz_level_mic = (self._viz_level_mic * 0.78) + (self._viz_target_mic * 0.22)
        self._viz_level_out = (self._viz_level_out * 0.78) + (self._viz_target_out * 0.22)
        self._draw_visualizer()
        self._viz_phase += 0.2 + (0.55 * max(self._viz_level_mic, self._viz_level_out))
        self._viz_job = self.after(max(24, UI_VIZ_FPS_MS), self._tick_visualizer)

    def _draw_visualizer(self) -> None:
        self._canvas.delete("viz")
        canvas_w = max(1, self._canvas.winfo_width())
        canvas_h = max(1, self._canvas.winfo_height())
        if self._timetable_canvas_active:
            self._draw_timetable_canvas_overlay(canvas_w, canvas_h)
            return

        cx = canvas_w / 2.0
        cy = canvas_h / 2.0 + 8
        min_side = float(min(canvas_w, canvas_h))
        activity = max(self._viz_level_mic, self._viz_level_out)

        self._canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill="#030303", outline="", tags=("viz",))
        self._canvas.create_line(0, 128, canvas_w, 128, fill="#2a211d", width=1, tags=("viz",))
        self._canvas.create_line(0, canvas_h - 62, canvas_w, canvas_h - 62, fill="#241c19", width=1, tags=("viz",))

        title_size = max(18, min(30, canvas_w // 32))
        subtitle_size = max(8, min(12, canvas_w // 76))
        self._title_font.configure(size=title_size)
        self._subtitle_font.configure(size=subtitle_size)
        self._status_font.configure(size=max(11, min(14, canvas_w // 58)))

        self._canvas.create_text(
            cx,
            42,
            text="CAS-E",
            fill="#fff4df",
            font=self._title_font,
            anchor="n",
            tags=("viz",),
        )
        self._canvas.create_text(
            cx,
            78,
            text="Campus Assistent of EC Dept",
            fill="#f4d7b4",
            font=self._subtitle_font,
            anchor="n",
            tags=("viz",),
        )

        speaking = self._viz_level_out > 0.02
        pulse = 1.0 + ((0.08 * math.sin(self._viz_phase * 1.8)) + (self._viz_level_out * 0.1) if speaking else 0.0)
        heart_scale = max(4.0, min_side * 0.0075 * pulse)
        points = []
        for i in range(160):
            t = (2.0 * math.pi) * (i / 160.0)
            x = 16.0 * (math.sin(t) ** 3)
            y = -(13.0 * math.cos(t) - 5.0 * math.cos(2 * t) - 2.0 * math.cos(3 * t) - math.cos(4 * t))
            points.extend((cx + x * heart_scale, cy + y * heart_scale))

        self._canvas.create_polygon(points, fill="#ff1b22", outline="", smooth=True, splinesteps=24, tags=("viz",))

        status_live = self._status_level == "live"
        status_error = self._status_level == "error"
        status_idle = self._status_level == "idle"
        status_warn = self._status_level == "warn"
        status_text = (
            "ONLINE"
            if status_live
            else ("SLEEP" if status_idle else ("ERROR" if status_error else ("RECONNECTING" if status_warn else "CONNECTING")))
        )
        status_fill = "#fff2d6" if status_live else ("#ff9b8f" if status_error else "#c9b9a0")
        dot_fill = "#f7efe0" if status_live else ("#ff6f61" if status_error else "#7f7668")
        status_y = min(canvas_h - 118, cy + max(116, min_side * 0.24))
        self._canvas.create_oval(cx - 48, status_y - 5, cx - 38, status_y + 5, outline=status_fill, width=2, tags=("viz",))
        self._canvas.create_oval(cx - 45, status_y - 2, cx - 41, status_y + 2, fill=dot_fill, outline="", tags=("viz",))
        self._canvas.create_text(
            cx + 18,
            status_y,
            text=status_text,
            fill=status_fill,
            font=self._status_font,
            anchor="center",
            tags=("viz",),
        )
        self.end_button.place(relx=0.5, y=status_y + 34, anchor="center")
        self.close_button.place(x=canvas_w - 18, y=18, anchor="ne")

        self._canvas.create_text(
            cx,
            canvas_h - 24,
            text="ROBOTICS AND AI DEPT | RIT KOTTAYAM",
            fill="#e7c7a2",
            font=self._footer_font,
            anchor="center",
            tags=("viz",),
        )

        self._place_transcript_widgets(canvas_w, canvas_h)
        self._canvas.tag_lower("viz")
        if self._timetable_window is not None:
            self._bring_timetable_to_front(self._timetable_window)

    def _hide_main_widgets(self) -> None:
        for widget in (self._left_text, self._right_text, self.end_button, self.close_button):
            with contextlib.suppress(tk.TclError):
                widget.place_forget()

    def _draw_timetable_canvas_overlay(self, canvas_w: int | None = None, canvas_h: int | None = None) -> None:
        if not self._timetable_canvas_active:
            return

        with contextlib.suppress(tk.TclError):
            self._canvas.lift()
        canvas_w = max(1, int(canvas_w or self._canvas.winfo_width() or self.winfo_screenwidth()))
        canvas_h = max(1, int(canvas_h or self._canvas.winfo_height() or self.winfo_screenheight()))
        cx = canvas_w / 2
        cy = canvas_h / 2
        pad = max(4, min(UI_TIMETABLE_MARGIN_PX // 2, 16))

        self._hide_main_widgets()
        self._canvas.delete("timetable")
        self._canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill="#030303", outline="", tags=("timetable",))
        if not self._timetable_photo:
            self._canvas.create_text(
                cx,
                cy,
                text=self._timetable_error_text or "Timetable could not be displayed.",
                fill="#fff2d6",
                font=self._status_font,
                anchor="center",
                width=max(260, int(canvas_w * 0.8)),
                tags=("timetable",),
            )
            self._canvas.tag_raise("timetable")
            return

        _original_photo, display_photo = self._timetable_photo
        image_w = max(1, display_photo.width())
        image_h = max(1, display_photo.height())
        left = max(0, cx - (image_w / 2) - pad)
        top = max(0, cy - (image_h / 2) - pad)
        right = min(canvas_w, cx + (image_w / 2) + pad)
        bottom = min(canvas_h, cy + (image_h / 2) + pad)
        self._canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill="#050505",
            outline="#f6d77a",
            width=3,
            tags=("timetable",),
        )
        self._canvas.create_image(cx, cy, image=display_photo, anchor="center", tags=("timetable",))
        self._canvas.tag_raise("timetable")

    def _place_transcript_widgets(self, canvas_w: int, canvas_h: int) -> None:
        top = 150
        bottom = 86
        outer_pad = 28
        center_gap = max(180, min(300, canvas_w // 3))
        lane_width = max(160, int((canvas_w - (outer_pad * 2) - center_gap) / 2))
        lane_height = max(120, canvas_h - top - bottom)
        right_x = canvas_w - outer_pad - lane_width

        self._left_text.place(x=outer_pad, y=top, width=lane_width, height=lane_height)
        self._right_text.place(x=right_x, y=top, width=lane_width, height=lane_height)
        with contextlib.suppress(tk.TclError):
            self._left_text.lift()
            self._right_text.lift()
            self.end_button.lift()
            self.close_button.lift()

    def _message_style(self, role: str, canvas_w: int) -> dict[str, object]:
        side = 24
        lane_width = int(max(170, min(canvas_w * 0.27, (canvas_w / 2) - 150)))
        if role == "you":
            return {
                "x": canvas_w - side,
                "anchor": "ne",
                "justify": "right",
                "fill": "#ffffff",
                "width": lane_width,
            }
        if role == "casie":
            return {
                "x": side,
                "anchor": "nw",
                "justify": "left",
                "fill": "#ffffff",
                "width": lane_width,
            }
        return {
            "x": canvas_w / 2,
            "anchor": "n",
            "justify": "center",
            "fill": "#9ab6ae",
            "width": int(max(240, canvas_w * 0.7)),
        }

    def _request_redraw(self) -> None:
        if self._redraw_job is not None:
            return
        self._redraw_job = self.after(max(20, UI_REDRAW_INTERVAL_MS), self._flush_redraw)

    def _flush_redraw(self) -> None:
        self._redraw_job = None
        self._redraw_messages()

    def _measure_message_height(self, role: str, text: str, canvas_w: int) -> int:
        style = self._message_style(role, canvas_w)
        width_px = max(80, int(style["width"]))
        chars_per_line = max(8, width_px // self._avg_char_width)
        lines = 0
        for segment in text.split("\n"):
            if not segment:
                lines += 1
            else:
                lines += max(1, math.ceil(len(segment) / chars_per_line))
        return max(self._line_height + 4, lines * self._line_height + 4)

    def _redraw_messages(self) -> None:
        if not hasattr(self, "_left_text"):
            return
        left_lines = [text for role, text in self._messages if role in {"casie", "sys"}]
        right_lines = [text for role, text in self._messages if role == "you"]
        self._render_text_widget(self._left_text, left_lines, "left")
        self._render_text_widget(self._right_text, right_lines, "right")

    def _render_text_widget(self, widget: tk.Text, lines: list[str], justify: str) -> None:
        widget.configure(state="normal")
        widget.tag_configure("body", justify=justify)
        widget.delete("1.0", "end")
        if lines:
            widget.insert("end", "\n\n".join(lines), ("body",))
        widget.see("end")
        widget.configure(state="disabled")

    def _stop_background_animation(self) -> None:
        if self._viz_job is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._viz_job)
            self._viz_job = None
        if self._redraw_job is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._redraw_job)
            self._redraw_job = None

    def _close_timetable(self) -> None:
        if self._timetable_close_job is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._timetable_close_job)
            self._timetable_close_job = None

        if self._timetable_canvas_active:
            self._timetable_canvas_active = False
            with contextlib.suppress(tk.TclError):
                self._canvas.delete("timetable")

        if self._timetable_window is not None:
            with contextlib.suppress(Exception):
                self._timetable_window.destroy()
            self._timetable_window = None
        self._timetable_photo = None
        self._timetable_error_text = ""
        self._request_redraw()

    def _bring_timetable_to_front(self, window: tk.Misc) -> None:
        try:
            exists = window.winfo_exists()
        except tk.TclError:
            return
        if not exists:
            return

        with contextlib.suppress(Exception):
            window.deiconify()
        with contextlib.suppress(Exception):
            window.attributes("-topmost", True)
        with contextlib.suppress(tk.TclError):
            window.lift()
            window.focus_force()

        if sys.platform.startswith("win"):
            hwnd = window.winfo_id()
            user32 = ctypes.windll.user32
            sw_shownormal = 1
            hwnd_topmost = -1
            swp_nosize = 0x0001
            swp_nomove = 0x0002
            swp_showwindow = 0x0040
            with contextlib.suppress(Exception):
                user32.ShowWindow(hwnd, sw_shownormal)
                user32.SetWindowPos(
                    hwnd,
                    hwnd_topmost,
                    0,
                    0,
                    0,
                    0,
                    swp_nomove | swp_nosize | swp_showwindow,
                )
                user32.SetForegroundWindow(hwnd)

    def _load_timetable_photo(self, image_path: str, max_w: int, max_h: int):
        max_w = max(1, int(max_w))
        max_h = max(1, int(max_h))
        if Image is not None and ImageOps is not None and ImageTk is not None:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGBA")
                image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(image)

        original_photo = tk.PhotoImage(file=image_path)
        shrink = max(
            1,
            math.ceil(max(original_photo.width() / max_w, original_photo.height() / max_h)),
        )
        if shrink > 1:
            return original_photo.subsample(shrink, shrink)
        return original_photo

    def _show_timetable_error(self, text: str, duration_ms: int) -> None:
        self._timetable_canvas_active = True
        self._timetable_photo = None
        self._timetable_error_text = text
        self._draw_timetable_canvas_overlay()
        self.update_idletasks()
        self._timetable_close_job = self.after(max(1000, int(duration_ms or 8000)), self._close_timetable)

    def _show_timetable(self, image_path: str, duration_ms: int = 8000) -> None:
        if not image_path or not os.path.exists(image_path):
            self._append_system_line("Timetable image not found.")
            print(f"Timetable image not found: {image_path}", flush=True)
            return

        self._close_timetable()
        self.update_idletasks()
        canvas_w = max(1, self._canvas.winfo_width(), self.winfo_width())
        canvas_h = max(1, self._canvas.winfo_height(), self.winfo_height())
        margin = max(0, UI_TIMETABLE_MARGIN_PX)
        max_w = max(1, int(canvas_w - (margin * 2)))
        max_h = max(1, int(canvas_h - (margin * 2)))
        try:
            display_photo = self._load_timetable_photo(image_path, max_w, max_h)
        except Exception as exc:
            self._show_timetable_error(f"Could not load timetable image: {exc}", duration_ms)
            print(f"Could not load timetable image {image_path}: {exc}", flush=True)
            return

        if self._is_linux:
            self._timetable_canvas_active = True
            self._timetable_photo = (display_photo, display_photo)
            self._draw_timetable_canvas_overlay(canvas_w, canvas_h)
            self.update_idletasks()
            print(
                f"Showing timetable canvas overlay: {image_path} "
                f"canvas={canvas_w}x{canvas_h} image={display_photo.width()}x{display_photo.height()}",
                flush=True,
            )
            self._timetable_close_job = self.after(max(1000, int(duration_ms or 8000)), self._close_timetable)
            return

        window = tk.Toplevel(self)
        window.title("Class Timetable")
        window.configure(bg="#030303")
        window.transient(self)
        with contextlib.suppress(tk.TclError):
            window.attributes("-topmost", True)

        label = tk.Label(window, image=display_photo, bg="#030303", bd=0)
        label.pack(padx=12, pady=12)
        window.protocol("WM_DELETE_WINDOW", self._close_timetable)
        window.update_idletasks()

        screen_w = max(1, self.winfo_screenwidth())
        screen_h = max(1, self.winfo_screenheight())
        x = max(0, int((screen_w - window.winfo_width()) / 2))
        y = max(0, int((screen_h - window.winfo_height()) / 2))
        window.geometry(f"+{x}+{y}")

        self._timetable_window = window
        self._timetable_photo = (display_photo, display_photo)
        self._bring_timetable_to_front(window)
        window.after(80, lambda: self._bring_timetable_to_front(window))
        window.after(300, lambda: self._bring_timetable_to_front(window))
        self._timetable_close_job = self.after(max(1000, int(duration_ms or 8000)), self._close_timetable)

    def _set_status(self, text: str, level: str) -> None:
        self._status_text = text
        self._status_level = level
        self.title(f"CASie - {text}")
        self._draw_visualizer()

    def _append_line(self, text: str, tag: str) -> None:
        clean = " ".join((text or "").split())
        if not clean:
            return
        self._messages.append((tag, clean))
        self._trim_messages()
        self._request_redraw()

    def _append_system_line(self, text: str) -> None:
        self._append_line(f"[System] {text}", "sys")
        self._last_transcript_role = None
        self._last_transcript_index = None
        self._last_transcript_text = ""

    def _trim_messages(self) -> None:
        max_messages = max(40, UI_MAX_MESSAGES)
        if len(self._messages) <= max_messages:
            return
        overflow = len(self._messages) - max_messages
        self._messages = self._messages[overflow:]
        if self._last_transcript_index is not None:
            self._last_transcript_index -= overflow
            if self._last_transcript_index < 0:
                self._last_transcript_index = None
                self._last_transcript_role = None
                self._last_transcript_text = ""

    def _merge_overlap_text(self, previous: str, current: str) -> str:
        prev = " ".join((previous or "").split())
        cur = " ".join((current or "").split())
        if not prev:
            return cur
        if not cur:
            return prev
        if cur == prev:
            return prev
        if cur.startswith(prev):
            return cur
        if prev.startswith(cur):
            return prev
        if prev.endswith(cur):
            return prev
        if cur.lower() in prev.lower():
            return prev

        max_overlap = min(len(prev), len(cur))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if prev[-size:].lower() == cur[:size].lower():
                overlap = size
                break

        if overlap > 0:
            merged = f"{prev}{cur[overlap:]}"
        else:
            sep = "" if cur[:1] in {".", ",", "!", "?", ":", ";"} else " "
            merged = f"{prev}{sep}{cur}"
        return " ".join(merged.split())

    def _append_transcript(self, role: str, text: str, update: bool = False) -> None:
        line = " ".join((text or "").split())
        if not line:
            return

        if (
            not update
            and self._last_transcript_role == role
            and self._last_transcript_index is not None
        ):
            update = True

        if (
            update
            and self._last_transcript_role == role
            and self._last_transcript_text
        ):
            line = self._merge_overlap_text(self._last_transcript_text, line)
        elif (
            not update
            and self._last_transcript_role == role
            and self._last_transcript_text == line
        ):
            return

        tag = "you" if role == "you" else "casie"

        if (
            update
            and self._last_transcript_role == role
            and self._last_transcript_index is not None
            and 0 <= self._last_transcript_index < len(self._messages)
        ):
            self._messages[self._last_transcript_index] = (tag, line)
        else:
            self._messages.append((tag, line))
            self._last_transcript_index = len(self._messages) - 1
            self._trim_messages()
            if self._last_transcript_index is None:
                self._last_transcript_index = len(self._messages) - 1

        self._last_transcript_role = role
        self._last_transcript_text = line
        self._request_redraw()

    def _reset_conversation_view(self) -> None:
        self._messages.clear()
        self._last_transcript_role = None
        self._last_transcript_index = None
        self._last_transcript_text = ""
        self._viz_target_mic = 0.0
        self._viz_target_out = 0.0
        self._viz_level_mic = 0.0
        self._viz_level_out = 0.0
        self._request_redraw()

    def _start_session(self) -> None:
        if self._running:
            return
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break
        self._worker = self._new_worker()
        self._active_worker_id = self._worker.worker_id
        self._reset_conversation_view()
        self._running = True
        self._set_status("Connecting...", "info")
        self.end_button.configure(state="disabled", text="Connecting...")
        self._worker.start()

    def _stop_session(self) -> None:
        if not self._running or self._worker is None:
            return
        self._close_after_stop = True
        self._set_status("Stopping...", "warn")
        self.end_button.configure(state="disabled", text="Sleeping...")
        self._worker.stop()

    def _toggle_session(self) -> None:
        if self._running:
            self._stop_session()
            return
        self._close_after_stop = False
        self._destroy_after_stop = False
        self._start_session()

    def _drain_events(self) -> None:
        max_events_per_tick = 96
        processed = 0
        while processed < max_events_per_tick:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            processed += 1

            event_type = event.get("type")
            worker_id = int(event.get("worker_id", 0) or 0)
            if worker_id and worker_id != self._active_worker_id:
                if event_type != "show_timetable":
                    continue
                print(
                    "Processing show_timetable from non-active worker "
                    f"{worker_id}; active worker is {self._active_worker_id}",
                    flush=True,
                )
            if event_type == "status":
                level = event.get("level", "info")
                self._set_status(event.get("text", "Status update"), level)
                if level == "live" and self._running:
                    self.end_button.configure(state="normal", text="Sleep")
                elif level in {"info", "warn"} and self._running:
                    self.end_button.configure(state="disabled", text="Connecting...")
            elif event_type == "transcript":
                self._append_transcript(event.get("role", "casie"), event.get("text", ""), update=False)
            elif event_type == "transcript_update":
                self._append_transcript(event.get("role", "casie"), event.get("text", ""), update=True)
            elif event_type == "levels":
                self._viz_target_mic = max(0.0, min(1.0, float(event.get("mic", 0.0) or 0.0)))
                self._viz_target_out = max(0.0, min(1.0, float(event.get("out", 0.0) or 0.0)))
            elif event_type == "show_timetable":
                print(
                    f"Received show_timetable event: {event.get('image_path', '')}",
                    flush=True,
                )
                self._show_timetable(
                    str(event.get("image_path", "") or ""),
                    int(event.get("duration_ms", 8000) or 8000),
                )
            elif event_type == "error":
                text = event.get("text", "Unknown error")
                self._set_status(text, "error")
                self._append_system_line(text)
            elif event_type == "auto_sleep":
                idle_seconds = int(event.get("idle_seconds", 0) or 0)
                text = f"Sleeping after {idle_seconds}s mic idle." if idle_seconds else "Sleeping after mic idle."
                self._set_status(text, "idle")
                self.end_button.configure(state="disabled", text="Sleeping...")
            elif event_type == "stopped":
                self._running = False
                self._worker = None
                if self._destroy_after_stop:
                    self._destroy_after_stop = False
                    self._close_after_stop = False
                    self._close_timetable()
                    self._stop_background_animation()
                    self.destroy()
                    return
                if self._close_after_stop:
                    self._close_after_stop = False
                self.end_button.configure(state="normal", text="Wake")
                self._set_status("Sleeping", "idle")

        self.after(max(8, UI_EVENT_POLL_MS), self._drain_events)

    def _on_close(self) -> None:
        if self._running and self._worker is not None:
            self._close_after_stop = True
            self._destroy_after_stop = True
            self._set_status("Stopping...", "warn")
            self.end_button.configure(state="disabled", text="Ending...")
            self._worker.stop()
            return
        self._close_timetable()
        self._stop_background_animation()
        self.destroy()


def main() -> None:
    app = CasieDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
