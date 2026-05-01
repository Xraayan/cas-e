import asyncio
import contextlib
import math
import os
import queue
import re
import threading
import time
import tkinter as tk
import tkinter.font as tkfont

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

import agent as core
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
UI_DEBUG_AUDIO = os.getenv("CASIE_UI_DEBUG_AUDIO", "false").strip().lower() in {"1", "true", "yes", "on"}
UI_REDRAW_INTERVAL_MS = int(os.getenv("CASIE_UI_REDRAW_INTERVAL_MS", "70"))
UI_MAX_MESSAGES = int(os.getenv("CASIE_UI_MAX_MESSAGES", "120"))
UI_LEVEL_EMIT_MS = int(os.getenv("CASIE_UI_LEVEL_EMIT_MS", "70"))
UI_VIZ_BARS = int(os.getenv("CASIE_UI_VIZ_BARS", "34"))
UI_VIZ_FPS_MS = int(os.getenv("CASIE_UI_VIZ_FPS_MS", "50"))
UI_EVENT_POLL_MS = int(os.getenv("CASIE_UI_EVENT_POLL_MS", "16"))


class LiveUiWorker:
    def __init__(self, event_sink):
        self._event_sink = event_sink
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
        self._event_sink({"type": event_type, **payload})

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
        input_stream_kwargs = core._stream_kwargs("input")
        output_stream_kwargs = core._stream_kwargs("output")
        run_context = RunContext(session_id="desktop-session")
        transcript_cache: dict[str, str] = {"you": "", "casie": ""}
        transcript_last_emit_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}
        transcript_last_chunk_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}
        transcript_last_raw_chunk: dict[str, str] = {"you": "", "casie": ""}
        transcript_last_raw_chunk_ts: dict[str, float] = {"you": 0.0, "casie": 0.0}

        self._emit("status", text="Connecting to CASie...", level="info")

        async with client.aio.live.connect(model=core.MODEL, config=config) as session:
            self._emit("status", text="Connected. Microphone is live.", level="live")
            if core.STARTUP_GREETING_ENABLED:
                await core._send_text_turn(session, core.GREETING_TEXT)

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
                pcm = (np.clip(indata[:, 0], -1.0, 1.0) * 32767).astype(np.int16).tobytes()

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

            async def watch_for_stop() -> None:
                while not stop_event.is_set():
                    if self._stop_requested.is_set():
                        stop_event.set()
                        return
                    await asyncio.sleep(0.1)

            async def listen_audio() -> None:
                blocksize = int(core.SAMPLE_RATE * (UI_INPUT_BLOCK_MS / 1000.0))
                with sd.InputStream(
                    samplerate=core.SAMPLE_RATE,
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

                while not stop_event.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic_queue.get(), timeout=0.08)
                    except asyncio.TimeoutError:
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
                    samplerate=core.OUTPUT_RATE,
                    channels=1,
                    dtype="float32",
                    device=output_device,
                    **output_stream_kwargs,
                )
                output_stream.start()
                current_rate = core.OUTPUT_RATE

                try:
                    while not stop_event.is_set():
                        raw, desired_rate = await speaker_queue.get()
                        if not raw:
                            level_state["out"] *= 0.9
                            continue

                        if desired_rate != current_rate:
                            output_stream = core._restart_output_stream(
                                output_stream,
                                desired_rate,
                                device_idx=output_device,
                                extra_kwargs=output_stream_kwargs,
                            )
                            current_rate = desired_rate

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

            watcher = asyncio.create_task(watch_for_stop())
            listener = asyncio.create_task(listen_audio())
            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive_audio())
            player = asyncio.create_task(play_audio())
            meter = asyncio.create_task(emit_levels())
            tasks = {watcher, listener, sender, receiver, player, meter}
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                core_stream_tasks = {"listen_audio", "send_audio", "receive_audio", "play_audio"}
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    coro_name = task.get_coro().__name__
                    if exc is None:
                        # Any core stream task finishing early means the session ended;
                        # return so outer loop can reconnect cleanly.
                        if self._stop_requested.is_set():
                            return
                        if coro_name not in core_stream_tasks:
                            continue
                        self._emit("status", text="Session dropped. Reconnecting...", level="warn")
                        return
                    if core._is_expected_disconnect(exc) or self._stop_requested.is_set():
                        self._emit("status", text="Connection dropped. Reconnecting...", level="warn")
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

        self._events = queue.Queue()
        self._worker = LiveUiWorker(self._events.put)
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

        self._build_layout()
        self._set_status("Ready", "idle")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(40, self._drain_events)

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

        self._canvas.create_text(
            cx,
            canvas_h - 24,
            text="ROBOTICS AND AI DEPT | RIT KOTTAYAM",
            fill="#e7c7a2",
            font=self._footer_font,
            anchor="center",
            tags=("viz",),
        )

        self._canvas.tag_lower("viz")
        self._canvas.tag_raise("msg")

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
        if not hasattr(self, "_canvas"):
            return

        self._canvas.delete("msg")
        canvas_w = max(1, self._canvas.winfo_width())
        canvas_h = max(1, self._canvas.winfo_height())
        top_pad = 150
        bottom_pad = 82
        line_gap = 12
        available_h = max(120, canvas_h - top_pad - bottom_pad)

        for role in ("casie", "you"):
            selected: list[tuple[str, int]] = []
            used = 0
            role_messages = [text for msg_role, text in self._messages if msg_role == role]
            for text in reversed(role_messages):
                h = min(self._measure_message_height(role, text, canvas_w), max(44, available_h // 3))
                need = h if not selected else h + line_gap
                if selected and used + need > available_h:
                    break
                selected.append((text, h))
                used += need

            y = max(top_pad, canvas_h - bottom_pad - used)
            style = self._message_style(role, canvas_w)
            for text, h in reversed(selected):
                max_chars = max(120, int(style["width"]) // max(1, self._avg_char_width) * max(2, h // self._line_height))
                display_text = text if len(text) <= max_chars else "..." + text[-max_chars:]
                self._canvas.create_text(
                    style["x"],
                    y,
                    text=display_text,
                    anchor=style["anchor"],
                    justify=style["justify"],
                    width=style["width"],
                    fill=style["fill"],
                    font=self._chat_font,
                    tags=("msg",),
                )
                y += h + line_gap

    def _stop_background_animation(self) -> None:
        if self._viz_job is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._viz_job)
            self._viz_job = None
        if self._redraw_job is not None:
            with contextlib.suppress(Exception):
                self.after_cancel(self._redraw_job)
            self._redraw_job = None

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
        self._reset_conversation_view()
        self._running = True
        self._set_status("Connecting...", "info")
        self.end_button.configure(state="disabled", text="Connecting...")
        self._worker.start()

    def _stop_session(self) -> None:
        if not self._running:
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
            elif event_type == "error":
                text = event.get("text", "Unknown error")
                self._set_status(text, "error")
                self._append_system_line(text)
            elif event_type == "stopped":
                self._running = False
                if self._destroy_after_stop:
                    self._destroy_after_stop = False
                    self._close_after_stop = False
                    self._stop_background_animation()
                    self.destroy()
                    return
                if self._close_after_stop:
                    self._close_after_stop = False
                self.end_button.configure(state="normal", text="Wake")
                self._set_status("Sleeping", "idle")

        self.after(max(8, UI_EVENT_POLL_MS), self._drain_events)

    def _on_close(self) -> None:
        if self._running:
            self._close_after_stop = True
            self._destroy_after_stop = True
            self._set_status("Stopping...", "warn")
            self.end_button.configure(state="disabled", text="Ending...")
            self._worker.stop()
            return
        self._stop_background_animation()
        self.destroy()


def main() -> None:
    app = CasieDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
