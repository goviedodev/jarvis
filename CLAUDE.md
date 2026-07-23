# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Spanish-language voice assistant. Everything runs locally: microphone → Silero VAD → Faster-Whisper (STT, CUDA) → Ollama (LLM) → Piper (TTS). Code, comments, docstrings, and console output are all in Spanish — match that when editing.

## The venv is relocated — do not activate it

The virtualenv was created at `/home/goviedo/tmp/jarvis/venv` and moved here, so every shebang-based script inside it points at a path that no longer exists. `source venv/bin/activate` and `venv/bin/pip` both fail.

Always invoke the interpreter directly:

```bash
./venv/bin/python3 jarvis.py --text     # run
./venv/bin/python3 -m pip list          # inspect/install packages
```

`run.sh` already does this and documents why. There is no `requirements.txt`; the dependency list lives in `run.sh`'s check block and in README.md.

## Commands

```bash
./run.sh                       # main entry: checks deps, downloads voice model, checks Ollama, then runs jarvis.py
./run.sh --vad                 # args pass through to jarvis.py
sudo ./run.sh --ptt            # push-to-talk needs root (see below)

./venv/bin/python3 jarvis.py --list-devices        # enumerate audio devices
./venv/bin/python3 jarvis.py --test-mic            # mic level check
./venv/bin/python3 jarvis.py --quick "texto"       # synthesize one string and exit — fastest TTS smoke test
./venv/bin/python3 stt_test.py                     # standalone STT (records + transcribes)
./venv/bin/python3 tts_test.py                     # standalone TTS
```

Input mode flags (`--vad`, `--ptt`, `--text`) are mutually exclusive; omitting all of them shows an interactive menu. Orthogonal flags: `--writer` (print tokens to console instead of speaking), `--pdev` (use `pi` CLI instead of Ollama), `--model`, `--whisper`.

### Tests

`pytest.ini` and ~1800 lines of tests exist under `tests/`, but **the suite cannot currently run**: pytest is not installed in the venv, and the system pytest (`~/.local/bin/pytest`) can't import `jarvis.py` because `pyaudio`/`numpy`/`requests` are venv-only. To restore it:

```bash
./venv/bin/python3 -m pip install pytest pytest-cov
./venv/bin/python3 -m pytest                                   # full suite, coverage per pytest.ini
./venv/bin/python3 -m pytest tests/test_jarvis_brain.py -k think_stream   # single test
./venv/bin/python3 -m pytest -m unit                           # markers: unit, integration, e2e, slow, streaming
```

`tests/conftest.py` imports `jarvis` at module level, then an autouse fixture patches `jarvis.pyaudio`, `jarvis.numpy`, `jarvis.requests`, `jarvis.subprocess`, `jarvis.wave`, and `jarvis.threading.Thread`. So tests need the real packages importable but never touch hardware, network, or Ollama. Note that patching `threading.Thread` means the TTS worker threads never actually start under test.

## Architecture

All application code is in `jarvis.py` (~1260 lines). Six classes, one orchestrator:

| Class | Responsibility |
|---|---|
| `AudioManager` | PyAudio capture/playback, device enumeration, amplitude-threshold recording |
| `SpeechRecognizer` | Faster-Whisper, lazy `load()`, CUDA FP16 only — no CPU fallback |
| `JarvisBrain` | LLM calls + conversation history (capped at 10 messages) |
| `VoiceSynthesizer` | Piper subprocess + two-stage audio worker pipeline |
| `VADManager` | Silero VAD, lazy `load()`, degrades to `available = False` on any failure |
| `Jarvis` | Wires the above; owns mode selection and the per-mode loops |

### The four LLM code paths

`JarvisBrain` has four near-duplicate request methods, and `Jarvis.process_query()` branches between them on `(pdev_mode, writer_mode)`. When changing prompt handling, history management, or error behavior, check all four — they do not share a helper:

- `think()` — non-streaming Ollama; largely superseded, kept for direct/simple use.
- `think_stream()` — streaming Ollama that yields **complete sentences**, buffering until it hits `.`/`?`/`!` *and* the buffer reaches `MIN_TTS_LENGTH` (100 chars, a local constant at `jarvis.py:416`). This grouping is deliberate: it trades a little first-audio latency for far fewer Piper invocations. Feeds TTS mode.
- `think_stream_tokens()` — streaming Ollama that yields **individual tokens**. Feeds writer mode.
- `think_pi()` — shells out to the `pi` CLI (`--pdev`). Not streaming: it flattens the whole history into one text prompt (`Usuario:`/`Jarvis:` lines) because `pi` has no chat-message API here, and returns a single string. So `--pdev --writer` prints the full answer at once rather than token-by-token.

History is appended only on success, in each method separately. Failures return/yield a Spanish apology string rather than raising — callers never see exceptions from the brain.

### TTS pipeline (double buffering)

`VoiceSynthesizer.__init__` starts two daemon threads immediately on construction, before any voice model check:

```
speak_nonblocking(text) → _queue → _synth_loop (Piper subprocess + resample)
                                 → _audio_queue (maxsize=3, backpressure)
                                 → _playback_loop (writes to persistent PyAudio stream)
```

Splitting synthesis from playback lets sentence N+1 synthesize while N plays. `_audio_queue`'s `maxsize=3` is the backpressure valve — raising it buys smoothness at the cost of memory and of a longer tail after an interrupt. `_ensure_audio_stream()` keeps one PyAudio instance and stream alive across utterances; `cleanup()` tears it down and the next call transparently rebuilds it, which is how playback errors self-heal.

Piper emits 22050 Hz; `_synthesize_text()` resamples to `TTS_OUTPUT_RATE` (48000) with `np.interp` because the target device expects 48k. A `paInvalidSampleRate` error means this constant doesn't match the device.

`speak()` (synchronous) and `speak_nonblocking()` (queued) both exist and bypass each other's ordering — don't mix them in one flow or audio will overlap. This overlap bug is exactly what the queue architecture was introduced to fix; see `docs/TTS_OPTIMIZACION.md`.

### VAD loop

`VADManager.listen_for_speech()` runs two phases against a live stream. Phase one waits for `VAD_MIN_SPEECH_CHUNKS` (6, ~180ms) consecutive speech chunks while holding them in a pre-buffer, so the activating audio isn't lost from the transcript; any non-speech chunk resets the counter and clears the buffer. Phase two records until `VAD_SILENCE_CHUNKS` (50, ~1.5s) of silence or `VAD_TIMEOUT_SECS`.

`VAD_CHUNK_SIZE` is 512 samples — this is Silero's hard minimum at 16 kHz, not a tunable. Note it differs from `CHUNK` (1024) used for ordinary recording. Full rationale and tuning guidance in `docs/VAD_IMPLEMENTACION.md`.

### Configuration

There is no config file. All tunables are module-level constants in `jarvis.py:39-91`, and only two are overridable by environment: `OLLAMA_HOST` and `JARVIS_MODEL` (plus `PDEV_MODEL` / `PDEV_SYSTEM_PROMPT` for `--pdev`). Adding a knob means adding a constant there and, if it should be runtime-settable, an argparse flag in `main()`.

Two system prompts exist and drift apart easily: `JARVIS_SYSTEM_PROMPT` (inline, `jarvis.py:80`) is used for Ollama; `prompts/pdev_system.md` is passed to `pi` via `--append-system-prompt` for `--pdev`.

## External dependencies that must be running

- **Ollama** — start separately (`ollama serve`). Not needed with `--pdev`. Connection failure is caught and surfaced as a spoken message; there is no retry.
- **CUDA GPU** — Faster-Whisper is hardcoded to CUDA FP16.
- **Piper CLI** — a subprocess (`piper` on PATH from `pip install piper-tts`), not a Python import.
- **`pi` CLI** — only for `--pdev`.
- Voice models live in `voices/`, Whisper models in `models/`. Both are gitignored and auto-download.

## Gotchas

- Push-to-talk uses the `keyboard` package, which needs root or `/dev/input` read access. `Jarvis._check_push_to_talk()` detects this and silently falls back to VAD, then to text.
- Writer mode (`--writer`) skips the Piper voice-file check and model load entirely — a TTS regression will not show up when testing in writer mode.
- `pyaudio` may need `portaudio19-dev` (Debian/Ubuntu) before pip install. A prebuilt `.deb` sits in the repo root as a fallback.

## Documentation state

`docs/VAD_IMPLEMENTACION.md`, `docs/TTS_OPTIMIZACION.md`, and `docs/RECOMENDACION_LLM_JARVIS.md` are substantial design write-ups worth reading before touching those subsystems (the LLM one benchmarks candidate Ollama models against a 12GB RTX 3060 and explains why `qwen2.5-coder:7b` is the default).

`README.md` is current: it documents `--pdev`, the real Whisper default (`large-v3-turbo`), the pytest suite, and the relocated-venv failure mode.

`AGENTS.md` has **not** been updated and has drifted: it cites 928 lines and config at lines 36-75, and predates `--pdev` entirely. Prefer the source when they disagree, and update docs alongside code changes.

The `pi` CLI package name is `@earendil-works/pi-coding-agent`. Earlier hints in `run.sh` and `jarvis.py` pointed at `@anthropic-ai/pi` and `@anthropic-ai/pi-coding-agent`, which both 404 on npm; they were corrected on 2026-07-22.
