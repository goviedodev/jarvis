# AGENTS.md

Spanish-language voice assistant ("JARVIS") — single-file Python project. Pipeline: microphone audio → Faster-Whisper STT → Ollama LLM → Piper TTS.

## Running

- `./run.sh` — main entry point. Handles venv activation, dependency check, voice model download, and Ollama check. Passes args through to `jarvis.py`.
- `python3 jarvis.py` — run directly if venv is already active. Args: `--vad`, `--ptt`, `--text`, `--list-devices`, `--test-mic`, `--quick "text"`, `--model <ollama-model>`, `--whisper <model-size>`.
- `python3 stt_test.py` — standalone STT test (records + transcribes).
- `python3 tts_test.py` — standalone TTS test (synthesizes + plays).

## Environment Variables

- `OLLAMA_HOST` — Ollama server URL (default: `http://localhost:11434`).
- `JARVIS_MODEL` — Ollama model name (default: `qwen2.5-coder:7b`).

## External Dependencies (must be running)

- **Ollama** — must be started separately (`ollama serve`). Check with `curl http://localhost:11434/api/tags`.
- **CUDA GPU** — Faster-Whisper runs on CUDA FP16. No CPU fallback is configured.
- **Piper CLI** — installed via pip (`piper-tts`), used as a subprocess for TTS.

## Setup

1. Create venv: `python3 -m venv venv`
2. Activate and install: `source venv/bin/activate && pip install faster-whisper pyaudio requests piper-tts keyboard sounddevice numpy silero-vad torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
3. Start Ollama in another terminal: `ollama serve`
4. Voice model and Whisper model auto-download on first run via `run.sh`.

## Key Architecture Notes

- All code lives in `jarvis.py` (928 lines). Five classes: `AudioManager`, `SpeechRecognizer`, `JarvisBrain`, `VoiceSynthesizer`, `VADManager`, orchestrated by `Jarvis`.
- Config is hardcoded constants in `jarvis.py` (lines 36-75), not a config file.
- Push-to-talk mode requires root (`sudo`) on Linux for global keyboard capture. Falls back to VAD mode or interactive text mode without root.
- VAD (Voice Activity Detection) mode uses Silero VAD for hands-free activation. See [`docs/VAD_IMPLEMENTACION.md`](docs/VAD_IMPLEMENTACION.md) for the complete guide — architecture, chunk sizing, buffer strategy, tuning, and lessons learned during development.
- TTS uses a queue-based architecture with a single worker thread to avoid audio overlap. PyAudio and streams are kept persistent to reduce overhead. See [`docs/TTS_OPTIMIZACION.md`](docs/TTS_OPTIMIZACION.md) for the complete optimization guide — queue pattern, persistent resources, sentence grouping, and tuning MIN_TTS_LENGTH.
- Conversation history is capped at 10 messages (5 exchanges) for Ollama context.
- TTS resamples Piper output from 22050 Hz to 48000 Hz for device playback.
- Whisper model downloads to `models/` (gitignored). Voice models in `voices/` (also gitignored).

## Gotchas

- `pyaudio` may need system packages (`portaudio19-dev` on Debian/Ubuntu) before pip install.
- Push-to-talk uses the `keyboard` Python package, which needs root or `/dev/input` access on Linux.
- The `run.sh` voice download fetches from HuggingFace (`rhasspy/piper-voices`). Requires internet on first run.
- Ollama connection failure is handled gracefully (returns a message), but no retry logic.
