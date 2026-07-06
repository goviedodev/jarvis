# AGENTS.md

Spanish-language voice assistant ("JARVIS") — single-file Python project. Pipeline: microphone audio → Faster-Whisper STT → Ollama LLM → Piper TTS.

## Running

- `./run.sh` — main entry point. Handles venv activation, dependency check, voice model download, and Ollama check. Passes args through to `jarvis.py`.
- `python3 jarvis.py` — run directly if venv is already active. Args: `--list-devices`, `--test-mic`, `--quick "text"`, `--model <ollama-model>`, `--whisper <model-size>`.
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
2. Activate and install: `source venv/bin/activate && pip install faster-whisper pyaudio requests piper-tts keyboard sounddevice numpy`
3. Start Ollama in another terminal: `ollama serve`
4. Voice model and Whisper model auto-download on first run via `run.sh`.

## Key Architecture Notes

- All code lives in `jarvis.py` (~658 lines). Four classes: `AudioManager`, `SpeechRecognizer`, `JarvisBrain`, `VoiceSynthesizer`, orchestrated by `Jarvis`.
- Config is hardcoded constants in `jarvis.py` (lines 28-54), not a config file.
- Push-to-talk mode requires root (`sudo`) on Linux for global keyboard capture. Falls back to interactive text mode without root.
- Conversation history is capped at 10 messages (5 exchanges) for Ollama context.
- TTS resamples Piper output from 22050 Hz to 48000 Hz for device playback.
- Whisper model downloads to `models/` (gitignored). Voice models in `voices/` (also gitignored).

## Gotchas

- `pyaudio` may need system packages (`portaudio19-dev` on Debian/Ubuntu) before pip install.
- Push-to-talk uses the `keyboard` Python package, which needs root or `/dev/input` access on Linux.
- The `run.sh` voice download fetches from HuggingFace (`rhasspy/piper-voices`). Requires internet on first run.
- Ollama connection failure is handled gracefully (returns a message), but no retry logic.
