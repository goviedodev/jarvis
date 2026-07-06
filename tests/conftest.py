"""
Shared test fixtures and mock factories for Jarvis tests.

All external dependencies (PyAudio, Faster-Whisper, Ollama, Piper, Silero VAD)
are mocked so tests run fully offline and without hardware requirements.
"""

import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock, Mock
from pathlib import Path

import pytest

# ─── Asegurar que el proyecto raíz está en sys.path ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar módulos bajo test (después de asegurar el path)
from jarvis import (
    AudioManager, SpeechRecognizer, JarvisBrain,
    VoiceSynthesizer, VADManager, Jarvis,
    WHISPER_MODEL_SIZE, OLLAMA_HOST, OLLAMA_MODEL,
    VOICE_MODEL, VOICE_CONFIG, SAMPLE_RATE, CHUNK,
    VAD_CHUNK_SIZE, VAD_SPEECH_THRESHOLD, VAD_SILENCE_CHUNKS,
    VAD_TIMEOUT_SECS, VAD_MIN_SPEECH_CHUNKS, Colors, BASE_DIR,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCKS GLOBALES (aplicados automáticamente)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def mock_external_packages():
    """
    Mock de todos los paquetes externos que requieren hardware o red.
    Se aplica automáticamente a todos los tests.
    """
    patches = [
        patch("jarvis.pyaudio", MagicMock()),
        patch("jarvis.numpy", MagicMock()),
        patch("jarvis.requests", MagicMock()),
        patch("jarvis.subprocess", MagicMock()),
        patch("jarvis.tempfile.NamedTemporaryFile", MagicMock()),
        patch("jarvis.wave", MagicMock()),
        patch("jarvis.threading.Thread", MagicMock()),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES DE DATOS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_audio_bytes():
    """Genera bytes de audio PCM int16 simulados (1 segundo de silencio a 16kHz)."""
    import struct
    return struct.pack(f"<{SAMPLE_RATE}h", *([0] * SAMPLE_RATE))


@pytest.fixture
def sample_transcript():
    """Texto de transcripción simulado."""
    return "hola jarvis cuál es el clima de hoy"


@pytest.fixture
def sample_llm_response():
    """Respuesta simulada de Ollama."""
    return "El clima de hoy es soleado con una temperatura de 25 grados."


@pytest.fixture
def sample_audio_path():
    """Crea un archivo WAV temporal simulado."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(b"fake_wav_data")
    tmp.close()
    yield tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK DE PYTHON PATH (para voice model)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_voice_files(tmp_path):
    """Crea archivos de voz simulados en un directorio temporal y parchea VOICE_MODEL."""
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    model_file = voice_dir / "es_ES-davefx-medium.onnx"
    config_file = voice_dir / "es_ES-davefx-medium.onnx.json"
    model_file.write_text("fake_model_data")
    config_file.write_text('{"key": "value"}')

    with (
        patch("jarvis.VOICE_MODEL", str(model_file)),
        patch("jarvis.VOICE_CONFIG", str(config_file)),
        patch("jarvis.VOICE_DIR", str(voice_dir)),
    ):
        yield model_file, config_file


@pytest.fixture
def mock_missing_voice(tmp_path):
    """Simula que no hay archivos de voz."""
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    fake_model = voice_dir / "nonexistent.onnx"

    with patch("jarvis.VOICE_MODEL", str(fake_model)):
        yield fake_model


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES DE INSTANCIAS (con dependencias mockeadas)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def audio_manager():
    """Fixture de AudioManager con PyAudio mockeado."""
    manager = AudioManager()
    manager.p = MagicMock()
    manager.p.get_sample_size.return_value = 2
    yield manager


@pytest.fixture
def speech_recognizer():
    """Fixture de SpeechRecognizer con Whisper mockeado."""
    recognizer = SpeechRecognizer()
    recognizer.model = MagicMock()
    yield recognizer


@pytest.fixture
def jarvis_brain():
    """Fixture de JarvisBrain con requests mockeado."""
    brain = JarvisBrain()
    # Resetear historial
    brain.conversation_history = []
    yield brain


@pytest.fixture
def voice_synthesizer(mock_voice_files):
    """Fixture de VoiceSynthesizer con archivos de voz mockeados.
    Pasa rutas explícitas porque los defaults de __init__ se evalúan
    en tiempo de definición de clase, no en tiempo de llamada.
    """
    model_file, config_file = mock_voice_files
    synth = VoiceSynthesizer(
        model_path=str(model_file),
        config_path=str(config_file),
    )
    yield synth


@pytest.fixture
def vad_manager():
    """Fixture de VADManager con Silero VAD mockeado."""
    manager = VADManager()
    manager.model = MagicMock()
    manager._available = True
    yield manager


@pytest.fixture
def jarvis_instance(audio_manager, speech_recognizer, jarvis_brain, voice_synthesizer, vad_manager):
    """
    Fixture de Jarvis completo con todas las dependencias mockeadas.
    Útil para tests de integración del orquestador.
    """
    jarvis = Jarvis()
    # Reemplazar con mocks
    jarvis.audio = audio_manager
    jarvis.stt = speech_recognizer
    jarvis.brain = jarvis_brain
    jarvis.tts = voice_synthesizer
    jarvis.vad = vad_manager
    jarvis._initialized = True
    jarvis.running = False
    return jarvis


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS PARA SIMULAR RESPUESTAS DE RED
# ═══════════════════════════════════════════════════════════════════════════════

def mock_ollama_response(text: str):
    """Crea una respuesta simulada de la API de Ollama (non-streaming)."""
    return {
        "message": {"role": "assistant", "content": text}
    }


def mock_ollama_stream_chunks(text: str):
    """
    Genera chunks SSE simulados para streaming de Ollama.
    Cada palabra es un chunk individual.
    """
    words = text.split()
    for i, word in enumerate(words):
        is_last = (i == len(words) - 1)
        chunk = {
            "message": {"role": "assistant", "content": word + " "},
            "done": is_last
        }
        yield json.dumps(chunk).encode("utf-8")
    if not words:
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode("utf-8")


def mock_ollama_stream_chunks_by_char(text: str):
    """
    Genera chunks SSE más realistas: carácter por carácter.
    """
    for i, char in enumerate(text):
        is_last = (i == len(text) - 1)
        chunk = {
            "message": {"role": "assistant", "content": char},
            "done": is_last
        }
        yield json.dumps(chunk).encode("utf-8")
    if not text:
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode("utf-8")
