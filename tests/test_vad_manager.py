"""
Tests unitarios para VADManager (Silero VAD).

Cubre:
- Carga del modelo Silero VAD
- Detección de voz (is_speech)
- Ciclo completo de escucha (listen_for_speech)
- Buffer de pre-activación (pre_buffer)
- Timeout de escucha
- Flag running para detener loops
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest
import numpy as np

from jarvis import (
    VADManager, VAD_CHUNK_SIZE, SAMPLE_RATE,
    VAD_SPEECH_THRESHOLD, VAD_MIN_SPEECH_CHUNKS,
    VAD_SILENCE_CHUNKS, VAD_TIMEOUT_SECS
)


class TestVADManager:
    """Suite de tests para VADManager."""

    @pytest.fixture
    def vad(self):
        """Fixture de VADManager con modelo mockeado."""
        manager = VADManager()
        manager.model = MagicMock()
        manager._available = True
        yield manager

    @pytest.fixture
    def mock_stream(self):
        """Fixture de stream de audio mockeado."""
        stream = MagicMock()
        # Por defecto, retorna silencio
        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()
        stream.read.return_value = silence_chunk
        return stream

    # ─── Constructor ──────────────────────────────────────────────────────

    def test_constructor_initial_state(self):
        """Debe inicializar en estado correcto."""
        vad = VADManager()
        assert vad.model is None
        assert vad._available is False
        assert vad.running is False

    # ─── load ─────────────────────────────────────────────────────────────

    def test_load_imports_silero_vad(self):
        """load() debe cargar silero_vad."""
        vad = VADManager()

        with patch("jarvis.silero_vad") as mock_silero:
            mock_silero.load_silero_vad.return_value = MagicMock()
            vad.load()

        assert vad.model is not None
        assert vad.available is True

    def test_load_handles_import_error_gracefully(self):
        """load() debe manejar errores de importación."""
        vad = VADManager()

        with patch("jarvis.silero_vad") as mock_silero:
            mock_silero.load_silero_vad.side_effect = Exception("No CUDA")

            vad.load()

        assert vad.available is False

    def test_load_does_not_reload_if_loaded(self):
        """load() no debe recargar si ya está cargado."""
        vad = VADManager()
        vad.model = MagicMock()

        with patch("jarvis.silero_vad") as mock_silero:
            vad.load()

        mock_silero.load_silero_vad.assert_not_called()

    # ─── is_speech ────────────────────────────────────────────────────────

    def test_is_speech_returns_true_for_voice(self, vad):
        """is_speech() debe retornar True si la probabilidad supera el umbral."""
        # Simular que Silero retorna una probabilidad alta
        mock_tensor = MagicMock()
        vad.model.return_value = mock_tensor
        mock_tensor.item.return_value = VAD_SPEECH_THRESHOLD + 0.2

        voice_chunk = np.ones(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        with patch("jarvis.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            mock_torch.from_numpy.return_value = MagicMock()

            result = vad.is_speech(voice_chunk)

        assert result is True

    def test_is_speech_returns_false_for_silence(self, vad):
        """is_speech() debe retornar False si la probabilidad es baja."""
        mock_tensor = MagicMock()
        vad.model.return_value = mock_tensor
        mock_tensor.item.return_value = VAD_SPEECH_THRESHOLD - 0.2

        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        with patch("jarvis.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            mock_torch.from_numpy.return_value = MagicMock()

            result = vad.is_speech(silence_chunk)

        assert result is False

    def test_is_speech_returns_false_if_no_model(self):
        """is_speech() debe retornar False si no hay modelo cargado."""
        vad = VADManager()  # model = None
        result = vad.is_speech(b"test")
        assert result is False

    def test_is_speech_converts_int16_to_float32_correctly(self, vad):
        """is_speech() debe convertir Int16 a Float32 [-1, 1] normalizado."""
        mock_tensor = MagicMock()
        vad.model.return_value = mock_tensor
        mock_tensor.item.return_value = 0.5

        # Valor máximo positivo en int16
        max_chunk = np.array([32767], dtype=np.int16).tobytes()

        with patch("jarvis.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            mock_torch.from_numpy = MagicMock()

            vad.is_speech(max_chunk)

        # Verificar que se convirtió dividiendo por 32768
        call_args = mock_torch.from_numpy.call_args
        if call_args:
            audio_array = call_args[0][0]
            assert np.max(audio_array) <= 1.0

    # ─── listen_for_speech ────────────────────────────────────────────────

    def test_listen_for_speech_returns_frames_when_speech_detected(self, vad, mock_stream):
        """listen_for_speech() debe retornar frames cuando detecta voz."""
        vad.running = True

        # Simular chunks de voz (probabilidad alta)
        speech_chunk = (np.ones(VAD_CHUNK_SIZE, dtype=np.int16) * 1000).tobytes()
        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        # Retornar voz durante VAD_MIN_SPEECH_CHUNKS + 1, luego silencio
        # VAD_MIN_SPEECH_CHUNKS = 6 raids de voz, luego 50+ raids de silencio
        speech_chunks_count = VAD_MIN_SPEECH_CHUNKS + 3
        silence_chunks_count = VAD_SILENCE_CHUNKS + 5
        mock_stream.read.side_effect = (
            [speech_chunk] * speech_chunks_count +
            [silence_chunk] * silence_chunks_count
        )

        # Configurar is_speech mock
        original_is_speech = vad.is_speech

        def mock_is_speech(data):
            if data == speech_chunk:
                return True
            return False

        vad.is_speech = mock_is_speech

        frames, total = vad.listen_for_speech(mock_stream, timeout_secs=10)

        assert len(frames) > 0
        assert total > VAD_MIN_SPEECH_CHUNKS

    def test_listen_for_speech_returns_empty_on_timeout(self, vad, mock_stream):
        """listen_for_speech() debe retornar vacío si nadie habla."""
        vad.running = True

        # Siempre silencio
        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()
        mock_stream.read.return_value = silence_chunk

        # Timeout rápido
        frames, total = vad.listen_for_speech(mock_stream, timeout_secs=0.1)

        assert len(frames) == 0
        assert total > 0  # Al menos procesó algunos chunks

    def test_listen_for_speech_stops_when_running_false(self, vad, mock_stream):
        """listen_for_speech() debe detenerse si running=False."""
        vad.running = True

        # Configurar running para que sea False después de algunos chunks
        call_count = 0

        def mock_read(size):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                vad.running = False
            return np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        mock_stream.read.side_effect = mock_read

        frames, total = vad.listen_for_speech(mock_stream, timeout_secs=10)

        assert len(frames) == 0  # No detectó voz, running se puso False

    def test_listen_for_speech_includes_pre_buffer(self, vad, mock_stream):
        """
        listen_for_speech() debe incluir los chunks de pre-activación
        en los frames retornados (para no perder el inicio de la frase).
        """
        vad.running = True

        # 2 chunks de voz que activan (menos de VAD_MIN_SPEECH_CHUNKS),
        # luego suficientes para activar, luego silencio
        speech_chunk = (np.ones(VAD_CHUNK_SIZE, dtype=np.int16) * 1000).tobytes()
        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        chunks_to_activate = VAD_MIN_SPEECH_CHUNKS + 2
        mock_stream.read.side_effect = (
            [speech_chunk] * chunks_to_activate +
            [silence_chunk] * (VAD_SILENCE_CHUNKS + 5)
        )

        original_is_speech = vad.is_speech

        def mock_is_speech(data):
            return data == speech_chunk

        vad.is_speech = mock_is_speech

        frames, total = vad.listen_for_speech(mock_stream, timeout_secs=10)

        # Debe incluir también los chunks de activación (pre_buffer)
        assert len(frames) >= VAD_MIN_SPEECH_CHUNKS

    def test_listen_for_speech_handles_empty_speech_frames(self, vad, mock_stream):
        """listen_for_speech() debe retornar lista vacía si no hay suficientes frames de voz."""
        vad.running = True

        speech_chunk = (np.ones(VAD_CHUNK_SIZE, dtype=np.int16) * 1000).tobytes()
        silence_chunk = np.zeros(VAD_CHUNK_SIZE, dtype=np.int16).tobytes()

        # Solo 1 chunk de voz (insuficiente para activar)
        mock_stream.read.side_effect = [speech_chunk] * 2 + [silence_chunk] * 100

        original_is_speech = vad.is_speech

        def mock_is_speech(data):
            return data == speech_chunk

        vad.is_speech = mock_is_speech

        frames, total = vad.listen_for_speech(mock_stream, timeout_secs=1)

        assert len(frames) == 0

    # ─── available property ───────────────────────────────────────────────

    def test_available_property(self):
        """available debe reflejar el estado de carga del modelo."""
        vad = VADManager()
        assert vad.available is False

        vad._available = True
        assert vad.available is True

        vad._available = False
        assert vad.available is False
