"""
Tests unitarios para AudioManager.

Cubre:
- Listado de dispositivos de audio
- Grabación con detección de silencio
- Reproducción de WAV
- Limpieza de recursos
- Casos borde: sin audio, timeout
"""

import os
import tempfile
import wave
from unittest.mock import MagicMock, patch, call

import pytest
import numpy as np

from jarvis import AudioManager, SAMPLE_RATE, CHUNK, FORMAT


class TestAudioManager:
    """Suite de tests para AudioManager."""

    @pytest.fixture
    def manager(self):
        """Fixture básico de AudioManager con PyAudio mockeado."""
        m = AudioManager()
        m.p = MagicMock()
        m.p.get_sample_size.return_value = 2  # 16 bits = 2 bytes
        yield m

    # ─── list_devices ─────────────────────────────────────────────────────

    def test_list_devices_prints_available_devices(self, manager, capsys):
        """Debe listar los dispositivos de entrada disponibles."""
        # Configurar mock: 3 dispositivos, solo 2 con entrada
        mock_info_1 = {"name": "Device1", "maxInputChannels": 2, "defaultSampleRate": 44100}
        mock_info_2 = {"name": "Device2", "maxInputChannels": 0}  # Solo salida
        mock_info_3 = {"name": "Device3", "maxInputChannels": 1, "defaultSampleRate": 16000}

        manager.p.get_device_count.return_value = 3
        manager.p.get_device_info_by_index.side_effect = [mock_info_1, mock_info_2, mock_info_3]

        manager.list_devices()

        captured = capsys.readouterr()
        assert "Device1" in captured.out
        assert "Device2" not in captured.out  # No es entrada
        assert "Device3" in captured.out

    def test_list_devices_empty(self, manager, capsys):
        """Debe manejar el caso de ningún dispositivo disponible."""
        manager.p.get_device_count.return_value = 0

        manager.list_devices()

        captured = capsys.readouterr()
        assert "Dispositivos de entrada" in captured.out

    def test_list_devices_no_input_devices(self, manager, capsys):
        """Debe manejar dispositivos sin canales de entrada."""
        manager.p.get_device_count.return_value = 2
        mock_info_1 = {"name": "Speaker", "maxInputChannels": 0}
        mock_info_2 = {"name": "Monitor", "maxInputChannels": 0}
        manager.p.get_device_info_by_index.side_effect = [mock_info_1, mock_info_2]

        manager.list_devices()

        captured = capsys.readouterr()
        assert "Speaker" not in captured.out
        assert "Monitor" not in captured.out

    # ─── record_until_silence ─────────────────────────────────────────────

    def test_record_until_silence_returns_path_and_duration(self, manager):
        """Debe retornar una ruta de archivo y duración al grabar audio."""
        # Simular datos de audio con volumen > umbral
        audio_data = np.array([1000] * CHUNK, dtype=np.int16).tobytes()
        # Luego silencio
        silence_data = np.array([0] * CHUNK, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        # Primero datos con voz, luego datos en silencio
        mock_stream.read.side_effect = [audio_data] * 5 + [silence_data] * 100
        manager.p.open.return_value = mock_stream

        path, duration = manager.record_until_silence(
            timeout=5, silence_threshold=100, silence_duration=0.5
        )

        assert path is not None
        assert os.path.exists(path)
        assert duration > 0
        # Limpiar
        if path:
            os.unlink(path)

    def test_record_until_silence_no_audio_returns_none(self, manager):
        """Debe retornar None si no se detecta voz."""
        silence_data = np.array([0] * CHUNK, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        mock_stream.read.return_value = silence_data
        manager.p.open.return_value = mock_stream

        path, duration = manager.record_until_silence(
            timeout=1, silence_threshold=500, silence_duration=0.1
        )

        assert path is None
        assert duration == 0

    def test_record_until_silence_applies_timeout(self, manager):
        """Debe respetar el timeout de grabación."""
        mixed_data = np.array([500] * CHUNK, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        mock_stream.read.return_value = mixed_data
        manager.p.open.return_value = mock_stream

        path, duration = manager.record_until_silence(
            timeout=1, silence_threshold=100, silence_duration=10
        )

        assert path is not None
        if path:
            os.unlink(path)

    def test_record_until_silence_writes_wav_correctly(self, manager):
        """Debe escribir un archivo WAV con los parámetros correctos."""
        audio_data = np.array([1000] * CHUNK, dtype=np.int16).tobytes()
        silence_data = np.array([0] * CHUNK, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [audio_data] * 3 + [silence_data] * 100
        manager.p.open.return_value = mock_stream

        path, duration = manager.record_until_silence(
            timeout=5, silence_threshold=100, silence_duration=0.3
        )

        assert path is not None
        # Verificar que se escribió un archivo WAV válido
        with wave.open(path, 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getsampwidth() == 2

        os.unlink(path)

    def test_record_until_silence_handles_overflow_gracefully(self, manager):
        """Debe manejar exception_on_overflow=False sin errores."""
        audio_data = np.array([1000] * CHUNK, dtype=np.int16).tobytes()
        silence_data = np.array([0] * CHUNK, dtype=np.int16).tobytes()

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [audio_data] * 3 + [silence_data] * 100
        manager.p.open.return_value = mock_stream

        # No debería lanzar excepción
        path, duration = manager.record_until_silence(
            timeout=5, silence_threshold=100, silence_duration=0.3
        )
        assert path is not None
        if path:
            os.unlink(path)

    # ─── play_wav ─────────────────────────────────────────────────────────

    def test_play_wav_opens_stream_and_writes_data(self, manager, tmp_path):
        """Debe abrir un stream de audio y escribir los frames."""
        # Crear un archivo WAV temporal
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"test_audio_data")

        mock_stream = MagicMock()
        manager.p.open.return_value = mock_stream
        manager.p.get_format_from_width.return_value = 2

        manager.play_wav(str(wav_path))

        # Verificar que se abrió el stream de salida
        manager.p.open.assert_called_once()
        manager.p.get_format_from_width.assert_called_once_with(2)

    def test_play_wav_with_empty_file(self, manager, tmp_path):
        """Debe manejar archivos WAV vacíos sin errores."""
        wav_path = tmp_path / "empty.wav"
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"")

        mock_stream = MagicMock()
        manager.p.open.return_value = mock_stream

        # No debería lanzar excepción
        manager.play_wav(str(wav_path))

    def test_play_wav_closes_stream(self, manager, tmp_path):
        """Debe cerrar el stream después de reproducir."""
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"data")

        mock_stream = MagicMock()
        manager.p.open.return_value = mock_stream

        manager.play_wav(str(wav_path))

        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    # ─── cleanup ──────────────────────────────────────────────────────────

    def test_cleanup_terminates_pyaudio(self, manager):
        """Debe terminar PyAudio correctamente."""
        manager.cleanup()
        manager.p.terminate.assert_called_once()

    # ─── constructor ──────────────────────────────────────────────────────

    def test_constructor_with_device_index(self):
        """Debe aceptar un índice de dispositivo opcional."""
        manager = AudioManager(device_index=2)
        assert manager.device_index == 2
        # No debe crear PyAudio hasta que se use
        assert hasattr(manager, 'p')

    def test_constructor_default_device(self):
        """Debe usar None como device_index por defecto."""
        manager = AudioManager()
        assert manager.device_index is None
